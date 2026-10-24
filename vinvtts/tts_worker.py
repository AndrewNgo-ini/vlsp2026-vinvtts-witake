"""VieNeu candidate generator behind serve.py (runs in VieNeu's own venv; serve.py's env has incompatible torch).

POST /gen {"text", "speaker_id" | "ref_wav_b64", "n"} -> {"sr", "wavs": [b64 PCM16 wav, ...]}   (n sampled candidates)

  --mode base : untuned VieNeu v3 Turbo, in-context ref codes, our tags -> its native tags (gen_zs.VN). Track A voice
                = the speaker's reference clip (shortest >= 4 s train clip, as mkeval.py).
  --mode mix  : both engines in one process; n/2 candidates from each, best-of-N picks per request (Track-B proxy:
                base wins SS, ft wins WER/NV -> complementary candidates). Needs --model (ft) too.
  --mode omni : OmniVoice LoRA merge (run_omni_ft.sh; CC-BY-NC weights -> only if the organisers allow). Run with the
                OmniVoice venv. Track A ref = speaker clip + its transcript; Track B ref transcript = OmniVoice's own ASR.
  --mode ft   : our LoRA-merged model, speaker embedding only, tags -> emotion tokens (prep_vieneu.TAG2EMO). Track A voice
                = mean of the speaker's training-clip embeddings (d-vector averaging, SV2TTS).

    /mnt/vinvtts/src/vieneu/.venv/bin/python tts_worker.py --mode base --train /mnt/vinvtts/data/vinv-tts/train.jsonl --port 8001
"""
import cpulimit  # noqa: F401
import argparse, base64, io, json, re, tempfile
import numpy as np, soundfile as sf, torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

TAGRE_W = re.compile(r"\[(laughter|breathing|sniff|throatclearing)\]")
VN = {"laughter": "[cười]", "throatclearing": "[hắng giọng]", "breathing": "[thở dài]", "sniff": ""}
app = FastAPI()


class Gen(BaseModel):
    text: str
    speaker_id: str | None = None
    ref_wav_b64: str | None = None
    track: str | None = None      # "A"/"B": --inpaint-k-a applies to A requests only
    gs: float | None = None       # omni guidance scale override (per track, set by prod.py)
    tag_w: float | None = None    # omni tag-CFG weight (None = worker default --tag-w; 0 = plain generation)
    ref_text: str | None = None   # transcript of the reference (Track A prompt / Track B optional)
    n: int = 1


class GenBatch(BaseModel):
    items: list[Gen]


def speaker_refs(train, rows=False):
    """speaker -> reference clip (shortest >= 4 s train clip, else any); rows=True returns the manifest row."""
    ref = {}
    rs = [json.loads(l) for l in open(train)]
    for r in rs:
        if "dur" not in r: r["dur"] = sf.info(r["audio"]).duration
    for r in sorted(rs, key=lambda r: r["dur"]):
        if r["dur"] >= 4 and r["speaker"] not in ref: ref[r["speaker"]] = r
    for r in rs: ref.setdefault(r["speaker"], r)   # speakers with only short clips
    return ref if rows else {s: r["audio"] for s, r in ref.items()}


class Worker:
    def __init__(self, a):
        if a.mode == "mix":
            self.mode, self.refs = "mix", speaker_refs(a.train)
            self.parts = [Worker(argparse.Namespace(**{**vars(a), "mode": "base", "model": None})),
                          Worker(argparse.Namespace(**{**vars(a), "mode": "ft"}))]
            return
        self.mode = a.mode
        if a.mode == "omni":
            from omnivoice import OmniVoice
            self.tag_w, self.tail_pad, self.tail_s = a.tag_w, a.tail_pad, a.tail_s
            self.inpaint_k_a, self.nv = a.inpaint_k_a, None
            if a.inpaint_k_a > 0:   # nvfix (verify + inpaint); builds sherpa before CUDA, owns the model
                from nvfix import NVFix
                self.nv = NVFix(a.model); self.om = self.nv.tts; self.om.load_asr_model()
            else:
                self.om = OmniVoice.from_pretrained(a.model, device_map="cuda:0", dtype=torch.float16, load_asr=True)
            import os
            from omnivoice.models.omnivoice import OmniVoiceGenerationConfig   # = gen_zs.py omni (the evaluated config)
            self.ref_sr = a.ref_sr
            self.gc = OmniVoiceGenerationConfig(guidance_scale=a.gs if a.gs is not None else float(os.environ.get("OMNI_GS", 2.0)), num_step=int(os.environ.get("OMNI_STEPS", 32)),
                                                class_temperature=float(os.environ.get("OMNI_CT", 0.0)),
                                                postprocess_output=os.environ.get("OMNI_POSTPROC", "1") == "1",
                                                denoise=os.environ.get("OMNI_DENOISE", "1") == "1",
                                                preprocess_prompt=os.environ.get("OMNI_PREPROMPT", "1") == "1")
            self.rows = speaker_refs(a.train, rows=True); self.refs = {s: r["audio"] for s, r in self.rows.items()}
            return
        from vieneu import Vieneu
        self.tts = Vieneu(mode="v3turbo", backbone_repo=(a.model if a.mode == "ft" else None) or "/mnt/vinvtts/models/VieNeu-TTS-v3-Turbo")
        self.refs = speaker_refs(a.train); self.emb = {}
        if a.mode == "ft":
            import pyarrow.parquet as pq
            t = pq.read_table(a.speakers, columns=["speaker", "speaker_embedding"]).to_pydict()
            by = {}
            for s, e in zip(t["speaker"], t["speaker_embedding"]): by.setdefault(s, []).append(np.asarray(e, np.float32))
            self.emb = {s: np.mean(v, 0) for s, v in by.items()}

    def gen(self, g):
        if self.mode == "mix":
            k = max(1, g.n // 2); a_ = self.parts[0].gen(g.model_copy(update={"n": k})); b_ = self.parts[1].gen(g.model_copy(update={"n": max(1, g.n - k)}))
            return {"sr": a_["sr"], "wavs": a_["wavs"] + b_["wavs"]}
        if g.speaker_id is None and g.ref_wav_b64 is None: raise HTTPException(400, "need speaker_id or ref_wav_b64")
        if g.ref_wav_b64 is None and g.speaker_id not in self.refs: raise HTTPException(404, f"unknown speaker_id {g.speaker_id!r}")
        tmp = None
        if g.ref_wav_b64 is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav")
            y, sr = sf.read(io.BytesIO(base64.b64decode(g.ref_wav_b64)), dtype="float32")
            sf.write(tmp.name, y.mean(1) if y.ndim > 1 else y, sr); tmp.flush()
        ref = tmp.name if tmp else self.refs[g.speaker_id]
        texts = [None] * max(1, g.n)
        if self.mode == "omni":
            return self.gen_batch([g])[0]
        if self.mode == "base":
            t = " ".join(re.sub(r"\[([a-z]+)\]", lambda m: VN[m.group(1)], g.text).split())
            wavs = self.tts.infer_batch([t] * len(texts), ref_audio=ref, apply_watermark=False)
        else:
            from prep_vieneu import to_vieneu
            if tmp: emb, _ = self.tts.engine.prepare_reference(ref, denoise=False, use_ref_codes=False)
            else: emb = self.emb.get(g.speaker_id)
            voice = {"speaker_emb": np.asarray(emb.float().cpu() if hasattr(emb, "cpu") else emb, np.float32).reshape(-1), "codes": None}
            wavs = self.tts.infer_batch([to_vieneu(g.text)] * len(texts), voice=voice, use_ref_codes=False, apply_watermark=False)
        return self._pack(wavs, self.tts.sample_rate)

    def _tail(self, t):
        """--tail-pad: text appended after a sentence-final tag so the model leaves room for the event."""
        t = " ".join(t.split())
        return f"{t} {self.tail_pad}" if self.tail_pad and re.search(r"\[[a-z]+\]$", t) else t

    def gen_batch(self, gs):
        """omni: all candidates of all requests in ONE generate call (GPU batching across requests)."""
        texts, refs, rts, tmps, k, cfg = [], [], [], [], [], []
        for g in gs:
            if g.ref_wav_b64 is not None:
                t = tempfile.NamedTemporaryFile(suffix=".wav"); tmps.append(t)
                y, sr = sf.read(io.BytesIO(base64.b64decode(g.ref_wav_b64)), dtype="float32")
                y = y.mean(1) if y.ndim > 1 else y
                if self.ref_sr and sr != self.ref_sr:   # default 0 = keep the prompt at its native rate (48 kHz prompts give higher SS)
                    import librosa; y = librosa.resample(y, orig_sr=sr, target_sr=self.ref_sr); sr = self.ref_sr
                sf.write(t.name, y, sr); t.flush(); ref, rt = t.name, g.ref_text
            else:
                if g.speaker_id not in self.refs: raise HTTPException(404, f"unknown speaker_id {g.speaker_id!r}")
                ref = self.refs[g.speaker_id]; rt = g.ref_text or " ".join(re.sub(r"\[[a-z]+\]", " ", self.rows[g.speaker_id]["text"]).split())
            if not rt: rt = self.om.transcribe(ref) if hasattr(self.om, "transcribe") else None
            tw = self.tag_w if g.tag_w is None else g.tag_w
            if self.nv is not None and (g.track or "A") == "A" and TAGRE_W.search(g.text):   # Track A: tag-CFG + inpaint k
                cfg.append((len(k), self._tail(g.text), ref, rt, max(1, g.n), tw, ("nv", g.gs))); k.append(0); continue
            if g.gs is not None and g.gs != self.gc.guidance_scale:   # per-request guidance -> own decode
                cfg.append((len(k), self._tail(g.text), ref, rt, max(1, g.n), tw, g.gs)); k.append(0); continue
            if tw > 0:   # tag-CFG (cfggen.py): one batched decode per request (3n rows)
                from cfggen import cfg_n
                cfg.append((len(k), self._tail(g.text), ref, rt, max(1, g.n), tw, None)); k.append(0); continue
            n = max(1, g.n); texts += [" ".join(self._tail(g.text).split())] * n; refs += [ref] * n; rts += [rt] * n; k.append(n)
        if not texts: wavs = []
        elif any(r is None for r in rts): wavs = [w for t, r, x in zip(texts, refs, rts) for w in self.om.generate(text=t, ref_audio=r, ref_text=x, generation_config=self.gc)]
        else: wavs = self.om.generate(text=texts, ref_audio=refs, ref_text=rts, generation_config=self.gc)
        out, i = [], 0
        for n in k: out.append(self._pack(wavs[i:i + n], 24000)); i += n
        from cfggen import cfg_n
        import copy
        for j, t, ref, rt, n, tw, gs in cfg:
            if isinstance(gs, tuple):
                self.nv.gc.guidance_scale = gs[1] if gs[1] is not None else self.gc.guidance_scale
                out[j] = self._pack(self.nv.synth_n(t, ref, rt, n, tag_w=tw, k=self.inpaint_k_a), 24000); continue
            gc = self.gc
            if gs is not None: gc = copy.copy(self.gc); gc.guidance_scale = gs
            out[j] = self._pack(cfg_n(self.om, gc, t, ref, rt, n, tw, self.tail_s), 24000)
        return out

    @staticmethod
    def _pack(wavs, sr):
        out = []
        for w in wavs:
            b = io.BytesIO(); sf.write(b, np.clip(np.asarray(w, np.float32).reshape(-1), -1, 1), sr, format="WAV", subtype="PCM_16")
            out.append(base64.b64encode(b.getvalue()).decode())
        return {"sr": sr, "wavs": out}


@app.post("/gen")
def gen(g: Gen):
    with LOCK, torch.inference_mode():
        seed([g]); return app.state.w.gen(g)


LOCK = __import__("threading").Lock()   # one generate at a time: the seed below must not be shared by concurrent calls


def seed(items):
    """Fixed seed per call (from the request texts) so the same input gives the same candidates."""
    import random, zlib
    k = zlib.crc32("|".join(f"{g.text}#{g.n}#{g.track}" for g in items).encode())
    random.seed(k); np.random.seed(k); torch.manual_seed(k)


@app.post("/gen_batch")
def gen_batch(b: GenBatch):
    with LOCK, torch.inference_mode():
        seed(b.items); return {"out": app.state.w.gen_batch(b.items)}


@app.get("/health")
def health():
    return {"status": "ok", "mode": app.state.w.mode, "speakers": len(app.state.w.refs)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["base", "ft", "mix", "omni"], default="base"); ap.add_argument("--model")
    ap.add_argument("--train", required=True, help="train manifest -> Track A reference clip per speaker")
    ap.add_argument("--speakers", help="(ft) training parquet -> mean speaker embeddings")
    ap.add_argument("--gs", type=float, default=None, help="omni guidance scale (default env OMNI_GS or 2.0; grid 4-8)")
    ap.add_argument("--ref-sr", type=int, default=0, help="resample uploaded prompts to this rate; 0 = native rate (48k B prompts)")
    ap.add_argument("--tag-w", type=float, default=3.0, help="omni tag-CFG weight; 0 = plain generation (fallback)")
    ap.add_argument("--inpaint-k-a", type=int, default=0, help="Track A: nvfix verify+inpaint rounds (2 = served config)")
    ap.add_argument("--tail-pad", default="", help="text appended after a sentence-final tag (e.g. ',')")
    ap.add_argument("--tail-s", type=float, default=0.0, help="tag-CFG: extra target seconds when the text ends with a tag")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8001)
    a = ap.parse_args()
    app.state.w = Worker(a)
    cpulimit.cap_gpu_memory()
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, workers=1)
