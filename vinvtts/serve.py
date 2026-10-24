"""ViNV-TTS HTTP API (the batch Docker entrypoint is run_batch.py).

POST /synthesize  {"text": "...", "speaker_id": "spk1"}                      -> audio/wav (Track A)
POST /synthesize  {"text": "...", "reference_wav_b64": "<base64 wav>"}       -> audio/wav (Track B)
GET  /health

Production (2026-09-30, prod.py: best-of-8 omft_tdl4000 -> official-weight selection -> NV splicing, decide=any):
    OMNI=/data/vinvtts/src/omnivoice/.venv/bin/python; $OMNI tts_worker.py --mode omni --model /data/vinvtts/runs/omni_ft_tdl/merged-4000 --train /data/vinvtts/data/vinv-tts/train.jsonl --port 8011 &
    /data/vinvtts/src/qwen3asr/.venv/bin/python judge_worker.py --kind qasr --port 8012 &
    /data/vinvtts/src/psed/.venv/bin/python judge_worker.py --kind psed --port 8013 &
    uv run serve.py --prod --worker http://127.0.0.1:8011 --port 8000
  request: {"text", "speaker_id"? (Track A), "reference_wav_b64"? (prompt clip, any sr; required for B), "reference_text"?,
            "track"? ("A"/"B"; default A if speaker_id else B)}. Requests are micro-batched (--batch, --wait).

    uv run serve.py --port 8000                                   # stub model (no GPU)
    /mnt/vinvtts/src/vieneu/.venv/bin/python tts_worker.py --mode base --train .../train.jsonl --port 8001 &
    uv run serve.py --worker http://127.0.0.1:8001 --n 8 --port 8000   # real: best-of-8 over the worker
    uv run serve.py --smoke     # starts the server in a subprocess, hits /health and /synthesize, stops it
"""
import argparse, asyncio, base64, io, re, socket, subprocess, sys, time
from contextlib import asynccontextmanager

import numpy as np, soundfile as sf
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

TAGS = ["laughter", "breathing", "sniff", "throatclearing"]
BAD_TAG = re.compile(r"\[(?!(" + "|".join(TAGS) + r")\])[^\]]*\]")
SR = 24000
MAX_CHARS = 1000
MAX_CONCURRENCY = 2  # one semaphore (non-prod path)


class StubSynth:
    """Stand-in: returns silence proportional to text length. Replace with F5-TTS (see README)."""
    sr = SR

    def synth(self, text, speaker_id=None, ref_wav=None):
        # TODO(model): load fine-tuned F5-TTS ckpt + extended vocab; map [tag] -> PUA codepoint (prep_f5.TAG2CP)
        # before tokenizing; Track A: speaker_id -> stored reference wav+text; Track B: use ref_wav directly.
        return np.zeros(int(SR * max(0.5, 0.06 * len(text))), dtype=np.float32)


class BestOfN:
    """tts_worker.py generates N sampled candidates; each is scored with the judge proxies and the best is returned
    (best-of-N, arXiv 2609.14231 / Llasa 2502.04128) under the objective part of Score A (bestof.utility):
    tag-ASR NV hits at the tagged positions, Zipformer WER, DNSMOS, ECAPA SS to the voice reference."""

    def __init__(self, worker, n, tagasr_ckpt, train, nvdet_dir=None, worker_b=None):
        import httpx, torch, tagasr
        from bench import Bench
        from tts_worker import speaker_refs
        # worker may be "url1,url2": candidates are split across engines (e.g. VieNeu-DPO + OmniVoice-DPO; the pooled
        # 8+8 best-of-N is the best evalF system, objV .476) and the selector picks across both.
        self.https = [httpx.Client(base_url=u, timeout=300) for u in worker.split(",")]; self.http = self.https[0]; self.n = n
        # Track B (uploaded reference) may use its own engine set: validated best = VieNeu base+ft (mix) + OmniVoice
        self.https_b = [httpx.Client(base_url=u, timeout=300) for u in worker_b.split(",")] if worker_b else self.https
        self.b = Bench("zipformer"); self.p, self.m = tagasr.load(tagasr_ckpt); self.m.eval(); self.tagasr = tagasr
        self.refs = speaker_refs(train); self.torch = torch
        self.nvdet = None
        if nvdet_dir:   # second NV scorer: selection averages tag-ASR and nvdet (bestof.utility)
            import pickle, nvdet
            self.nvdet = (nvdet, pickle.load(open(f"{nvdet_dir}/clf.pkl", "rb")), nvdet.Embedder())
        for h in self.https + self.https_b: h.get("/health").raise_for_status()
        self.sr = 48000   # workers must be up

    def synth(self, text, speaker_id=None, ref_wav=None):
        import torchaudio
        from judge import load16k, normalize, SR as SR16
        from speechmos import dnsmos
        from bestof import utility, W, W_B
        from tags import nv_utt
        import jiwer
        body = {"text": text, "n": self.n, "speaker_id": speaker_id}
        if ref_wav is not None:
            y, sr = ref_wav; b = io.BytesIO(); sf.write(b, y, sr, format="WAV"); body["ref_wav_b64"] = base64.b64encode(b.getvalue()).decode()
            ref16 = torchaudio.functional.resample(self.torch.from_numpy(np.asarray(y, np.float32)), sr, SR16)
        else:
            if speaker_id not in self.refs: raise HTTPException(404, f"unknown speaker_id {speaker_id!r}")
            ref16 = load16k(self.refs[speaker_id])
        from concurrent.futures import ThreadPoolExecutor
        hs = self.https_b if ref_wav is not None else self.https; k = len(hs)
        def ask(i):
            r = hs[i].post("/gen", json={**body, "n": max(1, self.n // k)})
            if r.status_code != 200: raise HTTPException(r.status_code, r.text)
            j = r.json()
            return [(sf.read(io.BytesIO(base64.b64decode(w)), dtype="float32")[0], j["sr"]) for w in j["wavs"]]
        with ThreadPoolExecutor(k) as ex: got = [c for part in ex.map(ask, range(k)) for c in part]
        self.sr = 24000 if any(sr != got[0][1] for _, sr in got) else got[0][1]   # mixed engines -> common 24 kHz
        import torchaudio as _ta
        wavs = [w if sr == self.sr else _ta.functional.resample(self.torch.from_numpy(w), sr, self.sr).numpy() for w, sr in got]
        if len(wavs) == 1: return wavs[0]
        w16 = [torchaudio.functional.resample(self.torch.from_numpy(w), self.sr, SR16).clamp(-1, 1) for w in wavs]
        hyps = self.b.asr(w16); tag_h = self.tagasr.transcribe(self.p, self.m, w16)
        e_ref = self.b.emb(ref16); best, bu = 0, -1e9
        for i, (w, h, th) in enumerate(zip(w16, hyps, tag_h)):
            hit, _, n, npred = nv_utt(text, th)
            c = {"nv_hit": hit, "nv_n": n, "nv_p": npred, "wer": min(1.0, jiwer.wer(normalize(text), normalize(h["text"]) or "-")),
                 **({"nvdet_hit": self._nvdet_hits(w, text)} if self.nvdet and n else {}),
                 "pmos": float(dnsmos.run(w.numpy(), sr=SR16)["ovrl_mos"]),
                 "ss": float(self.torch.nn.functional.cosine_similarity(self.b.emb(w), e_ref, dim=0))}
            u = utility(c, W_B if ref_wav is not None else W)   # Track B weights SS 0.2 / NV 0.25
            if u > bu: best, bu = i, u
        return wavs[best]

    def _nvdet_hits(self, w16, text):
        nd, clf, emb = self.nvdet
        gp, tags = nd.gen_gaps(self.b.rec, w16, text)   # gap 0 included (Sep-28 first_gap=1 bug fixed 09-30)
        segs = [nd.seg(w16, *gp[g]) if g in gp else w16[-int(nd.MIN_SEG * 16000):] for g, _ in tags]
        pred = clf.predict(emb(segs)) if segs else []
        return int(sum(nd.CLS[p] == t for p, (_, t) in zip(pred, tags)))


class Req(BaseModel):
    text: str
    speaker_id: str | None = None
    reference_wav_b64: str | None = None
    reference_text: str | None = None
    track: str | None = None


DIG = "không một hai ba bốn năm sáu bảy tám chín".split()


def norm(t):   # = mkpublic.norm (lowercase, digits spelled, tags spaced) + NFC (NFD Vietnamese input)
    import unicodedata
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"\[\s*(laughter|breathing|sniff|throatclearing)\s*\]", r"[\1]", t, flags=re.I)
    t = re.sub(r"\d", lambda m: f" {DIG[int(m.group())]} ", t)
    t = re.sub(r"(\[[a-z]+\])", r" \1 ", t.lower())
    return " ".join(t.split())


class Batcher:
    """Collect concurrent requests into batches of <= size (waiting <= wait s); `lanes` batches in flight so one batch's
    generation overlaps another's scoring."""
    def __init__(self, prod, size=8, wait=0.5, lanes=2):
        self.prod, self.size, self.wait, self.q = prod, size, wait, asyncio.Queue()
        self.tasks = [asyncio.create_task(self.loop()) for _ in range(lanes)]

    async def loop(self):
        while True:
            items = [await self.q.get()]; t_end = asyncio.get_event_loop().time() + self.wait
            while len(items) < self.size:
                try: items.append(await asyncio.wait_for(self.q.get(), max(0, t_end - asyncio.get_event_loop().time())))
                except asyncio.TimeoutError: break
            try:
                res = await asyncio.to_thread(self.prod, [r for r, _ in items])
                for (_, f), x in zip(items, res): f.set_result(x)
            except Exception as e:
                for _, f in items:
                    if not f.done(): f.set_exception(e)

    async def __call__(self, req):
        f = asyncio.get_event_loop().create_future(); await self.q.put((req, f)); return await f


@asynccontextmanager
async def lifespan(app):
    a = app.state.args
    if a.prod:
        from prod import Production
        from tts_worker import speaker_refs
        app.state.prod = Production(a.worker, a.qasr, a.psed, n=a.n, per_engine=a.per_engine, select=a.select); app.state.model = app.state.prod
        app.state.refs = speaker_refs(a.train, rows=True)
        app.state.batcher = Batcher(app.state.prod, a.batch, a.wait, a.lanes); app.state.model.sr = 24000
        yield; return
    app.state.model = BestOfN(a.worker, a.n, a.tagasr, a.train, a.nvdet, a.worker_b) if a.worker else StubSynth()  # warm models, loaded once
    app.state.sem = asyncio.Semaphore(MAX_CONCURRENCY)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": type(app.state.model).__name__}


@app.post("/synthesize")
async def synthesize(req: Req):
    text = req.text.strip()
    if not text or len(text) > MAX_CHARS:
        raise HTTPException(400, f"text must be 1..{MAX_CHARS} chars")
    if BAD_TAG.search(re.sub(r"\[\s*([a-zA-Z]+)\s*\]", lambda m: f"[{m.group(1).lower()}]", text)):
        raise HTTPException(400, f"unknown tag; allowed: {TAGS}")
    if not req.speaker_id and not req.reference_wav_b64:
        raise HTTPException(400, "need speaker_id or reference_wav_b64")
    ref = None
    if req.reference_wav_b64:
        try:
            y, sr = sf.read(io.BytesIO(base64.b64decode(req.reference_wav_b64)), dtype="float32")
            ref = (y.mean(1) if y.ndim > 1 else y, sr)
        except Exception as e:
            raise HTTPException(400, f"reference_wav_b64 is not a decodable wav: {e}")
    if getattr(app.state, "batcher", None):
        if ref is None:   # Track A without a prompt: fall back to the speaker's train clip
            row = app.state.refs.get(req.speaker_id)
            if row is None: raise HTTPException(404, f"unknown speaker_id {req.speaker_id!r}")
            y, sr = sf.read(row["audio"], dtype="float32"); ref = (y.mean(1) if y.ndim > 1 else y, sr)
            rt = req.reference_text or " ".join(re.sub(r"\[[a-z]+\]", " ", row["text"]).split())
        else: rt = req.reference_text
        rt = rt if rt and rt.strip() else None
        y, sr = ref
        if len(y) > 20 * sr:   # long prompt: keep the first 15 s; its transcript no longer matches -> model ASR
            y = y[: 15 * sr]; rt = None
        if len(y) < 0.5 * sr: raise HTTPException(400, "reference audio shorter than 0.5 s")
        pk = float(np.abs(y).max()) if len(y) else 0.0
        if pk < 1e-4: raise HTTPException(400, "reference audio is silent")
        ref = (y.astype(np.float32), sr)
        track = (req.track or ("A" if req.speaker_id else "B")).upper()
        t0 = time.time()
        wav, info = await app.state.batcher({"text": norm(text), "ref": ref, "ref_text": norm(rt) if rt else None, "track": track,
                                             "speaker": req.speaker_id, "key": f"{req.speaker_id}|{text}"})
        buf = io.BytesIO(); sf.write(buf, wav, 24000, format="WAV", subtype="PCM_16")
        import json as _j
        return Response(buf.getvalue(), media_type="audio/wav", headers={"X-Latency": f"{time.time() - t0:.2f}",
                        "X-Info": base64.b64encode(_j.dumps(info, ensure_ascii=False, default=str).encode()).decode()})
    async with app.state.sem:
        wav = await asyncio.to_thread(app.state.model.synth, text, req.speaker_id, ref)
    buf = io.BytesIO(); sf.write(buf, wav, getattr(app.state.model, "sr", SR), format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav")


def smoke():
    import httpx
    with socket.socket() as s:  # free port; fixed ports collide with whatever else is running locally
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    p = subprocess.Popen([sys.executable, __file__, "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(50):
            try:
                h = httpx.get(base + "/health"); break
            except httpx.ConnectError:
                time.sleep(0.2)
        else:
            raise RuntimeError("server did not start: " + p.stderr.read().decode()[-2000:])
        print("GET /health", h.status_code, h.json()); assert h.status_code == 200
        r = httpx.post(base + "/synthesize", json={"text": "Xin chào [laughter] các bạn", "speaker_id": "spk1"})
        wav, sr = sf.read(io.BytesIO(r.content)); print("POST /synthesize A", r.status_code, r.headers["content-type"], len(wav) / sr, "s")
        assert r.status_code == 200 and sr == SR and len(wav) > 0
        buf = io.BytesIO(); sf.write(buf, np.zeros(SR), SR, format="WAV")
        r = httpx.post(base + "/synthesize", json={"text": "Xin chào", "reference_wav_b64": base64.b64encode(buf.getvalue()).decode()})
        print("POST /synthesize B", r.status_code); assert r.status_code == 200
        r = httpx.post(base + "/synthesize", json={"text": "Xin chào [cough]", "speaker_id": "x"})
        print("POST bad tag", r.status_code, r.json()); assert r.status_code == 400
        r = httpx.post(base + "/synthesize", json={"text": "Xin chào"})
        print("POST no speaker", r.status_code); assert r.status_code == 400
        print("SMOKE OK")
    finally:
        p.terminate(); p.wait(5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--worker", help="tts_worker.py URL, e.g. http://127.0.0.1:8001; omit for the stub")
    ap.add_argument("--worker-b", help="Track-B worker URL(s) (reference-wav requests); default = --worker")
    ap.add_argument("--n", type=int, default=8, help="best-of-N candidates per request")
    ap.add_argument("--tagasr", default="/mnt/vinvtts/runs/tagasr")
    ap.add_argument("--nvdet", default="/mnt/vinvtts/runs/nvdet", help="'' to select on the tag-ASR alone")
    ap.add_argument("--train", default="/mnt/vinvtts/data/vinv-tts/train.jsonl", help="Track A reference clip per speaker")
    ap.add_argument("--prod", action="store_true", help="production recipe (prod.py); --worker = omni worker URL(s)")
    ap.add_argument("--qasr", default="http://127.0.0.1:8012"); ap.add_argument("--psed", default="http://127.0.0.1:8013")
    ap.add_argument("--per-engine", action="store_true", help="--n candidates from EACH --worker (else split)")
    ap.add_argument("--select", default="nvdet_v3", choices=["nvdet_v3", "v3_cj"])
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--wait", type=float, default=0.5); ap.add_argument("--lanes", type=int, default=2)
    a = ap.parse_args(); app.state.args = a
    if a.prod: a.train = a.train.replace("/mnt/vinvtts/", "/data/vinvtts/")
    if a.smoke:
        smoke()
    else:
        import uvicorn
        uvicorn.run(app, host=a.host, port=a.port, workers=1)
