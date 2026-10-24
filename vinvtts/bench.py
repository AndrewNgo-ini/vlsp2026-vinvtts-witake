"""Score generated wavs the way the VLSP judge does (proxies): WER (ASR, tags removed), SS (ECAPA cosine to the
speaker reference), pMOS (DNSMOS OVRL), NV (tag-event detection, see tags.py). Input JSONL rows:
{"gen": wav, "text": text-with-tags, "ref": reference wav of the target speaker}. Prints per-system means.

    uv run bench.py runs/gt.jsonl runs/f5vi.jsonl [--asr openai/whisper-large-v3]
"""
import cpulimit  # noqa: F401  (must precede torch/onnxruntime use)
import argparse, json
import jiwer, numpy as np, torch
from judge import load16k, normalize, SR


class Bench:
    def __init__(self, asr, device="cuda"):
        from speechbrain.inference.speaker import EncoderClassifier
        if asr == "zipformer":  # hynt 30M RNNT (VLSP2025 winner), closest public proxy to the judge's Zipformer
            import sherpa_onnx
            from huggingface_hub import snapshot_download
            d = snapshot_download("csukuangfj2/sherpa-onnx-zipformer-vi-30M-2026-02-09")
            rec = sherpa_onnx.OfflineRecognizer.from_transducer(encoder=f"{d}/encoder.onnx", decoder=f"{d}/decoder.onnx",
                joiner=f"{d}/joiner.onnx", tokens=f"{d}/tokens.txt", num_threads=min(2, int(__import__("os").environ.get("OMP_NUM_THREADS", "2"))), sample_rate=SR, feature_dim=80)
            self.rec = rec   # reused by nvdet (word gap times) so sherpa is built once, before any CUDA model
            def run(w):
                st = rec.create_stream(); st.accept_waveform(SR, w.numpy()); rec.decode_stream(st)
                return {"text": st.result.text.lower()}
            self.asr = lambda xs, **_: [run(x) for x in xs]
        else:
            from transformers import pipeline
            whisper = pipeline("automatic-speech-recognition", model=asr, device=device, torch_dtype=torch.float16)
            self.asr = lambda xs, **kw: whisper([{"raw": w.numpy(), "sampling_rate": SR} for w in xs], **kw,
                                                generate_kwargs={"language": "vi", "task": "transcribe"})
        self.spk = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
        self.dev = device

    @torch.no_grad()
    def emb(self, wav):
        return self.spk.encode_batch(wav[None].to(self.dev)).squeeze()

    def score(self, rows, bs=16):
        import dnsmos_gpu  # noqa: F401  (DNSMOS nets on CUDA via onnx2torch; exact parity with onnxruntime, 09-27)
        from speechmos import dnsmos
        wavs = [load16k(r["gen"]).clamp(-1, 1) for r in rows]  # VieNeu peaks can exceed 1; DNSMOS rejects that
        from concurrent.futures import ThreadPoolExecutor   # DNSMOS (GPU, 1 CPU for the mel) overlaps the CPU Zipformer (2 threads) -> <= 3 CPUs
        with ThreadPoolExecutor(int(__import__("os").environ.get("BENCH_WORKERS", "1"))) as ex:   # BENCH_WORKERS threads (default 1)
            mos = ex.map(lambda w: float(dnsmos.run(w.numpy(), sr=SR)["ovrl_mos"]), wavs)
            hyps = self.asr(wavs, batch_size=bs)
            mos = list(mos)
        out, ref_emb = [], {}   # rows share one reference per speaker: embed each reference once
        for r, w, h, pm in zip(rows, wavs, hyps, mos):
            ref, hyp = normalize(r["text"]), normalize(h["text"])
            s = {"wer": min(1.0, jiwer.wer(ref, hyp)), "cer": min(1.0, jiwer.cer(ref, hyp)),  # judge uses 1-WER in [0,1]
                 "pmos": pm, "dur": len(w) / SR}
            if r.get("ref"):
                if r["ref"] not in ref_emb: ref_emb[r["ref"]] = self.emb(load16k(r["ref"]))
                s["ss"] = float(torch.nn.functional.cosine_similarity(self.emb(w), ref_emb[r["ref"]], dim=0))
            out.append({**r, **s, "hyp": hyp})
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--asr", default="zipformer", help="zipformer | any HF whisper id")
    a = ap.parse_args()
    b = Bench(a.asr)
    for m in a.manifests:
        rows = [json.loads(l) for l in open(m)]
        res = b.score(rows)
        with open(m.replace(".jsonl", ".scored.jsonl"), "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in res)
        mean = {k: round(float(np.mean([r[k] for r in res if k in r])), 4) for k in ["wer", "cer", "ss", "pmos", "dur"] if any(k in r for r in res)}
        mean["wer_med"] = round(float(np.median([r["wer"] for r in res])), 4)
        print(m, len(res), json.dumps(mean))
