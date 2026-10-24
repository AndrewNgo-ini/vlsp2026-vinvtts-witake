"""Tag-aware ASR: fine-tune Whisper to transcribe Vietnamese with inline [laughter] [breathing] [sniff]
[throatclearing]. Used as the NVPA proxy (tags.py) and as the best-of-N selector.

    uv run tagasr.py train --train train.jsonl --dev dev.jsonl --out /mnt/vinvtts/runs/tagasr
    uv run tagasr.py train --v2 --epochs 8 --extra synth.jsonl ... --out runs/tagasr_v2    # balanced + perturbed
    uv run tagasr.py eval  --ckpt /mnt/vinvtts/runs/tagasr --dev dev.jsonl          # tag P/R/F1 + WER on dev truth
    uv run tagasr.py score --ckpt /mnt/vinvtts/runs/tagasr runs/zs/*.jsonl           # NVPA proxy on generated `gen` wavs
"""
import cpulimit  # noqa: F401  (must precede torch/onnxruntime use)
import argparse, json, random
import jiwer, numpy as np, torch
from judge import load16k, normalize
from tags import nv_score

TAGS = ["[laughter]", "[breathing]", "[sniff]", "[throatclearing]"]


def load(model_id):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    p = WhisperProcessor.from_pretrained(model_id)
    m = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda()
    if p.tokenizer.add_tokens(TAGS):  # atomic tag tokens; no-op when loading our own checkpoint
        m.resize_token_embeddings(len(p.tokenizer))
    m.generation_config.forced_decoder_ids = None
    return p, m


# post-hoc logit adjustment for the long-tailed tag tokens (Menon et al., ICLR 2021, arXiv 2007.07314): logit_t +=
# tau * log(pi_breathing / pi_t) with pi = train tag frequencies. tau = 0 -> unchanged. The trained tag-ASRs collapse to
# [breathing] (row S2: teacher-forced P(breathing) .35 > P(laughter) .12 at true laughter positions).
TRAIN_FREQ = {"[laughter]": 681, "[breathing]": 3733, "[sniff]": 117, "[throatclearing]": 115}
LA_TAU = float(__import__("os").environ.get("TAGASR_LA_TAU", "0"))


def _la_processor(p):
    import math
    from transformers import LogitsProcessor, LogitsProcessorList
    ids = {p.tokenizer.convert_tokens_to_ids(t): math.log(TRAIN_FREQ["[breathing]"] / f) for t, f in TRAIN_FREQ.items()}
    class LA(LogitsProcessor):
        def __call__(self, input_ids, scores):
            for i, b in ids.items(): scores[:, i] += LA_TAU * b
            return scores
    return LogitsProcessorList([LA()])


def transcribe(p, m, wavs, bs=16):
    out = []
    kw = {"logits_processor": _la_processor(p)} if LA_TAU else {}
    for i in range(0, len(wavs), bs):
        f = p.feature_extractor([w.numpy() for w in wavs[i:i + bs]], sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            ids = m.generate(f.cuda().to(torch.bfloat16), language="vi", task="transcribe", max_new_tokens=440, **kw)
        out += [" ".join(t.split()) for t in p.batch_decode(ids, skip_special_tokens=True)]
    return out


def sqrt_weights(rows):
    """Per-utt sampling weight = max over its tags of sqrt(f_max / f_tag) (square-root category sampling, NVVSpeech
    Track-1 system, arXiv 2609.23462)."""
    from collections import Counter
    f = Counter(t for r in rows for t in TAGS for _ in range(r["text"].count(t)))
    fm = max(f.values())
    return [max([(fm / f[t]) ** 0.5 for t in TAGS if t in r["text"]] or [1.0]) for r in rows]


def perturb(w):
    """Speed (0.9/1.0/1.1, resample) + additive noise at 10-30 dB SNR (Source-Adaptive NVV-ASR, arXiv 2609.09929)."""
    import torchaudio
    sp = random.choice([0.9, 1.0, 1.1])
    if sp != 1.0 and len(w) / sp < 29.5 * 16000:
        w = torchaudio.functional.resample(w, int(16000 * sp), 16000)
    if random.random() < 0.5:
        snr = random.uniform(10, 30); n = torch.randn_like(w)
        w = w + n * (w.pow(2).mean().sqrt() / (10 ** (snr / 20)) / n.pow(2).mean().sqrt())
    return w.clamp(-1, 1)


def train(a):
    p, m = load(a.base)
    rows = [json.loads(l) for l in open(a.train)]
    for x in a.extra or []: rows += [json.loads(l) for l in open(x)]
    if a.v2:   # sqrt category sampling + on-the-fly perturbation, features on GPU
        wts = sqrt_weights(rows); wavs = [load16k(r["audio"]) for r in rows]
        labels = [p.tokenizer(" " + r["text"]).input_ids for r in rows]
        steps_per_ep = len(rows) // a.bs
        def batches():
            for _ in range(steps_per_ep):
                idx = random.choices(range(len(rows)), wts, k=a.bs)
                ws = [perturb(wavs[i]).numpy() for i in idx]
                x = p.feature_extractor(ws, sampling_rate=16000, return_tensors="pt", device="cuda").input_features
                yield x, [labels[i] for i in idx]
    else:
        # oversample utts with rare tags x4 (sniff/throatclearing ~115 each)
        rows += [r for r in rows if "[sniff]" in r["text"] or "[throatclearing]" in r["text"]] * 3
        data = [(p.feature_extractor(load16k(r["audio"]).numpy(), sampling_rate=16000).input_features[0],
                 p.tokenizer(" " + r["text"]).input_ids) for r in rows]
        steps_per_ep = len(data) // a.bs
        def batches():
            random.shuffle(data)
            for i in range(0, len(data) - a.bs + 1, a.bs):
                b = data[i:i + a.bs]; yield torch.tensor(np.array([f for f, _ in b])), [t for _, t in b]
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * steps_per_ep
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    m.train(); step = 0
    for ep in range(a.epochs):
        for x, ys in batches():
            if step >= steps: break
            x = x.cuda().to(torch.bfloat16)
            L = max(len(t) for t in ys)
            y = torch.tensor([t + [-100] * (L - len(t)) for t in ys]).cuda()
            loss = m(input_features=x, labels=y).loss  # labels already start with <|sot|><|vi|>... from tokenizer prefix
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step % 50 == 0: print(f"ep {ep} step {step}/{steps} loss {loss.item():.4f}", flush=True)
    m.save_pretrained(a.out); p.save_pretrained(a.out)
    evaluate(argparse.Namespace(ckpt=a.out, dev=a.dev))


def evaluate(a):
    p, m = load(a.ckpt); m.eval()
    rows = [json.loads(l) for l in open(a.dev)]
    hyps = transcribe(p, m, [load16k(r["audio"]) for r in rows])
    s = nv_score([r["text"] for r in rows], hyps)
    s["wer"] = jiwer.wer([normalize(r["text"]) for r in rows], [normalize(h) for h in hyps])
    for t in TAGS:
        sub = [(r["text"], h) for r, h in zip(rows, hyps) if t in r["text"]]
        if sub: s["rec_" + t[1:-1]] = nv_score(*zip(*sub))["nvpa"]
    print("TAGASR_DEV", json.dumps({k: round(v, 4) for k, v in s.items()}))
    for r, h in list(zip(rows, hyps))[:3]:
        print("REF", r["text"][:200]); print("HYP", h[:200])


def score(a):
    """NVPA proxy per system: tag-ASR on each row's `gen` wav vs its tagged `text`; per-tag recall too."""
    p, m = load(a.ckpt); m.eval()
    for man in a.manifests:
        rows = [json.loads(l) for l in open(man)]
        hyps = transcribe(p, m, [load16k(r["gen"]) for r in rows])
        s = nv_score([r["text"] for r in rows], hyps)
        for t in TAGS:
            sub = [(r["text"], h) for r, h in zip(rows, hyps) if t in r["text"]]
            if sub: s["rec_" + t[1:-1]] = nv_score(*zip(*sub))["nvpa"]
        with open(man.replace(".jsonl", a.suffix), "w") as f:
            f.writelines(json.dumps({"id": r["id"], "text": r["text"], "tag_hyp": h}, ensure_ascii=False) + "\n" for r, h in zip(rows, hyps))
        print("NV", man, json.dumps({k: round(v, 4) for k, v in s.items()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "score"]); ap.add_argument("manifests", nargs="*")
    ap.add_argument("--base", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--train"); ap.add_argument("--dev"); ap.add_argument("--out"); ap.add_argument("--ckpt")
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--suffix", default=".tags.jsonl", help="score: output suffix (e.g. .tags_v2s.jsonl for a second tag-ASR)")
    ap.add_argument("--v2", action="store_true", help="sqrt category sampling + speed/noise perturbation")
    ap.add_argument("--extra", nargs="*", help="extra training manifests (e.g. splice synth)")
    a = ap.parse_args()
    {"train": train, "eval": evaluate, "score": score}[a.cmd](a)
