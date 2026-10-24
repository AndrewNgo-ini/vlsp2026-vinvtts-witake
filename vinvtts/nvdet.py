"""Acoustic NVPA proxy. The tag-ASR (tagasr.py) mostly predicts tags from the text prior -- base VieNeu, which cannot
make a sniff, scored sniff recall 0.62 -- so this scorer only looks at the audio where each tag should be:

  1. Zipformer word timestamps on the generated wav, Levenshtein-aligned to the reference words (align.py), give the
     time span of every word gap.
  2. Each gap segment (+-PAD) is embedded with AST (Gong et al. 2021, AudioSet; its ontology has Laughter, Breathing,
     Sniff, Throat clearing) = pooled embedding + AudioSet logits.
  3. A logistic-regression head, trained on the real event bank (align.py: tagged gaps -> tag, untagged gaps -> none),
     labels the segment. Tag realised = predicted label == tag. Untagged gaps predicted as an event = false positive.

    uv run nvdet.py train --events runs/events --train train.jsonl --dev-events runs/events_dev --dev dev.jsonl --out runs/nvdet
    uv run nvdet.py score --model runs/nvdet runs/eval/gt.jsonl runs/zs/ft1.jsonl ...
"""
import cpulimit  # noqa: F401
import argparse, json, pickle, random
from collections import Counter
from pathlib import Path
import jiwer, numpy as np, torch
from judge import load16k, SR
from tags import split

CLS = ["none", "laughter", "breathing", "sniff", "throatclearing"]
PAD = 0.08       # s of context on each side of the gap
MIN_SEG = 0.25   # s; shorter gaps are widened symmetrically (a missing event -> mostly the two word edges)
AST = "MIT/ast-finetuned-audioset-10-10-0.4593"


class Embedder:
    def __init__(self):
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        self.fe = ASTFeatureExtractor.from_pretrained(AST)
        self.m = ASTForAudioClassification.from_pretrained(AST, torch_dtype=torch.float16).cuda().eval()

    @torch.no_grad()
    def __call__(self, segs, bs=64):
        out = []
        for i in range(0, len(segs), bs):
            x = self.fe([s.numpy() for s in segs[i:i + bs]], sampling_rate=SR, return_tensors="pt").input_values.cuda().half()
            o = self.m(x, output_hidden_states=True)
            h = o.hidden_states[-1][:, :2].mean(1)   # cls + distillation token, as AST's own pooler
            out.append(torch.cat([h, o.logits], 1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768 + 527), np.float32)


def seg(wav, t0, t1):
    if t1 - t0 < MIN_SEG:
        c = (t0 + t1) / 2; t0, t1 = c - MIN_SEG / 2, c + MIN_SEG / 2
    a, b = max(0, int((t0 - PAD) * SR)), min(len(wav), int((t1 + PAD) * SR))
    a = min(a, max(0, len(wav) - 800))
    return wav[a:max(b, a + 800)]   # >= 50 ms: AST's kaldi fbank needs >= one 25 ms window


def bank(events_dir, manifest, max_neg_per_utt=3, seed=0):
    """Real segments: every located tagged event + up to N untagged aligned gaps per utt ('none')."""
    rnd = random.Random(seed)
    ev = [json.loads(l) for l in open(f"{events_dir}/events.jsonl")]
    gaps = json.load(open(f"{events_dir}/gaps.json"))
    tagged = Counter((e["audio"], e["gap"]) for e in ev)
    items = [(e["audio"], e["t0"], e["t1"], e["tag"]) for e in ev]
    texts = {r["audio"]: r["text"] for r in map(json.loads, open(manifest))}
    for a, gp in gaps.items():
        tg = {g for g, _ in split(texts[a])[1]}   # tagged gaps whose event failed to locate are not 'none' either
        free = [(int(g), t) for g, t in gp.items() if (a, int(g)) not in tagged and int(g) not in tg and 0 < int(g)]
        for g, (t0, t1) in rnd.sample(free, min(max_neg_per_utt, len(free))):
            items.append((a, t0, t1, "none"))
    return items


def vs_items(root, cap=1500, win=0.6, seed=0):
    """VocalSound (Gong et al., ICASSP 2022, CC BY-SA 4.0) clips as extra positives: <spk>_<i>_<class>.wav, one vocal
    sound per clip; keep laughter / sniff / throatclearing, take the max-RMS `win` s window of each clip."""
    import glob, random as _r
    keep = {"laughter": "laughter", "sniff": "sniff", "throatclearing": "throatclearing"}
    by = {k: [] for k in keep}
    for p in glob.glob(f"{root}/**/*.wav", recursive=True):
        c = Path(p).stem.split("_")[-1].lower()
        if c in keep: by[c].append(p)
    rnd = _r.Random(seed); out = []
    for c, ps in by.items():
        for p in rnd.sample(ps, min(cap, len(ps))):
            y = load16k(p).numpy(); n = int(win * SR)
            if len(y) < int(0.3 * SR): continue   # too short for a meaningful segment (and fbank's 25 ms window)
            if len(y) <= n: t0 = 0.0
            else:
                e = np.convolve(y ** 2, np.ones(n), "valid")[::160]; t0 = float(np.argmax(e) * 160 / SR)
            out.append((p, t0 + PAD, min(t0 + win, len(y) / SR) - PAD, keep[c]))
    print("vocalsound", {c: min(cap, len(ps)) for c, ps in by.items()}, flush=True)
    return out


def featurize(emb, items):
    cache, segs = {}, []
    for a, t0, t1, _ in items:
        if a not in cache:
            if len(cache) > 64: cache.clear()
            cache[a] = load16k(a)
        segs.append(seg(cache[a], t0, t1))
    return emb(segs), np.array([CLS.index(y) for *_, y in items])


def train(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    emb = Embedder()
    items = bank(a.events, a.train) + (vs_items(a.vocalsound) if a.vocalsound else [])
    Xtr, ytr = featurize(emb, items); print("train", Counter(CLS[i] for i in ytr), flush=True)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=a.C, max_iter=3000, class_weight="balanced"))
    clf.fit(Xtr, ytr)
    if a.dev_events:
        Xd, yd = featurize(emb, bank(a.dev_events, a.dev, seed=1))
        print(classification_report(yd, clf.predict(Xd), labels=range(len(CLS)), target_names=CLS, digits=3))
    Path(a.out).mkdir(parents=True, exist_ok=True); pickle.dump(clf, open(f"{a.out}/clf.pkl", "wb"))


def gen_gaps(rec, wav, text, first_gap=0):
    """ref word gaps of `text` -> (t0, t1) in the generated wav; gaps whose neighbours did not align are interpolated."""
    from align import words_with_times
    hw = words_with_times(rec, wav); rw, tags = split(text); dur = len(wav) / SR
    m = {}
    if hw and rw:
        for c in jiwer.process_words(" ".join(rw), " ".join(w for w, _, _ in hw)).alignments[0]:
            if c.type in ("equal", "substitute"):
                for k in range(c.ref_end_idx - c.ref_start_idx):
                    if c.hyp_start_idx + k < len(hw): m[c.ref_start_idx + k] = c.hyp_start_idx + k
    def span(g):
        left = next((m[i] for i in range(g - 1, -1, -1) if i in m), None)
        right = next((m[i] for i in range(g, len(rw)) if i in m), None)
        t0 = hw[left][2] if left is not None else 0.0
        t1 = hw[right][1] if right is not None else dur
        return t0, max(t1, t0)
    # gap 0 (tag before the first word) included: until 09-26 it was missing, so start-of-utterance tags were scored on
    # the LAST second of the audio (fallback below) -> the "start position" weakness in eda_errors was partly a scorer bug
    return {g: span(g) for g in range(first_gap, len(rw) + 1)}, tags   # first_gap=1 = old behaviour (frozen Sep-28 serve.py)


def score(a):
    from align import recognizer
    rec = recognizer()   # sherpa-onnx segfaults if created after torch/transformers CUDA init -> build it first
    clf = pickle.load(open(f"{a.model}/clf.pkl", "rb")); emb = Embedder()
    for man in a.manifests:
        rows = [json.loads(l) for l in open(man)]
        items, meta = [], []
        for r in rows:
            wav = load16k(r["gen"]).clamp(-1, 1)
            gp, tags = gen_gaps(rec, wav, r["text"])
            tg = {g for g, _ in tags}
            for g, t in tags:
                if g in gp: items.append(seg(wav, *gp[g])); meta.append((r["id"], g, t))
                else: items.append(wav[-int(MIN_SEG * SR):]); meta.append((r["id"], g, t))  # tag after the last word
            for g, (t0, t1) in gp.items():   # untagged pauses long enough to hold an event -> false-positive check
                if g not in tg and t1 - t0 >= 0.15: items.append(seg(wav, t0, t1)); meta.append((r["id"], g, "none"))
        P = clf.predict_proba(emb(items))
        pred = [CLS[i] for i in P.argmax(1)]
        hit = Counter(); n = Counter(); fp = nneg = 0; per = []
        for (i, g, t), p, pr in zip(meta, pred, P):
            if t == "none": nneg += 1; fp += p != "none"
            else: n[t] += 1; hit[t] += p == t; per.append({"id": i, "gap": g, "tag": t, "pred": p, "p_tag": round(float(pr[CLS.index(t)]), 3)})
        s = {"nvpa": sum(hit.values()) / max(1, sum(n.values())), "fp_rate": fp / max(1, nneg), "n": sum(n.values()),
             "p_tag": float(np.mean([x["p_tag"] for x in per])) if per else 0.0,
             **{f"rec_{t}": hit[t] / n[t] for t in CLS[1:] if n[t]}}
        with open(man.replace(".jsonl", ".nvdet.jsonl"), "w") as f: f.writelines(json.dumps(x) + "\n" for x in per)
        print("NVDET", man, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "score"]); ap.add_argument("manifests", nargs="*")
    ap.add_argument("--events"); ap.add_argument("--dev-events"); ap.add_argument("--train"); ap.add_argument("--dev"); ap.add_argument("--out"); ap.add_argument("--model")
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--vocalsound", help="VocalSound audio dir -> extra laughter/sniff/throatclearing positives")
    a = ap.parse_args()
    {"train": train, "score": score}[a.cmd](a)
