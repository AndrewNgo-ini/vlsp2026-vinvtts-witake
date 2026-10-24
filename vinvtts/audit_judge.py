"""AUDIT judge (never used by any generation / selection / splicing path): per-class NV presence at each tag's aligned
gap, from CLAP (LAION HTSAT audio encoder) embeddings + logistic regression. nvdet uses AST; PSED uses BEATs/ATST/M2D,
so the features here are independent of every pipeline judge (only the Zipformer gap alignment is shared).
Train: real train events + untagged-gap negatives (nvdet.bank) + VocalSound (laughter/sniff/throat) + optional
NonverbalTTS windows (--nvtts, see nvtts_items). Validate on dev events (events_dev).

    uv run audit_judge.py train --out runs/audit_judge [--nvtts]
    uv run audit_judge.py score --model runs/audit_judge <sys>.jsonl ...   -> <sys>.audit.jsonl + per class/position
"""
import cpulimit  # noqa: F401
import argparse, json, pickle, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, torch
from judge import load16k, SR
from nvdet import bank, vs_items, seg, gen_gaps, MIN_SEG
from tags import split

CLS = ["none", "laughter", "breathing", "sniff", "throatclearing"]
CLAP = "laion/clap-htsat-unfused"
R = "/data/vinvtts/runs"; D = "/data/vinvtts/data"


class WavLM:
    def __init__(self):
        from transformers import AutoFeatureExtractor, WavLMModel
        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus"); self.m = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").cuda().eval()

    @torch.no_grad()
    def __call__(self, segs):
        out = []
        for x in segs:
            f = self.fe(x.float().numpy(), sampling_rate=16000, return_tensors="pt").input_values.cuda()
            h = self.m(f, output_hidden_states=True).hidden_states
            out.append(torch.cat([h[i][0].mean(0) for i in (4, 8, 12)]).float().cpu().numpy())
        return np.stack(out) if out else np.zeros((0, 2304), np.float32)


class Both:
    """final audit features = [10 * CLAP(512, L2-normalised) | WavLM-base-plus layers 4/8/12 mean-pooled (2304)]"""
    def __init__(self): self.c, self.w = Embedder(), WavLM()
    def __call__(self, segs): return np.concatenate([self.c(segs) * 10, self.w(segs)], 1)


class Embedder:
    def __init__(self):
        from transformers import ClapModel, ClapProcessor
        self.p = ClapProcessor.from_pretrained(CLAP); self.m = ClapModel.from_pretrained(CLAP).cuda().eval()
        import torchaudio; self.rs = torchaudio.transforms.Resample(SR, 48000)

    @torch.no_grad()
    def __call__(self, segs, bs=64):
        out = []
        for i in range(0, len(segs), bs):
            x = [self.rs(s.float()).numpy() for s in segs[i:i + bs]]
            f = self.p(audio=x, sampling_rate=48000, return_tensors="pt")
            f = {k: v.cuda() for k, v in f.items()}
            e = self.m.get_audio_features(**f)
            e = e.pooler_output if hasattr(e, "pooler_output") else e
            out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def feats(emb, items):
    cache, segs = {}, []
    for a, t0, t1, _ in items:
        if a not in cache:
            if len(cache) > 64: cache.clear()
            cache[a] = load16k(a)
        segs.append(seg(cache[a], t0, t1))
    return emb(segs), np.array([CLS.index(y) for *_, y in items])


def nvtts_items(emb, clf, cap=600):
    """NonverbalTTS (deepvk, CC BY-NC-SA) clips carrying a sniff / throat-clearing / laughter / breathing label have no
    event times: multiple-instance pick = the 0.6 s window (hop .2 s) the first-pass classifier scores highest for that
    class; kept only if that probability >= .5 (self-training on confident windows)."""
    import glob, io, pandas as pd, soundfile as sf, torchaudio
    fs = sorted(glob.glob(f"{D}/nvtts/default/*/*.parquet"))
    KEY = {"sniff": ["\U0001f443"], "throatclearing": ["\U0001f5e3"]}   # NV emoji: nose = sniff, speaking head = throat clearing (NonverbalTTS guidelines)
    got = Counter(); out = []; wdir = Path(f"{D}/nvtts/wav"); wdir.mkdir(exist_ok=True)
    for f in fs:
        df = pd.read_parquet(f)
        tcol, acol = "Result", "audio"
        for i, row in df.iterrows():
            txt = str(row[tcol])
            cls = [c for c, ks in KEY.items() if any(k in txt for k in ks)]
            if len(cls) != 1 or got[cls[0]] >= cap: continue
            c = cls[0]; a = row[acol]; b = a["bytes"] if isinstance(a, dict) else a
            y, sr = sf.read(io.BytesIO(b), dtype="float32"); y = y if y.ndim == 1 else y.mean(1)
            y = torchaudio.functional.resample(torch.from_numpy(y), sr, SR)
            if len(y) < SR: continue
            W, H = int(.6 * SR), int(.2 * SR); st = list(range(0, max(1, len(y) - W), H))
            P = clf.predict_proba(emb([y[s:s + W] for s in st]))[:, CLS.index(c)]
            k = int(P.argmax())
            if P[k] < .5: continue
            p = wdir / f"{Path(f).parent.name}_{i}.wav"; sf.write(p, y.numpy(), SR)
            out.append((str(p), st[k] / SR + .08, (st[k] + W) / SR - .08, c)); got[c] += 1
    print("nvtts", dict(got), flush=True)
    return out


def train(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    emb = Embedder()
    items = bank(f"{R}/events", f"{D}/vinv-tts/train.jsonl") + vs_items(f"{D}/vocalsound/clean")
    X, y = feats(emb, items); print("train", Counter(CLS[i] for i in y), flush=True)
    Xd, yd = feats(emb, bank(f"{R}/events_dev", f"{D}/vinv-tts/dev.jsonl", seed=1))
    fit = lambda X, y: LogisticRegression(C=a.C, max_iter=5000, class_weight="balanced").fit(X, y)
    clf = fit(X, y)
    print(classification_report(yd, clf.predict(Xd), labels=range(5), target_names=CLS, digits=3), flush=True)
    if a.nvtts:
        extra = nvtts_items(emb, clf); Xe, ye = feats(emb, extra)
        clf = fit(np.concatenate([X, Xe]), np.concatenate([y, ye]))
        print("with NonverbalTTS"); print(classification_report(yd, clf.predict(Xd), labels=range(5), target_names=CLS, digits=3), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True); pickle.dump(clf, open(f"{a.out}/clf.pkl", "wb"))


def pos(g, n): return "start" if g == 0 else "end" if g >= n else "mid"


def score(a):
    from align import recognizer
    rec = recognizer(); clf = pickle.load(open(f"{a.model}/clf.pkl", "rb")); emb = Both()
    for man in a.manifests:
        rows = [json.loads(l) for l in open(man)]; items, meta = [], []
        for r in rows:
            wav = load16k(r["gen"]).clamp(-1, 1); gp, tags = gen_gaps(rec, wav, r["text"]); nw = len(split(r["text"])[0])
            for g, t in tags:
                items.append(seg(wav, *gp[g]) if g in gp else wav[-int(MIN_SEG * SR):]); meta.append((r["id"], g, t, pos(g, nw)))
        P = clf.predict_proba(emb(items)); h, n = Counter(), Counter(); per = []
        for (i, g, t, p), pr in zip(meta, P):
            ok = CLS[int(pr.argmax())] == t
            for k in [(t, p), (t, "all"), ("ALL", p), ("ALL", "all")]: n[k] += 1; h[k] += ok
            per.append({"id": i, "gap": g, "tag": t, "pos": p, "hit": ok, "p_tag": round(float(pr[CLS.index(t)]), 3)})
        with open(man.replace(".jsonl", ".audit.jsonl"), "w") as f: f.writelines(json.dumps(x) + "\n" for x in per)
        cells = {f"{c}_{p}": round(h[(c, p)] / n[(c, p)], 3) for c in CLS[1:] + ["ALL"] for p in ["start", "mid", "end", "all"] if n[(c, p)]}
        cells["macro"] = round(np.mean([h[(c, "all")] / n[(c, "all")] for c in CLS[1:] if n[(c, "all")]]), 3)
        print("AUDIT", man, json.dumps(cells), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["train", "score"]); ap.add_argument("manifests", nargs="*")
    ap.add_argument("--out", default=f"{R}/audit_judge"); ap.add_argument("--model", default=f"{R}/audit_judge")
    ap.add_argument("--C", type=float, default=1.0); ap.add_argument("--nvtts", action="store_true")
    a = ap.parse_args(); {"train": train, "score": score}[a.cmd](a)
