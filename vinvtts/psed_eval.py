"""Zero-shot rare-event detector check: PretrainedSED (Schmid et al., arXiv 2409.09546; github.com/fschmid56/PretrainedSED,
MIT; frame-level transformers trained on AudioSet-strong, 447 classes incl. Sniff / Throat clearing / Laughter /
Breathing, 40 ms frames). Our real events (train + dev event banks from align.py) are positives; every other located
event and every untagged word gap >= .12 s is a negative. Score of a segment = max frame prob of the class within
[t0 - .1, t1 + .1]. Reports precision at fixed thresholds and the best precision at recall >= .2 / .4, per class.
The model never saw our data, so train events are an unbiased test set too (232 sniff + throat events in total).
Adopt as a filter / reward only if precision >= .8 at usable recall. Also writes per-segment scores (psed_<model>.jsonl).

    /mnt/vinvtts/src/psed/.venv/bin/python psed_eval.py BEATs ATST-F
"""
import json, sys, os
from collections import defaultdict
import numpy as np, torch, librosa

PS = "/mnt/vinvtts/src/psed"; sys.path.insert(0, PS); os.chdir(PS)
torch.set_num_threads(2)
from data_util import audioset_classes  # noqa: E402
from models.prediction_wrapper import PredictionsWrapper  # noqa: E402

R = "/mnt/vinvtts/runs"; SR = 16000; FPS = 25.0
CLS = {"sniff": "Sniff", "throatclearing": "Throat clearing", "laughter": "Laughter", "breathing": "Breathing"}
names = list(audioset_classes.as_strong_train_classes); IDX = {k: names.index(v) for k, v in CLS.items()}


def load(name):
    if name == "BEATs":
        from models.beats.BEATs_wrapper import BEATsWrapper
        return PredictionsWrapper(BEATsWrapper(), checkpoint="BEATs_strong_1")
    if name == "ATST-F":
        from models.atstframe.ATSTF_wrapper import ATSTWrapper
        return PredictionsWrapper(ATSTWrapper(), checkpoint="ATST-F_strong_1")
    if name == "M2D":
        from models.m2d.M2D_wrapper import M2DWrapper
        m = M2DWrapper(); return PredictionsWrapper(m, checkpoint="M2D_strong_1", embed_dim=m.m2d.cfg.feature_d)
    raise ValueError(name)


@torch.no_grad()
def frame_probs(model, path):
    y, _ = librosa.load(path, sr=SR, mono=True); y = torch.from_numpy(y[None]).cuda(); seg = 10 * SR; out = []
    for i in range(0, y.shape[1], seg):
        c = y[:, i:i + seg]; n = c.shape[1]
        if n < seg: c = torch.nn.functional.pad(c, (0, seg - n))
        s, _ = model(model.mel_forward(c)); p = torch.sigmoid(s)[0]   # (447, 250)
        out.append(p[:, : max(1, int(round(n / SR * FPS)))])
    return torch.cat(out, 1)[[IDX[k] for k in CLS]].float().cpu().numpy()   # (4, T)


def segments():
    segs = []
    for d in ["events", "events_dev"]:
        ev = [json.loads(l) for l in open(f"{R}/{d}/events.jsonl")]
        tagged = defaultdict(set)
        for e in ev: segs.append((e["audio"], e["t0"], e["t1"], e["tag"], d)); tagged[e["audio"]].add(int(e["gap"]))
        for a, gp in json.load(open(f"{R}/{d}/gaps.json")).items():
            for g, (t0, t1) in gp.items():
                if int(g) not in tagged[a] and t1 - t0 >= 0.12: segs.append((a, t0, t1, "none", d))
    return segs


def main():
    segs = segments(); by = defaultdict(list)
    for s in segs: by[s[0]].append(s)
    print("segments", len(segs), {t: sum(s[3] == t for s in segs) for t in list(CLS) + ["none"]}, flush=True)
    for name in sys.argv[1:]:
        m = load(name).cuda().eval(); rows = []
        for k, (a, ss) in enumerate(by.items()):
            P = frame_probs(m, a)
            for (_, t0, t1, tag, d) in ss:
                i, j = max(0, int((t0 - .1) * FPS)), min(P.shape[1], int((t1 + .1) * FPS) + 1)
                sc = P[:, i:j].max(1) if j > i else np.zeros(len(CLS))
                rows.append({"audio": a, "t0": t0, "t1": t1, "tag": tag, "set": d, **{c: round(float(v), 4) for c, v in zip(CLS, sc)}})
            if k % 300 == 0: print(name, k, "/", len(by), flush=True)
        with open(f"{R}/psed_{name}.jsonl", "w") as f: f.writelines(json.dumps(r) + "\n" for r in rows)
        for c in CLS:
            y = np.array([r["tag"] == c for r in rows]); s = np.array([r[c] for r in rows]); o = np.argsort(-s)
            tp = np.cumsum(y[o]); prec = tp / np.arange(1, len(o) + 1); rec = tp / max(1, y.sum())
            ap = float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))
            best = lambda rr: float(prec[rec >= rr].max()) if (rec >= rr).any() else 0.0
            at = " ".join(f"p>={t}: P {y[s >= t].mean() if (s >= t).any() else 0:.2f} R {(y & (s >= t)).sum() / max(1, y.sum()):.2f}" for t in (.3, .5, .7))
            print(f"PSED {name:6s} {c:15s} n+={y.sum():4d} AP {ap:.3f} bestP@R.2 {best(.2):.2f} bestP@R.4 {best(.4):.2f} | {at}", flush=True)


if __name__ == "__main__":
    main()
