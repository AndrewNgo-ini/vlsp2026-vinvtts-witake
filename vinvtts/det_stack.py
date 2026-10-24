"""Rare-class detector v2 (option 2): stack PretrainedSED class probabilities (psed_eval.py; arXiv 2409.09546) with the
nvdet AST features (AST AudioSet embedding 768 + 527 logits, arXiv 2104.01778 -- the same segment features nvdet's LR
uses), evaluated with speaker-grouped 5-fold out-of-fold predictions on ALL real events (train + dev event banks).
Precision is measured on the SET {score >= t} over unique score values (tie-safe; row P3).

  emb   (GPU, project venv) : AST features for every segment of runs/psed_BEATs.jsonl (same order) -> runs/det_stack/ast.npy
  eval  (CPU)               : per class (sniff, throatclearing): AP and best set-precision at recall >= .1/.2/.3 for
                              PSED-only, AST-only (= nvdet-style LR), and the stack; fits + saves the best on all data.

    uv run det_stack.py emb ; uv run det_stack.py eval
"""
import cpulimit  # noqa: F401
import json, pickle, sys
from pathlib import Path
import numpy as np

R = "/mnt/vinvtts/runs"; O = Path(f"{R}/det_stack"); M = ["BEATs", "ATST-F", "M2D"]
C = ["sniff", "throatclearing", "laughter", "breathing"]


def rows():
    rr = [[json.loads(l) for l in open(f"{R}/psed_{m}.jsonl")] for m in M]
    assert all(len(x) == len(rr[0]) for x in rr)
    return rr


def emb():
    import torch
    from judge import load16k
    from nvdet import Embedder, seg
    rr = rows()[0]; E = Embedder(); out = np.zeros((len(rr), 768 + 527), np.float32)
    by = {}
    for i, r in enumerate(rr): by.setdefault(r["audio"], []).append(i)
    for k, (a, idx) in enumerate(by.items()):
        w = load16k(a).clamp(-1, 1)
        out[idx] = E([seg(w, rr[i]["t0"], rr[i]["t1"]) for i in idx])
        if k % 300 == 0: print("emb", k, "/", len(by), flush=True)
    O.mkdir(parents=True, exist_ok=True); np.save(O / "ast.npy", out); print("saved", out.shape)


def curve(y, s):
    ap_o = np.argsort(-s); tp = np.cumsum(y[ap_o]); prec = tp / np.arange(1, len(y) + 1); rec = tp / y.sum()
    ap = float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))
    pts = {}
    for t in np.unique(s)[::-1]:
        sel = s >= t; r_ = y[sel].sum() / y.sum(); p_ = y[sel].mean()
        for rr_ in (.1, .2, .3):
            if r_ >= rr_ and p_ > pts.get(rr_, 0): pts[rr_] = p_
    return ap, pts


def evaluate():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rr = rows(); r0 = rr[0]
    P = np.array([[x[i][c] for x in rr for c in C] for i in range(len(r0))])
    Xp = np.concatenate([P, np.log(np.clip(P, 1e-4, 1)), [[r["t1"] - r["t0"]] for r in r0]], 1)
    A = np.load(O / "ast.npy"); spk = np.array([r["audio"].split("/")[-2] for r in r0])
    sets = {"psed": Xp, "ast": A, "stack": np.concatenate([Xp, A], 1)}
    best = {}
    for c in ["sniff", "throatclearing"]:
        y = np.array([r["tag"] == c for r in r0])
        for name, X in sets.items():
            for Cr in (0.05, 0.5):
                mk = lambda: make_pipeline(StandardScaler(), *( [PCA(128, random_state=0)] if X.shape[1] > 200 else []),
                                           LogisticRegression(C=Cr, max_iter=3000))
                oof = np.zeros(len(y))
                for tr, te in GroupKFold(5).split(X, y, spk): oof[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                ap, pts = curve(y, oof)
                print(f"DET {c:15s} {name:6s} C={Cr:<5} OOF AP {ap:.3f} " + " ".join(f"P@R{k}={v:.2f}" for k, v in sorted(pts.items())), flush=True)
                if ap > best.get(c, (0,))[0]: best[c] = (ap, name, Cr, oof)
        ap, name, Cr, _ = best[c]; X = sets[name]
        m = make_pipeline(StandardScaler(), *([PCA(128, random_state=0)] if X.shape[1] > 200 else []), LogisticRegression(C=Cr, max_iter=3000)).fit(X, y)
        pickle.dump({"model": m, "features": name, "C": Cr}, open(O / f"{c}.pkl", "wb"))
        print(f"BEST {c}: {name} C={Cr} OOF AP {ap:.3f}")


if __name__ == "__main__":
    {"emb": emb, "eval": evaluate}[sys.argv[1]]()
