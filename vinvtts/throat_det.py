"""High-precision throat-clearing detector = logistic stack over 3 PretrainedSED models (row P2; Schmid et al. arXiv
2409.09546, MIT weights), used ONLY above its speaker-grouped out-of-fold precision >= .85 operating point.

  fit   : psed_{BEATs,ATST-F,M2D}.jsonl (all real train+dev events, psed_eval.py) -> runs/throat_det/{clf.pkl,thr.json}
  spans : <cands>.jsonl -> <cands>.spans.jsonl: time span of every [throatclearing] tag gap + every untagged word gap
          >= .15 s in the generated wav (nvdet.gen_gaps, Zipformer) -- the untagged gaps give the spurious-throat rate
  (psed_score.py in the PSED venv then writes <cands>.psed.jsonl: per span, 3 models x 4 class max probs)
  apply : <cands>.psed.jsonl -> <cands>.throat.jsonl {id, gap, tag, p, hit = p >= thr}; prints throat recall and the
          spurious-throat rate (fraction of untagged gaps with hit)

    uv run throat_det.py fit ; uv run throat_det.py spans X.jsonl ; <psed venv> psed_score.py X.jsonl ; uv run throat_det.py apply X.jsonl
"""
import cpulimit  # noqa: F401
import json, pickle, sys
from pathlib import Path
import numpy as np

R = "/mnt/vinvtts/runs"; O = Path(f"{R}/throat_det"); M = ["BEATs", "ATST-F", "M2D"]
C = ["sniff", "throatclearing", "laughter", "breathing"]; P_OP = float(__import__("os").environ.get("P_OP", "0.80"))   # coordinator: relaxed from .85 (no genuine .85 point, row P3)


def feats(probs, span):
    """probs: [model][class] max probs (3 x 4); span: seconds -> feature vector (same as the P2 stack)."""
    x = np.array([probs[m][c] for m in range(len(M)) for c in range(len(C))], float)
    return np.concatenate([x, np.log(np.clip(x, 1e-4, 1)), [span]])


def fit():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    rows = [[json.loads(l) for l in open(f"{R}/psed_{m}.jsonl")] for m in M]
    X = np.array([feats([[rr[i][c] for c in C] for rr in rows], rows[0][i]["t1"] - rows[0][i]["t0"]) for i in range(len(rows[0]))])
    y = np.array([r["tag"] == "throatclearing" for r in rows[0]]); spk = np.array([r["audio"].split("/")[-2] for r in rows[0]])
    mk = lambda: LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, spk): oof[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    # threshold by SET precision of {oof >= t} over unique values (tie-safe: saturated scores share one value); lowest t
    # whose set keeps OOF precision >= P_OP and holds >= 5 positives
    best = None
    for t in np.unique(oof)[::-1]:
        sel = oof >= t; npos = int(y[sel].sum())
        if npos >= 5 and y[sel].mean() >= P_OP: best = (float(t), float(y[sel].mean()), npos / y.sum())
    thr, pk, rk = best if best else (1.1, 0.0, 0.0); O.mkdir(parents=True, exist_ok=True)
    pickle.dump(mk().fit(X, y), open(O / "clf.pkl", "wb"))
    json.dump({"thr": thr, "oof_prec": pk, "oof_rec": rk, "n_pos": int(y.sum()), "p_op": P_OP}, open(O / "thr.json", "w"))
    print(f"THROAT_DET P_OP {P_OP} thr {thr:.6f} OOF set precision {pk:.2f} recall {rk:.2f} (n+ {y.sum()})")


def spans(man):
    from align import recognizer
    rec = recognizer()
    import nvdet
    from judge import load16k
    out = []
    for r in map(json.loads, open(man)):
        y = load16k(r["gen"]).clamp(-1, 1); gp, tags = nvdet.gen_gaps(rec, y, r["text"]); tg = dict(tags)
        for g, (t0, t1) in gp.items():
            t = tg.get(g)
            if t == "throatclearing" or (t is None and t1 - t0 >= 0.15):
                out.append({"id": r["id"], "gen": r["gen"], "gap": g, "tag": t or "none", "t0": round(t0, 3), "t1": round(t1, 3)})
        for g, t in tags:   # tag after the last aligned word -> last second (as nvdet)
            if t == "throatclearing" and g not in gp:
                d = len(y) / 16000; out.append({"id": r["id"], "gen": r["gen"], "gap": g, "tag": t, "t0": max(0, d - 1), "t1": d})
    with open(man.replace(".jsonl", ".spans.jsonl"), "w") as f: f.writelines(json.dumps(x) + "\n" for x in out)
    print("spans", len(out), "throat", sum(x["tag"] == "throatclearing" for x in out))


def apply(man):
    clf = pickle.load(open(O / "clf.pkl", "rb")); thr = json.load(open(O / "thr.json"))["thr"]
    rows = [json.loads(l) for l in open(man.replace(".jsonl", ".psed.jsonl"))]
    X = np.array([feats(r["probs"], r["t1"] - r["t0"]) for r in rows]) if rows else np.zeros((0, 25))
    p = clf.predict_proba(X)[:, 1] if len(rows) else []
    out = [{"id": r["id"], "gap": r["gap"], "tag": r["tag"], "p": round(float(q), 4), "hit": bool(q >= thr)} for r, q in zip(rows, p)]
    with open(man.replace(".jsonl", ".throat.jsonl"), "w") as f: f.writelines(json.dumps(x) + "\n" for x in out)
    tt = [x["hit"] for x in out if x["tag"] == "throatclearing"]; nn = [x["hit"] for x in out if x["tag"] == "none"]
    print(f"THROAT {Path(man).name:24s} recall@P.85 {np.mean(tt) if tt else 0:.3f} (n{len(tt)}) spurious {np.mean(nn) if nn else 0:.4f} (n{len(nn)})", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fit": fit()
    else:
        for m in sys.argv[2:]: {"spans": spans, "apply": apply}[cmd](m)
