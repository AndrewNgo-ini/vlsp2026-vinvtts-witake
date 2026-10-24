"""PretrainedSED judge for sniff / throat clearing at the tag positions of a system's audio (class-aware objective,
coordinator decision 09-27). Detector = the psed-only logistic stack of row P2 (3 PretrainedSED models x 4 class probs +
logs + span; arXiv 2409.09546), class_weight balanced, fitted on ALL real events; hit = p >= .5 (the balanced decision
boundary, analogous to nvdet's argmax). PSED was never a training reward except in the discarded GRPO-T (row G3).

  fit   : runs/psed_*.jsonl -> runs/rare_judge/{sniff,throatclearing}.pkl (+ speaker-grouped OOF P/R at p >= .5)
  spans : <sys>.jsonl -> <sys>.spans.jsonl  (only [sniff] / [throatclearing] tag gaps; nvdet.gen_gaps incl. gap 0)
  (psed_score.py in the PSED venv -> <sys>.psed.jsonl)
  apply : <sys>.psed.jsonl -> <sys>.rare.jsonl {id, gap, tag, p, hit}
"""
import cpulimit  # noqa: F401
import json, pickle, sys
from pathlib import Path
import numpy as np
from throat_det import feats, M, C, R

O = Path(f"{R}/rare_judge"); RARE = ["sniff", "throatclearing"]


def fit():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    rows = [[json.loads(l) for l in open(f"{R}/psed_{m}.jsonl")] for m in M]
    X = np.array([feats([[rr[i][c] for c in C] for rr in rows], rows[0][i]["t1"] - rows[0][i]["t0"]) for i in range(len(rows[0]))])
    spk = np.array([r["audio"].split("/")[-2] for r in rows[0]]); O.mkdir(parents=True, exist_ok=True)
    for c in RARE:
        y = np.array([r["tag"] == c for r in rows[0]]); mk = lambda: LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced")
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, spk): oof[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        sel = oof >= .5
        print(f"RARE_JUDGE {c}: OOF at p>=.5 recall {y[sel].sum() / y.sum():.2f} precision {y[sel].mean():.3f} (n+ {y.sum()}, flagged {sel.sum()})")
        pickle.dump(mk().fit(X, y), open(O / f"{c}.pkl", "wb"))


def spans(man):
    from align import recognizer
    rec = recognizer()
    import nvdet
    from judge import load16k
    out = []
    for r in map(json.loads, open(man)):
        if "[sniff]" not in r["text"] and "[throatclearing]" not in r["text"]: continue
        y = load16k(r["gen"]).clamp(-1, 1); gp, tags = nvdet.gen_gaps(rec, y, r["text"]); d = len(y) / 16000
        for g, t in tags:
            if t in RARE:
                t0, t1 = gp.get(g, (max(0, d - 1), d)); out.append({"id": r["id"], "gen": r["gen"], "gap": g, "tag": t, "t0": round(t0, 3), "t1": round(t1, 3)})
    with open(man.replace(".jsonl", ".spans.jsonl"), "w") as f: f.writelines(json.dumps(x) + "\n" for x in out)
    print("rare spans", man, len(out))


def apply(man):
    clf = {c: pickle.load(open(O / f"{c}.pkl", "rb")) for c in RARE}
    rows = [json.loads(l) for l in open(man.replace(".jsonl", ".psed.jsonl"))]
    out = []
    for r in rows:
        p = float(clf[r["tag"]].predict_proba(feats(r["probs"], r["t1"] - r["t0"])[None])[0, 1])
        out.append({"id": r["id"], "gap": r["gap"], "tag": r["tag"], "p": round(p, 4), "hit": p >= .5})
    with open(man.replace(".jsonl", ".rare.jsonl"), "w") as f: f.writelines(json.dumps(x) + "\n" for x in out)
    for c in RARE:
        h = [x["hit"] for x in out if x["tag"] == c]; print(f"RARE {Path(man).name:26s} {c} recall {np.mean(h) if h else 0:.3f} (n{len(h)})")


if __name__ == "__main__":
    if sys.argv[1] == "fit": fit()
    else:
        for m in sys.argv[2:]: {"spans": spans, "apply": apply}[sys.argv[1]](m)
