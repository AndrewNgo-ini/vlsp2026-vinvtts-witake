"""NV realism model fitted on REAL recordings only (no labels): per class, a Gaussian over envelope features of real NVs
(nvloud.py on train GT). Realism of a generated NV = Mahalanobis distance to its class (lower = more like real speech).
Features: dLvl (dB vs +-1 s of speech), log duration, bursts, onset rise, offset fall (dB/10 ms), boundary jump (dB).
    uv run nvreal.py fit <gt.loud.jsonl> <model.json>
    uv run nvreal.py score <model.json> <sys.loud.jsonl> ...      -> per class median distance, share inside the GT 90% region
"""
import json, sys
from collections import defaultdict
import numpy as np

F = ["d", "dur", "bursts", "rise", "fall", "jump"]


def vec(x):
    return [x["d"], np.log(max(x["dur"], 0.02)), min(x.get("bursts", 1), 4), x.get("rise", 0.0), x.get("fall", 0.0), x.get("jump", 0.0)]


def rows(p):
    c = defaultdict(list)
    for r in map(json.loads, open(p)):
        for x in r["nv"]:
            if "rise" in x: c[x["tag"]].append(vec(x))
    return {k: np.array(v, float) for k, v in c.items()}


class Real:
    def __init__(self, m): self.m = {k: (np.array(v["mu"]), np.array(v["P"]), v["q90"]) for k, v in m.items()}

    def dist(self, tag, x):
        if tag not in self.m: return 0.0
        mu, P, _ = self.m[tag]; z = np.asarray(vec(x)) - mu; return float(np.sqrt(max(z @ P @ z, 0.0)))


def fit(gt, out):
    m = {}
    for k, X in rows(gt).items():
        if len(X) < 20: continue
        mu = X.mean(0); C = np.cov(X.T) + np.eye(X.shape[1]) * 1e-2          # ridge: bursts is near-constant for some classes
        P = np.linalg.inv(C); d = np.sqrt(np.einsum("ij,jk,ik->i", X - mu, P, X - mu))
        m[k] = {"mu": mu.tolist(), "P": P.tolist(), "q90": float(np.quantile(d, .9)), "n": len(X),
                "gt_median": dict(zip(F, np.median(X, 0).round(2).tolist()))}
        print(k, "n", len(X), "q90", round(m[k]["q90"], 2), m[k]["gt_median"])
    json.dump(m, open(out, "w"), indent=1)


def score(model, mans):
    R = Real(json.load(open(model)))
    for man in mans:
        out = {}
        for k, X in rows(man).items():
            if k not in R.m: continue
            mu, P, q90 = R.m[k]; d = np.sqrt(np.einsum("ij,jk,ik->i", X - mu, P, X - mu))
            out[k] = {"median_dist": round(float(np.median(d)), 2), "inside_gt90": round(float(np.mean(d <= q90)), 3), "n": len(X)}
        mean_in = np.mean([v["inside_gt90"] for v in out.values()]) if out else float("nan")
        print(man, "realism (share of NVs inside the real 90% region, mean over classes):", round(float(mean_in), 3), json.dumps(out))


if __name__ == "__main__":
    if sys.argv[1] == "fit": fit(sys.argv[2], sys.argv[3])
    else: score(sys.argv[2], sys.argv[3:])
