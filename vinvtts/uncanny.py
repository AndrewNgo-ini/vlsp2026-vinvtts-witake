"""NV-naturalness objective (2026-10-01): evaluation of NV levelling.
Primary U (dB, lower better) = mean over classes of the Wasserstein-1 distance between a system's dLvl distribution
(NV level minus +-1 s of surrounding speech, nvloud.py) and real speech's (train GT). Distributional on purpose:
collapsing every NV to the GT median is not natural either, and W1 penalises it.
Secondary (logged): W1 of NV duration (s), boundary level jump (dB), steepest onset/offset (dB/10 ms) -> rhythm / cut-off;
multi-burst rate (share of NVs with >= 2 energy bursts: the 'double sniff') vs GT.
Guards vs a baseline run (same items): WER <= +.005, pMOS >= -.02, SS >= -.005, v3 NV >= -.03, audit NV(exS) >= -.03.
    uv run uncanny.py <gt.loud.jsonl> <base_dir> <cand_dir> [T ...]   (dirs hold <T>.{loud,scored,tags_qasr,audit}.jsonl)
Prints one JSON line per track + "U=<mean over tracks>" and "GUARDS=PASS|FAIL".
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from tags import nv_hits

CLS = ["breathing", "laughter", "sniff", "throatclearing"]
G = dict(wer=+.005, pmos=-.02, ss=-.005, nv=-.03, audit=-.03)


def w1(a, b):
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if not len(a) or not len(b): return float("nan")
    q = np.linspace(0, 1, 201); return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def loud(p):
    d = defaultdict(lambda: defaultdict(list))
    for r in map(json.loads, open(p)):
        for x in r["nv"]:
            for k in ("d", "dur", "jump", "rise", "fall", "bursts"):
                if k in x: d[k][x["tag"]].append(x[k])
    return d


def auto(dirp, T):
    sc = [json.loads(l) for l in open(dirp / f"{T}.scored.jsonl")]
    qa = {r["id"]: r["tag_hyp"] for r in map(json.loads, open(dirp / f"{T}.tags_qasr.jsonl"))}
    hits = [ok for r in sc if r["id"] in qa for _, ok in nv_hits(r["text"], qa[r["id"]])]
    au = [x for x in map(json.loads, open(dirp / f"{T}.audit.jsonl")) if x.get("pos") != "start"] if (dirp / f"{T}.audit.jsonl").exists() else []
    per = defaultdict(list)
    for x in au: per[x["tag"]].append(x["hit"])
    return dict(wer=np.mean([r["wer"] for r in sc]), pmos=np.mean([r["pmos"] for r in sc]), ss=np.mean([r["ss"] for r in sc]),
                nv=np.mean(hits), audit=np.mean([np.mean(v) for v in per.values()]) if per else float("nan"))


if __name__ == "__main__":
    gt = loud(sys.argv[1]); base, cand = Path(sys.argv[2]), Path(sys.argv[3]); Ts = sys.argv[4:] or ["A", "B"]
    Us, ok_all = [], True
    for T in Ts:
        c = loud(cand / f"{T}.loud.jsonl")
        U = float(np.nanmean([w1(c["d"][k], gt["d"][k]) for k in CLS]))
        sec = {m: round(float(np.nanmean([w1(c[m][k], gt[m][k]) for k in CLS])), 3) for m in ("dur", "jump", "rise", "fall")}
        multi = {k: (round(float(np.mean(np.array(c["bursts"][k]) >= 2)), 3), round(float(np.mean(np.array(gt["bursts"][k]) >= 2)), 3)) for k in CLS if c["bursts"][k] and gt["bursts"][k]}
        med = {k: round(float(np.median(c["d"][k])), 1) for k in CLS if c["d"][k]}
        a, b = auto(cand, T), auto(base, T)
        delta = {k: round(float(a[k] - b[k]), 4) for k in G}
        ok = {k: bool(delta[k] <= G[k]) if k == "wer" else bool(delta[k] >= G[k] or np.isnan(delta[k])) for k in G}
        ok_all &= all(ok.values()); Us.append(U)
        print(json.dumps({"T": T, "U_level_w1": round(U, 2), "secondary_w1": sec, "multi_burst_sys_vs_gt": multi, "median_dLvl": med,
                          "auto": {k: round(float(v), 4) for k, v in a.items()}, "delta_vs_base": delta, "guards": ok}))
    print(f"U={np.mean(Us):.2f}"); print("GUARDS=" + ("PASS" if ok_all else "FAIL"))
