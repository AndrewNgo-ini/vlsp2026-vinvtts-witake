"""PSED venv half of throat_det.py: <cands>.spans.jsonl -> <cands>.psed.jsonl, per span the max frame probability of
sniff / throat clearing / laughter / breathing within [t0 - .1, t1 + .1] for each of BEATs, ATST-F, M2D (psed_eval.py).

    /mnt/vinvtts/src/psed/.venv/bin/python psed_score.py runs/.../cands.jsonl
"""
import json, sys
from collections import defaultdict
import numpy as np
from psed_eval import load, frame_probs, FPS

MODELS = ["BEATs", "ATST-F", "M2D"]

for man in sys.argv[1:]:
    sp = [json.loads(l) for l in open(man.replace(".jsonl", ".spans.jsonl"))]
    by = defaultdict(list)
    for i, s in enumerate(sp): by[s["gen"]].append(i)
    probs = [[None] * len(MODELS) for _ in sp]
    for k, name in enumerate(MODELS):
        m = load(name).cuda().eval()
        for g, idx in by.items():
            P = frame_probs(m, g)
            for i in idx:
                a, b = max(0, int((sp[i]["t0"] - .1) * FPS)), min(P.shape[1], int((sp[i]["t1"] + .1) * FPS) + 1)
                probs[i][k] = [round(float(v), 4) for v in (P[:, a:b].max(1) if b > a else np.zeros(P.shape[0]))]
        del m
    with open(man.replace(".jsonl", ".psed.jsonl"), "w") as f:
        f.writelines(json.dumps({**s, "probs": pr}) + "\n" for s, pr in zip(sp, probs))
    print("psed", man, len(sp), flush=True)
