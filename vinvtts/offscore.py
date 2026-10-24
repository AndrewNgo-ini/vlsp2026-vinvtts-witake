"""Official-weight auto proxy (offA / offB) for a run_final.py output: <T>.scored.jsonl (bench.py) + <T>.tags_qasr.jsonl
(v3) [+ <T>.nvdet.jsonl, <T>.audit.jsonl if present]. Rebuilt 2026-10-01 (w1score's summariser was lost with the box);
reproduces x5_cleanft.md: off = (.3 NV_v3 + w_wer (1-WER) + w_mos pMOS/5 + w_ss SS) / .7, NV_v3 = micro tag hit rate.
    uv run offscore.py <dir>/A.jsonl <dir>/B.jsonl ...
"""
import json, sys
from pathlib import Path
import numpy as np
from tags import nv_hits

W = {"A": dict(nv=.3, wer=.15, mos=.15, ss=.1), "B": dict(nv=.3, wer=.1, mos=.1, ss=.2)}


def rows(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else None


for m in sys.argv[1:]:
    base = m[:-6]; T = "B" if Path(m).name.startswith("B") or "_B" in Path(m).name else "A"; w = W[T]
    sc, qa = rows(base + ".scored.jsonl"), rows(base + ".tags_qasr.jsonl")
    wer, mos, ss = (float(np.mean([r[k] for r in sc])) for k in ("wer", "pmos", "ss"))
    qa = {r["id"]: r for r in qa}; byid = {r["id"]: r for r in sc}
    hits = [ok for i, r in byid.items() if i in qa for _, ok in nv_hits(r["text"], qa[i]["tag_hyp"])]
    nv = float(np.mean(hits))
    off = (w["nv"] * nv + w["wer"] * (1 - wer) + w["mos"] * mos / 5 + w["ss"] * ss) / sum(w.values())
    out = dict(n=len(sc), WER=round(wer, 3), pMOS=round(mos, 2), SS=round(ss, 3), NV_v3=round(nv, 3), **{f"off{T}": round(off, 3)})
    nd = rows(base + ".nvdet.jsonl")
    if nd: out["NVj"] = round(float(np.mean([r["pred"] == r["tag"] for r in nd])), 3)
    print(m, json.dumps(out))
