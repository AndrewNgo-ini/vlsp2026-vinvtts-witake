"""NV loudness vs surrounding speech (user, 10-01: "the decibel between the text and the non-verbal is very detectable").
For every tag gap: Zipformer-align (nvdet.gen_gaps) -> NV span; level = 90th-percentile 20 ms frame RMS (dBFS) inside
the span vs in 1.0 s of speech on each side (span excluded). dLvl = NV - context (dB). Compare systems with the real
recordings (dev GT) per class; the "uncanny" gap = how far a system's dLvl distribution sits from GT's.
    uv run nvloud.py <name>=<manifest.jsonl> ...      (manifest rows: gen|audio, text) -> table + <manifest>.loud.jsonl
"""
import cpulimit  # noqa: F401
import json, sys
from collections import defaultdict
import numpy as np
from judge import load16k, SR
from nvdet import gen_gaps

CTX, FR, EDGE = 1.0, int(0.02 * SR), int(0.06 * SR)


def lvl(x):
    if len(x) < FR: return None
    f = x[: len(x) // FR * FR].reshape(-1, FR); r = np.sqrt((f ** 2).mean(1) + 1e-10)
    return float(20 * np.log10(np.percentile(r, 90) + 1e-10))


def shape(x):
    """NV envelope shape inside its span: number of energy bursts (peaks >= max-12 dB separated by a >= 6 dB dip) and the
    steepest 20 ms-frame rise / fall in dB per 10 ms (onset/offset abruptness)."""
    if len(x) < 3 * FR: return 1, 0.0, 0.0
    hop = FR // 2; n = (len(x) - FR) // hop + 1
    e = np.array([20 * np.log10(np.sqrt((x[i * hop:i * hop + FR] ** 2).mean()) + 1e-6) for i in range(n)])
    e = np.convolve(e, np.ones(3) / 3, mode="same"); top = e.max(); e = np.maximum(e, top - 25); bursts, inb, lo = 0, False, top
    for v in e:
        if not inb and v >= top - 12 and (bursts == 0 or v - lo >= 6): bursts += 1; inb = True
        elif inb and v < top - 18: inb, lo = False, v
        elif not inb: lo = min(lo, v)
    de = np.diff(e)   # dB per hop (10 ms)
    return bursts, float(de.max(initial=0)), float(-de.min(initial=0))


def measure(rec, wav, text):
    gp, tags = gen_gaps(rec, wav, text)
    return features(wav, gp, tags)


def features(wav, gp, tags):
    """per-tag level/shape features given alignment gaps (so callers that already aligned don't align twice)"""
    w = wav.numpy() if hasattr(wav, "numpy") else wav; out = []
    for g, t in tags:
        if g not in gp: continue
        t0, t1 = gp[g]; a, b = int(t0 * SR), int(t1 * SR)
        if b - a < FR: continue
        ctx = np.concatenate([w[max(0, a - int(CTX * SR)):a], w[b:b + int(CTX * SR)]])
        n, c = lvl(w[a:b]), lvl(ctx)
        # boundary abruptness: level step across each span edge (60 ms inside vs 60 ms outside), larger of the two
        e = [lvl(w[a:a + EDGE]), lvl(w[max(0, a - EDGE):a]), lvl(w[max(a, b - EDGE):b]), lvl(w[b:b + EDGE])]
        jump = max(abs(e[0] - e[1]) if None not in e[:2] else 0.0, abs(e[2] - e[3]) if None not in e[2:] else 0.0)
        if n is not None and c is not None: out.append({"gap": g, "tag": t, "nv_db": round(n, 1), "ctx_db": round(c, 1), "d": round(n - c, 1), "dur": round(t1 - t0, 2), "jump": round(jump, 1), **dict(zip(("bursts", "rise", "fall"), (lambda t: (t[0], round(t[1], 1), round(t[2], 1)))(shape(w[a:b]))))})
    return out


if __name__ == "__main__":
    from align import recognizer
    rec = recognizer()
    res = {}
    for arg in sys.argv[1:]:
        name, man = arg.split("=", 1); per = defaultdict(list); rows_out = []
        for r in map(json.loads, open(man)):
            m = measure(rec, load16k(r.get("gen") or r["audio"]), r["text"])
            rows_out.append({"id": r.get("id", r.get("audio")), "nv": m})
            for x in m: per[x["tag"]].append(x["d"])
        open(man.replace(".jsonl", ".loud.jsonl"), "w").writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows_out)
        res[name] = per
    tags = ["breathing", "laughter", "sniff", "throatclearing"]
    print("dLvl = NV level - context speech level (dB); median [IQR] n")
    print("system".ljust(12) + "".join(t[:12].ljust(24) for t in tags))
    for name, per in res.items():
        cells = []
        for t in tags:
            v = np.array(per.get(t, []))
            cells.append((f"{np.median(v):+.1f} [{np.percentile(v, 25):+.1f},{np.percentile(v, 75):+.1f}] {len(v)}" if len(v) else "-").ljust(24))
        print(name.ljust(12) + "".join(cells))
    json.dump({n: {t: v for t, v in p.items()} for n, p in res.items()}, open("/data/vinvtts/runs/nvloud_summary.json", "w"))
