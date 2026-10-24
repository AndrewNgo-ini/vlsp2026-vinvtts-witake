"""Independent laughter check (LaughterSegmentation, Omine et al. IS2024; AUC .97 on our dev per the crawl audit): for
each [laughter] tag, max frame laughter probability within the tag's word gap (+-0.3 s) of the system's audio.
Reports mean prob and the fraction >= 0.5 per system -- settles nvdet (.18) vs tag-ASR (.48) disagreement.
Writes <sys>.laugh.jsonl (id, gap, pos, p) next to each manifest; `tag_probs` is reused by distill.py.

    uv run laugh_check.py runs/evalF/gt.jsonl runs/zsF/dpo6b2.jsonl ...
"""
import cpulimit  # noqa: F401
import json, sys
import numpy as np
from judge import load16k, SR
from tags import split as _split


def tag_probs(rec, L, y, text):
    """-> [(gap, pos, p)] for every [laughter] tag of `text` in 16 kHz wav tensor `y`."""
    import nvdet
    gp, tags = nvdet.gen_gaps(rec, y, text); nw = len(_split(text)[0])
    P = L(y.numpy()); fps = len(P) / (len(y) / SR); out = []
    for g, t in tags:
        if t != "laughter": continue
        t0, t1 = gp.get(g, (len(y) / SR - 1.0, len(y) / SR))
        a, b = max(0, int((t0 - 0.3) * fps)), min(len(P), int((t1 + 0.3) * fps) + 1)
        out.append((g, "start" if g <= 1 else "end" if g >= nw - 1 else "mid", float(np.max(P[a:b])) if b > a else 0.0))
    return out


if __name__ == "__main__":
    from align import recognizer
    rec = recognizer()   # sherpa before any CUDA model
    from yt_detect import Laugh
    L = Laugh()
    FP = "--fp" in sys.argv
    for man in [m for m in sys.argv[1:] if m != "--fp"]:
        rows = [json.loads(l) for l in open(man)]
        if FP:   # spurious laughter: utterances WITHOUT a [laughter] tag whose max frame laughter prob >= .5 anywhere
            mx = [float(np.max(L(load16k(r["gen"]).clamp(-1, 1).numpy()))) for r in rows if "[laughter]" not in r["text"]]
            print(f"LAUGHFP {man.split('/')[-1]:20s} n_untagged_utts={len(mx)} frac_max>=.5={np.mean(np.array(mx) >= .5):.3f} mean_max={np.mean(mx):.3f}", flush=True)
            continue
        ps, pos, per = [], [], []
        for r in rows:
            if "[laughter]" not in r["text"]: continue
            for g, po, p in tag_probs(rec, L, load16k(r["gen"]).clamp(-1, 1), r["text"]):
                ps.append(p); pos.append(po); per.append({"id": r["id"], "gap": g, "pos": po, "p": round(p, 4)})
        with open(man.replace(".jsonl", ".laugh.jsonl"), "w") as f: f.writelines(json.dumps(x) + "\n" for x in per)
        ps = np.array(ps); pos = np.array(pos)
        bypos = " ".join(f"{k}={np.mean(ps[pos == k] >= .5):.2f}(n{int((pos == k).sum())})" for k in ["start", "mid", "end"])
        print(f"LAUGHPOS {man.split('/')[-1]:20s} frac>=.5 by position: {bypos}", flush=True)
        print(f"LAUGH {man.split('/')[-1]:20s} n={len(ps)} mean_p={ps.mean():.3f} frac>=.5={np.mean(ps >= .5):.3f} frac>=.3={np.mean(ps >= .3):.3f}", flush=True)
