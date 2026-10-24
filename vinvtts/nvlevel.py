"""NV level matcher (fix for the "uncanny" loud NVs, 2026-10-01). Our NVs sit 6-12 dB above where real recordings put them
relative to the surrounding speech (nvloud.py: GT laughter/throat ~-3.5 dB, sniff ~-18 dB; ours +3 / +3..+4 / -6..-10),
because tag-CFG, judge-ranked DPO and best-of-N all reward detectability.
Per tag: Zipformer-align the NV span (nvdet.gen_gaps), measure dLvl like nvloud.py, and if it is louder than the real
per-class target by more than TOL dB, attenuate that span (never boost) with raised-cosine ramps. Breathing is left alone.
    uv run nvlevel.py <run_dir>/<T>.jsonl <out_dir> [--targets laughter=-3.4,throatclearing=-3.5,sniff=-18] [--tol 2 --max-cut 12]
-> <out_dir>/<T>/<id>.wav + <out_dir>/<T>.jsonl (gen re-pointed) + per-tag gains in "nvlevel"
"""
import cpulimit  # noqa: F401
import argparse, json
from pathlib import Path
import numpy as np, soundfile as sf
from nvloud import measure, FR
from judge import SR as SR16

DEF = {"laughter": -3.4, "throatclearing": -3.5, "sniff": -18.0}
RAMP = 0.04


def parse(s):
    return {("throatclearing" if k.startswith("throat") else k): float(v) for k, v in (kv.split("=") for kv in s.split(","))} if s else DEF


class QMap:
    """1-D optimal transport per class: target(d) = Q_gt(F_sys(d)). F_sys/Q_gt from nvloud.py .loud.jsonl files."""
    def __init__(self, sys_loud, gt_loud):
        import collections
        def col(p):
            c = collections.defaultdict(list)
            for r in map(json.loads, open(p)):
                for x in r["nv"]: c[x["tag"]].append(x["d"])
            return {k: np.sort(v) for k, v in c.items()}
        self.s, self.g = col(sys_loud), col(gt_loud)

    def get(self, tag, d=None):
        if tag not in self.s or tag not in self.g or tag == "breathing": return None
        if d is None: return float(np.median(self.g[tag]))
        p = np.searchsorted(self.s[tag], d, side="right") / len(self.s[tag])
        return float(np.quantile(self.g[tag], min(max(p, 0.0), 1.0)))


def level(w, sr, text, rec, targets, tol=2.0, max_cut=12.0, alpha=1.0):
    """w: float32 mono at sr. Spans are measured on a 16 kHz copy, gains applied at sr."""
    import librosa
    w16 = librosa.resample(w, orig_sr=sr, target_sr=SR16) if sr != SR16 else w
    import torch
    m = measure(rec, torch.from_numpy(w16.astype(np.float32)), text)
    from nvdet import gen_gaps
    gp, _ = gen_gaps(rec, torch.from_numpy(w16.astype(np.float32)), text)
    g = np.ones(len(w), dtype=np.float32); log = []
    for x in m:
        tgt = targets.get(x["tag"], x["d"]) if isinstance(targets, QMap) else targets.get(x["tag"])
        if tgt is None or x["d"] <= tgt + tol: log.append({**x, "cut": 0.0}); continue
        al = alpha.get(x["tag"], 1.0) if isinstance(alpha, dict) else alpha
        cut = min(max_cut, al * (x["d"] - tgt)); t0, t1 = gp[x["gap"]]   # alpha < 1: partial transport toward the target
        a, b = int(t0 * sr), int(t1 * sr); r = min(int(RAMP * sr), max(1, (b - a) // 4))
        env = np.full(b - a, 10 ** (-cut / 20), dtype=np.float32)
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, r))          # 0 -> 1
        env[:r] = 1 - ramp * (1 - env[r] if r < len(env) else 1); env[-r:] = env[:r][::-1]
        g[a:b] = np.minimum(g[a:b], env); log.append({**x, "cut": round(float(cut), 1)})
    return w * g, log


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("manifest"); ap.add_argument("out")
    ap.add_argument("--targets", default=""); ap.add_argument("--tol", type=float, default=2.0); ap.add_argument("--max-cut", type=float, default=12.0)
    ap.add_argument("--alpha", default="1.0", help="partial transport fraction: a number, or per class e.g. throatclearing=0.6 (others 1)")
    ap.add_argument("--qmap", nargs=2, metavar=("SYS_LOUD", "GT_LOUD"), help="quantile-map (OT) targets instead of fixed medians")
    a = ap.parse_args()
    from align import recognizer
    rec = recognizer(); tg = QMap(*a.qmap) if a.qmap else parse(a.targets)
    T = Path(a.manifest).stem; od = Path(a.out) / T; od.mkdir(parents=True, exist_ok=True); rows = []
    for r in map(json.loads, open(a.manifest)):
        w, sr = sf.read(r["gen"], dtype="float32")
        if w.ndim > 1: w = w.mean(1)
        al = parse(a.alpha) if "=" in a.alpha else float(a.alpha)
        y, log = level(w, sr, r["text"], rec, tg, a.tol, a.max_cut, al)
        p = od / f"{r['id']}.wav"; sf.write(p, y, sr)
        rows.append({**r, "gen": str(p), "nvlevel": log})
    open(Path(a.out) / f"{T}.jsonl", "w").writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
    cuts = [x["cut"] for r in rows for x in r["nvlevel"] if x["cut"] > 0]
    print(T, len(rows), "rows;", len(cuts), "tags attenuated; median cut", round(float(np.median(cuts)), 1) if cuts else 0, "dB")
