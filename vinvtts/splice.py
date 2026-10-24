"""NV splicing post-processor: guarantee the right NV type at each tag position.

bank : cut every located train event (align.py event bank), trim to its active part, keep clean/audible ones
       (nvdet says its own class, SNR vs utterance floor, duration), store clip + ECAPA of its source utterance.
apply: TTS wav + tagged text -> Zipformer-align (nvdet.gen_gaps) -> per tag gap, if policy says so (always, or the
       nvdet head says the tag is missing there) replace the gap interior with pause + bank clip + pause, loudness-matched
       to the host speech (clip keeps its original event/utterance RMS ratio), 15 ms crossfades.
       Clip choice: same speaker if it has clips of that class (Track A), else most ECAPA-similar source speaker.

    uv run splice.py bank --out /data/vinvtts/runs/splice/bank
    uv run splice.py run --bank .../bank --policy missing base.jsonl out_dir      (writes out_dir.jsonl, gen -> spliced)
Server: sp = Splicer(bank_dir); y = sp(y, sr, text, ref_wav_path, speaker=None, policy={"sniff": "missing", ...})
"""
import cpulimit  # noqa: F401
import argparse, json, os, pickle, zlib
from collections import defaultdict
from pathlib import Path
import numpy as np, soundfile as sf, torch, torchaudio
from tags import split

CLS_POL = ("always", "missing", "never")
XF = 0.015          # s crossfade
PRE, POST = 0.08, 0.12   # s pauses around the inserted event (real events sit in short pauses)
EVENTS = "/data/vinvtts/backup/data/events/events.jsonl"
NVDET = "/data/vinvtts/runs/nvdet"
MINP = {"breathing": .5, "laughter": .5, "sniff": .4, "throatclearing": .4}


def rms(x): return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def frame_db(y, sr, hop=0.02):
    h = int(hop * sr); n = max(1, len(y) // h)
    return 20 * np.log10(np.sqrt((y[: n * h].reshape(n, h) ** 2).mean(1)) + 1e-9), h


def active(y, sr, thr=-32, pad=0.03, cap=1.2):
    e, h = frame_db(y, sr, 0.01); on = np.flatnonzero(e > e.max() + thr)
    if len(on) == 0: return y
    a, b = max(0, on[0] * h - int(pad * sr)), min(len(y), (on[-1] + 1) * h + int(pad * sr)); y = y[a:b]
    L = int(cap * sr)
    if len(y) > L:
        c = np.convolve(y ** 2, np.ones(L), "valid"); s = int(np.argmax(c)); y = y[s:s + L]
    return y


def speech_rms(y, sr):
    e, h = frame_db(y, sr); on = e > e.max() - 30
    return float(np.sqrt(np.mean(10 ** (e[on] / 10)))) if on.any() else rms(y)


def fix_path(p): return p.replace("/mnt/vinvtts/data/vinv-tts/", "/data/vinvtts/data/vinv-tts/")


def ecapa():
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cuda"})


def emb_of(enc, y, sr):
    x = torch.tensor(y, dtype=torch.float32)
    if sr != 16000: x = torchaudio.functional.resample(x, sr, 16000)
    with torch.no_grad(): e = enc.encode_batch(x[None].cuda()).squeeze().float().cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-9)


def build_bank(a):
    from nvdet import Embedder, seg, CLS
    from judge import load16k
    out = Path(a.out); (out / "clips").mkdir(parents=True, exist_ok=True)
    clf = pickle.load(open(f"{NVDET}/clf.pkl", "rb")); emb = Embedder(); enc = ecapa()
    ev = [json.loads(l) for l in open(EVENTS)]
    by = defaultdict(list)
    for e in ev: by[fix_path(e["audio"])].append(e)
    rows, spk_emb = [], defaultdict(list)
    for i, (p, es) in enumerate(by.items()):
        y, sr = sf.read(p, dtype="float32"); y = y if y.ndim == 1 else y.mean(1)
        w16 = load16k(p)
        P = clf.predict_proba(emb([seg(w16, e["t0"], e["t1"]) for e in es]))
        ue = emb_of(enc, y, sr); spk_emb[es[0]["speaker"]].append(ue)
        floor = np.percentile(frame_db(y, sr)[0], 10); sp = speech_rms(y, sr)
        for e, pr in zip(es, P):
            c = active(y[int(e["t0"] * sr):int(e["t1"] * sr)], sr)
            d = len(c) / sr; snr = 20 * np.log10(rms(c) + 1e-9) - floor; pt = float(pr[CLS.index(e["tag"])])
            ok = pt >= MINP[e["tag"]] and snr >= 12 and .12 <= d <= 1.2
            if not ok: continue
            name = f"{e['speaker']}_{Path(p).stem}_{e['gap']}_{e['tag']}.wav"
            sf.write(out / "clips" / name, c, sr)
            rows.append({"clip": str(out / "clips" / name), "speaker": e["speaker"], "tag": e["tag"], "dur": round(d, 3),
                         "p": round(pt, 3), "snr": round(float(snr), 1), "ratio": round(rms(c) / sp, 4), "src": p})
        if i % 200 == 0: print(i, len(by), len(rows), flush=True)
    with open(out / "bank.jsonl", "w") as f: f.writelines(json.dumps(r) + "\n" for r in rows)
    np.save(out / "spk_emb.npy", {k: np.mean(v, 0) / np.linalg.norm(np.mean(v, 0)) for k, v in spk_emb.items()})
    from collections import Counter
    print("kept", Counter(r["tag"] for r in rows), "of", Counter(e["tag"] for e in ev))


class Splicer:
    def __init__(self, bank_dir, rec=None, device="cuda", pre=PRE, post=POST, xf=XF, loud="ratio", pick="p",
                 breath_nopause=False):
        self.union = False
        self.pre, self.post, self.xf, self.loud, self.pmode, self.bnp = pre, post, xf, loud, pick, breath_nopause
        from align import recognizer
        self.rec = rec or recognizer()   # before any CUDA init (sherpa-onnx)
        from nvdet import Embedder
        self.clf = pickle.load(open(f"{NVDET}/clf.pkl", "rb")); self.emb = Embedder(); self.enc = ecapa()
        self.bank = [json.loads(l) for l in open(f"{bank_dir}/bank.jsonl")]
        for b in self.bank:   # bank moved (fresh box / HF download): clips live next to bank.jsonl
            if not os.path.exists(b["clip"]): b["clip"] = f"{bank_dir}/clips/{os.path.basename(b['clip'])}"
        self.spk = np.load(f"{bank_dir}/spk_emb.npy", allow_pickle=True).item()
        self.ref_cache = {}
        med = defaultdict(list)
        for b in self.bank: med[b["tag"]].append(b["ratio"])
        self.med = {k: float(np.median(v)) for k, v in med.items()}

    def detect(self, w16, gp, tags):
        from nvdet import seg, CLS
        segs = [seg(w16, *gp[g]) if g in gp else w16[-4000:] for g, _ in tags]
        P = self.clf.predict_proba(self.emb(segs))
        return [(CLS[p.argmax()], float(p[CLS.index(t)])) for p, (_, t) in zip(P, tags)]

    def pick(self, tag, speaker, ref, key, ctx_db=None):
        cand = [b for b in self.bank if b["tag"] == tag]
        vc = [b for b in cand if b.get("ref") == ref]   # Seed-VC-converted clips for this prompt (vc bank)
        if vc: cand, speaker = vc, None
        same = [b for b in cand if b["speaker"] == speaker]
        if vc: pool = cand
        elif len(same) >= 2 or (same and speaker):
            pool = same
        else:
            if ref not in self.ref_cache:
                y, sr = sf.read(ref, dtype="float32"); y = y if y.ndim == 1 else y.mean(1)
                self.ref_cache[ref] = emb_of(self.enc, y, sr)
            re_ = self.ref_cache[ref]
            sims = sorted({b["speaker"] for b in cand}, key=lambda s: -float(self.spk[s] @ re_))
            top = set(sims[:3]); pool = [b for b in cand if b["speaker"] in top]
        if self.pmode == "energy" and ctx_db is not None and "edge_db" in pool[0]:
            # continuity: clip whose onset level (rel. to its source speech) matches the host's local level at the gap
            pool = sorted(pool, key=lambda b: abs(b["edge_db"] - ctx_db) - 3 * b["p"])[:4]
            return pool[zlib.crc32(key.encode()) % len(pool)]
        pool = sorted(pool, key=lambda b: -b["p"])[:8]
        return pool[zlib.crc32(key.encode()) % len(pool)]

    def __call__(self, y, sr, text, ref, speaker=None, policy="missing", key="", force=None):
        """-> (y_out, info). policy: str or {tag: always|missing|never}."""
        from nvdet import gen_gaps
        import torchaudio.functional as AF
        pol = policy if isinstance(policy, dict) else {}; dflt = "missing" if isinstance(policy, dict) else policy
        w16 = AF.resample(torch.tensor(y, dtype=torch.float32), sr, 16000).clamp(-1, 1) if sr != 16000 else torch.tensor(y).clamp(-1, 1)
        gp, tags = gen_gaps(self.rec, w16, text)
        n_words = len(split(text)[0])
        if not tags: return y, []
        det = self.detect(w16, gp, tags)
        sp = speech_rms(y, sr); floor_rms = 10 ** (np.percentile(frame_db(y, sr)[0], 10) / 20)
        edits, info = [], []
        for (g, t), (pred, pt) in zip(tags, det):
            p = pol.get(t, dflt)
            miss = pred != t if force is None else ((g, t) in force or (self.union and pred != t))   # force = external judge's missing set
            do = p == "always" or (p == "missing" and miss)
            info.append({"gap": g, "tag": t, "pred": pred, "p_tag": round(pt, 3), "spliced": bool(do)})
            if not do: continue
            t0, t1 = gp.get(g, (len(y) / sr, len(y) / sr))
            if g >= n_words: t1 = max(t1, t0)  # sentence-final: insert after last word
            a, b = t0 + 0.02, max(t0 + 0.02, t1 - 0.02)   # replace gap interior, keep word edges
            if g == 0: a = min(a, b)
            e_, h_ = frame_db(y, sr); fi = min(len(e_) - 1, int(t0 * sr / h_)); ctx = float(e_[max(0, fi - 5):fi + 1].mean() - 20 * np.log10(sp))
            c = self.pick(t, speaker, ref, f"{key}:{g}", ctx)
            clip, csr = sf.read(c["clip"], dtype="float32")
            if csr != sr: clip = AF.resample(torch.tensor(clip), csr, sr).numpy()
            r_ = c["ratio"] if self.loud == "ratio" else self.med[t] if self.loud == "median" else c["ratio"] * float(self.loud)
            clip = clip * (r_ * sp / (rms(clip) + 1e-9))
            info[-1].update(clip=Path(c["clip"]).name, t0=round(a, 3), t1=round(b, 3))
            nop = self.bnp and t == "breathing" and 0 < g < n_words
            edits.append((a, b, clip, nop))
        if not edits: return y, info
        rng = np.random.default_rng(zlib.crc32(key.encode()))
        pause = lambda s: (rng.standard_normal(int(s * sr)) * floor_rms * .7).astype(np.float32)
        parts, pos = [], 0
        for a, b, clip, nop in sorted(edits, key=lambda e: e[0]):
            ia, ib = int(a * sr), int(b * sr)
            ia = max(ia, pos)
            pr, po = (0, 0) if nop else (self.pre, self.post)
            parts += [y[pos:ia], pause(pr), clip.astype(np.float32), pause(po)]; pos = max(ib, ia)
        parts.append(y[pos:])
        return xjoin([p for p in parts if len(p)], int(self.xf * sr)), info


def xjoin(parts, xf):
    out = parts[0]
    for p in parts[1:]:
        k = min(xf, len(out) // 2, len(p) // 2)
        if k > 0:
            w = np.linspace(0, 1, k, dtype=np.float32)
            out = np.concatenate([out[:-k], out[-k:] * (1 - w) + p[:k] * w, p[k:]])
        else: out = np.concatenate([out, p])
    return out


def judge_missing(base, mode):
    """Missing set per id from nvdet-independent judges already run on the base (w1score.sh):
    v3 = tag-ASR v3 misses the tag; ens = v3 miss AND class judge miss (laughter: LS p<.5, sniff/throat: PSED hit=False,
    breathing: v3 only); any = v3 miss OR class judge miss."""
    from tags import nv_hits
    L = lambda s, k: {k(r): r for r in map(json.loads, open(f"{base}.{s}.jsonl"))} if Path(f"{base}.{s}.jsonl").exists() else {}
    qa = L("tags_qasr", lambda r: r["id"]); ls = L("laugh", lambda r: (r["id"], r["gap"])); rj = L("rare", lambda r: (r["id"], r["gap"]))
    out = defaultdict(set)
    for r in map(json.loads, open(base + ".jsonl")):
        tg = split(r["text"])[1]
        v3 = [ok for _, ok in nv_hits(r["text"], qa[r["id"]]["tag_hyp"])] if r["id"] in qa else [False] * len(tg)
        for (g, t), ok in zip(tg, v3):
            if t == "laughter": cj = ls.get((r["id"], g), {}).get("p", 0) >= .5
            elif t in ("sniff", "throatclearing"): cj = bool(rj.get((r["id"], g), {}).get("hit", False))
            else: cj = ok
            m = {"v3": not ok, "ens": not ok and not cj, "any": not ok or not cj}[mode]
            if m: out[r["id"]].add((g, t))
    return out


def run(a):
    sp = Splicer(a.bank, pre=a.pre, post=a.post, xf=a.xf, loud=a.loud, pick=a.pick, breath_nopause=a.bnp)
    sp.union = a.decide.endswith("+nvdet")
    force = judge_missing(a.base[:-6], a.decide.replace("+nvdet", "")) if a.decide != "nvdet" else None
    rows = [json.loads(l) for l in open(a.base)]
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True); res = []
    pol = json.loads(a.policy) if a.policy.startswith("{") else a.policy
    for r in rows:
        y, sr = sf.read(r["gen"], dtype="float32"); y = y if y.ndim == 1 else y.mean(1)
        spk = r["speaker"] if a.track == "A" else None
        z, info = sp(y, sr, r["text"], r["ref"], speaker=spk, policy=pol, key=r["id"],
                     force=None if force is None else force.get(r["id"], set()))
        o = od / f"{r['id']}.wav"; sf.write(o, np.clip(z, -1, 1), sr)
        res.append({**r, "gen": str(o), "base_gen": r["gen"], "splice": info})
    with open(str(od) + ".jsonl", "w") as f: f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in res)
    n = sum(i["spliced"] for x in res for i in x["splice"]); m = sum(len(x["splice"]) for x in res)
    print(f"SPLICED {n}/{m} tags -> {od}.jsonl", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["bank", "run"])
    ap.add_argument("base", nargs="?"); ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--out"); ap.add_argument("--bank", default="/data/vinvtts/runs/splice/bank")
    ap.add_argument("--policy", default="missing"); ap.add_argument("--track", default="A")
    ap.add_argument("--pre", type=float, default=PRE); ap.add_argument("--post", type=float, default=POST)
    ap.add_argument("--xf", type=float, default=XF); ap.add_argument("--loud", default="ratio"); ap.add_argument("--pick", default="p")
    ap.add_argument("--bnp", action="store_true"); ap.add_argument("--decide", default="nvdet", choices=["nvdet", "v3", "ens", "any", "any+nvdet", "v3+nvdet"])
    a = ap.parse_args()
    if a.cmd == "run": a.out = a.out_dir
    {"bank": build_bank, "run": run}[a.cmd](a)
