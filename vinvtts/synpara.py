"""SynParaSpeech-style synthetic rare-class data (Bai et al., arXiv 2509.14946, ICASSP 2026): real paralinguistic event
audio -> zero-shot voice conversion of the EVENT ONLY to the host utterance's speaker (Seed-VC, as the paper; github
Plachtaa/seed-vc, GPL-3.0) -> merged into a real training utterance at a word gap -> tag inserted in the text.
Our additions: loudness matched to the per-class event/speech RMS ratio of our real events; automatic filter with the
two-detector agreement rule of row P5 (PretrainedSED stack AND AST-LR, both fitted on ALL real events) at the KNOWN inserted
span; tag-ASR v3 is NOT used (held-out judge).
Event source: VocalSound (Gong et al., ICASSP 2022; CC BY-SA 4.0; /mnt/vinvtts/data/vocalsound, 3504 clips per class).

  make   (Seed-VC venv)   : N per class hosts x events -> data/synpara/wav/*.wav + runs/synpara/cands.jsonl + .spans.jsonl
  (psed_score.py, PSED venv -> runs/synpara/cands.psed.jsonl)
  filter (project venv)   : AST features at the span + both detectors -> runs/synpara/kept.jsonl (train.jsonl rows)

    $SV synpara.py make --n 1500 ; $PS psed_score.py runs/synpara/cands.jsonl ; uv run synpara.py filter
"""
import argparse, glob, json, os, random, re, sys
from collections import defaultdict
import numpy as np

R = "/mnt/vinvtts/runs"; O = f"{R}/synpara"; W = "/mnt/vinvtts/data/synpara/wav"; SR = 24000
VS = "/mnt/vinvtts/data/vocalsound/audio_16k"; D = "/mnt/vinvtts/data/vinv-tts/train.jsonl"
TAG = re.compile(r"\[(laughter|breathing|sniff|throatclearing)\]")
# rule of row P5 (speaker-grouped OOF on real events): throat P .89 R .17; sniff P .80 R .11
RULE = {"throatclearing": (0.99, 0.5), "sniff": (0.999, 0.5)}


def split_words(text):
    words, tags = [], []
    for tok in TAG.sub(lambda m: f" [{m.group(1)}] ", text.lower()).split():
        m = TAG.fullmatch(tok)
        if m: tags.append((len(words), m.group(1)))
        elif re.sub(r"[^\w]", "", tok): words.append(tok)
    return words, tags


def with_tag(text, g, tag):   # = synth.with_tag (inlined: this runs in the Seed-VC venv)
    toks, n = TAG.sub(lambda m: f" {m.group(0)} ", text).split(), 0
    for i, t in enumerate(toks):
        if TAG.fullmatch(t): continue
        if re.sub(r"[^\w]", "", t.lower()):
            if n == g: return " ".join(toks[:i] + [f"[{tag}]"] + toks[i:])
            n += 1
    return " ".join(toks + [f"[{tag}]"])


def rms(x): return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def active(y, sr, thr_db=-30, pad=0.04, cap=0.9):   # real events: median .52 s, p75 .8 s
    """Trim a VocalSound clip to its active part (frames within thr_db of the peak), cap to the loudest `cap` s."""
    hop = int(0.01 * sr); n = len(y) // hop
    if n < 3: return y
    e = 20 * np.log10(np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9); on = np.flatnonzero(e > e.max() + thr_db)
    a, b = max(0, on[0] * hop - int(pad * sr)), min(len(y), (on[-1] + 1) * hop + int(pad * sr)); y = y[a:b]
    L = int(cap * sr)
    if len(y) > L:
        c = np.convolve(y ** 2, np.ones(L), "valid"); s = int(np.argmax(c)); y = y[s:s + L]
    return y


def durations():
    """Per-class real event durations (s) from our train event bank -> length target of each synthetic event
    (spot-check 09-27: a fixed trim cap made every event exactly 0.89 s, real median .52 s)."""
    out = defaultdict(list)
    for e in map(json.loads, open(f"{R}/events/events.jsonl")):
        if e["tag"] in RULE: out[e["tag"]].append(e["t1"] - e["t0"])
    return {k: np.array(v) for k, v in out.items()}


def ratios():
    """Per-class RMS(event span) / RMS(whole utterance) over our real train events (loudness target)."""
    import soundfile as sf
    out = defaultdict(list)
    for e in map(json.loads, open(f"{R}/events/events.jsonl")):
        if e["tag"] not in RULE: continue
        y, sr = sf.read(e["audio"], dtype="float32")
        seg = y[int(e["t0"] * sr): int(e["t1"] * sr)]
        if len(seg) > sr // 20: out[e["tag"]].append(rms(seg) / rms(y))
    return {k: np.array(v) for k, v in out.items()}


def make(a):
    import soundfile as sf, librosa, torch
    sys.path.insert(0, "/mnt/vinvtts/src/seedvc"); os.chdir("/mnt/vinvtts/src/seedvc")
    from seed_vc_wrapper import SeedVCWrapper
    vc = SeedVCWrapper()
    random.seed(a.seed); np.random.seed(a.seed); os.makedirs(O, exist_ok=True); os.makedirs(W, exist_ok=True)
    rows = [r for r in map(json.loads, open(D, encoding="utf-8")) if 3 <= r["dur"] <= 15]
    gaps = json.load(open(f"{R}/events/gaps.json")); rat = ratios(); durs = durations()
    print("event/speech RMS ratio medians", {k: round(float(np.median(v)), 3) for k, v in rat.items()}, flush=True)
    cands, spans = [], []; nskip = 0
    for tag in RULE:
        pool = sorted(glob.glob(f"{VS}/*_{tag}.wav"))
        for k in range(a.n):
            r = random.choice(rows); words, tg = split_words(r["text"]); used = {g for g, _ in tg}
            gp = gaps.get(r["audio"], {})
            free = [int(g) for g, (t0, t1) in gp.items() if int(g) not in used and int(g) - 1 not in used and int(g) + 1 not in used
                    and 0 < int(g) < len(words) and t1 - t0 >= 0.05]
            edge = [g for g in (0, len(words)) if g not in used and str(g) in gp]
            if not free and not edge: continue
            g = random.choice(edge) if edge and random.random() < 0.3 else random.choice(free or edge)
            t0, t1 = gp[str(g)]; y, sr = sf.read(r["audio"], dtype="float32")
            if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
            src = random.choice(pool); ev = active(librosa.load(src, sr=16000)[0], 16000, cap=float(np.clip(np.random.choice(durs[tag]), 0.15, 1.2)))
            if len(ev) < 1600: nskip += 1; continue   # empty / silent VocalSound clip (crashed Seed-VC's resampler, job 76)
            tmp = f"{W}/_src.wav"; sf.write(tmp, ev, 16000)
            try:
                gen = vc.convert_voice(tmp, r["audio"], diffusion_steps=a.steps, inference_cfg_rate=0.7, f0_condition=False, stream_output=False)
                try:
                    while True: next(gen)
                except StopIteration as e: out = e.value
            except Exception as e:
                nskip += 1; print("skip", src, type(e).__name__, flush=True); continue
            if out is None or len(out) < 400: continue
            out = librosa.resample(np.asarray(out, np.float32), orig_sr=22050, target_sr=SR)
            out = out * (np.random.choice(rat[tag]) * rms(y) / rms(out))
            f = int(0.01 * SR); ramp = np.linspace(0, 1, f, dtype=np.float32); out[:f] *= ramp; out[-f:] *= ramp[::-1]
            cut = int((t0 + t1) / 2 * SR); z = np.zeros(int(0.05 * SR), np.float32)
            wav = np.concatenate([y[:cut], z, out, z, y[cut:]]); ts = cut / SR + 0.05
            pk = float(np.abs(wav).max()); wav = wav * (0.95 / pk) if pk > 0.95 else wav   # peak-normalise, never hard-clip (19% of throat clips clipped)
            cid = f"sp_{tag[:5]}_{k:05d}"; p = f"{W}/{cid}.wav"; sf.write(p, wav, SR)
            text = with_tag(r["text"], g, tag)
            cands.append({"id": cid, "audio": p, "gen": p, "speaker": r["speaker"], "text": text, "dur": round(len(wav) / SR, 3),
                          "synth": "synpara", "host": r["audio"], "event": src, "gap": g})
            spans.append({"id": cid, "gen": p, "gap": g, "tag": tag, "t0": round(ts, 3), "t1": round(ts + len(out) / SR, 3)})
            if k % 200 == 0: print(tag, k, "/", a.n, flush=True)
    with open(f"{O}/cands.jsonl", "w", encoding="utf-8") as fh: fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in cands)
    with open(f"{O}/cands.spans.jsonl", "w", encoding="utf-8") as fh: fh.writelines(json.dumps(x) + "\n" for x in spans)
    print("synpara candidates", len(cands), "skipped", nskip)


def filt(a):
    import pickle
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from judge import load16k
    from nvdet import Embedder, seg
    from throat_det import feats, M, C
    rr = [[json.loads(l) for l in open(f"{R}/psed_{m}.jsonl")] for m in M]; r0 = rr[0]
    A = np.load(f"{R}/det_stack/ast.npy")
    lab = np.array([["none", "laughter", "breathing", "sniff", "throatclearing"].index(r["tag"]) for r in r0])
    ast = make_pipeline(StandardScaler(), PCA(128, random_state=0), LogisticRegression(C=0.5, max_iter=3000)).fit(A, lab)
    psed = {c: pickle.load(open(f"{R}/rare_judge/{c}.pkl", "rb")) for c in RULE}
    cands = {c["id"]: c for c in map(json.loads, open(f"{O}/cands.jsonl"))}
    ps = [json.loads(l) for l in open(f"{O}/cands.psed.jsonl")]; E = Embedder()
    segs = [seg(load16k(p["gen"]).clamp(-1, 1), p["t0"], p["t1"]) for p in ps]
    pa = ast.predict_proba(E(segs)); kept = []; stat = defaultdict(lambda: [0, 0])
    for p, q in zip(ps, pa):
        c = p["tag"]; tp, ta = RULE[c]; ci = ["none", "laughter", "breathing", "sniff", "throatclearing"].index(c)
        pp = float(psed[c].predict_proba(feats(p["probs"], p["t1"] - p["t0"])[None])[0, 1]); stat[c][0] += 1
        if pp >= tp and q[ci] >= ta:
            stat[c][1] += 1; x = cands[p["id"]]
            kept.append({k: x[k] for k in ("audio", "speaker", "text", "dur", "synth", "host", "event")})
    with open(f"{O}/kept.jsonl", "w", encoding="utf-8") as fh: fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in kept)
    print("SYNPARA kept", {c: f"{v[1]}/{v[0]}" for c, v in stat.items()}, "->", f"{O}/kept.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["make", "filter"])
    ap.add_argument("--n", type=int, default=1500); ap.add_argument("--steps", type=int, default=25); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); {"make": make, "filter": filt}[a.cmd](a)
