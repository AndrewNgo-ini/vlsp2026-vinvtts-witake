"""Laughter-targeted rejection-sampled distillation data (RS-SFT stage of arXiv 2609.11260; targeted per-class
augmentation of arXiv 2609.14231). Both engines (OmniVoice odpo_os, VieNeu dpo6b2) generate N candidates for every
training text that contains [laughter], in the row's own speaker voice but from a DIFFERENT clip of that speaker as
reference; a candidate is kept only if two independent acoustic detectors confirm every laughter tag
(LaughterSegmentation p >= LS_P at the tag gap AND nvdet argmax == laughter), nvdet realises every other tag (rare
tags with p >= .6, as rsft.py) and Zipformer WER <= .10. Best kept candidate per text by the Score-A quality terms.
Filter scorers = nvdet + LaughterSegmentation, so the evalF judge for models trained on this data is the arbiter
tag-ASR (tagasr_v2s), pre-registered.

    uv run distill.py prompts /mnt/vinvtts/data/vinv-tts/train.jsonl runs/distill/prompts.jsonl
    NBEST=4 <engine venv> gen_zs.py {omni|vieneu} runs/distill/prompts.jsonl runs/distill/c_<eng> <model>
    uv run distill.py filter runs/distill/kept.jsonl runs/distill/c_omni.jsonl runs/distill/c_vn.jsonl
"""
import cpulimit  # noqa: F401
import argparse, json, random, re, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import soundfile as sf
from tags import split

MAX_WER, P_RARE = 0.10, 0.6
strip = lambda t: " ".join(re.sub(r"\[[a-z]+\]", " ", t).split())


def prompts(a):
    random.seed(a.seed)
    rows = [json.loads(l) for l in open(a.inp)]
    by = defaultdict(list)
    for r in rows: by[r["speaker"]].append(r)
    out = []
    for i, r in enumerate(x for x in rows if "[laughter]" in x["text"]):
        cand = [x for x in by[r["speaker"]] if x["audio"] != r["audio"] and 3 <= x["dur"] <= 12] or [r]
        ref = random.choice(cand)
        out.append({"id": f"dl{i:04d}", "text": r["text"], "speaker": r["speaker"], "ref": ref["audio"],
                    "ref_text": strip(ref["text"]), "src": r["audio"], "self_ref": ref is r})
    with open(a.out, "w") as f: f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in out)
    print(len(out), "prompts;", sum(x["self_ref"] for x in out), "use their own clip as ref ->", a.out)


def trim(src, dst, sr_out=None, thr_db=-40, keep=0.0):
    """Drop leading/trailing digital silence and OmniVoice's zero pads (else prep_vieneu's pad-frame assert fires, as in
    rsft), keeping `keep` s of margin."""
    y, sr = sf.read(src, dtype="float32")
    win = int(0.01 * sr); n = len(y) // win
    e = 20 * np.log10(np.maximum(np.abs(y[: n * win]).reshape(n, win).max(1), 1e-8))
    on = np.flatnonzero(e > thr_db)
    if on.size: y = y[max(0, on[0] * win - int(keep * sr)): min(len(y), (on[-1] + 1) * win + int(keep * sr))]
    sf.write(dst, y, sr)
    return len(y) / sr


def filt(a):
    py = sys.executable
    for man in a.cands:
        base = man[:-len(".jsonl")]
        for suf, cmd in ((".scored.jsonl", ["bench.py", man]), (".nvdet.jsonl", ["nvdet.py", "score", "--model", a.nvdet, man]),
                         (".laugh.jsonl", ["laugh_check.py", man])):
            if not Path(base + suf).exists(): subprocess.run([py] + cmd, check=True)
    ok_by = defaultdict(list); stats = Counter()
    for man in a.cands:
        base = man[:-len(".jsonl")]; eng = Path(man).stem
        sc = {r["id"]: r for r in map(json.loads, open(base + ".scored.jsonl"))}
        nd = defaultdict(dict)
        for r in map(json.loads, open(base + ".nvdet.jsonl")): nd[r["id"]][r["gap"]] = r
        ls = defaultdict(dict)
        for r in map(json.loads, open(base + ".laugh.jsonl")): ls[r["id"]][r["gap"]] = r["p"]
        for i, r in sc.items():
            tags = split(r["text"])[1]; stats[eng, "cands"] += 1
            def tag_ok(g, t):
                d = nd[i].get(g)
                if d is None or d["pred"] != t: return False
                if t == "laughter": return ls[i].get(g, 0.0) >= a.ls_p
                return t == "breathing" or d["p_tag"] >= P_RARE
            lok = all(tag_ok(g, t) for g, t in tags if t == "laughter"); stats[eng, "laugh_ok"] += lok
            if lok and all(tag_ok(g, t) for g, t in tags) and r["wer"] <= MAX_WER:
                stats[eng, "kept_cand"] += 1; ok_by[r["base_id"]].append({**r, "eng": eng})
    out = Path(a.out).with_suffix(""); out.mkdir(parents=True, exist_ok=True); kept = []
    for bid, cs in ok_by.items():
        c = max(cs, key=lambda c: 0.1 * (1 - c["wer"]) + 0.1 * c["pmos"] / 5 + 0.1 * c["ss"])
        dst = out / f"{bid}.wav"; d = trim(c["gen"], dst)
        kept.append({"audio": str(dst), "speaker": c["speaker"], "text": c["text"], "dur": round(d, 3), "synth": "distill",
                     "eng": c["eng"], "src": c.get("src")})
    print("stats", dict(stats))
    print(f"kept {len(kept)} texts; by engine {dict(Counter(k['eng'] for k in kept))}; tags {dict(Counter(t for k in kept for _, t in split(k['text'])[1]))}")
    with open(a.out, "w") as f: f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in kept)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prompts", "filter"]); ap.add_argument("inp"); ap.add_argument("cands", nargs="*")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--nvdet", default="/mnt/vinvtts/runs/nvdet")
    ap.add_argument("--ls-p", type=float, default=0.5)
    if sys.argv[1] == "retrim":   # distill.py retrim kept.jsonl: re-trim kept wavs in place (first run kept 50 ms at -50 dB
        rows = [json.loads(l) for l in open(sys.argv[2])]   # -> 159/289 rows ended on the silence code, pad-frame assert)
        for r in rows: r["dur"] = round(trim(r["audio"], r["audio"]), 3)
        with open(sys.argv[2], "w") as f: f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
        print("retrimmed", len(rows)); raise SystemExit
    a = ap.parse_args()
    if a.cmd == "prompts": a.out = a.cands[0]
    else: a.out = a.inp
    {"prompts": prompts, "filter": filt}[a.cmd](a)
