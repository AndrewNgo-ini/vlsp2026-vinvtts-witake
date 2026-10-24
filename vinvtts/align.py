"""Event bank: locate every tagged non-verbal event in the real corpus.
Zipformer (sherpa-onnx) token timestamps -> word start times -> Levenshtein-align recognised words to the reference
words -> each tag sits in a word gap, so its audio is [end of the word before, start of the word after].
Writes events.jsonl rows {audio, speaker, tag, t0, t1, gap, n_words} and gaps.json {audio: {ref_gap: [t0, t1]}} (time span
of every word gap whose neighbours both aligned) for synth.py to splice events into.

    uv run align.py /mnt/vinvtts/data/vinv-tts/train.jsonl /mnt/vinvtts/runs/events
"""
import cpulimit  # noqa: F401  (must precede torch/onnxruntime use)
import json, sys
from collections import Counter
from pathlib import Path
import jiwer, numpy as np
from judge import load16k, SR
from tags import split

TOK_END = 0.2  # s from a word's last token start to its end (Zipformer frame = 40 ms; Vietnamese syllables ~0.2 s)
MIN_EVT, MAX_EVT = 0.12, 3.0  # s; outside this the gap is an alignment failure, not an event


def recognizer():
    import sherpa_onnx
    from huggingface_hub import snapshot_download
    d = snapshot_download("csukuangfj2/sherpa-onnx-zipformer-vi-30M-2026-02-09")
    return sherpa_onnx.OfflineRecognizer.from_transducer(encoder=f"{d}/encoder.onnx", decoder=f"{d}/decoder.onnx",
        joiner=f"{d}/joiner.onnx", tokens=f"{d}/tokens.txt", num_threads=min(2, int(__import__("os").environ.get("OMP_NUM_THREADS", "2"))), sample_rate=SR, feature_dim=80)


def words_with_times(rec, wav):
    """-> [(word, start_s, end_s)]. The transducer only emits token start times, so a word ends TOK_END after its last
    token starts (capped at the next word's start); a breath/laugh between words then shows up as a real gap."""
    st = rec.create_stream(); st.accept_waveform(SR, wav.numpy()); rec.decode_stream(st)
    toks, ts = st.result.tokens, st.result.timestamps
    ws = []
    for t, s in zip(toks, ts):
        new = t[:1] in (" ", "▁")   # this model marks word starts with a leading space (sentencepiece: "▁")
        t = "".join(t.lower().replace("▁", " ").split())
        if new or not ws: ws.append([t, s, s])
        else: ws[-1][0] += t; ws[-1][2] = s
    dur = len(wav) / SR
    return [(w, s, min(ws[i + 1][1] if i + 1 < len(ws) else dur, last + TOK_END)) for i, (w, s, last) in enumerate(ws) if w]


def events(rec, row):
    wav = load16k(row["audio"]); dur = len(wav) / SR
    hw = words_with_times(rec, wav)
    rw, tags = split(row["text"])
    if not hw or not rw: return [], {}
    # ref word index -> hyp word index for aligned (equal/substitute) words
    m = {}
    for c in jiwer.process_words(" ".join(rw), " ".join(w for w, _, _ in hw)).alignments[0]:
        if c.type in ("equal", "substitute"):
            for k in range(c.ref_end_idx - c.ref_start_idx):
                if c.hyp_start_idx + k < len(hw): m[c.ref_start_idx + k] = c.hyp_start_idx + k
    gaps = {}  # ref gap g (between ref words g-1 and g) -> [t0, t1], only when both neighbours aligned
    for g in range(len(rw) + 1):
        if (g > 0 and g - 1 not in m) or (g < len(rw) and g not in m): continue
        gaps[g] = [round(hw[m[g - 1]][2], 3) if g > 0 else 0.0, round(hw[m[g]][1], 3) if g < len(rw) else round(dur, 3)]
    out = []
    for g, tag in tags:
        if g not in gaps: continue
        t0, t1 = gaps[g]
        if MIN_EVT <= t1 - t0 <= MAX_EVT:
            out.append({"audio": row["audio"], "speaker": row["speaker"], "tag": tag, "t0": round(t0, 3), "t1": round(t1, 3),
                        "gap": g, "n_words": len(rw)})
    return out, gaps


if __name__ == "__main__":
    src, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(src)]
    rec = recognizer(); ev, gaps, want = [], {}, Counter()
    for i, r in enumerate(rows, 1):
        e, gp = events(rec, r); ev += e; gaps[r["audio"]] = gp
        want.update(t for _, t in split(r["text"])[1])
        if i % 200 == 0: print(f"{i}/{len(rows)} events {len(ev)}", flush=True)
    with open(out / "events.jsonl", "w") as f: f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in ev)
    json.dump(gaps, open(out / "gaps.json", "w"))
    got = Counter(e["tag"] for e in ev)
    for t in want:
        d = [e["t1"] - e["t0"] for e in ev if e["tag"] == t]
        print(f"{t:15s} located {got[t]}/{want[t]} ({got[t]/want[t]:.0%})  dur median {np.median(d) if d else 0:.2f}s")
