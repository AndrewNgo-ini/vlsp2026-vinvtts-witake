"""EDA of the ViNV-TTS drop: <root>/{train,dev}/spk_*/spk_*.json + *.flac -> manifest JSONL + stats on stdout.

    uv run eda.py ../data/vinv-tts
"""
import json, re, sys, unicodedata
from collections import Counter
from pathlib import Path
import numpy as np, soundfile as sf

root = Path(sys.argv[1])
TAG = re.compile(r"\[([^\]]+)\]")
for split in ["train", "dev"]:
    rows = []
    for meta in sorted(root.glob(f"{split}/spk_*/spk_*.json")):
        for r in json.load(open(meta)):
            p = meta.parent / r["audio"]
            i = sf.info(p)
            rows.append({"audio": str(p.resolve()), "speaker": meta.parent.name, "text": unicodedata.normalize("NFC", r["text"]),
                         "lang": r.get("language_id"), "sr": i.samplerate, "ch": i.channels, "dur": i.frames / i.samplerate})
    with open(root / f"{split}.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    d = np.array([r["dur"] for r in rows]); spk = Counter(r["speaker"] for r in rows)
    tags = Counter(t for r in rows for t in TAG.findall(r["text"]))
    words = [len(TAG.sub(" ", r["text"]).split()) for r in rows]
    pos = Counter()
    for r in rows:
        toks = TAG.sub(lambda m: f" <{m.group(1)}> ", r["text"]).split()
        for k, t in enumerate(toks):
            if t.startswith("<"): pos["start" if k == 0 else "end" if k == len(toks) - 1 else "mid"] += 1
        pos["adjacent"] += len(re.findall(r"\]\s*\[", r["text"]))
    spk_hours = Counter(); [spk_hours.__setitem__(r["speaker"], spk_hours[r["speaker"]] + r["dur"] / 3600) for r in rows]
    print(f"== {split}: {len(rows)} utts, {len(spk)} spk, {d.sum()/3600:.2f} h, dur mean {d.mean():.1f} min {d.min():.1f} "
          f"p50 {np.median(d):.1f} p95 {np.percentile(d,95):.1f} max {d.max():.1f}s")
    print("  sr", Counter(r["sr"] for r in rows), "ch", Counter(r["ch"] for r in rows), "lang", Counter(r["lang"] for r in rows))
    print("  tags", dict(tags.most_common()), "| utts w/o tag", sum(1 for r in rows if not TAG.search(r["text"])))
    print("  tag positions", dict(pos), "| words/utt mean", round(np.mean(words), 1), "| words/s", round(sum(words) / d.sum(), 2))
    print("  top spk hours", [(s, round(h, 2)) for s, h in spk_hours.most_common(12)])
    print("  spk with 1 utt", sum(1 for c in spk.values() if c == 1), "| <=5 utts", sum(1 for c in spk.values() if c <= 5))
    chars = Counter(c for r in rows for c in TAG.sub("", r["text"]))
    print("  non-alpha chars", {c: n for c, n in chars.most_common() if not c.isalpha()})
    print("  uppercase utts", sum(1 for r in rows if TAG.sub("", r["text"]) != TAG.sub("", r["text"]).lower()), "| digits utts", sum(1 for r in rows if re.search(r"\d", r["text"])))
tr = {json.loads(l)["speaker"] for l in open(root / "train.jsonl")}; dv = {json.loads(l)["speaker"] for l in open(root / "dev.jsonl")}
print("dev spk also in train:", len(dv & tr), "/", len(dv), "| dev-only:", sorted(dv - tr))
