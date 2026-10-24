"""Public-test manifests (HF trivitaai/vinv-tts public_test_{A,B}) -> runs/public/{A,B}.jsonl for gen_zs.py / bench.py.
Every text.txt line x every prompt speaker; rows: id, text (normalised like train: lowercase, digits spelled), text_raw,
ref (prompt clip, B resampled 48k -> 24k mono), ref_text, speaker.
    python mkpublic.py /mnt/vinvtts/data/vinv-tts-hf /mnt/vinvtts/runs/public
"""
import json, re, sys, subprocess
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
DIG = "không một hai ba bốn năm sáu bảy tám chín".split()


def norm(t):
    t = re.sub(r"\d", lambda m: f" {DIG[int(m.group())]} ", t)     # only single digits occur ("League 1")
    t = re.sub(r"(\[[a-z]+\])", r" \1 ", t.lower())
    return " ".join(t.split())


for tr in "AB":
    d = src / f"public_test_{tr}"; lines = [l.strip() for l in open(d / "text.txt", encoding="utf-8") if l.strip()]
    rows = []
    for m in map(json.loads, open(d / "metadata.jsonl", encoding="utf-8")):
        spk, ref = m["speaker"], d / m["file_name"]
        r24 = out / "refs" / tr / f"{spk}.wav"; r24.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(ref), "-ac", "1", "-ar", "24000", str(r24)], check=True)
        for i, l in enumerate(lines):
            rows.append({"id": f"{tr}_{spk}_{i:02d}", "text": norm(l), "text_raw": l, "ref": str(r24),
                         "ref_text": norm(m["text"]), "speaker": spk, "line": i})
    with open(out / f"{tr}.jsonl", "w", encoding="utf-8") as f: f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(tr, len(rows), "rows")
