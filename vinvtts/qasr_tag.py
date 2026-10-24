"""Class-balanced tag-ASR v3 (coordinator step 2): Qwen3-ASR-1.7B (Apache, Vietnamese; github.com/QwenLM/Qwen3-ASR,
official finetuning/qwen3_asr_sft.py) full SFT on our train set with inline tags, square-root category sampling
(NVVSpeech Track-1 4th place, arXiv 2609.23462: SQRT single-stage gave their best per-class F1 -- sniff .61,
throat .50, laughter .90, breath .63; the extra uniform stage hurt most classes, so it is skipped).

  prep  : train.jsonl -> runs/qasr/train.jsonl ({"audio", "text": "language Vietnamese<asr_text>..."}), each row
          repeated max over its tags of round(sqrt(f_max / f_tag)) (breathing 1, laughter 2, sniff/throat 6)
  score : <ckpt> <sys>.jsonl ... -> <sys>.tags_qasr.jsonl ({id, text, tag_hyp}) -- same format as tagasr.py, so
          tag_perclass.py / objc.py read it

    <qasr venv> qasr_tag.py prep ; <qasr venv> qasr_tag.py score <ckpt> runs/evalF/gt.jsonl ...
"""
import json, os, re, sys
from collections import Counter

R = "/mnt/vinvtts/runs"; O = f"{R}/qasr"; TAG = re.compile(r"\[(laughter|breathing|sniff|throatclearing)\]")
PREFIX = "language Vietnamese<asr_text>"


def prep():
    rows = [json.loads(l) for l in open("/mnt/vinvtts/data/vinv-tts/train.jsonl")]
    f = Counter(t for r in rows for t in TAG.findall(r["text"])); fm = max(f.values()); out = []
    for r in rows:
        k = max([round((fm / f[t]) ** 0.5) for t in set(TAG.findall(r["text"]))] or [1])
        out += [{"audio": r["audio"], "text": PREFIX + " ".join(r["text"].split())}] * k
    import random; random.seed(0); random.shuffle(out)
    os.makedirs(O, exist_ok=True)
    with open(f"{O}/train.jsonl", "w") as fh: fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in out)
    dev = [json.loads(l) for l in open("/mnt/vinvtts/data/vinv-tts/dev.jsonl")][:64]   # eval loss only (official script needs one)
    with open(f"{O}/dev.jsonl", "w") as fh: fh.writelines(json.dumps({"audio": r["audio"], "text": PREFIX + " ".join(r["text"].split())}, ensure_ascii=False) + "\n" for r in dev)
    print("qasr train rows", len(rows), "->", len(out), "tag rows", dict(Counter(t for x in out for t in set(TAG.findall(x["text"])))))


def score(ckpt, mans):
    import torch
    from qwen_asr import Qwen3ASRModel
    m = Qwen3ASRModel.from_pretrained(ckpt, dtype=torch.bfloat16, device_map="cuda:0")
    for man in mans:
        rows = [json.loads(l) for l in open(man)]; hyps = []
        for i in range(0, len(rows), 16):
            res = m.transcribe(audio=[r["gen"] for r in rows[i:i + 16]], language=["Vietnamese"] * len(rows[i:i + 16]))
            hyps += [" ".join(x.text.split()) for x in res]
        with open(man.replace(".jsonl", ".tags_qasr.jsonl"), "w") as fh:
            fh.writelines(json.dumps({"id": r["id"], "text": r["text"], "tag_hyp": h}, ensure_ascii=False) + "\n" for r, h in zip(rows, hyps))
        print("qasr scored", man, Counter(t for h in hyps for t in TAG.findall(h)), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prep": prep()
    else: score(sys.argv[2], sys.argv[3:])
