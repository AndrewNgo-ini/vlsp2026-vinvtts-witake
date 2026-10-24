#!/bin/bash
# Rejection-sampling SFT for OmniVoice (stage 1 of arXiv 2609.11260): the OmniVoice LoRA samples N candidates for
# training texts (rare-tag texts oversampled), keep the best candidate per text only if BOTH NV scorers confirm every
# tag and WER <= 0.1, then LoRA-fine-tune on real + kept (run_omni_ft.sh EXTRA=...). Train texts only -> no dev leakage.
export PATH=$HOME/.local/bin:$PATH HF_HOME=/mnt/vinvtts/hf OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 BENCH_WORKERS=1 TOKENIZERS_PARALLELISM=false UV_CACHE_DIR=/mnt/vinvtts/uvcache
R=/mnt/vinvtts/runs; D=/mnt/vinvtts/data/vinv-tts; PY=/mnt/vinvtts/src/omnivoice/.venv/bin/python; O=${ODIR:-$R/omni_rs}; SRC=${SRC:-$R/omni_ft/merged-2000}
N=${N:-4}; NP=${NP:-600}
cd ${VINV_CODE:-/opt/vinvtts-submission/vinvtts}; uv sync -q; mkdir -p $O
step() { echo "== $(date +%T) START $1"; shift; "$@"; local rc=$?; echo "== $(date +%T) END rc=$rc"; }
[ -f $O/prompts.jsonl ] || uv run python - <<PY
import json, random, re
from collections import Counter, defaultdict
random.seed(int("${SEED:-0}"))
rows = [r for r in map(json.loads, open("$D/train.jsonl")) if 3 <= r["dur"] <= 20]
by = defaultdict(list)
for r in rows: by[r["speaker"]].append(r)
f = Counter(t for r in rows for t in re.findall(r"\[([a-z]+)\]", r["text"])); fm = max(f.values())
w = [max([(fm / f[t]) ** 0.5 for t in set(re.findall(r"\[([a-z]+)\]", r["text"]))] or [1]) for r in rows]
out, seen = [], set()
while len(out) < $NP:
    r = random.choices(rows, w)[0]
    if r["audio"] in seen: continue
    seen.add(r["audio"])
    refs = sorted([x for x in by[r["speaker"]] if x is not r and x["dur"] >= 3], key=lambda x: x["dur"])
    if not refs: continue
    ref = refs[0]
    out.append({"id": f"rs{len(out):04d}", "text": r["text"], "speaker": r["speaker"], "ref": ref["audio"],
                "ref_text": " ".join(re.sub(r"\[[a-z]+\]", " ", ref["text"]).split())})
open("$O/prompts.jsonl", "w").writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in out)
print(len(out), "prompts", Counter(t for x in out for t in re.findall(r"\[([a-z]+)\]", x["text"])))
PY
[ -f $O/cands.jsonl ] || step gen env NBEST=$N $PY gen_zs.py omni $O/prompts.jsonl $O/cands $SRC
[ -f $O/cands.nvdet.jsonl ] || step nvdet uv run nvdet.py score --model $R/nvdet $O/cands.jsonl
step pick uv run bestof.py $O/cands.jsonl $O/best --ckpt $R/tagasr --nvdet $O/cands.nvdet.jsonl
uv run python - <<PY
import json, re
from collections import Counter
k = [c for c in map(json.loads, open("$O/best.jsonl")) if c["nv_n"] and c["nv_hit"] == c["nv_n"] and c.get("nvdet_hit") == c["nv_n"] and c["wer"] <= 0.1]
open("$O/kept.jsonl", "w").writelines(json.dumps({"audio": c["gen"], "speaker": c["speaker"], "text": c["text"], "dur": c["dur"]}, ensure_ascii=False) + "\n" for c in k)
print("kept", len(k), "tags", Counter(t for c in k for t in re.findall(r"\[([a-z]+)\]", c["text"])))
PY
[ -n "$PICK_ONLY" ] && { echo "== OMNIRS_DONE (pick only)"; exit 0; }
NAME=rs EXTRA=$O/kept.jsonl OVERSAMPLE=1 STEPS=${STEPS:-3000} CKPTS="${CKPTS:-1000 2000 3000}" bash run_omni_ft.sh
echo "== OMNIRS_DONE"
