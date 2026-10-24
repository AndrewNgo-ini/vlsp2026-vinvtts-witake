#!/bin/bash
# Masked-CE DPO for OmniVoice (dpo_omni.py; GenSR MGM-DPO 2508.17229 + VRPO 2505.19223 + anchor 2609.11260).
# Pairs = on-policy candidates of SRC for 600 TRAIN texts (run_omni_rs.sh output), ranked by the dual-scorer utility.
export PATH=$HOME/.local/bin:$PATH HF_HOME=/mnt/vinvtts/hf OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 BENCH_WORKERS=1 TOKENIZERS_PARALLELISM=false UV_CACHE_DIR=/mnt/vinvtts/uvcache
R=/mnt/vinvtts/runs; OPY=/mnt/vinvtts/src/omnivoice/.venv/bin/python; O=${ODIR:-$R/omni_rs}; SRC=${SRC:-$R/omni_ft/merged-2000}; RUN=${RUN:-odpo1}
cd ${VINV_CODE:-/opt/vinvtts-submission/vinvtts}; uv sync -q
step() { echo "== $(date +%T) START $1"; shift; "$@"; local rc=$?; echo "== $(date +%T) END rc=$rc"; }
[ -f $O/pairs.jsonl ] || step pairs uv run dpo_omni.py pairs $O/cands.scored.jsonl $O/cands.nvdet.jsonl $O/pairs.jsonl
step train_$RUN $OPY dpo_omni.py train $O/pairs.jsonl $R/omni_dpo/$RUN --model $SRC ${DPO_ARGS:-}
L=$R/zsF/$RUN.jsonl
step gen_$RUN $OPY gen_zs.py omni $R/evalF/eval.jsonl $R/zsF/$RUN $R/omni_dpo/$RUN
step bench uv run bench.py $L; step tags uv run tagasr.py score --ckpt $R/tagasr $L; step nvdet uv run nvdet.py score --model $R/nvdet $L
step v2s uv run tagasr.py score --ckpt $R/tagasr_v2s --suffix .tags_v2s.jsonl $L
uv run table.py $R/evalF/gt $R/zsF/omft2000 $R/zsF/$RUN
echo "== ODPO_DONE $RUN"
