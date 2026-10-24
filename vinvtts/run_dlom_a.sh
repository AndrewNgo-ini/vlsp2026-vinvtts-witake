#!/bin/bash
# Cross-engine laughter distillation VieNeu -> OmniVoice, part A (< 2 h): dpo6b2Lg3 x NBEST candidates for the 606
# [laughter] train texts (distill.py prompts: own speaker, other clip as ref) -> filter: LaughterSegmentation AND nvdet
# confirm every laugh, nvdet the other tags, WER <= .10 (distill.py; RS-SFT 2609.11260 across engines). v3 tag-ASR is
# NOT used (held-out judge).
export PATH=$HOME/.local/bin:$PATH HF_HOME=/mnt/vinvtts/hf OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 BENCH_WORKERS=1 TOKENIZERS_PARALLELISM=false UV_CACHE_DIR=/mnt/vinvtts/uvcache
R=/mnt/vinvtts/runs; VN=/mnt/vinvtts/src/vieneu/.venv/bin/python; O=$R/distill_g3; N=${N:-3}
cd ${VINV_CODE:-/opt/vinvtts-submission/vinvtts}; uv sync -q; mkdir -p $O
step() { echo "== $(date +%T) START $1"; shift; "$@"; local rc=$?; echo "== $(date +%T) END rc=$rc"; }
[ -f $O/c_g3.jsonl ] || step gen env NBEST=$N $VN gen_zs.py vieneu $R/distill/prompts.jsonl $O/c_g3 $R/vieneu_ft/dpo6b2Lg3/merged
[ -f $O/kept.jsonl ] || step filter uv run distill.py filter $O/kept.jsonl $O/c_g3.jsonl
echo "== DLOM_A_DONE"
