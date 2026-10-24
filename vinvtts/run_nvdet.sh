#!/bin/bash
# Acoustic NVPA proxy: event banks (train/dev) -> AST+LR detector -> score all systems incl. the no-tag negative control.
export PATH=$HOME/.local/bin:$PATH HF_HOME=/mnt/vinvtts/hf OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 BENCH_WORKERS=1 TOKENIZERS_PARALLELISM=false UV_CACHE_DIR=/mnt/vinvtts/uvcache
R=/mnt/vinvtts/runs; D=/mnt/vinvtts/data/vinv-tts; VN=/mnt/vinvtts/src/vieneu/.venv/bin/python
cd ${VINV_CODE:-/opt/vinvtts-submission/vinvtts}; uv sync -q
step() { echo "== $(date +%T) START $1"; shift; "$@"; local rc=$?; echo "== $(date +%T) END rc=$rc"; }
[ -f $R/events/events.jsonl ] || step align_train uv run align.py $D/train.jsonl $R/events
[ -f $R/events_dev/events.jsonl ] || step align_dev uv run align.py $D/dev.jsonl $R/events_dev
step gen_notag   env NOTAG=1 $VN gen_zs.py vieneu $R/eval/eval.jsonl $R/zs/ft1_notag $R/vieneu_ft/ft1/merged
step bench_notag uv run bench.py $R/zs/ft1_notag.jsonl $R/zs/vieneu.jsonl
step tag_notag   uv run tagasr.py score --ckpt $R/tagasr $R/zs/ft1_notag.jsonl
step nvdet_train uv run nvdet.py train --events $R/events --train $D/train.jsonl --dev-events $R/events_dev --dev $D/dev.jsonl --out $R/nvdet
S="$R/eval/gt.jsonl $R/zs/vieneu.jsonl $R/zs/ft1.jsonl $R/zs/ft1_notag.jsonl"; for x in omni f5vi; do [ -f $R/zs/$x.jsonl ] && S="$S $R/zs/$x.jsonl"; done
step nvdet_score uv run nvdet.py score --model $R/nvdet $S
echo "== NVDET_DONE"
