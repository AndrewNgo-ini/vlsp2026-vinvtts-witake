#!/bin/bash
# SynParaSpeech-style synthetic sniff / throat-clearing data (synpara.py; 2509.14946): VocalSound events -> Seed-VC to the
# host speaker -> inserted at a word gap of a real train utterance -> two-detector filter (row P5 rule) at the known span.
export PATH=$HOME/.local/bin:$PATH HF_HOME=/mnt/vinvtts/hf OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false UV_CACHE_DIR=/mnt/vinvtts/uvcache LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONIOENCODING=utf-8
R=/mnt/vinvtts/runs; SV=/mnt/vinvtts/src/seedvc/.venv/bin/python; PS=/mnt/vinvtts/src/psed/.venv/bin/python; O=$R/synpara
cd ${VINV_CODE:-/opt/vinvtts-submission/vinvtts}; uv sync -q
step() { echo "== $(date +%T) START $1"; shift; "$@"; local rc=$?; echo "== $(date +%T) END rc=$rc"; }
[ -f $O/cands.jsonl ] || step make $SV synpara.py make --n ${N:-1200}
[ -f $O/cands.psed.jsonl ] || step psed $PS psed_score.py $O/cands.jsonl
step filter uv run synpara.py filter
echo "== SYNPARA_DONE"
