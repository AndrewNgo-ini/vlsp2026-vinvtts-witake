#!/bin/bash
# Zero-shot PretrainedSED rare-event detector check on all our real events (psed_eval.py). ~30 min, 2 CPU threads.
export HF_HOME=/mnt/vinvtts/hf OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd ${VINV_CODE:-/opt/vinvtts-submission/vinvtts}
/mnt/vinvtts/src/psed/.venv/bin/python psed_eval.py BEATs ATST-F M2D
echo "== PSED_DONE"
