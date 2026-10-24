#!/bin/bash
# OmniVoice masked-CE DPO + SFT anchor (dpo_omni.py; the odpo_os recipe, beta 5) from the new OmniVoice main
# omft_spdl4000 (row SP4), fresh on-policy pairs (600 train texts x4). PART=1: candidates + scoring (run_omni_rs.sh
# PICK_ONLY); PART=2: pairs + DPO + evalF + held-out v3 / objC. Each part < 2 h; both skip finished outputs.
export PATH=$HOME/.local/bin:$PATH HF_HOME=/mnt/vinvtts/hf OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 BENCH_WORKERS=1 TOKENIZERS_PARALLELISM=false UV_CACHE_DIR=/mnt/vinvtts/uvcache LANG=C.UTF-8 LC_ALL=C.UTF-8
R=/mnt/vinvtts/runs; Z=$R/zsF; S=$R/omni_ft_spdl/merged-4000; O=$R/omni_rs_spdl; PS=/mnt/vinvtts/src/psed/.venv/bin/python
cd ${VINV_CODE:-/opt/vinvtts-submission/vinvtts}
if [ "${PART:-1}" = 1 ]; then ODIR=$O SRC=$S PICK_ONLY=1 SEED=7 bash run_omni_rs.sh; echo "== ODPO_SPDL_P1_DONE"; exit 0; fi
ODIR=$O SRC=$S RUN=odpo_spdl DPO_ARGS="--beta 5" bash run_dpo_omni.sh
X=$Z/odpo_spdl
uv run laugh_check.py $X.jsonl
uv run rare_judge.py spans $X.jsonl && $PS psed_score.py $X.jsonl && uv run rare_judge.py apply $X.jsonl
/mnt/vinvtts/src/qwen3asr/.venv/bin/python qasr_tag.py score $R/qasr/exp/checkpoint-270 $X.jsonl
uv run objc.py $R/evalF/gt $Z/omft_os5k4000 $Z/odpo_os $Z/omft_spdl4000 $Z/omft_spdl4000_r1 $X
echo "== ODPO_SPDL_DONE"
