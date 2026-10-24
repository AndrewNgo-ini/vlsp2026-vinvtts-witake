#!/bin/bash
# OmniVoice os5k4000 recipe on real + SynParaSpeech-v2 sniff/throat (row SP2) + g3-distilled laughter clips (row D2):
# does adding laughter-rich rows cancel the replicated laughter regression of omft_sp4000 (row SP3)? Two evalF draws.
export PATH=$HOME/.local/bin:$PATH HF_HOME=/mnt/vinvtts/hf OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 BENCH_WORKERS=1 TOKENIZERS_PARALLELISM=false UV_CACHE_DIR=/mnt/vinvtts/uvcache LANG=C.UTF-8 LC_ALL=C.UTF-8
R=/mnt/vinvtts/runs; Z=$R/zsF; PS=/mnt/vinvtts/src/psed/.venv/bin/python; OV=/mnt/vinvtts/src/omnivoice/.venv/bin/python
cd ${VINV_CODE:-/opt/vinvtts-submission/vinvtts}; uv sync -q
cat $R/synpara/kept.jsonl $R/distill_g3/kept.jsonl > $R/synpara/kept_plus_dl.jsonl
NAME=spdl OVERSAMPLE=1 STEPS=5000 CKPTS=4000 EXTRA=$R/synpara/kept_plus_dl.jsonl bash run_omni_ft.sh
S1=$Z/omft_spdl4000; S2=$Z/omft_spdl4000_r1
[ -f $S2.jsonl ] || $OV gen_zs.py omni $R/evalF/eval.jsonl $S2 $R/omni_ft_spdl/merged-4000
uv run bench.py $S2.jsonl
for S in $S1 $S2; do
  uv run nvdet.py score --model $R/nvdet $S.jsonl; uv run laugh_check.py $S.jsonl
  [ -f $S.tags_v2s.jsonl ] || uv run tagasr.py score --ckpt $R/tagasr_v2s --suffix .tags_v2s.jsonl $S.jsonl
  uv run rare_judge.py spans $S.jsonl && $PS psed_score.py $S.jsonl && uv run rare_judge.py apply $S.jsonl
done
/mnt/vinvtts/src/qwen3asr/.venv/bin/python qasr_tag.py score $R/qasr/exp/checkpoint-270 $S1.jsonl $S2.jsonl
uv run objc.py $R/evalF/gt $Z/omft_os5k4000 $Z/omft_sp4000 $Z/omft_sp4000_r1 $S1 $S2
echo "== SPDL_DONE"
