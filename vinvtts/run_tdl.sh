#!/bin/bash
# Throat-only synthetic variant (coordinator, row R1: synthetic sniff clips may teach a worse sniff): OmniVoice os5k
# recipe on real + SynParaSpeech-v2 THROAT clips + g3 laughter clips (no synthetic sniff) -> DPO (odpo recipe).
# PART=1: data + SFT (omft_tdl4000, 1 evalF draw); PART=2: DPO candidates; PART=3: DPO + 2 evalF draws + evalR.
export PATH=$HOME/.local/bin:$PATH HF_HOME=/mnt/vinvtts/hf OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 BENCH_WORKERS=1 TOKENIZERS_PARALLELISM=false UV_CACHE_DIR=/mnt/vinvtts/uvcache LANG=C.UTF-8 LC_ALL=C.UTF-8
R=/mnt/vinvtts/runs; Z=$R/zsF; PS=/mnt/vinvtts/src/psed/.venv/bin/python; OV=/mnt/vinvtts/src/omnivoice/.venv/bin/python
cd ${VINV_CODE:-/opt/vinvtts-submission/vinvtts}; uv sync -q
judge() { for S in "$@"; do
  [ -f $S.scored.jsonl ] || uv run bench.py $S.jsonl
  [ -f $S.tags_v2s.jsonl ] || uv run tagasr.py score --ckpt $R/tagasr_v2s --suffix .tags_v2s.jsonl $S.jsonl
  [ -f $S.nvdet.jsonl ] || uv run nvdet.py score --model $R/nvdet $S.jsonl
  [ -f $S.laugh.jsonl ] || uv run laugh_check.py $S.jsonl
  [ -f $S.rare.jsonl ] || { uv run rare_judge.py spans $S.jsonl && $PS psed_score.py $S.jsonl && uv run rare_judge.py apply $S.jsonl; }
  [ -f $S.tags_qasr.jsonl ] || /mnt/vinvtts/src/qwen3asr/.venv/bin/python qasr_tag.py score $R/qasr/exp/checkpoint-270 $S.jsonl
done; }
case ${PART:-1} in
1) uv run python -c "
import json
rows=[json.loads(l) for l in open('$R/synpara/kept.jsonl', encoding='utf-8')]
t=[r for r in rows if '/sp_throa_' in r['audio']]
d=[json.loads(l) for l in open('$R/distill_g3/kept.jsonl', encoding='utf-8')]
open('$R/synpara/kept_throat_dl.jsonl','w',encoding='utf-8').writelines(json.dumps(x,ensure_ascii=False)+'\n' for x in t+d)
print('throat synthetic', len(t), '+ laughter distilled', len(d))"
   NAME=tdl OVERSAMPLE=1 STEPS=5000 CKPTS=4000 EXTRA=$R/synpara/kept_throat_dl.jsonl bash run_omni_ft.sh
   judge $Z/omft_tdl4000; echo "== TDL_P1_DONE";;
2) ODIR=$R/omni_rs_tdl SRC=$R/omni_ft_tdl/merged-4000 PICK_ONLY=1 SEED=9 bash run_omni_rs.sh; echo "== TDL_P2_DONE";;
3) ODIR=$R/omni_rs_tdl SRC=$R/omni_ft_tdl/merged-4000 RUN=odpo_tdl DPO_ARGS="--beta 5" bash run_dpo_omni.sh
   [ -f $Z/odpo_tdl_r1.jsonl ] || $OV gen_zs.py omni $R/evalF/eval.jsonl $Z/odpo_tdl_r1 $R/omni_dpo/odpo_tdl
   judge $Z/odpo_tdl $Z/odpo_tdl_r1
   uv run objc.py $R/evalF/gt $Z/odpo_spdl $Z/odpo_spdl_r1 $Z/omft_tdl4000 $Z/odpo_tdl $Z/odpo_tdl_r1
   SYSTEMS="omft_tdl4000:omni:$R/omni_ft_tdl/merged-4000 odpo_tdl:omni:$R/omni_dpo/odpo_tdl" bash run_evalr.sh
   E=$R/evalR/zs; uv run objc.py $R/evalR/gt $E/omft_os5k4000 $E/omft_spdl4000 $E/odpo_spdl $E/omft_tdl4000 $E/odpo_tdl
   echo "== TDL_DONE";;
esac
