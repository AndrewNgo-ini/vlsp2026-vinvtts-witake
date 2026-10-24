#!/bin/bash
# w1score.sh GPU <dir>/A <dir>/B : judge proxies for run_final.py output (rebuilt 2026-10-01; original lost with the box).
#   bench (Zipformer WER, DNSMOS, ECAPA SS) -> v3 tag-ASR -> nvdet -> audit judge -> offscore.py summary
G=$1; shift; C=${VINV_CODE:-/opt/vinvtts-submission/vinvtts}; R=/data/vinvtts/runs
cd $C; export HF_HOME=/data/vinvtts/hf OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$G LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=$HOME/.local/bin:$PATH
M=(); for d in "$@"; do M+=("$d.jsonl"); done
uv run bench.py "${M[@]}"
/data/vinvtts/src/qwen3asr/.venv/bin/python qasr_tag.py score $R/qasr/exp/checkpoint-270 "${M[@]}"
uv run nvdet.py score --model $R/nvdet "${M[@]}"
[ -f $R/audit_judge/clf.pkl ] && uv run audit_judge.py score --model $R/audit_judge "${M[@]}"
uv run offscore.py "${M[@]}"
