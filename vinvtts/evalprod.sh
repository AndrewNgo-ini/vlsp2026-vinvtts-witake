#!/bin/bash
# evalprod.sh GPU MODEL NAME PORTBASE [OUTROOT] : single-engine production recipe (tag-CFG w3, best-of-8, splice any) on the
# public A/B test -> OUTROOT/final_NAME + w1score. Same recipe as X5 (HF data_backup/x5_meta/evalprod.sh), so cdpo1b must
# reproduce offA .882 / offB .875 here (environment drift check).
G=$1; M=$2; N=$3; P=$4; C=${5:-/data/vinvtts/runs/evalprod}; mkdir -p $C/logs
cd ${VINV_CODE:-/opt/vinvtts-submission/vinvtts}; export HF_HOME=/data/vinvtts/hf OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$G LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=$HOME/.local/bin:$PATH
/data/vinvtts/src/omnivoice/.venv/bin/python tts_worker.py --mode omni --tag-w 3 --model $M --train /data/vinvtts/data/vinv-tts/train.jsonl --port $((P+1)) > $C/logs/w_$N.log 2>&1 & W=$!
/data/vinvtts/src/qwen3asr/.venv/bin/python judge_worker.py --kind qasr --port $((P+2)) > $C/logs/q_$N.log 2>&1 & Q=$!
/data/vinvtts/src/psed/.venv/bin/python judge_worker.py --kind psed --port $((P+3)) > $C/logs/p_$N.log 2>&1 & S=$!
t=0; until curl -s 127.0.0.1:$((P+1))/health >/dev/null && curl -s 127.0.0.1:$((P+2))/health >/dev/null && curl -s 127.0.0.1:$((P+3))/health >/dev/null; do
  sleep 5; t=$((t+5)); [ $t -ge 1200 ] && { echo "workers not healthy, see $C/logs/*_$N.log"; kill $W $Q $S; exit 1; }; done
uv run serve.py --prod --worker http://127.0.0.1:$((P+1)) --qasr http://127.0.0.1:$((P+2)) --psed http://127.0.0.1:$((P+3)) --port $P > $C/logs/s_$N.log 2>&1 & V=$!
until curl -s 127.0.0.1:$P/health >/dev/null; do sleep 5; done
TR=${TRACKS:-AB}   # TRACKS=B: Track-B-only eval
uv run run_final.py --url http://127.0.0.1:$P --out $C/final_$N --conc 32 --tracks $TR > $C/logs/rf_$N.log 2>&1
kill $W $Q $S $V
bash w1score.sh $G $(for t in $(echo $TR | sed 's/./& /g'); do echo $C/final_$N/$t; done) > $C/logs/fs_$N.log 2>&1
tail -2 $C/logs/fs_$N.log; echo EVAL_DONE $N
