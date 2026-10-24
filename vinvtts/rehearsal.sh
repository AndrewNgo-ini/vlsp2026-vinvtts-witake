#!/bin/bash
# rehearsal.sh OUT : production dress rehearsal on a fresh box. launch.sh v4 defaults on GPUs 0+1 -> robust_test ->
# full public test (run_final) -> w1score. Expect ROBUST ALL OK and roughly offA .883 / offB .874 (single-draw noise ~±.01).
O=${1:-/data/vinvtts/runs/rehearsal}; mkdir -p $O; C=${VINV_CODE:-/opt/vinvtts-submission/vinvtts}; cd $C
export PATH=$HOME/.local/bin:$PATH
GPU=0 GPU2=1 PORT=8020 LOGS=$O/serve_logs setsid bash launch.sh > $O/launch.log 2>&1 & L=$!
t=0; until grep -q READY $O/launch.log; do sleep 10; t=$((t+10)); [ $t -ge 1800 ] && { echo "NOT READY"; tail -20 $O/launch.log; kill -- -$L; exit 1; }; done
echo "READY after ${t}s"
uv run robust_test.py --url http://127.0.0.1:8020 > $O/robust.log 2>&1; tail -3 $O/robust.log
uv run run_final.py --url http://127.0.0.1:8020 --out $O/final --conc 32 > $O/run_final.log 2>&1; tail -2 $O/run_final.log
kill -- -$L 2>/dev/null; sleep 10; pkill -f "[t]ts_worker.py --mode omni"; pkill -f "[j]udge_worker.py"; pkill -f "[s]erve.py --prod"
bash w1score.sh 0 $O/final/A $O/final/B > $O/score.log 2>&1; grep -h "offA\|offB" $O/score.log
echo REHEARSAL_DONE
