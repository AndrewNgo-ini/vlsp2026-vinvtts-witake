#!/bin/bash
# One-command production launcher: tag-CFG OmniVoice worker + v3 tag-ASR + PSED judges + serve.py --prod,
# each under an auto-restart loop, with health checks. Ctrl-C / kill of this script stops everything.
#   GPU=7 GPU2=3 PORT=8020 bash launch.sh
#     defaults = production (v5, 10-02 = v4 + NV levelling; NV_LEVEL=off -> v4): cdpo1b (A gs2/w3 + inpaint k2; B gs6/w5) + odpo_spdl (gs7/w3), 8+8 candidates,
#     nvdet+v3 selector, splice any, native-rate prompts.
#   per-track primary: MODEL_B=runs/omni_dpo/cdpo_real2 (MODEL -> Track A only, MODEL_B -> Track B only, MODEL2 both);
#   one engine only: MODEL2= ; plain fallback: TAG_W=0 TW_A= TW2_A= INPAINT_K_A=0
#   env: ROOT (default /data/vinvtts), GPU, PORT, TAG_W, GS, TAIL_PAD, TAIL_S, REF_SR, N (best-of-N), LOGS
set -u
ROOT=${ROOT:-/data/vinvtts}; GPU=${GPU:-0}; PORT=${PORT:-8020}; TAG_W=${TAG_W:-3}; GS=${GS:-2.0}; N=${N:-8}
TAIL_PAD=${TAIL_PAD:-}; TAIL_S=${TAIL_S:-0}; REF_SR=${REF_SR:-0}; INPAINT_K_A=${INPAINT_K_A:-2}; LOGS=${LOGS:-$ROOT/runs/serve_logs}
MODEL=${MODEL:-runs/omni_dpo/cdpo1b}; MODEL2=${MODEL2-runs/omni_dpo/odpo_spdl}; GPU2=${GPU2:-$GPU}; P_OMNI2=${P_OMNI2:-8014}
GS_A=${GS_A-2}; GS_B=${GS_B-6}; GS2_A=${GS2_A-7}; GS2_B=${GS2_B-7}; TW_A=${TW_A-3}; TW_B=${TW_B-5}; TW2_A=${TW2_A-3}; TW2_B=${TW2_B-3}; PER_ENGINE=${PER_ENGINE:-1}; SELECT=${SELECT:-nvdet_v3}
P_OMNI=${P_OMNI:-8011}; P_QASR=${P_QASR:-8012}; P_PSED=${P_PSED:-8013}
# Production NV levelling (v5, user-approved 2026-10-02): quantile-map NV loudness to real-speaker levels after splice.
# Disable with NV_LEVEL=off. Calibration/target files come from HF ar_uncanny_1001 (setup_box.sh links them).
export NV_LEVEL=${NV_LEVEL:-qmap} NV_LEVEL_ALPHA=${NV_LEVEL_ALPHA:-throatclearing=0.6}
export NV_LEVEL_SYS_A=${NV_LEVEL_SYS_A:-$ROOT/runs/ar_uncanny/calib_pooled.loud.jsonl} NV_LEVEL_SYS_B=${NV_LEVEL_SYS_B:-$ROOT/runs/ar_uncanny/calib_pooled.loud.jsonl}
export NV_LEVEL_GT=${NV_LEVEL_GT:-$ROOT/runs/ar_uncanny/gt_train.loud.jsonl}
unset CAND_DUMP
cd "$(dirname "$0")"; mkdir -p "$LOGS"
export HF_HOME=$ROOT/hf OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$GPU LANG=C.UTF-8 LC_ALL=C.UTF-8 TOKENIZERS_PARALLELISM=false PATH=$HOME/.local/bin:$PATH
PIDS=()
keep() {   # keep NAME CMD... : restart CMD whenever it exits
  local n=$1; shift
  ( while true; do echo "[$(date +%T)] start $n" >> "$LOGS/$n.log"; "$@" >> "$LOGS/$n.log" 2>&1; echo "[$(date +%T)] $n exited $?, restarting in 5 s" >> "$LOGS/$n.log"; sleep 5; done ) &
  PIDS+=($!)
}
wait_up() { local n=$1 p=$2 t=0; until curl -sf 127.0.0.1:$p/health >/dev/null; do sleep 5; t=$((t+5)); [ $t -ge 900 ] && { echo "$n not healthy after 15 min, see $LOGS/$n.log"; exit 1; }; done; echo "$n up on :$p"; }
trap 'kill ${PIDS[@]} 2>/dev/null; pkill -P $$; exit' INT TERM EXIT
EXTRA=(); [ -n "$TAIL_PAD" ] && EXTRA+=(--tail-pad "$TAIL_PAD")
keep omni $ROOT/src/omnivoice/.venv/bin/python tts_worker.py --mode omni --model $ROOT/$MODEL \
     --train $ROOT/data/vinv-tts/train.jsonl --tag-w $TAG_W --gs $GS --tail-s $TAIL_S --ref-sr $REF_SR --inpaint-k-a $INPAINT_K_A "${EXTRA[@]}" --port $P_OMNI
W="http://127.0.0.1:$P_OMNI"; [ -n "$GS_A" ] && W="$W@$GS_A/$GS_B"; [ -n "$TW_A" ] && W="$W@$TW_A/$TW_B"
# MODEL_B (optional, 2026-10-01): a Track-B-only primary engine; the MODEL engine then serves Track A only
MODEL_B=${MODEL_B:-}; P_OMNI3=${P_OMNI3:-8015}
if [ -n "$MODEL_B" ]; then
  # positional spec url@gs@tw@tracks: empty fields stay empty (plain fallback TW_A= must not shift "A" into the tw slot)
  W="http://127.0.0.1:$P_OMNI@${GS_A:+$GS_A/$GS_B}@${TW_A:+$TW_A/$TW_B}@A"
  keep omni3 $ROOT/src/omnivoice/.venv/bin/python tts_worker.py --mode omni --model $ROOT/$MODEL_B \
     --train $ROOT/data/vinv-tts/train.jsonl --tag-w $TAG_W --gs $GS --tail-s $TAIL_S --ref-sr $REF_SR "${EXTRA[@]}" --port $P_OMNI3
  W="$W,http://127.0.0.1:$P_OMNI3@${GS_B:+$GS_B/$GS_B}@${TW_B:+$TW_B/$TW_B}@B"
fi
if [ -n "$MODEL2" ]; then
  keep omni2 env CUDA_VISIBLE_DEVICES=$GPU2 $ROOT/src/omnivoice/.venv/bin/python tts_worker.py --mode omni --model $ROOT/$MODEL2 \
     --train $ROOT/data/vinv-tts/train.jsonl --tag-w $TAG_W --gs $GS --tail-s $TAIL_S --ref-sr $REF_SR "${EXTRA[@]}" --port $P_OMNI2
  W2="http://127.0.0.1:$P_OMNI2"; [ -n "$GS2_A" ] && W2="$W2@$GS2_A/$GS2_B"; [ -n "$TW2_A" ] && W2="$W2@$TW2_A/$TW2_B"; W="$W,$W2"
fi
keep qasr $ROOT/src/qwen3asr/.venv/bin/python judge_worker.py --kind qasr --port $P_QASR
keep psed $ROOT/src/psed/.venv/bin/python judge_worker.py --kind psed --port $P_PSED
wait_up omni $P_OMNI; [ -n "$MODEL_B" ] && wait_up omni3 $P_OMNI3; [ -n "$MODEL2" ] && wait_up omni2 $P_OMNI2; wait_up qasr $P_QASR; wait_up psed $P_PSED
PE=(); [ "$PER_ENGINE" = 1 ] && PE=(--per-engine)
keep serve uv run serve.py --prod --worker "$W" "${PE[@]}" --select $SELECT --qasr http://127.0.0.1:$P_QASR --psed http://127.0.0.1:$P_PSED --n $N --port $PORT
wait_up serve $PORT
echo "READY: POST http://127.0.0.1:$PORT/synthesize   (logs: $LOGS)"
# watchdog: a hung (not crashed) service -> kill it; its keep-loop restarts it
while true; do sleep 60
  for sp in omni:$P_OMNI $([ -n "$MODEL_B" ] && echo omni3:$P_OMNI3) $([ -n "$MODEL2" ] && echo omni2:$P_OMNI2) qasr:$P_QASR psed:$P_PSED serve:$PORT; do n=${sp%%:*}; p=${sp##*:}
    curl -sf -m 30 127.0.0.1:$p/health >/dev/null || { echo "[$(date +%T)] $n unhealthy -> restart" | tee -a "$LOGS/$n.log"; pkill -f -- "--port $p\b"; }
  done
done
