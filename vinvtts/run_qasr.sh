#!/bin/bash
# Tag-ASR v3 = Qwen3-ASR-1.7B full SFT + sqrt sampling (qasr_tag.py; 2609.23462). Part 1 (< 2 h): prep + train
# 5 epochs (official qwen3_asr_sft.py, lr 2e-5, bf16, sdpa), checkpoints every 200 steps. RESUME=1 continues a split run.
# Part 2 (EVAL=1): score evalF GT with the last checkpoint + per-class table vs the pre-registered bars.
export PATH=$HOME/.local/bin:$PATH HF_HOME=/mnt/vinvtts/hf OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false UV_CACHE_DIR=/mnt/vinvtts/uvcache
R=/mnt/vinvtts/runs; Q=/mnt/vinvtts/src/qwen3asr; QP=$Q/.venv/bin/python; O=$R/qasr
cd ${VINV_CODE:-/opt/vinvtts-submission/vinvtts}; mkdir -p $O
[ -f $O/dev.jsonl ] || $QP qasr_tag.py prep
if [ -z "$EVAL" ]; then
  $QP qasr_sft.py --model_path Qwen/Qwen3-ASR-1.7B --train_file $O/train.jsonl --eval_file $O/dev.jsonl --output_dir $O/exp \
    --batch_size ${BS:-8} --grad_acc ${GA:-4} --lr 2e-5 --epochs ${EPOCHS:-5} --save_steps 90 --save_total_limit 3 \
    --num_workers 2 --persistent_workers 0 --resume ${RESUME:-0}
fi
CK=$(ls -d $O/exp/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1); echo "== last checkpoint $CK"
if [ -n "$CK" ]; then
  $QP qasr_tag.py score $CK $R/evalF/gt.jsonl
  uv sync -q; uv run tag_perclass.py $R/evalF/gt.tags_v2s.jsonl $R/evalF/gt.tags_qasr.jsonl
fi
echo "== QASR_DONE"
