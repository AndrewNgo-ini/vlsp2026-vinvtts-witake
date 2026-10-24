#!/bin/bash
# OmniVoice LoRA fine-tune with the official recipe (examples/config/train_config_finetune_lora.json: r16, a32,
# lr 1e-4, audio_embeddings/heads trained fully), our four tags as literal text, 3000 steps (5000 = ~77 epochs of our
# 5.9 h), checkpoints judged on evalF metrics, not loss (VoxCPM2, arXiv 2606.26618). Weights are CC-BY-NC.
export PATH=$HOME/.local/bin:$PATH HF_HOME=/mnt/vinvtts/hf OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 BENCH_WORKERS=1 TOKENIZERS_PARALLELISM=false UV_CACHE_DIR=/mnt/vinvtts/uvcache
R=/mnt/vinvtts/runs; D=/mnt/vinvtts/data/vinv-tts; OV=/mnt/vinvtts/src/omnivoice; PY=$OV/.venv/bin/python; STEPS=${STEPS:-3000}
# HOLDOUT=1: Track-B check -- drop the mkprobe.py holdout speakers from training, evaluate on SET=trackB
if [ -n "$HOLDOUT" ]; then O=$R/omni_ftB; EV=trackB; ZD=$R/zs_trackB; HO=$R/trackB/holdout.txt; else O=$R/omni_ft; EV=evalF; ZD=$R/zsF; HO=/dev/null; fi
# NAME=<suffix>: separate run dir + system names; OVERSAMPLE=1: sqrt(f_max/f) rare-tag oversampling (2609.14231);
# EXTRA=<manifest>: extra train rows (e.g. rejection-sampled self-generated data, 2609.11260); INIT=<ckpt> to start from
[ -n "$NAME" ] && O=$R/omni_ft_$NAME; P=omft${NAME:+_$NAME}
cd ${VINV_CODE:-/opt/vinvtts-submission/vinvtts}; uv sync -q; mkdir -p $O
step() { echo "== $(date +%T) START $1"; shift; "$@"; local rc=$?; echo "== $(date +%T) END rc=$rc"; }
$PY -c "import peft" 2>/dev/null || step peft bash -c "cd $OV && uv pip install -q --python $PY peft"
for s in train dev; do
  $PY - > $O/$s.jsonl <<PY
import json, re
from collections import Counter
ho = set(open("$HO").read().split())
ev = set(open("/mnt/vinvtts/runs/evalR/holdout.txt").read().split()) if "${EXCL_EVALR:-}" else set()   # evalR clips out of training
rows = [r for r in map(json.loads, open("$D/$s.jsonl")) if r["speaker"] not in ho and r["audio"] not in ev]
if "$s" == "train" and "${EXTRA:-}": rows += [json.loads(l) for l in open("${EXTRA:-/dev/null}")]
rep = [1] * len(rows)
if "$s" == "train" and "${OVERSAMPLE:-}":
    f = Counter(t for r in rows for t in re.findall(r"\[([a-z]+)\]", r["text"])); fm = max(f.values())
    rep = [max([round((fm / f[t]) ** 0.5) for t in set(re.findall(r"\[([a-z]+)\]", r["text"]))] or [1]) for r in rows]
for i, (r, k) in enumerate(zip(rows, rep)):
    for j in range(k):
        print(json.dumps({"id": f"{i:05d}_{j}", "audio_path": r["audio"], "text": r["text"], "language_id": "vi"}, ensure_ascii=False))
PY
  [ -f $O/tokens/$s/data.lst ] || step tok_$s bash -c "cd $OV && PYTHONPATH=$OV $PY -m omnivoice.scripts.extract_audio_tokens --input_jsonl $O/$s.jsonl --tar_output_pattern '$O/tokens/$s/audios/shard-%06d.tar' --jsonl_output_pattern '$O/tokens/$s/txts/shard-%06d.jsonl' --tokenizer_path eustlb/higgs-audio-v2-tokenizer --nj_per_gpu 2 --shuffle True"
done
$PY - <<PY
import json
c = json.load(open("$OV/examples/config/train_config_finetune_" + ("sdpa" if "${FULL:-}" else "lora") + ".json"))   # sdpa = official full-FT config (lr 1e-5)
c.update(steps=$STEPS, save_steps=500, eval_steps=500, num_workers=2)
if "${INIT:-}": c["init_from_checkpoint"] = "${INIT:-}"
json.dump(c, open("$O/train_config.json", "w"), indent=1)
json.dump({"train": [{"manifest_path": ["$O/tokens/train/data.lst"]}], "dev": [{"manifest_path": ["$O/tokens/dev/data.lst"]}]}, open("$O/data_config.json", "w"))
PY
step train bash -c "cd $OV && PYTHONPATH=$OV $OV/.venv/bin/accelerate launch --num_processes 1 --mixed_precision bf16 -m omnivoice.cli.train --train_config $O/train_config.json --data_config $O/data_config.json --output_dir $O/exp"
for ck in ${CKPTS:-1000 2000 3000}; do
  [ -d $O/exp/checkpoint-$ck ] || continue
  if [ -n "$FULL" ]; then   # full fine-tune: checkpoint weights + config/tokenizers of a merged OmniVoice dir
    mkdir -p $O/merged-$ck; cp -rn $R/omni_ft/merged-2000/. $O/merged-$ck/; cp $O/exp/checkpoint-$ck/model.safetensors $O/merged-$ck/model.safetensors
  else
  step merge_$ck bash -c "cd $OV && PYTHONPATH=$OV $OV/.venv/bin/omnivoice-merge-lora --base_model ${INIT:-k2-fsa/OmniVoice} --lora_adapter $O/exp/checkpoint-$ck --output_dir $O/merged-$ck"
  fi
  step gen_$P$ck $PY gen_zs.py omni $R/$EV/eval.jsonl $ZD/$P$ck $O/merged-$ck
done
L=""; for ck in ${CKPTS:-1000 2000 3000}; do [ -f $ZD/$P$ck.jsonl ] && L="$L $ZD/$P$ck.jsonl"; done
step bench uv run bench.py $L; step tags uv run tagasr.py score --ckpt $R/tagasr $L; step nvdet uv run nvdet.py score --model $R/nvdet $L
step v2s uv run tagasr.py score --ckpt $R/tagasr_v2s --suffix .tags_v2s.jsonl $L
uv run table.py $R/$EV/gt $ZD/omft2000 $(for m in $L; do echo ${m%.jsonl}; done)
echo "== OMNIFT_DONE"
