#!/bin/bash
# Rebuild the production TTS stack (v5) on a fresh box: models, judges, data, venvs (docs/DEPLOY.md).
# Usage: HF_TOKEN=... bash setup_box.sh   (repo cloned at /opt/vinvtts-submission or VINV_CODE=<repo>/vinvtts; data root /data/vinvtts)
set -x
[ -n "$HF_TOKEN" ] || { echo "set HF_TOKEN (read access to the model repos)"; exit 1; }
ROOT=/data/vinvtts; C=${VINV_CODE:-/opt/vinvtts-submission/vinvtts}
export HF_HOME=$ROOT/hf UV_CACHE_DIR=$ROOT/uvcache PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=2 HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p $ROOT/{runs,data,src,hf,backup/data}
# older scripts and manifests use /mnt/vinvtts; keep it pointing at the data root
[ -e /mnt/vinvtts ] || { sudo mkdir -p /mnt && sudo ln -sfn $ROOT /mnt/vinvtts; }
command -v hf || uv tool install "huggingface_hub[cli]"

# --- models / scorers / bank ---
hf download trivitaai/vlsp2026-vinvtts-sprint --include "splice_bank/*" --include "reports/*" --include "models/omnivoice/cdpo1b/*" --local-dir $ROOT/hfsprint
hf download trivitaai/vlsp2026-vinvtts-models --include "omnivoice/odpo_spdl/*" --include "scorers/tagasr_v3_qwen3asr/*" --local-dir $ROOT/hfm
hf download hiieu/vinvtts-models --include "omnivoice/omft_tdl4000/*" --include "scorers/nvdet/*" --include "scorers/rare_judge_psed_stack/*" --include "data/events/*" --local-dir $ROOT/hfm
hf download trivitaai/vinv-tts --repo-type dataset --local-dir $ROOT/data/vinv-tts-hf
hf download trivitaai/vlsp2026-vinvtts-sprint-data --repo-type dataset --include "clean/*" --local-dir $ROOT/hfdata

hf download trivitaai/vlsp2026-vinvtts-sprint --include "scorers/audit_judge/*" --local-dir $ROOT/hfsprint
# uncanny-NV fix files (levelling calibration, real-NV realism model, real train NV features)
hf download trivitaai/vlsp2026-vinvtts-sprint --include "ar_uncanny_1001/nvreal.json" --include "ar_uncanny_1001/calib_pooled.loud.jsonl" --include "ar_uncanny_1001/gt_train.loud.jsonl" --local-dir $ROOT/hfsprint
R=$ROOT/runs; mkdir -p $R/omni_dpo $R/omni_ft_tdl $R/qasr/exp $R/splice
ln -sfn $ROOT/hfsprint/models/omnivoice/cdpo1b $R/omni_dpo/cdpo1b
ln -sfn $ROOT/hfm/omnivoice/odpo_spdl $R/omni_dpo/odpo_spdl
ln -sfn $ROOT/hfm/omnivoice/omft_tdl4000 $R/omni_ft_tdl/merged-4000
ln -sfn $ROOT/hfm/scorers/tagasr_v3_qwen3asr $R/qasr/exp/checkpoint-270
ln -sfn $ROOT/hfm/scorers/nvdet $R/nvdet
# nvfix (Track-A inpaint verify) reads a numpy export of nvdet's sklearn pipeline; it was never on HF -> rebuild it
cd $C && uv run python -c "import pickle,numpy as np; c=pickle.load(open('$R/nvdet/clf.pkl','rb')); s,l=c[0],c[-1]; np.savez('$R/nvdet/clf_np.npz',mean=s.mean_,scale=s.scale_,coef=l.coef_,intercept=l.intercept_,classes=np.array(l.classes_))"
ln -sfn $ROOT/hfm/scorers/rare_judge_psed_stack $R/rare_judge
ln -sfn $ROOT/hfsprint/splice_bank $R/splice/bank
ln -sfn $ROOT/hfsprint/scorers/audit_judge $R/audit_judge
ln -sfn $ROOT/hfm/data/events $ROOT/backup/data/events
mkdir -p $R/ar_uncanny; for f in nvreal.json calib_pooled.loud.jsonl gt_train.loud.jsonl; do ln -sfn $ROOT/hfsprint/ar_uncanny_1001/$f $R/ar_uncanny/$f; done
ln -sfn $ROOT/hfsprint/ar_uncanny_1001/nvreal.json $R/nvreal.json

# --- venvs ---
cd $C && uv sync
cd $ROOT/src && [ -d omnivoice ] || git clone https://github.com/k2-fsa/OmniVoice omnivoice
cd $ROOT/src/omnivoice && uv sync && uv pip install fastapi uvicorn librosa jiwer sherpa-onnx sherpa-onnx-core peft   # jiwer + sherpa: Track-A inpaint (align.py); peft: dpo_omni.py
mkdir -p $ROOT/src/qwen3asr && cd $ROOT/src/qwen3asr && { [ -d .venv ] || uv venv -p 3.11; } && uv pip install qwen-asr datasets fastapi uvicorn
cd $ROOT/src && [ -d psed ] || git clone https://github.com/fschmid56/PretrainedSED psed
cd $ROOT/src/psed && { [ -d .venv ] || uv venv -p 3.10; } && uv pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 && \
  { grep -v sed_scores_eval requirements.txt > req2.txt; uv pip install -r req2.txt fastapi uvicorn soundfile einops; }   # sed_scores_eval (eval-only) fails to build
command -v ffmpeg || { sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg; }
# --- manifests: corpus (Track-A reference clips) + public test ---
cd $C && uv run python eda.py $ROOT/data/vinv-tts-hf > /dev/null && ln -sfn $ROOT/data/vinv-tts-hf $ROOT/data/vinv-tts
uv run python mkpublic.py $ROOT/data/vinv-tts-hf $ROOT/runs/public
echo SETUP_DONE
