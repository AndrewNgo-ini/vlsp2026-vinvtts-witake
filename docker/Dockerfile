# syntax=docker/dockerfile:1
# VLSP 2026 ViNV-TTS submission image (team witake). Build: docker/build.sh (see docker/README.md).
# Layout = setup_box.sh: code /opt/vinvtts-submission/vinvtts, data root /data/vinvtts (/mnt/vinvtts -> /data/vinvtts).
# Torch in every venv is a CUDA 12.x build (grading driver 580 = CUDA <= 13.0).
FROM nvidia/cuda:12.6.3-base-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    UV_PYTHON_INSTALL_DIR=/opt/uvpython UV_CACHE_DIR=/tmp/uvcache UV_LINK_MODE=copy \
    ROOT=/data/vinvtts HF_HOME=/data/vinvtts/hf PATH=/opt/uv:$PATH
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg git ca-certificates curl build-essential libsndfile1 \
 && rm -rf /var/lib/apt/lists/* && mkdir -p /opt/uv && curl -LsSf https://astral.sh/uv/0.9.5/install.sh | env UV_INSTALL_DIR=/opt/uv UV_NO_MODIFY_PATH=1 sh \
 && mkdir -p /data/vinvtts/src /mnt && ln -s /data/vinvtts /mnt/vinvtts

COPY docker/cu126.sh /opt/cu126.sh
# --- main venv (pipeline, scoring) ---
COPY code/pyproject.toml code/uv.lock /opt/vinvtts-submission/vinvtts/
RUN --mount=type=cache,target=/tmp/uvcache cd /opt/vinvtts-submission/vinvtts && uv sync --frozen \
 && bash /opt/cu126.sh .venv
# --- OmniVoice venv (pinned commit) ---
RUN --mount=type=cache,target=/tmp/uvcache cd /data/vinvtts/src && git clone https://github.com/k2-fsa/OmniVoice omnivoice && cd omnivoice && git checkout 08be0b4ccbac3e13e374e86fbfead4b4cac343e2 \
 && uv sync --frozen && uv pip install fastapi uvicorn librosa jiwer sherpa-onnx sherpa-onnx-core peft   # torch 2.8.0+cu128 (its lock)
# --- Qwen3-ASR venv (tag-ASR v3 judge) ---
RUN --mount=type=cache,target=/tmp/uvcache mkdir -p /data/vinvtts/src/qwen3asr && cd /data/vinvtts/src/qwen3asr && uv venv -p 3.11 && uv pip install qwen-asr datasets fastapi uvicorn \
 && bash /opt/cu126.sh .venv
# --- PretrainedSED venv (rare-class judge; torch 2.5.1 = cu124) ---
RUN --mount=type=cache,target=/tmp/uvcache cd /data/vinvtts/src && git clone https://github.com/fschmid56/PretrainedSED psed && cd psed && git checkout 1aa47e482f \
 && uv venv -p 3.10 && uv pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 \
 && grep -v sed_scores_eval requirements.txt > req2.txt && uv pip install -r req2.txt fastapi uvicorn soundfile einops

# --- models, judges, splice bank, official corpus (staged by docker/build.sh) ---
COPY root/ /data/vinvtts/
RUN set -e; R=/data/vinvtts/runs; mkdir -p $R/omni_dpo $R/qasr/exp $R/splice $R/ar_uncanny /data/vinvtts/backup/data \
 && ln -sfn /data/vinvtts/hfsprint/models/omnivoice/cdpo1b $R/omni_dpo/cdpo1b \
 && ln -sfn /data/vinvtts/hfm/omnivoice/odpo_spdl $R/omni_dpo/odpo_spdl \
 && ln -sfn /data/vinvtts/hfm/scorers/tagasr_v3_qwen3asr $R/qasr/exp/checkpoint-270 \
 && ln -sfn /data/vinvtts/hfm/scorers/nvdet $R/nvdet \
 && ln -sfn /data/vinvtts/hfm/scorers/rare_judge_psed_stack $R/rare_judge \
 && ln -sfn /data/vinvtts/hfsprint/splice_bank $R/splice/bank \
 && ln -sfn /data/vinvtts/hfm/data/events /data/vinvtts/backup/data/events \
 && for f in nvreal.json calib_pooled.loud.jsonl gt_train.loud.jsonl; do ln -sfn /data/vinvtts/hfsprint/ar_uncanny_1001/$f $R/ar_uncanny/$f; done \
 && ln -sfn /data/vinvtts/hfsprint/ar_uncanny_1001/nvreal.json $R/nvreal.json

# --- manifests + prefetch of every model loaded at run time (the container runs with --network none). Uses a frozen copy
# of the few modules it needs (/opt/pf), so code changes below do not redo this layer. ---
COPY code/cpulimit.py code/judge.py code/tags.py code/nvdet.py code/yt_detect.py code/psed_eval.py code/eda.py docker/prefetch.py /opt/pf/
RUN set -e; V=/opt/vinvtts-submission/vinvtts/.venv/bin/python; cd /opt/pf \
 && $V eda.py /data/vinvtts/data/vinv-tts-hf > /dev/null && ln -sfn /data/vinvtts/data/vinv-tts-hf /data/vinvtts/data/vinv-tts \
 && $V -c "import pickle,numpy as np; R='/data/vinvtts/runs'; c=pickle.load(open(f'{R}/nvdet/clf.pkl','rb')); s,l=c[0],c[-1]; np.savez(f'{R}/nvdet/clf_np.npz',mean=s.mean_,scale=s.scale_,coef=l.coef_,intercept=l.intercept_,classes=np.array(l.classes_))" \
 && $V prefetch.py main \
 && /data/vinvtts/src/omnivoice/.venv/bin/python prefetch.py omni \
 && /data/vinvtts/src/qwen3asr/.venv/bin/python prefetch.py qasr \
 && /data/vinvtts/src/psed/.venv/bin/python prefetch.py psed \
 && for v in /opt/vinvtts-submission/vinvtts/.venv /data/vinvtts/src/*/.venv; do $v/bin/python -m compileall -q -j 8 $v/lib > /dev/null || true; done \
 && rm -rf /tmp/* && chmod -R a+rX /data /opt

# --- serving code ---
COPY code/ /opt/vinvtts-submission/vinvtts/
COPY docker/entrypoint.sh /opt/vinvtts-submission/docker/
WORKDIR /opt/vinvtts-submission/vinvtts

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/tmp/numba MPLCONFIGDIR=/tmp/mpl TORCHINDUCTOR_CACHE_DIR=/tmp/inductor TRITON_CACHE_DIR=/tmp/triton \
    PROD_REFS=/tmp/prod_refs LOGS=/tmp/logs OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
ENTRYPOINT ["/bin/bash", "/opt/vinvtts-submission/docker/entrypoint.sh"]
