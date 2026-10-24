"""Build-time prefetch (network on, no GPU): instantiate every model the serving stack loads at run time, on CPU, so
the weights land in HF_HOME / the repos' checkpoint dirs inside the image. Run once per venv:
    <venv python> prefetch.py main|omni|qasr|psed
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # /opt/pf: frozen copy of the modules used here
what = sys.argv[1]
R = "/data/vinvtts/runs"

if what in ("main", "omni"):   # both venvs: sherpa Zipformer (bench / align) + AST (nvdet)
    from huggingface_hub import snapshot_download
    snapshot_download("csukuangfj2/sherpa-onnx-zipformer-vi-30M-2026-02-09")
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    from nvdet import AST
    ASTFeatureExtractor.from_pretrained(AST); ASTForAudioClassification.from_pretrained(AST)

if what == "main":
    from speechbrain.inference.speaker import EncoderClassifier
    EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})
    import safetensors.torch  # noqa: F401
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2ForAudioFrameClassification
    from yt_detect import Laugh
    Wav2Vec2ForAudioFrameClassification.from_pretrained(Laugh.W2V, num_labels=1)
    hf_hub_download("omine-me/LaughterSegmentation", "model.safetensors")
    import numpy as np
    from speechmos import dnsmos
    print("dnsmos", dnsmos.run(np.random.default_rng(0).uniform(-.1, .1, 16000 * 3).astype(np.float32), sr=16000))

if what == "omni":
    from huggingface_hub import snapshot_download
    snapshot_download("openai/whisper-large-v3-turbo")
    for m in ("cdpo1b", "odpo_spdl"):
        assert os.path.isdir(f"{R}/omni_dpo/{m}/audio_tokenizer"), m
    snapshot_download("eustlb/higgs-audio-v2-tokenizer")   # fallback path in OmniVoice.from_pretrained

if what == "qasr":
    import torch
    from qwen_asr import Qwen3ASRModel
    Qwen3ASRModel.from_pretrained(f"{R}/qasr/exp/checkpoint-270", dtype=torch.bfloat16, device_map="cpu")

if what == "psed":
    import torch
    torch.Tensor.cuda = lambda self, *a, **k: self      # psed_eval.load only builds the models; keep them on CPU
    from psed_eval import load
    for n in ["BEATs", "ATST-F", "M2D"]: load(n)
print("PREFETCH OK", what)
