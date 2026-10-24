# Submission image (VLSP 2026 ViNV-TTS, team `witake`)

The image runs the served v5 pipeline (README.md) as a batch job: `/input/input.csv` -> `/output/audios/<id>.wav`
(24 kHz, mono, PCM 16-bit) + `/output/output.csv` (the input columns unchanged + `output_audio_path`).

```bash
docker load -i witake.tar
docker run --rm --gpus '"device=0"' --network none --shm-size=8g \
  -v /path/to/test_data:/input:ro -v /path/to/outputs/witake:/output witake:latest
```

## What runs
`entrypoint.sh` -> `vinvtts/run_batch.py`:
1. starts the services of `launch.sh` on the one visible GPU (`cuda:0`): cdpo1b and odpo_spdl OmniVoice workers, tag-ASR v3
   judge, PretrainedSED judge; each PyTorch allocator is capped (11 / 11 / 7.5 GB) so the whole stack stays under ~35 GB;
2. every row goes through `prod.Production` with the Track-B (reference prompt) settings: `ref_audio` + `ref_text` are
   the prompt, `target_text` is synthesised (unknown `[...]` tags are dropped, tags are case-insensitive);
3. rows run in chunks of 8; `output.csv` is rewritten after every chunk, so rows finished before a time-limit kill count;
4. a failed row is retried with plain generation, then left with an empty `output_audio_path`; exit code 0 unless the
   stack could not start at all;
5. time budget: candidates per engine (8) step down to 4 / 2 / 1 when the measured speed projects past 55 min, and no
   chunk is started that would end after 59 min (`TIME_BUDGET`, `HARD_LIMIT`, from container start);
6. fixed seeds: each generation call is seeded from its texts and runs alone on its worker (reruns are equivalent;
   GPU kernels are not bit-exact).

Offline: every weight is in the image (`HF_HUB_OFFLINE=1`); run-time writes go to `/output` and `/tmp` only.
Torch in every venv is a CUDA 12.x build (driver 580 / CUDA 13.0 on the grading machine).

Measured on one H100 (50-row test in the official format): 7.6 min wall including 20 s start-up, peak VRAM 34 GB.

## Build and package
```bash
STAGE=<dir with the hf downloads of setup_box.sh: hfsprint/ hfm/ data/vinv-tts-hf/> bash docker/build.sh   # -> witake:latest
bash docker/package.sh ~/witake_submission      # -> witake_submission.zip (witake.tar at the root) + .sha256
```
Pretrained models and data in the image: [MODELS_AND_DATA.md](../MODELS_AND_DATA.md) (§1 engines, §3 judges); also
openai/whisper-large-v3-turbo (OmniVoice's built-in prompt ASR, used only when a prompt is longer than 20 s), MIT licence.
