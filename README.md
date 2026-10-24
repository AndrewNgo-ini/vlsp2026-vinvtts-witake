# ViNV-TTS (VLSP 2026): Vietnamese TTS with inline non-verbal sounds

Team WITake (Trivita AI). Code for the VLSP 2026 ViNV-TTS task: Vietnamese speech synthesis that realises inline
`[laughter] [breathing] [sniff] [throatclearing]` tags at the tagged position.
- **Track A:** speakers seen in training.
- **Track B:** zero-shot cloning of unseen speakers from a reference clip.

The served system (**v5**) is an HTTP API built from our own fine-tunes of the open-source OmniVoice model.

## How it works (v5)

1. **Two fine-tuned OmniVoice engines** generate 8 candidates each:
   - **cdpo1b** (primary) = OmniVoice SFT on the official train set plus synthetic NV data, then class-aware DPO.
   - **odpo_spdl** = a second SFT + DPO line.
2. **Tag-CFG** during generation: classifier-free guidance against the same text with the tags removed, so the tagged NV
   is realised. Track A also re-generates ("inpaints") a missed tag up to 2 times.
3. **Best-of-16 selection** with the official score weights applied to automatic estimates: NV detection (v3 tag-ASR, AST-based detector), WER (Zipformer), DNSMOS, ECAPA speaker similarity.
4. **Splice:** where the judges still find a tag missing, a real NV clip from the official training data is inserted.
5. **NV levelling:** each NV's loudness relative to the surrounding speech is mapped onto the distribution measured on real recordings (1-D quantile/optimal-transport map per class). Before levelling, generated laughter and throat clearing were 3-4 dB too loud and sniffs 9-12 dB.

Public-test automatic score (official weights on the automatic terms only, 0-1): **A .863, B .879**. Human-rated terms are not included.

## Technical report

[report/main.pdf](report/main.pdf) (ACL format): method, ablations and negative results.

## Submission (Docker, batch)

The VLSP submission is a self-contained, offline Docker image (`witake:latest`) that runs this pipeline on
`/input/input.csv` and writes `/output/`: see [docker/README.md](docker/README.md).

## Quick start (serve)

Requirements: Linux, 2x NVIDIA H100 (or GPUs with 40 GB or more), ~150 GB disk, `uv`, `ffmpeg`, an `HF_TOKEN` with read access to the model repos.
```bash
sudo mkdir -p /opt /data && sudo chown $USER /data
git clone https://github.com/AndrewNgo-ini/vlsp2026-vinvtts-witake /opt/vinvtts-submission   # install path the scripts expect; or set VINV_CODE=<clone>/vinvtts
cd /opt/vinvtts-submission/vinvtts
HF_TOKEN=... bash setup_box.sh                         # models, judges, corpus, venvs, manifests (data root /data/vinvtts)
GPU=0 GPU2=1 PORT=8020 bash launch.sh                  # prints READY when all services are healthy
uv run robust_test.py --url http://127.0.0.1:8020      # edge cases: must print ROBUST ALL OK
```
API: `POST /synthesize` with `{"text", "track": "A"|"B", "speaker_id" (A), "reference_wav_b64", "reference_text"}`. It returns 24 kHz WAV. Details: [docs/DEPLOY.md](docs/DEPLOY.md).

To reproduce the public-test outputs and scores:
```bash
uv run run_final.py --url http://127.0.0.1:8020 --out /data/vinvtts/runs/final
bash w1score.sh 0 /data/vinvtts/runs/final/A /data/vinvtts/runs/final/B      # WER / DNSMOS / SS / NV proxies + offA/offB
```

## Repository layout

All code is in `vinvtts/`. It is a flat module directory: the scripts import each other by name, and `pyproject.toml` / `uv.lock` define the main environment.

| Group | Files |
|---|---|
| API and pipeline | `launch.sh`, `serve.py`, `prod.py`, `tts_worker.py`, `judge_worker.py`, `nvfix.py`, `cfggen.py`, `splice.py`, `nvlevel.py`, `nvloud.py`, `nvreal.py` |
| Judges / scorers | `nvdet.py`, `align.py`, `bench.py`, `judge.py`, `tags.py`, `yt_detect.py` (laughter judge), `throat_det.py`, `psed_eval.py`, `qasr_tag.py`, `dnsmos_gpu.py`, `audit_judge.py` |
| Setup, eval, tests | `setup_box.sh`, `rehearsal.sh`, `evalprod.sh`, `w1score.sh`, `offscore.py`, `uncanny.py`, `run_final.py`, `robust_test.py`, `mkpublic.py`, `eda.py`, `cpulimit.py`, `runwith.py` |
| Training recipes | see [docs/TRAINING.md](docs/TRAINING.md) |

## Models, data and licences

See [MODELS_AND_DATA.md](MODELS_AND_DATA.md) for every pretrained model, dataset and synthetic-data generator, with
licences. Third-party attribution notices are in [NOTICE](NOTICE).
