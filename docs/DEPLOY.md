# Deploying the ViNV-TTS API (v5)

## 1. Setup
```bash
sudo mkdir -p /opt /data && sudo chown $USER /data
git clone https://github.com/AndrewNgo-ini/vlsp2026-vinvtts-witake /opt/vinvtts-submission   # elsewhere: export VINV_CODE=<clone>/vinvtts
cd /opt/vinvtts-submission/vinvtts
HF_TOKEN=... bash setup_box.sh
```
`setup_box.sh` (logs: `/data/vinvtts/runs/setup_box.log`; it is idempotent, so a re-run is safe):
- downloads the engines, judges, splice bank, NV-levelling files and the official corpus into `/data/vinvtts`, and links them where the code expects them;
- builds four `uv` environments:

| Environment | Used by |
|---|---|
| `vinvtts/.venv` (pyproject + uv.lock) | serve, pipeline, scoring |
| `/data/vinvtts/src/omnivoice/.venv` | OmniVoice workers, DPO training |
| `/data/vinvtts/src/qwen3asr/.venv` | tag-ASR v3 judge |
| `/data/vinvtts/src/psed/.venv` | PretrainedSED rare-class judge |

- writes the corpus and public-test manifests.

## 2. Run
```bash
GPU=0 GPU2=1 PORT=8020 bash launch.sh
```
- Starts five services, each under an auto-restart loop:
  - two OmniVoice workers: cdpo1b on :8011 (GPU), odpo_spdl on :8014 (GPU2);
  - the tag-ASR v3 judge on :8012;
  - the PretrainedSED judge on :8013;
  - the API on :8020.
- Waits for every `/health` and prints `READY`. A watchdog restarts hung services.
- Logs go to `/data/vinvtts/runs/serve_logs/`.

Switches (environment variables; the defaults are the submitted v5 configuration):
| Variable | Default | Meaning |
|---|---|---|
| `MODEL`, `MODEL2` | `runs/omni_dpo/cdpo1b`, `runs/omni_dpo/odpo_spdl` | primary / second engine (`MODEL2=` = one engine) |
| `GS_A GS_B TW_A TW_B` | 2 6 3 5 | primary engine guidance scale and tag-CFG weight per track |
| `GS2_A GS2_B TW2_A TW2_B` | 7 7 3 3 | second engine |
| `INPAINT_K_A` | 2 | Track A verify-and-inpaint rounds (0 = off) |
| `N` | 8 | candidates per engine |
| `NV_LEVEL` | `qmap` | NV levelling (`off` = v4 behaviour) |
| `MODEL_B` | (unset) | optional Track-B-only primary engine |
| plain fallback | | `TAG_W=0 TW_A= TW_B= TW2_A= TW2_B= INPAINT_K_A=0` |

## 3. API
`POST /synthesize`
```json
{"text": "...[laughter]...", "track": "A" | "B",
 "speaker_id": "spk_0000",
 "reference_wav_b64": "<wav/flac, any rate, mono or stereo>",
 "reference_text": "..."}
```
- `speaker_id` is for Track A (optional when a prompt is sent). `reference_text` is optional; if missing, OmniVoice runs ASR on the prompt.
- **Response:** `audio/wav`, 24 kHz PCM16. The `X-Info` header holds base64 JSON: the chosen candidate's scores, splice and levelling decisions, timings.
- **Input handling:** text is NFC-normalised, lowercased and number-spelled; tags are case-insensitive.
- **Errors:** unknown tags, silent prompts and prompts under 0.5 s return 400. Prompts over 20 s are cut to 15 s and their transcript is dropped.
- **Robustness:** a failing batch is retried per request, then with plain generation.
- **Prompt length:** use prompts of 4-5 s or more. Very short prompts (~2 s) still return audio but degrade.
- **Throughput:** the public test (330 items) takes about 37 min on 2 H100 at 32 concurrent requests, p50 latency about 210 s.

## 4. Checks
```bash
uv run robust_test.py --url http://127.0.0.1:8020                               # must print ROBUST ALL OK
uv run run_final.py --url http://127.0.0.1:8020 --out /data/vinvtts/runs/final   # full public test
bash w1score.sh 0 /data/vinvtts/runs/final/A /data/vinvtts/runs/final/B          # automatic proxies + offA / offB
bash rehearsal.sh /data/vinvtts/runs/rehearsal                                   # all of the above in one go
```
