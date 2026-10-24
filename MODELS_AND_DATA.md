# Models, data and licences

## 1. Served TTS engines (all our own fine-tunes)
| Engine | Base | Training | Base licence |
|---|---|---|---|
| **cdpo1b** (primary) | k2-fsa/OmniVoice | SFT on the official train set + synthetic NV data (omft_tdl4000), then class-aware DPO ([docs/TRAINING.md](docs/TRAINING.md) §5–6) | code Apache-2.0; weights CC-BY-NC |
| **odpo_spdl** (second) | k2-fsa/OmniVoice | SFT (omft_spdl4000) + masked-CE DPO (§7) | same |
| OmniVoice audio tokenizer (bundled in both) | Boson AI Higgs Audio 2 tokenizer | used unmodified | Boson Higgs Audio 2 Community License (based on the Meta Llama 3 Community License), see [NOTICE](NOTICE) |

No untouched pretrained TTS model is served.

## 2. Training data and synthetic-data generators
| Item | Kind | Used for | Licence |
|---|---|---|---|
| ViNV-TTS official train set | official | SFT, DPO prompts and references, splice bank, judges, NV-level calibration | competition data |
| VocalSound | external dataset | NV events (throat clearing, sniff) inserted into official-train host speech | CC BY-SA 4.0 |
| Seed-VC (Plachtaa/seed-vc) | synthetic-data generator (voice conversion) | converting VocalSound events to the host voice | GPL-3.0 |
| VieNeu-TTS v3 Turbo, fine-tuned by us | synthetic-data generator (TTS) | laughter clips in the SFT mix, generated from official train texts | Apache-2.0 |
| Football-talk sentences (text only) | external text | 1660 of the 1921 DPO prompt texts for cdpo1b; transcribed by our ASR from public Vietnamese football-talk YouTube videos | YouTube content (not an open-licence dataset) |

No YouTube audio is used in any served model. Only the transcribed sentences were used, as text to synthesise.

## 3. Models used for selection, filtering and DPO pair ranking (they do not generate audio)
| Model | Role | Licence |
|---|---|---|
| Qwen/Qwen3-ASR-1.7B, fine-tuned by us (tag-ASR v3) | selector NV term, splice decision | Apache-2.0 |
| openai/whisper-large-v3-turbo, fine-tuned by us (tag-ASR v1) | ranked odpo_spdl's DPO pairs | MIT |
| openai/whisper-large-v3-turbo, unmodified (OmniVoice's built-in prompt ASR) | transcribes a reference prompt only when no usable transcript is given (prompts over 20 s) | MIT |
| MIT/ast-finetuned-audioset-10-10-0.4593 + our classifier head (nvdet) | selector NV term, DPO ranking, splice bank | BSD-3-Clause |
| omine-me/LaughterSegmentation | laughter judge (splice, DPO pairs, laughter-clip filter) | code MIT; trained model: research use only |
| PretrainedSED (fschmid56; BEATs / ATST-F / M2D) + our sniff/throat heads | rare-class judge | MIT |
| csukuangfj2/sherpa-onnx-zipformer-vi-30M (export of hynt/Zipformer-30M-RNNT-6000h) | WER term, NV alignment (splice, levelling); used unmodified | CC-BY-NC-ND 4.0 |
| speechbrain/spkrec-ecapa-voxceleb | speaker-similarity term, splice clip choice | Apache-2.0 |
| DNSMOS (Microsoft DNS-Challenge, via `speechmos`) | quality term | MIT (repository) |
| laion/clap-htsat-unfused + microsoft/wavlm-base-plus (audit judge) | evaluation only, not in the serving path | Apache-2.0 / Microsoft |

## 4. Released artefacts (Hugging Face, private; access on request)
| Artefact | Location |
|---|---|
| cdpo1b | `trivitaai/vlsp2026-vinvtts-sprint` → `models/omnivoice/cdpo1b` |
| odpo_spdl, tag-ASR v3 | `trivitaai/vlsp2026-vinvtts-models` → `omnivoice/odpo_spdl`, `scorers/tagasr_v3_qwen3asr` |
| omft_tdl4000, nvdet, rare-class judge, NV event bank | `hiieu/vinvtts-models` → `omnivoice/omft_tdl4000`, `scorers/{nvdet,rare_judge_psed_stack}`, `data/events` |
| Splice bank, NV-levelling calibration, public-test outputs (v5) | `trivitaai/vlsp2026-vinvtts-sprint` → `splice_bank/`, `ar_uncanny_1001/`, `public_final_v5_levelling/` |
| Synthetic NV clips, football-talk sentences | `trivitaai/vlsp2026-vinvtts-sprint-data` → `synpara_regen/`, `crawl2/meta/football_talk_sentences.txt` |
| VieNeu laughter generator (dpo6b2Lg3) | `hiieu/vinvtts-models` → `vieneu/dpo6b2Lg3` |
