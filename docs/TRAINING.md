# Training recipes for the served models

Every served TTS engine is our own fine-tune of `k2-fsa/OmniVoice`. The steps below rebuild each component. Run them from
`vinvtts/` with data root `/data/vinvtts`; `setup_box.sh` creates it and links `/mnt/vinvtts` to it. Released weights for
all components are listed in [../MODELS_AND_DATA.md](../MODELS_AND_DATA.md).

**Reproducibility note.** Each component's training data and recipe are documented here. Generative steps (candidate sampling,
Seed-VC voice conversion) are stochastic, so a rebuild matches the released weights in method, not bit for bit. The
released weights are the reference artefacts.

## 0. Shared: corpus manifests and NV event bank
```bash
uv run eda.py /data/vinvtts/data/vinv-tts-hf          # official train/dev -> train.jsonl / dev.jsonl
uv run align.py /data/vinvtts/data/vinv-tts/train.jsonl /data/vinvtts/runs/events
uv run align.py /data/vinvtts/data/vinv-tts/dev.jsonl   /data/vinvtts/runs/events_dev
```
`align.py` uses Zipformer word timestamps to cut each tagged gap of the official recordings into an NV event clip.

## 1. NV detector (`nvdet`), used by the selector, splice and DPO pair ranking
```bash
uv run nvdet.py train --events runs/events --train train.jsonl --dev-events runs/events_dev --dev dev.jsonl --out runs/nvdet
```
The model is an AST (`MIT/ast-finetuned-audioset-10-10-0.4593`) embedding with a logistic-regression head, trained on official train events only.
`setup_box.sh` exports `runs/nvdet/clf_np.npz` from it (needed by `nvfix.py`).

## 2. Rare-class judge (sniff / throat clearing)
```bash
$PSED_PY psed_eval.py BEATs ATST-F M2D        # PretrainedSED frame probabilities on the official events -> runs/psed_*.jsonl
uv run rare_judge.py fit                      # -> runs/rare_judge/{sniff,throatclearing}.pkl
```
`$PSED_PY` is the PretrainedSED venv that `setup_box.sh` builds. The helper scripts are `psed_score.py` and `throat_det.py`; `run_psed.sh` wraps both.

## 3. Tag-ASR v3 (NV-aware ASR judge)
```bash
$QP qasr_tag.py prep                          # official train with inline tags, sqrt class oversampling
$QP qasr_sft.py --model_path Qwen/Qwen3-ASR-1.7B --train_file runs/qasr/train.jsonl --eval_file runs/qasr/dev.jsonl \
    --output_dir runs/qasr/exp --lr 2e-5 --epochs 5 --save_steps 90       # served checkpoint: checkpoint-270
```
`qasr_sft.py` wraps the official Qwen3-ASR fine-tuning script. Clone `github.com/QwenLM/Qwen3-ASR` to `src/qwen3asr` first. `run_qasr.sh` has the full command.

## 4. Synthetic NV training data
- **Throat clearing / sniff (SynParaSpeech-style).**
  - VocalSound events are voice-converted to an official-train host voice with Seed-VC and inserted into host utterances.
  - Candidates are filtered with the rare-class judge and AST embeddings.
  - Commands: `synpara.py make` then `filter` (original path, `run_synpara.sh`), or `synpara_regen.py` (re-generation from the saved selection manifest).
  - The regenerated set (277 clips) is released in the data repo, under `synpara_regen/`.
- **Laughter clips.**
  - Sampled from our fine-tuned VieNeu-TTS model (`dpo6b2Lg3`, released) on official train texts: `distill.py prompts`, then `gen_zs.py vieneu`.
  - Filtered with LaughterSegmentation, nvdet and WER <= .10: `distill.py filter` (`run_dlom_a.sh`).

## 5. omft_tdl4000 (SFT; parent of cdpo1b)
```bash
NAME=tdl OVERSAMPLE=1 STEPS=5000 CKPTS=4000 EXTRA=<kept_throat_dl.jsonl> bash run_omni_ft.sh    # see run_tdl.sh
```
`kept_throat_dl.jsonl` holds the synthetic throat clips plus the laughter clips; `run_tdl.sh` builds it inline.
`run_omni_ft.sh` then:
- tokenises audio with the OmniVoice audio tokenizer (Higgs Audio 2);
- trains LoRA with the official OmniVoice recipe (`omnivoice.cli.train`, `train_config_finetune_lora.json`), official train data oversampled for rare classes;
- merges step 4000 (`omnivoice-merge-lora`).

## 6. cdpo1b (primary engine): class-aware DPO from omft_tdl4000
1. **Prompts:** `cdpo_mkprompts.py`, 1921 prompts.
   - 1660 are football-talk sentences with one NV tag inserted (1-2 tags; 15% at sentence start, 15% at sentence end).
     The sentences were transcribed from public football-talk videos (see MODELS_AND_DATA.md). Only text was used.
   - 261 are single-class official train texts.
   - References are 3-9 s clips of the same speaker from the official train set. No public-test text or audio was used.
2. **Candidates:** `cdpo_gen.sh` runs omft_tdl4000 with tag-CFG off, 4 candidates at gs 2 and 4 at gs 6 (8 per prompt).
3. **Per-tag judging:** nvdet, plus the class judge (LaughterSegmentation for laughter, the rare-class judge for sniff/throat clearing, nvdet for breathing).
4. **Pairs:** `cdpo_mkpairs.py`.
   - chosen = every tag realised, WER <= .15, SS >= median - .05; among those, the best by WER/SS/pMOS.
   - rejected = the same prompt with the target-class tag missing.
   - Class-balanced: 649 pairs (breathing 179, throat clearing 179, laughter 153, sniff 138).
5. **Training:** `dpo_omni.py train <pairs.jsonl> runs/omni_dpo/cdpo1b --model runs/omni_ft_tdl/merged-4000 --beta 5 --lr 5e-5 --epochs 2`.
   - This is masked-CE DPO for masked non-autoregressive TTS; the method and references are in `dpo_omni.py`.
   - 1298 steps, then merged.

The `cdpo_*` builder scripts in this repo are the round-3 versions of the builder, kept as backups. Rounds 1-2 (cdpo1b) used the same procedure with the parameters above.

## 7. odpo_spdl (second engine)
1. SFT: `NAME=spdl ... bash run_omni_ft.sh` (see `run_spdl_om.sh`). It uses the synthetic throat and sniff clips plus the laughter clips, giving omft_spdl4000.
2. On-policy candidates on 600 official train prompts x 4 (`run_odpo_spdl.sh` PART=1 -> `run_omni_rs.sh`). They are ranked by:
   - `bestof.py`, using our Whisper-based tag-ASR v1 (`tagasr.py`);
   - nvdet.
3. DPO: `run_odpo_spdl.sh` PART=2 -> `run_dpo_omni.sh` (`dpo_omni.py pairs`, then `train --beta 5`).

## 8. Splice bank
```bash
uv run splice.py bank --out runs/splice/bank      # real NV clips from the official train events, chosen by nvdet, with ECAPA speaker embeddings
```

## 9. NV levelling calibration
```bash
uv run nvloud.py GT_train=/data/vinvtts/data/vinv-tts/train.jsonl          # real NV level/shape distributions (official train)
uv run nvloud.py sA=<served A.jsonl> sB=<served B.jsonl>                   # the system's own NV level distribution (calibration)
uv run nvreal.py fit <train.loud.jsonl> runs/nvreal.json                    # NV realism model (optional selector term, off in v5)
```
`launch.sh` passes these files to the server (`NV_LEVEL_*`). The served calibration is the pooled distribution of two prod-like runs.
