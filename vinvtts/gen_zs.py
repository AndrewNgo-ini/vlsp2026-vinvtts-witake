"""Zero-shot baselines on eval.jsonl -> <out>/<id>.wav + <out>.jsonl (gen, text, ref) for bench.py.
Run with each engine's own venv:
    /mnt/vinvtts/src/vieneu/.venv/bin/python gen_zs.py vieneu eval.jsonl out/vieneu
    /mnt/vinvtts/src/f5vi/.venv/bin/python   gen_zs.py f5vi   eval.jsonl out/f5vi
    .../vieneu/.venv/bin/python gen_zs.py vieneu eval.jsonl out/ft <merged_model_dir>   # our fine-tune: emotion
        tokens from prep_vieneu.TAG2EMO, speaker embedding only (no ref codes), as in training
    /mnt/vinvtts/src/omnivoice/.venv/bin/python gen_zs.py omni eval.jsonl out/omni
Tags: VieNeu has native [cười] [hắng giọng] [thở dài]; OmniVoice has [laughter] [sigh]; F5-vi has none (dropped).
"""
import cpulimit  # noqa: F401  (must precede torch/onnxruntime use)
import json, re, sys, time
from pathlib import Path
import soundfile as sf

eng, man, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in open(man)]
NOTAG = __import__("os").environ.get("NOTAG") == "1"   # negative control: synthesise without tags, score against tagged text
VN = {"laughter": "[cười]", "throatclearing": "[hắng giọng]", "breathing": "[thở dài]", "sniff": ""}
strip = lambda t: " ".join(re.sub(r"\[[a-z]+\]", " ", t).split())

if eng == "vieneu":
    from vieneu import Vieneu
    ft = sys.argv[4] if len(sys.argv) > 4 else None
    tts = Vieneu(mode="v3turbo", backbone_repo=ft or "/mnt/vinvtts/models/VieNeu-TTS-v3-Turbo")
    if ft:
        from prep_vieneu import to_vieneu
        rc = __import__("os").environ.get("REFCODES") == "1"   # in-context ref codes at inference (not seen in training)
        prep = to_vieneu; kw = {"use_ref_codes": rc}
    else:
        prep = lambda t: re.sub(r"\[([a-z]+)\]", lambda m: VN[m.group(1)], t); kw = {}
    # MAXCHARS: SDK text-chunk cap (default 256 chars -> 45% of evalF is split into independently generated chunks with
    # comma->period rewrites, while train_lora trains on whole utterances up to 607 chars); large value = no chunking
    # (whole-utterance generation also needs a larger frame budget than the per-call default 300 = 24 s; fused buffer is 512)
    if __import__("os").environ.get("MAXCHARS"): kw.update(max_chars=int(__import__("os").environ["MAXCHARS"]), max_new_frames=450)
    synth = lambda r: (tts.infer(prep(r["text"]), ref_audio=r["ref"], apply_watermark=False, **kw), tts.sample_rate)
    # N candidates share every forward step (GPU static batching in the SDK)
    synth_n = lambda r, n: ([(w, tts.sample_rate) for w in tts.infer_batch([prep(r["text"])] * n, ref_audio=r["ref"], apply_watermark=False, **kw)])
elif eng == "omni":
    import torch
    from omnivoice import OmniVoice
    om = (sys.argv[4] if len(sys.argv) > 4 else "") or "k2-fsa/OmniVoice"   # or a merged LoRA fine-tune (run_omni_ft.sh)
    tts = OmniVoice.from_pretrained(om, device_map="cuda:0", dtype=torch.float16)
    OM = {"laughter": "[laughter]", "breathing": "[sigh]", "sniff": "", "throatclearing": ""}
    if om != "k2-fsa/OmniVoice": OM = {t: f"[{t}]" for t in OM}   # fine-tuned on our literal tags
    import os
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
    gc = OmniVoiceGenerationConfig(guidance_scale=float(os.environ.get("OMNI_GS", 2.0)), num_step=int(os.environ.get("OMNI_STEPS", 32)),
                                   class_temperature=float(os.environ.get("OMNI_CT", 0.0)),
                                   postprocess_output=os.environ.get("OMNI_POSTPROC", "1") == "1")   # 0: no edge trim/mid-silence cut
    if os.environ.get("OMNI_DURPRIOR") == "1":
        # NAR total length comes from a char-weight rule that counts "[breathing]" as 10 letters (~0.66 s at the median
        # ref rate) and "[sniff]" as 6 (~0.40 s). Duration prior instead: rule(text without tags) + per-class median event
        # span from the train event bank (runs/events: breathing .32, laughter/sniff/throat .52 s), via the SDK's own
        # target-length hook (same ref-token count as the default path).
        _MED = {"laughter": .52, "breathing": .32, "sniff": .52, "throatclearing": .52}; _orig = tts._estimate_target_tokens
        _fr = tts.audio_tokenizer.config.frame_rate
        def _est(text, ref_text, n, speed=1.0):
            tg = re.findall(r"\[(laughter|breathing|sniff|throatclearing)\]", text)
            return max(1, int(_orig(" ".join(re.sub(r"\[[a-z]+\]", " ", text).split()), ref_text, n, speed) + sum(_MED[t] for t in tg) * _fr))
        tts._estimate_target_tokens = _est
    omt = lambda r: " ".join(re.sub(r"\[([a-z]+)\]", lambda m: OM[m.group(1)], r["text"]).split())
    synth = lambda r: (tts.generate(text=omt(r), ref_audio=r["ref"], ref_text=r["ref_text"], generation_config=gc)[0], 24000)
    synth_n = lambda r, n: [(w, 24000) for w in tts.generate(text=[omt(r)] * n, ref_audio=[r["ref"]] * n, ref_text=[r["ref_text"]] * n, generation_config=gc)]
elif eng == "voxcpm":
    # VoxCPM2 (OpenBMB, Apache code+weights, 2B, 30 languages incl. Vietnamese, NVVSpeech Track-2 official baseline):
    # zero-shot "ultimate cloning" (prompt wav + transcript + reference wav). Native NV tags: [laughing], [sigh];
    # sniff / throat clearing have none (dropped). VOXCPM_LORA=<dir> loads a LoRA fine-tune.
    import os
    from voxcpm import VoxCPM
    tts = VoxCPM.from_pretrained(os.environ.get("VOXCPM_BASE","openbmb/VoxCPM2"), load_denoiser=False, lora_weights_path=os.environ.get("VOXCPM_LORA") or None)
    VX = {"laughter": "[laughing]", "breathing": "[sigh]", "sniff": "", "throatclearing": ""}
    if os.environ.get("VOXCPM_LORA"): VX = {"laughter": "[laughing]", "breathing": "[breathing]", "sniff": "[sniff]", "throatclearing": "[throatclearing]"}   # = vox_prep.py mapping
    vxt = lambda r: " ".join(re.sub(r"\[([a-z]+)\]", lambda m: VX[m.group(1)], r["text"]).split())
    VM = os.environ.get("VOXCPM_MODE", "ult")   # ult: prompt+transcript+reference | ref: reference only | cont: prompt+transcript only
    def synth(r):
        kw = dict(cfg_value=float(os.environ.get("VOXCPM_CFG", 2.0)), inference_timesteps=int(os.environ.get("VOXCPM_STEPS", 10)))
        if VM in ("ult", "cont"): kw.update(prompt_wav_path=r["ref"], prompt_text=r["ref_text"])
        if VM in ("ult", "ref"): kw.update(reference_wav_path=r["ref"])
        return tts.generate(text=vxt(r), **kw), tts.tts_model.sample_rate
else:
    from f5_tts.api import F5TTS
    m = "/mnt/vinvtts/models/F5-TTS-Vietnamese-ViVoice"
    ft = (sys.argv[4] if len(sys.argv) > 4 else None) or None   # fine-tuned dir from run_f5.sh: <ckpt>.pt + vocab.txt, tags as PUA chars
    if ft:
        TAG2CP = {f"[{t}]": chr(0xE000 + i) for i, t in enumerate(["laughter", "breathing", "sniff", "throatclearing"])}
        prep = lambda t: re.sub(r"\[[a-z]+\]", lambda x: TAG2CP.get(x.group(0), ""), t)
        tts = F5TTS(model="F5TTS_Base", ckpt_file=sys.argv[5] if len(sys.argv) > 5 else f"{ft}/model_last.pt", vocab_file=f"{ft}/vocab.txt")
    else:
        prep = strip
        tts = F5TTS(model="F5TTS_Base", ckpt_file=f"{m}/model_last.pt", vocab_file=f"{m}/config.json")
    synth = lambda r: tts.infer(ref_file=r["ref"], ref_text=r["ref_text"], gen_text=prep(r["text"]), remove_silence=False)[:2]

NBEST = int(__import__("os").environ.get("NBEST", "1"))   # >1: N sampled candidates per row for bestof.py
t0 = time.time(); res = []
for r in rows:
    q = {**r, "text": strip(r["text"])} if NOTAG else r
    outs = synth_n(q, NBEST) if NBEST > 1 and "synth_n" in globals() and __import__("os").environ.get("NBSEQ") != "1" else [synth(q) for _ in range(NBEST)]
    for k, (wav, sr) in enumerate(outs):
        i = r["id"] if NBEST == 1 else f"{r['id']}__{k}"
        p = out / f"{i}.wav"; sf.write(p, wav, sr)
        res.append({**r, "id": i, "base_id": r["id"], "gen": str(p)})
with open(f"{out}.jsonl", "w") as f: f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in res)
print(eng, len(res), "wavs", f"{(time.time()-t0)/len(res):.2f}s/wav")
