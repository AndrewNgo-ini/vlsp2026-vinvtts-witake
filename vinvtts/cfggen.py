"""Batched tag-CFG sampling for serving (nvfix.NVFix._decode, vectorised over n candidates of one request).
logp = c + g (c - u) + w_tag (c - c_notag); rows = [cond x n, uncond x n, notag-cond x n] in one forward per step.
Candidates differ through the SDK's position-temperature Gumbel noise (and class_temp if set)."""
import math, re
import torch

TAGRE = re.compile(r"\[(laughter|breathing|sniff|throatclearing)\]")


@torch.inference_mode()
def cfg_n(tts, gc, text, ref_audio, ref_text, n, tag_w=3.0, tail_s=0.0):
    from omnivoice.models.omnivoice import _get_time_steps, _gumbel_sample, _filter_top_k
    F = torch.nn.functional; M = tts.config.audio_mask_id; C = tts.config.num_audio_codebook; fr = tts.audio_tokenizer.config.frame_rate
    text = " ".join(text.split())
    notag = " ".join(TAGRE.sub(" ", text).split()) if TAGRE.search(text) else None
    task = tts._preprocess_all(text=text, ref_audio=ref_audio, ref_text=ref_text, preprocess_prompt=gc.preprocess_prompt)
    T = task.target_lens[0]
    if tail_s and TAGRE.search(text.split()[-1]): T += int(round(tail_s * fr))   # room after a sentence-final tag
    if T > gc.audio_chunk_threshold * fr or notag is None or not tag_w:   # long / untagged: plain SDK path
        return tts.generate(text=[text] * n, ref_audio=[ref_audio] * n, ref_text=[ref_text] * n, generation_config=gc)
    P = lambda t: tts._prepare_inference_inputs(t, T, task.ref_texts[0], task.ref_audio_tokens[0], task.langs[0], task.instructs[0], gc.denoise)["input_ids"]
    c0, c2 = P(text), P(notag); base = [c0, c0[..., -T:], c2]
    L = max(r.shape[-1] for r in base); R = 3; nra = task.ref_audio_tokens[0].shape[-1] if task.ref_audio_tokens[0] is not None else 0
    ids = torch.full((R * n, C, L), M, dtype=torch.long, device=tts.device)
    am = torch.zeros((R * n, L), dtype=torch.bool, device=tts.device); att = torch.zeros((R * n, 1, L, L), dtype=torch.bool, device=tts.device)
    lens = [r.shape[-1] for r in base]
    for i, r in enumerate(base):
        ln = lens[i]; na = T if i == 1 else T + nra; sl = slice(i * n, (i + 1) * n)
        ids[sl, :, :ln] = r[0]; am[sl, ln - na:ln] = True; att[sl, :, :ln, :ln] = True
        if ln < L: d = torch.arange(ln, L, device=tts.device); att[sl, :, d, d] = True
    tok = torch.full((n, C, T), M, dtype=torch.long, device=tts.device)
    ts = _get_time_steps(0.0, 1.0, gc.num_step, gc.t_shift).tolist(); masked = C * T; rem = masked; sched = []
    for s in range(gc.num_step):
        k = rem if s == gc.num_step - 1 else min(math.ceil(masked * (ts[s + 1] - ts[s])), rem); sched.append(int(k)); rem -= int(k)
    layer = torch.arange(C, device=tts.device).view(1, -1, 1)
    for s, k in enumerate(sched):
        if k <= 0: continue
        lg = tts(input_ids=ids, audio_mask=am, attention_mask=att).logits.to(torch.float32)
        c = F.log_softmax(lg[0:n, :, lens[0] - T:lens[0]], -1); u = F.log_softmax(lg[n:2 * n, :, :T], -1)
        x = c + gc.guidance_scale * (c - u) + tag_w * (c - F.log_softmax(lg[2 * n:, :, lens[2] - T:lens[2]], -1))
        lp = F.log_softmax(x, -1); lp[..., M] = -float("inf")
        pred = _gumbel_sample(_filter_top_k(lp, ratio=0.1), gc.class_temperature).argmax(-1) if gc.class_temperature > 0 else lp.argmax(-1)
        sc = lp.max(-1)[0] - layer * gc.layer_penalty_factor
        if gc.position_temperature > 0: sc = _gumbel_sample(sc, gc.position_temperature)
        sc.masked_fill_(tok != M, -float("inf"))
        idx = torch.topk(sc.flatten(1), k, dim=1).indices
        flat = tok.flatten(1).clone(); flat.scatter_(1, idx, pred.flatten(1).gather(1, idx)); tok = flat.view_as(tok)
        for i, ln in enumerate(lens): ids[i * n:(i + 1) * n, :, ln - T:ln] = tok
    wavs = tts.audio_tokenizer.decode(tok.to(tts.audio_tokenizer.device)).audio_values.cpu().numpy()   # (n, 1, N)
    return [tts._post_process_audio(w, ref_rms=task.ref_rms[0], gen_config=gc).squeeze(0) for w in wavs]
