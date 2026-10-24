"""Inference-time NV fixes for OmniVoice (masked-token NAR), usable from the server or as a CLI.

  1. tag-CFG: extra guidance term contrasting the tagged text with the same text without tags
        logp = c + g (c - u) + w_tag (c - c_notag)       (c/u/c_notag: log-softmax of cond / uncond / tagless-cond)
  2. generate -> verify -> inpaint: decode, Zipformer-align (align.py), classify every tag's gap with the nvdet
     AST+LR head (NOT the v3 tag-ASR judge, which stays held out); for each missed tag re-mask its gap +-PAD frames
     (plus INS inserted frames for room) and regenerate only those frames with every other audio token fixed,
     conditioned on the tagged text. Up to K rounds; keep the candidate with most verified tags (WER guard).

    <omni venv> nvfix.py <manifest.jsonl> <out_dir> <model_dir> [--tag-w 0] [--k 0]
    -> <out_dir>/<id>.wav + <out_dir>.jsonl (+ <out_dir>.nvfix.jsonl per-row log)
API: NVFix(model_dir).synth(text, ref_audio, ref_text, tag_w=.., k=..) -> (wav 24 kHz np.float32, info)
"""
import cpulimit  # noqa: F401
import argparse, json, math, os, re, sys, time
from pathlib import Path
import numpy as np

TAGS = ("laughter", "breathing", "sniff", "throatclearing")
TAGRE = re.compile(r"\[(laughter|breathing|sniff|throatclearing)\]")
CLS = ["none", "laughter", "breathing", "sniff", "throatclearing"]


class NVFix:
    def __init__(self, model_dir, nvdet="/data/vinvtts/runs/nvdet/clf_np.npz", device="cuda:0", verify=True):
        self.verify = verify
        if verify:   # sherpa-onnx must be created before torch CUDA init (nvdet.py note)
            from align import recognizer
            self.rec = recognizer()
        import torch
        from omnivoice import OmniVoice
        from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
        self.torch = torch
        self.tts = OmniVoice.from_pretrained(model_dir, device_map=device, dtype=torch.float16)
        import os   # OMNI_DENOISE / OMNI_PREPROMPT (default 1 = OmniVoice defaults): <|denoise|> token, prompt silence-trim
        self.gc = OmniVoiceGenerationConfig(guidance_scale=2.0, num_step=32, denoise=os.environ.get("OMNI_DENOISE", "1") == "1",
                                            preprocess_prompt=os.environ.get("OMNI_PREPROMPT", "1") == "1")
        self.fr = self.tts.audio_tokenizer.config.frame_rate
        self.M = self.tts.config.audio_mask_id; self.C = self.tts.config.num_audio_codebook
        if verify:
            from nvdet import Embedder
            self.emb = Embedder(); z = np.load(nvdet)
            self.lr = {k: z[k] for k in z.files}

    # ---------------- decoding with fixed tokens + tag-CFG ----------------
    def _decode(self, task, tokens, tag_w=0.0, notag=None, class_temp=0.0):
        torch, tts, gc = self.torch, self.tts, self.gc
        T = tokens.shape[-1]
        conds = [tts._prepare_inference_inputs(task.texts[0], T, task.ref_texts[0], task.ref_audio_tokens[0], task.langs[0], task.instructs[0], gc.denoise)["input_ids"]]
        if tag_w and notag is not None:
            conds.append(tts._prepare_inference_inputs(notag, T, task.ref_texts[0], task.ref_audio_tokens[0], task.langs[0], task.instructs[0], gc.denoise)["input_ids"])
        rows = [conds[0], conds[0][..., -T:]] + conds[1:]
        L = max(r.shape[-1] for r in rows); R = len(rows)
        ids = torch.full((R, self.C, L), self.M, dtype=torch.long, device=tts.device)
        am = torch.zeros((R, L), dtype=torch.bool, device=tts.device)
        att = torch.zeros((R, 1, L, L), dtype=torch.bool, device=tts.device)
        lens = []
        for i, r in enumerate(rows):
            n = r.shape[-1]; lens.append(n)
            ids[i, :, :n] = r[0]
            # audio part = ref tokens + target for cond rows; target only for the uncond row
            na = T if i == 1 else T + (task.ref_audio_tokens[0].shape[-1] if task.ref_audio_tokens[0] is not None else 0)
            am[i, n - na:n] = True
            att[i, :, :n, :n] = True
            if n < L: d = torch.arange(n, L, device=tts.device); att[i, :, d, d] = True
        tok = tokens.clone().to(tts.device)
        for i, n in enumerate(lens): ids[i, :, n - T:n] = tok
        masked = int((tok == self.M).sum())
        if masked == 0: return tok
        from omnivoice.models.omnivoice import _get_time_steps, _gumbel_sample, _filter_top_k
        ts = _get_time_steps(0.0, 1.0, gc.num_step, gc.t_shift).tolist(); rem = masked; sched = []
        for s in range(gc.num_step):
            k = rem if s == gc.num_step - 1 else min(math.ceil(masked * (ts[s + 1] - ts[s])), rem); sched.append(int(k)); rem -= int(k)
        layer = torch.arange(self.C, device=tts.device).view(1, -1, 1)
        F = torch.nn.functional
        for s in range(gc.num_step):
            k = sched[s]
            if k <= 0: continue
            with torch.no_grad():
                lg = tts(input_ids=ids, audio_mask=am, attention_mask=att).logits.to(torch.float32)
            c = F.log_softmax(lg[0:1, :, lens[0] - T:lens[0]], -1); u = F.log_softmax(lg[1:2, :, :T], -1)
            x = c + gc.guidance_scale * (c - u)
            if R == 3: x = x + tag_w * (c - F.log_softmax(lg[2:3, :, lens[2] - T:lens[2]], -1))
            lp = F.log_softmax(x, -1); lp[..., self.M] = -float("inf")
            pred = _gumbel_sample(_filter_top_k(lp, ratio=0.1), class_temp).argmax(-1) if class_temp > 0 else lp.argmax(-1)
            sc = lp.max(-1)[0] - layer * gc.layer_penalty_factor
            if gc.position_temperature > 0: sc = _gumbel_sample(sc, gc.position_temperature)
            cur = tok.unsqueeze(0)
            sc.masked_fill_(cur != self.M, -float("inf"))
            idx = torch.topk(sc.flatten(), k).indices
            flat = cur.flatten().clone(); flat[idx] = pred.flatten()[idx]; tok = flat.view_as(tok)
            for i, n in enumerate(lens): ids[i, :, n - T:n] = tok
        return tok

    def _decode_n(self, task, n, tag_w=0.0, notag=None):
        """n independent candidates of one text in one batch (k=0 path): rows = cond*n + uncond*n (+ notag*n)."""
        torch, tts, gc = self.torch, self.tts, self.gc
        T = task.target_lens[0]; rT = task.ref_audio_tokens[0].shape[-1] if task.ref_audio_tokens[0] is not None else 0
        c0 = tts._prepare_inference_inputs(task.texts[0], T, task.ref_texts[0], task.ref_audio_tokens[0], task.langs[0], task.instructs[0], gc.denoise)["input_ids"]
        kinds = [(c0, T + rT), (c0[..., -T:], T)]
        if tag_w and notag is not None:
            kinds.append((tts._prepare_inference_inputs(notag, T, task.ref_texts[0], task.ref_audio_tokens[0], task.langs[0], task.instructs[0], gc.denoise)["input_ids"], T + rT))
        L = max(r.shape[-1] for r, _ in kinds); K = len(kinds); R = K * n
        ids = torch.full((R, self.C, L), self.M, dtype=torch.long, device=tts.device)
        am = torch.zeros((R, L), dtype=torch.bool, device=tts.device); att = torch.zeros((R, 1, L, L), dtype=torch.bool, device=tts.device)
        lens = []
        for kk, (r, na) in enumerate(kinds):
            m = r.shape[-1]; lens.append(m)
            for j in range(n):
                i = kk * n + j; ids[i, :, :m] = r[0]; am[i, m - na:m] = True; att[i, :, :m, :m] = True
                if m < L: d = torch.arange(m, L, device=tts.device); att[i, :, d, d] = True
        tok = torch.full((n, self.C, T), self.M, dtype=torch.long, device=tts.device)
        from omnivoice.models.omnivoice import _get_time_steps, _gumbel_sample
        masked = T * self.C; ts = _get_time_steps(0.0, 1.0, gc.num_step, gc.t_shift).tolist(); rem = masked; sched = []
        for st in range(gc.num_step):
            k = rem if st == gc.num_step - 1 else min(math.ceil(masked * (ts[st + 1] - ts[st])), rem); sched.append(int(k)); rem -= int(k)
        layer = torch.arange(self.C, device=tts.device).view(1, -1, 1); F = torch.nn.functional
        for st in range(gc.num_step):
            k = sched[st]
            if k <= 0: continue
            lg = tts(input_ids=ids, audio_mask=am, attention_mask=att).logits.to(torch.float32)
            c = F.log_softmax(lg[0:n, :, lens[0] - T:lens[0]], -1); u = F.log_softmax(lg[n:2 * n, :, :T], -1)
            x = c + gc.guidance_scale * (c - u)
            if K == 3: x = x + tag_w * (c - F.log_softmax(lg[2 * n:3 * n, :, lens[2] - T:lens[2]], -1))
            lp = F.log_softmax(x, -1); lp[..., self.M] = -float("inf")
            pred = lp.argmax(-1); sc = lp.max(-1)[0] - layer * gc.layer_penalty_factor
            if gc.position_temperature > 0: sc = _gumbel_sample(sc, gc.position_temperature)
            sc.masked_fill_(tok != self.M, -float("inf"))
            idx = torch.topk(sc.view(n, -1), k, dim=1).indices
            ft = tok.view(n, -1); ft.scatter_(1, idx, pred.view(n, -1).gather(1, idx))
            for kk in range(K):
                ids[kk * n:(kk + 1) * n, :, lens[kk] - T:lens[kk]] = tok
        return tok

    def synth_n(self, text, ref_audio, ref_text, n, tag_w=0.0, k=0):
        """n candidates in one batched pass; k>0: then verify + inpaint only the candidates that miss a tag."""
        with self.torch.inference_mode():
            text = " ".join(text.split())
            task = self.tts._preprocess_all(text=text, ref_audio=ref_audio, ref_text=ref_text, preprocess_prompt=self.gc.preprocess_prompt)
            notag = " ".join(TAGRE.sub(" ", text).split()) if TAGRE.search(text) else None
            if task.target_lens[0] > self.gc.audio_chunk_threshold * self.fr:
                return [self.synth(text, ref_audio, ref_text, tag_w=tag_w)[0] for _ in range(n)]
            tok = self._decode_n(task, n, tag_w=tag_w, notag=notag)
            from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
            gcp = OmniVoiceGenerationConfig()
            outs = []
            for j in range(n):
                t, w = tok[j], self._wav(tok[j])
                if k and notag is not None and self.verify: t, w = self._fix(task, t, w, text, notag, tag_w, k)
                outs.append(self.tts._post_process_audio(w, ref_rms=task.ref_rms[0], gen_config=gcp).squeeze(0))
            return outs

    def _wav(self, tok):
        return self.tts.audio_tokenizer.decode(tok.unsqueeze(0).to(self.tts.audio_tokenizer.device)).audio_values[0].cpu().numpy()   # (1, N) 24 kHz

    # ---------------- verification (nvdet AST+LR on Zipformer gaps) ----------------
    def _check(self, wav24, text):
        import torch, torchaudio
        from nvdet import gen_gaps, seg, MIN_SEG, SR
        from tags import split
        w = torchaudio.functional.resample(torch.from_numpy(wav24[0]).float(), 24000, SR).clamp(-1, 1)
        gp, tags = gen_gaps(self.rec, w, text)
        segs = [seg(w, *gp[g]) if g in gp else w[-int(MIN_SEG * SR):] for g, _ in tags]
        if not segs: return [], gp, 0.0
        X = self.emb(segs); z = (X - self.lr["mean"]) / self.lr["scale"]; lo = z @ self.lr["coef"].T + self.lr["intercept"]
        P = np.exp(lo - lo.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        out = [{"gap": g, "tag": t, "hit": CLS[int(P[i].argmax())] == t, "p": float(P[i][CLS.index(t)]), "span": gp.get(g)} for i, (g, t) in enumerate(tags)]
        # alignment coverage as a cheap WER proxy: share of ref words recognised
        from align import words_with_times
        import jiwer
        rw = split(text)[0]; hw = [x[0] for x in words_with_times(self.rec, w)]
        wer = jiwer.wer(" ".join(rw), " ".join(hw) or "x") if rw else 0.0
        return out, gp, wer

    def _fix(self, task, tok, wav, text, notag, tag_w, k, pad=0.12, ins=0.3, info=None):
        """verify -> inpaint missed tags up to k rounds; returns the best (tok, wav)."""
        torch = self.torch; info = {} if info is None else info
        chk, gp, wer = self._check(wav, text); best = (sum(c["hit"] for c in chk), sum(c["p"] for c in chk), -wer)
        info["rounds"] = [{"hits": best[0], "n": len(chk), "wer": round(wer, 3)}]; cur_tok, cur_chk = tok, chk
        for r in range(k):
            miss = [c for c in cur_chk if not c["hit"]]
            if not miss: break
            # rebuild the token sequence: each missed gap -> [left | MASK * (gap + 2 pad + ins) | right]
            t = cur_tok.cpu(); pieces = []; last = 0
            for c in sorted(miss, key=lambda c: c["gap"]):
                t0, t1 = c["span"] if c["span"] else ((t.shape[-1] / self.fr) - 0.3, t.shape[-1] / self.fr)
                a = max(last, int((t0 - pad) * self.fr)); b = min(t.shape[-1], max(a, int(math.ceil((t1 + pad) * self.fr))))
                n_ins = int(ins * self.fr * (1 + 0.5 * r)) if (t1 - t0) < 0.35 else int(0.1 * self.fr)
                pieces += [t[:, last:a], torch.full((self.C, b - a + n_ins), self.M, dtype=torch.long)]; last = b
            pieces.append(t[:, last:]); t = torch.cat(pieces, -1)
            ntok = self._decode(task, t, tag_w=tag_w, notag=notag); nwav = self._wav(ntok)
            nchk, _, nwer = self._check(nwav, text)
            sc = (sum(c["hit"] for c in nchk), sum(c["p"] for c in nchk), -nwer)
            info["rounds"].append({"hits": sc[0], "n": len(nchk), "wer": round(nwer, 3)})
            if sc[0] > best[0] and nwer <= -best[2] + 0.1 or (sc[0] == best[0] and sc[1] > best[1] and nwer <= -best[2] + 0.02):
                best, tok, wav = sc, ntok, nwav
            cur_tok, cur_chk = ntok, nchk   # keep inpainting from the latest try (fresh context each round)
        info["final_hits"] = best[0]
        return tok, wav

    def fix_wav(self, wav24, text, ref_audio, ref_text, tag_w=3.0, k=2):
        """post-hoc fix of an already generated/selected 24 kHz wav (prod.py, after best-of-N selection, before splice):
        re-encode to audio tokens, verify with nvdet, inpaint missed tags (k rounds). -> (wav np.float32, info)."""
        torch = self.torch
        with torch.inference_mode():
            text = " ".join(text.split()); info = {}
            notag = " ".join(TAGRE.sub(" ", text).split()) if TAGRE.search(text) else None
            if notag is None: return wav24, info
            task = self.tts._preprocess_all(text=text, ref_audio=ref_audio, ref_text=ref_text, preprocess_prompt=self.gc.preprocess_prompt)
            x = np.asarray(wav24, dtype=np.float32).reshape(1, -1); hop = self.tts.audio_tokenizer.config.hop_length
            x = x[:, : x.shape[-1] - x.shape[-1] % hop]
            tok = self.tts.audio_tokenizer.encode(torch.from_numpy(x).to(self.tts.audio_tokenizer.device).unsqueeze(0)).audio_codes.squeeze(0).to(self.tts.device)
            wav = self._wav(tok)
            chk, _, _ = self._check(wav, text)
            if all(c["hit"] for c in chk): return wav24, {"rounds": [{"hits": len(chk), "n": len(chk)}]}   # nothing to fix: keep the original
            ntok, nwav = self._fix(task, tok, wav, text, notag, tag_w, k, info=info)
            if ntok is tok: return wav24, info   # no improvement -> original audio untouched
            return nwav[0].astype(np.float32), info

    def synth(self, *a, **kw):
        with self.torch.inference_mode(): return self._synth(*a, **kw)

    def _synth(self, text, ref_audio, ref_text, tag_w=0.0, k=0, pad=0.12, ins=0.3, postprocess=True):
        torch = self.torch
        text = " ".join(text.split())
        task = self.tts._preprocess_all(text=text, ref_audio=ref_audio, ref_text=ref_text, preprocess_prompt=self.gc.preprocess_prompt)
        T = task.target_lens[0]; info = {"T": T}
        notag = " ".join(TAGRE.sub(" ", text).split()) if TAGRE.search(text) else None
        if T > self.gc.audio_chunk_threshold * self.fr:   # long: SDK chunked path, no fixes
            w = self.tts.generate(text=text, ref_audio=ref_audio, ref_text=ref_text, generation_config=self.gc)[0]
            return w, {**info, "path": "chunked"}
        tok = self._decode(task, torch.full((self.C, T), self.M, dtype=torch.long), tag_w=tag_w, notag=notag)
        wav = self._wav(tok)
        if k and notag is not None and self.verify:
            tok, wav = self._fix(task, tok, wav, text, notag, tag_w, k, pad, ins, info)
        rms = task.ref_rms[0]
        from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
        gcp = OmniVoiceGenerationConfig(postprocess_output=postprocess)
        return self.tts._post_process_audio(wav, ref_rms=rms, gen_config=gcp).squeeze(0), info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("out"); ap.add_argument("model")
    ap.add_argument("--tag-w", type=float, default=0.0); ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--nbest", type=int, default=1)
    ap.add_argument("--fix-existing", action="store_true", help="manifest rows have 'gen' wavs: post-hoc fix_wav on each")
    ap.add_argument("--gs", type=float, default=2.0)
    ap.add_argument("--shard", default="0/1", help="i/n: this process handles rows i::n")
    a = ap.parse_args()
    import soundfile as sf
    si, sn = map(int, a.shard.split("/"))
    rows = [json.loads(l) for l in open(a.manifest)][si::sn]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    nv = NVFix(a.model, verify=a.k > 0)
    res, logs, t0 = [], [], time.time()
    nv.gc.guidance_scale = a.gs
    if a.fix_existing:
        for r in rows:
            p = out / f"{r['id']}.wav"; x, sr = sf.read(r["gen"], dtype="float32"); x = x if x.ndim == 1 else x.mean(1)
            assert sr == 24000, sr
            t1 = time.time(); w, info = nv.fix_wav(x, r["text"], r["ref"], r["ref_text"], tag_w=a.tag_w, k=a.k)
            sf.write(p, w, 24000); res.append({**r, "gen": str(p)}); logs.append({"id": r["id"], "sec": round(time.time() - t1, 2), **info})
        rows = []
    for r in rows:
        if a.nbest > 1:
            for j, w in enumerate(nv.synth_n(r["text"], r["ref"], r["ref_text"], a.nbest, tag_w=a.tag_w, k=a.k)):
                i = f"{r['id']}__{j}"; p = out / f"{i}.wav"; sf.write(p, w, 24000)
                res.append({**r, "id": i, "gen": str(p), "base_id": r["id"]}); logs.append({"id": i})
            continue
        for j in range(a.nbest):
            i = r["id"] if a.nbest == 1 else f"{r['id']}__{j}"; p = out / f"{i}.wav"
            w, info = nv.synth(r["text"], r["ref"], r["ref_text"], tag_w=a.tag_w, k=a.k)
            sf.write(p, w, 24000); res.append({**r, "id": i, "gen": str(p), "base_id": r["id"]}); logs.append({"id": i, **info})
    suf = "" if sn == 1 else f".s{si}"
    open(f"{a.out}{suf}.jsonl", "w").writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in res)
    open(f"{a.out}{suf}.nvfix.jsonl", "w").writelines(json.dumps(x) + "\n" for x in logs)
    print("nvfix", len(res), f"{(time.time() - t0) / max(1, len(res)):.2f}s/utt", flush=True)
