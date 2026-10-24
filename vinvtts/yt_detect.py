"""Stage B of the YouTube NV pipeline (GPU; no sherpa in this process). Reads Stage-A word files (yt_asr.py) and scores
EVERY inter-word gap >= MIN_GAP with three independent detectors:

  ast   AST (Gong 2021) embedding + AudioSet logits of the gap segment (nvdet.seg) -> classifiers in yt_label.py
        (nvdet v1 runs/nvdet/clf.pkl and a refit with all gaps as negatives).
  laugh LaughterSegmentation (Omine et al., Interspeech 2024; github.com/omine-me/LaughterSegmentation, weights
        hf omine-me/LaughterSegmentation, research use only): wav2vec2-large-xlsr-53 frame classifier, 7 s windows with
        2 s overlap, the repo's silence amplification + peak normalisation, frame sigmoid; gap score = max prob.
  tag   tag-ASR (tagasr.py checkpoint) as a FORCED scorer, not a free decoder (free decoding only ever emits breathing on
        external audio, results row 35): teacher-force the untagged Zipformer transcript of the chunk and read
        margin_k = log P(" " [tag_k] | prefix) - log P(next word | prefix) at each gap.

    uv run yt_detect.py --words /mnt/vinvtts/crawl/dev_words --out /mnt/vinvtts/crawl/feats/dev
    uv run yt_detect.py --words /mnt/vinvtts/crawl/train_words --out /mnt/vinvtts/crawl/feats/train --only ast
Output per file: <out>/<key>.npz {t0, t1, i (index of the word after the gap), ast (float16 N x 1295), laugh, tag (N x 4)}.
"""
import cpulimit  # noqa: F401
import argparse, json, math, os
from pathlib import Path
import numpy as np, torch
from judge import load16k, SR

MIN_GAP = 0.12   # s; align.MIN_EVT -- shorter gaps are never located as events in the training data
TAGS = ["laughter", "breathing", "sniff", "throatclearing"]


def gaps_of(d, edges):
    """(t0, t1, i): gap before word i (i = n -> after the last word). Lead/tail gaps only when `edges` (short clips)."""
    W = d["words"]; n = len(W); out = []
    for i in range(0 if edges else 1, n + 1 if edges else n):
        t0 = W[i - 1][2] if i > 0 else 0.0
        t1 = W[i][1] if i < n else d["dur"]
        if t1 - t0 >= MIN_GAP: out.append((t0, t1, i))
    return out


# ---------------------------------------------------------------- LaughterSegmentation (Omine et al. 2024)
class Laugh:
    W2V = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

    def __init__(self):
        import safetensors.torch
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2ForAudioFrameClassification
        self.m = Wav2Vec2ForAudioFrameClassification.from_pretrained(self.W2V, num_labels=1)
        sd = safetensors.torch.load_file(hf_hub_download("omine-me/LaughterSegmentation", "model.safetensors"))
        sd = {k.removeprefix("audio_model."): v for k, v in sd.items()}
        missing, unexpected = self.m.load_state_dict(sd, strict=False)
        assert not [k for k in missing if "masked_spec" not in k], missing
        self.m = self.m.cuda().half().eval()

    @staticmethod
    def amplify(y, mul=5.0):
        """= inference.custom_amplituder_small_portion: pydub detect_silence(270 ms, -35 dBFS) regions x5 with 150 ms
        ramps, then librosa.util.normalize. Silence search on a 10 ms grid instead of pydub's 1 ms seek step."""
        y = y.astype(np.float32).copy(); hop, win = 160, 27   # 10 ms hops, 270 ms windows
        n = len(y) // hop
        if n <= win: return y / max(1e-8, np.abs(y).max())
        e = np.concatenate([[0], np.cumsum((y[: n * hop].reshape(n, hop) ** 2).mean(1))])
        rms = np.sqrt(np.maximum((e[win:] - e[:-win]) / win, 1e-12)); sil = 20 * np.log10(rms) < -35
        segs, s = [], None
        for k, v in enumerate(sil):
            if v and s is None: s = k
            if not v and s is not None: segs.append((s, k - 1 + win)); s = None
        if s is not None: segs.append((s, len(sil) - 1 + win))
        f = int(SR * 0.15)
        for a, b in segs:
            a, b = a * hop, min(len(y), b * hop)
            if b - a > 2 * f:
                y[a:a + f] *= np.linspace(1, mul, f); y[a + f:b - f] *= mul; y[b - f:b] *= np.linspace(mul, 1, f)
            else: y[a:b] *= mul
        return y / max(1e-8, np.abs(y).max())

    @torch.no_grad()
    def __call__(self, y, bs=16, win=7.0, hop=5.0):
        """-> frame probs at 50 Hz (max over overlapping windows)."""
        y = self.amplify(y); L, H = int(win * SR), int(hop * SR)
        starts = list(range(0, max(1, len(y) - (L - H)), H)) or [0]
        nfr = int(math.ceil(len(y) / SR * 50)) + 1; P = np.zeros(nfr, np.float32)
        for b in range(0, len(starts), bs):
            x = np.stack([np.pad(y[s:s + L], (0, max(0, L - len(y[s:s + L])))) for s in starts[b:b + bs]])
            p = torch.sigmoid(self.m(torch.from_numpy(x).cuda().half()).logits.squeeze(-1).float()).cpu().numpy()
            for s, pr in zip(starts[b:b + bs], p):
                t = s / SR + np.arange(len(pr)) * (win / len(pr)); k = np.minimum((t * 50).astype(int), nfr - 1)
                np.maximum.at(P, k, pr)
        return P


def laugh_gap(P, t0, t1, pad=0.05):
    a, b = max(0, int((t0 - pad) * 50)), min(len(P), int(math.ceil((t1 + pad) * 50)) + 1)
    return float(P[a:b].max()) if b > a else 0.0


# ---------------------------------------------------------------- tag-ASR forced scoring
class TagScorer:
    def __init__(self, ckpt):
        import tagasr
        self.p, self.m = tagasr.load(ckpt); self.m.eval(); tk = self.p.tokenizer
        self.tag_ids = [tk.convert_tokens_to_ids(f"[{t}]") for t in TAGS]
        self.sp = tk(" ", add_special_tokens=False).input_ids[-1]            # 'Ġ' (220) precedes every tag token
        self.pre = tk("", add_special_tokens=True).input_ids[:-1]            # <|startoftranscript|> ... (no eot)
        self.eot = tk.eos_token_id; self.start = self.m.config.decoder_start_token_id

    @torch.no_grad()
    def __call__(self, y, words, gap_idx, bs=64):
        """y: chunk audio (<= 30 s); words: chunk words; gap_idx: gap positions i (before words[i]; len = after last).
        -> (len(gap_idx), 4) margins. Label layout = tokenizer(' ' + text) as in training; decoder input = [start] +
        labels[:-1] (HF shift_tokens_right), exactly what the model saw when trained with labels=."""
        tk = self.p.tokenizer
        ids = list(self.pre); pos = []
        for w in words:
            pos.append(len(ids)); ids += tk(" " + w, add_special_tokens=False).input_ids
        pos.append(len(ids)); ids.append(self.eot)
        f = self.p.feature_extractor(y, sampling_rate=16000, return_tensors="pt").input_features.cuda().to(torch.bfloat16)
        enc = self.m.model.encoder(f).last_hidden_state
        dec = torch.tensor([[self.start] + ids[:-1]]).cuda()
        lpA = torch.log_softmax(self.m.proj_out(self.m.model.decoder(input_ids=dec, encoder_hidden_states=enc).last_hidden_state).float(), -1)[0]
        out = np.zeros((len(gap_idx), 4), np.float32)
        for b in range(0, len(gap_idx), bs):
            g = gap_idx[b:b + bs]; seqs = [[self.start] + ids[:pos[i]] + [self.sp] for i in g]
            L = max(map(len, seqs)); x = torch.full((len(seqs), L), self.eot).cuda()
            for k, s in enumerate(seqs): x[k, :len(s)] = torch.tensor(s)
            h = self.m.model.decoder(input_ids=x, encoder_hidden_states=enc.expand(len(seqs), -1, -1)).last_hidden_state
            last = torch.tensor([len(s) - 1 for s in seqs]).cuda()
            lpB = torch.log_softmax(self.m.proj_out(h[torch.arange(len(seqs)).cuda(), last]).float(), -1)
            for k, i in enumerate(g):
                p = pos[i]   # decoder position p predicts ids[p] (= first token of word i, or eot)
                out[b + k] = (lpA[p, self.sp] + lpB[k, self.tag_ids] - lpA[p, ids[p]]).cpu().numpy()
        return out


def main(a):
    import nvdet
    only = set(a.only.split(","))
    emb = nvdet.Embedder() if "ast" in only else None
    lau = Laugh() if "laugh" in only else None
    tag = TagScorer(a.tagasr) if "tag" in only else None
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.words).glob("*.json"))
    for n, fp in enumerate(files, 1):
        o = out / (fp.stem + ".npz")
        if o.exists(): continue
        d = json.load(open(fp)); edges = d["dur"] <= 40; G = gaps_of(d, edges)
        if not d["words"]: np.savez(o, t0=np.zeros(0), t1=np.zeros(0), i=np.zeros(0, int)); continue
        y = load16k(d["audio"]).clamp(-1, 1)
        r = {"t0": np.array([g[0] for g in G], np.float32), "t1": np.array([g[1] for g in G], np.float32),
             "i": np.array([g[2] for g in G], np.int32)}
        if emb and G:
            r["ast"] = np.concatenate([emb([nvdet.seg(y, t0, t1) for t0, t1, _ in G[k:k + 512]]) for k in range(0, len(G), 512)]).astype(np.float16)
        if lau and G:
            P = lau(y.numpy()); r["laugh"] = np.array([laugh_gap(P, t0, t1) for t0, t1, _ in G], np.float32)
        if tag and G:
            T = np.full((len(G), 4), np.nan, np.float32); W = d["words"]
            for c0, c1 in d["chunks"]:
                wi = [k for k, w in enumerate(W) if c0 <= w[1] < c1]
                if not wi: continue
                lo, hi = wi[0], wi[-1]
                # gaps whose both neighbours lie in this chunk (+ the chunk's lead/tail gap for short clips)
                sel = [k for k, (_, _, i) in enumerate(G) if lo < i <= hi or (edges and (i == lo or i == hi + 1))]
                if not sel: continue
                cw = [W[k][0] for k in range(lo, hi + 1)]
                T[sel] = tag(y[int(c0 * SR):int(c1 * SR)].numpy(), cw, [G[k][2] - lo for k in sel])
            r["tag"] = T
        np.savez(o, **r)
        if n % 10 == 0 or n == len(files): print(f"{n}/{len(files)} {fp.stem} gaps {len(G)}", flush=True)
    print("DETECT_DONE", a.out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tagasr", default="/mnt/vinvtts/runs/tagasr"); ap.add_argument("--only", default="ast,laugh,tag")
    main(ap.parse_args())
