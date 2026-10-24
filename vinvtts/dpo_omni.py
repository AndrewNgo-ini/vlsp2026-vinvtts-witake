"""Masked-CE DPO for OmniVoice (masked non-autoregressive TTS).

  loss = -log sigmoid(beta * [(l_ref(y_w) - l_th(y_w)) - (l_ref(y_l) - l_th(y_l))]) + gamma * l_th(y_w)
  l(y) = mean over K stratified mask ratios of the codebook-weighted mean CE on the masked target tokens
         (the MGM-DPO likelihood surrogate of GenSR, arXiv 2508.17229; K-sample variance reduction of VRPO, 2505.19223;
         the SFT anchor gamma * l_th(y_w) of Anchored Flow-DPO, 2609.11260). Policy and reference share every mask.
Conditioning = exactly what inference feeds (OmniVoice._prepare_inference_inputs): style + ref_text/text + ref audio
tokens, target = the candidate's re-encoded tokens. Policy = LoRA on the LLM of SRC; reference = adapter disabled.

  pairs (our venv) : uv run dpo_omni.py pairs <cands.scored.jsonl> <cands.nvdet.jsonl> <pairs.jsonl>
  train (omni venv): $OPY dpo_omni.py train <pairs.jsonl> <out_dir> --model <SRC merged dir>
"""
import argparse, json, math, random, shutil, sys, time
from collections import defaultdict
from pathlib import Path

MIN_GAP = 0.02


def pairs(a):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bestof import utility
    sc = [json.loads(l) for l in open(a.inp)]
    hits = defaultdict(int)
    for r in map(json.loads, open(a.nvdet)): hits[r["id"]] += r["pred"] == r["tag"]
    by = defaultdict(list)
    for c in sc:
        c["nvdet_hit"] = hits[c["id"]]; c["u"] = utility(c); by[c["base_id"]].append(c)
    out = []
    for bid, cs in by.items():
        cs.sort(key=lambda c: c["u"])
        if cs[-1]["u"] - cs[0]["u"] >= MIN_GAP:
            w, l = cs[-1], cs[0]
            out.append({"id": bid, "text": w["text"], "ref": w["ref"], "ref_text": w["ref_text"], "chosen": w["gen"], "rejected": l["gen"],
                        "gap": w["u"] - l["u"]})
    with open(a.out, "w") as f: f.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in out)
    print(f"{len(out)} pairs from {len(by)} texts, mean gap {sum(p['gap'] for p in out) / max(1, len(out)):.3f}")


def train(a):
    import torch, torch.nn.functional as F
    from omnivoice import OmniVoice
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(0); random.seed(0)
    m = OmniVoice.from_pretrained(a.model, device_map="cuda:0", dtype=torch.float32)
    for p in m.parameters(): p.requires_grad_(False)
    m.llm = get_peft_model(m.llm, LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.0, bias="none",
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    params = [p for p in m.parameters() if p.requires_grad]
    print("trainable", sum(p.numel() for p in params) / 1e6, "M", flush=True)
    W = torch.tensor(m.normalized_audio_codebook_weights, device="cuda")
    C, MASK = m.config.num_audio_codebook, m.config.audio_mask_id

    P = [json.loads(l) for l in open(a.inp)]
    enc = {}
    def tokens(path):   # candidate / reference wav -> audio tokens (C, T), via OmniVoice's own prompt encoder
        if path not in enc:
            with torch.no_grad():
                enc[path] = m.create_voice_clone_prompt(path, ref_text="x", preprocess_prompt=False).ref_audio_tokens
        return enc[path]
    data = []
    for p in P:
        with torch.no_grad():
            ref = m.create_voice_clone_prompt(p["ref"], ref_text=p["ref_text"])
        data.append((p, ref, tokens(p["chosen"]), tokens(p["rejected"])))
    print("encoded", len(data), "pairs", flush=True)

    def build(p, ref, tgt):
        x = m._prepare_inference_inputs(text=p["text"], num_target_tokens=tgt.shape[-1], ref_text=ref.ref_text,
                                        ref_audio_tokens=ref.ref_audio_tokens, lang=None)
        ids = x["input_ids"].clone(); ids[0, :, -tgt.shape[-1]:] = tgt.to(ids.device)
        return ids, x["audio_mask"]

    def masked_ce(ids, amask, T, masks):
        """mean over masks of codebook-weighted masked CE on the last T (target) tokens -> (K,) -> mean."""
        K = len(masks)
        inp = ids.repeat(K, 1, 1); inp[:, :, -T:] = torch.where(masks, torch.full_like(inp[:, :, -T:], MASK), inp[:, :, -T:])
        logits = m(input_ids=inp, audio_mask=amask.repeat(K, 1)).logits[:, :, -T:]   # (K, C, T, V)
        ce = F.cross_entropy(logits.permute(0, 3, 1, 2).float(), ids[:, :, -T:].repeat(K, 1, 1), reduction="none")   # (K, C, T)
        per_layer = (ce * masks).sum(-1) / masks.sum(-1).clamp(min=1)   # (K, C)
        return (per_layer * W).sum(-1).mean()

    def draw_masks(T):
        r = [(k + random.random()) / a.K for k in range(a.K)]   # stratified mask ratios in (0, 1)
        return torch.stack([torch.rand(C, T, device="cuda") < max(ri, 0.05) for ri in r])

    ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    ref_l = []
    with torch.no_grad(), m.llm.disable_adapter(), ac():   # reference losses with fixed masks (stored for reuse)
        for p, ref, yw, yl in data:
            mw, ml = draw_masks(yw.shape[-1]), draw_masks(yl.shape[-1])
            iw, aw = build(p, ref, yw); il, al = build(p, ref, yl)
            ref_l.append((mw, ml, masked_ce(iw, aw, yw.shape[-1], mw).item(), masked_ce(il, al, yl.shape[-1], ml).item()))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = a.epochs * len(data); t0 = time.time(); acc_m = acc_a = 0.0
    for step in range(steps):
        i = random.randrange(len(data)); p, ref, yw, yl = data[i]; mw, ml, rw, rl = ref_l[i]
        iw, aw = build(p, ref, yw); il, al = build(p, ref, yl)
        with ac():
            tw = masked_ce(iw, aw, yw.shape[-1], mw); tl = masked_ce(il, al, yl.shape[-1], ml)
        margin = a.beta * ((rw - tw) - (rl - tl))
        loss = (-F.logsigmoid(margin) + a.gamma * tw) / a.accum
        loss.backward(); acc_m += margin.item(); acc_a += float(margin.item() > 0)
        if (step + 1) % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
        if (step + 1) % 50 == 0:
            print(f"step {step+1}/{steps} margin {acc_m/50:.4f} acc {acc_a/50:.2f} anchor {tw.item():.3f} {time.time()-t0:.0f}s", flush=True)
            acc_m = acc_a = 0.0
        if a.save_every and (step + 1) % a.save_every == 0 and step + 1 < steps:
            m.llm.save_pretrained(str(Path(a.out) / f"adapter-{step+1}")); print("saved adapter", step + 1, flush=True)
    m.llm.save_pretrained(str(Path(a.out) / f"adapter-{steps}"))
    m.llm = m.llm.merge_and_unload()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(str(out))
    for f in Path(a.model).iterdir():   # tokenizers / audio tokenizer / chat template from the source dir
        if not (out / f.name).exists():
            (shutil.copytree if f.is_dir() else shutil.copy)(f, out / f.name)
    print("saved", out)


def merge(adapter, out, model):
    """merge <adapter dir> <out dir> --model SRC: LoRA adapter (from --save-every) -> merged OmniVoice dir."""
    import torch
    from omnivoice import OmniVoice
    from peft import PeftModel
    m = OmniVoice.from_pretrained(model, device_map="cuda:0", dtype=torch.float32)
    m.llm = PeftModel.from_pretrained(m.llm, adapter).merge_and_unload()
    o = Path(out); o.mkdir(parents=True, exist_ok=True); m.save_pretrained(str(o))
    for f in Path(model).iterdir():
        if not (o / f.name).exists(): (shutil.copytree if f.is_dir() else shutil.copy)(f, o / f.name)
    print("merged", o)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pairs", "train", "merge"]); ap.add_argument("inp"); ap.add_argument("x"); ap.add_argument("y", nargs="?")
    ap.add_argument("--model"); ap.add_argument("--beta", type=float, default=5.0); ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--r", type=int, default=16); ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--save-every", type=int, default=0); ap.add_argument("--accum", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "pairs": a.nvdet, a.out = a.x, a.y; pairs(a)
    elif a.cmd == "merge": merge(a.inp, a.x, a.model)
    else: a.out = a.x; train(a)
