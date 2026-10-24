"""Best-of-N (ISCSLP-2026 NVVSpeech winner, arXiv 2609.14231; Llasa, arXiv 2502.04128): score every candidate from
`NBEST=N gen_zs.py` and keep the best per utterance under the objective part of the VLSP Score A weights:
    0.3 NVPA_utt (tag-ASR) + 0.1 (1 - WER) + 0.1 DNSMOS/5 + 0.1 SS
The tag-ASR used here is also an eval scorer, so report nvdet.py (independent acoustic detector) for the pick too.
Writes <cands>.scored.jsonl (all candidates, reusable as DPO pairs) and <out>.jsonl (one pick per utt).

    uv run bestof.py runs/zs/nb_vieneu.jsonl runs/zs/nb_vieneu_best --ckpt runs/tagasr
    uv run bestof.py pool runs/zsF/nb_vieneu.scored.jsonl runs/zsF/nb_ft1.scored.jsonl --out runs/zsF/mix_bo8
        # pick over the union of already-scored candidate sets (e.g. base + fine-tune engines), no re-scoring
"""
import cpulimit  # noqa: F401
import argparse, json
from collections import defaultdict
import numpy as np
from bench import Bench
from judge import load16k
from tags import nv_utt
import tagasr

W = {"nv": 0.3, "wer": 0.1, "dnsmos": 0.1, "ss": 0.1}      # Track A: objective part of Score A
W_B = {"nv": 0.25, "wer": 0.1, "dnsmos": 0.1, "ss": 0.2}   # Track B (zero-shot): 0.25 NVPA, 0.2 SS


def utility(c, w=W):
    """nv = tag-ASR hit rate, or the mean with nvdet when c["nvdet_hit"] is present (serve.py / --nvdet): the tag-ASR
    alone over-credits base VieNeu's breaths (evalF: selected tagASR .795 vs independent nvdet .665)."""
    nv = c["nv_hit"] / max(1, c["nv_n"])
    if "nv_p" in c and c["nv_n"]:   # class-aware F1 instead of recall: penalises spurious events (2609.14231 selects on F1;
        p = c["nv_hit"] / max(1, c["nv_p"])   # evalF: arbiter precision .702 -> .763, F1 .748 -> .779 at equal recall)
        nv = 2 * nv * p / max(nv + p, 1e-9)
    if "nvdet_hit" in c: nv = 0.5 * (nv + c["nvdet_hit"] / max(1, c["nv_n"]))
    return w["nv"] * nv + w["wer"] * (1 - c["wer"]) + w["dnsmos"] * c["pmos"] / 5 + w["ss"] * c["ss"]


def pool(files, out):
    by = defaultdict(list)
    for f in files:
        for c in map(json.loads, open(f)): c["src_set"] = f; by[c["base_id"]].append(c)
    pick = [max(v, key=lambda c: c["u"]) for v in by.values()]
    with open(out + ".jsonl", "w") as f:
        f.writelines(json.dumps({**c, "id": c["base_id"]}, ensure_ascii=False) + "\n" for c in pick)
    with open(out + ".scored.jsonl", "w") as f:   # bench-compatible, so table.py reads it without re-scoring
        f.writelines(json.dumps({**c, "id": c["base_id"]}, ensure_ascii=False) + "\n" for c in pick)
    with open(out + ".tags.jsonl", "w") as f:
        f.writelines(json.dumps({"id": c["base_id"], "text": c["text"], "tag_hyp": c["tag_hyp"]}, ensure_ascii=False) + "\n" for c in pick)
    from collections import Counter
    print("picked from", dict(Counter(c["src_set"].split("/")[-1] for c in pick)))


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "rebal":   # class-balanced utility: tag weight sqrt(f_max / f_tag) (sqrt rebalancing, 2609.14231)
        from collections import Counter
        from tags import nv_hits
        cands = [json.loads(l) for l in open(sys.argv[2])]
        f = Counter(t for c in cands if c["id"].endswith("__0") for t, _ in nv_hits(c["text"], c["text"]))
        wt = {t: (max(f.values()) / n) ** 0.5 for t, n in f.items()}; print("class weights", {t: round(v, 2) for t, v in wt.items()})
        for c in cands:
            h = nv_hits(c["text"], c["tag_hyp"]); den = sum(wt[t] for t, _ in h) or 1
            c["u"] = W["nv"] * sum(wt[t] for t, ok in h if ok) / den + W["wer"] * (1 - c["wer"]) + W["dnsmos"] * c["pmos"] / 5 + W["ss"] * c["ss"]
        with open(sys.argv[3], "w") as fh: fh.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in cands)
        raise SystemExit
    if sys.argv[1] == "lsrank":   # bestof.py lsrank <cands.scored.jsonl> <cands.laugh.jsonl> <out.jsonl> [--rebal]
        # laughter tags judged by LaughterSegmentation (p >= .5 at the tag gap; laugh_check.py) instead of the tag-ASR,
        # whose text prior credits VieNeu laughs that an acoustic detector does not hear (evalF: dpo6b2 LS .19 vs GT .75;
        # DPO rounds ranked by the tag-ASR lowered LS laughter ft6 .245 -> dpo6b2 .186). Other tags: tag-ASR as before.
        # --rebal: class weights sqrt(f_max/f) as `rebal` (2609.14231).
        from collections import Counter
        from tags import nv_hits
        cands = [json.loads(l) for l in open(sys.argv[2])]
        ls = defaultdict(list)
        for r in map(json.loads, open(sys.argv[3])): ls[r["id"]].append((r["gap"], r["p"]))
        f = Counter(t for c in cands if c["id"].endswith("__0") for t, _ in nv_hits(c["text"], c["text"]))
        wt = {t: (max(f.values()) / n) ** 0.5 if "--rebal" in sys.argv else 1.0 for t, n in f.items()}
        for c in cands:
            lp = iter(p for _, p in sorted(ls[c["id"]]))
            h = [(t, (next(lp, 0.0) >= .5) if t == "laughter" else ok) for t, ok in nv_hits(c["text"], c["tag_hyp"])]
            den = sum(wt[t] for t, _ in h) or 1
            c["u"] = W["nv"] * sum(wt[t] for t, ok in h if ok) / den + W["wer"] * (1 - c["wer"]) + W["dnsmos"] * c["pmos"] / 5 + W["ss"] * c["ss"]
        with open(sys.argv[4], "w") as fh: fh.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in cands)
        print("lsrank: class weights", {t: round(v, 2) for t, v in wt.items()}, len(cands), "cands")
        raise SystemExit
    if sys.argv[1] == "pool":
        ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("files", nargs="+"); ap.add_argument("--out", required=True)
        a = ap.parse_args(); pool(a.files, a.out); raise SystemExit
    ap = argparse.ArgumentParser(); ap.add_argument("cands"); ap.add_argument("out"); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--nvdet", help="<cands>.nvdet.jsonl from nvdet.py score: NV term = mean(tag-ASR, nvdet), as serve.py")
    ap.add_argument("--first", type=int, default=0, help="use only candidates __0..__K-1 (best-of-K from an N pool)")
    ap.add_argument("--trackB", action="store_true", help="Track-B utility weights (NV .25, SS .2)")
    a = ap.parse_args()
    sc_path = a.cands.replace(".jsonl", ".scored.jsonl")
    import os
    if os.path.exists(sc_path):   # candidates already scored (e.g. by an earlier N or --nvdet run): reuse
        sc = [json.loads(l) for l in open(sc_path)]
    else:
        rows = [json.loads(l) for l in open(a.cands)]
        sc = Bench("zipformer").score(rows)
        p, m = tagasr.load(a.ckpt); m.eval()
        hyps = tagasr.transcribe(p, m, [load16k(r["gen"]) for r in rows])
        for c, h in zip(sc, hyps):
            hit, _, n, npred = nv_utt(c["text"], h); c.update(tag_hyp=h, nv_hit=hit, nv_n=n, nv_p=npred); c["u"] = utility(c)
        with open(sc_path, "w") as f: f.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in sc)   # with u (dpo_vieneu reads it)
    if a.nvdet:
        hits = defaultdict(int)
        for r in map(json.loads, open(a.nvdet)): hits[r["id"]] += r["pred"] == r["tag"]
        for c in sc: c["nvdet_hit"] = hits[c["id"]]
    if a.first: sc = [c for c in sc if int(c["id"].rsplit("__", 1)[1]) < a.first]
    for c in sc:
        if "nv_p" not in c: c["nv_p"] = nv_utt(c["text"], c["tag_hyp"])[3]   # older .scored files
        c["u"] = utility(c, W_B if a.trackB else W)
    by = defaultdict(list)
    for c in sc: by[c["base_id"]].append(c)
    pick = [max(v, key=lambda c: c["u"]) for v in by.values()]
    first = [v[0] for v in by.values()]
    with open(a.out + ".jsonl", "w") as f:
        f.writelines(json.dumps({**c, "id": c["base_id"]}, ensure_ascii=False) + "\n" for c in pick)
    for name, s in [("first", first), ("best", pick)]:
        print(name, json.dumps({k: round(float(np.mean([c[k] for c in s])), 4) for k in ["u", "wer", "ss", "pmos"]}),
              "nvpa", round(sum(c["nv_hit"] for c in s) / max(1, sum(c["nv_n"] for c in s)), 4))
