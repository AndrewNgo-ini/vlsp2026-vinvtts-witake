"""NVPA proxy: an expected tag counts as realised when the tag-aware ASR hypothesis has the same tag type at the same
position. Position = word gap (number of words before the tag); ref gaps are mapped onto hyp gaps through the
word-level Levenshtein alignment, tolerance +-TOL words.

    uv run tags.py   # self-check
"""
import re
import jiwer

TAG = re.compile(r"\[(laughter|breathing|sniff|throatclearing)\]")
TOL = 1


def split(text):
    """-> (words, [(gap_index, tag)]) with tags removed from the words."""
    words, tags = [], []
    for tok in TAG.sub(lambda m: f" [{m.group(1)}] ", text.lower()).split():
        m = TAG.fullmatch(tok)
        if m: tags.append((len(words), m.group(1)))
        else:
            w = re.sub(r"[^\w]", "", tok)
            if w: words.append(w)
    return words, tags


def gap_map(rw, hw):
    """ref gap g (0..len(rw)) -> hyp gap."""
    if not rw or not hw:
        return lambda g: round(g * len(hw) / max(1, len(rw)))
    m = {0: 0}
    for c in jiwer.process_words(" ".join(rw), " ".join(hw)).alignments[0]:
        for k in range(c.ref_end_idx - c.ref_start_idx):
            m[c.ref_start_idx + k + 1] = min(c.hyp_start_idx + k + 1, c.hyp_end_idx) if c.type != "delete" else c.hyp_start_idx
    return lambda g: m.get(g, round(g * len(hw) / len(rw)))


def nv_utt(ref, hyp):
    rw, rt = split(ref); hw, ht = split(hyp); f = gap_map(rw, hw)
    free = list(ht); hit = hit_any = 0
    for g, t in rt:
        h = f(g)
        j = next((i for i, (hg, ht_) in enumerate(free) if ht_ == t and abs(hg - h) <= TOL), None)
        if j is not None: hit += 1; free.pop(j)
        hit_any += any(abs(hg - h) <= TOL for hg, _ in ht)
    return hit, hit_any, len(rt), len(ht)


def nv_hits(ref, hyp):
    """-> [(tag, hit)] per expected tag (same matching as nv_utt)."""
    rw, rt = split(ref); hw, ht = split(hyp); f = gap_map(rw, hw); free = list(ht); out = []
    for g, t in rt:
        h = f(g)
        j = next((i for i, (hg, ht_) in enumerate(free) if ht_ == t and abs(hg - h) <= TOL), None)
        if j is not None: free.pop(j)
        out.append((t, j is not None))
    return out


def nv_score(refs, hyps):
    h = a = n = p = 0
    for r, y in zip(refs, hyps):
        a_, b_, c_, d_ = nv_utt(r, y); h += a_; a += b_; n += c_; p += d_
    rec, prec = h / max(n, 1), h / max(p, 1)
    return {"nvpa": rec, "nv_prec": prec, "nv_f1": 2 * rec * prec / max(rec + prec, 1e-9), "nv_any": a / max(n, 1)}


if __name__ == "__main__":
    ref = "xin chào [laughter] các bạn hôm nay [breathing] trời đẹp"
    assert nv_utt(ref, ref) == (2, 2, 2, 2)
    assert nv_utt(ref, "xin chao [laughter] các bạn hôm nay trời đẹp [breathing]")[0] == 1   # breathing moved 2 words
    assert nv_utt(ref, "xin chào các [laughter] bạn hôm nay [sniff] trời đẹp")[:2] == (1, 2)  # wrong type counts in nv_any
    assert nv_utt(ref, "chào [laughter] các bạn nay [breathing] trời đẹp")[0] == 2            # deletions realign
    print("ok", nv_score([ref], [ref]))
