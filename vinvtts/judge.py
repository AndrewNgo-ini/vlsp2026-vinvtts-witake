"""Offline judge for ViNV-TTS outputs: WER/CER (Whisper), speaker cosine (ECAPA), UTMOS, tag-count stub.

Manifest format (one per line):  id|text_with_tags|ref_wav(optional)
Generated wavs are looked up as <wav_dir>/<id>.wav.

    uv run judge.py --wav_dir out/ --manifest test.txt --whisper openai/whisper-large-v3-turbo --device mps
    uv run judge.py --smoke        # synthesizes 2 Vietnamese wavs with macOS `say -v Linh`, runs on CPU with whisper-small
"""
import argparse, json, os, re, subprocess, sys, tempfile, unicodedata
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "data" / "models"
os.environ.setdefault("HF_HOME", str(MODELS / "hf"))
os.environ.setdefault("TORCH_HOME", str(MODELS / "torch"))

import numpy as np, soundfile as sf, torch, torchaudio, jiwer  # noqa: E402

TAGS = ["laughter", "breathing", "sniff", "throatclearing"]
TAG_RE = re.compile(r"\[(" + "|".join(TAGS) + r")\]")
SR = 16000  # every judge model (Whisper, ECAPA, UTMOS) consumes 16 kHz


def load16k(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(1))
    return torchaudio.functional.resample(wav, sr, SR) if sr != SR else wav


_DIG = "không một hai ba bốn năm sáu bảy tám chín".split()


def _num3(n, full):
    """0..999 in Vietnamese words; full=True when a higher group precedes (-> 'không trăm', 'linh')."""
    h, t, u = n // 100, n // 10 % 10, n % 10
    w = [_DIG[h], "trăm"] if h or full else []
    if t == 0:
        if u: w += (["linh"] if h or full else []) + [_DIG[u]]
    else:
        w += ["mười"] if t == 1 else [_DIG[t], "mươi"]
        if u: w.append("mốt" if u == 1 and t > 1 else "lăm" if u == 5 else "tư" if u == 4 and t > 1 else _DIG[u])
    return w


def num2words(s):
    """ASR (Zipformer) writes numbers as digits ('37 000', '2'); VLSP transcripts spell them out."""
    s = s.replace(".", "").replace(",", "")
    if len(s) > 1 and s[0] == "0" or len(s) > 12:
        return " ".join(_DIG[int(c)] for c in s)
    n = int(s)
    if n == 0: return "không"
    w, full = [], False
    for scale, name in ((10**9, "tỷ"), (10**6, "triệu"), (1000, "nghìn"), (1, "")):
        g = n // scale % 1000 if scale < 10**9 else n // scale
        if g: w += _num3(g, full) + ([name] if name else []); full = True
    return " ".join(w)


def normalize(text):
    text = unicodedata.normalize("NFC", TAG_RE.sub(" ", text)).lower()
    text = re.sub(r"\d+(?:[ .,]\d{3})+(?!\d)|\d+", lambda m: " " + num2words(re.sub(r"\D", "", m.group())) + " ", text)
    text = re.sub(r"[^\w\s]", " ", text)  # strip punctuation; digits verbalised above, no abbrev expansion
    return " ".join(text.split())


def tag_counts(text):
    c = {t: 0 for t in TAGS}
    for t in TAG_RE.findall(text):
        c[t] += 1
    return c


class Judge:
    def __init__(self, whisper, device, spk=True, mos=True):
        from transformers import pipeline
        self.asr = pipeline("automatic-speech-recognition", model=whisper, device=device, chunk_length_s=30)
        self.spk = self.mos = None
        if spk:
            from speechbrain.inference.speaker import EncoderClassifier
            self.spk = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(MODELS / "spkrec-ecapa-voxceleb"), run_opts={"device": "cpu"})
        if mos:  # UTMOS22 strong via torch.hub (tarepan/SpeechMOS); PyPI `speechmos` is DNSMOS, not UTMOS
            self.mos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()

    def transcribe(self, wav):
        out = self.asr({"raw": wav.numpy(), "sampling_rate": SR},
                       generate_kwargs={"language": "vi", "task": "transcribe"})
        return out["text"]

    @torch.no_grad()
    def embed(self, wav):
        return self.spk.encode_batch(wav[None]).squeeze()

    @torch.no_grad()
    def score(self, wav_path, text, ref_path=None):
        wav = load16k(wav_path)
        hyp, ref = normalize(self.transcribe(wav)), normalize(text)
        r = {"id": Path(wav_path).stem, "dur_s": round(len(wav) / SR, 2), "ref_text": ref, "hyp_text": hyp,
             "wer": jiwer.wer(ref, hyp) if ref else None, "cer": jiwer.cer(ref, hyp) if ref else None,
             "expected_tags": tag_counts(text), "n_tags": sum(tag_counts(text).values())}
        if self.spk and ref_path:
            r["spk_sim"] = float(torch.nn.functional.cosine_similarity(
                self.embed(wav), self.embed(load16k(ref_path)), dim=0))
        if self.mos:
            r["utmos"] = float(self.mos(wav[None], SR)[0])
        # TODO(tag-placement): detect realised non-verbal events in `wav` and align to expected_tags.
        # Candidate detectors (both AudioSet-527, clip-level, 16 kHz; slide a 1 s window for timestamps):
        #  - AST  HF `MIT/ast-finetuned-audioset-10-10-0.4593`: labels Laughter(16)/Giggle/Chuckle, Breathing(41),
        #    Sniff(50), Throat clearing(48), Cough(47). pipeline("audio-classification", ...) works on CPU.
        #  - BEATs `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt` (microsoft/unilm/beats), same label set, stronger.
        # Plan: window -> class prob per frame -> peaks above thr -> event list; compare type+order+count to
        # expected_tags and position vs Whisper word timestamps (return_timestamps="word") for placement.
        # Frame-level laughter-only alternative: jrgillick/laughter-detection (old torch deps).
        r["detected_tags"] = None
        return r


def read_manifest(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = line.rstrip("\n").split("|")
            rows.append((p[0], p[1], p[2] if len(p) > 2 and p[2] else None))
    return rows


def run(args):
    j = Judge(args.whisper, args.device, spk=not args.no_spk, mos=not args.no_mos)
    results = [j.score(Path(args.wav_dir) / f"{i}.wav", text, ref) for i, text, ref in read_manifest(args.manifest)]
    keys = ["wer", "cer", "spk_sim", "utmos"]
    mean = {k: float(np.mean(v)) for k in keys if (v := [r[k] for r in results if r.get(k) is not None])}
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    print("MEAN", json.dumps(mean))
    if args.out:
        Path(args.out).write_text(json.dumps({"samples": results, "mean": mean}, ensure_ascii=False, indent=1))
    return results, mean


def smoke():
    # macOS `say` as the TTS; swap for any public vi/en wav on Linux
    d = Path(tempfile.mkdtemp(prefix="vinvtts_smoke_"))
    say = lambda voice, text, name: subprocess.run(  # noqa: E731
        ["say", "-v", voice, "-o", str(d / name), "--data-format=LEI16@24000", text], check=True)
    t1 = "Hôm nay trời đẹp quá [laughter] mình đi chơi nhé [breathing]"
    t2 = "Tôi vừa mới đi làm về [sniff] hơi mệt một chút [throatclearing] nhưng vẫn vui."
    say("Linh", TAG_RE.sub("", t1), "a.wav"); say("Linh", TAG_RE.sub("", t2), "b.wav")
    say("Linh", "Xin chào các bạn, rất vui được gặp mọi người.", "ref_same.wav")
    say("Samantha", "Hello everyone, nice to meet you all today.", "ref_other.wav")
    (d / "manifest.txt").write_text(f"a|{t1}|{d/'ref_same.wav'}\nb|{t2}|{d/'ref_other.wav'}\n", encoding="utf-8")
    results, mean = run(argparse.Namespace(wav_dir=d, manifest=d / "manifest.txt", whisper="openai/whisper-small",
                                           device="cpu", no_spk=False, no_mos=False, out=None))
    a, b = results
    assert a["n_tags"] == 2 and b["expected_tags"] == {"laughter": 0, "breathing": 0, "sniff": 1, "throatclearing": 1}
    assert a["spk_sim"] > b["spk_sim"], "same-speaker ref must score higher than other-speaker ref"
    assert 0 <= a["cer"] < 0.5, f"CER too high for clean TTS: {a['cer']}"
    assert 1 <= a["utmos"] <= 5
    print("SMOKE OK", d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav_dir"); ap.add_argument("--manifest"); ap.add_argument("--out")
    ap.add_argument("--whisper", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--no_spk", action="store_true"); ap.add_argument("--no_mos", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        smoke()
    elif a.wav_dir and a.manifest:
        run(a)
    else:
        ap.error("need --wav_dir and --manifest, or --smoke")
