"""Robustness battery against a running serve.py --prod: every case must return 200 with non-silent audio
(or a clean 400 for invalid input). Writes /data/vinvtts/runs/public/robust/<case>.wav + report.json.

    uv run robust_test.py --url http://127.0.0.1:8020
"""
import argparse, base64, io, json, time, unicodedata
from pathlib import Path
import httpx, librosa, numpy as np, soundfile as sf

HF = Path("/data/vinvtts/data/vinv-tts-hf")


def enc(y, sr, fmt="WAV"):
    b = io.BytesIO(); sf.write(b, y, sr, format=fmt); return base64.b64encode(b.getvalue()).decode()


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mB = json.loads(open(HF / "public_test_B/metadata.jsonl").readline()); yB, srB = sf.read(HF / "public_test_B" / mB["file_name"], dtype="float32")
    yB = yB if yB.ndim == 1 else yB.mean(1)
    mA = json.loads(open(HF / "public_test_A/metadata.jsonl").readline())
    refA = base64.b64encode(open(HF / "public_test_A" / mA["file_name"], "rb").read()).decode()
    lines = [l.strip() for l in open(HF / "public_test_B/text.txt", encoding="utf-8") if l.strip()]
    longest = max(lines, key=lambda l: len(l.split()))
    B = lambda **kw: {"reference_wav_b64": enc(yB, srB), "reference_text": mB["text"], "track": "B", **kw}
    y16 = librosa.resample(yB, orig_sr=srB, target_sr=16000)
    cases = {
        "long_87w": B(text=longest),
        "very_long_2x": B(text=longest + " " + longest),
        "no_tag": B(text="Xin chào các bạn, hôm nay chúng ta sẽ nói về bóng đá Việt Nam."),
        "tags_in_row": B(text="Thật sự là [laughter] [laughter] [breathing] tôi không biết nói gì [sniff] [throatclearing] luôn."),
        "tag_only": B(text="[laughter]"),
        "tag_first_last": B(text="[throatclearing] Được rồi, chúng ta bắt đầu nhé [laughter]"),
        "empty_ref_text": B(text=lines[3], reference_text=""),
        "no_ref_text": {k: v for k, v in B(text=lines[3]).items() if k != "reference_text"},
        "short_prompt_2s": B(text=lines[4], reference_wav_b64=enc(yB[: 2 * srB], srB), reference_text=None),
        "long_prompt_30s": B(text=lines[4], reference_wav_b64=enc(np.tile(yB, 4)[: 30 * srB], srB)),
        "stereo_48k": B(text=lines[5], reference_wav_b64=enc(np.stack([yB, yB * .8], 1), srB)),
        "mono_16k": B(text=lines[5], reference_wav_b64=enc(y16, 16000)),
        "flac_prompt": B(text=lines[5], reference_wav_b64=enc(yB, srB, "FLAC")),
        "nfd_text": B(text=unicodedata.normalize("NFD", lines[6])),
        "upper_tag": B(text="Tôi nghĩ là [Laughter] không phải vậy."),
        "digits": B(text="Năm 2026 có 3 trận [breathing] đấu lớn."),
        "trackA_prompt": {"text": lines[7], "speaker_id": mA["speaker"], "reference_wav_b64": refA, "reference_text": mA["text"], "track": "A"},
        "trackA_id_only": {"text": lines[7], "speaker_id": mA["speaker"]},
        "bad_tag_400": B(text="Xin chào [cough] nhé"),
        "silent_prompt_400": B(text=lines[0], reference_wav_b64=enc(np.zeros(srB * 5, np.float32), srB)),
    }
    rep = {}
    with httpx.Client(base_url=a.url, timeout=3600) as c:
        import concurrent.futures as cf
        def run(item):
            k, body = item; t0 = time.time(); r = c.post("/synthesize", json=body); dt = time.time() - t0
            if r.status_code != 200: return k, {"status": r.status_code, "detail": r.text[:200], "s": round(dt, 1)}
            y, sr = sf.read(io.BytesIO(r.content), dtype="float32"); sf.write(out / f"{k}.wav", y, sr)
            rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
            info = json.loads(base64.b64decode(r.headers.get("X-Info", "e30=")))
            return k, {"status": 200, "dur": round(len(y) / sr, 2), "rms": round(rms, 4), "ok": len(y) > .3 * sr and rms > 1e-3,
                       "wer": info.get("wer"), "nv": info.get("nv"), "s": round(dt, 1)}
        with cf.ThreadPoolExecutor(8) as ex:
            for k, v in ex.map(run, cases.items()): rep[k] = v; print(k, v, flush=True)
    bad = [k for k, v in rep.items() if not (v.get("ok") or (k.endswith("_400") and v["status"] == 400))]
    json.dump(rep, open(out / "report.json", "w"), indent=1, ensure_ascii=False)
    print("ROBUST", "ALL OK" if not bad else f"FAILED {bad}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8020")
    ap.add_argument("--out", default="/data/vinvtts/runs/public/robust"); main(ap.parse_args())
