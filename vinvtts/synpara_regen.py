"""Regenerate the lost SynParaSpeech-v2 clips (backup/data/synpara/kept.jsonl) row by row: same host utterance, same
VocalSound event file, same insertion point and event length (from cands.spans.jsonl), Seed-VC re-run (stochastic, so
audio is not bit-identical). Loudness: the per-class RMS-ratio bank (runs/events) is lost -> fixed ratio RAT[tag].
    <seedvc_space venv> synpara_regen.py [K N]  -> /data/vinvtts/data/synpara_regen/wav + kept_regen_K.jsonl
"""
import json, os, sys
import numpy as np, soundfile as sf, librosa
B = "/data/vinvtts/backup/data/synpara"; O = "/data/vinvtts/data/synpara_regen"; SR = 24000
RAT = {"throatclearing": 0.6, "sniff": 0.35}
k, n = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
sys.path.insert(0, __import__("os").environ.get("VINV_CODE", __import__("os").path.dirname(__import__("os").path.abspath(__file__))))
from synpara import active, rms
spans = {s["id"]: s for s in map(json.loads, open(f"{B}/cands.spans.jsonl"))}
rows = [r for i, r in enumerate(map(json.loads, open(f"{B}/kept.jsonl"))) if i % n == k]
os.makedirs(f"{O}/wav", exist_ok=True); os.chdir("/data/vinvtts/src/seedvc_space"); sys.path.insert(0, ".")
from seed_vc_wrapper import SeedVCWrapper
vc = SeedVCWrapper(); out_rows = []
for r in rows:
    cid = os.path.basename(r["audio"])[:-4]; s = spans[cid]; tag = s["tag"]; L = s["t1"] - s["t0"]
    p = f"{O}/wav/{cid}.wav"
    if not os.path.exists(p):
        ev = librosa.load(r["event"].replace("/mnt/vinvtts/data/vocalsound/audio_16k", "/data/vinvtts/data/vocalsound/data_44k"), sr=16000)[0]
        ev = active(ev, 16000, cap=float(np.clip(L, 0.15, 1.2))); tmp = f"{O}/_src{k}.wav"; sf.write(tmp, ev, 16000)
        gen = vc.convert_voice(tmp, r["host"], diffusion_steps=25, inference_cfg_rate=0.7, f0_condition=False, stream_output=False)
        try:
            while True: next(gen)
        except StopIteration as e: out = e.value
        out = librosa.resample(np.asarray(out, np.float32), orig_sr=22050, target_sr=SR)[: int(L * SR)]
        y, sr = sf.read(r["host"], dtype="float32")
        if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        out = out * (RAT[tag] * rms(y) / rms(out))
        f = int(0.01 * SR); ramp = np.linspace(0, 1, f, dtype=np.float32); out[:f] *= ramp; out[-f:] *= ramp[::-1]
        cut = int(round((s["t0"] - 0.05) * SR)); z = np.zeros(int(0.05 * SR), np.float32)
        wav = np.concatenate([y[:cut], z, out, z, y[cut:]]); pk = float(np.abs(wav).max()); wav = wav * (0.95 / pk) if pk > 0.95 else wav
        sf.write(p, wav, SR)
    out_rows.append(dict(r, audio=p, dur=round(sf.info(p).duration, 3), regen_of=r["audio"]))
    print(cid, flush=True)
with open(f"{O}/kept_regen_{k}.jsonl", "w") as f: f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in out_rows)
print("REGEN_DONE", k, len(out_rows))
