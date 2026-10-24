"""End-to-end public-test run through the production API (serve.py --prod): every text.txt line x every prompt speaker,
sent as raw requests (cased text, original prompt file at its own sample rate + its transcript), `--conc` in flight.
Writes runs/public/final/{A,B}/<spk>/<line:02d>.wav + {A,B}.jsonl (scorable by w1score.sh) + timing.json.

    uv run run_final.py --url http://127.0.0.1:8000 --out /data/vinvtts/runs/public/final
"""
import argparse, asyncio, base64, io, json, time
from pathlib import Path
import httpx, numpy as np, soundfile as sf

HF = Path("/data/vinvtts/data/vinv-tts-hf"); PUB = Path("/data/vinvtts/runs/public")


async def main(a):
    out = Path(a.out); lat = []; t00 = time.time()
    pub = {r["id"]: r for T in "AB" for r in map(json.loads, open(PUB / f"{T}.jsonl"))}
    jobs = []
    for T in a.tracks:
        d = HF / f"public_test_{T}"; lines = [l.strip() for l in open(d / "text.txt", encoding="utf-8") if l.strip()]
        for m in map(json.loads, open(d / "metadata.jsonl", encoding="utf-8")):
            ref = base64.b64encode(open(d / m["file_name"], "rb").read()).decode()   # original file (flac / 48k wav)
            for i, l in enumerate(lines[: a.limit or None]):
                if a.final_only and not l.rstrip(" .,").endswith("]"): continue
                body = {"text": l, "reference_wav_b64": ref, "reference_text": m["text"], "track": T}
                if T == "A": body["speaker_id"] = m["speaker"]
                jobs.append((T, m["speaker"], i, body))
    sem = asyncio.Semaphore(a.conc); rows = {T: [] for T in a.tracks}
    async with httpx.AsyncClient(base_url=a.url, timeout=7200) as c:
        async def one(T, spk, i, body):
            async with sem:
                t0 = time.time(); r = await c.post("/synthesize", json=body); r.raise_for_status(); dt = time.time() - t0
            y, sr = sf.read(io.BytesIO(r.content), dtype="float32")
            p = out / T / spk / f"{i:02d}.wav"; p.parent.mkdir(parents=True, exist_ok=True); sf.write(p, y, sr)
            info = json.loads(base64.b64decode(r.headers["X-Info"])); lat.append(dt)
            pid = f"{T}_{spk}_{i:02d}"
            rows[T].append({**pub[pid], "gen": str(p), "prod": info, "latency": round(dt, 2)})
            if len(lat) % 20 == 0: print(f"{len(lat)}/{len(jobs)} done, {time.time() - t00:.0f}s", flush=True)
        await asyncio.gather(*[one(*j) for j in jobs])
    for T in a.tracks:
        with open(out / f"{T}.jsonl", "w") as f: f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in sorted(rows[T], key=lambda x: x["id"]))
    tot = time.time() - t00
    tm = {"n": len(lat), "wall_s": round(tot, 1), "utt_per_min": round(60 * len(lat) / tot, 2), "lat_p50": round(float(np.median(lat)), 1),
          "lat_p90": round(float(np.percentile(lat, 90)), 1), "conc": a.conc}
    json.dump(tm, open(out / "timing.json", "w"), indent=1); print("TIMING", json.dumps(tm))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default="/data/vinvtts/runs/public/final"); ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--tracks", default="AB"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--final-only", action="store_true", help="only lines whose last token is a tag")
    asyncio.run(main(ap.parse_args()))
