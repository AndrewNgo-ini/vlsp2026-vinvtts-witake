"""Judge workers for the production pipeline (each runs in its own venv; serve.py calls them over HTTP).

  --kind qasr : tag-ASR v3 (Qwen3-ASR SFT), qwen3asr venv.   POST /run {"wavs": [b64 wav]} -> {"hyps": [str]}
  --kind psed : PretrainedSED BEATs/ATST-F/M2D, psed venv.   POST /run {"wavs": [b64 wav], "spans": [[[t0,t1],..],..]}
                -> {"probs": [[ [model][4 class max probs] per span ] per wav]}   (= psed_score.py, window +-.1 s)

    /data/vinvtts/src/qwen3asr/.venv/bin/python judge_worker.py --kind qasr --port 8012
    /data/vinvtts/src/psed/.venv/bin/python judge_worker.py --kind psed --port 8013
"""
import argparse, base64, io, os, tempfile
import numpy as np, soundfile as sf, torch
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
QASR = "/data/vinvtts/runs/qasr/exp/checkpoint-270"


class Req(BaseModel):
    wavs: list[str]
    spans: list[list[list[float]]] | None = None


def to_files(wavs, d):
    ps = []
    for i, w in enumerate(wavs):
        p = f"{d}/{i}.wav"; open(p, "wb").write(base64.b64decode(w)); ps.append(p)
    return ps


@app.post("/run")
def run(r: Req):
    with tempfile.TemporaryDirectory() as d, torch.inference_mode():
        ps = to_files(r.wavs, d)
        if app.state.kind == "qasr":
            m, hyps = app.state.m, []
            bs = int(os.environ.get("QASR_BS", 16))
            for i in range(0, len(ps), bs):
                res = m.transcribe(audio=ps[i:i + bs], language=["Vietnamese"] * len(ps[i:i + bs]))
                hyps += [" ".join(x.text.split()) for x in res]
            return {"hyps": hyps}
        from psed_eval import frame_probs, FPS
        out = [[[None] * len(app.state.ms) for _ in sp] for sp in r.spans]
        for k, m in enumerate(app.state.ms):
            for i, (p, sp) in enumerate(zip(ps, r.spans)):
                if not sp: continue
                P = frame_probs(m, p)
                for j, (t0, t1) in enumerate(sp):
                    a, b = max(0, int((t0 - .1) * FPS)), min(P.shape[1], int((t1 + .1) * FPS) + 1)
                    out[i][j][k] = [round(float(v), 4) for v in (P[:, a:b].max(1) if b > a else np.zeros(P.shape[0]))]
        return {"probs": out}


@app.get("/health")
def health(): return {"status": "ok", "kind": app.state.kind}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", choices=["qasr", "psed"], required=True)
    ap.add_argument("--port", type=int, default=8012); a = ap.parse_args(); app.state.kind = a.kind
    if a.kind == "qasr":
        from qwen_asr import Qwen3ASRModel
        app.state.m = Qwen3ASRModel.from_pretrained(QASR, dtype=torch.bfloat16, device_map="cuda:0")
    else:
        from psed_eval import load
        app.state.ms = [load(n).cuda().eval() for n in ["BEATs", "ATST-F", "M2D"]]
    import cpulimit; cpulimit.cap_gpu_memory()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=a.port, workers=1)
