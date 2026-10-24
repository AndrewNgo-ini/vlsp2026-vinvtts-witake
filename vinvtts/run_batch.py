"""VLSP 2026 submission entrypoint (Docker): /input/input.csv -> /output/audios/<id>.wav + /output/output.csv.

Starts the v5 serving stack (launch.sh's services, one GPU), runs prod.Production on every row with the Track-B
(reference-prompt) recipe, and rewrites output.csv after every finished chunk, so rows done before a time-limit kill are
kept. Errors are per row (empty output_audio_path). Exit 0 unless nothing at all could be generated.
The per-row candidate count adapts to the remaining time budget (TIME_BUDGET s from container start, T0).

    python run_batch.py [--input /input] [--output /output]
"""
import cpulimit  # noqa: F401
import argparse, csv, os, re, subprocess, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np, soundfile as sf

ROOT = os.environ.get("ROOT", "/data/vinvtts")
T0 = float(os.environ.get("T0") or time.time())
BUDGET = float(os.environ.get("TIME_BUDGET", 3300))      # s from container start: target end of the last row
HARD_LIMIT = float(os.environ.get("HARD_LIMIT", 3600))   # organisers kill the container here
LOGS = os.environ.get("LOGS", "/tmp/logs")
TAGS = ("laughter", "breathing", "sniff", "throatclearing")
P_OMNI, P_OMNI2, P_QASR, P_PSED = 8011, 8014, 8012, 8013


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def start(name, cmd, env=None):
    f = open(f"{LOGS}/{name}.log", "ab")
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", **(env or {})})


def healthy(port):
    import httpx
    try: return httpx.get(f"http://127.0.0.1:{port}/health", timeout=5).status_code == 200
    except Exception: return False


def wait_up(procs, timeout=1200):
    """procs: {name: (Popen, port)} -> names that came up (a dead process is dropped, its log tail printed)."""
    up, t = set(), time.time()
    while time.time() - t < timeout and len(up) + sum(p.poll() is not None for n, (p, _) in procs.items() if n not in up) < len(procs):
        for n, (p, port) in procs.items():
            if n not in up and p.poll() is None and healthy(port): up.add(n); log(f"{n} up on :{port}")
        time.sleep(3)
    for n, (p, _) in procs.items():
        if n not in up:
            log(f"{n} FAILED to start (exit {p.poll()}); log tail:")
            try: print(open(f"{LOGS}/{n}.log", errors="replace").read()[-3000:], flush=True)
            except OSError: pass
    return up


def launch():
    """= launch.sh defaults (v5) on one GPU; the second engine is optional (dropped if it does not start)."""
    om = f"{ROOT}/src/omnivoice/.venv/bin/python"
    train = f"{ROOT}/data/vinv-tts/train.jsonl"
    common = ["--train", train, "--tag-w", "3", "--gs", "2.0", "--tail-s", "0", "--ref-sr", "0"]
    procs = {"omni": (start("omni", [om, "tts_worker.py", "--mode", "omni", "--model", f"{ROOT}/runs/omni_dpo/cdpo1b", *common,
                                      "--inpaint-k-a", os.environ.get("INPAINT_K_A", "2"), "--port", str(P_OMNI)], {"GPU_MEM_GB": os.environ.get("MEM_OMNI", "11")}), P_OMNI)}
    if os.environ.get("MODEL2", "1") != "":
        procs["omni2"] = (start("omni2", [om, "tts_worker.py", "--mode", "omni", "--model", f"{ROOT}/runs/omni_dpo/odpo_spdl", *common,
                                          "--port", str(P_OMNI2)], {"GPU_MEM_GB": os.environ.get("MEM_OMNI2", "11")}), P_OMNI2)
    procs["qasr"] = (start("qasr", [f"{ROOT}/src/qwen3asr/.venv/bin/python", "judge_worker.py", "--kind", "qasr", "--port", str(P_QASR)], {"GPU_MEM_GB": os.environ.get("MEM_QASR", "7.5"), "QASR_BS": os.environ.get("QASR_BS", "8")}), P_QASR)
    procs["psed"] = (start("psed", [f"{ROOT}/src/psed/.venv/bin/python", "judge_worker.py", "--kind", "psed", "--port", str(P_PSED)]), P_PSED)
    up = wait_up(procs)
    if not {"omni", "qasr", "psed"} <= up: return None, procs
    w = f"http://127.0.0.1:{P_OMNI}@2/6@3/5"                  # GS_A/GS_B @ TW_A/TW_B (launch.sh defaults)
    if "omni2" in up: w += f",http://127.0.0.1:{P_OMNI2}@7/7@3/3"
    from prod import Production
    prod = Production(w, f"http://127.0.0.1:{P_QASR}", f"http://127.0.0.1:{P_PSED}", n=int(os.environ.get("N", 8)),
                      per_engine=True, select="nvdet_v3")
    return prod, procs


def clean_text(t):
    """Lower-case known tags, drop unknown [..] tags (the API would 400 them), then serve.norm."""
    from serve import norm
    t = re.sub(r"\[\s*([^\]]*?)\s*\]", lambda m: f"[{m.group(1).lower()}]" if m.group(1).lower() in TAGS else " ", t)
    return norm(t)


def prepare(row, in_dir):
    """CSV row -> prod request (as serve.synthesize does for a Track-B request)."""
    from serve import norm
    text = clean_text(row["target_text"])
    if not text.strip(): raise ValueError("empty target_text")
    y, sr = sf.read(os.path.join(in_dir, row["ref_audio"]), dtype="float32", always_2d=True); y = y.mean(1)
    rt = (row.get("ref_text") or "").strip()
    rt = re.sub(r"\[[^\]]*\]", " ", rt)
    if len(y) > 20 * sr: y, rt = y[: 15 * sr], ""             # long prompt: first 15 s, transcript dropped -> model ASR
    if len(y) < 0.5 * sr: raise ValueError("reference audio shorter than 0.5 s")
    if float(np.abs(y).max()) < 1e-4: raise ValueError("reference audio is silent")
    return {"text": text, "ref": (y.astype(np.float32), sr), "ref_text": norm(rt) if rt.strip() else None, "track": "B",
            "speaker": None, "key": f"{row['id']}|{text}"}


class Out:
    def __init__(self, out_dir, fields, rows):
        self.dir, self.fields, self.rows = out_dir, fields, rows
        self.path = {}; self.lock = threading.Lock()
        os.makedirs(f"{out_dir}/audios", exist_ok=True); self.flush()

    def put(self, i, wav):
        rel = f"audios/{self.rows[i]['id']}.wav"
        sf.write(f"{self.dir}/{rel}", np.clip(np.asarray(wav, np.float32), -1, 1), 24000, subtype="PCM_16")
        with self.lock: self.path[i] = rel

    def flush(self):
        with self.lock:
            tmp = f"{self.dir}/.output.csv.tmp"
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(self.fields + ["output_audio_path"])
                for i, r in enumerate(self.rows): w.writerow([r[k] for k in self.fields] + [self.path.get(i, "")])
            os.replace(tmp, f"{self.dir}/output.csv")


def main(a):
    os.makedirs(LOGS, exist_ok=True)
    with open(f"{a.input}/input.csv", newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f); fields = list(rd.fieldnames); rows = [{k: (v if v is not None else "") for k, v in r.items()} for r in rd]
    log(f"{len(rows)} rows, columns {fields}")
    out = Out(a.output, fields, rows)
    prod, procs = launch()
    try:
        if prod is None:
            log("FATAL: serving stack did not start"); return 1
        log("pipeline ready")
        reqs, todo = {}, []
        for i, r in enumerate(rows):
            try: reqs[i] = prepare(r, a.input); todo.append(i)
            except Exception as e: log(f"row {r.get('id')}: invalid input, skipped: {e!r}")
        done = run(prod, reqs, todo, out, a)
        log(f"finished: {done}/{len(rows)} rows have audio")
        return 0 if done or not rows else 1
    finally:
        out.flush()
        for p, _ in procs.values(): p.terminate()
        for p, _ in procs.values():
            try: p.wait(10)
            except subprocess.TimeoutExpired: p.kill()


def run(prod, reqs, todo, out, a):
    """Chunks of a.chunk rows, a.lanes chunks in flight (one generates while another is scored).
    Cost model: wall s per (row x candidate) so far. n (candidates per engine) is the largest step whose projection for
    the rows left fits the budget; a chunk is only started if it is projected to end before the hard limit."""
    deadline, hard = T0 + BUDGET, T0 + HARD_LIMIT - 45
    steps = [prod.n] + [n for n in (4, 2, 1) if n < prod.n]; t_start = time.time(); work = [0]; done = [0]
    chunks = [todo[i:i + a.chunk] for i in range(0, len(todo), a.chunk)]

    def pick(left, size):
        if not work[0]: return steps[0]
        rate = (time.time() - t_start) / work[0]; now = time.time()
        for n in steps:
            if now + rate * n * left <= deadline and now + rate * n * size * a.lanes <= hard:
                if n != steps[0]: log(f"time budget: {n} candidates per engine (rate {rate:.2f} s/row/cand, {left} rows left)")
                return n
        return steps[-1] if now + rate * steps[-1] * size * a.lanes <= hard else None

    def one(ch, n):
        rs = [{**reqs[i], "n": n} for i in ch]
        try: res = prod(rs)
        except Exception as e:
            log(f"chunk {[out.rows[i]['id'] for i in ch]} failed: {e!r}"); traceback.print_exc(); res = []
            for i, r in zip(ch, rs):
                try: res.append(prod._run([r], tag_w=0.0)[0])
                except Exception as e2: log(f"row {out.rows[i]['id']} failed: {e2!r}"); res.append(None)
        for i, x in zip(ch, res):
            if x is None: continue
            try:
                out.put(i, x[0]); done[0] += 1
                inf = x[1]; log(f"row {out.rows[i]['id']}: ok util {inf.get('util')} wer {inf.get('wer')} ss {inf.get('ss')} nv {inf.get('nv')} t {inf.get('t')}")
            except Exception as e: log(f"row {out.rows[i]['id']}: write failed {e!r}")
        out.flush(); return len(ch) * n

    with ThreadPoolExecutor(a.lanes) as ex:
        fut, k, left = set(), 0, len(todo)
        while k < len(chunks) or fut:
            while k < len(chunks) and len(fut) < a.lanes:
                n = pick(left, len(chunks[k]))
                if n is None: log(f"time budget exhausted, {left} rows not generated"); k = len(chunks); break
                fut.add(ex.submit(one, chunks[k], n)); left -= len(chunks[k]); k += 1
            if not fut: break
            f = next(as_completed(fut)); fut.remove(f)
            try: work[0] += f.result()
            except Exception as e: log(f"chunk error {e!r}")
            log(f"progress {done[0]}/{len(todo)} rows")
    return done[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/input"); ap.add_argument("--output", default="/output")
    ap.add_argument("--chunk", type=int, default=int(os.environ.get("CHUNK", 8)))
    ap.add_argument("--lanes", type=int, default=int(os.environ.get("LANES", 2)))
    a = ap.parse_args()
    try: rc = main(a)
    except Exception:
        traceback.print_exc(); rc = 1
    sys.exit(rc)
