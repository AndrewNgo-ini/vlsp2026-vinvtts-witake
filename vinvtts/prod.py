"""Production recipe (2026-09-30): best-of-N from omft_tdl4000, selected by the official track weights on
judge proxies, then NV splicing (decide = any).

  1. gen    : tts_worker.py --mode omni /gen_batch, N candidates per request, all requests of a batch in one call
  2. score  : Zipformer WER, ECAPA SS to the given prompt, DNSMOS OVRL, NV = mean over tags of
              .5 tag-ASR v3 hit + .5 class judge (laughter: LaughterSegmentation p >= .5; sniff/throat: PSED rare_judge;
              breathing: v3 only). Word gaps from nvdet.gen_gaps with gap 0 included (Sep-28 first_gap=1 bug fixed).
  3. select : A = .3 NV + .15 (1-WER) + .15 pMOS/5 + .1 SS ; B = .3 NV + .1 (1-WER) + .1 pMOS/5 + .2 SS   (official formula)
  4. splice : splice.Splicer on the winner, splicing every tag that v3 OR the class judge calls missing.

Extra engines: pass more worker URLs (--worker a,b); candidates are split across them (multi-engine ensemble).
"""
import cpulimit  # noqa: F401
import base64, io, json, os, pickle, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, soundfile as sf, torch, torchaudio, jiwer

W_A = dict(nv=.3, wer=.15, mos=.15, ss=.1)
W_B = dict(nv=.3, wer=.1, mos=.1, ss=.2)
RARE = ("sniff", "throatclearing")


def b64wav(y, sr):
    b = io.BytesIO(); sf.write(b, np.clip(np.asarray(y, np.float32), -1, 1), sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(b.getvalue()).decode()


def unb64(s):
    y, sr = sf.read(io.BytesIO(base64.b64decode(s)), dtype="float32")
    return (y.mean(1) if y.ndim > 1 else y), sr


class Production:
    def __init__(self, workers, qasr_url, psed_url, n=8, bank="/data/vinvtts/runs/splice/bank", splice_policy="missing", per_engine=False, select="nvdet_v3"):
        import httpx
        from bench import Bench
        self.b = Bench("zipformer")                   # sherpa first (before CUDA models)
        import dnsmos_gpu  # noqa: F401
        from speechmos import dnsmos
        from yt_detect import Laugh
        from splice import Splicer
        from throat_det import feats
        self.dnsmos, self.L, self.feats = dnsmos, Laugh(), feats
        self.rare = {c: pickle.load(open(f"/data/vinvtts/runs/rare_judge/{c}.pkl", "rb")) for c in RARE}
        self.sp = Splicer(bank, rec=self.b.rec)
        # workers: "url[@gsA/gsB[@twA/twB[@tracks]]],..." -> per-engine, per-track guidance scale / tag-CFG weight; optional
        # tracks ("A", "B" or "AB", default AB) restricts an engine to those tracks (per-track engines, 2026-10-01)
        self.gen, self.gs, self.tw, self.tracks = [], [], [], []
        for w in workers.split(","):
            u, *rest = w.split("@"); self.gen.append(httpx.Client(base_url=u, timeout=1800))
            P = lambda x: dict(zip("AB", map(float, x.split("/"))))
            self.gs.append(P(rest[0]) if len(rest) > 0 and rest[0] else {}); self.tw.append(P(rest[1]) if len(rest) > 1 and rest[1] else {})
            self.tracks.append(set(rest[2]) if len(rest) > 2 and rest[2] else {"A", "B"})
        self.qasr = httpx.Client(base_url=qasr_url, timeout=1800); self.psed = httpx.Client(base_url=psed_url, timeout=1800)
        self.n, self.policy = n, splice_policy
        self.per_engine = per_engine      # True: n candidates from EACH engine (final config: 8 + 8)
        self.select = select              # "nvdet_v3" (served) | "v3_cj" (v3 + LS/PSED)
        # NV realism / NV levelling switches (2026-10-01):
        #   NV_REAL_W=<lambda> + NV_REAL=<nvreal.json>: add lambda * realism (share of the candidate's NVs inside the real-speech
        #     90% region of nvreal.py's per-class Gaussian) to the selection utility
        #   NV_LEVEL=qmap + NV_LEVEL_SYS_{A,B}=<sys .loud.jsonl> + NV_LEVEL_GT=<gt .loud.jsonl>: quantile-map NV levels after splice
        import os
        self.real_w = float(os.environ.get("NV_REAL_W", "0") or 0); self.real = None; self.qmap = None
        # NV_REAL_EPS=<eps>: constrained selection instead of a weighted term -- among candidates whose official utility is
        # within eps of the best, take the most realistic (lexicographic: max realism s.t. u >= u* - eps)
        self.real_eps = float(os.environ.get("NV_REAL_EPS", "0") or 0)
        if self.real_w > 0 or self.real_eps > 0:
            from nvreal import Real
            self.real = Real(json.load(open(os.environ["NV_REAL"])))
        if os.environ.get("NV_LEVEL") == "qmap":
            from nvlevel import QMap
            self.qmap = {T: QMap(os.environ[f"NV_LEVEL_SYS_{T}"], os.environ["NV_LEVEL_GT"]) for T in "AB"}
            from nvlevel import parse as _p
            a = os.environ.get("NV_LEVEL_ALPHA", "1.0")      # e.g. throatclearing=0.6 (it7); others 1
            self.level_alpha = _p(a) if "=" in a else float(a)
        self.ex = ThreadPoolExecutor(4)

    # ---------- 1. generation ----------
    def _gen(self, reqs, tag_w=None):
        k = len(self.gen); per = [self.n] * k if self.per_engine else [max(1, self.n // k)] * k
        if not self.per_engine: per[0] += self.n - sum(per)
        def ask(j):
            mine = [i for i, r in enumerate(reqs) if r["track"] in self.tracks[j]]
            out = [[] for _ in reqs]
            if not mine: return out
            items = [{"text": reqs[i]["text"], "ref_wav_b64": b64wav(*reqs[i]["ref"]), "ref_text": reqs[i].get("ref_text"), "n": reqs[i].get("n") or per[j], "track": reqs[i]["track"],
                      **({"gs": self.gs[j][reqs[i]["track"]]} if reqs[i]["track"] in self.gs[j] else {}),
                      **({"tag_w": tag_w} if tag_w is not None else {"tag_w": self.tw[j][reqs[i]["track"]]} if reqs[i]["track"] in self.tw[j] else {})} for i in mine]
            res = self.gen[j].post("/gen_batch", json={"items": items}); res.raise_for_status()
            for i, o in zip(mine, res.json()["out"]): out[i] = [unb64(w)[0] for w in o["wavs"]]
            return out
        parts = list(self.ex.map(ask, range(k)))
        return [[w for p in parts for w in p[i]] for i in range(len(reqs))]   # all engines emit 24 kHz

    # ---------- 2. scoring ----------
    def _score(self, reqs, cands):
        from judge import normalize
        from nvdet import gen_gaps
        from tags import nv_hits, split
        flat = [(i, w) for i, cs in enumerate(cands) for w in cs]
        w16 = [torchaudio.functional.resample(torch.from_numpy(w), 24000, 16000).clamp(-1, 1) for _, w in flat]
        # tag-ASR v3 (remote, GPU) overlaps the local CPU/GPU work
        fq = self.ex.submit(lambda: self.qasr.post("/run", json={"wavs": [b64wav(w.numpy(), 16000) for w in w16]}).json()["hyps"])
        hyps = self.b.asr(w16)
        gaps = [gen_gaps(self.b.rec, w, reqs[i]["text"]) for (i, _), w in zip(flat, w16)]
        mos = [float(self.dnsmos.run(w.numpy(), sr=16000)["ovrl_mos"]) for w in w16]
        remb = {}
        for i, r in enumerate(reqs): remb[i] = self.b.emb(torchaudio.functional.resample(torch.from_numpy(r["ref"][0]).float(), r["ref"][1], 16000))
        ss = [float(torch.nn.functional.cosine_similarity(self.b.emb(w), remb[i], dim=0)) for (i, _), w in zip(flat, w16)]
        # class judges
        cj = [dict() for _ in flat]; ps_idx, ps_w, ps_sp = [], [], []
        for c, ((i, _), w, (gp, tags)) in enumerate(zip(flat, w16, gaps)):
            d = len(w) / 16000
            if any(t == "laughter" for _, t in tags):
                P = self.L(w.numpy()); fps = len(P) / d
                for g, t in tags:
                    if t != "laughter": continue
                    t0, t1 = gp.get(g, (d - 1.0, d)); a, b = max(0, int((t0 - .3) * fps)), min(len(P), int((t1 + .3) * fps) + 1)
                    cj[c][g] = (float(np.max(P[a:b])) if b > a else 0.0) >= .5
            sp = [(g, t, *gp.get(g, (max(0, d - 1), d))) for g, t in tags if t in RARE]
            if sp: ps_idx.append((c, sp)); ps_w.append(b64wav(w.numpy(), 16000)); ps_sp.append([[s[2], s[3]] for s in sp])
        if ps_w:
            pr = self.psed.post("/run", json={"wavs": ps_w, "spans": ps_sp}).json()["probs"]
            for (c, sp), P in zip(ps_idx, pr):
                for (g, t, t0, t1), p in zip(sp, P):
                    cj[c][g] = float(self.rare[t].predict_proba(self.feats(p, t1 - t0)[None])[0, 1]) >= .5
        th = fq.result(); out = []
        if self.select == "nvdet_v3":   # nvdet p(tag) per tag (Splicer's AST+LR head)
            ndp = [[p for _, p in self.sp.detect(w, gp, tags)] if tags else [] for w, (gp, tags) in zip(w16, gaps)]
        for c, ((i, _), h, (gp, tags)) in enumerate(zip(flat, hyps, gaps)):
            text = reqs[i]["text"]
            v3 = [ok for _, ok in nv_hits(text, th[c])]
            hits = []; miss = set()
            for q, ((g, t), ok) in enumerate(zip(tags, v3)):
                j = cj[c].get(g, ok) if t != "breathing" else ok
                hits.append(.5 * ok + .5 * (ndp[c][q] if self.select == "nvdet_v3" else j))   # selection
                # splice decision (any) keeps v3 OR class judge (LS / PSED)
                if not ok or not j: miss.add((g, t))
            real = 1.0
            if self.real is not None and tags:
                from nvloud import features
                fx = features(w16[c], gp, tags)
                real = float(np.mean([self.real.dist(x["tag"], x) <= self.real.m[x["tag"]][2] if x["tag"] in self.real.m else 1.0 for x in fx])) if fx else 1.0
            out.append({"req": i, "wer": min(1.0, jiwer.wer(normalize(text), normalize(h["text"]) or "-")) if normalize(text).strip() else 0.0, "pmos": mos[c], "ss": ss[c],
                        "nv": float(np.mean(hits)) if hits else 1.0, "miss": miss, "tag_hyp": th[c], "real": real})
        return out

    # ---------- full batch ----------
    def __call__(self, reqs):
        """Robust wrapper: a failing batch is retried per request, then with plain generation (tag_w=0);
        silent/empty candidates are dropped before selection (see _run)."""
        try: return self._run(reqs)
        except Exception as e:
            print("PROD batch failed, retrying per request:", repr(e)[:300], flush=True)
        out = []
        for r in reqs:
            try: out.append(self._run([r])[0])
            except Exception as e:
                print("PROD request failed, plain fallback:", repr(e)[:300], flush=True)
                out.append(self._run([r], tag_w=0.0)[0])
        return out

    def _run(self, reqs, tag_w=None):
        """reqs: [{"text" (normalised, with tags), "ref": (wav, sr), "ref_text" | None, "track": "A"|"B", "key"}]
        -> [(wav24k, info)]"""
        t0 = time.time(); cands = self._gen(reqs, tag_w)
        ok = lambda w: len(w) > 0.3 * 24000 and float(np.sqrt(np.mean(w ** 2))) > 1e-3
        for i in range(len(reqs)):
            good = [w for w in cands[i] if ok(w)]
            if not good:   # every candidate silent/empty -> one plain-generation retry for this request
                good = [w for w in self._gen([reqs[i]], 0.0)[0] if ok(w)]
                if not good: raise RuntimeError(f"no non-silent candidate for {reqs[i].get('key')!r}")
            cands[i] = good
        t1 = time.time()
        sc = self._score(reqs, cands); t2 = time.time()
        best, us = {}, []
        for c, s in enumerate(sc):
            W = W_B if reqs[s["req"]]["track"] == "B" else W_A
            u = W["nv"] * s["nv"] + W["wer"] * (1 - s["wer"]) + W["mos"] * s["pmos"] / 5 + W["ss"] * s["ss"] + self.real_w * s["real"]
            us.append(u)
            if s["req"] not in best or u > best[s["req"]][0]: best[s["req"]] = (u, c)
        if self.real_eps > 0:
            for i in list(best):
                ub = best[i][0]; ok = [c for c, s in enumerate(sc) if s["req"] == i and us[c] >= ub - self.real_eps]
                c = max(ok, key=lambda c: (sc[c]["real"], us[c])); best[i] = (us[c], c)
        flatw = [w for cs in cands for w in cs]; res = []
        dump = os.environ.get("CAND_DUMP")   # paired offline selector eval: save every candidate + its scores + the pick
        if dump:
            import hashlib
            off = 0
            for i, r in enumerate(reqs):
                h = hashlib.sha1((r["track"] + "|" + r["text"] + "|" + str(len(r["ref"][0]))).encode()).hexdigest()[:16]
                d = os.path.join(dump, r["track"], h); os.makedirs(d, exist_ok=True)
                mine = [c for c, s in enumerate(sc) if s["req"] == i]
                for j, c in enumerate(mine): sf.write(os.path.join(d, f"c{j:02d}.wav"), flatw[c], 24000)
                json.dump({"text": r["text"], "track": r["track"], "key": r.get("key", ""), "pick": mine.index(best[i][1]),
                           "util": [us[c] for c in mine],
                           "sc": [{**{k: v for k, v in sc[c].items() if k != "miss"}, "miss": sorted([list(m) for m in sc[c]["miss"]])} for c in mine]},
                          open(os.path.join(d, "meta.json"), "w"), ensure_ascii=False)
        if True:
            for i, r in enumerate(reqs):
                u, c = best[i]; s = sc[c]; y = flatw[c]
                refp = ref_path(r["ref"])
                z, info = self.sp(y, 24000, r["text"], refp, speaker=r.get("speaker") if r["track"] == "A" else None,
                                  policy=self.policy, key=r.get("key", ""), force=s["miss"])
                lv = None
                if self.qmap is not None:
                    from nvlevel import level
                    z, lv = level(np.asarray(z, np.float32), 24000, r["text"], self.b.rec, self.qmap[r["track"]], tol=0.5, max_cut=18.0, alpha=self.level_alpha)
                res.append((np.clip(z, -1, 1).astype(np.float32), {"util": round(u, 4), "cand": c - sum(len(x) for x in cands[:i]),
                            **{k: round(s[k], 4) for k in ("wer", "pmos", "ss", "nv", "real")}, "tag_hyp": s["tag_hyp"], "splice": info,
                            **({"nvlevel": [{"tag": x["tag"], "cut": x["cut"]} for x in lv]} if lv is not None else {})}))
        t3 = time.time()
        for _, inf in res: inf["t"] = {"gen": round(t1 - t0, 2), "score": round(t2 - t1, 2), "splice": round(t3 - t2, 2), "batch": len(reqs)}
        return res


def ref_path(ref, d=os.environ.get("PROD_REFS", "/dev/shm/prod_refs")):
    """Splicer.pick caches ECAPA by path: content-addressed file per distinct prompt."""
    import hashlib
    os.makedirs(d, exist_ok=True); y, sr = ref
    p = f"{d}/{hashlib.md5(np.asarray(y, np.float32).tobytes()).hexdigest()}_{sr}.wav"
    if not os.path.exists(p): sf.write(p, y, sr)
    return p
