"""Run speechmos DNSMOS on the GPU: the two ONNX nets (sig/bak/ovr + P.808) are converted to torch with onnx2torch and
called on CUDA through a drop-in `.run(None, {name: x})` shim, so speechmos' windowing / polyfit code is unchanged (the
librosa mel front-end stays on CPU, single-threaded).
Import once before `speechmos.dnsmos.run`; `DNSMOS_GPU=0` disables. Parity vs onnxruntime-CPU checked by `python dnsmos_gpu.py`.
"""
import os
import numpy as np, torch

_DEV = os.environ.get("DNSMOS_DEV", "cuda" if torch.cuda.is_available() else "cpu")
_ON = os.environ.get("DNSMOS_GPU", "1") == "1" and (_DEV == "cuda" or os.environ.get("DNSMOS_DEV") == "cpu")


class _TorchSess:
    def __init__(self, path):
        import onnx
        from onnx2torch import convert
        self.m = convert(onnx.load(path)).eval().to(_DEV)   # a ModelProto, not the path: no temp file next to the model

    @torch.no_grad()
    def run(self, _out, feeds):
        x = torch.from_numpy(next(iter(feeds.values()))).to(_DEV)
        y = self.m(x)
        return [y.float().cpu().numpy()]


if _ON:
    import speechmos.dnsmos as D
    _init = D.DNSMOS.__init__

    def _gpu_init(self, primary_model_path, p808_model_path):
        self.primary_model_path = primary_model_path
        self.onnx_sess = _TorchSess(primary_model_path); self.p808_onnx_sess = _TorchSess(p808_model_path)

    D.DNSMOS.__init__ = _gpu_init


if __name__ == "__main__":   # parity check on 30 evalF GT clips: GPU (torch) vs onnxruntime CPU
    import json, sys, time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from judge import load16k
    import speechmos.dnsmos as D
    rows = [json.loads(l) for l in open("/mnt/vinvtts/runs/evalF/gt.jsonl")][:30]
    ws = [load16k(r["gen"]).clamp(-1, 1).numpy() for r in rows]
    t = time.time(); g = [D.run(w, sr=16000)["ovrl_mos"] for w in ws]; tg = time.time() - t
    D.DNSMOS.__init__ = _init; D.dnsmos = None
    t = time.time(); c = [D.run(w, sr=16000)["ovrl_mos"] for w in ws]; tc = time.time() - t
    print(f"DNSMOS parity: max |gpu-cpu| {np.max(np.abs(np.array(g) - np.array(c))):.5f} mean gpu {np.mean(g):.4f} cpu {np.mean(c):.4f} | {tg:.1f}s vs {tc:.1f}s")
