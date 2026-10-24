"""Import first in every job. torch / onnxruntime otherwise start one thread per visible core, which is very slow under a
CPU quota. Caps torch and every onnxruntime.InferenceSession (DNSMOS, ...) at N threads (OMP_NUM_THREADS, at most 2)."""
import os

N = min(2, int(os.environ.get("OMP_NUM_THREADS", "2")))
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, str(N))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch
    torch.set_num_threads(N); torch.set_num_interop_threads(1)
except (ImportError, RuntimeError):
    pass

try:
    import onnxruntime as ort
    _IS = ort.InferenceSession

    class _Capped(_IS):
        def __init__(self, path, sess_options=None, *a, **kw):
            so = sess_options or ort.SessionOptions()
            if so.intra_op_num_threads in (0, None) or so.intra_op_num_threads > N: so.intra_op_num_threads = N
            so.inter_op_num_threads = 1
            super().__init__(path, so, *a, **kw)
    ort.InferenceSession = _Capped
except ImportError:
    pass


def cap_gpu_memory():
    """GPU_MEM_GB=<x>: cap this process's PyTorch allocator at x GB (several services share one 40 GB GPU in the
    submission container). Call after the models are loaded (initialises CUDA)."""
    gb = float(os.environ.get("GPU_MEM_GB") or 0)
    if gb > 0:
        import torch
        tot = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 2 ** 30 / tot), 0)
        print(f"GPU memory cap {gb} GB of {tot / 2 ** 30:.0f} GB", flush=True)
