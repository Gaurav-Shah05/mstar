"""Microbenchmark of the pi05 SigLIP encoder forward (3 cameras), CUDA-event timed.
Run from an M* checkout under the GPU lock."""
import sys
import time

import torch

sys.path.insert(0, ".")
from mstar.model.pi05.pi05_model import Pi05Model  # noqa: E402

m = Pi05Model("lerobot/pi05_base")
sub = m.get_submodule("vit_encoder", device="cuda").to("cuda")
enc = sub.encoder.eval()
x = torch.rand(3, 3, 224, 224, device="cuda") * 2 - 1


def timeit(fn, n=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


with torch.no_grad():
    ref = enc(x)
    print("fp32 eager ms", timeit(lambda: enc(x)))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = enc(x)
    print("tf32 eager ms", timeit(lambda: enc(x)), "maxdiff", (out - ref).abs().max().item(), "refmax", ref.abs().max().item())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = enc(x)
        print("bf16 autocast eager ms", timeit(lambda: enc(x)), "maxdiff", (out.float() - ref).abs().max().item())
    torch.cuda.synchronize()
    sys.stdout.flush()
    try:
        c = torch.compile(enc, mode="max-autotune-no-cudagraphs")
        c(x)
        print("fp32 compiled ms", timeit(lambda: c(x)))
    except Exception as exc:  # noqa: BLE001
        print("compile failed:", type(exc).__name__, str(exc)[:300])
