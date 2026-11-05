"""Captured-graph GPU time of the pi05 ViT submodule forward for one request (3 cameras),
using the checkout on sys.path. usage: python vit_graph_bench.py [bs]"""
import sys

import torch

sys.path.insert(0, ".")
from mstar.model.pi05.pi05_model import Pi05Model  # noqa: E402
from mstar.model.submodule_base import ModelInputsFromEngine  # noqa: E402

bs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
m = Pi05Model("lerobot/pi05_base")
sub = m.get_submodule("vit_encoder", device="cuda").to("cuda").eval()
print({str(p.dtype) for p in sub.parameters()})
x = (torch.rand(bs, 3, 3, 224, 224, device="cuda") * 2 - 1)
ei = ModelInputsFromEngine(request_ids=list(range(bs)), per_request_info={})
fwd = torch.compile(sub.forward_batched, mode="max-autotune-no-cudagraphs")
with torch.no_grad():
    for _ in range(3):
        fwd("prefill", ei, x)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fwd("prefill", ei, x)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        out = fwd("prefill", ei, x)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(50):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    print(f"bs={bs} vit graph replay ms {a.elapsed_time(b) / 50:.3f} out dtype {out[0]['img_emb'][0].dtype}")

if len(sys.argv) > 2 and sys.argv[2] == "profile":
    from torch.profiler import ProfilerActivity, profile
    with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=25, max_name_column_width=90))
