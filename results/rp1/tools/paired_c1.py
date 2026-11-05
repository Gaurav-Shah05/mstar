"""Paired c=1 latency: alternate single DROID requests between two pi05 servers.

One request in flight at a time overall, so the servers never share the GPU, and
consecutive A/B requests see the same host conditions. Which side goes first
alternates per pair. Writes <out>/{A,B}/c1/results.json with the runner's keys.

usage (cwd = an M* checkout, venv active):
  python paired_c1.py <portA> <portB> <n_per_side> <outdir>
"""
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.getcwd())
from benchmark.dataset import DROIDDataset  # noqa: E402

port_a, port_b, n, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ds = DROIDDataset(local_file_dir="/workspace/results/rp1-pi05-tools/droid-cache", num_requests=n, task="pi05")
reqs = ds.get_requests()


def send(port: int, req) -> float:
    paths = [req.image_path, *req.extra_image_paths]
    files = [("files", (os.path.basename(p), open(p, "rb"), "image/png")) for p in paths]
    data = {"text": req.prompt, "input_modalities": "image,text", "output_modalities": "action",
            "model_kwargs": json.dumps(req.model_kwargs)}
    t0 = time.monotonic()
    r = requests.post(f"http://127.0.0.1:{port}/generate", data=data, files=files, timeout=120)
    got = False
    if "application/json" in r.headers.get("content-type", ""):
        got = bool(r.json().get("outputs", {}).get("action"))
    else:
        for line in r.iter_lines(chunk_size=1 << 20):
            if line and json.loads(line).get("modality") == "action":
                got = True
                break
    lat = time.monotonic() - t0
    for _, (_, fh, _) in files:
        fh.close()
    r.raise_for_status()
    if not got:
        raise RuntimeError("no action output")
    return lat


lat = {"A": [], "B": []}
fail = {"A": 0, "B": 0}
for i, req in enumerate(reqs):
    order = (("A", port_a), ("B", port_b)) if i % 2 == 0 else (("B", port_b), ("A", port_a))
    for side, port in order:
        try:
            lat[side].append(send(port, req))
        except Exception as exc:  # noqa: BLE001
            fail[side] += 1
            print(side, "FAILED", exc)


def pct(xs, p):
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (k - lo)) + s[hi] * (k - lo)


for side in "AB":
    ms = [x * 1000 for x in lat[side]]
    d = os.path.join(out, side, "c1")
    os.makedirs(d, exist_ok=True)
    res = {
        "completed": len(ms), "failed": fail[side],
        "jct_median_ms": sorted(ms)[len(ms) // 2], "jct_mean_ms": sum(ms) / len(ms),
        "jct_p99_ms": pct(ms, 99), "request_throughput": len(ms) / sum(lat[side]),
        "per_request": [{"request_id": i, "jct_ms": x} for i, x in enumerate(ms)],
    }
    json.dump(res, open(os.path.join(d, "results.json"), "w"))
    print(f"RESULT {side} c1 p50={res['jct_median_ms']:.2f} p99={res['jct_p99_ms']:.2f} "
          f"mean={res['jct_mean_ms']:.2f} thr={res['request_throughput']:.3f} ok={len(ms)} fail={fail[side]}")
