"""The gate's 16 DROID cases sent concurrently, repeated, to exercise batched rows.
usage (cwd = an M* checkout, venv active): gate_client_c.py <port> <out_prefix> <concurrency> <repeats>
Writes <out_prefix>_<rep>.npz in gate.npz's layout, in case order whatever order they ran in."""
import base64
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

sys.path.insert(0, os.getcwd())
from benchmark.dataset import DROIDDataset  # noqa: E402

port, prefix, c, reps = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
reqs = list(DROIDDataset(local_file_dir="/workspace/results/rp1-pi05-tools/droid-cache", num_requests=16, task="pi05").get_requests())


def one(item):
    i, req = item
    kw = dict(req.model_kwargs)
    kw["seed"] = i
    paths = [req.image_path, *req.extra_image_paths]
    files = [("files", (os.path.basename(p), open(p, "rb").read(), "image/png")) for p in paths]
    data = {"text": req.prompt, "input_modalities": "image,text", "output_modalities": "action",
            "model_kwargs": json.dumps(kw)}
    r = requests.post(f"http://127.0.0.1:{port}/generate", data=data, files=files, timeout=300)
    r.raise_for_status()
    raw = None
    if "application/json" in r.headers.get("content-type", ""):
        raw = base64.b64decode(r.json()["outputs"]["action"][0]["data"])
    else:
        for line in r.iter_lines():
            if line:
                msg = json.loads(line)
                if msg.get("modality") == "action" and msg.get("data"):
                    raw = base64.b64decode(msg["data"])
                    break
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 32)


with ThreadPoolExecutor(c) as pool:
    for rep in range(reps):
        acts = list(pool.map(one, enumerate(reqs)))
        np.savez(f"{prefix}_{rep}.npz", actions=np.stack(acts))
print("saved", reps)
