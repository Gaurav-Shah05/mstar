"""Send the 16 fixed DROID cases with seed=i to a running M* pi05 server; save actions.

usage (cwd = an M* checkout, venv active): python gate_client.py <port> <out.npz>
"""
import base64
import json
import os
import sys

import numpy as np
import requests

sys.path.insert(0, os.getcwd())
from benchmark.dataset import DROIDDataset  # noqa: E402

port, out = int(sys.argv[1]), sys.argv[2]
N = 16
ds = DROIDDataset(local_file_dir="/workspace/results/rp1-pi05-tools/droid-cache", num_requests=N, task="pi05")
acts = []
for i, req in enumerate(ds.get_requests()):
    kw = dict(req.model_kwargs)
    kw["seed"] = i
    paths = [req.image_path, *req.extra_image_paths]
    files = [("files", (os.path.basename(p), open(p, "rb"), "image/png")) for p in paths]
    data = {"text": req.prompt, "input_modalities": "image,text", "output_modalities": "action",
            "model_kwargs": json.dumps(kw)}
    r = requests.post(f"http://127.0.0.1:{port}/generate", data=data, files=files, timeout=120)
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
    a = np.frombuffer(raw, dtype=np.float32).reshape(-1, 32)
    acts.append(a)
np.savez(out, actions=np.stack(acts))
print("saved", out, np.stack(acts).shape)
