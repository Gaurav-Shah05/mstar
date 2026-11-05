"""Emit one mstar-report row per arm and concurrency of a round.

usage: python report_round.py <round logdir> <batch> <status> <base engine> <cand engine>
                              <base commit> <cand commit> <config> <hypothesis> [--dry]
Arm dirs are <i>_B / <i>_C; round number = the arm's index among arms of its kind.
"""
import datetime
import json
import os
import subprocess
import sys
from zoneinfo import ZoneInfo

logdir, batch, status, beng, ceng, bcommit, ccommit, config, hyp = sys.argv[1:10]
dry = "--dry" in sys.argv
arms = sorted((d for d in os.listdir(logdir) if d[0].isdigit()), key=lambda d: int(d.split("_")[0]))
seen = {"B": 0, "C": 0}
for arm in arms:
    kind = arm.split("_")[1]
    seen[kind] += 1
    base = kind == "B"
    path = os.path.join(logdir, arm)
    started = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(path, "load.txt")), ZoneInfo("America/New_York"))
    for c in (1, 4, 8):
        rj = os.path.join(path, f"c{c}", "results.json")
        if not os.path.exists(rj):
            continue
        d = json.load(open(rj))
        row = {
            "batch": batch, "family": "World models and robotics", "model": "pi0.5", "workload": "DROID",
            "gpus": 1, "engine": beng if base else ceng, "role": "baseline" if base else "candidate",
            "config": config, "c": c, "round": seen[kind],
            "throughput": round(d["request_throughput"], 3), "unit": "chunks/s", "latency": "Request",
            "p50_ms": round(d["jct_median_ms"], 2), "p99_ms": round(d["jct_p99_ms"], 2),
            "requests": d["completed"] + d["failed"], "failed": d["failed"], "status": status,
            "started": started.strftime("%Y-%m-%dT%H:%M%z")[:-2] + ":" + started.strftime("%z")[-2:],
            "error": "", "commit": bcommit if base else ccommit, "hypothesis": hyp,
        }
        line = json.dumps(row)
        print(line)
        if not dry:
            subprocess.run(["mstar-report", line], check=True)
