"""mstar-report rows for a paired round dir (<dir>/r<k>/{A,B}/c{1,4,8}/results.json).
usage: report_paired.py <dir> <batch> <status> <engA> <engB> <commitA> <commitB> <config> <hypothesis> [--dry]"""
import datetime
import json
import os
import subprocess
import sys
from zoneinfo import ZoneInfo

d, batch, status, ea, eb, ca, cb, config, hyp = sys.argv[1:10]
dry = "--dry" in sys.argv
rounds = sorted((r for r in os.listdir(d) if r.startswith("r") and r[1:].isdigit()), key=lambda r: int(r[1:]))
summary = {}
for r in rounds:
    k = int(r[1:])
    started = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(d, r, "load.txt")), ZoneInfo("America/New_York"))
    st = started.strftime("%Y-%m-%dT%H:%M%z")
    st = st[:-2] + ":" + st[-2:]
    for side, eng, commit, role in (("A", ea, ca, "baseline"), ("B", eb, cb, "candidate")):
        for c in (1, 4, 8):
            p = os.path.join(d, r, side, f"c{c}", "results.json")
            if not os.path.exists(p):
                continue
            x = json.load(open(p))
            summary.setdefault((c, side), []).append((round(x["jct_median_ms"], 2), round(x["request_throughput"], 2)))
            row = {"batch": batch, "family": "World models and robotics", "model": "pi0.5", "workload": "DROID",
                   "gpus": 1, "engine": eng, "role": role, "config": config, "c": c, "round": k,
                   "throughput": round(x["request_throughput"], 3), "unit": "chunks/s", "latency": "Request",
                   "p50_ms": round(x["jct_median_ms"], 2), "p99_ms": round(x["jct_p99_ms"], 2),
                   "requests": x["completed"] + x["failed"], "failed": x["failed"], "status": status,
                   "started": st, "error": "", "commit": commit, "hypothesis": hyp}
            if not dry:
                subprocess.run(["mstar-report", json.dumps(row)], check=True)
for (c, side), v in sorted(summary.items()):
    print(f"c={c} {'base' if side == 'A' else 'cand'}: p50 {[a for a, _ in v]} thr {[b for _, b in v]}")
