"""mstar-report rows for a main checkpoint from its .out RESULT lines (the round directories were overwritten).
usage: report_main_out.py <out file> <config label> <sha> [--dry]"""
import datetime, json, re, subprocess, sys
from zoneinfo import ZoneInfo
path, config, sha = sys.argv[1:4]; dry = "--dry" in sys.argv
rnd, st, summ = 0, "", {}
for line in open(path):
    m = re.match(r"=== round (\d+) (\S+)Z", line)
    if m:
        rnd = int(m[1]); t = datetime.datetime.fromisoformat(m[2]).replace(tzinfo=datetime.timezone.utc).astimezone(ZoneInfo("America/New_York"))
        st = t.strftime("%Y-%m-%dT%H:%M%z"); st = st[:-2] + ":" + st[-2:]; day = t.strftime("%m%d"); continue
    m = re.match(r"RESULT ([AB]) c(\d+) p50=([\d.]+) p99=([\d.]+) mean=([\d.]+) thr=([\d.]+) ok=(\d+) fail=(\d+)", line)
    if not m: continue
    side, c, p50, p99, thr, ok, fail = m[1], int(m[2]), float(m[3]), float(m[4]), float(m[6]), int(m[7]), int(m[8])
    summ.setdefault((c, side), []).append((p50, thr, p99))
    row = {"batch": f"h200_{day}", "family": "World models and robotics", "model": "pi0.5", "workload": "DROID", "gpus": 1,
           "engine": "M* main 9c6f1ad" if side == "A" else f"M* rp1-main {sha}", "role": "baseline" if side == "A" else "candidate",
           "config": config, "c": c, "round": rnd, "throughput": thr, "unit": "chunks/s", "latency": "Request", "p50_ms": p50, "p99_ms": p99,
           "requests": ok + fail, "failed": fail, "status": "counted", "started": st, "error": "", "commit": "9c6f1ad75" if side == "A" else sha, "hypothesis": "checkpoint on main"}
    if not dry:
        subprocess.run(["mstar-report", json.dumps(row)], check=True)
    else:
        print(row["batch"], row["started"], row["engine"], c, rnd, p50) if rnd == 1 and c == 1 else None
for (c, side), v in sorted(summ.items()):
    print(f"c={c} {'main' if side == 'A' else 'port'}: p50 {[a for a, _, _ in v]} p99 {[p for _, _, p in v]} thr {[b for _, b, _ in v]}")
