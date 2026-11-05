#!/usr/bin/env bash
# long c=1 run against one server; per-run p50 + cgroup throttle counters, to line up with cpu_sampler.
L=/workspace/results/rp1-pi05-logs/regime_watch
for i in $(seq 1 ${RUNS:-60}); do
  python -m benchmark.runner --url http://127.0.0.1:$PORT --model pi05 --request-type vision_language_action --profiling-type closed_loop --max-concurrency 1 --num-requests 100 --num-warmup 1 --local-cache /workspace/results/rp1-pi05-tools/droid-cache --output-dir $L/run > $L/last.log 2>&1
  python - "$L/run/results.json" >> $L/runs.txt <<'PY'
import json,sys,time
d=json.load(open(sys.argv[1]))
s=dict(l.split() for l in open('/sys/fs/cgroup/cpu.stat'))
mhz=[float(l.split()[-1]) for l in open('/proc/cpuinfo') if 'MHz' in l]
print(time.strftime('%H:%M:%S'),'p50 %.1f'%d['jct_median_ms'],'ok',d['completed'],'throttled',s['nr_throttled'],'avgMHz %.0f'%(sum(mhz)/len(mhz)),'load',open('/proc/loadavg').read().split()[0])
PY
done
