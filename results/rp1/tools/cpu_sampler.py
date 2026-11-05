"""Host-state calibration: fixed-work timings every 100 ms, summarised per 10 s.
usage: cpu_sampler.py <seconds> <outfile>"""
import statistics as st
import sys
import time

import torchvision
data = torchvision.io.read_file("/workspace/results/rp1-pi05-tools/droid-cache/ep0_cam0.png")
end = time.time() + float(sys.argv[1])
out = open(sys.argv[2], "a", buffering=1)
win = time.time(); dec, loop, wake = [], [], []
while time.time() < end:
    t = time.perf_counter(); torchvision.io.decode_png(data); dec.append(time.perf_counter() - t)
    t = time.perf_counter(); x = 0
    for i in range(20000):
        x += i
    loop.append(time.perf_counter() - t)
    t = time.perf_counter(); time.sleep(0.05); wake.append(time.perf_counter() - t - 0.05)
    if time.time() - win > 10:
        la = open("/proc/loadavg").read().split()[0]
        print(f"{time.strftime('%H:%M:%S')} decode_ms {st.median(dec)*1e3:.2f} py20k_ms {st.median(loop)*1e3:.3f} wake_late_us {st.median(wake)*1e6:.0f} load {la}", file=out)
        win = time.time(); dec, loop, wake = [], [], []
