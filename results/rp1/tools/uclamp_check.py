"""util_min of every thread under the two paired servers' process trees."""
import collections, ctypes, os, struct, subprocess, sys
libc = ctypes.CDLL(None, use_errno=True)
def um(tid):
    b = ctypes.create_string_buffer(56)
    return struct.unpack("IIQiIQQQII", b.raw)[8] if libc.syscall(315, tid, b, 56, 0) == 0 else None
for name, root in (("A", int(sys.argv[1])), ("B", int(sys.argv[2]))):
    kids = {root}; ps = [l.split() for l in subprocess.check_output(["ps","-eo","pid=,ppid="], text=True).splitlines()]
    for _ in range(4):
        kids |= {int(p) for p, q in ps if int(q) in kids}
    c = collections.Counter()
    for p in kids:
        try:
            for t in os.listdir(f"/proc/{p}/task"):
                c[um(int(t))] += 1
        except OSError: pass
    print(name, "procs", len(kids), "threads by util_min", dict(c))
