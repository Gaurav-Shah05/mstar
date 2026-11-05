# Diagnostic only: RP1_HOPS_DIR=<dir> records ingest and output hops of a c=1 request,
# one "<name> <time.monotonic()>" per event, flushed by a daemon thread (no file I/O in hooks).
import os, sys, threading, time

d = os.environ["RP1_HOPS_DIR"]
path = os.path.join(d, f"hops_{os.getpid()}.txt")
buf = []
TARGETS = [
    ("mstar.api_server.data_worker", "PreprocessWorkerThread", "_finish_input", "finish"),
    ("mstar.api_server.data_worker", "PreprocessWorkerThread", "_store_persisted", "store"),
    ("mstar.conductor.conductor", "Conductor", "_ingest_request", "ingest"),
    ("mstar.communication.tensors", "SharedMemoryCommunicationManager", "start_read_tensors", "read"),
    ("mstar.model.pi05.pi05_model", "Pi05Model", "postprocess", "post"),
]

def wrap(cls, meth, name):
    orig = getattr(cls, meth)
    def w(*a, **k):
        buf.append((name + "_in", time.monotonic()))
        try:
            return orig(*a, **k)
        finally:
            buf.append((name + "_out", time.monotonic()))
    setattr(cls, meth, w)

def loop():
    done = set()
    while True:
        for mod, cname, meth, name in TARGETS:
            if name in done:
                continue
            m = sys.modules.get(mod)
            cls = getattr(m, cname, None) if m else None
            if cls is not None and hasattr(cls, meth):
                wrap(cls, meth, name)
                done.add(name)
        time.sleep(0.05 if len(done) < len(TARGETS) else 1.0)
        if buf:
            items, buf[:] = buf[:], []
            with open(path, "a") as f:
                f.writelines(f"{a} {b:.6f}\n" for a, b in items)

threading.Thread(target=loop, daemon=True).start()
