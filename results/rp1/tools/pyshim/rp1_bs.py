# Diagnostic only: RP1_BS_DIR=<dir> logs "<monotonic> <node> <bs> <input tokens> <captured?>" per engine step, buffered.
import os, sys, threading, time
path = os.path.join(os.environ["RP1_BS_DIR"], f"bs_{os.getpid()}.txt")
buf = []
def loop():
    done = False
    while True:
        m = sys.modules.get("mstar.engine.engine")
        if m is not None and not done and hasattr(m, "Engine"):
            orig = m.Engine.exec
            def ex(self, batch, *a, _o=orig, **k):
                t0 = time.monotonic()
                try:
                    return _o(self, batch, *a, **k)
                finally:
                    toks = sum(i.input_seq_len or 0 for i in batch.inputs) if getattr(batch, "inputs", None) and hasattr(batch.inputs[0], "input_seq_len") else 0
                    lease = batch.step_context.slot_lease
                    buf.append(f"{t0:.6f} {batch.node_name} {len(batch.request_ids)} {toks} {lease is not None} {time.monotonic() - t0:.6f}\n")
            m.Engine.exec = ex
            done = True
        time.sleep(0.05 if not done else 0.5)
        if buf:
            items, buf[:] = buf[:], []
            with open(path, "a") as f:
                f.writelines(items)
threading.Thread(target=loop, daemon=True).start()
