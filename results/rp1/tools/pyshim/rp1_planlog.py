# Diagnostic only: RP1_PLANLOG_DIR=<dir> logs what KV and position plans held per step: node-independent,
# one line per plan call in worker processes, flushed by a daemon thread. Reads tensors, so it syncs.
import os, sys, threading, time

d = os.environ["RP1_PLANLOG_DIR"]
path = os.path.join(d, f"plan_{os.getpid()}.txt")
buf = []


def wrap_kv(cls):
    orig = cls.plan
    def plan(self, step, ctx):
        was = self._preplanned
        res = orig(self, step, ctx)
        rows = []
        for label, out in res.items():
            views = out.views if out.views is not None else []
            rows.append((label, [tuple(v)[:6] for v in views][:2]))
        buf.append(f"kv t={time.monotonic():.6f} walk={ctx.graph_walk} preplan={ctx.is_preplan} promoted={was} commit={step.commit} "
                   f"lease={(ctx.slot_lease.bucket.bs, ctx.slot_lease.bucket.num_tokens, ctx.slot_lease.slot) if ctx.slot_lease else None} rows={rows}\n")
        return res
    cls.plan = plan
    orig_c = cls.commit
    def commit(self, step, ctx):
        out = orig_c(self, step, ctx)
        buf.append(f"kv_commit t={time.monotonic():.6f} commit={step.commit} segs={[tuple(s)[:3] for s in step.segments][:2]}\n")
        return out
    cls.commit = commit


def wrap_pos(cls):
    orig = cls.plan
    def plan(self, step, ctx):
        was = self._preplanned
        res = orig(self, step, ctx)
        vals = {}
        for k, v in res.items():
            try:
                c = v.detach().cpu().flatten()
                vals[k] = (c[:3].tolist(), c[14:16].tolist(), int(c.numel()))
            except Exception as e:
                vals[k] = repr(e)
        buf.append(f"pos t={time.monotonic():.6f} preplan={ctx.is_preplan} promoted={was} ids={vals}\n")
        return res
    cls.plan = plan


def loop():
    done = set()
    while True:
        m = sys.modules.get("mstar.engine.resources.kv.manager")
        if m is not None and "kv" not in done and hasattr(m, "KVManager"):
            wrap_kv(m.KVManager); done.add("kv")
        m = sys.modules.get("mstar.engine.resources.position.manager")
        cls = next((getattr(m, n) for n in dir(m) if n.endswith("Manager") and hasattr(getattr(m, n), "_build_pos_ids")), None) if m else None
        if cls is not None and "pos" not in done:
            wrap_pos(cls); done.add("pos")
        time.sleep(0.05 if len(done) < 2 else 0.5)
        if buf:
            items, buf[:] = buf[:], []
            with open(path, "a") as f:
                f.writelines(items)


threading.Thread(target=loop, daemon=True).start()
