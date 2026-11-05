# Harness-only shim: the shared venv's torchaudio 2.11.1 does not load against
# torch 2.12.1 (undefined symbol), and transformers imports it eagerly. Mark it
# absent so find_spec() returns None and imports raise ImportError.
import sys

sys.modules["torchaudio"] = None

# Optional kernel profiler: with RP1_PROF_DIR set, every process that has
# initialized CUDA watches for $RP1_PROF_DIR/go; on it, it records CUDA kernel
# activity for RP1_PROF_SECONDS and writes a per-kernel table + chrome trace.
import os

if os.environ.get("RP1_PROF_DIR"):
    import threading
    import time

    def _watch():
        d = os.environ["RP1_PROF_DIR"]
        secs = float(os.environ.get("RP1_PROF_SECONDS", "5"))
        go = os.path.join(d, "go")
        while True:
            time.sleep(0.5)
            if not os.path.exists(go):
                continue
            torch = sys.modules.get("torch")
            if torch is None or not torch.cuda.is_initialized():
                time.sleep(5)
                continue
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                time.sleep(secs)
            pid = os.getpid()
            prof.export_chrome_trace(os.path.join(d, f"trace_{pid}.json"))
            with open(os.path.join(d, f"kernels_{pid}.txt"), "w") as f:
                f.write(prof.key_averages().table(sort_by="self_device_time_total", row_limit=60))
            return

    threading.Thread(target=_watch, daemon=True).start()

# Optional per-graph GPU timing: with RP1_GRAPH_TIMING=<dir>, wrap
# torch.cuda.CUDAGraph.replay with CUDA events (recorded on the replay's
# stream; resolved on a background thread, no sync on the step path) and
# write per-graph count / mean / total GPU ms to <dir>/graphs_<pid>.txt.
if os.environ.get("RP1_GRAPH_TIMING"):
    import collections
    import threading
    import time

    def _graph_timer():
        outdir = os.environ["RP1_GRAPH_TIMING"]
        while "torch.cuda.graphs" not in sys.modules:
            time.sleep(0.2)
        graphs = sys.modules["torch.cuda.graphs"]
        torch = sys.modules["torch"]
        pending = collections.deque()
        stats = collections.defaultdict(lambda: [0, 0.0, []])
        orig = graphs.CUDAGraph.replay

        def replay(self):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            orig(self)
            e.record()
            pending.append((id(self), s, e, time.time()))

        graphs.CUDAGraph.replay = replay
        last = time.time()
        while True:
            time.sleep(0.2)
            while pending:
                gid, s, e, t = pending[0]
                if not e.query():
                    break
                pending.popleft()
                st = stats[gid]
                st[0] += 1
                ms = s.elapsed_time(e)
                st[1] += ms
                st[2].append(round(ms, 3))
            if time.time() - last > 3 and stats:
                last = time.time()
                with open(os.path.join(outdir, f"graphs_{os.getpid()}.txt"), "w") as f:
                    for gid, (n, tot, xs) in sorted(stats.items(), key=lambda kv: -kv[1][1]):
                        xs_sorted = sorted(xs)
                        f.write(f"graph {gid} n={n} total_ms={tot:.1f} mean_ms={tot / n:.3f} "
                                f"p50_ms={xs_sorted[len(xs_sorted) // 2]:.3f} last={xs[-5:]}\n")

    threading.Thread(target=_graph_timer, daemon=True).start()

# Diagnostic only: RP1_WAIT_CAP_MS=<n> caps the worker/conductor idle wait
# (communicator.wait_for_work / poll_for_messages timeouts) at n ms, to test
# whether a readiness change that sends no wakeup leaves a process asleep.
if os.environ.get("RP1_WAIT_CAP_MS"):
    import threading
    import time

    def _cap_waits():
        cap = int(os.environ["RP1_WAIT_CAP_MS"])
        patched = set()
        while len(patched) < 2:
            for modname, clsname in (("mstar.communication.rust_communicator", "RustZMQCommunicator"),
                                     ("mstar.communication.communicator", "ZMQCommunicator")):
                mod = sys.modules.get(modname)
                cls = getattr(mod, clsname, None) if mod else None
                if cls is None or clsname in patched:
                    continue
                for meth in ("wait_for_work", "poll_for_messages"):
                    orig = getattr(cls, meth, None)
                    if orig is None:
                        continue

                    def wrapped(self, timeout_ms=50, _orig=orig):
                        return _orig(self, min(timeout_ms, cap))

                    setattr(cls, meth, wrapped)
                patched.add(clsname)
            time.sleep(0.05)

    threading.Thread(target=_cap_waits, daemon=True).start()

# Diagnostic only: RP1_TRACE_DIR=<dir> wraps pi05 model hooks (conductor-side
# get_partition_forward_pass_args, worker-side prepare_inputs per node/walk) and
# records "<hook> <walk> <time.monotonic()>"; a daemon thread flushes the
# in-memory buffer to <dir>/trace_<pid>.txt, so the hooks do no file I/O.
if os.environ.get("RP1_TRACE_DIR"):
    import threading
    import time

    def _trace_hooks():
        d = os.environ["RP1_TRACE_DIR"]
        path = os.path.join(d, f"trace_{os.getpid()}.txt")
        buf = []
        done = set()
        while True:
            m = sys.modules.get("mstar.model.pi05.pi05_model")
            if m is not None and "model" not in done and hasattr(m, "Pi05Model"):
                orig = m.Pi05Model.get_partition_forward_pass_args

                def gpfa(self, partition_name, partition_metadata, *a, _o=orig, **k):
                    buf.append(("switch", partition_metadata.graph_walk, time.monotonic()))
                    return _o(self, partition_name, partition_metadata, *a, **k)

                m.Pi05Model.get_partition_forward_pass_args = gpfa
                done.add("model")
            s = sys.modules.get("mstar.model.pi05.submodules")
            if s is not None and "sub" not in done and hasattr(s, "Pi05LLMSubmodule"):
                for cls_name in ("Pi05ViTEncoderSubmodule", "Pi05LLMSubmodule"):
                    cls = getattr(s, cls_name)
                    orig_p = cls.prepare_inputs

                    def prep(self, graph_walk, *a, _o=orig_p, _n=cls_name, **k):
                        buf.append((f"prep_{_n}", graph_walk, time.monotonic()))
                        return _o(self, graph_walk, *a, **k)

                    cls.prepare_inputs = prep
                done.add("sub")
            time.sleep(0.05 if len(done) < 2 else 1.0)
            if buf:
                items, buf[:] = buf[:], []
                with open(path, "a") as f:
                    f.writelines(f"{a} {b} {c:.6f}\n" for a, b, c in items)

    threading.Thread(target=_trace_hooks, daemon=True).start()

# Diagnostic only: with RP1_TRACE_DIR, API-side points of a request too.
if os.environ.get("RP1_TRACE_DIR"):
    import threading
    import time

    def _trace_api():
        d = os.environ["RP1_TRACE_DIR"]
        path = os.path.join(d, f"apitrace_{os.getpid()}.txt")
        buf = []
        done = set()
        def mark(n):
            buf.append((n, time.monotonic()))
        while True:
            r = sys.modules.get("starlette.requests")
            if r is not None and "form" not in done and hasattr(r, "Request"):
                o_form = r.Request._get_form

                async def _get_form(self, *a, _o=o_form, **k):
                    mark("form_start")
                    out = await _o(self, *a, **k)
                    mark("form_end")
                    return out

                r.Request._get_form = _get_form
                done.add("form")
            m = sys.modules.get("mstar.api_server.data_worker")
            if m is not None and "dw" not in done and hasattr(m, "PreprocessWorkerThread"):
                cls = m.PreprocessWorkerThread
                o_prep, o_media = cls._prepare_input, cls._load_media

                def _prepare_input(self, *a, _o=o_prep, **k):
                    mark("prep_start")
                    out = _o(self, *a, **k)
                    mark("prep_end")
                    return out

                def _load_media(self, *a, _o=o_media, **k):
                    out = _o(self, *a, **k)
                    mark("media_end")
                    return out

                cls._prepare_input, cls._load_media = _prepare_input, _load_media
                done.add("dw")
            time.sleep(0.05 if len(done) < 2 else 1.0)
            if buf:
                items, buf[:] = buf[:], []
                with open(path, "a") as f:
                    f.writelines(f"{a} {b:.6f}\n" for a, b in items)

    threading.Thread(target=_trace_api, daemon=True).start()

# Diagnostic only (proposal P1 evidence): RP1_PAR_DECODE=1 makes the API data
# worker decode a request's image files concurrently instead of one by one.
if os.environ.get("RP1_PAR_DECODE"):
    import threading
    import time

    def _par_decode():
        while True:
            m = sys.modules.get("mstar.api_server.data_worker")
            if m is not None and hasattr(m, "PreprocessWorkerThread"):
                break
            time.sleep(0.05)
        from concurrent.futures import ThreadPoolExecutor

        pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="rp1dec")
        cls = m.PreprocessWorkerThread
        orig = cls._load_media

        def _load_media(self, file_paths):
            if not file_paths or set(file_paths) != {"image"}:
                return orig(self, file_paths)
            m.check_file_counts({"image": len(file_paths["image"])}, self.model.max_input_files())
            budget = m.RequestMediaBudget(len(file_paths["image"]))
            futs = [pool.submit(self.model.load_image, p, self.device) for p in file_paths["image"]]
            outs = []
            for f in futs:
                with budget.file():
                    outs.append(f.result())
            return {"image_inputs": [o.data for o in outs]}, {"image_inputs": [o.metadata for o in outs]}

        cls._load_media = _load_media

    threading.Thread(target=_par_decode, daemon=True).start()

# Diagnostic only: ingest and output hop marks (rp1_hops.py).
if os.environ.get("RP1_HOPS_DIR"):
    import rp1_hops  # noqa: F401

# Diagnostic only: per-step plan contents (rp1_planlog.py).
if os.environ.get("RP1_PLANLOG_DIR"):
    import rp1_planlog  # noqa: F401

# Diagnostic only: per-step node and batch size (rp1_bs.py).
if os.environ.get("RP1_BS_DIR"):
    import rp1_bs  # noqa: F401
