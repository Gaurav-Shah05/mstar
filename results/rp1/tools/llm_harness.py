"""Standalone GPU harness for M*'s pi0.5 LLM node: no server, no worker.

Builds the node through the engine's own code (``EngineManager.build`` ->
``Engine.load_model``: real ``get_node_resources()`` KV / FlashInfer attention /
RoPE resources bound into the real ``Pi05LLMSubmodule`` with real weights, bf16
autocast, fp32 norms), then drives steps through ``Engine.prepare_inputs`` +
``Engine.exec`` (declare_step -> admit -> plan -> preprocess -> forward ->
commit), exactly as the worker's GPU thread does.

Modes (what runs as the forward of the timed walk):
  eager     submodule.forward_batched, eager
  compiled  torch.compile(forward_batched, **compile_kwargs(config.compile_mode))
            i.e. mode="max-autotune-no-cudagraphs", launched eagerly
  graph     Engine.warmup() captures the walk's config (only this walk, only
            this bs / token bucket) via AcceleratorGraphRunner; exec replays it

Timing: CUDA events around the forward call (eager/compiled) or around
``graph.replay()`` (graph), and separately around the whole ``Engine.exec``
(adds plan H2D copies, input staging, commit).

Run (GPU lock is mandatory on this host):
  source /workspace/venv/bin/activate
  flock /workspace/gpu.lock env PYTHONPATH=/workspace/results/rp1-pi05-tools/pyshim:/workspace/wt-rp1-pi05-best \\
      python /workspace/results/rp1-pi05-tools/llm_harness.py --walk action_gen --bs 1 --mode graph --profile
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--walk", choices=["action_gen", "prefill"], default="action_gen")
    p.add_argument("--bs", type=int, default=1, help="requests per step (each gets its own prefill)")
    p.add_argument("--mode", choices=["eager", "compiled", "graph"], default="graph")
    p.add_argument("--modes", default=None,
                   help="comma list to run several modes in one process, e.g. eager,compiled,graph "
                        "(graph is always run last since warmup replaces the engine's forwards)")
    p.add_argument("--profile", action="store_true", help="torch.profiler around --prof-iters steps")
    p.add_argument("--checkout", default="/workspace/wt-rp1-pi05-best",
                   help="M* checkout whose mstar package is imported")
    p.add_argument("--config", default=None,
                   help="serving YAML (default <checkout>/configs/pi05_droid.yaml: action_horizon=15)")
    p.add_argument("--image-tokens", type=int, default=768, help="3 cameras x 256")
    p.add_argument("--text-tokens", type=int, default=62)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--prof-iters", type=int, default=5)
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--trace-dir", default="/workspace/results/rp1-pi05-logs/harness")
    p.add_argument("--tag", default="", help="suffix for trace file names")
    return p.parse_args()


ARGS = parse_args()
# The checkout's mstar must win over anything else on sys.path.
sys.path.insert(0, os.path.abspath(ARGS.checkout))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/workspace/results/rp1-pi05-tools/ind-harness")

import torch  # noqa: E402
import yaml  # noqa: E402

import mstar  # noqa: E402
from mstar.communication.tensors import LocalTransferEngine  # noqa: E402
from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.distributed.communication import WorkerParallelGroups  # noqa: E402
from mstar.engine.accelerator_graph_runner import compile_kwargs  # noqa: E402
from mstar.engine.engine import ExecutingBatch  # noqa: E402
from mstar.engine.resources import StepContext  # noqa: E402
from mstar.engine.resources.kv.transfer import TransferEngineInfo  # noqa: E402
from mstar.model.registry import HF_MODELS, get_model_class  # noqa: E402
from mstar.platforms.compile import with_inductor_overrides  # noqa: E402
from mstar.worker.engine_manager import EngineManager  # noqa: E402

NODE = "LLM"
DEVICE = torch.device("cuda", 0)
DEFAULT_PREFILL_BUCKETS = (512, 1024, 1800)  # Pi05LLMSubmodule.PREFILL_TOKEN_BUCKETS


def log(msg: str) -> None:
    print(f"[harness {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Timer:
    """CUDA-event spans, read back once at the end (no syncs in the loop)."""

    def __init__(self):
        self.spans: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.enabled = False

    def wrap(self, fn):
        def timed(*args, **kwargs):
            if not self.enabled:
                return fn(*args, **kwargs)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn(*args, **kwargs)
            end.record()
            self.spans.append((start, end))
            return out
        return timed

    def take_ms(self) -> list[float]:
        torch.cuda.synchronize()
        out = [s.elapsed_time(e) for s, e in self.spans]
        self.spans.clear()
        return out


class Harness:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        cfg_path = args.config or os.path.join(args.checkout, "configs", "pi05_droid.yaml")
        self.model_config = yaml.safe_load(Path(cfg_path).read_text())
        model_cls = get_model_class(self.model_config["model"])
        log(f"mstar from {os.path.dirname(mstar.__file__)}; config {cfg_path}")
        self.model = model_cls(**HF_MODELS[self.model_config["model"]], **(self.model_config.get("model_kwargs") or {}))
        cfg = self.model.config
        log(f"action_horizon={cfg.action_horizon} num_flow_steps={cfg.num_flow_steps} layers={cfg.num_layers}")
        torch.cuda.set_device(DEVICE)
        t0 = time.perf_counter()
        # The worker's own engine build: specs + YAML overrides, model dtype,
        # get_submodule(...).to(device, node dtype) (fp32 norms), load_model.
        self.manager = EngineManager.build(
            node_names={NODE},
            device=DEVICE,
            model_config=self.model_config,
            parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
            transfer_engine_info=TransferEngineInfo(
                my_entity_id="harness", my_session_id="harness",
                transfer_engine=LocalTransferEngine("harness"),
            ),
            model=self.model,
            graph_runtime=None,
        )
        self.engine = self.manager.engine
        self.mgmt = self.engine._submodules[NODE]
        self.sub = self.mgmt.submodule
        log(f"engine built in {time.perf_counter() - t0:.0f}s; node dtype {self.mgmt.node_dtype}; "
            f"resources {sorted(self.mgmt.resources)}")
        # The time-embedding scratch is sized by max(ACTION_GEN_CAPTURE_BATCH_SIZES)
        # on first use, so set it before any step (bs=8 exceeds the served [1, 2, 4]).
        self.sub.ACTION_GEN_CAPTURE_BATCH_SIZES = [args.bs]
        self._eager_fwd = self.sub.forward_batched
        self._next_rid = 0
        self.fwd_timer = Timer()
        self.exec_timer = Timer()
        self.compiled = None
        self.graph_ready = False

    # -- request plumbing ---------------------------------------------------
    def _info(self, rid: int, walk: str) -> CurrentForwardPassInfo:
        return CurrentForwardPassInfo(
            request_id=f"r{rid}", rid_handle=rid, graph_walk=walk,
            fwd_index=0, random_seed=1234 + rid, max_tokens=1,
        )

    def new_requests(self, n: int) -> list[int]:
        rids = list(range(self._next_rid, self._next_rid + n))
        self._next_rid += n
        for rid in rids:
            self.manager.add_request(rid, {}, self._info(rid, "prefill"))
        return rids

    def remove(self, rids: list[int]) -> None:
        for rid in rids:
            self.manager.remove_request(rid)

    def _batch(self, rids: list[int], walk: str, tensors: dict) -> ExecutingBatch:
        batch = ExecutingBatch(
            node_name=NODE,
            per_request_info={rid: self._info(rid, walk) for rid in rids},
            step_context=StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False),
            per_request_input_tensors=tensors,
        )
        return batch

    def run_step(self, rids: list[int], walk: str) -> ExecutingBatch:
        a = self.args
        if walk == "prefill":
            g = torch.Generator(device=DEVICE).manual_seed(rids[0])
            tensors = {
                rid: {
                    # SigLIP output for 3 cameras; the ViT node runs in fp32
                    "img_emb": [torch.randn(a.image_tokens, self.model.config.hidden_size,
                                            device=DEVICE, generator=g) * 0.5],
                    "text_inputs": [torch.randint(1000, 200000, (a.text_tokens,), device=DEVICE, generator=g)],
                }
                for rid in rids
            }
        else:
            tensors = {rid: {"noisy_actions": []} for rid in rids}
        batch = self._batch(rids, walk, tensors)
        self.engine.prepare_inputs(batch)
        start = end = None
        if self.exec_timer.enabled:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        self.engine.exec(batch)
        if start is not None:
            end.record()
            self.exec_timer.spans.append((start, end))
        if batch.admit_error is not None or batch.failed_requests:
            raise RuntimeError(f"{walk} step failed: admit={batch.admit_error} failed={batch.failed_requests}")
        return batch

    # -- forward selection ----------------------------------------------------
    def set_mode(self, mode: str, walk: str) -> None:
        """Point the engine's forward for ``walk`` at the mode; other walks run eager."""
        if mode == "graph":
            if not self.graph_ready:
                self._capture(walk)
            # warmup() also wrapped the forwards in a default-mode compile;
            # undo that so the non-captured walk (prefill before action_gen) runs eager
            self.mgmt.forward_batched = self._eager_fwd
            runner = self.mgmt.accelerator_graph_runner
            if runner is None:
                raise RuntimeError("warmup captured no graph")
            if not getattr(runner, "_harness_wrapped", False):
                runner._replay = self.fwd_timer.wrap(runner._replay)
                runner._harness_wrapped = True
            return
        if mode == "compiled":
            if self.compiled is None:
                self.compiled = with_inductor_overrides(
                    torch.compile(self.sub.forward_batched, **compile_kwargs("max-autotune-no-cudagraphs")),
                )
            target = self.compiled
        else:
            target = self._eager_fwd
        eager, timed = self._eager_fwd, self.fwd_timer.wrap(target)

        def dispatch(graph_walk, *args, **kwargs):
            fn = timed if graph_walk == walk else eager
            return fn(graph_walk, *args, **kwargs)
        self.mgmt.forward_batched = dispatch

    def _capture(self, walk: str) -> None:
        bs = self.args.bs
        sub = self.sub
        # Capture only what this run replays: one walk, one batch size (and for
        # prefill one token bucket), so warmup is one max-autotune compile.
        # (ACTION_GEN_CAPTURE_BATCH_SIZES is set in __init__.)
        tokens = bs * (self.args.image_tokens + self.args.text_tokens)
        bucket = next((b for b in DEFAULT_PREFILL_BUCKETS if b >= tokens), None)
        if bucket is None:
            bucket = 256 * math.ceil(tokens / 256)
            log(f"prefill: {tokens} tokens exceeds the served buckets {DEFAULT_PREFILL_BUCKETS}; "
                f"capturing a non-served bucket of {bucket}")
        sub.PREFILL_TOKEN_BUCKETS = [bucket]
        sub.PREFILL_CAPTURE_BATCH_SIZES = [bs]
        cls_configs = type(sub).get_accelerator_graph_configs

        def only_this_walk(device, tp_world_size=1):
            return [c for c in cls_configs(sub, device, tp_world_size) if c.capture_graph_walk == walk]
        sub.get_accelerator_graph_configs = only_this_walk
        log(f"capturing {walk} bs={bs}" + (f" bucket={bucket} tokens" if walk == "prefill" else "")
            + " (max-autotune compile inside; minutes on a cold inductor cache)")
        t0 = time.perf_counter()
        self.manager.warmup_all()
        log(f"warmup/capture took {time.perf_counter() - t0:.0f}s; buckets: "
            f"{list(self.mgmt.accelerator_graph_runner._slots.keys()) if hasattr(self.mgmt.accelerator_graph_runner, '_slots') else '?'}")
        self.graph_ready = True

    # -- measurement ----------------------------------------------------------
    def measure(self, mode: str, walk: str) -> dict:
        a = self.args
        self.set_mode(mode, walk)
        bs = a.bs
        held: list[int] = []
        if walk == "action_gen":
            held = self.new_requests(bs)
            self.run_step(held, "prefill")  # eager prefill commits each request's KV

        def one_step():
            if walk == "action_gen":
                b = self.run_step(held, "action_gen")
                return b
            rids = self.new_requests(bs)
            b = self.run_step(rids, "prefill")
            torch.cuda.synchronize()  # pages go back on remove; keep it simple
            self.remove(rids)
            return b

        log(f"{mode} {walk} bs={bs}: {a.warmup} warmup steps")
        t0 = time.perf_counter()
        for _ in range(a.warmup):
            b = one_step()
        torch.cuda.synchronize()
        log(f"  warmup done in {time.perf_counter() - t0:.1f}s; lease={b.step_context.slot_lease}")
        if mode == "graph" and b.step_context.slot_lease is None:
            raise RuntimeError("graph mode but the step ran eager (no lease)")

        self.fwd_timer.enabled = self.exec_timer.enabled = True
        host = []
        for _ in range(a.iters):
            t = time.perf_counter()
            one_step()
            host.append((time.perf_counter() - t) * 1e3)
        self.fwd_timer.enabled = self.exec_timer.enabled = False
        fwd = self.fwd_timer.take_ms()
        ex = self.exec_timer.take_ms()
        res = {"mode": mode, "walk": walk, "bs": bs, "fwd": fwd, "exec": ex, "host": host}
        log(f"RESULT {mode:8s} {walk:10s} bs={bs}: forward GPU median {med(fwd):8.3f} ms "
            f"(min {min(fwd):.3f}, max {max(fwd):.3f}); whole exec GPU span median {med(ex):.3f} ms; "
            f"exec host wall median {med(host):.3f} ms")

        if a.profile:
            self.profile(mode, walk, one_step)
        if held:
            torch.cuda.synchronize()
            self.remove(held)
        return res

    def profile(self, mode: str, walk: str, one_step) -> None:
        a = self.args
        from torch.profiler import ProfilerActivity, profile
        os.makedirs(a.trace_dir, exist_ok=True)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(a.prof_iters):
                one_step()
            torch.cuda.synchronize()
        name = f"{walk}_bs{a.bs}_{mode}{('_' + a.tag) if a.tag else ''}"
        trace = os.path.join(a.trace_dir, f"{name}.json")
        prof.export_chrome_trace(trace)
        table = prof.key_averages().table(sort_by="self_device_time_total", row_limit=a.top, max_name_column_width=90)
        with open(os.path.join(a.trace_dir, f"{name}_table.txt"), "w") as f:
            f.write(table)
        print(table)
        kernel_summary(prof, a.prof_iters, a.top, os.path.join(a.trace_dir, f"{name}_kernels.txt"))
        log(f"trace -> {trace}")


def med(xs: list[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2]


def classify(name: str) -> str:
    n = name.lower()
    if "memcpy" in n or "memset" in n:
        return "memcpy/memset"
    if any(k in n for k in ("flashinfer", "batchprefill", "batchdecode", "attention", "flash_fwd", "fmha", "sdpa")):
        return "attention"
    if any(k in n for k in ("gemm", "cutlass", "xmma", "nvjet", "cublas", "triton_tem", "_mm_", "matmul", "sm90", "splitk")):
        return "gemm"
    if "triton_red" in n or "triton_per" in n or "rms" in n or "norm" in n:
        return "norm/reduction (fused)"
    if "triton_poi" in n or "elementwise" in n or "vectorized" in n or "unrolled" in n:
        return "elementwise"
    return "other"


def kernel_summary(prof, steps: int, top: int, path: str) -> None:
    """Device-side events only, per kernel: total, per call, calls per step."""
    rows = []
    for e in prof.key_averages():
        if e.device_type != torch.autograd.DeviceType.CUDA:
            continue
        t = e.self_device_time_total
        if t <= 0:
            continue
        rows.append((e.key, t / 1e3, e.count))
    rows.sort(key=lambda r: -r[1])
    total = sum(r[1] for r in rows)
    lines = [f"device kernels over {steps} steps: total {total:.3f} ms = {total / steps:.3f} ms/step"]
    lines.append(f"{'ms/step':>9} {'%':>6} {'us/call':>9} {'calls/step':>10}  {'class':<22} kernel")
    for name, ms, n in rows[:top]:
        lines.append(f"{ms / steps:9.3f} {100 * ms / total:6.2f} {1e3 * ms / n:9.2f} {n / steps:10.1f}  "
                     f"{classify(name):<22} {name[:150]}")
    cats = defaultdict(float)
    for name, ms, _ in rows:
        cats[classify(name)] += ms
    lines.append("by class (ms/step):")
    for c, ms in sorted(cats.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {c:<24} {ms / steps:8.3f}  ({100 * ms / total:5.1f}%)")
    text = "\n".join(lines)
    print(text)
    with open(path, "w") as f:
        f.write(text + "\n")


def main() -> None:
    a = ARGS
    modes = a.modes.split(",") if a.modes else [a.mode]
    # graph warmup rebinds the engine's forwards, so it goes last
    modes = sorted(modes, key=lambda m: m == "graph")
    h = Harness(a)
    results = [h.measure(m, a.walk) for m in modes]
    print("\nSUMMARY (CUDA-event medians, ms)")
    for r in results:
        print(f"  {r['walk']:10s} bs={r['bs']} {r['mode']:8s} forward {med(r['fwd']):8.3f}   "
              f"exec span {med(r['exec']):8.3f}   host {med(r['host']):8.3f}")
    h.manager.shutdown()


if __name__ == "__main__":
    with torch.no_grad():
        main()
