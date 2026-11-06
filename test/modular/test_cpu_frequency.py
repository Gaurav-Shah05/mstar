"""``full_cpu_frequency``: the clamp the kernel holds as requested on every thread
and process of a deployment, and the refusal that stops the server at startup."""
import ctypes
import errno
import logging
import multiprocessing
import os
import platform
import sys
import threading
from types import SimpleNamespace

import pytest

from mstar.utils import cpus, knobs

_SCHED_GETATTR = {"x86_64": 315, "aarch64": 275}
_NICE = 5


def _thread_attrs() -> tuple[int, int, int]:
    attr = cpus._SchedAttr()
    libc = ctypes.CDLL(None, use_errno=True)
    number = _SCHED_GETATTR[platform.machine()]
    assert libc.syscall(number, 0, ctypes.byref(attr), ctypes.sizeof(attr), 0) == 0, os.strerror(ctypes.get_errno())
    return attr.sched_util_min, attr.sched_policy, attr.sched_nice


def _report(queue) -> None:
    queue.put(_thread_attrs())


def _on_a_new_thread(queue) -> None:
    thread = threading.Thread(target=_report, args=(queue,))
    thread.start()
    thread.join()


def _deployment(queue) -> None:
    """A thread from before the request, a thread and a spawned process from
    after it, and a process from that spawned process, as conductor and workers are."""
    os.sched_setscheduler(0, os.SCHED_BATCH, os.sched_param(0))
    os.nice(_NICE)
    go = threading.Event()
    earlier = threading.Thread(target=lambda: (go.wait(), _report(queue)), daemon=True)
    earlier.start()
    cpus.run_at_full_frequency()
    _report(queue)
    go.set()
    earlier.join()
    _on_a_new_thread(queue)
    child = multiprocessing.get_context("spawn").Process(target=_spawn_again, args=(queue,))
    child.start()
    child.join(60)
    assert child.exitcode == 0


def _spawn_again(queue) -> None:
    _report(queue)
    process = multiprocessing.get_context("spawn").Process(target=_report, args=(queue,))
    process.start()
    process.join(60)
    assert process.exitcode == 0


def _refusal(queue) -> None:
    try:
        cpus.run_at_full_frequency()
        queue.put(None)
    except cpus.CpuPlanError as e:
        queue.put(str(e))


@pytest.mark.skipif(
    sys.platform != "linux" or platform.machine() not in _SCHED_GETATTR,
    reason="the sched_setattr number is known for Linux on x86_64 and aarch64 only",
)
def test_every_thread_and_process_of_a_deployment_holds_the_clamp():
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()

    # in children, so the pytest process keeps the default
    def in_a_child(target):
        process = ctx.Process(target=target, args=(queue,))
        process.start()
        process.join(120)
        assert process.exitcode == 0

    in_a_child(_refusal)
    refusal = queue.get(timeout=5)
    if refusal is not None:
        pytest.skip(f"this kernel refuses utilization clamps: {refusal}")
    in_a_child(_deployment)
    assert [queue.get(timeout=5) for _ in range(5)] == [(1024, os.SCHED_BATCH, _NICE)] * 5


class _Libc:
    """Fails the calls ``fail`` picks by position, with ``error``."""

    def __init__(self, error, fail=lambda call: True):
        self.error, self.fail, self.calls = error, fail, 0

    def syscall(self, *_args):
        self.calls += 1
        if self.fail(self.calls):
            ctypes.set_errno(self.error)
            return -1
        return 0


@pytest.fixture
def on_linux(monkeypatch):
    monkeypatch.setattr(cpus.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(cpus.sys, "platform", "linux")


@pytest.mark.parametrize("error", [errno.EOPNOTSUPP, errno.EPERM])
def test_a_refusal_names_what_the_kernel_needs(monkeypatch, on_linux, error):
    monkeypatch.setattr(cpus.ctypes, "CDLL", lambda *_a, **_k: _Libc(error))
    with pytest.raises(cpus.CpuPlanError, match="CONFIG_UCLAMP_TASK.*sched_setattr"):
        cpus.run_at_full_frequency()


def test_a_thread_that_exited_since_the_listing_is_not_a_refusal(monkeypatch, on_linux):
    libc = _Libc(errno.ESRCH, fail=lambda call: call == 1)
    monkeypatch.setattr(cpus.ctypes, "CDLL", lambda *_a, **_k: libc)
    monkeypatch.setattr(cpus.os, "listdir", lambda _path: ["101", "102"])
    cpus.run_at_full_frequency()
    assert libc.calls == 2


def _governor(root, name):
    (root / "cpu0/cpufreq").mkdir(parents=True)
    (root / "cpu0/cpufreq/scaling_governor").write_text(name + "\n")
    return root


def test_only_a_governor_that_ignores_the_clamp_warns(monkeypatch, on_linux, caplog, tmp_path):
    monkeypatch.setattr(cpus.ctypes, "CDLL", lambda *_a, **_k: _Libc(0, fail=lambda _call: False))
    monkeypatch.setattr(cpus.os, "sched_getaffinity", lambda _pid: {0, 1})
    with caplog.at_level(logging.INFO):
        cpus.run_at_full_frequency(_governor(tmp_path / "a", "schedutil"))
    assert [r.levelno for r in caplog.records] == [logging.INFO]
    caplog.clear()
    with caplog.at_level(logging.INFO):
        cpus.run_at_full_frequency(_governor(tmp_path / "b", "performance"))
    assert [r.levelno for r in caplog.records] == [logging.WARNING] and "governor performance" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.INFO):
        cpus.run_at_full_frequency(tmp_path / "no-cpufreq")
    assert [r.levelno for r in caplog.records] == [logging.WARNING] and "governor unknown" in caplog.text


def test_a_platform_without_the_syscall_is_refused(monkeypatch):
    monkeypatch.setattr(cpus.sys, "platform", "darwin")
    with pytest.raises(cpus.CpuPlanError, match="sched_setattr"):
        cpus.run_at_full_frequency()


@pytest.fixture
def entrypoint_module(monkeypatch):
    from mstar.api_server import entrypoint

    monkeypatch.delenv("MSTAR_FULL_CPU_FREQUENCY", raising=False)
    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "off")
    yield entrypoint
    knobs.configure(None)
    cpus.install(None)


def test_the_api_server_asks_only_when_the_knob_is_on(monkeypatch, entrypoint_module):
    asked = []
    monkeypatch.setattr(cpus, "run_at_full_frequency", lambda: asked.append(True))
    entrypoint_module.apply_cpu_settings({})
    assert asked == []
    entrypoint_module.apply_cpu_settings({"knobs": {"full_cpu_frequency": True}})
    assert asked == [True]


def test_a_refusal_stops_the_server_with_the_kernels_requirement(monkeypatch, on_linux, entrypoint_module):
    monkeypatch.setattr(cpus.ctypes, "CDLL", lambda *_a, **_k: _Libc(errno.EOPNOTSUPP))
    with pytest.raises(SystemExit, match="mstar: full_cpu_frequency.*CONFIG_UCLAMP_TASK"):
        entrypoint_module.apply_cpu_settings({"knobs": {"full_cpu_frequency": True}})


def test_both_entry_points_apply_the_settings_before_the_conductor_starts(monkeypatch, tmp_path):
    from mstar.api_server import entrypoint
    from mstar.integrations.dynamo import worker as dynamo
    from mstar.model import registry

    class AppliedError(Exception):
        pass

    def apply(_config):
        raise AppliedError

    def start(_process):
        raise AssertionError("the conductor started before the settings were applied")

    monkeypatch.setattr(entrypoint, "apply_cpu_settings", apply)
    monkeypatch.setattr(registry, "get_model_class", lambda _name: lambda **_kwargs: object())
    monkeypatch.setattr(multiprocessing.context.SpawnProcess, "start", start)
    # process-wide, and this process goes on to run other tests
    monkeypatch.setattr(entrypoint, "log_off_thread", lambda: None)
    monkeypatch.setattr(entrypoint, "quiet_noisy_loggers", lambda: None)
    monkeypatch.delenv("MSTAR_API_KEY", raising=False)
    config = tmp_path / "config.yaml"
    config.write_text("model: dummy\n")

    with pytest.raises(AppliedError):
        entrypoint.main(["--config", str(config)])
    with pytest.raises(AppliedError):
        dynamo.build_server(SimpleNamespace(config=str(config), cache_dir=None))
