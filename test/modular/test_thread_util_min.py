"""``set_thread_util_min``: what the kernel holds afterwards, read back on a thread of its own."""
import ctypes
import logging
import os
import platform
import sys
import threading

import pytest

from mstar.utils import threads

_SCHED_GETATTR = {"x86_64": 315, "aarch64": 275}

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or platform.machine() not in _SCHED_GETATTR,
    reason="sched_setattr is wrapped for Linux on x86_64 and aarch64 only",
)


def _read_util_min() -> int:
    attr = threads._SchedAttr()
    number = _SCHED_GETATTR[platform.machine()]
    assert ctypes.CDLL(None, use_errno=True).syscall(number, 0, ctypes.byref(attr), ctypes.sizeof(attr), 0) == 0
    return attr.sched_util_min


def _on_a_thread(fn):
    out = []
    thread = threading.Thread(target=lambda: out.append(fn()))
    thread.start()
    thread.join()
    return out[0]


def test_the_kernel_holds_the_floor_or_the_call_is_refused(caplog):
    def run():
        before = _read_util_min()
        threads.set_thread_util_min(512)
        return before, _read_util_min()

    with caplog.at_level(logging.WARNING):
        before, after = _on_a_thread(run)
    # A kernel without clamps, or a sandbox: a wrong struct or flag is EINVAL and fails below.
    if os.strerror(95) in caplog.text or os.strerror(1) in caplog.text:
        pytest.skip(f"this kernel refuses utilization clamps: {caplog.text}")
    assert before != 512
    assert after == 512


def test_zero_leaves_the_thread_alone(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("called the kernel")

    monkeypatch.setattr(threads.ctypes, "CDLL", fail)
    threads.set_thread_util_min(0)


def test_a_refusal_is_logged_not_raised(monkeypatch, caplog):
    class Libc:
        @staticmethod
        def syscall(*_args):
            return -1

    monkeypatch.setattr(threads.ctypes, "CDLL", lambda *_a, **_k: Libc)
    with caplog.at_level(logging.WARNING):
        threads.set_thread_util_min(1024)
    assert "refused" in caplog.text


def test_the_worker_and_the_api_server_apply_the_configured_floor(monkeypatch):
    from mstar.api_server import entrypoint
    from mstar.utils import knobs
    from mstar.worker import worker

    seen = []
    monkeypatch.setattr(worker, "set_thread_util_min", seen.append)
    monkeypatch.setattr(entrypoint, "set_thread_util_min", seen.append)
    monkeypatch.setitem(knobs._CONFIGURED, "step_thread_util_min", "1024")

    step = worker.Worker.__new__(worker.Worker)
    step._step_cpus = None
    step._pin_step_thread()

    api = entrypoint.APIServer.__new__(entrypoint.APIServer)
    api._result_cpu = None
    api.pin_result_thread()

    assert seen == [1024, 1024]
