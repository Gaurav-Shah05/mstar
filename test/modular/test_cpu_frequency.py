"""``cpus.run_at_full_frequency``: the uclamp hint is set on the calling thread
and inherited by processes it starts, without touching the scheduling policy."""
import ctypes
import errno
import multiprocessing
import os
import platform
import struct
import sys

import pytest

from mstar.utils import cpus

_SCHED_GETATTR = {"x86_64": 315, "aarch64": 275}


def _util_min_and_policy() -> tuple[int, int]:
    buf = ctypes.create_string_buffer(56)
    libc = ctypes.CDLL(None, use_errno=True)
    assert libc.syscall(_SCHED_GETATTR[platform.machine()], 0, buf, 56, 0) == 0, os.strerror(ctypes.get_errno())
    fields = struct.unpack("IIQiIQQQII", buf.raw)
    return fields[8], fields[1]


def _report(queue) -> None:
    queue.put(_util_min_and_policy())


def _in_fresh_process(target, *args):
    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(target=target, args=args)
    process.start()
    process.join(60)
    if process.is_alive():
        process.kill()
    assert process.exitcode == 0


def _set_then_spawn(queue) -> None:
    cpus.run_at_full_frequency()
    queue.put(_util_min_and_policy())
    # a spawned grandchild: exec keeps the attribute, as a deployment's workers need
    _in_fresh_process(_report, queue)


def test_full_frequency_reaches_spawned_processes_and_keeps_the_policy():
    if sys.platform != "linux" or platform.machine() not in _SCHED_GETATTR:
        pytest.skip("no sched_setattr number here")
    queue = multiprocessing.get_context("spawn").Queue()
    # in a child, so the pytest process's own threads keep the default
    _in_fresh_process(_set_then_spawn, queue)
    first = queue.get(timeout=5)
    if first[0] == 0:
        pytest.skip("this kernel or sandbox refuses uclamp")
    assert first == (1024, os.SCHED_OTHER)
    assert queue.get(timeout=5) == (1024, os.SCHED_OTHER)


def test_a_refused_request_warns_and_does_not_raise(monkeypatch, caplog):
    class _Libc:
        def syscall(self, *args):
            ctypes.set_errno(errno.EOPNOTSUPP)
            return -1

    monkeypatch.setattr(cpus.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(cpus.sys, "platform", "linux")
    monkeypatch.setattr(cpus.ctypes, "CDLL", lambda *a, **k: _Libc())
    cpus.run_at_full_frequency()
    assert "refused" in caplog.text


def test_the_knob_defaults_on_and_turns_off(monkeypatch):
    from mstar.utils import knobs

    monkeypatch.delenv("MSTAR_FULL_CPU_FREQUENCY", raising=False)
    assert knobs.full_cpu_frequency_enabled()
    monkeypatch.setenv("MSTAR_FULL_CPU_FREQUENCY", "0")
    assert not knobs.full_cpu_frequency_enabled()
