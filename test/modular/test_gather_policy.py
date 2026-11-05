"""A node's ready rows wait for company under its ``GatherPolicy``, only while the worker is busy."""
from types import SimpleNamespace

from mstar.model.submodule_base import GatherPolicy
from mstar.worker import micro_scheduler
from mstar.worker.micro_scheduler import MicroScheduler

KEY = ("audio_encoder", "prefill_audio")
POLICY = GatherPolicy(rows=8, wait_s=0.03, busy_requests=16)


def _scheduler(monkeypatch, policy=POLICY, requests=32, now=100.0):
    scheduler = MicroScheduler.__new__(MicroScheduler)
    asked = []

    def get_gather_policy(node, walk):
        asked.append((node, walk))
        return policy

    scheduler.engine_manager = SimpleNamespace(
        get_engine=lambda node: SimpleNamespace(get_gather_policy=get_gather_policy),
    )
    scheduler._gather_policies = {}
    scheduler._gathering_since = {}
    scheduler.combined_walk_of = {}
    scheduler.last_ran = {}
    scheduler._latest_step = {}
    scheduler._turns = 0
    scheduler.arrival = dict.fromkeys(range(requests), 0)
    clock = SimpleNamespace(now=now)
    monkeypatch.setattr(micro_scheduler.time, "monotonic", lambda: clock.now)
    return scheduler, clock, asked


def test_rows_wait_until_the_policy_lapses(monkeypatch):
    scheduler, clock, _ = _scheduler(monkeypatch)
    assert scheduler._gathering(KEY, 1)
    clock.now += 0.029
    assert scheduler._gathering(KEY, 1), "the wait counts from when they first could have run"
    clock.now += 0.002
    assert not scheduler._gathering(KEY, 1)


def test_enough_rows_run_at_once(monkeypatch):
    scheduler, _, _ = _scheduler(monkeypatch)
    assert not scheduler._gathering(KEY, 8)


def test_a_quiet_worker_does_not_wait(monkeypatch):
    scheduler, _, _ = _scheduler(monkeypatch, requests=15)
    assert not scheduler._gathering(KEY, 1)


def test_a_node_without_a_policy_runs_at_once_and_is_asked_once(monkeypatch):
    scheduler, _, asked = _scheduler(monkeypatch, policy=None)
    assert not scheduler._gathering(KEY, 1)
    assert not scheduler._gathering(KEY, 1)
    assert asked == [KEY]


def test_the_rows_after_a_step_wait_from_their_own_start(monkeypatch):
    scheduler, clock, _ = _scheduler(monkeypatch)
    assert scheduler._gathering(KEY, 1)
    clock.now += 0.05
    assert not scheduler._gathering(KEY, 1)
    step = SimpleNamespace(
        node_name=KEY[0], graph_walk=KEY[1], request_to_worker_graph={1: 0}, turn_order=None,
    )
    scheduler.note_ran(step)
    assert scheduler._gathering(KEY, 1)
