import asyncio

from py_inference_scheduler.speculative.config import DASPushConfig
from py_inference_scheduler.speculative.contracts import TrajectoryPush
from py_inference_scheduler.speculative.push_client import DASPushClient


class FakeRemoteMethod:
    def __init__(self, calls, *, fail=False):
        self._calls = calls
        self._fail = fail

    def remote(self, worker_id, batch):
        if self._fail:
            raise RuntimeError("service down")
        self._calls.append((worker_id, list(batch)))
        return object()  # stands in for the discarded ObjectRef


class FakeService:
    def __init__(self, *, fail=False):
        self.calls = []
        self.add_trajectories = FakeRemoteMethod(self.calls, fail=fail)


def make_push(i=0):
    return TrajectoryPush(problem_id=f"p{i}", token_ids=(1, 2, 3), prompt_len=1, server_id="s")


async def test_flush_on_batch_size():
    # Async context: a timer task is scheduled, so pushes batch up to size.
    service = FakeService()
    client = DASPushClient(service, "w1", DASPushConfig(batch_size=3, flush_interval_s=60))
    for i in range(3):
        client.enqueue(make_push(i))
    assert len(service.calls) == 1
    worker_id, batch = service.calls[0]
    assert worker_id == "w1"
    assert len(batch) == 3
    assert client.pushed == 3


async def test_flush_on_interval():
    service = FakeService()
    client = DASPushClient(service, "w1", DASPushConfig(batch_size=100, flush_interval_s=0.01))
    client.enqueue(make_push())
    assert service.calls == []
    await asyncio.sleep(0.05)
    assert len(service.calls) == 1


async def test_overflow_drops_with_counter():
    service = FakeService()
    # batch_size > capacity so nothing auto-flushes before the timer fires.
    client = DASPushClient(
        service, "w1", DASPushConfig(batch_size=100, queue_capacity=2, flush_interval_s=60)
    )
    for i in range(5):
        client.enqueue(make_push(i))
    assert client.dropped == 3
    assert len(client._buffer) == 2


def test_flush_errors_never_raise():
    service = FakeService(fail=True)
    client = DASPushClient(service, "w1", DASPushConfig(batch_size=1))
    client.enqueue(make_push())
    assert client.flush_errors == 1
    assert client.metrics()["flush_errors"] == 1


def test_flush_without_event_loop_is_synchronous():
    service = FakeService()
    client = DASPushClient(service, "w1", DASPushConfig(batch_size=100, flush_interval_s=60))
    # No running loop: _ensure_flusher must flush inline rather than schedule.
    client.enqueue(make_push())
    assert len(service.calls) == 1
