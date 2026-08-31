# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for AsyncLookupManager."""

import threading
from collections.abc import Iterable

from vllm.v1.kv_offload.base import OffloadKey, ReqContext, make_offload_key
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager


def _key(i: int) -> OffloadKey:
    return make_offload_key(str(i).encode(), 0)


def _ctx(req_id: str = "r1") -> ReqContext:
    return ReqContext(req_id=req_id)


class InMemoryLookupManager(AsyncLookupManager):
    """Test subclass backed by an in-memory set."""

    def __init__(self, existing_keys: set[OffloadKey] | None = None):
        super().__init__(tier_type="test")
        self._existing = existing_keys or set()
        self._results_ready = threading.Event()

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        results = [k in self._existing for k in keys]
        self._results_ready.set()
        return results


class TestAsyncLookupManager:
    def test_new_key_returns_none(self):
        mgr = InMemoryLookupManager()
        assert mgr.lookup(_key(1), _ctx()) is None
        mgr.shutdown()

    def test_found_key_returns_true(self):
        mgr = InMemoryLookupManager(existing_keys={_key(1)})
        assert mgr.lookup(_key(1), _ctx()) is None
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        assert mgr.lookup(_key(1), _ctx()) is True
        mgr.shutdown()

    def test_not_found_key_returns_false(self):
        mgr = InMemoryLookupManager(existing_keys=set())
        assert mgr.lookup(_key(1), _ctx()) is None
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        assert mgr.lookup(_key(1), _ctx()) is False
        mgr.shutdown()

    def test_multiple_keys_single_step(self):
        existing = {_key(1), _key(3)}
        mgr = InMemoryLookupManager(existing_keys=existing)
        ctx = _ctx()
        for i in range(1, 5):
            assert mgr.lookup(_key(i), ctx) is None
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        assert mgr.lookup(_key(1), ctx) is True
        assert mgr.lookup(_key(2), ctx) is False
        assert mgr.lookup(_key(3), ctx) is True
        assert mgr.lookup(_key(4), ctx) is False
        mgr.shutdown()

    def test_cleanup_removes_entries(self):
        mgr = InMemoryLookupManager(existing_keys={_key(1)})
        ctx = _ctx("req_a")
        mgr.lookup(_key(1), ctx)
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        assert mgr.lookup(_key(1), ctx) is True
        mgr.cleanup("req_a")
        assert _key(1) not in mgr._lookup_state
        mgr.shutdown()

    def test_cleanup_preserves_shared_entries(self):
        mgr = InMemoryLookupManager(existing_keys={_key(1)})
        ctx_a = _ctx("req_a")
        ctx_b = _ctx("req_b")
        mgr.lookup(_key(1), ctx_a)
        mgr.lookup(_key(1), ctx_b)
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        # Drain so result is applied
        mgr.lookup(_key(1), ctx_a)
        mgr.cleanup("req_a")
        # Key still present because req_b still references it
        assert _key(1) in mgr._lookup_state
        mgr.cleanup("req_b")
        assert _key(1) not in mgr._lookup_state
        mgr.shutdown()

    def test_flush_no_queue_post_when_empty(self):
        mgr = InMemoryLookupManager()
        mgr.flush()
        assert mgr._lookup_queue.empty()
        mgr.shutdown()

    def test_repeated_lookup_same_key_no_duplicate_batch(self):
        mgr = InMemoryLookupManager(existing_keys={_key(1)})
        ctx = _ctx()
        mgr.lookup(_key(1), ctx)
        mgr.lookup(_key(1), ctx)
        assert len(mgr._lookup_batch) == 1
        mgr.shutdown()

    def test_cleanup_unknown_req_id_is_noop(self):
        mgr = InMemoryLookupManager(existing_keys={_key(1)})
        ctx = _ctx("req_a")
        mgr.lookup(_key(1), ctx)
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        mgr.lookup(_key(1), ctx)
        mgr.cleanup("nonexistent")
        assert _key(1) in mgr._lookup_state
        mgr.shutdown()

    def test_multiple_flushes_across_steps(self):
        existing = {_key(1), _key(2), _key(3)}
        mgr = InMemoryLookupManager(existing_keys=existing)
        ctx = _ctx()

        # Step 1: lookup key 1, flush
        mgr.lookup(_key(1), ctx)
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()

        # Step 2: lookup keys 2 and 3, flush
        mgr.lookup(_key(2), ctx)
        mgr.lookup(_key(3), ctx)
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()

        # All results should be available
        assert mgr.lookup(_key(1), ctx) is True
        assert mgr.lookup(_key(2), ctx) is True
        assert mgr.lookup(_key(3), ctx) is True
        mgr.shutdown()

    def test_shutdown_unblocks_workers(self):
        mgr = InMemoryLookupManager()
        mgr.shutdown()
        assert all(not t.is_alive() for t in mgr._threads)

    def test_cancelled_request_skips_queued_scan(self):
        """cleanup() on a still-queued request must skip its scan entirely."""
        import time

        gate = threading.Event()

        class BlockingLookupManager(AsyncLookupManager):
            def __init__(self):
                super().__init__(tier_type="test")
                self.scanned_req_ids: list[str] = []

            def batch_lookup(self, keys, req_context):
                if req_context.req_id == "slow":
                    # Hold the single batch until we flip the gate.
                    gate.wait(timeout=5)
                self.scanned_req_ids.append(req_context.req_id)
                return [False] * len(keys)

        mgr = BlockingLookupManager()
        slow_ctx = _ctx("slow")
        mgr.lookup(_key(1), slow_ctx)
        mgr.flush()  # queued; worker now blocked inside batch_lookup

        live_ctx = _ctx("live")
        mgr.lookup(_key(2), live_ctx)
        mgr.cleanup("slow")  # abort the slow request while queued/blocked
        mgr.flush()

        gate.set()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and "slow" in mgr.scanned_req_ids:
            time.sleep(0.01)
        assert "slow" not in mgr.scanned_req_ids  # scan skipped, not executed

        mgr.lookup(_key(2), live_ctx)
        mgr.flush()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and mgr.lookup(_key(2), live_ctx) is None:
            mgr.drain_results()  # late-arriving worker results
            time.sleep(0.01)
        assert mgr.lookup(_key(2), live_ctx) is False  # live request resolved
        mgr.shutdown()

    def test_worker_pool_runs_batches_concurrently(self):
        """4 workers must chew 4 held batches at once (barrier proof)."""
        import time

        barrier = threading.Barrier(4, timeout=5)
        active_lock = threading.Lock()
        stats = {"active": 0, "max_active": 0}

        class PoolLookupManager(AsyncLookupManager):
            def __init__(self):
                super().__init__(tier_type="test", n_lookup_threads=4)

            def batch_lookup(self, keys, req_context):
                with active_lock:
                    stats["active"] += 1
                    stats["max_active"] = max(stats["max_active"], stats["active"])
                try:
                    barrier.wait()
                finally:
                    with active_lock:
                        stats["active"] -= 1
                return [False] * len(keys)

        mgr = PoolLookupManager()
        for i in range(4):
            mgr.lookup(_key(100 + i), _ctx(f"r{i}"))
            mgr.flush()
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            mgr.drain_results()  # late-arriving worker results
            if all(mgr.lookup(_key(100 + i), _ctx(f"r{i}")) is False
                   for i in range(4)):
                break
            time.sleep(0.01)
        assert stats["max_active"] >= 4
        mgr.shutdown()
