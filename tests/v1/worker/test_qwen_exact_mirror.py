# SPDX-License-Identifier: Apache-2.0
# Qwen38 P2c unit tests: merge_exact_computed_mirror.
#
# The merge folds an event-synced D2H snapshot of the exact GPU-side
# num_computed_tokens into the optimistic CPU mirror. Invariants:
#   1. drift correction: optimistic (scheduler) value lowered to exact
#   2. only rows in the copy-time batch AND still mapped to the same slot
#      are touched — finished / new / re-added rows keep scheduler values
#   3. min() semantics: a stale-HIGH snapshot (preemption rollback) never
#      raises the mirror
#   4. equality is a no-op (rows=0)

import numpy as np

from vllm.v1.worker.gpu.input_batch import merge_exact_computed_mirror

MAX = 8


def make_mirror(vals):
    return np.array(vals, dtype=np.int32)


def test_drift_correction_lowers_to_exact():
    # req-a: exact 1200, scheduler mirror drifted to 1200 + 3 steps * 4
    mirror = make_mirror([1212, 500, 0, 0, 0, 0, 0, 0])
    exact = make_mirror([1200, 500, 0, 0, 0, 0, 0, 0])
    rid2idx = {"a": 0, "b": 1}
    rows, lowered = merge_exact_computed_mirror(
        mirror, exact, [("a", 0), ("b", 1)], rid2idx
    )
    assert rows == 1 and lowered == 12
    assert mirror[0] == 1200 and mirror[1] == 500


def test_finished_row_skipped():
    # req-a finished between copy and merge; slot 0 holds its stale exact
    # value but req_id_to_index no longer knows it. Must not be touched.
    mirror = make_mirror([0, 4242, 0, 0, 0, 0, 0, 0])  # slot 0 reused? no: new req took slot 2
    exact = make_mirror([300, 4242, 0, 0, 0, 0, 0, 0])
    rid2idx = {"b": 1, "c": 2}
    rows, lowered = merge_exact_computed_mirror(
        mirror, exact, [("a", 0), ("b", 1)], rid2idx
    )
    assert rows == 0 and lowered == 0
    assert mirror[0] == 0  # untouched (a is gone)


def test_new_request_same_slot_not_clobbered():
    # req-a finished, req-new reused slot 0 (add_requests wrote exact 77).
    # The stale snapshot (300, from a) must not clobber it — min() would be
    # safe, but the req_id mapping check skips it entirely.
    mirror = make_mirror([77, 0, 0, 0, 0, 0, 0, 0])
    exact = make_mirror([300, 0, 0, 0, 0, 0, 0, 0])
    rid2idx = {"new": 0}
    rows, lowered = merge_exact_computed_mirror(
        mirror, exact, [("a", 0)], rid2idx
    )
    assert rows == 0
    assert mirror[0] == 77


def test_preemption_rollback_snapshot_high_keeps_scheduler_value():
    # req-a was preempted and re-added: add_requests rewrote the slot with
    # the rolled-back truth (400) while the snapshot still holds the
    # pre-rollback exact (900). min() keeps 400.
    mirror = make_mirror([400, 0, 0, 0, 0, 0, 0, 0])
    exact = make_mirror([900, 0, 0, 0, 0, 0, 0, 0])
    rid2idx = {"a": 0}
    rows, lowered = merge_exact_computed_mirror(
        mirror, exact, [("a", 0)], rid2idx
    )
    assert rows == 0 and lowered == 0
    assert mirror[0] == 400


def test_row_moved_slot_skipped():
    # Row identity moves to a different slot (condense-like remap): the
    # snapshot's slot association is stale for this req_id.
    mirror = make_mirror([0, 0, 555, 0, 0, 0, 0, 0])
    exact = make_mirror([999, 0, 555, 0, 0, 0, 0, 0])
    rid2idx = {"a": 2}
    rows, _ = merge_exact_computed_mirror(
        mirror, exact, [("a", 0)], rid2idx
    )
    assert rows == 0
    assert mirror[2] == 555 and mirror[0] == 0


def test_equality_noop_and_multi_row():
    mirror = make_mirror([100, 200, 308, 0, 0, 0, 0, 0])
    exact = make_mirror([100, 199, 308, 0, 0, 0, 0, 0])
    rid2idx = {"a": 0, "b": 1, "c": 2}
    rows, lowered = merge_exact_computed_mirror(
        mirror, exact, [("a", 0), ("b", 1), ("c", 2)], rid2idx
    )
    assert rows == 1 and lowered == 1
    assert list(mirror[:3]) == [100, 199, 308]
