# SPDX-License-Identifier: Apache-2.0
# LOCAL (qwen38): tests for chain-aware fs-tier eviction scoring.
import pytest

from vllm.v1.kv_offload.tiering.chain_registry import (
    _reset_chain_registry,
    get_chain_registry,
)

from tests.v1.kv_offload.tiering.test_fs_tier import (
    _make_quota_tier,
    _write_block_file,
)


def _h32(i: int) -> str:
    """64-char hex stem encoding int i in the last byte."""
    return f"{'0' * 62}{i:02x}"


@pytest.fixture()
def chain_env(monkeypatch):
    monkeypatch.setenv("QWEN_OFFLOAD_CHAIN_EVICT", "1")
    _reset_chain_registry()
    yield get_chain_registry()
    _reset_chain_registry()


def _register_two_chains(registry):
    """Chain A: hashes 10..13 depths 0..3; chain B: hashes 20..23."""
    registry.register([bytes.fromhex(_h32(i)) for i in range(10, 14)],
                      b"chain-a", start_depth=0)
    registry.register([bytes.fromhex(_h32(i)) for i in range(20, 24)],
                      b"chain-b", start_depth=0)


def test_anchor_blocks_survive_tail_first_eviction(tmp_path, chain_env):
    _register_two_chains(chain_env)
    # Every A block is older than every B block; within each chain depth
    # correlates with mtime (stores append in depth order).
    for i in range(10, 14):
        _write_block_file(tmp_path, "base_a", 0, 0, _h32(i), 10, 1000 + i)
    for i in range(20, 24):
        _write_block_file(tmp_path, "base_a", 0, 0, _h32(i), 10, 2000 + i)
    tier = _make_quota_tier(tmp_path, max_kv_bytes=85)
    tier._chain_evict = True
    try:
        # Plain mtime order would evict A10 (depth 0, the anchor) first.
        # Scoring skips anchors (depth <= min+2) for live chains, so the
        # oldest NON-anchor victim is A13 (chain A's deepest block).
        tier._evict_until_under_cap(incoming_bytes=10)
        evicted = _h32(13)
        assert not list(tmp_path.rglob(f"{evicted}.bin"))
        # Anchors survive: chain A keeps a contiguous usable base 10..12.
        for anchor in (10, 11, 12):
            assert list(tmp_path.rglob(f"{_h32(anchor)}.bin"))
        # Deeper blocks of the other chain also survive this pass.
        for i in (20, 21, 22, 23):
            assert list(tmp_path.rglob(f"{_h32(i)}.bin"))
    finally:
        tier.shutdown()


def test_all_anchors_falls_back_to_mtime_order(tmp_path, chain_env):
    # Only chain A exists; every candidate is an anchor (span=2 over 4
    # blocks means depths 0..2 anchored, 13.. wait -- depth 3 is not).
    chain_env.register([bytes.fromhex(_h32(i)) for i in range(10, 14)],
                       b"chain-a", start_depth=0)
    for i in range(10, 14):
        _write_block_file(tmp_path, "base_a", 0, 0, _h32(i), 10, 1000 + i)
    tier = _make_quota_tier(tmp_path, max_kv_bytes=45)
    tier._chain_evict = True
    try:
        # Force the fallback by anchoring everything: shrink the span.
        tier._CHAIN_ANCHOR_SPAN = 5
        tier._evict_until_under_cap(incoming_bytes=10)
        # I3: progress is made, in plain mtime order (A10 oldest first).
        assert not list(tmp_path.rglob(f"{_h32(10)}.bin"))
        for i in (11, 12, 13):
            assert list(tmp_path.rglob(f"{_h32(i)}.bin"))
    finally:
        tier.shutdown()


def test_disabled_scoring_evicts_plain_mtime(tmp_path, chain_env, monkeypatch):
    monkeypatch.setenv("QWEN_OFFLOAD_CHAIN_EVICT", "0")
    _register_two_chains(chain_env)
    for i in range(10, 14):
        _write_block_file(tmp_path, "base_a", 0, 0, _h32(i), 10, 1000 + i)
    tier = _make_quota_tier(tmp_path, max_kv_bytes=45)
    tier._chain_evict = False
    try:
        tier._evict_until_under_cap(incoming_bytes=10)
        # Off switch: pure LRU, the anchor A10 goes first.
        assert not list(tmp_path.rglob(f"{_h32(10)}.bin"))
    finally:
        tier.shutdown()


def test_eviction_releases_registry_refcount(tmp_path, chain_env):
    _register_two_chains(chain_env)
    for i in range(10, 14):
        _write_block_file(tmp_path, "base_a", 0, 0, _h32(i), 10, 1000 + i)
        chain_env.acquire(bytes.fromhex(_h32(i)), "fs")
    tier = _make_quota_tier(tmp_path, max_kv_bytes=45)
    tier._chain_evict = True
    try:
        tier._evict_until_under_cap(incoming_bytes=10)
        # A13 evicted and released; A10..A12 still held. Chain alive.
        assert chain_env.chain_alive(b"chain-a")
        assert chain_env.meta(bytes.fromhex(_h32(13))) is None
        assert chain_env.meta(bytes.fromhex(_h32(12))) is not None
    finally:
        tier.shutdown()
