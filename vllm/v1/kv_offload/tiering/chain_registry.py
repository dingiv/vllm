# SPDX-License-Identifier: Apache-2.0
# LOCAL (qwen38 project, 2026-09-02): chain-aware eviction support.
"""Chain registry: block-hash -> (chain_id, depth) metadata for
session-aware (chain-aware) KV eviction.

A "chain" is the hash-chained prefix of a request (all keys stored by one
request share the chain_id of its first block hash; shared prefixes
therefore share chains). Depth is the arrival-order position of the block
within the chain -- ordering only, contiguity is not required (checkpoint
strides and the eagle volatile-tail deferral may skip positions).

Eviction consumers:
  - ChainLRUCachePolicy (CPU primary tier): evicts whole chains
    oldest-chain-first, deepest-block-first inside a chain, so holes are
    never punched into a chain's usable prefix.
  - FileSystemTierManager: skips eviction candidates that anchor the low
    depths of a chain still alive in this tier.

Registry entries are keyed by block hash (not OffloadKey) so both the CPU
policy and the fs index (whose chunk_index stems are hash hex) share one
metadata table. Tier ownership is refcounted; an entry is dropped when no
tier holds the block anymore.

Single-threaded by construction (all call sites run on the scheduler
thread); a lock guards against future cross-thread callers at negligible
cost. See docs/vllm-链感知驱逐-施工图.md for the full design.
"""

import os
import threading
import time
from collections.abc import Sequence

_CHAIN_EVICT_ENV = "QWEN_OFFLOAD_CHAIN_EVICT"

# Registry instance, created lazily on first use and shared by the CPU
# policy and fs manager (both live in the scheduler process).
_registry: "ChainRegistry | None" = None


def chain_evict_enabled() -> bool:
    """Master switch for chain-aware eviction (CPU policy + fs scoring)."""
    return os.environ.get(_CHAIN_EVICT_ENV, "") == "1"


def get_chain_registry() -> "ChainRegistry":
    global _registry
    if _registry is None:
        _registry = ChainRegistry()
    return _registry


def _reset_chain_registry() -> None:
    """Test-only: drop the shared singleton."""
    global _registry
    _registry = None


class ChainRegistry:
    """Hash-keyed chain metadata with per-tier refcounts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # hash -> (chain_id, depth); keep-first on re-registration.
        self._hash_meta: dict[bytes, tuple[bytes, int]] = {}
        # chain_id -> number of live member hashes.
        self._chain_members: dict[bytes, int] = {}
        # chain_id -> last time any member was stored/touched/loaded.
        self._chain_last_touch: dict[bytes, float] = {}
        # hash -> number of tiers currently holding the block.
        self._tier_refs: dict[bytes, int] = {}

    def register(
        self, hashes: Sequence[bytes], chain_id: bytes, start_depth: int
    ) -> int:
        """Assign (chain_id, start_depth + i) to each new hash, keep-first.

        Returns the depth cursor after this batch so the caller can persist
        it per request. Re-registration of a known hash is a no-op (blocks
        are content-addressed; the first chain/depth wins).
        """
        depth = start_depth
        now = time.monotonic()
        with self._lock:
            for h in hashes:
                if h not in self._hash_meta:
                    self._hash_meta[h] = (chain_id, depth)
                    self._chain_members[chain_id] = (
                        self._chain_members.get(chain_id, 0) + 1
                    )
                self._chain_last_touch[chain_id] = now
                depth += 1
        return depth

    def meta(self, h: bytes) -> tuple[bytes, int] | None:
        with self._lock:
            return self._hash_meta.get(h)

    def is_empty(self) -> bool:
        with self._lock:
            return not self._hash_meta

    def touch(self, hashes: Sequence[bytes]) -> None:
        """Refresh chain recency for the chains of the given hashes."""
        now = time.monotonic()
        with self._lock:
            for h in hashes:
                meta = self._hash_meta.get(h)
                if meta is not None:
                    self._chain_last_touch[meta[0]] = now

    def chain_last_touch(self, chain_id: bytes) -> float:
        with self._lock:
            return self._chain_last_touch.get(chain_id, 0.0)

    def chain_alive(self, chain_id: bytes) -> bool:
        with self._lock:
            return self._chain_members.get(chain_id, 0) > 0

    def acquire(self, h: bytes, tier: str) -> None:
        del tier  # refcount is tier-agnostic; the label is documentation.
        with self._lock:
            self._tier_refs[h] = self._tier_refs.get(h, 0) + 1

    def release(self, h: bytes, tier: str) -> None:
        del tier
        with self._lock:
            refs = self._tier_refs.get(h, 0) - 1
            if refs > 0:
                self._tier_refs[h] = refs
                return
            self._tier_refs.pop(h, None)
            meta = self._hash_meta.pop(h, None)
            if meta is not None:
                chain_id = meta[0]
                remaining = self._chain_members.get(chain_id, 1) - 1
                if remaining > 0:
                    self._chain_members[chain_id] = remaining
                else:
                    self._chain_members.pop(chain_id, None)
                    self._chain_last_touch.pop(chain_id, None)
