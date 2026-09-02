# SPDX-License-Identifier: Apache-2.0
# LOCAL (qwen38 project, 2026-09-02): chain-aware eviction policy.
"""Chain-aware LRU cache policy for the CPU primary tier.

Extends LRUCachePolicy with chain-grouped victim selection: evict the
least-recently-touched CHAIN first, and inside a chain always evict the
DEEPEST blocks first (depth descending). Because a prefix-cache hit must
be contiguous from the chain root, evicting deepest-first never punches a
hole into a chain's remaining prefix -- the retained low-depth blocks
stay directly usable on the next request (SGLang RadixAttention's
evict-leaf-first, expressed over hash chains).

Keys without registry metadata (foreign data, disabled registration) form
a pseudo-chain that drains first in plain LRU order: unregistered blocks
are the coldest by definition, and their handling degrades exactly to the
upstream behavior.

Invariants (docs/vllm-链感知驱逐-施工图.md §二):
  I1 no holes  -- within a chain, depth i is never evicted while depth <i
                  of the same chain is evictable.
  I2 atomic    -- evict() mutates nothing unless it can return exactly n.
  I4 fallback  -- unregistered keys behave like upstream LRUCachePolicy.
"""

from collections.abc import Iterable

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey, ReqContext, get_offload_block_hash
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from vllm.v1.kv_offload.tiering.chain_registry import ChainRegistry, get_chain_registry

# Pseudo chain_id grouping unregistered keys; sorted first (oldest).
_NULL_CHAIN = b""


class ChainLRUCachePolicy(LRUCachePolicy):
    """LRU with chain-aware victim ordering. See module docstring."""

    def __init__(self, cache_capacity: int):
        super().__init__(cache_capacity)
        self._registry: ChainRegistry = get_chain_registry()

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self._registry.acquire(get_offload_block_hash(key), "cpu")
        super().insert(key, block)

    @override
    def remove(self, key: OffloadKey) -> None:
        super().remove(key)
        self._registry.release(get_offload_block_hash(key), "cpu")

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        super().touch(keys, req_context)
        self._registry.touch([get_offload_block_hash(k) for k in keys])

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []

        # Group non-protected evictable keys per chain, keeping the
        # OrderedDict (LRU) iteration order inside each group so that
        # depth ties fall back to LRU order (stable sort below).
        by_chain: dict[bytes, list[tuple[int, OffloadKey, BlockStatus]]] = {}
        total = 0
        for key in self.evictable_blocks:
            if key in protected:
                continue
            meta = self._registry.meta(get_offload_block_hash(key))
            chain_id, depth = meta if meta is not None else (_NULL_CHAIN, 0)
            block = self.blocks[key]
            assert block.ref_cnt == 0
            by_chain.setdefault(chain_id, []).append((depth, key, block))
            total += 1
        if total < n:
            return None

        # Chains ordered by last touch (NULL chain first); within a chain
        # deepest-first, LRU order on depth ties (stable sort).
        chain_order = sorted(
            by_chain,
            key=lambda cid: (
                cid != _NULL_CHAIN,
                self._registry.chain_last_touch(cid),
            ),
        )
        victims: list[tuple[OffloadKey, BlockStatus]] = []
        for cid in chain_order:
            group = sorted(by_chain[cid], key=lambda item: -item[0])
            for _, key, block in group:
                victims.append((key, block))
                if len(victims) == n:
                    for vkey, _ in victims:
                        del self.evictable_blocks[vkey]
                        del self.blocks[vkey]
                    return victims
        # Unreachable: total >= n guarantees enough candidates.
        raise AssertionError("chain_lru: victim selection came up short")
