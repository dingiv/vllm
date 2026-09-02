# SPDX-License-Identifier: Apache-2.0
# LOCAL (qwen38): tests for ChainLRUCachePolicy (chain-aware eviction).
import pytest

from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus
from vllm.v1.kv_offload.cpu.policies.chain_lru import ChainLRUCachePolicy
from vllm.v1.kv_offload.tiering.chain_registry import (
    _reset_chain_registry,
    get_chain_registry,
)


def _h(i: int) -> bytes:
    return i.to_bytes(32, "big")


def _k(i: int):
    return make_offload_key(_h(i), 0)


def _insert(policy, i: int) -> None:
    block = BlockStatus(block_id=i)
    block.ref_cnt = 0
    policy.insert(_k(i), block)


@pytest.fixture()
def registry():
    _reset_chain_registry()
    yield get_chain_registry()
    _reset_chain_registry()


def _register_chain(registry, name: bytes, indices: range, start: int = 0):
    registry.register([_h(i) for i in indices], name, start_depth=start)


def test_no_holes_chain_drains_deepest_first(registry):
    policy = ChainLRUCachePolicy(cache_capacity=100)
    # Chain A (hashes 10..14) registered first => older last_touch.
    _register_chain(registry, b"chain-a", range(10, 15))
    _register_chain(registry, b"chain-b", range(20, 25))
    for i in list(range(10, 15)) + list(range(20, 25)):
        _insert(policy, i)

    # I1: the first five victims are A's deepest-first; B is untouched.
    victims = policy.evict(5, protected=set())
    assert [k for k, _ in victims] == [_k(i) for i in (14, 13, 12, 11, 10)]
    # Next victim switches to chain B, deepest first.
    victims = policy.evict(1, protected=set())
    assert [k for k, _ in victims] == [_k(24)]


def test_touch_reorders_chain_victim_selection(registry):
    policy = ChainLRUCachePolicy(cache_capacity=100)
    _register_chain(registry, b"chain-a", range(10, 13))
    _register_chain(registry, b"chain-b", range(20, 23))
    for i in range(10, 13):
        _insert(policy, i)
    for i in range(20, 23):
        _insert(policy, i)

    # Touching chain A makes B the oldest chain: B drains first now.
    policy.touch([_k(10)], req_context=None)
    victims = policy.evict(3, protected=set())
    assert [k for k, _ in victims] == [_k(i) for i in (22, 21, 20)]


def test_null_chain_drains_first_in_lru_order(registry):
    policy = ChainLRUCachePolicy(cache_capacity=100)
    _register_chain(registry, b"chain-a", range(10, 12))
    # Unregistered keys (older by insertion) + registered ones.
    _insert(policy, 1)
    _insert(policy, 2)
    for i in range(10, 12):
        _insert(policy, i)

    victims = policy.evict(2, protected=set())
    # I4: unregistered keys first, in plain LRU order.
    assert [k for k, _ in victims] == [_k(1), _k(2)]


def test_atomic_refusal_leaves_state_untouched(registry):
    policy = ChainLRUCachePolicy(cache_capacity=100)
    _register_chain(registry, b"chain-a", range(10, 12))
    _insert(policy, 10)
    _insert(policy, 11)

    # I2: cannot satisfy n=3 -> None, and nothing was evicted.
    assert policy.evict(3, protected=set()) is None
    assert len(policy.evictable_blocks) == 2
    assert policy.get(_k(10)) is not None

    # Protected keys are excluded from the candidate count.
    assert policy.evict(2, protected={_k(10), _k(11)}) is None
    victims = policy.evict(1, protected={_k(10)})
    assert [k for k, _ in victims] == [_k(11)]


def test_insert_remove_drive_registry_refcounts(registry):
    policy = ChainLRUCachePolicy(cache_capacity=100)
    _register_chain(registry, b"chain-a", range(10, 12))
    registry.acquire(_h(10), "cpu")  # store-side claim

    _insert(policy, 10)
    _insert(policy, 11)
    policy.remove(_k(10))  # drops the policy-side ref
    assert registry.chain_alive(b"chain-a")
    assert registry.meta(_h(10)) is not None  # store-side ref still holds

    # After the last ref drops, metadata is gone.
    registry.release(_h(10), "cpu")
    assert registry.meta(_h(10)) is None
    policy.remove(_k(11))
    assert not registry.chain_alive(b"chain-a")


def test_remove_unknown_key_is_noop(registry):
    policy = ChainLRUCachePolicy(cache_capacity=100)
    with pytest.raises(KeyError):
        policy.remove(_k(99))
