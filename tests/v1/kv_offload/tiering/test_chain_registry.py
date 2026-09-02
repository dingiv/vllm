# SPDX-License-Identifier: Apache-2.0
# LOCAL (qwen38): tests for the chain registry (chain-aware eviction).
from vllm.v1.kv_offload.tiering.chain_registry import (
    ChainRegistry,
    get_chain_registry,
    _reset_chain_registry,
)


def _h(i: int) -> bytes:
    return i.to_bytes(32, "big")


def test_register_assigns_ascending_depths_keep_first():
    reg = ChainRegistry()
    chain = b"chain-a"
    end = reg.register([_h(1), _h(2), _h(3)], chain, start_depth=7)
    assert end == 10
    assert reg.meta(_h(1)) == (chain, 7)
    assert reg.meta(_h(2)) == (chain, 8)
    assert reg.meta(_h(3)) == (chain, 9)

    # Re-registration of a known hash keeps the first chain/depth, but the
    # cursor still advances past every key in the batch.
    end = reg.register([_h(2), _h(4)], b"chain-b", start_depth=0)
    assert end == 2
    assert reg.meta(_h(2)) == (chain, 8)
    assert reg.meta(_h(4)) == (b"chain-b", 1)
    assert reg.chain_alive(b"chain-a")
    assert reg.chain_alive(b"chain-b")


def test_release_lifecycle_drops_meta_and_chain():
    reg = ChainRegistry()
    chain = b"chain-a"
    reg.register([_h(1), _h(2)], chain, start_depth=0)
    reg.acquire(_h(1), "cpu")
    reg.acquire(_h(1), "fs")
    reg.acquire(_h(2), "cpu")

    reg.release(_h(1), "fs")
    assert reg.meta(_h(1)) == (chain, 0)  # still held by cpu
    reg.release(_h(1), "cpu")
    assert reg.meta(_h(1)) is None  # fully dropped
    assert reg.chain_alive(chain)  # _h(2) remains

    reg.release(_h(2), "cpu")
    assert reg.meta(_h(2)) is None
    assert not reg.chain_alive(chain)

    # Over-release is a safe no-op.
    reg.release(_h(2), "cpu")
    assert not reg.chain_alive(chain)


def test_touch_refreshes_last_touch():
    reg = ChainRegistry()
    reg.register([_h(1)], b"chain-a", start_depth=0)
    reg.register([_h(2)], b"chain-b", start_depth=0)
    before = reg.chain_last_touch(b"chain-a")
    reg.touch([_h(1)])
    assert reg.chain_last_touch(b"chain-a") >= before
    assert reg.chain_last_touch(b"chain-b") <= reg.chain_last_touch(b"chain-a")
    # Touch of an unknown hash is a no-op.
    reg.touch([b"\x00" * 32])


def test_is_empty_and_singleton_reset():
    _reset_chain_registry()
    reg = get_chain_registry()
    assert reg.is_empty()
    reg.register([_h(1)], b"c", start_depth=0)
    assert not reg.is_empty()
    # Singleton identity.
    assert get_chain_registry() is reg
    _reset_chain_registry()
    assert get_chain_registry() is not reg


def test_register_empty_batch_returns_cursor():
    reg = ChainRegistry()
    assert reg.register([], b"c", start_depth=5) == 5
    assert reg.is_empty()
