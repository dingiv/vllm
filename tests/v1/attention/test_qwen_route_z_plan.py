"""G2 builder-level equivalence: legacy per-step sync plan vs Route Z
(QWEN_MIXED_PLAN) bound-hint plan with device-side D2D correction.

Simulates a DFlash draft attention step shape: non-causal, uniform query
length per row, mixed prefill/decode rows, decode-row hint overshoot by the
rejected-token count, multi-step growth including exact and overshoot page
boundary crossings, and batch membership changes.

Bitwise output equality against the legacy sync path is the hard gate.
"""

import unittest.mock

import pytest
import torch

from tests.v1.attention.utils import create_common_attn_metadata, create_vllm_config
from vllm.config import set_current_vllm_config
from vllm.platforms import current_platform
from vllm.v1.attention.backends import flashinfer as flashinfer_backend
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.utils import PerLayerParameters
from vllm.v1.kv_cache_interface import FullAttentionSpec

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Route Z gate needs a CUDA device",
)

PAGE = 16
Q_PER_REQ = 4  # dflash draft query length per row
HEADS = 2
HEAD_DIM = 64
DTYPE = torch.bfloat16


def _mock_per_layer_params(vllm_config, layer_names, impl_cls):
    return {
        "placeholder": PerLayerParameters(
            window_left=-1, logits_soft_cap=0.0, sm_scale=0.1
        )
    }


def _make_builder(vllm_config, kv_cache_spec, device):
    with unittest.mock.patch(
        "vllm.v1.attention.backends.flashinfer.get_per_layer_parameters",
        _mock_per_layer_params,
    ):
        return flashinfer_backend.FlashInferMetadataBuilder(
            kv_cache_spec, ["placeholder"], vllm_config, device
        )


def _forward(attn_metadata, q, pool):
    wrapper = attn_metadata.prefill.wrapper
    return wrapper.forward(q, (pool, pool), causal=False, sm_scale=0.1)


@pytest.mark.parametrize(
    "steps",
    [8],
)
def test_route_z_matches_legacy_sync_plan(steps):
    # NOTE (P2c-v2 falsified, 2026-09-08): parametrizing this test with a
    # "trusted" fast path (skipping the exact sync on growth crossings and
    # relying on over-partition + D2D correction) DIVERGES at the first
    # overshoot-into-a-new-page step (max|diff|~0.76). Probes E/Z had only
    # verified within-page overshoot. Z-v1's crossing fallback is necessary;
    # the fix direction is making the fallback itself sync-free (mirror
    # algebra), not skipping it.
    from tests.v1.attention.utils import BatchSpec

    device = torch.device("cuda:0")
    import os

    local_model = os.environ.get(
        "QWEN_TEST_MODEL",
        "/home/div/Documents/codes/models/cyankiwi/Qwen3.8-27B-AWQ-INT4",
    )
    vllm_config = create_vllm_config(model_name=local_model, max_model_len=1024, block_size=PAGE)
    kv_cache_spec = FullAttentionSpec(
        block_size=PAGE,
        num_kv_heads=HEADS,
        head_size=HEAD_DIM,
        dtype=DTYPE,
    )

    # lens trajectory: [decode, decode, prefill, prefill]
    # decode rows grow by 3 (accepted); their CPU bound overshoots by 2
    # (unrejected drafts); prefill rows grow by 37 (chunk) with exact bound.
    decode_lens = [124, 61]
    prefill_lens = [200, 500]
    overshoot = 2

    max_rows = 4
    max_pages_per_row = (1024 + PAGE - 1) // PAGE
    num_blocks = max_rows * max_pages_per_row + 8
    pool = torch.randn(
        num_blocks, PAGE, HEADS, HEAD_DIM, device=device, dtype=DTYPE
    )
    # distinct per-row pages via arange block tables
    block_table = (
        torch.arange(max_rows * max_pages_per_row, dtype=torch.int32)
        .view(max_rows, max_pages_per_row)
        .to(device)
    )

    torch.manual_seed(3)
    with set_current_vllm_config(vllm_config):
        legacy_builder = _make_builder(vllm_config, kv_cache_spec, device)
        z_builder = _make_builder(vllm_config, kv_cache_spec, device)

        for step in range(steps):
            lens = decode_lens + prefill_lens
            bs = len(lens)
            exact = torch.tensor(lens, dtype=torch.int32)
            is_prefill = torch.tensor(
                [False, False][: len(decode_lens)] + [True] * len(prefill_lens)
            )
            hint = exact.clone()
            hint[~is_prefill] += overshoot  # decode-row bound overshoot
            assert (hint >= exact).all()

            spec = BatchSpec(seq_lens=lens, query_lens=[Q_PER_REQ] * bs)
            # ---- legacy: no hint, builder syncs seq_lens (exact path) ----
            common_leg = create_common_attn_metadata(
                spec, PAGE, device, arange_block_indices=True
            )
            common_leg = common_leg.replace(causal=False)
            common_leg.block_table_tensor = block_table[:bs]
            # force the sync path: no cached cpu copy, no hint
            common_leg._seq_lens_cpu = None
            common_leg.seq_lens_cpu_hint = None
            meta_leg = legacy_builder.build(0, common_leg)

            # ---- Route Z: bound hint + correction ----
            common_z = create_common_attn_metadata(
                spec, PAGE, device, arange_block_indices=True
            )
            common_z = common_z.replace(causal=False)
            common_z.block_table_tensor = block_table[:bs]
            common_z._seq_lens_cpu = None  # property must come from hint
            common_z.seq_lens_cpu_hint = hint
            common_z.seq_lens_cpu_hint_is_bound = True
            common_z.is_prefilling = is_prefill
            common_z.draft_row_sig = ("req", step)
            meta_z = z_builder.build(0, common_z)

            q = torch.randn(
                bs * Q_PER_REQ, HEADS, HEAD_DIM, device=device, dtype=DTYPE
            )
            out_leg = _forward(meta_leg, q, pool)
            out_z = _forward(meta_z, q, pool)
            torch.cuda.synchronize()

            assert torch.equal(out_leg, out_z), (
                f"step {step}: Route Z diverged from legacy sync plan "
                f"(max|diff|={(out_leg - out_z).abs().max().item()})"
            )
            # planned blocks must always equal the exact blocks (fallback
            # steps bake exact; non-fallback steps opt blocks == exact).
            exact_blocks = ((exact + PAGE - 1) // PAGE).numpy()
            assert (
                z_builder._fi_mixed_planned_blocks is not None
                and (z_builder._fi_mixed_planned_blocks == exact_blocks).all()
            ), f"step {step}: planned blocks diverged from exact"

            # grow
            decode_lens = [x + 3 for x in decode_lens]
            prefill_lens = [x + 37 for x in prefill_lens]

            # batch membership change mid-run: drop one prefill row
            if step == 3:
                prefill_lens.pop()

    # exercise sig change with same row count (decode swap) after loop
    lens = decode_lens + prefill_lens
    if len(lens) != 4:
        return
    with set_current_vllm_config(vllm_config):
        spec = BatchSpec(seq_lens=lens, query_lens=[Q_PER_REQ] * len(lens))
    common_z = create_common_attn_metadata(
        spec, PAGE, device, arange_block_indices=True
    ).replace(causal=False)
    common_z.block_table_tensor = block_table[: len(lens)]
    common_z._seq_lens_cpu = None
    exact = torch.tensor(lens, dtype=torch.int32)
    hint = exact.clone()
    hint[[0, 1]] += overshoot
    common_z.seq_lens_cpu_hint = hint
    common_z.seq_lens_cpu_hint_is_bound = True
    common_z.is_prefilling = torch.tensor(
        [False, False, True, True][: len(lens)]
    )
    common_z.draft_row_sig = ("different", 99)  # forces exact fallback
    with set_current_vllm_config(vllm_config):
        meta_z = z_builder.build(0, common_z)
    common_leg = create_common_attn_metadata(
        spec, PAGE, device, arange_block_indices=True
    ).replace(causal=False)
    common_leg.block_table_tensor = block_table[: len(lens)]
    common_leg._seq_lens_cpu = None
    common_leg.seq_lens_cpu_hint = None
    with set_current_vllm_config(vllm_config):
        meta_leg = legacy_builder.build(0, common_leg)
    q = torch.randn(
        len(lens) * Q_PER_REQ, HEADS, HEAD_DIM, device=device, dtype=DTYPE
    )
    assert torch.equal(_forward(meta_leg, q, pool), _forward(meta_z, q, pool))
