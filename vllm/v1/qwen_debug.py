# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LOCAL (qwen38 project): unified debug-logging switchboard.

All QWEN_* probe prints go through ``qwen_debug_log`` so there is one
place to flip them on/off. The master switch is ``QWEN_DEBUG``; each
probe keeps its historical env name as an alias:

    QWEN_DEBUG=1|offload|timing|all    master (comma-separated areas or "all")
    QWEN_OFFLOAD_PROBE=1               alias for area "offload"
    QWEN_TIMING=1|2                    alias for area "timing" (2 = sampled)

Probes that merely *extend* an existing upstream log line (e.g. extra
fields on an existing logger.info) stay inline and do not use this.
"""

from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

_MASTER = os.environ.get("QWEN_DEBUG", "").lower()


def _areas() -> frozenset[str]:
    if not _MASTER:
        return frozenset()
    parts = {p.strip() for p in _MASTER.split(",") if p.strip()}
    if "all" in parts or "1" in parts:
        return frozenset({"offload", "timing", "prefix"})
    return frozenset(parts)


_AREAS = _areas()

OFFLOAD_PROBE = "offload" in _AREAS or bool(os.environ.get("QWEN_OFFLOAD_PROBE"))

# Eviction/prefix-cache lifecycle probes ([FREEH]/[STRIP]/[PICK] in
# block_pool). Enabled via QWEN_DEBUG containing "prefix" (or =1/all).
PREFIX_PROBE = "prefix" in _AREAS

# 0=off, 1=every propose, 2=1-in-50 sampling. Master switch sets 1 unless the
# legacy alias gives a finer value.
_legacy_timing = int(os.environ.get("QWEN_TIMING") or 0)
TIMING = 1 if "timing" in _AREAS and _legacy_timing == 0 else _legacy_timing


def qwen_debug_log(area: str, msg: str, *args: object) -> None:
    """Emit a probe log line if ``area`` is enabled via QWEN_DEBUG."""
    if area in _AREAS:
        logger.info(msg, *args)
