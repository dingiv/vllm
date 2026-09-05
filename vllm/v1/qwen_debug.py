# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LOCAL (qwen38 project): unified debug-logging switchboard.

All QWEN_* probe prints go through ``qwen_debug_log`` so there is one
place to flip them on/off. The master switch is ``QWEN_DEBUG``; each
probe keeps its historical env name as an alias:

    qwen-server --debug offload,prefix,...   CLI (qwen38 fork serving pkg)
    QWEN_DEBUG=1|offload|timing|all    master (comma-separated areas or "all")
    QWEN_OFFLOAD_PROBE=1               alias for area "offload"
    QWEN_TIMING=1|2                    alias for area "timing" (2 = sampled)

Off by default. "schedule" is not an area here: that probe family
gates on QWEN_SCHED_PROBE (read at import by the scheduler files); the
CLI translates it.

Probes that merely *extend* an existing upstream log line (e.g. extra
fields on an existing logger.info) stay inline and do not use this.
"""

from __future__ import annotations

import os
import time

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

# Per-instance probe state maps are bounded by this; on overflow they reset
# (probe fidelity degrades to "first again" rather than growing unbounded).
_PROBE_STATE_LIMIT = 1024


class ProbeLogger:
    """Rate-limited probe logger bound to one debug area.

    Holds its own throttle state so call sites don't scatter module-level
    dicts. Create per class in ``__init__`` (or as a class attribute when
    several instances must share one budget, e.g. the 15 KV-group managers
    sampling the same request).
    """

    def __init__(self, area: str) -> None:
        self.area = area
        self._last_emit: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def _on(self) -> bool:
        return self.area in _AREAS

    def log(self, msg: str, *args: object) -> None:
        """Unthrottled passthrough (gated by area)."""
        if self._on():
            logger.info(msg, *args)

    def every(self, key: str, seconds: float, msg: str, *args: object) -> None:
        """Time-window throttle: at most one line per ``key`` per ``seconds``."""
        if not self._on():
            return
        now = time.monotonic()
        if now - self._last_emit.get(key, float("-inf")) < seconds:
            return
        if len(self._last_emit) >= _PROBE_STATE_LIMIT:
            self._last_emit.clear()
        self._last_emit[key] = now
        logger.info(msg, *args)

    def sample(
        self, key: str, first: int, every_n: int, msg: str, *args: object
    ) -> None:
        """Count-sampling: first ``first`` lines pass, then 1-in-``every_n``."""
        if not self._on():
            return
        n = self._counts.get(key, 0) + 1
        if n <= first or n % every_n == 0:
            if len(self._counts) >= _PROBE_STATE_LIMIT:
                self._counts.clear()
            self._counts[key] = n
            logger.info(msg, *args)
            return
        if len(self._counts) >= _PROBE_STATE_LIMIT:
            self._counts.clear()
        self._counts[key] = n


def qwen_debug_log(area: str, msg: str, *args: object) -> None:
    """Emit a probe log line if ``area`` is enabled via QWEN_DEBUG."""
    if area in _AREAS:
        logger.info(msg, *args)
