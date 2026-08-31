# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystemTierManager: Pure-Python file system secondary tier for KV cache offloading.

Store path:
    Data is written to a temp file (<dest_path.tmp>) via os.write,
    then os.replace'd to the final path (without .tmp).

Load path:
    Data is read from the block file directly via os.readv into the
    provided memoryview slice.

File naming:  <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)
"""

import functools
import json
import os
import time
from pathlib import Path
from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar

try:
    from vllm.fs_io_C import batch_lookup as batch_lookup_C

    _HAS_BATCH_LOOKUP_C = True
except ImportError:
    _HAS_BATCH_LOOKUP_C = False

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadKey,
    ReqContext,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.qwen_debug import qwen_debug_log, OFFLOAD_PROBE

from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.io import (
    batch_load_block,
    batch_store_block,
    probe_o_direct,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class FsAsyncLookupManager(AsyncLookupManager):
    """Async lookup manager for FileSystemTierManager."""

    def __init__(
        self,
        tier: "FileSystemTierManager",
        tier_type: str,
        n_lookup_threads: int = 4,
    ) -> None:
        super().__init__(tier_type=tier_type, n_lookup_threads=n_lookup_threads)
        self._tier = tier

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        import time as _time
        _t0 = _time.monotonic() if OFFLOAD_PROBE else 0.0
        paths = [self._tier.file_mapper.get_file_name(k) for k in keys]
        if _HAS_BATCH_LOOKUP_C:
            # C extension: GIL released for the entire faccessat() batch.
            _r = list(batch_lookup_C(paths))
        else:
            _r = list(os.path.exists(p) for p in paths)
        if OFFLOAD_PROBE:
            _ms = (_time.monotonic() - _t0) * 1000
            if _ms > 100:
                qwen_debug_log(
                    "offload",
                    "[QKV] fs.batch_lookup SLOW keys=%d %.1fms hits=%d",
                    len(keys), _ms, sum(1 for x in _r if x),
                )
        return iter(_r)


class FileSystemTierManager(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished_jobs() polls job completion and returns completed JobResults.

    Cross-process sharing:
        In order to enable KV cache sharing between multiple vLLM instances
        using the same ``root_dir`` (e.g., via a shared PVC) the environment
        variable ``PYTHONHASHSEED`` must be set to the same fixed value
        (e.g., "0") on all instances. Without this, each process initializes
        ``NONE_HASH`` (the chain-hash seed for block content hashes) with
        random bytes, producing different block filenames for identical token
        content.
    """

    medium: ClassVar[Medium] = Medium.STORAGE

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        enable_kv_events: bool = False,
        locality: str | None = None,
        max_kv_bytes: int = 0,
    ):
        """
        Args:
            offloading_spec: Contains normalized offloading configuration and
                blocks_per_chunk.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files.
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
            enable_kv_events: Emit BlockStored KV events for blocks
                successfully stored to this tier. Effective only when KV
                cache events are enabled globally (kv_events_config).
            locality: Whether this tier's storage is LOCAL or REMOTE relative
                to the publishing vLLM instance.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        self.locality = Locality(locality) if locality is not None else None

        self.events: list[OffloadingEvent] | None = None
        if enable_kv_events:
            if offloading_spec.kv_events_config.enable_kv_cache_events:
                self.events = []
            else:
                logger.warning(
                    "enable_kv_events is set on secondary tier '%s' but KV "
                    "cache events are disabled globally; the tier will not "
                    "emit events.",
                    tier_type,
                )
        # Keys of in-flight store jobs, tracked only when events are enabled.
        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]

        # Opt in; FileMapper enables it only for a parallelism-invariant block.
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            blocks_per_file=offloading_spec.blocks_per_chunk,
            parallel_agnostic=True,
        )

        # Write config file
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(
                    self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                )

        # Prefer O_DIRECT to bypass the page cache, but fall back to buffered
        # I/O on filesystems that reject it (e.g. overlayfs, some NFS mounts)
        # rather than failing every block.
        self._use_o_direct = probe_o_direct(os.path.dirname(config_path))
        if not self._use_o_direct:
            logger.warning(
                "O_DIRECT is not supported at '%s'; falling back to buffered "
                "I/O for the '%s' KV offload tier.",
                root_dir,
                tier_type,
            )

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        self._lookup_manager = FsAsyncLookupManager(
            tier=self, tier_type=self.tier_type, n_lookup_threads=4
        )

        # LOCAL (qwen38 project): capacity cap with chunk-granular LRU eviction.
        # 0 = unlimited (upstream behavior). Scope is the WHOLE root_dir (all
        # base-digest dirs and ranks), not just this spec's base_path, so
        # orphaned data from older engine configs still counts and gets
        # evicted. A startup scan seeds an in-memory chunk index; eviction
        # walks that index (O(need)) and only rescans the tree when the index
        # runs dry.
        self._max_kv_bytes = int(max_kv_bytes or 0)
        self._used_kv_bytes = 0
        self._root = Path(root_dir)
        self._chunk_index: dict[str, list] = {}
        self._evict_blocked: set[str] = set()
        self._oversize_warned = False
        self._last_evict_log = 0.0
        if self._max_kv_bytes:
            self._recount_existing_bytes()
            logger.info(
                "fs tier quota: max=%d bytes, existing usage=%d bytes "
                "(scope=%s)",
                self._max_kv_bytes, self._used_kv_bytes, self._root,
            )

    def _iter_block_files(self) -> Iterable[str]:
        """All block files under root_dir, any base digest, any rank.

        Layout is fixed by FileMapper.get_file_name:
        <root>/<base>_r<rank>/<hhh>/<hh>_g<group>/<hash>.bin
        """
        import glob as _glob
        return _glob.glob(
            os.path.join(str(self._root), "*_r[0-9]", "*", "*_g*", "*.bin")
        )

    def _recount_existing_bytes(self) -> None:
        """Scan root_dir for pre-existing block files (restart recovery),
        rebuild the chunk index and resync the usage counter with disk.
        Files left by a previous engine instance are still valid: keys are
        content hashes, so lookup finds them."""
        index: dict[str, list] = {}
        total = 0
        for path_str in self._iter_block_files():
            path = Path(path_str)
            try:
                st = path.stat()
            except OSError:
                continue
            total += st.st_size
            index.setdefault(path.stem, []).append((path, st.st_size, st.st_mtime))
        self._chunk_index = index
        self._evict_blocked.clear()
        self._used_kv_bytes = total

    def _evict_oldest_chunks(self, need_bytes: int) -> int:
        """Delete oldest chunks (min mtime across the chunk's group files)
        until at least need_bytes are freed or nothing evictable remains.
        Whole-chunk eviction: all group/rank files sharing one chunk hash go
        together (band coherence)."""
        freed = 0
        while freed < need_bytes:
            candidates = [
                (key, files)
                for key, files in self._chunk_index.items()
                if key not in self._evict_blocked
            ]
            if not candidates:
                self._recount_existing_bytes()
                candidates = [
                    (key, files)
                    for key, files in self._chunk_index.items()
                    if key not in self._evict_blocked
                ]
                if not candidates:
                    break
            key, files = min(
                candidates, key=lambda kv: min(t for _, _, t in kv[1])
            )
            chunk_bytes = sum(sz for _, sz, _ in files)
            removed = True
            for f, _, _ in files:
                try:
                    f.unlink()
                except OSError:
                    removed = False
                    break
            if not removed:
                self._evict_blocked.add(key)
                continue
            self._prune_empty_dirs(files)
            self._chunk_index.pop(key, None)
            self._used_kv_bytes -= chunk_bytes
            freed += chunk_bytes
        return freed

    def _prune_empty_dirs(self, files: list) -> None:
        """Remove now-empty hash/group dirs left behind by eviction,
        stopping at the tier root."""
        seen: set = set()
        for f, _, _ in files:
            d = f.parent
            while d != self._root and d not in seen:
                seen.add(d)
                try:
                    d.rmdir()
                except OSError:
                    break
                d = d.parent

    def _evict_until_under_cap(self, incoming_bytes: int) -> None:
        """Ensure used - evicted + incoming <= max, deleting oldest chunks."""
        if not self._max_kv_bytes:
            return
        if incoming_bytes > self._max_kv_bytes:
            if not self._oversize_warned:
                logger.warning(
                    "fs tier: single store job (%d bytes) exceeds quota "
                    "(%d bytes); letting it through unbounded",
                    incoming_bytes, self._max_kv_bytes,
                )
                self._oversize_warned = True
            return
        budget = self._max_kv_bytes - incoming_bytes
        if self._used_kv_bytes <= budget:
            return
        need = self._used_kv_bytes - budget
        freed = self._evict_oldest_chunks(need)
        now = time.monotonic()
        if now - self._last_evict_log >= 30.0:
            self._last_evict_log = now
            logger.warning(
                "fs tier eviction: freed %d bytes (need %d), usage now %d / %d",
                freed, need, self._used_kv_bytes, self._max_kv_bytes,
            )
        qwen_debug_log(
            "offload",
            "[QKV] fs.evict freed=%d usage=%d/%d",
            freed, self._used_kv_bytes, self._max_kv_bytes,
        )

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        return LookupResult.HIT if result else LookupResult.MISS

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        qwen_debug_log(
            "offload",
            "[QKV] fs.submit_store blocks=%d used=%d/%s",
            len(job_metadata.block_ids),
            self._used_kv_bytes,
            self._max_kv_bytes or "inf",
        )
        if self._max_kv_bytes:
            # Pre-evict based on bytes about to be written (block_size per key;
            # actual written size equals block_size -- full-block DMA).
            self._evict_until_under_cap(
                self._block_size * len(job_metadata.block_ids)
            )
        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = list(job_metadata.keys)
        task = functools.partial(
            batch_store_block,
            [self.file_mapper.get_file_name(key) for key in job_metadata.keys],
            self._primary_kv_view,
            [int(bid) * self._block_size for bid in job_metadata.block_ids],
            self._block_size,
            self._use_o_direct,
        )
        self._pool.enqueue_store(job_metadata.job_id, 1, [task])

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        task = functools.partial(
            batch_load_block,
            [self.file_mapper.get_file_name(key) for key in job_metadata.keys],
            self._primary_kv_view,
            [int(bid) * self._block_size for bid in job_metadata.block_ids],
            self._block_size,
            self._use_o_direct,
        )

        self._pool.enqueue_load(job_metadata.job_id, 1, [task])

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Collect completed jobs from the finished-jobs queue.
        """
        results = []
        for job_id, success in self._pool.get_finished():
            if success and self._max_kv_bytes:
                self._used_kv_bytes += self._block_size
            if self.events is not None:
                keys = self._store_job_keys.pop(job_id, None)
                if success and keys:
                    self.events.append(
                        OffloadingEvent(
                            keys=keys,
                            medium=self.medium,
                            removed=False,
                            locality=self.locality,
                        )
                    )
            results.append(JobResult(job_id=job_id, success=success))
        return results

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    @override
    def drain_jobs(self) -> None:
        """Block until all in-flight transfers in the threadpool finish."""
        self._pool.wait_idle()

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup_manager.flush()

    @override
    def shutdown(self) -> None:
        """
        Release resources held by this tier.

        Shuts down the lookup manager and the thread pool,
        clearing pending tasks and waiting for active threads to complete.
        """
        self._lookup_manager.shutdown()
        self._pool.shutdown(wait=True)
