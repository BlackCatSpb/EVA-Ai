#!/usr/bin/env python3
"""
Hybrid Paged Cache Benchmark
- Creates a synthetic paged store on SSD (data.bin + index.csv)
- Measures access throughput/latency for different patterns
- Compares:
  * RAM-only (all pages preloaded)
  * SSD-paged (no prefetch)
  * SSD-paged + async prefetch (depth k)

No external deps required.

Usage examples:
  python tools/benchmarks/hybrid_paged_cache_benchmark.py --workdir benchmark_cache/hybrid_cache --num-pages 2048 --page-size-mib 4 --pattern seq --prefetch-depth 0
  python tools/benchmarks/hybrid_paged_cache_benchmark.py --workdir benchmark_cache/hybrid_cache --num-pages 2048 --page-size-mib 4 --pattern zipf --prefetch-depth 3

Outputs a JSON summary with timings and hit/miss stats.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import struct
import sys
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from mmap import mmap, ACCESS_READ
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Tuple


@dataclass
class PageMeta:
    offset: int
    length: int


class PagedStore:
    """Simple read-only paged store backed by a single data.bin and an index.csv.
    Index format (CSV): page_id,offset,length
    """

    def __init__(self, root: Path):
        self.root = root
        self.data_path = root / "data.bin"
        self.index_path = root / "index.csv"
        if not self.data_path.exists() or not self.index_path.exists():
            raise FileNotFoundError("PagedStore files not found. Run with --create first.")
        # Load index
        self._pages: List[PageMeta] = []
        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                pid_s, off_s, len_s = line.strip().split(",")
                self._pages.append(PageMeta(int(off_s), int(len_s)))
        self._f = self.data_path.open("rb")
        self._m = mmap(self._f.fileno(), 0, access=ACCESS_READ)
        self.num_pages = len(self._pages)
        self._lock = threading.RLock()

    def close(self):
        with self._lock:
            try:
                self._m.close()
            finally:
                self._f.close()

    def read_page(self, page_id: int) -> bytes:
        meta = self._pages[page_id]
        with self._lock:
            return self._m[meta.offset : meta.offset + meta.length]


class LRUCache:
    """LRU cache with byte-size capacity."""

    def __init__(self, capacity_bytes: int):
        self.capacity = max(0, int(capacity_bytes))
        self._store: OrderedDict[int, bytes] = OrderedDict()
        self._size = 0
        self._lock = threading.RLock()

    def get(self, key: int) -> bytes | None:
        with self._lock:
            val = self._store.get(key)
            if val is not None:
                self._store.move_to_end(key)
            return val

    def put(self, key: int, value: bytes):
        sz = len(value)
        if self.capacity <= 0 or sz > self.capacity:
            return
        with self._lock:
            if key in self._store:
                old = self._store.pop(key)
                self._size -= len(old)
            self._store[key] = value
            self._size += sz
            self._evict_as_needed()

    def _evict_as_needed(self):
        while self._size > self.capacity and self._store:
            k, v = self._store.popitem(last=False)
            self._size -= len(v)

    @property
    def size_bytes(self) -> int:
        return self._size


class Prefetcher(threading.Thread):
    def __init__(self, store: PagedStore, cache: LRUCache, depth: int = 2):
        super().__init__(daemon=True)
        self.store = store
        self.cache = cache
        self.depth = max(0, int(depth))
        self.queue: Deque[int] = deque()
        self._stop = threading.Event()
        self._cv = threading.Condition()

    def run(self):
        while not self._stop.is_set():
            with self._cv:
                self._cv.wait(timeout=0.01)
                # Pull up to depth items
                pull: List[int] = []
                while self.queue and len(pull) < self.depth:
                    pid = self.queue.popleft()
                    pull.append(pid)
            for pid in pull:
                if self.cache.get(pid) is None:
                    data = self.store.read_page(pid)
                    self.cache.put(pid, data)

    def submit(self, upcoming: Iterable[int]):
        if self.depth <= 0:
            return
        with self._cv:
            for pid in upcoming:
                self.queue.append(pid)
            self._cv.notify()

    def stop(self):
        self._stop.set()
        with self._cv:
            self._cv.notify_all()


def create_paged_store(root: Path, num_pages: int, page_size_mib: int, seed: int = 42):
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / "data.bin"
    index_path = root / "index.csv"

    rnd = random.Random(seed)
    page_size = page_size_mib * 1024 * 1024
    with data_path.open("wb") as df, index_path.open("w", encoding="utf-8") as ix:
        offset = 0
        for pid in range(num_pages):
            # Make pseudo-random but deterministic page content
            buf = bytearray(page_size)
            val = rnd.getrandbits(32)
            struct.pack_into("<I", buf, 0, val)
            # fill chunked
            for i in range(4, page_size, 4096):
                buf[i : i + 4] = struct.pack("<I", (val + i) & 0xFFFFFFFF)
            df.write(buf)
            ix.write(f"{pid},{offset},{page_size}\n")
            offset += page_size


def gen_access_pattern(pattern: str, num_pages: int, total_requests: int, hot_ratio: float = 0.2, seed: int = 123) -> List[int]:
    rnd = random.Random(seed)
    if pattern == "seq":
        return [i % num_pages for i in range(total_requests)]
    if pattern == "stride":
        stride = max(1, num_pages // 97)
        return [(i * stride) % num_pages for i in range(total_requests)]
    if pattern == "random":
        return [rnd.randrange(num_pages) for _ in range(total_requests)]
    if pattern == "zipf":
        # 80/20 hot/cold split
        hot = max(1, int(num_pages * hot_ratio))
        hot_set = list(range(hot))
        cold_set = list(range(hot, num_pages))
        seq = []
        for _ in range(total_requests):
            if rnd.random() < 0.8:
                seq.append(hot_set[rnd.randrange(len(hot_set))])
            else:
                seq.append(cold_set[rnd.randrange(len(cold_set))])
        return seq
    raise ValueError(f"Unknown pattern: {pattern}")


def benchmark(store: PagedStore, access_seq: List[int], cache_bytes: int, prefetch_depth: int = 0) -> Dict[str, float]:
    cache = LRUCache(cache_bytes)
    prefetcher = Prefetcher(store, cache, depth=prefetch_depth)
    if prefetch_depth > 0:
        prefetcher.start()

    latencies: List[float] = []
    hits = 0
    misses = 0

    t0 = time.perf_counter()
    for i, pid in enumerate(access_seq):
        # simple announce of next few ids for prefetch
        if prefetch_depth > 0:
            nxt = access_seq[i + 1 : i + 1 + prefetch_depth]
            prefetcher.submit(nxt)

        t_read0 = time.perf_counter()
        data = cache.get(pid)
        if data is None:
            data = store.read_page(pid)
            cache.put(pid, data)
            misses += 1
        else:
            hits += 1
        t_read1 = time.perf_counter()
        latencies.append((t_read1 - t_read0) * 1000.0)  # ms

    t1 = time.perf_counter()

    if prefetch_depth > 0:
        prefetcher.stop()
        prefetcher.join(timeout=1.0)

    total_ms = (t1 - t0) * 1000.0
    total = len(access_seq)
    lat_sorted = sorted(latencies)
    def pct(p: float) -> float:
        if not lat_sorted:
            return 0.0
        k = min(len(lat_sorted) - 1, int(len(lat_sorted) * p))
        return lat_sorted[k]

    return {
        "total_requests": total,
        "cache_bytes": cache_bytes,
        "prefetch_depth": prefetch_depth,
        "hit_rate": hits / total if total else 0.0,
        "throughput_rps": total / (total_ms / 1000.0) if total_ms > 0 else 0.0,
        "lat_p50_ms": pct(0.50),
        "lat_p90_ms": pct(0.90),
        "lat_p99_ms": pct(0.99),
        "elapsed_ms": total_ms,
    }


def benchmark_ram_only(access_seq: List[int], page_size_bytes: int, cache_bytes: int) -> Dict[str, float]:
    # Simulate RAM-only by preloading all pages into RAM up to cache_bytes
    # We simply measure cache lookup path cost to compare upper bound.
    cache = LRUCache(cache_bytes)
    # Preload distinct first N pages within capacity
    distinct = []
    seen = set()
    for pid in access_seq:
        if pid not in seen:
            distinct.append(pid)
            seen.add(pid)
        if len(distinct) * page_size_bytes >= cache_bytes:
            break
    dummy = bytes(page_size_bytes)
    for pid in distinct:
        cache.put(pid, dummy)

    latencies: List[float] = []
    hits = 0
    misses = 0

    t0 = time.perf_counter()
    for pid in access_seq:
        t_read0 = time.perf_counter()
        data = cache.get(pid)
        if data is None:
            misses += 1
            # emulate a miss cost negligible in RAM-only scenario
            cache.put(pid, dummy)
        else:
            hits += 1
        t_read1 = time.perf_counter()
        latencies.append((t_read1 - t_read0) * 1000.0)
    t1 = time.perf_counter()

    total_ms = (t1 - t0) * 1000.0
    total = len(access_seq)
    lat_sorted = sorted(latencies)
    def pct(p: float) -> float:
        if not lat_sorted:
            return 0.0
        k = min(len(lat_sorted) - 1, int(len(lat_sorted) * p))
        return lat_sorted[k]

    return {
        "total_requests": total,
        "cache_bytes": cache_bytes,
        "prefetch_depth": 0,
        "hit_rate": hits / total if total else 0.0,
        "throughput_rps": total / (total_ms / 1000.0) if total_ms > 0 else 0.0,
        "lat_p50_ms": pct(0.50),
        "lat_p90_ms": pct(0.90),
        "lat_p99_ms": pct(0.99),
        "elapsed_ms": total_ms,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=str, required=True, help="Directory to place/read paged store files")
    ap.add_argument("--create", action="store_true", help="Create synthetic store if not exists")
    ap.add_argument("--num-pages", type=int, default=1024)
    ap.add_argument("--page-size-mib", type=int, default=4)
    ap.add_argument("--pattern", type=str, default="zipf", choices=["seq", "stride", "random", "zipf"])
    ap.add_argument("--total-requests", type=int, default=10000)
    ap.add_argument("--cache-mib", type=int, default=256, help="RAM cache capacity in MiB")
    ap.add_argument("--prefetch-depth", type=int, default=2)
    ap.add_argument("--mode", type=str, default="ssd", choices=["ram", "ssd", "ssd_prefetch"])
    args = ap.parse_args()

    root = Path(args.workdir)
    if args.create or not (root / "data.bin").exists():
        print(f"[create] num_pages={args.num_pages} page_size_mib={args.page_size_mib} -> {root}")
        create_paged_store(root, args.num_pages, args.page_size_mib)

    store = None
    try:
        store = PagedStore(root)
        access_seq = gen_access_pattern(args.pattern, store.num_pages, args.total_requests)
        cache_bytes = args.cache_mib * 1024 * 1024
        page_size_bytes = args.page_size_mib * 1024 * 1024

        if args.mode == "ram":
            res = benchmark_ram_only(access_seq, page_size_bytes, cache_bytes)
        elif args.mode == "ssd":
            res = benchmark(store, access_seq, cache_bytes, prefetch_depth=0)
        else:
            res = benchmark(store, access_seq, cache_bytes, prefetch_depth=args.prefetch_depth)

        out = {
            "mode": args.mode,
            "pattern": args.pattern,
            "num_pages": store.num_pages,
            "page_size_mib": args.page_size_mib,
            "cache_mib": args.cache_mib,
            "prefetch_depth": args.prefetch_depth,
            "results": res,
        }
        print(json.dumps(out, indent=2))
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    main()
