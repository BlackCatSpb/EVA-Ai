from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Tuple

import sys
from pathlib import Path

# Ensure project root on sys.path when running as a file from tools/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from cogniflex.adapters.torch_adapter import TorchBatchAdapter, Meta
from cogniflex.runtime.worker_pool import InferenceWorkerPool


def synth_item(seq_len: int) -> Dict[str, Any]:
    ids = torch.randint(low=1, high=1000, size=(seq_len,), dtype=torch.long)
    return {"input_ids": ids}


def run_trial(
    n_items: int,
    min_len: int,
    max_len: int,
    max_items: int,
    max_tokens: int,
    timeout_ms: int,
    num_workers: int,
    torch_threads: int,
    interop_threads: int,
    queue_maxsize: int,
) -> Dict[str, Any]:
    adapter = TorchBatchAdapter(
        max_items=max_items,
        max_tokens=max_tokens,
        timeout_ms=timeout_ms,
        key_for_length="input_ids",
    )
    pool = InferenceWorkerPool(
        model_fn_path="cogniflex.runtime.simple_model.example_model_fn",
        num_workers=num_workers,
        torch_threads=torch_threads,
        interop_threads=interop_threads,
        queue_maxsize=queue_maxsize,
    )

    submitted = 0
    received = 0
    t0 = time.perf_counter()
    pool.start()

    try:
        # Enqueue items and submit batches as they form
        for _ in range(n_items):
            L = random.randint(min_len, max_len)
            adapter.push(synth_item(L))
            b = adapter.try_pop_batch()
            if b is not None:
                pool.submit(b)
                submitted += len(b.metas)
        # Flush remaining
        b = adapter.flush()
        if b is not None:
            pool.submit(b)
            submitted += len(b.metas)

        # Collect all
        while received < submitted:
            _, res = pool.recv(block=True)
            received += int(res.get("logits").shape[0]) if isinstance(res, dict) and "logits" in res else 1
    finally:
        pool.stop()

    t1 = time.perf_counter()
    dur_s = t1 - t0
    tps = received / dur_s if dur_s > 0 else 0.0

    return {
        "submitted": submitted,
        "received": received,
        "duration_s": dur_s,
        "throughput_items_per_s": tps,
        "num_workers": num_workers,
        "torch_threads": torch_threads,
        "interop_threads": interop_threads,
        "batching": {"max_items": max_items, "max_tokens": max_tokens, "timeout_ms": timeout_ms},
    }


def main():
    p = argparse.ArgumentParser(description="Benchmark Torch adapter + worker pool")
    p.add_argument("--items", type=int, default=2000)
    p.add_argument("--min_len", type=int, default=8)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--max_items", type=int, default=32)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--timeout_ms", type=int, default=5)
    p.add_argument("--workers", type=str, default="1,2,4")
    p.add_argument("--threads", type=str, default="1,2")
    p.add_argument("--interop_threads", type=str, default="1")
    p.add_argument("--queue_maxsize", type=int, default=128)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--out", type=str, default="bench_results.json")
    args = p.parse_args()

    workers_list = [int(x) for x in args.workers.split(",") if x]
    threads_list = [int(x) for x in args.threads.split(",") if x]
    interop_list = [int(x) for x in args.interop_threads.split(",") if x]

    results: List[Dict[str, Any]] = []

    # Warmup one config to load torch/code paths
    _ = run_trial(
        n_items=min(200, args.items),
        min_len=args.min_len,
        max_len=args.max_len,
        max_items=args.max_items,
        max_tokens=args.max_tokens,
        timeout_ms=args.timeout_ms,
        num_workers=workers_list[0],
        torch_threads=threads_list[0],
        interop_threads=interop_list[0],
        queue_maxsize=args.queue_maxsize,
    )

    for w in workers_list:
        for t in threads_list:
            for it in interop_list:
                for r in range(args.repeats):
                    res = run_trial(
                        n_items=args.items,
                        min_len=args.min_len,
                        max_len=args.max_len,
                        max_items=args.max_items,
                        max_tokens=args.max_tokens,
                        timeout_ms=args.timeout_ms,
                        num_workers=w,
                        torch_threads=t,
                        interop_threads=it,
                        queue_maxsize=args.queue_maxsize,
                    )
                    res["repeat"] = r
                    results.append(res)
                    print(
                        f"workers={w} threads={t} interop={it} -> tps={res['throughput_items_per_s']:.1f} items/s, dur={res['duration_s']:.3f}s"
                    )

    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
