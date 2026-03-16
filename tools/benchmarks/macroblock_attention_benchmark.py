import argparse
import sys
from pathlib import Path
import time

# Bootstrap import path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogniflex.memory.paged_store import (
    HierarchicalIndex,
    BatchAdapter,
    MacroBlockSpec,
)


def main():
    ap = argparse.ArgumentParser(description="MacroBlock E2E benchmark: extent-merged reads per layer")
    ap.add_argument("hindex_root", type=str, help="Directory containing superblocks.jsonl/extents.jsonl/subblocks.jsonl and data.bin")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--window", type=int, default=64, help="Trailing token window size per layer")
    ap.add_argument("--dtype", type=str, default="float16")
    ap.add_argument("--tensor_kind", type=str, default="kv")
    ap.add_argument("--pin", action="store_true", help="Use pinned memory in BatchAdapter")
    args = ap.parse_args()

    root = Path(args.hindex_root)
    hidx = HierarchicalIndex(root)

    # Minimal store shim using data.bin and absolute reads
    class _Store:
        def __init__(self, root: Path):
            import mmap
            self._f = (root / "data.bin").open("r+b")
            self._m = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_WRITE)
            self._lock = __import__("threading").RLock()
        def read_abs(self, off: int, length: int) -> bytes:
            with self._lock:
                return self._m[off:off+length]
    store = _Store(root)

    adapter = BatchAdapter(store, prefetch_depth=0, use_pinned=args.pin)

    # Prepare specs: per-layer macroblock for trailing token window
    specs = []
    for layer in range(args.layers):
        spec = MacroBlockSpec(
            tensor_kind=args.tensor_kind,
            layer_id=layer,
            head_range=(0, args.heads),
            token_range=(max(0, args.tokens - args.window), args.tokens),
            dtype=args.dtype,
            shape=(args.heads, min(args.window, args.tokens)),
        )
        specs.append(spec)

    # Run benchmark
    t0 = time.perf_counter()
    results = []
    for spec in specs:
        mb = adapter.assemble_macroblock(hidx, spec, device=None)
        results.append(mb)
    t1 = time.perf_counter()

    # Aggregate metrics
    total_read = sum(r.bytes_read for r in results)
    total_used = sum(r.bytes_used for r in results)
    total_ops = sum(r.io_ops for r in results)

    print("MacroBlock E2E benchmark")
    print(f"layers={args.layers}, heads={args.heads}, tokens={args.tokens}, window={args.window}")
    print(f"total_io_ops={total_ops}")
    print(f"bytes_read={total_read}, bytes_used={total_used}, overfetch={total_read/total_used if total_used else 0:.2f}x")
    print(f"plan_ms_avg={sum(m.timings_ms['plan_ms'] for m in results)/len(results):.3f}")
    print(f"assemble_ms_avg={sum(m.timings_ms['assemble_ms'] for m in results)/len(results):.3f}")
    print(f"h2d_ms_avg={sum(m.timings_ms['h2d_ms'] for m in results)/len(results):.3f}")
    print(f"end_to_end_ms={(t1 - t0)*1e3:.3f}")


if __name__ == "__main__":
    main()
