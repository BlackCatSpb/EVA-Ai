import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path to import 'cogniflex' when running as a script from tools/
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from cogniflex.memory.macro_integration import MacroPrefetchConfig, MacroblockPrefetcher


def parse_args():
    ap = argparse.ArgumentParser(description="Macroblock prefetch + HotSet probe")
    ap.add_argument("--root", type=Path, default=Path("cogniflex_cache/hindex_demo"))
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--kind", type=str, default="kv")
    ap.add_argument("--hotset", action="store_true", default=True)
    ap.add_argument("--vram_frac", type=float, default=0.65)
    ap.add_argument("--lazy", action="store_true", default=True)
    ap.add_argument("--io_bps", type=int, default=128 * 1024 * 1024)
    ap.add_argument("--max_pending", type=int, default=8)
    ap.add_argument("--ram_cache", type=int, default=256 * 1024 * 1024)
    ap.add_argument("--ticks", type=int, default=12)
    ap.add_argument("--tick_ms", type=int, default=250)
    ap.add_argument("--log", type=Path, default=Path("tools/logs/macroblock_probe.log"))
    return ap.parse_args()


def ensure_logdir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    ensure_logdir(args.log)

    if not args.root.exists():
        print(json.dumps({"level": "error", "msg": f"root not found: {args.root}"}))
        sys.exit(1)

    cfg = MacroPrefetchConfig(
        root=args.root,
        window_tokens=args.window,
        max_layers_per_batch=args.layers,
        tensor_kind=args.kind,
        device=args.device,
        use_pinned=True,
        hotset_enabled=args.hotset and torch.cuda.is_available(),
        hotset_target_vram_frac=args.vram_frac,
        lazy_enabled=args.lazy,
        io_budget_bytes_per_sec=args.io_bps,
        max_pending=args.max_pending,
        ram_cache_bytes=args.ram_cache,
    )
    prefetcher = MacroblockPrefetcher(cfg)

    with args.log.open("w", encoding="utf-8") as lf:
        for i in range(args.ticks):
            out = prefetcher.build(items=[{"i": i}])
            s = prefetcher.stats()
            # Summarize emitted tensors
            tensors = {k: {"device": str(v.device), "dtype": str(v.dtype), "shape": list(v.shape)} for k, v in out.items()}
            line = {
                "tick": i,
                "emitted": list(tensors.keys()),
                "tensors": tensors,
                "stats": s,
            }
            j = json.dumps(line)
            print(j)
            lf.write(j + "\n")
            lf.flush()
            time.sleep(args.tick_ms / 1000.0)

        # Final HotSet stats snapshot
        final = {"final_stats": prefetcher.stats()}
        j = json.dumps(final)
        print(j)
        lf.write(j + "\n")


if __name__ == "__main__":
    main()
