import time
from dataclasses import dataclass
from typing import List, Tuple

from pathlib import Path
import sys

# Ensure project root is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogniflex.memory.paged_store import (
    HierarchicalIndex,
    SubBlockMeta,
    NeedSpec,
    plan_from_hindex,
)


@dataclass
class BenchCfg:
    layers: int = 8
    heads: int = 16
    tokens: int = 2048
    sub_block_tokens: int = 128  # size of token range per sub-block
    sub_block_heads: int = 4     # heads per sub-block
    tensor_kind: str = "kv"
    dtype: str = "float16"
    sub_block_bytes: int = 256 * 1024  # 256 KiB per sub-block


def build_mock_hindex(cfg: BenchCfg) -> HierarchicalIndex:
    # Create an empty index object and inject subblocks (no file IO required)
    hidx = HierarchicalIndex(Path("./nonexistent"))
    hidx.superblocks.clear()
    hidx.extents.clear()
    hidx.subblocks.clear()
    hidx._by_tensor_layer.clear()

    sub_id = 0
    file_off = 0
    for layer in range(cfg.layers):
        for head0 in range(0, cfg.heads, cfg.sub_block_heads):
            head1 = min(cfg.heads, head0 + cfg.sub_block_heads)
            for tok0 in range(0, cfg.tokens, cfg.sub_block_tokens):
                tok1 = min(cfg.tokens, tok0 + cfg.sub_block_tokens)
                sb = SubBlockMeta(
                    sub_block_id=sub_id,
                    superblock_id=0,
                    extent_id=layer,  # simple grouping by layer
                    file_offset=file_off,
                    length=cfg.sub_block_bytes,
                    tensor_kind=cfg.tensor_kind,
                    layer_id=layer,
                    head_range=(head0, head1),
                    token_range=(tok0, tok1),
                    dtype=cfg.dtype,
                    shape=(head1 - head0, tok1 - tok0),
                )
                hidx.subblocks.append(sb)
                hidx._by_tensor_layer.setdefault((cfg.tensor_kind, layer), []).append(sub_id)
                sub_id += 1
                file_off += cfg.sub_block_bytes
    return hidx


def build_needs(cfg: BenchCfg, batch_tokens: int = 64) -> List[NeedSpec]:
    # Simulate needs for the next-token step over last batch_tokens per head, per layer
    needs: List[NeedSpec] = []
    tok0 = max(0, cfg.tokens - batch_tokens)
    tok1 = cfg.tokens
    for layer in range(cfg.layers):
        needs.append(
            NeedSpec(
                tensor_kind=cfg.tensor_kind,
                layer_id=layer,
                head_range=(0, cfg.heads),
                token_range=(tok0, tok1),
                dtype=cfg.dtype,
                shape=(cfg.heads, tok1 - tok0),
            )
        )
    return needs


def naive_scatter_read_count(cfg: BenchCfg, needs: List[NeedSpec]) -> int:
    # Count how many sub-blocks would be touched without merging
    hidx = build_mock_hindex(cfg)
    subs = hidx.resolve(needs)
    return len(subs)


def extent_merge_read_count(hidx: HierarchicalIndex, needs: List[NeedSpec]) -> int:
    # Use a minimal stub store that triggers the hierarchical fallback path
    class _StubStore:
        blocks = []  # no entries; GatherPlanner will fail and fallback will be used

    plan = plan_from_hindex(_StubStore(), hidx, needs)
    return len(plan.reads)


def run_once(cfg: BenchCfg):
    needs = build_needs(cfg)
    hidx = build_mock_hindex(cfg)

    t0 = time.perf_counter()
    naive = naive_scatter_read_count(cfg, needs)
    t1 = time.perf_counter()

    merged = extent_merge_read_count(hidx, needs)
    t2 = time.perf_counter()

    print("Config:", cfg)
    print(f"Naive sub-block reads: {naive} (planned in {(t1 - t0)*1e3:.2f} ms)")
    print(f"Merged reads (contiguous): {merged} (planned in {(t2 - t1)*1e3:.2f} ms)")
    if naive > 0:
        print(f"Merge factor: {naive/merged:.2f}x fewer I/O ops")


if __name__ == "__main__":
    cfg = BenchCfg()
    run_once(cfg)
