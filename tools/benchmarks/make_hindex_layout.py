import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import sys

# Ensure project root is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogniflex.memory.paged_store import SuperblockMeta, ExtentMeta, SubBlockMeta  # type: ignore


@dataclass
class Cfg:
    out_dir: Path
    layers: int = 8
    heads: int = 16
    tokens: int = 2048
    sub_block_tokens: int = 128
    sub_block_heads: int = 4
    tensor_kind: str = "kv"
    dtype: str = "float16"
    extent_mib: int = 16
    superblock_mib: int = 128
    align: int = 1 * 1024 * 1024  # 1 MiB
    sub_block_bytes: int = 256 * 1024  # 256 KiB per sub-block payload


def align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            if hasattr(r, "__dict__"):
                data = r.__dict__
            else:
                data = r
            f.write(json.dumps(data) + "\n")


def build_layout(cfg: Cfg):
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)

    superblocks = []
    extents = []
    subblocks = []

    # Single superblock for MVP
    sb_id = 0
    sb_off = 0
    sb_len = cfg.superblock_mib * 1024 * 1024
    superblocks.append(SuperblockMeta(superblock_id=sb_id, file_offset=sb_off, length=sb_len, alignment=cfg.align))

    # Place one extent per layer; pack sub-blocks optimized for last-token access:
    # order head groups first across the last token chunk, then previous chunks, etc.
    file_off = align_up(sb_off, cfg.align)
    extent_size = cfg.extent_mib * 1024 * 1024
    sub_id = 0
    for layer in range(cfg.layers):
        ex_id = layer
        ex_off = align_up(file_off, cfg.align)
        ex_end = ex_off

        # Determine token chunks in order: prefer last chunk first
        tok_chunks = list(range(0, cfg.tokens, cfg.sub_block_tokens))
        if tok_chunks and tok_chunks[-1] != cfg.tokens - (cfg.tokens % cfg.sub_block_tokens or cfg.sub_block_tokens):
            pass
        tok_chunks.sort()
        # reorder to: last, last-1, ..., first (so last tokens grouped contiguously)
        tok_chunks = list(reversed(tok_chunks))

        # Head groups
        head_groups = list(range(0, cfg.heads, cfg.sub_block_heads))

        for tok0 in tok_chunks:
            tok1 = min(cfg.tokens, tok0 + cfg.sub_block_tokens)
            for h0 in head_groups:
                h1 = min(cfg.heads, h0 + cfg.sub_block_heads)
                sb = SubBlockMeta(
                    sub_block_id=sub_id,
                    superblock_id=sb_id,
                    extent_id=ex_id,
                    file_offset=ex_end,
                    length=cfg.sub_block_bytes,
                    tensor_kind=cfg.tensor_kind,
                    layer_id=layer,
                    head_range=(h0, h1),
                    token_range=(tok0, tok1),
                    dtype=cfg.dtype,
                    shape=(h1 - h0, tok1 - tok0),
                )
                subblocks.append(sb)
                ex_end += cfg.sub_block_bytes
                sub_id += 1

        ex_len = ex_end - ex_off
        # ensure extent is at least extent_size for alignment; not strictly required
        ex_len = max(ex_len, extent_size)
        extents.append(ExtentMeta(extent_id=ex_id, superblock_id=sb_id, file_offset=ex_off, length=ex_len, page_size=cfg.align))
        file_off = ex_off + ex_len

    # Write JSONL index files
    write_jsonl(out / "superblocks.jsonl", superblocks)
    write_jsonl(out / "extents.jsonl", extents)
    write_jsonl(out / "subblocks.jsonl", subblocks)

    # Create data.bin (zero-filled) sized to last file_off
    data_path = out / "data.bin"
    total_size = file_off
    with data_path.open("wb") as f:
        f.truncate(total_size)

    print(f"Written hierarchical index to: {out}")
    print(f"superblocks: {len(superblocks)}, extents: {len(extents)}, subblocks: {len(subblocks)}")
    print(f"data.bin size: {total_size / (1024*1024):.2f} MiB")


def main():
    ap = argparse.ArgumentParser(description="Build hierarchical index layout for KV with last-token-optimized packing")
    ap.add_argument("out_dir", type=str, help="Output directory for hindex files")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--sub_block_tokens", type=int, default=128)
    ap.add_argument("--sub_block_heads", type=int, default=4)
    ap.add_argument("--dtype", type=str, default="float16")
    ap.add_argument("--extent_mib", type=int, default=16)
    ap.add_argument("--superblock_mib", type=int, default=128)
    ap.add_argument("--align", type=int, default=1024*1024)
    ap.add_argument("--sub_block_bytes", type=int, default=256*1024)
    args = ap.parse_args()

    cfg = Cfg(
        out_dir=Path(args.out_dir),
        layers=args.layers,
        heads=args.heads,
        tokens=args.tokens,
        sub_block_tokens=args.sub_block_tokens,
        sub_block_heads=args.sub_block_heads,
        dtype=args.dtype,
        extent_mib=args.extent_mib,
        superblock_mib=args.superblock_mib,
        align=args.align,
        sub_block_bytes=args.sub_block_bytes,
    )
    build_layout(cfg)


if __name__ == "__main__":
    main()
