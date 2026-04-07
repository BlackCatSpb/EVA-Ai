from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional
import json

from .paged_store import SuperblockMeta, ExtentMeta, SubBlockMeta, HierarchicalIndex


@dataclass
class MacroArchiveState:
    version: int = 1
    total_bytes: int = 0
    superblocks: int = 0
    extents: int = 0
    subblocks: int = 0


class MacroArchiveWriter:
    """SSD-first archive builder for macroblocks (extent-coalesced layout).

    Writes:
      - superblocks.jsonl
      - extents.jsonl
      - subblocks.jsonl
      - data.bin (binary payload)
      - state.json (optional telemetry)
    """

    def __init__(self, out_dir: Path, align: int = 1 << 20):
        self.out_dir = Path(out_dir)
        self.align = align
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._superblocks: List[SuperblockMeta] = []
        self._extents: List[ExtentMeta] = []
        self._subblocks: List[SubBlockMeta] = []
        self._cur_off = 0

    def _align_up(self, x: int) -> int:
        a = self.align
        return (x + a - 1) // a * a

    def begin_superblock(self, length: int, alignment: Optional[int] = None) -> int:
        sb_id = len(self._superblocks)
        off = self._align_up(self._cur_off)
        self._superblocks.append(
            SuperblockMeta(superblock_id=sb_id, file_offset=off, length=length, alignment=alignment or self.align)
        )
        self._cur_off = off + length
        return sb_id

    def add_extent(self, superblock_id: int, length: int, page_size: Optional[int] = None) -> Tuple[int, int]:
        extent_id = len(self._extents)
        off = self._align_up(self._cur_off)
        self._extents.append(
            ExtentMeta(extent_id=extent_id, superblock_id=superblock_id, file_offset=off, length=length, page_size=page_size or self.align)
        )
        self._cur_off = off + length
        return extent_id, off

    def add_subblock(
        self,
        extent_id: int,
        superblock_id: int,
        file_offset: int,
        length: int,
        tensor_kind: str,
        layer_id: int,
        head_range: Tuple[int, int],
        token_range: Tuple[int, int],
        dtype: str,
        shape: Tuple[int, int],
    ) -> int:
        sbid = len(self._subblocks)
        self._subblocks.append(
            SubBlockMeta(
                sub_block_id=sbid,
                superblock_id=superblock_id,
                extent_id=extent_id,
                file_offset=file_offset,
                length=length,
                tensor_kind=tensor_kind,
                layer_id=layer_id,
                head_range=head_range,
                token_range=token_range,
                dtype=dtype,
                shape=shape,
            )
        )
        return sbid

    def finalize(self) -> MacroArchiveState:
        # Write JSONLs
        def _dump_jsonl(path: Path, rows):
            with path.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r.__dict__) + "\n")

        _dump_jsonl(self.out_dir / "superblocks.jsonl", self._superblocks)
        _dump_jsonl(self.out_dir / "extents.jsonl", self._extents)
        _dump_jsonl(self.out_dir / "subblocks.jsonl", self._subblocks)
        # Resize data.bin if needed
        data_path = self.out_dir / "data.bin"
        total = self._cur_off
        with data_path.open("wb") as f:
            f.truncate(total)
        state = MacroArchiveState(
            total_bytes=total,
            superblocks=len(self._superblocks),
            extents=len(self._extents),
            subblocks=len(self._subblocks),
        )
        with (self.out_dir / "state.json").open("w", encoding="utf-8") as f:
            json.dump(state.__dict__, f)
        return state


class MacroArchive:
    """Loader facade over HierarchicalIndex for SSD-first macroblock archive.

    Provides convenience to open existing archive directory and query via HierarchicalIndex.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.hindex = HierarchicalIndex(self.root)
        self.data_path = self.root / "data.bin"
        if not self.data_path.exists():
            raise FileNotFoundError(f"Missing data.bin in {self.root}")

    def open_store(self):
        # Minimal absolute-read store
        import mmap
        import threading
        f = self.data_path.open("r+b")
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)
        lock = threading.RLock()

        class _Store:
            def read_abs(self, off: int, length: int) -> bytes:
                with lock:
                    return m[off:off+length]
        return _Store()
