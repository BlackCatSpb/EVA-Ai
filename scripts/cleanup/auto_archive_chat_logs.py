#!/usr/bin/env python3
"""
Auto-archive chat logs from docs/chat_logs into a dated ZIP.

Default behavior:
- Source dir: docs/chat_logs/
- Destination ZIP: docs/chat_logs/chat_logs_YYYY-MM-DD_HHMMSS.zip
- Includes: all *.md files (recursively) under source
- Does NOT delete source files by default (safety). Use --delete-after to remove files after successful zip write.

Usage examples:
  python scripts/auto_archive_chat_logs.py
  python scripts/auto_archive_chat_logs.py --source docs/chat_logs --dest docs/chat_logs --delete-after
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import zipfile


def archive_chat_logs(source: Path, dest_dir: Path, delete_after: bool = False) -> Path:
    source = source.resolve()
    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    zip_path = dest_dir / f"chat_logs_{ts}.zip"

    md_files = sorted(source.rglob("*.md"))
    if not md_files:
        print(f"No .md files found under {source}")
        return zip_path

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in md_files:
            # Store relative paths inside the ZIP
            arcname = f.relative_to(source)
            zf.write(f, arcname)

    print(f"Archived {len(md_files)} files to {zip_path}")

    if delete_after:
        removed = 0
        for f in md_files:
            try:
                f.unlink(missing_ok=True)
                removed += 1
            except Exception as e:
                print(f"Failed to delete {f}: {e}")
        print(f"Deleted {removed} source files after archiving")

    return zip_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auto-archive chat logs into a dated ZIP")
    parser.add_argument("--source", type=Path, default=Path("docs/chat_logs"), help="Source directory with .md logs")
    parser.add_argument("--dest", type=Path, default=Path("docs/chat_logs"), help="Destination directory for ZIP")
    parser.add_argument("--delete-after", action="store_true", help="Delete source .md files after archiving")
    args = parser.parse_args(argv)

    try:
        zip_path = archive_chat_logs(args.source, args.dest, delete_after=args.delete_after)
        print(f"ZIP ready at: {zip_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
