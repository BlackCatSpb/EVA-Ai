#!/usr/bin/env python3
"""
NLTK Preload Helper

Usage:
  python tools/nltk_preload.py --dir ./nltk_data --packages basic
  python tools/nltk_preload.py --dir ./nltk_data --packages all
  python tools/nltk_preload.py --dir ./nltk_data --list

By default, downloads into ./nltk_data at the project root. You can override via --dir or NLTK_DATA env var.

Exit codes:
  0 on success, 1 on failure.
"""
import argparse
import os
import sys
from typing import List

try:
    import nltk
except Exception as e:
    print("[nltk_preload] ERROR: NLTK is not installed. Please `pip install nltk`.", file=sys.stderr)
    sys.exit(1)


BASIC_PACKAGES: List[str] = [
    # Tokenizers & sentence models
    "punkt",
    # WordNet and multilingual mappings
    "wordnet",
    "omw-1.4",
    # Stopwords
    "stopwords",
    # Taggers
    "averaged_perceptron_tagger",
    # Sentiment
    "vader_lexicon",
]

EXTENDED_PACKAGES: List[str] = BASIC_PACKAGES + [
    # Chunkers / NE
    "maxent_ne_chunker",
    "words",
    # Corpora used commonly in examples/tests
    "brown",
    "conll2000",
]

# Some environments (NLTK>=3.8) split punkt models; include if available
OPTIONAL_PACKAGES: List[str] = [
    "punkt_tab",
    "averaged_perceptron_tagger_eng",
]


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def _set_nltk_path(path: str) -> None:
    if path not in nltk.data.path:
        nltk.data.path.insert(0, path)


def _download(packages: List[str]) -> int:
    failures = 0
    for pkg in packages:
        try:
            print(f"[nltk_preload] Downloading: {pkg} ...")
            nltk.download(pkg, quiet=False)
        except Exception as e:
            failures += 1
            print(f"[nltk_preload] WARNING: Failed to download '{pkg}': {e}", file=sys.stderr)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Preload NLTK corpora/models for CI and local dev.")
    parser.add_argument("--dir", dest="target_dir", default=os.environ.get("NLTK_DATA", "./nltk_data"),
                        help="Directory to store downloaded NLTK data (default: ./nltk_data or $NLTK_DATA)")
    parser.add_argument("--packages", choices=["basic", "extended", "all"], default="basic",
                        help="Which package set to download")
    parser.add_argument("--list", action="store_true", help="List package sets and exit")
    args = parser.parse_args()

    if args.list:
        print("Package sets:")
        print("  basic:")
        for p in BASIC_PACKAGES:
            print(f"    - {p}")
        print("  extended:")
        for p in set(EXTENDED_PACKAGES) - set(BASIC_PACKAGES):
            print(f"    - {p}")
        print("  optional (auto-included for 'all' if available):")
        for p in OPTIONAL_PACKAGES:
            print(f"    - {p}")
        return 0

    target_dir = _ensure_dir(args.target_dir)
    _set_nltk_path(target_dir)
    os.environ["NLTK_DATA"] = target_dir
    print(f"[nltk_preload] Using NLTK_DATA at: {target_dir}")

    if args.packages == "basic":
        packages = BASIC_PACKAGES
    elif args.packages == "extended":
        packages = EXTENDED_PACKAGES
    else:
        packages = sorted(set(EXTENDED_PACKAGES + OPTIONAL_PACKAGES))

    failures = _download(packages)

    if failures:
        print(f"[nltk_preload] Completed with {failures} failure(s). Check logs above.", file=sys.stderr)
        return 1

    print("[nltk_preload] All requested packages are downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
