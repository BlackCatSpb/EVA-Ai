#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Heavy tokenization benchmark: baseline (transformers) vs CogniFlex UnifiedTextProcessor

Scenarios:
- Synthetic multi-page document
- Real file split into pages

Metrics:
- Latency per page (seq) and per batch
- Optional tokens/sec if tokens available

Usage examples (PowerShell):
  # Synthetic: 40 страниц x 1200 слов, батч 8
  python tools/bench_tokenization_heavy.py --pages 40 --words-per-page 1200 --batch-size 8 \
    --repeat 30 --model-path "c:\\Users\\black\\OneDrive\\Desktop\\CogniFlex\\cogniflex_models\\qwen25_1p5b_instruct"

  # From file: большой .txt
  python tools/bench_tokenization_heavy.py --file "C:\\path\\to\\big.txt" --chars-per-page 5000 \
    --batch-size 8 --repeat 30 --model-path "..."
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from typing import List, Dict, Any, Optional

# Ensure project root on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Try import CoreBrain
CoreBrain = None
try:
    from cogniflex.core.core_brain import CoreBrain as _CB  # type: ignore
    CoreBrain = _CB
except Exception as e:
    print(f"[bench-heavy] Warning: cannot import CoreBrain: {e}")

# Transformers baseline
AutoTokenizer = None
try:
    from transformers import AutoTokenizer  # type: ignore
except Exception as e:
    print(f"[bench-heavy] Warning: transformers not available: {e}")

LOREM = (
    "Искусственный интеллект использует разнообразные методы машинного обучения, "
    "включая нейронные сети, деревья решений и вероятностные графические модели. "
    "Прогресс в области вычислительных ресурсов и доступность больших данных ускорили развитие ИИ. "
    "Этические вопросы, прозрачность алгоритмов и интерпретируемость остаются приоритетными темами."
)


def make_synthetic_pages(pages: int, words_per_page: int) -> List[str]:
    # Build a base list of words
    base_words = re.findall(r"\w+", LOREM)
    if not base_words:
        base_words = ["слово"]
    out: List[str] = []
    for p in range(pages):
        words = []
        while len(words) < words_per_page:
            # vary content
            words.extend(base_words)
            words.append(str(p))
        out.append(" ".join(words[:words_per_page]))
    return out


def split_file_into_pages(path: str, chars_per_page: int) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()
    if not data:
        return []
    pages: List[str] = []
    for i in range(0, len(data), chars_per_page):
        pages.append(data[i:i+chars_per_page])
    return pages


def time_call(fn, *args, repeat: int = 1, **kwargs) -> Dict[str, Any]:
    latencies = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        latencies.append(time.perf_counter() - t0)
    return {
        "count": repeat,
        "p50": statistics.median(latencies),
        "mean": statistics.fmean(latencies),
        "p95": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
        "min": min(latencies),
        "max": max(latencies),
    }


def bench_baseline(model_path: str, pages: List[str], batch_size: int, repeat: int) -> Dict[str, Any]:
    if AutoTokenizer is None:
        return {"status": "skip", "reason": "transformers not installed"}
    try:
        tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except Exception as e:
        return {"status": "skip", "reason": f"cannot load tokenizer: {e}"}

    def one_by_one():
        for t in pages:
            tok(t, return_tensors=None)

    def in_batches():
        for i in range(0, len(pages), batch_size):
            batch = pages[i:i+batch_size]
            tok(batch, padding=False, truncation=False, return_tensors=None)

    res_seq = time_call(one_by_one, repeat=repeat)
    res_batch = time_call(in_batches, repeat=max(1, repeat // 5))
    return {"status": "ok", "seq": res_seq, "batch": res_batch, "pages": len(pages)}


def bench_cogniflex(pages: List[str], batch_size: int, repeat: int) -> Dict[str, Any]:
    if CoreBrain is None:
        return {"status": "skip", "reason": "CoreBrain import failed"}
    try:
        brain = CoreBrain()
        brain.initialize()
        ml = getattr(brain, "ml_unit", None)
        tp = getattr(ml, "text_processor", None) if ml else None
        if tp is None:
            return {"status": "skip", "reason": "UnifiedTextProcessor not available"}
    except Exception as e:
        return {"status": "skip", "reason": f"init failed: {e}"}

    has_async = hasattr(tp, "tokenize_async") and callable(getattr(tp, "tokenize_async"))

    def one_by_one():
        if has_async:
            for t in pages:
                tp.tokenize_async([t])
        else:
            for t in pages:
                tp.process_text(t)

    def in_batches():
        if has_async:
            for i in range(0, len(pages), batch_size):
                batch = pages[i:i+batch_size]
                tp.tokenize_async(batch)
        else:
            for i in range(0, len(pages), batch_size):
                batch = pages[i:i+batch_size]
                for t in batch:
                    tp.process_text(t)

    res_seq = time_call(one_by_one, repeat=repeat)
    res_batch = time_call(in_batches, repeat=max(1, repeat // 5))
    return {"status": "ok", "seq": res_seq, "batch": res_batch, "pages": len(pages)}


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--file", type=str, default=None, help="путь к большому .txt")
    ap.add_argument("--chars-per-page", type=int, default=5000)
    ap.add_argument("--pages", type=int, default=40)
    ap.add_argument("--words-per-page", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=30)
    ap.add_argument("--model-path", type=str, default=None)
    args = ap.parse_args()

    # Prepare pages
    pages: List[str]
    if args.file and os.path.isfile(args.file):
        pages = split_file_into_pages(args.file, args.chars_per_page)
        if not pages:
            print(json.dumps({"status": "error", "reason": "file empty or unreadable"}, ensure_ascii=False))
            return
        print(f"[bench-heavy] file pages: {len(pages)} (chars/page={args.chars_per_page})")
    else:
        pages = make_synthetic_pages(args.pages, args.words_per_page)
        print(f"[bench-heavy] synthetic pages: {len(pages)} (words/page={args.words_per_page})")

    model_path = args.model_path
    if not model_path:
        env_dir = os.environ.get("COGNIFLEX_MODELS_DIR")
        if env_dir:
            model_path = os.path.join(env_dir, os.environ.get("COGNIFLEX_QWEN_LOCAL_NAME", "qwen25_1p5b_instruct"))

    out: Dict[str, Any] = {
        "pages": len(pages),
        "batch_size": args.batch_size,
        "repeat": args.repeat,
        "mode": "file" if args.file else "synthetic"
    }

    # Baseline
    if model_path:
        out["baseline"] = bench_baseline(model_path, pages, args.batch_size, args.repeat)
    else:
        out["baseline"] = {"status": "skip", "reason": "no model-path"}

    # CogniFlex
    out["cogniflex"] = bench_cogniflex(pages, args.batch_size, args.repeat)

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
