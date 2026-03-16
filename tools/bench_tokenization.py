#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bench: tokenization baseline (transformers) vs CogniFlex UnifiedTextProcessor

Usage (Windows PowerShell):
  python tools/bench_tokenization.py --repeat 200 --batch-size 16 \
      --model-path "c:\\Users\\black\\OneDrive\\Desktop\\CogniFlex\\cogniflex_models\\qwen25_1p5b_instruct"

Notes:
- Baseline uses transformers.AutoTokenizer.from_pretrained(model_path)
- CogniFlex path uses CoreBrain -> MLUnit -> UnifiedTextProcessor
- If UnifiedTextProcessor is unavailable, CogniFlex path is skipped
- No network access required if model-path points to a local snapshot
"""
from __future__ import annotations
import argparse
import json
import os
import random
import statistics
import sys
import time
from typing import List, Dict, Any

# Ensure project root on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Try import CoreBrain
corebrain = None
try:
    from cogniflex.core.core_brain import CoreBrain  # type: ignore
    corebrain = CoreBrain
except Exception as e:
    CoreBrain = None  # type: ignore
    print(f"[bench] Warning: cannot import CoreBrain: {e}")

# Transformers baseline
AutoTokenizer = None
try:
    from transformers import AutoTokenizer  # type: ignore
except Exception as e:
    print(f"[bench] Warning: transformers not available: {e}")


def make_text_corpus(n: int) -> List[str]:
    base = (
        "Искусственный интеллект — это область информатики, занимающаяся созданием систем, "
        "способных выполнять задачи, требующие человеческого интеллекта, такие как обучение, "
        "распознавание образов и принятие решений."
    )
    # Vary lengths
    corpus = []
    for i in range(n):
        k = 1 + (i % 5)
        corpus.append((base + " ") * k)
    random.shuffle(corpus)
    return corpus


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


def bench_baseline(model_path: str, texts: List[str], batch_size: int, repeat: int) -> Dict[str, Any]:
    if AutoTokenizer is None:
        return {"status": "skip", "reason": "transformers not installed"}
    try:
        tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except Exception as e:
        return {"status": "skip", "reason": f"cannot load tokenizer: {e}"}

    def one_by_one():
        for t in texts:
            tok(t, return_tensors=None)

    def in_batches():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tok(batch, padding=False, truncation=False, return_tensors=None)

    res_seq = time_call(one_by_one, repeat=repeat)
    res_batch = time_call(in_batches, repeat=max(1, repeat // 5))
    return {"status": "ok", "seq": res_seq, "batch": res_batch}


def bench_cogniflex(texts: List[str], batch_size: int, repeat: int) -> Dict[str, Any]:
    if corebrain is None:
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

    # Prefer tokenize_async if available, else fallback to process_text per item
    has_async = hasattr(tp, "tokenize_async") and callable(getattr(tp, "tokenize_async"))

    def one_by_one():
        if has_async:
            # use async path with single-item list to keep flow similar
            for t in texts:
                tp.tokenize_async([t])
        else:
            for t in texts:
                tp.process_text(t)

    def in_batches():
        if has_async:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                tp.tokenize_async(batch)
        else:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                for t in batch:
                    tp.process_text(t)

    res_seq = time_call(one_by_one, repeat=repeat)
    res_batch = time_call(in_batches, repeat=max(1, repeat // 5))
    return {"status": "ok", "seq": res_seq, "batch": res_batch}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--texts", type=int, default=64, help="кол-во текстов в корпусе")
    parser.add_argument("--model-path", type=str, default=None, help="путь к локальному снапшоту модели для baseline")
    args = parser.parse_args()

    model_path = args.model_path
    if not model_path:
        env_dir = os.environ.get("COGNIFLEX_MODELS_DIR")
        if env_dir:
            model_path = os.path.join(env_dir, os.environ.get("COGNIFLEX_QWEN_LOCAL_NAME", "qwen25_1p5b_instruct"))

    texts = make_text_corpus(args.texts)

    print("[bench] starting tokenization benchmark...")
    out: Dict[str, Any] = {"repeat": args.repeat, "batch_size": args.batch_size, "texts": args.texts}

    # Baseline
    if model_path:
        print(f"[bench] baseline model path: {model_path}")
        out["baseline"] = bench_baseline(model_path, texts, args.batch_size, args.repeat)
    else:
        out["baseline"] = {"status": "skip", "reason": "no model-path provided"}

    # CogniFlex
    out["cogniflex"] = bench_cogniflex(texts, args.batch_size, args.repeat)

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
