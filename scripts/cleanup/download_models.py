#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скачивает указанные модели и токенизаторы в локальную директорию для офлайн-использования,
без загрузки весов в память PyTorch.

По умолчанию загружает:
- sberbank-ai/rugpt3small_based_on_gpt2
- Qwen/Qwen2.5-0.5B-Instruct

Пример:
  python scripts/download_models.py \
    --target "c:/Users/black/OneDrive/Desktop/CogniFlex/cogniflex/mlearning/cogniflex_models" \
    --models rugpt_small qwen_0_5b
"""
from __future__ import annotations
import os
import argparse
import shutil
import sys
from typing import Dict, List, Optional

# Включаем ускоренную загрузку, но если пакета hf_transfer нет — тихо отключаем
if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") is None:
    try:
        import hf_transfer  # type: ignore  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    except Exception:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

try:
    from huggingface_hub import snapshot_download
except Exception as e:
    print("[ERROR] Требуется пакет huggingface_hub: pip install huggingface_hub", file=sys.stderr)
    raise

MODEL_MAP: Dict[str, str] = {
    "rugpt_small": "sberbank-ai/rugpt3small_based_on_gpt2",
    "qwen_0_5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "rugpt3_large": "sberbank-ai/rugpt3large_based_on_gpt2",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_repo(
    repo_id: str,
    dest_dir: str,
    allow_fast: bool = True,
    allow_patterns: Optional[List[str]] = None,
) -> str:
    ensure_dir(dest_dir)
    # Скачиваем полную снапшот-копию репозитория в кэш HF и копируем в dest_dir
    print(f"[INFO] Загрузка репозитория {repo_id}...")
    try:
        local_cache = snapshot_download(
            repo_id=repo_id,
            local_files_only=False,
            allow_patterns=allow_patterns,
        )
    except ValueError as e:
        msg = str(e)
        if "HF_HUB_ENABLE_HF_TRANSFER=1" in msg or "hf_transfer" in msg or not allow_fast:
            print("[WARN] Быстрая загрузка недоступна, переключаемся на обычную…")
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
            local_cache = snapshot_download(
                repo_id=repo_id,
                local_files_only=False,
                allow_patterns=allow_patterns,
            )
        else:
            raise
    # Копируем (или синхронизируем) в целевую директорию
    print(f"[INFO] Копирование файлов в {dest_dir}...")
    # Очистим dest_dir перед копированием, чтобы избежать смешивания версий
    for item in os.listdir(dest_dir):
        p = os.path.join(dest_dir, item)
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)
    # Копирование дерева
    for root, dirs, files in os.walk(local_cache):
        rel = os.path.relpath(root, local_cache)
        tgt_root = os.path.join(dest_dir, rel) if rel != "." else dest_dir
        ensure_dir(tgt_root)
        for f in files:
            src = os.path.join(root, f)
            dst = os.path.join(tgt_root, f)
            shutil.copy2(src, dst)
    print(f"[OK] Загрузка завершена: {dest_dir}")
    return dest_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Каталог назначения для моделей")
    parser.add_argument("--no-fast-transfer", action="store_true", help="Отключить hf_transfer (обычная загрузка)")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rugpt_small", "qwen_0_5b"],
        choices=list(MODEL_MAP.keys()),
        help="Какие модели скачать",
    )
    parser.add_argument(
        "--only-torch",
        action="store_true",
        help="Скачивать только PyTorch/safetensors веса и файлы токенизатора (без Flax/TF)",
    )
    args = parser.parse_args()

    target_root = os.path.normpath(args.target)
    ensure_dir(target_root)

    if args.no_fast_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    # Разрешённые паттерны для только-PyTorch режима
    torch_patterns: Optional[List[str]] = None
    if args.only_torch:
        torch_patterns = [
            # веса
            "pytorch_model*.bin",
            "model*.safetensors",
            "*.bin",
            "*.safetensors",
            # конфиги
            "config.json",
            "generation_config.json",
            # токенизатор
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
        ]

    for key in args.models:
        repo = MODEL_MAP[key]
        subdir = key
        dest = os.path.join(target_root, subdir)
        ensure_dir(dest)
        download_repo(
            repo,
            dest,
            allow_fast=not args.no_fast_transfer,
            allow_patterns=torch_patterns,
        )

    print("\nВсе выбранные модели скачаны и готовы к офлайн-использованию.")


if __name__ == "__main__":
    main()
