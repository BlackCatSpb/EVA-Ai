#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI: Переупаковка модели в фрактальную структуру с извлечением графа знаний.

Пример использования:

  python scripts/repack_to_fractal.py \
    --model-path "hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct" \
    --output-path "cogniflex_cache/models/fractal_qwen" \
    --levels 5 \
    --block-size 64 \
    --device cpu

"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Разрешаем запуск скрипта из корня проекта
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogniflex.mlearning.storage.fractal_store import repack_model_to_fractal  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repack HF model to Fractal store")
    p.add_argument("--model-path", required=True, help="Путь к локальной директории модели HF")
    p.add_argument("--output-path", required=True, help="Папка, куда сохранить фрактальную структуру")
    p.add_argument("--levels", type=int, default=5, help="Количество уровней фрактала (>=1)")
    p.add_argument("--block-size", type=int, default=64, help="Базовый размер блока (рекомендуется 64)")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Устройство для загрузки/квантизации")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Уровень логирования")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    model_path = args.model_path
    output_path = args.output_path

    ok = repack_model_to_fractal(
        model_path=model_path,
        output_path=output_path,
        fractal_levels=max(1, int(args.levels)),
        block_size=max(1, int(args.block_size)),
        device=args.device,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
