#!/usr/bin/env python3
"""Скрипт для экспорта модели ruGPT-small в фрактальное хранилище."""

import sys
import os

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cogniflex.mlearning.storage.fractal_store import export_hf_model_to_fractal

def main():
    print("Экспорт модели ruGPT-small в фрактальное хранилище...")

    result = export_hf_model_to_fractal(
        hf_model_dir_or_id="sberbank-ai/rugpt3small_based_on_gpt2",
        output_path="./test_cache/ml_unit/fractal_storage/models/text-generation",
        model_id="rugpt3small",
        device="cpu",
        local_files_only=True
    )

    if result:
        print("✅ Модель успешно экспортирована в фрактальное хранилище")
    else:
        print("❌ Ошибка экспорта модели")
        sys.exit(1)

if __name__ == "__main__":
    main()
