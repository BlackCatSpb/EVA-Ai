#!/usr/bin/env python3
"""
Очистка и перезагрузка ruGPT-3 Medium с изоляцией существующей модели
"""

import os
import sys
import shutil
from pathlib import Path
import torch

# Добавляем путь к CogniFlex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clean_hf_cache():
    """Очищает кэш Hugging Face от проблемных загрузок"""
    print("🧹 Очистка кэша Hugging Face...")

    cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'

    if not cache_dir.exists():
        print("❌ Директория кэша не найдена")
        return False

    # Найдем модели ruGPT-3
    rugpt_models = []
    for model_dir in cache_dir.iterdir():
        if model_dir.is_dir() and 'rugpt3' in model_dir.name:
            size_mb = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / (1024*1024)
            rugpt_models.append((model_dir, size_mb))
            print(f"📁 {model_dir.name}: {size_mb:.1f} MB")

    # Проверим на поврежденные загрузки
    for model_dir, size_mb in rugpt_models:
        if 'rugpt3medium' in model_dir.name:
            # Проверим размер - должен быть около 1700 MB, но может быть меньше если недозагружен
            expected_size_mb = 1700
            if size_mb < expected_size_mb * 0.5:  # Если меньше половины ожидаемого размера
                print(f"⚠️ Модель {model_dir.name} кажется недозагруженной ({size_mb:.1f} MB < {expected_size_mb} MB)")
                print(f"🗑️ Удаление недозагруженной модели...")
                try:
                    shutil.rmtree(model_dir)
                    print(f"✅ Удалена: {model_dir.name}")
                except Exception as e:
                    print(f"❌ Ошибка удаления: {e}")
            else:
                print(f"✅ Модель {model_dir.name} выглядит полной ({size_mb:.1f} MB)")

    # Создадим резервную копию ruGPT-3 Small
    for model_dir, size_mb in rugpt_models:
        if 'rugpt3small' in model_dir.name:
            backup_dir = cache_dir / "backup_rugpt3small"
            print(f"💾 Создание резервной копии ruGPT-3 Small...")
            try:
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                shutil.copytree(model_dir, backup_dir)
                print(f"✅ Резервная копия создана: {backup_dir}")
            except Exception as e:
                print(f"❌ Ошибка создания резервной копии: {e}")

    print("✅ Очистка кэша завершена")
    return True

def isolate_fractal_models():
    """Изолирует существующие модели в фрактальном хранилище"""
    print("🔒 Изоляция существующих моделей в фрактальном хранилище...")

    fractal_base = Path("./fractal_storage")
    if not fractal_base.exists():
        print("ℹ️ Фрактальное хранилище не найдено, пропускаем")
        return True

    # Найдем существующие модели
    existing_models = []
    for model_dir in fractal_base.iterdir():
        if model_dir.is_dir() and any(model_dir.name.startswith(prefix) for prefix in ['rugpt3', 'gpt2', 'fractal']):
            existing_models.append(model_dir)
            print(f"📁 Найдена модель: {model_dir.name}")

    # Переместим их в архивную директорию
    archive_dir = fractal_base / "archive"
    archive_dir.mkdir(exist_ok=True)

    for model_dir in existing_models:
        new_path = archive_dir / model_dir.name
        print(f"📦 Перемещение {model_dir.name} в архив...")
        try:
            if new_path.exists():
                shutil.rmtree(new_path)
            shutil.move(str(model_dir), str(new_path))
            print(f"✅ Перемещено: {model_dir.name} -> archive/{model_dir.name}")
        except Exception as e:
            print(f"❌ Ошибка перемещения {model_dir.name}: {e}")

    print("✅ Изоляция моделей завершена")
    return True

def reload_rugpt3_medium():
    """Перезагружает ruGPT-3 Medium с чистой загрузкой"""
    print("🚀 Перезагрузка ruGPT-3 Medium...")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        model_name = "sberbank-ai/rugpt3medium_based_on_gpt2"

        # Загрузка токенизатора
        print("📝 Загрузка токенизатора...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
            force_download=True  # Принудительная перезагрузка
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"✅ Токенизатор загружен: {tokenizer.vocab_size} токенов")

        # Загрузка модели
        print("🧠 Загрузка модели...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="cpu",  # Загружаем на CPU для стабильности
            force_download=True  # Принудительная перезагрузка
        )

        # Информация о модели
        total_params = sum(p.numel() for p in model.parameters())
        print(f"🔢 Параметры: {total_params:,}")
        print(f"💾 Размер в памяти: {total_params * 4 / (1024**2):.1f} MB")

        # Сохранение локально
        save_path = "./models/rugpt3medium_clean"
        os.makedirs(save_path, exist_ok=True)

        print(f"💾 Сохранение модели локально...")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        print(f"✅ Модель сохранена в {save_path}")

        # Экспорт во фрактальное хранилище
        print(f"💾 Экспорт во фрактальное хранилище...")

        from cogniflex.mlearning.fractal_rugpt3_manager import FractalRuGPT3Manager

        fractal_manager = FractalRuGPT3Manager(
            brain=None,
            model_name="rugpt3medium",
            storage_path="./fractal_storage"
        )

        # Инициализируем менеджер с загруженной моделью
        fractal_manager.model = model
        fractal_manager.tokenizer = tokenizer
        fractal_manager.initialized = True

        export_path = "./fractal_storage/rugpt3medium"
        if fractal_manager.export_model(export_path):
            print(f"✅ Модель экспортирована во фрактальное хранилище: {export_path}")

            # Проверка размера экспорта
            total_size = sum(f.stat().st_size for f in Path(export_path).rglob('*') if f.is_file())
            print(f"💾 Размер фрактального экспорта: {total_size / (1024**2):.1f} MB")

            # Создание метаданных
            import json
            metadata = {
                "model_name": model_name,
                "description": "ruGPT-3 Medium (чистая перезагрузка)",
                "total_params": total_params,
                "vocab_size": tokenizer.vocab_size,
                "device": device,
                "export_path": export_path,
                "local_save_path": save_path,
                "timestamp": __import__('time').time()
            }

            with open(f"{export_path}/export_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            print(f"📋 Метаданные сохранены")

        else:
            print(f"❌ Не удалось экспортировать модель")

        # Очистка
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✅ Перезагрузка ruGPT-3 Medium завершена")
        return True

    except Exception as e:
        print(f"❌ Ошибка перезагрузки: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_tokenization():
    """Тестирует GPU токенизацию с перезагруженной моделью"""
    print("🚀 Тестирование GPU токенизации...")

    if not torch.cuda.is_available():
        print("❌ GPU не доступен")
        return False

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_path = "./models/rugpt3medium_clean"
        if not Path(model_path).exists():
            print("❌ Локальная модель не найдена")
            return False

        print(f"📦 Загрузка модели с {model_path} на GPU...")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        memory_used = torch.cuda.memory_allocated() / (1024**3)

        print(f"✅ Модель загружена на GPU")
        print(f"💾 Использование GPU: {memory_used:.1f}/{gpu_memory:.1f} GB")

        # Тест токенизации
        test_texts = [
            "Привет! Как дела?",
            "Что такое искусственный интеллект?",
            "Расскажи о фрактальном хранилище",
            "GPU токенизация в действии"
        ]

        print("🧪 Тест токенизации:")

        for text in test_texts:
            # CPU токенизация
            import time
            start_time = time.time()
            inputs_cpu = tokenizer.encode(text, return_tensors="pt")
            cpu_time = time.time() - start_time

            # GPU токенизация
            start_time = time.time()
            inputs_gpu = inputs_cpu.to("cuda")
            gpu_time = time.time() - start_time

            print(f"📝 '{text}' -> {inputs_cpu.numel()} токенов")
            print(f"   ⏱️ CPU: {cpu_time:.4f} сек, GPU: {gpu_time:.4f} сек")

        # Тест генерации
        print("🎯 Тест генерации на GPU:")
        test_query = "Расскажи о машинном обучении"

        start_time = time.time()
        with torch.no_grad():
            inputs = tokenizer.encode(test_query, return_tensors="pt").to("cuda")
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_time = time.time() - start_time
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"📝 Запрос: {test_query}")
        print(f"💬 Ответ: {response[:150]}...")
        print(f"⏱️ Время: {gen_time:.2f} сек")

        # Очистка
        del model, inputs, outputs
        torch.cuda.empty_cache()
        print("✅ GPU память очищена")

        return True

    except Exception as e:
        print(f"❌ Ошибка тестирования GPU: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Очистка и перезагрузка ruGPT-3 Medium")
    parser.add_argument("--clean", action="store_true", help="Очистить кэш и изолировать модели")
    parser.add_argument("--reload", action="store_true", help="Перезагрузить ruGPT-3 Medium")
    parser.add_argument("--gpu-test", action="store_true", help="Тестировать GPU токенизацию")

    args = parser.parse_args()

    if args.clean:
        clean_hf_cache()
        isolate_fractal_models()
    elif args.reload:
        reload_rugpt3_medium()
    elif args.gpu_test:
        test_gpu_tokenization()
    else:
        print("Используйте: --clean, --reload или --gpu-test")
        clean_hf_cache()
        isolate_fractal_models()
        reload_rugpt3_medium()
