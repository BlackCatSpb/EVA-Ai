#!/usr/bin/env python3
"""
Полная очистка кэша HF и перезагрузка ruGPT-3 Medium
"""

import os
import sys
import shutil
import time
from pathlib import Path

# Добавляем путь к CogniFlex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clean_all_hf_cache():
    """Полностью очищает весь кэш Hugging Face"""
    print("🧹 Полная очистка кэша Hugging Face...")

    cache_dir = Path.home() / '.cache' / 'huggingface'

    if not cache_dir.exists():
        print("ℹ️ Директория кэша не найдена")
        return True

    try:
        # Показываем что будем удалять
        total_size = 0
        items_to_remove = []

        for item in cache_dir.rglob('*'):
            if item.is_file():
                size = item.stat().st_size
                total_size += size
                items_to_remove.append(item)

        total_size_mb = total_size / (1024**2)
        print(f"📊 Найдено файлов: {len(items_to_remove)}")
        print(f"💾 Общий размер: {total_size_mb:.1f} MB")

        # Удаляем весь кэш
        print("🗑️ Удаление всего кэша...")
        shutil.rmtree(cache_dir)
        print("✅ Кэш Hugging Face полностью очищен")

        return True

    except Exception as e:
        print(f"❌ Ошибка очистки кэша: {e}")
        return False

def clean_local_models():
    """Очищает локальные директории моделей"""
    print("🧹 Очистка локальных директорий моделей...")

    dirs_to_clean = [
        Path('./models'),
        Path('./saved_models'),
        Path('./fractal_exports'),
        Path('./fractal_storage')
    ]

    for dir_path in dirs_to_clean:
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"✅ Удалена директория: {dir_path}")
            except Exception as e:
                print(f"⚠️ Ошибка удаления {dir_path}: {e}")

    print("✅ Локальные директории очищены")

def reload_rugpt3_medium_clean():
    """Перезагружает ruGPT-3 Medium с чистого состояния"""
    print("🚀 Перезагрузка ruGPT-3 Medium с чистого состояния...")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        model_name = "sberbank-ai/rugpt3medium_based_on_gpt2"

        print(f"📦 Загрузка {model_name}...")

        # Загрузка токенизатора
        print("📝 Загрузка токенизатора...")
        tokenizer_start = time.time()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer_time = time.time() - tokenizer_start
        print(f"✅ Токенизатор загружен за {tokenizer_time:.2f} сек")
        print(f"📝 Словарь: {tokenizer.vocab_size} токенов")

        # Загрузка модели
        print("🧠 Загрузка модели...")
        model_start = time.time()

        # Используем оптимальные параметры для загрузки
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="cpu"  # Загружаем на CPU для стабильности
        )

        model_time = time.time() - model_start
        print(f"✅ Модель загружена за {model_time:.2f} сек")

        # Информация о модели
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"🔢 Всего параметров: {total_params:,}")
        print(f"🎯 Обучаемых: {trainable_params:,}")
        print(f"💾 Размер в памяти: {total_params * 4 / (1024**2):.1f} MB")

        # Конфигурация
        config = model.config
        print(f"🧠 Тип: {config.model_type}")
        if hasattr(config, 'n_embd'):
            print(f"📐 Размерность: {config.n_embd}")
        if hasattr(config, 'n_layer'):
            print(f"📚 Слои: {config.n_layer}")
        if hasattr(config, 'n_head'):
            print(f"👥 Головы: {config.n_head}")

        # Тест токенизации
        print("\n🧪 Тест токенизации...")
        test_texts = [
            "Привет! Как дела?",
            "Что такое искусственный интеллект?",
            "Расскажи о машинном обучении и нейронных сетях",
            "Преимущества фрактального хранилища"
        ]

        for text in test_texts:
            start_time = time.time()
            tokens = tokenizer.encode(text, return_tensors="pt")
            token_time = time.time() - start_time

            print(f"📝 '{text[:40]}...' -> {tokens.numel()} токенов ({token_time:.4f} сек)")

        # Тест генерации
        print("\n🎯 Тест генерации...")
        for i, query in enumerate(test_texts[:3]):
            print(f"\n📝 Запрос {i+1}: {query}")

            start_time = time.time()

            with torch.no_grad():
                inputs = tokenizer.encode(query, return_tensors="pt")
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 80,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )

            gen_time = time.time() - start_time
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Очистка от запроса
            if response.startswith(query):
                response = response[len(query):].strip()

            print(f"💬 Ответ: {response[:150]}...")
            print(f"⏱️ Время: {gen_time:.2f} сек")

        # Сохранение локально
        save_path = "./models/rugpt3medium_fresh"
        os.makedirs(save_path, exist_ok=True)

        print(f"\n💾 Сохранение модели локально...")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        print(f"✅ Модель сохранена в {save_path}")

        # Проверка размера
        total_size = sum(f.stat().st_size for f in Path(save_path).rglob('*') if f.is_file())
        print(f"💾 Размер сохраненной модели: {total_size / (1024**2):.1f} MB")

        # Экспорт во фрактальное хранилище
        print(f"\n💾 Экспорт во фрактальное хранилище...")

        from cogniflex.mlearning.fractal_rugpt3_manager import FractalRuGPT3Manager

        # Создаем чистое фрактальное хранилище
        fractal_storage_path = "./fractal_storage_clean"
        if os.path.exists(fractal_storage_path):
            shutil.rmtree(fractal_storage_path)
        os.makedirs(fractal_storage_path, exist_ok=True)

        fractal_manager = FractalRuGPT3Manager(
            brain=None,
            model_name="rugpt3medium",
            storage_path=fractal_storage_path
        )

        # Инициализируем менеджер с загруженной моделью
        fractal_manager.model = model
        fractal_manager.tokenizer = tokenizer
        fractal_manager.initialized = True

        export_path = os.path.join(fractal_storage_path, "rugpt3medium")
        if fractal_manager.export_model(export_path):
            print(f"✅ Модель экспортирована во фрактальное хранилище: {export_path}")

            # Проверка размера экспорта
            export_size = sum(f.stat().st_size for f in Path(export_path).rglob('*') if f.is_file())
            print(f"💾 Размер фрактального экспорта: {export_size / (1024**2):.1f} MB")

            # Содержимое
            print("\n📁 Структура фрактального хранилища:")
            for root, dirs, files in os.walk(export_path):
                level = root.replace(export_path, '').count(os.sep)
                indent = '  ' * level
                folder_name = os.path.basename(root) if os.path.basename(root) else "root"
                print(f"{indent}📁 {folder_name}/")
                subindent = '  ' * (level + 1)
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path) / (1024**2)
                    if file_size < 1:
                        file_size_kb = os.path.getsize(file_path) / 1024
                        print(f"{subindent}📄 {file} ({file_size_kb:.1f} KB)")
                    else:
                        print(f"{subindent}📄 {file} ({file_size:.1f} MB)")

        else:
            print(f"❌ Не удалось экспортировать модель")

        # Создание метаданных
        metadata = {
            "model_name": model_name,
            "description": "ruGPT-3 Medium (чистая перезагрузка)",
            "total_params": total_params,
            "vocab_size": tokenizer.vocab_size,
            "load_time": {
                "tokenizer": tokenizer_time,
                "model": model_time,
                "total": tokenizer_time + model_time
            },
            "local_save_path": save_path,
            "fractal_export_path": export_path,
            "clean_reload": True,
            "timestamp": time.time()
        }

        import json
        with open(f"{export_path}/fresh_load_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"📋 Метаданные сохранены")

        # Очистка
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✅ Перезагрузка ruGPT-3 Medium завершена успешно!")

        # Итоговый отчет
        print("\n📊 Итоговый отчет:")
        print(f"  🤖 Модель: ruGPT-3 Medium")
        print(f"  🔢 Параметры: {total_params:,}")
        print(f"  ⏱️ Время загрузки: {tokenizer_time + model_time:.2f} сек")
        print(f"  💾 Локальное сохранение: {save_path}")
        print(f"  💾 Фрактальное хранилище: {export_path}")
        print(f"  ✅ Чистая перезагрузка: Выполнена")

        return True

    except Exception as e:
        print(f"❌ Ошибка перезагрузки: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_with_fresh_model():
    """Тестирует GPU токенизацию с свежезагруженной моделью"""
    print("🚀 Тестирование GPU токенизации с свежей моделью...")

    import torch

    model_path = "./models/rugpt3medium_fresh"
    if not Path(model_path).exists():
        print("❌ Свежая модель не найдена")
        return False

    if not torch.cuda.is_available():
        print("❌ GPU не доступен")
        return False

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"📦 Загрузка модели из {model_path} на GPU...")

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
            "GPU токенизация в действии",
            "Машинное обучение и глубокие нейронные сети"
        ]

        print("\n🧪 Тест токенизации:")
        cpu_times = []
        gpu_times = []

        for text in test_texts:
            # CPU токенизация
            start_time = time.time()
            inputs_cpu = tokenizer.encode(text, return_tensors="pt")
            cpu_time = time.time() - start_time
            cpu_times.append(cpu_time)

            # GPU токенизация
            start_time = time.time()
            inputs_gpu = inputs_cpu.to("cuda")
            gpu_time = time.time() - start_time
            gpu_times.append(gpu_time)

            print(f"📝 '{text[:35]}...' -> {inputs_cpu.numel()} токенов")
            print(f"   ⏱️ CPU: {cpu_time:.4f} сек, GPU: {gpu_time:.4f} сек")

        # Статистика
        avg_cpu = sum(cpu_times) / len(cpu_times)
        avg_gpu = sum(gpu_times) / len(gpu_times)

        print(f"\n📊 Статистика токенизации:")
        print(f"   ⏱️ Среднее CPU: {avg_cpu:.4f} сек")
        print(f"   ⚡ Среднее GPU: {avg_gpu:.4f} сек")

        if avg_gpu > 0:
            speedup = avg_cpu / avg_gpu
            print(f"   🚀 Ускорение: {speedup:.1f}x")

        # Тест генерации
        print(f"\n🎯 Тест генерации на GPU:")
        test_query = "Расскажи о машинном обучении"

        start_time = time.time()
        with torch.no_grad():
            inputs = tokenizer.encode(test_query, return_tensors="pt").to("cuda")
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 60,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_time = time.time() - start_time
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"📝 Запрос: {test_query}")
        print(f"💬 Ответ: {response[:150]}...")
        print(f"⏱️ Время генерации: {gen_time:.2f} сек")

        # Финальная память
        final_memory = torch.cuda.memory_allocated() / (1024**3)
        print(f"\n💾 Финальное использование GPU: {final_memory:.1f} GB")

        # Очистка
        del model, inputs, outputs
        torch.cuda.empty_cache()
        print("✅ GPU память очищена")

        print("\n✅ GPU токенизация протестирована успешно!")
        return True

    except Exception as e:
        print(f"❌ Ошибка тестирования GPU: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Полная очистка и перезагрузка ruGPT-3 Medium")
    parser.add_argument("--clean-all", action="store_true", help="Полностью очистить весь кэш HF")
    parser.add_argument("--reload", action="store_true", help="Перезагрузить ruGPT-3 Medium")
    parser.add_argument("--gpu-test", action="store_true", help="Тестировать GPU токенизацию")

    args = parser.parse_args()

    if args.clean_all:
        clean_all_hf_cache()
        clean_local_models()
    elif args.reload:
        reload_rugpt3_medium_clean()
    elif args.gpu_test:
        test_gpu_with_fresh_model()
    else:
        print("Используйте: --clean-all, --reload или --gpu-test")
        # Выполняем полную последовательность
        print("🚀 Выполнение полной последовательности...")
        if clean_all_hf_cache() and clean_local_models():
            if reload_rugpt3_medium_clean():
                test_gpu_with_fresh_model()
