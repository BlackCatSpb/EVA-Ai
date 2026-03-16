#!/usr/bin/env python3
"""
Проверка состояния загрузки ruGPT-3 Medium и тестирование GPU токенизации
"""

import os
import sys
import torch
from pathlib import Path

# Добавляем путь к CogniFlex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_model_loaded():
    """Проверяет, загружена ли модель ruGPT-3 Medium"""
    print("🔍 Проверка состояния загрузки ruGPT-3 Medium...")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_name = "sberbank-ai/rugpt3medium_based_on_gpt2"

        # Проверяем токенизатор
        print("📝 Проверка токенизатора...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"✅ Токенизатор доступен, словарь: {tokenizer.vocab_size}")

        # Проверяем модель (быстрая проверка)
        print("🧠 Проверка модели...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="cpu"
            )
            print("✅ Модель загружена успешно!")

            # Информация о модели
            total_params = sum(p.numel() for p in model.parameters())
            print(f"🔢 Параметры: {total_params:,}")
            print(f"💾 Размер в памяти: {total_params * 4 / (1024**2):.1f} MB")

            # Тест генерации
            print("\n🎯 Быстрый тест генерации...")
            with torch.no_grad():
                inputs = tokenizer("Привет!", return_tensors="pt")
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=20,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"💬 Тестовый ответ: {response}")

            # Очистка
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return True

        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def test_gpu_tokenization_with_loaded_model():
    """Тестирует GPU токенизацию с уже загруженной моделью"""
    print("🚀 Тестирование GPU токенизации...")

    if not torch.cuda.is_available():
        print("❌ GPU не доступен")
        return False

    gpu_name = torch.cuda.get_device_name()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print(f"✅ GPU: {gpu_name}")
    print(f"💾 Память: {gpu_memory:.1f} GB")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_name = "sberbank-ai/rugpt3medium_based_on_gpt2"

        # Загружаем на GPU
        print(f"\n📦 Загрузка {model_name} на GPU...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        print(f"✅ Модель загружена на GPU")

        # Память
        memory_used = torch.cuda.memory_allocated() / (1024**3)
        print(f"💾 Использование GPU: {memory_used:.1f}/{gpu_memory:.1f} GB ({memory_used/gpu_memory*100:.1f}%)")

        # Тест токенизации
        test_texts = [
            "Привет! Как дела?",
            "Что такое искусственный интеллект?",
            "Расскажи о фрактальном хранилище",
            "Преимущества GPU токенизации",
            "Машинное обучение и нейронные сети"
        ]

        print(f"\n🧪 Тест токенизации ({len(test_texts)} текстов)...")

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

            print(f"📝 '{text[:30]}...' -> {inputs_cpu.numel()} токенов")
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
        print(f"\n🎯 Тест генерации на GPU...")
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

        # Экспорт во фрактальное хранилище
        print(f"\n💾 Экспорт во фрактальное хранилище...")

        from cogniflex.mlearning.fractal_rugpt3_manager import FractalRuGPT3Manager

        fractal_manager = FractalRuGPT3Manager(
            brain=None,
            model_name="rugpt3medium",
            storage_path="./fractal_storage"
        )

        fractal_manager.model = model
        fractal_manager.tokenizer = tokenizer
        fractal_manager.initialized = True

        export_path = "./fractal_exports/rugpt3medium_gpu"
        if fractal_manager.export_model(export_path):
            print(f"✅ Модель экспортирована во фрактальное хранилище: {export_path}")

            # Проверка размера
            total_size = 0
            for root, dirs, files in os.walk(export_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)

            print(f"💾 Размер фрактального экспорта: {total_size / (1024**2):.1f} MB")
        else:
            print(f"❌ Не удалось экспортировать модель")

        # Очистка
        del model, inputs, outputs
        torch.cuda.empty_cache()
        print(f"✅ GPU память очищена")

        return True

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

import time

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Проверка загрузки ruGPT-3 Medium и GPU токенизации")
    parser.add_argument("--check", action="store_true", help="Проверить состояние загрузки модели")
    parser.add_argument("--gpu-test", action="store_true", help="Тестировать GPU токенизацию")

    args = parser.parse_args()

    if args.check:
        success = check_model_loaded()
        if success:
            print(f"\n✅ ruGPT-3 Medium успешно загружена!")
        else:
            print(f"\n❌ ruGPT-3 Medium не загружена")

    elif args.gpu_test:
        success = test_gpu_tokenization_with_loaded_model()
        if success:
            print(f"\n✅ GPU токенизация протестирована успешно!")
        else:
            print(f"\n❌ Ошибка тестирования GPU токенизации")

    else:
        print("Используйте: --check или --gpu-test")
        check_model_loaded()
