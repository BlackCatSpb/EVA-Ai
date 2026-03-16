#!/usr/bin/env python3
"""
Проверка готовности PyTorch для GPU и тестирование генерации
"""

import os
import sys
import time
from pathlib import Path

# Добавляем путь к CogniFlex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_pytorch_gpu():
    """Проверяет готовность PyTorch для GPU"""
    print("🔍 Проверка готовности PyTorch для GPU...")

    try:
        import torch
        print(f"✅ PyTorch версии: {torch.__version__}")

        # Проверка CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA доступна: {torch.version.cuda}")
            print(f"🎯 Количество GPU: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                print(f"  📱 GPU {i}: {gpu_props.name}")
                print(f"     💾 Память: {gpu_props.total_memory / (1024**3):.1f} GB")
                print(f"     🔢 Compute Capability: {gpu_props.major}.{gpu_props.minor}")
            
            # Тестирование GPU
            print("\n🧪 Тестирование GPU...")
            device = torch.device("cuda")
            
            # Создание тензора на GPU
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            
            # Тестовая операция
            start_time = time.time()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            print(f"✅ Тестовое умножение матриц на GPU: {gpu_time:.4f} сек")
            
            # Сравнение с CPU
            x_cpu = x.cpu()
            y_cpu = y.cpu()
            
            start_time = time.time()
            z_cpu = torch.matmul(x_cpu, y_cpu)
            cpu_time = time.time() - start_time
            
            print(f"⏱️ То же умножение на CPU: {cpu_time:.4f} сек")
            
            speedup = cpu_time / gpu_time
            print(f"🚀 Ускорение: {speedup:.1f}x")
            
            # Очистка
            del x, y, z, x_cpu, y_cpu, z_cpu
            torch.cuda.empty_cache()
            
            return True
            
        else:
            print("❌ CUDA недоступна")
            
            # Проверка доступных бэкендов
            print("\n🔍 Доступные бэкенды PyTorch:")
            for backend in torch.backends.cuda.is_built(), torch.backends.mps.is_built():
                print(f"  - {backend}")
            
            # Проверка ROCm (AMD GPU)
            if hasattr(torch.version, 'hip') and torch.version.hip:
                print(f"✅ ROCm доступен: {torch.version.hip}")
            else:
                print("❌ ROCm недоступен")
                
            return False

    except ImportError:
        print("❌ PyTorch не установлен")
        return False
    except Exception as e:
        print(f"❌ Ошибка проверки PyTorch: {e}")
        return False

def install_gpu_dependencies():
    """Устанавливает необходимые зависимости для GPU"""
    print("📦 Установка зависимостей для GPU...")

    dependencies = [
        "torch",
        "torchvision", 
        "torchaudio",
        "accelerate",
        "transformers",
        "xformers",
        "bitsandbytes"
    ]

    for dep in dependencies:
        print(f"🔍 Проверка {dep}...")
        try:
            if dep == "torch":
                import torch
                print(f"✅ {dep}: {torch.__version__}")
            elif dep == "torchvision":
                import torchvision
                print(f"✅ {dep}: {torchvision.__version__}")
            elif dep == "torchaudio":
                import torchaudio
                print(f"✅ {dep}: {torchaudio.__version__}")
            elif dep == "accelerate":
                import accelerate
                print(f"✅ {dep}: {accelerate.__version__}")
            elif dep == "transformers":
                import transformers
                print(f"✅ {dep}: {transformers.__version__}")
            elif dep == "xformers":
                import xformers
                print(f"✅ {dep}: {xformers.__version__}")
            elif dep == "bitsandbytes":
                import bitsandbytes
                print(f"✅ {dep}: {bitsandbytes.__version__}")
        except ImportError:
            print(f"❌ {dep} не установлен")
            install_cmd = f"pip install {dep}"
            if dep == "torch":
                # Для CUDA
                install_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            elif dep == "xformers":
                install_cmd = "pip install xformers --pre"
            print(f"📦 Установка: {install_cmd}")
            # os.system(install_cmd)  # Раскомментировать для реальной установки

def test_gpu_generation():
    """Тестирует генерацию на GPU с ruGPT-3"""
    print("🚀 Тестирование генерации на GPU...")

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if not torch.cuda.is_available():
            print("❌ GPU недоступен")
            return False

        model_path = "./models/rugpt3medium_fresh"
        if not Path(model_path).exists():
            print("❌ Модель не найдена")
            return False

        print(f"📦 Загрузка модели из {model_path} на GPU...")

        # Загрузка токенизатора
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Загрузка модели на GPU с оптимизацией
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # float16 для экономии памяти
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        memory_used = torch.cuda.memory_allocated() / (1024**3)

        print(f"✅ Модель загружена на GPU")
        print(f"💾 Использование GPU: {memory_used:.1f}/{gpu_memory:.1f} GB ({memory_used/gpu_memory*100:.1f}%)")

        # Тесты генерации
        test_prompts = [
            "Привет! Как дела?",
            "Что такое искусственный интеллект?",
            "Расскажи о машинном обучении",
            "Преимущества фрактального хранилища",
            "Как работает GPU токенизация?"
        ]

        print(f"\n🎯 Тесты генерации ({len(test_prompts)} запросов):")

        for i, prompt in enumerate(test_prompts):
            print(f"\n📝 Запрос {i+1}: {prompt}")

            start_time = time.time()

            with torch.no_grad():
                inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 60,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )

            gen_time = time.time() - start_time
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Очистка от запроса
            if response.startswith(prompt):
                response = response[len(prompt):].strip()

            print(f"💬 Ответ: {response[:150]}...")
            print(f"⏱️ Время генерации: {gen_time:.2f} сек")
            print(f"🔢 Токенов сгенерировано: {outputs.shape[1] - inputs.shape[1]}")
            print(f"⚡ Скорость: {(outputs.shape[1] - inputs.shape[1]) / gen_time:.1f} токен/сек")

        # Финальная статистика памяти
        final_memory = torch.cuda.memory_allocated() / (1024**3)
        max_memory = torch.cuda.max_memory_allocated() / (1024**3)

        print(f"\n📊 Статистика GPU:")
        print(f"💾 Текущее использование: {final_memory:.1f} GB")
        print(f"📈 Максимальное использование: {max_memory:.1f} GB")
        print(f"🔄 Пиковая эффективность: {max_memory/gpu_memory*100:.1f}%")

        # Очистка
        del model, inputs, outputs
        torch.cuda.empty_cache()
        print("✅ GPU память очищена")

        return True

    except Exception as e:
        print(f"❌ Ошибка тестирования генерации: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_gpu_performance():
    """Анализирует производительность GPU"""
    print("📊 Анализ производительности GPU...")

    try:
        import torch
        import torch.nn as nn

        if not torch.cuda.is_available():
            print("❌ GPU недоступен")
            return False

        device = torch.device("cuda")
        
        # Тестирование разных операций
        operations = [
            ("Матричное умножение", lambda: torch.matmul(
                torch.randn(1000, 1000).to(device),
                torch.randn(1000, 1000).to(device)
            )),
            ("Свертка", lambda: nn.Conv2d(3, 64, 3).to(device)(
                torch.randn(1, 3, 224, 224).to(device)
            )),
            ("ReLU активация", lambda: nn.ReLU().to(device)(
                torch.randn(1000, 1000).to(device)
            )),
            ("BatchNorm", lambda: nn.BatchNorm1d(1000).to(device)(
                torch.randn(32, 1000).to(device)
            ))
        ]

        print("🧪 Тестирование операций:")
        
        for op_name, op_func in operations:
            # Прогрев
            for _ in range(10):
                op_func()
            
            # Измерение
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(100):
                result = op_func()
            
            torch.cuda.synchronize()
            avg_time = (time.time() - start_time) / 100
            
            print(f"  📊 {op_name}: {avg_time*1000:.2f} мс")
            
            del result
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"❌ Ошибка анализа производительности: {e}")
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Проверка готовности PyTorch для GPU")
    parser.add_argument("--check", action="store_true", help="Проверить PyTorch GPU")
    parser.add_argument("--install", action="store_true", help="Установить зависимости")
    parser.add_argument("--test", action="store_true", help="Тестировать генерацию")
    parser.add_argument("--benchmark", action="store_true", help="Анализ производительности")

    args = parser.parse_args()

    if args.check:
        check_pytorch_gpu()
    elif args.install:
        install_gpu_dependencies()
    elif args.test:
        test_gpu_generation()
    elif args.benchmark:
        analyze_gpu_performance()
    else:
        print("🚀 Выполнение полной проверки...")
        check_pytorch_gpu()
        if torch.cuda.is_available():
            test_gpu_generation()
            analyze_gpu_performance()
