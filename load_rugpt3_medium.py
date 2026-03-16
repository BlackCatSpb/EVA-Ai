#!/usr/bin/env python3
"""
Загрузка и экспорт ruGPT-3 Medium с GPU токенизацией
"""

import os
import sys
import logging
import time
import torch
from typing import Dict, Any

# Добавляем путь к CogniFlex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniflex.core.utils import setup_logging
from cogniflex.mlearning.enhanced_rugpt3_manager import EnhancedRuGPT3ModelManager

def check_gpu_availability():
    """Проверяет доступность GPU"""
    print("🔍 Проверка доступности GPU...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        memory_total = torch.cuda.get_device_properties(current_device).total_memory
        memory_total_gb = memory_total / (1024**3)
        
        print(f"  ✅ GPU доступен: {device_name}")
        print(f"  📊 Устройств: {gpu_count}")
        print(f"  💾 Видеопамять: {memory_total_gb:.1f} GB")
        print(f"  📍 Текущее устройство: CUDA:{current_device}")
        
        return True, current_device, device_name, memory_total_gb
    else:
        print("  ❌ GPU не доступен, будет использоваться CPU")
        return False, None, None, 0

def load_and_export_rugpt3_medium():
    """Загружает и экспортирует ruGPT-3 Medium"""
    setup_logging(log_dir='logs')
    
    print("🚀 Загрузка и экспорт ruGPT-3 Medium...")
    
    # Проверка GPU
    gpu_available, gpu_id, gpu_name, gpu_memory = check_gpu_availability()
    
    # Настройка устройства
    device = "cuda" if gpu_available else "cpu"
    print(f"🎯 Устройство: {device}")
    
    # Создаем менеджер с ruGPT-3 Medium
    print("📦 Создание EnhancedRuGPT3ModelManager...")
    
    manager = EnhancedRuGPT3ModelManager(
        brain=None,
        model_name="rugpt3medium",  # ruGPT-3 Medium
        cache_dir="./cache",
        device=device,
        max_memory_gb=2.0 if gpu_available else 1.5,  # Больше памяти для GPU
        enable_gpu_tokenization=gpu_available,
        cache_tokens=True
    )
    
    print(f"🔧 Менеджер создан, инициализация...")
    
    # Замер времени инициализации
    init_start = time.time()
    
    if not manager.initialized:
        print("❌ Не удалось инициализировать менеджер")
        return False
    
    init_time = time.time() - init_start
    print(f"✅ Менеджер инициализирован за {init_time:.2f} сек")
    
    # Информация о модели
    model_info = manager.get_model_info()
    print(f"📊 Информация о модели:")
    for key, value in model_info.items():
        if key != 'model_info':
            print(f"  {key}: {value}")
    
    # Тест токенизации на GPU
    print(f"\n🧪 Тест токенизации...")
    test_texts = [
        "Привет! Как дела?",
        "Что такое искусственный интеллект?",
        "Расскажи о машинном обучении и нейронных сетях",
        "Преимущества и недостатки различных архитектур трансформеров",
        "Как работает механизм внимания в моделях типа GPT?"
    ]
    
    total_tokenization_time = 0
    total_tokens = 0
    
    for i, text in enumerate(test_texts):
        print(f"\n📝 Тест {i+1}: {text[:50]}...")
        
        # Токенизация с замером времени
        token_start = time.time()
        try:
            tokens = manager.tokenize_with_cache(text, return_tensors="pt")
            token_time = time.time() - token_start
            token_count = tokens.numel()
            
            total_tokenization_time += token_time
            total_tokens += token_count
            
            print(f"  ✅ Токенов: {token_count}")
            print(f"  ⏱️ Время: {token_time:.4f} сек")
            print(f"  🚀 Скорость: {token_count/token_time:.0f} ток/сек")
            print(f"  💾 Устройство токенов: {tokens.device}")
            
        except Exception as e:
            print(f"  ❌ Ошибка токенизации: {e}")
    
    # Статистика токенизации
    if total_tokenization_time > 0:
        avg_speed = total_tokens / total_tokenization_time
        print(f"\n📊 Статистика токенизации:")
        print(f"  🔄 Всего токенов: {total_tokens}")
        print(f"  ⏱️ Общее время: {total_tokenization_time:.4f} сек")
        print(f"  🚀 Средняя скорость: {avg_speed:.0f} ток/сек")
    
    # Тест генерации ответов
    print(f"\n🎯 Тест генерации ответов...")
    
    for i, query in enumerate(test_texts[:3]):  # Тест только первых 3
        print(f"\n📝 Запрос {i+1}: {query}")
        
        gen_start = time.time()
        try:
            response = manager.generate_response(
                query,
                max_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            gen_time = time.time() - gen_start
            
            print(f"  💬 Ответ: {response[:150]}...")
            print(f"  ⏱️ Время генерации: {gen_time:.2f} сек")
            
        except Exception as e:
            print(f"  ❌ Ошибка генерации: {e}")
            gen_time = time.time() - gen_start
            print(f"  ⏱️ Время ошибки: {gen_time:.2f} сек")
    
    # Статистика модели
    stats = manager.get_stats()
    print(f"\n📊 Статистика модели:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Использование памяти
    memory = manager.get_memory_usage()
    print(f"\n💾 Использование памяти:")
    for key, value in memory.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1f}")
        else:
            print(f"  {key}: {value}")
    
    # Экспорт модели
    print(f"\n📦 Экспорт модели во фрактальное хранилище...")
    export_start = time.time()
    
    export_path = "./fractal_exports/rugpt3medium"
    
    if manager.export_model(export_path):
        export_time = time.time() - export_start
        print(f"✅ Модель экспортирована в {export_path}")
        print(f"⏱️ Время экспорта: {export_time:.2f} сек")
        
        # Проверка размера экспорта
        import os
        export_size = 0
        for root, dirs, files in os.walk(export_path):
            for file in files:
                file_path = os.path.join(root, file)
                export_size += os.path.getsize(file_path)
        
        export_size_mb = export_size / (1024 * 1024)
        print(f"💾 Размер экспорта: {export_size_mb:.1f} MB")
        
    else:
        print(f"❌ Не удалось экспортировать модель")
    
    # Тест переключения моделей
    print(f"\n🔄 Тест переключения моделей...")
    
    # Переключение на GPT-2
    if manager.switch_model("gpt2"):
        print("✅ Переключено на GPT-2")
        
        # Быстрый тест
        quick_response = manager.generate_response("Тест переключения", max_tokens=50)
        print(f"💬 Ответ GPT-2: {quick_response[:100]}...")
        
        # Возврат к ruGPT-3
        if manager.switch_model("rugpt3medium"):
            print("✅ Возврат к ruGPT-3 Medium")
        else:
            print("⚠️ Не удалось вернуться к ruGPT-3 Medium")
    else:
        print("⚠️ Не удалось переключиться на GPT-2")
    
    # Очистка
    print(f"\n🧹 Очистка ресурсов...")
    manager.cleanup()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"✅ GPU кэш очищен")
    
    print(f"\n🎉 Тестирование ruGPT-3 Medium завершено!")
    
    # Итоговая статистика
    print(f"\n📋 Итоговая статистика:")
    print(f"  🤖 Модель: ruGPT-3 Medium")
    print(f"  🎯 Устройство: {device}")
    print(f"  💾 GPU память: {gpu_memory:.1f} GB" if gpu_available else "  💾 GPU память: Недоступно")
    print(f"  ⚡ Токенизация: {'GPU' if gpu_available else 'CPU'}")
    print(f"  📦 Экспорт: {'Успешно' if os.path.exists(export_path) else 'Ошибка'}")
    
    return True

def test_gpu_tokenization_performance():
    """Тестирует производительность GPU токенизации"""
    print("🚀 Тест производительности GPU токенизации...")
    
    if not torch.cuda.is_available():
        print("❌ GPU не доступен")
        return
    
    # Создаем тестовый менеджер
    manager = EnhancedRuGPT3ModelManager(
        brain=None,
        model_name="rugpt3medium",
        device="cuda",
        enable_gpu_tokenization=True
    )
    
    if not manager.initialized:
        print("❌ Не удалось инициализировать менеджер")
        return
    
    # Большой тестовый текст
    test_text = """
    Искусственный интеллект и машинное обучение революционизировали множество отраслей. 
    Нейронные сети, особенно трансформерные архитектуры, показали впечатляющие результаты 
    в обработке естественного языка, машинном переводе и генерации текста. Модели вроде GPT 
    используют механизм внимания для понимания контекста и генерации осмысленных ответов. 
    RuGPT-3 от Сбера специализируется на русском языке и показывает высокое качество 
    в понимании и генерации текстов на русском. Фрактальное хранилище позволяет эффективно 
    управлять весами моделей и оптимизировать использование памяти.
    """ * 5  # Умножаем для большего текста
    
    print(f"📝 Длина тестового текста: {len(test_text)} символов")
    
    # Тест токенизации
    iterations = 10
    times = []
    
    for i in range(iterations):
        start_time = time.time()
        tokens = manager.tokenize_with_cache(test_text, return_tensors="pt")
        token_time = time.time() - start_time
        times.append(token_time)
        
        if i == 0:
            print(f"🔄 Первая токенизация: {token_time:.4f} сек, токенов: {tokens.numel()}")
    
    # Статистика
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n📊 Статистика токенизации ({iterations} итераций):")
    print(f"  ⏱️ Среднее время: {avg_time:.4f} сек")
    print(f"  ⏱️ Минимальное: {min_time:.4f} сек")
    print(f"  ⏱️ Максимальное: {max_time:.4f} сек")
    
    # Сравнение с CPU
    print(f"\n🔄 Сравнение с CPU...")
    manager_cpu = EnhancedRuGPT3ModelManager(
        brain=None,
        model_name="gpt2",  # Используем более легкую модель для CPU
        device="cpu",
        enable_gpu_tokenization=False
    )
    
    if manager_cpu.initialized:
        start_time = time.time()
        tokens_cpu = manager_cpu.tokenize_with_cache(test_text[:1000], return_tensors="pt")  # Короче текст для CPU
        cpu_time = time.time() - start_time
        
        print(f"  ⏱️ CPU время: {cpu_time:.4f} сек")
        print(f"  ⚡ Ускорение GPU: {cpu_time/avg_time:.1f}x")
    
    manager.cleanup()
    manager_cpu.cleanup()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Загрузка ruGPT-3 Medium с GPU токенизацией")
    parser.add_argument("--load", action="store_true", help="Загрузить и экспортировать ruGPT-3 Medium")
    parser.add_argument("--gpu-test", action="store_true", help="Тест производительности GPU")
    parser.add_argument("--check", action="store_true", help="Проверить доступность GPU")
    
    args = parser.parse_args()
    
    if args.load:
        load_and_export_rugpt3_medium()
    elif args.gpu_test:
        test_gpu_tokenization_performance()
    elif args.check:
        check_gpu_availability()
    else:
        print("Используйте: --load, --gpu-test или --check")
        load_and_export_rugpt3_medium()
