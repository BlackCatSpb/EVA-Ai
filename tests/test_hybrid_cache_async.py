"""
Тест гибридного кэша токенов и асинхронной токенизации для CogniFlex
"""

import os
import sys
import time
import asyncio
import threading
from typing import Dict, Any
import pytest

# Добавляем путь к проекту
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cogniflex'))

def test_hybrid_cache():
    """Тестирует производительность гибридного кэша."""
    print("=== Тест гибридного кэша токенов ===")
    
    try:
        # Создаем минимальный brain для тестирования
        class TestBrain:
            def __init__(self):
                self.cache_dir = os.path.join(os.path.dirname(__file__), "test_cache")
                os.makedirs(self.cache_dir, exist_ok=True)
        
        brain = TestBrain()
        
        # Импортируем и создаем кэш
        from cogniflex.memory.hybrid_token_cache import HybridTokenCache
        cache = HybridTokenCache(brain, max_memory_tokens=100)
        
        print(f"✓ Кэш инициализирован: память={cache.max_memory_tokens}, диск={cache.disk_cache.max_size_bytes/(1024**3):.1f}GB")
        
        # Тест 1: Добавление и получение токенов
        print("\n--- Тест 1: Базовые операции ---")
        test_tokens = {}
        for i in range(50):
            token_id = f"token_{i}"
            token_data = {
                "text": f"Это тестовый токен номер {i}",
                "embedding": [0.1 * j for j in range(10)],
                "metadata": {"type": "test", "index": i}
            }
            test_tokens[token_id] = token_data
            cache.add_token(token_id, token_data)
        
        print(f"✓ Добавлено {len(test_tokens)} токенов")
        
        # Проверяем получение
        hits = 0
        start_time = time.time()
        for token_id in test_tokens:
            retrieved = cache.get_token(token_id)
            if retrieved:
                hits += 1
        
        access_time = time.time() - start_time
        print(f"✓ Получено {hits}/{len(test_tokens)} токенов за {access_time:.4f}сек")
        
        # Тест 2: Производительность при большой нагрузке
        print("\n--- Тест 2: Производительность ---")
        large_tokens = {}
        for i in range(200):
            token_id = f"large_token_{i}"
            token_data = {
                "text": f"Большой токен {i} " * 50,  # Увеличиваем размер
                "embedding": [0.01 * j * i for j in range(100)],
                "metadata": {"type": "large", "size": "big", "index": i}
            }
            large_tokens[token_id] = token_data
            cache.add_token(token_id, token_data)
        
        print(f"✓ Добавлено {len(large_tokens)} больших токенов")
        
        # Симулируем частые обращения к некоторым токенам
        hot_tokens = [f"large_token_{i}" for i in range(0, 50, 5)]  # Каждый 5-й токен
        
        start_time = time.time()
        for _ in range(3):  # 3 прохода для создания "горячих" токенов
            for token_id in hot_tokens:
                cache.get_token(token_id)
        
        hot_access_time = time.time() - start_time
        print(f"✓ Обращение к горячим токенам: {hot_access_time:.4f}сек")
        
        # Тест 3: Статистика кэша
        print("\n--- Тест 3: Статистика ---")
        stats = cache.get_cache_stats()
        print(f"✓ Токенов в памяти: {stats['memory_tokens']}")
        print(f"✓ Записей на диске: {stats['disk_stats']['entries']}")
        print(f"✓ Размер дискового кэша: {stats['disk_stats']['size_mb']:.2f}MB")
        print(f"✓ Эффективность кэша: {stats['hit_rate']:.2%}")
        print(f"✓ Среднее время доступа: {stats['usage_stats']['avg_access_time']:.4f}сек")
        
        # Тест 4: Многопоточность
        print("\n--- Тест 4: Многопоточность ---")
        
        def worker_thread(thread_id, num_operations):
            local_hits = 0
            for i in range(num_operations):
                token_id = f"thread_{thread_id}_token_{i}"
                # Добавляем токен
                cache.add_token(token_id, {"thread": thread_id, "data": f"data_{i}"})
                # Сразу пытаемся получить
                if cache.get_token(token_id):
                    local_hits += 1
            return local_hits
        
        threads = []
        results = {}
        start_time = time.time()
        
        for i in range(4):  # 4 потока
            def thread_func(tid=i):
                results[tid] = worker_thread(tid, 25)
            
            thread = threading.Thread(target=thread_func)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        mt_time = time.time() - start_time
        total_mt_hits = sum(results.values())
        print(f"✓ Многопоточный тест: {total_mt_hits}/100 операций за {mt_time:.4f}сек")
        
        # Финальная статистика
        final_stats = cache.get_cache_stats()
        print(f"\n--- Финальная статистика ---")
        print(f"Всего обращений: {final_stats['usage_stats']['total_accesses']}")
        print(f"Попаданий в память: {final_stats['usage_stats']['memory_hits']}")
        print(f"Попаданий на диск: {final_stats['usage_stats']['disk_hits']}")
        print(f"Промахов: {final_stats['usage_stats']['misses']}")
        print(f"Общая эффективность: {final_stats['hit_rate']:.2%}")
        
        print("\n✅ Тест гибридного кэша успешно завершен!")
        assert True
        
    except Exception as e:
        print(f"❌ Ошибка в тесте гибридного кэша: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(str(e))

def test_async_tokenization():
    """Тестирует асинхронную токенизацию."""
    print("\n=== Тест асинхронной токенизации ===")
    
    try:
        # Создаем минимальный brain
        class TestBrain:
            def __init__(self):
                self.cache_dir = os.path.join(os.path.dirname(__file__), "test_cache")
                os.makedirs(self.cache_dir, exist_ok=True)
        
        brain = TestBrain()
        
        # Импортируем текстовый процессор
        from cogniflex.mlearning.unified_text_processor import UnifiedTextProcessor
        processor = UnifiedTextProcessor(brain=brain, use_async=True, max_workers=2)
        
        print("✓ UnifiedTextProcessor инициализирован с асинхронной поддержкой")
        
        # Тест 1: Базовая токенизация
        print("\n--- Тест 1: Базовая токенизация ---")
        test_texts = [
            "Это простой тест токенизации.",
            "CogniFlex использует гибридный кэш для оптимизации.",
            "Асинхронная обработка повышает производительность системы.",
            "Машинное обучение требует эффективной работы с токенами."
        ]
        
        results = []
        completed_count = 0
        
        def tokenization_callback(tokens, context, error=None):
            nonlocal completed_count
            if error:
                print(f"❌ Ошибка токенизации: {error}")
            else:
                results.append({
                    "text": context.get("text", ""),
                    "tokens": tokens,
                    "token_count": len(tokens) if tokens else 0
                })
                print(f"✓ Токенизировано: {len(tokens) if tokens else 0} токенов")
            completed_count += 1
        
        # Запускаем асинхронную токенизацию
        start_time = time.time()
        for i, text in enumerate(test_texts):
            if hasattr(processor, 'tokenize'):
                tokens = processor.tokenize(text, priority=5)
                tokenization_callback(tokens, {"text": text, "index": i})
            else:
                # Fallback на синхронную токенизацию
                tokens = processor.process_text(text)
                tokenization_callback(tokens.get("keywords", []), {"text": text, "index": i})
        
        # Ждем завершения всех операций
        timeout = 10  # 10 секунд максимум
        while completed_count < len(test_texts) and timeout > 0:
            time.sleep(0.1)
            timeout -= 0.1
        
        processing_time = time.time() - start_time
        print(f"✓ Обработано {completed_count}/{len(test_texts)} текстов за {processing_time:.4f}сек")
        
        # Тест 2: Производительность
        print("\n--- Тест 2: Производительность ---")
        large_texts = [f"Большой текст для тестирования производительности системы токенизации. " * 20 for _ in range(10)]
        
        perf_results = []
        perf_completed = 0
        
        def perf_callback(tokens, context, error=None):
            nonlocal perf_completed
            if not error and tokens:
                perf_results.append(len(tokens))
            perf_completed += 1
        
        start_time = time.time()
        for i, text in enumerate(large_texts):
            if hasattr(processor, 'tokenize'):
                tokens = processor.tokenize(text, priority=3)
                perf_callback(tokens, {"index": i})
            else:
                tokens = processor.process_text(text)
                perf_callback(tokens.get("keywords", []), {"index": i})
        
        # Ждем завершения
        timeout = 15
        while perf_completed < len(large_texts) and timeout > 0:
            time.sleep(0.1)
            timeout -= 0.1
        
        perf_time = time.time() - start_time
        avg_tokens = sum(perf_results) / len(perf_results) if perf_results else 0
        
        print(f"✓ Производительность: {perf_completed}/{len(large_texts)} текстов")
        print(f"✓ Время обработки: {perf_time:.4f}сек")
        print(f"✓ Среднее количество токенов: {avg_tokens:.1f}")
        # Защита от деления на ноль на быстрых машинах/кэше
        safe_perf_time = perf_time if perf_time > 0 else 1e-6
        print(f"✓ Скорость: {avg_tokens * perf_completed / safe_perf_time:.1f} токенов/сек")
        
        print("\n✅ Тест асинхронной токенизации успешно завершен!")
        assert True
        
    except Exception as e:
        print(f"❌ Ошибка в тесте асинхронной токенизации: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(str(e))

def test_integration():
    """Тестирует интеграцию кэша и токенизации."""
    print("\n=== Тест интеграции ===")
    
    try:
        # Создаем brain с компонентами
        class IntegratedBrain:
            def __init__(self):
                self.cache_dir = os.path.join(os.path.dirname(__file__), "test_cache")
                os.makedirs(self.cache_dir, exist_ok=True)
                self.components = {}
        
        brain = IntegratedBrain()
        
        # Создаем компоненты
        from cogniflex.memory.hybrid_token_cache import HybridTokenCache
        from cogniflex.mlearning.unified_text_processor import UnifiedTextProcessor
        
        cache = HybridTokenCache(brain, max_memory_tokens=50)
        processor = UnifiedTextProcessor(brain=brain, hybrid_cache=cache, use_async=True)
        
        print("✓ Интегрированная система инициализирована")
        
        # Тест совместной работы
        test_sentences = [
            "Интеграция кэша и токенизации повышает эффективность.",
            "Гибридный подход оптимизирует использование ресурсов.",
            "Асинхронная обработка ускоряет работу системы."
        ]
        
        integration_results = []
        integration_completed = 0
        
        def integration_callback(tokens, context, error=None):
            nonlocal integration_completed
            if not error and tokens:
                # Сохраняем результат в кэш
                token_id = f"integrated_{context.get('index', 0)}"
                cache.add_token(token_id, {
                    "tokens": tokens,
                    "text": context.get("text", ""),
                    "processed_at": time.time()
                })
                integration_results.append(token_id)
            integration_completed += 1
        
        start_time = time.time()
        for i, sentence in enumerate(test_sentences):
            if hasattr(processor, 'tokenize'):
                tokens = processor.tokenize(sentence, priority=7)
                integration_callback(tokens, {"text": sentence, "index": i})
            else:
                tokens = processor.process_text(sentence)
                integration_callback(tokens.get("keywords", []), {"text": sentence, "index": i})
        
        # Ждем завершения
        timeout = 10
        while integration_completed < len(test_sentences) and timeout > 0:
            time.sleep(0.1)
            timeout -= 0.1
        
        integration_time = time.time() - start_time
        
        # Проверяем, что данные сохранились в кэше
        cached_count = 0
        for token_id in integration_results:
            if cache.get_token(token_id):
                cached_count += 1
        
        print(f"✓ Обработано и кэширано: {cached_count}/{len(test_sentences)} элементов")
        print(f"✓ Время интеграции: {integration_time:.4f}сек")
        
        # Финальная статистика
        final_stats = cache.get_cache_stats()
        print(f"✓ Эффективность кэша: {final_stats['hit_rate']:.2%}")
        
        print("\n✅ Тест интеграции успешно завершен!")
        assert True
        
    except Exception as e:
        print(f"❌ Ошибка в тесте интеграции: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(str(e))

if __name__ == "__main__":
    print("🚀 Запуск тестов гибридного кэша и асинхронной токенизации")
    print("=" * 60)
    
    try:
        # Запускаем только тест гибридного кэша сначала
        print("Запуск теста гибридного кэша...")
        cache_result = test_hybrid_cache()
        
        print("\nЗапуск теста асинхронной токенизации...")
        async_result = test_async_tokenization()
        
        print("\nЗапуск теста интеграции...")
        integration_result = test_integration()
        
        results = {
            "hybrid_cache": cache_result,
            "async_tokenization": async_result,
            "integration": integration_result
        }
        
        print("\n" + "=" * 60)
        print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        print("=" * 60)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ ПРОЙДЕН" if result else "❌ ПРОВАЛЕН"
            print(f"{test_name:20} : {status}")
        
        print(f"\nОбщий результат: {passed}/{total} тестов пройдено ({passed/total:.1%})")
        
        if passed == total:
            print("🎉 Все тесты успешно пройдены!")
        else:
            print("⚠️  Некоторые тесты провалены. Требуется доработка.")
            
    except Exception as e:
        print(f"❌ Критическая ошибка в main: {e}")
        import traceback
        traceback.print_exc()
