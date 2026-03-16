#!/usr/bin/env python3
"""
Анализ проблемы CUDA и исследование гибридного кеширования
"""
import sys
import os
import torch
import gc

# Добавляем путь к CogniFlex
sys.path.append('.')

def analyze_cuda_state():
    """Анализирует текущее состояние CUDA"""
    print("🔥 АНАЛИЗ СОСТОЯНИЯ CUDA")
    print("=" * 50)
    
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Количество GPU: {torch.cuda.device_count()}")
        print(f"Текущее устройство: {torch.cuda.current_device()}")
        print(f"Имя устройства: {torch.cuda.get_device_name()}")
        
        # Проверяем память
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        cached_memory = torch.cuda.memory_reserved(0)
        
        print(f"Всего памяти GPU: {total_memory / 1024**3:.2f} GB")
        print(f"Выделено памяти: {allocated_memory / 1024**3:.2f} GB")
        print(f"Кешировано памяти: {cached_memory / 1024**3:.2f} GB")
        print(f"Свободно памяти: {(total_memory - allocated_memory) / 1024**3:.2f} GB")
    
    return torch.cuda.is_available()

def test_model_loading():
    """Тестирует загрузку модели на разных устройствах"""
    print("\n🧪 ТЕСТ ЗАГРУЗКИ МОДЕЛИ")
    print("=" * 50)
    
    try:
        from cogniflex.mlearning.fractal_model_manager import FractalModelManager
        
        # Тестируем CPU
        print("📦 Тест загрузки на CPU...")
        manager_cpu = FractalModelManager()
        if hasattr(manager_cpu, 'device'):
            manager_cpu.device = 'cpu'
        
        if manager_cpu.initialize():
            print("✅ Модель успешно загружена на CPU")
            
            # Проверяем где находится модель
            if hasattr(manager_cpu, 'model') and manager_cpu.model:
                model_device = next(manager_cpu.model.parameters()).device
                print(f"📍 Устройство модели: {model_device}")
                
                # Тестируем генерацию
                try:
                    response = manager_cpu.generate_response("Привет!", max_tokens=30)
                    print(f"✅ Генерация на CPU успешна: {response[:50]}...")
                    return True
                except Exception as e:
                    print(f"❌ Ошибка генерации на CPU: {e}")
                    return False
            else:
                print("❌ Модель не загружена")
                return False
        else:
            print("❌ Не удалось инициализировать менеджер на CPU")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка теста: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_memory_usage():
    """Анализирует использование памяти"""
    print("\n💾 АНАЛИЗ ИСПОЛЬЗОВАНИЯ ПАМЯТИ")
    print("=" * 50)
    
    # Очищаем кэш
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Проверяем память до и после
    import psutil
    memory = psutil.virtual_memory()
    print(f"RAM всего: {memory.total / 1024**3:.2f} GB")
    print(f"RAM использовано: {memory.used / 1024**3:.2f} GB ({memory.percent}%)")
    print(f"RAM доступно: {memory.available / 1024**3:.2f} GB")

def investigate_hybrid_caching():
    """Исследует возможность гибридного кеширования"""
    print("\n🔄 ИССЛЕДОВАНИЕ ГИБРИДНОГО КЕШИРОВАНИЯ")
    print("=" * 50)
    
    try:
        from cogniflex.memory.hybrid_token_cache import HybridTokenCache
        
        print("📦 Тест HybridTokenCache...")
        
        # Создаем кэш с разными конфигурациями
        configs = [
            {"max_memory_gb": 0.5, "device": "cpu"},
            {"max_memory_gb": 1.0, "device": "cpu"}, 
            {"max_memory_gb": 0.1, "device": "cuda"} if torch.cuda.is_available() else None
        ]
        
        for config in configs:
            if config is None:
                continue
                
            print(f"🔧 Тест конфигурации: {config}")
            try:
                cache = HybridTokenCache(
                    brain=None,
                    max_memory_gb=config["max_memory_gb"],
                    device=config["device"]
                )
                print(f"✅ Кэш создан для {config['device']}")
                
                # Тестируем операции
                test_key = "test_key"
                test_data = torch.tensor([1, 2, 3, 4, 5])
                
                cache.put(test_key, test_data)
                retrieved = cache.get(test_key)
                
                if retrieved is not None:
                    print(f"✅ Операции кэша работают")
                else:
                    print(f"❌ Операции кэша не работают")
                    
            except Exception as e:
                print(f"❌ Ошибка конфигурации {config}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка исследования гибридного кэша: {e}")
        return False

def test_multiple_tokenizers():
    """Тестирует работу с несколькими токенизаторами"""
    print("\n🔤 ТЕСТ МНОГИХ ТОКЕНИЗАТОРОВ")
    print("=" * 50)
    
    try:
        from cogniflex.mlearning.local_rugpt3_loader import Localrugpt3largeLoader
        
        # Создаем загрузчики
        loaders = []
        
        # Тестируем разные пути
        paths = [
            "cogniflex_cache/ml_unit/fractal_storage/tokenizers/rugpt3_large_fractal",
            "cogniflex_cache/ml_unit/fractal_storage/models/rugpt3_large_fractal"
        ]
        
        for i, path in enumerate(paths):
            print(f"📦 Токенизатор {i+1}: {path}")
            try:
                loader = Localrugpt3largeLoader(storage_path=path)
                tokenizer = loader.create_tokenizer()
                
                if tokenizer:
                    print(f"✅ Токенизатор {i+1} создан")
                    loaders.append(tokenizer)
                    
                    # Тестируем токенизацию
                    test_text = "Привет мир!"
                    tokens = tokenizer.encode(test_text)
                    print(f"🔤 Токенов: {len(tokens)} из текста '{test_text}'")
                else:
                    print(f"❌ Токенизатор {i+1} не создан")
                    
            except Exception as e:
                print(f"❌ Ошибка токенизатора {i+1}: {e}")
        
        print(f"📊 Всего создано токенизаторов: {len(loaders)}")
        return len(loaders) > 0
        
    except Exception as e:
        print(f"❌ Ошибка теста токенизаторов: {e}")
        return False

def design_hybrid_solution():
    """Проектирует гибридное решение"""
    print("\n🏗️ ПРОЕКТИРОВАНИЕ ГИБРИДНОГО РЕШЕНИЯ")
    print("=" * 50)
    
    solution = {
        "hot_window_vram": {
            "description": "Горячее окно в VRAM для активных моделей",
            "size_gb": 0.5,
            "models": ["primary_model"],
            "benefits": ["Быстрый доступ", "Низкая задержка"],
            "limitations": ["Ограниченный размер", "Требует CUDA"]
        },
        "warm_window_ssd": {
            "description": "Теплое окно на SSD для часто используемых моделей",
            "size_gb": 2.0,
            "models": ["secondary_models"],
            "benefits": ["Быстрее чем HDD", "Больше места"],
            "limitations": ["Медленнее VRAM", "Износ SSD"]
        },
        "cold_storage": {
            "description": "Холодное хранилище для редких моделей",
            "size_gb": "Неограничено",
            "models": ["backup_models"],
            "benefits": ["Много места", "Дешево"],
            "limitations": ["Медленная загрузка", "Высокая задержка"]
        }
    }
    
    print("📋 Архитектура гибридного кеширования:")
    for name, config in solution.items():
        print(f"\n🔸 {name}:")
        for key, value in config.items():
            print(f"   {key}: {value}")
    
    return solution

def main():
    """Основная функция"""
    print("🔍 АНАЛИЗ ПРОБЛЕМЫ CUDA И ГИБРИДНОГО КЕШИРОВАНИЯ")
    print("=" * 70)
    
    # Анализируем текущее состояние
    cuda_available = analyze_cuda_state()
    analyze_memory_usage()
    
    # Тестируем загрузку моделей
    model_ok = test_model_loading()
    
    # Исследуем гибридное кеширование
    hybrid_ok = investigate_hybrid_caching()
    
    # Тестируем токенизаторы
    tokenizers_ok = test_multiple_tokenizers()
    
    # Проектируем решение
    solution = design_hybrid_solution()
    
    print(f"\n{'='*70}")
    print("📊 ИТОГИ АНАЛИЗА")
    print('='*70)
    print(f"CUDA доступна: {'✅' if cuda_available else '❌'}")
    print(f"Загрузка модели: {'✅' if model_ok else '❌'}")
    print(f"Гибридный кэш: {'✅' if hybrid_ok else '❌'}")
    print(f"Множественные токенизаторы: {'✅' if tokenizers_ok else '❌'}")
    
    if not model_ok:
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        print(f"1. Принудительно использовать CPU для стабильности")
        print(f"2. Реализовать гибридное кеширование с SSD")
        print(f"3. Создать менеджер горячих окон с переключением устройств")

if __name__ == "__main__":
    main()
