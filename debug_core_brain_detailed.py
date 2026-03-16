#!/usr/bin/env python3
"""
Детальное логирование CoreBrain для диагностики проблем
"""
import sys
import os
import logging
import traceback
from datetime import datetime

# Добавляем путь к CogniFlex
sys.path.append('.')

def setup_detailed_logging():
    """Настраивает детальное логирование"""
    # Создаем директорию для логов
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Имя файла с временной меткой
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"cogniflex_detailed_{timestamp}.log")
    
    # Настраиваем формат логирования
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Устанавливаем уровень для всех логгеров
    loggers = [
        'cogniflex.core_brain',
        'cogniflex.core_brain.query_processing',
        'cogniflex.mlearning.local_rugpt3_loader',
        'cogniflex.mlearning.fractal_model_manager',
        'cogniflex.mlearning.fractal_rugpt3_manager',
        'cogniflex.mlearning.enhanced_rugpt3_manager',
        'cogniflex.ml_unit',
        'cogniflex.component_initializer',
        'cogniflex.neuromorphic',
        'cogniflex.adaptation',
        'cogniflex.ethics',
        'cogniflex.contradiction',
        'cogniflex.analytics',
        'cogniflex.websearch',
        'cogniflex.memory',
        'cogniflex.knowledge'
    ]
    
    for logger_name in loggers:
        logger_obj = logging.getLogger(logger_name)
        logger_obj.setLevel(logging.DEBUG)
    
    print(f"📝 Детальное логирование настроено. Лог файл: {log_file}")
    return log_file

def test_ml_unit_models():
    """Тестирует какие модели видит ML Unit"""
    print("\n🔍 ТЕСТИРОВАНИЕ ML UNIT МОДЕЛЕЙ")
    print("=" * 50)
    
    try:
        from cogniflex.mlearning.ml_unit import MLUnit
        
        # Создаем ML Unit
        ml_unit = MLUnit()
        
        print(f"📊 ML Unit создан: {type(ml_unit)}")
        print(f"📋 Доступные атрибуты: {[attr for attr in dir(ml_unit) if not attr.startswith('_')]}")
        
        # Проверяем модели
        if hasattr(ml_unit, 'models'):
            print(f"🤖 Модели в ML Unit: {list(ml_unit.models.keys())}")
            for name, model in ml_unit.models.items():
                print(f"   - {name}: {type(model)}")
        else:
            print("❌ У ML Unit нет атрибута 'models'")
        
        # Проверяем фрактальные модели
        if hasattr(ml_unit, 'fractal_models'):
            print(f"🧊 Фрактальные модели: {list(ml_unit.fractal_models.keys())}")
        else:
            print("❌ У ML Unit нет атрибута 'fractal_models'")
        
        # Проверяем доступные модели
        if hasattr(ml_unit, 'get_available_models'):
            available = ml_unit.get_available_models()
            print(f"✅ Доступные модели: {available}")
        else:
            print("❌ У ML Unit нет метода 'get_available_models'")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования ML Unit: {e}")
        traceback.print_exc()
        return False

def test_fractal_model_manager():
    """Тестирует FractalModelManager"""
    print("\n🧊 ТЕСТИРОВАНИЕ FRACTAL MODEL MANAGER")
    print("=" * 50)
    
    try:
        from cogniflex.mlearning.fractal_model_manager import FractalModelManager
        
        # Создаем конфигурацию
        config = {
            "model_name": "rugpt3large",
            "device": "auto"
        }
        
        print(f"🔧 Конфигурация: {config}")
        
        # Создаем менеджер
        manager = FractalModelManager(config=config)
        
        print(f"📊 FractalModelManager создан: {type(manager)}")
        print(f"📋 Атрибуты: {[attr for attr in dir(manager) if not attr.startswith('_')]}")
        
        # Проверяем инициализацию
        if hasattr(manager, 'initialized'):
            print(f"✅ Инициализирован: {manager.initialized}")
        else:
            print("❌ Нет атрибута 'initialized'")
        
        # Проверяем токенизатор
        if hasattr(manager, 'tokenizer'):
            print(f"🔤 Токенизатор: {type(manager.tokenizer)}")
        else:
            print("❌ Нет токенизатора")
        
        # Проверяем модель
        if hasattr(manager, 'model'):
            print(f"🤖 Модель: {type(manager.model)}")
        else:
            print("❌ Нет модели")
        
        # Проверяем устройство
        if hasattr(manager, 'device'):
            print(f"💻 Устройство: {manager.device}")
        else:
            print("❌ Нет устройства")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования FractalModelManager: {e}")
        traceback.print_exc()
        return False

def test_local_rugpt3_loader():
    """Тестирует локальный загрузчик ruGPT-3"""
    print("\n📦 ТЕСТИРОВАНИЕ LOCAL RUGPT3 LOADER")
    print("=" * 50)
    
    try:
        from cogniflex.mlearning.local_rugpt3_loader import Localrugpt3largeLoader
        
        # Проверяем пути
        storage_paths = [
            "cogniflex_cache/ml_unit/fractal_storage/tokenizers/rugpt3_large_fractal",
            "cogniflex_cache/ml_unit/fractal_storage/models/rugpt3_large_fractal"
        ]
        
        print("🔍 Проверка путей:")
        for path in storage_paths:
            abs_path = os.path.abspath(path)
            exists = os.path.exists(abs_path)
            print(f"   {path}: {'✅' if exists else '❌'} ({abs_path})")
        
        # Пытаемся создать загрузчик
        try:
            loader = Localrugpt3largeLoader(
                storage_path="cogniflex_cache/ml_unit/fractal_storage/tokenizers/rugpt3_large_fractal"
            )
            print(f"✅ Загрузчик создан: {type(loader)}")
            
            # Проверяем атрибуты
            if hasattr(loader, 'storage_path'):
                print(f"📂 Storage path: {loader.storage_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка создания загрузчика: {e}")
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Ошибка импорта: {e}")
        traceback.print_exc()
        return False

def test_core_brain_initialization():
    """Тестирует инициализацию CoreBrain с детальным логированием"""
    print("\n🧠 ТЕСТИРОВАНИЕ CORE BRAIN ИНИЦИАЛИЗАЦИИ")
    print("=" * 50)
    
    try:
        from cogniflex.core.core_brain import CoreBrain
        
        print("🔧 Создание CoreBrain...")
        brain = CoreBrain()
        
        print(f"✅ CoreBrain создан: {type(brain)}")
        
        # Проверяем компоненты до инициализации
        print(f"📋 Компонентов до инициализации: {len(brain.components)}")
        for name in brain.components:
            print(f"   - {name}: {type(brain.components[name])}")
        
        print("\n🚀 Запуск инициализации...")
        init_result = brain.initialize()
        
        print(f"📊 Результат инициализации: {init_result}")
        
        # Проверяем компоненты после инициализации
        print(f"📋 Компонентов после инициализации: {len(brain.components)}")
        for name, component in brain.components.items():
            print(f"   - {name}: {type(component)}")
            if hasattr(component, 'initialized'):
                print(f"     Инициализирован: {component.initialized}")
            if hasattr(component, 'state'):
                print(f"     Состояние: {component.state}")
        
        # Проверяем ML Unit
        if hasattr(brain, 'ml_unit'):
            print(f"\n🤖 ML Unit в brain: {type(brain.ml_unit)}")
            if hasattr(brain.ml_unit, 'get_available_models'):
                models = brain.ml_unit.get_available_models()
                print(f"   Доступные модели: {models}")
        
        # Проверяем фрактальный менеджер
        if hasattr(brain, 'fractal_model_manager'):
            print(f"\n🧊 FractalModelManager в brain: {type(brain.fractal_model_manager)}")
            if hasattr(brain.fractal_model_manager, 'initialized'):
                print(f"   Инициализирован: {brain.fractal_model_manager.initialized}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка инициализации CoreBrain: {e}")
        traceback.print_exc()
        return False

def main():
    """Основная функция"""
    print("🚀 ДЕТАЛЬНАЯ ДИАГНОСТИКА COGNIFLEX")
    print("=" * 70)
    
    # Настраиваем логирование
    log_file = setup_detailed_logging()
    
    # Тестируем компоненты
    tests = [
        ("ML Unit модели", test_ml_unit_models),
        ("Fractal Model Manager", test_fractal_model_manager),
        ("Local ruGPT-3 Loader", test_local_rugpt3_loader),
        ("Core Brain инициализация", test_core_brain_initialization),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*70}")
        print(f"🧪 ТЕСТ: {test_name}")
        print('='*70)
        
        try:
            result = test_func()
            results[test_name] = result
            print(f"✅ Тест '{test_name}': {'УСПЕХ' if result else 'ПРОВАЛ'}")
        except Exception as e:
            results[test_name] = False
            print(f"❌ Тест '{test_name}' завершился с ошибкой: {e}")
            traceback.print_exc()
    
    # Итоги
    print(f"\n{'='*70}")
    print("📊 ИТОГИ ТЕСТИРОВАНИЯ")
    print('='*70)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ УСПЕХ" if result else "❌ ПРОВАЛ"
        print(f"{test_name}: {status}")
    
    print(f"\n📈 Пройдено тестов: {passed}/{total}")
    print(f"📝 Детальный лог: {log_file}")
    
    if passed == total:
        print("🎉 ВСЕ ТЕСТЫ УСПЕШНЫ!")
    else:
        print("⚠️ ЕСТЬ ПРОБЛЕМЫ, ТРЕБУЮЩИЕ ВНИМАНИЯ")

if __name__ == "__main__":
    main()
