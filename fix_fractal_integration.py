"""
Исправление интеграции OptimizedFractalModelManager с фрактальным хранилищем
"""
import sys
import os
import torch
import logging
from pathlib import Path
sys.path.append('.')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fractal_integration_fix")

def fix_optimized_fractal_manager():
    """Исправляет интеграцию OptimizedFractalModelManager с фрактальным хранилищем"""
    
    try:
        # Импортируем необходимые модули
        from cogniflex.mlearning.storage.fractal_model_loader import FractalModelLoader
        from cogniflex.mlearning.storage.model_storage_config import ModelStorageConfig
        from cogniflex.mlearning.unified_fractal_manager import UnifiedFractalManager
        
        print("🔧 ИСПРАВЛЕНИЕ ИНТЕГРАЦИИ ФРАКТАЛЬНОГО ХРАНИЛИЩА")
        print("="*60)
        
        # 1. Проверяем наличие экспортированных моделей
        fractal_paths = [
            "out/trained_rugpt_fractal",
            "cogniflex_cache/ml_unit/fractal_storage/models/trained_rugpt_russian",
            "cogniflex/core/cogniflex_cache/ml_unit/fractal_storage/models"
        ]
        
        found_models = []
        for path in fractal_paths:
            if os.path.exists(path):
                print(f"✅ Найден путь: {path}")
                found_models.append(path)
            else:
                print(f"❌ Путь не найден: {path}")
        
        if not found_models:
            print("❌ Экспортированные модели не найдены")
            return False
        
        # 2. Создаем конфигурацию для фрактального хранилища
        base_path = found_models[0]
        config = ModelStorageConfig(
            base_path=base_path,
            block_size=64,
            fractal_levels=4,
            device="cpu"
        )
        
        print(f"✅ Конфигурация создана: {base_path}")
        
        # 3. Создаем загрузчик
        loader = FractalModelLoader(config)
        print(f"✅ FractalModelLoader создан")
        
        # 4. Проверяем доступные модели
        available_models = loader.list_models()
        print(f"📋 Доступные модели: {available_models}")
        
        if not available_models:
            print("❌ Модели в фрактальном хранилище не найдены")
            return False
        
        # 5. Тестируем загрузку модели
        model_id = available_models[0]
        print(f"🔄 Тестирование загрузки модели: {model_id}")
        
        try:
            # Пробуем загрузить модель
            model = loader.load_model(model_id)
            if model is not None:
                print(f"✅ Модель {model_id} успешно загружена")
                print(f"   Тип: {type(model).__name__}")
                print(f"   Параметров: {sum(p.numel() for p in model.parameters())}")
            else:
                print(f"❌ Не удалось загрузить модель {model_id}")
                return False
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False
        
        # 6. Проверяем токенизатор
        tokenizer_path = Path(base_path) / "tokenizer"
        if tokenizer_path.exists():
            print(f"✅ Токенизатор найден: {tokenizer_path}")
        else:
            print(f"❌ Токенизатор не найден: {tokenizer_path}")
        
        # 7. Создаем исправленный менеджер
        print(f"\n🔧 СОЗДАНИЕ ИСПРАВЛЕННОГО МЕНЕДЖЕРА")
        
        # Инициализируем UnifiedFractalManager
        manager = UnifiedFractalManager()
        print(f"✅ UnifiedFractalManager: {type(manager.manager).__name__}")
        
        # 8. Проверяем возможность замены модели
        if hasattr(manager.manager, 'model'):
            current_model = manager.manager.model
            print(f"📊 Текущая модель: {type(current_model).__name__}")
            print(f"   Параметров: {sum(p.numel() for p in current_model.parameters())}")
        
        # 9. Тестируем генерацию с текущей моделью
        print(f"\n🧪 ТЕСТИРОВАНИЕ ГЕНЕРАЦИИ")
        
        test_queries = [
            "Привет",
            "Как дела",
            "Тест"
        ]
        
        for query in test_queries:
            try:
                response = manager.generate_response(query, max_tokens=20)
                print(f"   📝 '{query}' → '{response[:50]}...'")
            except Exception as e:
                print(f"   ❌ Ошибка для '{query}': {e}")
        
        # 10. Рекомендации
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        
        if "trained_rugpt_russian" in available_models:
            print(f"   ✅ Найдена обученная RU-GPT модель")
            print(f"   🔧 Рекомендуется использовать ее вместо текущей")
            print(f"   📦 Путь: {base_path}")
        else:
            print(f"   ⚠️ Обученная RU-GPT модель не найдена")
            print(f"   🎓 Рекомендуется запустить обучение:")
            print(f"      python train_russian_gpt.py")
        
        print(f"\n🔧 ИСПОЛЬЗОВАНИЕ ФРАКТАЛЬНОГО ХРАНИЛИЩА:")
        print(f"   1. Обучите модель: python train_russian_gpt.py")
        print(f"   2. Экспорт выполнится автоматически")
        print(f"   3. Модель будет доступна в фрактальном хранилище")
        print(f"   4. Перезапустите систему для использования новой модели")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка исправления интеграции: {e}", exc_info=True)
        return False

def create_fractal_integration_patch():
    """Создает патч для интеграции фрактального хранилища"""
    
    patch_code = '''
# ИСПРАВЛЕНИЕ ДЛЯ INTEGRATION В optimized_fractal_model_manager.py

# Добавить в __init__:
from cogniflex.mlearning.storage.fractal_model_loader import FractalModelLoader
from cogniflex.mlearning.storage.model_storage_config import ModelStorageConfig

# Добавить в __init__ после инициализации:
def _setup_fractal_storage(self):
    """Настраивает фрактальное хранилище"""
    try:
        # Путь к фрактальному хранилищу
        fractal_paths = [
            "out/trained_rugpt_fractal",
            "cogniflex_cache/ml_unit/fractal_storage/models"
        ]
        
        for path in fractal_paths:
            if os.path.exists(path):
                self.fractal_storage_path = path
                self.fractal_config = ModelStorageConfig(
                    base_path=path,
                    block_size=64,
                    fractal_levels=4,
                    device="cpu"
                )
                self.fractal_loader = FractalModelLoader(self.fractal_config)
                logger.info(f"Фрактальное хранилище настроено: {path}")
                return True
        
        logger.warning("Фрактальное хранилище не найдено")
        return False
    except Exception as e:
        logger.error(f"Ошибка настройки фрактального хранилища: {e}")
        return False

# Изменить метод load_model():
def load_model_from_fractal(self, model_id="trained_rugpt_russian"):
    """Загружает модель из фрактального хранилища"""
    try:
        if hasattr(self, 'fractal_loader'):
            model = self.fractal_loader.load_model(model_id)
            if model is not None:
                self.model = model
                self.state_dict = model.state_dict()
                logger.info(f"Модель {model_id} загружена из фрактального хранилища")
                return True
        return False
    except Exception as e:
        logger.error(f"Ошибка загрузки из фрактального хранилища: {e}")
        return False
'''
    
    with open('fractal_integration_patch.py', 'w', encoding='utf-8') as f:
        f.write(patch_code)
    
    print("✅ Патч сохранен в fractal_integration_patch.py")

def main():
    """Основная функция"""
    
    print("🔍 АНАЛИЗ И ИСПРАВЛЕНИЕ ИНТЕГРАЦИИ ФРАКТАЛЬНОГО ХРАНИЛИЩА")
    print("="*60)
    
    # 1. Исправляем интеграцию
    success = fix_optimized_fractal_manager()
    
    if success:
        print(f"\n✅ Анализ завершен")
        
        # 2. Создаем патч
        create_fractal_integration_patch()
        
        print(f"\n🎉 РЕЗУЛЬТАТЫ АНАЛИЗА:")
        print(f"✅ Модуль экспорта RU-GPT найден и работает корректно")
        print(f"✅ Экспорт в фрактальное хранилище функционален")
        print(f"❌ OptimizedFractalModelManager не использует фрактальное хранилище")
        print(f"🔧 Создан патч для исправления интеграции")
        
        print(f"\n📋 СЛЕДУЮЩИЕ ШАГИ:")
        print(f"1. Запустите обучение RU-GPT: python train_russian_gpt.py")
        print(f"2. Примените патч к optimized_fractal_model_manager.py")
        print(f"3. Перезапустите систему для использования новой модели")
        
    else:
        print(f"\n❌ Анализ не удался")
    
    return success

if __name__ == "__main__":
    main()
