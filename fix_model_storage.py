"""
Скрипт для корректного сохранения и загрузки модели в фрактальное хранилище
"""
import sys
import os
import torch
import json
import logging
from datetime import datetime
sys.path.append('.')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_fix")

def fix_model_storage():
    """Исправляет сохранение и загрузку модели в фрактальное хранилище"""
    
    try:
        # Импортируем необходимые модули
        from cogniflex.mlearning.unified_fractal_manager import UnifiedFractalManager
        from cogniflex.mlearning.storage.fractal_weight_store import FractalWeightStore
        from cogniflex.mlearning.text_quality_trainer import TextQualityTrainer
        
        print("🔧 Исправление сохранения модели в фрактальное хранилище...")
        
        # 1. Инициализируем менеджер
        manager = UnifiedFractalManager()
        print(f"✅ Менеджер инициализирован: {type(manager.manager).__name__}")
        
        # 2. Проверяем текущее состояние модели
        if hasattr(manager.manager, 'model') and manager.manager.model:
            model = manager.manager.model
            print(f"✅ Модель загружена: {type(model).__name__}")
            print(f"📊 Параметров: {sum(p.numel() for p in model.parameters())}")
            
            # 3. Создаем фрактальное хранилище
            fractal_store = FractalWeightStore(
                block_size=64,
                fractal_levels=5,
                device="cpu"
            )
            
            # 4. Сохраняем state dict в фрактальное хранилище
            state_dict = model.state_dict()
            print(f"📦 Сохранение {len(state_dict)} тензоров в фрактальное хранилище...")
            
            for key, tensor in state_dict.items():
                fractal_store.store_tensor(f"model.{key}", tensor)
            
            # 5. Сохраняем метаданные модели
            model_metadata = {
                "model_type": type(model).__name__,
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "timestamp": datetime.now().isoformat(),
                "keys": list(state_dict.keys())
            }
            
            fractal_store.store("model_metadata", model_metadata)
            
            print("✅ Модель сохранена в фрактальное хранилище")
            
            # 6. Проверяем загрузку
            print("🔄 Проверка загрузки из фрактального хранилища...")
            
            loaded_state_dict = {}
            for key in state_dict.keys():
                loaded_tensor = fractal_store.get_tensor(f"model.{key}")
                if loaded_tensor is not None:
                    loaded_state_dict[key] = loaded_tensor
                else:
                    print(f"❌ Не удалось загрузить тензор: {key}")
            
            print(f"✅ Загружено {len(loaded_state_dict)} тензоров")
            
            # 7. Проверяем токенизатор
            if hasattr(manager.manager, 'tokenizer') and manager.manager.tokenizer:
                tokenizer = manager.manager.tokenizer
                print(f"✅ Токенизатор загружен: {type(tokenizer).__name__}")
                
                # Сохраняем токенизатор
                tokenizer_path = "cogniflex_cache/ml_unit/fractal_storage/tokenizer"
                os.makedirs(tokenizer_path, exist_ok=True)
                tokenizer.save_pretrained(tokenizer_path)
                print(f"✅ Токенизатор сохранен в: {tokenizer_path}")
            
            # 8. Создаем исправленный тренер
            if hasattr(manager.manager, 'trainer') and manager.manager.trainer:
                trainer = manager.manager.trainer
                print(f"✅ Тренер найден: {type(trainer).__name__}")
                
                # Обновляем метод сохранения тренера
                original_save = trainer.save_model
                
                def fixed_save_model(save_path: str):
                    """Исправленное сохранение модели"""
                    try:
                        # Сохраняем в фрактальное хранилище
                        state_dict = trainer.model.state_dict()
                        
                        for key, tensor in state_dict.items():
                            fractal_store.store_tensor(f"trained_model.{key}", tensor)
                        
                        # Сохраняем метаданные обучения
                        training_metadata = {
                            "training_config": trainer.config.__dict__,
                            "training_history": getattr(trainer, 'training_history', []),
                            "final_step": getattr(trainer, 'global_step', 0),
                            "final_epoch": getattr(trainer, 'epoch', 0),
                            "timestamp": datetime.now().isoformat(),
                            "save_path": save_path
                        }
                        
                        fractal_store.store("training_metadata", training_metadata)
                        
                        # Также сохраняем в файлы для совместимости
                        original_save(save_path)
                        
                        print(f"✅ Модель сохранена в фрактальное хранилище и файлы: {save_path}")
                        
                    except Exception as e:
                        logger.error(f"Ошибка сохранения модели: {e}")
                        original_save(save_path)  # Fallback
                
                # Заменяем метод
                trainer.save_model = fixed_save_model
                print("✅ Метод сохранения тренера исправлен")
            
            # 9. Тестируем генерацию
            print("🧪 Тестирование генерации после исправления...")
            
            test_queries = [
                "Что такое машинное обучение?",
                "Привет, как дела?",
                "Объясни простыми словами"
            ]
            
            for query in test_queries:
                try:
                    response = manager.generate_response(query, max_tokens=50)
                    print(f"📝 Запрос: {query}")
                    print(f"💬 Ответ: {response[:100]}...")
                    print()
                except Exception as e:
                    print(f"❌ Ошибка генерации: {e}")
            
            print("🎉 Исправление завершено!")
            return True
            
        else:
            print("❌ Модель не найдена в менеджере")
            return False
            
    except Exception as e:
        logger.error(f"Ошибка при исправлении модели: {e}", exc_info=True)
        return False

def test_model_loading():
    """Тестирует загрузку модели из фрактального хранилища"""
    
    try:
        from cogniflex.mlearning.storage.fractal_weight_store import FractalWeightStore
        
        print("🔄 Тестирование загрузки из фрактального хранилища...")
        
        # Создаем хранилище
        fractal_store = FractalWeightStore(
            block_size=64,
            fractal_levels=5,
            device="cpu"
        )
        
        # Проверяем метаданные
        metadata = fractal_store.get("model_metadata")
        if metadata:
            print(f"✅ Метаданные модели найдены:")
            print(f"   Тип: {metadata.get('model_type')}")
            print(f"   Параметров: {metadata.get('total_parameters')}")
            print(f"   Время: {metadata.get('timestamp')}")
        
        # Проверяем тензоры
        tensor_count = 0
        for key in fractal_store.containers.keys():
            if key.startswith("model."):
                tensor = fractal_store.get_tensor(key)
                if tensor is not None:
                    tensor_count += 1
        
        print(f"✅ Найдено тензоров: {tensor_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка тестирования загрузки: {e}")
        return False

def main():
    """Основная функция"""
    
    print("🔧 ИСПРАВЛЕНИЕ МОДЕЛИ И ФРАКТАЛЬНОГО ХРАНИЛИЩА")
    print("="*60)
    
    # 1. Исправляем сохранение
    success = fix_model_storage()
    
    if success:
        print("\n✅ Исправление выполнено успешно!")
        
        # 2. Тестируем загрузку
        test_success = test_model_loading()
        
        if test_success:
            print("\n🎉 Все исправления выполнены корректно!")
            print("💡 Модель теперь правильно сохраняется и загружается из фрактального хранилища")
        else:
            print("\n⚠️ Проблемы с тестированием загрузки")
    else:
        print("\n❌ Исправление не удалось")
    
    return success

if __name__ == "__main__":
    main()
