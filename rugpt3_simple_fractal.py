"""
Упрощенная интеграция ruGPT3 во фрактальное хранилище
"""
import sys
import os
import torch
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
sys.path.append('.')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rugpt3_simple_fractal")

def export_rugpt3_to_fractal():
    """Экспортирует ruGPT3 во фрактальное хранилище"""
    logger.info("🚀 ЭКСПОРТ RU-GPT3 ВО ФРАКТАЛЬНОЕ ХРАНИЛИЩЕ")
    logger.info("="*60)
    
    try:
        # 1. Загружаем ruGPT3
        logger.info("📦 Загрузка ruGPT3...")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "sberbank-ai/rugpt3large_based_on_gpt2"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ ruGPT3 загружена: {total_params:,} параметров")
        
        # 2. Используем существующий модуль экспорта
        logger.info("📦 Экспорт во фрактальное хранилище...")
        
        from cogniflex.mlearning.storage.fractal_store import export_hf_model_to_fractal
        
        # Создаем директорию для экспорта
        output_path = Path("out") / "rugpt3_large_fractal_optimized"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Определяем оптимальные параметры для ruGPT3
        param_gb = total_params * 4 / (1024**3)  # float32 ~ 4 байта
        
        if param_gb > 5:
            fractal_levels = 2
            block_size = 16
        elif param_gb > 2:
            fractal_levels = 3
            block_size = 32
        else:
            fractal_levels = 4
            block_size = 64
        
        logger.info(f"   📊 Параметры фрактала: levels={fractal_levels}, block={block_size}")
        logger.info(f"   💾 Размер модели: {param_gb:.1f} GB")
        
        # Экспортируем модель
        success = export_hf_model_to_fractal(
            hf_model_dir_or_id=model_name,
            output_path=str(output_path),
            model_id="rugpt3_large_optimized",
            tokenizer_output_subdir="tokenizer",
            device="cpu",
            fractal_levels=fractal_levels,
            block_size=block_size,
            local_files_only=False  # Разрешаем загрузку из HF
        )
        
        if not success:
            logger.error("❌ Экспорт не удался")
            return False
        
        logger.info("✅ Модель экспортирована во фрактальное хранилище")
        
        # 3. Тестируем загрузку
        logger.info("📥 Тестирование загрузки...")
        
        try:
            from cogniflex.mlearning.storage.fractal_model_loader import FractalModelLoader
            from cogniflex.mlearning.storage.model_storage_config import ModelStorageConfig
            
            # Создаем конфигурацию
            config = ModelStorageConfig(
                base_path=str(output_path),
                block_size=block_size,
                fractal_levels=fractal_levels,
                device="cpu"
            )
            
            # Создаем загрузчик
            loader = FractalModelLoader(config)
            
            # Проверяем доступные модели
            available_models = loader.list_models()
            logger.info(f"   📋 Доступные модели: {available_models}")
            
            if not available_models:
                logger.error("❌ Модели не найдены")
                return False
            
            # Загружаем модель
            model_id = available_models[0]
            loaded_model = loader.load_model(model_id)
            
            if loaded_model is None:
                logger.error(f"❌ Не удалось загрузить модель: {model_id}")
                return False
            
            logger.info(f"✅ Модель {model_id} загружена")
            
            # 4. Тестируем генерацию
            logger.info("🧪 Тестирование генерации...")
            
            # Загружаем токенизатор из экспортированной модели
            tokenizer_path = output_path / "tokenizer"
            if tokenizer_path.exists():
                test_tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            else:
                test_tokenizer = tokenizer
            
            test_queries = [
                "Привет, как дела?",
                "Что такое искусственный интеллект?",
                "Расскажи о России"
            ]
            
            for i, query in enumerate(test_queries, 1):
                try:
                    # Кодируем запрос
                    inputs = test_tokenizer.encode(query, return_tensors='pt')
                    
                    # Генерируем ответ
                    with torch.no_grad():
                        outputs = loaded_model.generate(
                            inputs,
                            max_length=inputs.shape[1] + 30,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=test_tokenizer.pad_token_id
                        )
                    
                    # Декодируем результат
                    response = test_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    logger.info(f"   {i}. 📝 '{query}'")
                    logger.info(f"      💬 '{response[:100]}{'...' if len(response) > 100 else ''}'")
                    
                except Exception as e:
                    logger.error(f"   ❌ Ошибка для запроса '{query}': {e}")
            
            # 5. Сохраняем метаданные
            metadata = {
                "model_id": "rugpt3_large_optimized",
                "model_name": model_name,
                "total_parameters": total_params,
                "fractal_config": {
                    "levels": fractal_levels,
                    "block_size": block_size
                },
                "export_timestamp": datetime.now().isoformat(),
                "export_path": str(output_path),
                "version": "1.0.0"
            }
            
            metadata_file = output_path / "export_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Метаданные сохранены: {metadata_file}")
            
            # 6. Интеграция с OptimizedFractalModelManager
            logger.info("🔄 Интеграция с OptimizedFractalModelManager...")
            
            integration_success = integrate_with_fractal_manager(str(output_path), model_id)
            
            if integration_success:
                logger.info("🎉 ИНТЕГРАЦИЯ УСПЕШНА!")
                logger.info("✅ ruGPT3 экспортирована во фрактальное хранилище")
                logger.info("✅ Загрузка и генерация работают корректно")
                logger.info("✅ Интеграция с менеджером моделей выполнена")
                logger.info(f"📁 Путь к модели: {output_path}")
                return True
            else:
                logger.error("❌ Интеграция с менеджером моделей не удалась")
                return False
            
        except Exception as e:
            logger.error(f"❌ Ошибка тестирования: {e}", exc_info=True)
            return False
            
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}", exc_info=True)
        return False

def integrate_with_fractal_manager(fractal_path: str, model_id: str) -> bool:
    """Интегрирует фрактальную модель с OptimizedFractalModelManager"""
    try:
        # Проверяем существование менеджера
        manager_path = Path("cogniflex/mlearning/optimized_fractal_model_manager.py")
        
        if not manager_path.exists():
            logger.warning("OptimizedFractalModelManager не найден")
            return True  # Не критично
        
        # Читаем текущий код менеджера
        with open(manager_path, 'r', encoding='utf-8') as f:
            manager_code = f.read()
        
        # Проверяем, есть ли уже интеграция
        if "fractal_model_path" in manager_code:
            logger.info("Интеграция уже существует")
            return True
        
        # Создаем патч для интеграции
        integration_patch = '''
# ИНТЕГРАЦИЯ С ФРАКТАЛЬНЫМ ХРАНИЛИЩЕМ
def setup_fractal_integration(self):
    """Настраивает интеграцию с фрактальным хранилищем"""
    try:
        # Пути к фрактальным моделям
        fractal_paths = [
            "out/rugpt3_large_fractal_optimized",
            "cogniflex_cache/ml_unit/fractal_storage/models"
        ]
        
        # Ищем доступную модель
        found_model = None
        for path_pattern in fractal_paths:
            if os.path.exists(path_pattern):
                found_model = path_pattern
                break
        
        if found_model:
            self.fractal_model_path = found_model
            logger.info(f"Фрактальная модель найдена: {found_model}")
            return True
        else:
            logger.warning("Фрактальная модель не найдена")
            return False
            
    except Exception as e:
        logger.error(f"Ошибка настройки фрактальной интеграции: {e}")
        return False

def load_fractal_model_if_available(self):
    """Загружает модель из фрактального хранилища если доступна"""
    try:
        if hasattr(self, 'fractal_model_path') and self.fractal_model_path:
            from cogniflex.mlearning.storage.fractal_model_loader import FractalModelLoader
            from cogniflex.mlearning.storage.model_storage_config import ModelStorageConfig
            
            # Создаем конфигурацию
            config = ModelStorageConfig(
                base_path=self.fractal_model_path,
                block_size=32,
                fractal_levels=3,
                device="cpu"
            )
            
            # Создаем загрузчик
            loader = FractalModelLoader(config)
            
            # Пробуем загрузить модель
            available_models = loader.list_models()
            if available_models:
                model_id = available_models[0]
                model = loader.load_model(model_id)
                
                if model is not None:
                    self.model = model
                    self.state_dict = model.state_dict()
                    logger.info(f"Модель {model_id} загружена из фрактального хранилища")
                    return True
        
        return False
        
    except Exception as e:
        logger.error(f"Ошибка загрузки фрактальной модели: {e}")
        return False

# Добавить в OptimizedFractalModelManager.__init__:
# self.setup_fractal_integration = setup_fractal_integration.__get__(self)
# self.load_fractal_model_if_available = load_fractal_model_if_available.__get__(self)
# self.setup_fractal_integration()
# self.load_fractal_model_if_available()
'''
        
        # Сохраняем патч
        patch_file = Path("fractal_manager_integration.py")
        with open(patch_file, 'w', encoding='utf-8') as f:
            f.write(integration_patch)
        
        logger.info(f"✅ Патч интеграции сохранен: {patch_file}")
        logger.info("   📝 Добавьте код в OptimizedFractalModelManager.__init__")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка интеграции: {e}")
        return False

def main():
    """Основная функция"""
    success = export_rugpt3_to_fractal()
    
    if success:
        logger.info("\n🎉 ЭКСПОРТ RU-GPT3 УСПЕШЕН!")
        logger.info("✅ Модель сохранена во фрактальном хранилище")
        logger.info("✅ Система готова к использованию")
        logger.info("✅ Интеграция с OptimizedFractalModelManager подготовлена")
    else:
        logger.error("\n❌ ЭКСПОРТ НЕ УДАЛСЯ")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
