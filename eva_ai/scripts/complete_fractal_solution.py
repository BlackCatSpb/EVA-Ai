"""
Полное решение для очистки старых моделей и создания уникального фрактального токенизатора
"""
import sys
import os
import torch
import shutil
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
sys.path.append('.')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fractal_cleanup")

def cleanup_old_models():
    """Очищает старые модели и связанные файлы"""
    
    try:
        logger.info("🧹 ОЧИСТКА СТАРЫХ МОДЕЛЕЙ И ФАЙЛОВ")
        logger.info("="*50)
        
        # Пути для очистки
        cleanup_paths = [
            "out/rugpt3_small_fractal",
            "out/rugpt3_simple_fractal", 
            "eva_cache/ml_unit/fractal_storage/models/trained_rugpt_russian",
            "eva/core/eva_cache/ml_unit/fractal_storage/models",
            "training_output",
            "out/trained_rugpt_fractal"
        ]
        
        cleaned_count = 0
        total_size_freed = 0
        
        for path_str in cleanup_paths:
            path = Path(path_str)
            if path.exists():
                try:
                    # Считаем размер перед удалением
                    size_bytes = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                    size_mb = size_bytes / (1024 * 1024)
                    
                    logger.info(f"🗑️ Удаление: {path}")
                    logger.info(f"   📁 Размер: {size_mb:.1f} MB")
                    
                    # Удаляем директорию
                    shutil.rmtree(path)
                    cleaned_count += 1
                    total_size_freed += size_mb
                    
                    logger.info(f"   ✅ Удалено")
                    
                except Exception as e:
                    logger.error(f"   ERROR: Ошибка удаления {path}: {e}")
            else:
                logger.info(f"   ⚪ Путь не найден: {path}")
        
        logger.info(f"\n📊 ИТОГИ ОЧИСТКИ:")
        logger.info(f"   🗑️ Удалено директорий: {cleaned_count}")
        logger.info(f"   💾 Освобождено места: {total_size_freed:.1f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка очистки: {e}")
        return False

def create_unique_fractal_tokenizer():
    """Создает уникальный фрактальный токенизатор"""
    
    try:
        from transformers import AutoTokenizer, GPT2Tokenizer, GPT2TokenizerFast
        
        logger.info("🔤 СОЗДАНИЕ УНИКАЛЬНОГО ФРАКТАЛЬНОГО ТОКЕНИЗАТОРА")
        logger.info("="*50)
        
        # 1. Генерируем уникальный ID
        timestamp = datetime.now().isoformat()
        unique_id = hashlib.md5(timestamp.encode()).hexdigest()[:8]
        model_name = f"fractal_rugpt3_{unique_id}"
        
        logger.info(f"🆔 Уникальный ID модели: {model_name}")
        
        # 2. Создаем расширенный токенизатор на основе GPT-2
        base_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        
        # 3. Добавляем русские токены
        russian_tokens = [
            # Специальные токены для фрактальной структуры
            "<fractal_start>", "<fractal_end>", "<fractal_node>", "<fractal_edge>",
            "<fractal_level>", "<fractal_block>", "<fractal_compress>",
            "<fractal_expand>", "<fractal_memory>", "<fractal_cache>",
            
            # Русские слова и символы
            "Привет", "Здравствуйте", "Спасибо", "Пожалуйста", "Да", "Нет",
            "Что", "Как", "Почему", "Где", "Когда", "Кто", "Чей",
            "Россия", "Москва", "Санкт-Петербург", "Русский", "Язык",
            "Искусственный", "Интеллект", "Машина", "Обучение", "Нейронная", "Сеть",
            "Трансформер", "Модель", "Данные", "Алгоритм", "Программа", "Код",
            
            # Расширенные русские символы
            "ё", "Ё", "й", "Й", "ц", "Ц", "у", "У", "к", "К",
            "е", "Е", "н", "Н", "г", "Г", "ш", "Ш", "щ", "Щ",
            "з", "З", "х", "Х", "ъ", "Ъ", "ф", "Ф", "ы", "Ы",
            "в", "В", "а", "А", "п", "П", "р", "Р", "о", "О",
            "л", "Л", "д", "Д", "ж", "Ж", "э", "Э", "я", "Я",
            "ч", "Ч", "с", "С", "м", "М", "и", "И", "т", "Т",
            "ь", "Ь", "б", "Б", "ю", "Ю"
        ]
        
        # 4. Добавляем токены в токенизатор
        added_tokens = 0
        for token in russian_tokens:
            if token not in base_tokenizer.get_vocab():
                base_tokenizer.add_tokens([token])
                added_tokens += 1
        
        logger.info(f"➕ Добавлено русских токенов: {added_tokens}")
        
        # 5. Устанавливаем специальные токены
        special_tokens = {
            "pad_token": "<pad>",
            "eos_token": "<|endoftext|>",
            "bos_token": "<|startoftext|>",
            "unk_token": "<unk>",
            "sep_token": "<sep>",
            "mask_token": "<mask>"
        }
        
        base_tokenizer.add_special_tokens(special_tokens)
        
        # 6. Создаем директорию для токенизатора
        tokenizer_dir = Path("out") / f"{model_name}_tokenizer"
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        
        # 7. Сохраняем токенизатор
        base_tokenizer.save_pretrained(str(tokenizer_dir))
        
        logger.info(f"✅ Токенизатор сохранен: {tokenizer_dir}")
        logger.info(f"   📊 Размер словаря: {len(base_tokenizer.get_vocab()):,}")
        logger.info(f"   🔤 Специальных токенов: {len(base_tokenizer.all_special_tokens)}")
        
        # 8. Создаем метаданные токенизатора
        tokenizer_metadata = {
            "model_name": model_name,
            "tokenizer_class": "GPT2TokenizerFast",
            "vocab_size": len(base_tokenizer.get_vocab()),
            "special_tokens": base_tokenizer.all_special_tokens,
            "added_tokens": added_tokens,
            "russian_tokens": len(russian_tokens),
            "created_at": timestamp,
            "unique_id": unique_id,
            "fractal_features": [
                "fractal_start", "fractal_end", "fractal_node", "fractal_edge",
                "fractal_level", "fractal_block", "fractal_compress",
                "fractal_expand", "fractal_memory", "fractal_cache"
            ]
        }
        
        with open(tokenizer_dir / "fractal_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(tokenizer_metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Метаданные токенизатора сохранены")
        
        return model_name, str(tokenizer_dir), base_tokenizer
        
    except Exception as e:
        logger.error(f"Ошибка создания токенизатора: {e}", exc_info=True)
        return None, None, None

def download_and_export_rugpt3_with_custom_tokenizer():
    """Загружает ruGPT3 и экспортирует с кастомным токенизатором"""
    
    try:
        from transformers import AutoModelForCausalLM
        from eva_ai.mlearning.storage.fractal_store import export_hf_model_to_fractal
        
        logger.info("🚀 ЗАГРУЗКА И ЭКСПОРТ RU-GPT3 С КАСТОМНЫМ ТОКЕНИЗАТОРОМ")
        logger.info("="*60)
        
        # 1. Создаем кастомный токенизатор
        model_name, tokenizer_path, custom_tokenizer = create_unique_fractal_tokenizer()
        
        if not model_name or not custom_tokenizer:
            logger.info("❌ Не удалось создать токенизатор")
            return False
        
        # 2. Загружаем ruGPT3 модель
        model_id = "sberbank-ai/rugpt3large_based_on_gpt2"
        
        logger.info(f"\n📦 Загрузка модели: {model_id}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # Используем float32 для стабильности
            low_cpu_mem_usage=True
        )
        
        model.eval()
        
        logger.info(f"✅ Модель загружена")
        logger.info(f"   📊 Параметров: {sum(p.numel() for p in model.parameters()):,}")
        
        # 3. Адаптируем эмбеддинги под новый токенизатор
        logger.info(f"\n🔧 Адаптация эмбеддингов...")
        
        old_vocab_size = model.get_input_embeddings().weight.shape[0]
        new_vocab_size = len(custom_tokenizer.get_vocab())
        
        logger.info(f"   📊 Старый vocab: {old_vocab_size}")
        logger.info(f"   📊 Новый vocab: {new_vocab_size}")
        
        if new_vocab_size != old_vocab_size:
            # Изменяем размер эмбеддингов
            model.resize_token_embeddings(new_vocab_size)
            logger.info(f"✅ Эмбеддинги адаптированы")
        
        # 4. Тестируем генерацию
        logger.info(f"\n🧪 Тестирование генерации...")
        
        test_queries = [
            "Привет, как дела?",
            "Что такое искусственный интеллект?",
            "Расскажи о России"
        ]
        
        for query in test_queries:
            try:
                inputs = custom_tokenizer.encode(query, return_tensors='pt')
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 30,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=custom_tokenizer.pad_token_id
                    )
                
                response = custom_tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"   📝 '{query}'")
                logger.info(f"   💬 '{response}'")
                logger.info()
                
            except Exception as e:
                logger.error(f"   ERROR: Ошибка: {e}")
        
        # 5. Сохраняем модель временно для экспорта
        temp_model_dir = Path("out") / f"{model_name}_temp"
        temp_model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n💾 Сохранение модели для экспорта...")
        
        # Сохраняем модель
        model.save_pretrained(str(temp_model_dir))
        
        # Сохраняем кастомный токенизатор
        custom_tokenizer.save_pretrained(str(temp_model_dir))
        
        logger.info(f"✅ Модель сохранена: {temp_model_dir}")
        
        # 6. Экспортируем во фрактальное хранилище
        logger.info(f"\n📦 Экспорт во фрактальное хранилище...")
        
        output_path = Path("out") / f"{model_name}_fractal"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Используем оптимизированные параметры для экспорта
        success = export_hf_model_to_fractal(
            hf_model_dir_or_id=str(temp_model_dir),
            output_path=str(output_path),
            model_id=model_name,
            tokenizer_output_subdir="tokenizer",
            device="cpu",
            fractal_levels=3,      # Уменьшаем уровни для экономии памяти
            block_size=32,         # Уменьшаем размер блока
            local_files_only=True
        )
        
        if success:
            logger.info(f"✅ Модель экспортирована во фрактальное хранилище")
            
            # Проверяем структуру
            logger.info(f"\n📋 Структура экспорта:")
            
            file_count = 0
            total_size = 0
            
            for item in output_path.rglob("*"):
                if item.is_file():
                    size_mb = item.stat().st_size / (1024 * 1024)
                    total_size += size_mb
                    file_count += 1
                    
                    if item.suffix in ['.json', '.txt']:
                        logger.info(f"   📄 {item.relative_to(output_path)} ({size_mb:.1f} MB)")
            
            logger.info(f"   📁 Всего файлов: {file_count}")
            logger.info(f"   💾 Общий размер: {total_size:.1f} MB")
            
            # 7. Очищаем временные файлы
            try:
                shutil.rmtree(temp_model_dir)
                logger.info(f"🧹 Временные файлы очищены")
            except Exception as e:
                logger.error(f"WARNING: Ошибка очистки временных файлов: {e}")
            
            return True
        else:
            logger.info(f"❌ Экспорт не удался")
            return False
            
    except Exception as e:
        logger.error(f"Ошибка экспорта: {e}", exc_info=True)
        return False

def create_fractal_integration():
    """Создает интеграцию с фрактальным хранилищем"""
    
    try:
        integration_code = '''
# ИНТЕГРАЦИЯ ФРАКТАЛЬНОГО ХРАНИЛИЩА ДЛЯ ОПТИМИЗИРОВАННОГО МЕНЕДЖЕРА

import os
import logging
from pathlib import Path

def setup_fractal_integration(self):
    """Настраивает интеграцию с фрактальным хранилищем"""
    try:
        # Пути к фрактальным моделям
        fractal_paths = [
            "out/fractal_rugpt3_*_fractal",
            "eva_cache/ml_unit/fractal_storage/models"
        ]
        
        # Ищем доступную модель
        found_model = None
        for path_pattern in fractal_paths:
            if "*" in path_pattern:
                # Ищем по шаблону
                base_path = Path(path_pattern.split("*")[0])
                if base_path.exists():
                    for item in base_path.parent.glob(path_pattern.split("*")[1] + "*"):
                        if item.is_dir():
                            found_model = str(item)
                            break
            else:
                if os.path.exists(path_pattern):
                    found_model = path_pattern
                    break
            
            if found_model:
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
            from eva_ai.mlearning.storage.fractal_model_loader import FractalModelLoader
            from eva_ai.mlearning.storage.model_storage_config import ModelStorageConfig
            
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
        
        with open("fractal_integration.py", 'w', encoding='utf-8') as f:
            f.write(integration_code)
        
        logger.info("✅ Интеграционный код сохранен: fractal_integration.py")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка создания интеграции: {e}")
        return False

def main():
    """Основная функция"""
    
    logger.info("🚀 ПОЛНОЕ РЕШЕНИЕ: ОЧИСТКА + УНИКАЛЬНЫЙ ТОКЕНИЗАТОР + ЭКСПОРТ")
    logger.info("="*70)
    
    # 1. Очищаем старые модели
    logger.info("ШАГ 1: ОЧИСТКА СТАРЫХ МОДЕЛЕЙ")
    cleanup_success = cleanup_old_models()
    
    if not cleanup_success:
        logger.info("❌ Очистка не удалась")
        return False
    
    # 2. Создаем уникальный токенизатор и экспортируем модель
    logger.info("\nШАГ 2: СОЗДАНИЕ УНИКАЛЬНОГО ТОКЕНИЗАТОРА И ЭКСПОРТ")
    export_success = download_and_export_rugpt3_with_custom_tokenizer()
    
    if not export_success:
        logger.info("❌ Экспорт не удался")
        return False
    
    # 3. Создаем интеграцию
    logger.info("\nШАГ 3: СОЗДАНИЕ ИНТЕГРАЦИИ")
    integration_success = create_fractal_integration()
    
    if integration_success:
        logger.info("\n🎉 ВСЕ ОПЕРАЦИИ УСПЕШНЫ!")
        logger.info("✅ Старые модели очищены")
        logger.info("✅ Уникальный фрактальный токенизатор создан")
        logger.info("✅ ruGPT3 экспортирована во фрактальное хранилище")
        logger.info("✅ Интеграционный код создан")
        
        logger.info("\n📋 СЛЕДУЮЩИЕ ШАГИ:")
        logger.info("1. Примените интеграционный код к OptimizedFractalModelManager")
        logger.info("2. Перезапустите систему")
        logger.info("3. Система будет использовать новую фрактальную модель")
        
        return True
    else:
        logger.info("❌ Создание интеграции не удалось")
        return False

if __name__ == "__main__":
    main()
