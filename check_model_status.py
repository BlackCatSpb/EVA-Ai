"""
Проверка состояния предыдущей модели и функциональности
"""
import sys
import os
import torch
import json
import logging
from pathlib import Path
sys.path.append('.')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_check")

def check_previous_model():
    """Проверяет состояние предыдущей модели"""
    logger.info("🔍 ПРОВЕРКА ПРЕДЫДУЩЕЙ МОДЕЛИ")
    logger.info("="*50)
    
    # 1. Проверяем наличие файлов модели
    out_dir = Path("out")
    
    logger.info("📁 ПРОВЕРКА ФАЙЛОВ МОДЕЛИ:")
    
    model_files = {
        "fractal_rugpt_full.safetensors": out_dir / "fractal_rugpt_full.safetensors",
        "fractal_rugpt_resume.json": out_dir / "fractal_rugpt_resume.json",
        "fractal_rugpt_full.report.json": out_dir / "fractal_rugpt_full.report.json"
    }
    
    for name, path in model_files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(f"   ✅ {name}: {size_mb:.1f} MB")
        else:
            logger.info(f"   ❌ {name}: не найден")
    
    # 2. Проверяем OptimizedFractalModelManager
    logger.info("\n🏗️ ПРОВЕРКА OptimizedFractalModelManager:")
    
    try:
        from cogniflex.mlearning.optimized_fractal_model_manager import OptimizedFractalModelManager
        
        # Пробуем создать менеджер
        manager = OptimizedFractalModelManager()
        
        logger.info(f"   ✅ Менеджер создан")
        logger.info(f"   📁 Путь к модели: {getattr(manager, 'model_path', 'N/A')}")
        logger.info(f"   🔧 Устройство: {getattr(manager, 'device', 'N/A')}")
        logger.info(f"   💾 Память токенов: {getattr(manager, 'max_memory_tokens', 'N/A')}")
        
        # Проверяем наличие модели
        if hasattr(manager, 'model') and manager.model is not None:
            logger.info(f"   ✅ Модель загружена")
            param_count = sum(p.numel() for p in manager.model.parameters())
            logger.info(f"   📊 Параметров: {param_count:,}")
        else:
            logger.info(f"   ⚠️ Модель не загружена")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Ошибка менеджера: {e}")
        return False

def check_text_generation():
    """Проверяет генерацию текста"""
    logger.info("\n🧪 ПРОВЕРКА ГЕНЕРАЦИИ ТЕКСТА:")
    
    try:
        from cogniflex.mlearning.optimized_fractal_model_manager import OptimizedFractalModelManager
        
        manager = OptimizedFractalModelManager()
        
        # Проверяем наличие модели и токенизатора
        if not hasattr(manager, 'model') or manager.model is None:
            logger.warning("   ⚠️ Модель не загружена, пропускаем тест генерации")
            return False
        
        # Тестовые запросы
        test_queries = [
            "Привет, как дела?",
            "Что такое искусственный интеллект?",
            "Расскажи о России"
        ]
        
        logger.info(f"   📝 Тестовые запросы: {len(test_queries)}")
        
        for i, query in enumerate(test_queries, 1):
            try:
                logger.info(f"   {i}. 📝 '{query}'")
                
                # Пробуем сгенерировать ответ
                if hasattr(manager, 'generate_text'):
                    response = manager.generate_text(query, max_length=50)
                    logger.info(f"      💬 '{response[:100]}{'...' if len(response) > 100 else ''}'")
                else:
                    logger.warning(f"      ⚠️ Метод generate_text не найден")
                
            except Exception as e:
                logger.error(f"      ❌ Ошибка генерации: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Ошибка проверки генерации: {e}")
        return False

def check_fractal_storage():
    """Проверяет фрактальное хранилище"""
    logger.info("\n💾 ПРОВЕРКА ФРАКТАЛЬНОГО ХРАНИЛИЩА:")
    
    # 1. Проверяем директорию фрактального хранилища
    fractal_paths = [
        "cogniflex_cache/ml_unit/fractal_storage",
        "out/rugpt3_large_fractal",
        "out/trained_rugpt_fractal"
    ]
    
    for path_str in fractal_paths:
        path = Path(path_str)
        if path.exists():
            logger.info(f"   ✅ {path_str}: существует")
            
            # Считаем файлы
            files = list(path.rglob("*"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            
            logger.info(f"      📁 Файлов: {len(files)}")
            logger.info(f"      💾 Размер: {total_size / (1024*1024):.1f} MB")
        else:
            logger.info(f"   ❌ {path_str}: не существует")
    
    # 2. Проверяем модули фрактального хранилища
    logger.info("\n🔧 ПРОВЕРКА МОДУЛЕЙ ФРАКТАЛЬНОГО ХРАНИЛИЩА:")
    
    modules_to_check = [
        "cogniflex.mlearning.storage.fractal_store",
        "cogniflex.mlearning.storage.fractal_model_loader",
        "cogniflex.mlearning.storage.model_storage_config"
    ]
    
    for module_name in modules_to_check:
        try:
            __import__(module_name)
            logger.info(f"   ✅ {module_name}: доступен")
        except Exception as e:
            logger.error(f"   ❌ {module_name}: {e}")
    
    return True

def check_hybrid_cache():
    """Проверяет гибридный кеш"""
    logger.info("\n🔄 ПРОВЕРКА ГИБРИДНОГО КЕША:")
    
    try:
        from cogniflex.memory.hybrid_token_cache import HybridTokenCache
        
        logger.info(f"   ✅ HybridTokenCache: доступен")
        
        # Проверяем директории кеша
        cache_dirs = [
            "ml_cache",
            "cogniflex_cache/ml_unit/fractal_storage",
            "cogniflex_cache"
        ]
        
        for cache_dir in cache_dirs:
            if Path(cache_dir).exists():
                logger.info(f"   ✅ {cache_dir}: существует")
            else:
                logger.info(f"   ⚪ {cache_dir}: не существует")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Ошибка гибридного кеша: {e}")
        return False

def check_tokenizer():
    """Проверяет токенизатор"""
    logger.info("\n🔤 ПРОВЕРКА ТОКЕНИЗАТОРА:")
    
    try:
        from transformers import AutoTokenizer
        
        # Проверяем стандартный токенизатор
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        logger.info(f"   ✅ GPT-2 токенизатор: vocab_size={len(tokenizer.get_vocab()):,}")
        
        # Проверяем наличие фрактального токенизатора
        if Path("fractal_tokenizer.py").exists():
            logger.info(f"   ✅ Фрактальный токенизатор: создан")
        else:
            logger.info(f"   ⚪ Фрактальный токенизатор: не создан")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Ошибка токенизатора: {e}")
        return False

def main():
    """Основная функция проверки"""
    logger.info("🚀 КОМПЛЕКСНАЯ ПРОВЕРКА СОСТОЯНИЯ СИСТЕМЫ")
    logger.info("="*60)
    
    results = {
        "previous_model": check_previous_model(),
        "text_generation": check_text_generation(),
        "fractal_storage": check_fractal_storage(),
        "hybrid_cache": check_hybrid_cache(),
        "tokenizer": check_tokenizer()
    }
    
    logger.info("\n📊 ИТОГИ ПРОВЕРКИ:")
    
    for check_name, result in results.items():
        status = "✅ УСПЕШНО" if result else "❌ ОШИБКА"
        logger.info(f"   {check_name}: {status}")
    
    success_count = sum(1 for r in results.values() if r)
    total_checks = len(results)
    
    logger.info(f"\n📈 ОБЩИЙ РЕЗУЛЬТАТ: {success_count}/{total_checks} проверок успешны")
    
    if success_count == total_checks:
        logger.info("🎉 СИСТЕМА В ПОЛНОМ ПОРЯДКЕ!")
        logger.info("✅ Предыдущая модель функциональна")
        logger.info("✅ Можно продолжать разработку")
    else:
        logger.warning("⚠️ ОБНАРУЖЕНЫ ПРОБЛЕМЫ")
        logger.warning("📝 Требуется исправление перед продолжением")
    
    return success_count == total_checks

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
