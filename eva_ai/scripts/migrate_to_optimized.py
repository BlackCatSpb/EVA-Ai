"""
Скрипт миграции на OptimizedFractalModelManager
"""
import os
import sys
import logging

logger = logging.getLogger(__name__)
import json

# Добавляем путь к ЕВА
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def migrate_to_optimized():
    """Выполняет миграцию на оптимизированный менеджер"""
    
    logger.info("🔄 Миграция на OptimizedFractalModelManager")
    logger.info("=" * 50)
    
    try:
        # 1. Тестирование оптимизированного менеджера
        logger.info("\n🧪 Тестирование оптимизированного менеджера...")
        
        from eva_ai.mlearning.optimized_fractal_model_manager import OptimizedFractalModelManager
        
        manager = OptimizedFractalModelManager()
        
        if manager.initialized:
            logger.info("✅ Оптимизированный менеджер успешно инициализирован")
            
            # Тест генерации
            response = manager.generate_response_optimized("Привет, как дела?", max_tokens=50)
            logger.info(f"✅ Тест генерации: {response[:50]}...")
            
            # Статистика
            stats = manager.get_performance_stats()
            logger.info(f"✅ Статистика: cache_hit_rate={stats.get('cache_hit_rate', 0):.2%}")
            
        else:
            logger.info("❌ Оптимизированный менеджер не инициализирован")
            return False
        
        # 2. Обновление конфигурации
        logger.info("\n⚙️ Обновление конфигурации...")
        
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "config", "unified_config.json"
        )
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            config["manager_selection"]["use_optimized"] = True
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info("✅ Конфигурация обновлена")
        
        # 3. Создание symbolic link для легкого доступа
        logger.info("\n🔗 Создание symbolic link...")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.join(current_dir, "..", "mlearning", "current_manager.py")
        source_path = os.path.join(current_dir, "..", "mlearning", "optimized_fractal_model_manager.py")
        
        if os.path.exists(target_path):
            os.remove(target_path)
        
        # Windows не поддерживает symbolic links, используем копию
        import shutil
        shutil.copy2(source_path, target_path)
        
        logger.info("✅ Symbolic link создан")
        
        logger.info("\n" + "=" * 50)
        logger.info("🎉 МИГРАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
        logger.info("\n📝 Следующие шаги:")
        logger.info("1. Используйте UnifiedFractalManager для автоматического выбора")
        logger.info("2. Или импортируйте OptimizedFractalModelManager напрямую")
        logger.info("3. Проверьте производительность в GUI")
        
        return True
        
    except Exception as e:
        logger.error(f"ERROR: Ошибка миграции: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = migrate_to_optimized()
    
    if success:
        logger.info("\n✅ Миграция завершена успешно!")
    else:
        logger.info("\n❌ Миграция завершилась с ошибками")
    
    logger.info("\nНажмите Enter для выхода...")
    input()
