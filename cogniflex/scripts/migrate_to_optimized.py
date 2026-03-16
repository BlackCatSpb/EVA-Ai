"""
Скрипт миграции на OptimizedFractalModelManager
"""
import os
import sys
import logging
import json

# Добавляем путь к CogniFlex
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def migrate_to_optimized():
    """Выполняет миграцию на оптимизированный менеджер"""
    
    print("🔄 Миграция на OptimizedFractalModelManager")
    print("=" * 50)
    
    try:
        # 1. Тестирование оптимизированного менеджера
        print("\n🧪 Тестирование оптимизированного менеджера...")
        
        from cogniflex.mlearning.optimized_fractal_model_manager import OptimizedFractalModelManager
        
        manager = OptimizedFractalModelManager()
        
        if manager.initialized:
            print("✅ Оптимизированный менеджер успешно инициализирован")
            
            # Тест генерации
            response = manager.generate_response_optimized("Привет, как дела?", max_tokens=50)
            print(f"✅ Тест генерации: {response[:50]}...")
            
            # Статистика
            stats = manager.get_performance_stats()
            print(f"✅ Статистика: cache_hit_rate={stats.get('cache_hit_rate', 0):.2%}")
            
        else:
            print("❌ Оптимизированный менеджер не инициализирован")
            return False
        
        # 2. Обновление конфигурации
        print("\n⚙️ Обновление конфигурации...")
        
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
            
            print("✅ Конфигурация обновлена")
        
        # 3. Создание symbolic link для легкого доступа
        print("\n🔗 Создание symbolic link...")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.join(current_dir, "..", "mlearning", "current_manager.py")
        source_path = os.path.join(current_dir, "..", "mlearning", "optimized_fractal_model_manager.py")
        
        if os.path.exists(target_path):
            os.remove(target_path)
        
        # Windows не поддерживает symbolic links, используем копию
        import shutil
        shutil.copy2(source_path, target_path)
        
        print("✅ Symbolic link создан")
        
        print("\n" + "=" * 50)
        print("🎉 МИГРАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
        print("\n📝 Следующие шаги:")
        print("1. Используйте UnifiedFractalManager для автоматического выбора")
        print("2. Или импортируйте OptimizedFractalModelManager напрямую")
        print("3. Проверьте производительность в GUI")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка миграции: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = migrate_to_optimized()
    
    if success:
        print("\n✅ Миграция завершена успешно!")
    else:
        print("\n❌ Миграция завершилась с ошибками")
    
    print("\nНажмите Enter для выхода...")
    input()
