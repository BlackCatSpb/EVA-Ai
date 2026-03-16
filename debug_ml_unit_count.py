#!/usr/bin/env python3
"""
Отладка количества моделей в ML Unit
"""
import sys
import os

# Добавляем путь к CogniFlex
sys.path.append('.')

def debug_ml_unit_models():
    """Отлаживает почему ML Unit показывает 2 модели"""
    print("🔍 ОТЛАДКА КОЛИЧЕСТВА МОДЕЛЕЙ В ML UNIT")
    print("=" * 50)
    
    try:
        from cogniflex.core.core_brain import CoreBrain
        
        # Создаем ядро
        brain = CoreBrain()
        print("✅ CoreBrain создан")
        
        # Инициализируем
        if brain.initialize():
            print("✅ CoreBrain инициализирован")
            
            # Проверяем ML Unit
            if hasattr(brain, 'ml_unit') and brain.ml_unit:
                print(f"🤖 ML Unit: {type(brain.ml_unit)}")
                
                # Проверяем model_manager
                if hasattr(brain.ml_unit, 'model_manager'):
                    print(f"📊 Model Manager: {type(brain.ml_unit.model_manager)}")
                    
                    # Проверяем get_available_models
                    if hasattr(brain.ml_unit.model_manager, 'get_available_models'):
                        models = brain.ml_unit.model_manager.get_available_models()
                        print(f"📋 Модели из model_manager: {list(models.keys())}")
                        print(f"📊 Количество моделей: {len(models)}")
                        
                        # Детальная информация о моделях
                        for name, info in models.items():
                            print(f"   📝 {name}: {info.get('display_name', 'Unknown')}")
                    else:
                        print("❌ У model_manager нет get_available_models")
                
                # Проверяем model_metadata
                if hasattr(brain.ml_unit.model_manager, 'model_metadata'):
                    metadata = brain.ml_unit.model_manager.model_metadata
                    print(f"📋 Model metadata: {metadata}")
                
                # Проверяем другие менеджеры моделей
                for attr_name in ['fractal_model_manager', 'enhanced_manager', 'current_manager']:
                    if hasattr(brain, attr_name):
                        manager = getattr(brain, attr_name)
                        print(f"🔧 {attr_name}: {type(manager)}")
                        
                        if hasattr(manager, 'get_available_models'):
                            try:
                                models = manager.get_available_models()
                                print(f"   📊 Модели: {list(models.keys()) if isinstance(models, dict) else models}")
                            except Exception as e:
                                print(f"   ❌ Ошибка get_available_models: {e}")
                
                # Проверяем все компоненты
                print(f"\n📋 Все компоненты CoreBrain:")
                for name, component in brain.components.items():
                    print(f"   - {name}: {type(component).__name__}")
                    
                    # Если это менеджер моделей, проверяем его модели
                    if 'manager' in name.lower() and hasattr(component, 'get_available_models'):
                        try:
                            models = component.get_available_models()
                            print(f"     📊 Модели: {list(models.keys()) if isinstance(models, dict) else models}")
                        except Exception as e:
                            print(f"     ❌ Ошибка: {e}")
                
                return True
            else:
                print("❌ ML Unit недоступен")
                return False
        else:
            print("❌ CoreBrain не инициализирован")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Основная функция"""
    debug_ml_unit_models()

if __name__ == "__main__":
    main()
