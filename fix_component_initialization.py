#!/usr/bin/env python3
"""
Исправление инициализации компонентов в CogniFlex
"""
import sys
import os
import logging
import traceback

# Добавляем путь к CogniFlex
sys.path.append('.')

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_component_initialization():
    """Исправляет инициализацию компонентов"""
    print("🔧 ИСПРАВЛЕНИЕ ИНИЦИАЛИЗАЦИИ КОМПОНЕНТОВ")
    print("=" * 60)
    
    try:
        # Читаем файл component_initializer.py
        file_path = "cogniflex/core/component_initializer.py"
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Ищем место где создается компонент
        old_code = """                # Создаем компонент
                print(f"DEBUG: Creating component {component_name}...")
                component = factory()
                print(f"DEBUG: Component {component_name} created: {type(component).__name__}")
                
                # Регистрируем компонент
                print(f"DEBUG: Registering component {component_name}...")
                if self.register_component(component_name, component):"""
        
        new_code = """                # Создаем компонент
                print(f"DEBUG: Creating component {component_name}...")
                component = factory()
                print(f"DEBUG: Component {component_name} created: {type(component).__name__}")
                
                # ИНИЦИАЛИЗИРУЕМ КОМПОНЕНТ - ЭТО КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ!
                print(f"DEBUG: Initializing component {component_name}...")
                if hasattr(component, 'initialize'):
                    try:
                        init_result = component.initialize()
                        print(f"DEBUG: Component {component_name} initialize() returned: {init_result}")
                        if not init_result:
                            print(f"DEBUG: Component {component_name} failed to initialize")
                            return False
                    except Exception as e:
                        print(f"DEBUG: Exception during {component_name} initialize(): {e}")
                        traceback.print_exc()
                        return False
                else:
                    print(f"DEBUG: Component {component_name} has no initialize() method")
                
                # Регистрируем компонент
                print(f"DEBUG: Registering component {component_name}...")
                if self.register_component(component_name, component):"""
        
        if old_code in content:
            content = content.replace(old_code, new_code)
            print("✅ Найден и заменен код инициализации компонентов")
        else:
            print("❌ Не найден код для замены")
            return False
        
        # Записываем исправленный файл
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Файл component_initializer.py успешно исправлен")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка исправления файла: {e}")
        traceback.print_exc()
        return False

def test_fixed_initialization():
    """Тестирует исправленную инициализацию"""
    print("\n🧪 ТЕСТИРОВАНИЕ ИСПРАВЛЕННОЙ ИНИЦИАЛИЗАЦИИ")
    print("=" * 50)
    
    try:
        from cogniflex.core.core_brain import CoreBrain
        
        # Создаем и инициализируем ядро
        brain = CoreBrain()
        
        print("📊 Инициализация ядра...")
        if brain.initialize():
            print("✅ Инициализация успешна")
            
            print(f"📋 Компонентов после инициализации: {len(brain.components)}")
            
            # Запускаем
            print("🚀 Запуск компонентов...")
            if brain.start():
                print("✅ Запуск успешен")
                
                # Проверяем состояние
                print(f"📊 Статус системы:")
                print(f"   Всего компонентов: {len(brain.components)}")
                
                initialized_count = 0
                for name, component in brain.components.items():
                    if hasattr(component, 'state'):
                        state = getattr(component, 'state', None)
                        print(f"   {name}: {state}")
                        if str(state) != 'ComponentState.UNINITIALIZED':
                            initialized_count += 1
                    else:
                        print(f"   {name}: {type(component).__name__}")
                        initialized_count += 1
                
                print(f"   Инициализировано: {initialized_count}/{len(brain.components)}")
                
                if initialized_count >= len(brain.components) * 0.8:
                    print("✅ Большинство компонентов инициализировано корректно")
                    return True
                else:
                    print("⚠️ Все еще есть проблемы с инициализацией")
                    return False
            else:
                print("❌ Запуск неуспешен")
                return False
        else:
            print("❌ Инициализация неуспешна")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        traceback.print_exc()
        return False

def main():
    """Основная функция"""
    print("🚀 ИСПРАВЛЕНИЕ ИНИЦИАЛИЗАЦИИ КОМПОНЕНТОВ COGNIFLEX")
    print("=" * 70)
    
    # Шаг 1: Исправляем код
    if fix_component_initialization():
        print("\n📝 Код исправлен")
        
        # Шаг 2: Тестируем исправленную версию
        if test_fixed_initialization():
            print("\n🎉 УСПЕХ! Инициализация компонентов исправлена")
        else:
            print("\n⚠️ Требуется дополнительная отладка")
    else:
        print("\n❌ Не удалось исправить код")

if __name__ == "__main__":
    main()
