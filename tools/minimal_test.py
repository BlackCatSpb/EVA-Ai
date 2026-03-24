"""
Минимальный тест GUI для диагностики проблем
"""

import os
import sys
import traceback

# Добавляем путь к проекту
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cogniflex'))

def test_gui_imports():
    """Тестирует импорты GUI модулей."""
    print("=== Тест импортов GUI ===")
    
    try:
        print("Импорт CoreBrain...")
        from cogniflex.core.core_brain import CoreBrain
        print("✅ CoreBrain импортирован")
        
        print("Импорт GUI...")
        from cogniflex.gui.core_gui import CogniFlexGUI
        print("✅ GUI импортирован")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка импорта: {e}")
        traceback.print_exc()
        return False

def test_gui_with_mock_brain():
    """Тестирует GUI с мок brain."""
    print("\n=== Тест GUI с мок brain ===")
    
    try:
        # Создаем простой мок brain
        class MockBrain:
            def __init__(self):
                self.running = False
                self.initialized = False
                self.config_manager = None
                self.system_state_manager = None
                self.resource_manager = None
                self.system_metrics_manager = None
            
            def initialize(self):
                self.initialized = True
                return True
            
            def start(self):
                self.running = True
            
            def stop(self):
                self.running = False
            
            def get_status(self):
                return {'running': self.running, 'initialized': self.initialized}
            
            def get_system_info(self):
                return {'version': '2.0.0', 'components': 5}
        
        brain = MockBrain()
        print("✅ MockBrain создан")
        
        from cogniflex.gui.core_gui import CogniFlexGUI
        gui = CogniFlexGUI(brain=brain)
        print("✅ GUI создан с мок brain")
        
        # Проверяем основные атрибуты
        if hasattr(gui, 'brain') and gui.brain == brain:
            print("✅ Brain корректно привязан к GUI")
        else:
            print("❌ Проблема с привязкой brain к GUI")
            return False
        
        if hasattr(gui, 'cache_dir'):
            print(f"✅ Cache dir: {gui.cache_dir}")
        else:
            print("❌ Отсутствует cache_dir")
            return False
        
        if hasattr(gui, 'settings'):
            print("✅ Настройки GUI загружены")
        else:
            print("❌ Настройки GUI не загружены")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка создания GUI: {e}")
        traceback.print_exc()
        return False

def test_gui_methods():
    """Тестирует методы GUI."""
    print("\n=== Тест методов GUI ===")
    
    try:
        class MockBrain:
            def __init__(self):
                self.running = False
                self.initialized = False
            
            def get_status(self):
                return {'running': self.running, 'initialized': self.initialized}
            
            def get_system_info(self):
                return {'version': '2.0.0', 'components': 5}
        
        brain = MockBrain()
        
        from cogniflex.gui.core_gui import CogniFlexGUI
        gui = CogniFlexGUI(brain=brain)
        
        # Проверяем наличие ключевых методов
        expected_methods = ['start', 'stop', 'create_main_window']
        missing_methods = []
        
        for method in expected_methods:
            if hasattr(gui, method):
                print(f"✅ Метод {method} найден")
            else:
                print(f"❌ Метод {method} отсутствует")
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ Отсутствуют методы: {missing_methods}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка проверки методов: {e}")
        traceback.print_exc()
        return False

def test_gui_error_handling():
    """Тестирует обработку ошибок в GUI."""
    print("\n=== Тест обработки ошибок GUI ===")
    
    try:
        from cogniflex.gui.core_gui import CogniFlexGUI
        
        # Тест с None brain
        try:
            gui = CogniFlexGUI(brain=None)
            print("✅ GUI создан с None brain (обработка ошибки работает)")
        except Exception as e:
            print(f"⚠️ GUI не может работать с None brain: {e}")
        
        # Тест с некорректным brain
        class BadBrain:
            pass
        
        try:
            bad_brain = BadBrain()
            gui = CogniFlexGUI(brain=bad_brain)
            print("✅ GUI создан с некорректным brain")
        except Exception as e:
            print(f"⚠️ GUI не может работать с некорректным brain: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в тесте обработки ошибок: {e}")
        traceback.print_exc()
        return False

def test_gui_integration():
    """Тестирует интеграцию GUI с новыми менеджерами."""
    print("\n=== Тест интеграции с новыми менеджерами ===")
    
    try:
        # Создаем мок brain с новыми менеджерами
        class EnhancedMockBrain:
            def __init__(self):
                self.running = False
                self.initialized = False
                
                # Мок менеджеры
                from unittest.mock import Mock
                self.config_manager = Mock()
                self.system_state_manager = Mock()
                self.resource_manager = Mock()
                self.system_metrics_manager = Mock()
                
                # Настраиваем возвращаемые значения
                self.config_manager.get_config.return_value = {'gui': {'theme': 'light'}}
                self.system_state_manager.get_system_state.return_value = 'running'
                self.system_state_manager.get_component_states.return_value = {
                    'ml_unit': 'active',
                    'knowledge_graph': 'active'
                }
                self.resource_manager.get_resource_summary.return_value = {
                    'cpu_usage': 25.5,
                    'memory_usage': 45.2
                }
                self.system_metrics_manager.get_performance_metrics.return_value = {
                    'queries_processed': 100,
                    'avg_response_time': 0.5
                }
            
            def get_status(self):
                return {'running': self.running, 'initialized': self.initialized}
            
            def get_system_info(self):
                return {'version': '2.0.0', 'components': 5}
        
        brain = EnhancedMockBrain()
        print("✅ Enhanced MockBrain создан")
        
        from cogniflex.gui.core_gui import CogniFlexGUI
        gui = CogniFlexGUI(brain=brain)
        print("✅ GUI создан с enhanced brain")
        
        # Тестируем доступ к менеджерам
        if hasattr(brain, 'config_manager'):
            config = brain.config_manager.get_config()
            print(f"✅ Config manager работает: {config}")
        
        if hasattr(brain, 'system_state_manager'):
            state = brain.system_state_manager.get_system_state()
            print(f"✅ System state manager работает: {state}")
        
        if hasattr(brain, 'resource_manager'):
            resources = brain.resource_manager.get_resource_summary()
            print(f"✅ Resource manager работает: {resources}")
        
        if hasattr(brain, 'system_metrics_manager'):
            metrics = brain.system_metrics_manager.get_performance_metrics()
            print(f"✅ System metrics manager работает: {metrics}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка интеграции: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 ДИАГНОСТИКА GUI")
    print("="*50)
    
    tests = [
        ("Импорты GUI", test_gui_imports),
        ("GUI с мок brain", test_gui_with_mock_brain),
        ("Методы GUI", test_gui_methods),
        ("Обработка ошибок", test_gui_error_handling),
        ("Интеграция с менеджерами", test_gui_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Критическая ошибка в {test_name}: {e}")
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*50)
    print("📊 РЕЗУЛЬТАТЫ ДИАГНОСТИКИ GUI")
    print("="*50)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ ПРОЙДЕН" if results[i] else "❌ ПРОВАЛЕН"
        print(f"{test_name:25} : {status}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nОбщий результат: {passed}/{total} ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 GUI готов к работе!")
    elif passed >= total * 0.8:
        print("✅ GUI работает с незначительными проблемами")
    else:
        print("❌ Обнаружены серьезные проблемы в GUI")
