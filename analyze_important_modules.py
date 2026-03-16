#!/usr/bin/env python3
"""
Фокусированный анализ неинтегрированных модулей CogniFlex
Проверяет только важные модули, которые должны быть интегрированы
"""

import os
import sys
import importlib
import inspect
from typing import Dict, List, Tuple, Any

# Добавляем путь к CogniFlex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_important_modules():
    """Возвращает список важных модулей для проверки"""
    return {
        "adaptation": {
            "path": "cogniflex.adaptation",
            "description": "Модуль адаптации ответов",
            "should_integrate": True
        },
        "analytics": {
            "path": "cogniflex.analytics", 
            "description": "Модуль аналитики",
            "should_integrate": True
        },
        "contradiction": {
            "path": "cogniflex.contradiction",
            "description": "Модуль обнаружения противоречий",
            "should_integrate": True
        },
        "ethics": {
            "path": "cogniflex.ethics",
            "description": "Модуль этической оценки",
            "should_integrate": True
        },
        "learning": {
            "path": "cogniflex.learning",
            "description": "Модуль обучения",
            "should_integrate": True
        },
        "web": {
            "path": "cogniflex.web",
            "description": "Модуль веб-поиска",
            "should_integrate": True
        },
        "knowledge": {
            "path": "cogniflex.knowledge",
            "description": "Модуль графа знаний",
            "should_integrate": False  # Уже интегрирован
        },
        "memory": {
            "path": "cogniflex.memory",
            "description": "Модуль памяти",
            "should_integrate": False  # Уже интегрирован
        },
        "mlearning": {
            "path": "cogniflex.mlearning",
            "description": "Модуль машинного обучения",
            "should_integrate": False  # Уже интегрирован
        }
    }

def check_module_integration_status(module_name: str, module_info: Dict) -> Dict:
    """Проверяет статус интеграции модуля"""
    result = {
        "module": module_name,
        "description": module_info["description"],
        "should_integrate": module_info["should_integrate"],
        "integrated": False,
        "base_component": False,
        "event_support": False,
        "has_factory": False,
        "issues": [],
        "recommendations": []
    }
    
    try:
        # Проверяем наличие фабрики в ComponentInitializer
        class DummyBrain:
            def __init__(self):
                self.components = {}
            def register_component(self, name, component):
                self.components[name] = component
                return True
        
        dummy_brain = DummyBrain()
        
        from cogniflex.core.component_initializer import ComponentInitializer
        initializer = ComponentInitializer(dummy_brain)
        
        factory_name = f"create_{module_name}"
        if hasattr(initializer, factory_name):
            result["has_factory"] = True
            result["integrated"] = True
        
        # Анализируем основной файл модуля
        module_path = os.path.join(os.path.dirname(__file__), module_info["path"])
        if os.path.exists(module_path):
            # Ищем основной файл модуля
            main_file = None
            for file in os.listdir(module_path):
                if file.endswith(".py") and not file.startswith("__"):
                    if file == f"{module_name}.py" or file == f"{module_name}_core.py":
                        main_file = os.path.join(module_path, file)
                        break
            
            if not main_file:
                # Берем первый .py файл
                for file in os.listdir(module_path):
                    if file.endswith(".py") and not file.startswith("__"):
                        main_file = os.path.join(module_path, file)
                        break
            
            if main_file and os.path.exists(main_file):
                # Анализируем файл
                with open(main_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Проверяем наследование от BaseComponent
                if "BaseComponent" in content:
                    result["base_component"] = True
                
                # Проверяем поддержку событий
                if "EventBus" in content or "emit_event" in content or "subscribe" in content:
                    result["event_support"] = True
                
                # Проверяем на заглушки
                if "pass" in content and "TODO" in content:
                    result["issues"].append("Обнаружены заглушки (TODO)")
                
                if "NotImplemented" in content:
                    result["issues"].append("Обнаружены нереализованные методы")
                
                # Проверяем наличие методов интеграции
                if "def initialize(" in content:
                    result["base_component"] = True
                
                # Рекомендации
                if not result["integrated"] and module_info["should_integrate"]:
                    if not result["base_component"]:
                        result["recommendations"].append("Наследовать основной класс от BaseComponent")
                    if not result["event_support"]:
                        result["recommendations"].append("Добавить поддержку EventBus")
                    result["recommendations"].append("Создать фабрику в ComponentInitializer")
        
    except Exception as e:
        result["issues"].append(f"Ошибка анализа: {e}")
    
    return result

def analyze_component_initializers():
    """Анализирует текущие компоненты в ComponentInitializer"""
    print("🔧 Анализ текущих компонентов...")
    
    try:
        class DummyBrain:
            def __init__(self):
                self.components = {}
            def register_component(self, name, component):
                self.components[name] = component
                return True
        
        dummy_brain = DummyBrain()
        
        from cogniflex.core.component_initializer import ComponentInitializer
        initializer = ComponentInitializer(dummy_brain)
        
        components = []
        for attr_name in dir(initializer):
            if attr_name.startswith("create_") and callable(getattr(initializer, attr_name)):
                component_name = attr_name.replace("create_", "")
                components.append(component_name)
        
        print(f"   📋 Текущие компоненты ({len(components)}):")
        for comp in sorted(components):
            print(f"      - {comp}")
        
        return components
        
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return []

def check_core_brain_integration():
    """Проверяет интеграцию с CoreBrain"""
    print("\n🧠 Проверка интеграции с CoreBrain...")
    
    try:
        from cogniflex.core.core_brain import CoreBrain
        
        # Создаем экземпляр для анализа
        brain = CoreBrain()
        
        # Проверяем наличие компонентов
        if hasattr(brain, 'components'):
            components = list(brain.components.keys())
            print(f"   📋 Компоненты в CoreBrain ({len(components)}):")
            for comp in sorted(components):
                print(f"      - {comp}")
            
            return components
        else:
            print("   ⚠️ CoreBrain не имеет атрибута components")
            return []
            
    except Exception as e:
        print(f"   ❌ Ошибка анализа CoreBrain: {e}")
        return []

def main():
    """Основная функция анализа"""
    print("🚀 Фокусированный анализ неинтегрированных модулей CogniFlex")
    print("=" * 60)
    
    # 1. Анализ текущих компонентов
    current_components = analyze_component_initializers()
    
    # 2. Анализ CoreBrain
    brain_components = check_core_brain_integration()
    
    # 3. Анализ важных модулей
    print(f"\n📊 Анализ важных модулей...")
    important_modules = get_important_modules()
    results = []
    
    for module_name, module_info in important_modules.items():
        result = check_module_integration_status(module_name, module_info)
        results.append(result)
        
        status = "✅" if result["integrated"] else "⚠️" if not result["should_integrate"] else "❌"
        print(f"\n   {status} {module_name}")
        print(f"      📝 {result['description']}")
        
        if result["integrated"]:
            print(f"      ✅ Интегрирован (фабрика: create_{module_name})")
        elif result["should_integrate"]:
            print(f"      ❌ Требует интеграции")
        
        if result["base_component"]:
            print(f"      🔧 Поддерживает BaseComponent")
        if result["event_support"]:
            print(f"      📡 Поддерживает события")
        
        if result["issues"]:
            for issue in result["issues"]:
                print(f"      ⚠️ {issue}")
        
        if result["recommendations"]:
            for rec in result["recommendations"]:
                print(f"      💡 {rec}")
    
    # 4. Итоги
    print(f"\n📊 ИТОГИ АНАЛИЗА:")
    
    integrated_count = len([r for r in results if r["integrated"]])
    need_integration = [r for r in results if r["should_integrate"] and not r["integrated"]]
    have_issues = [r for r in results if r["issues"]]
    
    print(f"   ✅ Интегрировано: {integrated_count}/{len(results)}")
    print(f"   ❌ Требуют интеграции: {len(need_integration)}")
    print(f"   ⚠️ Имеют проблемы: {len(have_issues)}")
    
    # 5. Рекомендации
    if need_integration:
        print(f"\n💡 ПЛАН ИНТЕГРАЦИИ:")
        for result in need_integration:
            print(f"\n   🎯 {result['module']} ({result['description']})")
            for rec in result["recommendations"]:
                print(f"      - {rec}")
    
    return len(need_integration), len(have_issues)

if __name__ == "__main__":
    need_integration, have_issues = main()
    
    if need_integration > 0:
        print(f"\n⚠️ Требуется интеграция {need_integration} модулей")
        sys.exit(1)
    elif have_issues > 0:
        print(f"\n⚠️ Обнаружены проблемы в {have_issues} модулях")
        sys.exit(1)
    else:
        print(f"\n✅ Все важные модули корректно интегрированы!")
        sys.exit(0)
