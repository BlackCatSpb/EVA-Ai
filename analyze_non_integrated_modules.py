#!/usr/bin/env python3
"""
Анализ неинтегрированных модулей CogniFlex
Проверяет функциональность и совместимость с существующей структурой
"""

import os
import sys
import importlib
import inspect
from typing import Dict, List, Tuple, Any

# Добавляем путь к CogniFlex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_module_structure():
    """Анализирует структуру модулей"""
    print("🔍 Анализ структуры модулей CogniFlex...")
    
    modules_info = {}
    base_path = os.path.join(os.path.dirname(__file__), "cogniflex")
    
    for root, dirs, files in os.walk(base_path):
        if "__pycache__" in root:
            continue
            
        rel_path = os.path.relpath(root, base_path)
        module_name = rel_path.replace(os.sep, ".")
        
        if module_name == ".":
            module_name = "cogniflex"
        else:
            module_name = f"cogniflex.{module_name}"
        
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                file_path = os.path.join(root, file)
                module_file = file[:-3]  # Удаляем .py
                
                if module_name == "cogniflex":
                    full_module_name = f"cogniflex.{module_file}"
                else:
                    full_module_name = f"{module_name}.{module_file}"
                
                modules_info[full_module_name] = {
                    "path": file_path,
                    "size": os.path.getsize(file_path),
                    "module_name": full_module_name
                }
    
    return modules_info

def check_module_integration(module_name: str, module_info: Dict) -> Dict:
    """Проверяет интеграцию модуля"""
    result = {
        "module": module_name,
        "integratable": False,
        "conflicts": [],
        "dependencies": [],
        "base_component": False,
        "event_support": False,
        "core_brain_integration": False,
        "issues": []
    }
    
    try:
        # Пытаемся импортировать модуль
        spec = importlib.util.spec_from_file_location(
            module_name, 
            module_info["path"]
        )
        
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Анализируем классы
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Проверяем наследование от BaseComponent
                bases = [base.__name__ for base in obj.__bases__]
                if "BaseComponent" in bases:
                    result["base_component"] = True
                    result["integratable"] = True
                
                # Проверяем наличие методов интеграции
                methods = [method for method in dir(obj) if not method.startswith("_")]
                if "initialize" in methods:
                    result["core_brain_integration"] = True
                
                # Проверяем поддержку событий
                if hasattr(obj, "emit_event") or hasattr(obj, "subscribe"):
                    result["event_support"] = True
            
            # Анализируем функции
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if name.startswith("create_") or name.startswith("init_"):
                    result["integratable"] = True
            
            # Проверяем импорты для зависимостей
            with open(module_info["path"], "r", encoding="utf-8") as f:
                content = f.read()
                
                # Ищем потенциальные конфликты
                if "from cogniflex.core" in content:
                    result["dependencies"].append("core")
                if "from cogniflex.memory" in content:
                    result["dependencies"].append("memory")
                if "from cogniflex.learning" in content:
                    result["dependencies"].append("learning")
                
                # Проверяем на потенциальные проблемы
                if "import torch" in content and "torch" not in result["dependencies"]:
                    result["dependencies"].append("torch")
                
                # Проверяем на заглушки
                if "pass" in content and "TODO" in content:
                    result["issues"].append("Обнаружены заглушки (TODO)")
                
                if "NotImplemented" in content:
                    result["issues"].append("Обнаружены нереализованные методы")
        
    except Exception as e:
        result["issues"].append(f"Ошибка импорта: {e}")
    
    return result

def analyze_component_initializers():
    """Анализирует инициализаторы компонентов"""
    print("\n🔧 Анализ ComponentInitializer...")
    
    try:
        from cogniflex.core.component_initializer import ComponentInitializer
        
        # Создаем фиктивный CoreBrain для инициализации
        class DummyBrain:
            def __init__(self):
                self.components = {}
            def register_component(self, name, component):
                self.components[name] = component
                return True
        
        dummy_brain = DummyBrain()
        initializer = ComponentInitializer(dummy_brain)
        
        # Получаем все фабрики
        factories = {}
        for attr_name in dir(initializer):
            if attr_name.startswith("create_") and callable(getattr(initializer, attr_name)):
                component_name = attr_name.replace("create_", "")
                factories[component_name] = attr_name
        
        print(f"   📋 Найдено фабрик: {len(factories)}")
        for name, factory in factories.items():
            print(f"      - {name}: {factory}")
        
        return factories
        
    except Exception as e:
        print(f"   ❌ Ошибка анализа ComponentInitializer: {e}")
        return {}

def check_non_integrated_modules():
    """Проверяет неинтегрированные модули"""
    print("\n🔍 Поиск неинтегрированных модулей...")
    
    try:
        # Создаем фиктивный CoreBrain для инициализации
        class DummyBrain:
            def __init__(self):
                self.components = {}
            def register_component(self, name, component):
                self.components[name] = component
                return True
        
        dummy_brain = DummyBrain()
        
        from cogniflex.core.component_initializer import ComponentInitializer
        initializer = ComponentInitializer(dummy_brain)
        
        # Получаем все модули в системе
        all_modules = analyze_module_structure()
        integrated_modules = set()
        
        # Получаем интегрированные модули из ComponentInitializer
        for attr_name in dir(initializer):
            if attr_name.startswith("create_"):
                component_name = attr_name.replace("create_", "")
                integrated_modules.add(component_name)
        
        # Находим неинтегрированные
        non_integrated = []
        for module_name in all_modules:
            module_base = module_name.split(".")[-1]
            if module_base not in integrated_modules and module_base not in [
                "__init__", "component_initializer", "core_brain", 
                "base_component", "event_bus", "system_state"
            ]:
                non_integrated.append(module_name)
        
        print(f"   📊 Найдено модулей: {len(all_modules)}")
        print(f"   ✅ Интегрировано: {len(integrated_modules)}")
        print(f"   ⚠️ Не интегрировано: {len(non_integrated)}")
        
        return non_integrated, integrated_modules
        
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return [], []

def analyze_specific_modules(modules: List[str]):
    """Анализирует конкретные модули"""
    print("\n🔍 Детальный анализ модулей...")
    
    modules_info = analyze_module_structure()
    results = []
    
    for module_name in modules:
        if module_name in modules_info:
            result = check_module_integration(module_name, modules_info[module_name])
            results.append(result)
            
            status = "✅" if result["integratable"] else "⚠️"
            print(f"\n   {status} {module_name}")
            
            if result["base_component"]:
                print(f"      🔧 Наследует BaseComponent")
            if result["event_support"]:
                print(f"      📡 Поддерживает события")
            if result["core_brain_integration"]:
                print(f"      🧠 Интеграция с CoreBrain")
            
            if result["dependencies"]:
                print(f"      🔗 Зависимости: {', '.join(result['dependencies'])}")
            
            if result["issues"]:
                for issue in result["issues"]:
                    print(f"      ⚠️ {issue}")
    
    return results

def check_integration_conflicts():
    """Проверяет конфликты интеграции"""
    print("\n⚠️ Проверка конфликтов интеграции...")
    
    conflicts = []
    
    try:
        # Создаем фиктивный CoreBrain для инициализации
        class DummyBrain:
            def __init__(self):
                self.components = {}
            def register_component(self, name, component):
                self.components[name] = component
                return True
        
        dummy_brain = DummyBrain()
        
        from cogniflex.core.component_initializer import ComponentInitializer
        initializer = ComponentInitializer(dummy_brain)
        
        # Проверяем дублирование имен
        method_names = []
        for attr_name in dir(initializer):
            if attr_name.startswith("create_"):
                component_name = attr_name.replace("create_", "")
                if component_name in method_names:
                    conflicts.append(f"Дублирование компонента: {component_name}")
                method_names.append(component_name)
        
    except Exception as e:
        conflicts.append(f"Ошибка проверки ComponentInitializer: {e}")
    
    # Проверяем конфликты импортов
    base_path = os.path.join(os.path.dirname(__file__), "cogniflex")
    import_conflicts = {}
    
    for root, dirs, files in os.walk(base_path):
        if "__pycache__" in root:
            continue
            
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                        # Ищем импорты из других модулей CogniFlex
                        lines = content.split("\n")
                        for line in lines:
                            if "from cogniflex." in line and "import" in line:
                                if line not in import_conflicts:
                                    import_conflicts[line] = []
                                import_conflicts[line].append(file_path)
                except:
                    pass
    
    # Проверяем повторяющиеся импорты
    for import_stmt, files in import_conflicts.items():
        if len(files) > 5:  # Если импорт используется во многих файлах
            conflicts.append(f"Популярный импорт: {import_stmt} ({len(files)} файлов)")
    
    return conflicts

def main():
    """Основная функция анализа"""
    print("🚀 Анализ неинтегрированных модулей CogniFlex")
    print("=" * 60)
    
    # 1. Анализ структуры
    print("\n📊 1. Анализ структуры модулей...")
    modules_info = analyze_module_structure()
    print(f"   📁 Всего модулей: {len(modules_info)}")
    
    # 2. Анализ ComponentInitializer
    factories = analyze_component_initializers()
    
    # 3. Поиск неинтегрированных модулей
    non_integrated, integrated = check_non_integrated_modules()
    
    # 4. Детальный анализ неинтегрированных модулей
    if non_integrated:
        print(f"\n📋 2. Анализ {len(non_integrated)} неинтегрированных модулей...")
        results = analyze_specific_modules(non_integrated[:10])  # Ограничиваем для скорости
        
        # Классифицируем результаты
        integratable = [r for r in results if r["integratable"]]
        problematic = [r for r in results if r["issues"]]
        
        print(f"\n📊 Классификация:")
        print(f"   ✅ Готовы к интеграции: {len(integratable)}")
        print(f"   ⚠️ Требуют внимания: {len(problematic)}")
    else:
        print("\n✅ Все модули интегрированы!")
    
    # 5. Проверка конфликтов
    conflicts = check_integration_conflicts()
    if conflicts:
        print(f"\n⚠️ 3. Обнаружено конфликтов: {len(conflicts)}")
        for conflict in conflicts[:5]:  # Ограничиваем вывод
            print(f"   - {conflict}")
    else:
        print("\n✅ Конфликты не обнаружены")
    
    # 6. Рекомендации
    print(f"\n💡 4. Рекомендации:")
    
    if non_integrated:
        print(f"   🔧 Интегрировать модули:")
        for module in non_integrated[:5]:
            print(f"      - {module}")
    
    if conflicts:
        print(f"   ⚠️ Разрешить конфликты:")
        for conflict in conflicts[:3]:
            print(f"      - {conflict}")
    
    print(f"   📊 Создать отчет о состоянии интеграции")
    print(f"   🔍 Проверить тесты для интегрированных модулей")
    
    return len(non_integrated), len(conflicts)

if __name__ == "__main__":
    non_integrated_count, conflicts_count = main()
    sys.exit(0 if conflicts_count == 0 else 1)
