#!/usr/bin/env python3
"""
Анализ текущей архитектуры и планирование интеграции с событийной системой
"""

import os
import sys
import time
from typing import Dict, List, Any, Set

# Добавляем путь к CogniFlex
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def analyze_current_architecture():
    """Анализ текущей архитектуры для выявления конфликтов и дублирования"""
    
    print("🔍 Анализ текущей архитектуры CogniFlex...")
    
    # 1. Анализ методов компонентов
    print("\n1. 📋 Анализ методов компонентов...")
    
    try:
        from cogniflex.core.core_brain import CoreBrain
        from cogniflex.core.component_initializer import ComponentInitializer
        
        brain = CoreBrain()
        initializer = ComponentInitializer(brain)
        
        # Инициализируем компоненты
        initializer.initialize_components()
        
        # Собираем методы всех компонентов
        component_methods = {}
        method_conflicts = {}
        
        for comp_name, component in brain.components.items():
            if hasattr(component, '__class__'):
                methods = set()
                for attr_name in dir(component):
                    if not attr_name.startswith('_') and callable(getattr(component, attr_name)):
                        methods.add(attr_name)
                
                component_methods[comp_name] = methods
                
                # Проверяем конфликты
                for method in methods:
                    if method not in method_conflicts:
                        method_conflicts[method] = []
                    method_conflicts[method].append(comp_name)
        
        # Показываем компоненты и их методы
        print(f"   📊 Проанализировано компонентов: {len(component_methods)}")
        
        for comp_name, methods in sorted(component_methods.items()):
            print(f"   📋 {comp_name}: {len(methods)} методов")
            if len(methods) <= 5:
                for method in sorted(methods):
                    print(f"      - {method}")
        
        # Показываем конфликты
        print(f"\n   ⚠️ Конфликты методов:")
        conflicts_found = False
        for method, components in method_conflicts.items():
            if len(components) > 1:
                print(f"      ⚠️ {method}: {', '.join(components)}")
                conflicts_found = True
        
        if not conflicts_found:
            print("      ✅ Конфликтов методов не найдено")
        
    except Exception as e:
        print(f"   ❌ Ошибка анализа компонентов: {e}")
        return False
    
    # 2. Анализ зависимостей
    print("\n2. 🔗 Анализ зависимостей...")
    
    dependencies = initializer.component_dependencies
    dependency_graph = {}
    
    for comp, deps in dependencies.items():
        dependency_graph[comp] = deps
        print(f"   🔗 {comp}: {len(deps)} зависимостей")
        if deps:
            print(f"      -> {', '.join(deps)}")
    
    # Проверяем циклические зависимости
    def has_cycle(graph):
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if dfs(node):
                    return True
        return False
    
    if has_cycle(dependency_graph):
        print("   ⚠️ Обнаружены циклические зависимости!")
    else:
        print("   ✅ Циклических зависимостей не найдено")
    
    # 3. Анализ автономности
    print("\n3. 🤖 Анализ автономности...")
    
    autonomous_modules = [
        'analytics_manager', 'contradiction_manager', 'ethics_framework',
        'learning_manager', 'web_search_engine', 'knowledge_graph'
    ]
    
    autonomous_status = {}
    
    for module in autonomous_modules:
        if module in brain.components:
            component = brain.components[module]
            
            # Проверяем методы автономной работы
            autonomous_features = []
            
            if hasattr(component, 'start'):
                autonomous_features.append('start')
            if hasattr(component, 'stop'):
                autonomous_features.append('stop')
            if hasattr(component, 'run'):
                autonomous_features.append('run')
            if hasattr(component, 'process'):
                autonomous_features.append('process')
            if hasattr(component, 'analyze'):
                autonomous_features.append('analyze')
            if hasattr(component, 'monitor'):
                autonomous_features.append('monitor')
            
            autonomous_status[module] = {
                'available': True,
                'features': autonomous_features,
                'autonomous_score': len(autonomous_features) / 6.0
            }
            
            print(f"   🤖 {module}: {len(autonomous_features)}/6 функций автономности")
            if autonomous_features:
                print(f"      -> {', '.join(autonomous_features)}")
        else:
            autonomous_status[module] = {
                'available': False,
                'features': [],
                'autonomous_score': 0.0
            }
            print(f"   ❌ {module}: недоступен")
    
    # 4. Анализ событийной системы
    print("\n4. 📡 Анализ событийной системы...")
    
    event_system_components = []
    
    for comp_name, component in brain.components.items():
        event_methods = []
        
        if hasattr(component, 'emit_event'):
            event_methods.append('emit_event')
        if hasattr(component, 'subscribe'):
            event_methods.append('subscribe')
        if hasattr(component, 'unsubscribe'):
            event_methods.append('unsubscribe')
        if hasattr(component, 'handle_event'):
            event_methods.append('handle_event')
        if hasattr(component, 'register_handler'):
            event_methods.append('register_handler')
        
        if event_methods:
            event_system_components.append((comp_name, event_methods))
            print(f"   📡 {comp_name}: {len(event_methods)} методов событий")
            print(f"      -> {', '.join(event_methods)}")
    
    if not event_system_components:
        print("   ❌ Событийная система не найдена")
    
    # 5. Рекомендации по интеграции
    print("\n5. 🚀 Рекомендации по интеграции...")
    
    print("\n   📋 Приоритетные компоненты для интеграции:")
    
    # Базовые компоненты
    base_components = ['BaseComponent', 'SystemState']
    print("      🔧 Базовые:")
    for comp in base_components:
        print(f"         - {comp}")
    
    # Событийная система
    print("      📡 Событийная система:")
    print("         - EventBus - центральный шина событий")
    print("         - EventManager - менеджер событий")
    print("         - EventHandler - базовый обработчик")
    
    # Автономные модули
    print("      🤖 Автономные модули:")
    for module, status in autonomous_status.items():
        if status['available']:
            print(f"         - {module} (улучшить автономность: {status['autonomous_score']:.1%})")
        else:
            print(f"         - {module} (требуется интеграция)")
    
    # Стратегии
    print("      🧠 Стратегии:")
    print("         - ContradictionResolutionStrategy")
    print("         - LearningStrategy")
    print("         - AdaptationStrategy")
    
    # Фоновые задания
    print("      ⏰ Фоновые задания:")
    print("         - BackgroundJobManager")
    print("         - ScheduledTaskManager")
    
    return True

def plan_event_system_integration():
    """Планирование интеграции событийной системы"""
    
    print("\n🚀 Планирование интеграции событийной системы...")
    
    event_system_plan = {
        'core_components': [
            'EventBus - центральный шин событий',
            'EventManager - управление событиями',
            'EventHandler - базовый класс обработчиков',
            'EventTypes - типы событий'
        ],
        'integration_points': [
            'CoreBrain - центральный диспетчер',
            'ComponentInitializer - регистрация обработчиков',
            'AnalyticsManager - события аналитики',
            'ContradictionManager - события противоречий',
            'EthicsFramework - события этики',
            'LearningManager - события обучения'
        ],
        'autonomous_features': [
            'Автоматический запуск аналитики',
            'Самообучение на основе событий',
            'Обнаружение противоречий',
            'Этический мониторинг',
            'Веб-поиск по триггерам',
            'Обновление графа знаний'
        ],
        'event_types': [
            'system.start',
            'system.stop',
            'component.initialized',
            'component.error',
            'learning.completed',
            'contradiction.detected',
            'ethics.violation',
            'analytics.insight',
            'knowledge.updated',
            'web.search.completed'
        ]
    }
    
    print("\n   📋 План интеграции:")
    
    for category, items in event_system_plan.items():
        print(f"\n   📂 {category.replace('_', ' ').title()}:")
        for item in items:
            print(f"      - {item}")
    
    return event_system_plan

if __name__ == "__main__":
    print("🔍 Анализ архитектуры и планирование интеграции")
    print("=" * 60)
    
    success = analyze_current_architecture()
    
    if success:
        plan = plan_event_system_integration()
        
        print("\n🎉 Анализ завершен!")
        print("\n📋 Следующие шаги:")
        print("   1. Создать событийную систему")
        print("   2. Интегрировать BaseComponent")
        print("   3. Добавить SystemState")
        print("   4. Улучшить автономность модулей")
        print("   5. Интегрировать стратегии разрешения")
        
    else:
        print("❌ Ошибка при анализе архитектуры")
    
    sys.exit(0 if success else 1)
