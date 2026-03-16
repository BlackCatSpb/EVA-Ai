#!/usr/bin/env python3
"""
Диагностика компонентов CogniFlex - какие компоненты не запускаются
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

def diagnose_components():
    """Диагностика компонентов"""
    print("🔍 ДИАГНОСТИКА КОМПОНЕНТОВ COGNIFLEX")
    print("=" * 60)
    
    try:
        from cogniflex.core.core_brain import CoreBrain
        
        # Создаем ядро
        brain = CoreBrain()
        
        print("📊 ИНИЦИАЛИЗАЦИЯ ЯДРА:")
        print(f"   components dict: {len(brain.components)} компонентов")
        print(f"   components: {list(brain.components.keys())}")
        
        # Инициализируем
        print("\n🔧 ЗАПУСК ИНИЦИАЛИЗАЦИИ:")
        if brain.initialize():
            print("✅ Инициализация успешна")
            
            print(f"\n📋 КОМПОНЕНТЫ ПОСЛЕ ИНИЦИАЛИЗАЦИИ:")
            print(f"   Всего: {len(brain.components)}")
            for name, component in brain.components.items():
                status = "✅" if hasattr(component, 'start') else "⚠️"
                print(f"   {status} {name}: {type(component).__name__}")
            
            # Запускаем
            print(f"\n🚀 ЗАПУСК КОМПОНЕНТОВ:")
            if brain.start():
                print("✅ Запуск успешный")
            else:
                print("❌ Запуск неуспешный")
            
            # Проверяем состояние
            print(f"\n📊 СОСТОЯНИЕ СИСТЕМЫ:")
            health = brain.get_health_status()
            print(f"   Статус: {health.get('status', 'unknown')}")
            print(f"   Предупреждения: {health.get('warnings', [])}")
            print(f"   Ошибки: {health.get('errors', [])}")
            
        else:
            print("❌ Инициализация неуспешна")
            
    except Exception as e:
        print(f"❌ Ошибка диагностики: {e}")
        traceback.print_exc()

def check_component_imports():
    """Проверяет импорты компонентов"""
    print("\n🔍 ПРОВЕРКА ИМПОРТОВ КОМПОНЕНТОВ:")
    print("=" * 40)
    
    components_to_check = [
        ('ml_unit', 'cogniflex.mlearning.ml_unit.MLUnit'),
        ('memory_manager', 'cogniflex.memory.memory_manager.MemoryManager'),
        ('text_processor', 'cogniflex.nlp.text_processor.TextProcessor'),
        ('response_generator', 'cogniflex.core.response_generator.ResponseGenerator'),
        ('ethics_framework', 'cogniflex.ethics.ethics_framework.EthicsFramework'),
        ('knowledge_graph', 'cogniflex.knowledge.knowledge_graph.KnowledgeGraph'),
        ('contradiction_resolver', 'cogniflex.reasoning.contradiction_resolver.ContradictionResolver'),
        ('self_analyzer', 'cogniflex.self_analyzer.SelfAnalyzer'),
        ('analyzer_core', 'cogniflex.analyzer_core.AnalyzerCore'),
        ('neuromorphic_simulator', 'cogniflex.neuromorphic.simulator.NeuromorphicSimulator'),
    ]
    
    for name, import_path in components_to_check:
        try:
            module_path, class_name = import_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            component_class = getattr(module, class_name)
            print(f"✅ {name}: {component_class}")
        except Exception as e:
            print(f"❌ {name}: {e}")

def main():
    """Основная функция"""
    diagnose_components()
    check_component_imports()

if __name__ == "__main__":
    main()
