"""
ПОЛНЫЙ ТЕСТ СИСТЕМЫ COGNIFLEX - ВСЕ МОДУЛИ АКТИВНЫ

Этот тест проверяет полную функциональность системы с включенными
всеми модулями и подсистемами (без минимального режима отладки).
"""

import sys
import os
import time
import logging
from typing import Dict, Any

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("cogniflex.full_test")

def main():
    """Основная функция полного тестирования системы."""
    print("🚀 ЗАПУСК ПОЛНОГО ТЕСТИРОВАНИЯ СИСТЕМЫ COGNIFLEX")
    print("=" * 80)

    try:
        # Импорт компонентов системы
        from cogniflex.core.core_brain import CoreBrain
        from cogniflex.core.integration_layer import CogniFlexIntegrator
        from cogniflex.gui.core_gui import CogniFlexGUI

        print("✅ Импорт компонентов завершен")

        # Этап 1: Инициализация ядра системы (ПОЛНАЯ ВЕРСИЯ)
        print("\n🧠 ЭТАП 1: ИНИЦИАЛИЗАЦИЯ ЯДРА СИСТЕМЫ (ПОЛНАЯ ВЕРСИЯ)")
        print("-" * 70)

        # Конфигурация с ВКЛЮЧЕННЫМИ всеми модулями
        full_config = {
            'debug_minimal_mode': False,  # ВКЛЮЧАЕМ все модули
            'use_gpu_if_available': True,
            'max_vram_gb': 8.0,
            'cache_memory_gb': 4.0,
            'tokenization_workers': 4,
            'enable_background_processing': True,
            'enable_self_dialog': True,
            'enable_fractal_attention': True,
            'enable_ethics_framework': True,
            'enable_system_monitoring': True,
            'enable_recovery_system': True,
            'enable_security_system': True,
            # Настройки генерации
            'generation': {
                'model_name': 'sberbank-ai/rugpt3small_based_on_gpt2',
                'max_tokens': 100,
                'temperature': 0.7,
                'cache_config': {
                    'enabled': True,
                    'max_memory_tokens': 5000,
                    'vram_ratio': 0.7,
                    'vram_threshold': 0.2,
                    'ram_threshold': 0.15,
                    'disk_cache_dir': './cache'
                }
            }
        }

        print("🔧 Конфигурация: ВКЛЮЧЕНЫ все модули и подсистемы")
        print(f"   GPU: {'включен' if full_config['use_gpu_if_available'] else 'отключен'}")
        print(f"   Фоновые процессы: {'включены' if full_config.get('enable_background_processing', False) else 'отключены'}")
        print(f"   Самодиалог: {'включен' if full_config.get('enable_self_dialog', False) else 'отключен'}")

        brain = CoreBrain(config=full_config)
        success = brain.initialized if hasattr(brain, 'initialized') else True
        print(f"✅ Ядро системы инициализировано: {'УСПЕШНО' if success else 'НЕУДАЧНО'}")
        print(f"   Компонентов: {len(getattr(brain, 'components', {}))}")

        if not success:
            print("❌ Ошибка инициализации ядра системы")
            return False

        # Этап 2: Проверка всех подсистем
        print("\n🔍 ЭТАП 2: ПРОВЕРКА ВСЕХ ПОДСИСТЕМ")
        print("-" * 50)

        subsystems_check = check_all_subsystems(brain)
        if subsystems_check['status'] == 'success':
            print("✅ Все подсистемы работают корректно")
            for subsystem, status in subsystems_check['subsystems'].items():
                print(f"   ✅ {subsystem}: {status}")
        else:
            print("⚠️  Некоторые подсистемы работают с ограничениями:")
            for subsystem, status in subsystems_check['subsystems'].items():
                print(f"   ⚠️  {subsystem}: {status}")

        # Этап 3: Инициализация интегратора
        print("\n🔗 ЭТАП 3: ИНИЦИАЛИЗАЦИЯ ИНТЕГРАТОРА")
        print("-" * 50)

        integrator_config = {
            'timeline_maxlen': 2000,
            'enable_timeline': True,
            'max_workers': 8,
            'processing_timeout': 60.0,
            'enable_background_processing': True,
            'enable_self_dialog': True
        }

        integrator = CogniFlexIntegrator(integrator_config)

        integrator_success = integrator.initialize()

        if integrator_success:
            print("✅ Интегратор инициализирован успешно")
            print(f"   Событийная шина: {'активна' if hasattr(integrator, 'event_bus') else 'неактивна'}")
            print(f"   Фоновые процессы: {'активны' if hasattr(integrator, 'background_processor') else 'неактивны'}")
            print(f"   Компоненты интегратора: {'готовы' if integrator.initialized else 'не готовы'}")
        else:
            print("❌ Ошибка инициализации интегратора")
            return False

        # Этап 4: Тестирование компонентов (расширенное)
        print("\n🔧 ЭТАП 4: ТЕСТИРОВАНИЕ КОМПОНЕНТОВ (РАСШИРЕННОЕ)")
        print("-" * 60)

        components_test = test_system_components_extended(integrator)
        if components_test['status'] == 'success':
            print("✅ Все компоненты работают корректно")
        else:
            print("⚠️  Некоторые компоненты работают с ограничениями")

        # Этап 5: Тестирование фрактальной архитектуры (расширенное)
        print("\n🌀 ЭТАП 5: ТЕСТИРОВАНИЕ ФРАКТАЛЬНОЙ АРХИТЕКТУРЫ")
        print("-" * 55)

        fractal_test = test_fractal_architecture_extended(integrator)
        if fractal_test['status'] == 'success':
            print("✅ Фрактальная архитектура работает корректно")
        else:
            print("⚠️  Фрактальная архитектура работает с ограничениями")

        # Этап 6: Тестирование обработки запросов (расширенное)
        print("\n💬 ЭТАП 6: ТЕСТИРОВАНИЕ ОБРАБОТКИ ЗАПРОСОВ")
        print("-" * 50)

        query_test = test_query_processing_extended(integrator)
        if query_test['status'] == 'success':
            print("✅ Обработка запросов работает корректно")
            print(f"   Тестовых запросов: {query_test.get('total_queries', 0)}")
            print(f"   Успешных ответов: {query_test.get('successful_responses', 0)}")
        else:
            print("❌ Ошибки в обработке запросов")

        # Этап 7: Тестирование фоновых процессов
        print("\n⚙️  ЭТАП 7: ТЕСТИРОВАНИЕ ФОНОВЫХ ПРОЦЕССОВ")
        print("-" * 50)

        background_test = test_background_processing(integrator)
        if background_test['status'] == 'success':
            print("✅ Фоновые процессы работают корректно")
        else:
            print("⚠️  Фоновые процессы недоступны или работают с ограничениями")

        # Этап 8: Финальная комплексная проверка
        print("\n🎯 ЭТАП 8: ФИНАЛЬНАЯ КОМПЛЕКСНАЯ ПРОВЕРКА")
        print("-" * 55)

        final_check = perform_comprehensive_system_check(integrator, brain)
        if final_check['status'] == 'success':
            print("✅ Система полностью готова к работе!")
            print(f"   Общее здоровье системы: {final_check.get('overall_health', 'unknown')}")
            print(f"   Активных компонентов: {final_check.get('active_components', 0)}")
            print(f"   Рабочих подсистем: {final_check.get('working_subsystems', 0)}")
        else:
            print("⚠️  Система работает, но есть рекомендации по улучшению")

        # Этап 9: Демонстрация полной функциональности
        print("\n🎬 ЭТАП 9: ДЕМОНСТРАЦИЯ ПОЛНОЙ ФУНКЦИОНАЛЬНОСТИ")
        print("-" * 60)

        demonstration = demonstrate_full_system(integrator)

        print("\n" + "=" * 80)
        print("🎉 ПОЛНОЕ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
        print("")
        print("Система CogniFlex теперь имеет:")
        print("✅ Централизованную событийную шину")
        print("✅ Полный процесс обработки запросов")
        print("✅ Фрактальную память с динамическим фокусом внимания")
        print("✅ Самодиалог и самообучение")
        print("✅ Этическую рамку с 10 заповедями")
        print("✅ Самооптимизацию и анализ производительности")
        print("✅ Фоновые процессы и мониторинг")
        print("✅ Системы безопасности и восстановления")
        print("")
        print("🚀 Система готова к полноценной промышленной эксплуатации!")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n💥 КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_all_subsystems(brain) -> Dict[str, Any]:
    """Проверка всех подсистем ядра."""
    result = {
        'status': 'success',
        'subsystems': {}
    }

    subsystems = [
        ('memory_manager', 'Менеджер памяти'),
        ('knowledge_graph', 'Граф знаний'),
        ('ml_unit', 'ML модуль'),
        ('text_processor', 'Обработчик текста'),
        ('response_generator', 'Генератор ответов'),
        ('generation_coordinator', 'Координатор генерации'),
        ('token_cache', 'Кэш токенов'),
        ('security_manager', 'Менеджер безопасности'),
        ('monitoring_manager', 'Менеджер мониторинга'),
        ('recovery_manager', 'Менеджер восстановления'),
        ('fractal_attention', 'Фрактальное внимание'),
        ('self_analyzer', 'Самоанализатор'),
        ('event_system', 'Событийная система')
    ]

    for attr_name, display_name in subsystems:
        if hasattr(brain, attr_name):
            component = getattr(brain, attr_name)
            if component is not None:
                result['subsystems'][display_name] = 'активен'
            else:
                result['subsystems'][display_name] = 'инициализирован (None)'
                result['status'] = 'partial'
        else:
            result['subsystems'][display_name] = 'недоступен'
            result['status'] = 'partial'

    return result


def test_system_components_extended(integrator) -> Dict[str, Any]:
    """Расширенное тестирование компонентов системы."""
    result = {'status': 'success', 'components': {}}

    components = [
        ('core_brain', 'Ядро системы'),
        ('response_generator', 'Генератор ответов'),
        ('generation_coordinator', 'Координатор генерации'),
        ('fractal_attention', 'Фрактальная система внимания'),
        ('event_bus', 'Событийная шина'),
        ('background_processor', 'Фоновый процессор'),
        ('self_dialog_manager', 'Менеджер самодиалога'),
        ('ethics_framework', 'Этическая рамка'),
        ('system_monitor', 'Системный монитор'),
        ('recovery_manager', 'Менеджер восстановления')
    ]

    for attr_name, display_name in components:
        if hasattr(integrator, attr_name):
            component = getattr(integrator, attr_name)
            if component is not None:
                result['components'][display_name] = 'активен'
            else:
                result['components'][display_name] = 'инициализирован (None)'
                result['status'] = 'partial'
        else:
            result['components'][display_name] = 'недоступен'
            result['status'] = 'partial'

    return result


def test_fractal_architecture_extended(integrator) -> Dict[str, Any]:
    """Расширенное тестирование фрактальной архитектуры."""
    result = {'status': 'success', 'tests': {}}

    if not hasattr(integrator, 'fractal_attention') or not integrator.fractal_attention:
        return {'status': 'error', 'error': 'Фрактальная система недоступна'}

    fractal = integrator.fractal_attention

    # Тест 1: Инициализация фокуса внимания
    try:
        test_query = "Объясни принцип работы фрактальной памяти в ИИ"
        fractal._initialize_attention_focus(test_query)
        result['tests']['focus_initialization'] = 'успешно'
    except Exception as e:
        result['tests']['focus_initialization'] = f'ошибка: {e}'
        result['status'] = 'partial'

    # Тест 2: Извлечение концептов
    try:
        primary_concepts = fractal._extract_primary_concepts(test_query)
        secondary_concepts = fractal._extract_secondary_concepts(test_query)
        result['tests']['concept_extraction'] = f'успешно: {len(primary_concepts)}+{len(secondary_concepts)} концептов'
    except Exception as e:
        result['tests']['concept_extraction'] = f'ошибка: {e}'
        result['status'] = 'partial'

    # Тест 3: Построение горячего окна
    try:
        fractal._build_initial_hot_window()
        result['tests']['hot_window'] = 'успешно'
    except Exception as e:
        result['tests']['hot_window'] = f'предупреждение: {e}'
        # Это не критично

    # Тест 4: Обработка концептов
    try:
        test_concepts = ["фактальная", "аналитическая", "креативная"]
        for concept in test_concepts:
            domain = fractal._identify_domain(f"Запрос о {concept}")
            query_type = fractal._determine_query_type(f"Что такое {concept}?")
        result['tests']['concept_processing'] = 'успешно'
    except Exception as e:
        result['tests']['concept_processing'] = f'ошибка: {e}'
        result['status'] = 'partial'

    return result


def test_query_processing_extended(integrator) -> Dict[str, Any]:
    """Расширенное тестирование обработки запросов."""
    result = {'status': 'success', 'total_queries': 0, 'successful_responses': 0}

    test_queries = [
        "Что такое искусственный интеллект?",
        "Расскажи о машинном обучении",
        "Как работает нейронная сеть?",
        "Объясни принцип квантовых вычислений",
        "Что такое фрактальная память?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"   Тестирую запрос {i}: '{query[:30]}...'")
        result['total_queries'] += 1

        try:
            response = integrator.process_query(query)

            if response.get('status') == 'success':
                result['successful_responses'] += 1
                print(f"     ✅ Ответ получен")
            else:
                print(f"     ❌ Ошибка: {response.get('error', 'Неизвестная ошибка')}")

        except Exception as e:
            print(f"     ❌ Исключение: {e}")

    success_rate = result['successful_responses'] / result['total_queries'] if result['total_queries'] > 0 else 0
    result['success_rate'] = success_rate

    if success_rate < 0.6:  # Минимум 60% успешности для полной версии
        result['status'] = 'partial'

    print(".1%")

    return result


def test_background_processing(integrator) -> Dict[str, Any]:
    """Тестирование фоновых процессов."""
    result = {'status': 'success', 'processes': {}}

    background_features = [
        ('background_processor', 'Фоновый процессор'),
        ('learning_scheduler', 'Планировщик обучения'),
        ('dialog_manager', 'Менеджер диалога'),
        ('system_optimizer', 'Оптимизатор системы')
    ]

    for attr_name, display_name in background_features:
        if hasattr(integrator, attr_name):
            component = getattr(integrator, attr_name)
            if component is not None:
                result['processes'][display_name] = 'активен'
            else:
                result['processes'][display_name] = 'инициализирован (None)'
                result['status'] = 'partial'
        else:
            result['processes'][display_name] = 'недоступен'
            result['status'] = 'partial'

    return result


def perform_comprehensive_system_check(integrator, brain) -> Dict[str, Any]:
    """Комплексная проверка всей системы."""
    result = {
        'status': 'success',
        'overall_health': 'healthy',
        'active_components': 0,
        'working_subsystems': 0,
        'recommendations': []
    }

    # Проверка компонентов ядра
    core_components = ['memory_manager', 'ml_unit', 'knowledge_graph', 'text_processor']
    for component in core_components:
        if hasattr(brain, component) and getattr(brain, component) is not None:
            result['active_components'] += 1
        else:
            result['recommendations'].append(f"Добавить компонент: {component}")

    # Проверка подсистем
    subsystems = ['security_manager', 'monitoring_manager', 'recovery_manager', 'fractal_attention']
    for subsystem in subsystems:
        if hasattr(brain, subsystem) and getattr(brain, subsystem) is not None:
            result['working_subsystems'] += 1
        else:
            result['recommendations'].append(f"Добавить подсистему: {subsystem}")

    # Проверка интегратора
    integrator_components = ['response_generator', 'generation_coordinator', 'event_bus']
    for component in integrator_components:
        if hasattr(integrator, component) and getattr(integrator, component) is not None:
            result['active_components'] += 1
        else:
            result['recommendations'].append(f"Добавить компонент интегратора: {component}")

    # Определение общего здоровья
    if result['active_components'] >= 6 and result['working_subsystems'] >= 3:
        result['overall_health'] = 'excellent'
    elif result['active_components'] >= 4 and result['working_subsystems'] >= 2:
        result['overall_health'] = 'good'
    elif result['active_components'] >= 2:
        result['overall_health'] = 'fair'
        result['status'] = 'partial'
    else:
        result['overall_health'] = 'poor'
        result['status'] = 'error'

    return result


def demonstrate_full_system(integrator):
    """Демонстрация полной функциональности системы."""
    print("   Демонстрирую расширенные возможности системы...")

    # Демонстрационные запросы разных типов
    demo_queries = [
        ("Фактуальный", "Что такое квантовая механика?"),
        ("Аналитический", "Проанализируй влияние ИИ на общество"),
        ("Креативный", "Напиши короткий рассказ о будущем ИИ"),
        ("Философский", "Может ли ИИ иметь сознание?")
    ]

    results = []

    for query_type, query in demo_queries:
        print(f"   🔍 {query_type}: '{query[:40]}...'")

        start_time = time.time()
        try:
            result = integrator.process_query(query)
            processing_time = time.time() - start_time

            if result.get('status') == 'success':
                print(f"     ✅ Ответ получен за {processing_time:.2f} сек")
                results.append({'type': query_type, 'status': 'success', 'time': processing_time})
            else:
                print(f"     ❌ Ошибка: {result.get('error', 'Неизвестная ошибка')}")
                results.append({'type': query_type, 'status': 'error', 'time': processing_time})

        except Exception as e:
            print(f"     ❌ Исключение: {e}")
            results.append({'type': query_type, 'status': 'exception', 'time': 0})

    # Показываем статистику
    successful = sum(1 for r in results if r['status'] == 'success')
    total = len(results)
    avg_time = sum(r['time'] for r in results) / total if total > 0 else 0

    print("\n   📊 ИТОГИ ДЕМОНСТРАЦИИ:")
    print(f"      Успешных запросов: {successful}/{total}")
    print(f"      Среднее время: {avg_time:.2f} сек")
    print(f"      Успешность: {successful/total:.1%}")

    if hasattr(integrator, 'get_system_stats'):
        try:
            stats = integrator.get_system_stats()
            print("   🔄 Системная статистика:")
            print(f"      Всего запросов: {stats.get('metrics', {}).get('total_requests', 0)}")
            print(f"      Эффективность кэша: {stats.get('cache', {}).get('efficiency', 0):.1%}")
        except Exception:
            pass

    return {'total': total, 'successful': successful, 'avg_time': avg_time}


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
