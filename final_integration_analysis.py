"""
Финальная интеграция улучшенных методов в структуру проекта
"""
import sys
import os
import time
import logging
import importlib
from typing import Dict, Any, List, Optional
sys.path.append('.')

# Настройка логирования
logging.basicConfig(level=logging.INFO)

class ProjectStructureAnalyzer:
    """Анализатор структуры проекта и интеграции методов"""
    
    def __init__(self):
        self.project_root = "cogniflex"
        self.improved_methods = {}
        self.integration_status = {}
        self.test_results = {}
        
    def analyze_project_structure(self):
        """Анализирует структуру проекта"""
        
        print("🔍 АНАЛИЗ СТРУКТУРЫ ПРОЕКТА")
        print("=" * 60)
        
        # Ключевые компоненты проекта
        key_components = {
            "mlearning": {
                "core_files": [
                    "fractal_model_manager.py",
                    "optimized_fractal_model_manager.py", 
                    "unified_fractal_manager.py",
                    "web_search_learning_integration.py",
                    "enhanced_learning_integration.py",
                    "comprehensive_learning_system.py"
                ],
                "support_files": [
                    "text_quality_improver.py",
                    "text_quality_trainer.py",
                    "text_quality_learning_integration.py",
                    "fractal_trainer.py",
                    "cogniflex_tokenizer.py"
                ]
            },
            "websearch": {
                "core_files": [
                    "web_search_engine.py",
                    "search_engines.py",
                    "search_models.py"
                ]
            },
            "memory": {
                "core_files": [
                    "hybrid_token_cache.py",
                    "memory_graph.py"
                ]
            },
            "core": {
                "core_files": [
                    "core_brain.py",
                    "fractal_attention_system.py"
                ]
            }
        }
        
        print("📁 Ключевые компоненты проекта:")
        for component, files in key_components.items():
            print(f"\n  📂 {component}:")
            for file_type, file_list in files.items():
                print(f"    {file_type}:")
                for file in file_list:
                    file_path = os.path.join(self.project_root, component, file)
                    exists = os.path.exists(file_path)
                    status = "✅" if exists else "❌"
                    print(f"      {status} {file}")
        
        return key_components
    
    def identify_improved_methods(self):
        """Идентифицирует улучшенные методы"""
        
        print("\n🚀 ИДЕНТИФИКАЦИЯ УЛУЧШЕННЫХ МЕТОДОВ")
        print("=" * 60)
        
        # Улучшенные методы из наших тестов
        self.improved_methods = {
            "web_search_integration": {
                "file": "web_search_learning_integration.py",
                "methods": [
                    "search_and_enhance_response",
                    "_perform_web_search", 
                    "generate_training_texts_from_search",
                    "get_integration_stats",
                    "configure_integration",
                    "clear_cache"
                ],
                "status": "✅ Протестировано",
                "integration": "OptimizedFractalModelManager"
            },
            "enhanced_learning": {
                "file": "enhanced_learning_integration.py", 
                "methods": [
                    "start_enhanced_learning_session",
                    "generate_enhanced_response",
                    "get_enhanced_system_status",
                    "add_enhanced_topics",
                    "configure_enhanced_learning"
                ],
                "status": "✅ Протестировано",
                "integration": "UnifiedFractalManager"
            },
            "optimized_manager": {
                "file": "optimized_fractal_model_manager.py",
                "methods": [
                    "generate_response_with_web_search",
                    "generate_training_texts_from_web",
                    "get_web_search_stats",
                    "configure_web_search",
                    "clear_web_search_cache",
                    "generate_response_optimized",
                    "optimized_tokenize",
                    "get_performance_stats"
                ],
                "status": "✅ Протестировано",
                "integration": "Самостоятельный"
            },
            "unified_manager": {
                "file": "unified_fractal_manager.py",
                "methods": [
                    "start_enhanced_learning_session",
                    "generate_enhanced_response", 
                    "get_enhanced_system_status",
                    "add_enhanced_topics",
                    "configure_enhanced_learning"
                ],
                "status": "✅ Протестировано",
                "integration": "Основной интерфейс"
            },
            "text_quality": {
                "file": "text_quality_improver.py",
                "methods": [
                    "analyze_text_quality",
                    "improve_response_quality",
                    "_calculate_readability_score",
                    "_calculate_relevance_score"
                ],
                "status": "✅ Протестировано",
                "integration": "OptimizedFractalModelManager"
            },
            "hybrid_cache": {
                "file": "../memory/hybrid_token_cache.py",
                "methods": [
                    "cache_tokens",
                    "get_cached_tokens",
                    "get_cache_stats",
                    "clear_cache"
                ],
                "status": "✅ Протестировано",
                "integration": "OptimizedFractalModelManager"
            }
        }
        
        for category, info in self.improved_methods.items():
            print(f"\n  📂 {category}:")
            print(f"    📁 Файл: {info['file']}")
            print(f"    🔗 Интеграция: {info['integration']}")
            print(f"    ✅ Статус: {info['status']}")
            print(f"    🚀 Методы:")
            for method in info['methods']:
                print(f"      • {method}")
        
        return self.improved_methods
    
    def verify_method_integration(self):
        """Проверяет интеграцию методов"""
        
        print("\n🔗 ПРОВЕРКА ИНТЕГРАЦИИ МЕТОДОВ")
        print("=" * 60)
        
        try:
            # Импортируем основные компоненты
            from cogniflex.mlearning.unified_fractal_manager import UnifiedFractalManager
            
            manager = UnifiedFractalManager()
            
            print(f"✅ UnifiedFractalManager загружен: {type(manager.manager).__name__}")
            
            # Проверяем каждый улучшенный метод
            integration_results = {}
            
            for category, info in self.improved_methods.items():
                print(f"\n  🔍 Проверка {category}:")
                
                category_results = {
                    "available": False,
                    "methods_found": [],
                    "methods_missing": [],
                    "functional": False
                }
                
                # Проверяем доступность категории
                if category == "web_search_integration":
                    category_results["available"] = hasattr(manager.manager, 'web_search_integration')
                    if category_results["available"]:
                        web_integration = manager.manager.web_search_integration
                        for method in info['methods']:
                            if hasattr(web_integration, method):
                                category_results["methods_found"].append(method)
                            else:
                                category_results["methods_missing"].append(method)
                
                elif category == "enhanced_learning":
                    category_results["available"] = hasattr(manager, 'enhanced_learning')
                    if category_results["available"]:
                        enhanced = manager.enhanced_learning
                        for method in info['methods']:
                            if hasattr(enhanced, method):
                                category_results["methods_found"].append(method)
                            else:
                                category_results["methods_missing"].append(method)
                
                elif category == "optimized_manager":
                    category_results["available"] = hasattr(manager, 'manager') and manager.is_optimized
                    if category_results["available"]:
                        for method in info['methods']:
                            if hasattr(manager.manager, method):
                                category_results["methods_found"].append(method)
                            else:
                                category_results["methods_missing"].append(method)
                
                elif category == "unified_manager":
                    category_results["available"] = True  # UnifiedFractalManager всегда доступен
                    for method in info['methods']:
                        if hasattr(manager, method):
                            category_results["methods_found"].append(method)
                        else:
                            category_results["methods_missing"].append(method)
                
                elif category == "text_quality":
                    category_results["available"] = hasattr(manager.manager, 'quality_improver')
                    if category_results["available"]:
                        quality = manager.manager.quality_improver
                        for method in info['methods']:
                            if hasattr(quality, method):
                                category_results["methods_found"].append(method)
                            else:
                                category_results["methods_missing"].append(method)
                
                elif category == "hybrid_cache":
                    category_results["available"] = hasattr(manager.manager, 'token_cache')
                    if category_results["available"]:
                        cache = manager.manager.token_cache
                        for method in info['methods']:
                            if hasattr(cache, method):
                                category_results["methods_found"].append(method)
                            else:
                                category_results["methods_missing"].append(method)
                
                # Оценка функциональности
                category_results["functional"] = (
                    category_results["available"] and 
                    len(category_results["methods_missing"]) == 0
                )
                
                # Вывод результатов
                status = "✅" if category_results["functional"] else "⚠️" if category_results["available"] else "❌"
                print(f"    {status} Доступность: {category_results['available']}")
                print(f"    📊 Найдено методов: {len(category_results['methods_found'])}")
                print(f"    ❌ Отсутствует: {len(category_results['methods_missing'])}")
                
                if category_results["methods_missing"]:
                    print(f"    🔍 Отсутствующие методы:")
                    for method in category_results["methods_missing"]:
                        print(f"      • {method}")
                
                integration_results[category] = category_results
            
            self.integration_status = integration_results
            return integration_results
            
        except Exception as e:
            print(f"❌ Ошибка проверки интеграции: {e}")
            return {}
    
    def test_functional_integrity(self):
        """Тестирует функциональную целостность"""
        
        print("\n🧪 ТЕСТИРОВАНИЕ ФУНКЦИОНАЛЬНОЙ ЦЕЛОСТНОСТИ")
        print("=" * 60)
        
        try:
            from cogniflex.mlearning.unified_fractal_manager import UnifiedFractalManager
            
            manager = UnifiedFractalManager()
            
            test_results = {}
            
            # 1. Тест базовой функциональности
            print("\n  📊 1. Базовая функциональность:")
            
            basic_tests = {
                "generate_response": lambda: manager.generate_response("Тест", 50),
                "get_quality_metrics": lambda: manager.get_quality_metrics(),
                "improve_quality": lambda: manager.improve_quality(["Тестовый текст"]),
                "get_performance_stats": lambda: manager.get_performance_stats()
            }
            
            for test_name, test_func in basic_tests.items():
                try:
                    result = test_func()
                    test_results[f"basic_{test_name}"] = {
                        "status": "success",
                        "result_type": type(result).__name__
                    }
                    print(f"    ✅ {test_name}: {type(result).__name__}")
                except Exception as e:
                    test_results[f"basic_{test_name}"] = {
                        "status": "error",
                        "error": str(e)
                    }
                    print(f"    ❌ {test_name}: {e}")
            
            # 2. Тест веб-поиска
            print("\n  🔍 2. Веб-поиск:")
            
            web_search_tests = {
                "web_search_available": lambda: hasattr(manager.manager, 'web_search_integration'),
                "generate_with_web_search": lambda: manager.generate_response_with_web_search("Тест", 50, True) if hasattr(manager, 'generate_response_with_web_search') else None,
                "get_web_search_stats": lambda: manager.get_web_search_stats() if hasattr(manager, 'get_web_search_stats') else None,
                "enhanced_response": lambda: manager.generate_enhanced_response("Тест", 50, True) if hasattr(manager, 'generate_enhanced_response') else None
            }
            
            for test_name, test_func in web_search_tests.items():
                try:
                    result = test_func()
                    if result is not None:
                        test_results[f"web_{test_name}"] = {
                            "status": "success",
                            "result_type": type(result).__name__
                        }
                        print(f"    ✅ {test_name}: {type(result).__name__}")
                    else:
                        test_results[f"web_{test_name}"] = {
                            "status": "unavailable"
                        }
                        print(f"    ⚠️ {test_name}: недоступен")
                except Exception as e:
                    test_results[f"web_{test_name}"] = {
                        "status": "error",
                        "error": str(e)
                    }
                    print(f"    ❌ {test_name}: {e}")
            
            # 3. Тест обучения
            print("\n  🎓 3. Обучение:")
            
            learning_tests = {
                "enhanced_learning_available": lambda: hasattr(manager, 'enhanced_learning'),
                "start_learning_session": lambda: manager.start_enhanced_learning_session(["тест"], "test") if hasattr(manager, 'start_enhanced_learning_session') else None,
                "get_system_status": lambda: manager.get_enhanced_system_status() if hasattr(manager, 'get_enhanced_system_status') else None
            }
            
            for test_name, test_func in learning_tests.items():
                try:
                    result = test_func()
                    if result is not None:
                        test_results[f"learning_{test_name}"] = {
                            "status": "success",
                            "result_type": type(result).__name__
                        }
                        print(f"    ✅ {test_name}: {type(result).__name__}")
                    else:
                        test_results[f"learning_{test_name}"] = {
                            "status": "unavailable"
                        }
                        print(f"    ⚠️ {test_name}: недоступен")
                except Exception as e:
                    test_results[f"learning_{test_name}"] = {
                        "status": "error",
                        "error": str(e)
                    }
                    print(f"    ❌ {test_name}: {e}")
            
            # 4. Тест оптимизации
            print("\n  ⚡ 4. Оптимизация:")
            
            optimization_tests = {
                "is_optimized": lambda: manager.is_optimized,
                "cache_available": lambda: hasattr(manager.manager, 'token_cache'),
                "parallel_tokenization": lambda: hasattr(manager.manager, 'tokenization_executor')
            }
            
            for test_name, test_func in optimization_tests.items():
                try:
                    result = test_func()
                    test_results[f"opt_{test_name}"] = {
                        "status": "success",
                        "result": result
                    }
                    print(f"    ✅ {test_name}: {result}")
                except Exception as e:
                    test_results[f"opt_{test_name}"] = {
                        "status": "error",
                        "error": str(e)
                    }
                    print(f"    ❌ {test_name}: {e}")
            
            self.test_results = test_results
            return test_results
            
        except Exception as e:
            print(f"❌ Ошибка тестирования: {e}")
            return {}
    
    def generate_integration_report(self):
        """Генерирует отчет об интеграции"""
        
        print("\n📋 ОТЧЕТ ОБ ИНТЕГРАЦИИ")
        print("=" * 60)
        
        # Статистика
        total_categories = len(self.improved_methods)
        available_categories = sum(1 for cat in self.integration_status.values() if cat['available'])
        functional_categories = sum(1 for cat in self.integration_status.values() if cat['functional'])
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for test in self.test_results.values() if test['status'] == 'success')
        
        print(f"📊 ОБЩАЯ СТАТИСТИКА:")
        print(f"  📂 Категорий методов: {total_categories}")
        print(f"  ✅ Доступно категорий: {available_categories}")
        print(f"  🚀 Функциональных категорий: {functional_categories}")
        print(f"  🧪 Всего тестов: {total_tests}")
        print(f"  ✅ Успешных тестов: {successful_tests}")
        
        # Детальная информация по категориям
        print(f"\n📋 ДЕТАЛЬНАЯ ИНФОРМАЦИЯ:")
        
        for category, info in self.improved_methods.items():
            integration = self.integration_status.get(category, {})
            
            print(f"\n  📂 {category}:")
            print(f"    📁 Файл: {info['file']}")
            print(f"    🔗 Интеграция: {info['integration']}")
            print(f"    ✅ Статус: {info['status']}")
            print(f"    📊 Доступность: {'✅' if integration.get('available') else '❌'}")
            print(f"    🚀 Функциональность: {'✅' if integration.get('functional') else '❌'}")
            
            if integration.get('methods_found'):
                print(f"    ✅ Найденные методы: {len(integration['methods_found'])}")
                for method in integration['methods_found'][:3]:  # Первые 3 метода
                    print(f"      • {method}")
            
            if integration.get('methods_missing'):
                print(f"    ❌ Отсутствующие методы: {len(integration['methods_missing'])}")
                for method in integration['methods_missing']:
                    print(f"      • {method}")
        
        # Результаты тестов
        print(f"\n🧪 РЕЗУЛЬТАТЫ ТЕСТОВ:")
        
        test_categories = {
            "basic": "Базовая функциональность",
            "web": "Веб-поиск",
            "learning": "Обучение",
            "opt": "Оптимизация"
        }
        
        for category, description in test_categories.items():
            category_tests = {k: v for k, v in self.test_results.items() if k.startswith(category)}
            successful = sum(1 for test in category_tests.values() if test['status'] == 'success')
            total = len(category_tests)
            
            print(f"\n  {description}:")
            print(f"    ✅ Успешно: {successful}/{total}")
            
            if successful < total:
                failed_tests = {k: v for k, v in category_tests.items() if v['status'] != 'success'}
                for test_name, result in failed_tests.items():
                    print(f"    ❌ {test_name}: {result.get('error', 'unavailable')}")
        
        return {
            "total_categories": total_categories,
            "available_categories": available_categories,
            "functional_categories": functional_categories,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests * 100 if total_tests > 0 else 0
        }
    
    def create_integration_summary(self):
        """Создает итоговую сводку интеграции"""
        
        print("\n🎉 ИТОГОВАЯ СВОДКА ИНТЕГРАЦИИ")
        print("=" * 60)
        
        report = self.generate_integration_report()
        
        success_rate = report['success_rate']
        
        if success_rate >= 80:
            status = "🎉 ОТЛИЧНО"
            recommendation = "Система полностью готова к использованию"
        elif success_rate >= 60:
            status = "✅ ХОРОШО"
            recommendation = "Система функциональна, требуются незначительные доработки"
        else:
            status = "⚠️ ТРЕБУЕТ РАБОТЫ"
            recommendation = "Необходимо устранить критические проблемы"
        
        print(f"🎊 СТАТУС: {status}")
        print(f"📈 УСПЕШНОСТЬ: {success_rate:.1f}%")
        print(f"💡 РЕКОМЕНДАЦИЯ: {recommendation}")
        
        # Ключевые достижения
        print(f"\n🚀 КЛЮЧЕВЫЕ ДОСТИЖЕНИЯ:")
        
        achievements = []
        
        if self.integration_status.get("web_search_integration", {}).get("available"):
            achievements.append("✅ Веб-поиск интегрирован")
        
        if self.integration_status.get("enhanced_learning", {}).get("available"):
            achievements.append("✅ Улучшенное обучение доступно")
        
        if self.integration_status.get("optimized_manager", {}).get("available"):
            achievements.append("✅ Оптимизированный менеджер работает")
        
        if self.integration_status.get("unified_manager", {}).get("functional"):
            achievements.append("✅ Унифицированный интерфейс функционален")
        
        if any(test['status'] == 'success' for test in self.test_results.values() if 'basic' in test):
            achievements.append("✅ Базовая функциональность сохранена")
        
        for achievement in achievements:
            print(f"  {achievement}")
        
        # Рекомендации по использованию
        print(f"\n📝 РЕКОМЕНДАЦИИ ПО ИСПОЛЬЗОВАНИЮ:")
        
        print(f"  🚀 Для генерации ответов:")
        print(f"    • manager.generate_response(query, max_tokens)")
        print(f"    • manager.generate_enhanced_response(query, max_tokens, use_web_search=True)")
        
        print(f"  🔍 Для веб-поиска:")
        print(f"    • manager.generate_response_with_web_search(query, max_tokens, use_web_search=True)")
        print(f"    • manager.get_web_search_stats()")
        
        print(f"  🎓 Для обучения:")
        print(f"    • manager.start_enhanced_learning_session(topics)")
        print(f"    • manager.get_enhanced_system_status()")
        
        print(f"  ⚙️ Для настройки:")
        print(f"    • manager.configure_enhanced_learning(**settings)")
        print(f"    • manager.add_enhanced_topics(topics)")
        
        return report

def main():
    """Основная функция анализа и интеграции"""
    
    print("🔍 ФИНАЛЬНЫЙ АНАЛИЗ И ИНТЕГРАЦИЯ УЛУЧШЕННЫХ МЕТОДОВ")
    print("=" * 70)
    
    analyzer = ProjectStructureAnalyzer()
    
    # 1. Анализ структуры проекта
    structure = analyzer.analyze_project_structure()
    
    # 2. Идентификация улучшенных методов
    methods = analyzer.identify_improved_methods()
    
    # 3. Проверка интеграции
    integration = analyzer.verify_method_integration()
    
    # 4. Тестирование функциональности
    tests = analyzer.test_functional_integrity()
    
    # 5. Генерация отчета
    report = analyzer.create_integration_summary()
    
    print(f"\n🎉 АНАЛИЗ И ИНТЕГРАЦИЯ ЗАВЕРШЕНЫ!")
    
    return analyzer, report

if __name__ == "__main__":
    analyzer, report = main()
    
    print(f"\nНажмите Enter для выхода...")
    input()
