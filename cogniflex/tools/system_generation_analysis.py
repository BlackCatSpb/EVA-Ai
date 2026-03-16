"""
Комплексный анализ системы генерации текста CogniFlex
Изучение всех связанных модулей и их взаимодействия
"""
import sys
import os
import json
import logging
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Any
sys.path.append('.')

# Настройка логирования
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger("system_analysis")

class SystemAnalyzer:
    """Анализатор системы генерации текста"""
    
    def __init__(self):
        self.analysis_results = {}
        self.module_structure = {}
        self.dependency_graph = {}
        self.flow_analysis = {}
    
    def analyze_module_structure(self, module_path: str, module_name: str) -> Dict[str, Any]:
        """Анализирует структуру модуля"""
        logger.info(f"🔍 Анализ модуля: {module_name}")
        
        try:
            # Импортируем модуль
            if module_path.endswith('.py'):
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                module = importlib.import_module(module_path)
            
            # Анализируем структуру
            structure = {
                'name': module_name,
                'path': module_path,
                'classes': {},
                'functions': {},
                'imports': [],
                'dependencies': []
            }
            
            # Получаем все классы и функции
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and obj.__module__ == module.__name__:
                    structure['classes'][name] = {
                        'methods': [method for method in dir(obj) if not method.startswith('_')],
                        'doc': inspect.getdoc(obj) or "Нет документации",
                        'init_signature': str(inspect.signature(obj.__init__)) if hasattr(obj, '__init__') else "N/A"
                    }
                elif inspect.isfunction(obj) and obj.__module__ == module.__name__:
                    structure['functions'][name] = {
                        'signature': str(inspect.signature(obj)),
                        'doc': inspect.getdoc(obj) or "Нет документации"
                    }
            
            # Анализируем импорты
            try:
                source_lines = inspect.getsourcelines(module)
                if source_lines:
                    source_text = ''.join(source_lines)
                    
                    # Ищем импорты
                    import re
                    import_pattern = r'(?:from\s+(\S+)\s+import\s+(\S+)|import\s+(\S+))'
                    imports = re.findall(import_pattern, source_text)
                    
                    for imp in imports:
                        if imp[0]:  # from X import Y
                            structure['imports'].append(f"from {imp[0]} import {imp[1]}")
                        else:  # import X
                            structure['imports'].append(f"import {imp[2]}")
            except Exception as e:
                logger.warning(f"Ошибка анализа исходного кода: {e}")
            
            return structure
            
        except Exception as e:
            logger.error(f"Ошибка анализа модуля {module_name}: {e}")
            return {'error': str(e)}
    
    def analyze_generation_flow(self) -> Dict[str, Any]:
        """Анализирует поток генерации текста"""
        logger.info("🌊 Анализ потока генерации")
        
        flow = {
            'entry_points': [],
            'data_flow': [],
            'bottlenecks': [],
            'error_points': []
        }
        
        # 1. Анализ OptimizedFractalModelManager
        manager_analysis = self.analyze_module_structure(
            'cogniflex.mlearning.optimized_fractal_model_manager',
            'OptimizedFractalModelManager'
        )
        
        if 'error' not in manager_analysis:
            flow['entry_points'].append('OptimizedFractalModelManager')
            
            # Анализируем методы генерации
            classes = manager_analysis.get('classes', {})
            
            if 'OptimizedFractalModelManager' in classes:
                methods = classes['OptimizedFractalModelManager']['methods']
                
                generation_methods = [m for m in methods if 'generate' in m.lower()]
                flow['data_flow'].extend(generation_methods)
                
                logger.info(f"   📋 Методы генерации: {generation_methods}")
        
        # 2. Анализ зависимостей
        dependencies = [
            'cogniflex.mlearning.text_quality_trainer',
            'cogniflex.mlearning.text_quality_improver', 
            'cogniflex.web_search',
            'cogniflex.mlearning.web_search_integration',
            'cogniflex.memory.hybrid_token_cache',
            'transformers'
        ]
        
        for dep in dependencies:
            try:
                __import__(dep)
                flow['data_flow'].append(f"✅ {dep}")
            except ImportError as e:
                flow['error_points'].append(f"❌ {dep}: {e}")
                flow['bottlenecks'].append(dep)
        
        return flow
    
    def test_generation_pipeline(self) -> Dict[str, Any]:
        """Тестирует полный пайплайн генерации"""
        logger.info("🧪 Тестирование пайплайна генерации")
        
        test_results = {
            'manager_init': False,
            'model_load': False,
            'tokenizer_load': False,
            'generation_methods': False,
            'actual_generation': False,
            'errors': []
        }
        
        try:
            # 1. Тест инициализации менеджера
            from cogniflex.mlearning.optimized_fractal_model_manager import OptimizedFractalModelManager
            
            manager = OptimizedFractalModelManager()
            test_results['manager_init'] = True
            logger.info("   ✅ Менеджер инициализирован")
            
            # 2. Проверяем загрузку модели
            if hasattr(manager, 'model') and manager.model is not None:
                test_results['model_load'] = True
                param_count = sum(p.numel() for p in manager.model.parameters())
                logger.info(f"   ✅ Модель загружена: {param_count:,} параметров")
            else:
                test_results['errors'].append("Модель не загружена")
                logger.error("   ❌ Модель не загружена")
            
            # 3. Проверяем токенизатор
            if hasattr(manager, 'tokenizer') and manager.tokenizer is not None:
                test_results['tokenizer_load'] = True
                vocab_size = len(manager.tokenizer.get_vocab())
                logger.info(f"   ✅ Токенизатор загружен: {vocab_size:,} токенов")
            else:
                test_results['errors'].append("Токенизатор не загружен")
                logger.error("   ❌ Токенизатор не загружен")
            
            # 4. Проверяем методы генерации
            generation_methods = ['generate_text', 'generate_response', 'generate_response_optimized']
            available_methods = []
            
            for method in generation_methods:
                if hasattr(manager, method):
                    available_methods.append(method)
                    logger.info(f"   ✅ Метод {method} доступен")
                else:
                    test_results['errors'].append(f"Метод {method} отсутствует")
                    logger.error(f"   ❌ Метод {method} отсутствует")
            
            if available_methods:
                test_results['generation_methods'] = True
                logger.info(f"   ✅ Доступные методы: {available_methods}")
            
            # 5. Тест фактической генерации
            if test_results['manager_init'] and test_results['model_load'] and test_results['tokenizer_load']:
                try:
                    test_query = "Привет, как дела?"
                    
                    # Пробуем разные методы
                    for method in available_methods:
                        try:
                            method_func = getattr(manager, method)
                            response = method_func(test_query, max_length=50)
                            
                            if response and not response.startswith("Ошибка"):
                                test_results['actual_generation'] = True
                                logger.info(f"   ✅ {method}: '{response[:50]}...'")
                            else:
                                test_results['errors'].append(f"{method} вернул ошибку: {response}")
                                logger.error(f"   ❌ {method}: {response}")
                                
                        except Exception as e:
                            test_results['errors'].append(f"{method} исключение: {e}")
                            logger.error(f"   ❌ {method}: {e}")
                
                except Exception as e:
                    test_results['errors'].append(f"Ошибка тестирования генерации: {e}")
                    logger.error(f"   ❌ Ошибка тестирования: {e}")
            
        except Exception as e:
            test_results['errors'].append(f"Критическая ошибка: {e}")
            logger.error(f"   ❌ Критическая ошибка: {e}")
        
        return test_results
    
    def analyze_file_structure(self) -> Dict[str, Any]:
        """Анализирует структуру файлов системы"""
        logger.info("📁 Анализ структуры файлов")
        
        structure = {
            'core_files': {},
            'config_files': {},
            'cache_files': {},
            'model_files': {}
        }
        
        # Анализируем ключевые файлы
        key_files = {
            'optimized_fractal_model_manager': 'cogniflex/mlearning/optimized_fractal_model_manager.py',
            'text_quality_trainer': 'cogniflex/mlearning/text_quality_trainer.py',
            'text_quality_improver': 'cogniflex/mlearning/text_quality_improver.py',
            'hybrid_token_cache': 'cogniflex/memory/hybrid_token_cache.py',
            'fractal_store': 'cogniflex/mlearning/storage/fractal_store.py',
            'fractal_model_loader': 'cogniflex/mlearning/storage/fractal_model_loader.py'
        }
        
        for name, path in key_files.items():
            file_path = Path(path)
            if file_path.exists():
                structure['core_files'][name] = {
                    'path': str(file_path),
                    'size': file_path.stat().st_size,
                    'modified': file_path.stat().st_mtime,
                    'lines': 0
                }
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        structure['core_files'][name]['lines'] = len(f.readlines())
                except Exception as e:
                    logger.warning(f"Ошибка подсчета строк в {name}: {e}")
            else:
                structure['core_files'][name] = {'error': 'Файл не найден'}
        
        return structure
    
    def create_analysis_report(self) -> str:
        """Создает полный отчет анализа"""
        logger.info("📋 Создание отчета анализа")
        
        report = []
        report.append("# 🔍 АНАЛИЗ СИСТЕМЫ ГЕНЕРАЦИИ ТЕКСТА COGNIFLEX")
        report.append("=" * 80)
        report.append("")
        
        # 1. Анализ файловой структуры
        file_structure = self.analyze_file_structure()
        report.append("## 📁 СТРУКТУРА ФАЙЛОВ")
        report.append("")
        
        for name, info in file_structure['core_files'].items():
            if 'error' in info:
                report.append(f"- **❌ {name}**: {info['error']}")
            else:
                report.append(f"- **✅ {name}**:")
                report.append(f"  - 📁 Путь: `{info['path']}`")
                report.append(f"  - 📊 Размер: {info['size']:,} байт")
                report.append(f"  - 📝 Строк: {info['lines']:,}")
                report.append(f"  - 🕐 Изменен: {info['modified']}")
        
        # 2. Анализ потока генерации
        flow_analysis = self.analyze_generation_flow()
        report.append("")
        report.append("## 🌊 ПОТОК ГЕНЕРАЦИИ")
        report.append("")
        
        report.append("### 🎯 ТОЧКИ ВХОДА:")
        for point in flow_analysis['entry_points']:
            report.append(f"- ✅ {point}")
        
        report.append("")
        report.append("### 📦 ПОТОК ДАННЫХ:")
        for item in flow_analysis['data_flow']:
            report.append(f"- {item}")
        
        if flow_analysis['bottlenecks']:
            report.append("")
            report.append("### 🚫 УЗКИЕ МЕСТА:")
            for bottleneck in flow_analysis['bottlenecks']:
                report.append(f"- ⚠️ {bottleneck}")
        
        if flow_analysis['error_points']:
            report.append("")
            report.append("### ❌ ОШИБКИ ЗАВИСИМОСТЕЙ:")
            for error in flow_analysis['error_points']:
                report.append(f"- {error}")
        
        # 3. Тестирование пайплайна
        test_results = self.test_generation_pipeline()
        report.append("")
        report.append("## 🧪 ТЕСТИРОВАНИЕ ПАЙПЛАЙНА")
        report.append("")
        
        report.append("### 📊 РЕЗУЛЬТАТЫ ТЕСТОВ:")
        test_items = [
            ('Инициализация менеджера', test_results['manager_init']),
            ('Загрузка модели', test_results['model_load']),
            ('Загрузка токенизатора', test_results['tokenizer_load']),
            ('Доступность методов генерации', test_results['generation_methods']),
            ('Фактическая генерация', test_results['actual_generation'])
        ]
        
        for name, result in test_items:
            status = "✅ УСПЕХ" if result else "❌ ОШИБКА"
            report.append(f"- **{name}**: {status}")
        
        if test_results['errors']:
            report.append("")
            report.append("### 🚨 ОБНАРУЖЕННЫЕ ОШИБКИ:")
            for i, error in enumerate(test_results['errors'], 1):
                report.append(f"{i}. {error}")
        
        # 4. Анализ проблем
        report.append("")
        report.append("## 🔍 АНАЛИЗ ПРОБЛЕМ")
        report.append("")
        
        if not test_results['actual_generation']:
            report.append("### 🎯 ОСНОВНАЯ ПРОБЛЕМА:")
            report.append("Система **ВИДИТ** что работает, но **НЕ РАБОТАЕТ** на самом деле.")
            report.append("")
            report.append("#### 📋 Возможные причины:")
            report.append("1. **Несоответствие устройств** - модель и тензоры на разных устройствах")
            report.append("2. **Некорректная токенизация** - тензоры не правильно обрабатываются")
            report.append("3. **Отсутствие критических компонентов** - модули не загружаются")
            report.append("4. **Ошибки в методах генерации** - исключения при вызове")
            report.append("5. **Проблемы с конфигурацией** - неверные параметры")
        
        # 5. Рекомендации
        report.append("")
        report.append("## 💡 РЕКОМЕНДАЦИИ")
        report.append("")
        
        report.append("### 🔧 НЕМЕДЛЕННЫЕ ДЕЙСТВИЯ:")
        report.append("1. **Добавить детальное логирование** во все методы генерации")
        report.append("2. **Проверить совместимость устройств** на каждом шаге")
        report.append("3. **Создать fallback механизмы** для критических ошибок")
        report.append("4. **Улучшить обработку исключений** с конкретными сообщениями")
        report.append("5. **Добавить валидацию параметров** перед использованием")
        
        report.append("")
        report.append("### 📝 ДАЛЬНЕЙШИЕ ШАГИ:")
        report.append("1. Создать **единый интерфейс генерации** с унифицированной обработкой ошибок")
        report.append("2. Реализовать **пошаговую диагностику** с детальными логами")
        report.append("3. Добавить **тесты совместимости** для разных конфигураций")
        report.append("4. Создать **мониторинг производительности** в реальном времени")
        report.append("5. Реализовать **автоматическое восстановление** при ошибках")
        
        report.append("")
        report.append("---")
        report.append("*Отчет сгенерирован автоматически: " + str(datetime.now()))
        
        return '\n'.join(report)

def main():
    """Основная функция анализа"""
    logger.info("🚀 ЗАПУСК КОМПЛЕКСНОГО АНАЛИЗА СИСТЕМЫ")
    logger.info("=" * 80)
    
    try:
        # Импортируем необходимые модули
        import importlib.util
        import importlib
        from datetime import datetime
        
        # Создаем анализатор
        analyzer = SystemAnalyzer()
        
        # Создаем отчет
        report = analyzer.create_analysis_report()
        
        # Сохраняем отчет
        report_file = Path("system_generation_analysis.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"✅ Отчет сохранен: {report_file}")
        
        # Выводим краткие результаты
        logger.info("\n📊 КРАТКИЕ РЕЗУЛЬТАТЫ:")
        logger.info("📋 Анализ завершен")
        logger.info("📁 Отчет сохранен")
        logger.info("🔍 Проверьте system_generation_analysis.md")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка анализа: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
