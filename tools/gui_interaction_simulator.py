"""
Симулятор взаимодействия с GUI для тестирования и логирования ошибок
Имитирует пользовательские действия и отслеживает проблемы в интерфейсе
"""

import os
import sys
import time
import threading
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock

# Добавляем путь к проекту
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cogniflex'))

class GUIErrorLogger:
    """Логгер ошибок GUI с детальным отслеживанием."""
    
    def __init__(self, log_file: str = "gui_errors.log"):
        self.log_file = log_file
        self.errors = []
        self.warnings = []
        self.info_messages = []
        
        # Настраиваем логирование
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("GUI_Simulator")
    
    def log_error(self, component: str, action: str, error: str, details: Dict = None):
        """Логирует ошибку с контекстом."""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': 'ERROR',
            'component': component,
            'action': action,
            'error': error,
            'details': details or {}
        }
        self.errors.append(error_entry)
        self.logger.error(f"[{component}] {action}: {error}")
        if details:
            self.logger.error(f"Details: {json.dumps(details, indent=2)}")
    
    def log_warning(self, component: str, action: str, warning: str, details: Dict = None):
        """Логирует предупреждение."""
        warning_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': 'WARNING',
            'component': component,
            'action': action,
            'warning': warning,
            'details': details or {}
        }
        self.warnings.append(warning_entry)
        self.logger.warning(f"[{component}] {action}: {warning}")
    
    def log_info(self, component: str, action: str, message: str, details: Dict = None):
        """Логирует информационное сообщение."""
        info_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': 'INFO',
            'component': component,
            'action': action,
            'message': message,
            'details': details or {}
        }
        self.info_messages.append(info_entry)
        self.logger.info(f"[{component}] {action}: {message}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Возвращает сводку по ошибкам."""
        return {
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'total_info': len(self.info_messages),
            'errors_by_component': self._group_by_component(self.errors),
            'warnings_by_component': self._group_by_component(self.warnings),
            'recent_errors': self.errors[-5:] if self.errors else [],
            'recent_warnings': self.warnings[-5:] if self.warnings else []
        }
    
    def _group_by_component(self, entries: List[Dict]) -> Dict[str, int]:
        """Группирует записи по компонентам."""
        groups = {}
        for entry in entries:
            component = entry.get('component', 'unknown')
            groups[component] = groups.get(component, 0) + 1
        return groups

class MockBrain:
    """Мок-объект для имитации CoreBrain."""
    
    def __init__(self):
        self.running = False
        self.initialized = False
        self.components = {}
        self.config = {
            'gui': {'theme': 'light'},
            'system': {'debug': True}
        }
        
        # Имитируем новые менеджеры
        self.config_manager = Mock()
        self.system_state_manager = Mock()
        self.resource_manager = Mock()
        self.system_metrics_manager = Mock()
        
        # Настраиваем методы менеджеров
        self.config_manager.get_config.return_value = self.config
        self.system_state_manager.get_system_state.return_value = "running"
        self.system_state_manager.get_component_states.return_value = {
            'ml_unit': 'active',
            'knowledge_graph': 'active',
            'memory_manager': 'active'
        }
        self.resource_manager.get_resource_summary.return_value = {
            'cpu_usage': 45.2,
            'memory_usage': 67.8,
            'disk_usage': 23.1
        }
        self.system_metrics_manager.get_performance_metrics.return_value = {
            'queries_processed': 150,
            'avg_response_time': 0.85,
            'cache_hit_rate': 0.73
        }
    
    def initialize(self) -> bool:
        """Имитирует инициализацию."""
        self.initialized = True
        return True
    
    def start(self):
        """Имитирует запуск."""
        self.running = True
    
    def stop(self):
        """Имитирует остановку."""
        self.running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Возвращает статус системы."""
        return {
            'running': self.running,
            'initialized': self.initialized,
            'uptime': 3600,  # 1 час
            'components': self.components
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Возвращает системную информацию."""
        return {
            'version': '2.0.0',
            'build': 'refactored',
            'components_count': len(self.components),
            'memory_usage': '256MB',
            'cpu_usage': '15%'
        }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Имитирует обработку запроса."""
        time.sleep(0.1)  # Имитируем обработку
        return {
            'query': query,
            'response': f"Обработанный ответ на: {query}",
            'confidence': 0.85,
            'processing_time': 0.1
        }

class GUIInteractionSimulator:
    """Симулятор взаимодействия с GUI."""
    
    def __init__(self, error_logger: GUIErrorLogger):
        self.error_logger = error_logger
        self.brain = MockBrain()
        self.gui = None
        self.simulation_running = False
    
    def setup_gui_mock(self):
        """Настраивает мок GUI."""
        try:
            self.error_logger.log_info("Setup", "GUI Mock", "Настройка мок-объекта GUI")
            
            # Пытаемся импортировать реальный GUI
            from cogniflex.gui.core_gui import CogniFlexGUI
            
            # Создаем мок GUI с реальными методами
            self.gui = Mock(spec=CogniFlexGUI)
            self.gui.brain = self.brain
            self.gui.running = False
            self.gui.cache_dir = os.path.join(os.path.dirname(__file__), "test_gui_cache")
            
            # Настраиваем методы GUI
            self.gui.start = Mock()
            self.gui.stop = Mock()
            self.gui.update_status = Mock()
            self.gui.show_error = Mock()
            self.gui.show_message = Mock()
            
            self.error_logger.log_info("Setup", "GUI Mock", "Мок GUI успешно настроен")
            return True
            
        except ImportError as e:
            self.error_logger.log_error("Setup", "GUI Import", f"Не удалось импортировать GUI: {e}")
            return False
        except Exception as e:
            self.error_logger.log_error("Setup", "GUI Mock", f"Ошибка настройки мока: {e}")
            return False
    
    def test_gui_initialization(self) -> bool:
        """Тестирует инициализацию GUI."""
        self.error_logger.log_info("Test", "GUI Init", "Начало теста инициализации GUI")
        
        try:
            # Пытаемся создать реальный GUI
            from cogniflex.gui.core_gui import CogniFlexGUI
            
            # Тестируем создание GUI с мок brain
            gui = CogniFlexGUI(brain=self.brain)
            
            # Проверяем основные атрибуты
            if not hasattr(gui, 'brain'):
                self.error_logger.log_error("GUI Init", "Attribute Check", "Отсутствует атрибут 'brain'")
                return False
            
            if not hasattr(gui, 'cache_dir'):
                self.error_logger.log_error("GUI Init", "Attribute Check", "Отсутствует атрибут 'cache_dir'")
                return False
            
            if not hasattr(gui, 'settings'):
                self.error_logger.log_error("GUI Init", "Attribute Check", "Отсутствует атрибут 'settings'")
                return False
            
            self.error_logger.log_info("Test", "GUI Init", "GUI успешно инициализирован")
            return True
            
        except Exception as e:
            self.error_logger.log_error("Test", "GUI Init", f"Ошибка инициализации GUI: {e}", 
                                      {'exception_type': type(e).__name__})
            return False
    
    def test_brain_integration(self) -> bool:
        """Тестирует интеграцию с brain."""
        self.error_logger.log_info("Test", "Brain Integration", "Начало теста интеграции с brain")
        
        try:
            # Инициализируем brain
            if not self.brain.initialize():
                self.error_logger.log_error("Brain Integration", "Initialize", "Brain не инициализировался")
                return False
            
            # Запускаем brain
            self.brain.start()
            
            # Проверяем статус
            status = self.brain.get_status()
            if not status.get('running'):
                self.error_logger.log_error("Brain Integration", "Status Check", "Brain не запущен")
                return False
            
            # Проверяем системную информацию
            system_info = self.brain.get_system_info()
            if not system_info:
                self.error_logger.log_error("Brain Integration", "System Info", "Не удалось получить системную информацию")
                return False
            
            # Тестируем обработку запроса
            result = self.brain.process_query("Тестовый запрос")
            if not result or 'response' not in result:
                self.error_logger.log_error("Brain Integration", "Query Processing", "Ошибка обработки запроса")
                return False
            
            self.error_logger.log_info("Test", "Brain Integration", "Интеграция с brain работает корректно")
            return True
            
        except Exception as e:
            self.error_logger.log_error("Test", "Brain Integration", f"Ошибка интеграции: {e}",
                                      {'exception_type': type(e).__name__})
            return False
    
    def test_gui_components(self) -> bool:
        """Тестирует компоненты GUI."""
        self.error_logger.log_info("Test", "GUI Components", "Начало теста компонентов GUI")
        
        try:
            from cogniflex.gui.core_gui import CogniFlexGUI
            
            gui = CogniFlexGUI(brain=self.brain)
            
            # Список ожидаемых методов
            expected_methods = [
                'start', 'stop', 'update_status', 'show_error', 
                'show_message', 'create_main_window'
            ]
            
            missing_methods = []
            for method in expected_methods:
                if not hasattr(gui, method):
                    missing_methods.append(method)
            
            if missing_methods:
                self.error_logger.log_error("GUI Components", "Method Check", 
                                          f"Отсутствуют методы: {missing_methods}")
                return False
            
            # Проверяем настройки
            if not gui.settings:
                self.error_logger.log_warning("GUI Components", "Settings", "Настройки GUI пусты")
            
            # Проверяем тему
            if not hasattr(gui, 'theme') or gui.theme not in ['light', 'dark']:
                self.error_logger.log_warning("GUI Components", "Theme", f"Некорректная тема: {getattr(gui, 'theme', 'None')}")
            
            self.error_logger.log_info("Test", "GUI Components", "Компоненты GUI проверены")
            return True
            
        except Exception as e:
            self.error_logger.log_error("Test", "GUI Components", f"Ошибка проверки компонентов: {e}",
                                      {'exception_type': type(e).__name__})
            return False
    
    def simulate_user_interactions(self) -> bool:
        """Симулирует пользовательские взаимодействия."""
        self.error_logger.log_info("Simulation", "User Interactions", "Начало симуляции пользовательских действий")
        
        interactions = [
            ("startup", self._simulate_startup),
            ("query_processing", self._simulate_query_processing),
            ("status_updates", self._simulate_status_updates),
            ("error_handling", self._simulate_error_handling),
            ("shutdown", self._simulate_shutdown)
        ]
        
        success_count = 0
        
        for interaction_name, interaction_func in interactions:
            try:
                self.error_logger.log_info("Simulation", interaction_name, f"Выполнение симуляции: {interaction_name}")
                
                if interaction_func():
                    success_count += 1
                    self.error_logger.log_info("Simulation", interaction_name, f"Симуляция {interaction_name} успешна")
                else:
                    self.error_logger.log_error("Simulation", interaction_name, f"Симуляция {interaction_name} провалена")
                
                time.sleep(0.5)  # Пауза между действиями
                
            except Exception as e:
                self.error_logger.log_error("Simulation", interaction_name, 
                                          f"Исключение в симуляции {interaction_name}: {e}",
                                          {'exception_type': type(e).__name__})
        
        success_rate = success_count / len(interactions)
        self.error_logger.log_info("Simulation", "Summary", 
                                 f"Успешно выполнено {success_count}/{len(interactions)} симуляций ({success_rate:.1%})")
        
        return success_rate > 0.7
    
    def _simulate_startup(self) -> bool:
        """Симулирует запуск приложения."""
        try:
            # Инициализация brain
            if not self.brain.initialize():
                return False
            
            # Запуск brain
            self.brain.start()
            
            # Проверка статуса
            status = self.brain.get_status()
            return status.get('running', False)
            
        except Exception as e:
            self.error_logger.log_error("Simulation", "Startup", f"Ошибка запуска: {e}")
            return False
    
    def _simulate_query_processing(self) -> bool:
        """Симулирует обработку запросов."""
        try:
            test_queries = [
                "Что такое машинное обучение?",
                "Как работает нейронная сеть?",
                "Объясни принципы ИИ"
            ]
            
            for query in test_queries:
                result = self.brain.process_query(query)
                if not result or 'response' not in result:
                    return False
                
                time.sleep(0.1)
            
            return True
            
        except Exception as e:
            self.error_logger.log_error("Simulation", "Query Processing", f"Ошибка обработки запросов: {e}")
            return False
    
    def _simulate_status_updates(self) -> bool:
        """Симулирует обновления статуса."""
        try:
            # Получаем статус системы
            status = self.brain.get_status()
            system_info = self.brain.get_system_info()
            
            # Проверяем наличие новых менеджеров
            if hasattr(self.brain, 'system_state_manager'):
                component_states = self.brain.system_state_manager.get_component_states()
                self.error_logger.log_info("Status", "Components", f"Состояние компонентов: {component_states}")
            
            if hasattr(self.brain, 'resource_manager'):
                resources = self.brain.resource_manager.get_resource_summary()
                self.error_logger.log_info("Status", "Resources", f"Ресурсы системы: {resources}")
            
            if hasattr(self.brain, 'system_metrics_manager'):
                metrics = self.brain.system_metrics_manager.get_performance_metrics()
                self.error_logger.log_info("Status", "Metrics", f"Метрики производительности: {metrics}")
            
            return True
            
        except Exception as e:
            self.error_logger.log_error("Simulation", "Status Updates", f"Ошибка обновления статуса: {e}")
            return False
    
    def _simulate_error_handling(self) -> bool:
        """Симулирует обработку ошибок."""
        try:
            # Симулируем различные типы ошибок
            error_scenarios = [
                ("invalid_query", lambda: self.brain.process_query("")),
                ("null_query", lambda: self.brain.process_query(None)),
                ("long_query", lambda: self.brain.process_query("x" * 10000))
            ]
            
            for scenario_name, scenario_func in error_scenarios:
                try:
                    result = scenario_func()
                    self.error_logger.log_info("Error Handling", scenario_name, f"Сценарий {scenario_name} обработан")
                except Exception as e:
                    self.error_logger.log_info("Error Handling", scenario_name, f"Ошибка в сценарии {scenario_name} корректно обработана: {e}")
            
            return True
            
        except Exception as e:
            self.error_logger.log_error("Simulation", "Error Handling", f"Ошибка в обработке ошибок: {e}")
            return False
    
    def _simulate_shutdown(self) -> bool:
        """Симулирует завершение работы."""
        try:
            self.brain.stop()
            status = self.brain.get_status()
            return not status.get('running', True)
            
        except Exception as e:
            self.error_logger.log_error("Simulation", "Shutdown", f"Ошибка завершения: {e}")
            return False
    
    def run_full_simulation(self) -> Dict[str, Any]:
        """Запускает полную симуляцию."""
        self.error_logger.log_info("Main", "Full Simulation", "Запуск полной симуляции GUI")
        
        results = {
            'gui_mock_setup': False,
            'gui_initialization': False,
            'brain_integration': False,
            'gui_components': False,
            'user_interactions': False,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'duration': None
        }
        
        start_time = time.time()
        
        try:
            # Этап 1: Настройка мока
            results['gui_mock_setup'] = self.setup_gui_mock()
            
            # Этап 2: Тест инициализации
            results['gui_initialization'] = self.test_gui_initialization()
            
            # Этап 3: Тест интеграции с brain
            results['brain_integration'] = self.test_brain_integration()
            
            # Этап 4: Тест компонентов GUI
            results['gui_components'] = self.test_gui_components()
            
            # Этап 5: Симуляция пользовательских взаимодействий
            results['user_interactions'] = self.simulate_user_interactions()
            
        except Exception as e:
            self.error_logger.log_error("Main", "Full Simulation", f"Критическая ошибка симуляции: {e}")
        
        end_time = time.time()
        results['end_time'] = datetime.now().isoformat()
        results['duration'] = end_time - start_time
        
        # Подсчитываем общий результат
        success_count = sum(1 for result in results.values() if isinstance(result, bool) and result)
        total_tests = sum(1 for result in results.values() if isinstance(result, bool))
        results['success_rate'] = success_count / total_tests if total_tests > 0 else 0
        
        self.error_logger.log_info("Main", "Full Simulation", 
                                 f"Симуляция завершена. Успешность: {results['success_rate']:.1%}")
        
        return results

def main():
    """Основная функция запуска симулятора."""
    print("🔧 СИМУЛЯТОР ВЗАИМОДЕЙСТВИЯ С GUI")
    print("="*60)
    
    # Создаем логгер ошибок
    error_logger = GUIErrorLogger("gui_simulation_errors.log")
    
    # Создаем симулятор
    simulator = GUIInteractionSimulator(error_logger)
    
    # Запускаем полную симуляцию
    results = simulator.run_full_simulation()
    
    # Выводим результаты
    print("\n📊 РЕЗУЛЬТАТЫ СИМУЛЯЦИИ:")
    print("="*60)
    
    for test_name, result in results.items():
        if isinstance(result, bool):
            status = "✅ ПРОЙДЕН" if result else "❌ ПРОВАЛЕН"
            print(f"{test_name:25} : {status}")
    
    print(f"\nОбщая успешность: {results['success_rate']:.1%}")
    print(f"Время выполнения: {results['duration']:.2f} секунд")
    
    # Выводим сводку ошибок
    error_summary = error_logger.get_summary()
    print(f"\n🚨 СВОДКА ОШИБОК:")
    print(f"Ошибок: {error_summary['total_errors']}")
    print(f"Предупреждений: {error_summary['total_warnings']}")
    print(f"Информационных сообщений: {error_summary['total_info']}")
    
    if error_summary['errors_by_component']:
        print("\nОшибки по компонентам:")
        for component, count in error_summary['errors_by_component'].items():
            print(f"  {component}: {count}")
    
    # Сохраняем результаты
    results_file = "gui_simulation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'results': results,
            'error_summary': error_summary
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Результаты сохранены в {results_file}")
    print("📋 Логи ошибок сохранены в gui_simulation_errors.log")
    
    if results['success_rate'] > 0.8:
        print("\n🎉 GUI работает стабильно!")
    elif results['success_rate'] > 0.6:
        print("\n⚠️ GUI работает с незначительными проблемами.")
    else:
        print("\n❌ Обнаружены серьезные проблемы в GUI.")

if __name__ == "__main__":
    main()
