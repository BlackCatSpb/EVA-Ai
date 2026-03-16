#!/usr/bin/env python3
"""
Скрипт эмуляции GUI команд CogniFlex с логированием ошибок
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path

# Настройка логирования
log_file = f"gui_test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GUIEmulator:
    """Эмулятор GUI для тестирования команд"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.commands_tested = []
        self.brain = None
        self.integrator = None
        
    def log_error(self, command, error, context=""):
        """Логирование ошибки"""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'command': command,
            'error': str(error),
            'context': context
        }
        self.errors.append(error_entry)
        logger.error(f"[ERROR] Command: {command} | Error: {error} | Context: {context}")
        
    def log_warning(self, command, warning, context=""):
        """Логирование предупреждения"""
        warning_entry = {
            'timestamp': datetime.now().isoformat(),
            'command': command,
            'warning': str(warning),
            'context': context
        }
        self.warnings.append(warning_entry)
        logger.warning(f"[WARNING] Command: {command} | Warning: {warning} | Context: {context}")
        
    def test_command(self, command_name, command_func, *args, **kwargs):
        """Тестирование одной команды"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing command: {command_name}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        try:
            result = command_func(*args, **kwargs)
            elapsed = time.time() - start_time
            self.commands_tested.append({
                'command': command_name,
                'status': 'success',
                'elapsed': elapsed,
                'result': str(result)[:200] if result else None
            })
            logger.info(f"✓ Command '{command_name}' completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            self.log_error(command_name, e, f"Args: {args}, Kwargs: {kwargs}")
            self.commands_tested.append({
                'command': command_name,
                'status': 'error',
                'elapsed': elapsed,
                'error': str(e)
            })
            logger.error(f"✗ Command '{command_name}' failed after {elapsed:.3f}s")
            return None
            
    def initialize_system(self):
        """Инициализация системы"""
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: System Initialization")
        logger.info("="*60)
        
        # Тест 1: Импорт CoreBrain
        def test_import_corebrain():
            from cogniflex.core.core_brain import CoreBrain
            return CoreBrain
            
        CoreBrain = self.test_command("import_corebrain", test_import_corebrain)
        if not CoreBrain:
            return False
            
        # Тест 2: Создание экземпляра CoreBrain
        def test_create_brain():
            brain = CoreBrain()
            self.brain = brain
            return brain
            
        brain = self.test_command("create_brain", test_create_brain)
        if not brain:
            return False
            
        # Тест 3: Инициализация
        def test_initialize():
            return brain.initialize()
            
        result = self.test_command("initialize_brain", test_initialize)
        if not result:
            return False
            
        # Тест 4: Импорт интегратора
        def test_import_integrator():
            from cogniflex.core.integration_layer import CogniFlexIntegrator
            return CogniFlexIntegrator
            
        Integrator = self.test_command("import_integrator", test_import_integrator)
        if not Integrator:
            return False
            
        # Тест 5: Создание интегратора
        def test_create_integrator():
            integrator = Integrator(brain=brain)
            self.integrator = integrator
            return integrator
            
        integrator = self.test_command("create_integrator", test_create_integrator)
        if not integrator:
            return False
            
        return True
        
    def test_generation_commands(self):
        """Тестирование команд генерации"""
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: Generation Commands")
        logger.info("="*60)
        
        if not self.brain:
            self.log_error("test_generation", "Brain not initialized")
            return
            
        test_queries = [
            "Привет",
            "Как дела?",
            "Расскажи о себе",
            "Тест генерации текста",
            "Что такое искусственный интеллект?"
        ]
        
        for query in test_queries:
            def test_query(q=query):
                if hasattr(self.brain, 'process_query'):
                    return self.brain.process_query(q)
                elif self.integrator and hasattr(self.integrator, 'process_query'):
                    return self.integrator.process_query(q)
                else:
                    raise AttributeError("No process_query method found")
                    
            result = self.test_command(f"generate_{query[:20]}", test_query)
            
            # Анализ результата
            if result:
                response = result.get('response', result) if isinstance(result, dict) else result
                logger.info(f"Response preview: {str(response)[:100]}...")
                
                # Проверка на бессвязность
                if self._is_gibberish(str(response)):
                    self.log_warning(
                        f"generate_{query[:20]}",
                        "Response appears to be gibberish",
                        f"Response: {response[:200]}"
                    )
                    
    def _is_gibberish(self, text: str) -> bool:
        """Проверка на бессвязность текста"""
        import re
        
        # Признаки бессвязности:
        # 1. Много слитых слов без пробелов
        # 2. Много случайных символов
        # 3. Отсутствие осмысленных предложений
        
        # Считаем слова
        words = re.findall(r'\b\w+\b', text)
        if len(words) == 0:
            return True
            
        # Проверяем среднюю длину слова
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len > 15:  # Слишком длинные слова - подозрительно
            return True
            
        # Проверяем количество не-alphanumeric символов
        non_alpha = len(re.findall(r'[^\w\s]', text))
        if non_alpha / len(text) > 0.3:  # Слишком много спецсимволов
            return True
            
        # Проверяем на сплошные строки букв без пробелов
        long_words = len([w for w in words if len(w) > 20])
        if long_words > 2:
            return True
            
        return False
        
    def test_gui_commands(self):
        """Тестирование GUI команд"""
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: GUI Commands")
        logger.info("="*60)
        
        gui_commands = [
            ("switch_view_chat", lambda: self._gui_switch_view("chat")),
            ("switch_view_memory", lambda: self._gui_switch_view("memory")),
            ("switch_view_settings", lambda: self._gui_switch_view("settings")),
            ("get_system_status", lambda: self._gui_get_status()),
            ("get_health_report", lambda: self._gui_health_report()),
        ]
        
        for cmd_name, cmd_func in gui_commands:
            self.test_command(cmd_name, cmd_func)
            
    def _gui_switch_view(self, view_name):
        """Эмуляция переключения вида"""
        if self.brain and hasattr(self.brain, 'gui'):
            gui = self.brain.gui
            if hasattr(gui, '_switch_view'):
                return gui._switch_view(view_name)
        return None
        
    def _gui_get_status(self):
        """Получение статуса системы"""
        if self.brain:
            if hasattr(self.brain, 'get_system_health'):
                return self.brain.get_system_health()
            elif hasattr(self.brain, 'get_status'):
                return self.brain.get_status()
        return None
        
    def _gui_health_report(self):
        """Получение отчета о здоровье"""
        if self.brain and hasattr(self.brain, 'get_system_health'):
            return self.brain.get_system_health()
        return None
        
    def test_ml_components(self):
        """Тестирование ML компонентов"""
        logger.info("\n" + "="*60)
        logger.info("PHASE 4: ML Components")
        logger.info("="*60)
        
        ml_commands = [
            ("check_model_manager", lambda: self._check_component('model_manager')),
            ("check_tokenizer", lambda: self._check_component('tokenizer')),
            ("check_fractal_manager", lambda: self._check_component('fractal_model_manager')),
        ]
        
        for cmd_name, cmd_func in ml_commands:
            self.test_command(cmd_name, cmd_func)
            
    def _check_component(self, component_name):
        """Проверка наличия компонента"""
        if not self.brain:
            return None
            
        if hasattr(self.brain, 'components') and component_name in self.brain.components:
            comp = self.brain.components[component_name]
            return {
                'exists': True,
                'initialized': getattr(comp, 'is_initialized', False),
                'type': type(comp).__name__
            }
        elif hasattr(self.brain, component_name):
            comp = getattr(self.brain, component_name)
            return {
                'exists': True,
                'initialized': getattr(comp, 'initialized', getattr(comp, 'is_initialized', False)),
                'type': type(comp).__name__
            }
        return {'exists': False}
        
    def run_full_test(self):
        """Запуск полного тестирования"""
        logger.info("\n" + "="*80)
        logger.info("COGNIFLEX GUI EMULATOR - FULL SYSTEM TEST")
        logger.info(f"Started at: {datetime.now().isoformat()}")
        logger.info("="*80)
        
        # Phase 1: Initialization
        if not self.initialize_system():
            logger.error("CRITICAL: System initialization failed!")
            self.generate_report()
            return False
            
        # Phase 2: Generation
        self.test_generation_commands()
        
        # Phase 3: GUI
        self.test_gui_commands()
        
        # Phase 4: ML Components
        self.test_ml_components()
        
        # Generate report
        self.generate_report()
        
        return True
        
    def generate_report(self):
        """Генерация отчета"""
        report_file = f"gui_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_commands': len(self.commands_tested),
                'successful': len([c for c in self.commands_tested if c['status'] == 'success']),
                'failed': len([c for c in self.commands_tested if c['status'] == 'error']),
                'errors_count': len(self.errors),
                'warnings_count': len(self.warnings)
            },
            'commands': self.commands_tested,
            'errors': self.errors,
            'warnings': self.warnings
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        logger.info("\n" + "="*60)
        logger.info("TEST REPORT")
        logger.info("="*60)
        logger.info(f"Total commands tested: {report['summary']['total_commands']}")
        logger.info(f"Successful: {report['summary']['successful']}")
        logger.info(f"Failed: {report['summary']['failed']}")
        logger.info(f"Errors: {report['summary']['errors_count']}")
        logger.info(f"Warnings: {report['summary']['warnings_count']}")
        logger.info(f"\nDetailed report saved to: {report_file}")
        logger.info(f"Log file: {log_file}")
        
        return report

def main():
    """Main entry point"""
    emulator = GUIEmulator()
    
    try:
        emulator.run_full_test()
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        emulator.generate_report()
    except Exception as e:
        logger.critical(f"Critical error during testing: {e}", exc_info=True)
        emulator.generate_report()
        
if __name__ == "__main__":
    main()
