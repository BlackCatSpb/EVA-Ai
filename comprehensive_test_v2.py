#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive System Test for CogniFlex v2
Исправленная версия с улучшенной обработкой ошибок
"""

import os
import sys
import time
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional

# Настройка логирования с поддержкой UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('comprehensive_test_v2.log', encoding='utf-8')
    ]
)
# Принудительная установка UTF-8 для stdout
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

logger = logging.getLogger("cogniflex.comprehensive_test_v2")


class ComprehensiveSystemTestV2:
    """Комплексное тестирование системы CogniFlex - исправленная версия"""
    
    def __init__(self):
        self.test_results = {
            'initialization': {},
            'generation': {},
            'self_learning': {},
            'reasoning': {},
            'memory_graph': {},
            'integration': {},
            'summary': {}
        }
        self.metrics = {
            'start_time': time.time(),
            'components_initialized': [],
            'errors': [],
            'warnings': []
        }
        self.brain = None
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Запуск всех тестов системы"""
        logger.info("=" * 80)
        logger.info("НАЧАЛО КОМПЛЕКСНОГО ТЕСТИРОВАНИЯ COGNIFLEX v2")
        logger.info("=" * 80)
        logger.info("Время начала: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        test_methods = [
            ('Инициализация', self._test_initialization),
            ('Генерация текста', self._test_text_generation),
            ('Самообучение', self._test_self_learning),
            ('Рассуждение', self._test_reasoning),
            ('Граф памяти', self._test_memory_graph),
            ('Интеграция', self._test_integration)
        ]
        
        for test_name, test_method in test_methods:
            try:
                logger.info("\n[НАЧАЛО ТЕСТА] %s", test_name)
                test_method()
                logger.info("[ЗАВЕРШЕНО] %s", test_name)
            except Exception as e:
                logger.error("[ОШИБКА] %s: %s", test_name, str(e))
                logger.error(traceback.format_exc())
                self.metrics['errors'].append(f"{test_name}: {str(e)}")
        
        # Генерация отчёта
        self._generate_report()
        return self.test_results
    
    def _test_initialization(self):
        """Тест инициализации всех компонентов"""
        logger.info("Тестирование инициализации компонентов...")
        
        init_results = {
            'core_brain_import': False,
            'core_brain_created': False,
            'core_brain_initialized': False,
            'enhanced_learning': False,
            'memory_graph_ml': False,
            'reasoning_engine': False,
            'fractal_model': False,
            'knowledge_graph': False,
            'errors': []
        }
        
        try:
            # 1. Импорт CoreBrain
            logger.info("  [1/8] Импорт CoreBrain...")
            from cogniflex.core.core_brain import CoreBrain
            init_results['core_brain_import'] = True
            logger.info("  OK: CoreBrain импортирован")
            
            # 2. Создание экземпляра
            logger.info("  [2/8] Создание экземпляра...")
            config = {
                'learning': {'default_epochs': 2, 'batch_size': 2},
                'memory_graph_ml': {'embedding_dim': 64, 'fractal_levels': 3}
            }
            self.brain = CoreBrain(config)
            init_results['core_brain_created'] = True
            logger.info("  OK: CoreBrain создан")
            
            # 3. Инициализация
            logger.info("  [3/8] Инициализация CoreBrain...")
            init_result = self.brain.initialize()
            init_results['core_brain_initialized'] = init_result
            logger.info("  OK: CoreBrain.initialize() = %s", init_result)
            
            # 4. EnhancedLearning
            logger.info("  [4/8] Проверка EnhancedSelfLearningSystem...")
            if hasattr(self.brain, 'enhanced_learning') and self.brain.enhanced_learning:
                init_results['enhanced_learning'] = True
                self.metrics['components_initialized'].append('enhanced_learning')
                logger.info("  OK: EnhancedSelfLearningSystem доступен")
            else:
                logger.warning("  WARN: EnhancedSelfLearningSystem недоступен")
            
            # 5. MemoryGraphML
            logger.info("  [5/8] Проверка MemoryGraphML...")
            if hasattr(self.brain, 'memory_graph_ml') and self.brain.memory_graph_ml:
                init_results['memory_graph_ml'] = True
                self.metrics['components_initialized'].append('memory_graph_ml')
                logger.info("  OK: MemoryGraphML доступен")
            else:
                logger.warning("  WARN: MemoryGraphML недоступен")
            
            # 6. ReasoningEngine
            logger.info("  [6/8] Проверка ReasoningEngine...")
            try:
                from cogniflex.core.reasoning_engine import ReasoningEngine
                reasoning = ReasoningEngine(brain=self.brain)
                init_results['reasoning_engine'] = True
                self.metrics['components_initialized'].append('reasoning_engine')
                logger.info("  OK: ReasoningEngine создан")
            except Exception as e:
                logger.error("  FAIL: ReasoningEngine: %s", str(e))
                init_results['errors'].append(f"ReasoningEngine: {str(e)}")
            
            # 7. KnowledgeGraph
            logger.info("  [7/8] Проверка KnowledgeGraph...")
            if hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
                init_results['knowledge_graph'] = True
                self.metrics['components_initialized'].append('knowledge_graph')
                logger.info("  OK: KnowledgeGraph доступен")
            else:
                logger.warning("  WARN: KnowledgeGraph недоступен")
            
            # 8. Фрактальная модель
            logger.info("  [8/8] Проверка фрактальной модели...")
            if hasattr(self.brain, 'fractal_ready'):
                init_results['fractal_model'] = self.brain.fractal_ready
                if self.brain.fractal_ready:
                    self.metrics['components_initialized'].append('fractal_model')
                    logger.info("  OK: Фрактальная модель готова")
                else:
                    logger.warning("  WARN: Фрактальная модель не готова (fractal_ready=False)")
            else:
                logger.warning("  WARN: fractal_ready не найден")
                
        except Exception as e:
            logger.error("Критическая ошибка инициализации: %s", str(e))
            init_results['errors'].append(f"Critical: {str(e)}")
            traceback.print_exc()
        
        self.test_results['initialization'] = init_results
        
        # Подсчёт успешных
        success_count = sum(1 for k, v in init_results.items() 
                          if k != 'errors' and isinstance(v, bool) and v)
        total_count = sum(1 for k, v in init_results.items() 
                         if k != 'errors' and isinstance(v, bool))
        logger.info("Результат инициализации: %d/%d компонентов", success_count, total_count)
    
    def _test_text_generation(self):
        """Тест генерации текста"""
        logger.info("Тестирование генерации текста...")
        
        gen_results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'responses': [],
            'errors': []
        }
        
        if not self.brain:
            logger.error("Brain не инициализирован, пропускаем тест")
            gen_results['errors'].append("Brain not initialized")
            self.test_results['generation'] = gen_results
            return
        
        test_queries = [
            "Привет, как дела?",
            "Что такое искусственный интеллект?",
            "Расскажи о машинном обучении",
        ]
        
        for i, query in enumerate(test_queries, 1):
            gen_results['tests_run'] += 1
            logger.info("  [Тест %d/%d] Запрос: '%s'", i, len(test_queries), query[:40])
            
            try:
                start_time = time.time()
                
                if hasattr(self.brain, 'process_query'):
                    response = self.brain.process_query(query)
                    processing_time = time.time() - start_time
                    
                    if response and len(str(response)) > 10:
                        gen_results['tests_passed'] += 1
                        logger.info("  OK: Ответ получен за %.3fs", processing_time)
                        gen_results['responses'].append({
                            'query': query,
                            'response': str(response)[:150],
                            'time': processing_time
                        })
                    else:
                        gen_results['tests_failed'] += 1
                        logger.warning("  WARN: Некачественный ответ")
                else:
                    gen_results['tests_failed'] += 1
                    logger.error("  FAIL: process_query недоступен")
                    
            except Exception as e:
                gen_results['tests_failed'] += 1
                logger.error("  FAIL: Ошибка генерации: %s", str(e))
                gen_results['errors'].append(f"Query {i}: {str(e)}")
        
        self.test_results['generation'] = gen_results
        logger.info("Результат генерации: %d/%d", gen_results['tests_passed'], gen_results['tests_run'])
    
    def _test_self_learning(self):
        """Тест самообучения с эпохами"""
        logger.info("Тестирование самообучения...")
        
        learning_results = {
            'system_available': False,
            'data_added': False,
            'training_triggered': False,
            'epochs_completed': 0,
            'final_metrics': {},
            'errors': []
        }
        
        if not self.brain or not hasattr(self.brain, 'enhanced_learning'):
            logger.error("EnhancedLearning недоступен")
            learning_results['errors'].append("enhanced_learning not available")
            self.test_results['self_learning'] = learning_results
            return
        
        enhanced = self.brain.enhanced_learning
        learning_results['system_available'] = True
        logger.info("  OK: EnhancedSelfLearningSystem доступен")
        
        try:
            # Добавление данных
            logger.info("  Добавление тренировочных данных...")
            training_texts = [
                "Привет, как дела? Что нового?",
                "Искусственный интеллект - это область информатики",
                "Машинное обучение позволяет компьютерам учиться",
                "Нейронные сети имитируют работу мозга",
                "Python популярен для разработки ИИ",
                "Глубокое обучение использует многослойные сети",
                "Обработка естественного языка - важная область ИИ",
                "Компьютерное зрение позволяет машинам видеть",
                "Робототехника объединяет ИИ с устройствами",
                "Большие языковые модели показывают результаты"
            ]
            
            added_count = 0
            for text in training_texts:
                if enhanced.add_training_data(text, source="test"):
                    added_count += 1
            
            learning_results['data_added'] = added_count > 0
            logger.info("  OK: Добавлено %d/%d текстов", added_count, len(training_texts))
            
            # Проверка статистики
            pre_stats = enhanced.get_full_stats()
            logger.info("  Статистика до обучения: %d samples", 
                       pre_stats.get('system_status', {}).get('unprocessed_samples', 0))
            
            # Принудительное обучение
            logger.info("  Запуск обучения (2 эпохи)...")
            
            epoch_metrics_log = []
            
            def progress_callback(progress, session_data):
                logger.info("    Прогресс: %.1f%%", progress)
            
            def epoch_callback(metrics):
                epoch_metrics_log.append(metrics)
                logger.info("    Эпоха %d: loss=%.4f, acc=%.4f", 
                           metrics.get('epoch', 0),
                           metrics.get('loss', 0),
                           metrics.get('accuracy', 0))
            
            result = enhanced.force_training(
                epochs=2,
                progress_callback=progress_callback,
                epoch_callback=epoch_callback
            )
            
            learning_results['training_triggered'] = result.get('success', False)
            
            if result.get('success'):
                session = result.get('session', {})
                learning_results['epochs_completed'] = session.get('epochs_completed', 0)
                learning_results['final_metrics'] = {
                    'validation_accuracy': session.get('validation_accuracy', 0),
                    'final_loss': session.get('final_loss', 0),
                    'duration': session.get('duration', 0)
                }
                logger.info("  OK: Обучение завершено")
                logger.info("    Эпох: %d", learning_results['epochs_completed'])
                logger.info("    Точность: %.2f%%", learning_results['final_metrics']['validation_accuracy'] * 100)
            else:
                logger.error("  FAIL: Обучение не удалось: %s", result.get('message', 'Unknown'))
                learning_results['errors'].append(f"Training failed: {result.get('message')}")
                
        except Exception as e:
            logger.error("Ошибка тестирования самообучения: %s", str(e))
            learning_results['errors'].append(f"Exception: {str(e)}")
            traceback.print_exc()
        
        self.test_results['self_learning'] = learning_results
    
    def _test_reasoning(self):
        """Тест рассуждения"""
        logger.info("Тестирование рассуждения...")
        
        reasoning_results = {
            'engine_created': False,
            'tests_run': 0,
            'tests_passed': 0,
            'reasoning_times': [],
            'errors': []
        }
        
        if not self.brain:
            logger.error("Brain не инициализирован")
            reasoning_results['errors'].append("Brain not initialized")
            self.test_results['reasoning'] = reasoning_results
            return
        
        try:
            logger.info("  Создание ReasoningEngine...")
            from cogniflex.core.reasoning_engine import ReasoningEngine
            reasoning = ReasoningEngine(brain=self.brain, config={'enable_web_search': False})
            reasoning_results['engine_created'] = True
            logger.info("  OK: ReasoningEngine создан")
            
            test_queries = [
                "Что такое машинное обучение?",
                "Объясни разницу между ИИ и ML"
            ]
            
            logger.info("  Тестирование рассуждений...")
            for i, query in enumerate(test_queries, 1):
                reasoning_results['tests_run'] += 1
                
            # Добавление концептов
            test_concepts = [
                ('test_ai2', 'Искусственный интеллект', 'concept'),
                ('test_ml2', 'Машинное обучение', 'concept'),
            ]
                
            for concept_id, name, ctype in test_concepts:
                try:
                    if hasattr(kg, 'add_concept'):
                        kg.add_concept(id=concept_id, name=name, node_type=ctype, strength=0.8)
                        memory_results['entities_added'] += 1
                except Exception as e:
                    logger.debug("Ошибка добавления концепта: %s", str(e))
                
            logger.info("    Добавлено концептов: %d", memory_results['entities_added'])
        else:
            logger.warning("  WARN: KnowledgeGraph недоступен")
            
        # MemoryGraphML
        logger.info("  Проверка MemoryGraphML...")
        if hasattr(self.brain, 'memory_graph_ml') and self.brain.memory_graph_ml:
            memory_results['memory_graph_ml_available'] = True
            mgml = self.brain.memory_graph_ml
            logger.info("  OK: MemoryGraphML доступен")
                
            # Статистика
            if hasattr(mgml, 'get_stats'):
                stats = mgml.get_stats()
                memory_results['stats'] = stats
                logger.info("    Embeddings: %d, Паттернов: %d", 
                           stats.get('embeddings_count', 0),
                           stats.get('patterns_count', 0))
            logger.error("Ошибка тестирования рассуждений: %s", str(e))
            reasoning_results['errors'].append(f"Exception: {str(e)}")
        
        self.test_results['reasoning'] = reasoning_results
        logger.info("Результат: %d/%d", reasoning_results['tests_passed'], reasoning_results['tests_run'])
    
    def _test_memory_graph(self):
        """Тест графа памяти"""
        logger.info("Тестирование графа памяти...")
        
        memory_results = {
            'knowledge_graph_available': False,
            'memory_graph_ml_available': False,
            'context_retrieved': False,
            'fractal_context_retrieved': False,
            'entities_added': 0,
            'stats': {},
            'errors': []
        }
        
        if not self.brain:
            logger.error("Brain не инициализирован")
            memory_results['errors'].append("Brain not initialized")
            self.test_results['memory_graph'] = memory_results
            return
        
        try:
            # KnowledgeGraph
            logger.info("  Проверка KnowledgeGraph...")
            if hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
                memory_results['knowledge_graph_available'] = True
                kg = self.brain.knowledge_graph
                logger.info("  OK: KnowledgeGraph доступен")
                
                # Добавление концептов
                test_concepts = [
                    ('test_ai2', 'Искусственный интеллект', 'concept'),
                    ('test_ml2', 'Машинное обучение', 'concept'),
                ]
                
                for concept_id, name, ctype in test_concepts:
                    try:
                        if hasattr(kg, 'add_concept'):
                            kg.add_concept(id=concept_id, name=name, node_type=ctype, strength=0.8)
                            memory_results['entities_added'] += 1
                    except Exception as e:
                        logger.debug("Ошибка добавления концепта: %s", str(e))
                
                logger.info("    Добавлено концептов: %d", memory_results['entities_added'])
            else:
                logger.warning("  WARN: KnowledgeGraph недоступен")
            
            # MemoryGraphML
            logger.info("  Проверка MemoryGraphML...")
            if hasattr(self.brain, 'memory_graph_ml') and self.brain.memory_graph_ml:
                memory_results['memory_graph_ml_available'] = True
                mgml = self.brain.memory_graph_ml
                logger.info("  OK: MemoryGraphML доступен")
                
                # Статистика
                if hasattr(mgml, 'get_stats'):
                    stats = mgml.get_stats()
                    memory_results['stats'] = stats
                    logger.info("    Embeddings: %d, Паттернов: %d", 
                               stats.get('embeddings_count', 0),
                               stats.get('patterns_count', 0))
                
                # Получение контекста
                test_query = "нейронные сети и машинное обучение"
                
                if hasattr(mgml, 'get_context_for_query'):
                    context = mgml.get_context_for_query(test_query)
                    if context:
                        memory_results['context_retrieved'] = True
                        logger.info("  OK: Контекст получен (сущностей: %d)", 
                                   len(context.get('entities', [])))
                
                # Фрактальный контекст
                if hasattr(mgml, 'get_fractal_context'):
                    fractal_ctx = mgml.get_fractal_context(test_query, level=0)
                    if fractal_ctx:
                        memory_results['fractal_context_retrieved'] = True
                        logger.info("  OK: Фрактальный контекст получен")
            else:
                logger.warning("  WARN: MemoryGraphML недоступен")
                
        except Exception as e:
            logger.error("Ошибка тестирования графа: %s", str(e))
            memory_results['errors'].append(f"Exception: {str(e)}")
        
        self.test_results['memory_graph'] = memory_results
    
    def _test_integration(self):
        """Тест интеграции"""
        logger.info("Тестирование интеграции компонентов...")
        
        integration_results = {
            'brain_to_learning': False,
            'brain_to_memory_graph': False,
            'brain_to_reasoning': False,
            'learning_to_memory': False,
            'end_to_end': False,
            'errors': []
        }
        
        if not self.brain:
            logger.error("Brain не инициализирован")
            integration_results['errors'].append("Brain not initialized")
            self.test_results['integration'] = integration_results
            return
        
        try:
            # Brain -> EnhancedLearning
            logger.info("  Проверка Brain -> EnhancedLearning...")
            if self.brain and hasattr(self.brain, 'enhanced_learning'):
                integration_results['brain_to_learning'] = True
                logger.info("  OK: Связь Brain <-> EnhancedLearning")
            
            # Brain -> MemoryGraphML
            logger.info("  Проверка Brain -> MemoryGraphML...")
            if self.brain and hasattr(self.brain, 'memory_graph_ml'):
                integration_results['brain_to_memory_graph'] = True
                logger.info("  OK: Связь Brain <-> MemoryGraphML")
            
            # Brain -> Reasoning
            logger.info("  Проверка Brain -> Reasoning...")
            try:
                from cogniflex.core.reasoning_engine import ReasoningEngine
                reasoning = ReasoningEngine(brain=self.brain)
                integration_results['brain_to_reasoning'] = True
                logger.info("  OK: ReasoningEngine доступен")
            except Exception as e:
                logger.warning("  WARN: %s", str(e))
            
            # End-to-end
            logger.info("  End-to-End тест...")
            try:
                # Добавляем данные
                if self.brain and hasattr(self.brain, 'enhanced_learning') and self.brain.enhanced_learning:
                    self.brain.enhanced_learning.add_training_data("Тестовый текст", source="e2e_test")
                
                # Запрашиваем контекст
                if self.brain and hasattr(self.brain, 'memory_graph_ml') and self.brain.memory_graph_ml:
                    self.brain.memory_graph_ml.get_context_for_query("тест")
                
                # Генерируем ответ
                if hasattr(self.brain, 'process_query'):
                    self.brain.process_query("Тест")
                
                integration_results['end_to_end'] = True
                logger.info("  OK: End-to-End пройден")
            except Exception as e:
                logger.error("  FAIL: End-to-End: %s", str(e))
                integration_results['errors'].append(f"E2E: {str(e)}")
                
        except Exception as e:
            logger.error("Ошибка тестирования интеграции: %s", str(e))
            integration_results['errors'].append(f"Exception: {str(e)}")
        
        self.test_results['integration'] = integration_results
        
        passed = sum(1 for v in integration_results.values() if isinstance(v, bool) and v)
        total = sum(1 for v in integration_results.values() if isinstance(v, bool))
        logger.info("Результат интеграции: %d/%d", passed, total)
    
    def _generate_report(self):
        """Генерация отчёта"""
        logger.info("\n" + "=" * 80)
        logger.info("ИТОГОВЫЙ ОТЧЁТ")
        logger.info("=" * 80)
        
        end_time = time.time()
        total_duration = end_time - self.metrics['start_time']
        
        # Подсчёт по категориям
        def count_results(category):
            data = self.test_results.get(category, {})
            if isinstance(data, dict):
                passed = sum(1 for k, v in data.items() 
                           if k not in ['errors', 'responses', 'stats', 'final_metrics', 'reasoning_times', 'phases_executed'] 
                           and isinstance(v, bool) and v)
                total = sum(1 for k, v in data.items() 
                           if k not in ['errors', 'responses', 'stats', 'final_metrics', 'reasoning_times', 'phases_executed'] 
                           and isinstance(v, bool))
                return passed, total
            return 0, 0
        
        init_passed, init_total = count_results('initialization')
        
        gen_data = self.test_results.get('generation', {})
        gen_passed = gen_data.get('tests_passed', 0)
        gen_total = gen_data.get('tests_run', 0)
        
        learning_data = self.test_results.get('self_learning', {})
        learning_passed = 1 if learning_data.get('training_triggered') else 0
        learning_total = 1 if learning_data.get('system_available') else 0
        
        reasoning_data = self.test_results.get('reasoning', {})
        reasoning_passed = reasoning_data.get('tests_passed', 0)
        reasoning_total = reasoning_data.get('tests_run', 0)
        
        memory_data = self.test_results.get('memory_graph', {})
        memory_passed = sum([
            memory_data.get('knowledge_graph_available', False),
            memory_data.get('memory_graph_ml_available', False),
            memory_data.get('context_retrieved', False)
        ])
        memory_total = 3
        
        integration_passed, integration_total = count_results('integration')
        
        total_passed = init_passed + gen_passed + learning_passed + reasoning_passed + memory_passed + integration_passed
        total_tests = init_total + gen_total + learning_total + reasoning_total + memory_total + integration_total
        
        summary = {
            'test_duration_seconds': round(total_duration, 2),
            'components_initialized': len(self.metrics['components_initialized']),
            'total_tests': total_tests,
            'passed_tests': total_passed,
            'failed_tests': total_tests - total_passed,
            'success_rate': round(total_passed / total_tests * 100, 1) if total_tests > 0 else 0,
            'errors_count': len(self.metrics['errors']),
            'breakdown': {
                'initialization': {'passed': init_passed, 'total': init_total},
                'generation': {'passed': gen_passed, 'total': gen_total},
                'self_learning': {'passed': learning_passed, 'total': learning_total},
                'reasoning': {'passed': reasoning_passed, 'total': reasoning_total},
                'memory_graph': {'passed': memory_passed, 'total': memory_total},
                'integration': {'passed': integration_passed, 'total': integration_total}
            }
        }
        
        self.test_results['summary'] = summary
        
        # Вывод
        logger.info("ОБЩАЯ СТАТИСТИКА:")
        logger.info("  Длительность: %.2f секунд", total_duration)
        logger.info("  Компонентов инициализировано: %d", summary['components_initialized'])
        logger.info("  Всего тестов: %d", total_tests)
        logger.info("  Пройдено: %d", total_passed)
        logger.info("  Не пройдено: %d", summary['failed_tests'])
        logger.info("  Успешность: %.1f%%", summary['success_rate'])
        
        logger.info("\nРЕЗУЛЬТАТЫ ПО КАТЕГОРИЯМ:")
        logger.info("  Инициализация: %d/%d", init_passed, init_total)
        logger.info("  Генерация: %d/%d", gen_passed, gen_total)
        logger.info("  Самообучение: %d/%d", learning_passed, learning_total)
        logger.info("  Рассуждение: %d/%d", reasoning_passed, reasoning_total)
        logger.info("  Граф памяти: %d/%d", memory_passed, memory_total)
        logger.info("  Интеграция: %d/%d", integration_passed, integration_total)
        
        # Сохранение
        report_file = "comprehensive_test_report_v2_{}.json".format(datetime.now().strftime('%Y%m%d_%H%M%S'))
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2, default=str)
            logger.info("\nОтчёт сохранён: %s", report_file)
        except Exception as e:
            logger.error("Ошибка сохранения отчёта: %s", str(e))
        
        # Финальный статус
        logger.info("\n" + "=" * 80)
        if summary['success_rate'] >= 80:
            logger.info("СТАТУС: СИСТЕМА РАБОТОСПОСОБНА")
        elif summary['success_rate'] >= 50:
            logger.info("СТАТУС: СИСТЕМА ЧАСТИЧНО РАБОТОСПОСОБНА")
        else:
            logger.info("СТАТУС: СИСТЕМА ИМЕЕТ ПРОБЛЕМЫ")
        logger.info("=" * 80)


def main():
    """Главная функция"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SYSTEM TEST V2 FOR COGNIFLEX")
    print("=" * 80)
    
    test = ComprehensiveSystemTestV2()
    results = test.run_all_tests()
    
    summary = results.get('summary', {})
    print("\n" + "=" * 80)
    print("КРАТКАЯ СВОДКА:")
    print("  Успешность: {}%".format(summary.get('success_rate', 0)))
    print("  Пройдено: {}/{}".format(summary.get('passed_tests', 0), summary.get('total_tests', 0)))
    print("  Длительность: {}s".format(summary.get('test_duration_seconds', 0)))
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
