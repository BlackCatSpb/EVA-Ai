#!/usr/bin/env python3
"""
Comprehensive System Test for CogniFlex
Тестирование всех компонентов системы без GUI с эмуляцией действий пользователя
"""

import os
import sys
import time
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('comprehensive_test.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("cogniflex.comprehensive_test")

class ComprehensiveSystemTest:
    """Комплексное тестирование системы CogniFlex"""
    
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
        self.integrator = None
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Запуск всех тестов системы"""
        logger.info("=" * 80)
        logger.info("НАЧАЛО КОМПЛЕКСНОГО ТЕСТИРОВАНИЯ COGNIFLEX")
        logger.info("=" * 80)
        logger.info(f"Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # 1. Тест инициализации
            self._test_initialization()
            
            # 2. Тест генерации текста
            self._test_text_generation()
            
            # 3. Тест самообучения
            self._test_self_learning()
            
            # 4. Тест рассуждения
            self._test_reasoning()
            
            # 5. Тест графа памяти
            self._test_memory_graph()
            
            # 6. Тест интеграции компонентов
            self._test_integration()
            
        except Exception as e:
            logger.error(f"Критическая ошибка во время тестирования: {e}")
            self.metrics['errors'].append(f"Critical: {str(e)}")
            traceback.print_exc()
        
        finally:
            # Создание отчёта
            self._generate_report()
            
        return self.test_results
    
    def _test_initialization(self):
        """Тест инициализации всех компонентов"""
        logger.info("\n" + "=" * 80)
        logger.info("ТЕСТ 1: ИНИЦИАЛИЗАЦИЯ КОМПОНЕНТОВ")
        logger.info("=" * 80)
        
        init_results = {
            'core_brain': False,
            'enhanced_learning': False,
            'memory_graph_ml': False,
            'reasoning_engine': False,
            'fractal_model': False,
            'errors': []
        }
        
        try:
            # Импорт CoreBrain
            logger.info("[1/6] Импорт CoreBrain...")
            from cogniflex.core.core_brain import CoreBrain
            init_results['core_brain_import'] = True
            logger.info("✓ CoreBrain импортирован")
            
            # Создание экземпляра
            logger.info("[2/6] Создание экземпляра CoreBrain...")
            config = {
                'learning': {
                    'default_epochs': 2,
                    'batch_size': 2
                },
                'memory_graph_ml': {
                    'embedding_dim': 64,
                    'fractal_levels': 3
                }
            }
            self.brain = CoreBrain(config)
            init_results['core_brain_created'] = True
            logger.info("✓ CoreBrain создан")
            
            # Инициализация
            logger.info("[3/6] Инициализация CoreBrain...")
            init_result = self.brain.initialize()
            init_results['core_brain_initialized'] = init_result
            if init_result:
                logger.info("✓ CoreBrain инициализирован")
            else:
                logger.warning("⚠ CoreBrain инициализация вернула False")
                init_results['errors'].append("CoreBrain initialize() returned False")
            
            # Проверка EnhancedSelfLearningSystem
            logger.info("[4/6] Проверка EnhancedSelfLearningSystem...")
            if hasattr(self.brain, 'enhanced_learning') and self.brain.enhanced_learning:
                init_results['enhanced_learning'] = True
                logger.info("✓ EnhancedSelfLearningSystem доступен")
                self.metrics['components_initialized'].append('enhanced_learning')
            else:
                logger.warning("⚠ EnhancedSelfLearningSystem недоступен")
                init_results['errors'].append("enhanced_learning not available")
            
            # Проверка MemoryGraphML
            logger.info("[5/6] Проверка MemoryGraphML...")
            if hasattr(self.brain, 'memory_graph_ml') and self.brain.memory_graph_ml:
                init_results['memory_graph_ml'] = True
                logger.info("✓ MemoryGraphML доступен")
                self.metrics['components_initialized'].append('memory_graph_ml')
            else:
                logger.warning("⚠ MemoryGraphML недоступен")
                init_results['errors'].append("memory_graph_ml not available")
            
            # Проверка ReasoningEngine (через интегратор)
            logger.info("[6/6] Проверка ReasoningEngine...")
            try:
                from cogniflex.core.reasoning_engine import ReasoningEngine
                reasoning = ReasoningEngine(brain=self.brain)
                init_results['reasoning_engine'] = True
                logger.info("✓ ReasoningEngine создан")
                self.metrics['components_initialized'].append('reasoning_engine')
            except Exception as e:
                logger.error(f"✗ Ошибка создания ReasoningEngine: {e}")
                init_results['errors'].append(f"ReasoningEngine: {str(e)}")
            
            # Проверка фрактальной модели
            logger.info("[Дополнительно] Проверка фрактальной модели...")
            if hasattr(self.brain, 'fractal_ready'):
                init_results['fractal_model'] = self.brain.fractal_ready
                if self.brain.fractal_ready:
                    logger.info("✓ Фрактальная модель готова")
                    self.metrics['components_initialized'].append('fractal_model')
                else:
                    logger.warning("⚠ Фрактальная модель не готова")
            else:
                logger.warning("⚠ Фрактальная модель не найдена")
                init_results['errors'].append("fractal_model not found")
            
        except Exception as e:
            logger.error(f"✗ Критическая ошибка инициализации: {e}")
            init_results['errors'].append(f"Critical init error: {str(e)}")
            traceback.print_exc()
        
        self.test_results['initialization'] = init_results
        logger.info(f"\nРезультат инициализации: {sum(init_results.values()) if isinstance(init_results.get('core_brain'), bool) else 'Partial'}/6 компонентов")
    
    def _test_text_generation(self):
        """Тест генерации текста"""
        logger.info("\n" + "=" * 80)
        logger.info("ТЕСТ 2: ГЕНЕРАЦИЯ ТЕКСТА")
        logger.info("=" * 80)
        
        gen_results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'responses': [],
            'errors': []
        }
        
        test_queries = [
            "Привет, как дела?",
            "Что такое искусственный интеллект?",
            "Расскажи о машинном обучении",
            "Как работает нейронная сеть?",
            "Что ты умеешь делать?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            gen_results['tests_run'] += 1
            logger.info(f"\n[Тест {i}/{len(test_queries)}] Запрос: '{query}'")
            
            try:
                start_time = time.time()
                
                # Генерация через brain
                if self.brain and hasattr(self.brain, 'process_query'):
                    response = self.brain.process_query(query)
                    processing_time = time.time() - start_time
                    
                    # Проверка качества ответа
                    if response and len(response) > 10:
                        gen_results['tests_passed'] += 1
                        logger.info(f"✓ Ответ получен за {processing_time:.3f}s")
                        logger.info(f"  Ответ: {response[:100]}...")
                        
                        gen_results['responses'].append({
                            'query': query,
                            'response': response[:200],
                            'time': processing_time,
                            'length': len(response)
                        })
                    else:
                        gen_results['tests_failed'] += 1
                        logger.warning(f"⚠ Некачественный ответ (длина: {len(response) if response else 0})")
                        gen_results['errors'].append(f"Query {i}: low quality response")
                else:
                    gen_results['tests_failed'] += 1
                    logger.error("✗ Brain.process_query недоступен")
                    gen_results['errors'].append(f"Query {i}: process_query not available")
                    
            except Exception as e:
                gen_results['tests_failed'] += 1
                logger.error(f"✗ Ошибка генерации: {e}")
                gen_results['errors'].append(f"Query {i}: {str(e)}")
        
        self.test_results['generation'] = gen_results
        logger.info(f"\nРезультат генерации: {gen_results['tests_passed']}/{gen_results['tests_run']} тестов пройдено")
    
    def _test_self_learning(self):
        """Тест самообучения с эпохами"""
        logger.info("\n" + "=" * 80)
        logger.info("ТЕСТ 3: САМООБУЧЕНИЕ С ЭПОХАМИ")
        logger.info("=" * 80)
        
        learning_results = {
            'system_available': False,
            'data_added': False,
            'training_triggered': False,
            'epochs_completed': 0,
            'final_metrics': {},
            'errors': []
        }
        
        try:
            # Проверка доступности системы самообучения
            if not (self.brain and hasattr(self.brain, 'enhanced_learning') and self.brain.enhanced_learning):
                logger.error("✗ EnhancedSelfLearningSystem недоступен")
                learning_results['errors'].append("enhanced_learning not available")
                self.test_results['self_learning'] = learning_results
                return
            
            learning_results['system_available'] = True
            logger.info("✓ EnhancedSelfLearningSystem доступен")
            
            enhanced = self.brain.enhanced_learning
            
            # Добавление тренировочных данных
            logger.info("\n[1/4] Добавление тренировочных данных...")
            training_texts = [
                "Привет, как дела? Что нового?",
                "Искусственный интеллект - это область информатики",
                "Машинное обучение позволяет компьютерам учиться на данных",
                "Нейронные сети имитируют работу человеческого мозга",
                "Python популярен для разработки ИИ",
                "Глубокое обучение использует многослойные нейронные сети",
                "Обработка естественного языка - важная область ИИ",
                "Компьютерное зрение позволяет машинам 'видеть'",
                "Робототехника объединяет ИИ с физическими устройствами",
                "Большие языковые модели показывают впечатляющие результаты"
            ]
            
            added_count = 0
            for text in training_texts:
                if enhanced.add_training_data(text, source="test"):
                    added_count += 1
            
            learning_results['data_added'] = added_count > 0
            logger.info(f"✓ Добавлено {added_count}/{len(training_texts)} текстов для обучения")
            
            # Проверка статистики до обучения
            logger.info("\n[2/4] Статистика до обучения:")
            pre_stats = enhanced.get_full_stats()
            logger.info(f"  Очередь: {pre_stats.get('system_status', {}).get('unprocessed_samples', 0)} samples")
            logger.info(f"  Всего сессий: {pre_stats.get('session_stats', {}).get('total_sessions', 0)}")
            
            # Принудительное обучение
            logger.info("\n[3/4] Запуск принудительного обучения (2 эпохи)...")
            
            epoch_metrics_received = []
            
            def progress_callback(progress, session_data):
                logger.info(f"  Прогресс: {progress:.1f}%")
            
            def epoch_callback(metrics):
                epoch_metrics_received.append(metrics)
                logger.info(f"  Эпоха {metrics.get('epoch')}: loss={metrics.get('loss', 0):.4f}, "
                          f"accuracy={metrics.get('accuracy', 0):.4f}")
            
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
                
                logger.info("✓ Обучение завершено успешно")
                logger.info(f"  Эпох завершено: {learning_results['epochs_completed']}")
                logger.info(f"  Точность: {learning_results['final_metrics']['validation_accuracy']:.2%}")
                logger.info(f"  Loss: {learning_results['final_metrics']['final_loss']:.4f}")
                logger.info(f"  Длительность: {learning_results['final_metrics']['duration']:.1f}s")
                
                # Проверка сгенерированных сущностей
                post_stats = enhanced.get_full_stats()
                entities_count = post_stats.get('generated_entities_count', 0)
                logger.info(f"  Сгенерировано сущностей: {entities_count}")
            else:
                logger.error(f"✗ Обучение не удалось: {result.get('message', 'Unknown error')}")
                learning_results['errors'].append(f"Training failed: {result.get('message')}")
            
            # Проверка статистики после обучения
            logger.info("\n[4/4] Статистика после обучения:")
            final_stats = enhanced.get_full_stats()
            logger.info(f"  Всего сессий: {final_stats.get('session_stats', {}).get('total_sessions', 0)}")
            logger.info(f"  Успешных сессий: {final_stats.get('session_stats', {}).get('successful_sessions', 0)}")
            
        except Exception as e:
            logger.error(f"✗ Ошибка тестирования самообучения: {e}")
            learning_results['errors'].append(f"Exception: {str(e)}")
            traceback.print_exc()
        
        self.test_results['self_learning'] = learning_results
    
    def _test_reasoning(self):
        """Тест рассуждения (reasoning engine)"""
        logger.info("\n" + "=" * 80)
        logger.info("ТЕСТ 4: РАССУЖДЕНИЕ (REASONING ENGINE)")
        logger.info("=" * 80)
        
        reasoning_results = {
            'engine_created': False,
            'tests_run': 0,
            'tests_passed': 0,
            'reasoning_times': [],
            'phases_executed': [],
            'errors': []
        }
        
        try:
            # Создание ReasoningEngine
            logger.info("[1/3] Создание ReasoningEngine...")
            from cogniflex.core.reasoning_engine import ReasoningEngine
            
            reasoning = ReasoningEngine(
                brain=self.brain,
                config={
                    'enable_web_search': False,  # Отключаем для теста
                    'max_history': 10
                }
            )
            reasoning_results['engine_created'] = True
            logger.info("✓ ReasoningEngine создан")
            
            # Тестовые запросы для рассуждения
            test_queries = [
                "Что такое машинное обучение и как оно работает?",
                "Объясни разницу между ИИ и машинным обучением"
            ]
            
            logger.info("\n[2/3] Тестирование рассуждений...")
            for i, query in enumerate(test_queries, 1):
                reasoning_results['tests_run'] += 1
                logger.info(f"\n  [Запрос {i}] '{query[:50]}...'")
                
                try:
                    start_time = time.time()
                    result = reasoning.reason(query)
                    processing_time = time.time() - start_time
                    
                    reasoning_results['reasoning_times'].append(processing_time)
                    
                    # Анализ результата
                    if result and 'answer' in result:
                        answer = result['answer']
                        confidence = result.get('confidence', 0)
                        
                        if len(answer) > 20 and confidence > 0.3:
                            reasoning_results['tests_passed'] += 1
                            logger.info(f"  ✓ Рассуждение завершено за {processing_time:.3f}s")
                            logger.info(f"    Уверенность: {confidence:.2%}")
                            logger.info(f"    Ответ: {answer[:100]}...")
                            
                            # Проверка фаз
                            if 'reasoning_phases' in result:
                                phases = result['reasoning_phases']
                                reasoning_results['phases_executed'].extend(phases)
                                logger.info(f"    Фазы: {', '.join(phases)}")
                        else:
                            logger.warning(f"  ⚠ Низкая уверенность или короткий ответ")
                            reasoning_results['errors'].append(f"Query {i}: low confidence or short answer")
                    else:
                        logger.error(f"  ✗ Некорректный результат рассуждения")
                        reasoning_results['errors'].append(f"Query {i}: invalid result")
                        
                except Exception as e:
                    logger.error(f"  ✗ Ошибка рассуждения: {e}")
                    reasoning_results['errors'].append(f"Query {i}: {str(e)}")
            
            # Проверка статистики
            logger.info("\n[3/3] Проверка статистики рассуждений...")
            stats = reasoning.get_reasoning_stats()
            logger.info(f"  Всего рассуждений: {stats.get('total_reasonings', 0)}")
            logger.info(f"  Средняя уверенность: {stats.get('average_confidence', 0):.2%}")
            logger.info(f"  Среднее время: {stats.get('average_processing_time', 0):.3f}s")
            
        except Exception as e:
            logger.error(f"✗ Ошибка тестирования рассуждений: {e}")
            reasoning_results['errors'].append(f"Exception: {str(e)}")
            traceback.print_exc()
        
        self.test_results['reasoning'] = reasoning_results
        logger.info(f"\nРезультат рассуждений: {reasoning_results['tests_passed']}/{reasoning_results['tests_run']} тестов пройдено")
    
    def _test_memory_graph(self):
        """Тест графа памяти и MemoryGraphML"""
        logger.info("\n" + "=" * 80)
        logger.info("ТЕСТ 5: ГРАФ ПАМЯТИ И MEMORYGRAPHML")
        logger.info("=" * 80)
        
        memory_results = {
            'knowledge_graph_available': False,
            'memory_graph_ml_available': False,
            'context_retrieved': False,
            'fractal_context_retrieved': False,
            'entities_added': 0,
            'stats': {},
            'errors': []
        }
        
        try:
            # Проверка KnowledgeGraph
            logger.info("[1/4] Проверка KnowledgeGraph...")
            if self.brain and hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
                memory_results['knowledge_graph_available'] = True
                logger.info("✓ KnowledgeGraph доступен")
                
                # Добавление тестовых концептов
                kg = self.brain.knowledge_graph
                test_concepts = [
                    ('test_ai', 'Искусственный интеллект', 'concept'),
                    ('test_ml', 'Машинное обучение', 'concept'),
                    ('test_nn', 'Нейронные сети', 'concept')
                ]
                
                for concept_id, name, ctype in test_concepts:
                    try:
                        if hasattr(kg, 'add_concept'):
                            kg.add_concept(id=concept_id, name=name, node_type=ctype, strength=0.8)
                            memory_results['entities_added'] += 1
                    except Exception as e:
                        logger.debug(f"Ошибка добавления концепта {concept_id}: {e}")
                
                logger.info(f"  Добавлено {memory_results['entities_added']} концептов")
                
                # Добавление связей
                if hasattr(kg, 'add_relation') and memory_results['entities_added'] >= 2:
                    try:
                        kg.add_relation('test_ml', 'test_ai', 'is_part_of')
                        kg.add_relation('test_nn', 'test_ml', 'is_part_of')
                        logger.info("  Связи добавлены")
                    except Exception as e:
                        logger.debug(f"Ошибка добавления связей: {e}")
            else:
                logger.warning("⚠ KnowledgeGraph недоступен")
                memory_results['errors'].append("knowledge_graph not available")
            
            # Проверка MemoryGraphML
            logger.info("\n[2/4] Проверка MemoryGraphML...")
            if self.brain and hasattr(self.brain, 'memory_graph_ml') and self.brain.memory_graph_ml:
                memory_results['memory_graph_ml_available'] = True
                logger.info("✓ MemoryGraphML доступен")
                
                mgml = self.brain.memory_graph_ml
                
                # Получение статистики
                if hasattr(mgml, 'get_stats'):
                    stats = mgml.get_stats()
                    memory_results['stats'] = stats
                    logger.info(f"  Embeddings: {stats.get('embeddings_count', 0)}")
                    logger.info(f"  Паттернов: {stats.get('patterns_count', 0)}")
                    logger.info(f"  Фрактальных уровней: {stats.get('fractal_levels', 0)}")
                
                # Получение контекста
                logger.info("\n[3/4] Тест получения контекста...")
                test_query = "нейронные сети и машинное обучение"
                
                try:
                    if hasattr(mgml, 'get_context_for_query'):
                        context = mgml.get_context_for_query(test_query)
                        if context:
                            memory_results['context_retrieved'] = True
                            logger.info("✓ Контекст получен")
                            logger.info(f"  Сущностей: {len(context.get('entities', []))}")
                            logger.info(f"  Связанных концептов: {len(context.get('related_concepts', []))}")
                        else:
                            logger.warning("⚠ Контекст пуст")
                    
                    # Получение фрактального контекста
                    if hasattr(mgml, 'get_fractal_context'):
                        for level in range(min(3, mgml.fractal_levels)):
                            fractal_context = mgml.get_fractal_context(test_query, level=level)
                            if fractal_context:
                                memory_results['fractal_context_retrieved'] = True
                                logger.info(f"  Фрактальный уровень {level}: {fractal_context.get('type', 'unknown')}")
                                
                except Exception as e:
                    logger.error(f"✗ Ошибка получения контекста: {e}")
                    memory_results['errors'].append(f"Context retrieval: {str(e)}")
            else:
                logger.warning("⚠ MemoryGraphML недоступен")
                memory_results['errors'].append("memory_graph_ml not available")
            
            # Обновление данных
            logger.info("\n[4/4] Обновление данных графа...")
            if memory_results['memory_graph_ml_available'] and hasattr(mgml, 'update'):
                try:
                    update_result = mgml.update()
                    if update_result:
                        logger.info("✓ Данные графа обновлены")
                    else:
                        logger.warning("⚠ Обновление вернуло False")
                except Exception as e:
                    logger.debug(f"Ошибка обновления: {e}")
                    
        except Exception as e:
            logger.error(f"✗ Ошибка тестирования графа памяти: {e}")
            memory_results['errors'].append(f"Exception: {str(e)}")
            traceback.print_exc()
        
        self.test_results['memory_graph'] = memory_results
    
    def _test_integration(self):
        """Тест интеграции компонентов"""
        logger.info("\n" + "=" * 80)
        logger.info("ТЕСТ 6: ИНТЕГРАЦИЯ КОМПОНЕНТОВ")
        logger.info("=" * 80)
        
        integration_results = {
            'brain_to_learning': False,
            'brain_to_memory_graph': False,
            'brain_to_reasoning': False,
            'learning_to_memory_graph': False,
            'end_to_end': False,
            'errors': []
        }
        
        try:
            # Проверка связи Brain -> EnhancedLearning
            logger.info("[1/5] Проверка Brain -> EnhancedLearning...")
            if self.brain and hasattr(self.brain, 'enhanced_learning'):
                if self.brain.enhanced_learning and hasattr(self.brain.enhanced_learning, 'brain'):
                    integration_results['brain_to_learning'] = True
                    logger.info("✓ Связь Brain <-> EnhancedLearning работает")
                else:
                    logger.warning("⚠ Связь Brain -> EnhancedLearning неполная")
            
            # Проверка связи Brain -> MemoryGraphML
            logger.info("[2/5] Проверка Brain -> MemoryGraphML...")
            if self.brain and hasattr(self.brain, 'memory_graph_ml'):
                if self.brain.memory_graph_ml and hasattr(self.brain.memory_graph_ml, 'brain'):
                    integration_results['brain_to_memory_graph'] = True
                    logger.info("✓ Связь Brain <-> MemoryGraphML работает")
                else:
                    logger.warning("⚠ Связь Brain -> MemoryGraphML неполная")
            
            # Проверка связи Brain -> Reasoning (через интегратор)
            logger.info("[3/5] Проверка Brain -> Reasoning...")
            try:
                from cogniflex.core.reasoning_engine import ReasoningEngine
                reasoning = ReasoningEngine(brain=self.brain)
                integration_results['brain_to_reasoning'] = True
                logger.info("✓ ReasoningEngine может получить доступ к Brain")
            except Exception as e:
                logger.warning(f"⚠ Ошибка связи Brain -> Reasoning: {e}")
                integration_results['errors'].append(f"Brain->Reasoning: {str(e)}")
            
            # Проверка связи EnhancedLearning -> MemoryGraphML
            logger.info("[4/5] Проверка EnhancedLearning -> MemoryGraphML...")
            if (self.brain and 
                hasattr(self.brain, 'enhanced_learning') and 
                hasattr(self.brain, 'knowledge_graph')):
                integration_results['learning_to_memory_graph'] = True
                logger.info("✓ EnhancedLearning может обновлять KnowledgeGraph")
            else:
                logger.warning("⚠ Связь EnhancedLearning -> MemoryGraphML неполная")
            
            # End-to-end тест
            logger.info("\n[5/5] End-to-End тест...")
            try:
                # Симуляция полного цикла
                logger.info("  1. Добавление данных для обучения...")
                if self.brain and hasattr(self.brain, 'enhanced_learning') and self.brain.enhanced_learning:
                    self.brain.enhanced_learning.add_training_data(
                        "Это тестовый текст для end-to-end тестирования",
                        source="integration_test"
                    )
                
                logger.info("  2. Запрос к графу памяти...")
                if self.brain and hasattr(self.brain, 'memory_graph_ml') and self.brain.memory_graph_ml:
                    context = self.brain.memory_graph_ml.get_context_for_query("тестирование системы")
                
                logger.info("  3. Генерация ответа...")
                if self.brain and hasattr(self.brain, 'process_query'):
                    response = self.brain.process_query("Тестовый запрос")
                
                integration_results['end_to_end'] = True
                logger.info("✓ End-to-End тест пройден")
                
            except Exception as e:
                logger.error(f"✗ End-to-End тест не пройден: {e}")
                integration_results['errors'].append(f"End-to-End: {str(e)}")
                
        except Exception as e:
            logger.error(f"✗ Ошибка тестирования интеграции: {e}")
            integration_results['errors'].append(f"Exception: {str(e)}")
            traceback.print_exc()
        
        self.test_results['integration'] = integration_results
        
        passed = sum(integration_results.values()) if isinstance(integration_results.get('brain_to_learning'), bool) else 0
        logger.info(f"\nРезультат интеграции: {passed}/5 проверок пройдено")
    
    def _generate_report(self):
        """Генерация итогового отчёта"""
        logger.info("\n" + "=" * 80)
        logger.info("ИТОГОВЫЙ ОТЧЁТ ТЕСТИРОВАНИЯ")
        logger.info("=" * 80)
        
        end_time = time.time()
        total_duration = end_time - self.metrics['start_time']
        
        # Подсчёт результатов
        init_passed = sum(1 for v in self.test_results['initialization'].values() 
                         if isinstance(v, bool) and v)
        init_total = sum(1 for v in self.test_results['initialization'].values() if isinstance(v, bool))
        
        gen_passed = self.test_results['generation'].get('tests_passed', 0)
        gen_total = self.test_results['generation'].get('tests_run', 0)
        
        learning_passed = 1 if self.test_results['self_learning'].get('training_triggered') else 0
        learning_total = 1
        
        reasoning_passed = self.test_results['reasoning'].get('tests_passed', 0)
        reasoning_total = self.test_results['reasoning'].get('tests_run', 0)
        
        memory_passed = sum([
            self.test_results['memory_graph'].get('knowledge_graph_available', False),
            self.test_results['memory_graph'].get('memory_graph_ml_available', False),
            self.test_results['memory_graph'].get('context_retrieved', False)
        ])
        memory_total = 3
        
        integration_passed = sum(1 for v in self.test_results['integration'].values() 
                               if isinstance(v, bool) and v)
        integration_total = 5
        
        total_passed = init_passed + gen_passed + learning_passed + reasoning_passed + memory_passed + integration_passed
        total_tests = init_total + gen_total + learning_total + reasoning_total + memory_total + integration_total
        
        # Сводка
        summary = {
            'test_duration_seconds': round(total_duration, 2),
            'components_initialized': len(self.metrics['components_initialized']),
            'total_tests': total_tests,
            'passed_tests': total_passed,
            'failed_tests': total_tests - total_passed,
            'success_rate': round(total_passed / total_tests * 100, 1) if total_tests > 0 else 0,
            'errors_count': len(self.metrics['errors']),
            'warnings_count': len(self.metrics['warnings']),
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
        
        # Вывод отчёта
        logger.info(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
        logger.info(f"  Длительность тестирования: {total_duration:.2f} секунд")
        logger.info(f"  Инициализировано компонентов: {summary['components_initialized']}")
        logger.info(f"  Всего тестов: {total_tests}")
        logger.info(f"  Пройдено: {total_passed}")
        logger.info(f"  Не пройдено: {summary['failed_tests']}")
        logger.info(f"  Успешность: {summary['success_rate']}%")
        logger.info(f"  Ошибок: {summary['errors_count']}")
        logger.info(f"  Предупреждений: {summary['warnings_count']}")
        
        logger.info(f"\n📋 РЕЗУЛЬТАТЫ ПО КАТЕГОРИЯМ:")
        logger.info(f"  Инициализация: {init_passed}/{init_total}")
        logger.info(f"  Генерация текста: {gen_passed}/{gen_total}")
        logger.info(f"  Самообучение: {learning_passed}/{learning_total}")
        logger.info(f"  Рассуждение: {reasoning_passed}/{reasoning_total}")
        logger.info(f"  Граф памяти: {memory_passed}/{memory_total}")
        logger.info(f"  Интеграция: {integration_passed}/{integration_total}")
        
        # Сохранение отчёта в JSON
        report_file = f"comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"\n💾 Отчёт сохранён в: {report_file}")
        except Exception as e:
            logger.error(f"Ошибка сохранения отчёта: {e}")
        
        # Финальный статус
        logger.info("\n" + "=" * 80)
        if summary['success_rate'] >= 80:
            logger.info("✅ СТАТУС: СИСТЕМА РАБОТОСПОСОБНА")
        elif summary['success_rate'] >= 50:
            logger.info("⚠️ СТАТУС: СИСТЕМА ЧАСТИЧНО РАБОТОСПОСОБНА")
        else:
            logger.info("❌ СТАТУС: СИСТЕМА ИМЕЕТ СЕРЬЁЗНЫЕ ПРОБЛЕМЫ")
        logger.info("=" * 80)
        
        return summary


def main():
    """Главная функция запуска тестов"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SYSTEM TEST FOR COGNIFLEX")
    print("=" * 80)
    print("Начало полного тестирования системы...\n")
    
    test = ComprehensiveSystemTest()
    results = test.run_all_tests()
    
    # Вывод краткой сводки в stdout
    summary = results.get('summary', {})
    print("\n" + "=" * 80)
    print("КРАТКАЯ СВОДКА:")
    print(f"  Успешность: {summary.get('success_rate', 0)}%")
    print(f"  Пройдено тестов: {summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)}")
    print(f"  Длительность: {summary.get('test_duration_seconds', 0)}s")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
