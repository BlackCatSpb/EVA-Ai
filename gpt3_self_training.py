"""
Скрипт самообучения модели до уровня GPT3 с автозапуском
"""
import sys
import os
import time
import json
import logging
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime
sys.path.append('.')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gpt3_training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("gpt3_trainer")

class GPT3TrainingOrchestrator:
    """Оркестратор самообучения до уровня GPT3"""
    
    def __init__(self):
        """Инициализация оркестратора"""
        
        # Целевые показатели GPT3
        self.gpt3_targets = {
            "model_size": {
                "parameters": 175_000_000_000,  # 175B параметров
                "vocab_size": 50_257,           # Размер словаря
                "context_length": 2048,         # Длина контекста
                "layers": 96,                  # Количество слоев
                "heads": 96,                   # Количество голов внимания
                "d_model": 12288               # Размер модели
            },
            "performance": {
                "perplexity": 20.0,            # Целевая перплексия
                "quality_score": 0.85,         # Качество генерации
                "coherence": 0.90,             # Когерентность
                "diversity": 0.85,             # Разнообразие
                "grammar": 0.95                 # Грамматика
            },
            "data": {
                "training_tokens": 300_000_000_000,  # 300B токенов
                "training_texts": 10_000_000,        # 10M текстов
                "web_sources": 1_000_000,           # 1M веб-источников
                "knowledge_domains": 1000           # 1000 доменов знаний
            },
            "capabilities": {
                "reasoning": 0.80,              # Способность к рассуждениям
                "creativity": 0.75,              # Творческие способности
                "knowledge": 0.90,               # Общие знания
                "language_understanding": 0.85,   # Понимание языка
                "context_retention": 0.80        # Удержание контекста
            }
        }
        
        # Текущие показатели
        self.current_metrics = {
            "model_size": {
                "parameters": 0,
                "vocab_size": 0,
                "context_length": 0,
                "layers": 0,
                "heads": 0,
                "d_model": 0
            },
            "performance": {
                "perplexity": 100.0,
                "quality_score": 0.0,
                "coherence": 0.0,
                "diversity": 0.0,
                "grammar": 0.0
            },
            "data": {
                "training_tokens": 0,
                "training_texts": 0,
                "web_sources": 0,
                "knowledge_domains": 0
            },
            "capabilities": {
                "reasoning": 0.0,
                "creativity": 0.0,
                "knowledge": 0.0,
                "language_understanding": 0.0,
                "context_retention": 0.0
            }
        }
        
        # Статистика обучения
        self.training_stats = {
            "total_sessions": 0,
            "successful_sessions": 0,
            "total_training_time": 0.0,
            "total_web_searches": 0,
            "total_knowledge_extracted": 0,
            "quality_improvements": [],
            "start_time": time.time(),
            "current_session": None,
            "auto_training_active": False
        }
        
        # Пороги для автозапуска
        self.auto_thresholds = {
            "quality_drop_threshold": 0.1,      # Порог падения качества
            "performance_degradation": 0.15,    # Деградация производительности
            "data_stagnation_hours": 6,        # Часы без новых данных
            "min_improvement_rate": 0.01,      # Минимальная скорость улучшения
            "max_session_duration": 3600,      # Макс. длительность сессии (сек)
            "min_training_texts": 100          # Мин. текстов для обучения
        }
        
        # Менеджер
        self.manager = None
        self.training_thread = None
        self.monitoring_thread = None
        
        # Прогресс
        self.progress = {
            "overall_completion": 0.0,
            "model_size_completion": 0.0,
            "performance_completion": 0.0,
            "data_completion": 0.0,
            "capabilities_completion": 0.0,
            "estimated_completion_time": None,
            "current_phase": "initialization"
        }
        
        logger.info("GPT3TrainingOrchestrator инициализирован")
    
    def initialize_manager(self):
        """Инициализирует менеджер модели"""
        
        try:
            from cogniflex.mlearning.unified_fractal_manager import UnifiedFractalManager
            
            self.manager = UnifiedFractalManager()
            
            logger.info(f"Менеджер инициализирован: {type(self.manager.manager).__name__}")
            logger.info(f"Оптимизирован: {self.manager.is_optimized}")
            
            # Собираем текущие метрики
            self._collect_current_metrics()
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации менеджера: {e}")
            return False
    
    def _collect_current_metrics(self):
        """Собирает текущие метрики модели"""
        
        try:
            # Размер модели
            if hasattr(self.manager.manager, 'model'):
                model = self.manager.manager.model
                if hasattr(model, 'num_parameters'):
                    self.current_metrics["model_size"]["parameters"] = model.num_parameters()
                
                if hasattr(model, 'config'):
                    config = model.config
                    self.current_metrics["model_size"]["vocab_size"] = getattr(config, 'vocab_size', 0)
                    self.current_metrics["model_size"]["context_length"] = getattr(config, 'n_positions', 0)
                    self.current_metrics["model_size"]["layers"] = getattr(config, 'n_layer', 0)
                    self.current_metrics["model_size"]["heads"] = getattr(config, 'n_head', 0)
                    self.current_metrics["model_size"]["d_model"] = getattr(config, 'n_embd', 0)
            
            # Производительность
            quality_metrics = self.manager.get_quality_metrics()
            self.current_metrics["performance"].update(quality_metrics)
            
            # Данные
            if hasattr(self.manager, 'enhanced_learning') and self.manager.enhanced_learning:
                status = self.manager.get_enhanced_system_status()
                stats = status.get('statistics', {})
                self.current_metrics["data"]["training_texts"] = stats.get('total_training_texts', 0)
                self.current_metrics["data"]["total_web_searches"] = stats.get('total_web_searches', 0)
                self.current_metrics["data"]["total_knowledge_extracted"] = stats.get('total_knowledge_extracted', 0)
            
            logger.info("Текущие метрики собраны")
            
        except Exception as e:
            logger.error(f"Ошибка сбора метрик: {e}")
    
    def calculate_progress(self):
        """Рассчитывает прогресс обучения"""
        
        try:
            # Прогресс по размеру модели
            param_progress = min(1.0, self.current_metrics["model_size"]["parameters"] / self.gpt3_targets["model_size"]["parameters"])
            vocab_progress = min(1.0, self.current_metrics["model_size"]["vocab_size"] / self.gpt3_targets["model_size"]["vocab_size"])
            self.progress["model_size_completion"] = (param_progress + vocab_progress) / 2
            
            # Прогресс по производительности
            perf_metrics = self.current_metrics["performance"]
            target_perf = self.gpt3_targets["performance"]
            
            perf_progress = 0
            count = 0
            for metric, target_value in target_perf.items():
                if metric in perf_metrics and perf_metrics[metric] > 0:
                    progress = min(1.0, perf_metrics[metric] / target_value)
                    perf_progress += progress
                    count += 1
            
            self.progress["performance_completion"] = perf_progress / count if count > 0 else 0
            
            # Прогресс по данным
            data_metrics = self.current_metrics["data"]
            target_data = self.gpt3_targets["data"]
            
            data_progress = 0
            count = 0
            for metric, target_value in target_data.items():
                if metric in data_metrics and data_metrics[metric] > 0:
                    progress = min(1.0, data_metrics[metric] / target_value)
                    data_progress += progress
                    count += 1
            
            self.progress["data_completion"] = data_progress / count if count > 0 else 0
            
            # Общий прогресс
            self.progress["overall_completion"] = (
                self.progress["model_size_completion"] * 0.2 +
                self.progress["performance_completion"] * 0.4 +
                self.progress["data_completion"] * 0.4
            )
            
            # Оценка времени завершения
            if self.progress["overall_completion"] > 0:
                elapsed_time = time.time() - self.training_stats["start_time"]
                estimated_total = elapsed_time / self.progress["overall_completion"]
                self.progress["estimated_completion_time"] = self.training_stats["start_time"] + estimated_total
            
            logger.info(f"Прогресс: {self.progress['overall_completion']:.2%}")
            
        except Exception as e:
            logger.error(f"Ошибка расчета прогресса: {e}")
    
    def should_start_auto_training(self) -> bool:
        """Определяет, нужно ли запустить авто-обучение"""
        
        try:
            # Проверяем пороги качества
            quality_score = self.current_metrics["performance"]["quality_score"]
            if quality_score < (1.0 - self.auto_thresholds["quality_drop_threshold"]):
                logger.info(f"Качество упало до {quality_score:.3f}, запускаем авто-обучение")
                return True
            
            # Проверяем деградацию производительности
            if hasattr(self.manager, 'get_performance_stats'):
                perf_stats = self.manager.get_performance_stats()
                cache_hit_rate = perf_stats.get('cache_hit_rate', 0)
                if cache_hit_rate < (1.0 - self.auto_thresholds["performance_degradation"]):
                    logger.info(f"Производительность упала до {cache_hit_rate:.2%}, запускаем авто-обучение")
                    return True
            
            # Проверяем stagnation данных
            if self.training_stats["total_web_searches"] == 0:
                logger.info("Нет данных от веб-поиска, запускаем авто-обучение")
                return True
            
            # Проверяем улучшение качества
            if len(self.training_stats["quality_improvements"]) > 5:
                recent_improvements = self.training_stats["quality_improvements"][-5:]
                avg_improvement = sum(recent_improvements) / len(recent_improvements)
                if avg_improvement < self.auto_thresholds["min_improvement_rate"]:
                    logger.info(f"Улучшение качества замедлилось до {avg_improvement:.3f}, запускаем авто-обучение")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка проверки авто-обучения: {e}")
            return False
    
    def start_auto_training(self):
        """Запускает авто-обучение"""
        
        if self.training_stats["auto_training_active"]:
            logger.warning("Авто-обучение уже активно")
            return
        
        self.training_stats["auto_training_active"] = True
        self.training_thread = threading.Thread(target=self._auto_training_loop, daemon=True)
        self.training_thread.start()
        
        logger.info("Авто-обучение запущено")
    
    def _auto_training_loop(self):
        """Основной цикл авто-обучения"""
        
        while self.training_stats["auto_training_active"]:
            try:
                # Проверяем необходимость обучения
                if self.should_start_auto_training():
                    self._execute_training_session()
                
                # Обновляем метрики
                self._collect_current_metrics()
                self.calculate_progress()
                
                # Сохраняем прогресс
                self._save_progress()
                
                # Проверяем завершение
                if self.progress["overall_completion"] >= 0.95:  # 95% завершения
                    logger.info("Достигнут целевой уровень GPT3!")
                    self.training_stats["auto_training_active"] = False
                    break
                
                # Пауза между проверками
                time.sleep(300)  # 5 минут
                
            except Exception as e:
                logger.error(f"Ошибка в цикле авто-обучения: {e}")
                time.sleep(60)  # 1 минута при ошибке
    
    def _execute_training_session(self):
        """Выполняет сессию обучения"""
        
        try:
            session_start = time.time()
            session_id = f"auto_session_{int(session_start)}"
            
            logger.info(f"Запуск сессии авто-обучения: {session_id}")
            
            # Определяем темы для обучения
            learning_topics = self._select_learning_topics()
            
            # Запускаем сессию обучения
            if hasattr(self.manager, 'start_enhanced_learning_session'):
                self.training_stats["current_session"] = self.manager.start_enhanced_learning_session(
                    topics=learning_topics,
                    session_name=session_id
                )
                
                self.training_stats["total_sessions"] += 1
                
                # Ожидаем завершения с таймаутом
                session_timeout = self.auto_thresholds["max_session_duration"]
                session_start_time = time.time()
                
                while time.time() - session_start_time < session_timeout:
                    # Проверяем статус сессии
                    status = self.manager.get_enhanced_system_status()
                    
                    if 'sessions' in status and session_id in status['sessions']:
                        session_info = status['sessions'][session_id]
                        
                        if session_info['status'] == 'completed':
                            logger.info(f"Сессия {session_id} завершена успешно")
                            self.training_stats["successful_sessions"] += 1
                            
                            # Обновляем статистику
                            self._update_training_stats(session_info)
                            break
                        elif session_info['status'] == 'failed':
                            logger.warning(f"Сессия {session_id} завершилась с ошибкой")
                            break
                    
                    time.sleep(30)  # Проверяем каждые 30 секунд
                
                # Принудительно завершаем если нужно
                if self.training_stats["current_session"]:
                    logger.info(f"Завершение сессии {session_id} по таймауту")
            
            session_duration = time.time() - session_start
            self.training_stats["total_training_time"] += session_duration
            
            logger.info(f"Сессия {session_id} завершена за {session_duration:.1f}s")
            
        except Exception as e:
            logger.error(f"Ошибка выполнения сессии обучения: {e}")
    
    def _select_learning_topics(self) -> List[str]:
        """Выбирает темы для обучения"""
        
        # Базовые темы
        core_topics = [
            "машинное обучение",
            "нейронные сети",
            "искусственный интеллект",
            "глубокое обучение",
            "трансформеры",
            "NLP",
            "компьютерное зрение",
            "рекуррентные сети",
            "LSTM",
            "attention механизм"
        ]
        
        # Продвинутые темы
        advanced_topics = [
            "GPT модели",
            "BERT",
            "квантовые вычисления",
            "обучение с подкреплением",
            "transfer learning",
            "fine-tuning",
            "оптимизация гиперпараметров",
            "свёрточные сети",
            "генеративные модели",
            "мультимодальное обучение"
        ]
        
        # Специализированные темы
        specialized_topics = [
            "фрактальные нейронные сети",
            "нейроморфные вычисления",
            "квантовые нейронные сети",
            "графовые нейронные сети",
            "трансформеры с фрактальной архитектурой",
            "самоорганизующиеся системы",
            "адаптивное обучение",
            "мета-обучение",
            "обучение с небольшим количеством данных",
            "объяснимый ИИ"
        ]
        
        # Выбираем темы на основе прогресса
        if self.progress["overall_completion"] < 0.3:
            return core_topics[:5]
        elif self.progress["overall_completion"] < 0.6:
            return core_topics + advanced_topics[:5]
        elif self.progress["overall_completion"] < 0.8:
            return advanced_topics + specialized_topics[:5]
        else:
            return specialized_topics
    
    def _update_training_stats(self, session_info: Dict[str, Any]):
        """Обновляет статистику обучения"""
        
        try:
            # Обновляем данные
            self.current_metrics["data"]["training_texts"] += session_info.get('training_texts', 0)
            self.current_metrics["data"]["total_web_searches"] += session_info.get('web_searches', 0)
            self.current_metrics["data"]["total_knowledge_extracted"] += session_info.get('knowledge_extracted', 0)
            
            # Обновляем качество
            quality_improvement = session_info.get('quality_after', 0) - session_info.get('quality_before', 0)
            self.training_stats["quality_improvements"].append(quality_improvement)
            
            # Ограничиваем историю улучшений
            if len(self.training_stats["quality_improvements"]) > 100:
                self.training_stats["quality_improvements"] = self.training_stats["quality_improvements"][-50:]
            
            logger.info(f"Статистика обновлена: улучшение качества {quality_improvement:+.3f}")
            
        except Exception as e:
            logger.error(f"Ошибка обновления статистики: {e}")
    
    def _save_progress(self):
        """Сохраняет прогресс обучения"""
        
        try:
            progress_data = {
                "progress": self.progress,
                "current_metrics": self.current_metrics,
                "training_stats": self.training_stats,
                "timestamp": time.time(),
                "gpt3_targets": self.gpt3_targets
            }
            
            with open('gpt3_training_progress.json', 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Ошибка сохранения прогресса: {e}")
    
    def run_generation_tests(self) -> Dict[str, Any]:
        """Запускает тесты генерации"""
        
        test_results = {
            "status": "running",
            "tests": [],
            "overall_score": 0.0,
            "gpt3_comparison": {}
        }
        
        try:
            # Тестовые запросы разной сложности
            test_queries = [
                # Простые
                ("Что такое машинное обучение?", "simple"),
                ("Как работает нейронная сеть?", "simple"),
                ("Объясни концепцию искусственного интеллекта", "simple"),
                
                # Средние
                ("Сравни преимущества и недостатки различных архитектур трансформеров", "medium"),
                ("Как квантовые вычисления могут повлиять на будущее ИИ?", "medium"),
                ("Опиши процесс обучения с подкреплением на конкретном примере", "medium"),
                
                # Сложные
                ("Проанализируй взаимосвязь между фрактальной геометрией и архитектурой нейронных сетей", "complex"),
                ("Предложи методику создания самообучающейся ИИ-системы с минимальным человеческим вмешательством", "complex"),
                ("Оцени этические последствия создания сверхинтеллекта и предложи рамки регулирования", "complex")
            ]
            
            for query, complexity in test_queries:
                test_result = self._test_single_query(query, complexity)
                test_results["tests"].append(test_result)
            
            # Рассчитываем общий балл
            if test_results["tests"]:
                total_score = sum(test["score"] for test in test_results["tests"])
                test_results["overall_score"] = total_score / len(test_results["tests"])
            
            # Сравнение с GPT3
            test_results["gpt3_comparison"] = self._compare_with_gpt3(test_results)
            
            test_results["status"] = "completed"
            
        except Exception as e:
            logger.error(f"Ошибка тестирования генерации: {e}")
            test_results["status"] = "error"
            test_results["error"] = str(e)
        
        return test_results
    
    def _test_single_query(self, query: str, complexity: str) -> Dict[str, Any]:
        """Тестирует один запрос"""
        
        try:
            start_time = time.time()
            
            # Генерируем ответ
            response = self.manager.generate_enhanced_response(
                query, 
                max_tokens=200, 
                use_web_search=True
            )
            
            generation_time = time.time() - start_time
            
            # Оцениваем качество
            quality_score = self._evaluate_response_quality(response.get('response', ''), query)
            
            return {
                "query": query,
                "complexity": complexity,
                "response": response.get('response', ''),
                "generation_time": generation_time,
                "web_search_used": response.get('web_search_used', False),
                "search_results": len(response.get('search_results', [])),
                "score": quality_score,
                "metrics": response.get('quality_metrics', {})
            }
            
        except Exception as e:
            logger.error(f"Ошибка тестирования запроса '{query}': {e}")
            return {
                "query": query,
                "complexity": complexity,
                "error": str(e),
                "score": 0.0
            }
    
    def _evaluate_response_quality(self, response: str, query: str) -> float:
        """Оценивает качество ответа"""
        
        try:
            score = 0.0
            
            # Длина ответа
            if len(response) > 50:
                score += 0.1
            elif len(response) > 100:
                score += 0.1
            
            # Когерентность (простая проверка)
            sentences = response.split('.')
            if len(sentences) > 2:
                score += 0.1
            
            # Релевантность (проверка ключевых слов)
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            overlap = len(query_words & response_words)
            if overlap > 0:
                score += min(0.3, overlap / len(query_words))
            
            # Грамматика (простая проверка)
            if response.count('?') <= 1 and response.count('!') <= 1:
                score += 0.1
            
            # Структура
            if any(word in response.lower() for word in ['потому что', 'так как', 'например', 'во-первых']):
                score += 0.1
            
            # Информативность
            if len(response) > 200:
                score += 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Ошибка оценки качества: {e}")
            return 0.0
    
    def _compare_with_gpt3(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Сравнивает с характеристиками GPT3"""
        
        comparison = {
            "gpt3_equivalent": False,
            "score_difference": 0.0,
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }
        
        try:
            overall_score = test_results["overall_score"]
            
            # GPT3 ориентировочный балл
            gpt3_score = 0.85
            
            comparison["score_difference"] = overall_score - gpt3_score
            comparison["gpt3_equivalent"] = overall_score >= gpt3_score
            
            if overall_score >= 0.8:
                comparison["strengths"].append("Высокое качество генерации")
            if overall_score >= 0.7:
                comparison["strengths"].append("Хорошая когерентность")
            if overall_score >= 0.6:
                comparison["strengths"].append("Базовая функциональность")
            
            if overall_score < 0.5:
                comparison["weaknesses"].append("Низкое качество генерации")
            if overall_score < 0.6:
                comparison["weaknesses"].append("Проблемы с когерентностью")
            if overall_score < 0.7:
                comparison["weaknesses"].append("Ограниченные возможности")
            
            # Рекомендации
            if overall_score < 0.8:
                comparison["recommendations"].append("Продолжить обучение модели")
            if overall_score < 0.6:
                comparison["recommendations"].append("Увеличить объем обучающих данных")
            if overall_score < 0.5:
                comparison["recommendations"].append("Пересмотреть архитектуру модели")
            
        except Exception as e:
            logger.error(f"Ошибка сравнения с GPT3: {e}")
        
        return comparison
    
    def start_monitoring(self):
        """Запускает мониторинг прогресса"""
        
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Мониторинг прогресса запущен")
    
    def _monitoring_loop(self):
        """Цикл мониторинга"""
        
        while True:
            try:
                # Обновляем метрики
                self._collect_current_metrics()
                self.calculate_progress()
                
                # Сохраняем прогресс
                self._save_progress()
                
                # Выводим статус
                self._print_status()
                
                # Проверяем завершение
                if self.progress["overall_completion"] >= 0.95:
                    logger.info("🎉 ДОСТИГНУТ УРОВЕНЬ GPT3!")
                    self._generate_final_report()
                    break
                
                time.sleep(600)  # 10 минут
                
            except Exception as e:
                logger.error(f"Ошибка в цикле мониторинга: {e}")
                time.sleep(60)
    
    def _print_status(self):
        """Выводит статус обучения"""
        
        try:
            print("\n" + "="*80)
            print(f"🚀 GPT3 TRAINING STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            
            print(f"📊 Overall Progress: {self.progress['overall_completion']:.2%}")
            print(f"🤖 Model Size: {self.progress['model_size_completion']:.2%}")
            print(f"⚡ Performance: {self.progress['performance_completion']:.2%}")
            print(f"📚 Data: {self.progress['data_completion']:.2%}")
            
            print(f"\n📈 Current Metrics:")
            print(f"  Quality Score: {self.current_metrics['performance']['quality_score']:.3f}")
            print(f"  Training Texts: {self.current_metrics['data']['training_texts']:,}")
            print(f"  Web Searches: {self.current_metrics['data']['total_web_searches']:,}")
            
            print(f"\n⏱️ Training Stats:")
            print(f"  Sessions: {self.training_stats['total_sessions']}")
            print(f"  Successful: {self.training_stats['successful_sessions']}")
            print(f"  Training Time: {self.training_stats['total_training_time']:.1f}s")
            
            if self.progress["estimated_completion_time"]:
                eta = datetime.fromtimestamp(self.progress["estimated_completion_time"])
                print(f"  ETA: {eta.strftime('%Y-%m-%d %H:%M')}")
            
            print("="*80)
            
        except Exception as e:
            logger.error(f"Ошибка вывода статуса: {e}")
    
    def _generate_final_report(self):
        """Генерирует финальный отчет"""
        
        try:
            # Тесты генерации
            test_results = self.run_generation_tests()
            
            # Финальный отчет
            report = {
                "status": "completed",
                "completion_time": time.time(),
                "final_progress": self.progress,
                "final_metrics": self.current_metrics,
                "training_stats": self.training_stats,
                "test_results": test_results,
                "gpt3_achieved": test_results.get("gpt3_comparison", {}).get("gpt3_equivalent", False)
            }
            
            with open('gpt3_final_report.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info("🎉 Финальный отчет сохранен")
            
            # Выводим результаты
            print("\n" + "="*80)
            print("🎉 GPT3 TRAINING COMPLETED!")
            print("="*80)
            
            print(f"✅ Overall Progress: {self.progress['overall_completion']:.2%}")
            print(f"✅ Quality Score: {self.current_metrics['performance']['quality_score']:.3f}")
            print(f"✅ Training Texts: {self.current_metrics['data']['training_texts']:,}")
            print(f"✅ Total Sessions: {self.training_stats['total_sessions']}")
            print(f"✅ GPT3 Level: {'ACHIEVED' if report['gpt3_achieved'] else 'NOT ACHIEVED'}")
            
            print("\n📊 Test Results:")
            print(f"  Overall Score: {test_results.get('overall_score', 0):.3f}")
            print(f"  GPT3 Equivalent: {test_results.get('gpt3_comparison', {}).get('gpt3_equivalent', False)}")
            
            print("\n📁 Reports saved:")
            print("  - gpt3_training_progress.json")
            print("  - gpt3_final_report.json")
            print("  - gpt3_training.log")
            
            print("="*80)
            
        except Exception as e:
            logger.error(f"Ошибка генерации финального отчета: {e}")
    
    def run(self):
        """Запускает полный процесс обучения"""
        
        print("🚀 GPT3 SELF-TRAINING ORCHESTRATOR")
        print("="*80)
        print("Цель: Обучить модель до уровня GPT3 с автозапуском")
        print("="*80)
        
        # Инициализация
        if not self.initialize_manager():
            print("❌ Ошибка инициализации менеджера")
            return False
        
        # Запуск авто-обучения
        self.start_auto_training()
        
        # Запуск мониторинга
        self.start_monitoring()
        
        print("✅ Процесс обучения запущен")
        print("📊 Мониторинг активен")
        print("🔄 Авто-обучение будет запускаться по необходимости")
        print("📝 Прогресс сохраняется в gpt3_training_progress.json")
        print("\nНажмите Ctrl+C для остановки...")
        
        try:
            # Бесконечный цикл для поддержания работы
            while self.training_stats["auto_training_active"]:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n🛑 Остановка по запросу пользователя")
            self.training_stats["auto_training_active"] = False
        
        print("✅ Процесс обучения остановлен")
        return True

def main():
    """Основная функция"""
    
    orchestrator = GPT3TrainingOrchestrator()
    return orchestrator.run()

if __name__ == "__main__":
    main()
