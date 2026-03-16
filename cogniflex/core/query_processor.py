"""Модуль обработки запросов для CogniFlex"""
import time
import logging
import hashlib
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import torch  # для демо-интеграции пула инференса
    TORCH_AVAILABLE = True
except (ImportError, ModuleNotFoundError, RuntimeError):
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

# Настройка логгера для этого модуля
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)


class QueryProcessor:
    """Обрабатывает пользовательские запросы через конвейер обработки."""

    def __init__(self, brain):
        """Инициализирует процессор запросов.

        Args:
            brain: Ссылка на ядро системы
        """
        self.brain = brain
        # Инициализируем ссылки на общий гибридный кэш и общий executor
        self.hybrid_cache = None
        self.executor: Optional[ThreadPoolExecutor] = None
        self._own_executor = False

        try:
            # Получаем гибридный кэш из MemoryManager, затем из текстового процессора
            if getattr(self.brain, "memory_manager", None) and hasattr(self.brain.memory_manager, "get_hybrid_cache"):
                self.hybrid_cache = self.brain.memory_manager.get_hybrid_cache()
            if not self.hybrid_cache and getattr(self.brain, "text_processor", None):
                self.hybrid_cache = getattr(self.brain.text_processor, "hybrid_cache", None)
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"Ошибка инициализации hybrid_cache: {e}")

        try:
            # Переиспользуем executor из текстового процессора, если есть
            if getattr(self.brain, "text_processor", None) and getattr(self.brain.text_processor, "executor", None):
                self.executor = self.brain.text_processor.executor
            # Иначе создаём собственный пул
            if self.executor is None:
                self.executor = ThreadPoolExecutor(max_workers=4)
                self._own_executor = True
        except (AttributeError, TypeError, RuntimeError, OSError) as e:
            logger.debug(f"Ошибка инициализации executor: {e}")

    def process_query(self, query: str, user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Обрабатывает пользовательский запрос через конвейер обработки.

        Args:
            query: Текст запроса пользователя
            user_context: Контекст пользователя

        Returns:
            Dict[str, Any]: Структурированный ответ с результатами
        """
        start_time = time.time()
        result: Dict[str, Any] = {
            "response": None,
            "source": "none",
            "evidence": [],
            "metrics": {},
            "error": None,
            "contradictions": []
        }

        try:
            # 1) NLP preprocessing (с кэшированием)
            nlp_info = self._process_nlp(query)

            # 2) Извлечение концептов
            concept = self._extract_concept(query)

            # 3) Поиск в графе знаний
            nodes = self._search_knowledge_graph(query)
            if nodes:
                # Используем найденные узлы как основное доказательство и генерируем ответ через модель,
                # при необходимости дополняя веб-поиском
                evidence: List[Dict[str, Any]] = []
                try:
                    for n in nodes:
                        if hasattr(n, "__dict__"):
                            evidence.append({**getattr(n, "__dict__", {}), "_repr": str(n)})
                        else:
                            evidence.append({"repr": str(n)})
                except (AttributeError, TypeError, ValueError) as e:
                    logger.debug(f"Ошибка подготовки доказательств из узлов KG: {e}")

                # Опциональное дополнение веб-поиском при включённой конфигурации
                try:
                    augment_web = bool(getattr(self.brain, "config", {}).get("augment_with_web_on_kg", True))
                except (AttributeError, TypeError, ValueError) as e:
                    logger.debug(f"Ошибка получения конфигурации веб-дополнения: {e}")
                    augment_web = True

                if augment_web and self.brain.components.get("web_search_engine"):
                    try:
                        web_results = self.brain.components["web_search_engine"].search(query, max_results=3)
                        if isinstance(web_results, list):
                            evidence.extend(web_results)
                        elif web_results is not None:
                            evidence.append(web_results)
                    except Exception as e:
                        logger.warning(f"Ошибка веб-дополнения при наличии KG: {e}")

                # Генерируем ответ через ML-модуль, если доступен; иначе — краткий ответ из KG
                response = None
                if self.brain.components.get("ml_unit"):
                    response = self._generate_response(query, evidence, nlp_info, concept, user_context)

                if response:
                    result["response"] = response
                    result["source"] = "knowledge_graph+ml_unit"
                    result["evidence"] = evidence
                    result["metrics"] = {"time": time.time() - start_time}
                else:
                    # Фолбэк: короткий ответ из узлов KG
                    result = self._build_response_from_nodes(result, nodes, start_time)

                # Обновляем метрики (если есть менеджер метрик)
                try:
                    if hasattr(self.brain, "metrics_manager") and self.brain.metrics_manager:
                        self.brain.metrics_manager.update_request_metrics(time.time() - start_time, True)
                except (AttributeError, TypeError, ValueError) as e:
                    logger.debug(f"Ошибка обновления метрик после пути KG->генерация: {e}")

                return result

            # 4) Параллельный поиск в памяти и вебе (с кэшированием)
            evidence = self._parallel_search(query)
            result["evidence"] = evidence

            # 5) Генерация ответа
            response = self._generate_response(query, evidence, nlp_info, concept, user_context)
            result["response"] = response
            result["source"] = "ml_unit" if response else "none"

            # 6) Этическая проверка
            if response:
                ethics_result = self._check_ethics(response, nlp_info, user_context)
                result["ethics"] = ethics_result

            # 7) Проверка на противоречия
            contradictions = self._check_contradictions(query, response)
            if contradictions:
                result["contradictions"] = contradictions
                result["contradiction_detected"] = True

            # 8) Обновление метрик
            processing_time = time.time() - start_time
            # Собираем макроблок-метрики, если ядро их предоставляет
            macroblocks_stats = {}
            try:
                if hasattr(self.brain, "get_macroblocks_stats"):
                    logger.debug("[QueryProcessor] Collecting macroblocks stats from CoreBrain")
                    mb = self.brain.get_macroblocks_stats()
                    if isinstance(mb, dict):
                        macroblocks_stats = mb
                        try:
                            logger.debug("[QueryProcessor] Macroblocks stats modules=%s", list(mb.keys()))
                        except (AttributeError, TypeError) as e:
                            logger.debug(f"Ошибка логирования macroblocks stats: {e}")
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Ошибка получения macroblocks stats: {e}")
                macroblocks_stats = {}
            
            logger.debug("[QueryProcessor] Emitting metrics with macroblocks=%s", 
                        list(macroblocks_stats.keys()) if isinstance(macroblocks_stats, dict) else type(macroblocks_stats))
            result["metrics"] = {
                "time": processing_time,
                "nlp_info": nlp_info,
                "macroblocks": macroblocks_stats,
            }

            try:
                if hasattr(self.brain, "metrics_manager") and self.brain.metrics_manager:
                    self.brain.metrics_manager.update_request_metrics(processing_time, True)
                # Дополнительно эмитим нормализованные метрики
                self._emit_metrics([
                    {"name": "query_processor.requests_total", "component": "query_processor", "type": "counter", "value": 1.0, "labels": {"result": "success"}},
                    {"name": "query_processor.process_time_seconds", "component": "query_processor", "type": "summary", "value": float(processing_time)},
                ])
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Ошибка обновления метрик в конце process_query: {e}")

            return result

        except Exception as e:
            logger.exception("Критическая ошибка в process_query")
            try:
                if hasattr(self.brain, "metrics_manager") and self.brain.metrics_manager:
                    self.brain.metrics_manager.update_request_metrics(time.time() - start_time, False)
                self._emit_metrics([
                    {"name": "query_processor.requests_total", "component": "query_processor", "type": "counter", "value": 1.0, "labels": {"result": "error"}},
                ])
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Ошибка обновления метрик после исключения в process_query: {e}")
            result["error"] = str(e)
            result["response"] = f"Произошла ошибка: {e}"
            return result

    def _process_nlp(self, query: str) -> Dict[str, Any]:
        """Обрабатывает текст с помощью NLP.

        Args:
            query: Текст запроса

        Returns:
            Dict[str, Any]: Результаты NLP обработки
        """
        nlp_info = {
            "keywords": [],
            "entities": [],
            "intent": None,
            "sentiment": 0.0
        }

        try:
            cache_key = None
            if self.hybrid_cache:
                try:
                    cache_key = f"nlp:{hashlib.md5(query.encode('utf-8')).hexdigest()}"
                    cached = self.hybrid_cache.get(cache_key)
                    if isinstance(cached, dict) and cached.get("metadata", {}).get("processor"):
                        return cached
                except (AttributeError, TypeError, ValueError) as e:
                    logger.debug(f"Ошибка получения данных из кэша NLP: {e}")

            if self.brain.components.get("ml_unit"):
                try:
                    nlp_info = self.brain.components["ml_unit"].process_text(query)
                    # Сохраняем в кэш
                    if self.hybrid_cache and cache_key:
                        try:
                            self.hybrid_cache.set(cache_key, nlp_info)
                        except (AttributeError, TypeError, ValueError) as e:
                            logger.debug(f"Ошибка сохранения в кэш NLP: {e}")
                except Exception as e:
                    logger.warning(f"Ошибка NLP обработки: {e}")

            # Доп. демо-интеграция адаптера/пула инференса (опционально через конфиг)
            try:
                demo_enabled = bool(getattr(self.brain, "config", {}).get("nlp_demo_integration", False))
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Ошибка получения конфигурации nlp_demo_integration: {e}")
                demo_enabled = False
            
            if demo_enabled and torch is not None:
                try:
                    # Простейший item: в качестве input_ids используем длину запроса как диапазон [0..len-1]
                    length = max(1, len(query))
                    ids = torch.arange(length, dtype=torch.long)
                    item = {"input_ids": ids}
                    # Кладём в модуль "default"
                    if hasattr(self.brain, "nlp_enqueue"):
                        self.brain.nlp_enqueue(item, module="default")
                        # Форсируем выпуск оставшегося батча
                        self.brain.nlp_flush(module="default")
                        # Неблокирующий забор результатов (один раз для краткости)
                        res = self.brain.nlp_try_get_result(module="default", timeout_s=0.0)
                        if isinstance(res, dict):
                            # Сохраняем компактный след в метаданные nlp_info
                            try:
                                val = res.get("logits")
                                if hasattr(val, 'detach'):
                                    val = val.detach().cpu().flatten().tolist()[:3]
                                nlp_info.setdefault("pool_demo", {})["preview"] = val
                            except (AttributeError, TypeError, ValueError) as e:
                                logger.debug(f"Ошибка обработки результатов пула: {e}")
                                nlp_info.setdefault("pool_demo", {})["preview"] = "ok"
                except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                    logger.debug(f"Ошибка в демо-интеграции пула инференса: {e}")
                    # Демонстрационная часть не должна ломать основной путь
                    pass
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"Ошибка при попытке доступа к ml_unit в _process_nlp: {e}")

        return nlp_info

    def _extract_concept(self, query: str) -> Optional[str]:
        """Извлекает ключевой концепт из запроса.

        Args:
            query: Текст запроса

        Returns:
            Optional[str]: Извлеченный концепт или None
        """
        concept = None
        try:
            if self.brain.components.get("adaptation_manager"):
                try:
                    concept = self.brain.components["adaptation_manager"]._extract_concept_from_query(query)
                except Exception as e:
                    logger.warning(f"Ошибка извлечения концепта: {e}")
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"Ошибка при попытке доступа к adaptation_manager в _extract_concept: {e}")
        return concept

    def _search_knowledge_graph(self, query: str, limit: int = 3) -> List[Any]:
        """Выполняет безопасный поиск в графе знаний.

        Args:
            query: Запрос для поиска
            limit: Максимальное количество результатов

        Returns:
            List[Any]: Список найденных узлов
        """
        try:
            if not self.brain.components.get("knowledge_graph"):
                return []

            # Проверяем кэш
            cache_key = None
            if self.hybrid_cache:
                try:
                    cache_key = f"kg:{hashlib.md5((query + '|' + str(limit)).encode('utf-8')).hexdigest()}"
                    cached_nodes = self.hybrid_cache.get(cache_key)
                    if isinstance(cached_nodes, list) and cached_nodes:
                        return cached_nodes
                except (AttributeError, TypeError, ValueError, KeyError, IndexError) as e:
                    logger.debug(f"Ошибка получения данных из кэша KG: {e}")

            kg = self.brain.components["knowledge_graph"]

            # Проверяем доступные методы поиска
            for method_name in ['search_nodes', 'search', 'find_nodes', 'query_nodes']:
                if hasattr(kg, method_name):
                    try:
                        nodes = getattr(kg, method_name)(query, limit=limit)
                        if self.hybrid_cache and cache_key:
                            try:
                                self.hybrid_cache.set(cache_key, nodes)
                            except (AttributeError, TypeError, ValueError) as e:
                                logger.debug(f"Ошибка сохранения в кэш KG: {e}")
                        return nodes
                    except TypeError:
                        # Пытаемся вызвать без параметра limit
                        try:
                            nodes = getattr(kg, method_name)(query)
                            if self.hybrid_cache and cache_key:
                                try:
                                    self.hybrid_cache.set(cache_key, nodes)
                                except (AttributeError, TypeError, ValueError) as e:
                                    logger.debug(f"Ошибка кэширования узлов KG: {e}")
                            return nodes
                        except Exception as e:
                            logger.warning(f"Ошибка вызова {method_name}: {e}")
                            return []
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"Ошибка в _search_knowledge_graph: {e}")
        return []

    def _build_response_from_nodes(self, result: Dict, nodes: List[Any], start_time: float) -> Dict[str, Any]:
        """Строит ответ на основе найденных узлов графа знаний.

        Args:
            result: Текущий результат
            nodes: Найденные узлы
            start_time: Время начала обработки

        Returns:
            Dict[str, Any]: Обновленный результат
        """
        try:
            resp = "Найдены концепты:\n" + "\n".join(f"• {getattr(n, 'content', str(n))[:200]}" for n in nodes[:3])
            result.update({
                "response": resp,
                "source": "knowledge_graph",
                "evidence": [getattr(n, "__dict__", lambda: {"repr": str(n)})() for n in nodes],
                "metrics": {"time": time.time() - start_time}
            })
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.debug(f"Ошибка при сборке ответа из узлов графа: {e}")
        return result

    def _parallel_search(self, query: str) -> List[Dict[str, Any]]:
        """Выполняет параллельный поиск в памяти и вебе.

        Args:
            query: Текст запроса

        Returns:
            List[Dict[str, Any]]: Список результатов поиска
        """
        evidence: List[Dict[str, Any]] = []
        futures = []

        # Попытка получить из кэша заранее объединённые доказательства (если включено через конфиг)
        cache_key = None
        evidence_cache_enabled = True
        try:
            evidence_cache_enabled = bool(getattr(self.brain, "config", {}).get("evidence_cache_enabled", True))
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"Ошибка получения конфигурации evidence_cache_enabled: {e}")
            evidence_cache_enabled = True

        if self.hybrid_cache and evidence_cache_enabled:
            try:
                cache_key = f"evidence:{hashlib.md5(query.encode('utf-8')).hexdigest()}"
                cached = self.hybrid_cache.get(cache_key)
                if isinstance(cached, list) and cached:
                    return cached
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Ошибка получения кэша доказательств: {e}")

        try:
            exec_ref = self.executor
            if exec_ref is None:
                exec_ref = ThreadPoolExecutor(max_workers=2)
            
            # Поиск в памяти
            if self.brain.components.get("memory_manager") and hasattr(self.brain.components["memory_manager"], 'search'):
                try:
                    futures.append(exec_ref.submit(self.brain.components["memory_manager"].search, query))
                except (AttributeError, TypeError, RuntimeError) as e:
                    logger.debug(f"Ошибка добавления поиска memory_manager: {e}")

            # Веб-поиск
            if self.brain.components.get("web_search_engine"):
                try:
                    futures.append(exec_ref.submit(self.brain.components["web_search_engine"].search, query, max_results=3))
                except (AttributeError, TypeError, RuntimeError) as e:
                    logger.debug(f"Ошибка добавления веб-поиска: {e}")

            # Сбор результатов
            for future in as_completed(futures):
                try:
                    results = future.result()
                    # Нормализуем: если результат — список, расширяем, иначе добавляем как единицу доказательств
                    if isinstance(results, list):
                        evidence.extend(results)
                    elif results is not None:
                        evidence.append(results)
                except Exception as e:
                    logger.warning(f"Ошибка при асинхронном поиске: {e}")
        except (RuntimeError, TimeoutError, OSError) as e:
            logger.debug(f"Критическая ошибка в _parallel_search: {e}")

        # Сохраняем объединённые доказательства в кэш (если включено)
        if self.hybrid_cache and cache_key and evidence_cache_enabled:
            try:
                self.hybrid_cache.set(cache_key, evidence)
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Ошибка сохранения объединённых доказательств в кэш: {e}")

        return evidence

    def _generate_response(self, query: str, evidence: List, nlp_info: Dict, concept: Optional[str],
                          user_context: Optional[Dict]) -> Optional[str]:
        """Генерирует ответ на основе собранных данных.

        Args:
            query: Текст запроса
            evidence: Собранные данные
            nlp_info: Результаты NLP обработки
            concept: Извлеченный концепт
            user_context: Контекст пользователя

        Returns:
            Optional[str]: Сгенерированный ответ или None
        """
        try:
            # Используем унифицированный метод process_query
            if hasattr(self.brain, 'process_query'):
                gen_context = {
                    "nlp": nlp_info,
                    "evidence": evidence,
                    "concept": concept,
                    "user_context": user_context
                }
                
                result = self.brain.process_query(query, context=gen_context)
                if result and isinstance(result, dict):
                    return result.get('text', 'Ошибка обработки')
                elif isinstance(result, str):
                    return result
                else:
                    return "Ошибка обработки ответа"
            
            # Запасной вариант: используем старую логику, если GenerationCoordinator недоступен
            if not self.brain.components.get("ml_unit"):
                return "Извините, модуль генерации недоступен."
            
            # Формируем контекст для генерации
            gen_context = {
                "nlp": nlp_info,
                "evidence": evidence,
                "concept": concept,
                "user_context": user_context
            }

            # Генерируем ответ
            result = self.brain.process_query(query, context=gen_context)
            if result and isinstance(result, dict):
                return result.get('text', 'Ошибка обработки')
            elif isinstance(result, str):
                return result
            else:
                return "Ошибка обработки ответа"
                
        except Exception as e:
            logger.exception(f"Ошибка генерации ответа: {e}")
            return "Извините, произошла ошибка при генерации ответа. Пожалуйста, попробуйте еще раз."

    def _check_ethics(self, response: str, nlp_info: Dict, user_context: Optional[Dict]) -> Dict[str, Any]:
        """Проверяет ответ на соответствие этическим нормам.

        Args:
            response: Сгенерированный ответ
            nlp_info: Результаты NLP обработки
            user_context: Контекст пользователя

        Returns:
            Dict[str, Any]: Результаты этической проверки
        """
        ethics_result: Dict[str, Any] = {
            "score": 1.0,
            "violations": [],
            "recommendations": []
        }

        try:
            if self.brain.components.get("ethics_framework") and hasattr(self.brain.components["ethics_framework"], 'analyze_content'):
                try:
                    analysis = self.brain.components["ethics_framework"].analyze_content(
                        response,
                        context={"nlp": nlp_info, "user_context": user_context}
                    )
                    ethics_result = {
                        "score": getattr(analysis, "overall_score", 1.0),
                        "violations": getattr(analysis, "violations", []),
                        "recommendations": getattr(analysis, "recommendations", []),
                        "principle_scores": getattr(analysis, "principle_scores", {})
                    }
                except Exception as e:
                    logger.warning(f"Ошибка этической проверки: {e}")
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.debug(f"Ошибка при попытке доступа к ethics_framework в _check_ethics: {e}")

        return ethics_result

    def _check_contradictions(self, query: str, response: str) -> List[Dict[str, Any]]:
        """Проверяет ответ на наличие противоречий.

        Args:
            query: Текст запроса
            response: Сгенерированный ответ

        Returns:
            List[Dict[str, Any]]: Список обнаруженных противоречий
        """
        contradictions: List[Dict[str, Any]] = []

        try:
            if self.brain.components.get("contradiction_resolver"):
                try:
                    resolver = self.brain.components["contradiction_resolver"]
                    # Проверяем, есть ли метод для проверки конкретного ответа
                    if hasattr(resolver, "check_response_contradictions"):
                        contradictions = resolver.check_response_contradictions(query, response)
                    # Или проверяем все активные противоречия
                    elif hasattr(resolver, "get_active_contradictions"):
                        contradictions = resolver.get_active_contradictions()
                except Exception as e:
                    logger.warning(f"Ошибка проверки противоречий: {e}")
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.debug(f"Ошибка при попытке доступа к contradiction_resolver в _check_contradictions: {e}")

        return contradictions

    def _emit_metrics(self, metrics: List[Dict[str, Any]]):
        """Безопасная эмиссия метрик для QueryProcessor (через событийную шину и прямой вызов)."""
        try:
            if getattr(self, "brain", None):
                try:
                    if hasattr(self.brain, "events") and self.brain.events:
                        self.brain.events.trigger('metrics', metrics)
                except (AttributeError, TypeError, RuntimeError) as e:
                    logger.debug(f"Ошибка отправки метрик через события: {e}")
                try:
                    if hasattr(self.brain, "emit_metrics"):
                        self.brain.emit_metrics(metrics)
                except (AttributeError, TypeError, RuntimeError) as e:
                    logger.debug(f"Ошибка прямого вызова emit_metrics: {e}")
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"Ошибка в _emit_metrics: {e}")

    def __del__(self):
        try:
            if self._own_executor and self.executor:
                self.executor.shutdown(wait=False)
        except (AttributeError, TypeError, RuntimeError, OSError) as e:
            logger.debug(f"Ошибка при shutdown executor: {e}")