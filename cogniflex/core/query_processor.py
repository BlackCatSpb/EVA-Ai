"""Модуль обработки запросов для CogniFlex"""
import time
import logging
import hashlib
import re
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

try:
    import torch  # для демо-интеграции пула инференса
    TORCH_AVAILABLE = True
except (ImportError, ModuleNotFoundError, RuntimeError):
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    from cogniflex.knowledge.context_entity import EntityExtractor
    from cogniflex.knowledge.ambiguity_resolver import AmbiguityResolver
except ImportError as e:
    logger.warning(f"Failed to import EntityExtractor or AmbiguityResolver: {e}")
    EntityExtractor = None
    AmbiguityResolver = None


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
            if getattr(self.brain, "memory_manager", None) is not None and hasattr(self.brain.memory_manager, "get_hybrid_cache"):
                self.hybrid_cache = self.brain.memory_manager.get_hybrid_cache()
            if not self.hybrid_cache and getattr(self.brain, "text_processor", None) is not None:
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

        self.entity_extractor = EntityExtractor() if EntityExtractor else None
        self.ambiguity_resolver = AmbiguityResolver() if AmbiguityResolver else None

        # Model and tokenizer references (may be None if not loaded)
        self.model = None
        self.tokenizer = None

        # Embeddings cache with LRU size limit
        self.embeddings: OrderedDict = OrderedDict()
        self._embeddings_max_size = 1000

        # Current query for reasoning engine
        self.current_query = ""

        # State flags
        self.initialized = True
        self.running = True

        # Initialize model and tokenizer if available in brain
        self._initialize_model_components()
    
    def _initialize_model_components(self):
        """Initialize model and tokenizer from brain components."""
        try:
            if hasattr(self.brain, 'ml_unit') and self.brain.ml_unit:
                ml_unit = self.brain.ml_unit
                if hasattr(ml_unit, 'model'):
                    self.model = ml_unit.model
                    logger.debug("Model loaded into QueryProcessor")
                if hasattr(ml_unit, 'tokenizer'):
                    self.tokenizer = ml_unit.tokenizer
                    logger.debug("Tokenizer loaded into QueryProcessor")
        except Exception as e:
            logger.debug(f"Could not initialize model components: {e}")
    
    def _get_embedding(self, key: str) -> Optional[Any]:
        """Get embedding from cache with LRU eviction."""
        if key in self.embeddings:
            embedding = self.embeddings[key]
            if embedding is None:
                return None
            # Move to end (most recently used)
            self.embeddings.move_to_end(key)
            return embedding
        return None
    
    def _set_embedding(self, key: str, embedding: Any):
        """Set embedding in cache with LRU size limit."""
        if embedding is None:
            return
        # If key exists, update and move to end
        if key in self.embeddings:
            self.embeddings.move_to_end(key)
            self.embeddings[key] = embedding
            return
        # Evict oldest if at capacity
        if len(self.embeddings) >= self._embeddings_max_size:
            oldest_key = next(iter(self.embeddings))
            self.embeddings.pop(oldest_key, None)
            logger.debug(f"Evicted oldest embedding key: {oldest_key}")
        self.embeddings[key] = embedding

    def process_query(self, query: str, user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Обрабатывает пользовательский запрос через конвейер обработки.

        Args:
            query: Текст запроса пользователя
            user_context: Контекст пользователя

        Returns:
            Dict[str, Any]: Структурированный ответ с результатами
        """
        if not query or not query.strip():
            return {
                "response": "Пожалуйста, введите текст запроса.",
                "source": "none",
                "evidence": [],
                "metrics": {},
                "error": None,
                "contradictions": [],
                "ethics": {"score": 1.0, "violations": [], "recommendations": []},
                "ambiguities": None,
                "reasoning": "",
                "contradiction_detected": False
            }
        
        self.current_query = query
        start_time = time.time()
        result: Dict[str, Any] = {
            "response": None,
            "source": "none",
            "evidence": [],
            "metrics": {},
            "error": None,
            "contradictions": []
        }

        conversation_context = self._get_conversation_context()

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
                brain_config = getattr(self.brain, "config", {}) or {}
                augment_web = brain_config.get("augment_with_web_on_kg", True)

                if augment_web and self.brain.components.get("web_search_engine"):
                    try:
                        web_engine = self.brain.components["web_search_engine"]
                        if hasattr(web_engine, 'search'):
                            web_results = web_engine.search(query, max_results=3)
                        else:
                            web_results = None
                        if isinstance(web_results, list):
                            # Фильтруем web результаты - убираем мусор
                            for web_item in web_results:
                                if isinstance(web_item, dict):
                                    # Фильтруем snippet если есть
                                    if 'snippet' in web_item:
                                        snippet = web_item['snippet']
                                        # Убираем URL, HTML артефакты
                                        snippet = re.sub(r'https?://\S+', '', snippet)
                                        snippet = re.sub(r'<[^>]+>', '', snippet)
                                        snippet = re.sub(r'\s+', ' ', snippet).strip()
                                        # Проверяем минимальную длину
                                        if len(snippet) > 30:
                                            web_item['snippet'] = snippet[:200]  # Ограничиваем длину
                                        else:
                                            web_item['snippet'] = ''
                                    # Убираем весь result если он пустой
                                    if not any(web_item.values()):
                                        continue
                                evidence.append(web_item)
                        elif web_results is not None:
                            evidence.append(web_results)
                    except Exception as e:
                        logger.warning(f"Ошибка веб-дополнения при наличии KG: {e}")

                # Генерируем ответ через ML-модуль, если доступен; иначе — краткий ответ из KG
                response = None
                ml_unit = self.brain.components.get("ml_unit") if self.brain and self.brain.components else None
                if ml_unit is not None:
                    response = self._generate_response(query, evidence, nlp_info, concept, user_context)

                if response:
                    result["response"] = response
                    result["source"] = "knowledge_graph+ml_unit"
                    result["evidence"] = evidence
                    result["metrics"] = {"time": time.time() - start_time}
                    self._store_insight(query, response, nlp_info, concept)
                    self._store_conversation(query, response)
                else:
                    # Фолбэк: короткий ответ из узлов KG
                    result = self._build_response_from_nodes(result, nodes, start_time)

                # Обновляем метрики (если есть менеджер метрик)
                try:
                    if hasattr(self.brain, "metrics_manager") and self.brain.metrics_manager:
                        self.brain.metrics_manager.record_query_metrics(time.time() - start_time, True)
                except (AttributeError, TypeError, ValueError) as e:
                    logger.debug(f"Ошибка обновления метрик после пути KG->генерация: {e}")

                # Добавляем рассуждения для отображения в GUI
                self._add_reasoning_to_result(result)
                
                return result

            # 4) Параллельный поиск в памяти и вебе (с кэшированием)
            evidence = self._parallel_search(query)
            result["evidence"] = evidence

            # 5) Генерация ответа
            response = self._generate_response(query, evidence, nlp_info, concept, user_context)
            result["response"] = response
            result["source"] = "ml_unit" if response else "none"

            # 5a) Store insight for autonomous learning
            if response:
                self._store_insight(query, response, nlp_info, concept)
                self._store_conversation(query, response)

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
            # get_macroblocks_stats method doesn't exist in CoreBrain - removed
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
                    self.brain.metrics_manager.record_query_metrics(processing_time, True)
                # Дополнительно эмитим нормализованные метрики
                self._emit_metrics([
                    {"name": "query_processor.requests_total", "component": "query_processor", "type": "counter", "value": 1.0, "labels": {"result": "success"}},
                    {"name": "query_processor.process_time_seconds", "component": "query_processor", "type": "summary", "value": float(processing_time)},
                ])
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Ошибка обновления метрик в конце process_query: {e}")

            ambiguity_info = self._detect_ambiguity(query, nlp_info)
            if ambiguity_info.get("has_ambiguities"):
                result["ambiguities"] = ambiguity_info

            # 10) Добавляем рассуждения для отображения в GUI
            result["reasoning"] = self._get_reasoning_text()

            return result

        except Exception as e:
            logger.exception("Критическая ошибка в process_query")
            try:
                if hasattr(self.brain, "metrics_manager") and self.brain.metrics_manager:
                    self.brain.metrics_manager.record_query_metrics(time.time() - start_time, False)
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

            if self.brain and self.brain.components and self.brain.components.get("ml_unit"):
                try:
                    ml_unit = self.brain.components["ml_unit"]
                    if hasattr(ml_unit, 'process_text'):
                        nlp_info = ml_unit.process_text(query)
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
                        try:
                            self.brain.nlp_enqueue(item, module="default")
                            # Форсируем выпуск оставшегося батча
                            if hasattr(self.brain, "nlp_flush"):
                                self.brain.nlp_flush(module="default")
                            # Неблокирующий забор результатов (один раз для краткости)
                            if hasattr(self.brain, "nlp_try_get_result"):
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
                except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                    logger.debug(f"Ошибка в демо-интеграции пула инференса: {e}")
                    # Демонстрационная часть не должна ломать основной путь
                    pass
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"Ошибка при попытке доступа к ml_unit в _process_nlp: {e}")

        return nlp_info

    def _detect_ambiguity(self, query: str, nlp_result: Dict) -> Dict:
        """Detect ambiguous terms in query and return clarification info."""
        if not self.entity_extractor:
            return {"has_ambiguities": False, "clarifications": []}
        
        ambiguous = self.entity_extractor.extract_ambiguous_terms(query)
        clarifications = []
        
        for entity in ambiguous:
            if self.ambiguity_resolver:
                clarification = self.ambiguity_resolver.generate_clarification(entity, query)
                clarifications.append({
                    "term": entity.term,
                    "question": clarification.question if hasattr(clarification, 'question') else None,
                    "possible_meanings": entity.possible_meanings
                })
        
        return {
            "has_ambiguities": len(clarifications) > 0,
            "clarifications": clarifications
        }

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
                    adaptation_manager = self.brain.components["adaptation_manager"]
                    if hasattr(adaptation_manager, '_extract_concept_from_query'):
                        concept = adaptation_manager._extract_concept_from_query(query)
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
            cache_key = ""
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
            logger.error(f"Не удалось выполнить поиск в графе знаний для запроса '{query[:50]}...': {e}")
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
            evidence_list = []
            for n in nodes:
                try:
                    node_dict = getattr(n, "__dict__", None)
                    if node_dict is not None:
                        evidence_list.append({**node_dict, "_repr": str(n)})
                    else:
                        evidence_list.append({"repr": str(n)})
                except (AttributeError, TypeError, ValueError) as e:
                    evidence_list.append({"repr": str(n)})
            result.update({
                "response": resp,
                "source": "knowledge_graph",
                "evidence": evidence_list,
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
                try:
                    exec_ref = ThreadPoolExecutor(max_workers=2)
                    self._own_executor = True
                    self.executor = exec_ref
                except (OSError, RuntimeError) as e:
                    logger.warning(f"Не удалось создать ThreadPoolExecutor: {e}")
                    return evidence
            
            # Поиск в памяти
            if self.brain.components.get("memory_manager") and hasattr(self.brain.components["memory_manager"], 'search_memories_by_entity'):
                try:
                    entity_term = query.split()[0] if query else ""
                    futures.append(exec_ref.submit(self.brain.components["memory_manager"].search_memories_by_entity, entity_term))
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
            gen_context = {
                "nlp": nlp_info,
                "evidence": evidence,
                "concept": concept,
                "user_context": user_context
            }
            
            if not self.brain.components.get("ml_unit"):
                logger.debug("ml_unit not available in components")
                return "Извините, модуль генерации недоступен."
            
            # Validate model and tokenizer if they should be used
            if self.model is None:
                logger.debug("Self.model is None - model not loaded")
            if self.tokenizer is None:
                logger.debug("Self.tokenizer is None - tokenizer not loaded")
            
            # Используем ml_unit напрямую вместо brain.process_query() чтобы избежать бесконечного цикла
            ml_unit = self.brain.components.get("ml_unit")
            if ml_unit and hasattr(ml_unit, 'generate'):
                result = ml_unit.generate(query, context=gen_context)
                if result and isinstance(result, dict):
                    text = result.get('text')
                    if text and isinstance(text, str):
                        logger.info(f"Response generated successfully via ml_unit")
                        return text
                    generated = result.get('generated_text')
                    if generated and isinstance(generated, str):
                        logger.info(f"Response generated successfully via ml_unit (generated_text)")
                        return generated
                    return "Ошибка обработки"
                elif isinstance(result, str):
                    return result
                return "Ошибка обработки"
            
            # Fallback logging
            logger.info("Falling back to basic response - ml_unit generation failed")
            return "Извините, модуль генерации недоступен."
            
        except AttributeError as e:
            logger.error(f"Ошибка генерации ответа - отсутствует атрибут: {e}")
            return "Извините, компонент генерации недоступен."
        except TypeError as e:
            logger.error(f"Ошибка генерации ответа - некорректный тип данных: {e}")
            return "Извините, произошла внутренняя ошибка обработки."
        except ValueError as e:
            logger.error(f"Ошибка генерации ответа - некорректное значение: {e}")
            return "Извините, получены некорректные данные."
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
                    else:
                        logger.debug("contradiction_resolver found but no compatible methods, skipping")
                except Exception as e:
                    logger.warning(f"Ошибка проверки противоречий: {e}")
            else:
                logger.debug("No contradiction_resolver available, skipping contradiction check")
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.debug(f"Ошибка при попытке доступа к contradiction_resolver в _check_contradictions: {e}")

        return contradictions

    def _emit_metrics(self, metrics: List[Dict[str, Any]]):
        """Безопасная эмиссия метрик для QueryProcessor (через событийную шину)."""
        try:
            if getattr(self, "brain", None):
                if hasattr(self.brain, "events") and self.brain.events:
                    self.brain.events.trigger('metrics', metrics)
        except Exception as e:
            logger.debug(f"Ошибка в _emit_metrics: {e}")

    def _store_insight(self, query: str, response: str, nlp_info: Dict, concept: Optional[str]):
        """Stores query/response insights in fractal memory for autonomous learning."""
        try:
            if not hasattr(self.brain, 'memory_graph_ml') or not self.brain.memory_graph_ml:
                return
            
            mgml = self.brain.memory_graph_ml
            
            insight_text = f"Query: {query}\nResponse: {response}"
            if concept:
                insight_text += f"\nConcept: {concept}"
            
            metadata = {
                'entities': nlp_info.get('entities', []),
                'keywords': nlp_info.get('keywords', []),
                'concept': concept
            }
            
            if hasattr(mgml, 'add_insight'):
                mgml.add_insight(insight_text, query, metadata)
            logger.debug(f"Stored insight from query: {query[:50]}...")
            
        except Exception as e:
            logger.debug(f"Error storing insight: {e}")

    def _store_conversation(self, query: str, response: str):
        """Store conversation exchange in memory for context retention."""
        try:
            if hasattr(self.brain, 'memory_manager') and self.brain.memory_manager:
                self.brain.memory_manager.add_interaction(
                    user_id="default_user",
                    query=query,
                    response=response,
                    context={"source": "query_processor"}
                )
                logger.debug(f"Stored conversation in memory")
        except Exception as e:
            logger.debug(f"Error storing conversation: {e}")

    def _get_conversation_context(self) -> List[Dict]:
        try:
            if self.brain and hasattr(self.brain, 'memory_manager') and self.brain.memory_manager:
                memory_manager = self.brain.memory_manager
                if hasattr(memory_manager, 'get_recent_interactions'):
                    interactions = memory_manager.get_recent_interactions(limit=10)
                else:
                    interactions = None
                if not interactions:
                    return []
                return [{"query": i.get("query", ""), "response": i.get("response", "")} 
                        for i in interactions if i and isinstance(i, dict)]
        except Exception as e:
            logger.debug(f"Error getting conversation context: {e}")
        return []
    
    def _add_reasoning_to_result(self, result: Dict[str, Any]):
        """Добавляет рассуждения в результат для GUI."""
        try:
            reasoning_text = self._get_reasoning_text()
            if reasoning_text:
                result["reasoning"] = reasoning_text
        except Exception as e:
            logger.debug(f"Error adding reasoning to result: {e}")
    
    def _get_reasoning_text(self) -> str:
        """Извлекает текст рассуждений из brain/reasoning_engine."""
        try:
            if hasattr(self.brain, 'reasoning_engine') and self.brain.reasoning_engine:
                reasoning_engine = self.brain.reasoning_engine
                
                # Пробуем process_query если доступен
                if hasattr(reasoning_engine, 'process_query'):
                    try:
                        result = reasoning_engine.process_query(self.current_query if hasattr(self, 'current_query') else "")
                        if result:
                            if isinstance(result, dict):
                                return self._format_reasoning_dict(result)
                            elif isinstance(result, str):
                                return result
                    except Exception as e:
                        logger.debug(f"Error calling reasoning_engine.process_query: {e}")
                
                # Пробуем получить steps из result
                if hasattr(reasoning_engine, 'last_result') and reasoning_engine.last_result:
                    last_result = reasoning_engine.last_result
                    if isinstance(last_result, dict):
                        return self._format_reasoning_dict(last_result)
                
                # Пробуем dialogue.steps
                if hasattr(reasoning_engine, 'dialogue') and reasoning_engine.dialogue:
                    if hasattr(reasoning_engine.dialogue, 'steps') and reasoning_engine.dialogue.steps:
                        steps = reasoning_engine.dialogue.steps
                        if steps:
                            return self._format_steps(steps)
            
            if hasattr(self.brain, 'self_reasoning_engine') and self.brain.self_reasoning_engine:
                sre = self.brain.self_reasoning_engine
                if hasattr(sre, 'last_result') and sre.last_result:
                    last_result = sre.last_result
                    if isinstance(last_result, dict):
                        return self._format_reasoning_dict(last_result)
        except Exception as e:
            logger.debug(f"Error getting reasoning text: {e}")
        return ""
    
    def _format_reasoning_dict(self, reasoning_dict: Dict) -> str:
        """Форматирует словарь рассуждений в строку."""
        if not reasoning_dict:
            return ""
        
        lines = []
        
        if 'steps' in reasoning_dict and reasoning_dict['steps']:
            lines.append("Этапы рассуждения:")
            for i, step in enumerate(reasoning_dict['steps'][:5], 1):
                if isinstance(step, dict):
                    phase = step.get('phase', step.get('thought', f'Шаг {i}'))
                    thought = step.get('thought', '')
                    lines.append(f"  {i}. {phase}")
                    if thought:
                        lines.append(f"     {thought}")
                else:
                    lines.append(f"  {i}. {step}")
        
        if 'iterations' in reasoning_dict:
            lines.append(f"Итераций: {reasoning_dict['iterations']}")
        
        if 'confidence' in reasoning_dict:
            lines.append(f"Уверенность: {reasoning_dict['confidence']:.2f}")
        
        return "\n".join(lines) if lines else str(reasoning_dict)
    
    def _format_steps(self, steps) -> str:
        """Форматирует steps в строку для отображения."""
        if not steps:
            return ""
        
        lines = ["Этапы рассуждения:"]
        for i, step in enumerate(steps[:5], 1):
            try:
                if hasattr(step, 'phase'):
                    phase = step.phase.value if hasattr(step.phase, 'value') else str(step.phase)
                    lines.append(f"  {i}. {phase}")
                elif hasattr(step, 'thought'):
                    lines.append(f"  {i}. {step.thought}")
                else:
                    lines.append(f"  {i}. {step}")
            except Exception:
                lines.append(f"  {i}. {step}")
        
        return "\n".join(lines)

    def close(self):
        """Закрывает и освобождает ресурсы executor."""
        try:
            if self._own_executor and self.executor:
                self.executor.shutdown(wait=True)
                self.executor = None
                self._own_executor = False
        except (AttributeError, TypeError, RuntimeError, OSError) as e:
            logger.debug(f"Ошибка при shutdown executor: {e}")
        logger.debug("QueryProcessor ресурсы освобождены")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        try:
            if self._own_executor and self.executor:
                self.executor.shutdown(wait=False)
                self.executor = None
                self._own_executor = False
        except Exception:
            pass