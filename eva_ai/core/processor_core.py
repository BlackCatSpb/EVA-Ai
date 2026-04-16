"""Модуль обработки запросов для ЕВА — ядро, инициализация, жизненный цикл."""
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
    import torch
    TORCH_AVAILABLE = True
except (ImportError, ModuleNotFoundError, RuntimeError):
    torch = None
    TORCH_AVAILABLE = False

try:
    from eva_ai.knowledge.context_entity import EntityExtractor
    from eva_ai.knowledge.ambiguity_resolver import AmbiguityResolver
except ImportError as e:
    logger.warning(f"Failed to import EntityExtractor or AmbiguityResolver: {e}")
    EntityExtractor = None
    AmbiguityResolver = None


class QueryProcessor:
    """Обрабатывает пользовательские запросы через конвейер обработки."""

    def __init__(self, brain):
        self.brain = brain
        self.hybrid_cache = None
        self.executor: Optional[ThreadPoolExecutor] = None
        self._own_executor = False

        try:
            if self.brain is None:
                logger.warning("brain is None, QueryProcessor initialization limited")
                return
            if getattr(self.brain, "memory_manager", None) is not None and hasattr(self.brain.memory_manager, "get_hybrid_cache"):
                self.hybrid_cache = self.brain.memory_manager.get_hybrid_cache()
            if not self.hybrid_cache and getattr(self.brain, "text_processor", None) is not None:
                self.hybrid_cache = getattr(self.brain.text_processor, "hybrid_cache", None)
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"Ошибка инициализации hybrid_cache: {e}")

        try:
            import os
            if getattr(self.brain, "text_processor", None) and getattr(self.brain.text_processor, "executor", None):
                self.executor = self.brain.text_processor.executor
            if self.executor is None:
                # OpenVINO использует CPU, оставляем 2 ядра для системы
                workers = max(2, (os.cpu_count() or 4) - 2)
                self.executor = ThreadPoolExecutor(max_workers=workers)
                self._own_executor = True
        except (AttributeError, TypeError, RuntimeError, OSError) as e:
            logger.debug(f"Ошибка инициализации executor: {e}")

        self.entity_extractor = EntityExtractor() if EntityExtractor else None
        self.ambiguity_resolver = AmbiguityResolver() if AmbiguityResolver else None

        self.model = None
        self.tokenizer = None
        self.embeddings: OrderedDict = OrderedDict()
        self._embeddings_max_size = 1000
        self.current_query = ""
        self.initialized = True
        self.running = True

        self._initialize_model_components()

    def _initialize_model_components(self):
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
        if key in self.embeddings:
            embedding = self.embeddings[key]
            if embedding is None:
                return None
            self.embeddings.move_to_end(key)
            return embedding
        return None

    def _set_embedding(self, key: str, embedding: Any):
        if embedding is None:
            return
        if key in self.embeddings:
            self.embeddings.move_to_end(key)
            self.embeddings[key] = embedding
            return
        if len(self.embeddings) >= self._embeddings_max_size:
            oldest_key = next(iter(self.embeddings))
            self.embeddings.pop(oldest_key, None)
            logger.debug(f"Evicted oldest embedding key: {oldest_key}")
        self.embeddings[key] = embedding

    def process_query(self, query: str, user_context: Optional[Dict] = None) -> Dict[str, Any]:
        if not query or not query.strip():
            return {
                "response": "Пожалуйста, введите текст запроса.", "source": "none",
                "evidence": [], "metrics": {}, "error": None, "contradictions": [],
                "ethics": {"score": 1.0, "violations": [], "recommendations": []},
                "ambiguities": None, "reasoning": "", "contradiction_detected": False
            }

        if self.brain is None:
            return {
                "response": "Система не инициализирована.", "source": "none",
                "evidence": [], "metrics": {}, "error": "brain is None", "contradictions": [],
                "ethics": {"score": 1.0, "violations": [], "recommendations": []},
                "ambiguities": None, "reasoning": "", "contradiction_detected": False
            }

        self.current_query = query
        start_time = time.time()
        result: Dict[str, Any] = {
            "response": None, "source": "none", "evidence": [],
            "metrics": {}, "error": None, "contradictions": []
        }

        conversation_context = self._get_conversation_context()

        try:
            nlp_info = self._process_nlp(query)
            concept = self._extract_concept(query)
            nodes = self._search_knowledge_graph(query)

            if nodes:
                evidence: List[Dict[str, Any]] = []
                try:
                    for n in nodes:
                        if hasattr(n, "__dict__"):
                            evidence.append({**getattr(n, "__dict__", {}), "_repr": str(n)})
                        else:
                            evidence.append({"repr": str(n)})
                except (AttributeError, TypeError, ValueError) as e:
                    logger.debug(f"Ошибка подготовки доказательств из узлов KG: {e}")

                brain_config = getattr(self.brain, "config", {}) or {}
                augment_web = brain_config.get("augment_with_web_on_kg", True)

                if augment_web and self.brain.components.get("web_search_engine"):
                    try:
                        web_engine = self.brain.components["web_search_engine"]
                        web_results = web_engine.search(query, max_results=3) if hasattr(web_engine, 'search') else None
                        if isinstance(web_results, list):
                            for web_item in web_results:
                                if isinstance(web_item, dict):
                                    if 'snippet' in web_item:
                                        snippet = web_item['snippet']
                                        snippet = re.sub(r'https?://\S+', '', snippet)
                                        snippet = re.sub(r'<[^>]+>', '', snippet)
                                        snippet = re.sub(r'\s+', ' ', snippet).strip()
                                        if len(snippet) > 30:
                                            web_item['snippet'] = snippet[:200]
                                        else:
                                            web_item['snippet'] = ''
                                    if not any(web_item.values()):
                                        continue
                                evidence.append(web_item)
                        elif web_results is not None:
                            evidence.append(web_results)
                    except Exception as e:
                        logger.warning(f"Ошибка веб-дополнения при наличии KG: {e}")

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
                    result = self._build_response_from_nodes(result, nodes, start_time)

                try:
                    if hasattr(self.brain, "metrics_manager") and self.brain.metrics_manager:
                        self.brain.metrics_manager.record_query_metrics(time.time() - start_time, True)
                except (AttributeError, TypeError, ValueError) as e:
                    logger.debug(f"Ошибка обновления метрик после пути KG->генерация: {e}")

                self._add_reasoning_to_result(result)
                return result

            evidence = self._parallel_search(query)
            result["evidence"] = evidence

            response = self._generate_response(query, evidence, nlp_info, concept, user_context)
            result["response"] = response
            result["source"] = "ml_unit" if response else "none"

            if response:
                self._store_insight(query, response, nlp_info, concept)
                self._store_conversation(query, response)

            if response:
                ethics_result = self._check_ethics(response, nlp_info, user_context)
                result["ethics"] = ethics_result

            contradictions = self._check_contradictions(query, response)
            if contradictions:
                result["contradictions"] = contradictions
                result["contradiction_detected"] = True

            processing_time = time.time() - start_time
            macroblocks_stats = {}

            result["metrics"] = {
                "time": processing_time, "nlp_info": nlp_info, "macroblocks": macroblocks_stats,
            }

            try:
                if hasattr(self.brain, "metrics_manager") and self.brain.metrics_manager:
                    self.brain.metrics_manager.record_query_metrics(processing_time, True)
                self._emit_metrics([
                    {"name": "query_processor.requests_total", "component": "query_processor", "type": "counter", "value": 1.0, "labels": {"result": "success"}},
                    {"name": "query_processor.process_time_seconds", "component": "query_processor", "type": "summary", "value": float(processing_time)},
                ])
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Ошибка обновления метрик в конце process_query: {e}")

            ambiguity_info = self._detect_ambiguity(query, nlp_info)
            if ambiguity_info.get("has_ambiguities"):
                result["ambiguities"] = ambiguity_info

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

    def close(self):
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


