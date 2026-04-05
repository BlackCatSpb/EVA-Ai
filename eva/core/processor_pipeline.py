"""Модуль обработки запросов для ЕВА — конвейер обработки, стадии."""
import logging
import hashlib
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, ModuleNotFoundError, RuntimeError):
    torch = None
    TORCH_AVAILABLE = False


class QueryProcessor:
    """Placeholder for import compatibility — methods are on core class."""
    pass


class ProcessingPipeline:
    """Конвейер обработки запросов с этапами."""

    def __init__(self, parent):
        self.parent = parent

    def run_nlp_stage(self, query: str) -> Dict[str, Any]:
        nlp_info = {"keywords": [], "entities": [], "intent": None, "sentiment": 0.0}

        try:
            cache_key = None
            if self.parent.hybrid_cache:
                try:
                    cache_key = f"nlp:{hashlib.md5(query.encode('utf-8')).hexdigest()}"
                    cached = self.parent.hybrid_cache.get(cache_key)
                    if isinstance(cached, dict) and cached.get("metadata", {}).get("processor"):
                        return cached
                except (AttributeError, TypeError, ValueError) as e:
                    logger.debug(f"Ошибка получения данных из кэша NLP: {e}")

            if self.parent.brain and self.parent.brain.components and self.parent.brain.components.get("ml_unit"):
                try:
                    ml_unit = self.parent.brain.components["ml_unit"]
                    if hasattr(ml_unit, 'process_text'):
                        nlp_info = ml_unit.process_text(query)
                    if self.parent.hybrid_cache and cache_key:
                        try:
                            self.parent.hybrid_cache.set(cache_key, nlp_info)
                        except (AttributeError, TypeError, ValueError) as e:
                            logger.debug(f"Ошибка сохранения в кэш NLP: {e}")
                except Exception as e:
                    logger.warning(f"Ошибка NLP обработки: {e}")

            try:
                demo_enabled = bool(getattr(self.parent.brain, "config", {}).get("nlp_demo_integration", False))
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Ошибка получения конфигурации nlp_demo_integration: {e}")
                demo_enabled = False

            if demo_enabled and torch is not None:
                try:
                    length = max(1, len(query))
                    ids = torch.arange(length, dtype=torch.long)
                    item = {"input_ids": ids}
                    if hasattr(self.parent.brain, "nlp_enqueue"):
                        try:
                            self.parent.brain.nlp_enqueue(item, module="default")
                            if hasattr(self.parent.brain, "nlp_flush"):
                                self.parent.brain.nlp_flush(module="default")
                            if hasattr(self.parent.brain, "nlp_try_get_result"):
                                res = self.parent.brain.nlp_try_get_result(module="default", timeout_s=0.0)
                                if isinstance(res, dict):
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
                    pass
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"Ошибка при попытке доступа к ml_unit в _process_nlp: {e}")

        return nlp_info

    def run_concept_extraction(self, query: str) -> Optional[str]:
        concept = None
        try:
            if not self.parent.brain or not self.parent.brain.components:
                return concept
            if self.parent.brain.components.get("adaptation_manager"):
                try:
                    adaptation_manager = self.parent.brain.components["adaptation_manager"]
                    if hasattr(adaptation_manager, '_extract_concept_from_query'):
                        concept = adaptation_manager._extract_concept_from_query(query)
                except Exception as e:
                    logger.warning(f"Ошибка извлечения концепта: {e}")
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"Ошибка при попытке доступа к adaptation_manager: {e}")
        return concept

    def run_knowledge_graph_search(self, query: str, limit: int = 3) -> List[Any]:
        try:
            if not self.parent.brain or not self.parent.brain.components:
                return []
            if not self.parent.brain.components.get("knowledge_graph"):
                return []

            cache_key = ""
            if self.parent.hybrid_cache:
                try:
                    cache_key = f"kg:{hashlib.md5((query + '|' + str(limit)).encode('utf-8')).hexdigest()}"
                    cached_nodes = self.parent.hybrid_cache.get(cache_key)
                    if isinstance(cached_nodes, list) and cached_nodes:
                        return cached_nodes
                except (AttributeError, TypeError, ValueError, KeyError, IndexError) as e:
                    logger.debug(f"Ошибка получения данных из кэша KG: {e}")

            kg = self.parent.brain.components["knowledge_graph"]
            for method_name in ['search_nodes', 'search', 'find_nodes', 'query_nodes']:
                if hasattr(kg, method_name):
                    try:
                        nodes = getattr(kg, method_name)(query, limit=limit)
                        if self.parent.hybrid_cache and cache_key:
                            try:
                                self.parent.hybrid_cache.set(cache_key, nodes)
                            except (AttributeError, TypeError, ValueError) as e:
                                logger.debug(f"Ошибка сохранения в кэш KG: {e}")
                        return nodes
                    except TypeError:
                        try:
                            nodes = getattr(kg, method_name)(query)
                            if self.parent.hybrid_cache and cache_key:
                                try:
                                    self.parent.hybrid_cache.set(cache_key, nodes)
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

    def run_parallel_search(self, query: str) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        futures = []

        cache_key = None
        evidence_cache_enabled = True
        try:
            evidence_cache_enabled = bool(getattr(self.parent.brain, "config", {}).get("evidence_cache_enabled", True))
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"Ошибка получения конфигурации evidence_cache_enabled: {e}")
            evidence_cache_enabled = True

        if self.parent.hybrid_cache and evidence_cache_enabled:
            try:
                cache_key = f"evidence:{hashlib.md5(query.encode('utf-8')).hexdigest()}"
                cached = self.parent.hybrid_cache.get(cache_key)
                if isinstance(cached, list) and cached:
                    return cached
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Ошибка получения кэша доказательств: {e}")

        try:
            exec_ref = self.parent.executor
            if exec_ref is None:
                try:
                    exec_ref = ThreadPoolExecutor(max_workers=2)
                    self.parent._own_executor = True
                    self.parent.executor = exec_ref
                except (OSError, RuntimeError) as e:
                    logger.warning(f"Не удалось создать ThreadPoolExecutor: {e}")
                    return evidence

            if self.parent.brain.components.get("memory_manager") and hasattr(self.parent.brain.components["memory_manager"], 'search_memories_by_entity'):
                try:
                    entity_term = query.split()[0] if query else ""
                    futures.append(exec_ref.submit(self.parent.brain.components["memory_manager"].search_memories_by_entity, entity_term))
                except (AttributeError, TypeError, RuntimeError) as e:
                    logger.debug(f"Ошибка добавления поиска memory_manager: {e}")

            if self.parent.brain.components.get("web_search_engine"):
                try:
                    futures.append(exec_ref.submit(self.parent.brain.components["web_search_engine"].search, query, max_results=3))
                except (AttributeError, TypeError, RuntimeError) as e:
                    logger.debug(f"Ошибка добавления веб-поиска: {e}")

            for future in futures:
                try:
                    results = future.result()
                    if isinstance(results, list):
                        evidence.extend(results)
                    elif results is not None:
                        evidence.append(results)
                except Exception as e:
                    logger.warning(f"Ошибка при асинхронном поиске: {e}")
        except (RuntimeError, TimeoutError, OSError) as e:
            logger.debug(f"Критическая ошибка в _parallel_search: {e}")

        if self.parent.hybrid_cache and cache_key and evidence_cache_enabled:
            try:
                self.parent.hybrid_cache.set(cache_key, evidence)
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Ошибка сохранения объединённых доказательств в кэш: {e}")

        return evidence
