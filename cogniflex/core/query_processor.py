"""Модуль обработки запросов для CogniFlex"""
import time
import logging
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

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
            # 1) NLP preprocessing
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
                except Exception:
                    logger.exception("Ошибка подготовки доказательств из узлов KG")

                # Опциональное дополнение веб-поиском при включённой конфигурации
                try:
                    augment_web = bool(getattr(self.brain, "config", {}).get("augment_with_web_on_kg", True))
                except Exception:
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
                except Exception:
                    logger.exception("Ошибка обновления метрик после пути KG->генерация")

                return result

            # 4) Параллельный поиск в памяти и вебе
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
            result["metrics"] = {
                "time": processing_time,
                "nlp_info": nlp_info
            }

            try:
                if hasattr(self.brain, "metrics_manager") and self.brain.metrics_manager:
                    self.brain.metrics_manager.update_request_metrics(processing_time, True)
            except Exception:
                logger.exception("Ошибка обновления метрик в конце process_query")

            return result

        except Exception as e:
            logger.exception("Критическая ошибка в process_query")
            try:
                if hasattr(self.brain, "metrics_manager") and self.brain.metrics_manager:
                    self.brain.metrics_manager.update_request_metrics(time.time() - start_time, False)
            except Exception:
                logger.exception("Ошибка обновления метрик после исключения в process_query")
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
            if self.brain.components.get("ml_unit"):
                try:
                    nlp_info = self.brain.components["ml_unit"].process_text(query)
                except Exception as e:
                    logger.warning(f"Ошибка NLP обработки: {e}")
        except Exception:
            logger.exception("Ошибка при попытке доступа к ml_unit в _process_nlp")

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
        except Exception:
            logger.exception("Ошибка при попытке доступа к adaptation_manager в _extract_concept")
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

            kg = self.brain.components["knowledge_graph"]

            # Проверяем доступные методы поиска
            for method_name in ['search_nodes', 'search', 'find_nodes', 'query_nodes']:
                if hasattr(kg, method_name):
                    try:
                        return getattr(kg, method_name)(query, limit=limit)
                    except TypeError:
                        # Пытаемся вызвать без параметра limit
                        try:
                            return getattr(kg, method_name)(query)
                        except Exception as e:
                            logger.warning(f"Ошибка вызова {method_name}: {e}")
                            return []
        except Exception:
            logger.exception("Ошибка в _search_knowledge_graph")
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
        except Exception:
            logger.exception("Ошибка при сборке ответа из узлов графа")
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

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Поиск в памяти
                if self.brain.components.get("memory_manager") and hasattr(self.brain.components["memory_manager"], 'search'):
                    futures.append(executor.submit(self.brain.components["memory_manager"].search, query))

                # Веб-поиск
                if self.brain.components.get("web_search_engine"):
                    futures.append(executor.submit(self.brain.components["web_search_engine"].search, query, max_results=3))

                # Сбор результатов
                for future in as_completed(futures):
                    try:
                        results = future.result()
                        # Нормализуем: если результат — список, расширяем, иначе добавляем как единицу доказательств
                        if isinstance(results, list):
                            evidence.extend(results)
                        else:
                            evidence.append(results)
                    except Exception as e:
                        logger.warning(f"Ошибка при асинхронном поиске: {e}")
        except Exception:
            logger.exception("Критическая ошибка в _parallel_search")

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
        if not self.brain.components.get("ml_unit"):
            return "Извините, модуль генерации недоступен."

        try:
            # Формируем контекст для генерации
            gen_context = {
                "nlp": nlp_info,
                "evidence": evidence,
                "concept": concept,
                "user_context": user_context
            }

            # Генерируем ответ
            ml = self.brain.components["ml_unit"]
            if hasattr(ml, "process_query"):
                return ml.process_query(query, context=gen_context)
            elif hasattr(ml, "generate_response"):
                return ml.generate_response(query, context=gen_context)
            else:
                logger.warning("ml_unit не поддерживает process_query или generate_response")
                return "Извините, модуль генерации не поддерживает необходимые методы."
        except Exception:
            logger.exception("Ошибка генерации ответа")
            return None

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
        except Exception:
            logger.exception("Ошибка при попытке доступа к ethics_framework в _check_ethics")

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
        except Exception:
            logger.exception("Ошибка при попытке доступа к contradiction_resolver в _check_contradictions")

        return contradictions
