"""
SRE Recursive Module — recursive reasoning steps, semantic stability, fractal reasoning methods.
"""

import logging
from typing import Dict, Any, Optional, List

__all__ = [
    '_recursive_process_query',
    '_check_semantic_stability',
    '_recursive_reasoning_step',
    '_is_complex_query',
    'decompose_query',
    'retrieve_similar_reasoning',
    'build_recursive_context',
    '_synthesize_recursive_results',
    '_linear_process_query',
    '_init_retriever',
]

logger = logging.getLogger(__name__)


def _recursive_process_query(
    self,
    query: str,
    user_context: Optional[Dict],
    depth: int
) -> Dict[str, Any]:
    """Рекурсивная обработка сложных запросов"""
    self.recursive_calls += 1

    logger.info(f"Рекурсивный вызов depth={depth}, query={query[:30]}...")

    if depth >= self.max_recursion_depth:
        logger.info(f"Достигнута максимальная глубина {self.max_recursion_depth}")
        return self._linear_process_query(query, user_context)

    self._init_retriever()

    sub_queries = self.decompose_query(query)
    logger.info(f"Декомпозиция на {len(sub_queries)} подзадач")

    if not sub_queries:
        logger.info("Декомпозиция вернула пустой список, используем линейную обработку")
        return self._linear_process_query(query, user_context)

    sub_results = []
    for sq in sub_queries:
        if depth + 1 <= self.max_recursion_depth:
            sub_result = self._recursive_process_query(sq, user_context, depth=depth + 1)
        else:
            sub_result = self._linear_process_query(sq, user_context)
        sub_results.append(sub_result)

    similar_reasoning = []
    if self.fractal_retriever:
        similar_reasoning = self.retrieve_similar_reasoning(query)
        logger.info(f"Найдено {len(similar_reasoning)} похожих рассуждений")

    final_result = self._synthesize_recursive_results(
        query, sub_results, similar_reasoning, depth
    )

    return final_result


def _check_semantic_stability(self, responses: List[str], threshold: float = 0.7) -> bool:
    """Проверяет семантическую стабильность между несколькими ответами"""
    if len(responses) < 2:
        return True

    for i in range(len(responses) - 1):
        r1_lower = responses[i].lower()
        r2_lower = responses[i + 1].lower()

        words1 = set(r1_lower.split())
        words2 = set(r2_lower.split())

        if not words1 or not words2:
            continue

        overlap = len(words1.intersection(words2)) / max(len(words1), len(words2))
        if overlap < threshold:
            return False

    return True


def _recursive_reasoning_step(
    self,
    query: str,
    context: Dict[str, Any],
    depth: int
) -> Dict[str, Any]:
    """Один шаг рекурсивного рассуждения"""
    if depth >= self.max_recursion_depth:
        return self._linear_process_query(query, context)

    sub_queries = self.decompose_query(query)

    if not sub_queries:
        return self._linear_process_query(query, context)

    sub_results = []
    for sq in sub_queries:
        result = self._recursive_reasoning_step(sq, context, depth + 1)
        sub_results.append(result)

    similar = []
    if self.fractal_retriever:
        similar = self.retrieve_similar_reasoning(query)

    return self._synthesize_recursive_results(query, sub_results, similar, depth)


def _is_complex_query(self, query: str) -> bool:
    """Определить сложность запроса для выбора стратегии"""
    complexity_indicators = [
        " и ", " или ", " но ", " потому что ", " поэтому",
        " если ", " тогда ", " следовательно ", " значит ",
        " как ", "почему", "какие", "сравни", "различия", "отличие"
    ]

    query_lower = query.lower()
    complexity_score = sum(1 for ind in complexity_indicators if ind in query_lower)

    is_long = len(query.split()) > 15

    return complexity_score >= 2 or is_long


def decompose_query(self, query: str) -> List[str]:
    """Декомпозиция сложного запроса на подзадачи"""
    prompt = f"""Разбей запрос на 2-4 простых подзапроса.
Запрос: {query}
Верни только список подзапросов, каждый на новой строке:"""

    try:
        response = self._generate_with_qwen(prompt)
        if response:
            sub_queries = [
                line.strip() for line in response.split('\n')
                if line.strip() and len(line.strip()) > 10
            ]
            if len(sub_queries) >= 2:
                logger.info(f"Декомпозиция успешна: {len(sub_queries)} подзадач")
                return sub_queries
            else:
                logger.info(f"Декомпозиция вернула мало подзадач: {len(sub_queries)}, использую линейный режим")
    except Exception as e:
        logger.warning(f"Декомпозиция не удалась: {e}")

    return []


def retrieve_similar_reasoning(self, query: str) -> List[Dict]:
    """Найти похожие рассуждения из прошлого"""
    if not self.fractal_retriever or not self.fractal_storage:
        return []

    try:
        results = self.fractal_retriever.retrieve_with_embedding(query=query, top_k=5)
        similar = [r for r in results if r.get('node_type') == 'reasoning_step']
        return similar
    except Exception as e:
        logger.warning(f"Ошибка поиска похожих рассуждений: {e}")
        return []


def build_recursive_context(self, query: str) -> Dict:
    """Построить контекст из разных уровней хранилища"""
    context = {"level_0": [], "level_1": [], "level_2": [], "level_3": [], "similar": []}

    if not self.fractal_retriever:
        return context

    try:
        cross_level = self.fractal_retriever.retrieve_cross_level(query=query, levels=[0, 1, 2, 3])
        for level, nodes in cross_level.items():
            context[f"level_{level}"] = nodes
        context["similar"] = self.retrieve_similar_reasoning(query)
    except Exception as e:
        logger.warning(f"Ошибка построения контекста: {e}")

    return context


def _synthesize_recursive_results(
    self,
    query: str,
    sub_results: List[Dict],
    similar_reasoning: List[Dict],
    depth: int
) -> Dict[str, Any]:
    """Синтез результатов рекурсивной обработки с consistent confidence calculation"""
    combined_responses = [r.get("response", "") for r in sub_results]
    combined_confidences = [r.get("confidence", 0.0) for r in sub_results]

    if combined_confidences:
        weights = [r.get("iterations", 1) for r in sub_results]
        total_weight = sum(weights)
        avg_confidence = sum(c * w for c, w in zip(combined_confidences, weights)) / total_weight if total_weight > 0 else 0.5
    else:
        avg_confidence = 0.5

    if similar_reasoning:
        similar_confidences = [s.get("confidence", 0.5) for s in similar_reasoning]
        if similar_confidences:
            avg_similar = sum(similar_confidences) / len(similar_confidences)
            boost = min(0.1, avg_similar * 0.2)
            avg_confidence = min(1.0, avg_confidence + boost)

    avg_confidence = round(max(0.0, min(1.0, avg_confidence)), 3)

    prompt = f"""На основе подответов составь единый ответ на вопрос.
    Вопрос: {query}
    Подответы: {' '.join(combined_responses)}
    Дай финальный ответ:"""

    final_response = self._generate_with_qwen(prompt)

    return {
        "response": final_response or (combined_responses[0] if combined_responses else ""),
        "text": final_response or (combined_responses[0] if combined_responses else ""),
        "status": "ok",
        "confidence": avg_confidence,
        "source": "recursive_reasoning",
        "recursive_depth": depth + 1,
        "sub_queries_processed": len(sub_results),
        "similar_found": len(similar_reasoning),
        "reasoning": {
            "sub_results": sub_results,
            "similar_reasoning": similar_reasoning,
            "depth": depth
        }
    }


def _linear_process_query(
    self,
    query: str,
    user_context: Optional[Dict]
) -> Dict[str, Any]:
    """Стандартная линейная обработка (без рекурсии)"""
    from .reasoning_types import ReasoningStep, ReasoningResult, ReasoningPhase
    from .confidence_scorer import calculate_overall_confidence, should_terminate

    result = ReasoningResult(
        final_response="",
        confidence=0.0,
        iterations=0,
        query=query
    )

    current_query = query
    iteration = 0

    while iteration < self.max_iterations:
        iteration += 1
        result.iterations = iteration
        self.total_iterations += 1

        step = ReasoningStep(
            phase=ReasoningPhase.GENERATION.value,
            thought=f"Генерирую ответ на: {current_query[:30]}...",
            confidence=0.0
        )

        response = self._generate_with_qwen(current_query)
        result.steps.append(step)

        if not response or response.startswith("Ошибка"):
            result.final_response = response or "Ошибка генерации"
            break

        analysis = self._analyze_response(current_query, response)

        confidence = calculate_overall_confidence(
            ethics_result=analysis.ethics_result,
            contradiction_result=analysis.contradiction_result,
            knowledge_result=analysis.knowledge_result,
            query=current_query
        )

        step = ReasoningStep(
            phase=ReasoningPhase.FINAL_SYNTHESIS.value,
            thought=f"Анализ завершён. Уверенность: {confidence:.2f}",
            confidence=confidence
        )
        result.steps.append(step)

        result.confidence = confidence

        if should_terminate(confidence, self.confidence_threshold):
            result.final_response = response
            break

        if iteration < self.max_iterations:
            questions = self._generate_clarification(analysis, current_query)
            result.clarification_questions = questions

            if questions:
                result.final_response = f"Уточните, пожалуйста: {questions[0]}"
                break

    if iteration >= self.max_iterations and not result.final_response:
        result.final_response = "Извините, мне нужно больше информации."

    self._store_reasoning_chain(result)

    return {
        "response": result.final_response,
        "text": result.final_response,
        "status": "ok",
        "confidence": result.confidence,
        "reasoning": result.to_dict(),
        "source": "linear_reasoning",
        "processing_time": 0.0
    }


def _init_retriever(self):
    """Инициализация Retriever после подключения storage"""
    if self.fractal_storage and not self.fractal_retriever and self.fractal_embedder:
        try:
            from eva_ai.reasoning.fractal_ml.fractal_retriever import FractalRetriever
            self.fractal_retriever = FractalRetriever(
                storage=self.fractal_storage,
                embedder=self.fractal_embedder
            )
            logger.info("FractalRetriever инициализирован")
        except Exception as e:
            logger.warning(f"Не удалось инициализировать Retriever: {e}")
