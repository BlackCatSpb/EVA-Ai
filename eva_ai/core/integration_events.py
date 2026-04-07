"""
Integration Events - Event handling and pub/sub integration.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("eva_ai.integration")


def _setup_event_subscriptions(self):
    """Настройка подписок на события."""
    try:
        self.event_bus.subscribe(
            'query_received',
            self._handle_query_received,
            priority=10
        )

        self.event_bus.subscribe(
            'tokenize_request',
            self._handle_tokenize_request,
            priority=8
        )

        self.event_bus.subscribe(
            'tokens_ready',
            self._handle_tokens_ready,
            priority=7
        )

        self.event_bus.subscribe(
            'hot_window_ready',
            self._handle_hot_window_ready,
            priority=6
        )

        self.event_bus.subscribe(
            'response_generated',
            self._handle_response_generated,
            priority=5
        )

        self.event_bus.subscribe(
            'contradiction_detected',
            self._handle_contradiction_detected,
            priority=9
        )

        self.event_bus.subscribe(
            'learning_opportunity',
            self._handle_learning_opportunity,
            priority=4
        )

        self.event_bus.subscribe(
            'self_dialog_request',
            self._handle_self_dialog_request,
            priority=3
        )

        self.event_bus.subscribe(
            'ethical_check_request',
            self._handle_ethical_check_request,
            priority=8
        )
        
        self.event_bus.subscribe(
            'pipeline.start',
            self._handle_pipeline_start,
            priority=10
        )
        
        self.event_bus.subscribe(
            'pipeline.model_a.complete',
            self._handle_pipeline_model_a_complete,
            priority=9
        )
        
        self.event_bus.subscribe(
            'pipeline.web_search.complete',
            self._handle_pipeline_web_search_complete,
            priority=8
        )
        
        self.event_bus.subscribe(
            'pipeline.contradiction.check_complete',
            self._handle_pipeline_contradiction_check_complete,
            priority=8
        )
        
        self.event_bus.subscribe(
            'pipeline.ethics.check_complete',
            self._handle_pipeline_ethics_check_complete,
            priority=8
        )
        
        self.event_bus.subscribe(
            'pipeline.model_b.complete',
            self._handle_pipeline_model_b_complete,
            priority=7
        )
        
        self.event_bus.subscribe(
            'pipeline.relevance.check_complete',
            self._handle_pipeline_relevance_check_complete,
            priority=6
        )
        
        self.event_bus.subscribe(
            'pipeline.refinement.needed',
            self._handle_pipeline_refinement_needed,
            priority=7
        )
        
        self.event_bus.subscribe(
            'pipeline.refinement.attempt',
            self._handle_pipeline_refinement_attempt,
            priority=6
        )
        
        self.event_bus.subscribe(
            'pipeline.complete',
            self._handle_pipeline_complete,
            priority=5
        )
        
        self.event_bus.subscribe(
            'pipeline.failed',
            self._handle_pipeline_failed,
            priority=5
        )

        logger.info("Подписки на события настроены")

    except Exception as e:
        logger.error(f"Ошибка настройки подписок: {e}")


def _handle_pipeline_start(self, data: Dict[str, Any]):
    """Обработка начала pipeline."""
    try:
        logger.debug(f"Pipeline started: {data.get('query', '')[:50]}")
    except Exception as e:
        logger.error(f"Ошибка обработки pipeline.start: {e}")


def _handle_pipeline_model_a_complete(self, data: Dict[str, Any]):
    """Обработка завершения Model A."""
    try:
        logger.debug(f"Model A complete: {data.get('facts', '')[:50]}...")
    except Exception as e:
        logger.error(f"Ошибка обработки pipeline.model_a.complete: {e}")


def _handle_pipeline_web_search_complete(self, data: Dict[str, Any]):
    """Обработка завершения веб-поиска."""
    try:
        logger.debug(f"Web search complete: {data.get('results_count', 0)} results")
    except Exception as e:
        logger.error(f"Ошибка обработки pipeline.web_search.complete: {e}")


def _handle_pipeline_contradiction_check_complete(self, data: Dict[str, Any]):
    """Обработка проверки противоречий."""
    try:
        logger.debug(f"Contradiction check: {data.get('significant_count', 0)} significant")
    except Exception as e:
        logger.error(f"Ошибка обработки pipeline.contradiction.check_complete: {e}")


def _handle_pipeline_ethics_check_complete(self, data: Dict[str, Any]):
    """Обработка проверки этики."""
    try:
        logger.debug(f"Ethics check: violations={data.get('has_violations', False)}")
    except Exception as e:
        logger.error(f"Ошибка обработки pipeline.ethics.check_complete: {e}")


def _handle_pipeline_model_b_complete(self, data: Dict[str, Any]):
    """Обработка завершения Model B."""
    try:
        logger.debug(f"Model B complete: {data.get('response', '')[:50]}...")
    except Exception as e:
        logger.error(f"Ошибка обработки pipeline.model_b.complete: {e}")


def _handle_pipeline_relevance_check_complete(self, data: Dict[str, Any]):
    """Обработка проверки релевантности."""
    try:
        logger.debug(f"Relevance check: similarity={data.get('similarity', 0):.3f}, passes={data.get('passes', False)}")
    except Exception as e:
        logger.error(f"Ошибка обработки pipeline.relevance.check_complete: {e}")


def _handle_pipeline_refinement_needed(self, data: Dict[str, Any]):
    """Обработка необходимости уточнения."""
    try:
        logger.info(f"Refinement needed: {data.get('reason', 'unknown')}, count={data.get('count', 0)}")
    except Exception as e:
        logger.error(f"Ошибка обработки pipeline.refinement.needed: {e}")


def _handle_pipeline_refinement_attempt(self, data: Dict[str, Any]):
    """Обработка попытки уточнения."""
    try:
        logger.debug(f"Refinement attempt: {data.get('attempt', 0)}/{data.get('max_attempts', 0)}")
    except Exception as e:
        logger.error(f"Ошибка обработки pipeline.refinement.attempt: {e}")


def _handle_pipeline_complete(self, data: Dict[str, Any]):
    """Обработка успешного завершения pipeline."""
    try:
        logger.info(f"Pipeline complete: {data.get('query', '')[:50]}, time={data.get('processing_time', 0):.2f}s")
    except Exception as e:
        logger.error(f"Ошибка обработки pipeline.complete: {e}")


def _handle_pipeline_failed(self, data: Dict[str, Any]):
    """Обработка ошибки pipeline."""
    try:
        logger.warning(f"Pipeline failed: {data.get('reason', 'unknown')}")
    except Exception as e:
        logger.error(f"Ошибка обработки pipeline.failed: {e}")
