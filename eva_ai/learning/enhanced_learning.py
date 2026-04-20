"""
EnhancedLearningMixin - Продвинутые механизмы самообучения.

Включает:
1. Параллельные диалоговые роли
2. Запись паттернов ошибок
3. Проверка фактов в фоне
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("eva_ai.learning.enhanced_learning")


@dataclass
class ErrorPattern:
    """Структура для хранения статистики ошибок."""
    signature: str
    category: str
    count: int = 0
    last_seen: datetime = field(default_factory=datetime.now)


class EnhancedLearningMixin:
    """Миксин для продвинутых механизмов самообучения."""
    
    def __init__(self):
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self._fact_check_in_progress = False
    
    async def run_parallel_dialog_turn(
        self, 
        history: List[Dict], 
        query: str,
        roles: Dict[str, Any]
    ) -> Dict:
        """
        Выполняет анализ критика, рекомендации учителя и оценку наблюдателя параллельно.
        
        Args:
            history: История диалога.
            query: Текущий запрос.
            roles: Словарь ролей с методами analyze/recommend/evaluate.
            
        Returns:
            Словарь с результатами всех ролей.
        """
        if not roles:
            return {'error': 'No roles provided'}
        
        tasks = {}
        role_keys = []
        
        for role_name, role_handler in roles.items():
            if callable(role_handler):
                tasks[role_name] = asyncio.create_task(
                    self._safe_role_call(role_handler, history, query)
                )
                role_keys.append(role_name)
        
        if not tasks:
            return {'error': 'No callable roles'}
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        output = {}
        for key, res in zip(role_keys, results):
            if isinstance(res, Exception):
                output[key] = f"Error in {key}: {str(res)}"
            else:
                output[key] = res
                
        return output

    async def _safe_role_call(self, role_handler, history: List[Dict], query: str) -> str:
        """Безопасный вызов роли с таймаутом."""
        try:
            if asyncio.iscoroutinefunction(role_handler):
                return await asyncio.wait_for(role_handler(history, query), timeout=30.0)
            else:
                return role_handler(history, query)
        except asyncio.TimeoutError:
            return f"Timeout in {role_handler}"
        except Exception as e:
            logger.debug(f"Role call error: {e}")
            return str(e)

    def record_error_pattern(self, response_text: str, error_type: str):
        """
        Записывает паттерн ошибки для предотвращения повторения.
        
        Args:
            response_text: Текст ответа, содержащего ошибку.
            error_type: Тип ошибки ('repetition', 'logic', 'fact').
        """
        if not response_text:
            return
            
        signature = f"{error_type}:{response_text[:50].strip()}"
        
        if signature not in self.error_patterns:
            self.error_patterns[signature] = ErrorPattern(
                signature=signature, 
                category=error_type
            )
        
        self.error_patterns[signature].count += 1
        self.error_patterns[signature].last_seen = datetime.now()

    def get_system_prompt_with_error_avoidance(self, base_prompt: str) -> str:
        """
        Модифицирует системный промпт, добавляя инструкции по избеганию частых ошибок.
        
        Args:
            base_prompt: Базовый системный промпт.
            
        Returns:
            Усиленный промпт.
        """
        if not self.error_patterns:
            return base_prompt
        
        sorted_errors = sorted(
            self.error_patterns.values(), 
            key=lambda x: x.count, 
            reverse=True
        )[:5]
        
        avoidance_instructions = "\n\nВАЖНЫЕ ОГРАНИЧЕНИЯ (на основе прошлого опыта):\n"
        
        for err in sorted_errors:
            if err.category == 'repetition':
                avoidance_instructions += "- Избегай повторения одних и тех же фраз и структур.\n"
            elif err.category == 'logic':
                avoidance_instructions += "- Внимательно проверяй логические цепочки перед выводом.\n"
            elif err.category == 'fact':
                avoidance_instructions += "- Не утверждай факты как истину, если не уверен на 100%.\n"
            else:
                avoidance_instructions += f"- Избегай ошибок типа '{err.category}'.\n"
        
        return base_prompt + avoidance_instructions

    async def background_fact_checking_loop(
        self, 
        interval_seconds: int = 3600,
        fractal_graph=None,
        web_search=None
    ):
        """
        Фоновый цикл проверки фактов для узлов с низкой уверенностью.
        
        Args:
            interval_seconds: Интервал между проверками (по умолчанию 1 час).
            fractal_graph: Инстанс графа для проверки узлов.
            web_search: Функция веб-поиска.
        """
        if self._fact_check_in_progress:
            return
            
        self._fact_check_in_progress = True
        
        try:
            while True:
                await asyncio.sleep(interval_seconds)
                
                if not fractal_graph:
                    try:
                        from eva_ai.memory.fractal_graph_v2 import FractalGraphV2
                        fractal_graph = FractalGraphV2.get_instance()
                    except Exception:
                        break
                
                if not fractal_graph:
                    break
                    
                try:
                    low_conf_nodes = fractal_graph.get_nodes_by_confidence(max_conf=0.6, limit=5)
                except Exception:
                    logger.debug("get_nodes_by_confidence not available")
                    break
                
                for node in low_conf_nodes:
                    try:
                        content = node.content if hasattr(node, 'content') else str(node)
                        
                        if web_search:
                            search_results = await web_search(content, max_results=2)
                            
                            if self._verify_consistency(content, search_results):
                                fractal_graph.update_node_confidence(node.id, new_conf=0.85)
                            else:
                                fractal_graph.add_tag(node.id, "needs_review")
                        else:
                            logger.debug(f"No web_search, skipping fact check for {node.id}")
                            
                    except Exception as e:
                        logger.debug(f"Fact check failed for node: {e}")
                        
        except Exception as e:
            logger.error(f"Fact checking loop error: {e}")
        finally:
            self._fact_check_in_progress = False

    def _verify_consistency(self, claim: str, search_results: List[str]) -> bool:
        """Простая эвристика проверки соответствия поиска утверждению."""
        if not search_results:
            return False
            
        claim_words = set(claim.lower().split())
        if not claim_words:
            return False
            
        for res in search_results:
            res_words = set(res.lower().split())
            if len(claim_words & res_words) > len(claim_words) * 0.6:
                return True
        return False