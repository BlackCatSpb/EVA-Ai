"""Модуль обработки запросов для ЕВА — обработчики запросов, форматирование ответов."""
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Placeholder for import compatibility — methods are on core class."""
    pass


class ResponseHandler:
    """Обработчик ответов, генерация, этика, противоречия, рассуждения."""

    def __init__(self, parent):
        self.parent = parent

    def generate_response(self, query: str, evidence: List, nlp_info: Dict,
                          concept: Optional[str], user_context: Optional[Dict]) -> Optional[str]:
        try:
            gen_context = {
                "nlp": nlp_info, "evidence": evidence,
                "concept": concept, "user_context": user_context
            }

            if not self.parent.brain.components.get("ml_unit"):
                logger.debug("ml_unit not available in components")
                return "Извините, модуль генерации недоступен."

            if self.parent.model is None:
                logger.debug("Self.model is None - model not loaded")
            if self.parent.tokenizer is None:
                logger.debug("Self.tokenizer is None - tokenizer not loaded")

            ml_unit = self.parent.brain.components.get("ml_unit")
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

    def build_response_from_nodes(self, result: Dict, nodes: List[Any], start_time: float) -> Dict[str, Any]:
        import time as _time
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
                "response": resp, "source": "knowledge_graph",
                "evidence": evidence_list, "metrics": {"time": _time.time() - start_time}
            })
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.debug(f"Ошибка при сборке ответа из узлов графа: {e}")
        return result

    def check_ethics(self, response: str, nlp_info: Dict, user_context: Optional[Dict]) -> Dict[str, Any]:
        ethics_result: Dict[str, Any] = {"score": 1.0, "violations": [], "recommendations": []}
        try:
            if self.parent.brain.components.get("ethics_framework") and hasattr(self.parent.brain.components["ethics_framework"], 'analyze_content'):
                try:
                    analysis = self.parent.brain.components["ethics_framework"].analyze_content(
                        response, context={"nlp": nlp_info, "user_context": user_context}
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
            logger.debug(f"Ошибка при попытке доступа к ethics_framework: {e}")
        return ethics_result

    def check_contradictions(self, query: str, response: str) -> List[Dict[str, Any]]:
        contradictions: List[Dict[str, Any]] = []
        try:
            if self.parent.brain.components.get("contradiction_resolver"):
                try:
                    resolver = self.parent.brain.components["contradiction_resolver"]
                    if hasattr(resolver, "check_response_contradictions"):
                        contradictions = resolver.check_response_contradictions(query, response)
                    elif hasattr(resolver, "get_active_contradictions"):
                        contradictions = resolver.get_active_contradictions()
                    else:
                        logger.debug("contradiction_resolver found but no compatible methods, skipping")
                except Exception as e:
                    logger.warning(f"Ошибка проверки противоречий: {e}")
            else:
                logger.debug("No contradiction_resolver available, skipping contradiction check")
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.debug(f"Ошибка при попытке доступа к contradiction_resolver: {e}")
        return contradictions

    def detect_ambiguity(self, query: str, nlp_result: Dict) -> Dict:
        if not self.parent.entity_extractor:
            return {"has_ambiguities": False, "clarifications": []}

        ambiguous = self.parent.entity_extractor.extract_ambiguous_terms(query)
        clarifications = []

        for entity in ambiguous:
            if self.parent.ambiguity_resolver:
                clarification = self.parent.ambiguity_resolver.generate_clarification(entity, query)
                clarifications.append({
                    "term": entity.term,
                    "question": clarification.question if hasattr(clarification, 'question') else None,
                    "possible_meanings": entity.possible_meanings
                })

        return {"has_ambiguities": len(clarifications) > 0, "clarifications": clarifications}

    def emit_metrics(self, metrics: List[Dict[str, Any]]):
        try:
            if getattr(self.parent, "brain", None):
                if hasattr(self.parent.brain, "events") and self.parent.brain.events:
                    self.parent.brain.events.trigger('metrics', metrics)
        except Exception as e:
            logger.debug(f"Ошибка в _emit_metrics: {e}")

    def store_insight(self, query: str, response: str, nlp_info: Dict, concept: Optional[str]):
        try:
            if not hasattr(self.parent.brain, 'memory_graph_ml') or not self.parent.brain.memory_graph_ml:
                return
            mgml = self.parent.brain.memory_graph_ml
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

    def store_conversation(self, query: str, response: str):
        try:
            if hasattr(self.parent.brain, 'memory_manager') and self.parent.brain.memory_manager:
                self.parent.brain.memory_manager.add_interaction(
                    user_id="default_user", query=query, response=response,
                    context={"source": "query_processor"}
                )
                logger.debug(f"Stored conversation in memory")
        except Exception as e:
            logger.debug(f"Error storing conversation: {e}")

    def get_conversation_context(self) -> List[Dict]:
        try:
            if self.parent.brain and hasattr(self.parent.brain, 'memory_manager') and self.parent.brain.memory_manager:
                memory_manager = self.parent.brain.memory_manager
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

    def add_reasoning_to_result(self, result: Dict[str, Any]):
        try:
            reasoning_text = self.get_reasoning_text()
            if reasoning_text:
                result["reasoning"] = reasoning_text
        except Exception as e:
            logger.debug(f"Error adding reasoning to result: {e}")

    def get_reasoning_text(self) -> str:
        try:
            if hasattr(self.parent.brain, 'reasoning_engine') and self.parent.brain.reasoning_engine:
                reasoning_engine = self.parent.brain.reasoning_engine
                if hasattr(reasoning_engine, 'process_query'):
                    try:
                        result = reasoning_engine.process_query(self.parent.current_query if hasattr(self.parent, 'current_query') else "")
                        if result:
                            if isinstance(result, dict):
                                return self._format_reasoning_dict(result)
                            elif isinstance(result, str):
                                return result
                    except Exception as e:
                        logger.debug(f"Error calling reasoning_engine.process_query: {e}")

                if hasattr(reasoning_engine, 'last_result') and reasoning_engine.last_result:
                    last_result = reasoning_engine.last_result
                    if isinstance(last_result, dict):
                        return self._format_reasoning_dict(last_result)

                if hasattr(reasoning_engine, 'dialogue') and reasoning_engine.dialogue:
                    if hasattr(reasoning_engine.dialogue, 'steps') and reasoning_engine.dialogue.steps:
                        steps = reasoning_engine.dialogue.steps
                        if steps:
                            return self._format_steps(steps)

            if hasattr(self.parent.brain, 'self_reasoning_engine') and self.parent.brain.self_reasoning_engine:
                sre = self.parent.brain.self_reasoning_engine
                if hasattr(sre, 'last_result') and sre.last_result:
                    last_result = sre.last_result
                    if isinstance(last_result, dict):
                        return self._format_reasoning_dict(last_result)
        except Exception as e:
            logger.debug(f"Error getting reasoning text: {e}")
        return ""

    def _format_reasoning_dict(self, reasoning_dict: Dict) -> str:
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
