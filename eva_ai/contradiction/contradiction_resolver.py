"""
Основной модуль разрешения противоречий для ЕВА
Интегрирует все компоненты системы противоречий
"""
import logging
import time
from typing import Dict, List, Optional, Any
from .contradiction_core import OptimizedContradictionDetector, Contradiction
from .contradiction_resolution import ContradictionResolution

logger = logging.getLogger("eva_ai.contradiction.resolver")

class ContradictionResolver:
    """Основной класс для управления противоречиями в системе ЕВА"""
    
    def __init__(self, brain=None, knowledge_graph=None, cache_dir: Optional[str] = None):
        """
        Инициализирует резолвер противоречий
        
        Args:
            brain: Ссылка на ядро системы
            knowledge_graph: Ссылка на граф знаний
            cache_dir: Директория для кэша
        """
        self.brain = brain
        self.knowledge_graph = knowledge_graph
        self.cache_dir = cache_dir
        
        # Инициализируем детектор противоречий
        self.detector = OptimizedContradictionDetector(
            knowledge_graph=knowledge_graph,
            brain=brain,
            cache_dir=cache_dir
        )
        
        # Статистика работы
        self.stats = {
            "detected": 0,
            "resolved": 0,
            "failed_resolutions": 0,
            "last_scan": None
        }
        
        logger.info("ContradictionResolver инициализирован")

    def _emit_metrics(self, metrics: Dict[str, Any]) -> None:
        """Унифицированная отправка метрик: сначала через события, затем запасной путь.
        Схема: попытка brain.events.trigger("metrics", metrics) -> fallback brain.emit_metrics(metrics).
        Любые ошибки в этом пути не мешают основному потоку.
        """
        try:
            if not self.brain:
                return
            # Сначала пробуем через шину событий
            try:
                if hasattr(self.brain, "events") and hasattr(self.brain.events, "trigger"):
                    self.brain.events.trigger("metrics", metrics)
                    return
            except Exception as e:
                logger.debug(f"Не удалось отправить метрики через события: {e}", exc_info=True)
            # Запасной путь — прямой вызов emit_metrics
            try:
                if hasattr(self.brain, "emit_metrics"):
                    self.brain.emit_metrics(metrics)
            except Exception as e:
                logger.debug(f"Не удалось отправить метрики напрямую: {e}", exc_info=True)
        except Exception as e:
            logger.debug(f"Ошибка внутри _emit_metrics: {e}", exc_info=True)
    
    def start(self):
        """Запускает резолвер противоречий"""
        self.detector.start()
        logger.info("ContradictionResolver запущен")
    
    def stop(self):
        """Останавливает резолвер противоречий"""
        self.detector.stop()
        logger.info("ContradictionResolver остановлен")
    
    def detect_contradictions(self, concept: Optional[str] = None, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Обнаруживает противоречия в графе знаний
        
        Args:
            concept: Конкретный концепт для анализа
            domain: Домен для анализа
            
        Returns:
            List[Dict]: Список обнаруженных противоречий
        """
        try:
            start_time = time.time()
            contradictions = []
            
            if not self.knowledge_graph:
                logger.warning("Граф знаний недоступен для обнаружения противоречий")
                # Метрики о попытке без графа
                self._emit_metrics({
                    "type": "contradiction_detection",
                    "status": "no_knowledge_graph",
                    "concept": concept,
                    "domain": domain,
                    "count": 0,
                    "duration_ms": int((time.time() - start_time) * 1000),
                    "timestamp": int(time.time())
                })
                return contradictions
            
            # Если указан конкретный концепт
            if concept:
                contradictions.extend(self._analyze_concept_contradictions(concept))
            else:
                # Анализируем все узлы в графе
                nodes = self.knowledge_graph.get_all_nodes()
                for node in nodes[:50]:  # Ограничиваем для производительности
                    if domain and getattr(node, 'domain', 'general') != domain:
                        continue
                    if node and node.name:
                        contradictions.extend(self._analyze_concept_contradictions(node.name))
            
            self.stats["detected"] += len(contradictions)
            self.stats["last_scan"] = time.time()
            
            logger.info(f"Обнаружено {len(contradictions)} противоречий")
            # Отправляем метрики обнаружения
            self._emit_metrics({
                "type": "contradiction_detection",
                "status": "success",
                "concept": concept,
                "domain": domain,
                "count": len(contradictions),
                "duration_ms": int((time.time() - start_time) * 1000),
                "timestamp": int(time.time())
            })
            return contradictions
            
        except Exception as e:
            logger.error(f"Ошибка обнаружения противоречий: {e}", exc_info=True)
            # Метрики об ошибке
            self._emit_metrics({
                "type": "contradiction_detection",
                "status": "error",
                "concept": concept,
                "domain": domain,
                "error": str(e),
                "timestamp": int(time.time())
            })
            return []
    
    def _analyze_concept_contradictions(self, concept: str) -> List[Dict[str, Any]]:
        """Анализирует противоречия для конкретного концепта"""
        try:
            if not self.knowledge_graph:
                return []
            contradictions = []
            
            # Получаем все узлы, связанные с концептом
            related_nodes = self.knowledge_graph.search_nodes(concept, limit=10)
            
            if len(related_nodes) < 2:
                return contradictions
            
            # Группируем факты по концепту
            facts = []
            for node in related_nodes:
                fact = {
                    "value": node.description,
                    "source": getattr(node, 'meta', {}).get('source', 'unknown'),
                    "timestamp": getattr(node, 'timestamp', time.time()),
                    "strength": getattr(node, 'strength', 0.5)
                }
                facts.append(fact)
            
            # Пытаемся обнаружить противоречие
            contradiction = self.detector.detect_contradiction(
                concept=concept,
                facts=facts,
                metadata={"domain": getattr(related_nodes[0], 'domain', 'general')}
            )
            
            if contradiction:
                contradictions.append(contradiction.to_dict())
            
            return contradictions
            
        except Exception as e:
            logger.error(f"Ошибка анализа концепта {concept}: {e}", exc_info=True)
            return []
    
    def resolve_contradiction(self, contradiction_id: str, resolution_method: str = "auto") -> bool:
        """
        Разрешает противоречие
        
        Args:
            contradiction_id: ID противоречия
            resolution_method: Метод разрешения (auto, manual, weighted)
            
        Returns:
            bool: Успешно ли разрешено
        """
        try:
            start_time = time.time()
            # Получаем противоречие
            if contradiction_id not in self.detector.contradictions:
                logger.error(f"Противоречие {contradiction_id} не найдено")
                self._emit_metrics({
                    "type": "contradiction_resolution",
                    "status": "not_found",
                    "contradiction_id": contradiction_id,
                    "timestamp": int(time.time())
                })
                return False
            
            contradiction = self.detector.contradictions[contradiction_id]
            
            # Выбираем стратегию разрешения
            if resolution_method == "auto":
                resolution = self._auto_resolve(contradiction)
            elif resolution_method == "weighted":
                resolution = self._weighted_resolve(contradiction)
            else:
                resolution = self._manual_resolve(contradiction)
            
            if resolution:
                # Применяем разрешение
                success = self.detector.resolve_contradiction(
                    contradiction_id=contradiction_id,
                    resolution=resolution,
                    resolver="ContradictionResolver",
                    confidence=resolution.get("confidence", 0.7)
                )
                
                if success:
                    self.stats["resolved"] += 1
                    # Метрики об успешном разрешении
                    self._emit_metrics({
                        "type": "contradiction_resolution",
                        "status": "success",
                        "contradiction_id": contradiction_id,
                        "method": resolution.get("method"),
                        "confidence": resolution.get("confidence"),
                        "duration_ms": int((time.time() - start_time) * 1000),
                        "timestamp": int(time.time())
                    })
                    
                    # Уведомляем SelfAnalyzer о разрешении
                    if self.brain and hasattr(self.brain, 'self_analyzer'):
                        try:
                            self.brain.self_analyzer.add_learning_opportunity(
                                concept=f"resolved_contradiction_{contradiction.concept}",
                                opportunity_type="resolution",
                                priority=0.6,
                                domain=contradiction.metadata.get("domain", "general"),
                                evidence=[f"Разрешено противоречие: {contradiction_id}"],
                                suggested_actions=[
                                    "Обновить граф знаний с разрешенной информацией",
                                    "Проверить связанные концепты на наличие противоречий"
                                ]
                            )
                        except Exception as e:
                            logger.error(f"Ошибка уведомления SelfAnalyzer: {e}")
                    
                    logger.info(f"Противоречие {contradiction_id} успешно разрешено")
                    return True
                else:
                    self.stats["failed_resolutions"] += 1
                    self._emit_metrics({
                        "type": "contradiction_resolution",
                        "status": "failed_apply",
                        "contradiction_id": contradiction_id,
                        "method": resolution.get("method"),
                        "confidence": resolution.get("confidence"),
                        "duration_ms": int((time.time() - start_time) * 1000),
                        "timestamp": int(time.time())
                    })
                    return False
            else:
                self.stats["failed_resolutions"] += 1
                logger.warning(f"Не удалось создать разрешение для противоречия {contradiction_id}")
                self._emit_metrics({
                    "type": "contradiction_resolution",
                    "status": "no_resolution",
                    "contradiction_id": contradiction_id,
                    "method": resolution_method,
                    "timestamp": int(time.time())
                })
                return False
                
        except Exception as e:
            logger.error(f"Ошибка разрешения противоречия {contradiction_id}: {e}", exc_info=True)
            self.stats["failed_resolutions"] += 1
            self._emit_metrics({
                "type": "contradiction_resolution",
                "status": "error",
                "contradiction_id": contradiction_id,
                "error": str(e),
                "timestamp": int(time.time())
            })
            return False
    
    def _auto_resolve(self, contradiction: Contradiction) -> Optional[Dict[str, Any]]:
        """Автоматическое разрешение противоречия"""
        try:
            contradiction_type = contradiction.get_contradiction_type()
            
            if contradiction_type == "numeric_conflict":
                # Для числовых конфликтов берем среднее значение
                values = [
                    val for fact in contradiction.conflicting_facts
                    if (val := fact.get("value")) is not None and isinstance(val, (int, float))
                ]
                if values:
                    avg_value = sum(values) / len(values)
                    return {
                        "method": "average",
                        "resolved_value": avg_value,
                        "confidence": 0.7,
                        "reasoning": f"Использовано среднее значение из {len(values)} источников"
                    }
            
            elif contradiction_type == "boolean_conflict":
                # Для булевых конфликтов выбираем более надежный источник
                facts_with_strength = [(fact, fact.get("strength", 0.5)) 
                                     for fact in contradiction.conflicting_facts]
                best_fact = max(facts_with_strength, key=lambda x: x[1])
                return {
                    "method": "source_reliability",
                    "resolved_value": best_fact[0].get("value"),
                    "confidence": best_fact[1],
                    "reasoning": f"Выбран наиболее надежный источник (strength: {best_fact[1]})"
                }
            
            else:
                # Общий случай - выбираем факт с наибольшей силой
                best_fact = max(contradiction.conflicting_facts, 
                              key=lambda x: x.get("strength", 0.5))
                return {
                    "method": "strength_based",
                    "resolved_value": best_fact.get("value"),
                    "confidence": best_fact.get("strength", 0.5),
                    "reasoning": "Выбран факт с наибольшей силой"
                }
                
        except Exception as e:
            logger.error(f"Ошибка автоматического разрешения: {e}", exc_info=True)
            return None
    
    def _weighted_resolve(self, contradiction: Contradiction) -> Optional[Dict[str, Any]]:
        """Взвешенное разрешение противоречия"""
        try:
            # Учитываем силу, время и источник
            weighted_facts = []
            
            for fact in contradiction.conflicting_facts:
                weight = fact.get("strength", 0.5)
                
                # Учитываем время (более свежие факты важнее)
                timestamp = fact.get("timestamp", time.time())
                age_days = (time.time() - timestamp) / 86400
                time_weight = max(0.1, 1.0 - age_days / 365)  # Снижение веса за год
                
                # Учитываем источник
                source_weight = 1.0
                source = fact.get("source", "unknown")
                if "official" in source.lower() or "government" in source.lower():
                    source_weight = 1.2
                elif "wiki" in source.lower():
                    source_weight = 0.9
                elif "unknown" in source.lower():
                    source_weight = 0.5
                
                total_weight = weight * time_weight * source_weight
                weighted_facts.append((fact, total_weight, time_weight, source_weight))
            
            if not weighted_facts:
                return None

            # Выбираем факт с наибольшим весом
            best_fact, best_weight, best_time_weight, best_source_weight = max(weighted_facts, key=lambda x: x[1])
            
            return {
                "method": "weighted",
                "resolved_value": best_fact.get("value"),
                "confidence": min(0.95, best_weight),
                "reasoning": f"Взвешенный выбор (вес: {best_weight:.2f})",
                "weights": {
                    "strength": best_fact.get("strength", 0.5),
                    "time_factor": best_time_weight,
                    "source_factor": best_source_weight
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка взвешенного разрешения: {e}", exc_info=True)
            return None
    
    def _manual_resolve(self, contradiction: Contradiction) -> Optional[Dict[str, Any]]:
        """Подготовка для ручного разрешения"""
        return {
            "method": "manual",
            "status": "pending_manual_review",
            "confidence": 0.0,
            "reasoning": "Требуется ручное разрешение",
            "options": [
                {
                    "value": fact.get("value"),
                    "source": fact.get("source", "unknown"),
                    "strength": fact.get("strength", 0.5)
                }
                for fact in contradiction.conflicting_facts
            ]
        }
    
    def get_active_contradictions(self) -> List[Dict[str, Any]]:
        """Возвращает активные противоречия"""
        return self.detector.get_active_contradictions()
    
    def get_contradiction_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по противоречиям"""
        detector_stats = self.detector.get_contradiction_statistics()
        
        # Добавляем статистику резолвера
        detector_stats.update({
            "resolver_stats": self.stats,
            "resolution_rate": (
                self.stats["resolved"] / max(1, self.stats["detected"])
                if self.stats["detected"] > 0 else 0.0
            )
        })
        
        return detector_stats
    
    def generate_balanced_response(self, concept: str, language: str = "ru") -> str:
        """
        Генерирует сбалансированный ответ с учетом противоречий
        
        Args:
            concept: Концепт для анализа
            language: Язык ответа
            
        Returns:
            str: Сбалансированный ответ
        """
        try:
            # Ищем противоречия по концепту
            contradictions = []
            for contradiction in self.detector.contradictions.values():
                if concept.lower() in contradiction.concept.lower():
                    contradictions.append(contradiction)
            
            if not contradictions:
                return ""
            
            # Берем наиболее значимое противоречие
            main_contradiction = max(contradictions, key=lambda x: x.divergence_level)
            
            # Генерируем сбалансированный ответ
            return ContradictionResolution.generate_balanced_response(
                main_contradiction, language
            )
            
        except Exception as e:
            logger.error(f"Ошибка генерации сбалансированного ответа: {e}", exc_info=True)
            return ""
    
    def get_contradiction_summary(self) -> Dict[str, int]:
        """Возвращает краткую сводку по противоречиям"""
        return self.detector.get_contradiction_summary()
    
    def close(self):
        """Закрывает резолвер противоречий"""
        self.stop()
        logger.info("ContradictionResolver закрыт")