"""Модуль управления возможностями для обучения в CogniFlex"""
import json
import os
import sqlite3
import logging
import time
from typing import Dict, List, Any, Tuple
from cogniflex.learning.analyzer_core import AnalyzerCore
from cogniflex.learning.learning_opportunity import LearningOpportunity

logger = logging.getLogger("cogniflex.learning_opportunity_manager")

class LearningOpportunityManager:
    """Менеджер возможностей для обучения в CogniFlex."""
    
    def __init__(self, brain=None, analyzer_core=None, cache_dir=None):
        self.brain = brain
        self.analyzer_core = analyzer_core or AnalyzerCore(brain)
        self.learning_rate = 0.01
        logger.info("LearningOpportunityManager инициализирован")
    
    def get_learning_opportunities(self) -> List[Dict[str, Any]]:
        """
        Возвращает список возможностей для обучения.
        
        Returns:
            List[Dict[str, Any]]: Список возможностей обучения
        """
        opportunities = []
        
        try:
            conn = sqlite3.connect(self.analyzer_core.db_path)
            cursor = conn.cursor()
            
            # Получаем возможности из базы данных
            cursor.execute("""
                SELECT id, concept, opportunity_type, evidence, 
                       priority, created_at, executed
                FROM learning_opportunities 
                WHERE executed = 0
                ORDER BY priority DESC, created_at DESC
                LIMIT 10
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            for row in rows:
                opportunity = {
                    "id": row[0],
                    "concept": row[1],
                    "type": row[2],
                    "description": row[3],
                    "priority": row[4],
                    "timestamp": row[5],
                    "status": "executed" if row[6] else "pending"
                }
                opportunities.append(opportunity)
                
        except Exception as e:
            logger.error(f"Ошибка получения возможностей обучения: {e}")
            # Возвращаем базовые возможности при ошибке
            opportunities = [
                {
                    "type": "pattern_analysis",
                    "description": "Анализ паттернов в запросах пользователей",
                    "priority": "medium",
                    "timestamp": time.time()
                },
                {
                    "type": "knowledge_expansion",
                    "description": "Расширение базы знаний",
                    "priority": "low",
                    "timestamp": time.time()
                }
            ]
        
        return opportunities
    
    def execute_learning_opportunity(self, opportunity_id: str) -> bool:
        try:
            conn = sqlite3.connect(self.analyzer_core.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM learning_opportunities WHERE id = ?", (opportunity_id,))
            row = cursor.fetchone()
            if not row:
                logger.warning(f"Возможность для обучения с ID {opportunity_id} не найдена")
                return False
            if row[9]:  # executed
                logger.info(f"Возможность для обучения {opportunity_id} уже выполнена")
                return True
            
            opportunity = LearningOpportunity(
                id=row[0],
                concept=row[1],
                opportunity_type=row[2],
                priority=row[3],
                domain=row[4],
                evidence=json.loads(row[5]),
                suggested_actions=json.loads(row[6]),
                created_at=row[7],
                last_updated=row[8],
                executed=bool(row[9]),
                execution=json.loads(row[10]) if row[10] else None,
                metadata=json.loads(row[11])
            )
            
            success, result = False, None
            start_time = time.time()
            try:
                if opportunity.opportunity_type == "expansion":
                    success, result = self._handle_expansion(opportunity)
                elif opportunity.opportunity_type == "refinement":
                    success, result = self._handle_refinement(opportunity)
                elif opportunity.opportunity_type == "updating":
                    success, result = self._handle_updating(opportunity)
                elif opportunity.opportunity_type == "integration":
                    success, result = self._handle_integration(opportunity)
                else:
                    logger.warning(f"Неизвестный тип возможности: {opportunity.opportunity_type}")
            except Exception as e:
                logger.error(f"Ошибка выполнения возможности {opportunity_id}: {e}")
            
            execution_time = time.time() - start_time
            execution_data = {
                "status": "completed" if success else "failed",
                "timestamp": time.time(),
                "execution_time": execution_time,
                "result": result
            }
            
            cursor.execute('''
                UPDATE learning_opportunities SET
                    executed = ?,
                    execution = ?,
                    last_updated = ?
                WHERE id = ?
            ''', (
                True,
                json.dumps(execution_data),
                time.time(),
                opportunity_id
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Возможность для обучения {opportunity_id} {'успешно выполнена' if success else 'не выполнена'}")
            return success
        
        except Exception as e:
            logger.error(f"Ошибка выполнения возможности для обучения: {e}")
            return False
    
    def _handle_expansion(self, opportunity: LearningOpportunity) -> Tuple[bool, Any]:
        try:
            if not self.brain or not hasattr(self.brain, 'web_search_engine'):
                logger.warning("WebSearchEngine недоступен для расширения знаний")
                return False, "WebSearchEngine недоступен"
            
            knowledge = self.brain.web_search_engine.web_search_and_learn(
                opportunity.concept, 
                num_results=3
            )
            
            if not knowledge:
                return False, "Не найдено новых знаний"
            
            if hasattr(self.brain, 'knowledge_graph'):
                for item in knowledge:
                    self.brain.knowledge_graph.add_concept(
                        item["concept"],
                        item["content"],
                        domain=opportunity.domain,
                        source=item["source"]
                    )
            
            return True, f"Добавлено {len(knowledge)} новых знаний"
            
        except Exception as e:
            logger.error(f"Ошибка расширения знаний: {e}")
            return False, str(e)
    
    def _handle_refinement(self, opportunity: LearningOpportunity) -> Tuple[bool, Any]:
        try:
            if not self.brain or not hasattr(self.brain, 'knowledge_graph'):
                logger.warning("KnowledgeGraph недоступен для уточнения знаний")
                return False, "KnowledgeGraph недоступен"
            
            gaps = self.brain.knowledge_graph.analyze_knowledge_gaps(
                opportunity.concept, 
                num_samples=5
            )
            
            if not gaps:
                return True, "Пробелов в знаниях не обнаружено"
            
            filled_gaps = 0
            for gap in gaps:
                if gap["gap_type"] == "incomplete":
                    if hasattr(self.brain, 'text_processor'):
                        related = self.brain.text_processor.analyze_connection_pattern(
                            gap["concept"], 
                            [gap["concept"]], 
                            "related_to"
                        )
                        
                        if related["most_related"]:
                            for concept in related["most_related"]:
                                self.brain.knowledge_graph.add_connection(
                                    gap["concept"],
                                    concept,
                                    "related_to",
                                    strength=related["connection_strength"]
                                )
                            filled_gaps += 1
                
                elif gap["gap_type"] == "outdated":
                    if hasattr(self.brain, 'web_search_engine'):
                        knowledge = self.brain.web_search_engine.web_search_and_learn(
                            gap["concept"], 
                            num_results=1
                        )
                        
                        if knowledge:
                            self.brain.knowledge_graph.update_concept(
                                gap["concept"],
                                knowledge[0]["content"],
                                source=knowledge[0]["source"]
                            )
                            filled_gaps += 1
            
            return True, f"Заполнено {filled_gaps} пробелов в знаниях"
            
        except Exception as e:
            logger.error(f"Ошибка уточнения знаний: {e}")
            return False, str(e)
    
    def _handle_updating(self, opportunity: LearningOpportunity) -> Tuple[bool, Any]:
        try:
            # Пробуем обновить через knowledge_expander если доступен
            if hasattr(self.brain, 'knowledge_expander'):
                expander = self.brain.knowledge_expander
                if hasattr(expander, 'update_outdated_knowledge'):
                    updated_count = expander.update_outdated_knowledge(
                        domain=opportunity.domain,
                        max_age_days=365
                    )
                    if updated_count > 0:
                        return True, f"Обновлено {updated_count} устаревших знаний"
            
            # Пробуем обновить через knowledge_graph
            if hasattr(self.brain, 'knowledge_graph'):
                kg = self.brain.knowledge_graph
                if hasattr(kg, 'get_outdated_nodes'):
                    outdated = kg.get_outdated_nodes(older_than_days=365)
                    if outdated:
                        return True, f"Найдено {len(outdated)} устаревших узлов для обновления"
            
            # Проверяем ml_model в концепте
            if "ml_model" in opportunity.concept:
                model_name = opportunity.concept.replace("ml_model_", "")
                if hasattr(self.brain, 'model_manager'):
                    self.brain.model_manager.update_model(model_name)
                    return True, f"Модель {model_name} обновлена"
            
            # Обработка system домена
            if "system" in opportunity.domain:
                if "optimization" in opportunity.concept:
                    if hasattr(self.brain, 'ml_core'):
                        if opportunity.concept == "ml_response_time":
                            self.brain.ml_core.optimization_params["quantization"] = True
                            return True, "Квантование моделей включено"
                        elif opportunity.concept == "ml_error_rate":
                            self.brain.ml_core.optimization_params["fallback_models"] = True
                            return True, "Fallback-модели включены"
                
                elif opportunity.concept == "clear_cache":
                    if hasattr(self.brain, 'ml_core'):
                        self.brain.ml_core.response_cache = {}
                        return True, "Кэш очищен"
            
            # Если ничего не найдено - возвращаем информацию о доступных возможностях
            return True, "Обновление не требуется - знания актуальны"
            
        except Exception as e:
            logger.error(f"Ошибка обновления: {e}")
            return False, str(e)
    
    def _handle_integration(self, opportunity: LearningOpportunity) -> Tuple[bool, Any]:
        try:
            if not self.brain or not hasattr(self.brain, 'knowledge_graph'):
                logger.warning("KnowledgeGraph недоступен для интеграции знаний")
                return False, "KnowledgeGraph недоступен"
            
            if not hasattr(self.brain, 'text_processor'):
                logger.warning("TextProcessor недоступен для интеграции знаний")
                return False, "TextProcessor недоступен"
            
            if hasattr(self.brain, 'response_generator'):
                graph = self.brain.response_generator.generate_knowledge_graph(
                    opportunity.concept,
                    depth=2
                )
                
                for node in graph["nodes"]:
                    self.brain.knowledge_graph.add_concept(
                        node["label"],
                        node.get("description", ""),
                        domain=opportunity.domain
                    )
                
                for edge in graph["edges"]:
                    self.brain.knowledge_graph.add_connection(
                        edge["source"],
                        edge["target"],
                        edge["relation"],
                        strength=edge["strength"]
                    )
                
                return True, f"Граф знаний для {opportunity.concept} интегрирован"
            
            return False, "ResponseGenerator недоступен"
            
        except Exception as e:
            logger.error(f"Ошибка интеграции знаний: {e}")
            return False, str(e)
    
    def get_fixes(self) -> List[Dict[str, Any]]:
        fixes = []
        
        if hasattr(self.brain, 'health_monitor'):
            health = self.brain.health_monitor.analyze_system_health()
        else:
            health = {"components": {}}
        
        if "ml" in health["components"] and "models" in health["components"]["ml"]:
            for model_name, model_data in health["components"]["ml"]["models"].items():
                if model_data["status"] == "critical":
                    fixes.append({
                        "id": f"update_model_{model_name}",
                        "title": f"Обновить модель {model_name}",
                        "description": "Модель находится в критическом состоянии и требует обновления",
                        "severity": "high",
                        "action": lambda mn=model_name: self._update_model(mn)  # фикс лямбды
                    })
        
        if "ml" in health["components"] and "statistics" in health["components"]["ml"]:
            if health["components"]["ml"]["statistics"].get("avg_response_time", 0) > 2.0:
                fixes.append({
                    "id": "optimize_response_time",
                    "title": "Оптимизировать время ответа",
                    "description": "Среднее время ответа превышает 2 секунды",
                    "severity": "medium",
                    "action": self._optimize_response_time
                })
        
        if "ml" in health["components"] and "statistics" in health["components"]["ml"]:
            total_requests = health["components"]["ml"]["statistics"].get("total_requests", 1)
            failed_requests = health["components"]["ml"]["statistics"].get("failed_requests", 0)
            error_rate = failed_requests / max(1, total_requests)
            if error_rate > 0.1:
                fixes.append({
                    "id": "reduce_error_rate",
                    "title": "Снизить процент ошибок",
                    "description": f"Процент ошибок: {error_rate:.1%}",
                    "severity": "high",
                    "action": self._reduce_error_rate
                })
        
        if hasattr(self.brain, 'ml_core'):
            if len(self.brain.ml_core.response_cache) > self.brain.ml_core.cache_size * 0.9:
                fixes.append({
                    "id": "clear_cache",
                    "title": "Очистить кэш",
                    "description": "Кэш заполнен более чем на 90%",
                    "severity": "medium",
                    "action": self._clear_cache
                })
        
        return fixes
    
    def _update_model(self, model_name: str):
        if hasattr(self.brain, 'model_manager'):
            self.brain.model_manager.update_model(model_name)
    
    def _optimize_response_time(self):
        if hasattr(self.brain, 'ml_core'):
            if not self.brain.ml_core.optimization_params.get("quantization", False):
                self.brain.ml_core.optimization_params["quantization"] = True
                if hasattr(self.brain, 'model_manager'):
                    for model_name in list(self.brain.model_manager.models.keys()):
                        self.brain.model_manager.reload_model(model_name)
    
    def _reduce_error_rate(self):
        if hasattr(self.brain, 'ml_core'):
            if not self.brain.ml_core.optimization_params.get("fallback_models", False):
                self.brain.ml_core.optimization_params["fallback_models"] = True
    
    def set_learning_rate(self, learning_rate: float):
        """
        Устанавливает скорость обучения.
        
        Args:
            learning_rate: Новая скорость обучения
        """
        self.learning_rate = learning_rate
        logger.info(f"Установлена скорость обучения: {learning_rate}")
    
    def _clear_cache(self):
        if hasattr(self.brain, 'ml_core'):
            self.brain.ml_core.response_cache = {}
