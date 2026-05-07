"""Main contradiction learning class, initialization, lifecycle."""
import logging
import time
import json
import re
import random
import hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime, timedelta
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Lazy import for SentimentIntensityAnalyzer to avoid download errors
SentimentIntensityAnalyzer = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

from .learn_patterns import PatternExtractionMixin
from .learn_feedback import FeedbackProcessingMixin

logger = logging.getLogger("eva_ai.contradiction.learning")


class ContradictionLearningOpportunity(PatternExtractionMixin, FeedbackProcessingMixin):
    """Представляет возможность для обучения на основе противоречия."""
    
    def __init__(self, id: str, concept: str, type: str, priority: float,
                 description: str, evidence: List[str], 
                 suggested_actions: List[str], 
                 required_capabilities: Optional[List[str]] = None,
                 domain: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.id = id
        self.concept = concept
        self.type = type
        self.priority = priority
        self.description = description
        self.evidence = evidence
        self.suggested_actions = suggested_actions
        self.required_capabilities = required_capabilities or []
        self.domain = domain
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.last_updated = self.created_at
        self.status = "pending"
        self.progress = 0.0
        self.learning_tasks = []
        self.completed_tasks = []
        self.impact_assessment = {}
        self.learning_history = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует возможность обучения в словарь."""
        return {
            "id": self.id, "concept": self.concept, "type": self.type,
            "priority": self.priority, "description": self.description,
            "evidence": self.evidence, "suggested_actions": self.suggested_actions,
            "required_capabilities": self.required_capabilities, "domain": self.domain,
            "metadata": self.metadata, "created_at": self.created_at,
            "last_updated": self.last_updated, "status": self.status,
            "progress": self.progress, "learning_tasks": self.learning_tasks,
            "completed_tasks": self.completed_tasks,
            "impact_assessment": self.impact_assessment,
            "learning_history": self.learning_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContradictionLearningOpportunity':
        """Создает возможность обучения из словаря."""
        opportunity = cls(
            id=data["id"], concept=data["concept"], type=data["type"],
            priority=data["priority"], description=data["description"],
            evidence=data["evidence"], suggested_actions=data["suggested_actions"],
            required_capabilities=data.get("required_capabilities", []),
            domain=data.get("domain"), metadata=data.get("metadata", {})
        )
        opportunity.created_at = data.get("created_at", time.time())
        opportunity.last_updated = data.get("last_updated", opportunity.created_at)
        opportunity.status = data.get("status", "pending")
        opportunity.progress = data.get("progress", 0.0)
        opportunity.learning_tasks = data.get("learning_tasks", [])
        opportunity.completed_tasks = data.get("completed_tasks", [])
        opportunity.impact_assessment = data.get("impact_assessment", {})
        opportunity.learning_history = data.get("learning_history", [])
        return opportunity
    
    def assess_impact(self, knowledge_graph) -> Dict[str, float]:
        """Оценивает влияние обучения на систему знаний."""
        try:
            related_nodes = knowledge_graph.get_related_nodes(self.concept)
            node_count = len(related_nodes)
            depth = knowledge_graph.get_influence_depth(self.concept)
            importance = self._calculate_concept_importance(knowledge_graph)
            impact = (importance * 0.4) + (min(1.0, node_count / 50) * 0.3) + (min(1.0, depth / 10) * 0.3)
            self.impact_assessment = {
                "impact_score": min(1.0, impact), "node_count": node_count,
                "influence_depth": depth, "concept_importance": importance,
                "timestamp": time.time()
            }
            return self.impact_assessment
        except Exception as e:
            logger.error(f"Ошибка оценки влияния обучения: {e}")
            self.impact_assessment = {
                "impact_score": self.priority * 0.7, "node_count": 0,
                "influence_depth": 0, "concept_importance": self.priority,
                "timestamp": time.time()
            }
            return self.impact_assessment
    
    def _calculate_concept_importance(self, knowledge_graph) -> float:
        """Вычисляет важность концепта на основе его использования и значимости."""
        global SentimentIntensityAnalyzer
        if SentimentIntensityAnalyzer is None:
            try:
                from nltk.sentiment import SentimentIntensityAnalyzer as SIA
                SentimentIntensityAnalyzer = SIA
            except Exception:
                logger.warning("SentimentIntensityAnalyzer not available, skipping sentiment analysis")
                return 0.5  # default importance
        sentiment_analyzer = SentimentIntensityAnalyzer()
        stop_words = set(stopwords.words('english') + stopwords.words('russian'))
        
        usage_frequency = 0.3
        try:
            usage_count = knowledge_graph.get_concept_usage_count(self.concept)
            usage_frequency = min(0.5, 0.1 + np.log1p(usage_count) * 0.1)
        except Exception:
            pass
        
        concept_score = 0.0
        key_terms = ["наука", "технология", "искусственный интеллект", "этика", "безопасность",
                    "человек", "сознание", "разум", "жизнь", "вселенная", "время", "пространство"]
        
        related_nodes = knowledge_graph.get_related_nodes(self.concept)
        fact_content = " ".join(node.get("content", "") for node in related_nodes)
        fact_content = fact_content.lower()
        
        key_term_count = sum(1 for term in key_terms if term in fact_content)
        concept_score += min(0.3, key_term_count * 0.1)
        
        sentiment = sentiment_analyzer.polarity_scores(fact_content)
        neutrality = 1.0 - abs(sentiment['compound'])
        concept_score += neutrality * 0.2
        
        words = word_tokenize(fact_content)
        words = [word for word in words if word.isalnum() and word not in stop_words]
        unique_words = len(set(words))
        total_words = len(words)
        if total_words > 0:
            diversity = unique_words / total_words
            concept_score += min(0.2, diversity * 0.2)
        
        common_concepts = [
            "человек", "знание", "информация", "данные", "процесс", "система",
            "время", "пространство", "вселенная", "жизнь", "сознание", "разум"
        ]
        if self.concept.lower() in [c.lower() for c in common_concepts]:
            concept_score += 0.2
        
        importance = min(1.0, usage_frequency + concept_score)
        return importance
    
    def generate_learning_opportunity(self, contradiction, knowledge_graph) -> 'ContradictionLearningOpportunity':
        """Генерирует возможность для обучения на основе противоречия."""
        contradiction_type = self._get_contradiction_type(contradiction)
        domain = self._determine_domain(contradiction.concept, knowledge_graph)
        priority = contradiction.get_resolution_priority()
        evidence = [
            f"Обнаружено противоречие с уровнем расхождения {contradiction.divergence_level:.2f}",
            f"Тип противоречия: {contradiction_type}",
            f"Серьезность: {contradiction.get_severity()}"
        ]
        suggested_actions = [
            f"Исследовать '{contradiction.concept}' глубже",
            f"Проверить источники информации по '{contradiction.concept}'",
            f"Анализировать контекстные условия для разрешения противоречия"
        ]
        id = f"learn_{contradiction.contradiction_id}"
        return ContradictionLearningOpportunity(
            id=id, concept=contradiction.concept, type="contradiction_resolution",
            priority=priority, description=f"Разрешение противоречия в знаниях по '{contradiction.concept}'",
            evidence=evidence, suggested_actions=suggested_actions,
            required_capabilities=["knowledge_integration", "source_analysis"],
            domain=domain, metadata={
                "contradiction_id": contradiction.contradiction_id,
                "contradiction_type": contradiction_type,
                "divergence_level": contradiction.divergence_level,
                "severity": contradiction.get_severity()
            }
        )
    
    def get_learning_report(self) -> Dict[str, Any]:
        """Генерирует отчет о процессе обучения."""
        completed_count = len(self.completed_tasks)
        total_count = completed_count + len(self.learning_tasks)
        progress_details = []
        for task in self.completed_tasks:
            progress_details.append({
                "task": task["description"],
                "completion_time": task.get("completed_at", 0),
                "result": task.get("result", {})
            })
        return {
            "id": self.id, "concept": self.concept, "type": self.type,
            "priority": self.priority, "status": self.status,
            "progress": self.progress, "completed_tasks": completed_count,
            "total_tasks": total_count, "progress_details": progress_details,
            "impact_assessment": self.impact_assessment,
            "created_at": self.created_at, "last_updated": self.last_updated
        }


class ContradictionLearner:
    """Main class for managing contradiction-based learning in eva_ai."""
    
    def __init__(self, knowledge_graph=None, learning_rate: float = 0.1,
                 max_opportunities: int = 100):
        """
        Инициализирует обучатель противоречий.
        
        Args:
            knowledge_graph: Граф знаний системы
            learning_rate: Скорость обучения (0.0-1.0)
            max_opportunities: Максимальное количество возможностей обучения
        """
        self.knowledge_graph = knowledge_graph
        self.learning_rate = learning_rate
        self.max_opportunities = max_opportunities
        self.opportunities: Dict[str, ContradictionLearningOpportunity] = {}
        self.learning_history = []
        self.active = False
        logger.info("ContradictionLearner инициализирован")
    
    def create_opportunity(self, contradiction) -> ContradictionLearningOpportunity:
        """Создает возможность обучения из противоречия."""
        temp = ContradictionLearningOpportunity(
            id="", concept=contradiction.get("concept", "unknown"),
            type="contradiction_resolution", priority=0.5,
            description="", evidence=[], suggested_actions=[]
        )
        opportunity = temp.generate_learning_opportunity(contradiction, self.knowledge_graph)
        self.opportunities[opportunity.id] = opportunity
        if len(self.opportunities) > self.max_opportunities:
            oldest_id = min(self.opportunities, key=lambda k: self.opportunities[k].created_at)
            del self.opportunities[oldest_id]
        return opportunity
    
    def process_opportunity(self, opportunity_id: str) -> Dict[str, Any]:
        """Обрабатывает возможность обучения."""
        if opportunity_id not in self.opportunities:
            return {"error": "Opportunity not found"}
        
        opportunity = self.opportunities[opportunity_id]
        plan = opportunity.generate_learning_plan(self.knowledge_graph)
        for task in plan:
            opportunity.add_learning_task(task)
        
        opportunity.update_status("in_progress", progress=0.0)
        self.learning_history.append({
            "opportunity_id": opportunity_id,
            "action": "process",
            "timestamp": time.time(),
            "tasks_count": len(plan)
        })
        return {"status": "processed", "tasks": len(plan)}
    
    def apply_feedback(self, opportunity_id: str, feedback: Dict[str, Any]) -> bool:
        """Применяет пользовательскую обратную связь."""
        if opportunity_id not in self.opportunities:
            return False
        
        opportunity = self.opportunities[opportunity_id]
        if "task_id" in feedback and "result" in feedback:
            opportunity.complete_task(feedback["task_id"], feedback["result"])
        
        if "status" in feedback:
            opportunity.update_status(feedback["status"])
        
        self.learning_history.append({
            "opportunity_id": opportunity_id,
            "action": "feedback",
            "timestamp": time.time(),
            "feedback": feedback
        })
        return True
    
    def get_active_opportunities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Возвращает активные возможности обучения."""
        active = [o for o in self.opportunities.values() if o.status != "completed"]
        active.sort(key=lambda o: o.priority, reverse=True)
        return [o.to_dict() for o in active[:limit]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику обучения."""
        total = len(self.opportunities)
        completed = sum(1 for o in self.opportunities.values() if o.status == "completed")
        in_progress = sum(1 for o in self.opportunities.values() if o.status == "in_progress")
        pending = total - completed - in_progress
        
        by_type = defaultdict(int)
        for o in self.opportunities.values():
            by_type[o.type] += 1
        
        return {
            "total_opportunities": total,
            "completed": completed,
            "in_progress": in_progress,
            "pending": pending,
            "by_type": dict(by_type),
            "learning_history_count": len(self.learning_history),
            "learning_rate": self.learning_rate
        }
    
    def start(self):
        """Запускает обучатель."""
        self.active = True
        logger.info("ContradictionLearner запущен")
    
    def stop(self):
        """Останавливает обучатель."""
        self.active = False
        logger.info("ContradictionLearner остановлен")
