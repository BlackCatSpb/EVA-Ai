
import logging
import time
import json
import re
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime, timedelta
import random
import hashlib
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional dependencies
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    SentenceTransformer = None  # type: ignore
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

logger = logging.getLogger("cogniflex.contradiction.learning")

class ContradictionLearningOpportunity:
    """Представляет возможность для обучения на основе противоречия."""
    
    def __init__(self, id: str, concept: str, type: str, priority: float,
                 description: str, evidence: List[str], 
                 suggested_actions: List[str], 
                 required_capabilities: Optional[List[str]] = None,
                 domain: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Инициализирует возможность для обучения.
        
        Args:
            id: Уникальный ID возможности
            concept: Концепт, связанный с возможностью
            type: Тип возможности
            priority: Приоритет (0.0-1.0)
            description: Описание возможности
            evidence: Свидетельства
            suggested_actions: Предлагаемые действия
            required_capabilities: Требуемые возможности системы
            domain: Домен знаний
            metadata: Дополнительные метаданные
        """
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
            "id": self.id,
            "concept": self.concept,
            "type": self.type,
            "priority": self.priority,
            "description": self.description,
            "evidence": self.evidence,
            "suggested_actions": self.suggested_actions,
            "required_capabilities": self.required_capabilities,
            "domain": self.domain,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "status": self.status,
            "progress": self.progress,
            "learning_tasks": self.learning_tasks,
            "completed_tasks": self.completed_tasks,
            "impact_assessment": self.impact_assessment,
            "learning_history": self.learning_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContradictionLearningOpportunity':
        """Создает возможность обучения из словаря."""
        opportunity = cls(
            id=data["id"],
            concept=data["concept"],
            type=data["type"],
            priority=data["priority"],
            description=data["description"],
            evidence=data["evidence"],
            suggested_actions=data["suggested_actions"],
            required_capabilities=data.get("required_capabilities", []),
            domain=data.get("domain"),
            metadata=data.get("metadata", {})
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
    
    def update_status(self, new_status: str, progress: Optional[float] = None):
        """
        Обновляет статус возможности обучения.
        
        Args:
            new_status: Новый статус
            progress: Прогресс (опционально)
        """
        self.status = new_status
        self.last_updated = time.time()
        
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))
        
        # Добавляем запись в историю
        self.learning_history.append({
            "status": new_status,
            "progress": self.progress,
            "timestamp": self.last_updated
        })
    
    def add_learning_task(self, task: Dict[str, Any]):
        """
        Добавляет задачу обучения.
        
        Args:
            task: Задача обучения
        """
        self.learning_tasks.append(task)
        self.last_updated = time.time()
    
    def complete_task(self, task_id: str, result: Dict[str, Any]):
        """
        Помечает задачу как выполненную.
        
        Args:
            task_id: ID задачи
            result: Результат выполнения
        """
        for i, task in enumerate(self.learning_tasks):
            if task["task_id"] == task_id:
                completed_task = task.copy()
                completed_task["completed_at"] = time.time()
                completed_task["result"] = result
                self.completed_tasks.append(completed_task)
                del self.learning_tasks[i]
                
                # Обновляем прогресс
                total_tasks = len(self.completed_tasks) + len(self.learning_tasks)
                if total_tasks > 0:
                    self.progress = len(self.completed_tasks) / total_tasks
                
                self.last_updated = time.time()
                break
    
    def get_progress_report(self) -> Dict[str, Any]:
        """
        Возвращает отчет о прогрессе.
        
        Returns:
            Dict: Отчет о прогрессе
        """
        return {
            "id": self.id,
            "concept": self.concept,
            "type": self.type,
            "priority": self.priority,
            "status": self.status,
            "progress": self.progress,
            "total_tasks": len(self.learning_tasks) + len(self.completed_tasks),
            "completed_tasks": len(self.completed_tasks),
            "pending_tasks": len(self.learning_tasks),
            "impact_assessment": self.impact_assessment,
            "last_updated": self.last_updated
        }
    
    def assess_impact(self, knowledge_graph) -> Dict[str, float]:
        """
        Оценивает влияние обучения на систему знаний.
        
        Args:
            knowledge_graph: Граф знаний
            
        Returns:
            Dict: Оценка влияния
        """
        try:
            # Получаем количество связанных узлов
            related_nodes = knowledge_graph.get_related_nodes(self.concept)
            node_count = len(related_nodes)
            
            # Получаем глубину влияния
            depth = knowledge_graph.get_influence_depth(self.concept)
            
            # Оцениваем важность концепта
            importance = self._calculate_concept_importance(knowledge_graph)
            
            # Влияние = (важность * 0.4) + (количество узлов * 0.3) + (глубина * 0.3)
            impact = (importance * 0.4) + (min(1.0, node_count / 50) * 0.3) + (min(1.0, depth / 10) * 0.3)
            
            # Сохраняем оценку
            self.impact_assessment = {
                "impact_score": min(1.0, impact),
                "node_count": node_count,
                "influence_depth": depth,
                "concept_importance": importance,
                "timestamp": time.time()
            }
            
            return self.impact_assessment
            
        except Exception as e:
            logger.error(f"Ошибка оценки влияния обучения: {e}")
            # Базовая оценка влияния
            self.impact_assessment = {
                "impact_score": self.priority * 0.7,
                "node_count": 0,
                "influence_depth": 0,
                "concept_importance": self.priority,
                "timestamp": time.time()
            }
            return self.impact_assessment
    
    def _calculate_concept_importance(self, knowledge_graph) -> float:
        """
        Вычисляет важность концепта на основе его использования и значимости.
        
        Args:
            knowledge_graph: Граф знаний
            
        Returns:
            float: Важность концепта (0.0-1.0)
        """
        # Инициализируем анализатор тональности
        sentiment_analyzer = SentimentIntensityAnalyzer()
        stop_words = set(stopwords.words('english') + stopwords.words('russian'))
        
        # 1. Анализ частоты использования концепта в системе
        usage_frequency = 0.3  # Базовая оценка
        try:
            usage_count = knowledge_graph.get_concept_usage_count(self.concept)
            # Логарифмическая шкала для учета частоты использования
            usage_frequency = min(0.5, 0.1 + np.log1p(usage_count) * 0.1)
        except Exception:
            pass
        
        # 2. Анализ важности концепта через контент
        concept_score = 0.0
        key_terms = ["наука", "технология", "искусственный интеллект", "этика", "безопасность",
                    "человек", "сознание", "разум", "жизнь", "вселенная", "время", "пространство"]
        
        # Проверяем содержимое связанных узлов
        related_nodes = knowledge_graph.get_related_nodes(self.concept)
        fact_content = " ".join(node.get("content", "") for node in related_nodes)
        fact_content = fact_content.lower()
        
        # Считаем вхождения ключевых терминов
        key_term_count = sum(1 for term in key_terms if term in fact_content)
        concept_score += min(0.3, key_term_count * 0.1)
        
        # 3. Анализ тональности контента
        sentiment = sentiment_analyzer.polarity_scores(fact_content)
        neutrality = 1.0 - abs(sentiment['compound'])
        concept_score += neutrality * 0.2
        
        # 4. Анализ структуры текста
        words = word_tokenize(fact_content)
        words = [word for word in words if word.isalnum() and word not in stop_words]
        unique_words = len(set(words))
        total_words = len(words)
        
        if total_words > 0:
            diversity = unique_words / total_words
            concept_score += min(0.2, diversity * 0.2)
        
        # 5. Проверка на общий/специфичный концепт
        common_concepts = [
            "человек", "знание", "информация", "данные", "процесс", "система",
            "время", "пространство", "вселенная", "жизнь", "сознание", "разум"
        ]
        
        if self.concept.lower() in [c.lower() for c in common_concepts]:
            concept_score += 0.2
        
        # Общая важность концепта
        importance = min(1.0, usage_frequency + concept_score)
        return importance
    
    def generate_learning_plan(self, knowledge_graph) -> List[Dict[str, Any]]:
        """
        Генерирует план обучения для разрешения противоречия.
        
        Args:
            knowledge_graph: Граф знаний
            
        Returns:
            List: План обучения
        """
        # Определяем тип противоречия
        contradiction_type = self._determine_contradiction_type()
        
        # Генерируем задачи в зависимости от типа
        if contradiction_type == "numeric_conflict":
            return self._generate_numeric_conflict_learning_plan(knowledge_graph)
        elif contradiction_type == "boolean_conflict":
            return self._generate_boolean_conflict_learning_plan(knowledge_graph)
        elif contradiction_type == "exclusivity_conflict":
            return self._generate_exclusivity_conflict_learning_plan(knowledge_graph)
        elif contradiction_type == "hierarchy_conflict":
            return self._generate_hierarchy_conflict_learning_plan(knowledge_graph)
        elif contradiction_type == "response_conflict":
            return self._generate_response_conflict_learning_plan(knowledge_graph)
        else:
            return self._generate_general_learning_plan(knowledge_graph)
    
    def _determine_contradiction_type(self) -> str:
        """
        Определяет тип противоречия на основе метаданных.
        
        Returns:
            str: Тип противоречия
        """
        if "contradiction_type" in self.metadata:
            return self.metadata["contradiction_type"]
        
        # Определяем тип по описанию
        description = self.description.lower()
        
        if "число" in description or "значение" in description:
            return "numeric_conflict"
        elif "да" in description or "нет" in description or "верно" in description or "неверно" in description:
            return "boolean_conflict"
        elif "только" in description or "не только" in description:
            return "exclusivity_conflict"
        elif "иерархия" in description or "классификация" in description:
            return "hierarchy_conflict"
        elif "ответ" in description or "вопрос" in description:
            return "response_conflict"
        
        return "general"
    
    def _generate_numeric_conflict_learning_plan(self, knowledge_graph) -> List[Dict[str, Any]]:
        """Генерирует план обучения для числового противоречия."""
        tasks = [
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_numeric_1'.encode()).hexdigest()[:8]}",
                "task_type": "evidence_collection",
                "description": "Сбор данных из авторитетных источников для определения наиболее точного значения",
                "priority": self.priority * 0.9,
                "scheduled_time": time.time() + 60,  # Через 1 минуту
                "required_capabilities": ["web_search"],
                "parameters": {
                    "query": f"{self.concept} точное значение",
                    "num_results": 10
                },
                "dependencies": []
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_numeric_2'.encode()).hexdigest()[:8]}",
                "task_type": "analysis",
                "description": "Анализ методов измерения и определение контекстных условий",
                "priority": self.priority * 0.8,
                "scheduled_time": time.time() + 300,  # Через 5 минут
                "required_capabilities": ["nlp_analysis", "source_reputation"],
                "parameters": {
                    "analysis_type": "methodology_comparison"
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_numeric_1'.encode()).hexdigest()[:8]}"]
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_numeric_3'.encode()).hexdigest()[:8]}",
                "task_type": "integration",
                "description": "Интеграция данных и определение наиболее точного значения с учетом контекста",
                "priority": self.priority * 0.7,
                "scheduled_time": time.time() + 1800,  # Через 30 минут
                "required_capabilities": ["knowledge_integration"],
                "parameters": {
                    "integration_strategy": "contextual_weighting"
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_numeric_2'.encode()).hexdigest()[:8]}"]
            }
        ]
        
        return tasks
    
    def _generate_boolean_conflict_learning_plan(self, knowledge_graph) -> List[Dict[str, Any]]:
        """Генерирует план обучения для булева противоречия."""
        tasks = [
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_boolean_1'.encode()).hexdigest()[:8]}",
                "task_type": "context_analysis",
                "description": "Анализ условий, при которых каждое утверждение является верным",
                "priority": self.priority * 0.9,
                "scheduled_time": time.time() + 60,  # Через 1 минуту
                "required_capabilities": ["nlp_analysis", "context_detection"],
                "parameters": {
                    "analysis_type": "condition_extraction"
                },
                "dependencies": []
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_boolean_2'.encode()).hexdigest()[:8]}",
                "task_type": "evidence_collection",
                "description": "Сбор доказательств для подтверждения условий применения",
                "priority": self.priority * 0.8,
                "scheduled_time": time.time() + 300,  # Через 5 минут
                "required_capabilities": ["web_search"],
                "parameters": {
                    "query": f"{self.concept} условия применения",
                    "num_results": 8
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_boolean_1'.encode()).hexdigest()[:8]}"]
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_boolean_3'.encode()).hexdigest()[:8]}",
                "task_type": "synthesis",
                "description": "Создание контекстно-зависимого ответа",
                "priority": self.priority * 0.7,
                "scheduled_time": time.time() + 1800,  # Через 30 минут
                "required_capabilities": ["knowledge_integration", "response_generation"],
                "parameters": {
                    "synthesis_strategy": "conditional_response"
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_boolean_2'.encode()).hexdigest()[:8]}"]
            }
        ]
        
        return tasks
    
    def _generate_exclusivity_conflict_learning_plan(self, knowledge_graph) -> List[Dict[str, Any]]:
        """Генерирует план обучения для противоречия эксклюзивности."""
        tasks = [
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_exclusivity_1'.encode()).hexdigest()[:8]}",
                "task_type": "taxonomy_analysis",
                "description": "Анализ иерархии категорий и подкатегорий",
                "priority": self.priority * 0.9,
                "scheduled_time": time.time() + 60,  # Через 1 минуту
                "required_capabilities": ["taxonomy_analysis"],
                "parameters": {
                    "analysis_type": "category_hierarchy"
                },
                "dependencies": []
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_exclusivity_2'.encode()).hexdigest()[:8]}",
                "task_type": "evidence_collection",
                "description": "Сбор данных для определения взаимоисключающих условий",
                "priority": self.priority * 0.8,
                "scheduled_time": time.time() + 300,  # Через 5 минут
                "required_capabilities": ["web_search"],
                "parameters": {
                    "query": f"{self.concept} взаимоисключающие условия",
                    "num_results": 7
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_exclusivity_1'.encode()).hexdigest()[:8]}"]
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_exclusivity_3'.encode()).hexdigest()[:8]}",
                "task_type": "taxonomy_revision",
                "description": "Переработка иерархии категорий и разделение на подкатегории",
                "priority": self.priority * 0.7,
                "scheduled_time": time.time() + 1800,  # Через 30 минут
                "required_capabilities": ["knowledge_integration", "taxonomy_management"],
                "parameters": {
                    "revision_strategy": "subcategorization"
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_exclusivity_2'.encode()).hexdigest()[:8]}"]
            }
        ]
        
        return tasks
    
    def _generate_hierarchy_conflict_learning_plan(self, knowledge_graph) -> List[Dict[str, Any]]:
        """Генерирует план обучения для иерархического противоречия."""
        tasks = [
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_hierarchy_1'.encode()).hexdigest()[:8]}",
                "task_type": "hierarchy_analysis",
                "description": "Анализ иерархических связей и выявление циклических зависимостей",
                "priority": self.priority * 0.9,
                "scheduled_time": time.time() + 60,  # Через 1 минуту
                "required_capabilities": ["knowledge_graph_analysis"],
                "parameters": {
                    "analysis_type": "cycle_detection"
                },
                "dependencies": []
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_hierarchy_2'.encode()).hexdigest()[:8]}",
                "task_type": "taxonomy_review",
                "description": "Анализ взаимоисключающих классификаций",
                "priority": self.priority * 0.8,
                "scheduled_time": time.time() + 300,  # Через 5 минут
                "required_capabilities": ["taxonomy_analysis"],
                "parameters": {
                    "review_type": "classification_conflicts"
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_hierarchy_1'.encode()).hexdigest()[:8]}"]
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_hierarchy_3'.encode()).hexdigest()[:8]}",
                "task_type": "hierarchy_reconstruction",
                "description": "Перестройка иерархии для устранения противоречий",
                "priority": self.priority * 0.7,
                "scheduled_time": time.time() + 1800,  # Через 30 минут
                "required_capabilities": ["knowledge_integration", "taxonomy_management"],
                "parameters": {
                    "reconstruction_strategy": "hierarchy_flattening"
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_hierarchy_2'.encode()).hexdigest()[:8]}"]
            }
        ]
        
        return tasks
    
    def _generate_response_conflict_learning_plan(self, knowledge_graph) -> List[Dict[str, Any]]:
        """Генерирует план обучения для противоречия в ответах."""
        tasks = [
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_response_1'.encode()).hexdigest()[:8]}",
                "task_type": "audience_analysis",
                "description": "Анализ целевой аудитории и контекстных условий",
                "priority": self.priority * 0.9,
                "scheduled_time": time.time() + 60,  # Через 1 минуту
                "required_capabilities": ["audience_analysis", "context_detection"],
                "parameters": {
                    "analysis_type": "audience_segmentation"
                },
                "dependencies": []
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_response_2'.encode()).hexdigest()[:8]}",
                "task_type": "response_comparison",
                "description": "Сравнение ответов и определение границ применимости",
                "priority": self.priority * 0.8,
                "scheduled_time": time.time() + 300,  # Через 5 минут
                "required_capabilities": ["nlp_analysis", "semantic_comparison"],
                "parameters": {
                    "comparison_type": "contextual_boundaries"
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_response_1'.encode()).hexdigest()[:8]}"]
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_response_3'.encode()).hexdigest()[:8]}",
                "task_type": "response_integration",
                "description": "Создание условных ответов для разных сценариев",
                "priority": self.priority * 0.7,
                "scheduled_time": time.time() + 1800,  # Через 30 минут
                "required_capabilities": ["response_generation", "context_management"],
                "parameters": {
                    "integration_strategy": "scenario_based"
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_response_2'.encode()).hexdigest()[:8]}"]
            }
        ]
        
        return tasks
    
    def _generate_general_learning_plan(self, knowledge_graph) -> List[Dict[str, Any]]:
        """Генерирует общий план обучения."""
        tasks = [
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_general_1'.encode()).hexdigest()[:8]}",
                "task_type": "comprehensive_analysis",
                "description": "Определение типа противоречия и анализ источников информации",
                "priority": self.priority * 0.9,
                "scheduled_time": time.time() + 60,  # Через 1 минуту
                "required_capabilities": ["contradiction_analysis"],
                "parameters": {
                    "analysis_type": "contradiction_typing"
                },
                "dependencies": []
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_general_2'.encode()).hexdigest()[:8]}",
                "task_type": "context_research",
                "description": "Исследование контекстных условий и сбор дополнительных данных",
                "priority": self.priority * 0.8,
                "scheduled_time": time.time() + 300,  # Через 5 минут
                "required_capabilities": ["web_search", "context_analysis"],
                "parameters": {
                    "research_focus": "contextual_conditions"
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_general_1'.encode()).hexdigest()[:8]}"]
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_general_3'.encode()).hexdigest()[:8]}",
                "task_type": "solution_synthesis",
                "description": "Синтез решения и документирование",
                "priority": self.priority * 0.7,
                "scheduled_time": time.time() + 1800,  # Через 30 минут
                "required_capabilities": ["knowledge_integration", "solution_synthesis"],
                "parameters": {
                    "synthesis_type": "comprehensive"
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_general_2'.encode()).hexdigest()[:8]}"]
            }
        ]
        
        return tasks
    
    def get_learning_report(self) -> Dict[str, Any]:
        """
        Генерирует отчет о процессе обучения.
        
        Returns:
            Dict: Отчет о процессе обучения
        """
        completed_count = len(self.completed_tasks)
        total_count = completed_count + len(self.learning_tasks)
        
        # Анализируем прогресс
        progress_details = []
        for task in self.completed_tasks:
            progress_details.append({
                "task": task["description"],
                "completion_time": task.get("completed_at", 0),
                "result": task.get("result", {})
            })
        
        return {
            "id": self.id,
            "concept": self.concept,
            "type": self.type,
            "priority": self.priority,
            "status": self.status,
            "progress": self.progress,
            "completed_tasks": completed_count,
            "total_tasks": total_count,
            "progress_details": progress_details,
            "impact_assessment": self.impact_assessment,
            "created_at": self.created_at,
            "last_updated": self.last_updated
        }
    
    def generate_learning_opportunity(self, contradiction, knowledge_graph) -> 'ContradictionLearningOpportunity':
        """
        Генерирует возможность для обучения на основе противоречия.
        
        Args:
            contradiction: Противоречие
            knowledge_graph: Граф знаний
            
        Returns:
            ContradictionLearningOpportunity: Возможность для обучения
        """
        # Определяем тип противоречия
        contradiction_type = self._get_contradiction_type(contradiction)
        
        # Определяем домен
        domain = self._determine_domain(contradiction.concept, knowledge_graph)
        
        # Определяем приоритет
        priority = contradiction.get_resolution_priority()
        
        # Формируем свидетельства
        evidence = [
            f"Обнаружено противоречие с уровнем расхождения {contradiction.divergence_level:.2f}",
            f"Тип противоречия: {contradiction_type}",
            f"Серьезность: {contradiction.get_severity()}"
        ]
        
        # Формируем предполагаемые действия
        suggested_actions = [
            f"Исследовать '{contradiction.concept}' глубже",
            f"Проверить источники информации по '{contradiction.concept}'",
            f"Анализировать контекстные условия для разрешения противоречия"
        ]
        
        # Создаем уникальный ID
        id = f"learn_{contradiction.contradiction_id}"
        
        return ContradictionLearningOpportunity(
            id=id,
            concept=contradiction.concept,
            type="contradiction_resolution",
            priority=priority,
            description=f"Разрешение противоречия в знаниях по '{contradiction.concept}'",
            evidence=evidence,
            suggested_actions=suggested_actions,
            required_capabilities=["knowledge_integration", "source_analysis"],
            domain=domain,
            metadata={
                "contradiction_id": contradiction.contradiction_id,
                "contradiction_type": contradiction_type,
                "divergence_level": contradiction.divergence_level,
                "severity": contradiction.get_severity()
            }
        )
    
    def _get_contradiction_type(self, contradiction) -> str:
        """
        Определяет тип противоречия.
        
        Args:
            contradiction: Противоречие
            
        Returns:
            str: Тип противоречия
        """
        if "type" in contradiction.meta:
            return contradiction.metadata["type"]
        
        # Определяем тип по конфликтующим фактам
        if len(contradiction.conflicting_facts) == 2:
            fact1 = contradiction.conflicting_facts[0]
            fact2 = contradiction.conflicting_facts[1]
            
            # Проверяем числовые противоречия
            if isinstance(fact1.get("value"), (int, float)) and isinstance(fact2.get("value"), (int, float)):
                return "numeric_conflict"
            
            # Проверяем булевы противоречия
            if isinstance(fact1.get("value"), bool) and isinstance(fact2.get("value"), bool):
                return "boolean_conflict"
            
            # Проверяем противоречия в ответах
            if "response" in fact1.get("relation", "") and "response" in fact2.get("relation", ""):
                return "response_conflict"
        
        # Проверяем по метаданным
        if "relation_type" in contradiction.meta:
            if contradiction.metadata["relation_type"].startswith("only_") or contradiction.metadata["relation_type"].startswith("not_only_"):
                return "exclusivity_conflict"
            if contradiction.metadata["relation_type"] in ["is_a", "part_of", "member_of"]:
                return "hierarchy_conflict"
        
        return "unknown"
    
    def _determine_domain(self, concept: str, knowledge_graph) -> str:
        """
        Определяет домен знаний для концепта.
        
        Args:
            concept: Концепт
            knowledge_graph: Граф знаний
            
        Returns:
            str: Домен знаний
        """
        # Ключевые слова для определения домена
        domain_keywords = {
            "technology": ["программа", "алгоритм", "код", "технология", "разработка", "нейросеть",
                          "искусственный интеллект", "машинное обучение", "нейронная сеть", "квантовый"],
            "medicine": ["болезнь", "лечение", "врач", "диагноз", "симптом", "ген", "вирус",
                        "иммунитет", "лекарство", "здоровье", "медицина"],
            "philosophy": ["сознание", "этика", "свобода", "истина", "душа", "философия",
                          "мышление", "бытие", "смысл жизни", "мораль"],
            "art": ["картина", "музыка", "поэзия", "театр", "искусство", "литература",
                   "кино", "скульптура", "живопись", "творчество"],
            "science": ["наука", "физика", "химия", "биология", "астрономия", "математика",
                       "теория", "эксперимент", "гипотеза", "закон"],
            "general": ["знание", "информация", "данные", "процесс", "система",
                       "время", "пространство", "вселенная", "жизнь", "человек"]
        }
        
        # Проверяем связанные узлы в графе знаний
        related_nodes = knowledge_graph.get_related_nodes(concept)
        content = " ".join(node.get("content", "") for node in related_nodes).lower()
        
        # Считаем совпадения с ключевыми словами
        domain_scores = {domain: 0 for domain in domain_keywords}
        
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in content:
                    domain_scores[domain] += 1
        
        # Определяем домен с наибольшим количеством совпадений
        main_domain = max(domain_scores, key=lambda k: domain_scores[k])
        
        # Если нет явного домена, используем общий
        if domain_scores[main_domain] == 0:
            return "general"
        
        return main_domain
    
    def get_learning_recommendations(self) -> List[str]:
        """
        Генерирует рекомендации по обучению на основе противоречия.
        
        Returns:
            List[str]: Рекомендации
        """
        recommendations = []
        
        # Рекомендации на основе типа противоречия
        if "numeric_conflict" in self.metadata.get("contradiction_type", ""):
            recommendations.append(
                "Проведите дополнительный анализ для определения наиболее точного числового значения. "
                "Рассмотрите возможность усреднения показателей или выявления контекстных условий, "
                "при которых верно каждое значение."
            )
        
        elif "boolean_conflict" in self.metadata.get("contradiction_type", ""):
            recommendations.append(
                "Проверьте условия, при которых каждое утверждение является верным. "
                "Возможно, противоречие возникает из-за различия в контексте или условиях."
            )
        
        elif "exclusivity_conflict" in self.metadata.get("contradiction_type", ""):
            recommendations.append(
                "Проанализируйте, не являются ли утверждения 'только' и 'не только' "
                "применимыми в разных контекстах или подкатегориях."
            )
        
        elif "hierarchy_conflict" in self.metadata.get("contradiction_type", ""):
            recommendations.append(
                "Пересмотрите иерархию для устранения циклических зависимостей или "
                "взаимоисключающих классификаций. Возможно, некоторые связи должны "
                "быть заменены на другие типы отношений."
            )
        
        elif "response_conflict" in self.metadata.get("contradiction_type", ""):
            recommendations.append(
                "Проанализируйте контекст использования каждого ответа. Возможно, "
                "разные ответы применимы в разных сценариях или для разных аудиторий."
            )
        
        # Рекомендации на основе серьезности
        severity = self.metadata.get("severity", "medium")
        if severity == "high":
            recommendations.append(
                "Это высокосерьезное противоречие требует немедленного внимания. "
                "Рассмотрите возможность привлечения экспертов для разрешения."
            )
        elif severity == "medium":
            recommendations.append(
                "Это среднесерьезное противоречие важно для точности системы. "
                "Планируйте его разрешение в ближайшее время."
            )
        else:
            recommendations.append(
                "Это низкосерьезное противоречие имеет минимальное влияние на систему. "
                "Его можно разрешить в рамках регулярного процесса обновления знаний."
            )
        
        # Добавляем общие рекомендации
        recommendations.append(
            "Соберите дополнительные данные из авторитетных источников для подтверждения или "
            "опровержения конфликтующих утверждений."
        )
        
        recommendations.append(
            "Проведите анализ контекстных условий, при которых проявляется каждое утверждение."
        )
        
        return recommendations
    
    def is_high_priority(self) -> bool:
        """
        Проверяет, является ли возможность обучения высокоприоритетной.
        
        Returns:
            bool: Является ли высокоприоритетной
        """
        return self.priority > 0.7
    
    def get_time_since_creation(self) -> float:
        """
        Возвращает время с момента создания возможности обучения.
        
        Returns:
            float: Время в секундах
        """
        return time.time() - self.created_at
    
    def requires_immediate_attention(self) -> bool:
        """
        Проверяет, требует ли возможность обучения немедленного внимания.
        
        Returns:
            bool: Требует ли немедленного внимания
        """
        # Высокоприоритетные возможности требуют внимания в течение 24 часов
        if self.priority > 0.8 and self.get_time_since_creation() > 86400:
            return True
        
        # Среднеприоритетные возможности требуют внимания в течение 3 дней
        if self.priority > 0.6 and self.get_time_since_creation() > 3 * 86400:
            return True
        
"""Модуль обучения на основе противоречий в системе CogniFlex"""
import os
import logging
import time
import json
import re
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime, timedelta
import random
import hashlib
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional dependencies
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, ModuleNotFoundError, RuntimeError) as e:
    logger.debug(f"Failed to import sentence_transformers: {e}")
    SentenceTransformer = None  # type: ignore
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except (ImportError, ModuleNotFoundError, RuntimeError) as e:
    logger.debug(f"Failed to import torch: {e}")
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

logger = logging.getLogger("cogniflex.contradiction.learning")

class ContradictionLearningOpportunity:
    """Представляет возможность для обучения на основе противоречия."""
    
    def __init__(self, id: str, concept: str, type: str, priority: float,
                 description: str, evidence: List[str], 
                 suggested_actions: List[str], 
                 required_capabilities: Optional[List[str]] = None,
                 domain: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Инициализирует возможность для обучения.
        
        Args:
            id: Уникальный ID возможности
            concept: Концепт, связанный с возможностью
            type: Тип возможности
            priority: Приоритет (0.0-1.0)
            description: Описание возможности
            evidence: Свидетельства
            suggested_actions: Предлагаемые действия
            required_capabilities: Требуемые возможности системы
            domain: Домен знаний
            metadata: Дополнительные метаданные
        """
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
            "id": self.id,
            "concept": self.concept,
            "type": self.type,
            "priority": self.priority,
            "description": self.description,
            "evidence": self.evidence,
            "suggested_actions": self.suggested_actions,
            "required_capabilities": self.required_capabilities,
            "domain": self.domain,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "status": self.status,
            "progress": self.progress,
            "learning_tasks": self.learning_tasks,
            "completed_tasks": self.completed_tasks,
            "impact_assessment": self.impact_assessment,
            "learning_history": self.learning_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContradictionLearningOpportunity':
        """Создает возможность обучения из словаря."""
        opportunity = cls(
            id=data["id"],
            concept=data["concept"],
            type=data["type"],
            priority=data["priority"],
            description=data["description"],
            evidence=data["evidence"],
            suggested_actions=data["suggested_actions"],
            required_capabilities=data.get("required_capabilities", []),
            domain=data.get("domain"),
            metadata=data.get("metadata", {})
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
    
    def update_status(self, new_status: str, progress: Optional[float] = None):
        """
        Обновляет статус возможности обучения.
        
        Args:
            new_status: Новый статус
            progress: Прогресс (опционально)
        """
        self.status = new_status
        self.last_updated = time.time()
        
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))
        
        # Добавляем запись в историю
        self.learning_history.append({
            "status": new_status,
            "progress": self.progress,
            "timestamp": self.last_updated
        })
    
    def add_learning_task(self, task: Dict[str, Any]):
        """
        Добавляет задачу обучения.
        
        Args:
            task: Задача обучения
        """
        self.learning_tasks.append(task)
        self.last_updated = time.time()
    
    def complete_task(self, task_id: str, result: Dict[str, Any]):
        """
        Помечает задачу как выполненную.
        
        Args:
            task_id: ID задачи
            result: Результат выполнения
        """
        for i, task in enumerate(self.learning_tasks):
            if task["task_id"] == task_id:
                completed_task = task.copy()
                completed_task["completed_at"] = time.time()
                completed_task["result"] = result
                self.completed_tasks.append(completed_task)
                del self.learning_tasks[i]
                
                # Обновляем прогресс
                total_tasks = len(self.completed_tasks) + len(self.learning_tasks)
                if total_tasks > 0:
                    self.progress = len(self.completed_tasks) / total_tasks
                
                self.last_updated = time.time()
                break
    
    def get_progress_report(self) -> Dict[str, Any]:
        """
        Возвращает отчет о прогрессе.
        
        Returns:
            Dict: Отчет о прогрессе
        """
        return {
            "id": self.id,
            "concept": self.concept,
            "type": self.type,
            "priority": self.priority,
            "status": self.status,
            "progress": self.progress,
            "total_tasks": len(self.learning_tasks) + len(self.completed_tasks),
            "completed_tasks": len(self.completed_tasks),
            "pending_tasks": len(self.learning_tasks),
            "impact_assessment": self.impact_assessment,
            "last_updated": self.last_updated
        }
    
    def assess_impact(self, knowledge_graph) -> Dict[str, float]:
        """
        Оценивает влияние обучения на систему знаний.
        
        Args:
            knowledge_graph: Граф знаний
            
        Returns:
            Dict: Оценка влияния
        """
        try:
            # Получаем количество связанных узлов
            related_nodes = knowledge_graph.get_related_nodes(self.concept)
            node_count = len(related_nodes)
            
            # Получаем глубину влияния
            depth = knowledge_graph.get_influence_depth(self.concept)
            
            # Оцениваем важность концепта
            importance = self._calculate_concept_importance(knowledge_graph)
            
            # Влияние = (важность * 0.4) + (количество узлов * 0.3) + (глубина * 0.3)
            impact = (importance * 0.4) + (min(1.0, node_count / 50) * 0.3) + (min(1.0, depth / 10) * 0.3)
            
            # Сохраняем оценку
            self.impact_assessment = {
                "impact_score": min(1.0, impact),
                "node_count": node_count,
                "influence_depth": depth,
                "concept_importance": importance,
                "timestamp": time.time()
            }
            
            return self.impact_assessment
            
        except Exception as e:
            logger.error(f"Ошибка оценки влияния обучения: {e}")
            # Базовая оценка влияния
            self.impact_assessment = {
                "impact_score": self.priority * 0.7,
                "node_count": 0,
                "influence_depth": 0,
                "concept_importance": self.priority,
                "timestamp": time.time()
            }
            return self.impact_assessment
    
    def _calculate_concept_importance(self, knowledge_graph) -> float:
        """
        Вычисляет важность концепта на основе его использования и значимости.
        
        Args:
            knowledge_graph: Граф знаний
            
        Returns:
            float: Важность концепта (0.0-1.0)
        """
        # Инициализируем анализатор тональности
        sentiment_analyzer = SentimentIntensityAnalyzer()
        stop_words = set(stopwords.words('english') + stopwords.words('russian'))
        
        # 1. Анализ частоты использования концепта в системе
        usage_frequency = 0.3  # Базовая оценка
        try:
            usage_count = knowledge_graph.get_concept_usage_count(self.concept)
            # Логарифмическая шкала для учета частоты использования
            usage_frequency = min(0.5, 0.1 + np.log1p(usage_count) * 0.1)
        except Exception:
            pass
        
        # 2. Анализ важности концепта через контент
        concept_score = 0.0
        key_terms = ["наука", "технология", "искусственный интеллект", "этика", "безопасность",
                    "человек", "сознание", "разум", "жизнь", "вселенная", "время", "пространство"]
        
        # Проверяем содержимое связанных узлов
        related_nodes = knowledge_graph.get_related_nodes(self.concept)
        fact_content = " ".join(node.get("content", "") for node in related_nodes)
        fact_content = fact_content.lower()
        
        # Считаем вхождения ключевых терминов
        key_term_count = sum(1 for term in key_terms if term in fact_content)
        concept_score += min(0.3, key_term_count * 0.1)
        
        # 3. Анализ тональности контента
        sentiment = sentiment_analyzer.polarity_scores(fact_content)
        neutrality = 1.0 - abs(sentiment['compound'])
        concept_score += neutrality * 0.2
        
        # 4. Анализ структуры текста
        words = word_tokenize(fact_content)
        words = [word for word in words if word.isalnum() and word not in stop_words]
        unique_words = len(set(words))
        total_words = len(words)
        
        if total_words > 0:
            diversity = unique_words / total_words
            concept_score += min(0.2, diversity * 0.2)
        
        # 5. Проверка на общий/специфичный концепт
        common_concepts = [
            "человек", "знание", "информация", "данные", "процесс", "система",
            "время", "пространство", "вселенная", "жизнь", "сознание", "разум"
        ]
        
        if self.concept.lower() in [c.lower() for c in common_concepts]:
            concept_score += 0.2
        
        # Общая важность концепта
        importance = min(1.0, usage_frequency + concept_score)
        return importance
    
    def generate_learning_plan(self, knowledge_graph) -> List[Dict[str, Any]]:
        """
        Генерирует план обучения для разрешения противоречия.
        
        Args:
            knowledge_graph: Граф знаний
            
        Returns:
            List: План обучения
        """
        # Определяем тип противоречия
        contradiction_type = self._determine_contradiction_type()
        
        # Генерируем задачи в зависимости от типа
        if contradiction_type == "numeric_conflict":
            return self._generate_numeric_conflict_learning_plan(knowledge_graph)
        elif contradiction_type == "boolean_conflict":
            return self._generate_boolean_conflict_learning_plan(knowledge_graph)
        elif contradiction_type == "exclusivity_conflict":
            return self._generate_exclusivity_conflict_learning_plan(knowledge_graph)
        elif contradiction_type == "hierarchy_conflict":
            return self._generate_hierarchy_conflict_learning_plan(knowledge_graph)
        elif contradiction_type == "response_conflict":
            return self._generate_response_conflict_learning_plan(knowledge_graph)
        else:
            return self._generate_general_learning_plan(knowledge_graph)
    
    def _determine_contradiction_type(self) -> str:
        """
        Определяет тип противоречия на основе метаданных.
        
        Returns:
            str: Тип противоречия
        """
        if "contradiction_type" in self.metadata:
            return self.metadata["contradiction_type"]
        
        # Определяем тип по описанию
        description = self.description.lower()
        
        if "число" in description or "значение" in description:
            return "numeric_conflict"
        elif "да" in description or "нет" in description or "верно" in description or "неверно" in description:
            return "boolean_conflict"
        elif "только" in description or "не только" in description:
            return "exclusivity_conflict"
        elif "иерархия" in description or "классификация" in description:
            return "hierarchy_conflict"
        elif "ответ" in description or "вопрос" in description:
            return "response_conflict"
        
        return "general"
    
    def _generate_numeric_conflict_learning_plan(self, knowledge_graph) -> List[Dict[str, Any]]:
        """Генерирует план обучения для числового противоречия."""
        tasks = [
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_numeric_1'.encode()).hexdigest()[:8]}",
                "task_type": "evidence_collection",
                "description": "Сбор данных из авторитетных источников для определения наиболее точного значения",
                "priority": self.priority * 0.9,
                "scheduled_time": time.time() + 60,  # Через 1 минуту
                "required_capabilities": ["web_search"],
                "parameters": {
                    "query": f"{self.concept} точное значение",
                    "num_results": 10
                },
                "dependencies": []
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_numeric_2'.encode()).hexdigest()[:8]}",
                "task_type": "analysis",
                "description": "Анализ методов измерения и определение контекстных условий",
                "priority": self.priority * 0.8,
                "scheduled_time": time.time() + 300,  # Через 5 минут
                "required_capabilities": ["nlp_analysis", "source_reputation"],
                "parameters": {
                    "analysis_type": "methodology_comparison"
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_numeric_1'.encode()).hexdigest()[:8]}"]
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_numeric_3'.encode()).hexdigest()[:8]}",
                "task_type": "integration",
                "description": "Интеграция данных и определение наиболее точного значения с учетом контекста",
                "priority": self.priority * 0.7,
                "scheduled_time": time.time() + 1800,  # Через 30 минут
                "required_capabilities": ["knowledge_integration"],
                "parameters": {
                    "integration_strategy": "contextual_weighting"
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_numeric_2'.encode()).hexdigest()[:8]}"]
            }
        ]
        
        return tasks
    
    def _generate_boolean_conflict_learning_plan(self, knowledge_graph) -> List[Dict[str, Any]]:
        """Генерирует план обучения для булева противоречия."""
        tasks = [
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_boolean_1'.encode()).hexdigest()[:8]}",
                "task_type": "context_analysis",
                "description": "Анализ условий, при которых каждое утверждение является верным",
                "priority": self.priority * 0.9,
                "scheduled_time": time.time() + 60,  # Через 1 минуту
                "required_capabilities": ["nlp_analysis", "context_detection"],
                "parameters": {
                    "analysis_type": "condition_extraction"
                },
                "dependencies": []
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_boolean_2'.encode()).hexdigest()[:8]}",
                "task_type": "evidence_collection",
                "description": "Сбор доказательств для подтверждения условий применения",
                "priority": self.priority * 0.8,
                "scheduled_time": time.time() + 300,  # Через 5 минут
                "required_capabilities": ["web_search"],
                "parameters": {
                    "query": f"{self.concept} условия применения",
                    "num_results": 8
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_boolean_1'.encode()).hexdigest()[:8]}"]
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_boolean_3'.encode()).hexdigest()[:8]}",
                "task_type": "synthesis",
                "description": "Создание контекстно-зависимого ответа",
                "priority": self.priority * 0.7,
                "scheduled_time": time.time() + 1800,  # Через 30 минут
                "required_capabilities": ["knowledge_integration", "response_generation"],
                "parameters": {
                    "synthesis_strategy": "conditional_response"
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_boolean_2'.encode()).hexdigest()[:8]}"]
            }
        ]
        
        return tasks
    
    def _generate_exclusivity_conflict_learning_plan(self, knowledge_graph) -> List[Dict[str, Any]]:
        """Генерирует план обучения для противоречия эксклюзивности."""
        tasks = [
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_exclusivity_1'.encode()).hexdigest()[:8]}",
                "task_type": "taxonomy_analysis",
                "description": "Анализ иерархии категорий и подкатегорий",
                "priority": self.priority * 0.9,
                "scheduled_time": time.time() + 60,  # Через 1 минуту
                "required_capabilities": ["taxonomy_analysis"],
                "parameters": {
                    "analysis_type": "category_hierarchy"
                },
                "dependencies": []
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_exclusivity_2'.encode()).hexdigest()[:8]}",
                "task_type": "evidence_collection",
                "description": "Сбор данных для определения взаимоисключающих условий",
                "priority": self.priority * 0.8,
                "scheduled_time": time.time() + 300,  # Через 5 минут
                "required_capabilities": ["web_search"],
                "parameters": {
                    "query": f"{self.concept} взаимоисключающие условия",
                    "num_results": 7
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_exclusivity_1'.encode()).hexdigest()[:8]}"]
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_exclusivity_3'.encode()).hexdigest()[:8]}",
                "task_type": "taxonomy_revision",
                "description": "Переработка иерархии категорий и разделение на подкатегории",
                "priority": self.priority * 0.7,
                "scheduled_time": time.time() + 1800,  # Через 30 минут
                "required_capabilities": ["knowledge_integration", "taxonomy_management"],
                "parameters": {
                    "revision_strategy": "subcategorization"
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_exclusivity_2'.encode()).hexdigest()[:8]}"]
            }
        ]
        
        return tasks
    
    def _generate_hierarchy_conflict_learning_plan(self, knowledge_graph) -> List[Dict[str, Any]]:
        """Генерирует план обучения для иерархического противоречия."""
        tasks = [
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_hierarchy_1'.encode()).hexdigest()[:8]}",
                "task_type": "hierarchy_analysis",
                "description": "Анализ иерархических связей и выявление циклических зависимостей",
                "priority": self.priority * 0.9,
                "scheduled_time": time.time() + 60,  # Через 1 минуту
                "required_capabilities": ["knowledge_graph_analysis"],
                "parameters": {
                    "analysis_type": "cycle_detection"
                },
                "dependencies": []
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_hierarchy_2'.encode()).hexdigest()[:8]}",
                "task_type": "taxonomy_review",
                "description": "Анализ взаимоисключающих классификаций",
                "priority": self.priority * 0.8,
                "scheduled_time": time.time() + 300,  # Через 5 минут
                "required_capabilities": ["taxonomy_analysis"],
                "parameters": {
                    "review_type": "classification_conflicts"
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_hierarchy_1'.encode()).hexdigest()[:8]}"]
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_hierarchy_3'.encode()).hexdigest()[:8]}",
                "task_type": "hierarchy_reconstruction",
                "description": "Перестройка иерархии для устранения противоречий",
                "priority": self.priority * 0.7,
                "scheduled_time": time.time() + 1800,  # Через 30 минут
                "required_capabilities": ["knowledge_integration", "taxonomy_management"],
                "parameters": {
                    "reconstruction_strategy": "hierarchy_flattening"
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_hierarchy_2'.encode()).hexdigest()[:8]}"]
            }
        ]
        
        return tasks
    
    def _generate_response_conflict_learning_plan(self, knowledge_graph) -> List[Dict[str, Any]]:
        """Генерирует план обучения для противоречия в ответах."""
        tasks = [
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_response_1'.encode()).hexdigest()[:8]}",
                "task_type": "audience_analysis",
                "description": "Анализ целевой аудитории и контекстных условий",
                "priority": self.priority * 0.9,
                "scheduled_time": time.time() + 60,  # Через 1 минуту
                "required_capabilities": ["audience_analysis", "context_detection"],
                "parameters": {
                    "analysis_type": "audience_segmentation"
                },
                "dependencies": []
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_response_2'.encode()).hexdigest()[:8]}",
                "task_type": "response_comparison",
                "description": "Сравнение ответов и определение границ применимости",
                "priority": self.priority * 0.8,
                "scheduled_time": time.time() + 300,  # Через 5 минут
                "required_capabilities": ["nlp_analysis", "semantic_comparison"],
                "parameters": {
                    "comparison_type": "contextual_boundaries"
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_response_1'.encode()).hexdigest()[:8]}"]
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_response_3'.encode()).hexdigest()[:8]}",
                "task_type": "response_integration",
                "description": "Создание условных ответов для разных сценариев",
                "priority": self.priority * 0.7,
                "scheduled_time": time.time() + 1800,  # Через 30 минут
                "required_capabilities": ["response_generation", "context_management"],
                "parameters": {
                    "integration_strategy": "scenario_based"
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_response_2'.encode()).hexdigest()[:8]}"]
            }
        ]
        
        return tasks
    
    def _generate_general_learning_plan(self, knowledge_graph) -> List[Dict[str, Any]]:
        """Генерирует общий план обучения."""
        tasks = [
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_general_1'.encode()).hexdigest()[:8]}",
                "task_type": "comprehensive_analysis",
                "description": "Определение типа противоречия и анализ источников информации",
                "priority": self.priority * 0.9,
                "scheduled_time": time.time() + 60,  # Через 1 минуту
                "required_capabilities": ["contradiction_analysis"],
                "parameters": {
                    "analysis_type": "contradiction_typing"
                },
                "dependencies": []
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_general_2'.encode()).hexdigest()[:8]}",
                "task_type": "context_research",
                "description": "Исследование контекстных условий и сбор дополнительных данных",
                "priority": self.priority * 0.8,
                "scheduled_time": time.time() + 300,  # Через 5 минут
                "required_capabilities": ["web_search", "context_analysis"],
                "parameters": {
                    "research_focus": "contextual_conditions"
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_general_1'.encode()).hexdigest()[:8]}"]
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_general_3'.encode()).hexdigest()[:8]}",
                "task_type": "solution_synthesis",
                "description": "Синтез решения и документирование",
                "priority": self.priority * 0.7,
                "scheduled_time": time.time() + 1800,  # Через 30 минут
                "required_capabilities": ["knowledge_integration", "solution_synthesis"],
                "parameters": {
                    "synthesis_type": "comprehensive"
                },
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_general_2'.encode()).hexdigest()[:8]}"]
            }
        ]
        
        return tasks
    
    def get_learning_report(self) -> Dict[str, Any]:
        """
        Генерирует отчет о процессе обучения.
        
        Returns:
            Dict: Отчет о процессе обучения
        """
        completed_count = len(self.completed_tasks)
        total_count = completed_count + len(self.learning_tasks)
        
        # Анализируем прогресс
        progress_details = []
        for task in self.completed_tasks:
            progress_details.append({
                "task": task["description"],
                "completion_time": task.get("completed_at", 0),
                "result": task.get("result", {})
            })
        
        return {
            "id": self.id,
            "concept": self.concept,
            "type": self.type,
            "priority": self.priority,
            "status": self.status,
            "progress": self.progress,
            "completed_tasks": completed_count,
            "total_tasks": total_count,
            "progress_details": progress_details,
            "impact_assessment": self.impact_assessment,
            "created_at": self.created_at,
            "last_updated": self.last_updated
        }
    
    def generate_learning_opportunity(self, contradiction, knowledge_graph) -> 'ContradictionLearningOpportunity':
        """
        Генерирует возможность для обучения на основе противоречия.
        
        Args:
            contradiction: Противоречие
            knowledge_graph: Граф знаний
            
        Returns:
            ContradictionLearningOpportunity: Возможность для обучения
        """
        # Определяем тип противоречия
        contradiction_type = self._get_contradiction_type(contradiction)
        
        # Определяем домен
        domain = self._determine_domain(contradiction.concept, knowledge_graph)
        
        # Определяем приоритет
        priority = contradiction.get_resolution_priority()
        
        # Формируем свидетельства
        evidence = [
            f"Обнаружено противоречие с уровнем расхождения {contradiction.divergence_level:.2f}",
            f"Тип противоречия: {contradiction_type}",
            f"Серьезность: {contradiction.get_severity()}"
        ]
        
        # Формируем предполагаемые действия
        suggested_actions = [
            f"Исследовать '{contradiction.concept}' глубже",
            f"Проверить источники информации по '{contradiction.concept}'",
            f"Анализировать контекстные условия для разрешения противоречия"
        ]
        
        # Создаем уникальный ID
        id = f"learn_{contradiction.contradiction_id}"
        
        return ContradictionLearningOpportunity(
            id=id,
            concept=contradiction.concept,
            type="contradiction_resolution",
            priority=priority,
            description=f"Разрешение противоречия в знаниях по '{contradiction.concept}'",
            evidence=evidence,
            suggested_actions=suggested_actions,
            required_capabilities=["knowledge_integration", "source_analysis"],
            domain=domain,
            metadata={
                "contradiction_id": contradiction.contradiction_id,
                "contradiction_type": contradiction_type,
                "divergence_level": contradiction.divergence_level,
                "severity": contradiction.get_severity()
            }
        )
    
    def _get_contradiction_type(self, contradiction) -> str:
        """
        Определяет тип противоречия.
        
        Args:
            contradiction: Противоречие
            
        Returns:
            str: Тип противоречия
        """
        if "type" in contradiction.meta:
            return contradiction.metadata["type"]
        
        # Определяем тип по конфликтующим фактам
        if len(contradiction.conflicting_facts) == 2:
            fact1 = contradiction.conflicting_facts[0]
            fact2 = contradiction.conflicting_facts[1]
            
            # Проверяем числовые противоречия
            if isinstance(fact1.get("value"), (int, float)) and isinstance(fact2.get("value"), (int, float)):
                return "numeric_conflict"
            
            # Проверяем булевы противоречия
            if isinstance(fact1.get("value"), bool) and isinstance(fact2.get("value"), bool):
                return "boolean_conflict"
            
            # Проверяем противоречия в ответах
            if "response" in fact1.get("relation", "") and "response" in fact2.get("relation", ""):
                return "response_conflict"
        
        # Проверяем по метаданным
        if "relation_type" in contradiction.meta:
            if contradiction.metadata["relation_type"].startswith("only_") or contradiction.metadata["relation_type"].startswith("not_only_"):
                return "exclusivity_conflict"
            if contradiction.metadata["relation_type"] in ["is_a", "part_of", "member_of"]:
                return "hierarchy_conflict"
        
        return "unknown"
    
    def _determine_domain(self, concept: str, knowledge_graph) -> str:
        """
        Определяет домен знаний для концепта.
        
        Args:
            concept: Концепт
            knowledge_graph: Граф знаний
            
        Returns:
            str: Домен знаний
        """
        # Ключевые слова для определения домена
        domain_keywords = {
            "technology": ["программа", "алгоритм", "код", "технология", "разработка", "нейросеть",
                          "искусственный интеллект", "машинное обучение", "нейронная сеть", "квантовый"],
            "medicine": ["болезнь", "лечение", "врач", "диагноз", "симптом", "ген", "вирус",
                        "иммунитет", "лекарство", "здоровье", "медицина"],
            "philosophy": ["сознание", "этика", "свобода", "истина", "душа", "философия",
                          "мышление", "бытие", "смысл жизни", "мораль"],
            "art": ["картина", "музыка", "поэзия", "театр", "искусство", "литература",
                   "кино", "скульптура", "живопись", "творчество"],
            "science": ["наука", "физика", "химия", "биология", "астрономия", "математика",
                       "теория", "эксперимент", "гипотеза", "закон"],
            "general": ["знание", "информация", "данные", "процесс", "система",
                       "время", "пространство", "вселенная", "жизнь", "человек"]
        }
        
        # Проверяем связанные узлы в графе знаний
        related_nodes = knowledge_graph.get_related_nodes(concept)
        content = " ".join(node.get("content", "") for node in related_nodes).lower()
        
        # Считаем совпадения с ключевыми словами
        domain_scores = {domain: 0 for domain in domain_keywords}
        
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in content:
                    domain_scores[domain] += 1
        
        # Определяем домен с наибольшим количеством совпадений
        main_domain = max(domain_scores, key=lambda k: domain_scores[k])
        
        # Если нет явного домена, используем общий
        if domain_scores[main_domain] == 0:
            return "general"
        
        return main_domain
    
    def get_learning_recommendations(self) -> List[str]:
        """
        Генерирует рекомендации по обучению на основе противоречия.
        
        Returns:
            List[str]: Рекомендации
        """
        recommendations = []
        
        # Рекомендации на основе типа противоречия
        if "numeric_conflict" in self.metadata.get("contradiction_type", ""):
            recommendations.append(
                "Проведите дополнительный анализ для определения наиболее точного числового значения. "
                "Рассмотрите возможность усреднения показателей или выявления контекстных условий, "
                "при которых верно каждое значение."
            )
        
        elif "boolean_conflict" in self.metadata.get("contradiction_type", ""):
            recommendations.append(
                "Проверьте условия, при которых каждое утверждение является верным. "
                "Возможно, противоречие возникает из-за различия в контексте или условиях."
            )
        
        elif "exclusivity_conflict" in self.metadata.get("contradiction_type", ""):
            recommendations.append(
                "Проанализируйте, не являются ли утверждения 'только' и 'не только' "
                "применимыми в разных контекстах или подкатегориях."
            )
        
        elif "hierarchy_conflict" in self.metadata.get("contradiction_type", ""):
            recommendations.append(
                "Пересмотрите иерархию для устранения циклических зависимостей или "
                "взаимоисключающих классификаций. Возможно, некоторые связи должны "
                "быть заменены на другие типы отношений."
            )
        
        elif "response_conflict" in self.metadata.get("contradiction_type", ""):
            recommendations.append(
                "Проанализируйте контекст использования каждого ответа. Возможно, "
                "разные ответы применимы в разных сценариях или для разных аудиторий."
            )
        
        # Рекомендации на основе серьезности
        severity = self.metadata.get("severity", "medium")
        if severity == "high":
            recommendations.append(
                "Это высокосерьезное противоречие требует немедленного внимания. "
                "Рассмотрите возможность привлечения экспертов для разрешения."
            )
        elif severity == "medium":
            recommendations.append(
                "Это среднесерьезное противоречие важно для точности системы. "
                "Планируйте его разрешение в ближайшее время."
            )
        else:
            recommendations.append(
                "Это низкосерьезное противоречие имеет минимальное влияние на систему. "
                "Его можно разрешить в рамках регулярного процесса обновления знаний."
            )
        
        # Добавляем общие рекомендации
        recommendations.append(
            "Соберите дополнительные данные из авторитетных источников для подтверждения или "
            "опровержения конфликтующих утверждений."
        )
        
        recommendations.append(
            "Проведите анализ контекстных условий, при которых проявляется каждое утверждение."
        )
        
        return recommendations
    
    def is_high_priority(self) -> bool:
        """
        Проверяет, является ли возможность обучения высокоприоритетной.
        
        Returns:
            bool: Является ли высокоприоритетной
        """
        return self.priority > 0.7
    
    def get_time_since_creation(self) -> float:
        """
        Возвращает время с момента создания возможности обучения.
        
        Returns:
            float: Время в секундах
        """
        return time.time() - self.created_at
    
    def requires_immediate_attention(self) -> bool:
        """
        Проверяет, требует ли возможность обучения немедленного внимания.
        
        Returns:
            bool: Требует ли немедленного внимания
        """
        # Высокоприоритетные возможности требуют внимания в течение 24 часов
        if self.priority > 0.8 and self.get_time_since_creation() > 86400:
            return True
        
        # Среднеприоритетные возможности требуют внимания в течение 3 дней
        if self.priority > 0.6 and self.get_time_since_creation() > 3 * 86400:
            return True
        


class ContradictionLearning:
    def __init__(self, brain=None, knowledge_graph=None, cache_dir=None):
        self.brain = brain
        self.knowledge_graph = knowledge_graph
        self.cache_dir = cache_dir
        self.learning_opportunities = {}
        self.stats = {
            "opportunities_created": 0,
            "opportunities_completed": 0,
            "learning_tasks_generated": 0,
            "learning_tasks_completed": 0,
            "impact_assessments": 0,
            "last_learning_cycle": None
        }
    
    def create_learning_opportunity(self, contradiction, priority=None):
        return None
    
    def get_learning_opportunities(self, status=None, priority_min=None):
        return []
    
    def execute_learning_cycle(self):
        return {"processed_opportunities": 0, "generated_tasks": 0, "completed_tasks": 0, "errors": []}
    
    def get_learning_statistics(self):
        return self.stats.copy()
