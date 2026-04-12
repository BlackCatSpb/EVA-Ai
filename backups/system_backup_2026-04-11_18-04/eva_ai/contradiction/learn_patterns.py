"""Pattern extraction from contradictions, frequency analysis."""
import logging
import time
import hashlib
from typing import Dict, List, Any, Optional
from collections import defaultdict

logger = logging.getLogger("eva_ai.contradiction.learning.patterns")


class PatternExtractionMixin:
    """Mixin providing pattern extraction and learning plan generation."""
    
    def _determine_contradiction_type(self) -> str:
        """Определяет тип противоречия на основе метаданных."""
        if "contradiction_type" in self.metadata:
            return self.metadata["contradiction_type"]
        
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
    
    def _get_contradiction_type(self, contradiction) -> str:
        """Определяет тип противоречия."""
        if hasattr(contradiction, 'metadata') and isinstance(contradiction.metadata, dict) and "type" in contradiction.metadata:
            return contradiction.metadata["type"]
        
        if len(contradiction.conflicting_facts) == 2:
            fact1 = contradiction.conflicting_facts[0]
            fact2 = contradiction.conflicting_facts[1]
            if isinstance(fact1.get("value"), (int, float)) and isinstance(fact2.get("value"), (int, float)):
                return "numeric_conflict"
            if isinstance(fact1.get("value"), bool) and isinstance(fact2.get("value"), bool):
                return "boolean_conflict"
            if "response" in fact1.get("relation", "") and "response" in fact2.get("relation", ""):
                return "response_conflict"
        
        if hasattr(contradiction, 'metadata') and isinstance(contradiction.metadata, dict) and "relation_type" in contradiction.metadata:
            relation_type = contradiction.metadata["relation_type"]
            if relation_type.startswith("only_") or relation_type.startswith("not_only_"):
                return "exclusivity_conflict"
            if relation_type in ["is_a", "part_of", "member_of"]:
                return "hierarchy_conflict"
        
        return "unknown"
    
    def _determine_domain(self, concept: str, knowledge_graph) -> str:
        """Определяет домен знаний для концепта."""
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
        
        related_nodes = knowledge_graph.get_related_nodes(concept)
        content = " ".join(node.get("content", "") for node in related_nodes).lower()
        
        domain_scores = {domain: 0 for domain in domain_keywords}
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in content:
                    domain_scores[domain] += 1
        
        main_domain = max(domain_scores, key=lambda k: domain_scores[k])
        if domain_scores[main_domain] == 0:
            return "general"
        return main_domain
    
    def generate_learning_plan(self, knowledge_graph) -> List[Dict[str, Any]]:
        """Генерирует план обучения для разрешения противоречия."""
        contradiction_type = self._determine_contradiction_type()
        
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
    
    def _generate_numeric_conflict_learning_plan(self, knowledge_graph) -> List[Dict[str, Any]]:
        """Генерирует план обучения для числового противоречия."""
        tasks = [
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_numeric_1'.encode()).hexdigest()[:8]}",
                "task_type": "evidence_collection",
                "description": "Сбор данных из авторитетных источников для определения наиболее точного значения",
                "priority": self.priority * 0.9,
                "scheduled_time": time.time() + 60,
                "required_capabilities": ["web_search"],
                "parameters": {"query": f"{self.concept} точное значение", "num_results": 10},
                "dependencies": []
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_numeric_2'.encode()).hexdigest()[:8]}",
                "task_type": "analysis",
                "description": "Анализ методов измерения и определение контекстных условий",
                "priority": self.priority * 0.8,
                "scheduled_time": time.time() + 300,
                "required_capabilities": ["nlp_analysis", "source_reputation"],
                "parameters": {"analysis_type": "methodology_comparison"},
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_numeric_1'.encode()).hexdigest()[:8]}"]
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_numeric_3'.encode()).hexdigest()[:8]}",
                "task_type": "integration",
                "description": "Интеграция данных и определение наиболее точного значения с учетом контекста",
                "priority": self.priority * 0.7,
                "scheduled_time": time.time() + 1800,
                "required_capabilities": ["knowledge_integration"],
                "parameters": {"integration_strategy": "contextual_weighting"},
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
                "scheduled_time": time.time() + 60,
                "required_capabilities": ["nlp_analysis", "context_detection"],
                "parameters": {"analysis_type": "condition_extraction"},
                "dependencies": []
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_boolean_2'.encode()).hexdigest()[:8]}",
                "task_type": "evidence_collection",
                "description": "Сбор доказательств для подтверждения условий применения",
                "priority": self.priority * 0.8,
                "scheduled_time": time.time() + 300,
                "required_capabilities": ["web_search"],
                "parameters": {"query": f"{self.concept} условия применения", "num_results": 8},
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_boolean_1'.encode()).hexdigest()[:8]}"]
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_boolean_3'.encode()).hexdigest()[:8]}",
                "task_type": "synthesis",
                "description": "Создание контекстно-зависимого ответа",
                "priority": self.priority * 0.7,
                "scheduled_time": time.time() + 1800,
                "required_capabilities": ["knowledge_integration", "response_generation"],
                "parameters": {"synthesis_strategy": "conditional_response"},
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
                "scheduled_time": time.time() + 60,
                "required_capabilities": ["taxonomy_analysis"],
                "parameters": {"analysis_type": "category_hierarchy"},
                "dependencies": []
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_exclusivity_2'.encode()).hexdigest()[:8]}",
                "task_type": "evidence_collection",
                "description": "Сбор данных для определения взаимоисключающих условий",
                "priority": self.priority * 0.8,
                "scheduled_time": time.time() + 300,
                "required_capabilities": ["web_search"],
                "parameters": {"query": f"{self.concept} взаимоисключающие условия", "num_results": 7},
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_exclusivity_1'.encode()).hexdigest()[:8]}"]
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_exclusivity_3'.encode()).hexdigest()[:8]}",
                "task_type": "taxonomy_revision",
                "description": "Переработка иерархии категорий и разделение на подкатегории",
                "priority": self.priority * 0.7,
                "scheduled_time": time.time() + 1800,
                "required_capabilities": ["knowledge_integration", "taxonomy_management"],
                "parameters": {"revision_strategy": "subcategorization"},
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
                "scheduled_time": time.time() + 60,
                "required_capabilities": ["knowledge_graph_analysis"],
                "parameters": {"analysis_type": "cycle_detection"},
                "dependencies": []
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_hierarchy_2'.encode()).hexdigest()[:8]}",
                "task_type": "taxonomy_review",
                "description": "Анализ взаимоисключающих классификаций",
                "priority": self.priority * 0.8,
                "scheduled_time": time.time() + 300,
                "required_capabilities": ["taxonomy_analysis"],
                "parameters": {"review_type": "classification_conflicts"},
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_hierarchy_1'.encode()).hexdigest()[:8]}"]
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_hierarchy_3'.encode()).hexdigest()[:8]}",
                "task_type": "hierarchy_reconstruction",
                "description": "Перестройка иерархии для устранения противоречий",
                "priority": self.priority * 0.7,
                "scheduled_time": time.time() + 1800,
                "required_capabilities": ["knowledge_integration", "taxonomy_management"],
                "parameters": {"reconstruction_strategy": "hierarchy_flattening"},
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
                "scheduled_time": time.time() + 60,
                "required_capabilities": ["audience_analysis", "context_detection"],
                "parameters": {"analysis_type": "audience_segmentation"},
                "dependencies": []
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_response_2'.encode()).hexdigest()[:8]}",
                "task_type": "response_comparison",
                "description": "Сравнение ответов и определение границ применимости",
                "priority": self.priority * 0.8,
                "scheduled_time": time.time() + 300,
                "required_capabilities": ["nlp_analysis", "semantic_comparison"],
                "parameters": {"comparison_type": "contextual_boundaries"},
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_response_1'.encode()).hexdigest()[:8]}"]
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_response_3'.encode()).hexdigest()[:8]}",
                "task_type": "response_integration",
                "description": "Создание условных ответов для разных сценариев",
                "priority": self.priority * 0.7,
                "scheduled_time": time.time() + 1800,
                "required_capabilities": ["response_generation", "context_management"],
                "parameters": {"integration_strategy": "scenario_based"},
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
                "scheduled_time": time.time() + 60,
                "required_capabilities": ["contradiction_analysis"],
                "parameters": {"analysis_type": "contradiction_typing"},
                "dependencies": []
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_general_2'.encode()).hexdigest()[:8]}",
                "task_type": "context_research",
                "description": "Исследование контекстных условий и сбор дополнительных данных",
                "priority": self.priority * 0.8,
                "scheduled_time": time.time() + 300,
                "required_capabilities": ["web_search", "context_analysis"],
                "parameters": {"research_focus": "contextual_conditions"},
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_general_1'.encode()).hexdigest()[:8]}"]
            },
            {
                "task_id": f"task_{hashlib.md5(f'{self.id}_general_3'.encode()).hexdigest()[:8]}",
                "task_type": "solution_synthesis",
                "description": "Синтез решения и документирование",
                "priority": self.priority * 0.7,
                "scheduled_time": time.time() + 1800,
                "required_capabilities": ["knowledge_integration", "solution_synthesis"],
                "parameters": {"synthesis_type": "comprehensive"},
                "dependencies": [f"task_{hashlib.md5(f'{self.id}_general_2'.encode()).hexdigest()[:8]}"]
            }
        ]
        return tasks
