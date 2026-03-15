"""Главный модуль управления знаниями для CogniFlex - объединяет все компоненты"""
import os
import logging
import time
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger("cogniflex.knowledge_manager")

# Импортируем компоненты
from .knowledge_core import KnowledgeGraph
from .knowledge_analyzer import KnowledgeAnalyzer
from .knowledge_integrator import KnowledgeIntegrator

class KnowledgeManager:
    """Модуль управления знаниями для CogniFlex - объединяет все компоненты."""
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None):
        """
        Инициализирует модуль управления знаниями.
        
        Args:
            brain: Ссылка на ядро CogniFlex (опционально)
            cache_dir: Путь к директории кэша (опционально)
        """
        self.brain = brain
        self.cache_dir = cache_dir
        
        try:
            # Создаем компоненты
            self.knowledge_graph = KnowledgeGraph(cache_dir=cache_dir, brain=brain)
            self.knowledge_analyzer = KnowledgeAnalyzer(self.knowledge_graph, brain)
            self.knowledge_integrator = KnowledgeIntegrator(brain, knowledge_graph=self.knowledge_graph, knowledge_analyzer=self.knowledge_analyzer)
            
            # Устанавливаем ссылки между компонентами
            if brain:
                brain.knowledge_graph = self.knowledge_graph
                brain.knowledge_analyzer = self.knowledge_analyzer
                brain.knowledge_integrator = self.knowledge_integrator
            
            logger.info("KnowledgeManager полностью инициализирован")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации KnowledgeManager: {e}")
            raise
    
    def get_system_health(self) -> Dict[str, Any]:
        """Возвращает отчет о здоровье модуля управления знаниями."""
        return {
            "knowledge_graph_ready": self.knowledge_graph.is_ready(),
            "total_nodes": len(self.knowledge_graph.nodes),
            "total_edges": len(self.knowledge_graph.edges),
            "timestamp": time.time()
        }
    
    def generate_system_report(self) -> str:
        """Генерирует текстовый отчет о состоянии модуля управления знаниями."""
        health = self.get_system_health()
        return f"Knowledge Manager: {health['total_nodes']} nodes, {health['total_edges']} edges"
    
    def add_concept(self, concept: str, description: str, 
                   domain: str, strength: float = 1.0, 
                   source: Optional[str] = None, 
                   tags: Optional[List[str]] = None,
                   user_id: Optional[str] = None) -> str:
        """
        Добавляет концепт в граф знаний.
        
        Args:
            concept: Название концепта
            description: Описание концепта
            domain: Домен знаний
            strength: Сила знания (0.0-1.0)
            source: Источник информации
            tags: Теги для концепта
            user_id: ID пользователя, добавившего концепт
            
        Returns:
            str: ID добавленного концепта
        """
        if not concept or not description or not domain:
            raise ValueError("concept, description и domain обязательны")
        
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength должен быть между 0.0 и 1.0")
        
        from .knowledge_core import KnowledgeNode
        import hashlib
        import time
        
        node_id = hashlib.md5(f"{concept}_{domain}_{time.time()}".encode()).hexdigest()
        node = KnowledgeNode(
            id=node_id,
            content=description,
            node_type="concept",
            domain=domain,
            strength=strength,
            meta={"source": source, "tags": tags or [], "user_id": user_id}
        )
        
        if self.knowledge_graph.add_node(node):
            return node_id
        return ""
    
    def add_connection(self, source_id: str, target_id: str, 
                      relation: str, strength: float = 0.8, 
                      metadata: Optional[Dict[str, Any]] = None,
                      user_id: Optional[str] = None) -> str:
        """
        Добавляет связь между концептами.
        
        Args:
            source_id: ID исходного концепта
            target_id: ID целевого концепта
            relation: Тип связи
            strength: Сила связи (0.0-1.0)
            metadata: Дополнительные метаданные
            user_id: ID пользователя, добавившего связь
            
        Returns:
            str: ID добавленной связи
        """
        if not source_id or not target_id or not relation:
            raise ValueError("source_id, target_id и relation обязательны")
            
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength должен быть между 0.0 и 1.0")
        
        from .knowledge_core import KnowledgeEdge
        import hashlib
        import time
        
        edge_id = hashlib.md5(f"{source_id}_{target_id}_{relation}_{time.time()}".encode()).hexdigest()
        edge = KnowledgeEdge(
            id=edge_id,
            source=source_id,
            target=target_id,
            relation=relation,
            strength=strength,
            meta=metadata or {}
        )
        
        if self.knowledge_graph.add_edge(edge):
            return edge_id
        return ""
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Получает узел по ID.
        
        Args:
            node_id: ID узла
            
        Returns:
            Optional[Dict[str, Any]]: Узел или None
        """
        if not node_id:
            return None
            
        node = self.knowledge_graph.get_node(node_id)
        return node.to_dict() if node else None
    
    def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """
        Получает связь по ID.
        
        Args:
            edge_id: ID связи
            
        Returns:
            Optional[Dict[str, Any]]: Связь или None
        """
        if not edge_id:
            return None
            
        edge = self.knowledge_graph.get_edge(edge_id)
        return edge.to_dict() if edge else None
    
    def get_edges(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Получает все связи для узла.
        
        Args:
            node_id: ID узла
            
        Returns:
            List[Dict[str, Any]]: Список связей
        """
        if not node_id:
            return []
            
        edges = self.knowledge_graph.get_edges(node_id)
        return [edge.to_dict() for edge in edges]
    
    def get_related_nodes(self, node_id: str, max_distance: int = 2) -> List[Dict[str, Any]]:
        """
        Получает связанные узлы с указанным максимальным расстоянием.
        
        Args:
            node_id: ID узла
            max_distance: Максимальное расстояние
            
        Returns:
            List[Dict[str, Any]]: Список связанных узлов
        """
        if not node_id:
            return []
            
        nodes = self.knowledge_graph.get_related_nodes(node_id, max_distance)
        return [node.to_dict() for node in nodes]
    
    def search_nodes(self, query: str, domain: Optional[str] = None, 
                    limit: int = 10, min_strength: float = 0.3) -> List[Dict[str, Any]]:
        """
        Ищет узлы по запросу.
        
        Args:
            query: Поисковый запрос
            domain: Фильтр по домену
            limit: Максимальное количество результатов
            min_strength: Минимальная сила знания
            
        Returns:
            List[Dict[str, Any]]: Список найденных узлов
        """
        if not query:
            return []
            
        nodes = self.knowledge_graph.search_nodes(query, domain, limit, min_strength)
        return [node.to_dict() for node in nodes]
    
    def get_all_nodes(self, limit: int = 1000, min_strength: float = 0.3) -> List[Dict[str, Any]]:
        """
        Получает все узлы (с ограничением).
        
        Args:
            limit: Максимальное количество узлов
            min_strength: Минимальная сила знания
            
        Returns:
            List[Dict[str, Any]]: Список узлов
        """
        nodes = self.knowledge_graph.get_all_nodes(limit, min_strength)
        return [node.to_dict() for node in nodes]
    
    def update_concept(self, concept_id: str, new_description: str, 
                      strength: Optional[float] = None, 
                      source: Optional[str] = None,
                      user_id: Optional[str] = None) -> bool:
        """
        Обновляет концепт в графе знаний.
        
        Args:
            concept_id: ID концепта
            new_description: Новое описание
            strength: Новая сила знания (опционально)
            source: Новый источник (опционально)
            user_id: ID пользователя, внесшего изменения
            
        Returns:
            bool: Успешно ли обновлено
        """
        if not concept_id or not new_description:
            return False
            
        if strength is not None and not 0.0 <= strength <= 1.0:
            return False
            
        return self.knowledge_graph.update_concept(
            concept_id, new_description, strength, source, user_id)
    
    def get_node_details(self, node_id: str, max_distance: int = 2) -> Dict[str, Any]:
        """
        Получает подробную информацию об узле и его связях.
        
        Args:
            node_id: ID узла
            max_distance: Максимальное расстояние для получения связей
            
        Returns:
            Dict[str, Any]: Подробная информация об узле
        """
        if not node_id:
            return {}
            
        return self.knowledge_graph.get_node_details(node_id, max_distance)
    
    def analyze_knowledge_gaps(self, domain: Optional[str] = None, 
                             num_samples: int = 10) -> List[Dict[str, Any]]:
        """
        Анализирует пробелы в знаниях в указанной области.
        
        Args:
            domain: Область знаний (опционально)
            num_samples: Количество примеров для анализа
            
        Returns:
            List[Dict[str, Any]]: Выявленные пробелы в знаниях
        """
        return self.knowledge_analyzer.analyze_knowledge_gaps(domain, num_samples)
    
    def analyze_contradictions(self) -> List[Dict[str, Any]]:
        """
        Анализирует противоречия в знаниях.
        
        Returns:
            List[Dict[str, Any]]: Выявленные противоречия
        """
        return self.knowledge_analyzer.detect_contradictions()
    
    def analyze_knowledge_coverage(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Анализирует покрытие знаний в указанной области.
        
        Args:
            domain: Область знаний (опционально)
            
        Returns:
            Dict[str, Any]: Результат анализа
        """
        return self.knowledge_analyzer.analyze_knowledge_coverage(domain)
    
    def analyze_knowledge_recency(self) -> Dict[str, Any]:
        """
        Анализирует актуальность знаний.
        
        Returns:
            Dict[str, Any]: Результат анализа
        """
        return self.knowledge_analyzer.analyze_knowledge_recency()
    
    def analyze_knowledge_patterns(self) -> Dict[str, Any]:
        """
        Анализирует паттерны знаний для выявления возможностей.
        
        Returns:
            Dict[str, Any]: Результат анализа
        """
        return self.knowledge_analyzer.analyze_knowledge_patterns()
    
    def analyze_knowledge_evolution(self) -> Dict[str, Any]:
        """
        Анализирует эволюцию знаний со временем.
        
        Returns:
            Dict[str, Any]: Результат анализа
        """
        return self.knowledge_analyzer.analyze_knowledge_evolution()
    
    def analyze_concept_evolution(self, concept: str) -> Dict[str, Any]:
        """
        Анализирует эволюцию конкретного концепта со временем.
        
        Args:
            concept: Концепт для анализа
            
        Returns:
            Dict[str, Any]: Результат анализа
        """
        if not concept:
            return {}
            
        return self.knowledge_analyzer.analyze_concept_evolution(concept)
    
    def expand_knowledge(self, concept: str, depth: int = 1, 
                        num_results: int = 3) -> List[Dict[str, Any]]:
        """
        Расширяет знания по указанному концепту.
        
        Args:
            concept: Концепт для расширения
            depth: Глубина расширения
            num_results: Количество результатов
            
        Returns:
            List[Dict[str, Any]]: Расширенные знания
        """
        if not concept:
            return []
            
        return self.knowledge_updater.expand_knowledge(concept, depth, num_results)
    
    def update_outdated_knowledge(self, domain: Optional[str] = None, 
                                max_age_days: int = 365) -> int:
        """
        Обновляет устаревшие знания.
        
        Args:
            domain: Область знаний (опционально)
            max_age_days: Максимальный возраст знаний в днях
            
        Returns:
            int: Количество обновленных знаний
        """
        return self.knowledge_updater.update_outdated_knowledge(domain, max_age_days)
    
    def integrate_knowledge(self, concept: str, depth: int = 1) -> bool:
        """
        Интегрирует знания, заполняя пробелы и разрешая противоречия.
        
        Args:
            concept: Концепт для интеграции
            depth: Глубина интеграции
            
        Returns:
            bool: Успешно ли выполнено
        """
        if not concept:
            return False
            
        return self.knowledge_integrator.integrate_knowledge(concept, depth)
    
    def learn_from_user_feedback(self, feedback: Dict[str, Any], user_id: Optional[str] = None) -> bool:
        """
        Учится на основе пользовательского фидбэка.
        
        Args:
            feedback: Данные фидбэка
            user_id: ID пользователя, предоставившего фидбэк
            
        Returns:
            bool: Успешно ли выполнено
        """
        if not feedback:
            return False
            
        return self.knowledge_integrator.learn_from_user_feedback(feedback, user_id)
    
    def auto_update_knowledge(self, max_nodes: Optional[int] = None, 
                            min_strength: Optional[float] = None):
        """
        Автоматически обновляет знания.
        
        Args:
            max_nodes: Максимальное количество узлов
            min_strength: Минимальная сила знаний
        """
        self.knowledge_updater.auto_update_knowledge(max_nodes, min_strength)
    
    def generate_explanation(self, concept: str, level: float = 0.5) -> str:
        """
        Генерирует объяснение концепта с адаптированной детализацией.
        
        Args:
            concept: Концепт для объяснения
            level: Уровень детализации (0.0-1.0)
            
        Returns:
            str: Сгенерированное объяснение
        """
        if not concept:
            return ""
            
        if not 0.0 <= level <= 1.0:
            level = 0.5
            
        return self.knowledge_updater.generate_explanation(concept, level)
    
    def generate_comparison(self, concept1: str, concept2: str) -> str:
        """
        Генерирует сравнение двух концептов.
        
        Args:
            concept1: Первый концепт
            concept2: Второй концепт
            
        Returns:
            str: Сгенерированное сравнение
        """
        if not concept1 or not concept2:
            return ""
            
        return self.knowledge_updater.generate_comparison(concept1, concept2)
    
    def get_domain_statistics(self) -> Dict[str, Any]:
        """
        Получает статистику по доменам знаний.
        
        Returns:
            Dict[str, Any]: Статистика по доменам
        """
        return self.knowledge_graph.get_domain_statistics()
    
    def generate_knowledge_graph(self, concept: str, depth: int = 2) -> Dict[str, Any]:
        """
        Генерирует граф знаний для визуализации.
        
        Args:
            concept: Концепт для построения графа
            depth: Глубина графа
            
        Returns:
            Dict[str, Any]: Граф знаний в формате для визуализации
        """
        if not concept:
            return {}
            
        return self.knowledge_graph.generate_knowledge_graph(concept, depth)
    
    def get_history(self, node_id: Optional[str] = None, days: int = 30) -> List[Dict[str, Any]]:
        """
        Получает историю изменений за указанный период.
        
        Args:
            node_id: ID узла (опционально)
            days: Количество дней для истории
            
        Returns:
            List[Dict[str, Any]]: История изменений
        """
        return self.knowledge_core.get_history(node_id, days)
    
    def get_recent_changes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Получает недавние изменения в знаниях.
        
        Args:
            limit: Максимальное количество изменений
            
        Returns:
            List[Dict[str, Any]]: Недавние изменения
        """
        return self.knowledge_core.get_recent_changes(limit)
    
    def close(self):
        """Закрывает модуль управления знаниями и освобождает ресурсы."""
        logger.info("Закрытие модуля управления знаниями...")
        
        try:
            # Закрываем компоненты
            if hasattr(self, 'knowledge_graph'):
                self.knowledge_graph.close()
        except Exception as e:
            logger.error(f"Ошибка при закрытии: {e}")
        
        logger.info("Модуль управления знаниями закрыт")