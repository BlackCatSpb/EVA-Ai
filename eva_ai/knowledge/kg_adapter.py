"""
KG Compatibility Layer - перенаправляет вызовы Knowledge Graph на FractalGraph v2
Используется для обратной совместимости пока система полностью не перейдёт на FGv2
"""
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class KnowledgeGraphAdapter:
    """
    Адаптер для обеспечения совместимости с кодом, использующим KnowledgeGraph API,
    путём перенаправления вызовов на FractalGraph v2.
    """
    
    def __init__(self, fractal_graph):
        """
        Args:
            fractal_graph: Экземпляр FractalMemoryGraph
        """
        self._fg = fractal_graph
        self.stats = {
            'total_nodes': 0,
            'total_edges': 0,
            'migrated_from_kg': True
        }
    
    @property
    def nodes(self) -> Dict:
        """Возвращает узлы как словарь (совместимость с KG)."""
        return self._fg.storage.nodes
    
    @property
    def edges(self) -> Dict:
        """Возвращает связи как словарь."""
        return self._fg.storage.edges
    
    def get_recent_entities(self, limit: int = 20) -> List[Dict]:
        """Получить недавние сущности."""
        nodes = list(self._fg.storage.nodes.values())
        nodes.sort(key=lambda x: getattr(x, 'created_at', 0), reverse=True)
        return [
            {
                'name': getattr(n, 'content', str(n))[:50],
                'content': getattr(n, 'content', str(n)),
                'type': getattr(n, 'node_type', 'concept')
            }
            for n in nodes[:limit]
        ]
    
    def add_entity(self, name: str, entity_type: str = 'concept', properties: Dict = None) -> str:
        """Добавить сущность."""
        node = self._fg.add_node(
            content=name,
            node_type=entity_type,
            metadata=properties or {}
        )
        return node.id if node else name
    
    def add_concept(self, concept: str, content: str = None, domain: str = 'general', source: str = 'system') -> str:
        """Добавить концепцию."""
        content = content or concept
        node = self._fg.add_node(
            content=content,
            node_type='concept',
            metadata={'domain': domain, 'source': source, 'name': concept}
        )
        return node.id if node else concept
    
    def add_relation(self, source: str, target: str, relation_type: str = 'related') -> str:
        """Добавить связь между концепциями."""
        edge_id = f"{source}_{relation_type}_{target}"
        self._fg.add_edge(source, target, edge_type=relation_type)
        return edge_id
    
    def find_related(self, concept: str, limit: int = 5) -> List[Dict]:
        """Найти связанные концепции."""
        results = self._fg.semantic_search(concept, top_k=limit)
        return [
            {
                'name': r.get('content', '')[:50],
                'content': r.get('content', ''),
                'score': r.get('score', 0)
            }
            for r in results
        ]
    
    def get_related_concepts(self, concept: str) -> List[str]:
        """Получить список связанных концепций."""
        related = self.find_related(concept, limit=5)
        return [r['name'] for r in related]
    
    def find_path_between_concepts(self, concept1: str, concept2: str) -> List[str]:
        """Найти путь между концепциями (упрощённый)."""
        related1 = self.get_related_concepts(concept1)
        if concept2 in related1:
            return [concept1, concept2]
        return [concept1] + related1[:2] + [concept2]
    
    def get_entity_facts(self, entity: str) -> List[str]:
        """Получить факты о сущности."""
        results = self._fg.semantic_search(entity, top_k=3)
        return [r.get('content', '') for r in results]
    
    def search_nodes(self, query: str, limit: int = 10) -> List[Dict]:
        """Поиск узлов."""
        results = self._fg.semantic_search(query, top_k=limit)
        return results
    
    def get_stats(self) -> Dict[str, int]:
        """Получить статистику."""
        return {
            'total_nodes': len(self._fg.storage.nodes),
            'total_edges': len(self._fg.storage.edges),
            'total_groups': len(self._fg.storage.semantic_groups)
        }
    
    def __getattr__(self, name: str):
        """Перенаправление вызовов на FGv2."""
        if hasattr(self._fg, name):
            return getattr(self._fg, name)
        logger.debug(f"KG Adapter: метод {name} не найден, используется заглушка")
        return lambda *args, **kwargs: None
