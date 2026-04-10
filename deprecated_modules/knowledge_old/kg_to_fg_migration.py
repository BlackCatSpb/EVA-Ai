"""
Миграция данных из KnowledgeGraph в FractalGraphV2.
Переносит узлы и связи из KG в единый фрактальный граф.
"""
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("eva_ai.knowledge.migration")

class KnowledgeMigration:
    """
    Мигратор данных из KnowledgeGraph в FractalGraphV2.
    """
    
    def __init__(self, brain=None):
        self.brain = brain
        self.migrated_count = 0
        self.skipped_count = 0
        
    def migrate_all(self) -> Dict[str, Any]:
        """
        Перенести все данные из KG в FGv2.
        
        Returns:
            Dict с результатами миграции
        """
        logger.info("Начало миграции KnowledgeGraph -> FractalGraphV2")
        
        kg = getattr(self.brain, 'knowledge_graph', None)
        fg = getattr(self.brain, 'fractal_graph_v2', None)
        
        if not kg:
            logger.warning("KnowledgeGraph не найден на brain")
            return {'status': 'no_kg', 'migrated': 0}
        
        if not fg:
            logger.warning("FractalGraphV2 не найден на brain")
            return {'status': 'no_fg', 'migrated': 0}
        
        # Получаем узлы из KG
        try:
            kg_nodes = kg.get_all_nodes() if hasattr(kg, 'get_all_nodes') else []
        except Exception as e:
            logger.error(f"Ошибка получения узлов из KG: {e}")
            kg_nodes = []
        
        logger.info(f"Найдено {len(kg_nodes)} узлов в KnowledgeGraph")
        
        # Миграция узлов
        for node in kg_nodes:
            self._migrate_node(node, fg)
        
        # Миграция связей
        try:
            kg_edges = kg.get_all_edges() if hasattr(kg, 'get_all_edges') else []
        except Exception as e:
            logger.error(f"Ошибка получения связей из KG: {e}")
            kg_edges = []
        
        for edge in kg_edges:
            self._migrate_edge(edge, fg)
        
        logger.info(f"Миграция завершена: {self.migrated_count} перенесено, {self.skipped_count} пропущено")
        
        return {
            'status': 'complete',
            'migrated': self.migrated_count,
            'skipped': self.skipped_count,
            'kg_nodes': len(kg_nodes),
            'kg_edges': len(kg_edges)
        }
    
    def _migrate_node(self, node, fg) -> bool:
        """Мигрировать один узел."""
        try:
            # Извлекаем данные из узла
            if hasattr(node, 'id'):
                node_id = node.id
            elif isinstance(node, dict):
                node_id = node.get('id')
            else:
                self.skipped_count += 1
                return False
            
            if hasattr(node, 'name'):
                name = node.name
            elif isinstance(node, dict):
                name = node.get('name', node_id)
            else:
                name = str(node_id)
            
            if hasattr(node, 'description'):
                content = node.description
            elif isinstance(node, dict):
                content = node.get('description', node.get('content', ''))
            else:
                content = str(node)
            
            if hasattr(node, 'node_type'):
                node_type = node.node_type
            elif isinstance(node, dict):
                node_type = node.get('type', node.get('node_type', 'knowledge'))
            else:
                node_type = 'knowledge'
            
            if hasattr(node, 'domain'):
                domain = node.domain
            elif isinstance(node, dict):
                domain = node.get('domain', 'general')
            else:
                domain = 'general'
            
            # Пропускаем мусор
            if self._is_garbage(content, name):
                self.skipped_count += 1
                return False
            
            # Добавляем в FGv2
            if fg:
                try:
                    fg.add_node(
                        content=content,
                        node_type=node_type,
                        level=2,
                        metadata={
                            'domain': domain,
                            'source': 'kg_migration',
                            'original_id': node_id,
                            'name': name
                        }
                    )
                    self.migrated_count += 1
                    return True
                except Exception as e:
                    logger.debug(f"Ошибка добавления узла {node_id}: {e}")
                    self.skipped_count += 1
                    return False
            
            self.skipped_count += 1
            return False
            
        except Exception as e:
            logger.debug(f"Ошибка миграции узла: {e}")
            self.skipped_count += 1
            return False
    
    def _migrate_edge(self, edge, fg) -> bool:
        """Мигрировать связь."""
        try:
            if hasattr(edge, 'source_id'):
                source = edge.source_id
            elif isinstance(edge, dict):
                source = edge.get('source', edge.get('source_id'))
            else:
                return False
                
            if hasattr(edge, 'target_id'):
                target = edge.target_id
            elif isinstance(edge, dict):
                target = edge.get('target', edge.get('target_id'))
            else:
                return False
            
            if hasattr(edge, 'relation_type'):
                relation = edge.relation_type
            elif isinstance(edge, dict):
                relation = edge.get('relation', edge.get('relation_type', 'related'))
            else:
                relation = 'related'
            
            if not source or not target:
                return False
            
            # В FGv2 добавляем как knowledge
            if fg:
                try:
                    fg.add_knowledge(source, relation, target)
                    return True
                except Exception:
                    pass
            
            return False
            
        except Exception:
            return False
    
    def _is_garbage(self, content: str, name: str) -> bool:
        """Проверить является ли контент мусором."""
        garbage_patterns = [
            'продолжим разговор', 'перспективы развития',
            '###', '##', 'q:', 'a:', 'пример:',
            'особенности данного'
        ]
        
        text = (content + ' ' + name).lower()
        return any(p in text for p in garbage_patterns)


def migrate_knowledge_graph(brain) -> Dict[str, Any]:
    """Удобная функция для запуска миграции."""
    migrator = KnowledgeMigration(brain)
    return migrator.migrate_all()