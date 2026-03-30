"""Ядро системы знаний ЕВА с поддержкой анализа противоречий"""
import os
import logging
import time
import sqlite3
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from collections import defaultdict, deque
from enum import Enum
from .knowledge_hybrid_index import KnowledgeHybridIndex

logger = logging.getLogger("eva.knowledge.core")

class RelationType(Enum):
    """Типы отношений в графе знаний."""
    IS_A = "is_a"
    PART_OF = "part_of" 
    MEMBER_OF = "member_of"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    RELATED_TO = "related_to"

class KnowledgeNode:
    """Представляет узел в графе знаний."""
    
    def __init__(self, id: str, content: Any, node_type: str = "fact", 
                 domain: str = "general", strength: float = 0.5, 
                 timestamp: Optional[float] = None, meta: Optional[Dict] = None):
        """
        Инициализирует узел знаний.
        
        Args:
            id: Уникальный идентификатор узла
            content: Содержимое узла (текст, число, словарь и т.д.)
            node_type: Тип узла (fact, concept, belief и т.д.)
            domain: Домен знаний
            strength: Сила узла (0.0-1.0)
            timestamp: Временная метка создания
            meta: Метаданные узла
        """
        self.id = id
        self.content = content
        self.node_type = node_type
        self.domain = domain
        self.strength = max(0.0, min(1.0, strength))  # Ограничиваем диапазон 0.0-1.0
        self.timestamp = timestamp or time.time()
        self.meta = meta or {}
        self.context = self.meta.get('context', '')
        
        # Для совместимости с менеджером противоречий
        self.get_strength_factor = self.get_strength_factor
        self.update_strength = self.update_strength
    
    def get_strength_factor(self) -> float:
        """
        Возвращает фактор силы узла с учетом времени и подтверждений.
        
        Returns:
            float: Фактор силы (0.0-1.0)
        """
        # Сила уменьшается со временем (экспоненциальное затухание)
        time_factor = 0.95 ** ((time.time() - self.timestamp) / 86400)  # Затухание за день
        
        # Учитываем метаданные
        verification_factor = self.meta.get('verification_factor', 1.0)
        source_reputation = self.meta.get('source_reputation', 0.5)
        
        # Вычисляем итоговую силу
        strength = self.strength * time_factor * verification_factor * (0.5 + source_reputation * 0.5)
        return max(0.1, min(1.0, strength))
    
    def update_strength(self, verification_factor: float = 1.0, 
                        time_factor: float = 1.0, source_reputation: float = None):
        """
        Обновляет силу узла на основе новых данных.
        
        Args:
            verification_factor: Фактор подтверждения (0.0-2.0)
            time_factor: Фактор времени (обычно 1.0)
            source_reputation: Репутация источника (0.0-1.0)
        """
        # Базовая сила
        base_strength = self.strength
        
        # Учитываем репутацию источника, если она предоставлена
        if source_reputation is not None:
            if self.meta is None:
                self.meta = {}
            self.meta['source_reputation'] = source_reputation
            source_factor = 0.5 + source_reputation * 0.5
        else:
            source_factor = 1.0
        
        # Обновляем силу
        self.strength = min(1.0, base_strength * verification_factor * time_factor * source_factor)
        
        # Обновляем временную метку
        self.timestamp = time.time()
        
        # Сохраняем факторы в метаданных
        if self.meta is None:
            self.meta = {}
        self.meta['verification_factor'] = verification_factor
        self.meta['last_update'] = self.timestamp
    
    def add_context(self, context: str):
        """
        Добавляет контекст к узлу.
        
        Args:
            context: Контекстное описание
        """
        self.context = context
        if self.meta is None:
            self.meta = {}
        self.meta['context'] = context
    
    def get_content_summary(self, max_length: int = 100) -> str:
        """
        Возвращает краткое описание содержимого узла.
        
        Args:
            max_length: Максимальная длина описания
            
        Returns:
            str: Краткое описание
        """
        if isinstance(self.content, str):
            return self.content[:max_length] + "..." if len(self.content) > max_length else self.content
        elif isinstance(self.content, (int, float)):
            return str(self.content)
        elif isinstance(self.content, dict):
            # Пытаемся найти наиболее релевантное текстовое поле
            for field in ['description', 'text', 'content', 'value']:
                if field in self.content and isinstance(self.content[field], str):
                    return self.content[field][:max_length] + "..." if len(self.content[field]) > max_length else self.content[field]
            return str(self.content)[:max_length] + "..."
        else:
            return str(self.content)[:max_length] + "..."
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует узел в словарь для сериализации."""
        return {
            "id": self.id,
            "content": self.content,
            "node_type": self.node_type,
            "domain": self.domain,
            "strength": self.strength,
            "timestamp": self.timestamp,
            "meta": self.meta
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeNode':
        """Создает узел из словаря."""
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            node_type=data.get("node_type", "fact"),
            domain=data.get("domain", "general"),
            strength=data.get("strength", 0.5),
            timestamp=data.get("timestamp"),
            meta=data.get("meta")
        )

class KnowledgeEdge:
    """Представляет связь между узлами в графе знаний."""
    
    def __init__(self, id: str, source: str, target: str, relation: str,
                 strength: float = 0.5, timestamp: Optional[float] = None, 
                 meta: Optional[Dict] = None):
        """
        Инициализирует связь между узлами.
        
        Args:
            id: Уникальный идентификатор связи
            source: ID исходного узла
            target: ID целевого узла
            relation: Тип отношения
            strength: Сила связи (0.0-1.0)
            timestamp: Временная метка создания
            meta: Метаданные связи
        """
        self.id = id
        self.source = source
        self.target = target
        self.relation = relation
        self.strength = max(0.0, min(1.0, strength))
        self.timestamp = timestamp or time.time()
        self.meta = meta or {}
        
        # Для совместимости с менеджером противоречий
        self.get_priority = self.get_priority
    
    def get_priority(self) -> float:
        """
        Возвращает приоритет связи с учетом силы и времени.
        
        Returns:
            float: Приоритет связи
        """
        # Сила уменьшается со временем
        time_factor = 0.95 ** ((time.time() - self.timestamp) / 86400)
        return self.strength * time_factor
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует связь в словарь для сериализации."""
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "strength": self.strength,
            "timestamp": self.timestamp,
            "meta": self.meta
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEdge':
        """Создает связь из словаря."""
        return cls(
            id=data.get("id", ""),
            source=data.get("source", ""),
            target=data.get("target", ""),
            relation=data.get("relation", "related_to"),
            strength=data.get("strength", 0.5),
            timestamp=data.get("timestamp"),
            meta=data.get("meta")
        )

class KnowledgeGraph:
    """Граф знаний с поддержкой анализа противоречий."""
    
    def __init__(self, storage_path: Optional[str] = None, brain=None, cache_dir: Optional[str] = None):
        """
        Инициализирует граф знаний.
        
        Args:
            storage_path: Путь к файлу хранения
            brain: Ссылка на ядро системы (опционально)
            cache_dir: Путь к директории кэша (опционально)
        """
        self.brain = brain
        self.cache_dir = cache_dir
        self.storage_path = storage_path or os.path.join(cache_dir or os.getcwd(), "eva_knowledge.db")
        self.initialized = False
        self.nodes = {}  # id -> KnowledgeNode
        self.edges = {}  # id -> KnowledgeEdge
        self.node_edges = defaultdict(list)  # node_id -> list of edge ids
        self.domains = defaultdict(list)  # domain -> list of node ids
        self.contexts = defaultdict(list)  # context -> list of node ids
        
        # Статистика
        self.stats = {
            "total_nodes": 0,
            "total_edges": 0,
            "domains": set(),
            "last_update": time.time()
        }
        
        # Гибридный индекс (LRU + DiskCache) для узлов и связей графа знаний
        try:
            base_cache_dir = (
                getattr(self.brain, 'cache_dir', None)
                or self.cache_dir
                or os.path.join(os.getcwd(), 'eva_cache')
            )
            self.hybrid_index = KnowledgeHybridIndex(base_cache_dir=base_cache_dir, namespace="kg_index")
        except Exception as e:
            logger.warning(f"Ошибка инициализации гибридного индекса: {e}")
            self.hybrid_index = None

        logger.info("Граф знаний инициализирован")
    
    def is_ready(self) -> bool:
        """Проверяет готовность графа знаний к работе."""
        return self.initialized and len(self.nodes) >= 0
    
    def search_nodes(self, query: str, domain: Optional[str] = None, limit: int = 10, min_strength: float = 0.3) -> List[KnowledgeNode]:
        """Поиск узлов по запросу."""
        results = []
        query_lower = query.lower()
        
        for node in self.nodes.values():
            if domain and node.domain != domain:
                continue
            if node.strength < min_strength:
                continue
            
            content_str = str(node.content).lower()
            if query_lower in content_str:
                results.append(node)
                if len(results) >= limit:
                    break
        
        return results
    
    def initialize(self) -> bool:
        """Инициализирует граф знаний и загружает данные из хранилища."""
        if self.initialized:
            return True
        
        try:
            # Создаем базу данных
            self.db = self._init_database()
            
            # Загружаем данные
            self._load_from_storage()
            
            # Обновляем статистику
            self._update_stats()
            
            self.initialized = True
            logger.info("Граф знаний успешно инициализирован")
            return True
            
        except Exception as e:
            logger.critical(f"Критическая ошибка при инициализации графа знаний: {e}")
            return False
    
    def _init_database(self):
        """Инициализирует базу данных для хранения графа знаний."""
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            # Таблица для узлов
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                node_type TEXT NOT NULL,
                domain TEXT NOT NULL,
                strength REAL NOT NULL,
                timestamp REAL NOT NULL,
                meta TEXT
            )
            ''')
            
            # Таблица для связей
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                relation TEXT NOT NULL,
                strength REAL NOT NULL,
                timestamp REAL NOT NULL,
                meta TEXT,
                FOREIGN KEY (source) REFERENCES nodes (id),
                FOREIGN KEY (target) REFERENCES nodes (id)
            )
            ''')
            
            # Индексы для ускорения поиска
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_nodes_domain ON nodes (domain)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_edges_source ON edges (source)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_edges_target ON edges (target)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges (relation)')
            
            conn.commit()
            logger.debug("Структура базы данных графа знаний инициализирована")
            return conn
        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных графа знаний: {e}")
            raise
    
    def _load_from_storage(self):
        """Загружает данные графа знаний из хранилища."""
        try:
            cursor = self.db.cursor()
            
            # Загружаем узлы
            cursor.execute("SELECT id, content, node_type, domain, strength, timestamp, meta FROM nodes")
            for row in cursor.fetchall():
                node_id, content, node_type, domain, strength, timestamp, meta = row
                node = KnowledgeNode(
                    id=node_id,
                    content=json.loads(content),
                    node_type=node_type,
                    domain=domain,
                    strength=strength,
                    timestamp=timestamp,
                    meta=json.loads(meta) if meta else {}
                )
                self.nodes[node_id] = node
                self.domains[domain].append(node_id)
            
            # Загружаем связи
            cursor.execute("SELECT id, source, target, relation, strength, timestamp, meta FROM edges")
            for row in cursor.fetchall():
                edge_id, source, target, relation, strength, timestamp, meta = row
                edge = KnowledgeEdge(
                    id=edge_id,
                    source=source,
                    target=target,
                    relation=relation,
                    strength=strength,
                    timestamp=timestamp,
                    meta=json.loads(meta) if meta else {}
                )
                self.edges[edge_id] = edge
                self.node_edges[source].append(edge_id)
                self.node_edges[target].append(edge_id)
            
            logger.info(f"Загружено {len(self.nodes)} узлов и {len(self.edges)} связей")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки данных графа знаний: {e}")
    
    def _update_stats(self):
        """Обновляет статистику графа знаний."""
        self.stats["total_nodes"] = len(self.nodes)
        self.stats["total_edges"] = len(self.edges)
        self.stats["domains"] = set(self.domains.keys())
        self.stats["last_update"] = time.time()

    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает общую статистику по графу знаний (для GUI)."""
        try:
            # Подсчет типов узлов
            node_types: Dict[str, int] = {}
            for node in self.nodes.values():
                node_types[node.node_type] = node_types.get(node.node_type, 0) + 1

            return {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "node_types": node_types,
                "domains": list(self.domains.keys()),
                "last_updated": self.stats.get("last_update", time.time()),
            }
        except Exception as e:
            logger.error(f"Ошибка получения статистики графа знаний: {e}")
            return {"error": str(e)}

    def get_domain_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по доменам (для GUI вкладки знаний)."""
        try:
            from collections import defaultdict

            domains: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
                "nodes": 0,
                "edges": 0,
                "node_types": {}
            })

            # Подсчет узлов по доменам и типам
            for node in self.nodes.values():
                d = domains[node.domain]
                d["nodes"] += 1
                nt = d["node_types"]
                nt[node.node_type] = nt.get(node.node_type, 0) + 1

            # Подсчет связей по домену источника
            for edge in self.edges.values():
                src = self.nodes.get(edge.source)
                domain = src.domain if src else "unknown"
                domains[domain]["edges"] += 1

            # Преобразуем defaultdict в обычный словарь
            return {k: dict(v) for k, v in domains.items()}
        except Exception as e:
            logger.error(f"Ошибка получения статистики доменов: {e}")
            return {}
    
    def add_node(self, node: KnowledgeNode) -> bool:
        """
        Добавляет узел в граф знаний.
        
        Args:
            node: Узел для добавления
            
        Returns:
            bool: Успешно ли добавлен узел
        """
        if node.id in self.nodes:
            logger.warning(f"Узел с ID {node.id} уже существует")
            return False
        
        try:
            # Добавляем в память
            self.nodes[node.id] = node
            self.domains[node.domain].append(node.id)
            
            if node.context:
                self.contexts[node.context].append(node.id)
            
            # Добавляем в базу данных
            cursor = self.db.cursor()
            cursor.execute('''
            INSERT INTO nodes (id, content, node_type, domain, strength, timestamp, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                node.id,
                json.dumps(node.content),
                node.node_type,
                node.domain,
                node.strength,
                node.timestamp,
                json.dumps(node.meta)
            ))
            self.db.commit()
            
            # Индексируем в гибридном индексе
            if self.hybrid_index is not None:
                try:
                    self.hybrid_index.put_node(node)
                except Exception as e:
                    logger.warning(f"Ошибка индексации узла {node.id} в гибридном индексе: {e}")
            
            # Обновляем статистику
            self._update_stats()
            
            logger.debug(f"Узел {node.id} добавлен в граф знаний")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка добавления узла {node.id}: {e}")
            return False
    
    def add_edge(self, edge: KnowledgeEdge) -> bool:
        """
        Добавляет связь между узлами.
        
        Args:
            edge: Связь для добавления
            
        Returns:
            bool: Успешно ли добавлена связь
        """
        if edge.id in self.edges:
            logger.warning(f"Связь с ID {edge.id} уже существует")
            return False
        
        if edge.source not in self.nodes or edge.target not in self.nodes:
            logger.warning(f"Источник или цель связи {edge.id} не существуют в графе")
            return False
        
        try:
            # Добавляем в память
            self.edges[edge.id] = edge
            self.node_edges[edge.source].append(edge.id)
            self.node_edges[edge.target].append(edge.id)
            
            # Добавляем в базу данных
            cursor = self.db.cursor()
            cursor.execute('''
            INSERT INTO edges (id, source, target, relation, strength, timestamp, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                edge.id,
                edge.source,
                edge.target,
                edge.relation,
                edge.strength,
                edge.timestamp,
                json.dumps(edge.meta)
            ))
            self.db.commit()
            
            # Индексируем в гибридном индексе
            if self.hybrid_index is not None:
                try:
                    self.hybrid_index.put_edge(edge)
                except Exception as e:
                    logger.warning(f"Ошибка индексации связи {edge.id} в гибридном индексе: {e}")
            
            # Обновляем статистику
            self._update_stats()
            
            logger.debug(f"Связь {edge.id} добавлена в граф знаний")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка добавления связи {edge.id}: {e}")
            return False
    
    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """
        Возвращает узел по ID.
        
        Args:
            node_id: ID узла
            
        Returns:
            Optional[KnowledgeNode]: Узел или None
        """
        node = self.nodes.get(node_id)
        if node is not None:
            return node
        # Пытаемся достать из гибридного индекса
        if self.hybrid_index is not None:
            try:
                data = self.hybrid_index.get_node(node_id)
                if data:
                    node = KnowledgeNode.from_dict(data)
                    # Кладём в оперативные структуры (без повторной записи в БД)
                    self.nodes[node.id] = node
                    self.domains[node.domain].append(node.id)
                    if node.context:
                        self.contexts[node.context].append(node.id)
                    return node
            except Exception as e:
                logger.warning(f"Ошибка получения узла {node_id} из гибридного индекса: {e}")
        return None
    
    def get_edge(self, edge_id: str) -> Optional[KnowledgeEdge]:
        """
        Возвращает связь по ID.
        
        Args:
            edge_id: ID связи
            
        Returns:
            Optional[KnowledgeEdge]: Связь или None
        """
        edge = self.edges.get(edge_id)
        if edge is not None:
            return edge
        if self.hybrid_index is not None:
            try:
                data = self.hybrid_index.get_edge(edge_id)
                if data:
                    edge = KnowledgeEdge.from_dict(data)
                    self.edges[edge.id] = edge
                    self.node_edges[edge.source].append(edge.id)
                    self.node_edges[edge.target].append(edge.id)
                    return edge
            except Exception as e:
                logger.warning(f"Ошибка получения связи {edge_id} из гибридного индекса: {e}")
        return None
    
    def get_nodes_by_domain(self, domain: str) -> List[KnowledgeNode]:
        """
        Возвращает узлы указанного домена.
        
        Args:
            domain: Домен
            
        Returns:
            List[KnowledgeNode]: Список узлов
        """
        node_ids = self.domains.get(domain, [])
        return [self.nodes[node_id] for node_id in node_ids if node_id in self.nodes]
    
    def get_nodes_by_context(self, context: str) -> List[KnowledgeNode]:
        """
        Возвращает узлы с указанным контекстом.
        
        Args:
            context: Контекст
            
        Returns:
            List[KnowledgeNode]: Список узлов
        """
        node_ids = self.contexts.get(context, [])
        return [self.nodes[node_id] for node_id in node_ids if node_id in self.nodes]
    
    def get_edges(self, node_id: str, direction: str = "both") -> List[KnowledgeEdge]:
        """
        Возвращает связи узла.
        
        Args:
            node_id: ID узла
            direction: Направление связей (source, target, both)
            
        Returns:
            List[KnowledgeEdge]: Список связей
        """
        if node_id not in self.node_edges:
            return []
        
        edge_ids = self.node_edges[node_id]
        edges = [self.edges[edge_id] for edge_id in edge_ids if edge_id in self.edges]
        
        if direction == "source":
            return [edge for edge in edges if edge.source == node_id]
        elif direction == "target":
            return [edge for edge in edges if edge.target == node_id]
        else:
            return edges
    
    def get_related_nodes(self, node_id: str, max_distance: int = 2) -> List[Dict]:
        """
        Возвращает связанные узлы с информацией о связи.
        
        Args:
            node_id: ID узла
            max_distance: Максимальное расстояние в графе
            
        Returns:
            List[Dict]: Список связанных узлов с информацией о связи
        """
        if node_id not in self.nodes:
            return []
        
        # Используем BFS для поиска связанных узлов
        visited = {node_id}
        queue = deque([(node_id, 0)])
        related = []
        
        while queue:
            current_id, distance = queue.popleft()
            
            if distance >= max_distance:
                continue
                
            for edge_id in self.node_edges[current_id]:
                edge = self.edges[edge_id]
                neighbor_id = edge.target if edge.source == current_id else edge.source
                
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, distance + 1))
                    
                    # Добавляем информацию о связи
                    related.append({
                        "id": neighbor_id,
                        "distance": distance + 1,
                        "relation": edge.relation,
                        "strength": edge.strength
                    })
        
        return related
    
    def get_all_nodes(self, limit: Optional[int] = None) -> List[KnowledgeNode]:
        """
        Возвращает все узлы графа.
        
        Args:
            limit: Ограничение на количество узлов
            
        Returns:
            List[KnowledgeNode]: Список узлов
        """
        nodes = list(self.nodes.values())
        return nodes[:limit] if limit else nodes
    
    def get_all_concepts(self) -> List[Dict[str, Any]]:
        """
        Возвращает все концепты в формате для MemoryGraphML.
        
        Returns:
            List[Dict]: Список концептов с id, type, description
        """
        concepts = []
        for node in self.nodes.values():
            concepts.append({
                'id': node.id,
                'type': node.node_type,
                'description': node.description or node.meta.get('description', ''),
                'domain': node.domain,
                'properties': node.meta
            })
        
        for edge in self.edges.values():
            concepts.append({
                'id': edge.id,
                'type': 'relation',
                'description': f"{edge.source_id} -> {edge.target_id}: {edge.relation_type}",
                'domain': 'general',
                'properties': edge.meta
            })
        
        return concepts
    
    def get_all_edges(self, limit: Optional[int] = None) -> List[KnowledgeEdge]:
        """
        Возвращает все связи графа.
        
        Args:
            limit: Ограничение на количество связей
            
        Returns:
            List[KnowledgeEdge]: Список связей
        """
        edges = list(self.edges.values())
        return edges[:limit] if limit else edges
    
    def remove_node(self, node_id: str) -> bool:
        """
        Удаляет узел и все связанные с ним связи.
        
        Args:
            node_id: ID узла
            
        Returns:
            bool: Успешно ли удален узел
        """
        if node_id not in self.nodes:
            logger.warning(f"Узел {node_id} не найден")
            return False
        
        try:
            # Удаляем узел
            node = self.nodes.pop(node_id)
            
            # Удаляем доменные ссылки
            if node_id in self.domains[node.domain]:
                self.domains[node.domain].remove(node_id)
            
            if node.context and node_id in self.contexts[node.context]:
                self.contexts[node.context].remove(node_id)
            
            # Удаляем связанные связи
            edges_to_remove = self.node_edges.pop(node_id, [])
            for edge_id in edges_to_remove:
                if edge_id in self.edges:
                    del self.edges[edge_id]
            
            # Удаляем из базы данных
            cursor = self.db.cursor()
            cursor.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            cursor.execute("DELETE FROM edges WHERE source = ? OR target = ?", (node_id, node_id))
            self.db.commit()
            
            # Удаляем из гибридного индекса
            if self.hybrid_index is not None:
                try:
                    self.hybrid_index.remove_node(node_id)
                except Exception as e:
                    logger.warning(f"Ошибка удаления узла {node_id} из гибридного индекса: {e}")
            
            # Обновляем статистику
            self._update_stats()
            
            logger.info(f"Узел {node_id} и связанные с ним связи удалены")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка удаления узла {node_id}: {e}")
            return False
    
    def remove_edge(self, edge_id: str) -> bool:
        """
        Удаляет связь между узлами.
        
        Args:
            edge_id: ID связи
            
        Returns:
            bool: Успешно ли удалена связь
        """
        if edge_id not in self.edges:
            logger.warning(f"Связь {edge_id} не найдена")
            return False
        
        try:
            # Удаляем связь
            edge = self.edges.pop(edge_id)
            
            # Удаляем из списка связей узлов
            if edge_id in self.node_edges[edge.source]:
                self.node_edges[edge.source].remove(edge_id)
            if edge_id in self.node_edges[edge.target]:
                self.node_edges[edge.target].remove(edge_id)
            
            # Удаляем из базы данных
            cursor = self.db.cursor()
            cursor.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
            self.db.commit()
            
            # Удаляем из гибридного индекса
            if self.hybrid_index is not None:
                try:
                    self.hybrid_index.remove_edge(edge_id)
                except Exception as e:
                    logger.warning(f"Ошибка удаления связи {edge_id} из гибридного индекса: {e}")
            
            # Обновляем статистику
            self._update_stats()
            
            logger.info(f"Связь {edge_id} удалена")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка удаления связи {edge_id}: {e}")
            return False
    
    def update_node(self, node_id: str, **kwargs) -> bool:
        """
        Обновляет свойства узла.
        
        Args:
            node_id: ID узла
            **kwargs: Свойства для обновления
            
        Returns:
            bool: Успешно ли обновлен узел
        """
        if node_id not in self.nodes:
            logger.warning(f"Узел {node_id} не найден")
            return False
        
        try:
            node = self.nodes[node_id]
            
            # Обновляем свойства
            if "content" in kwargs:
                node.content = kwargs["content"]
            if "node_type" in kwargs:
                node.node_type = kwargs["node_type"]
            if "domain" in kwargs:
                # Обновляем доменные ссылки
                if node.domain in self.domains and node_id in self.domains[node.domain]:
                    self.domains[node.domain].remove(node_id)
                node.domain = kwargs["domain"]
                self.domains[node.domain].append(node_id)
            if "strength" in kwargs:
                node.strength = max(0.0, min(1.0, kwargs["strength"]))
            if "timestamp" in kwargs:
                node.timestamp = kwargs["timestamp"]
            if "meta" in kwargs:
                node.meta = kwargs["meta"]
                if node.meta is None:
                    node.meta = {}
                node.context = node.meta.get('context', '')
                
                # Обновляем контекстные ссылки
                if "context" in node.meta:
                    if node_id in self.contexts.get(node.context, []):
                        self.contexts[node.context].remove(node_id)
                    self.contexts[node.meta["context"]].append(node_id)
            
            # Обновляем в базе данных
            cursor = self.db.cursor()
            cursor.execute('''
            UPDATE nodes
            SET content = ?, node_type = ?, domain = ?, strength = ?, timestamp = ?, meta = ?
            WHERE id = ?
            ''', (
                json.dumps(node.content),
                node.node_type,
                node.domain,
                node.strength,
                node.timestamp,
                json.dumps(node.meta),
                node_id
            ))
            self.db.commit()
            
            logger.info(f"Узел {node_id} обновлен")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обновления узла {node_id}: {e}")
            return False
    
    def get_influence_depth(self, node_id: str, max_depth: int = 5) -> int:
        """
        Определяет глубину влияния узла в графе.
        
        Args:
            node_id: ID узла
            max_depth: Максимальная глубина для анализа
            
        Returns:
            int: Глубина влияния
        """
        if node_id not in self.nodes:
            return 0
        
        # Используем BFS для определения максимальной глубины
        visited = {node_id}
        queue = deque([(node_id, 0)])
        max_reached_depth = 0
        
        while queue:
            current_id, depth = queue.popleft()
            max_reached_depth = max(max_reached_depth, depth)
            
            if depth >= max_depth:
                continue
                
            for edge_id in self.node_edges[current_id]:
                edge = self.edges[edge_id]
                neighbor_id = edge.target if edge.source == current_id else edge.source
                
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))
        
        return max_reached_depth
    
    def get_concept_usage_frequency(self, concept: str) -> int:
        """
        Определяет частоту использования концепта в графе.
        
        Args:
            concept: Концепт для анализа
            
        Returns:
            int: Частота использования
        """
        count = 0
        
        # Ищем в содержимом узлов
        for node in self.nodes.values():
            content_str = str(node.content).lower()
            if concept.lower() in content_str:
                count += 1
        
        return count
    
    def find_concept_nodes(self, concept: str, max_results: int = 10) -> List[KnowledgeNode]:
        """
        Находит узлы, связанные с концептом.
        
        Args:
            concept: Концепт для поиска
            max_results: Максимальное количество результатов
            
        Returns:
            List[KnowledgeNode]: Список узлов
        """
        results = []
        concept_lower = concept.lower()
        
        for node in self.nodes.values():
            content_str = str(node.content).lower()
            if concept_lower in content_str:
                results.append(node)
                if len(results) >= max_results:
                    break
        
        return results
    
    def get_node_strength_history(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Возвращает историю изменения силы узла.
        
        Args:
            node_id: ID узла
            
        Returns:
            List[Dict[str, Any]]: История изменений
        """
        # В реальной системе здесь будет запрос к истории изменений
        # Для упрощения возвращаем заглушку
        if node_id not in self.nodes:
            return []
        
        node = self.nodes[node_id]
        return [{
            "timestamp": node.timestamp,
            "strength": node.strength,
            "reason": "initial_creation"
        }]
    
    def get_concept_relations(self, concept: str) -> Dict[str, List[Dict]]:
        """
        Возвращает все отношения для концепта.
        
        Args:
            concept: Концепт для анализа
            
        Returns:
            Dict[str, List[Dict]]: Отношения по типам
        """
        relations = defaultdict(list)
        concept_nodes = self.find_concept_nodes(concept)
        
        for node in concept_nodes:
            for edge_id in self.node_edges[node.id]:
                edge = self.edges[edge_id]
                neighbor_id = edge.target if edge.source == node.id else edge.source
                neighbor = self.nodes.get(neighbor_id)
                
                if neighbor:
                    relations[edge.relation].append({
                        "node_id": neighbor.id,
                        "content": neighbor.get_content_summary(),
                        "strength": edge.strength,
                        "direction": "out" if edge.source == node.id else "in"
                    })
        
        return dict(relations)
    
    def get_concept_contexts(self, concept: str) -> List[str]:
        """
        Возвращает контексты, в которых используется концепт.
        
        Args:
            concept: Концепт для анализа
            
        Returns:
            List[str]: Список контекстов
        """
        contexts = set()
        concept_nodes = self.find_concept_nodes(concept)
        
        for node in concept_nodes:
            if node.context:
                contexts.add(node.context)
        
        return list(contexts)
    
    def get_concept_domains(self, concept: str) -> List[str]:
        """
        Возвращает домены, в которых используется концепт.
        
        Args:
            concept: Концепт для анализа
            
        Returns:
            List[str]: Список доменов
        """
        domains = set()
        concept_nodes = self.find_concept_nodes(concept)
        
        for node in concept_nodes:
            domains.add(node.domain)
        
        return list(domains)
    
    def get_concept_influence_score(self, concept: str) -> float:
        """
        Вычисляет показатель влияния концепта в графе.
        
        Args:
            concept: Концепт для анализа
            
        Returns:
            float: Показатель влияния (0.0-1.0)
        """
        concept_nodes = self.find_concept_nodes(concept)
        if not concept_nodes:
            return 0.0
        
        # Базовая оценка на основе количества узлов
        node_count_score = min(1.0, len(concept_nodes) / 50)
        
        # Оценка на основе глубины влияния
        max_depth = 0
        for node in concept_nodes:
            depth = self.get_influence_depth(node.id)
            max_depth = max(max_depth, depth)
        
        depth_score = min(1.0, max_depth / 10)
        
        # Оценка на основе силы связей
        total_strength = 0
        total_relations = 0
        
        for node in concept_nodes:
            for edge_id in self.node_edges[node.id]:
                edge = self.edges[edge_id]
                total_strength += edge.strength
                total_relations += 1
        
        strength_score = min(1.0, total_strength / max(1, total_relations)) if total_relations > 0 else 0.0
        
        # Итоговая оценка
        influence_score = (node_count_score * 0.4) + (depth_score * 0.3) + (strength_score * 0.3)
        return influence_score
    
    def get_concept_reputation(self, concept: str) -> float:
        """
        Возвращает репутацию концепта на основе источников.
        
        Args:
            concept: Концепт для анализа
            
        Returns:
            float: Репутация концепта (0.0-1.0)
        """
        concept_nodes = self.find_concept_nodes(concept)
        if not concept_nodes:
            return 0.5  # Нейтральная репутация по умолчанию
        
        total_reputation = 0
        count = 0
        
        for node in concept_nodes:
            # Получаем репутацию источника из метаданных
            source_reputation = node.meta.get('source_reputation', 0.5)
            total_reputation += source_reputation
            count += 1
            
            # Если есть связи, учитываем репутацию связанных узлов
            for edge_id in self.node_edges[node.id]:
                edge = self.edges[edge_id]
                neighbor_id = edge.target if edge.source == node.id else edge.source
                neighbor = self.nodes.get(neighbor_id)
                
                if neighbor:
                    neighbor_reputation = neighbor.meta.get('source_reputation', 0.5)
                    total_reputation += neighbor_reputation
                    count += 1
        
        return total_reputation / max(1, count)
    
    def get_concept_time_relevance(self, concept: str) -> float:
        """
        Оценивает актуальность концепта во времени.
        
        Args:
            concept: Концепт для анализа
            
        Returns:
            float: Актуальность (0.0-1.0, где 1.0 - самый актуальный)
        """
        concept_nodes = self.find_concept_nodes(concept)
        if not concept_nodes:
            return 0.0
        
        # Находим самый новый узел
        newest_timestamp = max(node.timestamp for node in concept_nodes)
        
        # Вычисляем, как давно был обновлен концепт
        time_diff = time.time() - newest_timestamp
        days_diff = time_diff / 86400  # Секунды в дне
        
        # Актуальность уменьшается со временем
        # Полная актуальность в течение первых 7 дней, затем экспоненциальное затухание
        if days_diff <= 7:
            return 1.0
        else:
            # Экспоненциальное затухание после 7 дней
            decay_factor = 0.95 ** ((days_diff - 7) / 7)  # Затухание каждые 7 дней
            return max(0.1, decay_factor)
    
    def get_concept_analysis(self, concept: str) -> Dict[str, Any]:
        """
        Проводит комплексный анализ концепта.
        
        Args:
            concept: Концепт для анализа
            
        Returns:
            Dict[str, Any]: Результаты анализа
        """
        return {
            "concept": concept,
            "influence_score": self.get_concept_influence_score(concept),
            "reputation": self.get_concept_reputation(concept),
            "time_relevance": self.get_concept_time_relevance(concept),
            "domains": self.get_concept_domains(concept),
            "contexts": self.get_concept_contexts(concept),
            "relation_types": list(self.get_concept_relations(concept).keys()),
            "node_count": len(self.find_concept_nodes(concept)),
            "depth": max((self.get_influence_depth(node.id) for node in self.find_concept_nodes(concept)), default=0),
            "timestamp": time.time()
        }
    
    def close(self):
        """Закрывает соединение с базой данных."""
        if hasattr(self, 'db') and self.db:
            self.db.close()
            logger.info("Соединение с базой данных графа знаний закрыто")
    
    def __del__(self):
        """Деструктор для закрытия соединения с базой данных."""
        self.close()