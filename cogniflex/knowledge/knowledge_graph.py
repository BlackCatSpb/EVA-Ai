
def safe_json_loads(value):
    """Безопасная загрузка JSON с обработкой ошибок."""
    if not value:
        return {}
    try:
        if isinstance(value, str):
            return json.loads(value)
        elif isinstance(value, (bytes, bytearray)):
            return json.loads(value.decode('utf-8'))
        else:
            # Если значение не является строкой, возвращаем пустой dict
            return {}
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return {}

"""
Модуль графа знаний для CogniFlex - управление структурой знаний
Обновленная версия с поддержкой гибридного кэша, асинхронной токенизации и версионирования
"""
import os
import logging
import time
import re
import sqlite3
import json
import hashlib
import threading
import copy
import math
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger("cogniflex.knowledge_graph")

# Импорты для интеграции с другими модулями
try:
    from cogniflex.memory.hybrid_token_cache import HybridTokenCache
    logger.debug("HybridTokenCache импортирован успешно")
except ImportError:
    logger.warning("HybridTokenCache недоступен, кэширование будет ограничено")
    HybridTokenCache = None

try:
    from cogniflex.mlearning.unified_text_processor import UnifiedTextProcessor
    logger.debug("UnifiedTextProcessor импортирован успешно")
except ImportError:
    logger.warning("UnifiedTextProcessor недоступен, токенизация будет ограничена")
    UnifiedTextProcessor = None

try:
    from cogniflex.mlearning.ml_unit import MLUnit
    logger.debug("MLUnit импортирован успешно")
except ImportError:
    logger.warning("MLUnit недоступен, автоматическое обновление знаний ограничено")
    MLUnit = None

# Перечисления для типов узлов и связей
class NodeType(Enum):
    CONCEPT = "concept"
    ENTITY = "entity"
    FACT = "fact"
    EVENT = "event"
    RELATION = "relation"
    ATTRIBUTE = "attribute"
    PROCESS = "process"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    OTHER = "other"

class RelationType(Enum):
    IS_A = "is_a"
    PART_OF = "part_of"
    HAS_PROPERTY = "has_property"
    CAUSES = "causes"
    SUPPORTS = "supports"
    USED_FOR = "used_for"
    LOCATED_AT = "located_at"
    OCCURS_DURING = "occurs_during"
    PRECEDES = "precedes"
    RELATED_TO = "related_to"
    CONTRADICTS = "contradicts"
    SIMILAR_TO = "similar_to"
    DEPENDS_ON = "depends_on"
    OTHER = "other"

class KnowledgeNode:
    """Представляет узел в графе знаний с поддержкой версионирования и метаданных."""
    
    def __init__(self, id: str, name: str, description: str, 
                 node_type: str = "fact", domain: str = "general",
                 strength: float = 0.5, timestamp: Optional[float] = None,
                 meta: Optional[Dict] = None, version: int = 1,
                 spatial_info: Optional[Dict[str, Any]] = None,
                 temporal_info: Optional[Dict[str, Any]] = None):
        """
        Инициализирует узел знаний.
        
        Args:
            id: Уникальный идентификатор узла
            name: Название узла
            description: Описание узла
            node_type: Тип узла
            domain: Домен знаний
            strength: Сила знания (0.0-1.0)
            timestamp: Временная метка создания
            meta: Дополнительные метаданные
            version: Версия узла
            spatial_info: Пространственная информация
            temporal_info: Временная информация
        """
        self.id = id
        self.name = name
        self.description = description
        self.content = description  # Для совместимости с другими модулями
        self.node_type = node_type
        self.domain = domain
        self.strength = strength
        self.timestamp = timestamp or time.time()
        self.last_updated = self.timestamp
        self.meta = meta or {}
        self.version = version
        self.spatial_info = spatial_info or {}
        self.temporal_info = temporal_info or {}
        self.embedding = None  # Векторное представление узла
        
        # Для отслеживания истории изменений
        self.history = []
        if 'sources' not in self.meta:
            self.meta['sources'] = []
        
        # Для отслеживания противоречий
        self.contradictions = []
        
        # Для поддержки семантического поиска
        self.keyword_index = []
        self.concept_index = []
    
    def update(self, new_description: str, strength: Optional[float] = None,
              source: Optional[str] = None, user_id: Optional[str] = None,
              spatial_info: Optional[Dict[str, Any]] = None,
              temporal_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Обновляет узел знаний, создавая новую версию.
        
        Args:
            new_description: Новое описание
            strength: Новая сила связи
            source: Источник информации
            user_id: ID пользователя
            spatial_info: Новая пространственная информация
            temporal_info: Новая временная информация
        """
        # Сохраняем старое состояние для истории
        old_state = {
            "description": self.description,
            "strength": self.strength,
            "spatial_info": copy.deepcopy(self.spatial_info),
            "temporal_info": copy.deepcopy(self.temporal_info),
            "timestamp": self.last_updated,
            "version": self.version
        }
        
        # Обновляем узел
        self.description = new_description
        self.content = new_description  # Для совместимости
        self.last_updated = time.time()
        self.version += 1
        
        if strength is not None:
            self.strength = strength
        
        if spatial_info is not None:
            self.spatial_info = spatial_info
        
        if temporal_info is not None:
            self.temporal_info = temporal_info
        
        # Добавляем источник
        if source:
            self.meta['sources'].append({
                'source': source,
                'timestamp': time.time(),
                'user_id': user_id,
                'version': self.version
            })
        
        # Записываем в историю
        self.history.append({
            "timestamp": time.time(),
            "version": self.version,
            "changes": {
                "old": old_state,
                "new": {
                    "description": new_description,
                    "strength": self.strength,
                    "spatial_info": self.spatial_info,
                    "temporal_info": self.temporal_info
                },
                "source": source,
                "user_id": user_id
            }
        })
    
    def add_contradiction(self, contradictory_node_id: str, evidence: str, 
                         resolution: Optional[str] = None) -> None:
        """
        Добавляет информацию о противоречии.
        
        Args:
            contradictory_node_id: ID противоречивого узла
            evidence: Доказательства противоречия
            resolution: Предложенное разрешение
        """
        self.contradictions.append({
            "node_id": contradictory_node_id,
            "evidence": evidence,
            "timestamp": time.time(),
            "resolved": resolution is not None,
            "resolution": resolution
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует узел в словарь для сериализации."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "node_type": self.node_type,
            "domain": self.domain,
            "strength": self.strength,
            "timestamp": self.timestamp,
            "last_updated": self.last_updated,
            "version": self.version,
            "meta": self.meta,
            "spatial_info": self.spatial_info,
            "temporal_info": self.temporal_info,
            "history": self.history,
            "contradictions": self.contradictions,
            "keyword_index": self.keyword_index,
            "concept_index": self.concept_index
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeNode':
        """Создает узел из словаря."""
        node = cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            node_type=data["node_type"],
            domain=data["domain"],
            strength=data["strength"],
            timestamp=data["timestamp"],
            meta=data["meta"],
            version=data["version"],
            spatial_info=data["spatial_info"],
            temporal_info=data["temporal_info"]
        )
        
        node.last_updated = data["last_updated"]
        node.history = data["history"]
        node.contradictions = data["contradictions"]
        node.keyword_index = data["keyword_index"]
        node.concept_index = data["concept_index"]
        
        return node
    
    def __repr__(self) -> str:
        return f"KnowledgeNode(id={self.id}, name='{self.name}', type={self.node_type}, domain={self.domain})"

class KnowledgeEdge:
    """Представляет связь между узлами в графе знаний."""
    
    def __init__(self, id: str, source_id: str, target_id: str, 
                 relation_type: str, strength: float = 0.5,
                 timestamp: Optional[float] = None, meta: Optional[Dict] = None,
                 version: int = 1, spatial_info: Optional[Dict[str, Any]] = None,
                 temporal_info: Optional[Dict[str, Any]] = None):
        """
        Инициализирует связь между узлами.
        
        Args:
            id: Уникальный идентификатор связи
            source_id: ID исходного узла
            target_id: ID целевого узла
            relation_type: Тип связи
            strength: Сила связи (0.0-1.0)
            timestamp: Временная метка создания
            meta: Дополнительные метаданные
            version: Версия связи
            spatial_info: Пространственная информация
            temporal_info: Временная информация
        """
        self.id = id
        self.source_id = source_id
        self.target_id = target_id
        self.relation_type = relation_type
        self.strength = strength
        self.timestamp = timestamp or time.time()
        self.last_updated = self.timestamp
        self.meta = meta or {}
        self.version = version
        self.spatial_info = spatial_info or {}
        self.temporal_info = temporal_info or {}
        self.embedding = None  # Векторное представление связи
        
        # Для отслеживания истории изменений
        self.history = []
        if 'sources' not in self.meta:
            self.meta['sources'] = []
    
    def update(self, new_strength: Optional[float] = None,
              source: Optional[str] = None, user_id: Optional[str] = None,
              spatial_info: Optional[Dict[str, Any]] = None,
              temporal_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Обновляет связь, создавая новую версию.
        
        Args:
            new_strength: Новая сила связи
            source: Источник информации
            user_id: ID пользователя
            spatial_info: Новая пространственная информация
            temporal_info: Новая временная информация
        """
        # Сохраняем старое состояние для истории
        old_state = {
            "strength": self.strength,
            "spatial_info": copy.deepcopy(self.spatial_info),
            "temporal_info": copy.deepcopy(self.temporal_info),
            "timestamp": self.last_updated,
            "version": self.version
        }
        
        # Обновляем связь
        self.last_updated = time.time()
        self.version += 1
        
        if new_strength is not None:
            self.strength = new_strength
        
        if spatial_info is not None:
            self.spatial_info = spatial_info
        
        if temporal_info is not None:
            self.temporal_info = temporal_info
        
        # Добавляем источник
        if source:
            self.meta['sources'].append({
                'source': source,
                'timestamp': time.time(),
                'user_id': user_id,
                'version': self.version
            })
        
        # Записываем в историю
        self.history.append({
            "timestamp": time.time(),
            "version": self.version,
            "changes": {
                "old": old_state,
                "new": {
                    "strength": self.strength,
                    "spatial_info": self.spatial_info,
                    "temporal_info": self.temporal_info
                },
                "source": source,
                "user_id": user_id
            }
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует связь в словарь для сериализации."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "strength": self.strength,
            "timestamp": self.timestamp,
            "last_updated": self.last_updated,
            "version": self.version,
            "meta": self.meta,
            "spatial_info": self.spatial_info,
            "temporal_info": self.temporal_info,
            "history": self.history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEdge':
        """Создает связь из словаря."""
        edge = cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=data["relation_type"],
            strength=data["strength"],
            timestamp=data["timestamp"],
            meta=data["meta"],
            version=data["version"],
            spatial_info=data["spatial_info"],
            temporal_info=data["temporal_info"]
        )
        
        edge.last_updated = data["last_updated"]
        edge.history = data["history"]
        
        return edge
    
    def __repr__(self) -> str:
        return f"KnowledgeEdge(id={self.id}, {self.source_id} -> {self.target_id}, type={self.relation_type})"

class KnowledgeGraph:
    """Граф знаний для CogniFlex - хранит и управляет знаниями системы."""
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None,
                hybrid_cache: Optional[Any] = None, text_processor: Optional[Any] = None,
                max_workers: int = 4):
        """
        Инициализирует граф знаний.
        
        Args:
            brain: Ссылка на ядро CogniFlex
            cache_dir: Путь к директории кэша
            hybrid_cache: Экземпляр гибридного кэша
            text_processor: Экземпляр текстового процессора
            max_workers: Максимальное количество рабочих потоков
        """
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "cogniflex_knowledge_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Настройки пула потоков должны быть доступны раннее для зависимостей
        self.max_workers = max_workers

        # Подготовка структур и статистики до загрузки БД
        self.nodes = {}
        self.edges = {}
        self.stats = {
            "total_nodes": 0,
            "total_edges": 0,
            "node_creations": 0,
            "node_updates": 0,
            "edge_creations": 0,
            "edge_updates": 0,
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_processing_time": 0.0
        }

        # Инициализация компонентов для интеграции (могут зависеть от max_workers)
        self._init_integration_components(hybrid_cache, text_processor)

        # Путь к базе данных
        self.db_path = os.path.join(self.cache_dir, "knowledge_graph.db")

        # Инициализируем базу данных
        self._init_db()

        # Загружаем узлы и связи
        self._load_nodes()
        self._load_edges()

        # Инициализируем индексы
        self._init_indexes()

        # Пул потоков для асинхронных операций
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.futures: Dict[str, Future] = {}

        # Состояние
        self.initialized = True
        self.running = False
        self.stop_event = threading.Event()

        # Запускаем фоновые службы
        self._start_background_services()

        logger.info(f"KnowledgeGraph инициализирован с {len(self.nodes)} узлами и {len(self.edges)} связями")
    
    def _init_integration_components(self, hybrid_cache: Optional[Any], text_processor: Optional[Any]):
        """Инициализирует компоненты для интеграции с другими модулями."""
        # Гибридный кэш
        self.hybrid_cache = hybrid_cache
        if self.hybrid_cache is None and HybridTokenCache:
            try:
                # Используем поддерживаемые параметры HybridTokenCache
                self.hybrid_cache = HybridTokenCache(
                    brain=self.brain,
                    max_memory_tokens=10000,
                    disk_cache_dir="hybrid_cache"
                )
                logger.debug("Создан внутренний гибридный кэш")
            except Exception as e:
                logger.warning(f"Не удалось создать гибридный кэш: {e}")
        
        # Текстовый процессор
        self.text_processor = text_processor
        if self.text_processor is None and UnifiedTextProcessor:
            try:
                config = {
                    'model_name': "paraphrase-multilingual-MiniLM-L12-v2",
                    'use_async': True
                }
                self.text_processor = UnifiedTextProcessor(
                    brain=self.brain,
                    config=config
                )
                logger.debug("Создан внутренний текстовый процессор")
            except Exception as e:
                logger.warning(f"Не удалось создать текстовый процессор: {e}")
        
        # MLUnit
        self.ml_unit = None
        if self.brain and hasattr(self.brain, 'ml_unit'):
            self.ml_unit = self.brain.ml_unit
        elif MLUnit:
            try:
                self.ml_unit = MLUnit(brain=self.brain)
                logger.debug("Создан внутренний MLUnit")
            except Exception as e:
                logger.warning(f"Не удалось создать MLUnit: {e}")
    
    def _init_db(self):
        """Инициализирует базу данных для графа знаний."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Таблица узлов
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                node_type TEXT NOT NULL,
                domain TEXT NOT NULL,
                strength REAL NOT NULL,
                timestamp REAL NOT NULL,
                last_updated REAL NOT NULL,
                version INTEGER NOT NULL,
                meta TEXT DEFAULT '{}',
                spatial_info TEXT DEFAULT '{}',
                temporal_info TEXT DEFAULT '{}',
                history TEXT DEFAULT '[]',
                contradictions TEXT DEFAULT '[]',
                keyword_index TEXT DEFAULT '[]',
                concept_index TEXT DEFAULT '[]'
            )
            """)
            
            # Таблица связей
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                strength REAL NOT NULL,
                timestamp REAL NOT NULL,
                last_updated REAL NOT NULL,
                version INTEGER NOT NULL,
                meta TEXT DEFAULT '{}',
                spatial_info TEXT DEFAULT '{}',
                temporal_info TEXT DEFAULT '{}',
                history TEXT DEFAULT '[]',
                FOREIGN KEY(source_id) REFERENCES nodes(id),
                FOREIGN KEY(target_id) REFERENCES nodes(id)
            )
            """)
            
            # Таблица истории
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id TEXT PRIMARY KEY,
                node_id TEXT,
                edge_id TEXT,
                change_type TEXT NOT NULL,
                changes TEXT NOT NULL,
                timestamp REAL NOT NULL,
                user_id TEXT,
                source TEXT,
                version INTEGER NOT NULL
            )
            """)
            
            # Индексы для ускорения поиска
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_domain ON nodes(domain)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_timestamp ON history(timestamp)")
            
            conn.commit()
            conn.close()
            
            logger.debug("База данных графа знаний инициализирована")
        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных графа знаний: {e}", exc_info=True)
    
    def _load_nodes(self):
        """Загружает узлы из базы данных."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM nodes")
            
            self.nodes = {}
            for row in cursor.fetchall():
                node = KnowledgeNode(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    node_type=row[3],
                    domain=row[4],
                    strength=row[5],
                    timestamp=row[6],
                    meta=safe_json_loads(row[9]) if len(row) > 9 and row[9] else {},
                    version=row[8],
                    spatial_info=safe_json_loads(row[10]) if len(row) > 10 and row[10] else {},
                    temporal_info=safe_json_loads(row[11]) if len(row) > 11 and row[11] else {}
                )
                node.last_updated = row[7]
                node.history = safe_json_loads(row[12]) if len(row) > 12 and row[12] else []
                node.contradictions = safe_json_loads(row[13]) if len(row) > 13 and row[13] else []
                node.keyword_index = safe_json_loads(row[14]) if len(row) > 14 and row[14] else []
                node.concept_index = safe_json_loads(row[15]) if len(row) > 15 and row[15] else []
                
                self.nodes[node.id] = node
            
            # Обновляем статистику
            self.stats["total_nodes"] = len(self.nodes)
            
            logger.info(f"Загружено {len(self.nodes)} узлов в граф знаний")
            conn.close()
        except Exception as e:
            logger.error(f"Ошибка загрузки узлов графа знаний: {e}", exc_info=True)
            self.nodes = {}
    
    def _load_edges(self):
        """Загружает связи из базы данных."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM edges")
            
            self.edges = {}
            for row in cursor.fetchall():
                edge = KnowledgeEdge(
                    id=row[0],
                    source_id=row[1],
                    target_id=row[2],
                    relation_type=row[3],
                    strength=row[4],
                    timestamp=row[5],
                    meta=safe_json_loads(row[9]) if len(row) > 9 and row[9] else {},
                    version=row[8],
                    spatial_info=safe_json_loads(row[10]) if len(row) > 10 and row[10] else {},
                    temporal_info=safe_json_loads(row[11]) if len(row) > 11 and row[11] else {}
                )
                edge.last_updated = row[6]
                edge.history = safe_json_loads(row[12]) if len(row) > 12 and row[12] else []
                
                self.edges[edge.id] = edge
            
            # Обновляем статистику
            self.stats["total_edges"] = len(self.edges)
            
            logger.info(f"Загружено {len(self.edges)} связей в граф знаний")
            conn.close()
        except Exception as e:
            logger.error(f"Ошибка загрузки связей графа знаний: {e}", exc_info=True)
            self.edges = {}
    
    def _init_indexes(self):
        """Инициализирует индексы для быстрого поиска."""
        def safe_sort_key(item):
            """Безопасное извлечение временной метки для сортировки"""
            timestamp = item[0]
            if isinstance(timestamp, str):
                try:
                    # Пытаемся преобразовать строку в число
                    return float(timestamp)
                except (ValueError, TypeError):
                    # Если не получается - используем 0 как значение по умолчанию
                    return 0.0
            elif isinstance(timestamp, (int, float)):
                return float(timestamp)
            # Для других типов возвращаем 0
            return 0.0

        # Индекс по доменам
        self.domain_index = defaultdict(list)
        for node in self.nodes.values():
            self.domain_index[node.domain].append(node.id)
        
        # Индекс по типам узлов
        self.node_type_index = defaultdict(list)
        for node in self.nodes.values():
            self.node_type_index[node.node_type].append(node.id)
        
        # Индекс по связям
        self.relation_index = defaultdict(list)
        for edge in self.edges.values():
            self.relation_index[edge.relation_type].append(edge.id)
        
        # Индекс по временным меткам
        self.temporal_index = []
        for node in self.nodes.values():
            if node.temporal_info:
                self.temporal_index.append((node.timestamp, node.id, "node"))
            if node.last_updated != node.timestamp:
                self.temporal_index.append((node.last_updated, node.id, "node"))
        
        if edge:
            # Обновляем индекс связей
            self.relation_index[edge.relation_type] = [
                eid for eid in self.relation_index[edge.relation_type] if eid != edge.id
            ]
            self.relation_index[edge.relation_type].append(edge.id)
            
            # Обновляем временной индекс
            self.temporal_index = [
                item for item in self.temporal_index 
                if not (item[1] == edge.id and item[2] == "edge")
            ]
            self.temporal_index.append((edge.timestamp, edge.id, "edge"))
            if edge.last_updated != edge.timestamp:
                self.temporal_index.append((edge.last_updated, edge.id, "edge"))
            
            # Сортируем временной индекс с безопасной функцией сортировки
            self.temporal_index.sort(key=safe_sort_key)

    @property
    def graph(self) -> Dict[str, Any]:
        """Compatibility property - returns nodes as a dict-like structure."""
        return self.nodes

    def _update_indexes(self, node: Optional["KnowledgeNode"] = None,
                       edge: Optional["KnowledgeEdge"] = None):
        """
        Updates indexes after node/edge changes.
        
        Args:
            node: Node that was added/updated (optional)
            edge: Edge that was added/updated (optional)
        """
        if node is not None:
            self.domain_index[node.domain].append(node.id)
            self.node_type_index[node.node_type].append(node.id)
            if node.temporal_info:
                self.temporal_index.append((node.timestamp, node.id, "node"))
            if node.last_updated != node.timestamp:
                self.temporal_index.append((node.last_updated, node.id, "node"))
        
        if edge is not None:
            self.relation_index[edge.relation_type].append(edge.id)
            if edge.temporal_info:
                self.temporal_index.append((edge.timestamp, edge.id, "edge"))
            if edge.last_updated != edge.timestamp:
                self.temporal_index.append((edge.last_updated, edge.id, "edge"))
        
        if self.temporal_index:
            def safe_sort_key(item):
                ts = item[0]
                if isinstance(ts, (int, float)):
                    return float(ts)
                elif isinstance(ts, str):
                    try:
                        return float(ts)
                    except (ValueError, TypeError):
                        return 0.0
                return 0.0
            self.temporal_index.sort(key=safe_sort_key)
    
    def _start_background_services(self):
        """Запускает фоновые службы графа знаний."""
        # Останавливаем существующие потоки, если они есть
        self.stop()
        
        # Запускаем фоновый мониторинг
        self.monitoring_thread = threading.Thread(
            target=self._background_monitoring,
            name="KnowledgeGraphMonitoring",
            daemon=True
        )
        self.monitoring_thread.start()
        
        # Запускаем фоновую оптимизацию
        self.optimization_thread = threading.Thread(
            target=self._background_optimization,
            name="KnowledgeGraphOptimization",
            daemon=True
        )
        self.optimization_thread.start()
        
        self.running = True
        logger.info("Фоновые службы KnowledgeGraph запущены")
    
    def _background_monitoring(self):
        """Фоновый мониторинг состояния графа знаний."""
        while not self.stop_event.is_set():
            try:
                # Проверяем здоровье каждые 5 минут
                self.stop_event.wait(300)
                
                # Проверяем целостность графа
                self._check_graph_integrity()
                
                # Проверяем противоречия
                self._check_for_contradictions()
                
                # Проверяем устаревшие знания
                self._check_for_outdated_knowledge()
                
            except Exception as e:
                logger.error(f"Ошибка в фоновом мониторинге KnowledgeGraph: {e}", exc_info=True)
    
    def _background_optimization(self):
        """Фоновая оптимизация использования графа знаний."""
        while not self.stop_event.is_set():
            try:
                # Оптимизируем каждые 10 минут
                self.stop_event.wait(600)
                
                # Оптимизируем индексы
                self._optimize_indexes()
                
                # Очищаем устаревший кэш
                self._cleanup_cache()
                
            except Exception as e:
                logger.error(f"Ошибка в фоновой оптимизации KnowledgeGraph: {e}", exc_info=True)
    
    def _check_graph_integrity(self):
        """Проверяет целостность графа знаний."""
        try:
            # Проверяем отсутствующие узлы в связях
            missing_nodes = []
            for edge_id, edge in self.edges.items():
                if edge.source_id not in self.nodes:
                    missing_nodes.append(edge.source_id)
                if edge.target_id not in self.nodes:
                    missing_nodes.append(edge.target_id)
            
            if missing_nodes:
                logger.warning(f"Обнаружены {len(missing_nodes)} отсутствующих узлов в связях")
                
                # Попытка восстановления
                self._attempt_recovery(missing_nodes)
            
            # Проверяем дубликаты
            name_counts = defaultdict(int)
            for node in self.nodes.values():
                name_counts[node.name.lower()] += 1
            
            duplicates = [name for name, count in name_counts.items() if count > 1]
            if duplicates:
                logger.warning(f"Обнаружено {len(duplicates)} дублирующихся имен узлов")
                self._resolve_duplicates(duplicates)
            
        except Exception as e:
            logger.error(f"Ошибка проверки целостности графа: {e}", exc_info=True)
    
    def _attempt_recovery(self, missing_node_ids: List[str]):
        """Пытается восстановить отсутствующие узлы."""
        for node_id in set(missing_node_ids):
            # Попытка найти похожий узел
            similar_nodes = []
            for existing_id, node in self.nodes.items():
                # Простая проверка похожести (можно улучшить)
                if len(node_id) == len(existing_id) and sum(c1 == c2 for c1, c2 in zip(node_id, existing_id)) > len(node_id) * 0.8:
                    similar_nodes.append(node)
            
            if similar_nodes:
                # Выбираем наиболее подходящий узел
                best_match = max(similar_nodes, key=lambda n: n.strength)
                
                # Обновляем связи
                for edge in self.edges.values():
                    if edge.source_id == node_id:
                        edge.source_id = best_match.id
                        self._update_edge_in_db(edge)
                    if edge.target_id == node_id:
                        edge.target_id = best_match.id
                        self._update_edge_in_db(edge)
                
                logger.info(f"Восстановлена ссылка на узел {node_id} через {best_match.id}")
            else:
                # Удаляем поврежденные связи
                edges_to_remove = [
                    edge_id for edge_id, edge in self.edges.items()
                    if edge.source_id == node_id or edge.target_id == node_id
                ]
                
                for edge_id in edges_to_remove:
                    del self.edges[edge_id]
                    self._remove_edge_from_db(edge_id)
                
                logger.warning(f"Удалено {len(edges_to_remove)} связей с отсутствующим узлом {node_id}")
    
    def _resolve_duplicates(self, duplicate_names: List[str]):
        """Решает проблемы с дубликатами узлов."""
        for name in duplicate_names:
            nodes = [
                node for node in self.nodes.values()
                if node.name.lower() == name.lower()
            ]
            
            if len(nodes) <= 1:
                continue
            
            # Сортируем по силе и актуальности
            nodes.sort(key=lambda n: (n.strength, -n.last_updated), reverse=True)
            
            # Оставляем самый сильный и актуальный узел
            primary_node = nodes[0]
            duplicate_nodes = nodes[1:]
            
            # Переносим связи
            for duplicate in duplicate_nodes:
                for edge_id, edge in list(self.edges.items()):
                    if edge.source_id == duplicate.id:
                        # Перенаправляем связь на основной узел
                        edge.source_id = primary_node.id
                        self._update_edge_in_db(edge)
                    elif edge.target_id == duplicate.id:
                        # Перенаправляем связь на основной узел
                        edge.target_id = primary_node.id
                        self._update_edge_in_db(edge)
                
                # Объединяем информацию
                if duplicate.description and (not primary_node.description or len(duplicate.description) > len(primary_node.description)):
                    primary_node.description = duplicate.description
                    primary_node.content = duplicate.description
                
                # Объединяем метаданные
                for source in duplicate.meta.get('sources', []):
                    if 'sources' not in primary_node.meta:
                        primary_node.meta['sources'] = []
                    primary_node.meta['sources'].append(source)
                
                # Обновляем основной узел
                self._update_node_in_db(primary_node)
                
                # Удаляем дубликат
                del self.nodes[duplicate.id]
                self._remove_node_from_db(duplicate.id)
            
            logger.info(f"Объединено {len(duplicate_nodes)} дубликатов для '{name}'")
    
    def _check_for_contradictions(self):
        """Проверяет граф на наличие противоречий."""
        try:
            # Проверяем противоречивые связи
            for edge_id, edge in self.edges.items():
                if edge.relation_type == RelationType.CONTRADICTS.value:
                    # Проверяем, не противоречат ли они друг другу
                    source_node = self.nodes.get(edge.source_id)
                    target_node = self.nodes.get(edge.target_id)
                    
                    if source_node and target_node:
                        # Проверяем, не связаны ли они противоположными утверждениями
                        for other_edge in self.edges.values():
                            if (other_edge.source_id == edge.source_id and 
                                other_edge.target_id == edge.target_id and
                                other_edge.relation_type != RelationType.CONTRADICTS.value):
                                
                                # Возможное противоречие
                                if not self._is_resolved_contradiction(edge, other_edge):
                                    self._record_contradiction(
                                        edge.source_id, 
                                        edge.target_id,
                                        f"Противоречие между связями {edge.relation_type} и {other_edge.relation_type}",
                                        edge_id,
                                        other_edge.id
                                    )
            
            # Проверяем узлы на противоречивые описания
            for node_id, node in self.nodes.items():
                for other_id, other in self.nodes.items():
                    if node_id == other_id:
                        continue
                    
                    # Проверяем, не являются ли они противоположными концептами
                    if self._are_opposite_concepts(node, other):
                        # Проверяем, нет ли прямой связи противоречия
                        has_contradiction = False
                        for edge in self.edges.values():
                            if ((edge.source_id == node_id and edge.target_id == other_id) or
                                (edge.source_id == other_id and edge.target_id == node_id)) and \
                               edge.relation_type == RelationType.CONTRADICTS.value:
                                has_contradiction = True
                                break
                        
                        if not has_contradiction:
                            self._record_contradiction(
                                node_id,
                                other_id,
                                f"Противоположные концепты без явного противоречия: {node.name} и {other.name}",
                                None,
                                None
                            )
            
        except Exception as e:
            logger.error(f"Ошибка проверки противоречий: {e}", exc_info=True)
    
    def _are_opposite_concepts(self, node1: KnowledgeNode, node2: KnowledgeNode) -> bool:
        """
        Проверяет, являются ли концепты противоположными.
        
        Args:
            node1: Первый узел
            node2: Второй узел
            
        Returns:
            bool: Являются ли концепты противоположными
        """
        # Простая проверка (можно улучшить с использованием NLP)
        opposites = {
            "хороший": "плохой",
            "большой": "маленький",
            "быстрый": "медленный",
            "сильный": "слабый",
            "правда": "ложь",
            "истина": "ложь",
            "верно": "неверно",
            "положительный": "отрицательный",
            "да": "нет",
            "включено": "выключено"
        }
        
        # Проверяем имена
        name1 = node1.name.lower()
        name2 = node2.name.lower()
        
        for key, value in opposites.items():
            if (key in name1 and value in name2) or (value in name1 and key in name2):
                return True
        
        # Проверяем описания
        desc1 = node1.description.lower()
        desc2 = node2.description.lower()
        
        for key, value in opposites.items():
            if (key in desc1 and value in desc2) or (value in desc1 and key in desc2):
                return True
        
        return False
    
    def _is_resolved_contradiction(self, edge1: KnowledgeEdge, edge2: KnowledgeEdge) -> bool:
        """
        Проверяет, разрешено ли противоречие.
        
        Args:
            edge1: Первая связь
            edge2: Вторая связь
            
        Returns:
            bool: Разрешено ли противоречие
        """
        # Простая проверка (можно улучшить)
        return "resolution" in edge1.meta or "resolution" in edge2.meta
    
    def _record_contradiction(self, node_id1: str, node_id2: str, evidence: str,
                             edge_id1: Optional[str] = None, 
                             edge_id2: Optional[str] = None):
        """
        Записывает противоречие в систему.
        
        Args:
            node_id1: ID первого узла
            node_id2: ID второго узла
            evidence: Доказательства противоречия
            edge_id1: ID первой связи (опционально)
            edge_id2: ID второй связи (опционально)
        """
        # Добавляем информацию о противоречии в узлы
        if node_id1 in self.nodes:
            self.nodes[node_id1].add_contradiction(node_id2, evidence)
            self._update_node_in_db(self.nodes[node_id1])
        
        if node_id2 in self.nodes:
            self.nodes[node_id2].add_contradiction(node_id1, evidence)
            self._update_node_in_db(self.nodes[node_id2])
        
        # Отправляем в SelfAnalyzer для дальнейшего анализа
        if self.brain and hasattr(self.brain, 'self_analyzer'):
            try:
                self.brain.self_analyzer.add_learning_opportunity(
                    concept=f"knowledge_contradiction_{node_id1}_{node_id2}",
                    opportunity_type="updating",
                    priority=0.9,
                    domain="knowledge",
                    evidence=[evidence],
                    suggested_actions=[
                        "Проверить источники информации",
                        "Найти дополнительные доказательства",
                        "Разрешить противоречие на основе новых данных"
                    ]
                )
            except Exception as e:
                logger.error(f"Ошибка добавления возможности обучения: {e}", exc_info=True)
        
        logger.warning(f"Обнаружено противоречие между {node_id1} и {node_id2}: {evidence}")
    
    def _check_for_outdated_knowledge(self, days_threshold: int = 365):
        """
        Проверяет знания на устаревание.
        
        Args:
            days_threshold: Порог в днях для определения устаревания
        """
        try:
            outdated_count = 0
            threshold = time.time() - (days_threshold * 86400)
            
            for node_id, node in self.nodes.items():
                # Проверяем, не устарело ли знание
                if node.last_updated < threshold:
                    outdated_count += 1
                    
                    # Отмечаем как устаревшее
                    if 'status' not in node.meta:
                        node.meta['status'] = 'outdated'
                    
                    # Добавляем в SelfAnalyzer
                    if self.brain and hasattr(self.brain, 'self_analyzer'):
                        try:
                            self.brain.self_analyzer.add_learning_opportunity(
                                concept=f"outdated_knowledge_{node_id}",
                                opportunity_type="updating",
                                priority=0.7,
                                domain="knowledge",
                                evidence=[f"Знание не обновлялось более {days_threshold} дней"],
                                suggested_actions=[
                                    "Проверить актуальность информации",
                                    "Найти новые источники данных",
                                    "Обновить информацию из новых источников"
                                ]
                            )
                        except Exception as e:
                            logger.error(f"Ошибка добавления возможности обучения: {e}", exc_info=True)
            
            if outdated_count > 0:
                logger.info(f"Обнаружено {outdated_count} устаревших знаний")
        
        except Exception as e:
            logger.error(f"Ошибка проверки устаревших знаний: {e}", exc_info=True)
    
    def _optimize_indexes(self):
        """Оптимизирует индексы для повышения производительности."""
        try:
            # Перестраиваем временной индекс
            self.temporal_index = []
            for node in self.nodes.values():
                if node.temporal_info:
                    self.temporal_index.append((node.timestamp, node.id, "node"))
                if node.last_updated != node.timestamp:
                    self.temporal_index.append((node.last_updated, node.id, "node"))
            
            for edge in self.edges.values():
                if edge.temporal_info:
                    self.temporal_index.append((edge.timestamp, edge.id, "edge"))
                if edge.last_updated != edge.timestamp:
                    self.temporal_index.append((edge.last_updated, edge.id, "edge"))
            
            # Сортируем временной индекс
            # Сортируем временной индекс, приводя timestamp к float
            def sort_key(x):
                ts = x[0]
                if isinstance(ts, (int, float)):
                    return float(ts)
                elif isinstance(ts, str):
                    try:
                        return float(ts)
                    except (ValueError, TypeError):
                        return 0.0
                return 0.0
            self.temporal_index.sort(key=sort_key)
            
            logger.debug("Индексы графа знаний оптимизированы")
        except Exception as e:
            logger.error(f"Ошибка оптимизации индексов: {e}", exc_info=True)
    
    def _cleanup_cache(self):
        """Очищает устаревший кэш."""
        try:
            if self.hybrid_cache:
                # Очищаем устаревший кэш
                self.hybrid_cache.cleanup()
                logger.debug("Кэш графа знаний очищен")
        except Exception as e:
            logger.error(f"Ошибка очистки кэша: {e}", exc_info=True)
    
    def add_node(self, name: str, description: str, node_type: str = "fact", 
                domain: str = "general", strength: float = 0.5,
                meta: Optional[Dict] = None, spatial_info: Optional[Dict] = None,
                temporal_info: Optional[Dict] = None, user_id: Optional[str] = None,
                source: Optional[str] = None) -> str:
        """
        Добавляет новый узел в граф знаний.
        
        Args:
            name: Название узла
            description: Описание узла
            node_type: Тип узла
            domain: Домен знаний
            strength: Сила знания
            meta: Дополнительные метаданные
            spatial_info: Пространственная информация
            temporal_info: Временная информация
            user_id: ID пользователя
            source: Источник информации
            
        Returns:
            str: ID добавленного узла
        """
        start_time = time.time()
        node_id = f"node_{int(time.time())}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        
        try:
            # Проверяем на дубликаты
            existing_nodes = self.search_nodes(name, limit=1, domains=[domain])
            if existing_nodes:
                # Если найден существующий узел, обновляем его
                existing_node = existing_nodes[0]
                existing_node.update(
                    description, 
                    strength=strength,
                    source=source,
                    user_id=user_id,
                    spatial_info=spatial_info,
                    temporal_info=temporal_info
                )
                self._update_node_in_db(existing_node)
                self._update_indexes(node=existing_node)
                
                # Обновляем статистику
                self.stats["node_updates"] += 1
                self._update_statistics(start_time, True)
                
                logger.info(f"Обновлен существующий узел в графе знаний: {existing_node.id} ({name})")
                return existing_node.id
            
            # Создаем новый узел
            node = KnowledgeNode(
                id=node_id,
                name=name,
                description=description,
                node_type=node_type,
                domain=domain,
                strength=strength,
                meta=meta,
                spatial_info=spatial_info,
                temporal_info=temporal_info
            )
            
            # Добавляем источник
            if source:
                node.meta['sources'].append({
                    'source': source,
                    'timestamp': time.time(),
                    'user_id': user_id,
                    'version': node.version
                })
            
            # Сохраняем в базу данных
            self._save_node_to_db(node)
            
            # Добавляем в локальный кэш
            self.nodes[node_id] = node
            self._update_indexes(node=node)
            
            # Обновляем статистику
            self.stats["total_nodes"] += 1
            self.stats["node_creations"] += 1
            self._update_statistics(start_time, True)
            
            logger.debug(f"Добавлен узел в граф знаний: {node_id} ({name})")
            return node_id
            
        except Exception as e:
            logger.error(f"Ошибка добавления узла в граф знаний: {e}", exc_info=True)
            self._update_statistics(start_time, False)
            return ""

    def add_concept(self, concept: str, description: str,
                    domain: str = "general", strength: float = 1.0,
                    source: Optional[str] = None, tags: Optional[list] = None,
                    user_id: Optional[str] = None,
                    spatial_info: Optional[Dict] = None,
                    temporal_info: Optional[Dict] = None) -> str:
        """
        Обертка для обратной совместимости: добавляет концепт как узел типа "concept".

        Args:
            concept: Название концепта
            description: Описание концепта
            domain: Домен знаний
            strength: Сила знания (0.0-1.0)
            source: Источник информации
            tags: Теги (необязательно)
            user_id: ID пользователя (необязательно)
            spatial_info: Пространственные метаданные (необязательно)
            temporal_info: Временные метаданные (необязательно)

        Returns:
            str: ID созданного узла
        """
        meta = {"source": source, "tags": (tags or []), "user_id": user_id}
        return self.add_node(
            name=concept,
            description=description,
            node_type="concept",
            domain=domain,
            strength=strength,
            meta=meta,
            spatial_info=spatial_info,
            temporal_info=temporal_info,
            user_id=user_id,
            source=source,
        )
    
    def _update_statistics(self, start_time: float, success: bool):
        """Обновляет статистику запросов."""
        processing_time = time.time() - start_time
        self.stats["total_queries"] += 1
        
        if success:
            self.stats["successful_queries"] += 1
        else:
            self.stats["failed_queries"] += 1
        
        self.stats["total_processing_time"] += processing_time
    
    def _save_node_to_db(self, node: KnowledgeNode) -> bool:
        """
        Сохраняет узел в базу данных.
        
        Args:
            node: Узел для сохранения
            
        Returns:
            bool: Успешно ли сохранено
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT OR REPLACE INTO nodes 
            (id, name, description, node_type, domain, strength, timestamp, last_updated, 
            version, meta, spatial_info, temporal_info, history, contradictions, keyword_index, concept_index)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                node.id,
                node.name,
                node.description,
                node.node_type,
                node.domain,
                node.strength,
                node.timestamp,
                node.last_updated,
                node.version,
                json.dumps(node.meta),
                json.dumps(node.spatial_info),
                json.dumps(node.temporal_info),
                json.dumps(node.history),
                json.dumps(node.contradictions),
                json.dumps(node.keyword_index),
                json.dumps(node.concept_index)
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка сохранения узла {node.id} в БД: {e}", exc_info=True)
            return False
    
    def _update_node_in_db(self, node: KnowledgeNode) -> bool:
        """
        Обновляет узел в базе данных.
        
        Args:
            node: Узел для обновления
            
        Returns:
            bool: Успешно ли обновлено
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
            UPDATE nodes SET
                name = ?,
                description = ?,
                node_type = ?,
                domain = ?,
                strength = ?,
                last_updated = ?,
                version = ?,
                meta = ?,
                spatial_info = ?,
                temporal_info = ?,
                history = ?,
                contradictions = ?,
                keyword_index = ?,
                concept_index = ?
            WHERE id = ?
            """, (
                node.name,
                node.description,
                node.node_type,
                node.domain,
                node.strength,
                node.last_updated,
                node.version,
                json.dumps(node.meta),
                json.dumps(node.spatial_info),
                json.dumps(node.temporal_info),
                json.dumps(node.history),
                json.dumps(node.contradictions),
                json.dumps(node.keyword_index),
                json.dumps(node.concept_index),
                node.id
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обновления узла {node.id} в БД: {e}", exc_info=True)
            return False
    
    def _remove_node_from_db(self, node_id: str) -> bool:
        """
        Удаляет узел из базы данных.
        
        Args:
            node_id: ID узла для удаления
            
        Returns:
            bool: Успешно ли удалено
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Удаляем сначала связи
            cursor.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (node_id, node_id))
            
            # Удаляем узел
            cursor.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка удаления узла {node_id} из БД: {e}", exc_info=True)
            return False
    
    def _save_edge_to_db(self, edge: KnowledgeEdge) -> bool:
        """
        Сохраняет связь в базу данных.
        
        Args:
            edge: Связь для сохранения
            
        Returns:
            bool: Успешно ли сохранено
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT OR REPLACE INTO edges 
            (id, source_id, target_id, relation_type, strength, timestamp, 
            last_updated, version, meta, spatial_info, temporal_info, history)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                edge.id,
                edge.source_id,
                edge.target_id,
                edge.relation_type,
                edge.strength,
                edge.timestamp,
                edge.last_updated,
                edge.version,
                json.dumps(edge.meta),
                json.dumps(edge.spatial_info),
                json.dumps(edge.temporal_info),
                json.dumps(edge.history)
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка сохранения связи {edge.id} в БД: {e}", exc_info=True)
            return False
    
    def _update_edge_in_db(self, edge: KnowledgeEdge) -> bool:
        """
        Обновляет связь в базе данных.
        
        Args:
            edge: Связь для обновления
            
        Returns:
            bool: Успешно ли обновлено
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
            UPDATE edges SET
                source_id = ?,
                target_id = ?,
                relation_type = ?,
                strength = ?,
                last_updated = ?,
                version = ?,
                meta = ?,
                spatial_info = ?,
                temporal_info = ?,
                history = ?
            WHERE id = ?
            """, (
                edge.source_id,
                edge.target_id,
                edge.relation_type,
                edge.strength,
                edge.last_updated,
                edge.version,
                json.dumps(edge.meta),
                json.dumps(edge.spatial_info),
                json.dumps(edge.temporal_info),
                json.dumps(edge.history),
                edge.id
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обновления связи {edge.id} в БД: {e}", exc_info=True)
            return False
    
    def _remove_edge_from_db(self, edge_id: str) -> bool:
        """
        Удаляет связь из базы данных.
        
        Args:
            edge_id: ID связи для удаления
            
        Returns:
            bool: Успешно ли удалено
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка удаления связи {edge_id} из БД: {e}", exc_info=True)
            return False
    
    def _record_history(self, change_type: str, changes: Dict[str, Any], 
                       node_id: Optional[str] = None, edge_id: Optional[str] = None,
                       user_id: Optional[str] = None, source: Optional[str] = None):
        """
        Записывает изменение в историю.
        
        Args:
            change_type: Тип изменения
            changes: Изменения
            node_id: ID узла (опционально)
            edge_id: ID связи (опционально)
            user_id: ID пользователя
            source: Источник изменения
        """
        try:
            history_id = f"history_{int(time.time())}_{hashlib.md5(json.dumps(changes).encode()).hexdigest()[:8]}"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT INTO history
            (id, node_id, edge_id, change_type, changes, timestamp, user_id, source, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                history_id,
                node_id,
                edge_id,
                change_type,
                json.dumps(changes),
                time.time(),
                user_id,
                source,
                1  # Версия записи истории
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Ошибка записи в историю: {e}", exc_info=True)
    
    def add_edge(self, source_id: str, target_id: str, relation_type: str,
                strength: float = 0.5, meta: Optional[Dict] = None,
                spatial_info: Optional[Dict] = None,
                temporal_info: Optional[Dict] = None, user_id: Optional[str] = None,
                source: Optional[str] = None) -> str:
        """
        Добавляет связь между узлами в графе знаний.
        
        Args:
            source_id: ID исходного узла
            target_id: ID целевого узла
            relation_type: Тип связи
            strength: Сила связи
            meta: Дополнительные метаданные
            spatial_info: Пространственная информация
            temporal_info: Временная информация
            user_id: ID пользователя
            source: Источник информации
            
        Returns:
            str: ID добавленной связи
        """
        start_time = time.time()
        edge_id = f"edge_{int(time.time())}_{hashlib.md5(f'{source_id}_{target_id}'.encode()).hexdigest()[:8]}"
        
        try:
            # Проверяем существование узлов
            if source_id not in self.nodes:
                logger.warning(f"Исходный узел {source_id} не существует")
                self._update_statistics(start_time, False)
                return ""
            
            if target_id not in self.nodes:
                logger.warning(f"Целевой узел {target_id} не существует")
                self._update_statistics(start_time, False)
                return ""
            
            # Проверяем на дубликаты
            existing_edge = None
            for edge in self.edges.values():
                if (edge.source_id == source_id and edge.target_id == target_id and 
                    edge.relation_type == relation_type):
                    existing_edge = edge
                    break
            
            if existing_edge:
                # Если найдена существующая связь, обновляем её
                existing_edge.update(
                    new_strength=strength,
                    source=source,
                    user_id=user_id,
                    spatial_info=spatial_info,
                    temporal_info=temporal_info
                )
                self._update_edge_in_db(existing_edge)
                self._update_indexes(edge=existing_edge)
                
                # Обновляем статистику
                self.stats["edge_updates"] += 1
                self._update_statistics(start_time, True)
                
                logger.info(f"Обновлена существующая связь в графе знаний: {existing_edge.id}")
                return existing_edge.id
            
            # Создаем новую связь
            edge = KnowledgeEdge(
                id=edge_id,
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                strength=strength,
                meta=meta,
                spatial_info=spatial_info,
                temporal_info=temporal_info
            )
            
            # Добавляем источник
            if source:
                edge.meta['sources'].append({
                    'source': source,
                    'timestamp': time.time(),
                    'user_id': user_id,
                    'version': edge.version
                })
            
            # Сохраняем в базу данных
            self._save_edge_to_db(edge)
            
            # Добавляем в локальный кэш
            self.edges[edge_id] = edge
            self._update_indexes(edge=edge)
            
            # Обновляем статистику
            self.stats["total_edges"] += 1
            self.stats["edge_creations"] += 1
            self._update_statistics(start_time, True)
            
            logger.debug(f"Добавлена связь в граф знаний: {edge_id} ({source_id} -> {target_id})")
            return edge_id
            
        except Exception as e:
            logger.error(f"Ошибка добавления связи в граф знаний: {e}", exc_info=True)
            self._update_statistics(start_time, False)
            return ""
    
    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """
        Возвращает узел по ID.
        
        Args:
            node_id: ID узла
            
        Returns:
            Optional[KnowledgeNode]: Узел или None
        """
        return self.nodes.get(node_id)
    
    def get_relevant_nodes(self, query: str, limit: int = 50,
                           domains: Optional[List[str]] = None,
                           node_types: Optional[List[str]] = None,
                           min_strength: float = 0.0) -> List["KnowledgeNode"]:
        """
        Возвращает релевантные запросу узлы (быстрый отбор для расширения контекста).
        Использует встроенный поиск и последующую фильтрацию по силе/домену/типу.
        """
        try:
            # Базовый отбор через внутренний поиск (если доступен) или публичный search_nodes
            if hasattr(self, "_search_nodes_internal"):
                nodes = self._search_nodes_internal(
                    query=query, limit=limit * 2,
                    domains=domains, node_types=node_types,
                    min_strength=min_strength
                )
            elif hasattr(self, "search_nodes"):
                nodes = self.search_nodes(query, limit=limit * 2)
                # Фильтры
                if domains:
                    nodes = [n for n in nodes if n.domain in domains]
                if node_types:
                    nodes = [n for n in nodes if n.node_type in node_types]
                if min_strength > 0.0:
                    nodes = [n for n in nodes if getattr(n, "strength", 0.0) >= min_strength]
            else:
                return []

            # Ограничим до лимита после первичного отбора
            return nodes[:limit]
        except Exception:
            return []

    def update_node_weights(self, node_id: str, feedback: Dict[str, Any]):
        """Обновляет веса узла на основе обратной связи пользователя/системы."""
        try:
            node = self.get_node(node_id)
            if not node:
                return

            # Обновляем базовую силу на основе верификации (0..1)
            verification = feedback.get("verification")
            if isinstance(verification, (int, float)):
                node.strength = max(0.0, min(1.0, node.strength * (0.5 + 0.5 * float(verification))))

            # Обновляем временные метаданные и частоту использования
            now = time.time()
            node.timestamp = now
            meta = getattr(node, "meta", {}) or {}
            meta["last_used"] = now
            meta["usage_count"] = int(meta.get("usage_count", 0)) + 1
            node.meta = meta

            if hasattr(self, "_update_node_in_db"):
                self._update_node_in_db(node)
            else:
                # Если нет прямого метода — сохраняем в БД и в памяти
                self.nodes[node.id] = node
                if hasattr(self, "_update_node_in_db"):
                    self._update_node_in_db(node)
        except Exception:
            # Логирование уже есть на уровне вызывающих методов
            pass

    def prioritize_nodes(self, query: str, nodes: List["KnowledgeNode"]) -> List[Tuple["KnowledgeNode", float]]:
        """Возвращает узлы, отсортированные по динамическому весу для данного запроса."""
        scored: List[Tuple["KnowledgeNode", float]] = []
        for node in nodes:
            try:
                relevance = self._calculate_relevance(query, node)
                temporal_factor = self._calculate_temporal_factor(node)
                usage_factor = self._calculate_usage_factor(node)
                base_strength = getattr(node, "strength", 0.0) or 0.0

                total_weight = (base_strength * 0.3) + (relevance * 0.4) + (temporal_factor * 0.2) + (usage_factor * 0.1)
                scored.append((node, float(total_weight)))
            except Exception:
                # Если ошибка — понижаем вес
                scored.append((node, 0.0))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # === Вспомогательные оценки для приоритизации узлов ===
    def _calculate_relevance(self, query: str, node: "KnowledgeNode") -> float:
        """
        Приближенная оценка релевантности (0..1) по семантическому сходству.
        Реализация упрощена: использует совпадение ключевых слов, описаний и названия.
        При наличии продвинутых эмбеддингов метод может быть переопределён.
        """
        try:
            text = f"{getattr(node, 'name', '')} {getattr(node, 'description', '')}".lower()
            q = (query or "").lower().strip()
            if not q:
                return 0.0
            # Простая метрика: доля совпавших уникальных слов запроса в тексте
            q_tokens = [t for t in q.split() if len(t) > 2]
            if not q_tokens:
                return 0.0
            matches = sum(1 for t in set(q_tokens) if t in text)
            return max(0.0, min(1.0, matches / max(1, len(set(q_tokens)))))
        except Exception:
            return 0.0

    def _calculate_temporal_factor(self, node: "KnowledgeNode") -> float:
        """Учет актуальности: 1.0 для свежих, уменьшается со временем (в пределах ~30 дней)."""
        try:
            last = getattr(node, "last_updated", None) or getattr(node, "timestamp", None) or 0
            age = max(0.0, time.time() - float(last))
            # Нормализуем: <1 день ~1.0; ~30 дней ~0.5; >=180 дней ~0.1
            day = 86400.0
            if age <= day:
                return 1.0
            elif age <= 30 * day:
                return 0.5 + 0.5 * (1 - (age - day) / (29 * day))  # плавное снижение 1.0->0.5
            elif age <= 180 * day:
                return 0.1 + 0.4 * (1 - (age - 30 * day) / (150 * day))  # 0.5->0.1
            else:
                return 0.05
        except Exception:
            return 0.5

    def _calculate_usage_factor(self, node: "KnowledgeNode") -> float:
        """Нормализованная частота использования (0..1)."""
        try:
            meta = getattr(node, "meta", {}) or {}
            usage = int(meta.get("usage_count", 0))
            if usage <= 0:
                return 0.0
            # Логарифмическое насыщение: 1 при ~100+ обращениях
            import math
            return max(0.0, min(1.0, math.log10(1 + usage) / 2.0))
        except Exception:
            return 0.0

    def get_edge(self, edge_id: str) -> Optional[KnowledgeEdge]:
        """
        Возвращает связь по ID.
        
        Args:
            edge_id: ID связи
            
        Returns:
            Optional[KnowledgeEdge]: Связь или None
        """
        return self.edges.get(edge_id)
    
    def get_edges(self, node_id: str, direction: str = "both", 
                 relation_type: Optional[str] = None) -> List[KnowledgeEdge]:
        """
        Возвращает связи для узла.
        
        Args:
            node_id: ID узла
            direction: Направление связей (source, target, both)
            relation_type: Тип связи для фильтрации
            
        Returns:
            List[KnowledgeEdge]: Список связей
        """
        edges = []
        
        for edge in self.edges.values():
            if direction == "source" and edge.source_id == node_id:
                if relation_type is None or edge.relation_type == relation_type:
                    edges.append(edge)
            elif direction == "target" and edge.target_id == node_id:
                if relation_type is None or edge.relation_type == relation_type:
                    edges.append(edge)
            elif direction == "both" and (edge.source_id == node_id or edge.target_id == node_id):
                if relation_type is None or edge.relation_type == relation_type:
                    edges.append(edge)
                
        return edges
    
    def get_related_nodes(self, node_id: str, relation_type: Optional[str] = None, 
                         direction: str = "both", max_distance: int = 1) -> List[KnowledgeNode]:
        """
        Возвращает связанные узлы на указанном расстоянии.
        
        Args:
            node_id: ID узла
            relation_type: Тип связи для фильтрации
            direction: Направление связей
            max_distance: Максимальное расстояние
            
        Returns:
            List[KnowledgeNode]: Список связанных узлов
        """
        if max_distance < 1:
            return []
        
        visited = set([node_id])
        related_nodes = []
        
        # Сначала получаем прямые связи
        direct_edges = self.get_edges(node_id, direction=direction, relation_type=relation_type)
        for edge in direct_edges:
            neighbor_id = edge.target_id if edge.source_id == node_id else edge.source_id
            if neighbor_id not in visited:
                neighbor = self.get_node(neighbor_id)
                if neighbor:
                    related_nodes.append(neighbor)
                    visited.add(neighbor_id)
        
        # Если нужно больше уровней, рекурсивно получаем следующие уровни
        if max_distance > 1:
            for node in related_nodes.copy():
                deeper_nodes = self.get_related_nodes(
                    node.id, 
                    relation_type=relation_type,
                    direction=direction,
                    max_distance=max_distance - 1
                )
                for deeper_node in deeper_nodes:
                    if deeper_node.id not in visited:
                        related_nodes.append(deeper_node)
                        visited.add(deeper_node.id)
        
        return related_nodes
    
    def get_all_nodes(self) -> List[KnowledgeNode]:
        """
        Возвращает все узлы графа.
        
        Returns:
            List[KnowledgeNode]: Список всех узлов
        """
        return list(self.nodes.values())

    def get_nodes_by_domain(self, domain: str) -> List[KnowledgeNode]:
        """
        Возвращает узлы указанного домена.
        
        Args:
            domain: Домен для фильтрации узлов
            
        Returns:
            List[KnowledgeNode]: Список узлов указанного домена
        """
        return [node for node in self.nodes.values() if node.domain == domain]
    
    def get_all_edges(self) -> List[KnowledgeEdge]:
        """
        Возвращает все связи графа.
        
        Returns:
            List[KnowledgeEdge]: Список всех связей
        """
        return list(self.edges.values())
    
    def search_nodes(self, query: str, limit: int = 10, 
                   domains: Optional[List[str]] = None,
                   node_types: Optional[List[str]] = None,
                   min_strength: float = 0.0) -> List[KnowledgeNode]:
        """
        Ищет узлы в графе знаний по запросу с поддержкой гибридного кэша.
        
        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            domains: Фильтр по доменам
            node_types: Фильтр по типам узлов
            min_strength: Минимальная сила знания
            
        Returns:
            List[KnowledgeNode]: Список найденных узлов
        """
        start_time = time.time()
        
        # Формируем ключ кэша
        cache_key = self._generate_cache_key(query, domains, node_types, min_strength, limit)
        
        # Проверяем гибридный кэш
        if self.hybrid_cache:
            cached_result = self.hybrid_cache.get_token(cache_key)
            if cached_result:
                logger.debug(f"Найдены кэшированные результаты поиска для '{query}'")
                self._update_statistics(start_time, True)
                return cached_result
        
        # Выполняем поиск
        results = self._search_nodes_internal(query, limit, domains, node_types, min_strength)
        
        # Сохраняем в кэш
        if self.hybrid_cache and results:
            self.hybrid_cache.add_token(cache_key, results)
        
        self._update_statistics(start_time, True)
        return results
    
    def _generate_cache_key(self, query: str, domains: Optional[List[str]], 
                           node_types: Optional[List[str]], min_strength: float, 
                           limit: int) -> str:
        """
        Генерирует ключ кэша для поискового запроса.
        
        Args:
            query: Поисковый запрос
            domains: Домены
            node_types: Типы узлов
            min_strength: Минимальная сила
            limit: Лимит результатов
            
        Returns:
            str: Ключ кэша
        """
        config = {
            "query": query,
            "domains": sorted(domains) if domains else None,
            "node_types": sorted(node_types) if node_types else None,
            "min_strength": min_strength,
            "limit": limit
        }
        config_str = json.dumps(config, sort_keys=True)
        return f"search:{hashlib.md5(config_str.encode()).hexdigest()}"
    
    def _search_nodes_internal(self, query: str, limit: int = 10, 
                              domains: Optional[List[str]] = None,
                              node_types: Optional[List[str]] = None,
                              min_strength: float = 0.0) -> List[KnowledgeNode]:
        """
        Выполняет внутренний поиск узлов.
        
        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            domains: Фильтр по доменам
            node_types: Фильтр по типам узлов
            min_strength: Минимальная сила знания
            
        Returns:
            List[KnowledgeNode]: Список найденных узлов
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Подготавливаем запрос
            query_lower = query.lower()
            # Параметры должны соответствовать порядку плейсхолдеров в SQL:
            # (LIKE name, LIKE description, strength, [domains...], [node_types...], limit)
            params = [f"%{query_lower}%", f"%{query_lower}%", min_strength]
            
            sql = """
            SELECT id, name, description, node_type, domain, strength, timestamp, last_updated, 
                   version, meta, spatial_info, temporal_info, history, contradictions, 
                   keyword_index, concept_index
            FROM nodes
            WHERE (LOWER(name) LIKE ? OR LOWER(description) LIKE ?)
            AND strength >= ?
            """
            
            # Добавляем фильтр по доменам, если указан
            if domains and len(domains) > 0:
                placeholders = ",".join(["?" for _ in domains])
                sql += f" AND domain IN ({placeholders})"
                params.extend(domains)
            
            # Добавляем фильтр по типам узлов, если указан
            if node_types and len(node_types) > 0:
                placeholders = ",".join(["?" for _ in node_types])
                sql += f" AND node_type IN ({placeholders})"
                params.extend(node_types)
            
            # Добавляем лимит
            sql += " ORDER BY strength DESC LIMIT ?"
            params.append(limit)
            
            # Диагностика: проверяем соответствие количества плейсхолдеров и параметров
            placeholders_count = sql.count("?")
            if placeholders_count != len(params):
                logger.error(
                    f"Несоответствие параметров SQL: плейсхолдеров={placeholders_count}, параметров={len(params)}. "
                    f"SQL={sql} | params={params}"
                )
            else:
                logger.debug(
                    f"Выполняем SQL запрос. Плейсхолдеров={placeholders_count}, параметров={len(params)}. "
                    f"SQL={sql} | params={params}"
                )

            cursor.execute(sql, params)
            
            results = []
            for row in cursor.fetchall():
                node = KnowledgeNode(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    node_type=row[3],
                    domain=row[4],
                    strength=row[5],
                    timestamp=row[6],
                    meta=safe_json_loads(row[9]) if len(row) > 9 and row[9] else {},
                    version=row[8],
                    spatial_info=safe_json_loads(row[10]) if len(row) > 10 and row[10] else {},
                    temporal_info=safe_json_loads(row[11]) if len(row) > 11 and row[11] else {}
                )
                node.last_updated = row[7]
                node.history = safe_json_loads(row[12]) if len(row) > 12 and row[12] else []
                node.contradictions = safe_json_loads(row[13]) if len(row) > 13 and row[13] else []
                node.keyword_index = safe_json_loads(row[14]) if len(row) > 14 and row[14] else []
                node.concept_index = safe_json_loads(row[15]) if len(row) > 15 and row[15] else []
                
                results.append(node)
            
            logger.debug(f"Найдено {len(results)} узлов по запросу '{query}'")
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Ошибка поиска узлов в графе знаний: {e}", exc_info=True)
            return []
    
    def search_by_concept(self, concept: str, limit: int = 5) -> List[KnowledgeNode]:
        """
        Ищет узлы по концепту с использованием NLP.
        
        Args:
            concept: Концепт для поиска
            limit: Максимальное количество результатов
            
        Returns:
            List[KnowledgeNode]: Список найденных узлов
        """
        if not self.text_processor:
            return self.search_nodes(concept, limit)
        
        try:
            # Анализируем концепт
            analysis = self.text_processor.process_text(concept)
            
            # Используем ключевые слова для поиска
            keywords = [kw["word"] for kw in analysis.keywords[:3]] if analysis.keywords else [concept]
            
            # Выполняем поиск по каждому ключевому слову
            all_results = []
            for keyword in keywords:
                results = self.search_nodes(keyword, limit=limit)
                all_results.extend(results)
            
            # Уникализируем результаты
            seen = set()
            unique_results = []
            for result in all_results:
                if result.id not in seen:
                    seen.add(result.id)
                    unique_results.append(result)
            
            # Сортируем по релевантности
            unique_results.sort(key=lambda x: x.strength, reverse=True)
            
            return unique_results[:limit]
            
        except Exception as e:
            logger.error(f"Ошибка поиска по концепту: {e}", exc_info=True)
            return self.search_nodes(concept, limit)
    
    def update_node(self, node_id: str, new_description: str, 
                   strength: Optional[float] = None, 
                   source: Optional[str] = None,
                   user_id: Optional[str] = None,
                   spatial_info: Optional[Dict[str, Any]] = None,
                   temporal_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        Обновляет узел в графе знаний.
        
        Args:
            node_id: ID узла
            new_description: Новое описание
            strength: Новая сила связи
            source: Источник информации
            user_id: ID пользователя
            spatial_info: Новая пространственная информация
            temporal_info: Новая временная информация
            
        Returns:
            bool: Успешно ли обновлено
        """
        try:
            # Получаем текущий узел
            node = self.get_node(node_id)
            if not node:
                logger.warning(f"Узел с ID {node_id} не найден")
                return False
            
            # Обновляем узел
            node.update(
                new_description,
                strength=strength,
                source=source,
                user_id=user_id,
                spatial_info=spatial_info,
                temporal_info=temporal_info
            )
            
            # Сохраняем обновления
            self._update_node_in_db(node)
            
            # Обновляем индексы
            self._update_indexes(node=node)
            
            # Записываем в историю
            self._record_history(
                "node_updated",
                {
                    "node_id": node_id,
                    "new_description": new_description,
                    "new_strength": strength,
                    "spatial_info": spatial_info,
                    "temporal_info": temporal_info
                },
                node_id=node_id,
                user_id=user_id,
                source=source
            )
            
            logger.info(f"Узел '{node.name}' (ID: {node_id}) успешно обновлен")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обновления узла: {e}", exc_info=True)
            return False
    
    def update_edge(self, edge_id: str, new_strength: Optional[float] = None,
                  source: Optional[str] = None,
                  user_id: Optional[str] = None,
                  spatial_info: Optional[Dict[str, Any]] = None,
                  temporal_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        Обновляет связь в графе знаний.
        
        Args:
            edge_id: ID связи
            new_strength: Новая сила связи
            source: Источник информации
            user_id: ID пользователя
            spatial_info: Новая пространственная информация
            temporal_info: Новая временная информация
            
        Returns:
            bool: Успешно ли обновлено
        """
        try:
            # Получаем текущую связь
            edge = self.get_edge(edge_id)
            if not edge:
                logger.warning(f"Связь с ID {edge_id} не найдена")
                return False
            
            # Обновляем связь
            edge.update(
                new_strength=new_strength,
                source=source,
                user_id=user_id,
                spatial_info=spatial_info,
                temporal_info=temporal_info
            )
            
            # Сохраняем обновления
            self._update_edge_in_db(edge)
            
            # Обновляем индексы
            self._update_indexes(edge=edge)
            
            # Записываем в историю
            self._record_history(
                "edge_updated",
                {
                    "edge_id": edge_id,
                    "new_strength": new_strength,
                    "spatial_info": spatial_info,
                    "temporal_info": temporal_info
                },
                edge_id=edge_id,
                user_id=user_id,
                source=source
            )
            
            logger.info(f"Связь (ID: {edge_id}) успешно обновлена")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обновления связи: {e}", exc_info=True)
            return False
    
    def remove_node(self, node_id: str, user_id: Optional[str] = None) -> bool:
        """
        Удаляет узел из графа знаний.
        
        Args:
            node_id: ID узла
            user_id: ID пользователя
            
        Returns:
            bool: Успешно ли удалено
        """
        try:
            # Получаем текущий узел
            node = self.get_node(node_id)
            if not node:
                logger.warning(f"Узел с ID {node_id} не найден")
                return False
            
            # Удаляем все связанные связи
            edges_to_remove = []
            for edge_id, edge in self.edges.items():
                if edge.source_id == node_id or edge.target_id == node_id:
                    edges_to_remove.append(edge_id)
            
            for edge_id in edges_to_remove:
                del self.edges[edge_id]
                self._remove_edge_from_db(edge_id)
            
            # Удаляем узел
            del self.nodes[node_id]
            self._remove_node_from_db(node_id)
            
            # Записываем в историю
            self._record_history(
                "node_removed",
                {"node_id": node_id, "name": node.name},
                node_id=node_id,
                user_id=user_id
            )
            
            # Обновляем статистику
            self.stats["total_nodes"] -= 1
            
            logger.info(f"Узел '{node.name}' (ID: {node_id}) успешно удален")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка удаления узла: {e}", exc_info=True)
            return False
    
    def remove_all_concepts(self, user_id: Optional[str] = None) -> int:
        """
        Массово удаляет все узлы типа "concept" из графа знаний.

        Args:
            user_id: ID пользователя, инициировавшего удаление (опционально)

        Returns:
            int: Количество успешно удалённых узлов
        """
        try:
            # Собираем список ID узлов типа "concept" на текущий момент
            concept_ids = [n.id for n in list(self.nodes.values()) if getattr(n, "node_type", None) == "concept"]
            removed = 0
            for nid in concept_ids:
                try:
                    if self.remove_node(nid, user_id=user_id):
                        removed += 1
                except Exception:
                    # Локально подавляем, общий отчёт ниже
                    logger.error(f"Не удалось удалить концепт с ID: {nid}", exc_info=True)
            logger.info(f"Массовое удаление концептов завершено: удалено {removed} из {len(concept_ids)}")
            return removed
        except Exception as e:
            logger.error(f"Ошибка массового удаления концептов: {e}", exc_info=True)
            return 0
    
    def remove_edge(self, edge_id: str, user_id: Optional[str] = None) -> bool:
        """
        Удаляет связь из графа знаний.
        
        Args:
            edge_id: ID связи
            user_id: ID пользователя
            
        Returns:
            bool: Успешно ли удалено
        """
        try:
            # Получаем текущую связь
            edge = self.get_edge(edge_id)
            if not edge:
                logger.warning(f"Связь с ID {edge_id} не найдена")
                return False
            
            # Удаляем связь
            del self.edges[edge_id]
            self._remove_edge_from_db(edge_id)
            
            # Записываем в историю
            self._record_history(
                "edge_removed",
                {
                    "edge_id": edge_id,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relation_type": edge.relation_type
                },
                edge_id=edge_id,
                user_id=user_id
            )
            
            # Обновляем статистику
            self.stats["total_edges"] -= 1
            
            logger.info(f"Связь (ID: {edge_id}) успешно удалена")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка удаления связи: {e}", exc_info=True)
            return False
    
    def get_subgraph(self, center_node_id: str, depth: int = 2) -> Dict[str, Any]:
        """
        Возвращает подграф с центром в указанном узле.
        
        Args:
            center_node_id: ID центрального узла
            depth: Глубина подграфа
            
        Returns:
            Dict[str, Any]: Подграф (узлы и связи)
        """
        try:
            # Проверяем существование центрального узла
            if center_node_id not in self.nodes:
                logger.warning(f"Центральный узел {center_node_id} не существует")
                return {"nodes": [], "edges": []}
            
            # Используем BFS для построения подграфа
            visited = set([center_node_id])
            nodes = [self.nodes[center_node_id]]
            edges = []
            
            current_level = [center_node_id]
            for _ in range(depth):
                next_level = []
                
                for node_id in current_level:
                    # Получаем все исходящие связи
                    for edge in self.get_edges(node_id, direction="source"):
                        if edge.target_id not in visited:
                            visited.add(edge.target_id)
                            next_level.append(edge.target_id)
                            edges.append(edge)
                            
                            if edge.target_id in self.nodes:
                                nodes.append(self.nodes[edge.target_id])
                    
                    # Получаем все входящие связи
                    for edge in self.get_edges(node_id, direction="target"):
                        if edge.source_id not in visited:
                            visited.add(edge.source_id)
                            next_level.append(edge.source_id)
                            edges.append(edge)
                            
                            if edge.source_id in self.nodes:
                                nodes.append(self.nodes[edge.source_id])
                
                current_level = next_level
            
            return {
                "nodes": [node.to_dict() for node in nodes],
                "edges": [edge.to_dict() for edge in edges]
            }
            
        except Exception as e:
            logger.error(f"Ошибка построения подграфа: {e}", exc_info=True)
            return {"nodes": [], "edges": []}
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику по графу знаний.
        
        Returns:
            Dict[str, Any]: Статистика
        """
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "domains": self._get_domain_statistics(),
            "node_types": self._get_node_type_statistics(),
            "last_update": time.time(),
            "stats": self.stats.copy()
        }
    
    def _get_domain_statistics(self) -> Dict[str, int]:
        """Возвращает статистику по доменам."""
        stats = {}
        for node in self.nodes.values():
            if node.domain in stats:
                stats[node.domain] += 1
            else:
                stats[node.domain] = 1
        return stats
    
    def _get_node_type_statistics(self) -> Dict[str, int]:
        """Возвращает статистику по типам узлов."""
        stats = {}
        for node in self.nodes.values():
            if node.node_type in stats:
                stats[node.node_type] += 1
            else:
                stats[node.node_type] = 1
        return stats

    def _get_cache_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику кэша."""
        cache_stats = {
            "hybrid_cache_enabled": self.hybrid_cache is not None,
            "cache_dir": self.cache_dir,
            "cache_size_mb": 0.0,
            "cache_entries": 0,
            "cache_hit_rate": 0.0
        }
        
        try:
            # Размер директории кэша
            if os.path.exists(self.cache_dir):
                total_size = 0
                for root, dirs, files in os.walk(self.cache_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            total_size += os.path.getsize(file_path)
                        except OSError:
                            continue
                cache_stats["cache_size_mb"] = total_size / (1024 * 1024)
            
            # Статистика гибридного кэша
            if self.hybrid_cache and hasattr(self.hybrid_cache, 'get_stats'):
                try:
                    hybrid_stats = self.hybrid_cache.get_stats()
                    cache_stats.update({
                        "cache_entries": hybrid_stats.get("entries", 0),
                        "cache_hit_rate": hybrid_stats.get("hit_rate", 0.0),
                        "memory_usage_mb": hybrid_stats.get("memory_usage_mb", 0.0)
                    })
                except Exception as e:
                    logger.debug(f"Не удалось получить статистику гибридного кэша: {e}")
            
        except Exception as e:
            logger.debug(f"Ошибка при сборе статистики кэша: {e}")
        
        return cache_stats
    
    def _get_relation_type_statistics(self) -> Dict[str, int]:
        """Возвращает статистику по типам связей."""
        stats = {}
        for edge in self.edges.values():
            if edge.relation_type in stats:
                stats[edge.relation_type] += 1
            else:
                stats[edge.relation_type] = 1
        return stats
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Возвращает состояние системы графа знаний.
        
        Returns:
            Dict[str, Any]: Состояние системы
        """
        health = {
            "score": 0.0,
            "status": "critical",
            "components": {
                "nodes": {"status": "offline", "score": 0.0, "count": 0},
                "edges": {"status": "offline", "score": 0.0, "count": 0},
                "integrity": {"status": "offline", "score": 0.0},
                "resources": {"status": "offline", "score": 0.0}
            },
            "stats": self.stats.copy()
        }
        
        try:
            # Проверяем узлы
            node_score = 0.0
            if self.stats["total_nodes"] > 0:
                node_score = min(1.0, self.stats["total_nodes"] / 1000)  # Нормализуем к 0-1
                health["components"]["nodes"] = {
                    "status": "online" if node_score > 0.3 else "warning",
                    "score": node_score,
                    "count": self.stats["total_nodes"]
                }
            
            # Проверяем связи
            edge_score = 0.0
            if self.stats["total_edges"] > 0:
                edge_score = min(1.0, self.stats["total_edges"] / 5000)  # Нормализуем к 0-1
                health["components"]["edges"] = {
                    "status": "online" if edge_score > 0.3 else "warning",
                    "score": edge_score,
                    "count": self.stats["total_edges"]
                }
            
            # Проверяем целостность
            integrity_score = self._check_graph_integrity_score()
            health["components"]["integrity"] = {
                "status": "online" if integrity_score > 0.7 else "warning" if integrity_score > 0.4 else "critical",
                "score": integrity_score
            }
            
            # Проверяем ресурсы
            resource_score = self._check_resource_usage()
            health["components"]["resources"] = {
                "status": "online" if resource_score > 0.7 else "warning" if resource_score > 0.4 else "critical",
                "score": resource_score
            }
            
            # Вычисляем общий балл
            total_score = (node_score * 0.25 + 
                          edge_score * 0.25 + 
                          integrity_score * 0.3 + 
                          resource_score * 0.2)
            
            health["score"] = total_score
            
            # Определяем статус
            if total_score > 0.7:
                health["status"] = "healthy"
            elif total_score > 0.4:
                health["status"] = "warning"
            else:
                health["status"] = "critical"
                
            return health
            
        except Exception as e:
            logger.error(f"Ошибка проверки состояния системы: {e}", exc_info=True)
            return {
                "score": 0.0,
                "status": "critical",
                "error": str(e),
                "components": {},
                "stats": self.stats.copy()
            }
    
    def _check_graph_integrity_score(self) -> float:
        """Проверяет целостность графа и возвращает оценку."""
        try:
            # Проверяем отсутствующие узлы в связях
            missing_nodes = 0
            for edge in self.edges.values():
                if edge.source_id not in self.nodes:
                    missing_nodes += 1
                if edge.target_id not in self.nodes:
                    missing_nodes += 1
            
            # Проверяем дубликаты
            name_counts = defaultdict(int)
            for node in self.nodes.values():
                name_counts[node.name.lower()] += 1
            
            duplicate_count = sum(count - 1 for count in name_counts.values() if count > 1)
            
            # Рассчитываем оценку
            total_issues = missing_nodes + duplicate_count
            # Чем меньше проблем, тем лучше (максимум 1.0)
            return max(0.1, 1.0 - (total_issues / max(1, len(self.edges) * 2)))
            
        except Exception as e:
            logger.error(f"Ошибка проверки целостности графа: {e}", exc_info=True)
            return 0.5
    
    def _check_resource_usage(self) -> float:
        """Проверяет использование ресурсов и возвращает оценку."""
        try:
            import psutil
            
            # Проверяем использование памяти
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            
            # Проверяем использование диска
            disk = psutil.disk_usage(self.cache_dir)
            disk_usage = disk.percent / 100.0
            
            # Чем меньше использование, тем лучше
            resource_score = (1.0 - memory_usage) * 0.6 + (1.0 - disk_usage) * 0.4
            
            return max(0.1, min(1.0, resource_score))
            
        except Exception as e:
            logger.error(f"Ошибка проверки использования ресурсов: {e}", exc_info=True)
            return 0.7
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Возвращает метрики производительности.
        
        Returns:
            Dict[str, Any]: Метрики производительности
        """
        metrics = {
            "request_rate": self.stats["successful_queries"] / max(1, time.time() - self.stats.get("start_time", time.time())),
            "success_rate": self.stats["successful_queries"] / max(1, self.stats["total_queries"]),
            "avg_processing_time": self.stats["total_processing_time"] / max(1, self.stats["successful_queries"]),
            "system_health": self.get_system_health(),
            "node_statistics": self._get_node_type_statistics(),
            "edge_statistics": self._get_relation_type_statistics()
        }
        
        return metrics
    
    def get_detailed_health_report(self) -> Dict[str, Any]:
        """
        Возвращает детальный отчет о состоянии системы.
        
        Returns:
            Dict[str, Any]: Детальный отчет
        """
        report = {
            "timestamp": time.time(),
            "system_health": self.get_system_health(),
            "performance_metrics": self.get_performance_metrics(),
            "statistics": self.stats.copy(),
            "domain_statistics": self._get_domain_statistics(),
            "node_type_statistics": self._get_node_type_statistics(),
            "relation_type_statistics": self._get_relation_type_statistics(),
            "graph_integrity": self._get_graph_integrity_report()
        }
        
        return report
    
    def _get_graph_integrity_report(self) -> Dict[str, Any]:
        """Возвращает отчет о целостности графа."""
        try:
            # Проверяем отсутствующие узлы в связях
            missing_nodes = []
            for edge_id, edge in self.edges.items():
                if edge.source_id not in self.nodes:
                    missing_nodes.append((edge_id, "source", edge.source_id))
                if edge.target_id not in self.nodes:
                    missing_nodes.append((edge_id, "target", edge.target_id))
            
            # Проверяем дубликаты
            name_counts = defaultdict(list)
            for node_id, node in self.nodes.items():
                name_counts[node.name.lower()].append(node_id)
            
            duplicates = {name: ids for name, ids in name_counts.items() if len(ids) > 1}
            
            return {
                "missing_nodes_in_edges": len(missing_nodes),
                "missing_node_details": missing_nodes[:10],  # Ограничиваем детали
                "duplicate_nodes": len(duplicates),
                "duplicate_details": {name: ids[:3] for name, ids in list(duplicates.items())[:5]},  # Ограничиваем детали
                "total_issues": len(missing_nodes) + len(duplicates)
            }
            
        except Exception as e:
            logger.error(f"Ошибка формирования отчета о целостности: {e}", exc_info=True)
            return {"error": str(e)}
    
    def start(self):
        """Запускает фоновые процессы графа знаний."""
        if self.running:
            return
            
        self.stop_event.clear()
        self.running = True
        logger.info("KnowledgeGraph запущен")
    
    def stop(self):
        """Останавливает фоновые процессы графа знаний."""
        if not self.running:
            return
            
        self.stop_event.set()
        self.running = False
        
        # Дожидаемся завершения фоновых потоков
        if hasattr(self, 'monitoring_thread') and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        
        if hasattr(self, 'optimization_thread') and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=2.0)
        
        # Закрываем соединение с БД
        try:
            conn = sqlite3.connect(self.db_path)
            conn.close()
        except Exception as e:
            logger.error(f"Ошибка закрытия соединения с БД: {e}")
        
        # Закрываем пул потоков
        self.executor.shutdown(wait=True)
        
        logger.info("KnowledgeGraph остановлен")
    
    def close(self):
        """Закрывает граф знаний и освобождает ресурсы."""
        self.stop()
        
        # Освобождаем ресурсы
        self.nodes = {}
        self.edges = {}
        
        logger.info("KnowledgeGraph закрыт")
    
    def is_initialized(self) -> bool:
        """Проверяет, инициализирован ли граф знаний."""
        return self.initialized
    
    def is_running(self) -> bool:
        """Проверяет, запущен ли граф знаний."""
        return self.running
    
    def get_node_details(self, node_id: str, max_distance: int = 2) -> Dict[str, Any]:
        """
        Возвращает подробную информацию об узле и его связях.
        
        Args:
            node_id: ID узла
            max_distance: Максимальное расстояние для получения связей
            
        Returns:
            Dict[str, Any]: Подробная информация об узле
        """
        node = self.get_node(node_id)
        if not node:
            return {"error": "node_not_found"}
        
        # Получаем связанные узлы
        related_nodes = self.get_related_nodes(node_id, max_distance=max_distance)
        
        # Получаем связи
        edges = self.get_edges(node_id, direction="both")
        
        return {
            "node": node.to_dict(),
            "related_nodes": [n.to_dict() for n in related_nodes],
            "edges": [e.to_dict() for e in edges],
            "contradictions": node.contradictions,
            "history": node.history[-5:]  # Последние 5 изменений
        }
    
    def get_domain_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику по доменам.
        
        Returns:
            Dict[str, Any]: Статистика по доменам
        """
        stats = self._get_domain_statistics()
        
        # Добавляем дополнительную информацию
        detailed_stats = {}
        for domain, count in stats.items():
            # Получаем типы узлов в этом домене
            node_types = defaultdict(int)
            for node in self.nodes.values():
                if node.domain == domain:
                    node_types[node.node_type] += 1
            
            detailed_stats[domain] = {
                "total_nodes": count,
                "node_types": dict(node_types),
                "recent_updates": self._get_recent_updates_for_domain(domain, 10)
            }
        
        return detailed_stats
    
    def _get_recent_updates_for_domain(self, domain: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Возвращает недавние обновления для домена.
        
        Args:
            domain: Домен
            limit: Лимит результатов
            
        Returns:
            List[Dict[str, Any]]: Список обновлений
        """
        updates = []
        
        # Используем временной индекс для быстрого поиска
        for timestamp, item_id, item_type in reversed(self.temporal_index):
            if len(updates) >= limit:
                break
            
            if item_type == "node":
                node = self.nodes.get(item_id)
                if node and node.domain == domain and node.last_updated != node.timestamp:
                    updates.append({
                        "id": item_id,
                        "type": "node",
                        "name": node.name,
                        "last_updated": node.last_updated,
                        "changes": len(node.history)
                    })
            elif item_type == "edge":
                edge = self.edges.get(item_id)
        last_edge = None
        for edge in self.edges.values():
            last_edge = edge
        
        if last_edge is not None:
                    # Проверяем, относится ли связь к домену через узлы
                    source_node = self.nodes.get(edge.source_id)
                    target_node = self.nodes.get(edge.target_id)
                    
                    if (source_node and source_node.domain == domain) or (target_node and target_node.domain == domain):
                        updates.append({
                            "id": item_id,
                            "type": "edge",
                            "relation": edge.relation_type,
                            "last_updated": edge.last_updated,
                            "changes": len(edge.history)
                        })
        
        return updates
    
    def find_path(self, start_node_id: str, end_node_id: str, 
                 max_length: int = 5) -> List[List[str]]:
        """
        Находит пути между двумя узлами в графе.
        
        Args:
            start_node_id: ID начального узла
            end_node_id: ID конечного узла
            max_length: Максимальная длина пути
            
        Returns:
            List[List[str]]: Список найденных путей
        """
        if start_node_id not in self.nodes or end_node_id not in self.nodes:
            return []
        
        # Используем BFS для поиска путей
        paths = []
        queue = deque([(start_node_id, [start_node_id])])
        visited = set([start_node_id])
        
        while queue:
            current_node, path = queue.popleft()
            
            if current_node == end_node_id:
                paths.append(path)
                if len(paths) >= 10:  # Ограничиваем количество найденных путей
                    break
                continue
            
            if len(path) >= max_length:
                continue
            
            # Получаем все связанные узлы
            related_nodes = self.get_related_nodes(current_node)
            for node in related_nodes:
                if node.id not in visited:
                    visited.add(node.id)
                    queue.append((node.id, path + [node.id]))
        
        return paths
    
    def get_temporal_relations(self, start_time: Optional[float] = None, 
                              end_time: Optional[float] = None,
                              limit: int = 20) -> List[Dict[str, Any]]:
        """
        Возвращает временные отношения в указанном временном интервале.
        
        Args:
            start_time: Начальное время
            end_time: Конечное время
            limit: Лимит результатов
            
        Returns:
            List[Dict[str, Any]]: Список временных отношений
        """
        start_time = start_time or 0
        end_time = end_time or time.time()
        
        results = []
        
        # Используем временной индекс для быстрого поиска
        for timestamp, item_id, item_type in self.temporal_index:
            if timestamp < start_time:
                continue
            if timestamp > end_time:
                break
            
            if item_type == "node":
                node = self.nodes.get(item_id)
                if node and node.temporal_info:
                    results.append({
                        "type": "node",
                        "id": item_id,
                        "name": node.name,
                        "timestamp": timestamp,
                        "temporal_info": node.temporal_info,
                        "domain": node.domain
                    })
            elif item_type == "edge":
                edge = self.edges.get(item_id)
                if edge and edge.temporal_info:
                    results.append({
                        "type": "edge",
                        "id": item_id,
                        "relation_type": edge.relation_type,
                        "timestamp": timestamp,
                        "temporal_info": edge.temporal_info
                    })
            
            if len(results) >= limit:
                break
        
        return results
    
    def get_spatial_relations(self, location: Dict[str, float], 
                             max_distance: float = 100.0,
                             limit: int = 20) -> List[Dict[str, Any]]:
        """
        Возвращает пространственные отношения в указанном радиусе.
        
        Args:
            location: Координаты местоположения
            max_distance: Максимальное расстояние
            limit: Лимит результатов
            
        Returns:
            List[Dict[str, Any]]: Список пространственных отношений
        """
        results = []
        
        for node in self.nodes.values():
            if not node.spatial_info or "coordinates" not in node.spatial_info:
                continue
            
            # Вычисляем расстояние (упрощенный расчет)
            node_coords = node.spatial_info["coordinates"]
            distance = self._calculate_distance(
                location["lat"], location["lon"],
                node_coords["lat"], node_coords["lon"]
            )
            
            if distance <= max_distance:
                results.append({
                    "type": "node",
                    "id": node.id,
                    "name": node.name,
                    "distance": distance,
                    "spatial_info": node.spatial_info,
                    "domain": node.domain
                })
        
        for edge in self.edges.values():
            if not edge.spatial_info or "coordinates" not in edge.spatial_info:
                continue
            
            # Вычисляем расстояние
            edge_coords = edge.spatial_info["coordinates"]
            distance = self._calculate_distance(
                location["lat"], location["lon"],
                edge_coords["lat"], edge_coords["lon"]
            )
            
            if distance <= max_distance:
                results.append({
                    "type": "edge",
                    "id": edge.id,
                    "relation_type": edge.relation_type,
                    "distance": distance,
                    "spatial_info": edge.spatial_info
                })
        
        # Сортируем по расстоянию
        results.sort(key=lambda x: x["distance"])
        
        return results[:limit]
    
    def _calculate_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """
        Вычисляет расстояние между двумя точками на сфере.
        
        Args:
            lat1: Широта первой точки
            lon1: Долгота первой точки
            lat2: Широта второй точки
            lon2: Долгота второй точки
            
        Returns:
            float: Расстояние в километрах
        """
        # Упрощенный расчет (для небольших расстояний)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (dlat ** 2) + math.cos(lat1) * (dlon ** 2)
        distance = 6371 * math.sqrt(a)  # 6371 - радиус Земли в км
        return distance
    
    def generate_knowledge_graph(self, concept: str, depth: int = 2) -> Dict[str, Any]:
        """
        Генерирует граф знаний для визуализации.
        
        Args:
            concept: Концепт для построения графа
            depth: Глубина графа
            
        Returns:
            Dict[str, Any]: Граф знаний в формате для визуализации
        """
        # Ищем узел по концепту
        nodes = self.search_nodes(concept, limit=1)
        if not nodes:
            return {"nodes": [], "edges": []}
        
        center_node = nodes[0]
        
        # Получаем подграф
        subgraph = self.get_subgraph(center_node.id, depth)
        
        # Преобразуем для визуализации
        viz_nodes = []
        for node in subgraph["nodes"]:
            viz_nodes.append({
                "id": node["id"],
                "label": node["name"],
                "title": node["description"],
                "group": node["domain"],
                "shape": "ellipse",
                "font": {"size": 14}
            })
        
        viz_edges = []
        for edge in subgraph["edges"]:
            viz_edges.append({
                "from": edge["source_id"],
                "to": edge["target_id"],
                "label": edge["relation_type"],
                "arrows": "to",
                "font": {"align": "middle"}
            })
        
        return {
            "nodes": viz_nodes,
            "edges": viz_edges
        }
    
    def export_graph(self, format: str = "json") -> Any:
        """
        Экспортирует граф знаний в указанный формат.
        
        Args:
            format: Формат экспорта (json, graphml, gexf)
            
        Returns:
            Any: Экспортированные данные
        """
        if format == "json":
            return {
                "nodes": [node.to_dict() for node in self.nodes.values()],
                "edges": [edge.to_dict() for edge in self.edges.values()]
            }
        elif format == "graphml":
            # Реализация экспорта в GraphML
            return self._export_to_graphml()
        elif format == "gexf":
            # Реализация экспорта в GEXF
            return self._export_to_gexf()
        else:
            raise ValueError(f"Неподдерживаемый формат экспорта: {format}")
    
    def _export_to_graphml(self) -> str:
        """Экспортирует граф в формат GraphML."""
        from xml.etree.ElementTree import Element, SubElement, tostring
        import xml.dom.minidom
        
        # Создаем XML структуру
        graphml = Element('graphml', {
            'xmlns': 'http://graphml.graphdrawing.org/xmlns',
            'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
            'xsi:schemaLocation': 'http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd'
        })
        
        # Определяем ключи
        key_node = SubElement(graphml, 'key', {
            'id': 'd0',
            'for': 'node',
            'attr.name': 'description',
            'attr.type': 'string'
        })
        SubElement(key_node, 'default').text = ''
        
        key_domain = SubElement(graphml, 'key', {
            'id': 'd1',
            'for': 'node',
            'attr.name': 'domain',
            'attr.type': 'string'
        })
        SubElement(key_domain, 'default').text = 'general'
        
        key_strength = SubElement(graphml, 'key', {
            'id': 'd2',
            'for': 'node',
            'attr.name': 'strength',
            'attr.type': 'double'
        })
        SubElement(key_strength, 'default').text = '0.5'
        
        key_relation = SubElement(graphml, 'key', {
            'id': 'd3',
            'for': 'edge',
            'attr.name': 'relation_type',
            'attr.type': 'string'
        })
        SubElement(key_relation, 'default').text = 'related_to'
        
        # Создаем граф
        graph = SubElement(graphml, 'graph', {
            'id': 'G',
            'edgedefault': 'directed'
        })
        
        # Добавляем узлы
        for node in self.nodes.values():
            node_elem = SubElement(graph, 'node', {'id': node.id})
            SubElement(node_elem, 'data', {'key': 'd0'}).text = node.description
            SubElement(node_elem, 'data', {'key': 'd1'}).text = node.domain
            SubElement(node_elem, 'data', {'key': 'd2'}).text = str(node.strength)
        
        # Добавляем связи
        for edge in self.edges.values():
            edge_elem = SubElement(graph, 'edge', {
                'id': edge.id,
                'source': edge.source_id,
                'target': edge.target_id
            })
            SubElement(edge_elem, 'data', {'key': 'd3'}).text = edge.relation_type
        
        # Форматируем XML
        xml_str = tostring(graphml, encoding='utf-8')
        dom = xml.dom.minidom.parseString(xml_str)
        return dom.toprettyxml()
    
    def _export_to_gexf(self) -> str:
        """Экспортирует граф в формат GEXF."""
        from xml.etree.ElementTree import Element, SubElement, tostring
        import xml.dom.minidom
        
        # Создаем XML структуру
        gexf = Element('gexf', {
            'xmlns': 'http://www.gexf.net/1.2draft',
            'version': '1.2'
        })
        
        # Метаданные
        meta = SubElement(gexf, 'meta', {'lastmodified': datetime.now().isoformat()})
        SubElement(meta, 'creator').text = 'CogniFlex Knowledge Graph'
        SubElement(meta, 'description').text = 'Graph of knowledge exported from CogniFlex'
        
        # Визуализация
        SubElement(gexf, 'visualization')
        
        # Граф
        graph = SubElement(gexf, 'graph', {
            ' defaultedgetype': 'directed',
            'mode': 'static'
        })
        
        # Атрибуты узлов
        node_attributes = SubElement(graph, 'attributes', {'class': 'node'})
        SubElement(node_attributes, 'attribute', {
            'id': 'description',
            'title': 'Description',
            'type': 'string'
        })
        SubElement(node_attributes, 'attribute', {
            'id': 'domain',
            'title': 'Domain',
            'type': 'string'
        })
        SubElement(node_attributes, 'attribute', {
            'id': 'strength',
            'title': 'Strength',
            'type': 'float'
        })
        
        # Атрибуты связей
        edge_attributes = SubElement(graph, 'attributes', {'class': 'edge'})
        SubElement(edge_attributes, 'attribute', {
            'id': 'relation_type',
            'title': 'Relation Type',
            'type': 'string'
        })
        
        # Узлы
        nodes_elem = SubElement(graph, 'nodes')
        for node in self.nodes.values():
            node_elem = SubElement(nodes_elem, 'node', {
                'id': node.id,
                'label': node.name
            })
            attvalues = SubElement(node_elem, 'attvalues')
            SubElement(attvalues, 'attvalue', {
                'for': 'description',
                'value': node.description
            })
            SubElement(attvalues, 'attvalue', {
                'for': 'domain',
                'value': node.domain
            })
            SubElement(attvalues, 'attvalue', {
                'for': 'strength',
                'value': str(node.strength)
            })
        
        # Связи
        edges_elem = SubElement(graph, 'edges')
        for i, edge in enumerate(self.edges.values()):
            edge_elem = SubElement(edges_elem, 'edge', {
                'id': str(i),
                'source': edge.source_id,
                'target': edge.target_id
            })
            attvalues = SubElement(edge_elem, 'attvalues')
            SubElement(attvalues, 'attvalue', {
                'for': 'relation_type',
                'value': edge.relation_type
            })
        
        # Форматируем XML
        xml_str = tostring(gexf, encoding='utf-8')
        dom = xml.dom.minidom.parseString(xml_str)
        return dom.toprettyxml()
    
    def import_graph(self, data: Any, format: str = "json") -> bool:
        """
        Импортирует граф знаний из указанного формата.
        
        Args:
            data: Данные для импорта
            format: Формат данных (json, graphml, gexf)
            
        Returns:
            bool: Успешно ли импортировано
        """
        try:
            if format == "json":
                return self._import_from_json(data)
            elif format == "graphml":
                return self._import_from_graphml(data)
            elif format == "gexf":
                return self._import_from_gexf(data)
            else:
                raise ValueError(f"Неподдерживаемый формат импорта: {format}")
        except Exception as e:
            logger.error(f"Ошибка импорта графа: {e}", exc_info=True)
            return False
    
    def _import_from_json(self, data: Dict[str, Any]) -> bool:
        """Импортирует граф из JSON."""
        # Очищаем существующий граф (опционально)
        # self.nodes = {}
        # self.edges = {}
        
        # Импортируем узлы
        for node_data in data["nodes"]:
            node = KnowledgeNode.from_dict(node_data)
            self.nodes[node.id] = node
            self._save_node_to_db(node)
        
        # Импортируем связи
        for edge_data in data["edges"]:
            edge = KnowledgeEdge.from_dict(edge_data)
            self.edges[edge.id] = edge
            self._save_edge_to_db(edge)
        
        # Обновляем индексы
        self._init_indexes()
        
        logger.info(f"Импортировано {len(data['nodes'])} узлов и {len(data['edges'])} связей")
        return True
    
    def _import_from_graphml(self, data: str) -> bool:
        """Импортирует граф из GraphML."""
        import xml.etree.ElementTree as ET
        
        # Парсим XML
        root = ET.fromstring(data)
        namespace = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
        
        # Находим граф
        graph = root.find('graphml:graph', namespace)
        if graph is None:
            raise ValueError("Не найден элемент графа в GraphML")
        
        # Импортируем узлы
        for node in graph.findall('graphml:node', namespace):
            node_id = node.get('id')
            
            # Получаем данные
            description = ""
            domain = "general"
            strength = 0.5
            
            for data_elem in node.findall('graphml:data', namespace):
                key = data_elem.get('key')
                if key == 'd0':
                    description = data_elem.text or ""
                elif key == 'd1':
                    domain = data_elem.text or "general"
                elif key == 'd2':
                    try:
                        strength = float(data_elem.text or "0.5")
                    except ValueError:
                        logger.warning(f"Не удалось преобразовать значение в число: {data_elem.text}, установлено значение по умолчанию 0.5")
                        strength = 0.5
            
            # Создаем узел (упрощенно)
            name = node_id  # В реальном случае нужно извлекать из данных
            self.add_node(name, description, domain=domain, strength=strength)
        
        # Импортируем связи
        for edge in graph.findall('graphml:edge', namespace):
            source_id = edge.get('source')
            target_id = edge.get('target')
            edge_id = edge.get('id', f"edge_{hash(edge)}")
            
            # Получаем тип связи
            relation_type = "related_to"
            for data_elem in edge.findall('graphml:data', namespace):
                if data_elem.get('key') == 'd3':
                    relation_type = data_elem.text or "related_to"
            
            # Добавляем связь
            self.add_edge(source_id, target_id, relation_type)
        
        logger.info("Граф успешно импортирован из GraphML")
        return True
    
    def _import_from_gexf(self, data: str) -> bool:
        """Импортирует граф из GEXF."""
        import xml.etree.ElementTree as ET
        
        # Парсим XML
        root = ET.fromstring(data)
        namespace = {'gexf': 'http://www.gexf.net/1.2draft'}
        
        # Находим граф
        graph = root.find('gexf:graph', namespace)
        if graph is None:
            raise ValueError("Не найден элемент графа в GEXF")
        
        # Импортируем узлы
        nodes = graph.find('gexf:nodes', namespace)
        if nodes is not None:
            for node in nodes.findall('gexf:node', namespace):
                node_id = node.get('id')
                label = node.get('label', node_id)
                
                # Получаем атрибуты
                description = ""
                domain = "general"
                strength = 0.5
                
                attvalues = node.find('gexf:attvalues', namespace)
                if attvalues is not None:
                    for attvalue in attvalues.findall('gexf:attvalue', namespace):
                        for_attr = attvalue.get('for')
                        value = attvalue.get('value', '')
                        
                        if for_attr == 'description':
                            description = value
                        elif for_attr == 'domain':
                            domain = value
                        elif for_attr == 'strength':
                            try:
                                strength = float(value)
                            except ValueError:
                                logger.warning(f"Не удалось преобразовать значение в число: {value}, установлено значение по умолчанию 0.5")
                                strength = 0.5
                
                # Создаем узел
                self.add_node(label, description, domain=domain, strength=strength)
        
        # Импортируем связи
        edges = graph.find('gexf:edges', namespace)
        if edges is not None:
            for edge in edges.findall('gexf:edge', namespace):
                source_id = edge.get('source')
                target_id = edge.get('target')
                edge_id = edge.get('id', f"edge_{hash(edge)}")
                
                # Получаем тип связи
                relation_type = "related_to"
                attvalues = edge.find('gexf:attvalues', namespace)
                if attvalues is not None:
                    for attvalue in attvalues.findall('gexf:attvalue', namespace):
                        if attvalue.get('for') == 'relation_type':
                            relation_type = attvalue.get('value', 'related_to')
                
                # Добавляем связь
                self.add_edge(source_id, target_id, relation_type)
        
        logger.info("Граф успешно импортирован из GEXF")
        return True
    
    def analyze_knowledge_gaps(self, domain: str, num_samples: int = 10) -> List[Dict[str, Any]]:
        """
        Анализирует пробелы в знаниях в указанной области.
        
        Args:
            domain: Область знаний
            num_samples: Количество примеров для анализа
            
        Returns:
            List[Dict[str, Any]]: Выявленные пробелы в знаниях
        """
        if not domain:
            return []
        
        try:
            if not self.initialized:
                return []
            
            if not self.brain or not hasattr(self.brain, 'knowledge_graph'):
                logger.warning("KnowledgeGraph недоступен для анализа пробелов в знаниях")
                return []
            
            # Получаем узлы из области знаний
            nodes = self.search_nodes("", domain=domain, limit=num_samples)
            gaps = []
            
            for node in nodes:
                # Анализируем связи узла
                node_id = getattr(node, 'id', '')
                edges = self.get_edges(node_id)
                
                # Если мало связей, это может быть пробелом
                if len(edges) < 3:
                    content = getattr(node, 'content', str(node))
                    gaps.append({
                        "concept": content,
                        "gap_type": "incomplete",
                        "severity": 0.7,
                        "evidence": [f"Концепт '{content}' имеет мало связей ({len(edges)})"],
                        "suggested_actions": [
                            f"Добавить связи для концепта '{content}'",
                            "Расширить информацию о концепте"
                        ]
                    })
                
                # Проверяем актуальность
                if hasattr(node, 'last_updated'):
                    time_since_update = time.time() - node.last_updated
                    if time_since_update > 365 * 86400:  # Больше года
                        gaps.append({
                            "concept": node.name,
                            "gap_type": "outdated",
                            "severity": 0.6,
                            "evidence": [f"Информация не обновлялась более года"],
                            "suggested_actions": [
                                "Проверить актуальность информации",
                                "Найти новые источники данных"
                            ]
                        })
            
            # Проверяем, есть ли вообще узлы в домене
            if not nodes:
                gaps.append({
                    "concept": domain,
                    "gap_type": "missing",
                    "severity": 0.9,
                    "evidence": [f"Нет узлов в домене '{domain}'"],
                    "suggested_actions": [
                        f"Добавить базовые концепты для домена '{domain}'",
                        "Импортировать знания из внешних источников"
                    ]
                })
            
            logger.debug(f"Выявлено {len(gaps)} пробелов в знаниях для домена '{domain}'")
            return gaps
            
        except Exception as e:
            logger.error(f"Ошибка анализа пробелов в знаниях: {e}", exc_info=True)
            return []
    
    def resolve_contradictions(self, max_contradictions: int = 5) -> List[Dict[str, Any]]:
        """
        Пытается разрешить противоречия в графе знаний.
        
        Args:
            max_contradictions: Максимальное количество противоречий для разрешения
            
        Returns:
            List[Dict[str, Any]]: Разрешенные противоречия
        """
        resolved = []
        
        try:
            # Получаем узлы с противоречиями
            contradictory_nodes = [
                node for node in self.nodes.values() 
                if node.contradictions and not all(c["resolved"] for c in node.contradictions)
            ]
            
            # Ограничиваем количество
            contradictory_nodes = contradictory_nodes[:max_contradictions]
            
            for node in contradictory_nodes:
                for contradiction in node.contradictions:
                    if contradiction["resolved"]:
                        continue
                    
                    try:
                        # Пытаемся найти решение с помощью ML
                        resolution = self._find_contradiction_resolution(
                            node.id, 
                            contradiction["node_id"],
                            contradiction["evidence"]
                        )
                        
                        if resolution:
                            # Применяем решение
                            self._apply_contradiction_resolution(
                                node.id,
                                contradiction["node_id"],
                                resolution
                            )
                            
                            resolved.append({
                                "node_id": node.id,
                                "contradictory_node_id": contradiction["node_id"],
                                "resolution": resolution,
                                "evidence": contradiction["evidence"]
                            })
                            
                            logger.info(f"Разрешено противоречие между {node.id} и {contradiction['node_id']}")
                        else:
                            logger.debug(f"Не найдено решение для противоречия {node.id} и {contradiction['node_id']}")
                    
                    except Exception as e:
                        logger.error(f"Ошибка разрешения противоречия: {e}", exc_info=True)
            
            return resolved
            
        except Exception as e:
            logger.error(f"Ошибка разрешения противоречий: {e}", exc_info=True)
            return []
    
    def _find_contradiction_resolution(self, node_id: str, contradictory_node_id: str, 
                                     evidence: str) -> Optional[str]:
        """
        Пытается найти решение для противоречия с помощью ML.
        
        Args:
            node_id: ID узла
            contradictory_node_id: ID противоречивого узла
            evidence: Доказательства противоречия
            
        Returns:
            Optional[str]: Предложенное решение или None
        """
        if not self.ml_unit:
            return None
        
        try:
            # Получаем узлы
            node = self.get_node(node_id)
            contradictory_node = self.get_node(contradictory_node_id)
            
            if not node or not contradictory_node:
                return None
            
            # Формируем промпт для ML
            prompt = (
                f"В системе знаний обнаружено противоречие между двумя концептами:\n\n"
                f"Концепт 1: {node.name}\n"
                f"Описание: {node.description}\n\n"
                f"Концепт 2: {contradictory_node.name}\n"
                f"Описание: {contradictory_node.description}\n\n"
                f"Доказательства противоречия: {evidence}\n\n"
                f"Проанализируйте информацию и предложите решение противоречия. "
                f"Решение должно быть кратким (1-2 предложения) и содержать конкретные рекомендации."
            )
            
            # Генерируем ответ
            response = self.ml_unit.generate_response(
                prompt=prompt,
                max_length=200,
                temperature=0.3,
                top_p=0.9,
                task="text-generation"
            )
            
            resolution = response.get("text", "").strip()
            
            if resolution and len(resolution) > 10:
                return resolution
        
        except Exception as e:
            logger.error(f"Ошибка генерации решения противоречия: {e}", exc_info=True)
        
        return None
    
    def _apply_contradiction_resolution(self, node_id: str, contradictory_node_id: str, 
                                      resolution: str):
        """
        Применяет решение противоречия.
        
        Args:
            node_id: ID узла
            contradictory_node_id: ID противоречивого узла
            resolution: Решение
        """
        # Обновляем узлы
        if node_id in self.nodes:
            self.nodes[node_id].add_contradiction(
                contradictory_node_id, 
                "Resolved", 
                resolution=resolution
            )
            self._update_node_in_db(self.nodes[node_id])
        
        if contradictory_node_id in self.nodes:
            self.nodes[contradictory_node_id].add_contradiction(
                node_id, 
                "Resolved", 
                resolution=resolution
            )
            self._update_node_in_db(self.nodes[contradictory_node_id])
        
        # Записываем в историю
        self._record_history(
            "contradiction_resolved",
            {
                "node_id": node_id,
                "contradictory_node_id": contradictory_node_id,
                "resolution": resolution
            }
        )
    
    def update_from_external_source(self, source_name: str, data: Any, 
                                  source_type: str = "web", 
                                  update_policy: str = "merge") -> Dict[str, Any]:
        """
        Обновляет граф знаний из внешнего источника.
        
        Args:
            source_name: Название источника
            data: Данные от источника
            source_type: Тип источника
            update_policy: Политика обновления (merge, replace, incremental)
            
        Returns:
            Dict[str, Any]: Результат обновления
        """
        start_time = time.time()
        results = {
            "new_nodes": 0,
            "updated_nodes": 0,
            "new_edges": 0,
            "updated_edges": 0,
            "source": source_name,
            "source_type": source_type,
            "timestamp": time.time(),
            "processing_time": 0.0
        }
        
        try:
            # В зависимости от типа источника обрабатываем данные
            if source_type == "web":
                processed_data = self._process_web_data(data)
            elif source_type == "api":
                processed_data = self._process_api_data(data)
            elif source_type == "file":
                processed_data = self._process_file_data(data)
            else:
                raise ValueError(f"Неизвестный тип источника: {source_type}")
            
            # Применяем политику обновления
            if update_policy == "merge":
                self._merge_data(processed_data, source_name, results)
            elif update_policy == "replace":
                self._replace_data(processed_data, source_name, results)
            elif update_policy == "incremental":
                self._incremental_update(processed_data, source_name, results)
            else:
                raise ValueError(f"Неизвестная политика обновления: {update_policy}")
            
            # Проверяем противоречия после обновления
            self._check_for_contradictions()
            
            # Обновляем статистику
            processing_time = time.time() - start_time
            results["processing_time"] = processing_time
            self._update_statistics(start_time, True)
            
            logger.info(f"Граф знаний обновлен из источника {source_name}. "
                       f"Новые узлы: {results['new_nodes']}, "
                       f"Обновленные узлы: {results['updated_nodes']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка обновления из внешнего источника: {e}", exc_info=True)
            self._update_statistics(start_time, False)
            results["error"] = str(e)
            return results
    
    def _process_web_data(self, data: Any) -> Dict[str, Any]:
        """Обрабатывает данные из веб-источника."""
        # Здесь будет реализация обработки веб-данных
        # Например, извлечение сущностей и отношений из текста
        if not self.text_processor:
            return {"nodes": [], "edges": []}
        
        try:
            # Анализируем текст
            analysis = self.text_processor.process_text(data)
            
            # Извлекаем концепты и отношения
            nodes = []
            for concept in analysis.concepts:
                nodes.append({
                    "name": concept["concept"],
                    "description": "",
                    "node_type": concept["type"],
                    "domain": "general",
                    "strength": concept["relevance"]
                })
            
            # Извлекаем ключевые слова как концепты
            for keyword in analysis.keywords[:10]:
                nodes.append({
                    "name": keyword["word"],
                    "description": "",
                    "node_type": "concept",
                    "domain": "general",
                    "strength": keyword["score"]
                })
            
            # Создаем связи между концептами
            edges = []
            if len(nodes) > 1:
                for i in range(len(nodes) - 1):
                    edges.append({
                        "source": nodes[i]["name"],
                        "target": nodes[i+1]["name"],
                        "relation_type": "related_to",
                        "strength": 0.7
                    })
            
            return {"nodes": nodes, "edges": edges}
            
        except Exception as e:
            logger.error(f"Ошибка обработки веб-данных: {e}", exc_info=True)
            return {"nodes": [], "edges": []}
    
    def _process_api_data(self, data: Any) -> Dict[str, Any]:
        """Обрабатывает данные из API."""
        # Здесь будет реализация обработки данных API
        # Например, преобразование JSON в структуру графа
        return data
    
    def _process_file_data(self, data: Any) -> Dict[str, Any]:
        """Обрабатывает данные из файла."""
        # Здесь будет реализация обработки файловых данных
        # Например, парсинг CSV или других форматов
        return data
    
    def _merge_data(self, data: Dict[str, Any], source_name: str, results: Dict[str, Any]):
        """Слияние новых данных с существующим графом."""
        # Добавляем новые узлы
        for node_data in data["nodes"]:
            # Проверяем, существует ли узел
            existing_nodes = self.search_nodes(node_data["name"], limit=1)
            
            if existing_nodes:
                # Обновляем существующий узел
                existing_node = existing_nodes[0]
                updated = False
                
                # Обновляем описание, если оно пустое или слабое
                if not existing_node.description or existing_node.strength < node_data["strength"]:
                    existing_node.description = node_data.get("description", existing_node.description)
                    existing_node.strength = max(existing_node.strength, node_data["strength"])
                    updated = True
                
                # Обновляем домен, если он не задан
                if not existing_node.domain and "domain" in node_data:
                    existing_node.domain = node_data["domain"]
                    updated = True
                
                if updated:
                    self._update_node_in_db(existing_node)
                    results["updated_nodes"] += 1
            else:
                # Добавляем новый узел
                node_id = self.add_node(
                    name=node_data["name"],
                    description=node_data.get("description", ""),
                    node_type=node_data.get("node_type", "concept"),
                    domain=node_data.get("domain", "general"),
                    strength=node_data.get("strength", 0.5),
                    source=source_name
                )
                if node_id:
                    results["new_nodes"] += 1
        
        # Добавляем новые связи
        for edge_data in data["edges"]:
            # Находим узлы по имени
            source_nodes = self.search_nodes(edge_data["source"], limit=1)
            target_nodes = self.search_nodes(edge_data["target"], limit=1)
            
            if source_nodes and target_nodes:
                source_id = source_nodes[0].id
                target_id = target_nodes[0].id
                
                # Проверяем, существует ли связь
                existing_edge = None
                for edge in self.get_edges(source_id, direction="source"):
                    if edge.target_id == target_id and edge.relation_type == edge_data["relation_type"]:
                        existing_edge = edge
                        break
                
                if existing_edge:
                    # Обновляем силу связи
                    if existing_edge.strength < edge_data["strength"]:
                        self.update_edge(
                            existing_edge.id,
                            new_strength=edge_data["strength"],
                            source=source_name
                        )
                        results["updated_edges"] += 1
                else:
                    # Добавляем новую связь
                    edge_id = self.add_edge(
                        source_id=source_id,
                        target_id=target_id,
                        relation_type=edge_data["relation_type"],
                        strength=edge_data["strength"],
                        source=source_name
                    )
                    if edge_id:
                        results["new_edges"] += 1
    
    def _replace_data(self, data: Dict[str, Any], source_name: str, results: Dict[str, Any]):
        """Замена существующих данных новыми."""
        # Удаляем все узлы и связи из этого источника
        # (в реальной системе нужно отслеживать источник для каждого узла/связи)
        
        # Добавляем новые данные
        self._merge_data(data, source_name, results)
    
    def _incremental_update(self, data: Dict[str, Any], source_name: str, results: Dict[str, Any]):
        """Инкрементальное обновление с минимальными изменениями."""
        # Анализируем изменения
        changes = self._analyze_changes(data)
        
        # Применяем только значимые изменения
        for change in changes:
            if change["significance"] > 0.3:  # Порог значимости
                if change["type"] == "node":
                    if change["operation"] == "update":
                        self.update_node(
                            change["id"],
                            new_description=change["new_value"],
                            strength=change["strength"],
                            source=source_name
                        )
                        results["updated_nodes"] += 1
                    elif change["operation"] == "add":
                        node_id = self.add_node(
                            name=change["name"],
                            description=change["description"],
                            node_type=change["node_type"],
                            domain=change["domain"],
                            strength=change["strength"],
                            source=source_name
                        )
                        if node_id:
                            results["new_nodes"] += 1
                elif change["type"] == "edge":
                    # Аналогично для связей
                    pass
    
    def _analyze_changes(self, new_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Анализирует изменения между новыми и существующими данными."""
        changes = []
        
        # Анализ узлов
        for new_node in new_data["nodes"]:
            existing_nodes = self.search_nodes(new_node["name"], limit=1)
            
            if existing_nodes:
                existing = existing_nodes[0]
                
                # Проверяем изменения в описании
                if existing.description != new_node.get("description", ""):
                    significance = self._calculate_change_significance(
                        existing.description, 
                        new_node.get("description", "")
                    )
                    changes.append({
                        "type": "node",
                        "id": existing.id,
                        "operation": "update",
                        "field": "description",
                        "old_value": existing.description,
                        "new_value": new_node.get("description", ""),
                        "significance": significance,
                        "strength": new_node.get("strength", 0.5)
                    })
            else:
                # Новый узел
                changes.append({
                    "type": "node",
                    "operation": "add",
                    "name": new_node["name"],
                    "description": new_node.get("description", ""),
                    "node_type": new_node.get("node_type", "concept"),
                    "domain": new_node.get("domain", "general"),
                    "strength": new_node.get("strength", 0.5),
                    "significance": 1.0  # Новые узлы всегда значимы
                })
        
        return changes
    
    def _calculate_change_significance(self, old_text: str, new_text: str) -> float:
        """
        Рассчитывает значимость изменения текста.
        
        Args:
            old_text: Старый текст
            new_text: Новый текст
            
        Returns:
            float: Значимость изменения (0.0-1.0)
        """
        if not self.text_processor:
            # Простая проверка, если текстовый процессор недоступен
            old_len = len(old_text)
            new_len = len(new_text)
            if old_len == 0 or new_len == 0:
                return 1.0 if old_len != new_len else 0.0
            
            common = sum(1 for a, b in zip(old_text, new_text) if a == b)
            return 1.0 - (common / max(old_len, new_len))
        
        try:
            # Используем эмбеддинги для более точной оценки
            old_embedding = self.text_processor.get_embedding(old_text)
            new_embedding = self.text_processor.get_embedding(new_text)
            
            if old_embedding is None or new_embedding is None:
                return 0.5  # Неуверенная оценка
            
            # Вычисляем косинусное сходство
            similarity = self.text_processor.calculate_similarity(old_text, new_text)
            
            # Значимость = 1 - сходство
            return max(0.0, min(1.0, 1.0 - similarity))
            
        except Exception as e:
            logger.error(f"Ошибка расчета значимости изменения: {e}", exc_info=True)
            return 0.5
    
    def get_history(self, node_id: Optional[str] = None, 
                  edge_id: Optional[str] = None,
                  days: int = 30,
                  limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Получает историю изменений за указанный период.
        
        Args:
            node_id: ID узла (опционально)
            edge_id: ID связи (опционально)
            days: Количество дней для истории
            limit: Лимит результатов
            
        Returns:
            List[Dict[str, Any]]: История изменений
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Подготавливаем запрос
            start_time = time.time() - (days * 86400)
            params = [start_time]
            
            sql = "SELECT * FROM history WHERE timestamp >= ?"
            
            # Фильтр по узлу
            if node_id:
                sql += " AND node_id = ?"
                params.append(node_id)
            
            # Фильтр по связи
            if edge_id:
                sql += " AND edge_id = ?"
                params.append(edge_id)
            
            # Сортировка
            sql += " ORDER BY timestamp DESC"
            
            # Лимит
            if isinstance(limit, int) and limit > 0:
                sql += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(sql, params)
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    "id": row[0],
                    "node_id": row[1],
                    "edge_id": row[2],
                    "change_type": row[3],
                    "changes": safe_json_loads(row[4]) if len(row) > 4 and row[4] else {},
                    "timestamp": row[5],
                    "user_id": row[6],
                    "source": row[7],
                    "version": row[8]
                })
            
            conn.close()
            return history
            
        except Exception as e:
            logger.error(f"Ошибка получения истории: {e}", exc_info=True)
            return []
    
    def get_recent_changes(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Возвращает недавние изменения в графе знаний.
        
        Args:
            limit: Лимит результатов
            
        Returns:
            List[Dict[str, Any]]: Недавние изменения
        """
        return self.get_history(days=7, limit=limit)
    
    def search_temporal(self, start_time: float, end_time: float, 
                       entity: Optional[str] = None, 
                       event_type: Optional[str] = None,
                       limit: int = 10) -> List[Dict[str, Any]]:
        """
        Выполняет временной поиск в графе знаний.
        
        Args:
            start_time: Начальное время
            end_time: Конечное время
            entity: Сущность для фильтрации
            event_type: Тип события для фильтрации
            limit: Лимит результатов
            
        Returns:
            List[Dict[str, Any]]: Результаты временного поиска
        """
        results = []
        
        # Ищем узлы с временной информацией
        for node in self.nodes.values():
            if not node.temporal_info:
                continue
            
            # Проверяем временные рамки
            node_start = node.temporal_info.get("start", 0)
            node_end = node.temporal_info.get("end", float('inf'))
            
            if (node_start <= end_time and node_end >= start_time):
                # Проверяем фильтры
                if entity and entity.lower() not in node.name.lower():
                    continue
                
                if event_type and node.temporal_info.get("type") != event_type:
                    continue
                
                results.append({
                    "type": "node",
                    "id": node.id,
                    "name": node.name,
                    "description": node.description,
                    "temporal_info": node.temporal_info,
                    "domain": node.domain,
                    "relevance": self._calculate_temporal_relevance(
                        start_time, end_time, node_start, node_end
                    )
                })
        
        # Ищем связи с временной информацией
        for edge in self.edges.values():
            if not edge.temporal_info:
                continue
            
            # Проверяем временные рамки
            edge_start = edge.temporal_info.get("start", 0)
            edge_end = edge.temporal_info.get("end", float('inf'))
            
            if (edge_start <= end_time and edge_end >= start_time):
                # Проверяем фильтры
                if entity:
                    source_node = self.get_node(edge.source_id)
                    target_node = self.get_node(edge.target_id)
                    
                    entity_found = False
                    if source_node and entity.lower() in source_node.name.lower():
                        entity_found = True
                    if target_node and entity.lower() in target_node.name.lower():
                        entity_found = True
                    
                    if not entity_found:
                        continue
                
                if event_type and edge.temporal_info.get("type") != event_type:
                    continue
                
                results.append({
                    "type": "edge",
                    "id": edge.id,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relation_type": edge.relation_type,
                    "temporal_info": edge.temporal_info,
                    "relevance": self._calculate_temporal_relevance(
                        start_time, end_time, edge_start, edge_end
                    )
                })
        
        # Сортируем по релевантности
        results.sort(key=lambda x: x["relevance"], reverse=True)
        
        return results[:limit]
    
    def _calculate_temporal_relevance(self, query_start: float, query_end: float,
                                   item_start: float, item_end: float) -> float:
        """
        Рассчитывает релевантность временного элемента запросу.
        
        Args:
            query_start: Начало запроса
            query_end: Конец запроса
            item_start: Начало элемента
            item_end: Конец элемента
            
        Returns:
            float: Релевантность (0.0-1.0)
        """
        # Вычисляем пересечение
        intersection_start = max(query_start, item_start)
        intersection_end = min(query_end, item_end)
        
        if intersection_start >= intersection_end:
            return 0.0  # Нет пересечения
        
        # Длина пересечения
        intersection_length = intersection_end - intersection_start
        
        # Длина запроса
        query_length = query_end - query_start
        
        # Релевантность пропорциональна доле пересечения
        relevance = intersection_length / query_length
        
        return min(1.0, relevance)
    
    def search_spatial(self, location: Dict[str, float], 
                      radius: float, 
                      entity: Optional[str] = None,
                      spatial_type: Optional[str] = None,
                      limit: int = 10) -> List[Dict[str, Any]]:
        """
        Выполняет пространственный поиск в графе знаний.
        
        Args:
            location: Координаты центра поиска
            radius: Радиус поиска в км
            entity: Сущность для фильтрации
            spatial_type: Тип пространственных данных
            limit: Лимит результатов
            
        Returns:
            List[Dict[str, Any]]: Результаты пространственного поиска
        """
        results = []
        
        # Ищем узлы с пространственной информацией
        for node in self.nodes.values():
            if not node.spatial_info or "coordinates" not in node.spatial_info:
                continue
            
            # Вычисляем расстояние
            node_coords = node.spatial_info["coordinates"]
            distance = self._calculate_distance(
                location["lat"], location["lon"],
                node_coords["lat"], node_coords["lon"]
            )
            
            if distance <= radius:
                # Проверяем фильтры
                if entity and entity.lower() not in node.name.lower():
                    continue
                
                if spatial_type and node.spatial_info.get("type") != spatial_type:
                    continue
                
                results.append({
                    "type": "node",
                    "id": node.id,
                    "name": node.name,
                    "description": node.description,
                    "spatial_info": node.spatial_info,
                    "distance": distance,
                    "domain": node.domain
                })
        
        # Ищем связи с пространственной информацией
        for edge in self.edges.values():
            if not edge.spatial_info or "coordinates" not in edge.spatial_info:
                continue
            
            # Вычисляем расстояние
            edge_coords = edge.spatial_info["coordinates"]
            distance = self._calculate_distance(
                location["lat"], location["lon"],
                edge_coords["lat"], edge_coords["lon"]
            )
            
            if distance <= radius:
                # Проверяем фильтры
                if entity:
                    source_node = self.get_node(edge.source_id)
                    target_node = self.get_node(edge.target_id)
                    
                    entity_found = False
                    if source_node and entity.lower() in source_node.name.lower():
                        entity_found = True
                    if target_node and entity.lower() in target_node.name.lower():
                        entity_found = True
                    
                    if not entity_found:
                        continue
                
                if spatial_type and edge.spatial_info.get("type") != spatial_type:
                    continue
                
                results.append({
                    "type": "edge",
                    "id": edge.id,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relation_type": edge.relation_type,
                    "spatial_info": edge.spatial_info,
                    "distance": distance
                })
        
        # Сортируем по расстоянию
        results.sort(key=lambda x: x["distance"])
        
        return results[:limit]
    
    def build_semantic_network(self, concept: str, depth: int = 2) -> Dict[str, Any]:
        """
        Строит семантическую сеть вокруг концепта.
        
        Args:
            concept: Концепт для построения сети
            depth: Глубина сети
            
        Returns:
            Dict[str, Any]: Семантическая сеть
        """
        # Ищем узел по концепту
        nodes = self.search_nodes(concept, limit=1)
        if not nodes:
            return {"nodes": [], "edges": []}
        
        center_node = nodes[0]
        
        # Получаем подграф
        subgraph = self.get_subgraph(center_node.id, depth)
        
        # Добавляем семантические связи
        semantic_edges = []
        for i, node1 in enumerate(subgraph["nodes"]):
            for j, node2 in enumerate(subgraph["nodes"]):
                if i >= j:
                    continue
                
                # Вычисляем семантическое сходство
                similarity = self._calculate_semantic_similarity(node1, node2)
                
                if similarity > 0.3:  # Порог сходства
                    semantic_edges.append({
                        "source": node1["id"],
                        "target": node2["id"],
                        "relation_type": "semantic_similarity",
                        "strength": similarity
                    })
        
        # Объединяем с существующими связями
        all_edges = subgraph["edges"] + semantic_edges
        
        return {
            "nodes": subgraph["nodes"],
            "edges": all_edges
        }
    
    def _calculate_semantic_similarity(self, node1: Dict[str, Any], 
                                     node2: Dict[str, Any]) -> float:
        """
        Рассчитывает семантическое сходство между узлами.
        
        Args:
            node1: Первый узел
            node2: Второй узел
            
        Returns:
            float: Семантическое сходство (0.0-1.0)
        """
        if not self.text_processor:
            # Простая проверка, если текстовый процессор недоступен
            text1 = (node1.get("name", "") + " " + node1.get("description", "")).lower()
            text2 = (node2.get("name", "") + " " + node2.get("description", "")).lower()
            
            common_words = set(text1.split()) & set(text2.split())
            total_words = len(set(text1.split()) | set(text2.split()))
            
            return len(common_words) / max(1, total_words) if total_words > 0 else 0.0
        
        try:
            # Используем эмбеддинги для более точной оценки
            text1 = f"{node1.get('name', '')} {node1.get('description', '')}"
            text2 = f"{node2.get('name', '')} {node2.get('description', '')}"
            
            return self.text_processor.calculate_similarity(text1, text2)
            
        except Exception as e:
            logger.error(f"Ошибка расчета семантического сходства: {e}", exc_info=True)
            return 0.0
    
    def get_concept_hierarchy(self, concept: str, max_depth: int = 3) -> Dict[str, Any]:
        """
        Возвращает иерархию концептов для указанного концепта.
        
        Args:
            concept: Концепт для построения иерархии
            max_depth: Максимальная глубина иерархии
            
        Returns:
            Dict[str, Any]: Иерархия концептов
        """
        # Ищем узел по концепту
        nodes = self.search_nodes(concept, limit=1)
        if not nodes:
            return {"error": "concept_not_found"}
        
        root_node = nodes[0]
        
        # Строим иерархию
        hierarchy = {
            "root": {
                "id": root_node.id,
                "name": root_node.name,
                "description": root_node.description
            },
            "children": self._build_hierarchy_children(root_node.id, max_depth)
        }
        
        return hierarchy
    
    def _build_hierarchy_children(self, node_id: str, depth: int) -> List[Dict[str, Any]]:
        """
        Рекурсивно строит дочерние элементы иерархии.
        
        Args:
            node_id: ID узла
            depth: Текущая глубина
            
        Returns:
            List[Dict[str, Any]]: Дочерние элементы
        """
        if depth <= 0:
            return []
        
        children = []
        
        # Ищем связи "is_a" и "part_of"
        for edge in self.get_edges(node_id, direction="target", relation_type="is_a"):
            child_node = self.get_node(edge.source_id)
            if child_node:
                children.append({
                    "id": child_node.id,
                    "name": child_node.name,
                    "description": child_node.description,
                    "children": self._build_hierarchy_children(child_node.id, depth - 1)
                })
        
        for edge in self.get_edges(node_id, direction="target", relation_type="part_of"):
            child_node = self.get_node(edge.source_id)
            if child_node:
                children.append({
                    "id": child_node.id,
                    "name": child_node.name,
                    "description": child_node.description,
                    "children": self._build_hierarchy_children(child_node.id, depth - 1)
                })
        
        return children
    
    def get_concept_network(self, concept: str, max_nodes: int = 50) -> Dict[str, Any]:
        """
        Возвращает сеть концептов, связанных с указанным концептом.
        
        Args:
            concept: Концепт для построения сети
            max_nodes: Максимальное количество узлов
            
        Returns:
            Dict[str, Any]: Сеть концептов
        """
        # Ищем узел по концепту
        nodes = self.search_nodes(concept, limit=1)
        if not nodes:
            return {"nodes": [], "edges": []}
        
        center_node = nodes[0]
        
        # Получаем связанные узлы
        related_nodes = self.get_related_nodes(center_node.id, max_distance=2)
        
        # Ограничиваем количество узлов
        if len(related_nodes) > max_nodes:
            # Сортируем по силе связи и оставляем самые сильные
            related_nodes.sort(key=lambda n: self._get_node_strength(center_node.id, n.id), reverse=True)
            related_nodes = related_nodes[:max_nodes]
        
        # Создаем узлы для визуализации
        viz_nodes = [{
            "id": center_node.id,
            "label": center_node.name,
            "title": center_node.description,
            "group": "center",
            "shape": "dot",
            "size": 25
        }]
        
        for node in related_nodes:
            strength = self._get_node_strength(center_node.id, node.id)
            viz_nodes.append({
                "id": node.id,
                "label": node.name,
                "title": node.description,
                "group": node.domain,
                "shape": "dot",
                "size": 10 + strength * 15  # Размер зависит от силы связи
            })
        
        # Создаем связи
        viz_edges = []
        for node in related_nodes:
            strength = self._get_node_strength(center_node.id, node.id)
            viz_edges.append({
                "from": center_node.id,
                "to": node.id,
                "value": strength,
                "title": f"Сила: {strength:.2f}"
            })
            
            # Добавляем связи между связанными узлами
            for other_node in related_nodes:
                if node.id == other_node.id:
                    continue
                
                other_strength = self._get_node_strength(node.id, other_node.id)
                if other_strength > 0.2:  # Порог для отображения
                    viz_edges.append({
                        "from": node.id,
                        "to": other_node.id,
                        "value": other_strength * 0.5,
                        "dashes": True
                    })
        
        return {
            "nodes": viz_nodes,
            "edges": viz_edges
        }
    
    def _get_node_strength(self, node1_id: str, node2_id: str) -> float:
        """
        Возвращает силу связи между двумя узлами.
        
        Args:
            node1_id: ID первого узла
            node2_id: ID второго узла
            
        Returns:
            float: Сила связи
        """
        max_strength = 0.0
        
        # Проверяем прямые связи
        for edge in self.get_edges(node1_id, direction="both"):
            if (edge.source_id == node1_id and edge.target_id == node2_id) or \
               (edge.source_id == node2_id and edge.target_id == node1_id):
                max_strength = max(max_strength, edge.strength)
        
        # Если прямых связей нет, проверяем через общий узел
        if max_strength == 0.0:
            node1_edges = self.get_edges(node1_id, direction="both")
            node2_edges = self.get_edges(node2_id, direction="both")
            
            # Ищем общий узел
            common_nodes = set()
            for edge in node1_edges:
                common_nodes.add(edge.source_id)
                common_nodes.add(edge.target_id)
            
            for edge in node2_edges:
                if edge.source_id in common_nodes:
                    common_nodes = {edge.source_id}
                    break
                if edge.target_id in common_nodes:
                    common_nodes = {edge.target_id}
                    break
            
            # Если найден общий узел, вычисляем косвенную силу
            if common_nodes:
                node1_strength = max((e.strength for e in node1_edges), default=0.0)
                node2_strength = max((e.strength for e in node2_edges), default=0.0)
                max_strength = (node1_strength + node2_strength) / 2 * 0.7  # Понижаем вес косвенной связи
        
        return max_strength
    
    def get_domain_knowledge_map(self, domain: str, max_nodes: int = 100) -> Dict[str, Any]:
        """
        Возвращает карту знаний для указанной области.
        
        Args:
            domain: Область знаний
            max_nodes: Максимальное количество узлов
            
        Returns:
            Dict[str, Any]: Карта знаний
        """
        # Получаем узлы в домене
        nodes = [
            node for node in self.nodes.values() 
            if node.domain == domain
        ]
        
        # Ограничиваем количество
        if len(nodes) > max_nodes:
            # Сортируем по силе и актуальности
            nodes.sort(key=lambda n: (n.strength, -n.last_updated), reverse=True)
            nodes = nodes[:max_nodes]
        
        # Создаем узлы для визуализации
        viz_nodes = []
        for node in nodes:
            viz_nodes.append({
                "id": node.id,
                "label": node.name,
                "title": node.description,
                "group": node.node_type,
                "shape": "dot",
                "size": 10 + node.strength * 20
            })
        
        # Создаем связи
        viz_edges = []
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i >= j:
                    continue
                
                # Проверяем, есть ли прямая связь
                has_direct_connection = False
                for edge in self.get_edges(node1.id, direction="both"):
                    if (edge.source_id == node1.id and edge.target_id == node2.id) or \
                       (edge.source_id == node2.id and edge.target_id == node1.id):
                        viz_edges.append({
                            "from": node1.id,
                            "to": node2.id,
                            "value": edge.strength,
                            "title": edge.relation_type
                        })
                        has_direct_connection = True
                        break
                
                # Если нет прямой связи, проверяем косвенную
                if not has_direct_connection:
                    strength = self._get_node_strength(node1.id, node2.id)
                    if strength > 0.3:  # Порог для косвенной связи
                        viz_edges.append({
                            "from": node1.id,
                            "to": node2.id,
                            "value": strength * 0.5,
                            "dashes": True,
                            "title": f"Косвенная связь: {strength:.2f}"
                        })
        
        return {
            "nodes": viz_nodes,
            "edges": viz_edges
        }
    
    def get_knowledge_evolution(self, concept: str, 
                              time_intervals: int = 5) -> List[Dict[str, Any]]:
        """
        Возвращает эволюцию знаний по концепту во времени.
        
        Args:
            concept: Концепт для анализа
            time_intervals: Количество временных интервалов
            
        Returns:
            List[Dict[str, Any]]: Эволюция знаний
        """
        # Ищем узел по концепту
        nodes = self.search_nodes(concept, limit=1)
        if not nodes:
            return []
        
        node = nodes[0]
        
        # Получаем историю изменений
        history = []
        for change in node.history:
            if "changes" in change and "new" in change["changes"]:
                new_desc = change["changes"]["new"].get("description")
                if new_desc:
                    history.append({
                        "timestamp": change["timestamp"],
                        "description": new_desc,
                        "version": change["version"]
                    })
        
        # Если нет истории, используем текущее состояние
        if not history:
            history.append({
                "timestamp": node.timestamp,
                "description": node.description,
                "version": 1
            })
        
        # Сортируем по времени
        history.sort(key=lambda x: x["timestamp"])
        
        # Разделяем на временные интервалы
        if len(history) <= time_intervals:
            return history
        
        interval_size = len(history) // time_intervals
        sampled_history = []
        
        for i in range(time_intervals):
            # Берем средний элемент в интервале
            idx = min((i + 1) * interval_size - 1, len(history) - 1)
            sampled_history.append(history[idx])
        
        return sampled_history
    
    def get_knowledge_density(self, domain: Optional[str] = None) -> Dict[str, float]:
        """
        Рассчитывает плотность знаний в указанной области.
        
        Args:
            domain: Область знаний (опционально)
            
        Returns:
            Dict[str, float]: Плотность знаний
        """
        total_nodes = 0
        total_edges = 0
        domain_nodes = 0
        domain_edges = 0
        
        # Считаем общее количество
        total_nodes = len(self.nodes)
        total_edges = len(self.edges)
        
        # Считаем для домена, если указан
        if domain:
            for node in self.nodes.values():
                if node.domain == domain:
                    domain_nodes += 1
            
            for edge in self.edges.values():
                source_node = self.get_node(edge.source_id)
                target_node = self.get_node(edge.target_id)
                
                if (source_node and source_node.domain == domain) or \
                   (target_node and target_node.domain == domain):
                    domain_edges += 1
        
        # Рассчитываем плотность
        global_density = total_edges / max(1, total_nodes) if total_nodes > 0 else 0
        domain_density = domain_edges / max(1, domain_nodes) if domain_nodes > 0 else 0
        
        return {
            "global_density": global_density,
            "domain_density": domain_density,
            "completeness": domain_density / max(0.1, global_density) if global_density > 0 else 0
        }
    
    def get_knowledge_coverage(self, required_concepts: List[str]) -> Dict[str, Any]:
        """
        Оценивает покрытие обязательных концептов.
        
        Args:
            required_concepts: Список обязательных концептов
            
        Returns:
            Dict[str, Any]: Покрытие знаний
        """
        covered = []
        missing = []
        
        for concept in required_concepts:
            nodes = self.search_nodes(concept, limit=1)
            if nodes:
                covered.append(concept)
            else:
                missing.append(concept)
        
        coverage = len(covered) / max(1, len(required_concepts))
        
        return {
            "coverage": coverage,
            "covered_concepts": covered,
            "missing_concepts": missing,
            "recommendations": self._generate_coverage_recommendations(missing)
        }
    
    def _generate_coverage_recommendations(self, missing_concepts: List[str]) -> List[str]:
        """
        Генерирует рекомендации по заполнению пробелов.
        
        Args:
            missing_concepts: Отсутствующие концепты
            
        Returns:
            List[str]: Рекомендации
        """
        recommendations = []
        
        if missing_concepts:
            recommendations.append(
                f"Добавьте {len(missing_concepts)} отсутствующих концептов: " +
                ", ".join(missing_concepts[:3]) + ("..." if len(missing_concepts) > 3 else "")
            )
            
            # Анализируем похожие концепты
            similar_concepts = []
            for concept in missing_concepts[:5]:  # Ограничиваем анализ
                nodes = self.search_nodes(concept, limit=3)
                for node in nodes:
                    if node.name.lower() != concept.lower():
                        similar_concepts.append((concept, node.name))
            
            if similar_concepts:
                recommendations.append(
                    "Рассмотрите похожие концепты: " +
                    ", ".join([f"'{orig}' -> '{similar}'" for orig, similar in similar_concepts])
                )
            
            recommendations.append(
                "Используйте модуль расширения знаний для автоматического добавления недостающих концептов"
            )
        
        return recommendations
    
    def get_knowledge_quality(self, domain: Optional[str] = None) -> Dict[str, float]:
        """
        Оценивает качество знаний в указанной области.
        
        Args:
            domain: Область знаний (опционально)
            
        Returns:
            Dict[str, float]: Качество знаний
        """
        # Получаем узлы
        nodes = list(self.nodes.values())
        if domain:
            nodes = [node for node in nodes if node.domain == domain]
        
        if not nodes:
            return {
                "completeness": 0.0,
                "consistency": 0.0,
                "accuracy": 0.0,
                "timeliness": 0.0,
                "overall": 0.0
            }
        
        # Оцениваем полноту
        completeness = len(nodes) / 1000  # Нормализуем к 0-1 (условно)
        completeness = min(1.0, completeness)
        
        # Оцениваем согласованность (отсутствие противоречий)
        inconsistent_nodes = sum(1 for node in nodes if node.contradictions)
        consistency = 1.0 - (inconsistent_nodes / len(nodes))
        
        # Оцениваем точность (на основе силы знаний)
        avg_strength = sum(node.strength for node in nodes) / len(nodes)
        accuracy = min(1.0, avg_strength * 1.2)  # Небольшое увеличение для компенсации
        
        # Оцениваем своевременность
        current_time = time.time()
        avg_age = sum(current_time - node.last_updated for node in nodes) / len(nodes)
        timeliness = max(0.0, 1.0 - (avg_age / (365 * 86400)))  # Нормализуем к 0-1 (год)
        
        # Общий показатель
        overall = (
            completeness * 0.3 +
            consistency * 0.3 +
            accuracy * 0.2 +
            timeliness * 0.2
        )
        
        return {
            "completeness": completeness,
            "consistency": consistency,
            "accuracy": accuracy,
            "timeliness": timeliness,
            "overall": overall
        }
    
    def get_knowledge_gaps_analysis(self, domain: str) -> Dict[str, Any]:
        """
        Проводит анализ пробелов в знаниях для указанной области.
        
        Args:
            domain: Область знаний
            
        Returns:
            Dict[str, Any]: Анализ пробелов
        """
        # Получаем узлы в домене
        domain_nodes = [node for node in self.nodes.values() if node.domain == domain]
        
        # Анализируем плотность связей
        avg_connections = 0
        if domain_nodes:
            total_connections = sum(len(self.get_edges(node.id)) for node in domain_nodes)
            avg_connections = total_connections / len(domain_nodes)
        
        # Определяем минимально необходимое количество связей
        min_connections = 3
        weakly_connected = [
            node for node in domain_nodes 
            if len(self.get_edges(node.id)) < min_connections
        ]
        
        # Анализируем устаревшие знания
        outdated_threshold = time.time() - (180 * 86400)  # 6 месяцев
        outdated = [
            node for node in domain_nodes 
            if node.last_updated < outdated_threshold
        ]
        
        # Анализируем противоречия
        contradictory = [
            node for node in domain_nodes 
            if node.contradictions and not all(c["resolved"] for c in node.contradictions)
        ]
        
        # Формируем анализ
        analysis = {
            "domain": domain,
            "total_nodes": len(domain_nodes),
            "weakly_connected_nodes": len(weakly_connected),
            "outdated_nodes": len(outdated),
            "contradictory_nodes": len(contradictory),
            "gap_severity": self._calculate_gap_severity(
                len(weakly_connected), 
                len(outdated), 
                len(contradictory),
                len(domain_nodes)
            ),
            "recommendations": []
        }
        
        # Формируем рекомендации
        if weakly_connected:
            analysis["recommendations"].append(
                f"Укрепите связи для {len(weakly_connected)} концептов с малым количеством связей"
            )
        
        if outdated:
            analysis["recommendations"].append(
                f"Обновите информацию для {len(outdated)} устаревших концептов"
            )
        
        if contradictory:
            analysis["recommendations"].append(
                f"Разрешите противоречия для {len(contradictory)} концептов"
            )
        
        if not analysis["recommendations"]:
            analysis["recommendations"].append(
                "Пробелов в знаниях не обнаружено"
            )
        
        return analysis
    
    def _calculate_gap_severity(self, weak_count: int, outdated_count: int, 
                             contradictory_count: int, total_count: int) -> float:
        """
        Рассчитывает серьезность пробелов в знаниях.
        
        Args:
            weak_count: Количество слабо связанных концептов
            outdated_count: Количество устаревших концептов
            contradictory_count: Количество противоречивых концептов
            total_count: Общее количество концептов
            
        Returns:
            float: Серьезность пробелов (0.0-1.0)
        """
        if total_count == 0:
            return 0.0
        
        # Веса для разных типов пробелов
        weights = {
            "weak": 0.3,
            "outdated": 0.4,
            "contradictory": 0.3
        }
        
        # Нормализованные значения
        weak_ratio = weak_count / total_count
        outdated_ratio = outdated_count / total_count
        contradictory_ratio = contradictory_count / total_count
        
        # Рассчитываем общий показатель
        severity = (
            weak_ratio * weights["weak"] +
            outdated_ratio * weights["outdated"] +
            contradictory_ratio * weights["contradictory"]
        )
        
        return min(1.0, severity)
    
    def get_knowledge_trends(self, domain: Optional[str] = None, 
                           period: str = "month") -> Dict[str, Any]:
        """
        Анализирует тренды в знаниях за указанный период.
        
        Args:
            domain: Область знаний (опционально)
            period: Период анализа (day, week, month, year)
            
        Returns:
            Dict[str, Any]: Тренды в знаниях
        """
        # Определяем временные интервалы
        intervals = self._get_time_intervals(period)
        
        # Собираем данные по интервалам
        trends = {
            "intervals": intervals,
            "new_nodes": [0] * len(intervals),
            "updated_nodes": [0] * len(intervals),
            "contradictions": [0] * len(intervals),
            "sources": defaultdict(lambda: [0] * len(intervals))
        }
        
        # Анализируем узлы
        for node in self.nodes.values():
            if domain and node.domain != domain:
                continue
            
            # Определяем интервал для создания
            for i, (start, end) in enumerate(intervals):
                if start <= node.timestamp <= end:
                    trends["new_nodes"][i] += 1
                    
                    # Учитываем источник
                    for source in node.meta.get("sources", []):
                        source_name = source.get("source", "unknown")
                        trends["sources"][source_name][i] += 1
            
            # Определяем интервалы для обновлений
            for change in node.history:
                for i, (start, end) in enumerate(intervals):
                    if start <= change["timestamp"] <= end:
                        trends["updated_nodes"][i] += 1
            
            # Считаем противоречия
            for contradiction in node.contradictions:
                for i, (start, end) in enumerate(intervals):
                    if start <= contradiction["timestamp"] <= end:
                        trends["contradictions"][i] += 1
        
        # Преобразуем defaultdict в обычный dict
        trends["sources"] = dict(trends["sources"])
        
        return trends
    
    def _get_time_intervals(self, period: str) -> List[Tuple[float, float]]:
        """
        Возвращает временные интервалы для анализа трендов.
        
        Args:
            period: Период (day, week, month, year)
            
        Returns:
            List[Tuple[float, float]]: Список интервалов (начало, конец)
        """
        current_time = time.time()
        intervals = []
        
        if period == "day":
            # Последние 30 дней
            for i in range(29, -1, -1):
                start = current_time - (i + 1) * 86400
                end = current_time - i * 86400
                intervals.append((start, end))
        
        elif period == "week":
            # Последние 12 недель
            for i in range(11, -1, -1):
                start = current_time - (i + 1) * 7 * 86400
                end = current_time - i * 7 * 86400
                intervals.append((start, end))
        
        elif period == "month":
            # Последние 12 месяцев
            for i in range(11, -1, -1):
                # Приблизительный расчет (30 дней в месяце)
                start = current_time - (i + 1) * 30 * 86400
                end = current_time - i * 30 * 86400
                intervals.append((start, end))
        
        elif period == "year":
            # Последние 10 лет
            for i in range(9, -1, -1):
                start = current_time - (i + 1) * 365 * 86400
                end = current_time - i * 365 * 86400
                intervals.append((start, end))
        
        return intervals
    
    def get_knowledge_sources_analysis(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Анализирует источники знаний для указанной области.
        
        Args:
            domain: Область знаний (опционально)
            
        Returns:
            Dict[str, Any]: Анализ источников
        """
        source_stats = defaultdict(lambda: {
            "node_count": 0,
            "total_strength": 0.0,
            "unique_concepts": set(),
            "contradictions": 0
        })
        
        # Собираем статистику по источникам
        for node in self.nodes.values():
            if domain and node.domain != domain:
                continue
            
            for source in node.meta.get("sources", []):
                source_name = source.get("source", "unknown")
                
                # Обновляем статистику
                source_stats[source_name]["node_count"] += 1
                source_stats[source_name]["total_strength"] += node.strength
                source_stats[source_name]["unique_concepts"].add(node.name.lower())
                
                # Проверяем противоречия
                if node.contradictions:
                    source_stats[source_name]["contradictions"] += 1
        
        # Преобразуем в удобный формат
        analysis = {
            "total_sources": len(source_stats),
            "sources": []
        }
        
        for source, stats in source_stats.items():
            avg_strength = stats["total_strength"] / stats["node_count"] if stats["node_count"] > 0 else 0
            contradiction_rate = stats["contradictions"] / stats["node_count"] if stats["node_count"] > 0 else 0
            
            analysis["sources"].append({
                "name": source,
                "node_count": stats["node_count"],
                "average_strength": avg_strength,
                "unique_concepts": len(stats["unique_concepts"]),
                "contradiction_rate": contradiction_rate,
                "reliability": max(0.1, 1.0 - contradiction_rate) * avg_strength
            })
        
        # Сортируем по надежности
        analysis["sources"].sort(key=lambda x: x["reliability"], reverse=True)
        
        return analysis
    
    def get_knowledge_evolution_map(self, concept: str, 
                                  time_horizon: str = "year") -> Dict[str, Any]:
        """
        Возвращает карту эволюции знаний по концепту.
        
        Args:
            concept: Концепт для анализа
            time_horizon: Горизонт времени (month, quarter, year)
            
        Returns:
            Dict[str, Any]: Карта эволюции
        """
        # Ищем узел по концепту
        nodes = self.search_nodes(concept, limit=1)
        if not nodes:
            return {"error": "concept_not_found"}
        
        node = nodes[0]
        
        # Получаем историю изменений
        history = []
        for change in node.history:
            if "changes" in change and "new" in change["changes"]:
                new_desc = change["changes"]["new"].get("description")
                if new_desc:
                    history.append({
                        "timestamp": change["timestamp"],
                        "description": new_desc,
                        "version": change["version"]
                    })
        
        # Если нет истории, используем текущее состояние
        if not history:
            history.append({
                "timestamp": node.timestamp,
                "description": node.description,
                "version": 1
            })
        
        # Сортируем по времени
        history.sort(key=lambda x: x["timestamp"])
        
        # Определяем временные интервалы
        intervals = self._get_time_intervals_for_evolution(history[0]["timestamp"], 
                                                         history[-1]["timestamp"], 
                                                         time_horizon)
        
        # Группируем изменения по интервалам
        interval_changes = [[] for _ in intervals]
        for change in history:
            for i, (start, end) in enumerate(intervals):
                if start <= change["timestamp"] <= end:
                    interval_changes[i].append(change)
                    break
        
        # Создаем карту эволюции
        evolution_map = []
        current_description = node.description
        
        for i, changes in enumerate(interval_changes):
            if changes:
                # Берем последнее изменение в интервале
                latest_change = changes[-1]
                current_description = latest_change["description"]
            
            evolution_map.append({
                "interval": f"{datetime.fromtimestamp(intervals[i][0]).strftime('%Y-%m-%d')} - {datetime.fromtimestamp(intervals[i][1]).strftime('%Y-%m-%d')}",
                "description": current_description,
                "change_count": len(changes)
            })
        
        return {
            "concept": concept,
            "evolution_map": evolution_map,
            "total_intervals": len(intervals),
            "change_density": len(history) / max(1, len(intervals))
        }
    
    def _get_time_intervals_for_evolution(self, start_time: float, 
                                       end_time: float, 
                                       time_horizon: str) -> List[Tuple[float, float]]:
        """
        Возвращает временные интервалы для карты эволюции.
        
        Args:
            start_time: Начальное время
            end_time: Конечное время
            time_horizon: Горизонт времени
            
        Returns:
            List[Tuple[float, float]]: Список интервалов
        """
        intervals = []
        total_duration = end_time - start_time
        
        if time_horizon == "month":
            interval_duration = 30 * 86400  # 30 дней
        elif time_horizon == "quarter":
            interval_duration = 90 * 86400  # 90 дней
        else:  # year
            interval_duration = 365 * 86400  # 365 дней
        
        # Создаем интервалы
        current = start_time
        while current < end_time:
            next_time = min(current + interval_duration, end_time)
            intervals.append((current, next_time))
            current = next_time
        
        return intervals
    
    def get_knowledge_integration_points(self, domain1: str, 
                                      domain2: str) -> List[Dict[str, Any]]:
        """
        Находит точки интеграции между двумя областями знаний.
        
        Args:
            domain1: Первая область знаний
            domain2: Вторая область знаний
            
        Returns:
            List[Dict[str, Any]]: Точки интеграции
        """
        integration_points = []
        
        # Получаем узлы в обеих областях
        domain1_nodes = [node for node in self.nodes.values() if node.domain == domain1]
        domain2_nodes = [node for node in self.nodes.values() if node.domain == domain2]
        
        # Ищем общие концепты
        for node1 in domain1_nodes:
            for node2 in domain2_nodes:
                # Проверяем семантическое сходство
                similarity = self._calculate_semantic_similarity(node1.to_dict(), node2.to_dict())
                
                if similarity > 0.6:  # Высокое сходство
                    integration_points.append({
                        "concept1": node1.name,
                        "concept2": node2.name,
                        "similarity": similarity,
                        "type": "common_concept",
                        "evidence": f"Семантическое сходство: {similarity:.2f}"
                    })
                
                # Проверяем связи между областями
                for edge in self.get_edges(node1.id, direction="both"):
                    if edge.target_id == node2.id or edge.source_id == node2.id:
                        integration_points.append({
                            "concept1": node1.name,
                            "concept2": node2.name,
                            "relation": edge.relation_type,
                            "strength": edge.strength,
                            "type": "direct_connection",
                            "evidence": f"Прямая связь: {edge.relation_type} (сила: {edge.strength})"
                        })
        
        # Ищем опосредованные связи
        for node1 in domain1_nodes:
            for node2 in domain2_nodes:
                # Проверяем наличие общего узла
                node1_edges = self.get_edges(node1.id, direction="both")
                node2_edges = self.get_edges(node2.id, direction="both")
                
                common_nodes = set()
                for edge in node1_edges:
                    common_nodes.add(edge.source_id)
                    common_nodes.add(edge.target_id)
                
                for edge in node2_edges:
                    if edge.source_id in common_nodes or edge.target_id in common_nodes:
                        # Найдена опосредованная связь
                        node1_strength = max((e.strength for e in node1_edges), default=0.0)
                        node2_strength = max((e.strength for e in node2_edges), default=0.0)
                        indirect_strength = (node1_strength + node2_strength) / 2 * 0.7
                        
                        if indirect_strength > 0.3:
                            integration_points.append({
                                "concept1": node1.name,
                                "concept2": node2.name,
                                "indirect_strength": indirect_strength,
                                "type": "indirect_connection",
                                "evidence": f"Опосредованная связь через общий концепт"
                            })
        
        return integration_points
    
    def integrate_knowledge_domains(self, domain1: str, domain2: str) -> Dict[str, Any]:
        """
        Интегрирует две области знаний.
        
        Args:
            domain1: Первая область знаний
            domain2: Вторая область знаний
            
        Returns:
            Dict[str, Any]: Результат интеграции
        """
        result = {
            "domain1": domain1,
            "domain2": domain2,
            "integration_points": [],
            "new_concepts": [],
            "new_relations": 0,
            "status": "completed"
        }
        
        try:
            # Находим точки интеграции
            integration_points = self.get_knowledge_integration_points(domain1, domain2)
            result["integration_points"] = integration_points
            
            # Создаем новые концепты для интеграции
            for point in integration_points:
                if point["type"] == "common_concept":
                    # Создаем интегрированный концепт
                    integrated_concept = self._create_integrated_concept(
                        point["concept1"], 
                        point["concept2"],
                        domain1,
                        domain2
                    )
                    result["new_concepts"].append(integrated_concept)
                
                elif point["type"] == "direct_connection":
                    # Усиливаем связь
                    self._enhance_connection(
                        point["concept1"], 
                        point["concept2"], 
                        point["relation"], 
                        point["strength"] * 1.2
                    )
                    result["new_relations"] += 1
            
            # Добавляем новые связи между интегрированными концептами
            for i in range(len(result["new_concepts"])):
                for j in range(i + 1, len(result["new_concepts"])):
                    self.add_edge(
                        result["new_concepts"][i]["id"],
                        result["new_concepts"][j]["id"],
                        "integrates",
                        strength=0.7
                    )
                    result["new_relations"] += 1
            
            # Обновляем статистику
            result["status"] = "completed"
            logger.info(f"Области знаний '{domain1}' и '{domain2}' успешно интегрированы")
            
        except Exception as e:
            logger.error(f"Ошибка интеграции областей знаний: {e}", exc_info=True)
            result["status"] = "failed"
            result["error"] = str(e)
        
        return result
    
    def _create_integrated_concept(self, concept1: str, concept2: str, 
                                 domain1: str, domain2: str) -> Dict[str, Any]:
        """
        Создает интегрированный концепт из двух концептов.
        
        Args:
            concept1: Первый концепт
            concept2: Второй концепт
            domain1: Первая область
            domain2: Вторая область
            
        Returns:
            Dict[str, Any]: Интегрированный концепт
        """
        # Генерируем имя интегрированного концепта
        integrated_name = f"{concept1}+{concept2}"
        
        # Создаем новый узел
        node_id = self.add_node(
            name=integrated_name,
            description=f"Интеграция концептов {concept1} и {concept2}",
            node_type="concept",
            domain=f"{domain1}_{domain2}",
            strength=0.85
        )
        
        # Создаем связи с основными концептами
        concept1_node = self.search_nodes(concept1, limit=1)
        concept2_node = self.search_nodes(concept2, limit=1)
        
        if concept1_node:
            self.add_edge(node_id, concept1_node[0].id, "integrates", strength=0.8)
        
        if concept2_node:
            self.add_edge(node_id, concept2_node[0].id, "integrates", strength=0.8)
        
        return {
            "id": node_id,
            "name": integrated_name,
            "concept1": concept1,
            "concept2": concept2,
            "domain": f"{domain1}_{domain2}"
        }
    
    def _enhance_connection(self, concept1: str, concept2: str, 
                          relation_type: str, new_strength: float):
        """
        Усиливает существующую связь между концептами.
        
        Args:
            concept1: Первый концепт
            concept2: Второй концепт
            relation_type: Тип связи
            new_strength: Новая сила связи
        """
        # Находим узлы
        node1 = self.search_nodes(concept1, limit=1)
        node2 = self.search_nodes(concept2, limit=1)
        
        if not node1 or not node2:
            return
        
        # Находим существующую связь
        for edge in self.get_edges(node1[0].id, direction="both"):
            if (edge.source_id == node1[0].id and edge.target_id == node2[0].id) or \
               (edge.source_id == node2[0].id and edge.target_id == node1[0].id):
                # Обновляем силу связи
                self.update_edge(edge.id, new_strength=new_strength)
                return
        
        # Если связи нет, создаем новую
        self.add_edge(node1[0].id, node2[0].id, relation_type, strength=new_strength)
    
    def get_knowledge_abstraction_levels(self, domain: str) -> Dict[str, Any]:
        """
        Возвращает уровни абстракции для указанной области знаний.
        
        Args:
            domain: Область знаний
            
        Returns:
            Dict[str, Any]: Уровни абстракции
        """
        # Получаем узлы в домене
        domain_nodes = [node for node in self.nodes.values() if node.domain == domain]
        
        if not domain_nodes:
            return {"error": "no_nodes_in_domain"}
        
        # Строим иерархию абстракции
        abstraction_levels = self._build_abstraction_hierarchy(domain_nodes)
        
        # Формируем результат
        result = {
            "domain": domain,
            "total_concepts": len(domain_nodes),
            "abstraction_levels": []
        }
        
        # Добавляем уровни
        for i, level in enumerate(abstraction_levels):
            result["abstraction_levels"].append({
                "level": i + 1,
                "concepts": [node.name for node in level],
                "count": len(level),
                "examples": [node.name for node in level[:3]]  # Первые 3 примера
            })
        
        return result
    
    def _build_abstraction_hierarchy(self, nodes: List[KnowledgeNode]) -> List[List[KnowledgeNode]]:
        """
        Строит иерархию абстракции для списка узлов.
        
        Args:
            nodes: Список узлов
            
        Returns:
            List[List[KnowledgeNode]]: Иерархия абстракции
        """
        # Сначала ищем самые абстрактные концепты (те, у которых много "is_a" связей)
        abstraction_hierarchy = []
        remaining_nodes = nodes.copy()
        
        while remaining_nodes:
            # Находим концепты с минимальным количеством "is_a" связей (самые конкретные)
            concrete_nodes = []
            for node in remaining_nodes:
                is_a_count = sum(1 for edge in self.get_edges(node.id, direction="source") 
                                if edge.relation_type == "is_a")
                if is_a_count == 0:
                    concrete_nodes.append(node)
            
            # Если нет конкретных концептов, берем все оставшиеся
            if not concrete_nodes:
                concrete_nodes = remaining_nodes
            
            # Добавляем на текущий уровень
            abstraction_hierarchy.append(concrete_nodes)
            
            # Удаляем из оставшихся
            for node in concrete_nodes:
                if node in remaining_nodes:
                    remaining_nodes.remove(node)
        
        # Инвертируем, чтобы самые абстрактные были первыми
        abstraction_hierarchy.reverse()
        
        return abstraction_hierarchy
    
    def get_knowledge_evolution_trends(self, domain: str, 
                                     period: str = "year") -> Dict[str, Any]:
        """
        Анализирует тренды эволюции знаний в указанной области.
        
        Args:
            domain: Область знаний
            period: Период анализа
            
        Returns:
            Dict[str, Any]: Тренды эволюции
        """
        # Получаем узлы в домене
        domain_nodes = [node for node in self.nodes.values() if node.domain == domain]
        
        if not domain_nodes:
            return {"error": "no_nodes_in_domain"}
        
        # Определяем временные интервалы
        intervals = self._get_time_intervals(period)
        
        # Анализируем изменения
        trends = {
            "intervals": [f"{datetime.fromtimestamp(start).strftime('%Y-%m')} - {datetime.fromtimestamp(end).strftime('%Y-%m')}" 
                         for start, end in intervals],
            "new_concepts": [0] * len(intervals),
            "concept_modifications": [0] * len(intervals),
            "concept_deprecations": [0] * len(intervals),
            "average_modification_frequency": 0.0
        }
        
        total_modifications = 0
        
        # Анализируем каждый узел
        for node in domain_nodes:
            # Новые концепты
            for i, (start, end) in enumerate(intervals):
                if start <= node.timestamp <= end:
                    trends["new_concepts"][i] += 1
            
            # Модификации
            for change in node.history:
                for i, (start, end) in enumerate(intervals):
                    if start <= change["timestamp"] <= end:
                        trends["concept_modifications"][i] += 1
                        total_modifications += 1
            
            # Депрекации (если есть пометка)
            if "status" in node.meta and node.meta["status"] == "deprecated":
                for i, (start, end) in enumerate(intervals):
                    if start <= node.last_updated <= end:
                        trends["concept_deprecations"][i] += 1
        
        # Средняя частота модификаций
        if len(domain_nodes) > 0:
            trends["average_modification_frequency"] = total_modifications / len(domain_nodes)
        
        return trends
    
    def get_knowledge_coherence(self, domain: Optional[str] = None) -> Dict[str, float]:
        """
        Оценивает связность знаний в указанной области.
        
        Args:
            domain: Область знаний (опционально)
            
        Returns:
            Dict[str, float]: Связность знаний
        """
        # Получаем узлы
        nodes = list(self.nodes.values())
        if domain:
            nodes = [node for node in nodes if node.domain == domain]
        
        if len(nodes) < 2:
            return {"coherence": 1.0, "reason": "not_enough_nodes"}
        
        # Рассчитываем среднее количество связей на узел
        total_connections = sum(len(self.get_edges(node.id)) for node in nodes)
        avg_connections = total_connections / len(nodes)
        
        # Идеальное количество связей для полного графа
        ideal_connections = len(nodes) - 1
        
        # Нормализуем к 0-1
        coherence = avg_connections / ideal_connections if ideal_connections > 0 else 1.0
        
        # Учитываем противоречия
        contradictory_nodes = sum(1 for node in nodes if node.contradictions)
        contradiction_penalty = contradictory_nodes / len(nodes) * 0.3
        
        # Итоговая связность
        final_coherence = max(0.0, min(1.0, coherence - contradiction_penalty))
        
        return {
            "coherence": final_coherence,
            "average_connections": avg_connections,
            "contradiction_penalty": contradiction_penalty,
            "reason": "good" if final_coherence > 0.7 else "moderate" if final_coherence > 0.4 else "poor"
        }
    
    def get_knowledge_diversity(self, domain: Optional[str] = None) -> Dict[str, float]:
        """
        Оценивает разнообразие знаний в указанной области.
        
        Args:
            domain: Область знаний (опционально)
            
        Returns:
            Dict[str, float]: Разнообразие знаний
        """
        # Получаем узлы
        nodes = list(self.nodes.values())
        if domain:
            nodes = [node for node in nodes if node.domain == domain]
        
        if not nodes:
            return {"diversity": 0.0, "reason": "no_nodes"}
        
        # Анализируем разнообразие по типам узлов
        type_counts = defaultdict(int)
        for node in nodes:
            type_counts[node.node_type] += 1
        
        # Рассчитываем энтропию
        total = len(nodes)
        entropy = 0.0
        for count in type_counts.values():
            p = count / total
            entropy -= p * math.log2(p) if p > 0 else 0
        
        # Нормализуем энтропию к 0-1
        max_entropy = math.log2(len(type_counts)) if len(type_counts) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1
        
        # Анализируем разнообразие источников
        source_counts = defaultdict(int)
        for node in nodes:
            for source in node.meta.get("sources", []):
                source_name = source.get("source", "unknown")
                source_counts[source_name] += 1
        
        total_sources = len(source_counts)
        source_entropy = 0.0
        for count in source_counts.values():
            p = count / total
            source_entropy -= p * math.log2(p) if p > 0 else 0
        
        max_source_entropy = math.log2(total_sources) if total_sources > 1 else 1
        normalized_source_entropy = source_entropy / max_source_entropy if max_source_entropy > 0 else 1
        
        # Итоговое разнообразие
        diversity = (normalized_entropy * 0.6 + normalized_source_entropy * 0.4)
        
        return {
            "diversity": diversity,
            "type_diversity": normalized_entropy,
            "source_diversity": normalized_source_entropy,
            "node_types": dict(type_counts),
            "sources": dict(source_counts)
        }
    
    def get_knowledge_application_potential(self, domain: str) -> Dict[str, Any]:
        """
        Оценивает потенциал применения знаний в указанной области.
        
        Args:
            domain: Область знаний
            
        Returns:
            Dict[str, Any]: Потенциал применения
        """
        # Получаем узлы в домене
        domain_nodes = [node for node in self.nodes.values() if node.domain == domain]
        
        if not domain_nodes:
            return {"error": "no_nodes_in_domain"}
        
        # Оцениваем полноту
        completeness = len(domain_nodes) / 1000  # Условная нормализация
        completeness = min(1.0, completeness)
        
        # Оцениваем связность
        coherence = self.get_knowledge_coherence(domain)["coherence"]
        
        # Оцениваем актуальность
        current_time = time.time()
        avg_age = sum(current_time - node.last_updated for node in domain_nodes) / len(domain_nodes)
        timeliness = max(0.0, 1.0 - (avg_age / (180 * 86400)))  # Нормализация к 0-1 (6 месяцев)
        
        # Оцеживаем разнообразие
        diversity = self.get_knowledge_diversity(domain)["diversity"]
        
        # Рассчитываем общий потенциал
        application_potential = (
            completeness * 0.3 +
            coherence * 0.3 +
            timeliness * 0.2 +
            diversity * 0.2
        )
        
        # Формируем рекомендации
        recommendations = []
        if completeness < 0.5:
            recommendations.append("Расширьте базу знаний, добавив основные концепты области")
        if coherence < 0.5:
            recommendations.append("Улучшите связность знаний, добавив связи между концептами")
        if timeliness < 0.5:
            recommendations.append("Обновите устаревшие знания")
        if diversity < 0.5:
            recommendations.append("Увеличьте разнообразие источников и типов знаний")
        
        return {
            "domain": domain,
            "application_potential": application_potential,
            "completeness": completeness,
            "coherence": coherence,
            "timeliness": timeliness,
            "diversity": diversity,
            "recommendations": recommendations
        }
    
    def get_knowledge_gap_analysis_report(self, domain: str) -> Dict[str, Any]:
        """
        Генерирует подробный отчет об анализе пробелов в знаниях.
        
        Args:
            domain: Область знаний
            
        Returns:
            Dict[str, Any]: Отчет об анализе пробелов
        """
        # Получаем базовый анализ
        gap_analysis = self.get_knowledge_gaps_analysis(domain)
        
        # Получаем дополнительные метрики
        knowledge_quality = self.get_knowledge_quality(domain)
        knowledge_coverage = self.get_knowledge_coverage(self._get_required_concepts(domain))
        
        # Формируем отчет
        report = {
            "timestamp": time.time(),
            "domain": domain,
            "gap_analysis": gap_analysis,
            "knowledge_quality": knowledge_quality,
            "knowledge_coverage": knowledge_coverage,
            "recommendations": [],
            "priority_actions": []
        }
        
        # Формируем рекомендации на основе анализа
        if gap_analysis["weakly_connected_nodes"] > 0:
            report["recommendations"].append(
                f"Укрепите связи для {gap_analysis['weakly_connected_nodes']} концептов с малым количеством связей"
            )
            report["priority_actions"].append(
                "Добавьте связи между слабо связанными концептами"
            )
        
        if gap_analysis["outdated_nodes"] > 0:
            report["recommendations"].append(
                f"Обновите информацию для {gap_analysis['outdated_nodes']} устаревших концептов"
            )
            report["priority_actions"].append(
                "Проверьте актуальность информации для устаревших концептов"
            )
        
        if gap_analysis["contradictory_nodes"] > 0:
            report["recommendations"].append(
                f"Разрешите противоречия для {gap_analysis['contradictory_nodes']} концептов"
            )
            report["priority_actions"].append(
                "Анализ и разрешение выявленных противоречий"
            )
        
        # Добавляем рекомендации из покрытия
        report["recommendations"].extend(knowledge_coverage["recommendations"])
        
        # Определяем общий статус
        if gap_analysis["gap_severity"] < 0.3:
            report["status"] = "healthy"
            report["summary"] = "Пробелов в знаниях не обнаружено. Область знаний хорошо заполнена."
        elif gap_analysis["gap_severity"] < 0.6:
            report["status"] = "warning"
            report["summary"] = "Обнаружены незначительные пробелы в знаниях. Требуется небольшая корректировка."
        else:
            report["status"] = "critical"
            report["summary"] = "Обнаружены серьезные пробелы в знаниях. Требуется срочное заполнение."
        
        return report
    
    def _get_required_concepts(self, domain: str) -> List[str]:
        """
        Возвращает список обязательных концептов для области.
        
        Args:
            domain: Область знаний
            
        Returns:
            List[str]: Список обязательных концептов
        """
        # В реальной системе здесь будет более сложная логика
        # Например, загрузка из справочника или через ML
        required_concepts = {
            "medicine": [
                "анатомия", "физиология", "патология", "диагностика", 
                "лечение", "препараты", "симптомы", "профилактика"
            ],
            "technology": [
                "программирование", "алгоритмы", "базы данных", "сети",
                "искусственный интеллект", "кибербезопасность", "аппаратное обеспечение"
            ],
            "finance": [
                "инвестиции", "бухгалтерия", "анализ", "рынки", "акции",
                "облигации", "портфель", "риск"
            ]
        }
        
        return required_concepts.get(domain.lower(), [])
    
    def get_knowledge_evolution_path(self, start_concept: str, 
                                   end_concept: str) -> Dict[str, Any]:
        """
        Находит путь эволюции от одного концепта к другому.
        
        Args:
            start_concept: Начальный концепт
            end_concept: Конечный концепт
            
        Returns:
            Dict[str, Any]: Путь эволюции
        """
        # Ищем узлы
        start_nodes = self.search_nodes(start_concept, limit=1)
        end_nodes = self.search_nodes(end_concept, limit=1)
        
        if not start_nodes or not end_nodes:
            return {"error": "concepts_not_found"}
        
        start_node = start_nodes[0]
        end_node = end_nodes[0]
        
        # Ищем кратчайший путь
        paths = self.find_path(start_node.id, end_node.id, max_length=5)
        
        if not paths:
            return {"error": "no_evolution_path"}
        
        # Выбираем самый короткий путь
        shortest_path = min(paths, key=len)
        
        # Формируем результат
        evolution_path = []
        for node_id in shortest_path:
            node = self.get_node(node_id)
            if node:
                evolution_path.append({
                    "id": node.id,
                    "name": node.name,
                    "description": node.description,
                    "domain": node.domain
                })
        
        return {
            "start_concept": start_concept,
            "end_concept": end_concept,
            "path_length": len(shortest_path) - 1,
            "evolution_path": evolution_path,
            "historical_changes": self._get_historical_changes(shortest_path)
        }
    
    def _get_historical_changes(self, path: List[str]) -> List[Dict[str, Any]]:
        """
        Возвращает исторические изменения вдоль пути.
        
        Args:
            path: Путь в графе
            
        Returns:
            List[Dict[str, Any]]: Исторические изменения
        """
        changes = []
        
        for i in range(len(path) - 1):
            node1 = self.get_node(path[i])
            node2 = self.get_node(path[i + 1])
            
            if node1 and node2:
                # Ищем связь между узлами
                for edge in self.get_edges(node1.id, direction="source"):
                    if edge.target_id == node2.id:
                        changes.append({
                            "from": node1.name,
                            "to": node2.name,
                            "relation": edge.relation_type,
                            "strength": edge.strength,
                            "timestamp": edge.timestamp
                        })
                        break
        
        return changes
    
    def get_knowledge_innovation_potential(self, domain: str) -> Dict[str, float]:
        """
        Оценивает потенциал инноваций в указанной области знаний.
        
        Args:
            domain: Область знаний
            
        Returns:
            Dict[str, float]: Потенциал инноваций
        """
        # Получаем узлы в домене
        domain_nodes = [node for node in self.nodes.values() if node.domain == domain]
        
        if not domain_nodes:
            return {"innovation_potential": 0.0, "reason": "no_nodes"}
        
        # Оцениваем плотность знаний
        knowledge_density = self.get_knowledge_density(domain)
        
        # Оцениваем разнообразие
        knowledge_diversity = self.get_knowledge_diversity(domain)
        
        # Оцениваем динамику обновлений
        knowledge_trends = self.get_knowledge_trends(domain)
        recent_activity = sum(knowledge_trends["new_nodes"][-3:]) + sum(knowledge_trends["updated_nodes"][-3:])
        
        # Оцениваем наличие пробелов (высокие пробелы могут указывать на потенциал для инноваций)
        gap_analysis = self.get_knowledge_gaps_analysis(domain)
        
        # Рассчитываем потенциал инноваций
        # Высокая плотность и разнообразие + активность обновлений + наличие пробелов = высокий потенциал
        innovation_potential = (
            knowledge_density["domain_density"] * 0.2 +
            knowledge_diversity["diversity"] * 0.3 +
            min(1.0, recent_activity / 10) * 0.2 +  # Нормализуем активность
            gap_analysis["gap_severity"] * 0.3  # Чем больше пробелов, тем выше потенциал
        )
        
        return {
            "innovation_potential": innovation_potential,
            "knowledge_density": knowledge_density["domain_density"],
            "diversity": knowledge_diversity["diversity"],
            "recent_activity": recent_activity,
            "gap_severity": gap_analysis["gap_severity"]
        }
    
    def get_knowledge_synergy_analysis(self, domains: List[str]) -> Dict[str, Any]:
        """
        Анализирует синергетический эффект между областями знаний.
        
        Args:
            domains: Список областей знаний
            
        Returns:
            Dict[str, Any]: Анализ синергии
        """
        if len(domains) < 2:
            return {"error": "need_at_least_two_domains"}
        
        analysis = {
            "domains": domains,
            "pairwise_synergy": {},
            "overall_synergy": 0.0,
            "integration_opportunities": []
        }
        
        # Анализируем каждую пару областей
        total_synergy = 0.0
        pair_count = 0
        
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                domain1 = domains[i]
                domain2 = domains[j]
                
                # Получаем точки интеграции
                integration_points = self.get_knowledge_integration_points(domain1, domain2)
                
                # Рассчитываем синергетический эффект
                synergy_score = self._calculate_synergy_score(integration_points, domain1, domain2)
                
                # Сохраняем результат
                analysis["pairwise_synergy"][f"{domain1}-{domain2}"] = {
                    "synergy_score": synergy_score,
                    "integration_points_count": len(integration_points),
                    "integration_points": integration_points[:3]  # Первые 3 точки
                }
                
                total_synergy += synergy_score
                pair_count += 1
                
                # Сохраняем возможности интеграции
                if integration_points:
                    analysis["integration_opportunities"].append({
                        "domains": [domain1, domain2],
                        "synergy_score": synergy_score,
                        "integration_points": integration_points
                    })
        
        # Рассчитываем общий синергетический эффект
        analysis["overall_synergy"] = total_synergy / pair_count if pair_count > 0 else 0.0
        
        # Сортируем возможности интеграции по синергии
        analysis["integration_opportunities"].sort(
            key=lambda x: x["synergy_score"], 
            reverse=True
        )
        
        return analysis
    
    def _calculate_synergy_score(self, integration_points: List[Dict[str, Any]], 
                              domain1: str, domain2: str) -> float:
        """
        Рассчитывает синергетический эффект между областями.
        
        Args:
            integration_points: Точки интеграции
            domain1: Первая область
            domain2: Вторая область
            
        Returns:
            float: Синергетический эффект
        """
        if not integration_points:
            return 0.0
        
        # Веса для разных типов точек интеграции
        weights = {
            "common_concept": 0.4,
            "direct_connection": 0.3,
            "indirect_connection": 0.3
        }
        
        # Рассчитываем средний вес
        total_weight = 0.0
        for point in integration_points:
            total_weight += weights.get(point["type"], 0.2)
        
        # Нормализуем к 0-1
        synergy_score = min(1.0, total_weight / len(integration_points))
        
        return synergy_score
    
    def get_knowledge_application_scenarios(self, domain: str) -> List[Dict[str, Any]]:
        """
        Генерирует сценарии применения знаний в указанной области.
        
        Args:
            domain: Область знаний
            
        Returns:
            List[Dict[str, Any]]: Сценарии применения
        """
        if not self.ml_unit:
            return []
        
        try:
            # Получаем информацию о домене
            domain_stats = self.get_domain_statistics(domain)
            knowledge_quality = self.get_knowledge_quality(domain)
            
            # Формируем промпт для генерации сценариев
            prompt = (
                f"Сгенерируйте 5 практических сценариев применения знаний в области '{domain}'.\n\n"
                f"Статистика области:\n"
                f"- Количество концептов: {domain_stats['total_nodes']}\n"
                f"- Количество связей: {domain_stats['total_edges']}\n"
                f"- Качество знаний: {knowledge_quality['overall']:.2f}\n\n"
                f"Сценарии должны быть:\n"
                f"- Практическими и применимыми в реальных условиях\n"
                f"- Соответствовать текущему уровню знаний в области\n"
                f"- Учитывать пробелы в знаниях (если они есть)\n"
                f"- Иметь четкие шаги реализации\n\n"
                f"Формат вывода:\n"
                f"1. Название сценария\n"
                f"   - Описание: краткое описание сценария\n"
                f"   - Шаги: список шагов для реализации\n"
                f"   - Ожидаемый результат: что будет достигнуто\n"
                f"   - Потребность в знаниях: какие знания требуются и достаточно ли их в системе\n"
            )
            
            # Генерируем ответ
            response = self.ml_unit.generate_response(
                prompt=prompt,
                max_length=800,
                temperature=0.7,
                top_p=0.9,
                task="text-generation"
            )
            
            # Парсим ответ
            scenarios = self._parse_application_scenarios(response.get("text", ""))
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Ошибка генерации сценариев применения: {e}", exc_info=True)
            return []
    
    def _parse_application_scenarios(self, text: str) -> List[Dict[str, Any]]:
        """
        Парсит сгенерированные сценарии применения.
        
        Args:
            text: Текст сгенерированных сценариев
            
        Returns:
            List[Dict[str, Any]]: Распарсенные сценарии
        """
        scenarios = []
        current_scenario = None
        
        # Разбиваем текст на строки
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Проверяем, начинается ли строка с номера сценария
            if re.match(r'^\d+\.', line):
                # Сохраняем предыдущий сценарий
                if current_scenario:
                    scenarios.append(current_scenario)
                
                # Начинаем новый сценарий
                scenario_name = line.split('.', 1)[1].strip()
                current_scenario = {
                    "name": scenario_name,
                    "description": "",
                    "steps": [],
                    "expected_result": "",
                    "knowledge_requirements": ""
                }
            
            # Проверяем разделы
            elif current_scenario and line.startswith("Описание:"):
                current_scenario["description"] = line.replace("Описание:", "").strip()
            elif current_scenario and line.startswith("Шаги:"):
                # Следующие строки будут шагами
                pass
            elif current_scenario and line.startswith("- "):
                # Это шаг в списке
                current_scenario["steps"].append(line[2:].strip())
            elif current_scenario and line.startswith("Ожидаемый результат:"):
                current_scenario["expected_result"] = line.replace("Ожидаемый результат:", "").strip()
            elif current_scenario and line.startswith("Потребность в знаниях:"):
                current_scenario["knowledge_requirements"] = line.replace("Потребность в знаниях:", "").strip()
        
        # Добавляем последний сценарий
        if current_scenario:
            scenarios.append(current_scenario)
        
        return scenarios
    
    def get_knowledge_evolution_forecast(self, domain: str, 
                                       time_horizon: int = 12) -> Dict[str, Any]:
        """
        Прогнозирует эволюцию знаний в указанной области.
        
        Args:
            domain: Область знаний
            time_horizon: Горизонт прогноза в месяцах
            
        Returns:
            Dict[str, Any]: Прогноз эволюции
        """
        if not self.ml_unit:
            return {"error": "ml_unit_unavailable"}
        
        try:
            # Получаем исторические данные
            trends = self.get_knowledge_trends(domain, period="month")
            
            # Формируем промпт для прогноза
            prompt = (
                f"Проанализируйте историю развития области знаний '{domain}' и сделайте прогноз на ближайшие {time_horizon} месяцев.\n\n"
                f"Исторические данные (количество новых и обновленных концептов по месяцам):\n"
            )
            
            # Добавляем исторические данные
            for i, (interval, new_nodes, updated_nodes) in enumerate(zip(
                trends["intervals"], 
                trends["new_nodes"], 
                trends["updated_nodes"]
            )):
                prompt += f"- {interval}: {new_nodes} новых, {updated_nodes} обновленных\n"
            
            prompt += (
                f"\nНа основе этих данных:\n"
                f"1. Определите текущие тренды в развитии области\n"
                f"2. Прогнозируйте количество новых и обновленных концептов на каждый из следующих {time_horizon} месяцев\n"
                f"3. Выявите потенциальные точки роста и инновации\n"
                f"4. Определите возможные риски и пробелы в знаниях\n"
                f"5. Предложите рекомендации по развитию области\n\n"
                f"Формат вывода:\n"
                f"Текущие тренды: краткое описание текущих трендов\n"
                f"Прогноз: таблица с месяцами и прогнозируемым количеством новых и обновленных концептов\n"
                f"Точки роста: список потенциальных точек роста и инноваций\n"
                f"Риски: список возможных рисков и пробелов\n"
                f"Рекомендации: конкретные рекомендации по развитию области\n"
            )
            
            # Генерируем ответ
            response = self.ml_unit.generate_response(
                prompt=prompt,
                max_length=1000,
                temperature=0.5,
                top_p=0.9,
                task="text-generation"
            )
            
            # Парсим ответ
            forecast = self._parse_evolution_forecast(response.get("text", ""), time_horizon)
            
            return forecast
            
        except Exception as e:
            logger.error(f"Ошибка прогнозирования эволюции знаний: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _parse_evolution_forecast(self, text: str, time_horizon: int) -> Dict[str, Any]:
        """
        Парсит прогноз эволюции знаний.
        
        Args:
            text: Текст прогноза
            time_horizon: Горизонт прогноза
            
        Returns:
            Dict[str, Any]: Распарсенный прогноз
        """
        forecast = {
            "current_trends": "",
            "forecast_table": [],
            "growth_opportunities": [],
            "risks": [],
            "recommendations": []
        }
        
        # Разбиваем текст на секции
        sections = re.split(r'\n(?=[А-Я][а-я]+:)', text)
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # Определяем тип секции
            if section.startswith("Текущие тренды:"):
                forecast["current_trends"] = section.replace("Текущие тренды:", "").strip()
            
            elif section.startswith("Прогноз:"):
                # Парсим таблицу
                lines = section.split('\n')[1:]
                for line in lines:
                    match = re.search(r'(\d+)-й месяц: (\d+) новых, (\d+) обновленных', line)
                    if match:
                        month = int(match.group(1))
                        new_nodes = int(match.group(2))
                        updated_nodes = int(match.group(3))
                        forecast["forecast_table"].append({
                            "month": month,
                            "new_nodes": new_nodes,
                            "updated_nodes": updated_nodes
                        })
            
            elif section.startswith("Точки роста:"):
                items = [item.strip() for item in section.replace("Точки роста:", "").split('\n-') if item.strip()]
                forecast["growth_opportunities"] = items
            
            elif section.startswith("Риски:"):
                items = [item.strip() for item in section.replace("Риски:", "").split('\n-') if item.strip()]
                forecast["risks"] = items
            
            elif section.startswith("Рекомендации:"):
                items = [item.strip() for item in section.replace("Рекомендации:", "").split('\n-') if item.strip()]
                forecast["recommendations"] = items
        
        return forecast
    
    def get_knowledge_gap_closure_plan(self, domain: str) -> Dict[str, Any]:
        """
        Генерирует план закрытия пробелов в знаниях для указанной области.
        
        Args:
            domain: Область знаний
            
        Returns:
            Dict[str, Any]: План закрытия пробелов
        """
        # Получаем анализ пробелов
        gap_analysis = self.get_knowledge_gap_analysis_report(domain)
        
        if "error" in gap_analysis:
            return gap_analysis
        
        if not self.ml_unit:
            return {"error": "ml_unit_unavailable"}
        
        try:
            # Формируем промпт для генерации плана
            prompt = (
                f"Создайте детальный план закрытия пробелов в знаниях для области '{domain}'.\n\n"
                f"Анализ пробелов:\n"
                f"- Общая серьезность: {gap_analysis['gap_analysis']['gap_severity']:.2f}\n"
                f"- Слабо связанные концепты: {gap_analysis['gap_analysis']['weakly_connected_nodes']}\n"
                f"- Устаревшие концепты: {gap_analysis['gap_analysis']['outdated_nodes']}\n"
                f"- Противоречивые концепты: {gap_analysis['gap_analysis']['contradictory_nodes']}\n"
                f"- Качество знаний: {gap_analysis['knowledge_quality']['overall']:.2f}\n"
                f"- Покрытие обязательных концептов: {gap_analysis['knowledge_coverage']['coverage']:.2f}\n\n"
                f"План должен включать:\n"
                f"1. Приоритизированный список действий по закрытию пробелов\n"
                f"2. Оценку трудоемкости каждого действия\n"
                f"3. Рекомендуемую последовательность выполнения\n"
                f"4. Ожидаемые результаты после реализации плана\n"
                f"5. Метрики для оценки успешности плана\n\n"
                f"Формат вывода:\n"
                f"Приоритетные действия:\n"
                f"- [Действие 1]: [Краткое описание], трудоемкость: [низкая/средняя/высокая]\n"
                f"Последовательность:\n"
                f"1. [Действие]\n"
                f"2. [Действие]\n"
                f"...\n"
                f"Ожидаемые результаты:\n"
                f"- [Результат 1]\n"
                f"- [Результат 2]\n"
                f"Метрики успеха:\n"
                f"- [Метрика 1]: [Целевое значение]\n"
            )
            
            # Генерируем ответ
            response = self.ml_unit.generate_response(
                prompt=prompt,
                max_length=800,
                temperature=0.4,
                top_p=0.9,
                task="text-generation"
            )
            
            # Парсим ответ
            plan = self._parse_gap_closure_plan(response.get("text", ""))
            
            # Добавляем информацию из анализа
            plan["domain"] = domain
            plan["current_gap_severity"] = gap_analysis["gap_analysis"]["gap_severity"]
            
            return plan
            
        except Exception as e:
            logger.error(f"Ошибка генерации плана закрытия пробелов: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _parse_gap_closure_plan(self, text: str) -> Dict[str, Any]:
        """
        Парсит план закрытия пробелов.
        
        Args:
            text: Текст плана
            
        Returns:
            Dict[str, Any]: Распарсенный план
        """
        plan = {
            "priority_actions": [],
            "sequence": [],
            "expected_results": [],
            "success_metrics": []
        }
        
        # Разбиваем текст на секции
        sections = re.split(r'\n(?=[А-Я][а-я]+:)', text)
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # Определяем тип секции
            if section.startswith("Приоритетные действия:"):
                items = re.findall(r'-\s*\[(.*?)\]:\s*(.*?),\s*трудоемкость:\s*(\w+)', section)
                for item in items:
                    plan["priority_actions"].append({
                        "action": item[0],
                        "description": item[1],
                        "effort": item[2]
                    })
            
            elif section.startswith("Последовательность:"):
                items = re.findall(r'\d+\.\s*(.*)', section)
                plan["sequence"] = [item.strip() for item in items if item.strip()]
            
            elif section.startswith("Ожидаемые результаты:"):
                items = [item.strip() for item in section.replace("Ожидаемые результаты:", "").split('\n-') if item.strip()]
                plan["expected_results"] = items
            
            elif section.startswith("Метрики успеха:"):
                items = re.findall(r'-\s*\[(.*?)\]:\s*(.*)', section)
                for item in items:
                    plan["success_metrics"].append({
                        "metric": item[0],
                        "target": item[1]
                    })
        
        return plan
    
    def get_knowledge_evolution_impact_analysis(self, domain: str, 
                                              changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Анализирует влияние предполагаемых изменений на эволюцию знаний.
        
        Args:
            domain: Область знаний
            changes: Список предполагаемых изменений
            
        Returns:
            Dict[str, Any]: Анализ влияния
        """
        analysis = {
            "domain": domain,
            "changes": changes,
            "impact_summary": "",
            "positive_effects": [],
            "negative_effects": [],
            "risk_level": "low",  # low, medium, high
            "recommendations": []
        }
        
        try:
            # Анализируем каждое изменение
            total_impact = 0.0
            risks = []
            
            for change in changes:
                # Оцениваем влияние на полноту
                completeness_impact = self._estimate_completeness_impact(change)
                
                # Оцениваем влияние на связность
                coherence_impact = self._estimate_coherence_impact(change)
                
                # Оцениваем влияние на актуальность
                timeliness_impact = self._estimate_timeliness_impact(change)
                
                # Оцениваем риски
                risk_level, risk_description = self._assess_risk(change)
                
                # Сохраняем детали
                change["impact"] = {
                    "completeness": completeness_impact,
                    "coherence": coherence_impact,
                    "timeliness": timeliness_impact,
                    "risk_level": risk_level,
                    "risk_description": risk_description
                }
                
                # Накапливаем общий эффект
                total_impact += (completeness_impact + coherence_impact + timeliness_impact) / 3
                
                # Сохраняем риски
                if risk_level == "high":
                    risks.append(risk_description)
            
            # Определяем общий уровень риска
            if risks:
                if len(risks) > len(changes) * 0.5:
                    analysis["risk_level"] = "high"
                else:
                    analysis["risk_level"] = "medium"
            else:
                analysis["risk_level"] = "low"
            
            # Формируем сводку
            if total_impact > 0.5:
                analysis["impact_summary"] = "Предложенные изменения значительно улучшат знания в области."
            elif total_impact > 0.2:
                analysis["impact_summary"] = "Предложенные изменения умеренно улучшат знания в области."
            else:
                analysis["impact_summary"] = "Предложенные изменения окажут незначительное влияние на знания в области."
            
            # Формируем рекомендации
            if analysis["risk_level"] == "high":
                analysis["recommendations"].append(
                    "Высокий уровень риска. Рекомендуется пересмотреть некоторые изменения или добавить меры по снижению рисков."
                )
            elif analysis["risk_level"] == "medium":
                analysis["recommendations"].append(
                    "Средний уровень риска. Рекомендуется мониторинг после внедрения изменений."
                )
            
            if total_impact < 0.3:
                analysis["recommendations"].append(
                    "Рассмотрите дополнительные изменения для более значительного улучшения знаний."
                )
            
        except Exception as e:
            logger.error(f"Ошибка анализа влияния изменений: {e}", exc_info=True)
            analysis["error"] = str(e)
        
        return analysis
    
    def _estimate_completeness_impact(self, change: Dict[str, Any]) -> float:
        """
        Оценивает влияние изменения на полноту знаний.
        
        Args:
            change: Изменение
            
        Returns:
            float: Влияние на полноту (0.0-1.0)
        """
        # Простая эвристика
        if change["type"] == "add_node":
            return 0.3
        elif change["type"] == "update_node":
            return 0.1
        elif change["type"] == "add_edge":
            return 0.2
        else:
            return 0.0
    
    def _estimate_coherence_impact(self, change: Dict[str, Any]) -> float:
        """
        Оценивает влияние изменения на связность знаний.
        
        Args:
            change: Изменение
            
        Returns:
            float: Влияние на связность (0.0-1.0)
        """
        # Простая эвристика
        if change["type"] == "add_edge":
            return 0.4
        elif change["type"] == "update_node" and "strength" in change.get("changes", {}):
            return 0.2
        else:
            return 0.0
    
    def _estimate_timeliness_impact(self, change: Dict[str, Any]) -> float:
        """
        Оценивает влияние изменения на актуальность знаний.
        
        Args:
            change: Изменение
            
        Returns:
            float: Влияние на актуальность (0.0-1.0)
        """
        # Простая эвристика
        if change["type"] == "update_node" and "description" in change.get("changes", {}):
            return 0.5
        else:
            return 0.0
    
    def _assess_risk(self, change: Dict[str, Any]) -> Tuple[str, str]:
        """
        Оценивает риски, связанные с изменением.
        
        Args:
            change: Изменение
            
        Returns:
            Tuple[str, str]: Уровень риска и описание
        """
        risk_level = "low"
        description = "Низкий риск изменений."
        
        if change["type"] == "update_node":
            if "description" in change.get("changes", {}):
                # Проверяем, не противоречит ли новое описание существующим связям
                if self._check_for_new_contradictions(change):
                    risk_level = "high"
                    description = "Изменение может создать противоречия с существующими знаниями."
                else:
                    risk_level = "medium"
                    description = "Изменение может потребовать обновления связанных концептов."
        
        return risk_level, description
    
    def _check_for_new_contradictions(self, change: Dict[str, Any]) -> bool:
        """
        Проверяет, не создаст ли изменение новые противоречия.
        
        Args:
            change: Изменение
            
        Returns:
            bool: Создаст ли противоречия
        """
        # В реальной системе здесь будет более сложная проверка
        # Например, сравнение семантического сходства с связанными узлами
        return False
    
    def get_knowledge_evolution_simulation(self, domain: str, 
                                         scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Симулирует эволюцию знаний под воздействием различных сценариев.
        
        Args:
            domain: Область знаний
            scenarios: Список сценариев изменений
            
        Returns:
            Dict[str, Any]: Результаты симуляции
        """
        simulation_results = {
            "domain": domain,
            "scenarios": [],
            "comparison": {
                "best_scenario": None,
                "worst_scenario": None
            }
        }
        
        try:
            for scenario in scenarios:
                # Анализируем влияние сценария
                impact_analysis = self.get_knowledge_evolution_impact_analysis(domain, scenario["changes"])
                
                # Сохраняем результат
                scenario_result = {
                    "scenario_name": scenario["name"],
                    "changes": scenario["changes"],
                    "impact_analysis": impact_analysis,
                    "predicted_outcome": self._predict_outcome(impact_analysis)
                }
                
                simulation_results["scenarios"].append(scenario_result)
            
            # Определяем лучший и худший сценарии
            if simulation_results["scenarios"]:
                best = max(simulation_results["scenarios"], 
                          key=lambda x: self._calculate_scenario_score(x["impact_analysis"]))
                worst = min(simulation_results["scenarios"], 
                           key=lambda x: self._calculate_scenario_score(x["impact_analysis"]))
                
                simulation_results["comparison"]["best_scenario"] = best["scenario_name"]
                simulation_results["comparison"]["worst_scenario"] = worst["scenario_name"]
        
        except Exception as e:
            logger.error(f"Ошибка симуляции эволюции знаний: {e}", exc_info=True)
            simulation_results["error"] = str(e)
        
        return simulation_results
    
    def _calculate_scenario_score(self, impact_analysis: Dict[str, Any]) -> float:
        """
        Рассчитывает общий балл сценария на основе анализа влияния.
        
        Args:
            impact_analysis: Анализ влияния
            
        Returns:
            float: Балл сценария
        """
        # Простая эвристика для расчета балла
        score = 0.0
        
        # Оцениваем позитивное влияние
        if "changes" in impact_analysis:
            for change in impact_analysis["changes"]:
                if "impact" in change:
                    impact = change["impact"]
                    score += (impact["completeness"] + impact["coherence"] + impact["timeliness"]) / 3
        
        # Учитываем риски
        risk_penalty = 0.0
        if impact_analysis["risk_level"] == "high":
            risk_penalty = 0.5
        elif impact_analysis["risk_level"] == "medium":
            risk_penalty = 0.2
        
        return max(0.0, score - risk_penalty)
    
    def _predict_outcome(self, impact_analysis: Dict[str, Any]) -> str:
        """
        Прогнозирует результат применения сценария.
        
        Args:
            impact_analysis: Анализ влияния
            
        Returns:
            str: Прогнозируемый результат
        """
        if impact_analysis["risk_level"] == "high":
            return "Высокий риск негативного влияния. Не рекомендуется к применению без модификации."
        elif impact_analysis["risk_level"] == "medium":
            if self._calculate_scenario_score(impact_analysis) > 0.5:
                return "Умеренный риск, но значительная польза. Рекомендуется с мониторингом."
            else:
                return "Умеренный риск при небольшой пользе. Требуется доработка сценария."
        else:
            if self._calculate_scenario_score(impact_analysis) > 0.5:
                return "Низкий риск и значительная польза. Рекомендуется к применению."
            elif self._calculate_scenario_score(impact_analysis) > 0.2:
                return "Низкий риск и умеренная польза. Может быть применен."
            else:
                return "Низкий риск, но незначительная польза. Рассмотрите альтернативы."
    
    def get_knowledge_evolution_strategy(self, domain: str) -> Dict[str, Any]:
        """
        Генерирует стратегию эволюции знаний для указанной области.
        
        Args:
            domain: Область знаний
            
        Returns:
            Dict[str, Any]: Стратегия эволюции
        """
        if not self.ml_unit:
            return {"error": "ml_unit_unavailable"}
        
        try:
            # Получаем анализ текущего состояния
            gap_analysis = self.get_knowledge_gap_analysis_report(domain)
            trends = self.get_knowledge_trends(domain)
            
            # Формируем промпт для генерации стратегии
            prompt = (
                f"Разработайте стратегию эволюции знаний для области '{domain}'.\n\n"
                f"Текущее состояние:\n"
                f"- Серьезность пробелов: {gap_analysis['gap_analysis']['gap_severity']:.2f}\n"
                f"- Качество знаний: {gap_analysis['knowledge_quality']['overall']:.2f}\n"
                f"- Покрытие: {gap_analysis['knowledge_coverage']['coverage']:.2f}\n"
                f"- Направления: {', '.join(gap_analysis['priority_actions'][:3])}\n\n"
                f"Исторические тренды (последние 6 месяцев):\n"
            )
            
            # Добавляем последние 6 месяцев трендов
            for i in range(-6, 0):
                prompt += f"- {trends['intervals'][i]}: {trends['new_nodes'][i]} новых, {trends['updated_nodes'][i]} обновленных\n"
            
            prompt += (
                f"\nСтратегия должна включать:\n"
                f"1. Краткосрочные действия (1-3 месяца)\n"
                f"2. Среднесрочные инициативы (3-6 месяцев)\n"
                f"3. Долгосрочное видение (6-12 месяцев)\n"
                f"4. Распределение ресурсов\n"
                f"5. Метрики успеха и контрольные точки\n\n"
                f"Формат вывода:\n"
                f"Краткосрочные действия:\n"
                f"- [Действие 1]: [Описание], [срок], [ожидаемый результат]\n"
                f"Среднесрочные инициативы:\n"
                f"- [Инициатива 1]: [Описание], [срок], [ожидаемый результат]\n"
                f"Долгосрочное видение:\n"
                f"- [Видение 1]: [Описание], [срок], [ожидаемый результат]\n"
                f"Распределение ресурсов:\n"
                f"- [Ресурс]: [Процент], [обоснование]\n"
                f"Метрики успеха:\n"
                f"- [Метрика 1]: [Целевое значение], [контрольная точка]\n"
            )
            
            # Генерируем ответ
            response = self.ml_unit.generate_response(
                prompt=prompt,
                max_length=1200,
                temperature=0.5,
                top_p=0.9,
                task="text-generation"
            )
            
            # Парсим ответ
            strategy = self._parse_evolution_strategy(response.get("text", ""))
            
            # Добавляем информацию о домене
            strategy["domain"] = domain
            strategy["timestamp"] = time.time()
            
            return strategy
            
        except Exception as e:
            logger.error(f"Ошибка генерации стратегии эволюции: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _parse_evolution_strategy(self, text: str) -> Dict[str, Any]:
        """
        Парсит стратегию эволюции знаний.
        
        Args:
            text: Текст стратегии
            
        Returns:
            Dict[str, Any]: Распарсенная стратегия
        """
        strategy = {
            "short_term_actions": [],
            "medium_term_initiatives": [],
            "long_term_vision": [],
            "resource_allocation": [],
            "success_metrics": []
        }
        
        # Разбиваем текст на секции
        sections = re.split(r'\n(?=[А-Я][а-я]+:)', text)
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # Определяем тип секции
            if section.startswith("Краткосрочные действия:"):
                items = re.findall(r'-\s*\[(.*?)\]:\s*(.*?),\s*([^\]]+),\s*(.*)', section)
                for item in items:
                    strategy["short_term_actions"].append({
                        "action": item[0],
                        "description": item[1],
                        "timeline": item[2],
                        "expected_result": item[3]
                    })
            
            elif section.startswith("Среднесрочные инициативы:"):
                items = re.findall(r'-\s*\[(.*?)\]:\s*(.*?),\s*([^\]]+),\s*(.*)', section)
                for item in items:
                    strategy["medium_term_initiatives"].append({
                        "initiative": item[0],
                        "description": item[1],
                        "timeline": item[2],
                        "expected_result": item[3]
                    })
            
            elif section.startswith("Долгосрочное видение:"):
                items = re.findall(r'-\s*\[(.*?)\]:\s*(.*?),\s*([^\]]+),\s*(.*)', section)
                for item in items:
                    strategy["long_term_vision"].append({
                        "vision": item[0],
                        "description": item[1],
                        "timeline": item[2],
                        "expected_result": item[3]
                    })
            
            elif section.startswith("Распределение ресурсов:"):
                items = re.findall(r'-\s*\[(.*?)\]:\s*([^\s]+),\s*(.*)', section)
                for item in items:
                    strategy["resource_allocation"].append({
                        "resource": item[0],
                        "allocation": item[1],
                        "rationale": item[2]
                    })
            
            elif section.startswith("Метрики успеха:"):
                items = re.findall(r'-\s*\[(.*?)\]:\s*([^\s]+),\s*(.*)', section)
                for item in items:
                    strategy["success_metrics"].append({
                        "metric": item[0],
                        "target": item[1],
                        "checkpoint": item[2]
                    })

        return strategy

    def get_domain_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по доменам знаний."""
        try:
            domains = {}
            for node_id, node in self.nodes.items():
                domain = getattr(node, "meta", {}).get("domain", "unknown")
                if domain not in domains:
                    domains[domain] = {
                        "nodes": 0,
                        "edges": 0,
                        "concepts": 0,
                        "entities": 0
                    }
                
                domains[domain]["nodes"] += 1
                if node.node_type == NodeType.CONCEPT:
                    domains[domain]["concepts"] += 1
                elif node.node_type == NodeType.ENTITY:
                    domains[domain]["entities"] += 1
            
            # Подсчитываем связи по доменам
            for edge in self.edges.values():
                src_node = self.nodes.get(edge.source_id)
                source_domain = getattr(src_node, "meta", {}).get("domain", "unknown") if src_node else "unknown"
                domains.setdefault(source_domain, {"nodes": 0, "edges": 0, "concepts": 0, "entities": 0})
                domains[source_domain]["edges"] += 1
            
            return domains
        except Exception as e:
            logger.error(f"Ошибка получения статистики доменов: {e}")
            return {}

    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает общую статистику графа знаний."""
        try:
            result = {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "node_types": {
                    "concepts": sum(1 for n in self.nodes.values() if n.node_type == NodeType.CONCEPT),
                    "entities": sum(1 for n in self.nodes.values() if n.node_type == NodeType.ENTITY),
                    "facts": sum(1 for n in self.nodes.values() if n.node_type == NodeType.FACT),
                    "supports": sum(1 for e in self.edges.values() if e.relation_type == RelationType.SUPPORTS),
                    "contradicts": sum(1 for e in self.edges.values() if e.relation_type == RelationType.CONTRADICTS),
                    "causes": sum(1 for e in self.edges.values() if e.relation_type == RelationType.CAUSES)
                },
                "domains": list(set(getattr(n, "meta", {}).get("domain", "unknown") for n in self.nodes.values())),
                "last_updated": max((n.last_updated for n in self.nodes.values()), default=time.time()),
                "cache_stats": self._get_cache_statistics()
            }
            # Эмитим нормализованные метрики через ядро (если доступно)
            try:
                if getattr(self, "brain", None) and hasattr(self.brain, "emit_metrics"):
                    metrics = [
                        {
                            "name": "knowledge_graph.total_nodes",
                            "component": "knowledge_graph",
                            "type": "gauge",
                            "value": float(result["total_nodes"]),
                        },
                        {
                            "name": "knowledge_graph.total_edges",
                            "component": "knowledge_graph",
                            "type": "gauge",
                            "value": float(result["total_edges"]),
                        },
                        {
                            "name": "knowledge_graph.domains_count",
                            "component": "knowledge_graph",
                            "type": "gauge",
                            "value": float(len(result.get("domains", [])))
                        },
                    ]
                    # Безопасный вызов
                    try:
                        self.brain.emit_metrics(metrics)
                    except Exception:
                        pass
            except Exception:
                pass
            return result
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            # Возвращаем совместимый по схеме объект, чтобы GUI не падал
            return {
                "total_nodes": len(getattr(self, "nodes", {})),
                "total_edges": len(getattr(self, "edges", {})),
                "node_types": {},
                "domains": [],
                "last_updated": time.time(),
                "cache_stats": {},
                "error": str(e)
            }

 