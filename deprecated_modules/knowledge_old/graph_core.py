"""
Graph Core module for EVA Knowledge Graph
Main class definition, initialization, lifecycle, DB loading, background services, integrity checks.
"""
import os
import logging
import time
import sqlite3
import json
import hashlib
import threading
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from collections import defaultdict, deque

logger = logging.getLogger("eva_ai.knowledge_graph")

def _get_knowledge_graph_types():
    """Lazy import для типов графа знаний."""
    from eva_ai.knowledge.knowledge_graph_types import (
        KnowledgeNode,
        KnowledgeEdge,
        NodeType,
        RelationType,
        safe_json_loads
    )
    return KnowledgeNode, KnowledgeEdge, NodeType, RelationType, safe_json_loads

def _get_hybrid_token_cache():
    """Lazy import для гибридного кэша."""
    try:
        from eva_ai.memory.hybrid_token_cache import HybridTokenCache, get_shared_cache
        return HybridTokenCache, get_shared_cache
    except ImportError:
        logger.warning("HybridTokenCache недоступен, кэширование будет ограничено")
        return None, None

def _get_unified_text_processor():
    """Lazy import для текстового процессора."""
    try:
        from eva_ai.mlearning.unified_text_processor import UnifiedTextProcessor
        return UnifiedTextProcessor
    except ImportError:
        logger.warning("UnifiedTextProcessor недоступен, токенизация будет ограничена")
        return None

def _get_ml_unit():
    """Lazy import для MLUnit."""
    try:
        from eva_ai.mlearning.ml_unit import MLUnit
        return MLUnit
    except ImportError:
        logger.warning("MLUnit недоступен, автоматическое обновление знаний ограничено")
        return None

def _get_entity_extractor():
    """Lazy import для экстрактора сущностей."""
    try:
        from eva_ai.knowledge.context_entity import EntityExtractor
        return EntityExtractor
    except ImportError:
        return None

KnowledgeNode = None
KnowledgeEdge = None
NodeType = None
RelationType = None
safe_json_loads = None
HybridTokenCache = None
get_shared_cache = None
UnifiedTextProcessor = None
MLUnit = None
EntityExtractor = None

def _ensure_imports():
    """Ensures all lazy imports are loaded."""
    global KnowledgeNode, KnowledgeEdge, NodeType, RelationType, safe_json_loads
    global HybridTokenCache, get_shared_cache
    global UnifiedTextProcessor, MLUnit, EntityExtractor
    
    if KnowledgeNode is None:
        KnowledgeNode, KnowledgeEdge, NodeType, RelationType, safe_json_loads = _get_knowledge_graph_types()
    if HybridTokenCache is None:
        HybridTokenCache, get_shared_cache = _get_hybrid_token_cache()
    if UnifiedTextProcessor is None:
        UnifiedTextProcessor = _get_unified_text_processor()
    if MLUnit is None:
        MLUnit = _get_ml_unit()
    if EntityExtractor is None:
        EntityExtractor = _get_entity_extractor()


class KnowledgeGraphCore:
    """Base class for EVA Knowledge Graph - initialization, lifecycle, DB management, background services."""
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None,
                hybrid_cache: Optional[Any] = None, text_processor: Optional[Any] = None,
                max_workers: int = 4):
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "eva_knowledge_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.max_workers = max_workers

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

        self._init_integration_components(hybrid_cache, text_processor)

        self.db_path = os.path.join(self.cache_dir, "knowledge_graph.db")

        self._init_db()

        self._load_nodes()
        self._load_edges()

        self._init_indexes()

        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.futures: Dict[str, Future] = {}

        self.initialized = True
        self.running = False
        self.stop_event = threading.Event()

        self._start_background_services()

        logger.info(f"KnowledgeGraph инициализирован с {len(self.nodes)} узлами и {len(self.edges)} связями")
    
    def _init_integration_components(self, hybrid_cache: Optional[Any], text_processor: Optional[Any]):
        """Инициализирует компоненты для интеграции с другими модулями."""
        _ensure_imports()
        
        self.hybrid_cache = hybrid_cache
        if self.hybrid_cache is None:
            try:
                _, gs_cache = _get_hybrid_token_cache()
                if gs_cache and self.brain is not None:
                    self.hybrid_cache = gs_cache(self.brain, "knowledge_graph")
            except Exception as e:
                logger.warning(f"Не удалось получить гибридный кэш: {e}")
        
        self.text_processor = text_processor
        if self.text_processor is None and UnifiedTextProcessor:
            try:
                config = {
                    'model_name': "eva_ai/core/hf_cache/multilingual-e5-base",
                    'use_async': True
                }
                self.text_processor = UnifiedTextProcessor(
                    brain=self.brain,
                    config=config
                )
                logger.debug("Создан внутренний текстовый процессор")
            except Exception as e:
                logger.warning(f"Не удалось создать текстовый процессор: {e}")
        
        self.ml_unit = None
        if self.brain and hasattr(self.brain, 'ml_unit'):
            self.ml_unit = self.brain.ml_unit
        elif MLUnit:
            try:
                self.ml_unit = MLUnit(brain=self.brain)
                logger.debug("Создан внутренний MLUnit")
            except Exception as e:
                logger.warning(f"Не удалось создать MLUnit: {e}")
        
        self.entity_extractor = EntityExtractor() if EntityExtractor else None
    
    def _init_db(self):
        """Инициализирует базу данных для графа знаний."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
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
            raise RuntimeError(f"Не удалось инициализировать базу данных графа знаний: {e}") from e
    
    def _load_nodes(self):
        """Загружает узлы из базы данных."""
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                    try:
                        node.history = safe_json_loads(row[12]) if len(row) > 12 and row[12] else []
                        node.contradictions = safe_json_loads(row[13]) if len(row) > 13 and row[13] else []
                        node.keyword_index = safe_json_loads(row[14]) if len(row) > 14 and row[14] else []
                        node.concept_index = safe_json_loads(row[15]) if len(row) > 15 and row[15] else []
                    except (IndexError, TypeError):
                        node.history = []
                        node.contradictions = []
                        node.keyword_index = []
                        node.concept_index = []
                    
                    self.nodes[node.id] = node
                
                self.stats["total_nodes"] = len(self.nodes)
                
                logger.info(f"Загружено {len(self.nodes)} узлов в граф знаний")
        except Exception as e:
            logger.error(f"Ошибка загрузки узлов графа знаний: {e}", exc_info=True)
            self.nodes = {}
    
    def _load_edges(self):
        """Загружает связи из базы данных."""
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                
                self.stats["total_edges"] = len(self.edges)
                
                logger.info(f"Загружено {len(self.edges)} связей в граф знаний")
        except Exception as e:
            logger.error(f"Ошибка загрузки связей графа знаний: {e}", exc_info=True)
            self.edges = {}
    
    def _init_indexes(self):
        """Инициализирует индексы для быстрого поиска."""
        def safe_sort_key(item):
            timestamp = item[0]
            if isinstance(timestamp, str):
                try:
                    return float(timestamp)
                except (ValueError, TypeError):
                    return 0.0
            elif isinstance(timestamp, (int, float)):
                return float(timestamp)
            return 0.0

        self.domain_index = defaultdict(list)
        for node in self.nodes.values():
            self.domain_index[node.domain].append(node.id)
        
        self.node_type_index = defaultdict(list)
        for node in self.nodes.values():
            self.node_type_index[node.node_type].append(node.id)
        
        self.relation_index = defaultdict(list)
        for edge in self.edges.values():
            if edge and hasattr(edge, 'relation_type'):
                self.relation_index[edge.relation_type].append(edge.id)
        
        self.temporal_index = []
        for node in self.nodes.values():
            if node.temporal_info:
                self.temporal_index.append((node.timestamp, node.id, "node"))
            if node.last_updated != node.timestamp:
                self.temporal_index.append((node.last_updated, node.id, "node"))

    @property
    def graph(self) -> Dict[str, Any]:
        """Compatibility property - returns nodes as a dict-like structure."""
        return self.nodes

    def _update_indexes(self, node: Optional["KnowledgeNode"] = None,
                       edge: Optional["KnowledgeEdge"] = None):
        """Updates indexes after node/edge changes."""
        if node is not None:
            if node.id not in self.domain_index.get(node.domain, []):
                self.domain_index.setdefault(node.domain, []).append(node.id)
            if node.id not in self.node_type_index.get(node.node_type, []):
                self.node_type_index.setdefault(node.node_type, []).append(node.id)
            if node.temporal_info:
                self.temporal_index.append((node.timestamp, node.id, "node"))
            if node.last_updated != node.timestamp:
                self.temporal_index.append((node.last_updated, node.id, "node"))
        
        if edge is not None:
            if edge.id not in self.relation_index.get(edge.relation_type, []):
                self.relation_index.setdefault(edge.relation_type, []).append(edge.id)
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
        self.stop()
        
        self.monitoring_thread = threading.Thread(
            target=self._background_monitoring,
            name="KnowledgeGraphMonitoring",
            daemon=True
        )
        self.monitoring_thread.start()
        
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
        retry_count = 0
        max_retries = 3
        while not self.stop_event.is_set():
            try:
                self.stop_event.wait(300)
                
                self._check_graph_integrity()
                self._check_for_contradictions()
                self._check_for_outdated_knowledge()
                
                retry_count = 0
                
            except Exception as e:
                retry_count += 1
                logger.error(f"Ошибка в фоновом мониторинге KnowledgeGraph (попытка {retry_count}/{max_retries}): {e}", exc_info=True)
                if retry_count >= max_retries:
                    logger.critical(f"Превышено максимальное количество попыток мониторинга. Требуется вмешательство.")
                    retry_count = 0
    
    def _background_optimization(self):
        """Фоновая оптимизация использования графа знаний."""
        retry_count = 0
        max_retries = 3
        while not self.stop_event.is_set():
            try:
                self.stop_event.wait(600)
                
                self._optimize_indexes()
                self._cleanup_cache()
                
                retry_count = 0
                
            except Exception as e:
                retry_count += 1
                logger.error(f"Ошибка в фоновой оптимизации KnowledgeGraph (попытка {retry_count}/{max_retries}): {e}", exc_info=True)
                if retry_count >= max_retries:
                    logger.critical(f"Превышено максимальное количество попыток оптимизации. Требуется вмешательство.")
                    retry_count = 0
    
    def _check_graph_integrity(self):
        """Проверяет целостность графа знаний."""
        try:
            missing_nodes = []
            for edge_id, edge in self.edges.items():
                if edge.source_id not in self.nodes:
                    missing_nodes.append(edge.source_id)
                if edge.target_id not in self.nodes:
                    missing_nodes.append(edge.target_id)
            
            if missing_nodes:
                logger.warning(f"Обнаружены {len(missing_nodes)} отсутствующих узлов в связях")
                self._attempt_recovery(missing_nodes)
            
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
        logger.info(f"Начало восстановления {len(missing_node_ids)} отсутствующих узлов")
        for node_id in set(missing_node_ids):
            logger.debug(f"Попытка восстановления узла: {node_id}")
            similar_nodes = []
            for existing_id, node in self.nodes.items():
                if len(node_id) == len(existing_id) and sum(c1 == c2 for c1, c2 in zip(node_id, existing_id)) > len(node_id) * 0.8:
                    similar_nodes.append(node)
            
            if similar_nodes:
                logger.debug(f"Найдено {len(similar_nodes)} похожих узлов для {node_id}")
                best_match = max(similar_nodes, key=lambda n: n.strength)
                logger.debug(f"Выбран лучший матч: {best_match.id} (strength={best_match.strength})")
                
                edges_updated = 0
                for edge in self.edges.values():
                    try:
                        if edge.source_id == node_id:
                            edge.source_id = best_match.id
                            self._update_edge_in_db(edge)
                            edges_updated += 1
                        if edge.target_id == node_id:
                            edge.target_id = best_match.id
                            self._update_edge_in_db(edge)
                            edges_updated += 1
                    except Exception as e:
                        logger.warning(f"Ошибка обновления связи при восстановлении: {e}")
                
                logger.info(f"Восстановлена ссылка на узел {node_id} через {best_match.id} ({edges_updated} связей)")
            else:
                logger.debug(f"Похожие узлы не найдены для {node_id}, удаляем связанные edges")
                edges_to_remove = [
                    edge_id for edge_id, edge in self.edges.items()
                    if edge.source_id == node_id or edge.target_id == node_id
                ]
                
                for edge_id in edges_to_remove:
                    try:
                        del self.edges[edge_id]
                        self._remove_edge_from_db(edge_id)
                    except Exception as e:
                        logger.warning(f"Ошибка удаления связи при восстановлении: {e}")
                
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
            
            nodes.sort(key=lambda n: (n.strength, -n.last_updated), reverse=True)
            
            primary_node = nodes[0]
            duplicate_nodes = nodes[1:]
            
            for duplicate in duplicate_nodes:
                for edge_id, edge in list(self.edges.items()):
                    if edge.source_id == duplicate.id:
                        edge.source_id = primary_node.id
                        self._update_edge_in_db(edge)
                    elif edge.target_id == duplicate.id:
                        edge.target_id = primary_node.id
                        self._update_edge_in_db(edge)
                
                if duplicate.description and (not primary_node.description or len(duplicate.description) > len(primary_node.description)):
                    primary_node.description = duplicate.description
                    primary_node.content = duplicate.description
                
                if duplicate.meta and isinstance(duplicate.meta, dict):
                    if not primary_node.meta or not isinstance(primary_node.meta, dict):
                        primary_node.meta = {}
                    for source in duplicate.meta.get('sources', []):
                        if 'sources' not in primary_node.meta:
                            primary_node.meta['sources'] = []
                        primary_node.meta['sources'].append(source)
                
                self._update_node_in_db(primary_node)
                
                del self.nodes[duplicate.id]
                self._remove_node_from_db(duplicate.id)
            
            logger.info(f"Объединено {len(duplicate_nodes)} дубликатов для '{name}'")
    
    def _check_for_contradictions(self):
        """Проверяет граф на наличие противоречий."""
        try:
            for edge_id, edge in self.edges.items():
                if edge.relation_type == RelationType.CONTRADICTS.value:
                    source_node = self.nodes.get(edge.source_id)
                    target_node = self.nodes.get(edge.target_id)
                    
                    if source_node and target_node:
                        for other_edge in self.edges.values():
                            if (other_edge.source_id == edge.source_id and 
                                other_edge.target_id == edge.target_id and
                                other_edge.relation_type != RelationType.CONTRADICTS.value):
                                
                                if not self._is_resolved_contradiction(edge, other_edge):
                                    self._record_contradiction(
                                        edge.source_id, 
                                        edge.target_id,
                                        f"Противоречие между связями {edge.relation_type} и {other_edge.relation_type}",
                                        edge_id,
                                        other_edge.id
                                    )
            
            for node_id, node in self.nodes.items():
                for other_id, other in self.nodes.items():
                    if node_id == other_id:
                        continue
                    
                    if self._are_opposite_concepts(node, other):
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
    
    def _are_opposite_concepts(self, node1: "KnowledgeNode", node2: "KnowledgeNode") -> bool:
        """Проверяет, являются ли концепты противоположными."""
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
        
        name1 = node1.name.lower()
        name2 = node2.name.lower()
        
        for key, value in opposites.items():
            if (key in name1 and value in name2) or (value in name1 and key in name2):
                return True
        
        desc1 = node1.description.lower()
        desc2 = node2.description.lower()
        
        for key, value in opposites.items():
            if (key in desc1 and value in desc2) or (value in desc1 and key in desc2):
                return True
        
        return False
    
    def _is_resolved_contradiction(self, edge1: "KnowledgeEdge", edge2: "KnowledgeEdge") -> bool:
        """Проверяет, разрешено ли противоречие."""
        return "resolution" in edge1.meta or "resolution" in edge2.meta
    
    def _record_contradiction(self, node_id1: str, node_id2: str, evidence: str,
                             edge_id1: Optional[str] = None, 
                             edge_id2: Optional[str] = None):
        """Записывает противоречие в систему."""
        if node_id1 in self.nodes:
            self.nodes[node_id1].add_contradiction(node_id2, evidence)
            self._update_node_in_db(self.nodes[node_id1])
        
        if node_id2 in self.nodes:
            self.nodes[node_id2].add_contradiction(node_id1, evidence)
            self._update_node_in_db(self.nodes[node_id2])
        
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
        """Проверяет знания на устаревание."""
        try:
            outdated_count = 0
            threshold = time.time() - (days_threshold * 86400)
            
            for node_id, node in self.nodes.items():
                if node.last_updated < threshold:
                    outdated_count += 1
                    
                    if node.meta and isinstance(node.meta, dict):
                        if 'status' not in node.meta:
                            node.meta['status'] = 'outdated'
                    
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
                self.hybrid_cache.cleanup()
                logger.debug("Кэш графа знаний очищен")
        except Exception as e:
            logger.error(f"Ошибка очистки кэша: {e}", exc_info=True)
    
    def _update_statistics(self, start_time: float, success: bool):
        """Обновляет статистику запросов."""
        processing_time = time.time() - start_time
        self.stats["total_queries"] += 1
        
        if success:
            self.stats["successful_queries"] += 1
        else:
            self.stats["failed_queries"] += 1
        
        self.stats["total_processing_time"] += processing_time
    
    def _record_history(self, change_type: str, changes: Dict[str, Any], 
                       node_id: Optional[str] = None, edge_id: Optional[str] = None,
                       user_id: Optional[str] = None, source: Optional[str] = None):
        """Записывает изменение в историю."""
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
                1
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Ошибка записи в историю: {e}", exc_info=True)
