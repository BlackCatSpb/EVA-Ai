"""
FractalGraph L1/L2 Extensions - Расширения для Pie Architecture

L1: Activation Profile - центроиды профилей активаций
L2: Routing Rule - правила маршрутизации

Использует SQLite для хранения узлов и эмбеддингов.
"""

import numpy as np
import time
import json
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass


@dataclass
class ActivationProfileData:
    """Данные профиля активации."""
    node_id: str
    domain: str
    model_id: str
    quant_profile: str
    centroid: np.ndarray
    avg_entropy: float
    avg_latency_ms: float
    avg_quality: float
    sample_count: int
    last_accessed: float


@dataclass
class RoutingRuleData:
    """Данные правила маршрутизации."""
    node_id: str
    domain: str
    temperature: float
    repeat_penalty: float
    max_tokens: int
    quant_profile: str
    fallback_chain: List[str]
    priority: float
    access_count: int
    success_count: int
    created_at: float
    last_used: Optional[float]


class FractalGraphL1L2:
    """
    Расширение FractalGraph для L1/L2 слоёв Pie Architecture.
    
    Работает с SQLite напрямую для хранения:
    - activation_profile: центроиды (768 float) + метаданные
    - routing_rule: параметры генерации + статистика
    """
    
    def __init__(self, db_path: str):
        """
        Args:
            db_path: Путь к SQLite файлу графа
        """
        self.db_path = db_path
        self.embedding_dim = 768
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Создать таблицы если не существуют."""
        with sqlite3.connect(self.db_path) as conn:
            # Таблица для всех узлов (если не существует)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    level INTEGER DEFAULT 1,
                    embedding BLOB,
                    metadata TEXT,
                    temporal_weight REAL DEFAULT 1.0,
                    domain_lambda REAL DEFAULT 0.05,
                    created_at REAL DEFAULT (CAST(strftime('%s', 'now') AS REAL)),
                    updated_at REAL DEFAULT (CAST(strftime('%s', 'now') AS REAL))
                )
            """)
            
            # Таблица для связей
            conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    created_at REAL DEFAULT (CAST(strftime('%s', 'now') AS REAL)),
                    FOREIGN KEY (source_id) REFERENCES nodes(node_id),
                    FOREIGN KEY (target_id) REFERENCES nodes(node_id)
                )
            """)
            
            # Индексы
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_content ON nodes(content)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
    
    # ==================== L1: Activation Profile ====================
    
    def create_activation_profile(
        self,
        domain: str,
        model_id: str,
        quant_profile: str = "Q4_K_M",
        initial_embedding: Optional[np.ndarray] = None
    ) -> str:
        """
        Создать узел activation_profile.
        
        Args:
            domain: Домен (например, "astrophysics")
            model_id: ID модели (model_a, model_b)
            quant_profile: Профиль квантизации
            initial_embedding: Начальный центроид (768 float)
            
        Returns:
            node_id: ID созданного узла
        """
        import uuid
        
        node_id = f"profile_{domain}_{model_id}_{uuid.uuid4().hex[:8]}"
        content = f"profile_{domain}_{model_id}_{quant_profile}"
        
        if initial_embedding is None:
            initial_embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        else:
            initial_embedding = initial_embedding.astype(np.float32)
            if len(initial_embedding) != self.embedding_dim:
                # Pad or truncate to 768
                if len(initial_embedding) < self.embedding_dim:
                    initial_embedding = np.pad(
                        initial_embedding, 
                        (0, self.embedding_dim - len(initial_embedding))
                    )
                else:
                    initial_embedding = initial_embedding[:self.embedding_dim]
        
        metadata = {
            "domain": domain,
            "model_id": model_id,
            "quant_profile": quant_profile,
            "avg_entropy": 0.0,
            "avg_latency_ms": 0.0,
            "avg_quality": 0.0,
            "sample_count": 0,
            "last_accessed": time.time()
        }
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO nodes 
                (node_id, content, node_type, level, embedding, metadata, temporal_weight, domain_lambda)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                node_id,
                content,
                "activation_profile",
                2,  # L2 уровень
                initial_embedding.tobytes(),
                json.dumps(metadata),
                1.0,
                0.05
            ))
        
        return node_id
    
    def get_activation_profile(
        self,
        domain: str,
        model_id: str
    ) -> Optional[ActivationProfileData]:
        """
        Получить профиль по домену и модели.
        
        Args:
            domain: Домен
            model_id: ID модели
            
        Returns:
            Данные профиля или None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT node_id, embedding, metadata 
                FROM nodes 
                WHERE node_type = 'activation_profile'
                AND json_extract(metadata, '$.domain') = ?
                AND json_extract(metadata, '$.model_id') = ?
                LIMIT 1
            """, (domain, model_id))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            node_id, embedding_bytes, metadata_json = row
            metadata = json.loads(metadata_json)
            centroid = np.frombuffer(embedding_bytes, dtype=np.float32)
            
            return ActivationProfileData(
                node_id=node_id,
                domain=metadata["domain"],
                model_id=metadata["model_id"],
                quant_profile=metadata["quant_profile"],
                centroid=centroid,
                avg_entropy=metadata.get("avg_entropy", 0.0),
                avg_latency_ms=metadata.get("avg_latency_ms", 0.0),
                avg_quality=metadata.get("avg_quality", 0.0),
                sample_count=metadata.get("sample_count", 0),
                last_accessed=metadata.get("last_accessed", 0.0)
            )
    
    def update_activation_profile(
        self,
        profile_id: str,
        new_fingerprint: np.ndarray,
        entropy: float,
        latency_ms: float,
        quality: float = 0.0
    ) -> bool:
        """
        Инкрементально обновить профиль.
        
        new_centroid = (old_centroid * n + new_fingerprint) / (n + 1)
        
        Args:
            profile_id: ID профиля
            new_fingerprint: Новый fingerprint
            entropy: Энтропия генерации
            latency_ms: Задержка в мс
            quality: Оценка качества
            
        Returns:
            True если успешно
        """
        with sqlite3.connect(self.db_path) as conn:
            # Получаем текущие данные
            cursor = conn.execute(
                "SELECT embedding, metadata FROM nodes WHERE node_id = ? AND node_type = 'activation_profile'",
                (profile_id,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return False
            
            embedding_bytes, metadata_json = row
            metadata = json.loads(metadata_json)
            
            n = metadata.get("sample_count", 0)
            old_centroid = np.frombuffer(embedding_bytes, dtype=np.float32)
            new_fingerprint = new_fingerprint.astype(np.float32)
            
            # Нормализуем размерность
            if len(new_fingerprint) != self.embedding_dim:
                if len(new_fingerprint) < self.embedding_dim:
                    new_fingerprint = np.pad(
                        new_fingerprint, 
                        (0, self.embedding_dim - len(new_fingerprint))
                    )
                else:
                    new_fingerprint = new_fingerprint[:self.embedding_dim]
            
            # Вычисляем новый центроид
            if n == 0:
                new_centroid = new_fingerprint
            else:
                new_centroid = (old_centroid * n + new_fingerprint) / (n + 1)
            
            # Обновляем метаданные
            metadata["avg_entropy"] = (metadata.get("avg_entropy", 0.0) * n + entropy) / (n + 1)
            metadata["avg_latency_ms"] = (metadata.get("avg_latency_ms", 0.0) * n + latency_ms) / (n + 1)
            metadata["avg_quality"] = (metadata.get("avg_quality", 0.0) * n + quality) / (n + 1)
            metadata["sample_count"] = n + 1
            metadata["last_accessed"] = time.time()
            
            # Сохраняем
            conn.execute("""
                UPDATE nodes 
                SET embedding = ?, metadata = ?, updated_at = ?, temporal_weight = MIN(1.0, temporal_weight + 0.01)
                WHERE node_id = ?
            """, (
                new_centroid.tobytes(),
                json.dumps(metadata),
                time.time(),
                profile_id
            ))
            
            return True
    
    def find_similar_profiles(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Найти похожие профили по косинусному сходству.
        
        Args:
            query_embedding: Вектор запроса
            top_k: Сколько вернуть
            
        Returns:
            Список (node_id, similarity)
        """
        query_embedding = query_embedding.astype(np.float32)
        if len(query_embedding) != self.embedding_dim:
            if len(query_embedding) < self.embedding_dim:
                query_embedding = np.pad(
                    query_embedding, 
                    (0, self.embedding_dim - len(query_embedding))
                )
            else:
                query_embedding = query_embedding[:self.embedding_dim]
        
        # Нормализуем query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT node_id, embedding FROM nodes WHERE node_type = 'activation_profile'"
            )
            
            for row in cursor:
                node_id, embedding_bytes = row
                centroid = np.frombuffer(embedding_bytes, dtype=np.float32)
                centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
                
                similarity = np.dot(query_norm, centroid_norm)
                results.append((node_id, float(similarity)))
        
        # Сортируем по similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def list_activation_profiles(self) -> List[ActivationProfileData]:
        """Получить все профили активации."""
        profiles = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT node_id, embedding, metadata FROM nodes WHERE node_type = 'activation_profile'"
            )
            
            for row in cursor:
                node_id, embedding_bytes, metadata_json = row
                metadata = json.loads(metadata_json)
                centroid = np.frombuffer(embedding_bytes, dtype=np.float32)
                
                profiles.append(ActivationProfileData(
                    node_id=node_id,
                    domain=metadata["domain"],
                    model_id=metadata["model_id"],
                    quant_profile=metadata["quant_profile"],
                    centroid=centroid,
                    avg_entropy=metadata.get("avg_entropy", 0.0),
                    avg_latency_ms=metadata.get("avg_latency_ms", 0.0),
                    avg_quality=metadata.get("avg_quality", 0.0),
                    sample_count=metadata.get("sample_count", 0),
                    last_accessed=metadata.get("last_accessed", 0.0)
                ))
        
        return profiles
    
    # ==================== L2: Routing Rule ====================
    
    def create_routing_rule(
        self,
        domain: str,
        temperature: float = 0.3,
        repeat_penalty: float = 1.8,
        max_tokens: int = 1024,
        quant_profile: str = "Q4_K_M",
        fallback_chain: Optional[List[str]] = None,
        priority: float = 1.0
    ) -> str:
        """
        Создать правило маршрутизации.
        
        Args:
            domain: Домен
            temperature: Температура генерации
            repeat_penalty: Штраф за повторы
            max_tokens: Максимум токенов
            quant_profile: Профиль квантизации
            fallback_chain: Цепочка fallback'ов
            priority: Приоритет правила
            
        Returns:
            node_id: ID созданного правила
        """
        import uuid
        
        if fallback_chain is None:
            fallback_chain = ["L3_memory", "keyword_response"]
        
        node_id = f"rule_{domain}_{uuid.uuid4().hex[:8]}"
        content = f"rule_{domain}"
        
        metadata = {
            "domain": domain,
            "temperature": temperature,
            "repeat_penalty": repeat_penalty,
            "max_tokens": max_tokens,
            "quant_profile": quant_profile,
            "fallback_chain": fallback_chain,
            "priority": priority,
            "access_count": 0,
            "success_count": 0,
            "created_at": time.time(),
            "last_used": None
        }
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO nodes 
                (node_id, content, node_type, level, embedding, metadata, temporal_weight, domain_lambda)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                node_id,
                content,
                "routing_rule",
                2,  # L2 уровень
                None,  # routing_rule не имеет embedding
                json.dumps(metadata),
                1.0,
                0.02
            ))
        
        return node_id
    
    def get_routing_rule(
        self,
        domain: str
    ) -> Optional[RoutingRuleData]:
        """
        Получить правило по домену.
        
        Args:
            domain: Домен
            
        Returns:
            Данные правила или None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT node_id, metadata 
                FROM nodes 
                WHERE node_type = 'routing_rule'
                AND json_extract(metadata, '$.domain') = ?
                ORDER BY json_extract(metadata, '$.priority') DESC
                LIMIT 1
            """, (domain,))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            node_id, metadata_json = row
            metadata = json.loads(metadata_json)
            
            return RoutingRuleData(
                node_id=node_id,
                domain=metadata["domain"],
                temperature=metadata["temperature"],
                repeat_penalty=metadata["repeat_penalty"],
                max_tokens=metadata["max_tokens"],
                quant_profile=metadata["quant_profile"],
                fallback_chain=metadata.get("fallback_chain", ["L3_memory"]),
                priority=metadata.get("priority", 1.0),
                access_count=metadata.get("access_count", 0),
                success_count=metadata.get("success_count", 0),
                created_at=metadata.get("created_at", 0.0),
                last_used=metadata.get("last_used")
            )
    
    def update_routing_rule_stats(
        self,
        rule_id: str,
        success: bool
    ) -> bool:
        """
        Обновить статистику использования правила.
        
        Args:
            rule_id: ID правила
            success: Был ли ответ успешным
            
        Returns:
            True если успешно
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT metadata FROM nodes WHERE node_id = ? AND node_type = 'routing_rule'",
                (rule_id,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return False
            
            metadata = json.loads(row[0])
            
            metadata["access_count"] = metadata.get("access_count", 0) + 1
            if success:
                metadata["success_count"] = metadata.get("success_count", 0) + 1
            
            # Пересчитываем priority
            acc = metadata["access_count"]
            if acc > 0:
                metadata["priority"] = metadata["success_count"] / acc
            
            metadata["last_used"] = time.time()
            
            conn.execute("""
                UPDATE nodes 
                SET metadata = ?, updated_at = ?, temporal_weight = MIN(1.0, temporal_weight + 0.02)
                WHERE node_id = ?
            """, (
                json.dumps(metadata),
                time.time(),
                rule_id
            ))
            
            return True
    
    def list_routing_rules(self) -> List[RoutingRuleData]:
        """Получить все правила маршрутизации."""
        rules = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT node_id, metadata FROM nodes WHERE node_type = 'routing_rule'"
            )
            
            for row in cursor:
                node_id, metadata_json = row
                metadata = json.loads(metadata_json)
                
                rules.append(RoutingRuleData(
                    node_id=node_id,
                    domain=metadata["domain"],
                    temperature=metadata["temperature"],
                    repeat_penalty=metadata["repeat_penalty"],
                    max_tokens=metadata["max_tokens"],
                    quant_profile=metadata["quant_profile"],
                    fallback_chain=metadata.get("fallback_chain", ["L3_memory"]),
                    priority=metadata.get("priority", 1.0),
                    access_count=metadata.get("access_count", 0),
                    success_count=metadata.get("success_count", 0),
                    created_at=metadata.get("created_at", 0.0),
                    last_used=metadata.get("last_used")
                ))
        
        return rules
    
    # ==================== Relations ====================
    
    def link_rule_to_domain(
        self,
        rule_id: str,
        domain_concept_id: str,
        weight: float = 1.0
    ) -> bool:
        """
        Связать правило с концептом домена.
        
        Args:
            rule_id: ID правила
            domain_concept_id: ID концепта домена
            weight: Вес связи
            
        Returns:
            True если успешно
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO edges (source_id, target_id, relation, weight)
                    VALUES (?, ?, 'applies_to_domain', ?)
                """, (rule_id, domain_concept_id, weight))
            return True
        except sqlite3.IntegrityError:
            return False
    
    def link_profile_to_model(
        self,
        profile_id: str,
        model_root_id: str,
        weight: float = 1.0
    ) -> bool:
        """
        Связать профиль с корневым узлом модели.
        
        Args:
            profile_id: ID профиля
            model_root_id: ID корневого узла
            weight: Вес связи
            
        Returns:
            True если успешно
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO edges (source_id, target_id, relation, weight)
                    VALUES (?, ?, 'derived_from', ?)
                """, (profile_id, model_root_id, weight))
            return True
        except sqlite3.IntegrityError:
            return False
    
    def get_related_rules(
        self,
        domain_concept_id: str
    ) -> List[RoutingRuleData]:
        """
        Получить правила, связанные с концептом домена.
        
        Args:
            domain_concept_id: ID концепта
            
        Returns:
            Список правил
        """
        rules = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT n.node_id, n.metadata 
                FROM nodes n
                JOIN edges e ON n.node_id = e.source_id
                WHERE e.target_id = ? AND e.relation = 'applies_to_domain'
                AND n.node_type = 'routing_rule'
                ORDER BY json_extract(n.metadata, '$.priority') DESC
            """, (domain_concept_id,))
            
            for row in cursor:
                node_id, metadata_json = row
                metadata = json.loads(metadata_json)
                
                rules.append(RoutingRuleData(
                    node_id=node_id,
                    domain=metadata["domain"],
                    temperature=metadata["temperature"],
                    repeat_penalty=metadata["repeat_penalty"],
                    max_tokens=metadata["max_tokens"],
                    quant_profile=metadata["quant_profile"],
                    fallback_chain=metadata.get("fallback_chain", ["L3_memory"]),
                    priority=metadata.get("priority", 1.0),
                    access_count=metadata.get("access_count", 0),
                    success_count=metadata.get("success_count", 0),
                    created_at=metadata.get("created_at", 0.0),
                    last_used=metadata.get("last_used")
                ))
        
        return rules


def create_l1l2_graph(db_path: str) -> FractalGraphL1L2:
    """Фабричный метод создания графа."""
    return FractalGraphL1L2(db_path)
