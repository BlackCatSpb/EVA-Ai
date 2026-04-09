"""
Fractal Graph V2 - Основной API фрактального графа памяти
"""

import os
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple

from .types import FractalNode, FractalEdge, SemanticGroup, NodeType, RelationType
from .storage import FractalGraphV2, create_fractal_graph
from .embeddings import EmbeddingsManager, create_embeddings_manager
from .gguf_parser import parse_gguf_model, extract_to_graph
from .gguf_extractor import GGUFKnowledgeExtractor, create_extractor
from .gguf_shadow import GGUFShadowProfiler, create_gguf_shadow_profiler
from .hybrid_tokenizer import HybridTokenizer, create_hybrid_tokenizer
from .eva_generator import EVAGenerator, create_eva_generator, GenerationRequest, GenerationResult
from .semantic_context_cache import SemanticContextCache, create_semantic_context_cache
from .eva_container import EVAContainer, create_eva_container, load_eva_container
from .tokenizer import GraphTokenizer, create_graph_tokenizer

logger = logging.getLogger("eva_ai.fractal_graph_v2")

__all__ = [
    # Storage
    'FractalMemoryGraph',
    'FractalGraphV2',
    'create_fractal_memory_graph',
    
    # Types
    'FractalNode',
    'FractalEdge', 
    'SemanticGroup',
    'NodeType',
    'RelationType',
    
    # GGUF
    'parse_gguf_model',
    'extract_to_graph',
    'create_extractor',
    'GGUFKnowledgeExtractor',
    
    # GGUF Shadow (гибридная интеграция)
    'GGUFShadowProfiler',
    'create_gguf_shadow_profiler',
    
    # Hybrid Tokenizer (для EVA контейнера)
    'HybridTokenizer',
    'create_hybrid_tokenizer',
    
    # EVA Generator (единый генератор с виртуальными токенами)
    'EVAGenerator',
    'create_eva_generator',
    'GenerationRequest',
    'GenerationResult',
    
    # Semantic Context Cache (CPU-based semantic search)
    'SemanticContextCache',
    'create_semantic_context_cache',
    
    # EVA Container (unified .eva format)
    'EVAContainer',
    'create_eva_container',
    'load_eva_container',
    
    # Tokenizer
    'GraphTokenizer',
    'create_graph_tokenizer',
    
    # Test Generation
    'GraphMemorySystem',
    'GraphBasedGenerator',
    'GenerationResult',
    'QueryContext',
    'create_graph_memory_system',
]


# ============================================================================
# Main API: FractalMemoryGraph
# ============================================================================

class FractalMemoryGraph:
    
    def __init__(
        self,
        storage_dir: str = None,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        embedding_device: str = "cuda",
        embedding_dim: int = 384
    ):
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(__file__), "fractal_graph_v2_data"
        )
        
        # Инициализация хранилища
        self.storage = create_fractal_graph(
            storage_dir=self.storage_dir,
            embedding_dim=embedding_dim
        )
        
        # Инициализация эмбеддингов
        self.embeddings = create_embeddings_manager(
            model_name=embedding_model,
            device=embedding_device
        )
        
        self._background_thread = None
        self._running = False
        
        logger.info(f"FractalMemoryGraph инициализирован: {self.storage_dir}")
    
    # === ОСНОВНЫЕ ОПЕРАЦИИ ===
    
    def add_node(
        self,
        content: str,
        node_type: str = "concept",
        level: int = 1,
        confidence: float = 0.5,
        metadata: Optional[Dict] = None,
        auto_vectorize: bool = True,
        auto_cluster: bool = True,
        cluster_threshold: float = 0.6
    ) -> FractalNode:
        """
        Добавить узел в граф.
        
        Args:
            content: Текстовое содержание
            node_type: Тип узла (concept, fact, detail, attribute и др.)
            level: Фрактальный уровень (0 - самый глубокий)
            confidence: Уверенность (0-1)
            metadata: Дополнительные метаданные
            auto_vectorize: Автоматически вычислить эмбеддинг
            auto_cluster: Автоматически присоединить к ближайшей группе
            cluster_threshold: Порог similarity для кластеризации
            
        Returns:
            FractalNode
        """
        node = self.storage.add_node(
            content=content,
            node_type=node_type,
            level=level,
            confidence=confidence,
            metadata=metadata,
            auto_cluster=False  # Кластеризация после векторизации
        )
        
        if auto_vectorize:
            self._vectorize_single_node(node.id)
            
            # Инкрементальная кластеризация после векторизации
            if auto_cluster and node.embedding:
                best_group = self.storage._find_nearest_group(
                    node.embedding, level, cluster_threshold
                )
                if best_group:
                    node.parent_group_id = best_group
                    self.storage._save_node(node)
                    if best_group in self.storage.semantic_groups:
                        self.storage.semantic_groups[best_group].member_count += 1
        
        return node
    
    def add_knowledge(
        self,
        subject: str,
        relation: str,
        object_: str,
        subject_level: int = 1,
        object_level: int = 1,
        confidence: float = 0.5
    ) -> Tuple[FractalNode, FractalNode, FractalEdge]:
        """
        Добавить знание в формате S-P-O (Subject-Predicate-Object).
        
        Args:
            subject: Субъект
            relation: Отношение (is_a, part_of, attribute_of и др.)
            object_: Объект
            subject_level: Уровень субъекта
            object_level: Уровень объекта
            
        Returns:
            (subject_node, object_node, edge)
        """
        # Добавляем субъект
        subject_node = self.add_node(
            content=subject,
            node_type="concept",
            level=subject_level,
            confidence=confidence
        )
        
        # Добавляем объект
        object_node = self.add_node(
            content=object_,
            node_type="concept",
            level=object_level,
            confidence=confidence
        )
        
        # Добавляем связь
        edge = self.storage.add_edge(
            source_id=subject_node.id,
            target_id=object_node.id,
            relation_type=relation,
            weight=confidence
        )
        
        return subject_node, object_node, edge
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 0.5
    ) -> Optional[FractalEdge]:
        """Добавить связь между узлами."""
        return self.storage.add_edge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight
        )
    
    def create_group(
        self,
        name: str,
        member_ids: List[str],
        level: int = 2
    ) -> SemanticGroup:
        """Создать семантическую группу (образ)."""
        return self.storage.create_semantic_group(
            name=name,
            member_ids=member_ids,
            level=level
        )
    
    # === ПОИСК ===
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        min_level: int = 2,
        min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Семантический поиск по запросу.
        
        Args:
            query: Текстовый запрос
            top_k: Количество результатов
            min_level: Минимальный уровень для поиска
            min_similarity: Минимальная схожесть (по умолчанию 0.5 - повышено)
            
        Returns:
            List of {node, similarity, group}
        """
        # Векторизуем запрос
        query_emb = self.embeddings.encode_single(query, normalize=True)
        
        if query_emb is None:
            return []
        
        # Поиск в графе (берём больше для фильтрации)
        results = self.storage.semantic_search(
            query_embedding=query_emb.tolist(),
            top_k=min(top_k * 3, 30),  # Берём больше для фильтрации
            min_level=min_level
        )
        
        # Форматируем и фильтруем результаты
        formatted = []
        for node_id_or_group_id, similarity, group_id in results:
            # Фильтр по минимальной схожести
            if similarity < min_similarity:
                continue
                
            if node_id_or_group_id in self.storage.nodes:
                node = self.storage.nodes[node_id_or_group_id]
                formatted.append({
                    "type": "node",
                    "id": node.id,
                    "content": node.content,
                    "node_type": node.node_type,
                    "level": node.level,
                    "confidence": node.confidence,
                    "similarity": similarity,
                    "group_id": group_id
                })
            elif node_id_or_group_id in self.storage.semantic_groups:
                group = self.storage.semantic_groups[node_id_or_group_id]
                # Получаем членов группы
                members = self.storage.get_group_members(group.id)
                formatted.append({
                    "type": "group",
                    "id": group.id,
                    "name": group.name,
                    "member_count": len(members),
                    "avg_confidence": group.avg_confidence,
                    "similarity": similarity,
                    "members": [m.content[:50] for m in members[:5]]
                })
        
        return formatted
    
    def keyword_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[FractalNode]:
        """Поиск по ключевым словам."""
        node_ids = self.storage.keyword_search(query, top_k)
        return [self.storage.nodes[nid] for nid in node_ids if nid in self.storage.nodes]
    
    def get_context(self, node_id: str) -> Dict[str, Any]:
        """Получить контекст узла (группа, связи, атрибуты)."""
        return self.storage.get_node_context(node_id)
    
    # === GGUF МОДЕЛИ ===
    
    def load_gguf_knowledge(self, model_path: str) -> Dict[str, Any]:
        """
        Извлечь знания из GGUF модели и добавить в граф.
        
        Args:
            model_path: Путь к GGUF файлу
            
        Returns:
            Результат добавления
        """
        return extract_to_graph(model_path, self.storage)
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """Получить информацию о GGUF модели."""
        info = parse_gguf_model(model_path)
        return {
            "architecture": info.architecture,
            "model_type": info.model_type,
            "vocab_size": info.vocab_size,
            "hidden_size": info.hidden_size,
            "num_layers": info.num_layers,
            "num_attention_heads": info.num_attention_heads,
            "max_context": info.max_position_embeddings,
            "file_size": info.file_size
        }
    
    # === ВЕКТОРИЗАЦИЯ ===
    
    def _vectorize_single_node(self, node_id: str):
        """Векторизовать один узел."""
        if node_id not in self.storage.nodes:
            return
        
        node = self.storage.nodes[node_id]
        if node.embedding is not None:
            return  # Уже векторизован
        
        emb = self.embeddings.encode_single(node.content, normalize=True)
        if emb is not None:
            node.embedding = emb.tolist()
            self.storage._save_node(node)
    
    def vectorize_all(self, level_filter: int = None):
        """Векторизовать все узлы."""
        nodes_to_vectorize = []
        
        for node_id, node in self.storage.nodes.items():
            if node.embedding is None:
                if level_filter is None or node.level >= level_filter:
                    nodes_to_vectorize.append(node)
        
        if not nodes_to_vectorize:
            logger.info("Все узлы уже векторизованы")
            return
        
        logger.info(f"Векторизация {len(nodes_to_vectorize)} узлов...")
        
        texts = [node.content for node in nodes_to_vectorize]
        embeddings = self.embeddings.encode(texts, normalize=True)
        
        for node, emb in zip(nodes_to_vectorize, embeddings):
            node.embedding = emb.tolist()
            self.storage._save_node(node)
        
        logger.info("Векторизация завершена")
    
    def vectorize_groups(self):
        """Векторизовать семантические группы (вычислить центроиды)."""
        for group_id, group in self.storage.semantic_groups.items():
            members = self.storage.get_group_members(group_id)
            
            embeddings = []
            for member in members:
                if member.embedding:
                    import numpy as np
                    embeddings.append(np.array(member.embedding))
            
            if embeddings:
                import numpy as np
                centroid = np.mean(embeddings, axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                
                group.embedding = centroid.tolist()
                group.avg_confidence = np.mean([m.confidence for m in members])
                self.storage._save_group(group)
        
        logger.info("Группы векторизованы")
    
    # === КЛАСТЕРИЗАЦИЯ ===
    
    def auto_cluster(
        self,
        level: int = 1,
        threshold: float = 0.5,
        method: str = "agglomerative"
    ):
        """
        Автоматическая кластеризация узлов уровня.
        
        Args:
            level: Уровень для кластеризации
            threshold: Порог similarity для объединения в группу
            method: Метод кластеризации (agglomerative, dbscan, simple)
        """
        clusters = self.storage.cluster_nodes(
            level=level,
            threshold=threshold,
            method=method
        )
        
        created_groups = 0
        for cluster_name, member_ids in clusters.items():
            if not member_ids:
                continue
            
            # Создаём группу
            group = self.storage.create_semantic_group(
                name=cluster_name,
                member_ids=member_ids,
                level=level + 1
            )
            
            # Векторизуем группу (центроид)
            members = [self.storage.nodes[mid] for mid in member_ids if mid in self.storage.nodes]
            if members:
                import numpy as np
                embeddings = [np.array(m.embedding) for m in members if m.embedding]
                if embeddings:
                    centroid = np.mean(embeddings, axis=0)
                    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                    group.embedding = centroid.tolist()
                    self.storage._save_group(group)
            
            created_groups += 1
        
        logger.info(f"Создано {created_groups} групп методом {method}")
        
        return created_groups
    
    # === ПРОТИВОРЕЧИЯ ===
    
    def check_contradiction(self, content: str, group_id: str = None) -> Dict[str, Any]:
        """
        Проверить текст на противоречие с группой.
        
        Args:
            content: Текст для проверки
            group_id: ID группы (опционально)
            
        Returns:
            {is_contradiction, distance, suggestions}
        """
        # Векторизуем текст
        emb = self.embeddings.encode_single(content, normalize=True)
        
        if emb is None:
            return {"is_contradiction": False, "error": "no embedding"}
        
        # Если группа не указана - ищем ближайшую
        if group_id is None:
            results = self.storage.semantic_search(emb.tolist(), top_k=1, min_level=1)
            if results and results[0][2]:  # group_id
                group_id = results[0][2]
            # Или ищем ближайший узел
            elif results:
                node_id = results[0].get('id')
                if node_id:
                    node = self.storage.nodes.get(node_id)
                    group_id = node.parent_group_id if node else None
        
        if group_id:
            is_contr, distance = self.storage.detect_contradiction(
                emb.tolist(), group_id, threshold=0.7
            )
            return {
                "is_contradiction": is_contr,
                "distance": distance,
                "group_id": group_id
            }
        
        return {"is_contradiction": False, "reason": "no group found"}
    
    def resolve_contradiction(self, node_id: str, resolution: str = "remove"):
        """
        Разрешить противоречие.
        
        Args:
            node_id: ID противоречивого узла
            resolution: one of "remove", "merge", "keep"
        """
        if node_id not in self.storage.nodes:
            return {"error": "node not found"}
        
        node = self.storage.nodes[node_id]
        
        if resolution == "remove":
            self.storage.mark_contradiction(node_id, "removed by resolution")
            return {"status": "removed", "node_id": node_id}
        
        elif resolution == "keep":
            node.is_contradiction = False
            node.confidence = 1.0  # Подтверждено
            self.storage._save_node(node)
            return {"status": "confirmed", "node_id": node_id}
        
        return {"status": "unknown_resolution"}
    
    def self_dialogue(self, new_knowledge: str) -> Dict[str, Any]:
        """
        Самодиалог - автоматическая верификация нового знания.
        
        Согласно спецификации:
        1. Проверяем противоречие с существующими группами
        2. Ищем подтверждающие или опровергающие факты в графе
        3. Разрешаем противоречие на основе анализа
        
        Args:
            new_knowledge: Новое знание для проверки
            
        Returns:
            {confirmed, action, reasoning, new_nodes}
        """
        # 1. Проверяем на противоречие
        check_result = self.check_contradiction(new_knowledge)
        
        if not check_result.get("is_contradiction"):
            # Нет противоречия - просто добавляем знание
            return {
                "confirmed": True,
                "action": "add",
                "reasoning": "No contradiction detected",
                "new_nodes": []
            }
        
        # 2. Противоречие найдено - анализируем
        group_id = check_result.get("group_id")
        group = self.storage.semantic_groups.get(group_id)
        
        if not group:
            return {"confirmed": False, "action": "reject", "reasoning": "No group found"}
        
        # 3. Ищем связанные факты в графе
        related_facts = self._search_related_facts(new_knowledge, group_id)
        
        # 4. Анализ: есть ли подтверждающие факты?
        confirming_facts = [f for f in related_facts if f.get("similarity", 0) > 0.7]
        
        if len(confirming_facts) >= 2:
            # Много подтверждений - вероятно знание верное
            # Добавляем как новый контекстный узел
            new_node = self.add_node(
                content=new_knowledge,
                node_type="context",
                level=2,
                confidence=0.3,  # Низкая уверенность initially
                metadata={"source": "self_dialogue", "parent_group": group_id}
            )
            
            return {
                "confirmed": True,
                "action": "add_as_context",
                "reasoning": f"Found {len(confirming_facts)} confirming facts",
                "new_nodes": [new_node.id]
            }
        
        elif len(confirming_facts) == 1:
            # Один подтверждающий факт - неопределённо
            # Добавляем с низкой уверенностью
            new_node = self.add_node(
                content=new_knowledge,
                node_type="context",
                level=1,
                confidence=0.2,
                metadata={"source": "self_dialogue", "needs_verification": True}
            )
            
            return {
                "confirmed": False,
                "action": "add_uncertain",
                "reasoning": f"Only {len(confirming_facts)} confirming fact",
                "new_nodes": [new_node.id]
            }
        
        else:
            # Нет подтверждений - скорее всего ошибка
            # Помечаем как противоречие
            return {
                "confirmed": False,
                "action": "reject",
                "reasoning": "No confirming facts found",
                "new_nodes": []
            }
    
    def _search_related_facts(self, query: str, group_id: str, top_k: int = 10) -> List[Dict]:
        """Поиск связанных фактов в группе."""
        results = self.semantic_search(query, top_k=top_k, min_level=1)
        
        # Фильтруем только факты из той же группы
        related = []
        for r in results:
            node_id = r.get("id")
            if node_id and node_id in self.storage.nodes:
                node = self.storage.nodes[node_id]
                if node.parent_group_id == group_id:
                    related.append(r)
        
        return related
    
    # === СТАТИСТИКА ===
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику графа."""
        base_stats = self.storage.get_stats()
        
        # Дополнительная статистика
        base_stats["embedding_cache_size"] = self.embeddings.get_cache_size()
        
        # Распределение по уровням
        level_dist = {}
        for node_id, node in self.storage.nodes.items():
            level_dist[node.level] = level_dist.get(node.level, 0) + 1
        base_stats["nodes_by_level"] = level_dist
        
        return base_stats
    
    def get_node(self, node_id: str) -> Optional[FractalNode]:
        """Получить узел по ID."""
        return self.storage.nodes.get(node_id)
    
    def get_all_nodes(self, level: int = None, node_type: str = None) -> List[FractalNode]:
        """Получить все узлы с фильтрацией."""
        nodes = list(self.storage.nodes.values())
        
        if level is not None:
            nodes = [n for n in nodes if n.level == level]
        
        if node_type is not None:
            nodes = [n for n in nodes if n.node_type == node_type]
        
        return nodes
    
    def get_groups(self, level: int = None) -> List[SemanticGroup]:
        """Получить семантические группы."""
        groups = list(self.storage.semantic_groups.values())
        
        if level is not None:
            groups = [g for g in groups if g.level == level]
        
        return groups
    
    # === ИНТЕГРАЦИЯ СО СТАРОЙ СИСТЕМОЙ ===
    
    def save_experience(
        self,
        query: str,
        response: str,
        model_used: str,
        quality_score: float = 0.5
    ) -> str:
        """
        Сохранить опыт (query/response) в граф.
        Аналог UnifiedFractalMemory.save_experience()
        
        Args:
            query: Запрос пользователя
            response: Ответ системы
            model_used: Какая модель использовалась (model_a, model_b, model_c, web_ui)
            quality_score: Оценка качества (0-1)
            
        Returns:
            ID созданного query узла
        """
        # Создаём узел для query
        query_node = self.add_node(
            content=query[:500],  # Ограничиваем длину
            node_type="query",
            level=2,
            confidence=quality_score,
            metadata={
                "source": "experience",
                "model": model_used,
                "timestamp": time.time()
            },
            auto_vectorize=True
        )
        
        # Создаём узел для response
        if response and len(response) > 3:
            response_node = self.add_node(
                content=response[:1000],
                node_type="response",
                level=2,
                confidence=quality_score,
                metadata={
                    "source": "experience", 
                    "model": model_used,
                    "timestamp": time.time()
                },
                auto_vectorize=True
            )
            
            # Связываем query -> response
            self.storage.add_edge(
                source_id=query_node.id,
                target_id=response_node.id,
                relation_type="generated_by",
                weight=quality_score
            )
        
        return query_node.id
    
    def get_context_for_query(self, query: str, max_length: int = 512, min_similarity: float = 0.5) -> str:
        """
        Получить контекст для запроса.
        Аналог UnifiedFractalMemory.get_context_for_query()
        
        Args:
            query: Запрос пользователя
            max_length: Максимальная длина контекста
            min_similarity: Минимальная схожесть для включения в контекст
            
        Returns:
            Текстовый контекст из графа
        """
        # Семантический поиск с фильтрацией по схожести
        results = self.semantic_search(query, top_k=10, min_level=1, min_similarity=min_similarity)
        
        if not results:
            return ""
        
        # Фильтрация мусора и формирование контекста
        context_parts = []
        template_patterns = [
            'продолжим разговор', 'перспективы развития',
            '###', '##', 'особенности данного',
            'q:', 'a:', 'пример:'
        ]
        
        for r in results:
            content = r.get('content', '')
            if not content:
                continue
            
            # Проверка на мусор
            content_lower = content.lower()
            is_garbage = any(p in content_lower for p in template_patterns)
            if is_garbage:
                continue
            
            # Проверка минимальной длины
            if len(content) < 30:
                continue
            
            context_parts.append(content)
        
        context = "\n".join(context_parts)
        
        # Обрезаем по длине
        if len(context) > max_length:
            context = context[:max_length] + "..."
        
        return context
    
    def retrieve_knowledge(self, query: str, top_k: int = 5, min_similarity: float = 0.5) -> List[Dict[str, Any]]:
        """
        Извлечь знания по запросу.
        Аналог UnifiedFractalMemory.retrieve_knowledge()
        
        Args:
            query: Запрос
            top_k: Количество результатов
            min_similarity: Минимальная схожесть
            
        Returns:
            List of {node_id, content, similarity, level}
        """
        results = self.semantic_search(query, top_k=top_k, min_level=1, min_similarity=min_similarity)
        
        knowledge = []
        template_patterns = [
            'продолжим разговор', 'перспективы развития',
            '###', '##', 'особенности данного',
            'q:', 'a:', 'пример:'
        ]
        
        for r in results:
            content = r.get('content', '')
            if not content:
                continue
            
            content_lower = content.lower()
            if any(p in content_lower for p in template_patterns):
                continue
            if len(content) < 30:
                continue
            
            knowledge.append({
                "node_id": r.get("id"),
                "content": content,
                "similarity": r.get("similarity"),
                "level": r.get("level")
            })
        
        return knowledge
    
    # === УПРАВЛЕНИЕ LLAMA ИНСТАНСАМИ ===
    
    def register_model_instance(self, model_type: str, llama_instance):
        """
        Зарегистрировать Llama инстанс модели.
        Аналог UnifiedFractalMemory.register_model_instance()
        
        Args:
            model_type: Тип модели (model_a, model_b, model_c)
            llama_instance: Llama инстанс из llama_cpp
        """
        if not hasattr(self, '_model_instances'):
            self._model_instances: Dict[str, Any] = {}
        
        self._model_instances[model_type] = llama_instance
        logger.info(f"Зарегистрирован Llama инстанс: {model_type}")
    
    def get_model_instance(self, model_type: str):
        """Получить Llama инстанс модели."""
        if hasattr(self, '_model_instances'):
            return self._model_instances.get(model_type)
        return None
    
    def get_model_context(self, model_type: str) -> Dict[str, Any]:
        """
        Получить контекст для конкретной модели.
        Аналог UnifiedFractalMemory.get_model_context()
        """
        # Ищем узлы связанные с этой моделью
        model_nodes = self.get_all_nodes(node_type=model_type)
        
        return {
            "model_type": model_type,
            "nodes_count": len(model_nodes),
            "nodes": [
                {"id": n.id, "content": n.content[:50], "level": n.level}
                for n in model_nodes[:5]
            ]
        }
    
    def get_static_models(self) -> List[Dict[str, Any]]:
        """
        Получить информацию о статичных моделях.
        Аналог UnifiedFractalMemory.get_static_models()
        """
        models = []
        for model_type in ['model_a', 'model_b', 'model_c']:
            nodes = self.get_all_nodes(node_type=model_type)
            if nodes:
                models.append({
                    "type": model_type,
                    "content": nodes[0].content[:100],
                    "nodes_count": len(nodes)
                })
        
        return models


def create_fractal_memory_graph(
    storage_dir: str = None,
    embedding_device: str = "cuda"
) -> FractalMemoryGraph:
    """Фабричная функция для создания фрактального графа памяти."""
    return FractalMemoryGraph(
        storage_dir=storage_dir,
        embedding_device=embedding_device
    )


# ============================================================================
# Test Generation System (separate module)
# ============================================================================

# Lazy import to avoid circular dependencies
def get_graph_memory_system_class():
    """Get GraphMemorySystem class without circular import."""
    from .test_generation import create_graph_memory_system as _create
    return _create()


# === ЭКСПОРТ ТИПОВ ===
__all__ = [
    # Main
    'FractalMemoryGraph',
    'create_fractal_memory_graph',
    
    # Types
    'FractalNode',
    'FractalEdge', 
    'SemanticGroup',
    'NodeType',
    'RelationType',
    
    # GGUF
    'parse_gguf_model',
    'extract_to_graph',
    'create_extractor',
    'GGUFKnowledgeExtractor',
    
    # Tokenizer
    'GraphTokenizer',
    'create_graph_tokenizer',
    
    # Test System (lazy)
    'get_graph_memory_system_class',
]