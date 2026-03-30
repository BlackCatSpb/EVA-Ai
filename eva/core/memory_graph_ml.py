"""
MemoryGraphML - Машинное обучение на основе графа памяти ЕВА
Интегрирует фрактальную структуру графа памяти в процесс обучения моделей
"""

import os
import sys
import time
import json
import logging
import threading
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Set, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

if TYPE_CHECKING:
    from eva.fractal.entity_fractal_store import EntityFractalStore

# Torch imports for GPU-based embeddings
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    SentenceTransformer = None

logger = logging.getLogger("eva.memory_graph_ml")


@dataclass
class GraphEmbedding:
    """Векторное представление узла графа"""
    node_id: str
    node_type: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AmbiguousEntity:
    """Сущность с неоднозначным значением"""
    term: str
    possible_meanings: List[str]
    context: str
    timestamp: float = field(default_factory=time.time)
    clarification_history: List[Dict] = field(default_factory=list)
    resolved_meaning: Optional[str] = None
    confidence: float = 0.5


@dataclass
class ClarificationRequest:
    """Запрос на уточнение значения сущности"""
    entity_term: str
    question: str
    context: str
    possible_meanings: List[str]
    timestamp: float = field(default_factory=time.time)
    answered: bool = False
    selected_meaning: Optional[str] = None


@dataclass
class GraphPattern:
    """Паттерн, извлеченный из графа памяти"""
    pattern_id: str
    nodes: List[str]
    relations: List[Tuple[str, str, str]]  # (from, to, relation_type)
    frequency: int
    confidence: float
    context: str
    embedding: Optional[np.ndarray] = None


class MemoryGraphML:
    """
    ML система на основе графа памяти ЕВА
    
    Особенности:
    - Извлечение паттернов из фрактальной структуры графа
    - Векторные представления узлов (embeddings)
    - Обучение на связях и отношениях
    - Интеграция с генерацией и рассуждением
    """
    
    def __init__(self, brain, config: Optional[Dict] = None):
        self.brain = brain
        self.config = config or {}
        
        # Хранилище данных
        self.embeddings: Dict[str, GraphEmbedding] = {}
        self.patterns: List[GraphPattern] = []
        self.training_data: List[Dict] = []
        
        # Параметры
        self.embedding_dim = self.config.get('embedding_dim', 128)
        self.max_patterns = self.config.get('max_patterns', 1000)
        self.min_pattern_frequency = self.config.get('min_pattern_frequency', 3)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        
        # Состояние
        self.is_initialized = False
        self.last_update = 0
        self.update_interval = self.config.get('update_interval', 300)  # 5 минут
        
        # Фрактальная структура
        self.fractal_levels = self.config.get('fractal_levels', 3)
        self.level_weights = [1.0, 0.7, 0.5, 0.3]  # Веса уровней
        
        # Гибридный кэш для токенизации
        self._hybrid_cache = None
        
        # Sentence-transformer для CPU embeddings
        self._st_model = None
        self._st_model_name = self.config.get('st_model', 'paraphrase-multilingual-MiniLM-L12-v2')
        self._init_st_model()
        
        # Graph property for external access compatibility
        self._graph = self.embeddings
        
        logger.info("MemoryGraphML инициализирован")
    
    @property
    def graph(self):
        """Property for external access to the graph structure."""
        return self._graph
    
    def _init_st_model(self):
        """Инициализирует sentence-transformer для CPU embeddings."""
        if not ST_AVAILABLE:
            return
        try:
            self._st_model = SentenceTransformer(self._st_model_name, device='cpu')
            self.embedding_dim = self._st_model.get_sentence_embedding_dimension()
            logger.info(f"Sentence-transformer загружен: {self._st_model_name}, dim={self.embedding_dim}")
        except Exception as e:
            logger.warning(f"Не удалось загрузить sentence-transformer: {e}")
            self._st_model = None
    
    def initialize(self) -> bool:
        """Инициализация и загрузка данных из графа памяти"""
        try:
            if not self.brain or not hasattr(self.brain, 'knowledge_graph'):
                logger.warning("KnowledgeGraph недоступен")
                return False
            
            # Пытаемся получить гибридный кэш
            self._get_hybrid_cache()
            
            self._load_graph_structure()
            self._extract_patterns()
            self._compute_embeddings()
            
            self.is_initialized = True
            self.last_update = time.time()
            
            logger.info(f"MemoryGraphML инициализирован: {len(self.embeddings)} embeddings, {len(self.patterns)} паттернов")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации MemoryGraphML: {e}")
            return False
    
    def _load_graph_structure(self):
        """Загрузка структуры графа"""
        try:
            if not hasattr(self, 'brain') or not self.brain:
                return
            kg = getattr(self.brain, 'knowledge_graph', None)
            if not kg:
                return
            
            # Получаем все концепты
            all_concepts = kg.get_all_concepts() if hasattr(kg, 'get_all_concepts') else []
            
            logger.debug(f"Загружено {len(all_concepts)} концептов из графа")
            
        except Exception as e:
            logger.debug(f"Ошибка загрузки структуры: {e}")
    
    def _extract_patterns(self):
        """Извлечение паттернов из графа"""
        try:
            if not hasattr(self, 'brain') or not self.brain:
                return
            kg = getattr(self.brain, 'knowledge_graph', None)
            if not kg:
                return
            
            # Ищем частые пути (триады)
            patterns_found = []
            
            if hasattr(kg, 'get_all_relations'):
                relations = kg.get_all_relations()
                
                # Строим индекс
                node_relations = defaultdict(list)
                for rel in relations:
                    from_node = rel.get('from') or rel.get('source')
                    to_node = rel.get('to') or rel.get('target')
                    rel_type = rel.get('type') or rel.get('relation_type')
                    
                    if from_node and to_node and rel_type:
                        node_relations[from_node].append((to_node, rel_type))
                
                # Ищем паттерны (пути длины 2)
                pattern_counts = defaultdict(int)
                for start_node, first_level in node_relations.items():
                    for mid_node, rel1 in first_level:
                        if mid_node in node_relations:
                            for end_node, rel2 in node_relations[mid_node]:
                                if end_node != start_node:  # Избегаем циклов
                                    pattern_key = f"{rel1}->{rel2}"
                                    pattern_counts[pattern_key] += 1
                
                # Создаем паттерны
                for pattern_key, count in pattern_counts.items():
                    if count >= self.min_pattern_frequency:
                        parts = pattern_key.split("->")
                        if len(parts) == 2:
                            patterns_found.append(GraphPattern(
                                pattern_id=f"pattern_{hash(pattern_key) % 100000}",
                                nodes=[],
                                relations=[("A", "B", parts[0]), ("B", "C", parts[1])],
                                frequency=count,
                                confidence=min(0.95, 0.5 + count * 0.05),
                                context=f"Frequent pattern: {pattern_key}"
                            ))
            
            # Сортируем по частоте и ограничиваем
            patterns_found.sort(key=lambda p: p.frequency, reverse=True)
            self.patterns = patterns_found[:self.max_patterns]
            
            logger.debug(f"Извлечено {len(self.patterns)} паттернов")
            
        except Exception as e:
            logger.debug(f"Ошибка извлечения паттернов: {e}")
    
    def _compute_embeddings(self):
        """Вычисление векторных представлений узлов"""
        try:
            if not hasattr(self, 'brain') or not self.brain:
                return
            kg = getattr(self.brain, 'knowledge_graph', None)
            if not kg:
                return
            
            if hasattr(kg, 'get_all_concepts'):
                concepts = kg.get_all_concepts()
                
                for concept in concepts:
                    node_id = str(concept.get('id') if isinstance(concept, dict) else concept)
                    node_type = str(concept.get('type', 'unknown') if isinstance(concept, dict) else 'concept')
                    description = str(concept.get('description', '') if isinstance(concept, dict) else '')
                    
                    # Используем sentence-transformer для реальных embeddings
                    if self._st_model is not None and description:
                        try:
                            vector = self._st_model.encode([description], convert_to_numpy=True, normalize_embeddings=True)[0]
                        except Exception:
                            vector = self._compute_fallback_embedding(node_id, description)
                    else:
                        vector = self._compute_fallback_embedding(node_id, description)
                    
                    self.embeddings[node_id] = GraphEmbedding(
                        node_id=node_id,
                        node_type=node_type,
                        vector=vector,
                        metadata=concept if isinstance(concept, dict) else {}
                    )
            
            logger.debug(f"Вычислено {len(self.embeddings)} embeddings")
            
        except Exception as e:
            logger.debug(f"Ошибка вычисления embeddings: {e}")
    
    def _compute_fallback_embedding(self, node_id: str, description: str) -> np.ndarray:
        """Fallback вычисление embedding когда sentence-transformer недоступен"""
        if description:
            words = description.lower().split()
            vectors = []
            for word in words:
                if word in self.embeddings:
                    vectors.append(self.embeddings[word].vector)
            if vectors:
                return np.mean(vectors, axis=0)
        
        np.random.seed(hash(node_id) % 2**32)
        vector = np.random.randn(self.embedding_dim).astype(np.float32)
        return vector / np.linalg.norm(vector)
    
    def get_context_for_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Получение релевантного контекста из графа для запроса"""
        try:
            # Извлекаем сущности из запроса
            entities = self._extract_entities_from_query(query)
            
            # Находим ближайшие узлы
            relevant_nodes = []
            for entity in entities:
                if entity in self.embeddings:
                    relevant_nodes.append(self.embeddings[entity])
            
            # Если нет прямых совпадений - ищем по схожести
            if not relevant_nodes and self.embeddings:
                query_embedding = self._compute_query_embedding(query)
                similarities = []
                for node_id, embedding in self.embeddings.items():
                    sim = self._cosine_similarity(query_embedding, embedding.vector)
                    similarities.append((node_id, sim))
                
                similarities.sort(key=lambda x: x[1], reverse=True)
                for node_id, sim in similarities[:top_k]:
                    if sim > self.similarity_threshold:
                        relevant_nodes.append(self.embeddings[node_id])
            
            # Получаем окружение узлов (соседей)
            context = {
                'entities': entities,
                'relevant_nodes': [n.node_id for n in relevant_nodes],
                'related_concepts': [],
                'patterns': []
            }
            
            for node in relevant_nodes:
                # Добавляем связанные концепты
                related = self._get_related_concepts(node.node_id)
                context['related_concepts'].extend(related)
                
                # Находим релевантные паттерны
                patterns = self._find_relevant_patterns(node.node_id)
                context['patterns'].extend(patterns)
            
            # Убираем дубликаты
            context['related_concepts'] = list(set(context['related_concepts']))
            context['patterns'] = list(set(context['patterns']))
            
            return context
            
        except Exception as e:
            logger.debug(f"Ошибка получения контекста: {e}")
            return {'entities': [], 'relevant_nodes': [], 'related_concepts': [], 'patterns': []}
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Извлечение сущностей из запроса"""
        import re
        
        entities = []
        # Имена собственные
        proper_nouns = re.findall(r'\b[А-Я][а-я]+(?:\s+[А-Я][а-я]+)*\b', query)
        entities.extend(proper_nouns)
        
        # Проверяем наличие в графе
        found = [e for e in entities if e in self.embeddings]
        
        return found if found else entities[:3]
    
    def _compute_query_embedding(self, query: str) -> np.ndarray:
        """Вычисление эмбеддинга запроса"""
        # Простая сумма эмбеддингов слов
        words = query.lower().split()
        vectors = []
        
        for word in words:
            if word in self.embeddings:
                vectors.append(self.embeddings[word].vector)
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            # Случайный вектор если ничего не найдено
            np.random.seed(hash(query) % 2**32)
            vec = np.random.randn(self.embedding_dim).astype(np.float32)
            return vec / np.linalg.norm(vec)
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Косинусное сходство"""
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    def _get_related_concepts(self, node_id: str, depth: int = 2) -> List[str]:
        """Получение связанных концептов с учетом глубины"""
        related = []
        
        try:
            if not hasattr(self, 'brain') or not self.brain:
                return related
            kg = getattr(self.brain, 'knowledge_graph', None)
            if not kg:
                return related
            
            if hasattr(kg, 'get_related_concepts'):
                direct = kg.get_related_concepts(node_id)
                related.extend(direct)
                
                # Рекурсивно для глубины > 1
                if depth > 1:
                    for r in direct[:5]:  # Ограничиваем
                        deeper = self._get_related_concepts(r, depth - 1)
                        related.extend(deeper)
        
        except Exception as e:
            logger.debug(f"Ошибка получения связанных: {e}")
        
        return list(set(related))
    
    def _find_relevant_patterns(self, node_id: str) -> List[str]:
        """Поиск паттернов, содержащих узел"""
        relevant = []
        
        for pattern in self.patterns:
            # Проверяем, участвует ли узел в паттерне
            if any(node_id in rel for rel in pattern.relations):
                relevant.append(pattern.pattern_id)
        
        return relevant
    
    def generate_training_sample(self, concept_id: str = None) -> Optional[Dict]:
        """Генерация тренировочного примера из графа"""
        try:
            if not concept_id and self.embeddings:
                concept_id = np.random.choice(list(self.embeddings.keys()))
            
            if concept_id not in self.embeddings:
                return None
            
            embedding = self.embeddings[concept_id]
            
            # Получаем окружение
            related = self._get_related_concepts(concept_id, depth=1)
            
            # Формируем текстовое описание
            description = f"Концепт {embedding.node_id}"
            if related:
                description += f" связан с: {', '.join(related[:5])}"
            
            sample = {
                'input': embedding.node_id,
                'context': description,
                'related_concepts': related,
                'embedding': embedding.vector.tolist(),
                'timestamp': time.time()
            }
            
            return sample
            
        except Exception as e:
            logger.debug(f"Ошибка генерации примера: {e}")
            return None
    
    def get_fractal_context(self, query: str, level: int = 0) -> Dict[str, Any]:
        """
        Получение фрактального контекста с учетом уровня
        
        level 0: Прямые связи
        level 1: Связи второго порядка
        level 2: Паттерны и абстракции
        level 3: Мета-структуры
        """
        base_context = self.get_context_for_query(query)
        
        weight = self.level_weights[min(level, len(self.level_weights) - 1)]
        
        if level == 0:
            return {
                **base_context,
                'fractal_level': level,
                'weight': weight,
                'type': 'direct_connections'
            }
        
        elif level == 1:
            # Расширяем контекст
            extended = []
            for node in base_context.get('relevant_nodes', []):
                extended.extend(self._get_related_concepts(node, depth=2))
            
            return {
                **base_context,
                'extended_concepts': list(set(extended)),
                'fractal_level': level,
                'weight': weight,
                'type': 'extended_connections'
            }
        
        elif level >= 2:
            # Добавляем паттерны и абстракции
            patterns = []
            for pattern in self.patterns[:20]:
                patterns.append({
                    'pattern_id': pattern.pattern_id,
                    'relations': pattern.relations,
                    'frequency': pattern.frequency,
                    'confidence': pattern.confidence
                })
            
            return {
                **base_context,
                'patterns': patterns,
                'fractal_level': level,
                'weight': weight,
                'type': 'pattern_based'
            }
    
    def integrate_into_reasoning(self, reasoning_engine):
        """Интеграция с движком рассуждений"""
        try:
            # Добавляем методы для доступа к графовому контексту
            reasoning_engine.memory_graph_ml = self
            reasoning_engine.get_graph_context = self.get_context_for_query
            reasoning_engine.get_fractal_context = self.get_fractal_context
            
            logger.info("MemoryGraphML интегрирован в ReasoningEngine")
            
        except Exception as e:
            logger.error(f"Ошибка интеграции: {e}")
    
    def update(self) -> bool:
        """Обновление данных из графа памяти"""
        current_time = time.time()
        
        if current_time - self.last_update < self.update_interval:
            return True  # Не время обновлять
        
        try:
            self._extract_patterns()
            self._compute_embeddings()
            self.last_update = current_time
            
            logger.debug("MemoryGraphML обновлен")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обновления: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики"""
        return {
            'is_initialized': self.is_initialized,
            'embeddings_count': len(self.embeddings),
            'patterns_count': len(self.patterns),
            'training_samples': len(self.training_data),
            'embedding_dim': self.embedding_dim,
            'fractal_levels': self.fractal_levels,
            'last_update': self.last_update,
            'graph_coverage': len(self.embeddings) / max(len(self.patterns), 1)
        }
    
    def add_insight(self, insight: str, source_query: str, metadata: Optional[Dict] = None) -> bool:
        """
        Добавляет результат рассуждения (insight) в фрактальный граф памяти.
        Токенизирует текст и создаёт векторные представления на GPU.
        
        Args:
            insight: Текст инсайта/рассуждения
            source_query: Исходный запрос пользователя
            metadata: Дополнительные метаданные
            
        Returns:
            bool: True если успешно добавлено
        """
        try:
            if not insight or len(insight) < 5:
                logger.debug(f"Insight слишком короткий: {insight[:50]}")
                return False
            
            # Токенизируем insight через text_processor
            tokens = self._tokenize_text(insight)
            if not tokens:
                logger.debug(f"Не удалось токенизировать insight: {insight[:50]}")
                return False
            
            # Вычисляем embedding на GPU
            embedding = self._compute_embedding_on_gpu(insight, tokens)
            
            # Создаём узел в графе
            node_id = f"insight_{hash(insight) % 1000000}"
            
            # Добавляем во fractal structure (уровни)
            fractal_level = self._determine_fractal_level(insight)
            
            # Сохраняем embedding
            self.embeddings[node_id] = GraphEmbedding(
                node_id=node_id,
                node_type='insight',
                vector=embedding,
                metadata={
                    'insight': insight[:500],
                    'source_query': source_query,
                    'tokens_count': len(tokens),
                    'fractal_level': fractal_level,
                    'timestamp': time.time(),
                    **(metadata or {})
                }
            )
            
            # Обновляем graph property
            self._graph = self.embeddings
            
            # Добавляем связь с исходным запросом
            query_node_id = f"query_{hash(source_query) % 1000000}"
            self._add_fractal_relation(query_node_id, node_id, 'generated_from')
            
            # Генерируем данные для обучения если есть model_manager
            self._generate_training_data(insight, source_query, embedding)
            
            logger.debug(f"Insight добавлен в граф: level={fractal_level}, tokens={len(tokens)}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка добавления insight: {e}")
            return False
    
    def _tokenize_text(self, text: str) -> List[int]:
        """
        Токенизирует текст через систему токенизации.
        Использует hybrid_cache для кэширования токенов.
        """
        try:
            # Пробуем сначала получить токены из гибридного кэша
            cache_key = f"tokenize:{hash(text) % 10000000}"
            if hasattr(self, '_hybrid_cache') and self._hybrid_cache:
                cached = self._hybrid_cache.get(cache_key)
                if cached is not None:
                    return cached if isinstance(cached, list) else []
            
            tokens = None
            
            # Пробуем через text_processor
            if hasattr(self.brain, 'text_processor') and self.brain.text_processor:
                tokenizer = getattr(self.brain.text_processor, 'tokenizer', None)
                if tokenizer and hasattr(tokenizer, 'encode'):
                    tokens = tokenizer.encode(text)
            
            # Fallback: пробуем через memory_manager
            if not tokens and hasattr(self.brain, 'memory_manager') and self.brain.memory_manager:
                mm = self.brain.memory_manager
                if hasattr(mm, 'hybrid_cache') and mm.hybrid_cache:
                    tokenizer = getattr(mm.hybrid_cache, 'tokenizer', None)
                    if tokenizer and hasattr(tokenizer, 'encode'):
                        tokens = tokenizer.encode(text)
            
            # Fallback: простая токенизация
            if not tokens:
                tokens = [hash(word) % 50000 for word in text.split() if word]
            
            # Кэшируем результат в hybrid_cache
            if tokens and hasattr(self, '_hybrid_cache') and self._hybrid_cache:
                try:
                    self._hybrid_cache.put(cache_key, tokens)
                except Exception:
                    pass
            
            return tokens if tokens else []
            
        except Exception as e:
            logger.debug(f"Ошибка токенизации: {e}")
            return []
    
    def _get_hybrid_cache(self):
        """Получает гибридный кэш из различных источников"""
        if hasattr(self, '_hybrid_cache') and self._hybrid_cache:
            return self._hybrid_cache
        
        # Пробуем получить из brain
        if self.brain:
            if hasattr(self.brain, 'hybrid_cache') and self.brain.hybrid_cache:
                self._hybrid_cache = self.brain.hybrid_cache
                return self._hybrid_cache
            
            # Пробуем через memory_manager
            if hasattr(self.brain, 'memory_manager') and self.brain.memory_manager:
                mm = self.brain.memory_manager
                if hasattr(mm, 'hybrid_cache') and mm.hybrid_cache:
                    self._hybrid_cache = mm.hybrid_cache
                    return mm.hybrid_cache
                if hasattr(mm, 'get_hybrid_cache'):
                    hc = mm.get_hybrid_cache()
                    if hc:
                        self._hybrid_cache = hc
                        return hc
        
        return None
    
    def _compute_embedding_on_gpu(self, text: str, tokens: List[int]) -> np.ndarray:
        """
        Вычисляет embedding с приоритетом: GPU > sentence-transformer (CPU) > random.
        """
        try:
            # Приоритет 1: sentence-transformer (лучшее качество на CPU)
            if self._st_model is not None:
                embedding = self._st_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
                return self._fractal_transform(
                    torch.from_numpy(embedding).float()
                ).cpu().numpy()
            
            # Приоритет 2: GPU tensor ops
            if TORCH_AVAILABLE:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                if tokens:
                    token_tensor = torch.tensor(tokens[:self.embedding_dim], dtype=torch.long, device=device)
                    if hasattr(self, '_embedding_layer') and self._embedding_layer is not None:
                        with torch.no_grad():
                            embedding = self._embedding_layer(token_tensor).mean(dim=0)
                    else:
                        embedding = torch.randn(self.embedding_dim, dtype=torch.float32, device=device)
                    embedding = self._fractal_transform(embedding)
                    return (embedding / torch.norm(embedding)).cpu().numpy()
            
            # Fallback: детерминированный случайный вектор
            np.random.seed(hash(text) % 2**32)
            vector = np.random.randn(self.embedding_dim).astype(np.float32)
            return vector / np.linalg.norm(vector)
                
        except Exception as e:
            logger.debug(f"Ошибка embedding: {e}")
            np.random.seed(hash(text) % 2**32)
            return np.random.randn(self.embedding_dim).astype(np.float32) / np.linalg.norm(np.random.randn(self.embedding_dim))
    
    def _fractal_transform(self, embedding):
        """
        Применяет фрактальное преобразование к embedding.
        Создаёт иерархические представления для разных уровней графа.
        """
        try:
            if not TORCH_AVAILABLE or not isinstance(embedding, torch.Tensor):
                # CPU fallback - numpy-based
                emb = np.array(embedding)
                chunk_size = max(1, self.embedding_dim // self.fractal_levels)
                result_parts = []
                for level in range(self.fractal_levels):
                    start_idx = level * chunk_size
                    end_idx = min((level + 1) * chunk_size, self.embedding_dim)
                    level_part = emb[start_idx:end_idx] * self.level_weights[level]
                    result_parts.append(level_part)
                result = np.concatenate(result_parts)
                result = np.tanh(result)
                return torch.tensor(result, dtype=torch.float32) if TORCH_AVAILABLE else result
            
            # Фрактальное преобразование - разбиваем на уровни
            level_embeddings = []
            chunk_size = max(1, self.embedding_dim // self.fractal_levels)
            
            for level in range(self.fractal_levels):
                start_idx = level * chunk_size
                end_idx = min((level + 1) * chunk_size, self.embedding_dim)
                
                # Применяем вес уровня
                level_part = embedding[start_idx:end_idx] * self.level_weights[level]
                level_embeddings.append(level_part)
            
            # Объединяем обратно
            result = torch.cat(level_embeddings)
            
            # Добавляем нелинейность
            result = torch.tanh(result)
            
            return result
            
        except Exception as e:
            logger.debug(f"Ошибка фрактального преобразования: {e}")
            return embedding
    
    def _determine_fractal_level(self, text: str) -> int:
        """Определяет уровень фрактальной структуры для текста"""
        # Чем важнее/сложнее текст - тем выше уровень
        text_length = len(text)
        
        if text_length < 50:
            return 0  # Surface level
        elif text_length < 200:
            return 1  # Middle level
        elif text_length < 500:
            return 2  # Deep level
        else:
            return 3  # Core level
    
    def _add_fractal_relation(self, from_node: str, to_node: str, relation_type: str):
        """Добавляет связь между узлами во фрактальной структуре"""
        try:
            # Проверяем существование узлов
            if from_node not in self.embeddings:
                # Создаём placeholder узел
                np.random.seed(hash(from_node) % 2**32)
                self.embeddings[from_node] = GraphEmbedding(
                    node_id=from_node,
                    node_type='query',
                    vector=np.random.randn(self.embedding_dim).astype(np.float32),
                    metadata={}
                )
            
            # Добавляем связь в метаданные
            if 'relations' not in self.embeddings[from_node].metadata:
                self.embeddings[from_node].metadata['relations'] = []
            
            self.embeddings[from_node].metadata['relations'].append({
                'to': to_node,
                'type': relation_type,
                'timestamp': time.time()
            })
            
        except Exception as e:
            logger.debug(f"Ошибка добавления связи: {e}")
    
    def _generate_training_data(self, insight: str, query: str, embedding: np.ndarray):
        """Генерирует данные для обучения из insight"""
        try:
            # Создаём обучающую выборку
            sample = {
                'input': query,
                'output': insight,
                'embedding': embedding.tolist(),
                'timestamp': time.time(),
                'type': 'reasoning_insight'
            }
            
            self.training_data.append(sample)
            
            # Ограничиваем размер
            max_training_samples = self.config.get('max_training_samples', 10000)
            if len(self.training_data) > max_training_samples:
                self.training_data = self.training_data[-max_training_samples:]
            
            # Если есть model_manager - пробуем обучить
            if hasattr(self.brain, 'model_manager') and self.brain.model_manager:
                self._trigger_training_if_needed()
                
        except Exception as e:
            logger.debug(f"Ошибка генерации данных для обучения: {e}")
    
    def _trigger_training_if_needed(self):
        """Триггерит обучение если накопилось достаточно данных"""
        if self.config.get('training_disabled', True):
            return
        try:
            min_samples = self.config.get('min_training_samples', 100)
            if len(self.training_data) >= min_samples:
                # Пробуем инициировать обучение
                if hasattr(self.brain, 'training_orchestrator'):
                    logger.info(f"Накоплено {len(self.training_data)} обучающих выборок, готово к обучению")
        except Exception as e:
            logger.debug(f"Ошибка триггера обучения: {e}")

    def add_ambiguous_entity(self, entity: AmbiguousEntity, query_context: str):
        """Store extracted entity with its query context."""
        try:
            entity_id = f"entity_{hash(entity.term) % 1000000}_{int(time.time())}"
            
            self.embeddings[entity_id] = GraphEmbedding(
                node_id=entity_id,
                node_type='ambiguous_entity',
                vector=self._compute_fallback_embedding(entity_id, entity.context),
                metadata={
                    'term': entity.term,
                    'possible_meanings': entity.possible_meanings,
                    'context': query_context,
                    'resolved_meaning': entity.resolved_meaning,
                    'confidence': entity.confidence,
                    'clarification_history': entity.clarification_history,
                    'timestamp': entity.timestamp
                }
            )
            
            self._link_entity_to_graph(entity_id, entity.term)
            
            entity_fractal_store = getattr(self.brain, 'entity_fractal_store', None) if hasattr(self, 'brain') and self.brain else None
            if entity_fractal_store and hasattr(entity_fractal_store, 'store_entity'):
                try:
                    entity_fractal_store.store_entity(entity, query_context)
                except Exception:
                    pass
            
            logger.debug(f"Ambiguous entity added: {entity.term}")
            
        except Exception as e:
            logger.error(f"Error adding ambiguous entity: {e}")
    
    def get_entity_history(self, entity_term: str) -> List[AmbiguousEntity]:
        """Get history of how an entity was used/clarified."""
        history = []
        try:
            for emb_id, embedding in self.embeddings.items():
                if embedding.node_type == 'ambiguous_entity':
                    if embedding.metadata.get('term', '').lower() == entity_term.lower():
                        history.append(AmbiguousEntity(
                            term=embedding.metadata.get('term', ''),
                            possible_meanings=embedding.metadata.get('possible_meanings', []),
                            context=embedding.metadata.get('context', ''),
                            timestamp=embedding.metadata.get('timestamp', time.time()),
                            clarification_history=embedding.metadata.get('clarification_history', []),
                            resolved_meaning=embedding.metadata.get('resolved_meaning'),
                            confidence=embedding.metadata.get('confidence', 0.5)
                        ))
            
            history.sort(key=lambda x: x.timestamp, reverse=True)
            
        except Exception as e:
            logger.debug(f"Error getting entity history: {e}")
        
        return history
    
    def store_clarification(self, request: ClarificationRequest, answer: str):
        """Store clarification Q&A pair for learning."""
        try:
            request.answered = True
            request.selected_meaning = answer
            
            entity_id = f"entity_{hash(request.entity_term) % 1000000}"
            
            if entity_id in self.embeddings:
                self.embeddings[entity_id].metadata.setdefault('clarification_history', []).append({
                    'question': request.question,
                    'answer': answer,
                    'timestamp': time.time()
                })
                
                if not self.embeddings[entity_id].metadata.get('resolved_meaning'):
                    self.embeddings[entity_id].metadata['resolved_meaning'] = answer
                    self.embeddings[entity_id].metadata['confidence'] = 1.0
            
            entity_fractal_store = getattr(self.brain, 'entity_fractal_store', None) if hasattr(self, 'brain') and self.brain else None
            if entity_fractal_store and hasattr(entity_fractal_store, 'update_clarification'):
                try:
                    entity_fractal_store.update_clarification(
                        request.entity_term, request.question, answer
                    )
                except Exception:
                    pass
            
            if hasattr(self.brain, 'knowledge_graph'):
                try:
                    self._link_clarification_to_graph(request.entity_term, answer)
                except Exception:
                    pass
            
            logger.debug(f"Clarification stored for: {request.entity_term}")
            
        except Exception as e:
            logger.error(f"Error storing clarification: {e}")
    
    def _link_entity_to_graph(self, entity_id: str, entity_term: str):
        """Link entity node to knowledge graph."""
        try:
            if not hasattr(self, 'brain') or not self.brain:
                return
            kg = getattr(self.brain, 'knowledge_graph', None)
            if not kg:
                return
            
            similar_nodes = kg.search_nodes(entity_term, limit=5) if hasattr(kg, 'search_nodes') else []
            for node in similar_nodes:
                if hasattr(node, 'id'):
                    kg.add_edge(
                        source_id=entity_id,
                        target_id=node.id,
                        relation_type='related_to',
                        strength=0.6,
                        meta={'link_type': 'entity_similarity'}
                    )
            
        except Exception as e:
            logger.debug(f"Error linking entity to graph: {e}")
    
    def _link_clarification_to_graph(self, entity_term: str, resolved_meaning: str):
        """Link clarification result to knowledge graph."""
        try:
            if not hasattr(self, 'brain') or not self.brain:
                return
            kg = getattr(self.brain, 'knowledge_graph', None)
            if not kg:
                return
            
            meaning_node_id = kg.add_node(
                name=entity_term,
                description=f"Clarified meaning: {resolved_meaning}",
                node_type='entity',
                meta={'clarification': True, 'resolved_meaning': resolved_meaning}
            )
            
            entity_nodes = kg.search_nodes(entity_term, limit=1)
            if entity_nodes:
                kg.add_edge(
                    source_id=meaning_node_id,
                    target_id=entity_nodes[0].id,
                    relation_type='clarifies',
                    strength=1.0,
                    meta={'clarification_result': True}
                )
            
        except Exception as e:
            logger.debug(f"Error linking clarification to graph: {e}")
