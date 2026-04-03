"""
GraphLearningLoop — обучение через граф опыта

Архитектура:
1. Каждый Q&A сохраняется как ExperienceNode с embedding и quality
2. При запросе DynamicContextBuilder собирает релевантный контекст из графа
3. Фоновый цикл анализирует успешные ответы, находит паттерны, создаёт концепты
4. SnapshotManager экспортирует/импортирует слепки знаний

Цикл обучения:
  Запрос → Модель отвечает → Опыт сохраняется → Граф обновляется
  → Фоновый цикл находит паттерны → Создаёт концепты → Граф становится "умнее"
  → Следующий запрос получает лучший контекст → Модель отвечает лучше
"""

import os
import json
import time
import hashlib
import logging
import threading
from typing import Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExperienceNode:
    """Узел опыта — один Q&A с метаданными"""
    id: str
    query: str
    response: str
    model_used: str  # model_a, model_b
    quality_score: float
    embedding: List[float] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    usage_count: int = 0  # сколько раз этот опыт помог
    related_experiences: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'query': self.query,
            'response': self.response,
            'model_used': self.model_used,
            'quality_score': self.quality_score,
            'embedding': self.embedding,
            'tags': self.tags,
            'timestamp': self.timestamp,
            'usage_count': self.usage_count,
            'related_experiences': self.related_experiences
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperienceNode':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ConceptNode:
    """Концепт — обобщённый паттерн из множества опытов"""
    id: str
    title: str
    description: str
    experience_ids: List[str] = field(default_factory=list)
    embedding: List[float] = field(default_factory=list)
    confidence: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'experience_ids': self.experience_ids,
            'embedding': self.embedding,
            'confidence': self.confidence,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConceptNode':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class DynamicContextBuilder:
    """
    Собирает контекст из графа при каждом запросе.
    
    Алгоритм:
    1. Находит релевантные опыты по embedding similarity
    2. Находит релевантные концепты
    3. Строит структурированный контекст: концепты + лучшие опыты
    4. Контекст подаётся модели как "память"
    """
    
    def __init__(self, fractal_memory, max_experiences: int = 5, max_concepts: int = 3):
        self.fractal_memory = fractal_memory
        self.max_experiences = max_experiences
        self.max_concepts = max_concepts
    
    def build_context(self, query: str, query_embedding: List[float] = None) -> str:
        """Строит контекст из графа для запроса"""
        experiences = self._find_relevant_experiences(query, query_embedding)
        concepts = self._find_relevant_concepts(query, query_embedding)
        
        parts = []
        
        # Сначала концепты (обобщённые знания)
        if concepts:
            parts.append("Общие знания по теме:")
            for c in concepts:
                parts.append(f"- {c.title}: {c.description}")
        
        # Затем конкретные опыты
        if experiences:
            parts.append("\nПредыдущий опыт:")
            for exp in experiences:
                # Только если качество высокое
                if exp.quality_score > 0.6:
                    parts.append(f"- Q: {exp.query[:100]}")
                    parts.append(f"  A: {exp.response[:200]}")
        
        return "\n".join(parts) if parts else ""
    
    def _find_relevant_experiences(self, query: str, query_embedding: List[float] = None) -> List[ExperienceNode]:
        """Находит релевантные опыты"""
        # Пока используем простое текстовое совпадение
        # В будущем — cosine similarity по embeddings
        experiences = self._load_experiences()
        
        if not experiences:
            return []
        
        # Сортируем по usage_count и quality_score
        scored = []
        for exp in experiences:
            # Простое текстовое совпадение
            query_words = set(query.lower().split())
            exp_words = set(exp.query.lower().split())
            overlap = len(query_words & exp_words) / max(len(query_words), 1)
            
            # Комбинированный скор: 50% релевантность, 50% качество, + бонус за использование
            score = overlap * 0.5 + exp.quality_score * 0.3 + min(exp.usage_count / 10, 1) * 0.2
            scored.append((score, exp))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        
        # Увеличиваем usage_count у топ результатов
        for score, exp in scored[:self.max_experiences]:
            exp.usage_count += 1
            self._save_experience(exp)
        
        return [exp for _, exp in scored[:self.max_experiences]]
    
    def _find_relevant_concepts(self, query: str, query_embedding: List[float] = None) -> List[ConceptNode]:
        """Находит релевантные концепты"""
        concepts = self._load_concepts()
        
        if not concepts:
            return []
        
        scored = []
        for concept in concepts:
            query_words = set(query.lower().split())
            concept_words = set((concept.title + ' ' + concept.description).lower().split())
            overlap = len(query_words & concept_words) / max(len(query_words), 1)
            score = overlap * 0.6 + concept.confidence * 0.4
            scored.append((score, concept))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return [c for _, c in scored[:self.max_concepts]]
    
    def _load_experiences(self) -> List[ExperienceNode]:
        """Загружает все опыты из графа"""
        exp_dir = os.path.join(self.fractal_memory.storage_dir, "experiences")
        os.makedirs(exp_dir, exist_ok=True)
        
        experiences = []
        for fname in os.listdir(exp_dir):
            if fname.endswith('.json'):
                fpath = os.path.join(exp_dir, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        experiences.append(ExperienceNode.from_dict(data))
                except Exception as e:
                    logger.warning(f"Ошибка загрузки опыта {fname}: {e}")
        
        return experiences
    
    def _load_concepts(self) -> List[ConceptNode]:
        """Загружает все концепты из графа"""
        concept_dir = os.path.join(self.fractal_memory.storage_dir, "concepts")
        os.makedirs(concept_dir, exist_ok=True)
        
        concepts = []
        for fname in os.listdir(concept_dir):
            if fname.endswith('.json'):
                fpath = os.path.join(concept_dir, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        concepts.append(ConceptNode.from_dict(data))
                except Exception as e:
                    logger.warning(f"Ошибка загрузки концепта {fname}: {e}")
        
        return concepts
    
    def _save_experience(self, exp: ExperienceNode):
        """Сохраняет опыт"""
        exp_dir = os.path.join(self.fractal_memory.storage_dir, "experiences")
        os.makedirs(exp_dir, exist_ok=True)
        fpath = os.path.join(exp_dir, f"{exp.id}.json")
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(exp.to_dict(), f, ensure_ascii=False, indent=2)
    
    def _save_concept(self, concept: ConceptNode):
        """Сохраняет концепт"""
        concept_dir = os.path.join(self.fractal_memory.storage_dir, "concepts")
        os.makedirs(concept_dir, exist_ok=True)
        fpath = os.path.join(concept_dir, f"{concept.id}.json")
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(concept.to_dict(), f, ensure_ascii=False, indent=2)


class GraphLearningLoop:
    """
    Фоновый цикл обучения через граф.
    
    Работает в отдельном потоке:
    1. Собирает новые опыты (Q&A пары)
    2. Кластеризует по темам
    3. Создаёт концепты из кластеров
    4. Обновляет связи между узлами
    """
    
    def __init__(self, fractal_memory, context_builder: DynamicContextBuilder, 
                 min_quality: float = 0.7, cluster_interval: int = 300):
        self.fractal_memory = fractal_memory
        self.context_builder = context_builder
        self.min_quality = min_quality
        self.cluster_interval = cluster_interval  # секунд между кластеризацией
        
        self._running = False
        self._thread = None
        self._pending_experiences: List[ExperienceNode] = []
        self._lock = threading.Lock()
    
    def start(self):
        """Запускает фоновый цикл обучения"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="GraphLearningLoop")
        self._thread.start()
        logger.info("GraphLearningLoop запущен")
    
    def stop(self):
        """Останавливает цикл"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("GraphLearningLoop остановлен")
    
    def add_experience(self, query: str, response: str, model_used: str, quality_score: float) -> str:
        """Добавляет новый опыт в очередь обучения"""
        exp_id = hashlib.sha256(f"{query}:{response}:{time.time()}".encode()).hexdigest()[:16]
        
        exp = ExperienceNode(
            id=f"exp_{exp_id}",
            query=query,
            response=response,
            model_used=model_used,
            quality_score=quality_score
        )
        
        with self._lock:
            self._pending_experiences.append(exp)
        
        # Сохраняем сразу
        self.context_builder._save_experience(exp)
        
        logger.info(f"Добавлен опыт: {exp_id[:8]}... (quality={quality_score:.2f})")
        return exp.id
    
    def _loop(self):
        """Основной цикл обучения"""
        last_cluster_time = 0
        
        while self._running:
            try:
                now = time.time()
                
                # Кластеризация каждые cluster_interval секунд
                if now - last_cluster_time > self.cluster_interval and self._pending_experiences:
                    self._cluster_experiences()
                    last_cluster_time = now
                
                time.sleep(10)  # Проверка каждые 10 секунд
                
            except Exception as e:
                logger.error(f"Ошибка в цикле обучения: {e}")
                time.sleep(30)
    
    def _cluster_experiences(self):
        """Кластеризует опыты и создаёт концепты"""
        with self._lock:
            experiences = list(self._pending_experiences)
            self._pending_experiences.clear()
        
        if not experiences:
            return
        
        # Фильтруем только качественные
        good_exps = [e for e in experiences if e.quality_score >= self.min_quality]
        if not good_exps:
            logger.info(f"Нет качественных опытов для кластеризации ({len(experiences)} всего, min={self.min_quality})")
            return
        
        # Простая кластеризация по ключевым словам
        clusters = self._simple_cluster(good_exps)
        
        for cluster_exps in clusters:
            if len(cluster_exps) >= 2:  # Минимум 2 опыта для концепта
                self._create_concept(cluster_exps)
        
        logger.info(f"Кластеризация: {len(clusters)} кластеров из {len(good_exps)} опытов")
    
    def _simple_cluster(self, experiences: List[ExperienceNode]) -> List[List[ExperienceNode]]:
        """Простая кластеризация по пересечению ключевых слов"""
        # Извлекаем ключевые слова
        stop_words = {'что', 'это', 'как', 'для', 'или', 'и', 'в', 'на', 'не', 'по', 'из', 'а', 'но', 'the', 'is', 'to', 'of', 'and'}
        
        def get_keywords(text: str) -> set:
            words = set(text.lower().split())
            return {w for w in words if len(w) > 3 and w not in stop_words}
        
        # Группируем по ключевым словам
        clusters = []
        used = set()
        
        for exp in experiences:
            if exp.id in used:
                continue
            
            cluster = [exp]
            used.add(exp.id)
            exp_keywords = get_keywords(exp.query)
            
            for other in experiences:
                if other.id in used:
                    continue
                other_keywords = get_keywords(other.query)
                overlap = len(exp_keywords & other_keywords) / max(len(exp_keywords | other_keywords), 1)
                if overlap > 0.2:  # 20% пересечение
                    cluster.append(other)
                    used.add(other.id)
            
            clusters.append(cluster)
        
        return clusters
    
    def _create_concept(self, cluster: List[ExperienceNode]):
        """Создаёт концепт из кластера опытов"""
        # Заголовок — самый частый запрос
        titles = [e.query[:80] for e in cluster]
        title = max(set(titles), key=titles.count)
        
        # Описание — агрегация из лучших ответов
        sorted_exps = sorted(cluster, key=lambda e: e.quality_score, reverse=True)
        best_responses = [e.response[:200] for e in sorted_exps[:3]]
        description = best_responses[0] if best_responses else ""
        
        # Средняя уверенность
        confidence = sum(e.quality_score for e in cluster) / len(cluster)
        
        concept_id = hashlib.sha256(title.encode()).hexdigest()[:16]
        concept = ConceptNode(
            id=f"concept_{concept_id}",
            title=title,
            description=description,
            experience_ids=[e.id for e in cluster],
            confidence=confidence
        )
        
        self.context_builder._save_concept(concept)
        logger.info(f"Создан концепт: {title[:60]}... (confidence={confidence:.2f}, опытов={len(cluster)})")


class SnapshotManager:
    """
    Экспорт/импорт слепков знаний.
    
    Слепок = сериализованное состояние графа опыта (experiences + concepts).
    Можно экспортировать, импортировать, сравнивать.
    """
    
    def __init__(self, fractal_memory, context_builder: DynamicContextBuilder):
        self.fractal_memory = fractal_memory
        self.context_builder = context_builder
        self.snapshots_dir = os.path.join(fractal_memory.storage_dir, "snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)
    
    def export_snapshot(self, name: str = None) -> str:
        """Экспортирует текущее состояние графа как слепок"""
        if not name:
            name = f"snapshot_{int(time.time())}"
        
        experiences = self.context_builder._load_experiences()
        concepts = self.context_builder._load_concepts()
        
        snapshot = {
            'name': name,
            'created_at': time.time(),
            'experiences': [e.to_dict() for e in experiences],
            'concepts': [c.to_dict() for c in concepts],
            'stats': {
                'total_experiences': len(experiences),
                'total_concepts': len(concepts),
                'avg_quality': sum(e.quality_score for e in experiences) / max(len(experiences), 1),
                'top_topics': self._get_top_topics(experiences)
            }
        }
        
        fpath = os.path.join(self.snapshots_dir, f"{name}.json")
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Слепок экспортирован: {name} ({len(experiences)} опытов, {len(concepts)} концептов)")
        return fpath
    
    def import_snapshot(self, snapshot_path: str) -> Dict[str, int]:
        """Импортирует слепок в граф"""
        with open(snapshot_path, 'r', encoding='utf-8') as f:
            snapshot = json.load(f)
        
        imported_exps = 0
        imported_concepts = 0
        
        for exp_data in snapshot.get('experiences', []):
            exp = ExperienceNode.from_dict(exp_data)
            self.context_builder._save_experience(exp)
            imported_exps += 1
        
        for concept_data in snapshot.get('concepts', []):
            concept = ConceptNode.from_dict(concept_data)
            self.context_builder._save_concept(concept)
            imported_concepts += 1
        
        logger.info(f"Слепок импортирован: {imported_exps} опытов, {imported_concepts} концептов")
        return {'experiences': imported_exps, 'concepts': imported_concepts}
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """Список доступных слепков"""
        snapshots = []
        for fname in os.listdir(self.snapshots_dir):
            if fname.endswith('.json'):
                fpath = os.path.join(self.snapshots_dir, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        snapshots.append({
                            'name': data.get('name', fname),
                            'created_at': data.get('created_at', 0),
                            'experiences': data.get('stats', {}).get('total_experiences', 0),
                            'concepts': data.get('stats', {}).get('total_concepts', 0),
                            'avg_quality': data.get('stats', {}).get('avg_quality', 0)
                        })
                except Exception:
                    pass
        
        snapshots.sort(key=lambda x: x['created_at'], reverse=True)
        return snapshots
    
    def _get_top_topics(self, experiences: List[ExperienceNode]) -> List[str]:
        """Получает топ тем из опытов"""
        from collections import Counter
        stop_words = {'что', 'это', 'как', 'для', 'или', 'и', 'в', 'на', 'не', 'по', 'из', 'а', 'но'}
        
        words = []
        for exp in experiences:
            for w in exp.query.lower().split():
                if len(w) > 3 and w not in stop_words:
                    words.append(w)
        
        return [w for w, _ in Counter(words).most_common(10)]
