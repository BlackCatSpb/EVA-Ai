"""
ContradictionMiner - модуль проактивного обнаружения логических противоречий в FGv2

Обнаруживает пары/кластеры узлов с:
1. Высокой семантической близостью (cosine similarity >= 0.75)
2. Логическим противоречием (NLI contradiction >= 0.65)

Создаёт ContradictionNode для самодиалога и веб-поиска.

По спецификации: Доработка FG.txt
"""

import os
import time
import json
import logging
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("eva_ai.contradiction.miner")


class ContradictionStatus(Enum):
    """Жизненный цикл узла-противоречия"""
    ACTIVE = "active"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ARCHIVED = "archived"


@dataclass
class ContradictionCandidate:
    """Кандидат на противоречие (кластер конфликтных узлов)"""
    id: str
    cluster_id: str
    node_ids: List[str]
    nodes_data: List[Dict]
    
    # Метрики
    avg_similarity: float
    max_contradiction: float
    priority: float
    
    # Сгенерированное содержимое
    title: str = ""
    description: str = ""
    resolution_question: str = ""
    
    status: str = "active"
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    resolution_node_id: Optional[str] = None
    
    # Метаданные
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContradictionMiner:
    """
    Модуль проактивного обнаружения противоречий во фрактальном графе.
    
    Работает на низком уровне с FGv2:
    - Анализ пар узлов: семантическая близость + логическое противоречие
    - Кластеризация конфликтных пар
    - Генерация формулировки противоречия через LLM
    - Создание ContradictionNode
    
    Отличие от ContradictionGenerator:
    - ContradictionGenerator: создаёт противоречия для концептов (шаблоны)
    - ContradictionMiner: обнаруживает реальные противоречия в графе (анализ)
    """

    DEFAULT_CONFIG = {
        "enabled": True,
        "dry_run": False,
        "sim_threshold": 0.75,  # τ_sim - семантическая близость
        "contra_threshold": 0.65,  # τ_contra - логическое противоречие
        "min_confidence": 0.4,
        "max_candidates_per_cycle": 5,
        "priority_coefficients": {
            "alpha": 0.4,  # вес размера кластера
            "beta": 0.3,   # вес средней уверенности
            "gamma": 0.3   # вес максимального противоречия
        },
        "llm_temperature": 0.25,
        "llm_repeat_penalty": 1.5,
        "llm_max_tokens": 200,
        "enable_web_search_for_resolution": True,
        "check_interval_seconds": 3600,  # Проверка раз в час
    }

    SYSTEM_PROMPT = """Ты — аналитический модуль когнитивной системы EVA-Ai.
Твоя задача — выявить и чётко сформулировать логическое противоречие между несколькими утверждениями, которые относятся к одной теме.

Сформулируй:
1. Краткий заголовок противоречия (вопрос или тезис).
2. Развёрнутое описание сути противоречия, цитируя ключевые моменты из предоставленных узлов.
3. Конкретный вопрос, ответ на который помог бы разрешить противоречие (для последующего самодиалога или веб-поиска).

Формат вывода строго:
CONTRADICTION_TITLE: [заголовок]
DESCRIPTION: [описание, до 100 слов]
RESOLUTION_QUESTION: [вопрос]"""

    def __init__(
        self,
        brain=None,
        event_bus=None,
        deferred_system=None,
        config: Dict = None
    ):
        self.brain = brain
        self.event_bus = event_bus
        self.deferred_system = deferred_system

        self.config = {**self.DEFAULT_CONFIG}
        if config:
            self.config.update(config)

        self._running = False
        self._subscription_ids = []
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ContradictionMiner")

        # Кэш для пар (u,v) → sim/contra чтобы не пересчитывать
        self._similarity_cache: Dict[str, float] = {}
        self._contradiction_cache: Dict[str, float] = {}
        
        # Активные кандидаты
        self._candidates: Dict[str, ContradictionCandidate] = {}
        self._load_candidates()

        self._last_check_time = 0
        self._checking_in_progress = False

        self._metrics = {
            "candidates_detected_total": 0,
            "nodes_created_total": 0,
            "resolved_total": 0,
            "avg_cluster_size": 0.0,
            "processing_time_ms": 0.0,
            "pairs_checked": 0,
            "cache_hits": 0,
        }

        logger.info("ContradictionMiner инициализирован")

    def _load_candidates(self):
        """Загрузка кандидатов из storage"""
        try:
            storage_dir = self._get_storage_dir()
            candidates_file = os.path.join(storage_dir, "contradiction_candidates.json")
            if os.path.exists(candidates_file):
                with open(candidates_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for cid, cdata in data.items():
                        self._candidates[cid] = ContradictionCandidate(**cdata)
                logger.info(f"Загружено {len(self._candidates)} кандидатов противоречий")
        except Exception as e:
            logger.warning(f"Не удалось загрузить кандидатов: {e}")

    def _save_candidates(self):
        """Сохранение кандидатов"""
        try:
            storage_dir = self._get_storage_dir()
            os.makedirs(storage_dir, exist_ok=True)
            candidates_file = os.path.join(storage_dir, "contradiction_candidates.json")
            data = {cid: asdict(c) for cid, c in self._candidates.items()}
            with open(candidates_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Не удалось сохранить кандидатов: {e}")

    def _get_storage_dir(self) -> str:
        """Директория для хранения"""
        if self.brain and hasattr(self.brain, 'cache_dir'):
            return os.path.join(self.brain.cache_dir, "contradiction_miner_data")
        return os.path.join(os.path.dirname(__file__), "contradiction_miner_data")

    def start(self):
        """Запуск модуля"""
        if not self.config.get("enabled", True):
            logger.info("ContradictionMiner отключён")
            return

        if self._running:
            return

        self._running = True

        if self.event_bus:
            self._subscribe_to_events()

        logger.info("ContradictionMiner запущен")

    def stop(self):
        """Остановка модуля"""
        self._running = False

        if self.event_bus and self._subscription_ids:
            for sub_id in self._subscription_ids:
                try:
                    self.event_bus.unsubscribe(sub_id)
                except Exception:
                    pass
            self._subscription_ids.clear()

        self._executor.shutdown(wait=False)
        self._save_candidates()

        logger.info("ContradictionMiner остановлен")

    def _subscribe_to_events(self):
        """Подписка на события"""
        if not self.event_bus:
            return

        events_to_subscribe = [
            ("memory.node_created", "_on_node_created"),
            ("memory.graph_updated", "_on_graph_updated"),
            ("system.idle", "_on_system_idle"),
        ]

        for event_type, handler_name in events_to_subscribe:
            try:
                if hasattr(self, handler_name):
                    handler = getattr(self, handler_name)
                    if hasattr(self.event_bus, 'subscribe'):
                        try:
                            sub_id = self.event_bus.subscribe(event_type, handler, priority=7)
                            self._subscription_ids.append(sub_id)
                        except Exception as e:
                            logger.debug(f"Ошибка подписки: {e}")
            except Exception as e:
                logger.warning(f"Не удалось подписаться на {event_type}: {e}")

    def _on_node_created(self, event):
        """При создании узла - проверить через некоторое время"""
        # Не запускаем сразу, а ждём накопления
        if time.time() - self._last_check_time > self.config.get("check_interval_seconds", 3600):
            self._schedule_check()

    def _on_graph_updated(self, event):
        """При обновлении графа"""
        if time.time() - self._last_check_time > self.config.get("check_interval_seconds", 3600):
            self._schedule_check()

    def _on_system_idle(self, event):
        """При простое системы - идеальное время для проверки"""
        self._schedule_check()

    def _schedule_check(self):
        """Планирование проверки"""
        if not self._running or self._checking_in_progress:
            return

        min_interval = self.config.get("check_interval_seconds", 3600)
        if time.time() - self._last_check_time < min_interval:
            return

        self._last_check_time = time.time()

        if self.deferred_system:
            try:
                from eva_ai.core.deferred_command_system import CommandPriority
                self.deferred_system.add_command(
                    command=self._detection_cycle,
                    priority=CommandPriority.NORMAL,
                    command_id=f"contradiction_check_{int(time.time())}"
                )
            except Exception:
                self._executor.submit(self._detection_cycle)
        else:
            self._executor.submit(self._detection_cycle)

    def _detection_cycle(self):
        """Основной цикл обнаружения противоречий"""
        if self._checking_in_progress:
            return

        self._checking_in_progress = True
        start_time = time.time()

        try:
            logger.info("Начат цикл обнаружения противоречий")

            # 1. Поиск кандидатов
            pairs = self._detect_candidate_pairs()
            
            if not pairs:
                logger.debug("Противоречий не обнаружено")
                return

            logger.info(f"Найдено {len(pairs)} пар-кандидатов")

            # 2. Кластеризация
            candidates = self._cluster_pairs(pairs)
            logger.info(f"Сформировано {len(candidates)} кластеров")

            # 3. Фильтрация и приоритизация
            filtered = self._filter_and_prioritize(candidates)
            
            # 4-6. Генерация, валидация, создание узлов
            for candidate in filtered[:self.config.get("max_candidates_per_cycle", 5)]:
                self._generate_formulation(candidate)
                if self._validate_candidate(candidate):
                    if not self.config.get("dry_run", False):
                        self._create_contradiction_node(candidate)
                    else:
                        logger.info(f"[DRY RUN] Противоречие: {candidate.title}")

            # Обновление метрик
            processing_time = (time.time() - start_time) * 1000
            self._metrics["processing_time_ms"] = processing_time
            self._metrics["candidates_detected_total"] += len(filtered)

            self._save_candidates()

            logger.info(f"Цикл завершён за {processing_time:.1f}ms")

        except Exception as e:
            logger.error(f"Ошибка цикла обнаружения: {e}", exc_info=True)
        finally:
            self._checking_in_progress = False

    def _detect_candidate_pairs(self) -> List[Tuple[str, str, float, float]]:
        """
        Этап 1: Поиск пар-кандидатов (u, v) с:
        - sim(u,v) >= τ_sim
        - contra(u,v) >= τ_contra
        """
        pairs = []
        
        # Получаем узлы из FGv2
        nodes = self._get_graph_nodes()
        if len(nodes) < 2:
            return pairs

        sim_threshold = self.config.get("sim_threshold", 0.75)
        contra_threshold = self.config.get("contra_threshold", 0.65)
        min_confidence = self.config.get("min_confidence", 0.4)

        node_list = list(nodes.items())
        checked_pairs = set()

        for i, (id_i, node_i) in enumerate(node_list):
            # Пропускаем узлы с низкой уверенностью
            conf_i = node_i.get('confidence', 0.5)
            if conf_i < min_confidence:
                continue

            emb_i = node_i.get('embedding')
            if emb_i is None:
                continue

            for j, (id_j, node_j) in enumerate(node_list[i+1:], i+1):
                # Уникальность пары
                pair_key = tuple(sorted([id_i, id_j]))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                conf_j = node_j.get('confidence', 0.5)
                if conf_j < min_confidence:
                    continue

                emb_j = node_j.get('embedding')
                if emb_j is None:
                    continue

                # Проверка семантической близости
                similarity = self._compute_similarity(emb_i, emb_j)
                if similarity < sim_threshold:
                    continue

                # Проверка логического противоречия
                contradiction = self._compute_contradiction(node_i, node_j)
                if contradiction < contra_threshold:
                    continue

                # Проверка отношений, исключающих противоречие
                if self._has_excluding_relation(id_i, id_j):
                    continue

                pairs.append((id_i, id_j, similarity, contradiction))
                self._metrics["pairs_checked"] += 1

        return pairs

    def _get_graph_nodes(self) -> Dict[str, Dict]:
        """Получение узлов из FGv2"""
        nodes = {}
        
        try:
            fg = None
            if self.brain:
                fg = getattr(self.brain, 'fractal_graph_v2', None)
                if fg is None:
                    kg = getattr(self.brain, 'knowledge_graph', None)
                    if kg and hasattr(kg, '_fg'):
                        fg = kg._fg

            if fg and hasattr(fg, 'storage') and hasattr(fg.storage, 'nodes'):
                for node_id, node in fg.storage.nodes.items():
                    nodes[node_id] = {
                        'id': node_id,
                        'content': getattr(node, 'content', ''),
                        'embedding': getattr(node, 'embedding', None),
                        'confidence': getattr(node, 'confidence', 0.5),
                        'node_type': getattr(node, 'node_type', 'unknown')
                    }
        except Exception as e:
            logger.warning(f"Не удалось получить узлы: {e}")

        return nodes

    def _compute_similarity(self, emb1, emb2) -> float:
        """Косинусное сходство эмбеддингов"""
        try:
            # Проверка кэша
            cache_key = f"{hash(str(emb1))}_{hash(str(emb2))}"
            if cache_key in self._similarity_cache:
                self._metrics["cache_hits"] += 1
                return self._similarity_cache[cache_key]

            v1 = np.array(emb1)
            v2 = np.array(emb2)
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 < 1e-8 or norm2 < 1e-8:
                return 0.0
            
            sim = float(np.dot(v1, v2) / (norm1 * norm2))
            
            self._similarity_cache[cache_key] = sim
            return sim
        except Exception:
            return 0.0

    def _compute_contradiction(self, node1: Dict, node2: Dict) -> float:
        """
        Оценка логического противоречия между узлами.
        
        В реальной реализации должна использоваться NLI-модель (deberta-v3-xsmall-mnli).
        Здесь - эвристика на основе ключевых слов как fallback.
        """
        content1 = node1.get('content', '').lower()
        content2 = node2.get('content', '').lower()
        
        # Простая эвристика: антонимы и противоположности
        contradictions_map = {
            'быстрый': ['медленный', 'медленнее'],
            'медленный': ['быстрый', 'быстрее'],
            'хороший': ['плохой', 'худший'],
            'плохой': ['хороший', 'лучший'],
            'высокий': ['низкий', 'ниже'],
            'низкий': ['высокий', 'выше'],
            'большой': ['маленький', 'меньше'],
            'маленький': ['большой', 'больше'],
            'да': ['нет', 'не'],
            'нет': ['да'],
            'всегда': ['никогда', 'редко'],
            'никогда': ['всегда', 'часто'],
        }
        
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        contradiction_score = 0.0
        
        for word in words1:
            if word in contradictions_map:
                antonyms = contradictions_map[word]
                if any(ant in content2 for ant in antonyms):
                    contradiction_score = 0.7  # Найдено противоречие
                    break
        
        # Дополнительно: проверка на отрицание
        negations = ['не ', 'нет ', 'без ', 'отсутствует']
        has_negation1 = any(n in content1 for n in negations)
        has_negation2 = any(n in content2 for n in negations)
        
        # Если один утверждает, другой отрицает - возможно противоречие
        if has_negation1 != has_negation2:
            # Проверим похожесть остального текста
            clean1 = ' '.join([w for w in content1.split() if w not in negations])
            clean2 = ' '.join([w for w in content2.split() if w not in negations])
            if clean1 and clean2 and len(set(clean1.split()) & set(clean2.split())) > 2:
                contradiction_score = max(contradiction_score, 0.65)
        
        return min(1.0, contradiction_score)

    def _has_excluding_relation(self, id1: str, id2: str) -> bool:
        """Проверка отношений, исключающих противоречие"""
        excluding_relations = {'contextualizes', 'supersedes', 'version_of', 'replaces'}
        
        try:
            fg = None
            if self.brain:
                fg = getattr(self.brain, 'fractal_graph_v2', None)
                if fg is None:
                    kg = getattr(self.brain, 'knowledge_graph', None)
                    if kg and hasattr(kg, '_fg'):
                        fg = kg._fg

            if fg and hasattr(fg, 'storage') and hasattr(fg.storage, 'edges'):
                for edge in fg.storage.edges.values():
                    source = getattr(edge, 'source', None)
                    target = getattr(edge, 'target', None)
                    edge_type = getattr(edge, 'edge_type', '')
                    
                    if (source == id1 and target == id2) or (source == id2 and target == id1):
                        if edge_type in excluding_relations:
                            return True
        except Exception:
            pass
        
        return False

    def _cluster_pairs(self, pairs: List[Tuple[str, str, float, float]]) -> List[ContradictionCandidate]:
        """
        Этап 2: Кластеризация пар через транзитивное замыкание.
        Если (A,B) и (B,C) — кандидаты, то формируется кластер {A,B,C}.
        """
        if not pairs:
            return []

        # Строим граф конфликтов
        graph = defaultdict(set)
        pair_contra = {}
        
        for id1, id2, sim, contra in pairs:
            graph[id1].add(id2)
            graph[id2].add(id1)
            pair_key = tuple(sorted([id1, id2]))
            pair_contra[pair_key] = contra

        # Находим связные компоненты
        visited = set()
        clusters = []
        
        for node in graph:
            if node in visited:
                continue
            
            # BFS для нахождения компоненты
            cluster = []
            queue = [node]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                cluster.append(current)
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            if len(cluster) >= 2:
                clusters.append(cluster)

        # Создаём кандидатов
        nodes_data = self._get_graph_nodes()
        candidates = []
        
        for i, cluster in enumerate(clusters):
            cluster_nodes = [nodes_data.get(nid, {'id': nid, 'content': ''}) for nid in cluster]
            
            # Вычисляем метрики кластера
            avg_sim = np.mean([pair_contra.get(tuple(sorted([cluster[j], cluster[k]])), 0) 
                              for j in range(len(cluster)) for k in range(j+1, len(cluster))])
            max_contra = max([pair_contra.get(tuple(sorted([cluster[j], cluster[k]])), 0) 
                             for j in range(len(cluster)) for k in range(j+1, len(cluster))])
            
            candidate = ContradictionCandidate(
                id=f"contra_cand_{int(time.time() * 1000)}_{i}",
                cluster_id=f"cluster_{i}",
                node_ids=cluster,
                nodes_data=cluster_nodes,
                avg_similarity=avg_sim,
                max_contradiction=max_contra,
                priority=0.0  # Будет рассчитано позже
            )
            candidates.append(candidate)

        return candidates

    def _filter_and_prioritize(self, candidates: List[ContradictionCandidate]) -> List[ContradictionCandidate]:
        """
        Этап 3: Приоритизация кандидатов.
        priority = α·|C| + β·avg_confidence + γ·max_contra_score
        """
        coeffs = self.config.get("priority_coefficients", {"alpha": 0.4, "beta": 0.3, "gamma": 0.3})
        alpha = coeffs.get("alpha", 0.4)
        beta = coeffs.get("beta", 0.3)
        gamma = coeffs.get("gamma", 0.3)

        for candidate in candidates:
            cluster_size = len(candidate.node_ids)
            avg_confidence = np.mean([n.get('confidence', 0.5) for n in candidate.nodes_data])
            max_contra = candidate.max_contradiction
            
            priority = (alpha * cluster_size + 
                       beta * avg_confidence + 
                       gamma * max_contra)
            
            candidate.priority = priority
            candidate.metadata['priority_calc'] = {
                'alpha_term': alpha * cluster_size,
                'beta_term': beta * avg_confidence,
                'gamma_term': gamma * max_contra
            }

        # Сортируем по приоритету
        candidates.sort(key=lambda x: x.priority, reverse=True)
        
        return candidates

    def _generate_formulation(self, candidate: ContradictionCandidate):
        """Этап 4: Генерация формулировки противоречия через LLM"""
        try:
            # Формируем user prompt
            nodes_formatted = "\n".join([
                f"Узел [{n.get('id', '?')}]: \"{n.get('content', '')[:150]}\""
                for n in candidate.nodes_data
            ])

            user_prompt = f"""Проанализируй следующие утверждения из базы знаний EVA-Ai:

{nodes_formatted}

Все эти утверждения относятся к одной теме, но содержат логическое противоречие.
Сформулируй противоречие по указанному формату."""

            response = self._call_pipeline(user_prompt)

            if response:
                lines = response.split('\n')
                for line in lines:
                    if line.startswith("CONTRADICTION_TITLE:"):
                        candidate.title = line.replace("CONTRADICTION_TITLE:", "").strip()
                    elif line.startswith("DESCRIPTION:"):
                        candidate.description = line.replace("DESCRIPTION:", "").strip()
                    elif line.startswith("RESOLUTION_QUESTION:"):
                        candidate.resolution_question = line.replace("RESOLUTION_QUESTION:", "").strip()

                # Если не удалось распарсить - берём первую строку как заголовок
                if not candidate.title:
                    candidate.title = response.split('\n')[0][:100]
                    candidate.description = response[:300]
                    candidate.resolution_question = "Какое из утверждений более корректно?"

        except Exception as e:
            logger.error(f"Ошибка генерации формулировки: {e}")
            candidate.title = f"Противоречие между узлами {candidate.node_ids[:2]}"
            candidate.description = "Обнаружено логическое противоречие"
            candidate.resolution_question = "Как разрешить данное противоречие?"

    def _call_pipeline(self, user_prompt: str) -> str:
        """
        Вызов EVAGenerator через HybridPipelineAdapter.
        
        НЕ используем GGUF pipeline - только EVAGenerator (основная система).
        """
        try:
            # Основной метод: HybridPipelineAdapter (использует EVAGenerator)
            if self.brain and hasattr(self.brain, 'hybrid_adapter') and self.brain.hybrid_adapter:
                result = self.brain.hybrid_adapter.generate(
                    prompt=user_prompt,
                    system_prompt=self.SYSTEM_PROMPT,
                    temperature=self.config.get("llm_temperature", 0.25),
                    repeat_penalty=self.config.get("llm_repeat_penalty", 1.5),
                    max_tokens=self.config.get("llm_max_tokens", 200)
                )
                if result and isinstance(result, dict):
                    return result.get('text', result.get('response', ''))
                return str(result) if result else ""

            # Fallback: прямой доступ к fractal_pipeline
            elif self.brain and hasattr(self.brain, 'fractal_pipeline') and self.brain.fractal_pipeline:
                result = self.brain.fractal_pipeline.process_query(
                    query=user_prompt,
                    max_tokens=self.config.get("llm_max_tokens", 200),
                    temperature=self.config.get("llm_temperature", 0.25)
                )
                return result.get('response', '') if isinstance(result, dict) else str(result)

            else:
                logger.debug("EVAGenerator не доступен, пропускаем генерацию")
                return ""

        except Exception as e:
            logger.warning(f"Ошибка вызова EVAGenerator: {e}")

        return ""

    def _validate_candidate(self, candidate: ContradictionCandidate) -> bool:
        """Этап 5: Валидация кандидата"""
        # Проверяем, что сформулировано что-то осмысленное
        if not candidate.title or len(candidate.title) < 5:
            return False
        
        if not candidate.description or len(candidate.description) < 10:
            return False

        # Проверка на дубликаты
        for existing in self._candidates.values():
            if set(existing.node_ids) == set(candidate.node_ids):
                logger.debug("Дубликат противоречия, пропускаем")
                return False

        return True

    def _create_contradiction_node(self, candidate: ContradictionCandidate):
        """Этап 6: Создание ContradictionNode в FGv2"""
        try:
            fg = None
            if self.brain:
                fg = getattr(self.brain, 'fractal_graph_v2', None)
                if fg is None:
                    kg = getattr(self.brain, 'knowledge_graph', None)
                    if kg and hasattr(kg, '_fg'):
                        fg = kg._fg

            if not fg or not hasattr(fg, 'add_node'):
                return

            # Создаём узел-противоречие
            node_content = {
                'title': candidate.title,
                'description': candidate.description,
                'resolution_question': candidate.resolution_question
            }

            node = fg.add_node(
                content=json.dumps(node_content, ensure_ascii=False),
                node_type='contradiction',
                metadata={
                    'status': 'active',
                    'cluster_size': len(candidate.node_ids),
                    'max_contra_score': candidate.max_contradiction,
                    'priority': candidate.priority,
                    'source_nodes': candidate.node_ids,
                    'candidate_id': candidate.id
                }
            )

            if node:
                node_id = node.id if hasattr(node, 'id') else str(node)
                candidate.metadata['node_id'] = node_id
                
                # Создаём связи contradicts
                if hasattr(fg, 'add_edge'):
                    for source_id in candidate.node_ids:
                        fg.add_edge(node_id, source_id, edge_type='contradicts')
                        fg.add_edge(source_id, node_id, edge_type='contradicted_by')

                self._candidates[candidate.id] = candidate
                self._metrics["nodes_created_total"] += 1

                logger.info(f"Создан ContradictionNode: {candidate.title}")

                # Публикуем событие
                self._publish_event("contradiction.node_created", {
                    "node_id": node_id,
                    "title": candidate.title,
                    "priority": candidate.priority
                })

                # Добавляем в очередь самодиалога
                if hasattr(self.brain, 'self_dialog_learning') and self.brain.self_dialog_learning:
                    self.brain.self_dialog_learning.queue_contradiction_for_resolution(
                        contr_id=node_id,
                        concept=candidate.title,
                        priority=candidate.priority
                    )

        except Exception as e:
            logger.error(f"Ошибка создания узла-противоречия: {e}")

    def _publish_event(self, event_type: str, data: Dict):
        """Публикация события"""
        if self.event_bus and hasattr(self.event_bus, 'publish'):
            try:
                self.event_bus.publish(event_type, data)
            except Exception as e:
                logger.debug(f"Ошибка публикации: {e}")

    def get_metrics(self) -> Dict:
        """Получение метрик"""
        return {
            **self._metrics,
            "active_candidates": len(self._candidates),
            "status": "running" if self._running else "stopped"
        }

    def get_candidates(self, status: str = None) -> List[Dict]:
        """Получение кандидатов"""
        if status:
            return [asdict(c) for c in self._candidates.values() if c.status == status]
        return [asdict(c) for c in self._candidates.values()]

    def resolve_contradiction(self, candidate_id: str, resolution_node_id: str):
        """Отметить противоречие как разрешённое"""
        if candidate_id in self._candidates:
            candidate = self._candidates[candidate_id]
            candidate.status = "resolved"
            candidate.resolved_at = time.time()
            candidate.resolution_node_id = resolution_node_id
            
            self._metrics["resolved_total"] += 1
            self._save_candidates()
            
            self._publish_event("contradiction.resolved", {
                "candidate_id": candidate_id,
                "resolution_node_id": resolution_node_id
            })

    def force_check(self):
        """Принудительный запуск проверки"""
        self._schedule_check()


def create_contradiction_miner(
    brain=None,
    event_bus=None,
    deferred_system=None,
    config: Dict = None
) -> ContradictionMiner:
    """Фабрика создания ContradictionMiner"""
    return ContradictionMiner(
        brain=brain,
        event_bus=event_bus,
        deferred_system=deferred_system,
        config=config
    )


__all__ = [
    'ContradictionMiner',
    'ContradictionStatus',
    'ContradictionCandidate',
    'create_contradiction_miner'
]
