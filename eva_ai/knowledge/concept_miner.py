"""
ConceptMiner - модуль автономного концептуального вывода для EVA-Ai
Реализует проактивное обнаружение семантических лакун во фрактальном графе памяти (FGv2)

По спецификации: Техническая спецификация модуля ACI (ConceptMiner) v3.1

Интеграция с ConceptExtractor:
- ConceptExtractor: быстрое извлечение из запросов/ответов (поверхностный уровень)
- ConceptMiner: глубокий анализ кластеров FGv2 (семантические лакуны)
"""

import os
import time
import json
import logging
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("eva_ai.knowledge.concept_miner")


class ConceptStatus(Enum):
    """Жизненный цикл концепта"""
    PROVISIONAL = "provisional"
    CONFIRMED = "confirmed"
    STABLE = "stable"
    ARCHIVED = "archived"


@dataclass
class PhantomCandidate:
    """Кандидат концепта (фантомная сущность)"""
    id: str
    cluster_id: str
    centroid: List[float]
    nodes: List[Dict]
    variance: float
    semantic_gap: float

    title: str = ""
    definition: str = ""
    rationale: str = ""

    status: str = "provisional"
    confidence: float = 0.0

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    confirmations: int = 0
    rejections: int = 0

    parent_group_id: Optional[str] = None

    validation_nli: Optional[Dict] = None
    validation_ontology: Optional[Dict] = None
    validation_ethics: Optional[Dict] = None
    web_verification: Optional[Dict] = None


class ConceptMiner:
    """
    Модуль автономного концептуального вывода (ACI / ConceptMiner)
    
    Работает на глубоком уровне с FGv2:
    - Детекция семантических лакун через анализ кластеров
    - Генерация гипотез через EVAGenerator (основная система генерации)
    - Валидация (NLI, Ontology, Ethics, Web)
    - Жизненный цикл концептов
    
    Отличие от ConceptExtractor:
    - ConceptExtractor: быстрое извлечение из текста запроса (поверхностно)
    - ConceptMiner: анализ структуры графа (семантические лакуны в кластерах)
    """

    DEFAULT_CONFIG = {
        "enabled": True,
        "dry_run": False,  # Теперь реально интегрируем
        "base_threshold": 0.30,
        "dedup_radius": 0.15,
        "max_candidates_per_cycle": 3,
        "priority_queue": "NORMAL",
        "llm_temperature": 0.35,
        "llm_repeat_penalty": 1.8,
        "max_llm_tokens": 128,
        "enable_web_search_validation": True,
        "idle_threshold_seconds": 10.0,
        "cpu_threshold_soft": 0.80,
        "cpu_threshold_hard": 0.90,
        "cycles_before_stable": 5,
        "confidence_threshold_confirm": 0.75,
        "confidence_threshold_archive": 0.25,
        "variance_k": 2.0
    }

    SYSTEM_PROMPT = """Ты — модуль концептуального анализа когнитивной системы EVA-Ai.
Твоя задача — выявить обобщающее понятие для предоставленного семантического кластера.
Ответ должен быть точным, логически обоснованным и соответствовать общепринятым нормам классификации.
Не выдумывай несуществующие термины. Если обобщение невозможно или кластер семантически разнороден, верни строго: "NULL"."""

    def __init__(
        self,
        brain=None,
        event_bus=None,
        deferred_system=None,
        background_coordinator=None,
        config: Dict = None
    ):
        self.brain = brain
        self.event_bus = event_bus
        self.deferred_system = deferred_system
        self.background_coordinator = background_coordinator

        self.config = {**self.DEFAULT_CONFIG}
        if config:
            self.config.update(config)

        self._running = False
        self._subscription_ids = []
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ConceptMiner")

        self._candidates: Dict[str, PhantomCandidate] = {}
        self._load_candidates()

        self._last_mining_time = 0
        self._mining_in_progress = False

        self._metrics = {
            "phantom_detection_rate": 0.0,
            "hypothesis_confirmation_ratio": 0.0,
            "graph_coherence_delta": 0.0,
            "dry_run_skipped": 0,
            "validation_rejection_rate": 0.0,
            "total_mining_cycles": 0,
            "candidates_generated": 0,
            "candidates_confirmed": 0,
            "candidates_archived": 0,
            "candidates_rejected": 0
        }

        self._audit_log: List[Dict] = []
        self._load_audit_log()

        logger.info("ConceptMiner инициализирован")

    def _load_audit_log(self):
        """Загрузка аудит-лога отклонённых кандидатов"""
        try:
            storage_dir = self._get_storage_dir()
            audit_file = os.path.join(storage_dir, "phantom_audit_log.json")
            if os.path.exists(audit_file):
                with open(audit_file, 'r', encoding='utf-8') as f:
                    self._audit_log = json.load(f)
                logger.info(f"Загружено {len(self._audit_log)} записей аудит-лога")
        except Exception as e:
            logger.warning(f"Не удалось загрузить аудит-лог: {e}")

    def _save_audit_log(self):
        """Сохранение аудит-лога"""
        try:
            storage_dir = self._get_storage_dir()
            os.makedirs(storage_dir, exist_ok=True)
            audit_file = os.path.join(storage_dir, "phantom_audit_log.json")
            with open(audit_file, 'w', encoding='utf-8') as f:
                json.dump(self._audit_log, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Не удалось сохранить аудит-лог: {e}")

    def _log_rejection(self, candidate: PhantomCandidate, reason: str):
        """Запись отклонённого кандидата в аудит-лог"""
        entry = {
            "id": candidate.id,
            "cluster_hash": hash(candidate.cluster_id) % 10000,
            "reason": reason,
            "variance": candidate.variance,
            "confidence": candidate.confidence,
            "timestamp": time.time()
        }
        self._audit_log.append(entry)
        self._save_audit_log()

    def _load_candidates(self):
        """Загрузка кандидатов из storage"""
        try:
            storage_dir = self._get_storage_dir()
            candidates_file = os.path.join(storage_dir, "phantom_candidates.json")
            if os.path.exists(candidates_file):
                with open(candidates_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for cid, cdata in data.items():
                        self._candidates[cid] = PhantomCandidate(**cdata)
                logger.info(f"Загружено {len(self._candidates)} кандидатов")
        except Exception as e:
            logger.warning(f"Не удалось загрузить кандидатов: {e}")

    def _save_candidates(self):
        """Сохранение кандидатов в storage"""
        try:
            storage_dir = self._get_storage_dir()
            os.makedirs(storage_dir, exist_ok=True)
            candidates_file = os.path.join(storage_dir, "phantom_candidates.json")
            data = {cid: asdict(c) for cid, c in self._candidates.items()}
            with open(candidates_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Не удалось сохранить кандидатов: {e}")

    def _get_storage_dir(self) -> str:
        """Получение директории для хранения"""
        if self.brain and hasattr(self.brain, 'cache_dir'):
            return os.path.join(self.brain.cache_dir, "concept_miner_data")
        return os.path.join(os.path.dirname(__file__), "concept_miner_data")

    def start(self):
        """Запуск модуля - подписка на события"""
        if not self.config.get("enabled", True):
            logger.info("ConceptMiner отключён в конфигурации")
            return

        if self._running:
            logger.warning("ConceptMiner уже запущен")
            return

        self._running = True

        if self.event_bus:
            self._subscribe_to_events()

        logger.info("ConceptMiner запущен")

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
        self._save_audit_log()

        logger.info("ConceptMiner остановлен")

    def _subscribe_to_events(self):
        """Подписка на события EventBus"""
        if not self.event_bus:
            return

        # Универсальная подписка на события
        events_to_subscribe = [
            ("memory.graph_updated", "_on_memory_graph_updated"),
            ("memory.clustering_complete", "_on_memory_clustering_complete"),
            ("pipeline.complete", "_on_pipeline_complete"),
            ("system.ready", "_on_system_ready"),
            ("system.idle", "_on_system_idle"),
        ]

        for event_type, handler_name in events_to_subscribe:
            try:
                if hasattr(self, handler_name):
                    handler = getattr(self, handler_name)
                    # Пробуем подписаться через разные API
                    if hasattr(self.event_bus, 'subscribe'):
                        try:
                            sub_id = self.event_bus.subscribe(event_type, handler, priority=8)
                            self._subscription_ids.append(sub_id)
                            logger.debug(f"Подписка на {event_type}: {sub_id}")
                        except Exception as e:
                            logger.debug(f"Ошибка подписки на {event_type}: {e}")
            except Exception as e:
                logger.warning(f"Не удалось подписаться на {event_type}: {e}")

        logger.info(f"ConceptMiner подписан на {len(self._subscription_ids)} событий")

    def _on_memory_graph_updated(self, event):
        """Обработка обновления графа памяти - запуск майнинга"""
        self._schedule_mining_if_needed()

    def _on_memory_clustering_complete(self, event):
        """Обработка завершения кластеризации"""
        data = event.data if hasattr(event, 'data') else event if isinstance(event, dict) else {}
        cluster_data = data.get('clusters', {}) if isinstance(data, dict) else {}

        if cluster_data:
            self._executor.submit(self._mine_concepts_from_clusters, cluster_data)

    def _on_pipeline_complete(self, event):
        """После завершения пайплайна - проверка idle"""
        self._schedule_mining_if_needed()

    def _on_system_ready(self, event):
        """При готовности системы"""
        self._schedule_mining_if_needed()

    def _on_system_idle(self, event):
        """При простое системы - основное время для майнинга"""
        self._schedule_mining_if_needed()

    def _schedule_mining_if_needed(self):
        """Планирование майнинга если система в простое"""
        if not self._running or self._mining_in_progress:
            return

        if not self._can_mine():
            return

        min_interval = 60.0
        if time.time() - self._last_mining_time < min_interval:
            return

        self._last_mining_time = time.time()

        if self.deferred_system and self.config.get("priority_queue") != "DISABLED":
            try:
                from eva_ai.core.deferred_command_system import CommandPriority

                priority_map = {
                    "CRITICAL": CommandPriority.CRITICAL,
                    "HIGH": CommandPriority.HIGH,
                    "NORMAL": CommandPriority.NORMAL,
                    "LOW": CommandPriority.LOW
                }
                priority = priority_map.get(self.config.get("priority_queue", "NORMAL"), CommandPriority.NORMAL)

                self.deferred_system.add_command(
                    command=self._mining_cycle,
                    priority=priority,
                    max_retries=2,
                    retry_delay=5.0,
                    command_id=f"concept_mining_{int(time.time())}"
                )
                logger.info("Concept mining запланирован через DeferredCommandSystem")
            except Exception as e:
                logger.debug(f"Deferred system не доступен: {e}")
                self._executor.submit(self._mining_cycle)
        else:
            self._executor.submit(self._mining_cycle)

    def _can_mine(self) -> bool:
        """Проверка возможности майнинга (ресурсы + idle)"""
        if self.background_coordinator:
            try:
                if hasattr(self.background_coordinator, '_can_run_background'):
                    if not self.background_coordinator._can_run_background():
                        return False
            except Exception:
                pass

        if self.config.get("cpu_threshold_hard"):
            try:
                if self.brain and hasattr(self.brain, 'resource_manager'):
                    rm = self.brain.resource_manager
                    if hasattr(rm, 'get_cpu_usage'):
                        import psutil
                        cpu = psutil.cpu_percent(interval=0.1)
                        if cpu > self.config.get("cpu_threshold_hard", 0.90):
                            logger.debug(f"CPU слишком высок: {cpu:.1%}, майнинг отложен")
                            return False
            except Exception:
                pass

        return True

    def _mining_cycle(self):
        """Основной цикл майнинга концептов"""
        if self._mining_in_progress:
            return

        self._mining_in_progress = True

        try:
            logger.info("Начат цикл майнинга концептов")
            self._metrics["total_mining_cycles"] += 1

            clusters = self._get_clusters()
            if not clusters:
                logger.debug("Нет кластеров для анализа")
                return

            candidates = self._detect_semantic_gaps(clusters)

            for candidate in candidates[:self.config.get("max_candidates_per_cycle", 3)]:
                self._generate_hypothesis(candidate)
                self._validate_candidate(candidate)

                if not self.config.get("dry_run", True):
                    self._integrate_candidate(candidate)
                else:
                    self._metrics["dry_run_skipped"] += 1

            self._update_lifecycle()
            self._save_candidates()

            logger.info(f"Цикл майнинга завершён: {len(candidates)} кандидатов")

        except Exception as e:
            logger.error(f"Ошибка майнинга: {e}", exc_info=True)
        finally:
            self._mining_in_progress = False

    def _mine_concepts_from_clusters(self, clusters: Dict):
        """Майнинг концептов из предоставленных кластеров"""
        if not self._running or self._mining_in_progress:
            return

        self._mining_in_progress = True

        try:
            logger.info(f"Майнинг из {len(clusters)} кластеров")
            self._metrics["total_mining_cycles"] += 1

            candidates = self._detect_semantic_gaps(clusters)

            for candidate in candidates[:self.config.get("max_candidates_per_cycle", 3)]:
                self._generate_hypothesis(candidate)
                self._validate_candidate(candidate)

                if not self.config.get("dry_run", True):
                    self._integrate_candidate(candidate)
                else:
                    self._metrics["dry_run_skipped"] += 1

            self._update_lifecycle()
            self._save_candidates()

            logger.info(f"Майнинг из кластеров завершён: {len(candidates)} кандидатов")
        except Exception as e:
            logger.error(f"Ошибка майнинга из кластеров: {e}", exc_info=True)
        finally:
            self._mining_in_progress = False

    def _get_clusters(self) -> Dict:
        """Получение кластеров из FractalGraph v2"""
        clusters = {}

        try:
            # Получаем доступ к FGv2
            fg = None
            if self.brain:
                fg = getattr(self.brain, 'fractal_graph_v2', None)
                if fg is None:
                    fg = getattr(self.brain, 'knowledge_graph', None)
                    if fg and hasattr(fg, '_fg'):
                        fg = fg._fg

            if fg and hasattr(fg, 'storage'):
                storage = fg.storage
                
                # Пробуем получить семантические группы как кластеры
                if hasattr(storage, 'semantic_groups'):
                    for group_id, group in storage.semantic_groups.items():
                        if hasattr(group, 'node_ids'):
                            clusters[group_id] = group.node_ids
                        elif isinstance(group, dict) and 'node_ids' in group:
                            clusters[group_id] = group['node_ids']
                
                # Если нет групп, используем кэшированные кластеры из FractalGraph
                if not clusters and hasattr(storage, 'nodes'):
                    fg = getattr(self, 'fractal_graph', None) or getattr(self, '_fg', None)
                    if fg and hasattr(fg, 'get_clusters'):
                        clusters = fg.get_clusters()
                    else:
                        logger.warning("FractalGraph.get_clusters() не доступен, используем O(n²) вычисление")
                        if nodes_with_embeddings := [
                            (nid, np.array(emb)) for nid, n in storage.nodes.items() 
                            if (emb := getattr(n, 'embedding', None)) is not None
                        ]:
                            visited = set()
                            for i, (nid_i, emb_i) in enumerate(nodes_with_embeddings):
                                if nid_i in visited:
                                    continue
                                cl_nodes = [nid_i]
                                visited.add(nid_i)
                                for j, (nid_j, emb_j) in enumerate(nodes_with_embeddings[i+1:], i+1):
                                    if nid_j in visited:
                                        continue
                                    n_i, n_j = np.linalg.norm(emb_i), np.linalg.norm(emb_j)
                                    if n_i > 0 and n_j > 0 and np.dot(emb_i, emb_j) / (n_i * n_j) > 0.7:
                                        cl_nodes.append(nid_j)
                                        visited.add(nid_j)
                                if len(cl_nodes) >= 3:
                                    clusters[f"auto_cluster_{len(clusters)}"] = cl_nodes

        except Exception as e:
            logger.warning(f"Не удалось получить кластеры: {e}")

        return clusters

    def _detect_semantic_gaps(self, clusters: Dict) -> List[PhantomCandidate]:
        """
        Детекция семантических лакун (алгоритм по спецификации)

        Расчёт центроида: μC = (1/|C|)Σv
        Семантический разрыв: ΔC = min(1 - cos(μC, v))
        Внутрикластерная дисперсия: σ²C = (1/|C|)Σ cos_dist(μC, v)
        Адаптивный порог: τ = τ_base * (1 + variance_k * σC)
        """
        candidates = []
        base_threshold = self.config.get("base_threshold", 0.30)
        dedup_radius = self.config.get("dedup_radius", 0.15)
        variance_k = self.config.get("variance_k", 2.0)

        for cluster_id, node_ids in clusters.items():
            if len(node_ids) < 3:
                continue

            try:
                nodes_data = self._get_cluster_nodes_data(node_ids)
                if not nodes_data or len(nodes_data) < 3:
                    continue

                embeddings = [n.get('embedding') for n in nodes_data if n.get('embedding')]
                if not embeddings:
                    continue

                embeddings = [np.array(e) for e in embeddings if len(e) > 0]
                if not embeddings:
                    continue

                centroid = np.mean(embeddings, axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 1e-8:
                    centroid = centroid / centroid_norm

                gaps = []
                for emb in embeddings:
                    emb_norm = np.linalg.norm(emb)
                    if emb_norm > 1e-8:
                        cos_sim = np.dot(centroid, emb) / emb_norm
                    else:
                        cos_sim = 0.0
                    gap = 1.0 - cos_sim
                    gaps.append(gap)

                if not gaps:
                    continue

                variance = float(np.std(gaps))
                semantic_gap = float(min(gaps))

                threshold = base_threshold * (1 + variance_k * variance)

                if semantic_gap > threshold:
                    if not self._is_duplicate(centroid, dedup_radius):
                        candidate = PhantomCandidate(
                            id=f"phantom_{int(time.time() * 1000)}_{hash(cluster_id) % 10000}",
                            cluster_id=cluster_id,
                            centroid=centroid.tolist(),
                            nodes=nodes_data,
                            variance=variance,
                            semantic_gap=semantic_gap
                        )
                        candidates.append(candidate)

            except Exception as e:
                logger.warning(f"Ошибка анализа кластера {cluster_id}: {e}")

        self._metrics["phantom_detection_rate"] = len(candidates) / max(len(clusters), 1)

        return candidates

    def _get_cluster_nodes_data(self, node_ids: List[str]) -> List[Dict]:
        """Получение данных узлов кластера из FGv2"""
        nodes_data = []

        try:
            fg = None
            if self.brain:
                fg = getattr(self.brain, 'fractal_graph_v2', None)
                if fg is None:
                    kg = getattr(self.brain, 'knowledge_graph', None)
                    if kg and hasattr(kg, '_fg'):
                        fg = kg._fg

            if fg and hasattr(fg, 'storage') and hasattr(fg.storage, 'nodes'):
                for nid in node_ids:
                    node = fg.storage.nodes.get(nid)
                    if node:
                        nodes_data.append({
                            'id': getattr(node, 'id', nid),
                            'content': getattr(node, 'content', ''),
                            'embedding': getattr(node, 'embedding', None),
                            'node_type': getattr(node, 'node_type', 'unknown')
                        })
        except Exception as e:
            logger.warning(f"Не удалось получить данные узлов: {e}")

        return nodes_data

    def _is_duplicate(self, centroid: np.ndarray, radius: float) -> bool:
        """Проверка на дубликат среди существующих концептов"""
        for candidate in self._candidates.values():
            if candidate.status in ["confirmed", "stable"] and candidate.centroid:
                existing = np.array(candidate.centroid)
                existing_norm = np.linalg.norm(existing)
                if existing_norm > 1e-8:
                    cos_sim = np.dot(centroid, existing) / existing_norm
                else:
                    cos_sim = 0.0
                if (1.0 - cos_sim) < radius:
                    return True
        return False

    def _generate_hypothesis(self, candidate: PhantomCandidate):
        """
        Генерация гипотезы концепта через EVAGenerator (основная система генерации)
        """
        self._publish_event("concept.mining.start", {"candidate_id": candidate.id})

        try:
            cluster_nodes_formatted = "\n".join([
                f"- {n.get('content', '')[:100]}" for n in candidate.nodes[:10]
            ])

            user_prompt = f"""Анализируй следующий семантический кластер узлов фрактальной памяти:
{cluster_nodes_formatted}

Типы связей в кластере: {', '.join(set(n.get('node_type', 'fact') for n in candidate.nodes))}
Семантическая дисперсия: {candidate.variance:.3f}

Сформулируй одно обобщающее понятие (концепт), которое логически объединяет данные элементы.
В ответе укажи:
1. Название концепта (1–3 слова)
2. Краткое определение (до 50 слов)
3. Обоснование объединения (1 предложение)

Формат вывода строго:
CONCEPT: [название]
DEFINITION: [определение]
RATIONALE: [обоснование]"""

            response = self._call_pipeline(user_prompt, self.SYSTEM_PROMPT)

            if response and "NULL" not in response:
                lines = response.split('\n')
                for line in lines:
                    if line.startswith("CONCEPT:"):
                        candidate.title = line.replace("CONCEPT:", "").strip()
                    elif line.startswith("DEFINITION:"):
                        candidate.definition = line.replace("DEFINITION:", "").strip()
                    elif line.startswith("RATIONALE:"):
                        candidate.rationale = line.replace("RATIONALE:", "").strip()

                if not candidate.title:
                    candidate.title = response.split('\n')[0][:50]
                    candidate.definition = response[:200]

                candidate.confidence = 0.5
                self._metrics["candidates_generated"] += 1

                lexical_entropy = self._calculate_lexical_entropy(candidate.definition)
                if lexical_entropy < 2.1:
                    logger.info(f"Низкая энтропия ({lexical_entropy:.2f}), повторный проход с повышенной температурой")
                    response2 = self._call_pipeline(user_prompt, self.SYSTEM_PROMPT, temp_override=0.65)
                    if response2 and response2 != response:
                        candidate.definition = response2[:200]
                        candidate.confidence = 0.6

        except Exception as e:
            logger.error(f"Ошибка генерации гипотезы: {e}")

        self._publish_event("concept.candidate.generated", {
            "candidate_id": candidate.id, 
            "title": candidate.title
        })

    def _publish_event(self, event_type: str, data: Dict):
        """Публикация события"""
        if self.event_bus and hasattr(self.event_bus, 'publish'):
            try:
                self.event_bus.publish(event_type, data)
            except Exception as e:
                logger.debug(f"Ошибка публикации события {event_type}: {e}")

    def _call_pipeline(self, user_prompt: str, system_prompt: str = None, temp_override: float = None) -> str:
        """
        Вызов EVAGenerator через HybridPipelineAdapter.
        
        НЕ используем GGUF pipeline - только EVAGenerator (основная система).
        Если EVAGenerator недоступен - возвращаем пустую строку.
        """
        temp = temp_override if temp_override else self.config.get('llm_temperature', 0.35)

        try:
            # Основной метод: HybridPipelineAdapter (использует EVAGenerator)
            if self.brain and hasattr(self.brain, 'hybrid_adapter') and self.brain.hybrid_adapter:
                result = self.brain.hybrid_adapter.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=temp,
                    repeat_penalty=self.config.get('llm_repeat_penalty', 1.8),
                    max_tokens=self.config.get('max_llm_tokens', 128)
                )
                if result and isinstance(result, dict):
                    return result.get('text', result.get('response', ''))
                return str(result) if result else ""

            # Fallback: прямой доступ к fractal_pipeline если есть
            elif self.brain and hasattr(self.brain, 'fractal_pipeline') and self.brain.fractal_pipeline:
                result = self.brain.fractal_pipeline.process_query(
                    query=user_prompt,
                    max_tokens=self.config.get('max_llm_tokens', 128),
                    temperature=temp
                )
                return result.get('response', '') if isinstance(result, dict) else str(result)

            else:
                logger.debug("EVAGenerator не доступен, пропускаем генерацию")
                return ""

        except Exception as e:
            logger.warning(f"Ошибка вызова EVAGenerator: {e}")

        return ""

    def _calculate_lexical_entropy(self, text: str) -> float:
        """Расчёт лексической энтропии для адаптации параметров"""
        if not text:
            return 0.0

        words = text.lower().split()
        if len(words) < 2:
            return 0.0

        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        total = len(words)
        entropy = 0.0
        for freq in word_freq.values():
            p = freq / total
            if p > 0:
                entropy -= p * (p ** 0.5)

        return entropy

    def _validate_candidate(self, candidate: PhantomCandidate):
        """
        Валидация кандидата (многоуровневая):
        1. NLI-когерентность
        2. Онтологическая совместимость
        3. Этическая фильтрация
        4. Веб-верификация (опционально)
        """
        context_summary = " ".join([n.get('content', '')[:50] for n in candidate.nodes[:5]])

        candidate.validation_nli = self._check_nli_coherence(context_summary, candidate)
        candidate.validation_ontology = self._check_ontology_compliance(candidate)
        candidate.validation_ethics = self._check_ethics(candidate)

        if candidate.validation_nli.get("status") == "contradiction":
            candidate.rejections += 1
            self._metrics["candidates_rejected"] += 1
            candidate.status = "archived"
            self._log_rejection(candidate, "NLI contradiction")
            logger.info(f"Кандидат {candidate.id} отклонён: противоречие NLI")
            return

        if not candidate.validation_ontology.get("compliant", True):
            candidate.rejections += 1
            self._metrics["candidates_rejected"] += 1
            candidate.status = "archived"
            reason = candidate.validation_ontology.get("reason", "Ontology violation")
            self._log_rejection(candidate, reason)
            logger.info(f"Кандидат {candidate.id} отклонён: онтологическое несоответствие")
            return

        if candidate.validation_ethics.get("risk_level") == "high":
            candidate.rejections += 1
            self._metrics["candidates_rejected"] += 1
            candidate.status = "archived"
            reason = candidate.validation_ethics.get("reason", "Ethics violation")
            self._log_rejection(candidate, reason)
            logger.info(f"Кандидат {candidate.id} отклонён: этика")
            return

        if self.config.get("enable_web_search_validation", True):
            candidate.web_verification = self._verify_web(candidate)

        candidate.status = "provisional"
        candidate.updated_at = time.time()

        self._candidates[candidate.id] = candidate

        self._publish_event("concept.validation.complete", {
            "candidate_id": candidate.id, 
            "status": candidate.status
        })

    def _check_nli_coherence(self, context: str, candidate: PhantomCandidate) -> Dict:
        """Проверка NLI-когерентности"""
        nli_prompt = f"""Оцени логическую согласованность гипотетического концепта с существующим фрагментом графа знаний.

Контекст кластера: {context}
Гипотеза: {candidate.title} — {candidate.definition}

Вопрос: Противоречит ли гипотеза установленным связям в кластере или нарушает онтологические правила?

Ответь строго в формате JSON:
{{"status": "entailment" | "neutral" | "contradiction", "confidence": 0.0-1.0, "reason": "краткое объяснение"}}"""

        response = self._call_pipeline(nli_prompt)

        try:
            for line in response.split('\n'):
                if line.startswith('{') and line.endswith('}'):
                    result = json.loads(line)
                    status = result.get("status", "neutral")
                    conf = result.get("confidence", 0.5)

                    if status == "entailment":
                        candidate.confidence = min(1.0, candidate.confidence + 0.25)

                    return result
        except Exception:
            pass

        return {"status": "neutral", "confidence": 0.5, "reason": "Ошибка проверки"}

    def _check_ethics(self, candidate: PhantomCandidate) -> Dict:
        """Этическая фильтрация"""
        ethics_prompt = f"""Проверь гипотетический концепт "{candidate.title}" и его определение на соответствие этическим принципам:
1. Без насилия
2. Честность
3. Проверка фактов
4. Безопасный код
5. Блокировка рисков
6. Контроль вывода

Если концепт потенциально содержит вредоносные, дискриминационные или дезинформирующие элементы,
верни строго: {{"risk_level": "high", "block": true, "reason": "..."}}
В противном случае: {{"risk_level": "low", "block": false}}"""

        response = self._call_pipeline(ethics_prompt)

        try:
            for line in response.split('\n'):
                if line.startswith('{') and line.endswith('}'):
                    return json.loads(line)
        except Exception:
            pass

        return {"risk_level": "low", "block": False}

    def _check_ontology_compliance(self, candidate: PhantomCandidate) -> Dict:
        """Онтологическая проверка концепта"""
        # Простая проверка: минимальное количество связей
        min_connections = 3
        if len(candidate.nodes) < min_connections:
            return {
                "compliant": False,
                "reason": f"Недостаточно связей в кластере: {len(candidate.nodes)} < {min_connections}"
            }

        return {"compliant": True, "reason": "Все онтологические проверки пройдены"}

    def _verify_web(self, candidate: PhantomCandidate) -> Dict:
        """Веб-верификация концепта"""
        web_prompt = f"""В системе обнаружена семантическая лакуна, соответствующая концепту "{candidate.title}".
Текущее определение: "{candidate.definition}".

Сгенерируй оптимизированный поисковый запрос на русском языке для проверки существования и корректности данного термина.

Верни строго JSON: {{"query": "...", "domains": ["..."], "language": "ru"}}"""

        response = self._call_pipeline(web_prompt)

        try:
            for line in response.split('\n'):
                if line.startswith('{') and line.endswith('}'):
                    search_data = json.loads(line)

                    # Пробуем веб-поиск
                    if self.brain and hasattr(self.brain, 'web_search') and self.brain.web_search:
                        try:
                            results = self.brain.web_search.search(
                                query=search_data.get('query', ''),
                                language='ru'
                            )
                            return {"verified": True, "results_count": len(results), "source_quality": "medium"}
                        except Exception:
                            pass

                    return {"verified": False, "confidence_delta": 0.0, "source_quality": "low"}
        except Exception as e:
            logger.warning(f"Веб-верификация не удалась: {e}")

        return {"verified": False, "confidence_delta": 0.0, "source_quality": "low"}

    def _integrate_candidate(self, candidate: PhantomCandidate):
        """Интеграция кандидата в FGv2"""
        try:
            # Получаем FGv2
            fg = None
            if self.brain:
                fg = getattr(self.brain, 'fractal_graph_v2', None)
                if fg is None:
                    kg = getattr(self.brain, 'knowledge_graph', None)
                    if kg and hasattr(kg, '_fg'):
                        fg = kg._fg

            if fg and hasattr(fg, 'add_node'):
                node = fg.add_node(
                    content=candidate.title,
                    node_type='concept',
                    metadata={
                        'definition': candidate.definition,
                        'rationale': candidate.rationale,
                        'confidence': candidate.confidence,
                        'source': 'concept_miner',
                        'semantic_gap': candidate.semantic_gap,
                        'variance': candidate.variance,
                        'cluster_id': candidate.cluster_id
                    }
                )

                if node:
                    candidate.parent_group_id = node.id if hasattr(node, 'id') else str(node)
                    candidate.status = "confirmed"
                    candidate.confirmations += 1
                    self._metrics["candidates_confirmed"] += 1

                    logger.info(f"Концепт '{candidate.title}' интегрирован в FGv2")

                    # Добавляем в очередь самодиалога и триггерим
                    if hasattr(self.brain, 'self_dialog_learning') and self.brain.self_dialog_learning:
                        self.brain.self_dialog_learning.queue_concept_for_dialog(
                            candidate.title,
                            priority=candidate.confidence
                        )
                        # Триггер для запуска self-learning по требованию
                        try:
                            if hasattr(self.brain.self_dialog_learning, 'trigger_self_dialog'):
                                self.brain.self_dialog_learning.trigger_self_dialog(reason='concept_mined')
                        except Exception as e:
                            logger.debug(f"Trigger error: {e}")

        except Exception as e:
            logger.error(f"Ошибка интеграции концепта: {e}")

    def _update_lifecycle(self):
        """
        Обновление жизненного цикла концептов:
        - provisional → confirmed: веб-поиск подтвердил или confidence >= 0.75
        - confirmed → stable: отсутствие противоречий в течение 5 циклов
        - archived: confidence < 0.25 в течение 3 циклов или явный отказ
        """
        cycles_stable = self.config.get("cycles_before_stable", 5)
        threshold_confirm = self.config.get("confidence_threshold_confirm", 0.75)
        threshold_archive = self.config.get("confidence_threshold_archive", 0.25)

        for candidate in self._candidates.values():
            if candidate.status == "provisional":
                if candidate.web_verification and candidate.web_verification.get("verified"):
                    candidate.confidence = min(1.0, candidate.confidence + 0.25)
                if candidate.confidence >= threshold_confirm:
                    candidate.status = "confirmed"
                    self._integrate_candidate(candidate)

            elif candidate.status == "confirmed":
                if candidate.confirmations >= cycles_stable:
                    candidate.status = "stable"

            elif candidate.status != "archived":
                if candidate.confidence < threshold_archive:
                    candidate.status = "archived"
                    self._metrics["candidates_archived"] += 1

    def get_metrics(self) -> Dict:
        """Получение метрик модуля"""
        total = self._metrics["candidates_generated"]
        if total > 0:
            self._metrics["hypothesis_confirmation_ratio"] = self._metrics["candidates_confirmed"] / total
            self._metrics["validation_rejection_rate"] = (
                self._metrics.get("candidates_rejected", 0) / total
            )

        return {
            **self._metrics,
            "active_candidates": len(self._candidates),
            "audit_log_size": len(self._audit_log),
            "status": "running" if self._running else "stopped"
        }

    def get_candidates(self, status: str = None) -> List[Dict]:
        """Получение списка кандидатов"""
        if status:
            return [asdict(c) for c in self._candidates.values() if c.status == status]
        return [asdict(c) for c in self._candidates.values()]

    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        """Получение аудит-лога отклонённых кандидатов"""
        return self._audit_log[-limit:]

    def force_mining_cycle(self):
        """Принудительный запуск цикла майнинга"""
        if self._can_mine():
            self._executor.submit(self._mining_cycle)
        else:
            logger.warning("Невозможно запустить майнинг: недостаточно ресурсов")


def create_concept_miner(
    brain=None,
    event_bus=None,
    deferred_system=None,
    background_coordinator=None,
    config: Dict = None
) -> ConceptMiner:
    """Фабрика создания ConceptMiner"""
    return ConceptMiner(
        brain=brain,
        event_bus=event_bus,
        deferred_system=deferred_system,
        background_coordinator=background_coordinator,
        config=config
    )


__all__ = [
    'ConceptMiner',
    'ConceptStatus',
    'PhantomCandidate',
    'create_concept_miner'
]
