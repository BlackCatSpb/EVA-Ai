"""
ConceptMiner - модуль автономного концептуального вывода для EVA-Ai
Реализует проактивное обнаружение семантических лакун во фрактальном графе памяти

По спецификации: Техническая спецификация модуля ACI (ConceptMiner) v3.1
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
    validation_ethics: Optional[Dict] = None
    web_verification: Optional[Dict] = None


class ConceptMiner:
    """
    Модуль автономного концептуального вывода (ACI / ConceptMiner)
    
    Функции:
    - Подписка на события шины EventBus
    - Детекция семантических лакун через анализ кластеров
    - Генерация гипотез через GGUF Pipeline
    - Валидация (NLI, Ethics, Web)
    - Жизненный цикл концептов
    """
    
    DEFAULT_CONFIG = {
        "enabled": True,
        "dry_run": True,
        "base_threshold": 0.30,
        "dedup_radius": 0.15,
        "max_candidates_per_cycle": 5,
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
        "confidence_threshold_archive": 0.25
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
            "candidates_archived": 0
        }
        
        logger.info("ConceptMiner инициализирован")
    
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
        if hasattr(self.brain, '_project_root'):
            project_root = self.brain._project_root
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(project_root, "eva_ai", "knowledge", "concept_miner_data")
    
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
        
        logger.info("ConceptMiner остановлен")
    
    def _subscribe_to_events(self):
        """Подписка на события EventBus"""
        if not self.event_bus:
            return
        
        from eva_ai.core.event_bus import EventTypes
        
        events_to_subscribe = [
            EventTypes.MEMORY_GRAPH_UPDATED,
            EventTypes.MEMORY_CLUSTERING_COMPLETE,
            EventTypes.PIPELINE_COMPLETE,
            EventTypes.SYSTEM_READY,
            EventTypes.SYSTEM_IDLE,
        ]
        
        for event_type in events_to_subscribe:
            try:
                handler_name = f"_on_{event_type.value.replace('.', '_')}"
                if hasattr(self, handler_name):
                    handler = getattr(self, handler_name)
                    sub_id = self.event_bus.subscribe(event_type, handler, priority=8)
                    self._subscription_ids.append(sub_id)
                    logger.debug(f"Подписка на {event_type}: {sub_id}")
            except Exception as e:
                logger.warning(f"Не удалось подписаться на {event_type}: {e}")
        
        logger.info(f"ConceptMiner подписан на {len(self._subscription_ids)} событий")
    
    def _on_memory_graph_updated(self, event):
        """Обработка обновления графа памяти - запуск майнинга"""
        self._schedule_mining_if_needed()
    
    def _on_memory_clustering_complete(self, event):
        """Обработка завершения кластеризации"""
        data = event.data if hasattr(event, 'data') else {}
        cluster_data = data.get('clusters', {})
        
        if cluster_data:
            self._executor.submit(self._mine_concepts_from_clusters, cluster_data)
    
    def _on_pipeline_complete(self, event):
        """После завершения пайплайна - проверка idle"""
        self._schedule_mining_if_needed()
    
    def _on_system_ready(self, event):
        """При готовности системы"""
        self._schedule_mining_if_needed()
    
    def _on_system_idle(self, event):
        """При простое системы"""
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
                if hasattr(self.brain, 'resource_manager'):
                    rm = self.brain.resource_manager
                    if hasattr(rm, 'get_cpu_usage'):
                        cpu = float(rm.get_cpu_usage())
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
            
            for candidate in candidates[:self.config.get("max_candidates_per_cycle", 5)]:
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
    
    def _get_clusters(self) -> Dict:
        """Получение кластеров из фрактальной памяти"""
        clusters = {}
        
        try:
            if hasattr(self.brain, 'fractal_memory'):
                fm = self.brain.fractal_memory
                
                if hasattr(fm, 'cluster_nodes'):
                    clusters = fm.cluster_nodes(level=1, threshold=0.5)
                elif hasattr(fm, 'storage') and hasattr(fm.storage, 'cluster_nodes'):
                    clusters = fm.storage.cluster_nodes(level=1, threshold=0.5)
            
            if hasattr(self.brain, 'memory_manager'):
                mm = self.brain.memory_manager
                
                if hasattr(mm, 'fractal_graph') and hasattr(mm.fractal_graph, 'cluster_nodes'):
                    clusters = mm.fractal_graph.cluster_nodes(level=1, threshold=0.5)
                elif hasattr(mm, 'graph') and hasattr(mm.graph, 'cluster_nodes'):
                    clusters = mm.graph.cluster_nodes(level=1, threshold=0.5)
                    
        except Exception as e:
            logger.warning(f"Не удалось получить кластеры: {e}")
        
        return clusters
    
    def _detect_semantic_gaps(self, clusters: Dict) -> List[PhantomCandidate]:
        """
        Детекция семантических лакун (алгоритм по спецификации)
        
        Расчёт центроида: μC = (1/|C|)Σv
        Семантический разрыв: ΔC = min(1 - cos(μC, v))
        Адаптивный порог: τ = τ_base * (1 + σ)
        """
        candidates = []
        base_threshold = self.config.get("base_threshold", 0.30)
        dedup_radius = self.config.get("dedup_radius", 0.15)
        
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
                centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                
                gaps = []
                for emb in embeddings:
                    cos_sim = np.dot(centroid, emb) / (np.linalg.norm(emb) + 1e-8)
                    gap = 1.0 - cos_sim
                    gaps.append(gap)
                
                variance = float(np.std(gaps))
                semantic_gap = float(min(gaps))
                
                threshold = base_threshold * (1 + variance)
                
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
        """Получение данных узлов кластера"""
        nodes_data = []
        
        try:
            if hasattr(self.brain, 'fractal_memory'):
                fm = self.brain.fractal_memory
                if hasattr(fm, 'storage') and hasattr(fm.storage, 'nodes'):
                    for nid in node_ids:
                        node = fm.storage.nodes.get(nid)
                        if node:
                            nodes_data.append({
                                'id': getattr(node, 'id', nid),
                                'content': getattr(node, 'content', ''),
                                'embedding': getattr(node, 'embedding', None)
                            })
        except Exception as e:
            logger.warning(f"Не удалось получить данные узлов: {e}")
        
        return nodes_data
    
    def _is_duplicate(self, centroid: np.ndarray, radius: float) -> bool:
        """Проверка на дубликат среди существующих концептов"""
        for candidate in self._candidates.values():
            if candidate.status in ["confirmed", "stable"] and candidate.centroid:
                existing = np.array(candidate.centroid)
                cos_sim = np.dot(centroid, existing) / (np.linalg.norm(existing) + 1e-8)
                if (1.0 - cos_sim) < radius:
                    return True
        return False
    
    def _generate_hypothesis(self, candidate: PhantomCandidate):
        """
        Генерация гипотезы концепта через GGUF Pipeline
        
        Использует существующий RecursiveModelPipeline для унификации
        """
        from eva_ai.core.event_bus import Event, EventTypes
        
        self.event_bus.publish(Event(
            event_type=EventTypes.CONCEPT_MINING_START,
            source="concept_miner",
            data={"candidate_id": candidate.id}
        ))
        
        try:
            cluster_nodes_formatted = "\n".join([
                f"- {n.get('content', '')[:100]}" for n in candidate.nodes[:10]
            ])
            
            user_prompt = f"""Анализируй следующий семантический кластер узлов фрактальной памяти:
{cluster_nodes_formatted}

Типы связей в кластере: {', '.join(set(n.get('type', 'fact') for n in candidate.nodes))}
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
        
        self.event_bus.publish(Event(
            event_type=EventTypes.CONCEPT_CANDIDATE_GENERATED,
            source="concept_miner",
            data={"candidate_id": candidate.id, "title": candidate.title}
        ))
    
    def _call_pipeline(self, user_prompt: str, system_prompt: str = None, temp_override: float = None) -> str:
        """Вызов существующего GGUF Pipeline для генерации (каскад Model A + Model B)"""
        
        temp_a = temp_override if temp_override else self.config.get('llm_temperature', 0.35)
        
        params = {
            'model_a': {
                'temperature': temp_a,
                'repeat_penalty': self.config.get('llm_repeat_penalty', 1.8),
                'max_tokens': self.config.get('max_llm_tokens', 128)
            },
            'model_b': {
                'temperature': 0.25,
                'repeat_penalty': 2.0,
                'max_tokens': 256
            }
        }
        
        try:
            if hasattr(self.brain, 'pipeline') and self.brain.pipeline:
                result = self.brain.pipeline.process_query(user_prompt, gen_params=params)
                
                model_a_response = result.get('model_a_result', {}).get('natural_response', '')
                model_b_response = result.get('model_b_result', {}).get('natural_response', '')
                
                if model_b_response:
                    return model_b_response
                return model_a_response
            
            elif hasattr(self.brain, 'process_query'):
                result = self.brain.process_query(user_prompt)
                return result.get('response', '')
                
        except Exception as e:
            logger.warning(f"Ошибка вызова Pipeline: {e}")
        
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
        Валидация кандидата:
        1. NLI-когерентность
        2. Этическая фильтрация
        3. Веб-верификация (опционально)
        """
        from eva_ai.core.event_bus import Event, EventTypes
        
        context_summary = " ".join([n.get('content', '')[:50] for n in candidate.nodes[:5]])
        
        candidate.validation_nli = self._check_nli_coherence(context_summary, candidate)
        
        candidate.validation_ethics = self._check_ethics(candidate)
        
        if candidate.validation_nli.get("status") == "contradiction":
            candidate.rejections += 1
            candidate.status = "archived"
            logger.info(f"Кандидат {candidate.id} отклонён: противоречие")
            return
        
        if candidate.validation_ethics.get("risk_level") == "high":
            candidate.rejections += 1
            candidate.status = "archived"
            logger.info(f"Кандидат {candidate.id} отклонён: этика")
            return
        
        if self.config.get("enable_web_search_validation", True):
            candidate.web_verification = self._verify_web(candidate)
        
        candidate.status = "provisional"
        candidate.updated_at = time.time()
        
        self._candidates[candidate.id] = candidate
        
        self.event_bus.publish(Event(
            event_type=EventTypes.CONCEPT_VALIDATION_COMPLETE,
            source="concept_miner",
            data={"candidate_id": candidate.id, "status": candidate.status}
        ))
    
    def _check_nli_coherence(self, context: str, candidate: PhantomCandidate) -> Dict:
        """Проверка NLI-когерентности (логическая согласованность)"""
        nli_prompt = f"""Оцени логическую согласованность гипотетического концепта с существующим фрагментом графа знаний.

Контекст кластера: {context}
Гипотеза: {candidate.title} — {candidate.definition}

Вопрос: Противоречит ли гипотеза установленным связям в кластере или нарушает онтологические правила?

Ответь строго в формате JSON:
{{"status": "entailment" | "neutral" | "contradiction", "confidence": 0.0-1.0, "reason": "краткое объяснение"}}"""
        
        response = self._call_pipeline(nli_prompt)
        
        try:
            import json
            for line in response.split('\n'):
                if line.startswith('{') and line.endswith('}'):
                    return json.loads(line)
        except Exception:
            pass
        
        return {"status": "neutral", "confidence": 0.5, "reason": "Ошибка проверки"}
    
    def _check_ethics(self, candidate: PhantomCandidate) -> Dict:
        """Этическая фильтрация"""
        ethics_prompt = f"""Проверь гипотетический концепт "{candidate.title}" и его определение на соответствие этическим принципам EVA-Ai:
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
            import json
            for line in response.split('\n'):
                if line.startswith('{') and line.endswith('}'):
                    return json.loads(line)
        except Exception:
            pass
        
        return {"risk_level": "low", "block": False}
    
    def _verify_web(self, candidate: PhantomCandidate) -> Dict:
        """Веб-верификация концепта"""
        web_prompt = f"""В системе обнаружена семантическая лакуна, соответствующая концепту "{candidate.title}". 
Текущее определение: "{candidate_definition}". 

Сгенерируй оптимизированный поисковый запрос на русском языке для проверки существования и корректности данного термина.

Верни строго JSON: {{"query": "...", "domains": ["..."], "language": "ru"}}"""
        
        response = self._call_pipeline(web_prompt)
        
        try:
            import json
            for line in response.split('\n'):
                if line.startswith('{') and line.endswith('}'):
                    search_data = json.loads(line)
                    
                    if hasattr(self.brain, 'web_search') and self.brain.web_search:
                        results = self.brain.web_search.search(
                            query=search_data.get('query', ''),
                            language='ru'
                        )
                        
                        verify_prompt = f"""Проанализируй результаты веб-поиска по запросу "{search_data.get('query')}".
Контекст: Гипотетический концепт "{candidate.title}" с определением "{candidate.definition}".

Найденные данные: {results[:3]}

Задача:
1. Подтверждает ли источник существование и корректность термина?
2. Требуется ли корректировка определения?
3. Оцени уверенность системы (0.0–1.0) на основе авторитетности источников.

Ответ строго в формате JSON:
{{"verified": true/false, "updated_definition": "...", "confidence_delta": -0.2 to +0.4, "source_quality": "low|medium|high"}}"""
                        
                        verify_response = self._call_pipeline(verify_prompt)
                        
                        for line in verify_response.split('\n'):
                            if line.startswith('{') and line.endswith('}'):
                                return json.loads(line)
        except Exception as e:
            logger.warning(f"Веб-верификация не удалась: {e}")
        
        return {"verified": False, "confidence_delta": 0.0, "source_quality": "low"}
    
    def _integrate_candidate(self, candidate: PhantomCandidate):
        """Интеграция кандидата в граф"""
        try:
            if hasattr(self.brain, 'knowledge_graph'):
                kg = self.brain.knowledge_graph
                
                if hasattr(kg, 'add_node'):
                    node_id = kg.add_node(
                        name=candidate.title,
                        description=candidate.definition,
                        node_type="concept",
                        metadata={
                            "source": "concept_miner",
                            "confidence": candidate.confidence,
                            "parent_cluster": candidate.cluster_id,
                            "rationale": candidate.rationale
                        }
                    )
                    
                    candidate.parent_group_id = node_id
                    candidate.status = "confirmed"
                    candidate.confirmations += 1
                    self._metrics["candidates_confirmed"] += 1
                    
                    logger.info(f"Концепт '{candidate.title}' интегрирован в граф")
                    
        except Exception as e:
            logger.error(f"Ошибка интеграции концепта: {e}")
    
    def _update_lifecycle(self):
        """Обновление жизненного цикла концептов"""
        cycles_stable = self.config.get("cycles_before_stable", 5)
        threshold_confirm = self.config.get("confidence_threshold_confirm", 0.75)
        threshold_archive = self.config.get("confidence_threshold_archive", 0.25)
        
        for candidate in self._candidates.values():
            if candidate.status == "provisional":
                if candidate.web_verification and candidate.web_verification.get("verified"):
                    candidate.confidence = min(1.0, candidate.confidence + 0.3)
                    if candidate.confidence >= threshold_confirm:
                        candidate.status = "confirmed"
                        self._integrate_candidate(candidate)
                        
                elif candidate.confirmations >= 2:
                    candidate.status = "confirmed"
                    
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
            "status": "running" if self._running else "stopped"
        }
    
    def get_candidates(self, status: str = None) -> List[Dict]:
        """Получение списка кандидатов"""
        if status:
            return [asdict(c) for c in self._candidates.values() if c.status == status]
        return [asdict(c) for c in self._candidates.values()]
    
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
