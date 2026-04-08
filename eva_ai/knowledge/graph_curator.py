"""Модернизированный куратор графа знаний для ЕВА.

Возможности:
- Фоновая работа с адаптивным интервалом
- Подключение к event bus и deferred commands
- Метрики качества графа и производительности
- "Растворение QWEN" - извлечение знаний из GGUF моделей
- Поддержание порядка: чистка, ре-кластеризация
"""
import logging
import time
import threading
import os
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger("eva_ai.graph_curator")

class CuratorState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

class CuratorEventType(Enum):
    CURATION_START = "curator.started"
    CURATION_COMPLETE = "curator.completed"
    CURATION_ERROR = "curator.error"
    GRAPH_OPTIMIZED = "curator.graph_optimized"
    KNOWLEDGE_EXTRACTED = "curator.knowledge_extracted"
    CLEANUP_COMPLETE = "curator.cleanup_done"
    METRICS_UPDATED = "curator.metrics_updated"

@dataclass
class CuratorMetrics:
    """Метрики куратора."""
    # Работа
    cycles_completed: int = 0
    nodes_curated: int = 0
    links_created: int = 0
    links_removed: int = 0
    groups_created: int = 0
    groups_merged: int = 0
    
    # Качество графа
    total_nodes: int = 0
    total_edges: int = 0
    total_groups: int = 0
    orphan_nodes: int = 0
    duplicate_clusters: int = 0
    avg_cluster_quality: float = 0.0
    
    # Производительность
    avg_cycle_time: float = 0.0
    last_cycle_time: float = 0.0
    total_processing_time: float = 0.0
    
    # QWEN extraction
    knowledge_extracted: int = 0
    last_extraction_time: float = 0.0
    
    # Состояние
    state: str = "idle"
    last_run: float = 0.0
    next_run: float = 0.0

@dataclass
class CuratorConfig:
    """Конфигурация куратора."""
    enabled: bool = True
    min_interval: int = 60       # Минимальный интервал между циклами (сек)
    max_interval: int = 600      # Максимальный интервал
    adaptive_interval: bool = True
    
    # Метрики и оптимизация
    check_graph_health: bool = True
    cleanup_orphans: bool = True
    cleanup_duplicates: bool = True
    recluster_threshold: float = 0.4
    
    # QWEN extraction
    extract_from_gguf: bool = True
    gguf_models_dir: str = None
    extraction_interval: int = 3600  # Раз в час
    
    # Ограничения
    max_nodes_per_cycle: int = 100
    max_links_per_cycle: int = 200
    max_groups_created: int = 10
    
    # Event bus
    publish_events: bool = True
    subscribe_to_events: bool = True
    
    # Deferred commands
    use_deferred: bool = True

class GraphCurator:
    """
    Модернизированный куратор графа знаний.
    
    Обеспечивает:
    - Адаптивную фоновую работу
    - Интеграцию с event bus и deferred commands
    - Метрики и мониторинг
    - Извлечение знаний из GGUF моделей
    - Поддержание порядка в графе
    """
    
    TEMPLATE_PATTERNS = [
        'продолжим разговор о перспективах',
        'давайте продолжим разговор',
        'перспективы развития искусственного интеллекта',
        'развитие машинного обучения и нейронных сетей',
        '### перспективы разработка',
        'настоящее развитие технологий в области',
    ]
    
    LOW_QUALITY_INDICATORS = [
        'q:', 'a:', 'вопрос:', 'ответ:',
        'пример:', 'обратите внимание',
        'особенности данного предложения',
    ]
    
    SEMANTIC_ASSOCIATIONS = {
        'снег': ['белый', 'холодный', 'зимний', 'искрящийся', 'пушистый', 'первый', 'сугроб'],
        'дождь': ['мокрый', 'холодный', 'ливень', 'капли', 'зонт', 'облака'],
        'солнце': ['яркое', 'тёплое', 'жёлтое', 'лето', 'рассвет', 'закат'],
        'море': ['синее', 'голубое', 'солёное', 'волны', 'пляж'],
        'лес': ['зелёный', 'густой', 'тихий', 'деревья', 'грибы'],
        'человек': ['мужчина', 'женщина', 'ребёнок', 'взрослый', 'друг'],
        'друг': ['близкий', 'верный', 'надёжный', 'поддержка'],
        'мама': ['забота', 'любовь', 'тёплая', 'семья'],
        'счастье': ['радость', 'удовольствие', 'улыбка', 'смех'],
        'грусть': ['печаль', 'тоска', 'одиночество', 'слёзы'],
        'страх': ['ужас', 'тревога', 'опасность', 'беспокойство'],
        'утро': ['рассвет', 'свежесть', 'кофе', 'начало'],
        'вечер': ['закат', 'отдых', 'тишина', 'темнота'],
        'ночь': ['темнота', 'звёзды', 'луна', 'сон'],
        'еда': ['вкусная', 'горячая', 'полезная', 'сытная'],
        'кофе': ['горячий', 'ароматный', 'бодрящий', 'крепкий'],
        'кот': ['мягкий', 'пушистый', 'ласковый', 'игривый'],
        'собака': ['верный', 'преданный', 'дружелюбный', 'умный'],
        'музыка': ['мелодичная', 'ритмичная', 'классическая'],
        'картина': ['красивая', 'яркая', 'выразительная'],
        'книга': ['интересная', 'познавательная', 'художественная'],
    }
    
    def __init__(self, brain=None, config: Optional[Dict] = None):
        self.brain = brain
        self.config = CuratorConfig(**(config or {}))
        
        # Состояние
        self.state = CuratorState.IDLE
        self.running = False
        self.paused = False
        self.stop_event = threading.Event()
        
        # Потоки
        self.curator_thread = None
        self.monitor_thread = None
        
        # Метрики
        self.metrics = CuratorMetrics()
        self.metrics_history: List[CuratorMetrics] = []
        self._metrics_lock = threading.RLock()
        
        # Event bus и deferred commands
        self._event_bus = None
        self._deferred_system = None
        
        # Адаптивный интервал
        self._current_interval = self.config.min_interval
        self._last_activity = time.time()
        
        # GGUF модели для извлечения
        self._gguf_models: List[str] = []
        self._last_extraction = 0
        
        logger.info("GraphCurator инициализирован (модернизированный)")
    
    # === УПРАВЛЕНИЕ ===
    
    def start(self) -> bool:
        """Запустить куратора."""
        if not self.config.enabled:
            logger.info("GraphCurator отключён в конфигурации")
            return False
        
        if self.running:
            logger.warning("GraphCurator уже запущен")
            return False
        
        # Подключаемся к event bus
        self._connect_to_event_bus()
        
        # Подключаемся к deferred commands
        self._connect_to_deferred_system()
        
        # Сканируем GGUF модели
        if self.config.extract_from_gguf:
            self._scan_gguf_models()
        
        self.running = True
        self.stop_event.clear()
        self.state = CuratorState.RUNNING
        
        self.curator_thread = threading.Thread(target=self._curation_loop, daemon=True, name="GraphCurator")
        self.curator_thread.start()
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True, name="CuratorMonitor")
        self.monitor_thread.start()
        
        self._publish_event(CuratorEventType.CURATION_START, {"state": "started"})
        logger.info("GraphCurator запущен")
        
        return True
    
    def stop(self):
        """Остановить куратора."""
        if not self.running:
            return
        
        self.stop_event.set()
        self.running = False
        self.state = CuratorState.IDLE
        
        if self.curator_thread and self.curator_thread.is_alive():
            self.curator_thread.join(timeout=5)
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=3)
        
        logger.info("GraphCurator остановлен")
    
    def pause(self):
        """Приостановить куратора."""
        self.paused = True
        self.state = CuratorState.PAUSED
        logger.info("GraphCurator приостановлен")
    
    def resume(self):
        """Возобновить куратора."""
        self.paused = False
        self.state = CuratorState.RUNNING
        self._last_activity = time.time()
        logger.info("GraphCurator возобновлён")
    
    def is_running(self) -> bool:
        return self.running and not self.paused
    
    def get_state(self) -> Dict[str, Any]:
        """Получить состояние куратора."""
        return {
            "state": self.state.value,
            "running": self.running,
            "paused": self.paused,
            "metrics": self.get_metrics(),
            "config": {
                "enabled": self.config.enabled,
                "adaptive_interval": self.config.adaptive_interval,
                "current_interval": self._current_interval
            }
        }
    
    # === МЕТРИКИ ===
    
    def get_metrics(self) -> Dict[str, Any]:
        """Получить метрики куратора."""
        with self._metrics_lock:
            return {
                "cycles_completed": self.metrics.cycles_completed,
                "nodes_curated": self.metrics.nodes_curated,
                "links_created": self.metrics.links_created,
                "links_removed": self.metrics.links_removed,
                "groups_created": self.metrics.groups_created,
                "groups_merged": self.metrics.groups_merged,
                "total_nodes": self.metrics.total_nodes,
                "total_edges": self.metrics.total_edges,
                "total_groups": self.metrics.total_groups,
                "orphan_nodes": self.metrics.orphan_nodes,
                "duplicate_clusters": self.metrics.duplicate_clusters,
                "avg_cluster_quality": round(self.metrics.avg_cluster_quality, 3),
                "avg_cycle_time": round(self.metrics.avg_cycle_time, 3),
                "last_cycle_time": round(self.metrics.last_cycle_time, 3),
                "knowledge_extracted": self.metrics.knowledge_extracted,
                "state": self.metrics.state,
                "next_run": max(0, self.metrics.next_run - time.time())
            }
    
    def update_graph_metrics(self, graph):
        """Обновить метрики графа."""
        with self._metrics_lock:
            try:
                if hasattr(graph, 'get_stats'):
                    stats = graph.get_stats()
                    self.metrics.total_nodes = stats.get('total_nodes', 0)
                    self.metrics.total_edges = stats.get('total_edges', 0)
                    self.metrics.total_groups = stats.get('total_groups', 0)
            except Exception as e:
                logger.debug(f"Ошибка обновления метрик графа: {e}")
    
    def _adjust_interval(self, activity: float):
        """Адаптивная корректировка интервала."""
        if not self.config.adaptive_interval:
            return
        
        if activity > 0.8:
            self._current_interval = max(self.config.min_interval, self._current_interval * 0.7)
        elif activity < 0.2:
            self._current_interval = min(self.config.max_interval, self._current_interval * 1.3)
        
        self.metrics.next_run = time.time() + self._current_interval
    
    # === EVENT BUS ===
    
    def _connect_to_event_bus(self):
        """Подключиться к event bus."""
        if not self.config.subscribe_to_events:
            return
        
        try:
            if self.brain and hasattr(self.brain, 'event_bus'):
                self._event_bus = self.brain.event_bus
                if self._event_bus and self.config.publish_events:
                    self._subscribe_to_system_events()
                    logger.info("GraphCurator подключён к event bus")
        except Exception as e:
            logger.warning(f"Не удалось подключиться к event bus: {e}")
    
    def _subscribe_to_system_events(self):
        """Подписаться на системные события."""
        if not self._event_bus:
            return
        
        try:
            self._event_bus.subscribe("query_received", self._on_query_received, priority=3)
            self._event_bus.subscribe("knowledge_added", self._on_knowledge_added, priority=2)
            self._event_bus.subscribe("system_health_check", self._on_health_check, priority=8)
        except Exception as e:
            logger.debug(f"Ошибка подписки на события: {e}")
    
    def _on_query_received(self, data: Any):
        """Обработчик события нового запроса."""
        self._last_activity = time.time()
        if self.paused:
            self.resume()
    
    def _on_knowledge_added(self, data: Any):
        """Обработчик добавления знаний."""
        if self.config.adaptive_interval:
            self._current_interval = max(self.config.min_interval, self._current_interval * 0.8)
    
    def _on_health_check(self, data: Any):
        """Обработчик проверки здоровья."""
        self._publish_event(CuratorEventType.METRICS_UPDATED, self.get_metrics())
    
    def _publish_event(self, event_type: CuratorEventType, data: Dict):
        """Опубликовать событие куратора."""
        if not self.config.publish_events or not self._event_bus:
            return
        
        try:
            self._event_bus.trigger(event_type.value, data)
            
            # Публикуем в EventBus для внешних подписчиков (ConceptMiner)
            from eva_ai.core.event_bus import Event, EventTypes
            if event_type == CuratorEventType.GRAPH_OPTIMIZED:
                self._event_bus.publish(Event(
                    event_type=EventTypes.MEMORY_CLUSTERING_COMPLETE,
                    source="graph_curator",
                    data=data
                ))
        except Exception as e:
            logger.debug(f"Ошибка публикации события: {e}")
    
    # === DEFERRED COMMANDS ===
    
    def _connect_to_deferred_system(self):
        """Подключиться к системе отложенных команд."""
        if not self.config.use_deferred:
            return
        
        try:
            if self.brain and hasattr(self.brain, 'deferred_system'):
                self._deferred_system = self.brain.deferred_system
                if self._deferred_system:
                    logger.info("GraphCurator подключён к deferred commands")
        except Exception as e:
            logger.warning(f"Не удалось подключиться к deferred system: {e}")
    
    def schedule_deferred(self, command: callable, priority: str = "NORMAL", delay: float = 0):
        """Запланировать отложенную команду."""
        if not self._deferred_system:
            return
        
        try:
            from eva_ai.core.deferred_command_system import CommandPriority
            priority_map = {
                "CRITICAL": CommandPriority.CRITICAL,
                "HIGH": CommandPriority.HIGH,
                "NORMAL": CommandPriority.NORMAL,
                "LOW": CommandPriority.LOW
            }
            cmd_priority = priority_map.get(priority.upper(), CommandPriority.NORMAL)
            
            self._deferred_system.add_command(
                command=command,
                priority=cmd_priority,
                max_retries=2,
                retry_delay=5.0,
                command_id=f"curator_{int(time.time()*1000)}"
            )
        except Exception as e:
            logger.debug(f"Ошибка планирования команды: {e}")
    
    # === GGUF EXTRACTION ===
    
    def _scan_gguf_models(self):
        """Сканировать директорию с GGUF моделями."""
        models_dir = self.config.gguf_models_dir
        if not models_dir:
            # Default: ищем в eva/mlearning
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            models_dir = os.path.join(base_dir, "eva", "mlearning", "eva_models")
        
        if not os.path.exists(models_dir):
            logger.debug(f"Директория GGUF моделей не найдена: {models_dir}")
            return
        
        for root, dirs, files in os.walk(models_dir):
            for f in files:
                if f.endswith('.gguf'):
                    full_path = os.path.join(root, f)
                    if full_path not in self._gguf_models:
                        self._gguf_models.append(full_path)
        
        if self._gguf_models:
            logger.info(f"Найдено {len(self._gguf_models)} GGUF моделей для извлечения")
    
    def _extract_knowledge_from_gguf(self):
        """Извлечь знания из GGUF моделей."""
        if not self._gguf_models or not self.brain:
            return
        
        # Проверяем интервал
        if time.time() - self._last_extraction < self.config.extraction_interval:
            return
        
        knowledge_graph = self._get_knowledge_graph()
        if not knowledge_graph or not hasattr(knowledge_graph, 'load_gguf_knowledge'):
            return
        
        extracted_count = 0
        for model_path in self._gguf_models:
            try:
                result = knowledge_graph.load_gguf_knowledge(model_path)
                if result and result.get('added_nodes'):
                    extracted_count += result['added_nodes']
            except Exception as e:
                logger.debug(f"Ошибка извлечения из {model_path}: {e}")
        
        if extracted_count > 0:
            self.metrics.knowledge_extracted += extracted_count
            self.metrics.last_extraction_time = time.time()
            self._publish_event(CuratorEventType.KNOWLEDGE_EXTRACTED, {
                "count": extracted_count,
                "models": len(self._gguf_models)
            })
            logger.info(f"Извлечено {extracted_count} знаний из GGUF моделей")
        
        self._last_extraction = time.time()
    
    # === ОСНОВНОЙ ЦИКЛ ===
    
    def _curation_loop(self):
        """Основной цикл куратора."""
        logger.info("Запущен цикл куратора графа")
        
        while not self.stop_event.is_set():
            try:
                if self.paused:
                    self.stop_event.wait(timeout=5)
                    continue
                
                cycle_start = time.time()
                
                # Выполняем курацию
                self._run_curation_cycle()
                
                # Извлекаем знания из GGUF
                if self.config.extract_from_gguf:
                    self._extract_knowledge_from_gguf()
                
                cycle_time = time.time() - cycle_start
                
                with self._metrics_lock:
                    self.metrics.last_cycle_time = cycle_time
                    self.metrics.cycles_completed += 1
                    self.metrics.total_processing_time += cycle_time
                    self.metrics.avg_cycle_time = self.metrics.total_processing_time / self.metrics.cycles_completed
                    self.metrics.last_run = time.time()
                    self.metrics.state = "running"
                
                # Адаптивный интервал
                activity = min(1.0, (cycle_time / self._current_interval))
                self._adjust_interval(activity)
                
                self._publish_event(CuratorEventType.CURATION_COMPLETE, {
                    "cycle_time": cycle_time,
                    "total_cycles": self.metrics.cycles_completed
                })
                
            except Exception as e:
                logger.error(f"Ошибка куратора: {e}", exc_info=True)
                self.state = CuratorState.ERROR
                self.metrics.state = "error"
                self._publish_event(CuratorEventType.CURATION_ERROR, {"error": str(e)})
            
            # Ждём следующий цикл
            self.stop_event.wait(timeout=self._current_interval)
    
    def _run_curation_cycle(self):
        """Выполнить один цикл курации."""
        knowledge_graph = self._get_knowledge_graph()
        if not knowledge_graph:
            return
        
        # Обновляем метрики графа
        self.update_graph_metrics(knowledge_graph)
        
        # 1. Проверка здоровья графа
        if self.config.check_graph_health:
            self._check_graph_health(knowledge_graph)
        
        # 2. Семантические связи
        self._create_semantic_links(knowledge_graph)
        
        # 3. Чистка сирот
        if self.config.cleanup_orphans:
            self._cleanup_orphans(knowledge_graph)
        
        # 4. Чистка дубликатов
        if self.config.cleanup_duplicates:
            self._cleanup_duplicates(knowledge_graph)
        
        # 5. Чистка мусора (шаблоны, низкое качество)
        self._cleanup_garbage(knowledge_graph)
        
        # 5. Ре-кластеризация
        if self.config.recluster_threshold > 0:
            self._recluster_if_needed(knowledge_graph)
    
    def _get_knowledge_graph(self):
        """Получить граф знаний."""
        if not self.brain:
            return None
        
        # Пробуем разные источники
        if hasattr(self.brain, 'fractal_graph_v2'):
            return self.brain.fractal_graph_v2
        if hasattr(self.brain, 'knowledge_graph'):
            return self.brain.knowledge_graph
        if hasattr(self.brain, 'fractal_memory'):
            return self.brain.fractal_memory
        
        return None
    
    # === ОПЕРАЦИИ КУРАЦИИ ===
    
    def _check_graph_health(self, kg):
        """Проверить здоровье графа."""
        try:
            # Получаем все узлы
            nodes = self._get_all_nodes(kg)
            
            orphans = 0
            for node in nodes:
                # Проверяем связи
                edges = self._get_node_edges(kg, node)
                if not edges:
                    orphans += 1
            
            with self._metrics_lock:
                self.metrics.orphan_nodes = orphans
                self.metrics.avg_cluster_quality = 1.0 - (orphans / max(1, len(nodes)))
            
        except Exception as e:
            logger.debug(f"Ошибка проверки здоровья: {e}")
    
    def _create_semantic_links(self, kg):
        """Создать семантические связи."""
        try:
            nodes = self._get_all_nodes(kg)
            if not nodes:
                return
            
            new_links = 0
            
            for node in nodes[:self.config.max_nodes_per_cycle]:
                node_name = getattr(node, 'name', '') or getattr(node, 'content', '') or ''
                if not node_name or len(node_name) < 2:
                    continue
                
                node_lower = node_name.lower().strip()
                
                if node_lower in self.SEMANTIC_ASSOCIATIONS:
                    for assoc in self.SEMANTIC_ASSOCIATIONS[node_lower]:
                        if new_links >= self.config.max_links_per_cycle:
                            break
                        
                        if not self._link_exists(kg, node_name, assoc):
                            self._create_link(kg, node_name, assoc, 'related')
                            new_links += 1
            
            with self._metrics_lock:
                self.metrics.links_created += new_links
                self.metrics.nodes_curated += len(nodes[:self.config.max_nodes_per_cycle])
            
            if new_links > 0:
                logger.debug(f"Создано {new_links} семантических связей")
                
        except Exception as e:
            logger.debug(f"Ошибка создания связей: {e}")
    
    def _cleanup_orphans(self, kg):
        """Удалить сиротские узлы (без связей)."""
        try:
            nodes = self._get_all_nodes(kg)
            removed = 0
            
            for node in nodes:
                edges = self._get_node_edges(kg, node)
                if not edges:
                    # Удаляем если нет связей и низкая уверенность
                    if hasattr(node, 'confidence') and node.confidence < 0.3:
                        self._remove_node(kg, node)
                        removed += 1
            
            with self._metrics_lock:
                self.metrics.links_removed += removed
            
            if removed > 0:
                logger.debug(f"Удалено сирот: {removed}")
                self._publish_event(CuratorEventType.CLEANUP_COMPLETE, {"orphans_removed": removed})
                
        except Exception as e:
            logger.debug(f"Ошибка чистки сирот: {e}")
    
    def _cleanup_duplicates(self, kg):
        """Найти и объединить дубликаты."""
        try:
            nodes = self._get_all_nodes(kg)
            if not nodes or not hasattr(nodes[0], 'embedding') or not nodes[0].embedding:
                return
            
            # Простой поиск дубликатов по сходству эмбеддингов
            merged = 0
            
            for i, node in enumerate(nodes):
                if not hasattr(node, 'embedding') or not node.embedding:
                    continue
                
                for j in range(i + 1, len(nodes)):
                    other = nodes[j]
                    if not hasattr(other, 'embedding') or not other.embedding:
                        continue
                    
                    # Косинусное сходство
                    emb1 = np.array(node.embedding)
                    emb2 = np.array(other.embedding)
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
                    
                    if similarity > 0.95:
                        # Объединяем - удаляем other, добавляем связь к node
                        self._merge_nodes(kg, node, other)
                        merged += 1
                        
                        if merged >= 10:
                            break
                
                if merged >= 10:
                    break
            
            with self._metrics_lock:
                self.metrics.groups_merged += merged
                self.metrics.duplicate_clusters = max(0, self.metrics.duplicate_clusters - merged)
            
            if merged > 0:
                logger.debug(f"Объединено дубликатов: {merged}")
                
        except Exception as e:
            logger.debug(f"Ошибка чистки дубликатов: {e}")
    
    def _cleanup_garbage(self, kg):
        """Удалить мусорные узлы (шаблоны, низкое качество, повторы)."""
        try:
            nodes = self._get_all_nodes(kg)
            removed = 0
            
            for node in nodes:
                content = getattr(node, 'content', '') or getattr(node, 'name', '') or ''
                if not content:
                    continue
                
                content_lower = content.lower()
                is_garbage = False
                
                # 1. Проверка на шаблоны
                for pattern in self.TEMPLATE_PATTERNS:
                    if pattern.lower() in content_lower:
                        is_garbage = True
                        break
                
                # 2. Проверка на индикаторы низкого качества
                if not is_garbage:
                    indicator_count = sum(1 for ind in self.LOW_QUALITY_INDICATORS if ind in content_lower)
                    if indicator_count >= 3:
                        is_garbage = True
                
                # 3. Проверка на очень короткий контент с ссылками
                if not is_garbage and len(content) < 50 and ('http' in content_lower or 'www' in content_lower):
                    is_garbage = True
                
                # 4. Проверка на повторяющиеся символы
                if not is_garbage:
                    if '###' in content or '##' in content:
                        if content.count('###') > 3 or content.count('##') > 5:
                            is_garbage = True
                
                if is_garbage:
                    self._remove_node(kg, node)
                    removed += 1
                    
                    if removed >= 20:
                        break
            
            if removed > 0:
                logger.info(f"Удалено мусора: {removed}")
                with self._metrics_lock:
                    self.metrics.links_removed += removed
                self._publish_event(CuratorEventType.CLEANUP_COMPLETE, {"garbage_removed": removed})
                
        except Exception as e:
            logger.debug(f"Ошибка чистки мусора: {e}")
    
    def _recluster_if_needed(self, kg):
        """Пере кластеризация при необходимости."""
        try:
            if not hasattr(kg, 'auto_cluster'):
                return
            
            # Проверяем количество групп
            groups = getattr(kg, 'semantic_groups', None)
            if not groups or len(groups) < 2:
                return
            
            # Запускаем кластеризацию с низким порогом
            created = kg.auto_cluster(
                level=1,
                threshold=self.config.recluster_threshold,
                method="simple"
            )
            
            if created > 0:
                with self._metrics_lock:
                    self.metrics.groups_created += created
                logger.debug(f"Создано групп: {created}")
                self._publish_event(CuratorEventType.GRAPH_OPTIMIZED, {"groups_created": created})
                
        except Exception as e:
            logger.debug(f"Ошибка ре-кластеризации: {e}")
    
    # === ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ===
    
    def _get_all_nodes(self, kg) -> List:
        """Получить все узлы из графа."""
        try:
            if hasattr(kg, 'get_all_nodes'):
                return kg.get_all_nodes()
            elif hasattr(kg, 'nodes'):
                return list(kg.nodes.values()) if hasattr(kg.nodes, 'values') else []
            return []
        except Exception:
            return []
    
    def _get_node_edges(self, kg, node) -> List:
        """Получить связи узла."""
        try:
            node_id = getattr(node, 'id', None) or getattr(node, 'name', None)
            if not node_id:
                return []
            
            if hasattr(kg, 'get_edges'):
                return kg.get_edges(node_id)
            elif hasattr(kg, 'edges'):
                return [e for e in kg.edges.values() 
                       if e.source_id == node_id or e.target_id == node_id]
            return []
        except Exception:
            return []
    
    def _link_exists(self, kg, source: str, target: str) -> bool:
        """Проверить существование связи."""
        try:
            if hasattr(kg, 'edge_exists'):
                return kg.edge_exists(source, target)
            return False
        except Exception:
            return False
    
    def _create_link(self, kg, source: str, target: str, relation: str = 'related'):
        """Создать связь."""
        try:
            if hasattr(kg, 'add_edge'):
                kg.add_edge(source, target, relation=relation, weight=0.8)
        except Exception:
            pass
    
    def _remove_node(self, kg, node):
        """Удалить узел."""
        try:
            node_id = getattr(node, 'id', None)
            if node_id and hasattr(kg, 'remove_node'):
                kg.remove_node(node_id)
        except Exception:
            pass
    
    def _merge_nodes(self, kg, node1, node2):
        """Объединить два узла."""
        try:
            node2_id = getattr(node2, 'id', None)
            if node2_id and hasattr(kg, 'remove_node'):
                kg.remove_node(node2_id)
        except Exception:
            pass
    
    # === МОНИТОРИНГ ===
    
    def _monitor_loop(self):
        """Мониторинг состояния куратора."""
        while not self.stop_event.is_set():
            try:
                # Проверяем состояние
                if self.running and not self.paused:
                    with self._metrics_lock:
                        idle_time = time.time() - self.metrics.last_run
                        
                        if idle_time > self._current_interval * 2:
                            logger.warning(f"Куратор бездействует {idle_time:.1f}s")
                
                # Проверяем GGUF модели
                if self.config.extract_from_gguf:
                    self._scan_gguf_models()
                
                time.sleep(30)
                
            except Exception as e:
                logger.debug(f"Ошибка мониторинга: {e}")
                time.sleep(60)
    
    # === PUBLIC API ===
    
    def force_curation(self):
        """Принудительно запустить курацию."""
        if self.running and not self.paused:
            logger.info("Принудительная курация уже запущена")
            return
        
        def _run():
            self._run_curation_cycle()
        
        self.schedule_deferred(_run, priority="HIGH")
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Получить статистику графа."""
        kg = self._get_knowledge_graph()
        if not kg:
            return {}
        
        if hasattr(kg, 'get_stats'):
            return kg.get_stats()
        
        return {"nodes": len(self._get_all_nodes(kg))}


# === ФАБРИКА ===

def create_graph_curator(brain=None, config: Optional[Dict] = None) -> GraphCurator:
    """Создать куратор графа."""
    return GraphCurator(brain=brain, config=config)