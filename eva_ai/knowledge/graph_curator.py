"""
Graph Curator для FractalGraph v2
Оптимизация, консолидация и обслуживание графа знаний

Features:
- Защита важных узлов
- Консолидация по уровням и типам
- Промоут/демоут между слоями
- Очистка мусора
- Поддержание когерентности групп
- Интеграция с FractalGraphV2 temporal decay
"""
import logging
import time
import threading
import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from collections import defaultdict

from eva_ai.core.event_bus import Event, EventPriority

logger = logging.getLogger("eva_ai.graph_curator")


class CuratorState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class GraphCurator:
    """
    Куратор графа - интеллектуальный оптимизатор для FractalGraph v2.
    
    Функции:
    1. Защита важных узлов (концепты, противоречия, статичные)
    2. Консолидация узлов по уровням и типам
    3. Промоут/демоут узлов между слоями
    4. Очистка мусора (осиротевшие, устаревшие с низкой уверенностью)
    5. Поддержание когерентности групп
    """
    
    # Типы узлов, которые НЕ удаляем
    PROTECTED_TYPES = {
        'concept', 'contradiction', 'model_a', 'model_b', 'model_c', 
        'model_root', 'semantic_group', 'domain_profile'
    }
    
    # Минимальная эффективная уверенность для сохранения
    MIN_EFFECTIVE_CONFIDENCE = 0.15
    
    # Пороги для промоута/демоута
    PROMOTE_THRESHOLD = 0.8  # Уверенность для повышения уровня
    DEMOTE_THRESHOLD = 0.2   # Уверенность для понижения уровня
    
    def __init__(self, brain=None, config: Optional[Dict] = None):
        self.brain = brain
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        
        # State
        self.state = CuratorState.IDLE
        self._running = False
        self._paused = False
        self._thread = None
        self._lock = threading.Lock()
        
        # Metrics
        self.metrics = {
            'cycles_completed': 0,
            'nodes_promoted': 0,
            'nodes_demoted': 0,
            'nodes_consolidated': 0,
            'nodes_removed': 0,
            'groups_created': 0,
            'groups_merged': 0,
            'state': 'idle',
            'last_run': 0,
            'next_run': time.time() + 300,
            'last_error': None
        }
        
        # Config
        self.check_interval = self.config.get('check_interval', 600)  # 10 min
        self.consolidation_enabled = self.config.get('consolidation_enabled', True)
        self.cleanup_enabled = self.config.get('cleanup_enabled', True)
        self.promotion_enabled = self.config.get('promotion_enabled', True)
        
        # EventBus
        self._event_bus = None
        if self.brain and hasattr(self.brain, 'events'):
            self._event_bus = self.brain.events
            self._subscribe_to_events()
        
        # DeferredCommandSystem
        self._deferred_system = None
        if self.brain and hasattr(self.brain, 'deferred_system'):
            self._deferred_system = self.brain.deferred_system
        
        logger.info(f"GraphCurator initialized (FGv2)")
    
    def _subscribe_to_events(self):
        """Подписка на события EventBus."""
        if not self._event_bus:
            return
        
        try:
            self._event_bus.subscribe("system.idle", self._on_system_idle, priority=3)
            self._event_bus.subscribe("memory.graph_updated", self._on_graph_updated, priority=5)
            self._event_bus.subscribe("memory.node_created", self._on_node_created, priority=5)
            logger.debug("GraphCurator subscribed to events")
        except Exception as e:
            logger.debug(f"Event subscription error: {e}")
    
    def _on_system_idle(self, event):
        """Обработка события простоя системы - запуск курирования."""
        if not self._running or self._paused:
            return
        
        try:
            logger.debug("System idle - running curation")
            self._do_curation()
        except Exception as e:
            import traceback
            logger.error(f"Curation error on idle: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
    
    def _on_graph_updated(self, event):
        """Обработка обновления графа - отложенный запуск."""
        if not self._running or self._paused:
            return
        
        # Безопасная обработка event объекта
        try:
            if hasattr(event, 'data') and event.data is not None:
                if isinstance(event.data, dict):
                    data = event.data
                else:
                    data = {}
            elif isinstance(event, dict):
                data = event
            else:
                data = {}
        except Exception:
            data = {}
        
        if not isinstance(data, dict):
            data = {}
        
        if data.get('skip_curation'):
            return
        
        if self._deferred_system:
            from eva_ai.core.deferred_command_system import CommandPriority
            self._deferred_system.add_command(
                command=self._do_curation,
                priority=CommandPriority.NORMAL
            )
        else:
            self.force_curation()
    
    def _on_node_created(self, event):
        """Обработка создания узла - отложенная оптимизация."""
        if not self._running or self._paused:
            return
        
        # Безопасная обработка event объекта
        try:
            if hasattr(event, 'data') and event.data is not None:
                if isinstance(event.data, dict):
                    data = event.data
                else:
                    data = {}
            elif isinstance(event, dict):
                data = event
            else:
                data = {}
        except Exception:
            data = {}
        
        # Проверка на валидный dict
        if not isinstance(data, dict):
            logger.debug(f"Skipping curation: invalid event data type {type(data)}")
            return
        
        node_type = data.get('node_type', '')
        
        if node_type in self.PROTECTED_TYPES:
            return
        
        if self._deferred_system:
            from eva_ai.core.deferred_command_system import CommandPriority
            self._deferred_system.add_command(
                command=self._do_curation,
                priority=CommandPriority.LOW
            )
    
    def _get_fractal_graph(self):
        """Получить ссылку на FGv2"""
        if self.brain:
            return getattr(self.brain, 'fractal_graph_v2', None)
        return None
    
    def _is_protected_node(self, node) -> bool:
        """
        Проверка защиты узла от удаления/модификации.
        
        Защищены:
        - Статичные узлы (is_static=True)
        - Узлы типа concept, contradiction и др.
        - Узлы с is_contradiction=True
        - Узлы с высокой уверенностью (>0.7) и частым доступом
        """
        # Статичные узлы
        if getattr(node, 'is_static', False):
            return True
        
        # Типы под защитой
        node_type = getattr(node, 'node_type', '')
        if node_type in self.PROTECTED_TYPES:
            return True
        
        # Узлы помеченные как противоречие
        if getattr(node, 'is_contradiction', False):
            return True
        
        # Высокоуверенные узлы с активным использованием
        if getattr(node, 'confidence', 0) > 0.7 and getattr(node, 'access_count', 0) > 10:
            return True
        
        return False
    
    def start(self):
        """Запустить куратор"""
        if self._running:
            logger.debug("GraphCurator already running")
            return
        
        self._running = True
        self._paused = False
        self.state = CuratorState.RUNNING
        self.metrics['state'] = 'running'
        
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("GraphCurator started")
    
    def stop(self):
        """Остановить куратор"""
        self._running = False
        self.state = CuratorState.IDLE
        self.metrics['state'] = 'idle'
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("GraphCurator stopped")
    
    def is_running(self) -> bool:
        """Проверить, запущен ли куратор."""
        return self._running and self._thread is not None and self._thread.is_alive()
    
    def _run_loop(self):
        """Главный цикл куратора с ПРИНУДИТЕЛЬНЫМ запуском каждые 5 минут."""
        # Фиксированный интервал 5 минут (300 сек)
        forced_interval = 300  # 5 minutes
        logger.info(f"[Curator] Loop started. Forced run every {forced_interval}s")
        
        while self._running:
            try:
                if not self._paused:
                    current_time = time.time()
                    # Запускаем если:
                    # 1. Прошло 5 минут с последнего запуска
                    # 2. ИЛИ пришло событие system.idle
                    time_since_last = current_time - self.metrics.get('last_run', 0)
                    
                    if time_since_last >= forced_interval:
                        logger.info(f"[Curator] Forced run (5 min interval passed)")
                        self._do_curation()
                    elif self._event_bus:
                        # Ждем события idle (если есть EventBus)
                        time.sleep(10)  # Проверяем каждые 10 сек
                        continue
                else:
                    time.sleep(10)
                    
            except Exception as e:
                logger.error(f"GraphCurator error: {e}")
                self.metrics['last_error'] = str(e)
                self.state = CuratorState.ERROR
                time.sleep(60)
    
    def _compute_adaptive_interval(self) -> float:
        """Вычисляет адаптивный интервал на основе активности системы."""
        base = self.check_interval
        
        # Увеличиваем интервал при высокой нагрузке
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Если система под нагрузкой - увеличиваем интервал
            if cpu_percent > 80 or memory.percent > 85:
                return base * 2.0
            if cpu_percent > 60 or memory.percent > 70:
                return base * 1.5
            
        except ImportError:
            pass
        
        # Уменьшаем интервал если граф большой и активный
        fg = self._get_fractal_graph()
        if fg and hasattr(fg, 'storage'):
            node_count = len(getattr(fg.storage, 'nodes', {}))
            
            # Большой граф требует более частого курирования
            if node_count > 10000:
                return base * 0.5  # Вдвое чаще
            elif node_count > 5000:
                return base * 0.75  # На 25% чаще
        
        return base
    
    def _do_curation(self):
        """Выполнить цикл курирования"""
        with self._lock:
            try:
                logger.debug("_do_curation: started")
                
                # Ensure metrics is always a dict with required keys
                if not isinstance(self.metrics, dict):
                    logger.warning(f"metrics is not a dict: {type(self.metrics)}, resetting")
                    self.metrics = {}
                
                # Ensure all required keys exist
                required_keys = ['cycles_completed', 'nodes_promoted', 'nodes_demoted', 
                               'nodes_consolidated', 'nodes_removed', 'last_run', 'last_error']
                for key in required_keys:
                    if key not in self.metrics:
                        self.metrics[key] = 0 if key != 'last_error' else None
                
                self.metrics['last_run'] = time.time()

                fg = self._get_fractal_graph()
                if not fg or not hasattr(fg, 'storage'):
                    logger.warning("FGv2 not available for curation")
                    logger.debug("_do_curation: FGv2 not available, returning")
                    return

                storage = fg.storage
                
                # Validate storage has expected attributes and is not an Event object
                if not hasattr(storage, 'nodes') or not hasattr(storage, 'semantic_groups'):
                    logger.warning(f"storage doesn't have expected attributes: {type(storage)}")
                    return
                
                # Extra check: ensure storage is not an Event object
                if hasattr(storage, 'event_type'):  # Event objects have event_type
                    logger.warning(f"storage appears to be an Event object: {type(storage)}")
                    return
                
                logger.debug(f"_do_curation: storage has {len(storage.nodes)} nodes")

                # 1. Очистка мусора (только незащищенных узлов)
                if self.cleanup_enabled:
                    self._cleanup_garbage(storage)

                # 2. Промоут/демоут узлов между уровнями
                if self.promotion_enabled:
                    self._process_level_promotions(storage)

                # 3. Консолидация узлов в группы
                if self.consolidation_enabled:
                    self._consolidate_nodes(storage)

                # 4. Обновление метрик
                self._update_metrics(storage)

                self.metrics['cycles_completed'] += 1
                self.metrics['state'] = 'running'

                logger.debug(f"Curation completed: cycle #{self.metrics['cycles_completed']}")

                if self._event_bus:
                    self._event_bus.publish(Event(
                        event_type="curator.curation_complete",
                        source="graph_curator",
                        data={
                            "cycles": self.metrics['cycles_completed'],
                            "nodes_promoted": self.metrics.get('nodes_promoted', 0),
                            "nodes_demoted": self.metrics.get('nodes_demoted', 0),
                            "nodes_consolidated": self.metrics.get('nodes_consolidated', 0)
                        },
                        priority=EventPriority.NORMAL
                    ))
                logger.debug("_do_curation: completed successfully")

            except AttributeError as ae:
                import traceback
                tb = traceback.format_exc()
                logger.error(f"Curation error: {ae}")
                logger.error(f"Traceback: {tb}")
                self.metrics['last_error'] = f"AttributeError: {ae}"
                # Debug: log metrics type
                logger.error(f"DEBUG: self.metrics type = {type(self.metrics)}")
                if hasattr(self, '_fractal_graph'):
                    logger.error(f"DEBUG: fg type = {type(self._fractal_graph)}")
                    if hasattr(self._fractal_graph, 'storage'):
                        logger.error(f"DEBUG: storage type = {type(self._fractal_graph.storage)}")
            except Exception as e:
                import traceback
                logger.error(f"Curation error: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                self.metrics['last_error'] = str(e)
    
    def _cleanup_garbage(self, storage):
        """
        Очистка мусора: удаление осиротевших и устаревших узлов.
        
        Удаляем только если:
        - Не защищенный тип
        - Нет связей (orphan)
        - Эффективная уверенность < MIN_EFFECTIVE_CONFIDENCE
        - Создан более 7 дней назад
        """
        nodes_to_remove = []
        
        for node_id, node in list(storage.nodes.items()):
            # Пропускаем защищенные узлы
            if self._is_protected_node(node):
                continue
            
            # Проверяем наличие связей
            has_edges = False
            if hasattr(storage, 'edges'):
                for edge in storage.edges.values():
                    if (getattr(edge, 'source_id', None) == node_id or 
                        getattr(edge, 'target_id', None) == node_id):
                        has_edges = True
                        break
            
            if has_edges:
                continue
            
            # Проверяем эффективную уверенность
            effective_conf = 0.5
            if hasattr(node, 'get_effective_confidence'):
                effective_conf = node.get_effective_confidence()
            else:
                # Fallback расчет
                conf = getattr(node, 'confidence', 0.5)
                last_access = getattr(node, 'last_accessed', time.time())
                delta_days = (time.time() - last_access) / 86400
                effective_conf = conf * np.exp(-0.01 * delta_days)
            
            if effective_conf < self.MIN_EFFECTIVE_CONFIDENCE:
                # Проверяем возраст (не удаляем свежие)
                created_at = getattr(node, 'created_at', time.time())
                age_days = (time.time() - created_at) / 86400
                
                if age_days > 7:  # Удаляем только старые
                    nodes_to_remove.append(node_id)
        
        # Удаляем отобранные узлы
        for node_id in nodes_to_remove[:50]:  # Максимум 50 за раз
            try:
                del storage.nodes[node_id]
                self.metrics['nodes_removed'] += 1
            except Exception:
                pass
        
        if nodes_to_remove:
            logger.info(f"Cleaned up {len(nodes_to_remove)} garbage nodes")
            if self._event_bus:
                self._event_bus.publish(Event(
                    event_type="curator.cleanup_done",
                    source="graph_curator",
                    data={
                        "nodes_removed": len(nodes_to_remove),
                        "total_removed": self.metrics.get('nodes_removed', 0)
                    },
                    priority=EventPriority.LOW
                ))
    
    def _process_level_promotions(self, storage):
        """
        Промоут/демоут узлов между фрактальными уровнями.
        
        Логика:
        - Узлы с confidence > PROMOTE_THRESHOLD → повышаем уровень (укрупняем)
        - Узлы с confidence < DEMOTE_THRESHOLD → понижаем уровень (детализируем)
        - Узлы с высоким access_count → повышаем уровень (важность)
        """
        promoted = 0
        demoted = 0
        
        for node_id, node in list(storage.nodes.items()):
            # Пропускаем защищенные
            if self._is_protected_node(node):
                continue
            
            current_level = getattr(node, 'level', 1)
            confidence = getattr(node, 'confidence', 0.5)
            access_count = getattr(node, 'access_count', 0)
            
            # Промоут: высокая уверенность и частый доступ
            if confidence > self.PROMOTE_THRESHOLD and access_count > 5:
                if current_level < 3:  # Максимум уровень 3
                    node.level = current_level + 1
                    node.version += 1
                    promoted += 1
            
            # Демоут: низкая уверенность и редкий доступ
            elif confidence < self.DEMOTE_THRESHOLD and access_count < 2:
                if current_level > 0:  # Минимум уровень 0
                    node.level = current_level - 1
                    node.version += 1
                    demoted += 1
        
        self.metrics['nodes_promoted'] += promoted
        self.metrics['nodes_demoted'] += demoted
        
        if promoted or demoted:
            logger.info(f"Level changes: {promoted} promoted, {demoted} demoted")
    
    def _consolidate_nodes(self, storage):
        """
        Консолидация узлов: создание и обновление семантических групп.
        
        Логика:
        1. Находим кластеры узлов по косинусному сходству
        2. Создаем/обновляем SemanticGroup для кластеров
        3. Перемещаем узлы в группы
        """
        if not hasattr(storage, 'cluster_nodes'):
            return
        
        try:
            # Кластеризация для каждого уровня
            for level in [1, 2]:
                clusters = storage.cluster_nodes(
                    level=level,
                    threshold=0.6,
                    method="simple"
                )
                
                for cluster_name, node_ids in clusters.items():
                    if len(node_ids) < 3:
                        continue
                    
                    # Проверяем существование группы
                    existing_group = None
                    for group in storage.semantic_groups.values():
                        if group.name == cluster_name:
                            existing_group = group
                            break
                    
                    if existing_group:
                        # Обновляем существующую группу
                        self._update_group(existing_group, node_ids, storage)
                    else:
                        # Создаем новую группу
                        if hasattr(storage, 'create_semantic_group'):
                            storage.create_semantic_group(
                                name=cluster_name,
                                member_ids=node_ids,
                                level=level + 1
                            )
                            self.metrics['groups_created'] += 1
            
            # Объединение пересекающихся групп
            self._merge_overlapping_groups(storage)
            
        except Exception as e:
            logger.error(f"Consolidation error: {e}")
    
    def _update_group(self, group, node_ids, storage):
        """Обновление существующей группы новыми членами"""
        current_members = set()
        for node_id in node_ids:
            if node_id in storage.nodes:
                node = storage.nodes[node_id]
                if getattr(node, 'parent_group_id', None) == group.id:
                    current_members.add(node_id)
        
        # Добавляем новых членов
        new_members = set(node_ids) - current_members
        for node_id in new_members:
            if node_id in storage.nodes:
                storage.nodes[node_id].parent_group_id = group.id
                group.member_count += 1
        
        # Пересчитываем центроид
        if hasattr(group, 'embedding') and node_ids:
            embeddings = []
            for nid in node_ids[:10]:  # Берем первые 10
                if nid in storage.nodes and storage.nodes[nid].embedding:
                    embeddings.append(np.array(storage.nodes[nid].embedding))
            
            if embeddings:
                group.embedding = np.mean(embeddings, axis=0).tolist()
        
        group.updated_at = time.time()
    
    def _merge_overlapping_groups(self, storage):
        """Объединение пересекающихся семантических групп"""
        if not hasattr(storage, 'semantic_groups'):
            return
        
        groups = list(storage.semantic_groups.values())
        merged = set()
        
        for i, group1 in enumerate(groups):
            if group1.id in merged:
                continue
            
            for group2 in groups[i+1:]:
                if group2.id in merged:
                    continue
                
                # Проверяем пересечение членов
                members1 = set()
                members2 = set()
                
                for nid, node in storage.nodes.items():
                    if getattr(node, 'parent_group_id', None) == group1.id:
                        members1.add(nid)
                    elif getattr(node, 'parent_group_id', None) == group2.id:
                        members2.add(nid)
                
                # Если пересечение > 70% - объединяем
                if members1 and members2:
                    intersection = members1 & members2
                    if len(intersection) / min(len(members1), len(members2)) > 0.7:
                        # Переносим членов group2 в group1
                        for nid in members2:
                            if nid in storage.nodes:
                                storage.nodes[nid].parent_group_id = group1.id
                        
                        # Обновляем group1
                        group1.member_count += len(members2)
                        group1.updated_at = time.time()
                        
                        # Удаляем group2
                        del storage.semantic_groups[group2.id]
                        merged.add(group2.id)
                        self.metrics['groups_merged'] += 1
    
    def _update_metrics(self, storage):
        """Обновление метрик куратора"""
        try:
            self.metrics['total_nodes'] = len(storage.nodes)
            self.metrics['total_groups'] = len(storage.semantic_groups)
            
            # Подсчет по типам
            type_counts = defaultdict(int)
            for node in storage.nodes.values():
                node_type = getattr(node, 'node_type', 'unknown')
                type_counts[node_type] += 1
            
            self.metrics['nodes_by_type'] = dict(type_counts)
            
        except Exception as e:
            logger.debug(f"Metrics update error: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Получить метрики куратора"""
        return self.metrics.copy()
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Получить статистику графа"""
        fg = self._get_fractal_graph()
        if fg and hasattr(fg, 'storage'):
            storage = fg.storage
            return {
                'total_nodes': len(storage.nodes),
                'total_edges': len(storage.edges),
                'total_groups': len(storage.semantic_groups),
                'protected_nodes': sum(1 for n in storage.nodes.values() if self._is_protected_node(n))
            }
        return {}
    
    def force_curation(self):
        """Принудительный запуск курирования"""
        threading.Thread(target=self._do_curation, daemon=True).start()
    
    # === ИЕРАРХИЧЕСКАЯ ОПТИМИЗАЦИЯ ===
    
    def build_hierarchical_index(self, force: bool = False):
        """Построить иерархический индекс для быстрого поиска."""
        fg = self._get_fractal_graph()
        if not fg or not hasattr(fg, 'storage'):
            logger.warning("FGv2 not available for hierarchical index")
            return
        
        storage = fg.storage
        
        if hasattr(storage, 'build_hierarchical_index'):
            if force or not hasattr(storage, '_hierarchical_index') or not storage._hierarchical_index:
                logger.info("Building hierarchical index...")
                storage.build_hierarchical_index()
                logger.info("Hierarchical index built successfully")
        
        if hasattr(fg, '_hierarchical_index'):
            fg._hierarchical_index = storage._hierarchical_index
    
    def optimize_hierarchical_index(self):
        """Оптимизировать иерархический индекс - обновить при изменениях графа."""
        fg = self._get_fractal_graph()
        if not fg:
            return
        
        storage = getattr(fg, 'storage', None)
        if not storage:
            return
        
        if hasattr(storage, '_hierarchical_index') and storage._hierarchical_index:
            try:
                storage._hierarchical_index.build_from_graph(
                    storage.nodes,
                    storage.semantic_groups,
                    storage.nodes_by_level
                )
                logger.debug("Hierarchical index optimized")
            except Exception as e:
                logger.warning(f"Hierarchical index optimization failed: {e}")
    
    # === ОБЪЕДИНЕННЫЕ МЕТОДЫ (из FCP Curator) ===
    
    def detect_contradictions(self, storage) -> int:
        """Обнаружение противоречий в графе."""
        if not hasattr(storage, 'detect_contradiction'):
            return 0
        
        contradictions_found = 0
        groups = list(storage.semantic_groups.values())
        
        for group in groups[:20]:
            if not group.embedding:
                continue
            
            try:
                is_contr, distance = storage.detect_contradiction(
                    group.embedding,
                    group.id,
                    threshold=0.7
                )
                if is_contr:
                    contradictions_found += 1
                    storage.mark_contradiction(group.id, "Auto-detected by curator")
            except Exception as e:
                logger.debug(f"Contradiction detection error: {e}")
        
        self.metrics['contradictions_found'] = (
            self.metrics.get('contradictions_found', 0) + contradictions_found
        )
        logger.info(f"Detected {contradictions_found} potential contradictions")
        return contradictions_found
    
    def prune_duplicates(self, storage, threshold: float = 0.95) -> int:
        """Удаление дубликатов - объединение очень похожих узлов."""
        if not hasattr(storage, 'nodes'):
            return 0
        
        removed = 0
        nodes_by_type = defaultdict(list)
        
        for node_id, node in storage.nodes.items():
            if node.embedding:
                nodes_by_type[node.node_type].append((node_id, node))
        
        for node_type, nodes in nodes_by_type.items():
            if len(nodes) < 2:
                continue
            
            merged = set()
            for i, (id1, node1) in enumerate(nodes):
                if id1 in merged:
                    continue
                
                for j, (id2, node2) in enumerate(nodes[i+1:], i+1):
                    if id2 in merged or not node2.embedding:
                        continue
                    
                    try:
                        emb1 = np.array(node1.embedding)
                        emb2 = np.array(node2.embedding)
                        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
                        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
                        
                        sim = float(np.dot(emb1, emb2))
                        if sim >= threshold:
                            if node1.confidence >= node2.confidence:
                                storage.mark_contradiction(id2, f"Duplicate of {id1} (sim={sim:.2f})")
                                merged.add(id2)
                            else:
                                storage.mark_contradiction(id1, f"Duplicate of {id2} (sim={sim:.2f})")
                                merged.add(id1)
                            removed += 1
                    except Exception:
                        pass
        
        self.metrics['duplicates_pruned'] = self.metrics.get('duplicates_pruned', 0) + removed
        logger.info(f"Pruned {removed} duplicate nodes")
        return removed
    
    def decay_nodes(self, storage) -> int:
        """Временной распад - снижение уверенности старых узлов."""
        decayed = 0
        now = time.time()
        
        for node_id, node in storage.nodes.items():
            if self._is_protected_node(node):
                continue
            
            last_access = getattr(node, 'last_accessed', now)
            days_inactive = (now - last_access) / 86400
            
            if days_inactive > 30:
                lambda_decay = getattr(node, 'domain_lambda', 0.01)
                decay_factor = np.exp(-lambda_decay * days_inactive)
                new_confidence = node.confidence * decay_factor
                
                if new_confidence < 0.1:
                    node.confidence = 0.1
                else:
                    node.confidence = new_confidence
                
                if hasattr(storage, '_save_node'):
                    storage._save_node(node)
                decayed += 1
        
        self.metrics['nodes_decayed'] = self.metrics.get('nodes_decayed', 0) + decayed
        logger.info(f"Decayed {decayed} inactive nodes")
        return decayed
    
    # === GC-1: Интеграция с FG2 Temporal Decay ===
    
    def integrate_with_fg2_decay(self, force: bool = False) -> Dict:
        """
        GC-1: Использовать методы временного распада из FractalGraphV2.
        
        Args:
            force: Если True - применить распад, иначе dry_run
            
        Returns:
            Результаты применения распада
        """
        fg = self._get_fractal_graph()
        if not fg or not hasattr(fg, 'storage'):
            return {"error": "FG2 not available"}
        
        storage = fg.storage
        
        if not hasattr(storage, 'apply_temporal_decay'):
            return {"error": "apply_temporal_decay not available"}
        
        try:
            result = storage.apply_temporal_decay(
                lambda_base=0.01,
                min_confidence=0.15,
                dry_run=not force
            )
            
            logger.info(f"FG2 decay integration: {result}")
            return result
            
        except Exception as e:
            logger.error(f"FG2 decay integration error: {e}")
            return {"error": str(e)}
    
    def get_fg2_decay_statistics(self) -> Dict:
        """GC-1: Получить статистику распада из FG2."""
        fg = self._get_fractal_graph()
        if not fg or not hasattr(fg, 'storage'):
            return {}
        
        storage = fg.storage
        
        if hasattr(storage, 'get_decay_statistics'):
            return storage.get_decay_statistics()
        return {}
    
    # === GC-2: Асинхронное курирование ===
    
    async def async_curation(self):
        """
        GC-2: Асинхронный цикл курирования для лучшей интеграции с EventBus.
        """
        await asyncio.sleep(0)
        
        with self._lock:
            try:
                fg = self._get_fractal_graph()
                if not fg or not hasattr(fg, 'storage'):
                    return
                
                storage = fg.storage
                
                tasks = []
                
                if self.cleanup_enabled:
                    tasks.append(self._async_cleanup_garbage(storage))
                
                if self.promotion_enabled:
                    tasks.append(self._async_promotions(storage))
                
                if tasks:
                    await asyncio.gather(*tasks)
                
                self.metrics['cycles_completed'] += 1
                
            except Exception as e:
                logger.error(f"Async curation error: {e}")
    
    async def _async_cleanup_garbage(self, storage):
        """GC-2: Асинхронная очистка мусора."""
        await asyncio.sleep(0)
        self._cleanup_garbage(storage)
    
    async def _async_promotions(self, storage):
        """GC-2: Асинхронный промоут/демоут."""
        await asyncio.sleep(0)
        self._process_level_promotions(storage)
    
    # === GC-3: Расширенная статистика графа ===
    
    def get_extended_graph_stats(self) -> Dict:
        """
        GC-3: Расширенная статистика графа с FG2 decay данными.
        
        Returns:
            {
                "total_nodes": int,
                "by_type": dict,
                "by_level": dict,
                "decay_stats": dict,
                "protected_count": int,
                "contradictions": int
            }
        """
        fg = self._get_fractal_graph()
        if not fg or not hasattr(fg, 'storage'):
            return {}
        
        storage = fg.storage
        
        stats = {
            "total_nodes": len(storage.nodes),
            "total_edges": len(storage.edges),
            "total_groups": len(storage.semantic_groups)
        }
        
        type_counts = defaultdict(int)
        level_counts = defaultdict(int)
        
        for node_id, node in storage.nodes.items():
            node_type = getattr(node, 'node_type', 'unknown')
            level = getattr(node, 'level', 0)
            type_counts[node_type] += 1
            level_counts[level] += 1
        
        stats["by_type"] = dict(type_counts)
        stats["by_level"] = dict(level_counts)
        stats["protected_count"] = sum(1 for n in storage.nodes.values() if self._is_protected_node(n))
        stats["contradictions"] = sum(1 for n in storage.nodes.values() if getattr(n, 'is_contradiction', False))
        
        if hasattr(storage, 'get_decay_statistics'):
            stats["decay_stats"] = storage.get_decay_statistics()
        
        return stats
    
    def get_curation_recommendations(self) -> List[Dict]:
        """
        GC-3: Получить рекомендации по оптимизации графа.
        
        Returns:
            List of {issue, recommendation, priority}
        """
        recommendations = []
        stats = self.get_extended_graph_stats()
        
        if not stats:
            return recommendations
        
        decay_stats = stats.get("decay_stats", {})
        if decay_stats:
            age_range = decay_stats.get("age_range_days", 0)
            if age_range > 90:
                recommendations.append({
                    "issue": f"Graph contains nodes older than {int(age_range)} days",
                    "recommendation": "Run FG2 temporal decay to clean up old nodes",
                    "priority": "medium"
                })
        
        nodes_by_type = stats.get("by_type", {})
        unknown_count = nodes_by_type.get("unknown", 0)
        if unknown_count > stats.get("total_nodes", 1) * 0.1:
            recommendations.append({
                "issue": f"High count of unknown type nodes: {unknown_count}",
                "recommendation": "Review and categorize untyped nodes",
                "priority": "low"
            })
        
        protected_pct = stats.get("protected_count", 0) / max(stats.get("total_nodes", 1), 1)
        if protected_pct < 0.1:
            recommendations.append({
                "issue": f"Very few protected nodes ({protected_pct:.1%})",
                "recommendation": "Consider marking important concepts as protected",
                "priority": "low"
            })
        
        return recommendations
    
    # === РАСШИРЕННЫЙ ЦИКЛ КУРИРОВАНИЯ ===
    
    def _do_extended_curation(self):
        """Расширенный цикл курирования с иерархической оптимизацией."""
        with self._lock:
            try:
                fg = self._get_fractal_graph()
                if not fg or not hasattr(fg, 'storage'):
                    return
                
                storage = fg.storage
                
                # Стандартные операции
                if self.cleanup_enabled:
                    self._cleanup_garbage(storage)
                if self.promotion_enabled:
                    self._process_level_promotions(storage)
                if self.consolidation_enabled:
                    self._consolidate_nodes(storage)
                
                # Расширенные операции (низкий приоритет)
                if self.metrics['cycles_completed'] % 3 == 0:
                    self.detect_contradictions(storage)
                
                if self.metrics['cycles_completed'] % 5 == 0:
                    self.prune_duplicates(storage)
                    self.decay_nodes(storage)
                
                # Обновление иерархического индекса
                if self.metrics['cycles_completed'] % 10 == 0:
                    self.optimize_hierarchical_index()
                
                self._update_metrics(storage)
                self.metrics['cycles_completed'] += 1
                
            except Exception as e:
                logger.error(f"Extended curation error: {e}")


def create_graph_curator(brain=None, config=None) -> GraphCurator:
    """Factory function для создания GraphCurator"""
    return GraphCurator(brain=brain, config=config)
