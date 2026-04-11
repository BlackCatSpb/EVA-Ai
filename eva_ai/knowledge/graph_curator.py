"""
Graph Curator для FractalGraph v2
Оптимизация, консолидация и обслуживание графа знаний
"""
import logging
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from collections import defaultdict

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
        
        logger.info(f"GraphCurator initialized (FGv2)")
    
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
    
    def _run_loop(self):
        """Главный цикл куратора"""
        while self._running:
            try:
                if not self._paused:
                    self._do_curation()
                
                self.metrics['next_run'] = time.time() + self.check_interval
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"GraphCurator error: {e}")
                self.metrics['last_error'] = str(e)
                self.state = CuratorState.ERROR
                time.sleep(60)
    
    def _do_curation(self):
        """Выполнить цикл курирования"""
        with self._lock:
            try:
                self.metrics['last_run'] = time.time()
                
                fg = self._get_fractal_graph()
                if not fg or not hasattr(fg, 'storage'):
                    logger.warning("FGv2 not available for curation")
                    return
                
                storage = fg.storage
                
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
                
            except Exception as e:
                logger.error(f"Curation error: {e}")
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
            except:
                pass
        
        if nodes_to_remove:
            logger.info(f"Cleaned up {len(nodes_to_remove)} garbage nodes")
    
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


def create_graph_curator(brain=None, config=None) -> GraphCurator:
    """Factory function для создания GraphCurator"""
    return GraphCurator(brain=brain, config=config)
