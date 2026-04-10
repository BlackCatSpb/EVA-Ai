"""
Graph Curator для FractalGraph v2
Запускает фоновые задачи по оптимизации и обслуживанию графа
"""
import logging
import time
import threading
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger("eva_ai.graph_curator")


class CuratorState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class GraphCurator:
    """
    Куратор графа - фоновый оптимизатор для FractalGraph v2
    """
    
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
            'nodes_curated': 0,
            'links_created': 0,
            'links_removed': 0,
            'state': 'idle',
            'last_run': 0,
            'next_run': time.time() + 300,
            'last_error': None
        }
        
        # Config
        self.check_interval = self.config.get('check_interval', 300)  # 5 min
        self.cleanup_enabled = self.config.get('cleanup_enabled', True)
        
        logger.info(f"GraphCurator initialized (FGv2)")
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def running(self) -> bool:
        return self._running
    
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
    
    def pause(self):
        self._paused = True
        self.state = CuratorState.PAUSED
        self.metrics['state'] = 'paused'
    
    def resume(self):
        self._paused = False
        self.state = CuratorState.RUNNING
        self.metrics['state'] = 'running'
    
    def force_curation(self):
        """Принудительный запуск курирования"""
        threading.Thread(target=self._do_curation, daemon=True).start()
    
    def get_state(self) -> str:
        return self.state.value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Получить метрики"""
        return {
            'cycles_completed': self.metrics['cycles_completed'],
            'nodes_curated': self.metrics['nodes_curated'],
            'links_created': self.metrics['links_created'],
            'links_removed': self.metrics['links_removed'],
            'state': self.metrics['state'],
            'last_run': self.metrics['last_run'],
            'next_run': self.metrics['next_run'],
            'last_error': self.metrics.get('last_error')
        }
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Получить статистику графа"""
        fg = self._get_fractal_graph()
        if fg:
            return {
                'total_nodes': len(fg.storage.nodes),
                'total_edges': len(fg.storage.edges),
                'total_groups': len(fg.storage.semantic_groups)
            }
        return {}
    
    def _get_fractal_graph(self):
        """Получить ссылку на FGv2"""
        if self.brain:
            return getattr(self.brain, 'fractal_graph_v2', None)
        return None
    
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
        """Выполнить курирование"""
        with self._lock:
            try:
                self.metrics['last_run'] = time.time()
                
                fg = self._get_fractal_graph()
                if not fg:
                    logger.warning("FGv2 not available for curation")
                    return
                
                # Update stats
                nodes_count = len(fg.storage.nodes)
                edges_count = len(fg.storage.edges)
                groups_count = len(fg.storage.semantic_groups)
                
                self.metrics['nodes_curated'] = nodes_count
                
                # Cleanup if enabled
                if self.cleanup_enabled:
                    self._cleanup_orphans(fg)
                
                self.metrics['cycles_completed'] += 1
                self.metrics['state'] = 'running'
                
                logger.debug(f"Curation: {nodes_count} nodes, {edges_count} edges, {groups_count} groups")
                
            except Exception as e:
                logger.error(f"Curation error: {e}")
                self.metrics['last_error'] = str(e)
    
    def _cleanup_orphans(self, fg):
        """Очистка изолированных узлов"""
        # Проверяем узлы без связей и удаляем если их слишком много
        orphan_count = 0
        for node_id, node in list(fg.storage.nodes.items()):
            edges = fg.storage.get_edges_for_node(node_id) if hasattr(fg.storage, 'get_edges_for_node') else []
            if not edges and len(fg.storage.nodes) > 100:
                # Удаляем старые изолированные узлы (старше 7 дней)
                created_at = getattr(node, 'created_at', 0)
                if time.time() - created_at > 7 * 24 * 3600:
                    try:
                        del fg.storage.nodes[node_id]
                        orphan_count += 1
                    except:
                        pass
        
        self.metrics['links_removed'] = orphan_count
