"""
MemoryConsolidator - фоновая консолидация краткосрочной памяти.
В периоды CPU < 50% запускает консолидацию: объединяет фрагментированные 
узлы сессий в долгосрочные семантические группы, удаляет дубли.
"""
import logging
import threading
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import psutil

logger = logging.getLogger("eva_ai.memory_consolidator")

@dataclass
class ConsolidationResult:
    """Результат консолидации."""
    nodes_merged: int
    duplicates_removed: int
    groups_created: int
    processing_time: float

class MemoryConsolidator:
    """
    Фоновая консолидация памяти.
    Запускается при CPU < 50%, объединяет фрагменты сессий.
    """
    
    def __init__(self, brain=None, config: Optional[Dict] = None):
        self.brain = brain
        self.config = config or {}
        
        # Пороги
        self.cpu_threshold = self.config.get('cpu_threshold', 50.0)  # %
        self.min_idle_time = self.config.get('min_idle_time', 60)  # секунд
        self.max_nodes_per_cycle = self.config.get('max_nodes_per_cycle', 50)
        
        # Состояние
        self.running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
    def start(self):
        """Запустить фоновую консолидацию."""
        if self.running:
            logger.warning("MemoryConsolidator already running")
            return
            
        self.running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._consolidation_loop, daemon=True)
        self._thread.start()
        logger.info("MemoryConsolidator started")
        
    def stop(self):
        """Остановить консолидацию."""
        self.running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("MemoryConsolidator stopped")
        
    def _consolidation_loop(self):
        """Основной цикл консолидации."""
        while not self._stop_event.is_set():
            try:
                # Проверяем CPU
                cpu_percent = psutil.cpu_percent(interval=1)
                
                if cpu_percent >= self.cpu_threshold:
                    # CPU высокий - ждём
                    logger.debug(f"CPU high ({cpu_percent}%), skipping consolidation")
                    self._stop_event.wait(timeout=30)
                    continue
                
                # CPU низкий - выполняем консолидацию
                logger.info(f"CPU low ({cpu_percent}%), starting consolidation")
                result = self._run_consolidation()
                
                if result:
                    logger.info(f"Consolidation done: merged={result.nodes_merged}, "
                              f"removed={result.duplicates_removed}, groups={result.groups_created}")
                    
            except Exception as e:
                logger.error(f"Consolidation error: {e}")
                
            # Ждём перед следующим циклом
            self._stop_event.wait(timeout=300)  # 5 минут
        
    def _run_consolidation(self) -> Optional[ConsolidationResult]:
        """Выполнить один цикл консолидации."""
        start_time = time.time()
        
        if not self.brain:
            return None
            
        # Получаем граф
        fractal_graph = getattr(self.brain, 'fractal_graph_v2', None)
        if not fractal_graph:
            return None
            
        nodes_merged = 0
        duplicates_removed = 0
        groups_created = 0
        
        try:
            # 1. Поиск и объединение дубликатов
            duplicates_removed = self._merge_duplicates(fractal_graph)
            
            # 2. Создание семантических групп из фрагментов сессий
            groups_created = self._create_semantic_groups(fractal_graph)
            
            # 3. Удаление устаревших временных узлов
            nodes_removed = self._remove_old_temp_nodes(fractal_graph)
            
        except Exception as e:
            logger.error(f"Consolidation cycle error: {e}")
            
        return ConsolidationResult(
            nodes_merged=nodes_merged,
            duplicates_removed=duplicates_removed,
            groups_created=groups_created,
            processing_time=time.time() - start_time
        )
    
    def _merge_duplicates(self, graph) -> int:
        """Найти и объединить дубликаты."""
        # Упрощённая реализация - в реальности нужна проверка embedding similarity
        return 0
    
    def _create_semantic_groups(self, graph) -> int:
        """Создать семантические группы из фрагментов."""
        # Упрощённая реализация
        return 0
    
    def _remove_old_temp_nodes(self, graph) -> int:
        """Удалить старые временные узлы."""
        # Упрощённая реализация
        return 0
    
    def trigger_now(self) -> Optional[ConsolidationResult]:
        """Принудительно запустить консолидацию (для API)."""
        if not self.running:
            return self._run_consolidation()
        return None


def create_memory_consolidator(brain=None, config: Optional[Dict] = None) -> MemoryConsolidator:
    """Создать инстанс консолидатора."""
    return MemoryConsolidator(brain, config)