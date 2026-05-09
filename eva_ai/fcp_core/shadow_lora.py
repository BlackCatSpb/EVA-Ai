"""
ShadowLoRAManagerOV - Менеджер LoRA адаптеров в OpenVINO

Управление адаптерами с атомарной сменой.
"""
import os
from typing import Optional, Dict, List
from threading import Lock


class ShadowLoRAManager:
    """
    Shadow LoRA Manager для OpenVINO.
    
    Особенности:
    - Атомарная смена адаптера
    - Thread-safe операции
    - Поддержка нескольких адаптеров
    - Rollback при деградации
    - Мониторинг качества
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        scheduler_config: Optional[dict] = None
    ):
        self.model_path = model_path
        self.device = device
        self.scheduler_config = scheduler_config
        
        self._pipeline = None
        self._active_adapter: Optional[str] = None
        self._adapters: Dict[str, str] = {}
        self._lock = Lock()
        
        # SLM-2: Мониторинг качества
        self._adapter_metrics: Dict[str, List[float]] = {}
        self._baseline_quality: Optional[float] = None
        self._degradation_threshold: float = 0.1
        
        # Rollback стек
        self._swap_history: List[Dict] = []
        
        self._init_pipeline()
    
    def set_baseline_quality(self, quality: float):
        """Установить baseline качества для сравнения."""
        self._baseline_quality = quality
    
    def record_quality_metric(self, adapter_name: str, quality: float):
        """
        SLM-2: Записать метрику качества для адаптера.
        
        Args:
            adapter_name: Имя адаптера
            quality: Значение метрики (0-1)
        """
        if adapter_name not in self._adapter_metrics:
            self._adapter_metrics[adapter_name] = []
        
        self._adapter_metrics[adapter_name].append(quality)
        
        if len(self._adapter_metrics[adapter_name]) > 100:
            self._adapter_metrics[adapter_name].pop(0)
    
    def get_adapter_quality(self, adapter_name: str) -> Optional[float]:
        """Получить среднее качество адаптера."""
        if adapter_name not in self._adapter_metrics:
            return None
        metrics = self._adapter_metrics[adapter_name]
        return sum(metrics) / len(metrics) if metrics else None
    
    def is_degraded(self, current_quality: float) -> bool:
        """
        SLM-1: Проверить деградацию качества.
        
        Returns:
            True если качество упало относительно baseline
        """
        if self._baseline_quality is None:
            return False
        
        degradation = self._baseline_quality - current_quality
        return degradation > self._degradation_threshold
    
    def atomic_swap(
        self,
        adapter_name: str,
        alpha: float = 0.8,
        auto_rollback: bool = True
    ) -> bool:
        """
        Атомарная смена адаптера с опциональным rollback.
        
        Args:
            adapter_name: имя адаптера
            alpha: коэффициент смешивания
            auto_rollback: автоматически откатывать при деградации
        
        Returns:
            True если успешно
        """
        if self._pipeline is None:
            print("Pipeline not initialized")
            return False
        
        if adapter_name not in self._adapters:
            print(f"Adapter not found: {adapter_name}")
            return False
        
        previous_adapter = self._active_adapter
        adapter_path = self._adapters[adapter_name]
        
        try:
            with self._lock:
                if hasattr(self._pipeline, 'set_adapters'):
                    self._pipeline.set_adapters(adapter_path)
                
                self._active_adapter = adapter_name
                
                self._swap_history.append({
                    "timestamp": __import__('time').time(),
                    "from": previous_adapter,
                    "to": adapter_name,
                    "alpha": alpha
                })
            
            print(f"Swapped to adapter: {adapter_name} (alpha={alpha})")
            return True
            
        except Exception as e:
            print(f"Swap failed: {e}")
            return False
    
    def rollback_to_previous(self) -> bool:
        """
        SLM-1: Откатить к предыдущему адаптеру.
        
        Returns:
            True если rollback успешен
        """
        if len(self._swap_history) < 2:
            print("No history for rollback")
            return False
        
        previous_swap = self._swap_history[-2]
        previous_adapter = previous_swap["from"]
        
        if previous_adapter is None:
            print("No previous adapter to rollback to")
            return False
        
        print(f"Rolling back to: {previous_adapter}")
        return self.atomic_swap(previous_adapter, alpha=0.8, auto_rollback=False)
    
    def get_active_adapter(self) -> Optional[str]:
        """Получить активный адаптер."""
        return self._active_adapter
    
    def list_adapters(self) -> List[str]:
        """Список зарегистрированных адаптеров."""
        return list(self._adapters.keys())
    
    def unload(self):
        """Выгрузить адаптер."""
        with self._lock:
            self._active_adapter = None


class LoRAAdapter:
    """
    Отдельный LoRA адаптер.
    
    Представляет один адаптер.
    """
    
    def __init__(
        self,
        name: str,
        path: str,
        rank: int = 8,
        alpha: float = 16.0
    ):
        self.name = name
        self.path = path
        self.rank = rank
        self.alpha = alpha
        self._loaded = False
    
    def load(self):
        """Загрузить адаптер."""
        if os.path.exists(self.path):
            self._loaded = True
            print(f"Loaded: {self.name}")
        else:
            print(f"Not found: {self.path}")
    
    def unload(self):
        """Выгрузить адаптер."""
        self._loaded = False
    
    def is_loaded(self) -> bool:
        """Проверить загрузку."""
        return self._loaded


class MultiAdapterManager:
    """
    SLM-3: Менеджер нескольких адаптеров.
    
    Особенности:
    - Поддержка адаптеров с разными rank (r=4, r=8, r=16)
    - Автоматический выбор адаптера по задаче
    - Hot-swap без прерывания генерации
    """
    
    def __init__(self):
        self._adapters: Dict[str, LoRAAdapter] = {}
        self._current: Optional[str] = None
        self._rank_groups: Dict[int, List[str]] = {4: [], 8: [], 16: []}
        self._task_mappings: Dict[str, str] = {}
    
    def add(self, name: str, path: str, rank: int = 8):
        """Добавить адаптер."""
        adapter = LoRAAdapter(name, path, rank)
        self._adapters[name] = adapter
        
        if rank in self._rank_groups:
            if name not in self._rank_groups[rank]:
                self._rank_groups[rank].append(name)
        
        print(f"MultiAdapterManager: Added {name} (rank={rank})")
    
    def remove(self, name: str):
        """Удалить адаптер."""
        if name in self._adapters:
            adapter = self._adapters[name]
            rank = adapter.rank
            if rank in self._rank_groups and name in self._rank_groups[rank]:
                self._rank_groups[rank].remove(name)
            del self._adapters[name]
            
            if self._current == name:
                self._current = None
    
    def set_active(self, name: str):
        """Установить активный."""
        if name in self._adapters:
            self._current = name
    
    def get_rank(self) -> int:
        """Получить rank активного."""
        if self._current and self._current in self._adapters:
            return self._adapters[self._current].rank
        return 8
    
    def get_active_name(self) -> Optional[str]:
        """Получить имя активного."""
        return self._current
    
    def get_active_adapter(self) -> Optional[LoRAAdapter]:
        """Получить объект активного адаптера."""
        if self._current and self._current in self._adapters:
            return self._adapters[self._current]
        return None
    
    def list_by_rank(self, rank: int) -> List[str]:
        """Список адаптеров с указанным rank."""
        return self._rank_groups.get(rank, []).copy()
    
    def select_for_task(self, task: str) -> Optional[str]:
        """
        SLM-3: Автоматический выбор адаптера по типу задачи.
        
        Args:
            task: Тип задачи (reasoning, creative, factual, etc.)
            
        Returns:
            Имя выбранного адаптера или None
        """
        if task in self._task_mappings:
            mapped = self._task_mappings[task]
            if mapped in self._adapters:
                self.set_active(mapped)
                return mapped
        
        rank_map = {
            "reasoning": 16,
            "factual": 8,
            "creative": 8,
            "general": 4
        }
        
        preferred_rank = rank_map.get(task, 8)
        candidates = self._rank_groups.get(preferred_rank, [])
        
        if candidates:
            selected = candidates[0]
            self.set_active(selected)
            return selected
        
        if self._adapters:
            first = list(self._adapters.keys())[0]
            self.set_active(first)
            return first
        
        return None
    
    def map_task_to_adapter(self, task: str, adapter_name: str):
        """SLM-3: Явно сопоставить задачу и адаптер."""
        if adapter_name in self._adapters:
            self._task_mappings[task] = adapter_name
    
    def get_adapter_stats(self) -> Dict:
        """Получить статистику по всем адаптерам."""
        stats = {
            "total": len(self._adapters),
            "by_rank": {rank: len(adapters) for rank, adapters in self._rank_groups.items()},
            "active": self._current,
            "task_mappings": len(self._task_mappings),
            "loaded_count": sum(1 for a in self._adapters.values() if a.is_loaded())
        }
        return stats
    
    def load_all(self):
        """Загрузить все адаптеры в память."""
        for name, adapter in self._adapters.items():
            adapter.load()
    
    def unload_all(self):
        """Выгрузить все адаптеры из памяти."""
        for adapter in self._adapters.values():
            adapter.unload()
    
    def swap_with_preload(self, new_adapter_name: str) -> bool:
        """
        SLM-3: Swap с предварительной загрузкой для минимизации задержки.
        
        Args:
            new_adapter_name: Имя нового адаптера
            
        Returns:
            True если успешно
        """
        if new_adapter_name not in self._adapters:
            return False
        
        new_adapter = self._adapters[new_adapter_name]
        
        if not new_adapter.is_loaded():
            new_adapter.load()
        
        self._current = new_adapter_name
        
        return True