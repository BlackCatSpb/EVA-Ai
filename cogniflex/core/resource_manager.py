"""
Менеджер ресурсов для CogniFlex
"""

import os
import psutil
import threading
import time
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class ResourceManager:
    """Управляет системными ресурсами CogniFlex."""
    
    def __init__(self, config_manager=None):
        """Инициализирует менеджер ресурсов.
        
        Args:
            config_manager: Менеджер конфигурации
        """
        self.config = config_manager
        self.monitoring_active = False
        self.monitoring_thread = None
        self.resource_lock = threading.RLock()
        
        # Текущие метрики ресурсов
        self.current_metrics = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "memory_available_gb": 0.0,
            "disk_usage_percent": 0.0,
            "disk_free_gb": 0.0,
            "gpu_usage": 0.0,
            "gpu_memory": 0.0
        }
        
        # Пороги предупреждений (увеличены для уменьшения частоты уведомлений)
        self.warning_thresholds = {
            "cpu_percent": 90.0,
            "memory_percent": 95.0,  # Увеличен с 85% до 95%
            "disk_usage_percent": 95.0,
            "gpu_memory": 95.0
        }
        
        # Критические пороги
        self.critical_thresholds = {
            "cpu_percent": 98.0,
            "memory_percent": 98.0,  # Увеличен с 95% до 98%
            "disk_usage_percent": 98.0,
            "gpu_memory": 98.0
        }
        
        # Счетчик для дебаунса уведомлений
        self.last_warning_time = {}
        self.warning_cooldown = 300  # 5 минут между одинаковыми предупреждениями
        
        # Счетчики использования
        self.resource_usage_history = []
        self.max_history_size = 5000
        
        logger.info("ResourceManager инициализирован")
    
    def start_monitoring(self, interval: float = 5.0):
        """Запускает мониторинг ресурсов.
        
        Args:
            interval: Интервал мониторинга в секундах
        """
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Мониторинг ресурсов запущен с интервалом {interval}с")
    
    def stop_monitoring(self):
        """Останавливает мониторинг ресурсов."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Мониторинг ресурсов остановлен")
    
    def _monitoring_loop(self, interval: float):
        """Основной цикл мониторинга ресурсов."""
        while self.monitoring_active:
            try:
                self._update_metrics()
                self._check_thresholds()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Ошибка в мониторинге ресурсов: {e}")
                time.sleep(interval * 2)  # Увеличиваем интервал при ошибке
    
    def _update_metrics(self):
        """Обновляет метрики ресурсов."""
        try:
            with self.resource_lock:
                # CPU
                self.current_metrics["cpu_percent"] = psutil.cpu_percent(interval=1)
                
                # Память
                memory = psutil.virtual_memory()
                self.current_metrics["memory_percent"] = memory.percent
                self.current_metrics["memory_available_gb"] = memory.available / (1024**3)
                
                # Диск
                disk = psutil.disk_usage('/')
                self.current_metrics["disk_usage_percent"] = disk.percent
                self.current_metrics["disk_free_gb"] = disk.free / (1024**3)
                
                # GPU (если доступен)
                self._update_gpu_metrics()
                
                # Добавляем в историю
                self._add_to_history()
                
        except Exception as e:
            logger.error(f"Ошибка обновления метрик: {e}")
    
    def _update_gpu_metrics(self):
        """Обновляет метрики GPU."""
        try:
            import torch
        except ImportError:
            self.current_metrics["gpu_usage"] = 0.0
            self.current_metrics["gpu_memory"] = 0.0
            return
        
        try:
            if torch.cuda.is_available():
                # Использование GPU памяти
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                self.current_metrics["gpu_memory"] = (gpu_memory_allocated / gpu_memory_total) * 100
                self.current_metrics["gpu_usage"] = 0.0
            else:
                self.current_metrics["gpu_usage"] = 0.0
                self.current_metrics["gpu_memory"] = 0.0
        except ImportError:
            # PyTorch не установлен
            self.current_metrics["gpu_usage"] = 0.0
            self.current_metrics["gpu_memory"] = 0.0
        except Exception as e:
            logger.debug(f"Ошибка получения GPU метрик: {e}")
    
    def _add_to_history(self):
        """Добавляет текущие метрики в историю."""
        history_entry = {
            "timestamp": time.time(),
            "metrics": self.current_metrics.copy()
        }
        
        self.resource_usage_history.append(history_entry)
        
        # Ограничиваем размер истории
        if len(self.resource_usage_history) > self.max_history_size:
            self.resource_usage_history = self.resource_usage_history[-self.max_history_size:]
    
    def _check_thresholds(self):
        """Проверяет пороги использования ресурсов с дебаунсом."""
        current_time = time.time()
        
        for metric, value in self.current_metrics.items():
            if metric in self.critical_thresholds:
                # Проверяем критический уровень
                if value > self.critical_thresholds[metric]:
                    warning_key = f"critical_{metric}"
                    if (warning_key not in self.last_warning_time or 
                        current_time - self.last_warning_time[warning_key] > self.warning_cooldown):
                        logger.critical(f"КРИТИЧЕСКИЙ уровень {metric}: {value:.1f}%")
                        self.last_warning_time[warning_key] = current_time
                
                # Проверяем уровень предупреждения
                elif value > self.warning_thresholds[metric]:
                    warning_key = f"warning_{metric}"
                    if (warning_key not in self.last_warning_time or 
                        current_time - self.last_warning_time[warning_key] > self.warning_cooldown):
                        logger.warning(f"Высокий уровень {metric}: {value:.1f}%")
                        self.last_warning_time[warning_key] = current_time
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Возвращает текущие метрики ресурсов.
        
        Returns:
            Словарь с текущими метриками
        """
        with self.resource_lock:
            return self.current_metrics.copy()

    def get_cpu_usage(self) -> float:
        """Возвращает загрузку CPU как долю от 0.0 до 1.0."""
        with self.resource_lock:
            try:
                return float(self.current_metrics.get("cpu_percent", 0.0)) / 100.0
            except Exception:
                return 0.0

    def get_memory_usage(self) -> float:
        """Возвращает загрузку RAM как долю от 0.0 до 1.0."""
        with self.resource_lock:
            try:
                return float(self.current_metrics.get("memory_percent", 0.0)) / 100.0
            except Exception:
                return 0.0
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Возвращает сводку по ресурсам.
        
        Returns:
            Сводка использования ресурсов
        """
        with self.resource_lock:
            metrics = self.current_metrics.copy()
            
            # Определяем общий статус
            status = "healthy"
            alerts = []
            
            for metric, value in metrics.items():
                if metric in self.critical_thresholds:
                    if value > self.critical_thresholds[metric]:
                        status = "critical"
                        alerts.append(f"{metric}: {value:.1f}% (критический)")
                    elif value > self.warning_thresholds[metric]:
                        if status != "critical":
                            status = "warning"
                        alerts.append(f"{metric}: {value:.1f}% (предупреждение)")
            
            return {
                "status": status,
                "metrics": metrics,
                "alerts": alerts,
                "monitoring_active": self.monitoring_active,
                "history_size": len(self.resource_usage_history)
            }
    
    def get_resource_history(self, last_minutes: int = 10) -> List[Dict[str, Any]]:
        """Возвращает историю использования ресурсов.
        
        Args:
            last_minutes: Количество минут истории
            
        Returns:
            Список записей истории
        """
        cutoff_time = time.time() - (last_minutes * 60)
        
        with self.resource_lock:
            return [
                entry for entry in self.resource_usage_history
                if entry["timestamp"] > cutoff_time
            ]
    
    def check_resource_availability(self, required_memory_gb: float = 1.0, required_disk_gb: float = 1.0) -> bool:
        """Проверяет доступность ресурсов для операции.
        
        Args:
            required_memory_gb: Требуемая память в ГБ
            required_disk_gb: Требуемое место на диске в ГБ
            
        Returns:
            True если ресурсы доступны
        """
        with self.resource_lock:
            memory_available = self.current_metrics["memory_available_gb"]
            disk_available = self.current_metrics["disk_free_gb"]
            
            if memory_available < required_memory_gb:
                logger.warning(f"Недостаточно памяти: требуется {required_memory_gb}ГБ, доступно {memory_available:.1f}ГБ")
                return False
            
            if disk_available < required_disk_gb:
                logger.warning(f"Недостаточно места на диске: требуется {required_disk_gb}ГБ, доступно {disk_available:.1f}ГБ")
                return False
            
            return True
    
    def optimize_resources(self):
        """Выполняет оптимизацию использования ресурсов."""
        try:
            import gc
            
            # Принудительная сборка мусора
            collected = gc.collect()
            logger.info(f"Сборка мусора: освобождено {collected} объектов")
            
            # Очистка кэшей PyTorch если доступен
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("Очищен кэш GPU")
            except ImportError:
                pass
            
            # Ограничиваем историю ресурсов
            with self.resource_lock:
                if len(self.resource_usage_history) > self.max_history_size:
                    self.resource_usage_history = self.resource_usage_history[-self.max_history_size//2:]
                    logger.info("Очищена история ресурсов")
            
        except Exception as e:
            logger.error(f"Ошибка оптимизации ресурсов: {e}")
    
    def set_thresholds(self, warning: Dict[str, float] = None, critical: Dict[str, float] = None):
        """Устанавливает пороги предупреждений и критических значений.
        
        Args:
            warning: Пороги предупреждений
            critical: Критические пороги
        """
        if warning:
            self.warning_thresholds.update(warning)
            logger.info(f"Обновлены пороги предупреждений: {warning}")
        
        if critical:
            self.critical_thresholds.update(critical)
            logger.info(f"Обновлены критические пороги: {critical}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Возвращает информацию о системе.
        
        Returns:
            Информация о системе
        """
        try:
            return {
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "disk_total_gb": psutil.disk_usage('/').total / (1024**3),
                "boot_time": psutil.boot_time(),
                "platform": os.name,
                "python_version": os.sys.version,
                "gpu_available": self._is_gpu_available()
            }
        except Exception as e:
            logger.error(f"Ошибка получения системной информации: {e}")
            return {}
    
    def _is_gpu_available(self) -> bool:
        """Проверяет доступность GPU."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
