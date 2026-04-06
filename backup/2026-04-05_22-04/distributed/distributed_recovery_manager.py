"""Модуль управления восстановлением для ЕВА - восстановление системы после сбоев"""
import os
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import threading
from .cluster_manager import ClusterNode

logger = logging.getLogger("eva.distributed.recovery")

class RecoveryManager:
    """Менеджер восстановления системы после сбоев"""
    
    CHECKPOINT_DIR = "checkpoints"
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None):
        """
        Инициализирует менеджер восстановления.
        
        Args:
            brain: Ссылка на ядро ЕВА
            cache_dir: Путь к директории кэша
        """
        self.brain = brain
        self.cache_dir = cache_dir
        self.initialized = False
        self.running = False
        self.stop_event = threading.Event()
        
        # Создаем директорию кэша если нужно
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.checkpoint_dir = os.path.join(self.cache_dir, self.CHECKPOINT_DIR)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        else:
            self.checkpoint_dir = self.CHECKPOINT_DIR
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Регистры
        self.checkpoints = {}
        self.recovery_history = []
        
        # Загружаем существующие контрольные точки
        self._load_checkpoints()
        
        self.initialized = True
        logger.info(f"RecoveryManager инициализирован. Директория контрольных точек: {self.checkpoint_dir}")
    
    def start(self):
        """Запускает фоновые процессы менеджера восстановления."""
        if not self.initialized:
            logger.error("Невозможно запустить неинициализированный RecoveryManager")
            return False
        
        self.running = True
        self.stop_event.clear()
        logger.info("RecoveryManager запущен")
        return True
    
    def stop(self):
        """Останавливает фоновые процессы менеджера восстановления."""
        self.running = False
        self.stop_event.set()
        logger.info("RecoveryManager остановлен")
    
    def _load_checkpoints(self):
        """Загружает существующие контрольные точки"""
        if not os.path.exists(self.checkpoint_dir):
            logger.info(f"Директория контрольных точек не существует: {self.checkpoint_dir}")
            return
            
        try:
            for filename in os.listdir(self.checkpoint_dir):
                if filename.endswith('.json'):
                    checkpoint_id = filename[:-5]  # Убираем .json
                    self.checkpoints[checkpoint_id] = os.path.join(self.checkpoint_dir, filename)
            logger.info(f"Загружено {len(self.checkpoints)} контрольных точек")
        except Exception as e:
            logger.error(f"Ошибка загрузки контрольных точек: {e}")
    
    def create_checkpoint(self, checkpoint_id: str = None) -> Optional[str]:
        """Создает контрольную точку состояния системы"""
        if not self.initialized:
            logger.error("Невозможно создать контрольную точку: RecoveryManager не инициализирован")
            return None
            
        if not checkpoint_id:
            checkpoint_id = f"cp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        try:
            # Проверяем доступность необходимых компонентов
            if not self.brain or not hasattr(self.brain, 'memory_manager') or not self.brain.memory_manager:
                logger.error("MemoryManager недоступен для создания контрольной точки")
                return None
                
            if not hasattr(self.brain, 'cluster_manager') or not self.brain.cluster_manager:
                logger.error("ClusterManager недоступен для создания контрольной точки")
                return None
            
            # Собираем критически важные данные для восстановления
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "memory_state": {
                    "working_memory": self._get_working_memory_state(),
                    "semantic_memory": self._get_semantic_memory_state(),
                    "episodic_memory": self._get_episodic_memory_state(),
                    "user_profiles": self._get_user_profiles_state()
                },
                "cluster_state": self._get_cluster_state(),
                "system_status": self._get_system_status()
            }
            
            # Сохраняем контрольную точку
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                
            self.checkpoints[checkpoint_id] = checkpoint_path
            logger.info(f"Создана контрольная точка: {checkpoint_id}")
            
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Ошибка создания контрольной точки: {str(e)}", exc_info=True)
            return None
    
    def _get_working_memory_state(self) -> List[Dict[str, Any]]:
        """Получает состояние рабочей памяти."""
        if not self.brain or not self.brain.memory_manager:
            return []
        return self.brain.memory_manager.working_memory
    
    def _get_semantic_memory_state(self) -> List[Dict[str, Any]]:
        """Получает состояние семантической памяти."""
        if not self.brain or not self.brain.memory_manager:
            return []
        return self.brain.memory_manager.semantic_memory
    
    def _get_episodic_memory_state(self) -> List[Dict[str, Any]]:
        """Получает состояние эпизодической памяти."""
        if not self.brain or not self.brain.memory_manager:
            return []
        return self.brain.memory_manager.episodic_memory
    
    def _get_user_profiles_state(self) -> Dict[str, Any]:
        """Получает состояние профилей пользователей."""
        if not self.brain or not self.brain.memory_manager:
            return {}
        return self.brain.memory_manager.user_profiles
    
    def _get_cluster_state(self) -> Dict[str, Any]:
        """Получает состояние кластера."""
        if not self.brain or not self.brain.cluster_manager:
            return {"nodes": [], "status": "unknown"}
        return {
            "nodes": [node.to_dict() for node in self.brain.cluster_manager.nodes.values()],
            "status": self.brain.cluster_manager.get_cluster_status()
        }
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Получает статус системы."""
        if not self.brain:
            return {"timestamp": time.time()}
        
        status = {
            "timestamp": time.time(),
            "memory_status": self.brain.memory_manager.get_memory_status() if hasattr(self.brain, 'memory_manager') else {},
            "cluster_status": self.brain.cluster_manager.get_cluster_status() if hasattr(self.brain, 'cluster_manager') else {},
            "ethics_status": self.brain.ethics_framework.get_system_status() if hasattr(self.brain, 'ethics_framework') else {}
        }
        
        return status
    
    def restore_from_checkpoint(self, checkpoint_id: str) -> bool:
        """Восстанавливает систему из контрольной точки"""
        if not self.initialized:
            logger.error("Невозможно восстановить систему: RecoveryManager не инициализирован")
            return False
            
        if checkpoint_id not in self.checkpoints:
            logger.error(f"Контрольная точка не найдена: {checkpoint_id}")
            return False
            
        try:
            # Загружаем данные контрольной точки
            with open(self.checkpoints[checkpoint_id], 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                
            # Проверяем доступность необходимых компонентов
            if not self.brain:
                logger.error("Ядро системы недоступно для восстановления")
                return False
                
            if not hasattr(self.brain, 'memory_manager') or not self.brain.memory_manager:
                logger.error("MemoryManager недоступен для восстановления")
                return False
                
            if not hasattr(self.brain, 'cluster_manager') or not self.brain.cluster_manager:
                logger.error("ClusterManager недоступен для восстановления")
                return False
            
            # Восстанавливаем состояние памяти
            self._restore_memory_state(checkpoint_data["memory_state"])
            
            # Восстанавливаем состояние кластера
            self._restore_cluster_state(checkpoint_data["cluster_state"])
            
            logger.info(f"Система восстановлена из контрольной точки: {checkpoint_id}")
            self._add_to_recovery_history(checkpoint_id, "success")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка восстановления из контрольной точки {checkpoint_id}: {str(e)}", exc_info=True)
            self._add_to_recovery_history(checkpoint_id, "failure", str(e))
            return False
    
    def _restore_memory_state(self, memory_state: Dict[str, Any]):
        """Восстанавливает состояние памяти из контрольной точки."""
        # Восстанавливаем рабочую память
        if "working_memory" in memory_state and isinstance(memory_state["working_memory"], list):
            with self.brain.memory_manager.memory_locks["working"]:
                self.brain.memory_manager.working_memory = memory_state["working_memory"]
        
        # Восстанавливаем семантическую память
        if "semantic_memory" in memory_state and isinstance(memory_state["semantic_memory"], list):
            with self.brain.memory_manager.memory_locks["semantic"]:
                self.brain.memory_manager.semantic_memory = memory_state["semantic_memory"]
        
        # Восстанавливаем эпизодическую память
        if "episodic_memory" in memory_state and isinstance(memory_state["episodic_memory"], list):
            with self.brain.memory_manager.memory_locks["episodic"]:
                self.brain.memory_manager.episodic_memory = memory_state["episodic_memory"]
        
        # Восстанавливаем профили пользователей
        if "user_profiles" in memory_state and isinstance(memory_state["user_profiles"], dict):
            with self.brain.memory_manager.memory_locks["user_profiles"]:
                self.brain.memory_manager.user_profiles = memory_state["user_profiles"]
    
    def _restore_cluster_state(self, cluster_state: Dict[str, Any]):
        """Восстанавливает состояние кластера из контрольной точки."""
        if not self.brain.cluster_manager or "nodes" not in cluster_state:
            return
            
        # Очищаем текущие узлы
        self.brain.cluster_manager.nodes.clear()
        
        # Восстанавливаем узлы
        for node_data in cluster_state["nodes"]:
            try:
                node = ClusterNode.from_dict(node_data)
                self.brain.cluster_manager.nodes[node.node_id] = node
            except Exception as e:
                logger.error(f"Ошибка восстановления узла: {e}")
    
    def _add_to_recovery_history(self, checkpoint_id: str, status: str, error: Optional[str] = None):
        """Добавляет запись в историю восстановлений."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "from_checkpoint": checkpoint_id,
            "status": status
        }
        if error:
            entry["error"] = error
            
        self.recovery_history.append(entry)
        # Ограничиваем историю последними 100 записями
        if len(self.recovery_history) > 100:
            self.recovery_history = self.recovery_history[-100:]
    
    def auto_recovery(self) -> bool:
        """Автоматическое восстановление после сбоя"""
        if not self.initialized:
            logger.error("Невозможно выполнить автоматическое восстановление: RecoveryManager не инициализирован")
            return False
            
        # Находим самую свежую контрольную точку
        if not self.checkpoints:
            logger.error("Нет доступных контрольных точек для восстановления")
            return False
            
        # Сортируем контрольные точки по времени
        sorted_checkpoints = sorted(
            self.checkpoints.keys(),
            key=lambda x: datetime.strptime(x[3:], "%Y%m%d_%H%M%S") if x.startswith("cp_") else datetime.min,
            reverse=True
        )
        
        latest_checkpoint = sorted_checkpoints[0]
        return self.restore_from_checkpoint(latest_checkpoint)
    
    def get_recovery_history(self) -> List[Dict[str, Any]]:
        """Возвращает историю восстановлений"""
        return self.recovery_history
    
    def get_system_health(self) -> Dict[str, Any]:
        """Возвращает отчет о здоровье системы восстановления."""
        try:
            # Рассчитываем общий показатель здоровья
            health_score = 100.0
            
            # Учитываем количество доступных контрольных точек
            checkpoint_count = len(self.checkpoints)
            if checkpoint_count < 1:
                health_score -= 30
            elif checkpoint_count < 5:
                health_score -= 15
                
            # Учитываем последнее успешное восстановление
            if self.recovery_history:
                last_recovery = self.recovery_history[-1]
                if last_recovery["status"] == "failure":
                    health_score -= 20
                elif (datetime.now() - datetime.fromisoformat(last_recovery["timestamp"])).total_seconds() > 86400:
                    # Последнее восстановление было более 24 часов назад
                    health_score -= 10
            
            # Формируем список проблем
            problem_areas = []
            if checkpoint_count < 1:
                problem_areas.append("Отсутствуют контрольные точки")
            elif checkpoint_count < 5:
                problem_areas.append("Недостаточно контрольных точек")
                
            if self.recovery_history and self.recovery_history[-1]["status"] == "failure":
                problem_areas.append("Последнее восстановление завершилось неудачей")
                
            # Формируем рекомендации
            recommendations = []
            if checkpoint_count < 1:
                recommendations.append("Настройте регулярное создание контрольных точек")
            elif checkpoint_count < 5:
                recommendations.append("Увеличьте частоту создания контрольных точек")
                
            if not self.recovery_history:
                recommendations.append("Выполните тестовое восстановление для проверки работоспособности")
                
            return {
                "health_score": max(0, min(100, health_score)),
                "checkpoint_count": checkpoint_count,
                "last_recovery": self.recovery_history[-1] if self.recovery_history else None,
                "problem_areas": problem_areas,
                "recommendations": recommendations,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Ошибка получения информации о здоровье системы восстановления: {e}")
            return {
                "health_score": 0,
                "error": str(e),
                "timestamp": time.time()
            }