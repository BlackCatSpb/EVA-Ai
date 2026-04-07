"""Основной модуль машинного обучения для ЕВА - ядро системы"""
import os
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger("eva_ai.ml_core")

@dataclass
class ModelHealth:
    """Хранит информацию о состоянии модели."""
    
    def __init__(self, model_name: str, model_type: str = "unknown", 
                 status: str = "operational", health_score: float = 1.0,
                 usage_count: int = 0, error_count: int = 0, 
                 response_time: float = 0.0, last_used: float = 0.0,
                 memory_usage: float = 0.0, metadata: Dict[str, Any] = None):
        """
        Инициализирует состояние модели.
        
        Args:
            model_name: Имя модели
            model_type: Тип модели (text-generation, summarization, и т.д.)
            status: Статус модели (operational, degraded, critical)
            health_score: Оценка здоровья модели (0.0-1.0)
            usage_count: Количество использований
            error_count: Количество ошибок
            response_time: Среднее время ответа
            last_used: Временная метка последнего использования
            memory_usage: Использование памяти
            metadata: Дополнительные метаданные
        """
        self.model_name = model_name
        self.model_type = model_type
        self.status = status
        self.health_score = health_score
        self.usage_count = usage_count
        self.error_count = error_count
        self.response_time = response_time
        self.last_used = last_used or time.time()
        self.memory_usage = memory_usage
        self.metadata = metadata or {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelHealth':
        """Создает состояние модели из словаря."""
        return cls(
            model_name=data["model_name"],
            health_score=data["health_score"],
            last_used=data["last_used"],
            usage_count=data["usage_count"],
            error_count=data["error_count"],
            memory_usage=data["memory_usage"],
            response_time=data["response_time"],
            status=data["status"],
            metadata=data.get("metadata", {})
        )

class MLCore:
    """Основной интерфейс модуля машинного обучения для ЕВА."""
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None, hybrid_cache=None, token_streamer=None):

        """
        Инициализирует ядро модуля машинного обучения.
        
        Args:
            brain: Ссылка на ядро ЕВА (опционально)
            cache_dir: Путь к директории кэша (опционально)
        """
        self.brain = brain
        self.hybrid_cache = hybrid_cache
        self.token_streamer = token_streamer
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "eva_ml_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Статистика
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_processed": 0,
            "avg_response_time": 0.0,
            "last_request": 0,
            "model_switches": 0,
            "cache_hits": 0
        }
        
        # Параметры оптимизации
        self.optimization_params = {
            "quantization": False,
            "pruning": False,
            "cache_enabled": True,
            "model_rotation": True,
            "fallback_models": True
        }
        
        # Кэш результатов
        self.response_cache = {}
        self.cache_size = 1000
        self.cache_ttl = 3600  # 1 час
        
        # Блокировка ресурсов
        self.lock = threading.Lock()
        
        logger.info("MLCore инициализирован")
    
    def update_stats(self, success: bool, response_time: float, tokens_processed: int = 0):
        """Обновляет статистику работы модуля."""
        with self.lock:
            self.stats["total_requests"] += 1
            if success:
                self.stats["successful_requests"] += 1
                self.stats["total_tokens_processed"] += tokens_processed
            else:
                self.stats["failed_requests"] += 1
            
            # Обновляем среднее время ответа
            if self.stats["total_requests"] > 1:
                self.stats["avg_response_time"] = (
                    (self.stats["avg_response_time"] * (self.stats["total_requests"] - 1) + response_time) 
                    / self.stats["total_requests"]
                )
            else:
                self.stats["avg_response_time"] = response_time
            
            self.stats["last_request"] = time.time()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает текущую статистику модуля."""
        return self.stats.copy()
    
    def enable_cache(self, enable: bool = True):
        """Включает или выключает кэширование результатов."""
        self.optimization_params["cache_enabled"] = enable
        logger.info(f"Кэширование {'включено' if enable else 'выключено'}")
    
    def set_cache_size(self, size: int):
        """Устанавливает размер кэша."""
        self.cache_size = max(100, size)
        logger.info(f"Размер кэша установлен на {self.cache_size} записей")
    
    def set_cache_ttl(self, ttl: int):
        """Устанавливает время жизни кэша в секундах."""
        self.cache_ttl = max(300, ttl)  # минимум 5 минут
        logger.info(f"Время жизни кэша установлено на {self.cache_ttl} секунд")
    
    def cleanup_cache(self):
        """Очищает устаревший кэш."""
        current_time = time.time()
        old_entries = [key for key, value in self.response_cache.items() 
                      if current_time - value["timestamp"] > self.cache_ttl]
        
        for key in old_entries:
            del self.response_cache[key]
        
        # Если кэш все еще слишком большой, удаляем наименее используемые записи
        if len(self.response_cache) > self.cache_size:
            sorted_cache = sorted(self.response_cache.items(), 
                               key=lambda x: x[1]["usage_count"])
            for key, _ in sorted_cache[:-self.cache_size//2]:
                del self.response_cache[key]
    
    def get_cached_response(self, cache_key: str) -> Optional[str]:
        """Получает ответ из кэша, если он актуален."""
        if not self.optimization_params["cache_enabled"]:
            return None
            
        if cache_key in self.response_cache:
            cache_entry = self.response_cache[cache_key]
            if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                cache_entry["usage_count"] += 1
                self.stats["cache_hits"] += 1
                logger.debug(f"Использован кэш для ключа: {cache_key[:30]}...")
                return cache_entry["response"]
        
        return None
    
    def cache_response(self, cache_key: str, response: str, model_name: str):
        """Кэширует ответ."""
        if not self.optimization_params["cache_enabled"]:
            return
            
        self.response_cache[cache_key] = {
            "response": response,
            "timestamp": time.time(),
            "usage_count": 1,
            "model": model_name
        }
        
        # Ограничиваем размер кэша
        if len(self.response_cache) > self.cache_size:
            # Удаляем наименее используемые записи
            sorted_cache = sorted(self.response_cache.items(), 
                               key=lambda x: x[1]["usage_count"])
            for key, _ in sorted_cache[:-self.cache_size//2]:
                del self.response_cache[key]
    
    def get_model_health_report(self) -> Dict[str, Any]:
        """Возвращает отчет о здоровье моделей."""
        try:
            # Здесь будут реальные данные от ModelManager
            model_health = {}
            if self.brain and hasattr(self.brain, 'model_manager'):
                model_health = self.brain.model_manager.get_model_health()
            
            report = {
                "status": "unknown",
                "health_score": 0.0,
                "models": model_health,
                "statistics": {
                    "total_models": len(model_health),
                    "healthy_models": 0,
                    "warning_models": 0,
                    "critical_models": 0,
                    "total_requests": self.stats["total_requests"],
                    "successful_requests": self.stats["successful_requests"],
                    "failed_requests": self.stats["failed_requests"],
                    "avg_response_time": self.stats["avg_response_time"],
                    "cache_hits": self.stats["cache_hits"]
                },
                "timestamp": time.time()
            }
            
            # Анализируем состояние моделей
            model_scores = []
            for model_name, health in model_health.items():
                if health["status"] == "healthy":
                    report["statistics"]["healthy_models"] += 1
                elif health["status"] == "warning":
                    report["statistics"]["warning_models"] += 1
                else:
                    report["statistics"]["critical_models"] += 1
                
                model_scores.append(health["health_score"])
            
            # Рассчитываем общий показатель здоровья
            if model_scores:
                report["health_score"] = sum(model_scores) / len(model_scores)
            else:
                report["health_score"] = 50.0  # Значение по умолчанию
            
            # Определяем статус
            if report["health_score"] > 0.7:
                report["status"] = "healthy"
            elif report["health_score"] > 0.4:
                report["status"] = "warning"
            else:
                report["status"] = "critical"
            
            return report
            
        except Exception as e:
            logger.error(f"Ошибка формирования отчета о здоровье моделей: {e}")
            return {
                "status": "error",
                "health_score": 0.0,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Возвращает отчет о здоровье MLUnit."""
        return self.get_model_health_report()
    
    def get_ml_health_dashboard_data(self) -> Dict[str, Any]:
        """Возвращает данные для дашборда здоровья MLUnit."""
        model_health = self.get_model_health_report()
        
        return {
            "model_health": model_health,
            "statistics": self.stats,
            "timestamp": time.time()
        }
    
    def close(self):
        """Закрывает MLCore и освобождает ресурсы."""
        logger.info("Закрытие MLCore...")
        
        # Очищаем кэш
        self.response_cache = {}
        
        logger.info("MLCore закрыт")
    
    def generate_system_report(self) -> str:
        """Генерирует текстовый отчет о состоянии MLCore."""
        health = self.get_model_health_report()
        stats = self.stats
        
        report = "ОТЧЕТ О СОСТОЯНИИ ML CORE\n"
        report += "=" * 50 + "\n\n"
        
        # Общая оценка
        report += f"ОБЩАЯ ОЦЕНКА: {health['health_score']:.1f}/100\n"
        if health['health_score'] > 80:
            report += "Состояние MLCore: Отличное\n"
        elif health['health_score'] > 60:
            report += "Состояние MLCore: Хорошее\n"
        elif health['health_score'] > 40:
            report += "Состояние MLCore: Удовлетворительное\n"
        else:
            report += "Состояние MLCore: Требует внимания\n"
        
        report += "\n"
        
        # Статистика
        report += "СТАТИСТИКА ML CORE:\n"
        report += f"- Всего запросов: {stats['total_requests']}\n"
        report += f"- Успешных запросов: {stats['successful_requests']}\n"
        report += f"- Неудачных запросов: {stats['failed_requests']}\n"
        report += f"- Обработано токенов: {stats['total_tokens_processed']}\n"
        report += f"- Среднее время ответа: {stats['avg_response_time']:.2f} сек\n"
        report += f"- Попаданий в кэш: {stats['cache_hits']}\n"
        
        return report