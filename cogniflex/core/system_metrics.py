"""Модуль для управления системными метриками CogniFlex"""
import time
import json
from typing import Dict, Any, List, Optional, Tuple
 
# Пытаемся импортировать psutil для получения реальных системных метрик
try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # graceful fallback

class SystemMetricsManager:
    """Управляет сбором и обновлением системных метрик."""
    
    def __init__(self):
        """Инициализирует менеджер метрик."""
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.request_durations = []
        
        # Инициализация метрик
        self.metrics = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "active_tasks": 0,
            "contradictions": 0,
            "learning_progress": 0.0,
            "request_throughput": 0.0,
            "response_time": 0.0,
            "error_rate": 0.0,
            "uptime": 0.0
        }

        # Буфер для нормализованных метрик (универсальная схема)
        self._buffer: List[Dict[str, Any]] = []
        # Карантин для метрик с ошибками схемы
        self._quarantine: List[Dict[str, Any]] = []
        # Агрегаты по нормализованным метрикам
        # key: (name, component, type, frozenset(labels.items())) -> aggregate dict
        self._aggregates: Dict[Tuple[str, str, str, Tuple[Tuple[str, Any], ...]], Dict[str, Any]] = {}
    
    def start_tracking(self):
        """Начинает отслеживание метрик."""
        self.start_time = time.time()
        self._update_uptime()
    
    def update_request_metrics(self, duration: float, success: bool):
        """Обновляет метрики, связанные с обработкой запросов.
        
        Args:
            duration: Время обработки запроса
            success: Успешно ли обработан запрос
        """
        self.total_requests += 1
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.request_durations.append(duration)
        
        # Обновляем метрики
        self._update_request_metrics()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Возвращает текущие системные метрики.
        
        Returns:
            Dict[str, Any]: Словарь с метриками системы
        """
        self._update_uptime()
        # Обновляем метрики CPU/памяти в момент запроса, если доступен psutil
        try:
            if psutil is not None:
                # Значения приводим к долям (0..1) — интерфейс аналитики умножает их на 100
                self.metrics["cpu_usage"] = float(psutil.cpu_percent(interval=None)) / 100.0
                self.metrics["memory_usage"] = float(psutil.virtual_memory().percent) / 100.0
        except Exception:
            # Безопасно игнорируем любые ошибки чтения системных метрик
            pass
        return self.metrics.copy()

    # ========= Универсальные метрики (нормализованные) =========
    def emit(self, metric: Dict[str, Any]) -> bool:
        """Принимает одну нормализованную метрику и добавляет её в буфер.

        Ожидаемая схема (минимум):
        {
          "name": str,
          "component": str,
          "type": "counter"|"gauge"|"histogram"|"summary",
          "value": int|float,
          "timestamp": float (epoch seconds),
          "labels": {k: v} (опционально)
        }
        """
        try:
            m = self._normalize_metric(metric)
            if m is None:
                # Некорректная метрика — в карантин и false
                self._quarantine.append(metric)
                return False
            self._buffer.append(m)
            # Обновляем агрегаты на лету
            try:
                self._update_aggregates(m)
            except Exception:
                # Агрегация не должна ломать поток
                pass
            return True
        except Exception:
            # Никаких исключений наружу
            self._quarantine.append(metric)
            return False

    def emit_many(self, metrics: List[Dict[str, Any]]) -> int:
        """Принимает список метрик, нормализует и добавляет в буфер. Возвращает число принятых."""
        accepted = 0
        for m in metrics or []:
            if self.emit(m):
                accepted += 1
        return accepted

    def flush(self) -> List[Dict[str, Any]]:
        """Возвращает и очищает буфер метрик."""
        out, self._buffer = self._buffer, []
        return out

    def get_quarantine(self) -> List[Dict[str, Any]]:
        """Возвращает и очищает карантин некорректных метрик."""
        out, self._quarantine = self._quarantine, []
        return out

    def _normalize_metric(self, metric: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Проверяет/дополняет схему. Возвращает нормализованную метрику либо None при ошибке."""
        if not isinstance(metric, dict):
            return None
        name = metric.get("name")
        component = metric.get("component")
        mtype = metric.get("type")
        value = metric.get("value")
        if not name or not isinstance(name, str):
            return None
        if not component or not isinstance(component, str):
            return None
        if mtype not in ("counter", "gauge", "histogram", "summary"):
            return None
        if not isinstance(value, (int, float)):
            return None
        # Таймстамп по умолчанию
        ts = metric.get("timestamp")
        if not isinstance(ts, (int, float)):
            ts = time.time()
        # Опциональные поля
        labels = metric.get("labels")
        labels = labels if isinstance(labels, dict) else {}
        unit = metric.get("unit")
        unit = unit if isinstance(unit, str) else None
        subsystem = metric.get("subsystem")
        subsystem = subsystem if isinstance(subsystem, str) else None
        # Возвращаем нормализованную структуру
        norm = {
            "name": name,
            "component": component,
            "type": mtype,
            "value": float(value),
            "timestamp": float(ts),
            "labels": labels,
        }
        if unit:
            norm["unit"] = unit
        if subsystem:
            norm["subsystem"] = subsystem
        return norm

    def validate_metric_schema(self, metric: Dict[str, Any]) -> bool:
        """Публичная валидация схемы метрики без модификаций состояния.

        Возвращает True, если метрика соответствует ожидаемой схеме
        нормализованных метрик, иначе False. Метод не изменяет буферы,
        карантин и агрегаты — используется только для явной проверки
        перед эмиссией.
        """
        try:
            return self._normalize_metric(metric) is not None
        except Exception:
            return False

    def validate_many(self, metrics: List[Dict[str, Any]]) -> int:
        """Возвращает количество метрик, прошедших валидацию схемы.

        Полезно для предварительной проверки батчей метрик перед отправкой.
        """
        ok = 0
        for m in metrics or []:
            try:
                if self._normalize_metric(m) is not None:
                    ok += 1
            except Exception:
                # Игнорируем ошибки, считаем метрику невалидной
                pass
        return ok
    
    def _update_request_metrics(self):
        """Обновляет метрики, связанные с запросами."""
        # Вычисляем среднее время ответа
        if self.request_durations:
            self.metrics["response_time"] = sum(self.request_durations[-100:]) / min(len(self.request_durations), 100)
        
        # Вычисляем коэффициент ошибок
        if self.total_requests > 0:
            self.metrics["error_rate"] = self.failed_requests / self.total_requests
        
        # Вычисляем пропускную способность
        uptime = self.metrics["uptime"]
        if uptime > 0:
            self.metrics["request_throughput"] = self.total_requests / uptime
    
    def _update_uptime(self):
        """Обновляет время работы системы."""
        self.metrics["uptime"] = time.time() - self.start_time

    # ========= Аггрегация нормализованных метрик =========
    def _key(self, m: Dict[str, Any]) -> Tuple[str, str, str, Tuple[Tuple[str, Any], ...]]:
        labels = m.get("labels") or {}
        # Стабильный ключ по отсортированным лейблам
        return (
            m.get("name", ""),
            m.get("component", ""),
            m.get("type", ""),
            tuple(sorted(labels.items())),
        )

    def _update_aggregates(self, m: Dict[str, Any]) -> None:
        key = self._key(m)
        mt = m["type"]
        agg = self._aggregates.get(key)
        if agg is None:
            # Базовые структуры
            if mt == "counter":
                agg = {"value": 0.0}
            elif mt == "gauge":
                agg = {"value": 0.0}
            elif mt == "histogram":
                # Минимальная реализация: count, sum, min, max
                agg = {"count": 0, "sum": 0.0, "min": None, "max": None}
            elif mt == "summary":
                # count/sum для среднего, плюс min/max
                agg = {"count": 0, "sum": 0.0, "min": None, "max": None}
            else:
                agg = {"value": 0.0}
            # Постоянные поля для репорта
            agg["name"] = m["name"]
            agg["component"] = m["component"]
            agg["type"] = mt
            agg["labels"] = m.get("labels", {})
            self._aggregates[key] = agg

        v = float(m.get("value", 0.0))
        if mt == "counter":
            agg["value"] = float(agg.get("value", 0.0)) + v
        elif mt == "gauge":
            agg["value"] = v
        elif mt in ("histogram", "summary"):
            agg["count"] = int(agg.get("count", 0)) + 1
            agg["sum"] = float(agg.get("sum", 0.0)) + v
            if agg.get("min") is None or v < agg["min"]:
                agg["min"] = v
            if agg.get("max") is None or v > agg["max"]:
                agg["max"] = v

    def get_snapshot(self) -> Dict[str, Any]:
        """Снимок текущего состояния метрик и агрегатов.

        Returns:
            Dict[str, Any]: {"system": ..., "aggregates": [...], "buffer_size": int, "quarantine_size": int}
        """
        try:
            system = self.get_metrics()
        except Exception:
            system = {}
        aggs = list(self._aggregates.values())
        return {
            "system": system,
            "aggregates": aggs,
            "buffer_size": len(self._buffer),
            "quarantine_size": len(self._quarantine),
            "uptime": system.get("uptime", 0.0),
            "timestamp": time.time(),
        }

    def reset_aggregates(self) -> None:
        """Очищает агрегаты нормализованных метрик."""
        self._aggregates = {}

    # ========= Экспортеры =========
    def export_snapshot_to_stdout(self, include_buffer_sizes: bool = True) -> None:
        """Печатает снимок метрик в stdout в компактном JSON-формате."""
        snap = self.get_snapshot()
        if not include_buffer_sizes:
            snap.pop("buffer_size", None)
            snap.pop("quarantine_size", None)
        try:
            print("[METRICS_SNAPSHOT] " + json.dumps(snap, ensure_ascii=False))
        except Exception:
            # Фолбэк на безопасную печать
            print("[METRICS_SNAPSHOT] <unserializable>")

    def export_snapshot_to_json(self, file_path: str) -> bool:
        """Сохраняет снимок метрик в JSON-файл. Возвращает True при успехе."""
        try:
            snap = self.get_snapshot()
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(snap, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    def record_system_startup(self, total_time):
        # Логируем время старта
        self.metrics["startup_time"] = total_time
        print(f"[METRICS] Система запустилась за {total_time:.2f} сек")

    def record_error(self, error_code):
        # Логируем код ошибки
        if "errors" not in self.metrics:
            self.metrics["errors"] = []
        self.metrics["errors"].append(error_code)
        print(f"[METRICS] Ошибка: {error_code}")
    
    def record_warning(self, warning_code):
        """Регистрирует предупреждение для системных метрик.
        Совместимо с вызовами в CoreBrain.start() и других местах.
        """
        try:
            if "warnings" not in self.metrics:
                self.metrics["warnings"] = []
            self.metrics["warnings"].append(warning_code)
            print(f"[METRICS] Предупреждение: {warning_code}")
        except Exception:
            # Грейсфул-фолбэк, чтобы метрики не ломали основной поток
            pass
    
    def record_system_shutdown(self, total_time):
        # Логируем время остановки
        self.metrics["shutdown_time"] = total_time
        print(f"[METRICS] Система остановлена за {total_time:.2f} сек")
    
    def record_system_reboot(self, total_time):
        # Логируем время перезагрузки
        self.metrics["reboot_time"] = total_time
        print(f"[METRICS] Система перезагружена за {total_time:.2f} сек")
    
    def record_query_metrics(self, query_length, response_length, processing_time, tokens_processed):
        # Записываем метрики запроса
        self.update_request_metrics(processing_time, True)
        self.metrics["last_query_length"] = query_length
        self.metrics["last_response_length"] = response_length
        self.metrics["last_tokens_processed"] = tokens_processed
        print(f"[METRICS] Обработан запрос: {query_length} символов -> {response_length} символов за {processing_time:.2f} сек")

