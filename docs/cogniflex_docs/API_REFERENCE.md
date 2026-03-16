# CogniFlex — API Reference (актуализировано)

Актуальная выдержка публичных API по исходникам проекта. Пути и сигнатуры сверены по коду.

## Core

Файл: `cogniflex/core/core_brain.py`

- __Класс__: `CoreBrain`
- __Публичные методы (выдержка)__:
  - `initialize() -> bool`
  - `start() -> bool`
  - `stop(preserve_ml: bool = False) -> None`
  - `soft_reload(reload_gui: bool = False) -> bool`
  - `process_query(query: str, user_context: Optional[Dict] = None) -> Dict[str, Any]`
  - `get_status() -> Dict[str, Any]`
  - `get_system_health() -> Dict[str, Any]`
  - `get_metrics() -> Dict[str, Any]`
  - `get_resource_snapshot() -> Dict[str, Any]` — безопасная обёртка над `GlobalResourceQueue.snapshot()`, при недоступности возвращает `{}`
  - `get_cache_stats() -> Dict[str, Any]` — безопасная обёртка над `HybridTokenCache.get_cache_stats()`, при недоступности возвращает `{}`
  - `get_response_metadata(query: str) -> Dict[str, Any]`
  - `start_module(name: str) -> bool`
  - `stop_module(name: str) -> bool`
  - `get_module_status(name: str) -> Dict[str, Any]`
  - `get_available_models() -> List[Dict[str, Any]]`
  - `ensure_model_available(model_id: str, wait: bool = False, timeout_s: float = 0.0) -> Dict[str, Any]`
  - `add_deferred_command(command: callable, *args, **kwargs) -> None`
  - `nlp_enqueue(item: Dict[str, Any], module: str = "default") -> bool`
  
- __Интерфейс управления ресурсами (обёртка GRQ)__:
  - `request_memory(nbytes: int, timeout: Optional[float] = None) -> bool`
  - `release_memory(nbytes: int) -> None`
  - `request_cpu(n: int = 1, timeout: Optional[float] = None) -> bool`
  - `release_cpu(n: int = 1) -> None`

Примечание: в объекте `CoreBrain` ожидается опциональный атрибут `resource_queue` (экземпляр `GlobalResourceQueue`).

Также, при доступном `token_cache` (`HybridTokenCache`) метод `get_cache_stats()` предоставляет агрегированную телеметрию (RAM/диск, hit-rate, политика кэширования). Оба метода безопасны к ошибкам и возвращают пустой словарь при недоступности компонентов.

## GlobalResourceQueue (GRQ)

Файл: `cogniflex/core/global_resource_queue.py`

- __Класс__: `GlobalResourceQueue`
- __Конструктор__:
  - `GlobalResourceQueue(brain, max_memory_bytes: int, cpu_tokens: int, io_rate_bps: float, io_burst_factor: float = 1.5)`
- __RAM__:
  - `acquire_memory(nbytes: int, timeout: Optional[float] = None, priority: int = 0) -> bool`
  - `release_memory(nbytes: int) -> None`
- __CPU__:
  - `acquire_cpu(n: int = 1, timeout: Optional[float] = None, priority: int = 0) -> bool`
  - `release_cpu(n: int = 1) -> None`
- __IO__:
  - `acquire_io(nbytes: int, priority: int = 0) -> None`  (блокирующий троттлинг)
- __Диагностика__:
  - `snapshot() -> dict`

## HybridTokenCache

Файл: `cogniflex/memory/hybrid_token_cache.py`

- __Класс__: `HybridTokenCache`
- __Конструктор (основные аргументы)__:
  - `HybridTokenCache(brain, max_memory_tokens: int = 10000, disk_cache_dir: str = "token_cache", target_memory_gb: float = 50.0, dynamic_memory_limit: bool = True, system_free_mem_threshold: float = 0.15, memory_pressure_interval_s: float = 2.0, pressure_offload_batch: int = 200, disk_write_mb_s: float = 40.0, disk_read_mb_s: float = 200.0, disk_burst_factor: float = 2.0)`
- __Публичные методы__:
  - `get_token(token_id: str) -> Optional[Dict]`
  - `add_token(token_id: str, token_data: Any) -> None`
  - Совместимый KV‑интерфейс:
    - `exists(key: str) -> bool`
    - `get(key: str) -> Optional[Any]`
    - `set(key: str, value: Any) -> None`

Примечания:
- RAM‑дублирование выполняется только при успешном окне RAM через `CoreBrain.request_memory()`; при вытеснении память освобождается через `CoreBrain.release_memory()`.
- Диск всегда является источником истины при включённой агрессивной стратегии; `DiskCache` троттлит IO локально и, при наличии, через GRQ.

## DiskCache

Файл: `cogniflex/memory/disk_cache.py`

- __Класс__: `DiskCache`
- __Конструктор__:
  - `DiskCache(cache_dir: str, max_size_gb: float = 10.0, write_mb_s: float = 40.0, read_mb_s: float = 200.0, burst_factor: float = 2.0, resource_queue: Optional[object] = None)`
- __Публичные методы__:
  - `put(key: str, data: Any) -> None`
  - `get(key: str) -> Optional[Any]`
  - `get_stats() -> dict`
  - `close() -> None`

Примечание: Внутренне использует SQLite (`cache_metadata.db`) и zlib‑сжатие; троттлинг чтения/записи и опциональный системный троттлинг через GRQ `acquire_io()`.
