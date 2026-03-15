# API Reference

Краткая справка по ключевым публичным классам/методам CogniFlex. Пути и API проверены по текущему репозиторию.

## ModelManager
- Файл: `cogniflex/mlearning/model_manager.py`
- Класс: `ModelManager`
- Основные аргументы конструктора:
  - `brain: Optional[Any]`
  - `cache_dir: Optional[str]`
  - `model_dir: Optional[str]`
  - `use_gpu: bool = True`
  - `max_workers: int = 4`
  - `hybrid_cache_size: int = 50000`
  - `autoload: bool = True` (отключает/включает фоновые службы и автозагрузку)
- Ключевые методы:
  - `scan_models_directory() -> int`
  - `load_model(model_id: str) -> bool`
  - `unload_model(model_id: str) -> bool` (если присутствует в файле; см. реализацию)
  - `get_available_models()` / доступ к `model_metadata`
  - События: `_on_text_processor_ready(self, text_processor)`
- Поведение офлайн:
  - Внутренний метод `_is_offline()` учитывает `TRANSFORMERS_OFFLINE` и `HF_HUB_OFFLINE`
  - Все вызовы `from_pretrained(..., local_files_only=_is_offline())`

## UnifiedTextProcessor
- Файл: `cogniflex/mlearning/unified_text_processor.py`
- Класс: `UnifiedTextProcessor`
- Основные аргументы конструктора:
  - `brain=None, cache_dir: Optional[str]=None, use_gpu: bool=False, model_name: str="paraphrase-multilingual-MiniLM-L12-v2", max_workers: Optional[int]=None, use_async: bool=True, hybrid_cache: Optional[Any]=None`
- Ключевые методы:
  - `process_text(text: str) -> Dict[str, Any]`
  - `tokenize(text: str) -> List[str]`
  - `lemmatize(tokens: List[str]) -> List[str]`
  - `extract_keywords(text: str, tokens: List[str]) -> List[Tuple[str, float]]`
  - `analyze_sentiment(text: str) -> Dict[str, float]`
  - `get_embeddings(text: str) -> np.ndarray`
  - `tokenize_async(text: str) -> Dict[str, Any]`
  - `is_ready() -> bool`
  - `shutdown()`
- Интеграция:
  - Триггерит событие `text_processor_ready` через `brain.events` или колбэки совместимости
  - Интегрируется с гибридным кэшем памяти (если доступен)

## TokenProcessor
- Файл: `cogniflex/core/token_processor.py`
- Класс: `TokenProcessor`
- Основные методы:
  - `tokenize_query(query: str, context: Optional[Dict]=None) -> List[Dict]`
  - `get_token_statistics() -> Dict[str, Any]`
  - `prewarm_tokens_async(texts: List[str], priority: int = 5, batch_size: int = 100) -> bool`
  - `health_check() -> Dict[str, Any]`
  - `recover() -> bool`

## GUI: LearningModule
- Файл: `cogniflex/gui/learning_module.py`
- Класс: `LearningModule`
- Важные действия:
  - Кнопка «Обучить модель» в `_create_learning_opportunities_panel()` → обработчик `_start_model_training()`
  - `_start_model_training()` ищет `memory_graph_trainer` и запускает `train_async()` напрямую или через `DeferredCommandSystem`
  - Метрики обучения читаются через `trainer.get_training_stats()` (если доступно)

## Ядро (Core) — публичные методы
- Публичные методы ядра упоминаются/используются в GUI и тестах:
  - `get_system_health()`
  - `get_system_metrics()`
  - `get_system_dashboard_data()`
- Конкретные классы менеджеров (ConfigManager, SystemStateManager, ResourceManager, SystemMetricsManager) могут быть реализованы в ядре и доступны через `brain`.

Примечание: Некоторые внутренние/защитные методы намеренно не документированы. См. исходники для деталей и актуального состояния API.
