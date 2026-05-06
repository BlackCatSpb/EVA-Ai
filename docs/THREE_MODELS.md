# Eva Pie Architecture - Три модели

## Конфигурация (Вариант 1)

### Модели:
1. **LOGIC** - RuadaptQwen3-4B (condensed, n_ctx=4096)
2. **CONTEXT** - RuadaptQwen3-4B (extended, n_ctx=32768) 
3. **CODER** - Qwen Coder 1.5B

### Файлы:
```
LOGIC: ruadapt_qwen3_4b_q4_k_m.gguf (2.32 GB)
CONTEXT: ruadapt_qwen3_4b_q4_k_m.gguf (тот же файл, но с n_ctx=32768)
CODER: qwen2.5-coder-1.5b-instruct-q4_k_m.gguf (1.04 GB)
```

### L2 Роутинг:
- **CODER_KEYWORDS** → CODER модель (код, функции, алгоритмы)
- **CONTEXT_KEYWORDS** → CONTEXT модель (подробно, детально, анализ)
- **default** → LOGIC модель (логика, рассуждения)

### Обновленные файлы:
- `eva_ai/core/unified_generator.py` - три ModelType
- `eva_ai/core/pipeline_adapter.py` - поддержка трех моделей
- `eva_ai/core/brain_components.py` - инициализация с coder_path
- `brain_config.json` - пути к трем моделям

### Использование:
```python
# Автоматический выбор
result = generator.generate("Привет!")  # → LOGIC
result = generator.generate("Напиши функцию")  # → CODER
result = generator.generate("Объясни подробно")  # → CONTEXT
```
