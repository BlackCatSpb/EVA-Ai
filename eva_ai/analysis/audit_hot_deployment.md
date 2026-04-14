# Аудит HotDeployment подсистемы EVA AI

**Дата аудита:** Tue Apr 14 2026  
**Аудитор:** EVA AI System Auditor  
**Версия:** 1.0

---

## Резюме

| Параметр | Значение |
|----------|----------|
| Всего файлов | 12 |
| Активных (используются) | 2 |
| Мёртвый код | 10 |
| Дубликатов классов | 6 |
| EventBus интеграция | **НЕТ** |
| **Оценка** | **3/10** |

---

## 1. Файлы HotDeployment подсистемы

### 1.1 Полный список файлов

`
eva_ai/mlearning/hot_deployment/
+-- __init__.py                 (685 lines) - HotDeploymentManager, FractalGraph, GraphNode
+-- optimized_inference.py      (354 lines) - OptimizedQwenGenerator, MultiBatchGenerator
+-- openvino_inference.py       (389 lines) - OpenVINOGenerator (ДУБЛИКАТ)
+-- openvino_via_optimum.py     (147 lines) - Конвертация через optimum (НЕ ИСПОЛЬЗУЕТСЯ)
+-- openvino_convert.py         (154 lines) - Конвертация через CLI (НЕ ИСПОЛЬЗУЕТСЯ)
+-- onnx_runtime.py             (395 lines) - OnnxRuntimeGenerator (ДУБЛИКАТ)
+-- onnx_optimizer.py          (433 lines) - OnnxOptimizer, HotDeploymentOnnx (НЕ ИСПОЛЬЗУЕТСЯ)
+-- llama_cpp_wrapper.py       (384 lines) - LlamaCppGenerator (ДУБЛИКАТ)
+-- llama_cpp_hot.py            (400 lines) - LlamaCppHotDeployment (**ИСПОЛЬЗУЕТСЯ**)
+-- download_gguf.py            (150 lines) - Скачивание GGUF (НЕ ИСПОЛЬЗУЕТСЯ)
+-- convert_to_gguf.py          (254 lines) - Конвертация в GGUF (НЕ ИСПОЛЬЗУЕТСЯ)
+-- export_onnx.py              (133 lines) - ONNX export (ДУБЛИКАТ)
`

---

## 2. Анализ использования файлов

### 2.1 ИСПОЛЬЗУЮТСЯ (Активные)

| Файл | Класс | Где используется |
|------|-------|-----------------|
| llama_cpp_hot.py | LlamaCppHotDeployment | rain_components.py:206, ractal_model_manager.py:101 |

### 2.2 НЕ ИСПОЛЬЗУЮТСЯ (Мёртвый код)

| Файл | Класс | Статус |
|------|-------|--------|
| __init__.py | HotDeploymentManager, FractalGraph | Импортируется но НЕ используется |
| optimized_inference.py | OptimizedQwenGenerator, MultiBatchGenerator | **НИКОГДА НЕ ИМПОРТИРОВАЛСЯ** |
| openvino_inference.py | OpenVINOGenerator | **НИКОГДА НЕ ИМПОРТИРОВАЛСЯ** |
| openvino_via_optimum.py | - | **НИКОГДА НЕ ИМПОРТИРОВАЛСЯ** |
| openvino_convert.py | - | **НИКОГДА НЕ ИМПОРТИРОВАЛСЯ** |
| onnx_runtime.py | OnnxRuntimeGenerator | **НИКОГДА НЕ ИМПОРТИРОВАЛСЯ** |
| onnx_optimizer.py | OnnxOptimizer, HotDeploymentOnnx | Импортируется в себе |
| llama_cpp_wrapper.py | LlamaCppGenerator | **НИКОГДА НЕ ИМПОРТИРОВАЛСЯ** |
| download_gguf.py | - | **НИКОГДА НЕ ИМПОРТИРОВАЛСЯ** |
| convert_to_gguf.py | - | **НИКОГДА НЕ ИМПОРТИРОВАЛСЯ** |
| xport_onnx.py | - | **НИКОГДА НЕ ИМПОРТИРОВАЛСЯ** |

---

## 3. Дубликаты классов

### 3.1 OpenVINOGenerator - 3 версии

| Путь | Размер | Статус |
|------|--------|--------|
| hot_deployment/openvino_inference.py | 389 lines | **МЁРТВЫЙ КОД** |
| hot_deployment/openvino_via_optimum.py | 147 lines | **МЁРТВЫЙ КОД** |
| core/openvino_generator.py | 1128 lines | **АКТИВЕН** |

### 3.2 LlamaCppGenerator - 2 версии

| Путь | Класс | Статус |
|------|-------|--------|
| hot_deployment/llama_cpp_wrapper.py | LlamaCppGenerator | **МЁРТВЫЙ КОД** |
| hot_deployment/llama_cpp_hot.py | LlamaCppHotDeployment | **АКТИВЕН** |

### 3.3 HotDeploymentManager - 2 версии

| Путь | Статус |
|------|--------|
| hot_deployment/__init__.py | Импортируется но НЕ используется |
| hot_deployment/onnx_optimizer.py (HotDeploymentOnnx) | **МЁРТВЫЙ КОД** |

---

## 4. EventBus Интеграция

### 4.1 Текущее состояние

**ВЫВОД: EventBus НЕ ИНТЕГРИРОВАН в HotDeployment подсистему**

### 4.2 Поиск EventBus

`
grep -r event_bus eva_ai/mlearning/hot_deployment/
# Результат: НИЧЕГО НЕ НАЙДЕНО
`

### 4.3 Что ДОЛЖНО быть

1. **Публиковать события:**
   - hot_deployment.ready
   - hot_deployment.model_loaded
   - hot_deployment.node_activated
   - hot_deployment.generation_complete

2. **Подписываться на события:**
   - system.idle
   - memory.graph_updated
   - model.unload_requested

---

## 5. LlamaCppHotDeployment (единственный активный)

### 5.1 Положительные черты

| Черта | Оценка |
|-------|--------|
| Синглтон паттерн | Хорошо |
| Интеграция с EVA ethics | Хорошо |
| Метод unload() | Хорошо |
| Форматирование промптов | Хорошо |
| Удаление повторений | Хорошо |

### 5.2 Проблемы

| Проблема | Серьёзность |
|----------|-------------|
| Нет EventBus интеграции | Средняя |
| Не наследуется от HotDeploymentManager | Средняя |
| Модель hardcoded пути | Средняя |

---

## 6. Рекомендации

### Критические (C)

| ID | Рекомендация | Файлы |
|----|--------------|-------|
| C1 | **Удалить мёртвый код** | Все кроме llama_cpp_hot.py |
| C2 | **Добавить EventBus интеграцию** | llama_cpp_hot.py |
| C3 | **Вызвать _init_llama_cpp()** | rain_components.py |

### Высокие приоритеты (H)

| ID | Рекомендация |
|----|--------------|
| H1 | Удалить openvino_inference.py - копия core/openvino_generator.py |
| H2 | Удалить llama_cpp_wrapper.py - копия llama_cpp_hot.py |
| H3 | Удалить onnx_optimizer.py - HotDeploymentOnnx не используется |

### Средние приоритеты (M)

| ID | Рекомендация |
|----|--------------|
| M1 | Удалить optimized_inference.py |
| M2 | Удалить openvino_via_optimum.py и openvino_convert.py |
| M3 | Удалить download_gguf.py и convert_to_gguf.py |
| M4 | Удалить xport_onnx.py и onnx_runtime.py |

---

## 7. Файлы после чистки

### Удалить (10 файлов)

`
hot_deployment/__init__.py
hot_deployment/optimized_inference.py
hot_deployment/openvino_inference.py
hot_deployment/openvino_via_optimum.py
hot_deployment/openvino_convert.py
hot_deployment/onnx_runtime.py
hot_deployment/onnx_optimizer.py
hot_deployment/llama_cpp_wrapper.py
hot_deployment/download_gguf.py
hot_deployment/convert_to_gguf.py
hot_deployment/export_onnx.py
`

### Оставить (2 файла)

`
hot_deployment/__init__.py          # Только экспорты
hot_deployment/llama_cpp_hot.py      # АКТИВЕН
`

---

## 8. Финальная оценка

| Критерий | Балл | Максимум | Комментарий |
|----------|------|----------|-------------|
| Использование кода | 1 | 3 | Только 2 из 12 файлов |
| Уникальность | 2 | 3 | 6 дубликатов классов |
| EventBus интеграция | 0 | 2 | Полностью отсутствует |
| Техническое качество | 1 | 2 | LlamaCppHotDeployment хорош |
| **ИТОГО** | **4** | **10** | **3/10** |

### Шкала оценки:

| Оценка | Значение |
|--------|----------|
| 9-10 | Отлично |
| 7-8 | Хорошо |
| 5-6 | Удовлетворительно |
| 3-4 | Плохо |
| 1-2 | Критично |

---

## 9. План действий

### Phase 1: Критическая чистка (1 час)
1. Удалить все 10 файлов мёртвого кода
2. Переделать __init__.py только для экспортов

### Phase 2: EventBus интеграция (2 часа)
1. Добавить EventBus в LlamaCppHotDeployment
2. Публиковать события: ready, model_loaded, generation_complete
3. Подписываться на: system.idle

### Phase 3: Интеграция с brain (1 час)
1. Вызвать _init_llama_cpp() в правильном месте
2. Проверить конфигурацию

---

**Отчёт сгенерирован:** Tue Apr 14 2026  
**Следующий аудит:** Через 2 недели после чистки
