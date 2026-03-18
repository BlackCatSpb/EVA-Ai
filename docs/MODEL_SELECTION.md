# CogniFlex — Выбор модели: анализ вариантов

> Составлен: 2026-03-18

---

## Текущее состояние

Сейчас система жёстко привязана к **ruGPT3Large** (`sberbank-ai/rugpt3large_based_on_gpt2`):
- ~800MB весов
- Требует GPU для приемлемой скорости
- Только русский язык
- Архитектура GPT-2, контекст 1024 токена

---

## Анализ альтернатив

### 1. BitNet b1.58 (Microsoft) — для CPU

**Что это**: 1.58-битные веса (ternary: -1, 0, 1), линейные слои без умножений с плавающей точкой.

**Модели**:
- `microsoft/bitnet-b1.58-2B-4T` — 2B параметров, опубликован 2025
- `microsoft/bitnet-b1.58-3B-4T` — 3B параметров

**Характеристики**:
| Параметр | Значение |
|---|---|
| Размер весов | ~700MB (2B) / ~1GB (3B) |
| RAM в инференсе | 2-3GB |
| Скорость на CPU (Apple M) | ~10-15 tok/s |
| Скорость на CPU (x86 AVX2) | ~5-10 tok/s |
| Контекст | 4096 токенов |
| Языки | Многоязычный (en-centric) |

**Плюсы**:
- ✅ Работает на CPU без GPU — ключевое преимущество
- ✅ Низкое потребление памяти
- ✅ Простой инференс (llama.cpp / bitnet.cpp поддержка)
- ✅ Открытый код и веса

**Минусы**:
- ❌ Не специализирован под русский язык
- ❌ Качество ниже FP16 аналогов
- ❌ Нет официального GGUF для llama.cpp (2025)

**Вывод**: **Отличный вариант для CPU-только деплоя**. При хорошем русском fine-tune может заменить ruGPT3 на слабом железе.

---

### 2. Qwen2.5 (Alibaba) — баланс качество/скорость

**Модели**: `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen2.5-3B-Instruct`, `Qwen2.5-7B-Instruct`

**Характеристики** (3B вариант):
| Параметр | Значение |
|---|---|
| Размер весов | ~1.8GB (BF16) / ~0.9GB (Q4) |
| RAM на CPU | 4-6GB |
| Скорость CPU (M-series) | ~15-20 tok/s (Q4) |
| Контекст | 32768 токенов |
| Языки | 29 языков включая русский |

**Плюсы**:
- ✅ Отличная поддержка русского языка (обучен на CulturaX RU)
- ✅ Большой контекст (32K vs 1K у ruGPT3)
- ✅ Chat / Instruct режим из коробки
- ✅ GGUF доступен (ollama, llama.cpp)
- ✅ Активно поддерживается

**Минусы**:
- ❌ 7B версия требует GPU для нормальной скорости
- ❌ 1.5B — качество значительно хуже

**Вывод**: **Лучший вариант для замены ruGPT3** при наличии 6+ GB RAM. Особенно 3B-instruct.

---

### 3. Llama 3.2 / 3.1 (Meta) — производительность

**Модели**: `meta-llama/Llama-3.2-1B-Instruct`, `3.2-3B-Instruct`, `3.1-8B-Instruct`

**Характеристики** (3B вариант):
| Параметр | Значение |
|---|---|
| Размер весов | ~1.8GB (BF16) / ~1.2GB (Q4) |
| Контекст | 131072 токенов (!) |
| Языки | Многоязычный, русский OK |

**Плюсы**:
- ✅ Огромный контекст
- ✅ Хорошее качество для размера
- ✅ Широкая экосистема (vLLM, ollama, llama.cpp)

**Минусы**:
- ❌ Русский хуже, чем у Qwen
- ❌ Требует принятия Meta лицензии

---

### 4. Saiga (IlyaGusev) — русскоязычные instruct-модели

**Что это**: Fine-tune Mistral / LLaMA под русский язык.

**Модели**:
- `IlyaGusev/saiga_llama3_8b` — Llama3 + русский SFT
- `IlyaGusev/saiga_mistral_7b_lora` — Mistral + LoRA
- `IlyaGusev/saiga_nemo_12b` — Mistral NeMo 12B

**Плюсы**:
- ✅ Лучшее качество на русском из открытых моделей
- ✅ Специально обучен на русских диалогах
- ✅ Активно развивается

**Минусы**:
- ❌ 7B+ требует GPU или много времени на CPU
- ❌ Нет малых вариантов (< 3B)

**Вывод**: **Лучший выбор для русскоязычного качества** при наличии GPU.

---

### 5. RWKV-6 — для CPU без трансформеров

**Что это**: RNN-архитектура с трансформерным качеством. Постоянное O(1) потребление памяти при инференсе.

**Модели**:
- `RWKV/v6-Finch-1B6` — 1.6B параметров
- `RWKV/v6-Finch-3B` — 3B параметров

**Плюсы**:
- ✅ Постоянный memory footprint (нет KV-cache роста)
- ✅ Быстрый на CPU
- ✅ Хорошо для длинных последовательностей

**Минусы**:
- ❌ Нестандартная архитектура (нет llama.cpp поддержки)
- ❌ Русский язык слабее

---

### 6. Phi-4 mini / Phi-3.5 mini (Microsoft)

**Модели**: `microsoft/Phi-3.5-mini-instruct` (3.8B)

**Характеристики**:
| Параметр | Значение |
|---|---|
| Размер весов | ~2.2GB (Q4) |
| Контекст | 128K токенов |
| Качество | Высокое для размера |

**Плюсы**:
- ✅ Высокое качество рассуждений для малого размера
- ✅ Большой контекст
- ✅ GGUF доступен

**Минусы**:
- ❌ Слабый русский язык
- ❌ Нужен fine-tune для русского

---

## Сравнительная таблица

| Модель | Размер | CPU viable | Русский | Контекст | Рекомендация |
|---|---|---|---|---|---|
| ruGPT3Large (текущий) | 800MB | ❌ медленно | ✅ натив | 1K | Заменить |
| BitNet 2B | 700MB | ✅✅ | ⚠️ средне | 4K | CPU-only deploy |
| Qwen2.5-3B | 1.8GB | ✅ | ✅ хорошо | 32K | **Рекомендован** |
| Saiga Llama3 8B | 4.5GB | ❌ | ✅✅ | 8K | При наличии GPU |
| Llama3.2-3B | 1.8GB | ✅ | ⚠️ средне | 128K | Альтернатива |
| RWKV-6 3B | 1.7GB | ✅✅ | ⚠️ | ∞ | Для слабых машин |
| Phi-3.5 mini | 2.2GB | ✅ | ❌ слабо | 128K | Только с fine-tune |

---

## Рекомендуемая архитектура модельного слоя

```
ModelManager (фасад)
├── detect_hardware() → CPU/GPU/MPS profile
└── load_best_model(hardware_profile)
    ├── GPU (8GB+ VRAM) → Saiga-Llama3-8B или Qwen2.5-7B
    ├── GPU (4-8GB VRAM) → Qwen2.5-3B или BitNet-3B
    ├── CPU (16GB+ RAM) → Qwen2.5-3B Q4 через llama.cpp
    ├── CPU (8-16GB RAM) → BitNet-2B или Qwen2.5-1.5B
    └── CPU (< 8GB RAM) → ruGPT3 (текущий, минимальный)
```

### Реализация через llama.cpp / ollama

```python
# Предлагаемый интерфейс бэкенда
class ModelBackend(Protocol):
    def generate(self, prompt: str, max_tokens: int, **kwargs) -> str: ...
    def get_embeddings(self, text: str) -> list[float]: ...
    @property
    def context_length(self) -> int: ...
    @property
    def supports_streaming(self) -> bool: ...

class LlamaCppBackend(ModelBackend):
    """Для BitNet, Qwen GGUF, Llama GGUF через llama-cpp-python"""

class TransformersBackend(ModelBackend):
    """Для HuggingFace моделей (текущий ruGPT3)"""

class OllamaBackend(ModelBackend):
    """Для моделей запущенных в Ollama (простейший деплой)"""
```

---

## Конкретные рекомендации для CogniFlex

### Краткосрочно (сохранить ruGPT3 + добавить опцию)

1. Оставить ruGPT3 как дефолт для совместимости
2. Добавить поддержку **Qwen2.5-3B** через llama-cpp-python (GGUF)
3. Добавить переключение в `brain_config.json`:
   ```json
   "model": {
     "backend": "transformers",
     "name": "rugpt3",
     "alternatives": {
       "cpu_preferred": "qwen2.5-3b-q4",
       "gpu_preferred": "saiga-llama3-8b"
     }
   }
   ```

### Долгосрочно (новая архитектура)

1. Полностью заменить ruGPT3 на **Qwen2.5-3B** (русский + многоязычный + instruct)
2. Для CPU-only режима — **BitNet 2B** когда появится русскоязычный fine-tune
3. Для лучшего русского качества — **Saiga Llama3 8B** при наличии GPU 8GB+

### Для эмбеддингов (вместо отключённого SentenceTransformer)

- `intfloat/multilingual-e5-small` — 117MB, работает на CPU, хороший русский
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — 118MB, быстрый

---

## Что нужно для интеграции BitNet

1. Установить `bitnet.cpp` или использовать через llama.cpp (поддержка добавлена в 2025)
2. Конвертировать модель в GGUF: `python convert_hf_to_gguf.py microsoft/bitnet-b1.58-2B-4T`
3. Реализовать `BitNetBackend` в `cogniflex/mlearning/backends/bitnet_backend.py`
4. Добавить автоопределение CPU capabilities (AVX2/AVX512 для оптимальной скорости)

```python
# cogniflex/mlearning/hardware_detector.py
def get_optimal_backend() -> str:
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram >= 8: return "saiga_8b"
        if vram >= 4: return "qwen_3b_fp16"
    if platform.processor() and "arm" in platform.processor().lower():
        return "qwen_3b_q4"  # Apple Silicon
    ram_gb = psutil.virtual_memory().total / 1e9
    if ram_gb >= 16: return "qwen_3b_q4"
    if ram_gb >= 8: return "bitnet_2b"
    return "rugpt3"  # fallback
```
