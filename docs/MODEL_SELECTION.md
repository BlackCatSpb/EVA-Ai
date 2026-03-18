# CogniFlex — Выбор модели: анализ вариантов

> Обновлено: 2026-03-18

---

## Текущее состояние

Система жёстко привязана к **ruGPT3Large** (`sberbank-ai/rugpt3large_based_on_gpt2`):
- ~800MB весов
- Требует GPU для приемлемой скорости
- Только русский язык
- Архитектура GPT-2, контекст **1024 токена**
- Обучение отключено в конфиге

В кодовой базе **уже есть заготовки**:
- `cogniflex/mlearning/bitnet_model_manager.py` — стабы для BitNet 2B/3B
- `cogniflex/mlearning/qwen_model_manager.py` — **полностью реализован** (0.8B–9B, 4-bit поддержка)
- `cogniflex/mlearning/model_config.py` — устарел, только GPT-2/ruGPT-3

---

## 1. Qwen3.5 (Alibaba) — рекомендован ✅

> Именно **Qwen3.5**, не Qwen2.5 или Qwen3.

**HuggingFace IDs:**
- `Qwen/Qwen3.5-0.8B-Instruct`
- `Qwen/Qwen3.5-2B-Instruct`
- `Qwen/Qwen3.5-4B-Instruct`
- `Qwen/Qwen3.5-9B-Instruct`
- `Qwen/Qwen3.5-27B`, `Qwen/Qwen3.5-35B-A3B` (MoE)

**Дата выхода:** Март 2026
**Контекст:** 262K токенов native, до 1M+ с RoPE/YaRN
**Языки:** 201 язык/диалект, включая **русский**
**Архитектура:** Gated Delta Networks + sparse MoE

| Модель | FP16 | Q4 | RAM на CPU | Рекомендация |
|---|---|---|---|---|
| **0.8B** | 1.6 GB | 0.5 GB | 2 GB | IoT / мобильный |
| **2B** | 4 GB | 1 GB | 3 GB | **CPU без GPU** ✅ |
| **4B** | 8 GB | 2.5 GB | 5 GB | Баланс качество/скорость |
| **9B** | 18 GB | 5 GB | 9 GB | GPU 8GB+ / RAM 16GB+ |

**Плюсы:**
- ✅ **Уже реализован** в `qwen_model_manager.py` — только включить
- ✅ Отличный русский (201 язык из обучения)
- ✅ Огромный контекст (262K vs 1K у ruGPT3)
- ✅ GGUF доступен — `ollama pull qwen3.5:2b`
- ✅ Instruct-режим из коробки

**Минусы:**
- ❌ 9B+ требует GPU или много времени на CPU

**Вывод:** **Приоритет #1 для замены ruGPT3.** Уже поддерживается в коде — снять запрет в конфиге.

---

## 2. Saiga (IlyaGusev) — лучший русский 🇷🇺

**HuggingFace IDs:**
- `IlyaGusev/saiga_llama3_8b`
- `IlyaGusev/saiga_llama3_8b_gguf`
- `IlyaGusev/saiga_nemo_12b`

**Контекст:** 8K–32K токенов | **Языки:** Русский (SFT)

**Плюсы:**
- ✅ **Лучшее качество русского** из всех открытых моделей
- ✅ Специально обучен на русских диалогах

**Минусы:**
- ❌ 8B+ — нужен GPU или долго на CPU

**Вывод:** Выбор при наличии GPU 8GB+, когда качество критично.

---

## 3. Gemma 3 (Google DeepMind) — продакшн + мультимодальность

**HuggingFace IDs:**
- `google/gemma-3-1b-it`, `google/gemma-3-4b-it`, `google/gemma-3-12b-it`, `google/gemma-3-27b-it`

**Контекст:** 128K | **Языки:** 140+ (русский заявлен) | **Дата:** 2026
**Мультимодальность:** Изображения + текст во всех размерах

| Модель | BF16 | Q4 QAT | CPU |
|---|---|---|---|
| **1B** | 2 GB | 0.5 GB | ✅ отлично |
| **4B** | 8 GB | 2.6 GB | ✅ хорошо |
| **12B** | 24 GB | 6.6 GB | ⚠️ медленно |

**Плюсы:**
- ✅ QAT-варианты — 3x меньше памяти без потери качества
- ✅ Мультимодальность с 1B
- ✅ Долгосрочная поддержка Google
- ✅ llama.cpp / ollama / gemma.cpp

**Минусы:**
- ❌ Русский язык документально не подтверждён (заявлен, оценок нет)

---

## 4. GLM-4.7-Flash (Zhipu AI) — НЕ рекомендован ❌

**HuggingFace ID:** `zai-org/GLM-4.7-Flash`
**Параметры:** 31B total / 3B active (MoE) | **Контекст:** 200K | **Дата:** Январь 2026

| Формат | Размер |
|---|---|
| BF16 | 60 GB |
| Q4_K_M | 15–20 GB |

**Проблемы:**
- ❌ MoE — нужен GPU (CPU нереалистично медленно)
- ❌ **Не оптимизирован под русский** (en-centric, code generation)
- ❌ Минимум 24 GB unified memory

**Вывод:** Не подходит для CogniFlex.

---

## 5. BitNet b1.58 (Microsoft) — экспериментальный ⚗️

**HuggingFace IDs:**
- `microsoft/bitnet-b1.58-2B-4T`
- `microsoft/bitnet-b1.58-2B-4T-gguf`

**Архитектура:** 1.58-бит веса (ternary: -1, 0, 1) | **Дата:** Апрель 2025
**Контекст:** ~2K–4K | **Языки:** English-primary, **русский слабый**

| Формат | Размер | Примечание |
|---|---|---|
| 1-bit native | 0.4 GB | Нужен bitnet.cpp |
| GGUF | 0.5–0.8 GB | **НЕ совместим с llama.cpp** |

**Актуальный статус (2026):**
- ⚠️ **GGUF не совместим с llama.cpp** — требует отдельный `bitnet.cpp`
- ⚠️ Критический баг на ARM Linux (Apple Silicon работает, x86 AVX2 работает)
- ✅ Скорость на CPU: 40–48 tok/s — рекордная для размера
- ✅ Заготовка `bitnet_model_manager.py` уже есть в коде

**Вывод:** Революционная идея, **не для продакшна в 2026**. Вернуться когда дозреет llama.cpp интеграция.

---

## 6. SmolLM2 (HuggingFace) — микромодели для тестов

**HuggingFace IDs:**
- `HuggingFaceTB/SmolLM2-135M-Instruct`
- `HuggingFaceTB/SmolLM2-360M-Instruct`
- `HuggingFaceTB/SmolLM2-1.7B-Instruct`

**Контекст:** 8K | **Языки:** English-primary | **Лицензия:** Apache 2.0

| Модель | Q4 | Применение |
|---|---|---|
| 135M | 0.1 GB | IoT / смартфоны |
| 360M | 0.2 GB | Edge |
| 1.7B | 0.8 GB | CPU-тесты без крупных весов |

**Вывод:** Полезны для тестирования pipeline без загрузки тяжёлых весов.

---

## Итоговое сравнение

| Модель | Размер (Q4) | CPU | Русский | Контекст | Статус | Приоритет |
|---|---|---|---|---|---|---|
| ruGPT3Large (текущий) | 800 MB | ⚠️ | ✅ натив | 1K | Рабочий | Заменить |
| **Qwen3.5-2B** | **1 GB** | **✅✅** | **✅** | **262K** | **Продакшн** | **#1** |
| **Qwen3.5-4B** | **2.5 GB** | **✅** | **✅** | **262K** | **Продакшн** | **#2** |
| Saiga-Llama3-8B | 4.5 GB | ❌ | ✅✅ лучший | 8K | Продакшн | При GPU |
| Gemma3-1B | 0.5 GB | ✅✅ | ⚠️ | 128K | Продакшн | #3 |
| Gemma3-4B | 2.6 GB | ✅ | ⚠️ | 128K | Продакшн | #4 |
| BitNet-2B | 0.5 GB | ✅✅✅ | ❌ слабый | ~4K | Эксперимент | Backlog |
| SmolLM2-1.7B | 0.8 GB | ✅✅ | ❌ | 8K | Продакшн | Тесты |
| GLM-4.7-Flash | 15–20 GB | ❌ | ❌ | 200K | Продакшн | Не нужен |

---

## Рекомендуемая стратегия интеграции

```
ModelManager (фасад)
└── detect_hardware() → профиль CPU/GPU/MPS
    ├── GPU 8GB+ VRAM    → Saiga-Llama3-8B
    ├── GPU 4-8GB VRAM   → Qwen3.5-4B Q4
    ├── CPU Apple Silicon → Qwen3.5-2B Q4 (llama-cpp-python)
    ├── CPU x86 16GB+    → Qwen3.5-2B Q4
    ├── CPU x86 8-16GB   → Qwen3.5-0.8B Q4 или Gemma3-1B
    └── CPU < 8GB RAM    → ruGPT3 (fallback)
```

### brain_config.json — многопрофильная конфигурация

```json
"model": {
  "backend": "auto",
  "profiles": {
    "gpu_8gb":  { "name": "saiga_llama3_8b",  "backend": "transformers" },
    "gpu_4gb":  { "name": "qwen3.5-4b",       "backend": "llama_cpp" },
    "cpu_16gb": { "name": "qwen3.5-2b",       "backend": "llama_cpp" },
    "cpu_8gb":  { "name": "qwen3.5-0.8b",     "backend": "llama_cpp" },
    "fallback": { "name": "rugpt3",            "backend": "transformers" }
  }
}
```

### Фазы внедрения

| Фаза | Действие | Сложность |
|---|---|---|
| **0 (сейчас)** | Снять блокировку Qwen в конфиге; активировать `qwen_model_manager.py` | Низкая |
| **1** | Добавить `hardware_detector.py`; Gemma3 менеджер | Средняя |
| **2** | Дождаться зрелости BitNet в llama.cpp | — |

---

## Эмбеддинги (вместо отключённого SentenceTransformer)

| Модель | Размер | CPU | Русский |
|---|---|---|---|
| `intfloat/multilingual-e5-small` | 117 MB | ✅ | ✅ хорошо |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 118 MB | ✅ | ✅ хорошо |
| `BAAI/bge-m3` | 570 MB | ⚠️ медленно | ✅✅ отлично |

**Рекомендация:** `multilingual-e5-small` — минимальный размер, хорошее качество.
