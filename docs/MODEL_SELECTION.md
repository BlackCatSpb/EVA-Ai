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

> Именно **Qwen3.5**, не Qwen3.

**HuggingFace IDs:**
- `Qwen/Qwen3.5-0.8B` / `Qwen/Qwen3.5-0.8B-Instruct`
- `Qwen/Qwen3.5-2B` / `Qwen/Qwen3.5-2B-Instruct`
- `Qwen/Qwen3.5-4B` / `Qwen/Qwen3.5-4B-Instruct`
- `Qwen/Qwen3.5-9B` / `Qwen/Qwen3.5-9B-Instruct`
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

**Вывод:** **Приоритет #1 для замены ruGPT3.** Модель уже поддерживается в коде — нужно только снять запрет в конфиге.

---

## 2. Saiga (IlyaGusev) — лучший русский 🇷🇺

**HuggingFace IDs:**
- `IlyaGusev/saiga_llama3_8b` — Llama3 + русский SFT
- `IlyaGusev/saiga_nemo_12b` — Mistral NeMo 12B
- `IlyaGusev/saiga_llama3_8b_gguf` — GGUF вариант

**Контекст:** 8K–32K токенов
**Языки:** Русский (специально обучен на русских диалогах)

**Плюсы:**
- ✅ **Лучшее качество русского** из всех открытых моделей
- ✅ Специально обучен на русских разговорных данных
- ✅ Активно поддерживается

**Минусы:**
- ❌ 8B+ — нужен GPU или долго на CPU

**Вывод:** Выбор при наличии GPU 8GB+, когда качество русского критично.

---

## 3. Gemma 3 (Google DeepMind) — продакшн + мультимодальность

**HuggingFace IDs:**
- `google/gemma-3-1b-it` — 1B instruct
- `google/gemma-3-4b-it` — 4B instruct
- `google/gemma-3-12b-it` — 12B instruct
- `google/gemma-3-27b-it` — 27B instruct

**Дата выхода:** 2026
**Контекст:** 128K токенов
**Языки:** 140+ языков (русский заявлен)
**Мультимодальность:** Изображения + текст во всех размерах

| Модель | BF16 | Q4 QAT | CPU |
|---|---|---|---|
| **1B** | 2 GB | 0.5 GB | ✅ отлично |
| **4B** | 8 GB | 2.6 GB | ✅ хорошо |
| **12B** | 24 GB | 6.6 GB | ⚠️ медленно |
| **27B** | 54 GB | 14.1 GB | ❌ |

**Плюсы:**
- ✅ QAT-варианты — 3x меньше памяти без потери качества
- ✅ Мультимодальность с 1B
- ✅ llama.cpp / ollama / gemma.cpp поддержка
- ✅ Долгосрочная поддержка Google

**Минусы:**
- ❌ Русский язык документально не подтверждён (140 языков заявлено, оценки не опубликованы)

**Вывод:** Хороший выбор если нужна мультимодальность или долгосрочная поддержка.

---

## 4. GLM-4.7-Flash (Zhipu AI) — НЕ рекомендован ❌

**HuggingFace ID:** `zai-org/GLM-4.7-Flash`
**Параметры:** 31B total / 3B active (MoE)
**Дата выхода:** Январь 2026
**Контекст:** 200K токенов (MLA)

| Формат | Размер | Примечание |
|---|---|---|
| BF16 | 60 GB | Нереалистично |
| Q4_K_M | 15–20 GB | Минимум |

**Проблемы:**
- ❌ Требует GPU для нормальной скорости (MoE)
- ❌ **Не оптимизирован под русский** — en-centric
- ❌ Ориентирован на code generation, не диалог
- ❌ Минимум 24 GB unified memory

**Вывод:** Не подходит для CogniFlex. Заточен под coding на мощном железе.

---

## 5. BitNet b1.58 (Microsoft) — экспериментальный ⚗️

**HuggingFace IDs:**
- `microsoft/bitnet-b1.58-2B-4T` — 2B, обучен на 4T токенах
- `microsoft/bitnet-b1.58-2B-4T-gguf` — GGUF вариант
- `QuantFactory/bitnet_b1_58-3B-GGUF` — community 3B

**Архитектура:** 1.58-бит веса (тернарные: -1, 0, 1), без умножений FP
**Дата выхода:** Апрель 2025
**Контекст:** ~2K–4K токенов
**Языки:** English-primary, **русский слабый**

| Формат | Размер | Примечание |
|---|---|---|
| 1-bit native | 0.4 GB | Нужен bitnet.cpp |
| GGUF (bitnet.cpp) | 0.5–0.8 GB | НЕ совместим с llama.cpp |

**Актуальный статус (2026):**
- ⚠️ **GGUF-формат НЕ совместим с llama.cpp** — требует отдельный `bitnet.cpp`
- ⚠️ Критический баг на ARM Linux (Apple Silicon M1–M4 работает, x86 AVX2 работает)
- ⚠️ Инференс-фреймворк в ранней стадии
- ✅ Скорость на CPU: 40–48 tok/s — рекордная для класса
- ✅ Заготовка `bitnet_model_manager.py` уже есть в коде

**Когда использовать:** CPU-only деплой на слабом железе (RAM < 8GB), если сделать русскоязычный fine-tune.

**Вывод:** Революционная архитектура, но **не для продакшна в 2026**. Экосистема незрелая, русский язык слабый. Вернуться когда появится зрелая llama.cpp поддержка.

---

## 6. SmolLM2 (HuggingFace) — микромодели

**HuggingFace IDs:**
- `HuggingFaceTB/SmolLM2-135M-Instruct`
- `HuggingFaceTB/SmolLM2-360M-Instruct`
- `HuggingFaceTB/SmolLM2-1.7B-Instruct`

**Контекст:** 8K токенов
**Языки:** English-primary, русский слабый

| Модель | FP16 | Q4 | Применение |
|---|---|---|---|
| 135M | 0.3 GB | 0.1 GB | IoT / смартфоны |
| 360M | 0.7 GB | 0.2 GB | Edge-устройства |
| 1.7B | 3.4 GB | 0.8 GB | Лёгкие CPU |

**Лицензия:** Apache 2.0
**Вывод:** Полезны для тестирования пайплайна без весов модели. **Не для русского языка.**

---

## Итоговое сравнение

| Модель | Размер (Q4) | CPU viable | Русский | Контекст | Статус | Приоритет |
|---|---|---|---|---|---|---|
| ruGPT3Large (текущий) | 800 MB | ⚠️ медленно | ✅ натив | 1K | Рабочий | Заменить |
| **Qwen3.5-2B** | **1 GB** | **✅✅** | **✅ хорошо** | **262K** | **Продакшн** | **#1** |
| **Qwen3.5-4B** | **2.5 GB** | **✅** | **✅ хорошо** | **262K** | **Продакшн** | **#2** |
| Saiga-Llama3-8B | 4.5 GB | ❌ | ✅✅ лучший | 8K | Продакшн | При GPU |
| Gemma3-1B | 0.5 GB | ✅✅ | ⚠️ не верифицирован | 128K | Продакшн | #3 |
| Gemma3-4B | 2.6 GB | ✅ | ⚠️ | 128K | Продакшн | #4 |
| BitNet-2B | 0.5 GB | ✅✅✅ | ❌ слабый | ~4K | Эксперимент | Backlog |
| SmolLM2-1.7B | 0.8 GB | ✅✅ | ❌ | 8K | Продакшн | Тесты |
| GLM-4.7-Flash | 15–20 GB | ❌ | ❌ | 200K | Продакшн | Не нужен |

---

## Рекомендуемая стратегия интеграции

```
ModelManager (фасад)
└── detect_hardware() → профиль CPU/GPU/MPS
    ├── GPU 8GB+ VRAM    → Saiga-Llama3-8B (лучший русский)
    ├── GPU 4-8GB VRAM   → Qwen3.5-4B Q4
    ├── CPU Apple Silicon → Qwen3.5-2B Q4 (через llama-cpp-python)
    ├── CPU x86 16GB+    → Qwen3.5-2B Q4
    ├── CPU x86 8-16GB   → Qwen3.5-0.8B Q4 или Gemma3-1B
    └── CPU < 8GB RAM    → ruGPT3 (fallback, текущий)
```

### Фазы внедрения

**Немедленно (Фаза 0):**
1. Снять блокировку в `brain_config.json` → разрешить Qwen3.5
2. Активировать `qwen_model_manager.py` (уже реализован)
3. Тест с `Qwen/Qwen3.5-2B-Instruct` Q4

**Краткосрочно (Фаза 1):**
1. Добавить `hardware_detector.py` для автовыбора модели
2. Добавить Gemma3 менеджер (для мультимодальности)
3. Сделать `brain_config.json` поддерживающим несколько профилей:
```json
"model": {
  "backend": "auto",
  "profiles": {
    "gpu_8gb":  { "name": "saiga_llama3_8b", "backend": "transformers" },
    "gpu_4gb":  { "name": "qwen3.5-4b",      "backend": "llama_cpp" },
    "cpu_16gb": { "name": "qwen3.5-2b",      "backend": "llama_cpp" },
    "cpu_8gb":  { "name": "qwen3.5-0.8b",    "backend": "llama_cpp" },
    "fallback": { "name": "rugpt3",           "backend": "transformers" }
  }
}
```

**Среднесрочно (Фаза 2):**
1. Дождаться зрелости BitNet llama.cpp интеграции
2. Добавить русский fine-tune BitNet когда появится
3. Интеграция SmolLM2 для тестовой среды без весов

---

## Эмбеддинги (вместо отключённого SentenceTransformer)

| Модель | Размер | CPU | Русский |
|---|---|---|---|
| `intfloat/multilingual-e5-small` | 117 MB | ✅ | ✅ хорошо |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 118 MB | ✅ | ✅ хорошо |
| `BAAI/bge-m3` | 570 MB | ⚠️ медленно | ✅✅ отлично |

**Рекомендация:** `multilingual-e5-small` — минимальный размер при хорошем качестве.
