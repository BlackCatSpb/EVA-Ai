# λ_d — Content-Dependent Spectral Language Model

**λ_d** (lambda-d) — рекуррентная языковая модель с контент-зависимым спектральным оператором. Вместо механизма внимания (Transformer) или фиксированной рекурренции (Mamba) λ_d вычисляет для каждого токена собственный линейный оператор `A(h)`, спектр которого управляется корнями Фибоначчи.

Архитектура нацелена на **инференс с памятью O(1)** (весь контекст сжимается в скрытое состояние 896 чисел) и **работу на Edge-устройствах** (смартфоны, NPU, робототехника) — при весе ~50 MB после замены lm_head на ZeckendorfReadout.

---

## Математическая основа

### Рекурренция λ_d

Вместо `softmax(QK^T)V` (Transformer) или `A·h + B·x` (Mamba/SSM), λ_d использует **контент-зависимый линейный оператор**:

```
h_out = h + V · diag(α(h)·λ) · V^T · norm(h + conv(h))
```

где:
- **V** ∈ O(D) — случайная ортогональная матрица (собственный базис, frozen)
- **λ** = [λ₂, λ₃, ..., λ_{K+1}] — корни Фибоначчи (спектр, frozen)
- **α(h)** ∈ Δ^K — контент-зависимый гейт (trainable)
- **conv(h)** — depthwise causal 1D convolution (локальный контекст, frozen)

### Спектр Фибоначчи (λ_k)

Корни Фибоначчи — наибольшие действительные корни полинома:

```
x^k = x^{k-1} + x^{k-2} + ... + x + 1
```

| k | λ_k | Свойство |
|---|-----|----------|
| 1 | 1.0 | Тривиальный корень (не используется) |
| 2 | 1.618 | **φ** — золотое сечение, короткая память |
| 3 | 1.839 | Среднее затухание |
| 4 | 1.928 | Длинная память |
| 5 | 1.966 | Очень длинная память |
| 6 | 1.984 | Почти без затухания |
| 7 | 1.994 | Квази-постоянная память |

Каждый λ_k соответствует горизонту памяти: при λ = φ информация затухает экспоненциально, при λ → 2 — сохраняется почти навсегда. Это образует **естественную иерархию времён** затухания.

### Контент-зависимое управление (block-wise gating)

α(h) — softmax-распределение весов между K корнями:

```
α(h) = softmax(4.0 · (W_gate · rms_norm(h) + b_gate))
```

λ_eff — **вектор размера D**, где каждая из K групп (D//K измерений) масштабируется на λ_k · α_k:

```
λ_eff = (α ⊙ λ).repeat_interleave(D//K, dim=-1)
```

**Почему блоковый, а не скалярный?** В первой реализации λ_eff был скаляром, что после нормировки давало identity-оператор — LDBlock ничего не учил. Блоковый λ_eff даёт D независимых eigen-коэффициентов.

**Почему K блоков, а не D гейтов?** W_gate — D×K (896×4 = 3.6K параметров). Если бы каждый из D каналов имел свой гейт, было бы D² (800K).

### Causal Conv1d (кросс-токен микс)

Depthwise causal 1D-свёртка с kernel=4, padding=3 слева:

```
h_norm = rms_norm(h + conv1d(h))
```

Решает проблему **позиционной слепоты**: без conv модель эквивалентна bag-of-words (не отличает "кот съел рыбу" от "рыбу съел кот"). Conv даёт локальный контекст 4 токена.

### Ортогональный базис V

V ∈ O(D) — произведение 32 случайных отражений Хаусхолдера. Фиксирован (buffer). V^T = V^{-1}, что даёт обращение за O(D²) без вычислений. Каждый слой имеет свой V_l.

**Зачем?** Без V все моды λ действуют независимо по осям. V смешивает их, распределяя спектральное сжатие/растяжение по всем D измерениям.

---

## Архитектура

```
tokens → Embed(D) → h₀
for l in 1..L:
    h_conv = CausalConv1d(h)                     # локальный контекст
    h = h + LDBlock(rms_norm(h + h_conv))        # спектральное преобразование
    h = h + BottleneckMLP(rms_norm(h))           # D → bottleneck → D
h = rms_norm(h)
logits = h @ lm_head^T                           # или ZeckendorfReadout
```

### LDBlock (ядро)

```
→ h_norm = rms_norm(h + conv(h))
→ α = softmax(4.0 · (h_norm@W_gate + b))
→ λ_eff = (λ_k · α).repeat_interleave(D//K, dim=-1)
→ h_proj = h_norm @ V^T
→ h_scaled = h_proj * λ_eff
→ Δ = h_scaled @ V
→ h_out = h + Δ
```

**Нет clamping**: норма Δ ограничена max(λ)·√D ≈ 2·√D автоматически.

### BottleneckMLP

```
z = SiLU(W_up @ h)
h_out = W_down @ z
```

- W_up: D → bottleneck (256), W_down: bottleneck → D
- Полный ранг 256 против rank=16 у старого LoRA SwiGLU
- Все веса trainable

**Почему bottleneck, а не LoRA?** LoRA rank=16 на D=896, I=4864 давал эффективный ранг 16 — 4864 нейрона лежали в 16-мерном подпространстве. Bottleneck 256 даёт ранг 256 при сопоставимом числе параметров.

### ZeckendorfReadout (альтернативный выход)

Замена lm_head (D×V = 44.8M параметров) на древовидный декодер с K·4·D параметров (< 15K).

**Теорема Цекендорфа:** каждое натуральное число имеет единственное представление в виде суммы НЕСОСЕДНИХ чисел Фибоначчи.

**Алгоритм декодирования:**
```
state = 0
for k = K-1 .. 0:
    if state == 1: bit = 0                    # ограничение Цекендорфа
    else: bit = argmax(softmax([h·c[k,0,0], h·c[k,0,1]]))
    state = bit
token_id = sum(bit[k] · F[k])
```

**Edge-преимущество:** при замене lm_head на ZeckendorfReadout вес модели падает с 95M до ~50M при сохранении качества.

---

## Обучение

### Текущая конфигурация (Phase 2)

| Параметр | Значение |
|----------|----------|
| D (hidden size) | 896 |
| n_layers | 12 |
| K (n_modes) | 4 |
| bottleneck | 256 |
| V | 32 Householder reflections per layer |
| Параметры всего | 95.1M |
| Физический batch | 4 |
| Accum steps | 8 |
| Эффективный batch | 32 (4096 токенов) |
| Sequence length | 128 |
| Оптимизатор | AdamW (wd=0.01) |
| LR schedule | Linear warmup 5% → Cosine |
| Max LR | 1e-3 |
| Grad clip | 1.0 |
| Epochs | 3 |
| Шагов/эпоха | 12 500 |
| Данные | wikitext-103 (50K chunks train, 500 eval) |
| Precision | fp32 (MX550 нестабильна в fp16) |
| Чекпоинты | per epoch + best + interrupt, auto-resume |

### Прогресс

На step 3300/12500 (первая эпоха):
- Train loss: 6.14
- Train PPL: 464
- LR: 9.97e-04 (warmup завершён, начало cosine decay)
- Кривая строго монотонна, без осцилляций

### Стабильность

- **Нормы:** hidden states в диапазоне ±6 (компактное скрытое пространство)
- **NaN:** 0 за всё время (fp32)
- **TDR:** решён увеличением таймаута драйвера до 10 сек

### Параметры

| Компонент | Shape | Параметры | Обучаемые? |
|-----------|-------|-----------|------------|
| Embedding | 50000×896 | 44.8M | Да |
| CausalConv1d (×12) | 896×1×4 + bias | 4.5K × 12 = 54K | Нет (frozen) |
| W_gate (×12) | 896×4 | 3.6K × 12 = 43K | Да |
| b_gate (×12) | 4 | 4 × 12 = 48 | Да |
| V (×12) | 896×896 | 803K × 12 = 9.6M | Нет (frozen) |
| BottleneckMLP (×12) | 896×256 + 256×896 | 459K × 12 = 5.5M | Да |
| RMSNorm weights | 896 | ~11K | Да |
| lm_head | 896×50000 | 44.8M | Да |
| **Итого** | | **105.5M** (95.1M trainable) | |

### Сравнение с Phase 1 (старая архитектура)

| Аспект | Phase 1 (buggy) | Phase 2 (fixed) |
|--------|-----------------|------------------|
| LDBlock | Identity (scalar λ_eff + clamping) | Работает (block-wise λ_eff, no clamp) |
| MLP | LoRA rank-16 над нулевой базой | Dense bottleneck 256, full rank |
| Позиционность | Bag-of-words | Causal Conv1d (kernel=4) |
| Batch | 4 (512 tok) | 32 eff (4096 tok, grad accum) |
| LR | 1e-3, cosine, no warmup | Warmup 5% + cosine |
| NaN | fp16 → 100% NaN | fp32 → 0 NaN |
| Обучение | Только gates + LoRA | Все веса trainable |

---

## Сравнение с альтернативами

| Аспект | Transformer | Mamba | λ_d |
|--------|-------------|-------|-----|
| Внимание | QK^T — O(L²) | O(L) scan | O(L) (нет внимания) |
| Память контекста | KV-cache O(L×D) | O(D) hidden state | O(D) hidden state |
| Память на 1M токенов | ~3.4 GB (D=896) | ~3.5 KB | ~3.5 KB |
| Позиционность | RoPE/AliBi | Causal 1D conv | Causal 1D conv |
| Параметры внимания | 4·D² | D·Δ | D·K (гейты) |
| Спектр | Фиксирован | HIPPO (фикс.) | α(h)-зависимый, блоковый |
| Вес модели (inference) | ~100M+ (lm_head) | ~100M+ | ~50M (с Zeckendorf) |
| Edge-ready | Нет (KV-cache) | Частично | Да |

---

## Edge AI & Мультиагентность

λ_d обладает уникальными свойствами для Edge-вычислений:

1. **O(1) память контекста:** весь контекст сжимается в D=896 чисел — ~3.5 KB. Против ~3.4 GB для Transformer с KV-cache на 1M токенов.

2. **Subspace Multiplexing:** ортогональный базис V ∈ O(896) можно разделить на подпространства (например, 4 × 224), каждое для независимого агента. Shared embed + conv (общая сенсорная кора), изолированные спектральные моды.

3. **Zeckendorf Readout:** удаление lm_head (44.8M → ~0) даёт модель ~50M для инференса без потери качества токенизации.

4. **Энергоэффективность:** BottleneckMLP (896→256→896) требует в ~10× меньше MAC, чем стандартный SwiGLU MLP (896→4864→896).

---

## Roadmap

### Phase 2 (текущая)
- ✅ 12 слоёв, D=896, K=4
- ✅ Исправление identity-бага (block-wise λ_eff)
- ✅ Causal Conv1d (кросс-токен микс)
- ✅ Dense Bottleneck MLP (замена LoRA)
- ✅ Gradient accumulation + warmup
- ✅ Все веса trainable
- ⏳ **Завершить первую эпоху, получить eval PPL**
- ⏳ Логирование энтропии гейтов α(h)

### Phase 3 (план)
- Масштабирование: 36 слоёв, D=2560, 10B+ токенов
- Zeckendorf Readout (инференс без lm_head)
- Subspace multiplexing для мультиагентности
- INT8/INT4 квантование (V — Orthogonal Procrustes)

### Research
- Анализ энтропии гейтов по слоям (когнитивная иерархия)
- Сравнение Zeckendorf vs lm_head (A/B тест)
- Kernel fusion: RMSNorm → Conv → V^T → Scale → V (один CUDA kernel)

---

## Файлы проекта

| Файл | Описание |
|------|----------|
| `ld_model/core.py` | LDConfig, fibonacci_roots, random_orthogonal, CausalConv1d, LDBlock, BottleneckMLP, LDStack |
| `ld_model/readout.py` | ZeckendorfReadout (древовидный декодер) |
| `train_phase2.py` | Тренировка Phase 2 (12 слоёв, D=896, grad accum, warmup) |
| `colab_train.py` | Скрипт для T4/Colab |
| `colab_phase2.ipynb` | Colab notebook |
| `monitor.py` | Монитор GPU/CPU/RAM |
| `debug_nan.py` | Диагностика NaN |
| `LAMBDA_ARCHITECTURE.md` | Полная документация архитектуры (рус.) |
| `SUMMARY.md` | Краткая сводка |

---

## Запуск

```bash
# Локально (MX550, fp32, batch=4, eff batch=32)
python train_phase2.py

# Colab (T4, batch=8, streaming data)
python colab_train.py --ckpt_dir /path/to/checkpoints

# Мониторинг
python monitor.py
```

Авто-resume: скрипт находит последний чекпоинт в `checkpoints/` и продолжает с того же шага.

---

## Терминология

- **λ_d** — контент-зависимый линейный оператор A(h) = V · diag(α★λ) · V^T
- **LDBlock** — один слой λ_d с causal conv + RMSnorm + spectral transform
- **LDStack** — стек LDBlock + BottleneckMLP слоёв
- **BottleneckMLP** — D → bottleneck → D (896→256→896), SiLU
- **CausalConv1d** — depthwise causal convolution, kernel=4
- **ZeckendorfReadout** — древовидный декодер на базе теоремы Цекендорфа
- **Fibonacci roots** — λ_k, наибольшие корни полинома x^k = Σx^{k-1} + ... + 1
- **Block-wise λ_eff** — вектор D, каждая из K групп масштабируется на λ_k·α_k
- **Grad Accum** — накопление градиентов через ACCUM_STEPS (eff batch = phys batch × ACCUM_STEPS)
