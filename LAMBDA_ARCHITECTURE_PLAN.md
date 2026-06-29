# λ_d: План построения новой архитектуры

## Преамбула: что приняли

После разбора установлено:

1. **λ_d(x) = φ^x/√5 — непрерывный предел Binet.** Для больших x эквивалентен `exp(x·ln φ)`, что даёт `softmax(T=2.08)`. НО это только одна сторона. Нелинейность приходит из **Zeckendorf-разложения**: logit → бинарное дерево с φ-основанием и запретом соседних единиц. Это дискретная (не continuous) структура, которая не сводится к температуре.
2. **Резонанс φ:** одна константа пронизывает три уровня — спектр A_d (`sr=φ`), размерность Zeckendorf-разрядов (`F_k ≈ φ^k`), и основание readout (`φ^x`). Это matching scales, а не нумерология.
3. **A_d — линейный RNN** (как SSM/Mamba/RWKV). Content-dependent A(h) добавляет data-dependent routing.
4. **MSE per-layer → CE end-to-end.** Текущий подход (MSE на attention deltas) не композируется в генерацию. Нужен сквозной CE-loss.
5. **Frozen MLP = конфликт.** 67% параметров заморожены в режиме, для которого не оптимизированы. Нужна адаптация (LoRA) или fine-tune.
6. **Старт с чистого листа.** Qwen — источник идей (архитектура блока, веса для инициализации), но не цель дистилляции. Доказываем работоспособность λ_d-архитектуры как самостоятельной модели.

---

## Архитектура слоя (окончательная)

```
Вход: h ∈ ℝᴰ

  ┌─ 1. RMSNorm (пред-λ_d) ───────────────────────────────┐
  │   h_norm = rms_norm(h, w_gamma)                       │
  └────────────────────────────────────────────────────────┘
                           │
  ┌─ 2. Content-dependent A(h) ────────────────────────────┐
  │                                                        │
  │   2a. Gating: α = softmax(W_gate · h_norm)             │
  │       α ∈ ℝᴷ (K = 3~8 режимов Фибоначчи)              │
  │                                                        │
  │   2b. Смесь спектров:                                  │
  │       Λ_eff = diag( Σ α_k · λ_k )                     │
  │       λ_k = корень xᵏ = xᵏ⁻¹ + ... + 1               │
  │       (предвычислены, frozen)                          │
  │                                                        │
 │   2c. Применение через собственный базис:               │
 │       delta = V_l · Λ_eff · V_l⁻¹ · h_norm             │
 │       V_l ∈ ℝᴰˣᴰ — learned orthogonal (на слой)       │
 │       (Cayley init: V = expm(S), S skew-sym)           │
 │                                                        │
 │   2d. Clamp: scale = 1/max(1, |sr(Λ_eff)|)            │
  │       delta = scale · delta                            │
  └────────────────────────────────────────────────────────┘
                           │
  ┌─ 3. Residual ──────────────────────────────────────────┐
  │   h = h + delta                                        │
  └────────────────────────────────────────────────────────┘
                           │
  ┌─ 4. RMSNorm (пост-λ_d) ───────────────────────────────┐
  │   h_norm = rms_norm(h, w_gamma2)                      │
  └────────────────────────────────────────────────────────┘
                           │
  ┌─ 5. SwiGLU MLP ───────────────────────────────────────┐
  │   gate = silu(h_norm @ W_g)                           │
  │   up   = h_norm @ W_u                                 │
  │   out  = (gate × up) @ W_d                            │
  │                                                        │
 │   + LoRA адаптер (rank=256, если MLP frozen):         │
 │   ΔW_g = A_g @ B_g, ΔW_u = A_u @ B_u, ΔW_d = B_d @ A_d│
 │   Или: разморозить последние 12 слоёв MLP полностью   │
  └────────────────────────────────────────────────────────┘
                           │
  ┌─ 6. Residual ──────────────────────────────────────────┐
  │   h = h + out                                          │
  └────────────────────────────────────────────────────────┘
                           │
Выход: h' ∈ ℝᴰ
```

### Параметры слоя

| Компонент | Размерность | Параметров | Trainable |
|-----------|-------------|-----------|-----------|
| RMS нормы (×2) | D × 2 | 2D = 5K | опционально |
| V_l (базис, на слой) | D×D | D² = 6.55M | frozen (инит через Qwen SVD) |
| W_gate + b_l | D×K + K | D·K + K ≈ 20K | да |
| λ_k (K значений) | скаляры | K = 3~8 | нет (frozen) |
| LoRA gate (rank=r=256) | (D×r + r×I) × 2 | 2·D·r ≈ 1.3M | да |
| LoRA up/down (rank=256) | (D×r + r×I) × 2 | 2·D·r ≈ 1.3M | да |
| **Итого trainable** | | **~2.6M/слой** | |
| **Итого trainable (36 слоёв)** | | **~94M** | |
| MLP frozen | 2560×9728 × 3 | 74.7M | нет |

**Сравнение:** 94M trainable vs 236M (текущий A_d) vs 943M (Qwen attention).  
В 10× меньше полного внимания. В 2.5× меньше плотного A_d.

**Важно:** 36 × V_l = 36 × 6.55M = 236M frozen параметров. Они не trainable, но занимают память. На T4 (16GB): 236M × 2 bytes = 472MB — приемлемо.

### 2e (дополнение). Zeckendorf Tree Readout (inference only)

**Обучение:** tied lm_head = `h·W_embed^T` (стандартный, 374M frozen).  
**Inference:** Zeckendorf-дерево (128K trainable, ~25·D FLOPs).

```
Разряды Фибоначчи: F_k ∈ {1, 2, 3, 5, 8, 13, 21, ..., F_K}
Каждый токен i → Zeckendorf-код: i = Σ b_k · F_k,  b_k ∈ {0,1}, ∀k: b_k·b_{k+1}=0

Центроиды (learned, не mean!):
  c_{k,s,d} ∈ ℝᴰ — обучаемый вектор для (уровень k, состояние s, цифра d)
  Всего: K × 2 × 2 × D = 25 × 4 × 2560 = 256K параметров

Вероятность:
  P(i|h) = Πₖ P(bₖ | h, stateₖ)

  где:
    state_0 = 0
    P(b=1|h, s) = φ^{h·c_{k,s,1}} / (φ^{h·c_{k,s,0}} + φ^{h·c_{k,s,1}})
    state_{k+1} = bₖ  (0 → любая цифра, 1 → принудительно 0)

  Запрет соседних единиц — нелинейность:
    bₖ = 1 ⇒ b_{k+1} = 0  (принудительно)
```

Стоимость: K = log_φ(V) ≈ 25 уровней, на каждом 2 h·c = 50·D = 128K FLOPs.  
**В 3000× дешевле** полного lm_head (374M FLOPs).

**Резонанс:** та же φ в спектре A_d, в размерности разрядов F_k и в основании
readout. Token на расстоянии i со вкладом φ^{-i} попадает в Zeckendorf-разряд
того же масштаба φ^i.

### Вычислительная стоимость (на токен)

| Операция | FLOPs |
|----------|-------|
| RMSNorm (×2) | 4·D = 10K |
| W_gate · h + b (gating) | D·K + K ≈ 20K |
| V_l · (Λ · (V_l⁻¹ · h)) | 2·D² + D = 13.1M |
| Clamp sr | 0 (power iteration раз в N шагов) |
| LoRA forward (+rank=256 | 2·D·rank + 2·I·rank = 1.3M + 5M ≈ 6.3M |
| MLP frozen | 4·D·I = 99.6M |
| **Итого слой** | **~119M** |
| **Итого 36 слоёв** | **~4.3B** |
| lm_head readout | D·V = 374M |
| Zeckendorf readout | K·D ≈ 128K (inference) |
| **Обучение (токен)** | **~4.7B** (через lm_head) |
| **Inference (токен)** | **~4.3B** (через Zeckendorf) |

Сравнение: Qwen attention при L=1024 — 13.4B FLOPs (только decode).  
λ_d — 4.5B FLOPs на токен, независимо от L.  
С Zeckendorf readout: ~4.1B.

---

## План реализации (10 недель)

### Фаза 0: Фундамент (неделя 1) — ✅ ЗАВЕРШЕНА

**Цель (достигнута):** Инфраструктура готова, стабильность подтверждена на случайных данных.

**Выполнено:**
1. ✅ Создан `ld_model/` — чистый код, без копипасты
2. ✅ Ядро: `LDBlock` + `LDStack` с content-dependent A(h)
3. ✅ Базис V_l: learned orthogonal (Householder reflections, 32 отражения)
4. ✅ λ_k: предвычислены корни x^k = x^{k-1} + ... + 1 для k=2..8
5. ✅ **Gate differentiation:** α(h₁) ≠ α(h₂), cos=0.80 (критерий < 0.9 выполнен)
6. ✅ Тест стабильности: no NaN, no Inf, норма ~16 (D=256, 6 слоёв)
7. ✅ Тест последовательной генерации: 32 шага, норма стабильна
8. ✅ Тест α-энтропии: gate_scale=4 → H=1.26 vs scale=1 → H=1.38 (разреженнее)
9. ✅ Zeckendorf readout: сравнение с lm_head, проверка запрета соседних единиц
10. ✅ Документация: `ARCHITECTURE_ANALYSIS.md`, `LAMBDA_ARCHITECTURE_PLAN.md`

**Результаты тестов (D=256, 6 слоёв, K=4, float16):**

| Тест | Результат | Статус |
|------|-----------|--------|
| Стабильность (NaN/Inf) | Нет NaN, нет Inf | ✅ |
| Отношение норм out/in | 1.00 | ✅ |
| Дифференциация врат cos(h₁,h₂) | 0.80 (< 0.9) | ✅ |
| Энтропия α (scale=4) | 1.26 / 1.39 | ✅ |
| Последовательная генерация 32 шага | норма 16.0 ± 0.0 | ✅ |
| Временнáя дисперсия врат | 0.00001 (низкая — ожидается на тексте) | ⚠️ |
| Gate_scale=4 vs scale=1 энтропия | 1.26 < 1.38 | ✅ |
| Zeckendorf top-10 overlap с lm_head | 0.1/10 (случ. центроиды — ожидаемо) | ⚠️ |
| Ограничение Цекендорфа | все токены валидны | ✅ |

**Файлы:**
- `ld_model/core.py` — LDBlock, LDStack, LoRALinear, LDMLP, fibonacci_roots, random_orthogonal
- `ld_model/readout.py` — ZeckendorfReadout (обучаемые центроиды, обход дерева)
- `tests/test_synthetic.py` — 4 теста: стабильность, дифф. врат, распред. врат, генерация
- `tests/test_alpha_zeckendorf.py` — 5 тестов: α-энтропия, дифф. врат, послойное, Zeckendorf

---

### Фаза 1: Single-layer CE (неделя 2)

**Цель:** Обучить один LDBlock на next-token prediction. Доказать, что loss падает.

**Задачи:**
1. Обучить один слой: embed → A(h) → MLP → readout → CE
2. Dataset: 10K отрывков из Wikipedia/FineWeb (128 токенов каждый)
3. Сравнить с baseline: один слой Qwen attention (заморожен) vs один слой λ_d (trainable A(h))
4. Grid search по K (2, 3, 4, 6, 8) — какой порядок Фибоначчи даёт лучший loss
5. Grid search по LoRA rank (0, 16, 32, 64) — нужна ли адаптация MLP

**Метрики:**
- Perplexity на validation (должна падать)
- Gate distribution (не коллапсирует ли в один режим)
- sr(A_eff) post-training (стабилен ли)

**Критерий:** ppl < baseline (random init) и ppl падает стабильно 5+ эпох.

---

### Фаза 2: Multi-layer stack (недели 3-4)

**Цель:** 12 слоёв, CE через стек. Проверить композицию без взрыва.

**Задачи:**
1. 12 слоёв, каждый со своим W_gate (но общий V)
2. Full CE: `CE(softmax(h_36 · W_embed^T), target)`
3. Gradient checkpointing (backprop через 12 слоёв = 12 × 13M = 156M активаций, надо чекпоинтить)
4. Мониторинг: gradient norm, layer-wise update ratio, sr per layer
5. Сравнить с 12 слоями Qwen (same MLP, attention вместо A_d)

**Ключевые вопросы:**
- Разные ли gates у разных слоёв? (Layer 0 выбирает d=2, Layer 11 выбирает d=8?)
- Есть ли коллапс: все α_k → 1/K?
- Не взрывается ли градиент через 12 A_d?

**Критерий:** ppl падает, gradient norms в пределах 10× друг от друга, gates не коллапсируют.

---

### Фаза 3: Full-scale (недели 5-6)

**Цель:** 36 слоёв, полный D=2560, 100M+ токенов.

**Задачи:**
1. 36 слоёв, полный размер (V=146260, D=2560)
2. Инициализация MLP из Qwen (веса frozen, только LoRA trainable)
3. Dataset: 100M токенов (Wikipedia + FineWeb, русский + английский)
4. Оптимизатор: AdamW, lr=3e-4, cosine schedule, warmup 1000 steps
5. Mixed precision (bf16)

**Проблемы:**
- 36 × 680K trainable = 24.5M → AdamW: 24.5M × 4 bytes × 3 = 294MB — влезает
- V (DCT) = 0 trainable, 0 memory (аналитический, не хранится)
- MLP forward: 36 × 99.6M = 3.6B FLOPs — bottleneck на T4 (~10 tok/s)

**Критерий:** ppl < 30 на русском тексте (baseline Qwen3-4B: ppl ≈ 12).  
Ожидание: 30-50 (в 2-4× хуже полного трансформера, что ожидаемо).

---

### Фаза 4: Fine-tune и анализ (недели 7-8)

**Цель:** Понять, что работает, что нет.

**Задачи:**
1. Frozen MLP vs full fine-tune vs LoRA-only — ablation
2. K=2 vs K=4 vs K=8 — влияние числа режимов
3. Позиция: как gates меняются при перестановке токенов во фразе
4. Layer-wise α-распределения: есть ли hierarchy (low→local, high→semantic)?
5. Длина контекста: как ppl меняется с L (128, 512, 1024, 4096)
6. Zeckendorf readout vs lm_head: KL и top-k overlap после обучения

**Ожидаемые гипотезы:**
- Нижние слои выбирают малые d (короткая память, локальная обработка)
- Верхние слои выбирают большие d (длинная память, семантическая композиция)
- Learned V необходим (DCT не диагонализует shift operators)
- LoRA rank=256 нужно, rank=512 не даёт улучшения

**Критерий:** Понимание, на что способна архитектура и где её границы.

---

### Фаза 5: Inference и интеграция (недели 9-10)

**Цель:** Быстрый inference, интеграция в EVA/FCF.

**Задачи:**
1. Fuse V и V⁻¹: `A_eff = V · diag(α·λ) · V⁻¹` → одна матрица, пересчитывается при смене α
2. Оптимизация: если α меняется редко (каждые N токенов, не на каждом), можно кэшировать A_eff
3. Batch size: при seq=1024, λ_d: 4.5B FLOPs/токен. На T4 (65 TFLOPS): ~14 tok/s. На A100 (312 TFLOPS): ~70 tok/s.
4. FCF bridge: λ_d-стек как альтернатива colloc-generation
5. Export: torch.compile + CUDA graph → минимальный latency

**Критерий:** 10+ tok/s на T4, интеграция в FCF без изменения API.

---

## Roadmap

```
Неделя 1:  ████████████████████  Фаза 0 (фундамент, стабильность) ✅
Неделя 2:  ░░░░░░░░░░░░░░░░░░░░  Фаза 1 (single layer CE) — следующая
Неделя 3-4: ░░░░░░░░░░░░░░░░░░░░  Фаза 2 (12-layer stack, проверка композиции)
Неделя 5-6: ░░░░░░░░░░░░░░░░░░░░  Фаза 3 (36-layer, 100M tokens)
Неделя 7-8: ░░░░░░░░░░░░░░░░░░░░  Фаза 4 (ablation, анализ)
Неделя 9-10:░░░░░░░░░░░░░░░░░░░░  Фаза 5 (inference, FCF интеграция)
```

---

## Синтетическая верификация: трассировка промпта

Промпт: **"The future of AI is"** (5 токенов → t₀..t₄)

### Шаг 0: Embedding

```
W[t₀], W[t₁], W[t₂], W[t₃], W[t₄] ∈ ℝᵛ → ℝᴰ
v₀ = W[t₄]  (состояние после токена "is")
```

До этого: 4 шага рекуррентности (t₀→t₁→t₂→t₃→t₄).  
Каждый шаг: `v_{n+1} = A(h_n) · v_n + W[t_{n+1}]`.

### Шаг 1: предсказание следующего токена (P5)

```
1a. h_norm = RMSNorm(v₄)
    Норма: ||h_norm||² = D = 2560 (RMSNorm гарантирует)
    Направление: кодирует "The future of AI is"

1b. α = softmax(W_gate · h_norm + b_l)
    α ∈ ℝᴷ. Для слоя l: какие режимы Фибоначчи активированы.
    Ожидание: разные слои выбирают разные d.
    Layer 0: α ≈ [0.8, 0.1, 0.05, ...] → d≈2 (локальная обработка)
    Layer 35: α ≈ [0.1, 0.2, 0.3, 0.4] → d≈5+ (семантическая композиция)

1c. Λ_eff = diag(Σ αₖ · λₖ)
    λ₂ ≈ 1.618, λ₃ ≈ 1.839, λ₄ ≈ 1.928, λ₅ ≈ 1.966, λ₆ ≈ 1.984
    d_eff = α · [2, 3, 4, 5, 6, ...]
    Для layer 0 (α≈d=2): d_eff ≈ 2.1 → λ_eff ≈ 1.65
    Для layer 35 (α≈d=5): d_eff ≈ 4.7 → λ_eff ≈ 1.95

1d. delta = V_l · Λ_eff · V_l⁻¹ · h_norm
    V_l — слой-специфичный базис. V_l⁻¹ · h_norm — координаты h в этом базисе.
    Λ_eff растягивает/сжимает каждую координату: xₖ → λ_effₖ · xₖ
    V_l преобразует обратно в residual stream.

1e. scale = clamp(sr(Λ_eff))
    sr = max_k |λ_effₖ · αₖ| ≈ 1.95 для layer 35, ≤ 1.65 для layer 0.
    Если sr > 1: scale = 1.65⁻¹ или 1.95⁻¹ → delta stabilised.

1f. v₅ = v₄ + scale · delta
    v₅ — обновлённое состояние. delta — "замена attention output".

1g. post_norm = RMSNorm(v₅)
    h_norm₂ = rms_norm(v₅, w_gamma2)

1h. MLP forward:
    gate = silu(h_norm₂ @ (W_g + A_g@B_g))  — LoRA-адаптированный gate
    up   = h_norm₂ @ (W_u + A_u@B_u)
    out  = (gate · up) @ (W_d + B_d@A_d)
    v₅' = v₅ + out
```

**Вопрос:** Чем delta_layer_0 отличается от delta_layer_35?

- Layer 0: d_eff ≈ 2 → быстрая смена, внимание к последнему токену.
  delta обновляет v с акцентом на биграммы (pos-1, pos).
  Аналог: shallow pattern detection ("is" → "a"/"the"/"not").

- Layer 35: d_eff ≈ 5 → долгая память, attention ко всем 5 токенам.
  delta обновляет v с учётом полного контекста.
  Аналог: semantic composition ("future of AI is" → "bright"/"uncertain").

### Шаг 2: Readout (предсказание токена)

**Обучение (tied lm_head):**
```
logits = v_36 @ W_embed^T    (1, V)
loss = CE(softmax(logits), target)
```

**Inference (Zeckendorf tree):**
```
h = v_36  (последнее состояние после 36 слоёв)

for k = K-1 down to 0:     # 25 уровней (F₂₅ ≈ 150K > 146260)
    p1 = φ^{h·c_{k,state,1}} / (φ^{h·c_{k,state,0}} + φ^{h·c_{k,state,1}})
    if state == 1: digit = 0     # запрет соседних единиц
    else: digit = sample(p1)
    state = digit
    bits.append(digit)

token_id = Σ bits[k] · F_{k}   # Zeckendorf → token
```

**Резонанс φ:** Поскольку v_36 = A(h_35)·v_35 + W[t₄], и spectral radius
A(h) ≈ Σ αₖ·λₖ ≈ φ^{d_eff-1}, вклад токена tᵢ в v_36 пропорционален
φ^{-(36-i)·(d_eff-1)}. Это же φ стоит в основании readout (φ^{h·c}).

Token на расстоянии i имеет "вес" φ^{-i·(d_eff-1)} в скрытом состоянии.
При чтении через Zeckendorf: k-й разряд F_k ≈ φ^k даёт "маску" для
масштаба контекста φ^k. Согласование: на каком расстоянии токен,
в такой разряд он и попадает.

### Проверка осмысленности

| Свойство | Ожидание | Как проверить |
|----------|----------|--------------|
| α(h₁) ≠ α(h₂) для разных фраз | gates различны | Gate cos similarity < 0.9 |
| Gates разные по слоям | Layer 0: d≈2, Layer 35: d≈5 | α_entropy_by_layer |
| delta норма < residual норма | ||A(h)@h|| < ||h|| | sr ≤ 1 clamp |
| MLP не расходится | Loss падает | CE(train), CE(val) |
| Top-5 осмыслен | "bright", "not", "already" | Human eval |
| Position sensitivity | Gates меняются при перестановке | α("future of AI") ≠ α("AI of future") |
| Zeckendorf ≈ lm_head | KL < 0.1 на val | Сравнение распределений |

| Риск | Вероятность | Влияние | Митигация |
|------|------------|---------|-----------|
| α-коллапс (все gates одинаковы) | высокая | высокое | α-entropy loss + layer-specific bias b_l |
| sr > 1 на смеси спектров | средняя | высокое | Explicit clamp: sr = max(|λ_k|), scale = 1/sr |
| Frozen MLP блокирует learning | высокая | высокое | LoRA rank=256; если не хватит → разморозить top-12 слоёв MLP |
| Learned V не выучивается | средняя | среднее | Cayley init + orthogonal regularization |
| Gradient scaling (36 A_d) | средняя | среднее | Layer-wise lr decay ×0.9, gradient clipping 1.0 |
| V_l⁻¹ численно нестабилен | низкая | высокое | Householder reflexions вместо полного V (V = Π(I - 2·u·uᵀ)) |
| Zeckendorf readout хуже lm_head | высокая | низкое | Оставить lm_head для обучения, Zeckendorf только для inference |
| Gate не дифференцирует входы | средняя | критическое | Если α(h₁) ≈ α(h₂) для разных фраз → вся архитектура не имеет смысла |
| Gate низкая временнáя дисперсия | высокая | среднее | На случайном шуме var=0.00001; на реальном тексте ожидается рост. Если нет → добавить α-entropy loss |

---

## Что не вошло в план (отложено)

1. **Фазовая модуляция** (раздел 4 анализа) — слишком сложна для первой итерации.
2. **FPGA/ASIC inference** — после доказательства концепции.
3. **Multi-modal** — FCF пока text-only, расширение позже.
4. **Distributed training** — 24.5M trainable влезает в один GPU, распределение не нужно.

---

## Сводка

```
Архитектура:  v_{n+1} = A(h_n) · v_n + W[t_n]
             A(h) = V · diag(α(h)·λ) · V⁻¹
             α(h) = softmax(W_gate · h + b_l)  (b_l — layer-specific bias)
Trainable:   24.5M (W_gate + LoRA rank=256 + Zeckendorf centroids)
Frozen:      3.06B — embed, MLP, нормы
Loss:        CE end-to-end (через tied lm_head), Zeckendorf readout — inference only
Complexity:  4.5B FLOPs/token (O(D²), не зависит от L)
Memory:      5KB state (= KV cache eliminated)
Цель:        Доказать, что линейный RNN с content-dependent
             routing и φ-спектром работает как языковая модель.
```
