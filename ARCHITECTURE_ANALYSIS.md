# Qwen3-4B → λ_d: Полный архитектурный разбор

## 1. Qwen3-4B RuAdapt: спецификация

| Параметр | Значение |
|----------|----------|
| hidden_size (D) | 2560 |
| num_hidden_layers | 36 |
| num_attention_heads | 32 (Q), 8 (KV) |
| head_dim | 128 |
| intermediate_size | 9728 (SwiGLU) |
| vocab_size | 146260 |
| norm | RMSNorm (no bias) |
| activation | SwiGLU |
| position encoding | RoPE (θ=10⁷) |
| lm_head | tied with embed_tokens |
| Вес | ~4.0B параметров (bf16: 8GB) |

### Распределение параметров

| Компонент | Параметров | % |
|-----------|-----------|---|
| Embedding (tied lm_head) | 374.4M | 9.3% |
| Attention (QKVO + нормы) | 943.7M | 23.6% |
| MLP (gate + up + down) | 2689.6M | 67.1% |
| RMS нормы (все) | 0.2M | <0.1% |
| **Итого** | **~4008M** | 100% |

Per-layer: Attention = 26.2M, MLP = 74.7M, norms = 5K. **Каждый слой идентичен по архитектуре.**

---

## 2. Покомпонентная карта Qwen → λ_d

### 2.1. Embedding Layer

| Компонент | Qwen | λ_d | Параметры | Статус |
|-----------|------|-----|-----------|--------|
| embed_tokens | W ∈ ℝ¹⁴⁶²⁶⁰×²⁵⁶⁰ | W (тот же) | 374.4M | frozen |
| lm_head (readout) | h·Wᵀ | h·Wᵀ (та же операция) | 0 (tied) | идентично |

**Вывод:** прямое использование, без изменений.

---

### 2.2. Per-Layer Block (×36)

```
Qwen layer:
  x → RMSNorm → Q_proj  ← 10.5M
              → K_proj  ←  2.6M       QK_norm → RoPE → Q·K^T/√d → softmax → @V
              → V_proj  ←  2.6M
                                         → O_proj ← 10.5M → +x → RMSNorm → SwiGLU MLP (74.7M) → +x
                                                                                ↑ заморожен

λ_d layer:
  x → RMSNorm → A_d @ x ← 6.6M → +x → RMSNorm → SwiGLU MLP (тот же, 74.7M) → +x
                ↑ trainable                           ↑ frozen
```

#### 2.2.1. Input RMSNorm

| | Qwen | λ_d |
|---|---|---|
| Формула | `x / rms(x) * w` | та же |
| Параметры | 2560 (frozen) | 2560 (frozen) |
| Проблем | нет | нет |

#### 2.2.2. Q, K, V проекции — ELIMINATED (15.7M/слой)

```
Qwen:
  Q = h @ W_Q   (2560×4096 = 10.5M)
  K = h @ W_K   (2560×1024 =  2.6M)
  V = h @ W_V   (2560×1024 =  2.6M)
  ---
  Итого: 15.7M на слой, 565M всего
  Функция: content-dependent routing — каждый токен проецируется
  в три различных пространства для вычисления парных взаимодействий.

λ_d:
  A_d @ h        (2560×2560 = 6.6M)
  ---
  Итого: 6.6M на слой, 236M всего
  Функция: фиксированное линейное отображение, не зависит от h
```

**Потеря:** Единственный механизм, где содержимое токена определяет,
на какие другие токены он "обращает внимание". Без него модель
не может выборочно извлекать информацию из контекста.

#### 2.2.3. GQA Expand (Grouped Query Attention) — ELIMINATED

```
Qwen:
  8 KV-голов → expand до 32 → 4 Q-головы на 1 KV-голову

  Каждая пара (K,V) учит свой "аспект" контекста.
  8 независимых аспектов, 32 Q-головы их по-разному "читают".

λ_d:
  Нет расширения. A_d — одна матрица, один "аспект".
```

**Потеря:** Multi-head diversity. Qwen: 32 параллельных attention
с разными проекциями. λ_d: одна проекция.

#### 2.2.4. RoPE — ELIMINATED

```
Qwen:
  x ← rotate(x, pos)  для каждой головы
  Q·K^T содержит сигнал (pos_i - pos_j) — относительную позицию

λ_d:
  A_d^n — экспоненциальный рост/затухание
  При sr(A_d) = 1: A_d^n — вращение на единичной сфере без масштаба
  При sr(A_d) = φ: A_d^n → ∞ при n → ∞
```

**Потеря:** Distance-aware encoding. Qwen: `Q·K^T = f(pos_i - pos_j)`.
λ_d: A_d^n @ h₀ не даёт относительного расстояния — эргодическое
перемешивание на сфере. A_d^1 и A_d^100 неразличимы по норме.

#### 2.2.5. Scaled Dot-Product Attention — ELIMINATED (ключевая замена)

```
Qwen (L=1024):
  scores = Q·K^T / √128    — O(32 × 1024² × 128) = 4.3 GFLOPS
  weights = softmax(scores) — нелинейная селекция
  out = weights @ V @ W_O   — взвешенное среднее values
  ---
  Вычислительная сложность: O(L² · h)

λ_d:
  delta = A_d @ h_norm      — O(2560²) = 6.5 MFLOPS
  ---
  Вычислительная сложность: O(D²)
```

**Потеря:** Attention — это **pairwise interaction** + **content-conditional
aggregation**. Q·K^T даёт каждому токену оценку релевантности всех
остальных токенов. softmax превращает её в разреженное окно.
@V агрегирует только релевантные токены. A_d — фиксированная матрица
без выбора. Это фундаментальный tradeoff: O(L²) expressivity vs O(1) efficiency.

#### 2.2.6. Output Projection — ELIMINATED (10.5M/слой)

```
Qwen:
  W_O ∈ ℝ³²×¹²⁸ → ²⁵⁶⁰
  Смешивает 32 head-выхода обратно в residual stream.

λ_d:
  Нет W_O. A_d напрямую отображает h_norm → delta размерности D.
```

**Потеря:** 4:1 компрессия head-выходов в один stream. Без неё
нет разделения на "аспекты" даже на уровне residual.

#### 2.2.7. Post-Attention RMSNorm

| | Qwen | λ_d |
|---|---|---|
| Параметры | 2560 (frozen) | 2560 (frozen) |
| Проблем | нет | нет |

#### 2.2.8. SwiGLU MLP — FROZEN (74.7M/слой)

```
Qwen:
  gate = silu(h @ W_g)  — 2560→9728 (24.9M)
  up   = h @ W_u        — 2560→9728 (24.9M)
  out  = (gate × up) @ W_d — 9728→2560 (24.9M)

  Всего: 74.7M/слой × 36 = 2.69B (67.1% модели)

λ_d:
  Те же веса, frozen. MLP не переобучается.
```

**Проблема:** MLP обучен на `h + attention(h)`, а получает
`h + A_d@h`. Attention и A_d дают структурно разные дельты:

- `attention(h)` — взвешенное среднее values (self-normalizing, bounded)
- `A_d@h` — линейная проекция (не bounded, не среднее)

Frozen MLP функционально не совпадает с ожидаемым входом.
Это **центральный конфликт** архитектуры: 67% параметров
заморожены в режиме, для которого не оптимизированы.

#### 2.2.9. Residual Connections

```
Qwen:
  h = h + attention(h)  — bounded: ||attn(h)|| ≤ max_i ||V_i||
  h = h + mlp(h)

λ_d:
  h = h + A_d@h_norm    — unbounded: ||A_d@h|| может быть > ||h||
  h = h + mlp(h)
```

**Проблема:** В Qwen attention — выпуклая комбинация (self-normalising).
Норма attention выхода гарантированно ≤ нормы values.
A_d — линейный оператор. Без контроля sr(A_d) норма дельты
может превышать норму входа → взрыв состояния за 36 слоёв.

**Решение:** Clamp sr(A_d) ≤ 1. Тогда ||A_d@h|| ≤ ||h||,
и residual стабилен. Но sr=1 теряет position signal.

---

### 2.3. Final RMSNorm

| | Qwen | λ_d |
|---|---|---|
| Параметры | 2560 (frozen) | 2560 (frozen) |
| Проблем | нет | нет |

---

## 3. Сводная таблица: что уходит, что остаётся

| Компонент Qwen | Параметры (M) | λ_d | Параметры (M) | Δ |
|---|---|---|---|---|
| embed_tokens | 374.4 | frozen | 374.4 | 0 |
| lm_head | 0 (tied) | идентично | 0 | 0 |
| **Q_proj** | **377.9** | — | **0** | **−378** |
| **K_proj** | **94.4** | — | **0** | **−94** |
| **V_proj** | **94.4** | — | **0** | **−94** |
| **O_proj** | **377.9** | — | **0** | **−378** |
| Q_norm | 0.15 | — | 0 | −0.15 |
| K_norm | 0.04 | — | 0 | −0.04 |
| **A_d (36 слоёв)** | **0** | **trainable** | **236.2** | **+236** |
| input_layernorm (×36) | 0.09 | frozen | 0.09 | 0 |
| post_attn_layernorm (×36) | 0.09 | frozen | 0.09 | 0 |
| gate_proj (×36) | 896.5 | frozen | 896.5 | 0 |
| up_proj (×36) | 896.5 | frozen | 896.5 | 0 |
| down_proj (×36) | 896.5 | frozen | 896.5 | 0 |
| final_norm | 0.003 | frozen | 0.003 | 0 |
| **Итого** | **4008** | | **3300** | **−708** (17.6%) |

В λ_d модели из 3300M параметров:
- 236M trainable (6.5%) — только A_d
- 3064M frozen (93.5%) — embed + MLP + нормы

---

## 4. Ключевые проблемы при масштабировании

### P0: Нет content-dependent routing (фундаментальная)

```
Qwen: out = softmax(Q·K^T/√d) @ V @ W_O
      ↑ токен решает, какие токены влияют на его обновление

λ_d:  delta = A_d @ h_norm
      ↑ все токены получают одинаковое преобразование
```

**Следствие:** модель не может "посмотреть в прошлое" по содержанию.
Не может найти релевантный контекст по смыслу. Token salad после
2-3 токенов — прямое следствие. При любом масштабе (0.8B–72B)
эта проблема остаётся, если A_d не сделать data-dependent.

### P0: Frozen MLP обучен на attention-дельтах, не на A_d

```
Qwen training: MLP(f(x + attention(x)))   → веса оптимизированы
λ_d inference: MLP(f(x + A_d@x))           → те же веса, другой вход
```

67% модели (2.69B frozen) работают не в том режиме.
При масштабировании проблема усугубляется: чем больше параметров
заморожено, тем жёстче ограничение на форму дельты.

### P1: Position signal = 0 при sr=1

| sr | A_d^n @ h₀ | Position info |
|----|-------------|---------------|
| 1.618 | ||A_d^n·h₀|| → ∞ при n→∞ | взрыв |
| 1.0 | ||A_d^n·h₀|| = const, ergodic sphere mixing | нет расстояний |
| 0.5 | ||A_d^n·h₀|| → 0 при n→∞ | информация теряется |

При любом sr < ∞, A_d^n — это **одна траектория на сфере**.
Модель не знает, насколько далеко i от j — только "был до" или "был после".
RoPE даёт точное относительное расстояние через частоты вращения.

### P1: Capacity bottleneck (4:1)

| Scale | D | A_d/слой | Qwen attn/слой | Ratio |
|-------|---|----------|----------------|-------|
| 4B | 2560 | 6.6M | 26.2M | 4.0× |
| 7B | 4096 | 16.8M | 67.1M | 4.0× |
| 14B | 5120 | 26.2M | 104.9M | 4.0× |
| 72B | 8192 | 67.1M | 268.4M | 4.0× |

A_d всегда в 4× меньше по параметрам, чем QKVO, при одинаковой
асимптотике O(D²). На 72B: A_d = 67M/слой × 80 = 5.4B trainable —
27% всех параметров. При этом A_d всё равно не выразительнее,
чем при D=2560 — просто больше того же.

### P2: Memory bandwidth на генерации

С per-layer load/forward/free:
- PCIe 4.0 x16: 32 GB/s
- A_d: 6.6M × 2 bytes = 13.2 MB
- Время загрузки: 13.2/32000 ≈ 0.4 ms
- 36 слоёв: 14.4 ms overhead на токен (при 3.8 TFLOPS A100: 6.5M FLOPs = 0.002 ms compute)

Генерация на 100 токенов: 1.4s overhead на загрузку A_d.

При D=8192 (72B): 67M × 2 = 134 MB/слой, 80 × 134 / 32000 = 335 ms на токен.
Только transfer, без compute. Нужен fused pipeline или on-device A_d.

### P2: Нормализация дельты

```
Qwen attention: out@W_O — bounded (выпуклая комбинация)
λ_d A_d@h: unbounded (линейный оператор)
```

Текущее решение: learn A_d с MSE, надеяться на sr ≈ φ → не работает.
Правильное: explicit `A_d @ rms_norm(h)` + output clamp.
Норма rms_norm(h) = ||w||² = 2560. A_d@rms_norm(h) ≈ sr·2560.
При sr близком к любому значению — стабильно. Но:

```
h_out = h + A_d@rms_norm(h) = h + sr * rms_dir(h)
```

Если A_d — identity + perturbation, h_out ≈ (1 + sr/D)·h.
Это работает. Если A_d — dense random, h_out = h + O(||h||) — ок.
Проблема только если sr > 1 и много шагов.

---

## 5. Анализ expressivity: что может A_d, что не может attention

### Может:

1. **Кодировать позицию** через A_d^n (ограниченно)
2. **Смешивать измерения** residual stream (один dense matmul)
3. **Давать затухание/рост** исторических токенов через sr(A_d)
4. **Быть обучен** под конкретный слой (layer-specific transformation)

### Не может:

1. **Content-dependent routing** — выбирать какие токены влияют
2. **Pairwise interaction** — Q·K^T матрица парных сходств
3. **Multi-head diversity** — 32 разных проекции одного входа
4. **Distance-aware encoding** — знать расстояние между токенами
5. **Softmax selection** — разреженное нелинейное окно
6. **Variable-length awareness** — attention масштабируется с L, A_d — нет

### Компенсируется MLP?

MLP может частично компенсировать (1), (2), (3) через нелинейное
преобразование h. Но MLP оперирует каждым токеном независимо
(position-wise). Attention — единственное место в трансформере,
где токены обмениваются информацией. A_d + MLP = токены "видят"
только прошлые состояния через A_d^n, но не друг друга.

---

## 6. Пути развития

### Вариант A: A_d data-dependent (Mamba-style)

```
A_d(h) = A_base + B·h     — добавляет D² = 6.6M/слой
A_d(h) @ h_norm — дельта зависит от h
```

- Content-dependent routing появляется
- +6.6M/слой = +236M (удвоение trainable)
- Frozen MLP получает более релевантные дельты
- **Риск:** градиент через 36·A_d тяжелее (backprop через data-dependence)

### Вариант B: Multi-head A_d

```
h_split = split(h, k=8)   — 8×320 dim
delta_k = A_d_k @ h_split_k — 8 × 320² = 819K/слой
delta = merge(delta_k)     — 8×320 → 2560
```

- Multi-head diversity без data-dependence
- 0.82M/слой (8× меньше оригинального A_d)
- 8 разных "аспектов" внимания
- **Риск:** без data-dependence каждый head всё равно фиксирован

### Вариант C: Разморозить MLP + CE end-to-end

- Отказаться от дистилляции Qwen
- Инициализировать MLP из Qwen, но обучать на CE
- A_d + MLP как единая модель
- **Риск:** стоимость обучения — ~$50K на 100B токенов

### Вариант D: Гибрид (low-rank + learnable MLP)

- A_d = U·V^T + diag(s) — low-rank + diagonal
- rank = 128 (как attention heads): 128·2560 + 128·2560 + 2560 = 657K/слой
- MLP разморозить только адаптеры (LoRA на gate/up/down)
- **Минимальные параметры:** ~1M/слой trainable

---

## 7. Резюме: что на самом деле происходит

```
Qwen:         h → RMSNorm → [Q·K^T → softmax → @V → W_O] → +h → MLP → +h
                              ↑ O(L²) pairwise, content-conditional

λ_d:          h → RMSNorm → [A_d] → +h → MLP → +h
                              ↑ O(D²) fixed, unconditional

Разница:      pairwise selection → single matrix
              content-conditional → content-independent
              multi-head → single-head
              O(L²) → O(D²)
```

Сравнение с существующими подходами:

| Модель | Замена attention | Data-dep? | Multi-head? | CE end-to-end? |
|--------|-----------------|-----------|-------------|----------------|
| λ_d (текущий) | A_d (dense, fixed) | нет | нет | нет |
| Mamba-2 | SSM + selection | да | да | да |
| RWKV-6 | WKV + gating | да | нет | да |
| Linear Attn | φ(Q)·φ(K)^T | да (через Q,K) | да | да |
| **λ_d (Var A)** | A_d + B·h | **да** | опционально | нужно |

**Главный вывод:** Текущая λ_d архитектура — это линейный RNN
с инициализацией из Qwen. Она работает как RNN, не как трансформер.
Для рабочей языковой модели нужно либо:
1. Добавить data-dependence в A_d (вариант A)
2. Или принять RNN-природу и обучать end-to-end большим corpus (вариант C)
