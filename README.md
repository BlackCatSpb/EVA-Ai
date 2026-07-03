# λ_d — Content-Dependent Spectral Language Model

**λ_d** (lambda-d) — рекуррентная языковая модель, которая заменяет механизм внимания
(Transformer) на **контент-зависимый спектральный оператор** A(h). Память контекста —
**O(1)** (3.5 KB), сложность — **O(L·D)**, контекст — **∞** (рекуррентная, без окна).

Вместо `softmax(QK^T)V` λ_d настраивает для каждого токена 4 «струны»
(корни Фибоначчи — спектральные моды), и информация распространяется через
частоты, а не через попарное внимание.

---

## Быстрый старт

```bash
# Анализ чекпоинта (ASCII)
python analyze_model.py

# HTML-отчёт (открыть в браузере)
python analyze_model.py --html

# Генерация текста
python generate.py --ckpt checkpoints/model_step30000.pt --prompt "Москва — столица"

# Тренировка с нуля (тест, 5000 шагов)
python train_phase2.py --data russian
```

---

## Как это работает

### Рекурренция λ_d

```
h_out = h + V_eff · diag(α(h)·λ) · V_eff^T · norm(h + conv(h))
```

Где:
- **V_eff = V_frozen @ R** — ортогональный базис с поворотом Кэли (R ∈ O(D))
- **λ = [λ₂, λ₃, ..., λ_{K+1}]** — корни Фибоначчи (спектр)
- **α(h)** ∈ Δ^K — контент-зависимый гейт (4 числа — натяжение каждой струны)
- **conv(h)** — depthwise causal 1D convolution (локальный контекст 4 токена)

### Спектр Фибоначчи

| k | λ_k | Затухание | Роль |
|---|-----|-----------|------|
| 2 | 1.618 | Быстрое (φ^{-t}) | Локальная грамматика |
| 3 | 1.839 | Умеренное | Синтаксис |
| 4 | 1.928 | Медленное | Семантика |
| 5 | 1.966 | Почти постоянное | Глобальный контекст |

Каждый токен получает свою gate-композицию: союз «и» активирует только высокие
частоты, слово «инвестиционный» уходит в бас (глобальный контекст).

### Learnable V (Cayley rotation)

V_frozen — случайная ортогональная матрица (32 отражения Хаусхолдера).
Поверх неё каждый слой учит **ортогональный поворот** R:

```
R = solve(I − S, I + S),  S = A·B^T − B·A^T  (skew-symmetric, rank 2r)
```

Свойства:
- **Аналитическая ортогональность**: ||R^T R − I|| < 1e-9 (machine epsilon)
- **Не требует orth_loss или clipping**
- **r=8** (эффективный ранг ~14, запас 2–5×)

### Adaptive Depth (per-token routing)

Gate spread (σ(α) — стандартное отклонение гейтов) — zero-cost salience signal:
- **Низкий spread** (< 0.08) — токен поверхностный (союзы, предлоги → мелко)
- **Высокий spread** (> 0.25) — токен важный (сущности, термины → глубоко)

На обучении: soft routing (взвешенная сумма). На инференсе: hard routing
(пороговое отсечение, 0 overhead).

```
Слой 0: [и, ——, Москва, —, столица, .]   100% проходят
Слой 6: [и, ——, Москва, —, столица, .]   ~70% (мелкие заморожены)
Слой 11:[и, ——, Москва, —, столица, .]   ~20% (только ключевые)
```

Пороги — обучаемые: `depth_logits` с инициализацией linspace(0.25 → 0.45).

---

## Архитектура (подробно)

### LDBlock (ядро)

```
→ h_norm = rms_norm(h + conv(h))
→ α = softmax(4.0 · (h_norm @ W_gate + b_gate))
→ R = solve(I − S, I + S),  S = V_cay_A @ V_cay_B^T − V_cay_B @ V_cay_A^T
→ V_eff = V @ R
→ h_proj = h_norm @ V_eff^T
→ h_scaled = h_proj * (λ ⊙ α).repeat_interleave(D//K)
→ Δ = h_scaled @ V_eff
→ h_out = h + Δ
```

### BottleneckMLP

```
z = SiLU(W_up @ h)         # D → 256
h_out = W_down @ z          # 256 → D
```

Dense bottleneck (полный ранг 256) вместо LoRA rank-16.

### Полный стек

```
embed → [LDBlock + BottleneckMLP] × 12 → RMSNorm → lm_head
```

---

## Результаты

### Phase 2 (wikitext-103, 12×896, 95M)

| Метрика | Значение |
|---------|----------|
| Train PPL (эпоха 3) | 49.78 |
| Eval PPL (эпоха 3) | 131.53 |
| NaN | 0 |
| Экстраполяция (8× длины) | PPL 43.3 → 116.7 (+169%) |

### Phase 2 Russian (12×896, 844M токенов)

| Step | Loss | PPL | LR |
|------|------|-----|----|
| 25000 | 4.93 | 137.9 | 1e-3 |
| 30000 | 4.84 | 125.9 | 1e-3 |
| 30600 | 4.83 | 125.3 | 1e-3 |

### Adaptive depth (25K шагов)

| Уровень | Spread | Тип токенов |
|:-------:|:------:|-------------|
| L0 | < 0.08 | Союзы, предлоги, пунктуация |
| L1 | 0.08–0.15 | Обычные слова |
| L2 | 0.15–0.25 | Значимые существительные |
| L3 | > 0.25 | Ключевые сущности |

98% токенов проходят L0, 59% L7, 30% L10.

### Learnable V

- Orth error: **< 1e-9** на всех 12 слоях
- |A+B| нормы: ~0.17 (свежий init, растут с обучением)
- Eff rank S: ~14 (r=8 даёт запас)

---

## Сравнение с Transformer

| Аспект | Transformer | λ_d |
|--------|-------------|-----|
| Память контекста | KV-cache O(L×D) | **O(D)** — 3.5 KB |
| Сложность | O(L²·D) | **O(L·D)** |
| Контекст 1M токенов | 3.4 GB | **3.5 KB** |
| Параметры внимания | 4·D² (3.2M) | **D·K (3.6K)** |
| Edge-ready | Нет (KV-cache) | **Да** |
| Инференс на телефоне | Нет | **Да** |

---

## Анализатор модели

```bash
# ASCII-отчёт с гейтами, Cayley, нормами, adaptive depth
python analyze_model.py

# HTML-отчёт (открыть в браузере)
python analyze_model.py --html

# Указать чекпоинт
python analyze_model.py --ckpt path/to/model.pt --html
```

Вывод включает:
- Gate composition per layer (с цветными барами)
- Cayley orth error и эффективный ранг
- Adaptive depth pass rate
- Hidden state dynamics
- Parameter breakdown

---

## Файлы проекта

| Файл | Назначение |
|------|-----------|
| `ld_model/core.py` | LDConfig, fibonacci_roots, random_orthogonal, CausalConv1d, LDBlock, BottleneckMLP, LDStack |
| `train_phase2.py` | Тренировка Phase 2 (12×896, grad accum, Cayley, adaptive_depth, |A+B| логирование) |
| `analyze_model.py` | Per-layer анализатор (ASCII + HTML) |
| `generate.py` | Генерация русского текста из чекпоинта |
| `colab_phase2_ru.ipynb` | Colab notebook (Russian, mmap, auto-resume, fp16 ckpt) |
| `test_fresh_start.py` | Тест fresh start (5000 шагов, отдельное окно) |
| `LAMBDA_ARCHITECTURE.md` | Полная документация архитектуры (рус.) |
| `ROADMAP.md` | Пофазный roadmap |
| `SUMMARY.md` | Краткая сводка экспериментов |

---

## История: ZeckendorfReadout (эксперимент 2026-07)

Пытались заменить lm_head (44.8M) на древовидный декодер Фибоначчи (86K).
На замороженном stack Zeckendorf побеждал (PPL 1,435 vs 32,652), но при
co-training градиенты через ZK оказались шумны → stack уходил в сторону.
**Закрыт.** lm_head — 0.04% модели — оставлен.

---

## Запуск

```bash
# Локально (MX550/T4)
python train_phase2.py --data russian

# Colab (T4, mmap, auto-resume)
colab_phase2_ru.ipynb

# Анализ
python analyze_model.py --html

# Генерация
python generate.py --ckpt checkpoints/model_step30000.pt
```

Авто-resume: находит последний `model_step*.pt` и продолжает.

---

*Подробнее: [LAMBDA_ARCHITECTURE.md](LAMBDA_ARCHITECTURE.md) — математика,
[SUMMARY.md](SUMMARY.md) — эксперименты, [ROADMAP.md](ROADMAP.md) — планы.*
