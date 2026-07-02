# λ_d — Content-Dependent Spectral Language Model

## Goal
Self-contained language model with O(1) memory context, based on content-dependent spectral operator A(h) = V·diag(α★λ)·V⁻¹, using Fibonacci spectrum + causal convolution + bottleneck MLP.

## Phase 2 — Results (Confirmed)

**Architecture validated. 12-layer LDStack, D=896, K=4, 95.1M params, trained from scratch on 50K chunks wikitext-103.**

| Metric | Epoch 1 | Epoch 2 (start) |
|--------|---------|-----------------|
| Train PPL | **175.3** | **77.5** |
| Eval PPL | **149.7** | — |
| NaN count | 0 | 0 |
| Stack norm range | ±6.3 | ±6.3 (stable) |

**Ключевые выводы:**
- Eval PPL (149.7) < Train PPL (175.3) — модель обобщает, нет переобучения
- Монотонная кривая без осцилляций (grad accum + warmup)
- Гомеостаз норм скрытых состояний (±6.3) — спектральное пространство стабильно
- Ни одного NaN за всю эпоху (fp32, MX550)
- Архитектура подтверждена: λ_d — работающая языковая модель

## Phase 2 Russian — In Progress

**12-layer LDStack, D=896, K=4, 95.1M, Russian Wikipedia + books (844M tok, 6.6M chunks).**

| Step | Loss | PPL | lr |
|------|------|-----|----|
| 20200 | 5.01 | 150.1 | 1e-3 |
| 25000 | 4.93 | 137.9 | 9.97e-4 |
| 29000 | 4.88 | 132.1 | 9.95e-4 |

**Генерация (step 25000, temp=0.8, top_k=40):**
```
Привет, как дела?..
В городе действует ещё 7,5 млн м³.
В результате этого было сказано, что в конечном счете, как это был ещё в 1999 г. в США,
в возрасте 15 лет. После этого она начала выступать в чемпионате мира в рамках Кубка мира.
В том сезоне 2012/13 стал победителем Лиги чемпионов УЕФА.
В августе 2010-х годов в Москве.
Прибыв на могиле установлен памятник.
Действие происходит в Москве.
```

**Gate analysis (step 25000):** Энтропия гейтов 0.151 (max=1.386), слои сильно специализированы:
- Layer 5: Mode 3 at 99.6% (полностью захвачен одной модой)
- Layer 6: Mode 3 at 95.7%
- Layer 9: Mode 0 at 92.4%
- Layer 7: Mode 0 at 42.0% (самый равномерный)

V-energy anisotropy: ratio 1.3e9 в Layer 11 — сигнал сконцентрирован в нескольких направлениях.

**Extrapolation test (step 25000):**

| Length | PPL | Degradation |
|--------|-----|-------------|
| 128 | 43.3 | baseline |
| 256 | 63.4 | +46% |
| 512 | 99.1 | +129% |
| 1024 | 116.7 | +169% |

Модель generalizes beyond training length (128), но PPL растёт — мотивация для HierarchyLD.

## Эксперименты

### 6. Token importance → adaptive depth
Gate spread (σ(α) = std of mode weights) — zero-cost salience signal.

| Уровень | Spread | Режим | Доминантная мода |
|:-------:|:------:|-------|:----------------:|
| L0 (синтаксис) | < 0.08 | Равномерное распределение | m1/m4 (~40% каждая) |
| L1 (модификаторы) | 0.08-0.15 | Слабый выбор | m4 ~33% |
| L2 (конкретные) | 0.15-0.25 | Уверенный выбор | m4 ~41% |
| L3 (абстрактные) | > 0.25 | Решительный выбор | m4 ~60% (93% доминанта) |

Adaptive depth routing: `h = σ(α)·h_mlp + (1-σ(α))·h` (soft, trainable) / `h = h_mlp if σ(α) > τ` (hard, inference). Обучаемые пороги τₗ, инициализированы 0.25→0.45. На чекпоинте: 98% токенов проходят L0, 59% L7, 30% L10.

### 7. Learnable V (low-rank delta)
V_eff = V_frozen + U·Vᵀ (r=8), norm clip 0.3.

Косинусная симметрия V_delta между слоями (200 шагов тренировки, D=256):
```
    L0    L1    L2    L3    L4    L5
L0 1.00  0.20  0.35  0.38  0.49  0.62  ← поздние конвергируют
L1 0.20  1.00  0.35  0.47  0.47  0.58
L5 0.62  0.58  0.60  0.65  0.56  1.00  ← ранние дивергируют
```

Эффективный ранг: L0=4.4, L5=1.5 (при r=16). Снижен до r=8. 
Все слои упёрлись в норму 0.1 — поднято до 0.3.

### 8. Pipeline integration
- adaptive_depth=True, learnable_V=True по умолчанию
- clip_v_delta() после optimizer.step()
- |Vd| логирование в шагах
- Обратная совместимость: load_state_dict(strict=False)

### 1. Spectral gates vs Learned gates
- Spectral energy gates (`alpha_k = ||V_k^T h||^2 / sum`): PPL 125.5 → 413.0 (3.3x хуже)
- L∞ variant (tau=5): PPL 3967.7 (полный коллапс)
- Причина: frozen V изотропен на входе, gates не могут построить анизотропию
- Вывод: learned gates необходимы, они = attention head routing

### 2. Dimensionality sweep
| D | Params | PPL@200 steps | Статус |
|---|--------|---------------|--------|
| 512 | 54M | 140.1 | OK |
| 768 | 82M | 74.7 | OK |
| 896 | 95M | 56.7 | OK (with ckpt) |
| 1024 | 109M | 52.1 | OK |
| 1536 | 163M | 48.8 | OK |
| 2048 | 217M | ~3883@50 | OK (OOM на 200) |

Все размерности стабильны, 0 NaN.

### 3. 36-layer forward test
- 106.2M params (всего +11.7M за +24 слоя — V frozen)
- VRAM: 875MB (batch=1)
- Forward: 52.4ms/tok (vs 19ms для 12-layer)
- NaN/Inf: 0, Grad norm: 3.74
- **Вывод**: 36 слоёв стабильны, влезут на T4

### 4. HierarchyLD forward test
- 3 уровня (8+6+4 слоя, D=2048)
- 244.8M params
- Forward L=4096 на CPU: 6 сек
- Train: loss=10.97, grad_norm=2.13
- NaN: 0
- **Вывод**: HierarchyLD работает, можно обучать

### 5. V-energy anisotropy
- Embedding: ratio 1.33 (почти изотропно)
- Layer 11: ratio 1.3e9 (экстремально анизотропно)
- Вывод: gates строят структуру в V-базисе через bootstrapping

## Решённые проблемы

| Проблема | Решение |
|----------|---------|
| Identity LDBlock (скалярный λ_eff + clamping) | Блоковый λ_eff (вектор D), без clamping |
| Ранг 16 в LoRA MLP (16/4864 = 0.3%) | Dense bottleneck 896→256→896, полный ранг |
| Позиционная слепота (bag-of-words) | Causal Conv1d kernel=4 в каждом LDBlock |
| Высокий шум градиента (batch=4) | Gradient accumulation ×8 (eff batch=32) |
| Разрушение начальных весов | Linear warmup 5% от total steps |
| TDR (GPU timeout) | Registry: TdrDelay = 10s |
| Crash при save_checkpoint | Убран scheduler.state_dict() из сохранения |
| FCF (VSA bind/unbind) не работает | Заменён на HierarchyLD (стек LDBlock + Linear compression) |
| mean_pool теряет информацию | Опционально: Cross-Attention / λ_d-based pooling |

## Архитектура (кратко)

- **CausalConv1d**: depthwise, kernel=4, frozen — локальные n-граммы
- **LDBlock**: rms_norm → V·diag(λ_eff)·V^T, content-dependent gating
  - V = V_frozen + U·Vᵀ (learnable low-rank delta, r=8, norm clip 0.3)
- **BottleneckMLP**: D→256→D, SiLU, trainable
- **Adaptive depth**: gate spread → per-token routing (soft/hard)
- **Выход**: lm_head (96M на обучение, ~50M на инференс с ZeckendorfReadout)

## Файлы

| Файл | Назначение |
|------|-----------|
| `ld_model/core.py` | CausalConv1d, LDBlock, BottleneckMLP, LDStack |
| `ld_model/readout.py` | ZeckendorfReadout (древовидный декодер) |
| `train_phase2.py` | Тренировка (12×896, grad accum, warmup, entropy logging) |
| `colab_train.py` | Colab/T4 скрипт |
| `distill_zeckendorf.py` | Дистилляция ZeckendorfReadout поверх обученного ствола |
| `colab_phase2_ru.ipynb` | Colab notebook (Russian, mmap, auto-resume, fp16 ckpt) |
| `experiment_spectral_gates.py` | Сравнение spectral vs learned gates |
| `experiment_gate_analysis.py` | Gate entropy + V-energy anisotropy анализ |
| `experiment_extrapolation.py` | Context extrapolation тест |
| `test_dimensionality.py` | D-свап тест (D=512..2048) |
| `test_generate.py` | Генерация текста из чекпоинта |
| `LAMBDA_ARCHITECTURE.md` | Полная документация (рус.) |
| `ROADMAP.md` | 4-фазный roadmap (скорректированный после аудита) |
| `WHITEPAPER.md` | 11-секционный whitepaper |

## Сравнение

| Аспект | Transformer | Mamba | λ_d |
|--------|-------------|-------|-----|
| Память контекста | KV-cache O(L×D) | O(D) | **O(D) — 3.5 KB** |
| Длина контекста | 8K-128K | 128K+ | **∞ (RNN)** |
| Параметры внимания | 4·D² | D·Δ | **D·K (K=4..6)** |
| Спектр | Фиксирован | HIPPO | **α(h)-зависимый, блоковый** |
| Инференс (1.3B) | 100M+ lm_head | 100M+ | **~50M c Zeckendorf** |
| Edge-ready | Нет (KV-cache) | Частично | **Да** |

## Next: Phase 2.5 → Phase 3

1. **Phase 2.5**: Scaling law sweep (D=512/768/896/1024) + Transformer baseline → нужно RTX 3090
2. **Phase 3.0**: HierarchyLD 3-level (8+6+4, D=2048) — код готов, forward tested
3. **Phase 3.5**: HierarchyLD 1.2B, ∞ context — 4×A100, ~$5K
4. **Phase 4**: 2.5B — 8×A100, ~$15K (скорректировано после аудита FLOPs)
