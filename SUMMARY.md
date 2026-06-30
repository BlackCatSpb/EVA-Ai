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

## Архитектура (кратко)

- **CausalConv1d**: depthwise, kernel=4, frozen — локальные n-граммы
- **LDBlock**: rms_norm → V·diag(λ_eff)·V^T, content-dependent gating
- **BottleneckMLP**: D→256→D, SiLU, trainable
- **Выход**: lm_head (96M на обучение, ~50M на инференс с ZeckendorfReadout)

## Файлы

| Файл | Назначение |
|------|-----------|
| `ld_model/core.py` | CausalConv1d, LDBlock, BottleneckMLP, LDStack |
| `ld_model/readout.py` | ZeckendorfReadout (древовидный декодер) |
| `train_phase2.py` | Тренировка (12×896, grad accum, warmup, entropy logging) |
| `colab_train.py` | Colab/T4 скрипт |
| `distill_zeckendorf.py` | Дистилляция ZeckendorfReadout поверх обученного ствола |
| `monitor.py` | GPU/CPU/RAM монитор |
| `colab_phase2.ipynb` | Colab notebook |
| `LAMBDA_ARCHITECTURE.md` | Полная документация (рус.) |

## Сравнение

| Аспект | Transformer | Mamba | λ_d |
|--------|-------------|-------|-----|
| Память контекста | KV-cache O(L×D) | O(D) | **O(D) — 3.5 KB** |
| Длина контекста | 8K-128K | 128K+ | **∞ (RNN)** |
| Параметры внимания | 4·D² | D·Δ | **D·K (K=4..6)** |
| Спектр | Фиксирован | HIPPO | **α(h)-зависимый, блоковый** |
| Инференс (1.3B) | 100M+ lm_head | 100M+ | **~50M c Zeckendorf** |
| Edge-ready | Нет (KV-cache) | Частично | **Да** |

## Next: Phase 3 (36×2560, 1.3B, 50B tok)
