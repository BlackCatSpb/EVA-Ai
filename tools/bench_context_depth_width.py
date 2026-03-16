#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark: Context Understanding (Width & Depth) — Baseline vs CogniFlex

- Width: извлечение отдельных фактов из длинного контекста (много несвязанных разделов)
- Depth: многошаговые вопросы с межразделочными ссылками (coref, 2-3 хопа)

Режимы:
- baseline:
    * Попытка использовать transformers (если установлены и есть локальная модель)
    * Иначе: эвристический rule-based baseline (поиск ответа по ключам)
- cogniflex:
    * CoreBrain -> ResponseGenerator, контекст подставляется в prompt
    * Флаги: --use-knowledge, --use-websearch, --safe-test-mode

Метрики:
- accuracy_exact (точное совпадение ответа), accuracy_sub (подстрока)
- latency p50/p95 (на запрос)

Usage (PowerShell examples):
  # Синтетический датасет 20/20 (width/depth), без внешней сети
  $env:TRANSFORMERS_OFFLINE="1"; $env:HF_HUB_OFFLINE="1"
  python tools/bench_context_depth_width.py --width 20 --depth 20 --batch 1 \
    --model-path "c:\\Users\\black\\OneDrive\\Desktop\\CogniFlex\\cogniflex_models\\qwen25_1p5b_instruct" \
    --safe-test-mode 1 --use-knowledge 0 --use-websearch 0 > bench_ctx_wd_safe.json

  # Тот же тест в полном режиме (когда будут веса)
  python tools/bench_context_depth_width.py --width 20 --depth 20 --batch 1 \
    --model-path "c:\\...\\qwen25_1p5b_instruct" --safe-test-mode 0 > bench_ctx_wd_full.json
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from typing import List, Dict, Any, Tuple

# Ensure project root on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Optional transformers
AutoTokenizer = None
AutoModelForCausalLM = None
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
except Exception:
    pass

# CogniFlex
CoreBrain = None
try:
    from cogniflex.core.core_brain import CoreBrain as _CB  # type: ignore
    CoreBrain = _CB
except Exception as e:
    print(f"[bench-ctx] Warn: cannot import CoreBrain: {e}")


def make_width_dataset(n: int) -> Tuple[str, List[Tuple[str, str]]]:
    """Создаёт один длинный документ с n фактами и вопросы к каждому факту.
    Returns: (document, [(question, answer), ...])
    """
    facts = []
    qas: List[Tuple[str, str]] = []
    chunks = []
    for i in range(n):
        city = f"Город-{i}"
        river = f"Река-{i}"
        pop = 100000 + i * 7
        fact = f"{city} расположен на берегу реки {river}. Население {city} составляет {pop} человек."
        chunks.append(f"### Раздел {i}\n{fact}\n")
        qas.append((f"Как называется река в городе {city}?", river))
    document = "\n\n".join(chunks)
    return document, qas


def make_depth_dataset(n: int) -> Tuple[str, List[Tuple[str, str]]]:
    """Создаёт многошаговые зависимости между персонажами/городами.
    Пример: Алиса живёт в Город-7, Город-7 находится в стране Z, столица страны Z — Омега. Вопрос: столица страны проживания Алисы? Ответ: Омега.
    """
    names = ["Алиса", "Борис", "Виктор", "Галина", "Дмитрий", "Елена", "Жанна", "Захар", "Ирина", "Константин"]
    countries = ["Страна-X", "Страна-Y", "Страна-Z", "Страна-W"]
    capitals = {"Страна-X": "Капиталис", "Страна-Y": "Альфа", "Страна-Z": "Омега", "Страна-W": "Сигма"}

    chunks = []
    qas: List[Tuple[str, str]] = []
    for i in range(n):
        name = names[i % len(names)] + f"-{i}"
        city = f"Город-{(i*3)%37}"
        country = countries[i % len(countries)]
        capital = capitals[country]
        s1 = f"{name} проживает в {city}."
        s2 = f"{city} расположен в государстве {country}."
        s3 = f"Столицей государства {country} является {capital}."
        s4 = f"В {city} протекает река Ре-{i}."
        chunks.append(f"### История {i}\n{s1} {s2} {s3} {s4}\n")
        q = f"Как называется столица страны проживания {name}?"
        qas.append((q, capital))
    document = "\n\n".join(chunks)
    return document, qas


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def score_answer(pred: str, gold: str) -> Tuple[bool, bool]:
    p = normalize_text(pred)
    g = normalize_text(gold)
    exact = p == g
    sub = (g in p) or (p in g) or any(w in p for w in g.split())
    return exact, sub


def baseline_answer(tokenizer, model, question: str, context: str) -> str:
    # If model available, do simple concat prompting
    if tokenizer is not None and model is not None:
        prompt = f"Вопрос: {question}\nКонтекст: {context}\nКраткий ответ:"
        try:
            input_ids = tokenizer(prompt, return_tensors="pt")
            out = model.generate(**input_ids, max_new_tokens=32, do_sample=False)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            resp = text.split("Краткий ответ:")[-1].strip()
            return resp
        except Exception:
            pass
    # Fallback: rule-based (search gold-like spans by simple heuristics)
    # Look for nearest capitalized phrase near keywords
    m = re.search(r"столицей.*?является\s+([A-ЯA-ZЁ][^\s,.!?]+)", context)
    if m:
        return m.group(1)
    m = re.search(r"реки\s+([A-ЯA-ZЁ][^\s,.!?]+)", context)
    if m:
        return m.group(1)
    # last resort: first capitalized token
    cap = re.findall(r"\b[А-ЯЁ][а-яёA-ЯЁA-Za-z0-9\-]+\b", context)
    return cap[0] if cap else ""


def cogniflex_answer(brain, question: str, context: str) -> str:
    try:
        prompt = (
            "Ты аналитик. Прочитай контекст и ответь кратко.\n" \
            + "Контекст:" + "\n" + context + "\n" \
            + "Вопрос: " + question + "\nКраткий ответ:"
        )
        rg = getattr(brain, "response_generator", None)
        if rg is None:
            return ""
        resp = rg.generate_response(prompt=prompt, max_length=48, temperature=0.0, top_p=0.9, task="text-generation")
        txt = (resp or {}).get("text", "").strip()
        # post-trim to answer part
        if "Краткий ответ:" in txt:
            txt = txt.split("Краткий ответ:")[-1].strip()
        return txt
    except Exception:
        return ""


def run_suite(mode: str, items: List[Tuple[str, str]], context: str, model_path: str, brain) -> Dict[str, Any]:
    lat = []
    exact_hits = 0
    sub_hits = 0
    tokenizer = None
    model = None
    if mode == "baseline" and AutoTokenizer is not None and AutoModelForCausalLM is not None and model_path:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        except Exception as e:
            print(f"[bench-ctx] baseline HF load failed: {e}")
            tokenizer = None
            model = None

    for (q, gold) in items:
        t0 = time.perf_counter()
        if mode == "baseline":
            pred = baseline_answer(tokenizer, model, q, context)
        else:
            pred = cogniflex_answer(brain, q, context)
        lat.append(time.perf_counter() - t0)
        ex, sb = score_answer(pred, gold)
        exact_hits += int(ex)
        sub_hits += int(sb)

    res = {
        "count": len(items),
        "p50": statistics.median(lat) if lat else 0.0,
        "mean": statistics.fmean(lat) if lat else 0.0,
        "p95": statistics.quantiles(lat, n=20)[18] if len(lat) >= 20 else (max(lat) if lat else 0.0),
        "exact": exact_hits,
        "exact_acc": exact_hits / max(1, len(items)),
        "substr": sub_hits,
        "substr_acc": sub_hits / max(1, len(items)),
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=20)
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1, help="зарезервировано")
    ap.add_argument("--model-path", type=str, default=None)
    ap.add_argument("--safe-test-mode", type=int, default=1)
    ap.add_argument("--use-knowledge", type=int, default=0)
    ap.add_argument("--use-websearch", type=int, default=0)
    args = ap.parse_args()

    # Prepare documents
    width_doc, width_qas = make_width_dataset(args.width)
    depth_doc, depth_qas = make_depth_dataset(args.depth)

    # Baseline
    out: Dict[str, Any] = {"width": {}, "depth": {}}
    out["width"]["baseline"] = run_suite("baseline", width_qas, width_doc, args.model_path, brain=None)
    out["depth"]["baseline"] = run_suite("baseline", depth_qas, depth_doc, args.model_path, brain=None)

    # CogniFlex init
    brain = None
    if CoreBrain is not None:
        # Respect flags via env for safe mode
        if args.safe_test_mode:
            os.environ["COGNIFLEX_TEST_MODE"] = "1"
            os.environ["COGNIFLEX_SAFE_TEST_MODE"] = "1"
        else:
            os.environ["COGNIFLEX_TEST_MODE"] = "0"
            os.environ["COGNIFLEX_SAFE_TEST_MODE"] = "0"
        try:
            brain = CoreBrain()
            # Optionally toggle features here if needed
            # e.g., brain.config.update({"use_knowledge": bool(args.use_knowledge), ...}) if supported
            brain.initialize()
        except Exception as e:
            print(f"[bench-ctx] CoreBrain init failed: {e}")
            brain = None

    # CogniFlex runs
    if brain is not None:
        out["width"]["cogniflex"] = run_suite("cogniflex", width_qas, width_doc, args.model_path, brain)
        out["depth"]["cogniflex"] = run_suite("cogniflex", depth_qas, depth_doc, args.model_path, brain)
    else:
        out["width"]["cogniflex"] = {"status": "skip", "reason": "brain init failed"}
        out["depth"]["cogniflex"] = {"status": "skip", "reason": "brain init failed"}

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
