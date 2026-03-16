#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Smoke-тест загрузки и генерации для русских моделей на конфигурациях с ограниченной VRAM.
Модели:
- ruGPT3 Small: sberbank-ai/rugpt3small_based_on_gpt2
- Qwen (многоязычная, поддерживает русский): Qwen/Qwen2.5-0.5B-Instruct

Особенности:
- Автовыбор устройства (CUDA/CPU)
- Осторожное использование памяти (device_map="auto", fp16 на CUDA)
- Попытка 8-битного квантования при очень малой VRAM, если доступен bitsandbytes
- Для Qwen: trust_remote_code=True и безопасный вызов tie_weights()

Запуск:
  python scripts/smoke_russian_models.py --model both
  python scripts/smoke_russian_models.py --model rugpt
  python scripts/smoke_russian_models.py --model qwen
"""
from __future__ import annotations
import os
import argparse
import time
from typing import Optional

# Важно: улучшает поведение аллокатора CUDA
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

RUGPT_ID = "sberbank-ai/rugpt3small_based_on_gpt2"
QWEN_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def _gpu_info_gb() -> float:
    try:
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()  # type: ignore
            return float(free_b) / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def _maybe_quant_kwargs() -> dict:
    """Если VRAM < 1 ГБ и есть bitsandbytes — вернуть quantization_config."""
    kw = {}
    free_gb = _gpu_info_gb()
    if free_gb > 0.0 and free_gb < 1.0:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            kw["device_map"] = "auto"
        except Exception:
            # нет bitsandbytes — хотя бы device_map=auto
            kw["device_map"] = "auto"
    return kw


def _common_load_kwargs(trust_remote_code: bool = False) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {
        "device_map": "auto" if device == "cuda" else None,
        "trust_remote_code": trust_remote_code,
    }
    if device == "cuda":
        kw["torch_dtype"] = torch.float16
    else:
        kw["low_cpu_mem_usage"] = True
    kw.update(_maybe_quant_kwargs())
    return kw


def load_rugpt3_small() -> tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    tokenizer = AutoTokenizer.from_pretrained(RUGPT_ID)
    model = AutoModelForCausalLM.from_pretrained(RUGPT_ID, **_common_load_kwargs())
    prompt = (
        "Ты — умный помощник. Ответь кратко на русском.\n\n"
        "Вопрос: Чем Санкт-Петербург отличается от Москвы?\nОтвет:"
    )
    return model, tokenizer, prompt


def load_qwen_small() -> tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(QWEN_ID, **_common_load_kwargs(trust_remote_code=True))
    # tie_weights для некоторых Qwen-вариантов
    try:
        if hasattr(model, "tie_weights"):
            model.tie_weights()  # type: ignore
    except Exception:
        pass
    prompt = (
        "Ты — умный ассистент. Отвечай по-русски кратко и по делу.\n\n"
        "Вопрос: Назови три главные реки России.\nОтвет:"
    )
    return model, tokenizer, prompt


def generate(model, tokenizer, prompt: str) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen = model.generate(
        **inputs,
        max_new_tokens=48,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(gen[0], skip_special_tokens=True)
    return text


def run(model_kind: str) -> None:
    kinds = [model_kind] if model_kind != "both" else ["rugpt", "qwen"]
    for kind in kinds:
        t0 = time.time()
        print(f"\n=== Загрузка модели: {kind} ===")
        if kind == "rugpt":
            model, tok, prompt = load_rugpt3_small()
        elif kind == "qwen":
            model, tok, prompt = load_qwen_small()
        else:
            raise ValueError("model_kind must be one of: rugpt, qwen, both")
        t1 = time.time()
        print(f"Загрузка заняла: {t1 - t0:.2f} c")

        print("Генерация...")
        out = generate(model, tok, prompt)
        print("--- Результат ---")
        print(out)

        # Очистка CUDA-кэша для снижения фрагментации после генерации
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rugpt", "qwen", "both"], default="both")
    args = parser.parse_args()
    run(args.model)
