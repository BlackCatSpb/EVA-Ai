#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Импортирует данные из каталога HuggingFace ruGPT (или совместимого) в фрактальное хранилище графа памяти.
Фокус: без запуска большой модели, используем только файлы токенайзера и конфига.

- Узлы: токены (`node_type='token'`), конфиг (`node_type='config'`)
- Рёбра: BPE-мерджи, связи конфиг->токен (опц.)

Пример:
  python scripts/import_hf_rugpt_to_fractal.py \
    --hf-dir hf_cache/hub/models--sberbank-ai--ruGPT3Large \
    --out-dir ml_cache/models/fractal_rugpt \
    --model-id ruGPT3Large \
    --graph-name rugpt_tokenizer
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

# Гарантируем, что корень проекта доступен в PYTHONPATH при запуске скрипта напрямую
try:
    PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
except Exception:
    pass

from cogniflex.mlearning.storage.fractal_store import FractalWeightStore

logger = logging.getLogger("cogniflex.scripts.import_hf_rugpt_to_fractal")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_tokenizer(hf_dir: Path) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Возвращает (tokens, merges) из tokenizer.json или пары (vocab.json, merges.txt).
    merges — список пар символов (a,b). Если merges недоступны, вернём пустой список.
    """
    tok_json = hf_dir / "tokenizer.json"
    tok_cfg_json = hf_dir / "tokenizer_config.json"
    vocab_json = hf_dir / "vocab.json"
    merges_txt = hf_dir / "merges.txt"

    tokens: List[str] = []
    merges: List[Tuple[str, str]] = []

    # Если есть tokenizer_config.json с указанием tokenizer_file — используем его
    if (not tok_json.exists()) and tok_cfg_json.exists():
        cfg = _read_json(tok_cfg_json)
        tok_file = cfg.get("tokenizer_file") or cfg.get("tokenizer_json")
        if isinstance(tok_file, str):
            cand = (hf_dir / tok_file) if not os.path.isabs(tok_file) else Path(tok_file)
            if cand.exists():
                tok_json = cand

    if tok_json.exists():
        data = _read_json(tok_json)
        # Пробуем токен-лист по стандарту Tokenizers
        try:
            toks = data.get("model", {}).get("vocab") or data.get("vocab")
            if isinstance(toks, dict):
                # словарь token->id
                def _to_int(v: Any) -> int:
                    try:
                        return int(v)
                    except Exception:
                        return 10**9
                tokens = sorted(toks, key=lambda k: _to_int(toks[k]))
            elif isinstance(toks, list):
                tokens = [str(t) for t in toks]
            # Добавляем добавленные токены, если они есть в структуре
            added = data.get("added_tokens")
            if isinstance(added, list):
                for item in added:
                    if isinstance(item, dict) and "content" in item:
                        tokens.append(str(item["content"]))
        except Exception:
            pass
        # merges
        try:
            merges_list = data.get("model", {}).get("merges") or data.get("merges") or []
            for m in merges_list or []:
                if isinstance(m, str):
                    parts = m.split()
                    if len(parts) == 2:
                        merges.append((parts[0], parts[1]))
        except Exception:
            pass
        # Если merges не нашлись в tokenizer.json — попробуем merges.txt рядом
        if not merges and merges_txt.exists():
            try:
                with merges_txt.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split()
                        if len(parts) == 2:
                            merges.append((parts[0], parts[1]))
            except Exception:
                pass
        return tokens, merges

    # Fallback: vocab.json + merges.txt (BPE)
    if vocab_json.exists():
        voc = _read_json(vocab_json)
        if isinstance(voc, dict):
            # сортируем по id, если возможно
            try:
                tokens = sorted(voc.keys(), key=lambda k: int(voc[k]))
            except Exception:
                tokens = list(voc.keys())
        logger.info(f"Loaded {len(tokens)} tokens from vocab.json")
    if merges_txt.exists():
        try:
            with merges_txt.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) == 2:
                        merges.append((parts[0], parts[1]))
            logger.info(f"Loaded {len(merges)} merges from merges.txt")
        except Exception:
            pass
    return tokens, merges


def _build_kg_from_tokenizer(tokens: List[str], merges: List[Tuple[str, str]], config: Dict[str, Any], limit_tokens: int | None = None) -> Dict[str, Any]:
    """
    Собирает минимальный граф знаний из токенов/мерджей и конфига.
    Структура: {"nodes": [...], "edges": [...]}
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    # Обрежем при необходимости, чтобы экономить память
    if limit_tokens is not None and limit_tokens > 0:
        tokens = tokens[:limit_tokens]

    # Узлы-токены
    for t in tokens:
        nid = f"tok::{t}"
        nodes.append({
            "id": nid,
            "node_type": "token",
            "content": t,
        })

    # Узел-конфиг (усечённый для компактности)
    cfg_small: Dict[str, Any] = {}
    for k in ("model_type", "vocab_size", "n_positions", "n_ctx", "n_embd", "n_layer", "n_head"):
        if k in config:
            cfg_small[k] = config[k]
    nodes.append({
        "id": "config::hf_model",
        "node_type": "config",
        "content": cfg_small or {"model": config.get("model_type", "hf")},
    })

    # Рёбра-мерджи: простой граф парных связей
    # Ограничим количество рёбер, чтобы не взорваться на памяти
    max_merges = min(len(merges), 200_000)  # достаточно для структуры
    for a, b in merges[:max_merges]:
        src = f"tok::{a}"
        dst = f"tok::{b}"
        edges.append({
            "source": src,
            "target": dst,
            "relation_type": "merge",
        })

    # Рёбра конфиг -> некоторые частотные токены (первые N)
    for t in tokens[: min(128, len(tokens))]:
        edges.append({
            "source": "config::hf_model",
            "target": f"tok::{t}",
            "relation_type": "defines",
        })

    return {"nodes": nodes, "edges": edges}


def import_hf_to_fractal(hf_dir: str, out_dir: str, model_id: str, graph_name: str | None = None, limit_tokens: int | None = None) -> str:
    """
    Основной процесс импорта: читаем HF токенайзер/конфиг, строим KG, упаковываем во фрактал и сохраняем атомарно.
    Возвращает путь к каталогу сохранения.
    """
    hf_path = Path(hf_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1) Загрузка токенов/мерджей и конфига
    tokens, merges = _load_tokenizer(hf_path)
    config = _read_json(hf_path / "config.json")

    if not tokens:
        logger.warning("Не удалось загрузить токены из токенайзера HF. Импорт продолжится только с узлом конфига.")

    # 2) Построение минимального графа
    kg = _build_kg_from_tokenizer(tokens, merges, config, limit_tokens=limit_tokens)

    # 3) Упаковка в фрактальное хранилище
    store = FractalWeightStore()
    # ВАЖНО: установить model_id до упаковки, чтобы id контейнеров корректно маркировались
    store.model_id = model_id
    store.pack_knowledge_graph(kg)

    # 4) Валидация упаковки
    try:
        validation = store.validate_knowledge_graph_packing()
        if not validation.get("ok", False):
            logger.warning("Валидация упаковки графа: проблемы=%s", validation.get("issues"))
    except Exception as e:
        logger.warning("Валидация упаковки недоступна или завершилась с исключением: %s", e)

    # 5) Сохранение атомарно
    # 5) Сохранение атомарно (на директорию)
    report = store.save_to_disk_atomic(str(out_path))
    if not report.get("ok"):
        raise RuntimeError(f"Ошибка сохранения фрактального стора: {report.get('error')}")
    saved_path = report.get("path", str(out_path))

    logger.info("Импорт HF->Fractal завершён. Сохранено в: %s (checksum=%s)", saved_path, report.get("checksum"))
    return saved_path


def main() -> None:
    p = argparse.ArgumentParser(description="Импорт HuggingFace ruGPT в фрактальный граф памяти (токенайзер+конфиг)")
    p.add_argument("--hf-dir", required=True, help="Каталог модели HF (содержит tokenizer.json или vocab.json/merges.txt, config.json)")
    p.add_argument("--out-dir", required=True, help="Каталог для сохранения фрактального стора")
    p.add_argument("--model-id", required=True, help="Идентификатор модели (используется в метаданных и именах контейнеров)")
    p.add_argument("--graph-name", default=None, help="Метка/имя графа для сохранения")
    p.add_argument("--limit-tokens", type=int, default=None, help="Ограничить число импортируемых токенов (для экономии памяти)")
    args = p.parse_args()

    import_hf_to_fractal(
        hf_dir=args.hf_dir,
        out_dir=args.out_dir,
        model_id=args.model_id,
        graph_name=args.graph_name,
        limit_tokens=args.limit_tokens,
    )


if __name__ == "__main__":
    main()
