#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Драйвер-сценарий для безопасного сессионного переноса и инкрементального сохранения
весов ruGPT small (или совместимой GPT2-подобной модели) в фрактальную структуру.

Возможности:
- Загрузка модели из локального HF-кэша (или указанного каталога модели)
- Упаковка весов в `FractalWeightStore.pack_model_weights()`
- Инкрементальное сохранение с поддержкой resume, прогресс-бара и лимита контейнеров за сессию
- Опциональная очистка временных артефактов: мягкая и полная
- Автонастройка batch_size при необходимости

Примеры:
  # 1) Обычный запуск с прогресс-баром и возобновлением, лимит 15000 контейнеров за сессию
  python scripts/rugpt_session_driver.py \
    --hf-model-dir hf_cache/hub/models--sberbank-ai--rugpt3small_based_on_gpt2 \
    --out-dir ml_cache/models/fractal_rugpt \
    --model-id ruGPT3Small \
    --resume \
    --max-items-per-session 15000 \
    --batch-size 6000 \
    --min-batch-size 1025 \
    --progress

  # 2) Мягкая очистка артефактов незавершённого сохранения
  python scripts/rugpt_session_driver.py --out-dir ml_cache/models/fractal_rugpt --soft-clean

  # 3) Полная очистка (начать сначала)
  python scripts/rugpt_session_driver.py --out-dir ml_cache/models/fractal_rugpt --fresh-clean
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time
from typing import Optional

# Гарантируем PYTHONPATH проекта при прямом запуске
try:
    PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
except Exception:
    pass

from cogniflex.mlearning.storage.fractal_store import FractalWeightStore

logger = logging.getLogger("cogniflex.scripts.rugpt_session_driver")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")


def _load_model_from_dir(model_dir: str):
    """Загружает модель из локального каталога HF. Требует transformers.
    Возвращает (model, model_id_str).
    """
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
    except Exception as e:
        raise RuntimeError("Не найден пакет transformers. Установите его: pip install transformers") from e

    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Каталог модели не найден: {model_path}")

    # Конфиг для model_id
    cfg = None
    try:
        cfg = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    except Exception:
        cfg = None

    # Загрузка самой модели (локально)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    model.eval()

    # Предпочитаем название из конфига
    model_id = None
    if cfg is not None:
        # попытка взять полезный идентификатор
        for key in ("name_or_path", "model_type", "_name_or_path"):
            val = getattr(cfg, key, None)
            if isinstance(val, str) and val:
                model_id = val
                break
    if not model_id:
        model_id = model_path.name

    return model, str(model_id)


def run_session(
    hf_model_dir: Optional[str],
    out_dir: str,
    model_id: Optional[str],
    resume: bool,
    batch_size: Optional[int],
    min_batch_size: int,
    max_items_per_session: Optional[int],
    progress: bool,
    compress: bool,
    offload_after_write: bool,
    auto_batch: bool,
    auto_resume_loop: bool,
    max_cycles: Optional[int],
    sleep_between_seconds: float,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    store = FractalWeightStore()

    # Загрузка модели и упаковка весов
    if hf_model_dir:
        model, detected_id = _load_model_from_dir(hf_model_dir)
        mid = model_id or detected_id
        logger.info("Упаковка весов модели '%s'...", mid)
        ok = store.pack_model_weights(model, model_id=mid)
        if not ok:
            raise RuntimeError("Упаковка весов завершилась неуспешно")
    else:
        if not model_id:
            raise ValueError("Если не указан --hf-model-dir, необходимо явно указать --model-id")
        # Предполагаем, что store уже подготовлен пользователем до запуска, но на всякий случай проверим
        if not store.containers:
            raise RuntimeError("Store пуст. Укажите --hf-model-dir для загрузки модели или подготовьте store заранее.")

    # Автонастройка batch_size
    if auto_batch or (batch_size is None):
        guessed = store.auto_adjust_batch_size()
        if batch_size is None:
            batch_size = guessed
        else:
            # берём минимум для безопасности
            batch_size = min(batch_size, guessed)
        logger.info("Выбран batch_size=%d (auto)", batch_size)

    # Запуск инкрементального сохранения
    logger.info(
        "Старт инкрементального сохранения: out_dir=%s, resume=%s, batch_size=%s, min_batch_size=%d, max_items_per_session=%s, progress=%s, compress=%s, offload=%s",
        out_path,
        resume,
        batch_size,
        min_batch_size,
        max_items_per_session,
        progress,
        compress,
        offload_after_write,
    )
    try:
        cycles = 0
        while True:
            cycles += 1
            ok = store.save_to_disk_incremental(
                output_path=str(out_path),
                knowledge_graph=None,
                batch_size=int(batch_size),
                resume=bool(resume or cycles > 1),
                by_level=True,
                compress=bool(compress),
                state_filename="incremental_state.json",
                graph_manifest_records=None,
                graph_manifest_dir=None,
                max_items_per_session=max_items_per_session,
                show_progress=progress,
                min_batch_size=int(min_batch_size),
                offload_after_write=bool(offload_after_write),
            )
            if not ok:
                logger.error("Инкрементальное сохранение завершилось с ошибкой на цикле %d.", cycles)
                return False

            # Если state-файл удалён — работа завершена
            state_path = out_path / "incremental_state.json"
            if not state_path.exists():
                logger.info("Инкрементальное сохранение полностью завершено за %d цикл(ов).", cycles)
                return True

            # Если auto-loop выключен — выходим после первого успешного прерывания по лимиту
            if not auto_resume_loop:
                logger.info("Достигнут сессионный лимит и auto-resume-loop выключен — остановка после %d цикла.", cycles)
                return True

            # Проверяем лимит количества циклов
            if max_cycles is not None and cycles >= int(max_cycles):
                logger.info("Достигнут лимит циклов (%d). Останавливаемся, можно продолжить запуском скрипта снова.", max_cycles)
                return True

            # Переходим к следующей итерации
            logger.info("Переходим к следующему циклу (%d) через %.2f сек...", cycles + 1, float(sleep_between_seconds))
            try:
                time.sleep(float(sleep_between_seconds))
            except Exception:
                pass
    except KeyboardInterrupt:
        logger.warning("Процесс прерван пользователем. Состояние сохранено, можно возобновить с --resume.")
        return False


def main() -> None:
    p = argparse.ArgumentParser(description="Сессионный перенос ruGPT small в фрактальную структуру с инкрементальным сохранением")
    p.add_argument("--hf-model-dir", default=None, help="Каталог локальной HF-модели (например, hf_cache/hub/models--sberbank-ai--rugpt3small_based_on_gpt2)")
    p.add_argument("--out-dir", required=True, help="Каталог вывода для фрактальной структуры")
    p.add_argument("--model-id", default=None, help="Идентификатор модели для метаданных (если не указан, берётся из конфига HF)")

    # Режимы очистки
    grp_clean = p.add_mutually_exclusive_group()
    grp_clean.add_argument("--soft-clean", action="store_true", help="Мягкая очистка: удалить *.tmp и state, не трогая шарды")
    grp_clean.add_argument("--fresh-clean", action="store_true", help="Полная очистка: удалить shards/, манифест и state")

    # Параметры сессии
    p.add_argument("--resume", action="store_true", help="Возобновить незавершённую сессию (если есть state)")
    p.add_argument("--max-items-per-session", type=int, default=None, help="Ограничить число контейнеров, обрабатываемых за один запуск")
    p.add_argument("--batch-size", type=int, default=None, help="Размер батча (если не указан, будет подобран автоматически)")
    p.add_argument("--min-batch-size", type=int, default=1025, help="Минимально допустимый размер батча")
    p.add_argument("--no-progress", dest="progress", action="store_false", help="Отключить прогресс-бар")
    p.add_argument("--progress", dest="progress", action="store_true", help="Включить прогресс-бар")
    p.set_defaults(progress=True)

    p.add_argument("--no-compress", dest="compress", action="store_false", help="Отключить сжатие npz")
    p.add_argument("--compress", dest="compress", action="store_true", help="Включить сжатие npz (по умолчанию)")
    p.set_defaults(compress=True)

    p.add_argument("--no-offload", dest="offload", action="store_false", help="Не выгружать данные контейнеров из памяти после записи батча")
    p.add_argument("--offload", dest="offload", action="store_true", help="Выгружать данные из памяти после записи батча (по умолчанию)")
    p.set_defaults(offload=True)

    p.add_argument("--auto-batch", action="store_true", help="Подобрать безопасный batch_size автоматически (и ограничить пользовательский, если задан)")
    p.add_argument("--auto-resume-loop", action="store_true", help="Автоматически перезапускать сессию при достижении лимита, пока не завершится полностью")
    p.add_argument("--max-cycles", type=int, default=None, help="Максимальное число циклов автоперезапуска (по умолчанию без лимита)")
    p.add_argument("--sleep-between-seconds", type=float, default=1.0, help="Пауза между циклами автоперезапуска")

    args = p.parse_args()

    # Очистка, если запрошена
    if args.soft_clean or args.fresh_clean:
        store = FractalWeightStore()
        report = store.cleanup_incremental_artifacts(output_path=args.out_dir, fresh=bool(args.fresh_clean))
        if not report.get("ok", False):
            logger.error("Очистка завершилась с ошибкой: %s", report.get("error"))
            sys.exit(2)
        removed = report.get("removed", [])
        if removed:
            logger.info("Удалено артефактов: %d", len(removed))
            for pth in removed[:10]:
                logger.info(" - %s", pth)
            if len(removed) > 10:
                logger.info(" ... и ещё %d", len(removed) - 10)
        else:
            logger.info("Артефактов для удаления не найдено")
        # Если была только очистка — завершаем
        if not args.hf_model_dir:
            return

    ok = run_session(
        hf_model_dir=args.hf_model_dir,
        out_dir=args.out_dir,
        model_id=args.model_id,
        resume=bool(args.resume),
        batch_size=args.batch_size,
        min_batch_size=int(args.min_batch_size),
        max_items_per_session=args.max_items_per_session,
        progress=bool(args.progress),
        compress=bool(args.compress),
        offload_after_write=bool(args.offload),
        auto_batch=bool(args.auto_batch),
        auto_resume_loop=bool(args.auto_resume_loop),
        max_cycles=args.max_cycles,
        sleep_between_seconds=float(args.sleep_between_seconds),
    )

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
