from __future__ import annotations

import argparse
import json
from pathlib import Path
import logging
import os
import tempfile
import gc
from typing import Optional, Dict, List, Set

import torch

try:
    from safetensors.torch import save_file as save_safetensors  # type: ignore
    HAVE_SAFETENSORS = True
except Exception:
    HAVE_SAFETENSORS = False

from cogniflex.mlearning.storage.fractal_store import FractalWeightStore


def _load_existing_state(out_path: Path) -> Dict[str, torch.Tensor]:
    if not out_path.exists():
        return {}
    try:
        if out_path.suffix == ".safetensors" and HAVE_SAFETENSORS:
            # Загружаем и сразу клонируем тензоры, чтобы отвязаться от mmap-файла
            from safetensors.torch import load_file as load_safetensors  # type: ignore
            _mapped = load_safetensors(str(out_path))
            try:
                cloned = {k: v.detach().clone() for k, v in _mapped.items()}
            finally:
                # Убираем ссылки на мэппинг и форсируем GC, чтобы освободить файловый дескриптор
                del _mapped
                gc.collect()
            return cloned
        else:
            return torch.load(str(out_path), map_location="cpu")
    except Exception:
        return {}


def _save_state(out_path: Path, state: Dict[str, torch.Tensor]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_suffix = ".tmp"
    if out_path.suffix == ".safetensors" and HAVE_SAFETENSORS:
        cpu_state = {k: v.detach().cpu().contiguous() for k, v in state.items()}
        # Пишем во временный файл и атомарно заменяем целевой (Windows-friendly)
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(out_path.parent), suffix=tmp_suffix) as tf:
            tmp_name = tf.name
        try:
            save_safetensors(cpu_state, tmp_name)
            os.replace(tmp_name, str(out_path))
        finally:
            try:
                if os.path.exists(tmp_name):
                    os.remove(tmp_name)
            except Exception:
                pass
    else:
        # Аналогично через временный файл для .pt
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(out_path.parent), suffix=tmp_suffix) as tf:
            tmp_name = tf.name
        try:
            torch.save({k: v.detach().cpu() for k, v in state.items()}, tmp_name)
            os.replace(tmp_name, str(out_path))
        finally:
            try:
                if os.path.exists(tmp_name):
                    os.remove(tmp_name)
            except Exception:
                pass


def build_state(
    fractal_dir: Path,
    out_path: Path,
    dtype: str = "float32",
    device: str = "cpu",
    limit_tensors: Optional[int] = None,
    include_params: Optional[List[str]] = None,
    resume_file: Optional[Path] = None,
    hot_window_mb: Optional[int] = None,
    iterative: bool = False,
) -> Dict[str, torch.Tensor]:
    # Быстрые проверки целостности, чтобы не читать гигантский манифест впустую
    manifest = fractal_dir / "shards_manifest.jsonl"
    shards_dir = fractal_dir / "shards"
    if not manifest.exists():
        raise FileNotFoundError(f"Не найден {manifest}. Убедитесь, что указан каталог фрактала.")
    npz_list = []
    try:
        if shards_dir.exists():
            npz_list = [p for p in shards_dir.glob("*.npz")]  # возможно сжатые шард-файлы
    except Exception:
        npz_list = []
    if not npz_list:
        raise RuntimeError(
            f"Каталог шардов пуст: {shards_dir}. Нет *.npz файлов. Похоже, шардированное сохранение не завершено «data shards»."
        )

    # Инициализация стора с учётом устройства
    store = FractalWeightStore(device=device)
    if hot_window_mb is not None and hot_window_mb > 0:
        try:
            store.hot_window_size = int(hot_window_mb) * 1024 * 1024
        except Exception:
            pass
    ok = store.load_from_disk(str(fractal_dir), lazy=True)
    if not ok:
        raise RuntimeError(f"Не удалось лениво загрузить фрактал из {fractal_dir}")

    # Резюме прогресса
    processed: Set[str] = set()
    if resume_file is not None and resume_file.exists():
        try:
            data = json.loads(resume_file.read_text(encoding="utf-8"))
            processed = set(data.get("processed", []))
        except Exception:
            processed = set()

    merged_state = _load_existing_state(out_path)

    total_collected = 0
    last_key: Optional[str] = None

    while True:
        state_batch = store.reconstruct_state_dict(
            output_dtype=dtype,
            device=device,
            limit_tensors=limit_tensors,
            include_params=include_params,
            processed_params=processed,
            resume_from=last_key,
        )
        if not state_batch:
            break
        # Обновляем итоговый словарь и сохраняем на диск (перезапись файла)
        merged_state.update(state_batch)
        _save_state(out_path, merged_state)

        # Обновляем прогресс
        keys_sorted = sorted(state_batch.keys())
        if keys_sorted:
            last_key = keys_sorted[-1]
        processed.update(state_batch.keys())
        total_collected += len(state_batch)

        # Запишем резюме файл
        if resume_file is not None:
            try:
                resume_file.parent.mkdir(parents=True, exist_ok=True)
                resume_file.write_text(
                    json.dumps({"processed": sorted(processed), "last_key": last_key}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass

        # Если не итеративный режим — выходим после одного батча
        if not iterative:
            break

        # В итеративном режиме продолжаем, пока есть что собирать
        if limit_tensors is None:
            # защитный выход, чтобы избежать бесконечного цикла
            break

    # Сохраним служебный отчёт рядом
    report = {
        "fractal_dir": str(fractal_dir),
        "out_file": str(out_path),
        "dtype": dtype,
        "device": device,
        "limit_tensors": limit_tensors,
        "param_count": len(merged_state),
        "new_params": total_collected,
        "keys_sample": list(merged_state.keys())[:50],
        "resume_file": str(resume_file) if resume_file is not None else None,
    }
    with (out_path.parent / (out_path.stem + ".report.json")).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return merged_state


def main() -> None:
    # Настройка логирования до разбора аргументов, чтобы видеть ранние логи
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    p = argparse.ArgumentParser(description="Сборка state_dict из фрактального графа памяти")
    p.add_argument("fractal_dir", type=str, help="Каталог фрактала (с shards_manifest.jsonl)")
    p.add_argument("out", type=str, help="Путь для сохранения (.safetensors или .pt)")
    p.add_argument("--dtype", type=str, default="float32", choices=["float16","bfloat16","float32","float64"], help="Выходной dtype")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"], help="Устройство для итоговых тензоров")
    p.add_argument("--limit", type=int, default=None, help="Ограничить число параметров в одном батче")
    p.add_argument("--only-params", type=str, default=None, help="Собирать только параметры, имена которых содержат перечисленные подстроки (через запятую)")
    p.add_argument("--resume-file", type=str, default=None, help="Путь к JSON-файлу прогресса для возобновления")
    p.add_argument("--hot-window-mb", type=int, default=1500, help="Размер горячего окна в МБ (по умолчанию 1500)")
    p.add_argument("--iterative", action="store_true", help="Итеративно собирать батчи, обновляя выходной файл до исчерпания")
    args = p.parse_args()

    fractal_dir = Path(args.fractal_dir)
    out_path = Path(args.out)

    include_params: Optional[List[str]] = None
    if args.only_params:
        include_params = [s.strip() for s in str(args.only_params).split(",") if s.strip()]

    resume_file = Path(args.resume_file) if args.resume_file else None

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Запрошено устройство CUDA, но CUDA недоступна")

    build_state(
        fractal_dir,
        out_path,
        dtype=args.dtype,
        device=args.device,
        limit_tensors=args.limit,
        include_params=include_params,
        resume_file=resume_file,
        hot_window_mb=args.hot_window_mb,
        iterative=bool(args.iterative),
    )
    print(f"OK: сохранено в {out_path}")


if __name__ == "__main__":
    main()
