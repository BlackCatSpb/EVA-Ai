import os
import sys
import time
import argparse
import json
import platform


def main():
    # Гарантируем доступность проектного корня для импорта пакета cogniflex
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="CogniFlex ML smoke test: list models and run a tiny generation")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="На каком устройстве инициализировать MLUnit")
    parser.add_argument("--prompt", default="Проверка связи.", help="Тестовый промпт")
    parser.add_argument("--max_length", type=int, default=32, help="Максимальная длина генерации")
    parser.add_argument("--offline", action="store_true", help="Принудительно оффлайн-режим для HF (HF_HUB_OFFLINE=1)")
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    # Вывод системной информации
    sys_info = {
        "python": sys.version.split(" ")[0],
        "platform": platform.platform(),
        "device_requested": args.device,
        "offline": bool(os.environ.get("HF_HUB_OFFLINE")),
    }
    print("[INFO] System:", json.dumps(sys_info, ensure_ascii=False))

    # Импорт MLUnit
    t0 = time.time()
    try:
        from cogniflex.mlearning.ml_unit import MLUnit  # type: ignore
    except Exception as e:
        print(f"[ERROR] Не удалось импортировать MLUnit: {e}")
        return 2
    t_import = time.time() - t0
    print(f"[INFO] MLUnit импортирован за {t_import:.2f}с")

    # Инициализация MLUnit
    try:
        use_gpu = args.device == "gpu"
        ml = MLUnit(use_gpu=use_gpu)
        print(f"[INFO] MLUnit инициализирован (use_gpu={use_gpu})")
    except Exception as e:
        print(f"[ERROR] Ошибка инициализации MLUnit: {e}")
        return 3

    # Получение списка доступных моделей
    try:
        models = []
        if hasattr(ml, "get_available_models"):
            models = ml.get_available_models()  # ожидается List[Dict]
        else:
            print("[WARN] MLUnit.get_available_models отсутствует — список моделей может быть пуст")
        print("[INFO] Доступные модели (первые 5):")
        for m in (models or [])[:5]:
            try:
                if isinstance(m, dict):
                    print("  -", json.dumps(m, ensure_ascii=False))
                else:
                    name = getattr(m, 'name', None) or getattr(m, 'model_name', None)
                    mid = getattr(m, 'id', None) or getattr(m, 'model_id', None)
                    src = getattr(m, 'source', None)
                    info = {k: v for k, v in {"name": name, "id": mid, "source": src} .items() if v is not None}
                    if info:
                        print("  -", json.dumps(info, ensure_ascii=False))
                    else:
                        print("  -", repr(m))
            except Exception:
                print("  -", repr(m))
        print(f"[INFO] Всего моделей: {len(models or [])}")
    except Exception as e:
        print(f"[ERROR] Ошибка получения списка моделей: {e}")

    # Пробная генерация (ленивая загрузка модели)
    try:
        prompt = args.prompt
        print(f"[INFO] Пробная генерация: '{prompt}' (max_length={args.max_length})")
        tg0 = time.time()
        result = ml.generate_response(prompt=prompt, max_length=args.max_length, task="text-generation")
        dt = time.time() - tg0
        ok = bool(result and result.get("text"))
        tokens = result.get("tokens") if isinstance(result, dict) else None
        print("[RESULT] success=", ok, ", time=", f"{dt:.2f}s", sep="")
        if ok:
            print("[TEXT]", result.get("text", "")[:200].replace("\n", " "))
            if tokens is not None:
                print("[TOKENS]", len(tokens))
        else:
            print("[DEBUG] full_result=", json.dumps(result, ensure_ascii=False))
        return 0 if ok else 4
    except Exception as e:
        print(f"[ERROR] Ошибка на стадии генерации (вероятно загрузка модели): {e}")
        return 5


if __name__ == "__main__":
    sys.exit(main())
