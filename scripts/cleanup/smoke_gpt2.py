import os
import logging
import sys
from pathlib import Path

# Убедимся, что используем CPU для простоты теста
os.environ.setdefault("COGNIFLEX_DEVICE_MAP", "cpu")

# Гарантируем, что корень проекта в sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogniflex.mlearning.model_manager import ModelManager

logging.basicConfig(level=logging.INFO)


def main():
    mm = ModelManager(use_gpu=False, autoload=False, safe_test_mode=False)

    # Регистрируем GPT2 как простую модель генерации текста
    model_id = "gpt2_cpu_smoke"
    mm.register_model(
        model_id=model_id,
        model_path="gpt2",
        model_type="gpt",
        priority=50,
        name="GPT-2 (CPU Smoke)",
        tags=["test", "gpt2", "cpu"],
    )

    # Запускаем менеджер и загружаем модель
    mm.start()
    ok = mm.load_model(model_id)
    if not ok:
        raise SystemExit("[FAIL] Не удалось инициировать загрузку модели")

    # Подождём немного загрузку (простая синхронная проверка)
    import time
    for _ in range(30):  # до ~30 секунд
        inst = mm.get_model(model_id)
        if inst:
            break
        time.sleep(1)

    inst = mm.get_model(model_id)
    if not inst:
        raise SystemExit("[FAIL] Модель не стала доступной за отведённое время")

    # Проверяем, что в модели нет meta-тензоров
    if any(getattr(p, "is_meta", False) for p in inst.model.parameters()):
        raise SystemExit("[FAIL] В модели остались meta-тензоры")

    # Генерация
    result = mm.generate_response(
        prompt="Привет! Расскажи короткий факт о космосе.",
        model_id=model_id,
        max_length=80,
        temperature=0.8,
        top_p=0.9,
        task="text-generation",
    )

    if "error" in result:
        raise SystemExit(f"[FAIL] Ошибка генерации: {result['error']}")

    print("[OK] Тест пройден. Ответ:")
    print(result["text"]) 


if __name__ == "__main__":
    main()
