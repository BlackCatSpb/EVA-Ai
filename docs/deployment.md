# Руководство по развертыванию и DevOps

Этот документ описывает практики развертывания CogniFlex в онлайн/офлайн средах, управление зависимостями и контейнеризацией.

## Предпосылки
- Python 3.11+ (рекомендуется 3.12/3.13 при совместимости зависимостей)
- Windows/Linux/macOS
- Доступ к GPU (опционально) и CUDA/cuDNN, если используется
- Зависимости из `cogniflex/requirements.txt`

## Офлайн режим (без скачиваний из сети)
1. Установите переменные окружения перед запуском процесса:
   - PowerShell:
     ```powershell
     $env:TRANSFORMERS_OFFLINE="1"
     $env:HF_HUB_OFFLINE="1"
     ```
   - Bash:
     ```bash
     export TRANSFORMERS_OFFLINE=1
     export HF_HUB_OFFLINE=1
     ```
2. Инициализируйте `ModelManager` с `autoload=False` для предотвращения фоновых загрузок:
   ```python
   from cogniflex.mlearning.model_manager import ModelManager
   mm = ModelManager(brain=None, use_gpu=False, max_workers=1, autoload=False)
   ```
3. При необходимости понизьте приоритет больших моделей (Qwen 7B/14B и пр.) в БД `models.db` (поле `priority=-1`).

## Конфигурация окружения
- Управляйте путями к кэшу/моделям параметрами `ModelManager` (`cache_dir`, `model_dir`).
- Следите за потреблением памяти/CPU/GPU через `SystemMetricsManager` и метод `get_system_health()` ядра.

## Логи и метрики
- Логи запуска/GUI: файлы в корне проекта (`gui_simulation_errors.log`, `import_debug.log` и т.п.).
- Системные метрики доступны через интерфейсы ядра (`get_system_metrics()`, `get_system_dashboard_data()`).

## Контейнеризация (Docker)
Пример минимального Dockerfile (CPU, офлайн):
```dockerfile
FROM python:3.11-slim
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY cogniflex/ /app/cogniflex/
COPY README.md /app/
RUN pip install --no-cache-dir -r /app/cogniflex/requirements.txt
CMD ["python", "-m", "cogniflex.system_selftest"]
```

docker build / run:
```bash
docker build -t cogniflex:offline .
docker run --rm -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 cogniflex:offline
```

## Запуск GUI
- Локально: `python run_gui.py`
- В контейнере: рекомендуется запускать headless части (API/обработчики). GUI в контейнере требует X11/Wayland проброса или VNC.

## CI/CD рекомендации
- Кэшируйте директории модели/токенов между сборками.
- Выполняйте smoke‑тесты: `system_smoke_test.py`, короткие юнит‑тесты `test_*`.
- Строгая проверка офлайн‑режима в CI (env‑переменные + `autoload=False`).

## Безопасность
- Не сохраняйте токены/ключи в репозитории.
- Проверяйте лицензии моделей перед публикацией артефактов.
