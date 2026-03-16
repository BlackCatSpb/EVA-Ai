#!/bin/bash
# CogniFlex Startup Script
# Оптимизированный запуск

cd "$(dirname "$0")"

# Обход проверки CVE в PyTorch
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Активировать виртуальное окружение
if [ -d ".venv311" ]; then
    source .venv311/Scripts/activate
fi

# Запуск
python -m cogniflex.run
