# FMF EVA Container

Универсальный контейнер для FMF Interactive EVA System.

## Структура

```
FMF_EVA/
├── model.ov/          # OpenVINO модель (Qwen2 3B)
├── lora/              # LoRA адаптеры
├── data/              # Граф памяти
│   └── graph.db
├── src/               # Исходный код
│   └── fmf_cli.py
├── config.json        # Конфигурация
└── requirements.txt  # Зависимости
```

## Установка

```bash
# Клонировать репозиторий
git clone https://github.com/yourrepo/FMF_EVA.git
cd FMF_EVA

# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows

# Установить зависимости
pip install -r requirements.txt
```

## Использование

### CLI

```bash
# Один запрос
fmf run "Привет, как дела?"

# Интерактивный чат
fmf chat

# Запуск API сервера
fmf serve --port 7860

# Проверка статуса
fmf status
```

### Python API - Базовая

```python
from src.fmf_cli import FMFGeneratorInteractive

gen = FMFGeneratorInteractive("model.ov", "data/graph.db", "CPU")
result = gen.generate("Твой запрос", max_tokens=2048)
print(result["response"])
```

### Python API - Универсальный адаптер (замена PyTorch/GGUF)

```python
from src.fmf_adapter import FMFAdapter, FMFPipeline, from_pretrained

# Способ 1: Как transformers (from_pretrained)
model = from_pretrained("model.ov", device="CPU")
result = model.generate("Привет!")

# Способ 2: Как pipeline
pipeline = create_pipeline("model.ov")
result = pipeline("Привет!")

# Способ 3: Прямой вызов
adapter = FMFAdapter("model.ov")
print(adapter("Привет!"))  # Возвращает только текст
```

## Требования

- Python 3.10+
- OpenVINO 2026.1+
- Windows / Linux / macOS

## Лицензия

MIT