# Backup ЕВА системы модернизации

Дата создания: 2026-04-01

## Структура бекапа

```
backup/
├── configs/
│   ├── brain_config.json      # Основная конфигурация
│   └── Update.md              # План модернизации (источник истины)
├── modules/
│   ├── contradiction/         # 14 файлов модуля противоречий
│   ├── ethics/                # 11 файлов модуля этики
│   ├── websearch/            # 9 файлов + 1 подкаталог модуля поиска
│   ├── analytics/            # 4 файла модуля аналитики
│   ├── preprocess/           # 2 файла модуля препроцессинга
│   └── learning/             # 24 файла модуля обучения
├── integration/
│   ├── core_brain.py         # Основной мозг системы
│   ├── generation_coordinator.py  # Координатор генерации
│   └── server.py             # API сервер веб-интерфейса
├── test_contradiction_manager.py  # Тест модуля противоречий
└── README.md                  # Этот файл
```

## Инструкция по восстановлению

### 1. Восстановление конфигов
```bash
cp backup/configs/brain_config.json ./
cp backup/configs/Update.md ./
```

### 2. Восстановление модулей
```bash
# Противоречия
cp backup/modules/contradiction/*.py eva/contradiction/

# Этика
cp backup/modules/ethics/*.py eva/ethics/

# Веб-поиск
cp backup/modules/websearch/*.py eva/websearch/
cp -r backup/modules/websearch/cogniflex_web_search_cache eva/websearch/

# Аналитика
cp backup/modules/analytics/*.py eva/analytics/

# Препроцессинг
cp backup/modules/preprocess/*.py eva/preprocess/

# Обучение
cp backup/modules/learning/*.py eva/learning/
```

### 3. Восстановление интеграции
```bash
cp backup/integration/core_brain.py eva/core/
cp backup/integration/generation_coordinator.py eva/core/
cp backup/integration/server.py eva/gui/web_gui/
```

## Что изменяется при модернизации

### Новое (добавляется)
1. **Два экземпляра GGUF** - Модель А (факты) + Модель Б (уточнение)
2. **bge-small-rus** - для расчёта релевантности (косинусное сходство)
3. **ContradictionAnalyzer** - новый модуль аналитики противоречий
4. **Конвейер по этапам** - Факты → Поиск → Противоречия → Этика → Уточнение → Релевантность
5. **Рекурсивное уточнение** - до 3 попыток при пороге < 0.85

### Старое (заменяется)
- `core_brain.py` - обновлённая логика two-model pipeline
- `generation_coordinator.py` - поддержка Model A и Model B
- Модуль аналитики - новый `contradiction_analyzer.py`

## Метрики после модернизации

| Метрика | Описание | Норма |
|---------|----------|-------|
| semantic_weight | Косинусное сходство query→answer | ≥0.85 |
| generation_time_A | Время GGUF-A (мс) | Зависит от CPU |
| generation_time_B | Время GGUF-B (мс) | Зависит от CPU |
| search_useful_rate | Доля релевантных сниппетов | ≥0.70 |
| contradiction_count | Количество выявленных конфликтов | 0-1 |
| refinement_attempts | Количество попыток уточнения | 1-3 |

## Контакты

При вопросах обращаться к документации в Update.md
