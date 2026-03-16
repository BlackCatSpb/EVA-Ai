# CogniFlex - Документация проекта

## Обзор

CogniFlex — это интеллектуальная система с фрактальной архитектурой, объединяющая современные технологии машинного обучения, обработки естественного языка, распределенных вычислений и этического ИИ.

## Архитектура системы

### 1. Ядро системы (`cogniflex/core/`)

#### Основные компоненты:
- **CoreBrain** (`core_brain.py`) - Центральный контроллер системы
- **Integration Layer** (`integration_layer.py`) - Интеграционный слой для обработки запросов
- **Event Bus** (`event_bus.py`) - Событийная шина для компонентов
- **Base Component** (`base_component.py`) - Базовый класс для всех компонентов
- **Component Initializer** (`component_initializer.py`) - Инициализатор компонентов

#### Функциональность:
- Многоуровневая обработка запросов
- Событийная архитектура с 7 этапами обработки
- Автоматическая инициализация и управление компонентами
- Fallback-механизмы для отказоустойчивости

### 2. Машинное обучение (`cogniflex/mlearning/`)

#### Ключевые модули:
- **Fractal Model Manager** - Управление фрактальными моделями
- **Generation Coordinator** - Координация генерации ответов
- **Parallel Tokenization** - Параллельная токенизация
- **Model Manager** - Управление моделями ML
- **Tokenizer** - Токенизация текста с фрактальными метаданными

#### Хранилище:
- **Fractal Store** (`storage/`) - Фрактальное хранилище данных
- **Unified Storage** - Унифицированное хранилище
- **Memory Graph Store** - Графовое хранилище памяти

### 3. Память (`cogniflex/memory/`)

- **Hybrid Token Cache** - Гибридный кэш токенов (VRAM → RAM → SSD)
- **Memory Manager** - Управление памятью системы
- **Multi-tier Caching** - Многоуровневое кэширование

### 4. Знания (`cogniflex/knowledge/`)

- **Knowledge Graph** - Граф знаний системы
- **Knowledge Integration** - Интеграция знаний
- **Concept Extractor** - Извлечение концептов
- **Semantic Engine** - Семантический движок

### 5. Этика (`cogniflex/ethics/`)

- **Ethics Framework** - Этическая рамка
- **Ethics Integrated** - Интегрированный модуль этики
- **Violation Manager** - Управление нарушениями
- **Principles Management** - Управление этическими принципами

Поддерживаемые категории принципов:
- Safety (безопасность)
- Privacy (приватность)
- Fairness (справедливость)
- Transparency (прозрачность)
- Autonomy (автономия)
- Beneficence (благотворительность)
- Accountability (ответственность)

### 6. Обучение (`cogniflex/learning/`)

- **Learning Manager** - Менеджер обучения
- **Learning Processor** - Процессор обучения
- **Analyzer Core** - Ядро анализа
- **Opportunity Manager** - Управление возможностями обучения
- **Self Analyzer** - Самоанализ системы

### 7. Противоречия (`cogniflex/contradiction/`)

- **Contradiction Manager** - Управление противоречиями
- **Contradiction Detection** - Обнаружение противоречий
- **Contradiction Resolution** - Разрешение противоречий
- **Resolution Strategies** - Стратегии разрешения:
  - ConservativeStrategy - Консервативная стратегия
  - MajorityVoteStrategy - Стратегия большинства
  - ConfidenceBasedStrategy - На основе уверенности

### 8. Аналитика (`cogniflex/analytics/`)

- **Analytics Manager** - Менеджер аналитики
- **Metrics Collection** - Сбор метрик
- **Performance Monitoring** - Мониторинг производительности

### 9. GUI (`cogniflex/gui/`)

- **Integrated GUI** - Интегрированный интерфейс
- **Chat Module** - Модуль чата
- **Memory Module** - Модуль памяти
- **Core GUI** - Основной GUI

### 10. Распределенная система (`cogniflex/distributed/`)

- **Distributed System** - Распределенная система
- **Task Scheduler** - Планировщик задач
- **Knowledge Sync** - Синхронизация знаний
- **Recovery Manager** - Менеджер восстановления
- **Cluster Manager** - Управление кластером

### 11. Веб-поиск (`cogniflex/websearch/`)

- **Web Search Engine** - Поисковый движок
- **Search Cache** - Кэш поиска
- **Result Analyzer** - Анализатор результатов

### 12. Адаптация (`cogniflex/adaptation/`)

- **Adaptation Manager** - Менеджер адаптации
- **User Profiling** - Профилирование пользователей
- **Response Adaptation** - Адаптация ответов

### 13. Безопасность (`cogniflex/security/`)

- **Security Framework** - Рамки безопасности
- **Access Control** - Контроль доступа
- **Encryption** - Шифрование данных

## Поток обработки запросов

1. **Query Received** - Получение запроса
2. **Tokenize Request** - Токенизация запроса
3. **Tokens Ready** - Токены готовы
4. **Hot Window Ready** - Горячее окно готово
5. **Response Generated** - Генерация ответа
6. **Ethics Check** - Этическая проверка
7. **Response Delivered** - Доставка ответа

## Возможности системы

### Обработка естественного языка
- Мультиязычная поддержка (русский, английский)
- Контекстуальное понимание
- Семантический анализ
- Извлечение сущностей и концептов

### Генерация ответов
- На основе RuGPT3/GPT2
- Фрактальная генерация
- Контролируемая генерация
- Адаптация стиля ответов

### Обучение
- Самообучение на основе обратной связи
- Онлайн-обучение
- Обучение с подкреплением
- Обнаружение возможностей для обучения

### Этика
- Проверка запросов на этичность
- Обнаружение потенциальных нарушений
- Генерация рекомендаций
- Управление нарушениями

### Память
- Многоуровневое кэширование
- Долгосрочное хранение знаний
- Ассоциативная память
- Граф знаний

### Масштабируемость
- Распределенная обработка
- Кластеризация
- Балансировка нагрузки
- Отказоустойчивость

## Конфигурация

### Файлы конфигурации:
- `brain_config.json` - Конфигурация CoreBrain
- `fractal_model_config.json` - Конфигурация фрактальных моделей
- `gpt3_config.json` - Конфигурация GPT-3
- `requirements.txt` - Зависимости Python

### Кэш и данные:
- `cogniflex_cache/` - Кэш системы
- `ethics_cache/` - Кэш этики
- `fractal_storage/` - Фрактальное хранилище
- `cache/` - Общий кэш

## API и интерфейсы

### Основные классы:

```python
# CoreBrain - центральный контроллер
from cogniflex.core.core_brain import CoreBrain

# Интеграционный слой
from cogniflex.core.integration_layer import CogniFlexIntegrator

# Этическая рамка
from cogniflex.ethics.ethics_framework import EthicsFramework

# Менеджер моделей
from cogniflex.mlearning.fractal_model_manager import FractalModelManager
```

### Запуск системы:

```bash
python -m cogniflex.run
```

## Требования

### Python:
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- SQLite3
- NumPy, SciPy

### Опциональные зависимости:
- NLTK (NLP)
- spaCy (NLP)
- scikit-learn (ML)
- psutil (мониторинг)

## Мониторинг и логирование

### Логи:
- `logs/cogniflex_app.log` - Основной лог
- `logs/` - Директория с логами

### Метрики:
- Производительность системы
- Использование памяти
- Статистика запросов
- Этические нарушения
- Показатели обучения

## Разработка

### Структура проекта:
```
cogniflex/
├── core/           # Ядро системы
├── mlearning/      # Машинное обучение
├── memory/         # Память
├── knowledge/      # Знания
├── ethics/         # Этика
├── learning/       # Обучение
├── contradiction/  # Противоречия
├── analytics/      # Аналитика
├── gui/            # Интерфейс
├── distributed/    # Распределенная система
├── websearch/      # Веб-поиск
├── adaptation/     # Адаптация
├── security/       # Безопасность
└── config/         # Конфигурация
```

### Тестирование:
- Модульные тесты в `tests/`
- Интеграционные тесты
- Скрипты проверки в корневой директории

## Будущие улучшения

- Улучшение фрактальной архитектуры
- Расширение мультимодальной поддержки
- Усиление приватности (федеративное обучение)
- Оптимизация производительности
- Расширение языковой поддержки

---

**Версия**: 1.0  
**Последнее обновление**: 2024  
**Автор**: CogniFlex Team
