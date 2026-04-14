# Отчёт: Tools & Security

## 1. Структура

### eva_ai/tools/
```
eva_ai/tools/
├── __init__.py                 # Экспорты: import_pipeline, document_reader
├── document_reader.py          # Чтение текстовых файлов (.txt, .md, .log, .json, .xml, .csv, .yaml)
├── import_pipeline.py         # Импорт TXT, PDF, EPUB с OCR fallback
├── dependency_scan.py          # AST-анализ зависимостей, детекция циклов
├── system_generation_analysis.py  # Анализ системы генерации текста
└── layer_expertise_analysis.py    # Анализ экспертизы слоёв Qwen 2.5 3B
```

### eva_ai/security/
```
eva_ai/security/
├── __init__.py                 # Экспорты: SecurityManager, AuthenticationManager, RateLimiter
└── security_framework.py      # 356 строк - полная система безопасности
```

### eva_ai/ethics/
```
eva_ai/ethics/
├── __init__.py                 # Экспорт EthicsFramework
├── ethics_framework.py         # Импорт из модулей (рефаторинг)
├── framework_core.py          # Ядро EthicsFramework
├── framework_principles.py    # EthicalPrinciple, управление принципами
├── framework_checks.py         # EthicsChecksMixin - проверки по категориям
├── framework_violations.py     # EthicsViolationsMixin - нарушения и статистика
├── violation_id_manager.py     # Генерация/парсинг ID нарушений
├── situations_db.py            # База этических ситуаций
├── situations_evaluation.py    # Оценка ситуаций
├── situations_scenarios.py      # Сценарии ситуаций
├── ethical_situations.py       # Работа с ситуациями
├── risk_assessment.py          # Оценка рисков
├── principles_manager.py        # Менеджер принципов
├── ethics_integrated.py        # Интеграция этики
├── ethics_core.py              # Ядро этики
├── reference_scenarios.json    # Референсные сценарии
└── eva_ethics_cache/          # Кэш этической рамки
    ├── ethical_issues.json
    ├── ethical_reviews.json
    ├── ethical_solutions.json
    └── ethics_principles.db
```

---

## 2. Инструменты

### 2.1 DocumentTextReader (document_reader.py)
**Назначение:** Чтение текстовых файлов для отображения в чате

**Поддерживаемые форматы:**
- `.txt`, `.md`, `.log`, `.json`, `.xml`, `.csv`, `.yaml`, `.yml`

**Возможности:**
- Автоопределение кодировки (utf-8, utf-8-sig, cp1251, koi8-r, iso-8859-5)
- Ограничение по размеру (max_chars)
- Метаданные: размер, расширение, кодировка, количество строк/символов
- Метод `read_as_messages()` для интеграции с чатом

**Классы:**
- `DocumentContent` (dataclass) - filename, filepath, content, lines, metadata
- `DocumentTextReader` - основной класс чтения

---

### 2.2 ImportPipeline (import_pipeline.py)
**Назначение:** Импорт документов с нормализацией и чанкованием

**Поддерживаемые форматы:**
- `.txt`, `.md`, `.log` - прямое чтение
- `.pdf` - pdfminer/pypdf, опционально OCR (Tesseract)
- `.epub` - ebooklib + BeautifulSoup

**Основные компоненты:**
- `ImportedDocument` (dataclass) - id, source_path, metadata, _segments
- `ImportPipeline`:
  - `chunk_tokens: int = 512` - размер чанка
  - `overlap_tokens: int = 64` - перекрытие между чанками
  - `max_doc_chars: int = 2_000_000` - лимит документа

**Методы:**
- `import_path(path, doc_id)` - основной API
- `_normalize_text(text)` - нормализация через UnifiedTextProcessor или fallback
- `_chunk_text(text)` - сегментация с учётом токенов
- `_read_txt/pdf/epub()` - ридеры для разных форматов

---

### 2.3 Dependency Scan (dependency_scan.py)
**Назначение:** AST-анализ зависимостей модулей ЕВА

**Возможности:**
- Обход файловой системы с исключением: `.git`, `__pycache__`, `venv`, `site-packages`, `eva_models`
- Парсинг `import` и `from ... import` через AST
- Построение графа зависимостей
- **Детекция циклов** через DFS (WHITE/GRAY/BLACK)
- Генерация отчётов:
  - `dependency_report.log` - текстовый отчёт
  - `dependency_graph.json` - JSON граф
  - `dependency_graph.dot` - GraphViz

**Выявляет:**
- Missing internal imports (отсутствующие внутренние модули)
- Циклические зависимости
- Top modules by in-degree

---

### 2.4 SystemGenerationAnalysis (system_generation_analysis.py)
**Назначение:** Комплексный анализ системы генерации текста

**Класс:** `SystemAnalyzer`

**Методы:**
- `analyze_module_structure()` - структура модуля (классы, функции, импорты)
- `analyze_generation_flow()` - поток генерации
- `test_generation_pipeline()` - тестирование пайплайна
- `analyze_file_structure()` - анализ файлов
- `create_analysis_report()` - генерация Markdown-отчёта

**Анализирует:**
- OptimizedFractalModelManager
- text_quality_trainer, text_quality_improver
- hybrid_token_cache
- fractal_store, fractal_model_loader

---

### 2.5 LayerExpertiseAnalysis (layer_expertise_analysis.py)
**Назначение:** Анализ экспертизы слоёв Qwen 2.5 3B

**Тестовые категории:**
- `syntax` - синтаксис/грамматика
- `facts` - факты/знания
- `logic` - логика/рассуждения
- `code` - программирование
- `creative` - креатив/стиль

**Процесс:**
1. Загрузка модели с `output_hidden_states=True`
2. Сбор активаций слоёв для каждой категории
3. Mean pooling по последовательности
4. KMeans кластеризация (5 кластеров)
5. Назначение меток кластерам

**Вывод:** JSON с профилями слоёв, доминирующими категориями, кластерами

---

## 3. Безопасность

### 3.1 Security Framework (security_framework.py)

#### User & SecurityEvent
```python
@dataclass
class User:
    id: str
    username: str
    role: str  # admin, user, guest
    is_active: bool
    created_at: datetime
    last_login: datetime

@dataclass  
class SecurityEvent:
    event_type: str
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    timestamp: datetime
    details: Dict
```

#### RateLimiter
**Лимиты:**
- `requests_per_minute: int = 60` - общий лимит
- `burst_limit: int = 10` - лимит на последнюю секунду

**Механизм:** Скользящее окно (60 секунд), очистка старых записей

#### AuthenticationManager
**Функции:**
- `authenticate(username, password)` → session_token или None
- `validate_session(session_token)` → session dict или None
- `create_user(username, password, role)` → User
- `logout(session_token)`

**Сессия:**
- Токен: `secrets.token_hex(32)`
- TTL: 24 часа
- Хранение: in-memory словарь `sessions`

**Пароль:** SHA256 хэш (демо), комментарий о bcrypt для продакшена

#### AuthorizationManager
**Роли и права:**
| Роль | Разрешения |
|------|------------|
| admin | read, write, delete, admin, system |
| user | read, write, chat |
| guest | read, chat |

#### SecurityManager (главный класс)
**Методы:**
- `authenticate_request()` - аутентификация + rate limit
- `authorize_request()` - проверка разрешений
- `validate_request()` - комплексная валидация
- `_log_event()` - логирование событий
- `get_security_events(limit)` - история событий
- `get_rate_limit_status()` - статус rate limit

**Интеграция:**
```python
@require_authentication(permission="user")
def protected_function(...):
    ...
```

---

### 3.2 Ethics Framework

#### EthicalPrinciple
```python
@dataclass
class EthicalPrinciple:
    name: str
    description: str
    phrase: str
    weight: float = 1.0
    threshold: float = 0.8
    category: str = "general"
    priority: int = 5
    last_updated: float
    active: bool = True
```

#### Базовые принципы (по умолчанию)
| Принцип | Категория | Вес | Порог | Приоритет |
|---------|-----------|-----|-------|-----------|
| no_violence | safety | 1.5 | 0.6 | 10 |
| honesty | integrity | 1.2 | 0.7 | 9 |
| fact_verification | accuracy | 1.2 | 0.7 | 9 |
| safe_code | security | 1.5 | 0.6 | 10 |
| risk_blocking | safety | 1.3 | 0.65 | 10 |
| output_control | quality | 1.0 | 0.75 | 8 |

#### Категории оценки
- **privacy** - личные данные, конфиденциальность
- **safety** - насилие, оружие, наркотики, вред
- **fairness** - дискриминация, расизм, сексизм
- **transparency** - прозрачность, манипуляции
- **autonomy** - автономия, контроль мыслей
- **beneficence** - польза, предотвращение вреда
- **accountability** - подотчётность, ответственность

#### EthicalDecision
```python
@dataclass
class EthicalDecision:
    approved: bool
    principle: str
    severity: float
    description: str
    context: Dict
    timestamp: float
    resolved: bool
    resolution: Optional[str]
    resolution_timestamp: Optional[float]
    source: str  # system
    violation_id: str  # уникальный ID
```

#### EthicsChecksMixin - Методы проверки
- `analyze_request(request, context)` → Dict с violations, principle_scores
- `analyze_response(query, response)` → EthicsAnalysisResult
- `analyze_content(content, context)` → EthicsAnalysisResult
- `check_with_context(text, query, context)` → подробный результат
- `generate_regeneration_prompt()` - промпт для исправления

#### EthicsViolationsMixin - Управление нарушениями
- `_load_violations_and_stats()` - загрузка из JSON
- `_save_violations()` / `_save_principles()` / `_save_stats()`
- `get_violation_history(limit, principle)`
- `resolve_violation(violation_id, resolution)`
- `get_active_violations()`
- `get_ethics_statistics()`
- `export_ethics_data()` / `import_ethics_data()`
- `_analyze_ethical_trends()`
- `generate_ethics_report()`

#### Фоновые службы
- `_violation_monitor_thread` - проверка каждые 60 сек (авторазрешение старых >7 дней)
- `_principle_check_thread` - проверка каждые 3600 сек

---

## 4. Оценка

### 4.1 Tools

| Критерий | Оценка | Комментарий |
|----------|--------|-------------|
| Полнота | ★★★★☆ | 5 инструментов для разных задач |
| Покрытие форматов | ★★★★☆ | TXT, PDF, EPUB, код, документы |
| Интеграция | ★★★☆☆ | Требуют brain для полной работы |
| Безопасность | ★★★★☆ | Есть проверки в import_pipeline |
| Документация | ★★★☆☆ | Есть docstrings, нет unit-тестов |

**Замечания:**
- `dependency_scan.py` - полезный инструмент для анализа архитектуры
- `system_generation_analysis.py` - ценен для отладки, но захардкожены пути
- `layer_expertise_analysis.py` - требует sklearn, numpy, torch

### 4.2 Security Framework

| Критерий | Оценка | Комментарий |
|----------|--------|-------------|
| Аутентификация | ★★★☆☆ | SHA256 (демо), нет bcrypt |
| Авторизация | ★★★★☆ | RBAC с 3 ролями |
| Rate Limiting | ★★★★☆ | Скользящее окно, burst limit |
| Логирование | ★★★★☆ | Централизованное, SecurityEvent |
| Сессии | ★★★☆☆ | In-memory, нет refresh token |
| Шифрование | ★★☆☆☆ | Минимальное (SHA256 хэш) |

**Уязвимости:**
1. Пароли хранятся как SHA256 (не bcrypt/argon2)
2. Сессии в памяти (не persistent)
3. Нет защиты от brute-force
4. Дефолтный admin:admin

**Рекомендации:**
- Использовать `bcrypt` или `argon2` для паролей
- Добавить персистентное хранилище сессий
- Реализовать lockout после N неудачных попыток
- Добавить HTTPS/TLS

### 4.3 Ethics Framework

| Критерий | Оценка | Комментарий |
|----------|--------|-------------|
| Принципы | ★★★★★ | 6 базовых + расширяемые |
| Категории | ★★★★☆ | 7 категорий покрывают основное |
| Проверки | ★★★★☆ | Keyword-based + контекстные |
| Нарушения | ★★★★★ | Полный жизненный цикл |
| Статистика | ★★★★☆ | Тренды, экспорт/импорт |
| Автоматизация | ★★★★☆ | Фоновые мониторинг-потоки |

**Сильные стороны:**
- Разделение на миксины (принципы, проверки, нарушения)
- Персистентность в JSON/SQLite
- Анализ трендов нарушений
- Генерация этических отчётов

**Слабые стороны:**
- Keyword-based проверки (можно обойти)
- Нет интеграции с LLM для семантической проверки
- Нет NLI (Neutral Language Inference) валидации

---

## Итоговый вердикт

### Tools: ★★★★☆ (4/5)
Система предоставляет полный набор инструментов для работы с документами, анализа зависимостей и исследования модели. Основная проблема - жёсткие пути и зависимости.

### Security: ★★★☆☆ (3/5)  
Базовая система безопасности реализована, но требует доработки для продакшена. Главные риски: слабое хэширование паролей, отсутствие защиты от brute-force.

### Ethics: ★★★★☆ (4/5)
Мощная система этических проверок с хорошей архитектурой. Keyword-based подход эффективен для базовой фильтрации, но для продвинутой семантической проверки нужна интеграция с LLM/NLI.

---

*Отчёт сгенерирован: 2026-04-14*
