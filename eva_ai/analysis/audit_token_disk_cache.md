# АУДИТ TokenDiskCache Системы в EVA AI

**Дата аудита:** 14.04.2026  
**Аудитор:** EVA AI Analysis System  
**Версия системы:** CogniFlex

---

## СОДЕРЖАНИЕ

1. [Резюме](#резюме)
2. [Архитектура TokenDiskCache](#архитектура)
3. [1. Использование Pickle](#1-использование-pickle)
4. [2. Thread Safety](#2-thread-safety)
5. [3. Система TTL](#3-система-ttl)
6. [4. Интеграция с HybridCache](#4-интеграция-с-hybridcache)
7. [Дополнительные находки](#дополнительные-находки)
8. [Рекомендации](#рекомендации)
9. [Оценка по 10-балльной шкале](#оценка-по-10-балльной-шкале)

---

## Резюме

| Компонент | Статус | Критичность |
|-----------|--------|-------------|
| Реализация TokenDiskCache | НАЙДЕНА | - |
| Использование Pickle | ДА | КРИТИЧЕСКАЯ |
| Thread Safety | ДА | - |
| TTL в дисковом кэше | НЕТ | ВЫСОКАЯ |
| Интеграция с HybridCache | ДА | - |

**Общая оценка: 5/10**

---

## Архитектура TokenDiskCache

### Найденные файлы

`
eva_ai/memory/
├── cache_disk.py          # ОСНОВНАЯ реализация TokenDiskCache (pickle, thread-safe)
├── token_disk_cache.py    # ПРОСТАЯ реализация (JSON, НЕ thread-safe, НЕ ИСПОЛЬЗУЕТСЯ)
├── cache_core.py          # HybridTokenCache - интегрирует TokenDiskCache
├── cache_ram.py           # LRUCache - RAM кэш
└── cache_eviction.py      # Логика вытеснения и TTL для контента
`

### Иерархия кэширования

`
HybridTokenCache (cache_core.py)
├── vram_cache: LRUCache (GPU VRAM)
├── ram_cache: LRUCache (RAM) 
└── disk_cache: TokenDiskCache (cache_disk.py) ← Дисковый кэш
`

**Поток данных:**
`
get_token() → VRAM → RAM → DISK → (miss)
put_token() → RAM + DISK (backup)
`

---

## 1. Использование Pickle

### Вердикт: КРИТИЧЕСКАЯ УЯЗВИМОСТЬ

### Детали

**Файл:** va_ai/memory/cache_disk.py

Pickle используется в трёх местах:

#### 1.1 Десериализация (строки 82-106)
`python
def get(self, token_id: str) -> Optional[Dict]:
    # ...
    import pickle
    # ...
    try:
        token_data = pickle.loads(data, fix_imports=True, encoding='bytes', errors='strict')
        if not isinstance(token_data, dict):
            logger.error(f"Invalid token data type: {type(token_data)}")
            self._remove_file(token_id)
            return None
    except (pickle.UnpicklingError, AttributeError, ValueError) as e:
        logger.error(f"Ошибка десериализации токена {token_id}: {e}")
        self._remove_file(token_id)
        return None
`

#### 1.2 Сериализация (строки 118-127)
`python
def put(self, token_id: str, token_data: Dict) -> bool:
    with self._lock:
        try:
            import pickle
            data = pickle.dumps(token_data)
            data_size = len(data)
            # ...
`

### Проблемы с Pickle

| Проблема | Описание | Серьёзность |
|----------|----------|------------|
| **Удалённое выполнение кода** | pickle может десериализовать вредоносные объекты | КРИТИЧЕСКАЯ |
| **Небезопасная дефолтная кодировка** | Используется ncoding='bytes' | ВЫСОКАЯ |
| **Нет валидации схемы** | Принимается любой dict без проверки структуры | СРЕДНЯЯ |

### Защитные меры (имеющиеся)

Автор добавил некоторые проверки:

`python
# Валидация token_id (строки 85-89)
SAFE_TOKEN_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,128}$')
if not SAFE_TOKEN_ID_PATTERN.match(token_id):
    logger.error(f"Invalid token_id format: {token_id}")
    self._remove_file(token_id)
    return None

# Проверка размера данных (строки 91-94)
if len(data) > 100 * 1024 * 1024:  # 100 MB
    logger.error(f"Token data too large: {len(data)} bytes")
    self._remove_file(token_id)
    return None

# Проверка типа (строки 98-101)
if not isinstance(token_data, dict):
    logger.error(f"Invalid token data type: {type(token_data)}")
    self._remove_file(token_id)
    return None
`

**Но:** Эти меры НЕ защищают от эксплойта через специально созданные pickle-данные.

### Простая альтернатива (НЕ ИСПОЛЬЗУЕТСЯ)

В том же каталоге есть 	oken_disk_cache.py с JSON:

`python
# token_disk_cache.py (НЕ ИСПОЛЬЗУЕТСЯ!)
class TokenDiskCache:
    def load_token(self, token_hash: str) -> Dict[str, Any]:
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)  # Безопасный JSON!
    
    def save_token(self, token_hash: str, token_data: Dict[str, Any]) -> None:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(token_data, f, ensure_ascii=False, indent=2)
`

**Проблема:** 	oken_disk_cache.py НЕ используется системой, не имеет thread safety и не интегрирован.

---

## 2. Thread Safety

### Вердикт: ХОРОШО (8/10)

### Детали

**Файл:** va_ai/memory/cache_disk.py

#### Используемый механизм

`python
class TokenDiskCache:
    def __init__(self, cache_dir: str, max_size_gb: float = 50.0):
        # ...
        self._lock = threading.RLock()  # Рекурсивная блокировка
`

#### Покрытие блокировками

| Метод | Блокировка | Строки |
|-------|------------|--------|
| __init__ | Нет (конструктор) | 15-29 |
| _load_index | Нет (вызывается из __init) | 31-42 |
| _save_index | Нет (-private) | 44-54 |
| _get_file_path | Нет (вызывается под lock) | 56-60 |
| get | with self._lock | 63-116 |
| put | with self._lock | 119-155 |
| _evict_lru | Нет (вызывается под lock) | 157-164 |
| _remove_file | Нет (вызывается под lock) | 166-181 |
| emove | with self._lock | 183-189 |
| clear | with self._lock | 191-208 |
| get_stats | with self._lock | 210-219 |
| __contains__ | with self._lock | 221-223 |
| __len__ | with self._lock | 225-227 |
| get_recent | with self._lock | 229-249 |
| search | НЕТ БЛОКИРОВКИ! | 251-275 |

### Проблемы с Thread Safety

#### 1. Метод search() без блокировки (строки 251-275)

`python
def search(self, query: str, limit: int = 10) -> List[Dict]:
    results = []
    # ...
    with self._lock:
        for token_id, meta in self.file_index.items():
            # ...
            data = self.get(token_id)  # ← Вложенный get() с блокировкой
            # ...
`

**Проблема:** Итерация по ile_index без блокировки + вложенный get() может вызвать deadlock (RLock НЕ поддерживает повторный вход из того же потока).

**Решение:** Метод get() уже имеет блокировку, поэтому вложенный вызов безопасен, но итерация по ile_index без защиты - потенциальная проблема.

#### 2. Вспомогательные методы без блокировок

Методы _load_index, _save_index, _get_file_path, _evict_lru, _remove_file не имеют собственных блокировок, но вызываются только из методов с блокировкой - это допустимо.

### Дополнительные блокировки в HybridTokenCache

`python
# cache_core.py
class HybridTokenCache:
    def __init__(self, ...):
        self.metadata_lock = threading.RLock()
        self.stats_lock = threading.RLock()
        self._lock = threading.RLock()
`

В cache_eviction.py используются все три блокировки:

`python
def _get_token_impl(cache, token_id: str) -> Optional[Dict]:
    with cache._lock:                              # Основная блокировка
        with cache.stats_lock:                     # Статистика
            cache.usage_stats["total_accesses"] += 1
            # ...
`

---

## 3. Система TTL

### Вердикт: ОТСУТСТВУЕТ в TokenDiskCache (2/10)

### Анализ TTL в системе

#### 3.1 HybridTokenCache - cache_ttl НЕ ИСПОЛЬЗУЕТСЯ

`python
# cache_core.py, строка 166
self.cache_settings = {
    "max_memory_size": self.max_memory_tokens,
    "disk_cache_threshold": max(1, self.max_memory_tokens // 2),
    "eviction_policy": eviction_policy,
    "cache_ttl": int(hc.get("cache_ttl", 86400)),  # ← 86400 сек = 24 часа
    "min_relevance_score": float(hc.get("min_relevance_score", 0.3)),
    "max_context_tokens": int(hc.get("max_context_tokens", 1000)),
}
`

**Проблема:** cache_ttl сохраняется в cache_settings, но НИКОГДА не применяется к TokenDiskCache!

#### 3.2 TTL реализован только на уровне данных

В cache_eviction.py TTL проверяется только для特定ных типов данных:

`python
# Контексты (строки 196-207)
def _get_context_impl(cache, session_id: str) -> Optional[Dict]:
    data = cache.get_token(context_key)
    if data and isinstance(data, dict):
        timestamp = data.get("timestamp", 0)
        ttl = data.get("ttl", 3600)  # ← TTL из данных!
        if time.time() - timestamp < ttl:
            return data
    return None

# Документы (строки 247-258)
def _get_document_impl(cache, session_id: str, doc_id: str) -> Optional[Dict]:
    # Аналогичная проверка TTL...

# Поисковые результаты (строки 322-333)
def _get_search_results_impl(cache, query_hash: str) -> Optional[Dict]:
    # Аналогичная проверка TTL...
`

#### 3.3 TokenDiskCache - НЕТ TTL

В cache_disk.py **нет никакого механизма TTL**:

`python
class TokenDiskCache:
    def put(self, token_id: str, token_data: Dict) -> bool:
        # Нет TTL параметра
        # Данные живут вечно пока не вытеснены по размеру
        
    def _evict_lru(self):
        # Только LRU по времени последнего доступа
        # НЕ по возрасту данных!
`

#### 3.4 Сравнение с другими кэшами

| Кэш | TTL | Механизм |
|-----|-----|----------|
| TokenDiskCache | НЕТ | Только LRU по размеру |
| LRUCacheWithTTL (fractal_graph_v2) | ДА (300с) | Проверка при get() |
| ChainCache | ДА (3600с) | _cached_at timestamp |

### Последствия отсутствия TTL

1. **Устаревшие данные** - токены могут жить бесконечно
2. **Неэффективное использование места** - 50GB могут быть заполнены мусором
3. **Несогласованность** - cache_ttl настраивается, но не работает

---

## 4. Интеграция с HybridCache

### Вердикт: ХОРОШО (7/10)

### Схема интеграции

`
HybridTokenCache (cache_core.py)
    │
    ├── vram_cache: LRUCache (max_tokens~384)
    ├── ram_cache: LRUCache (max_tokens~256000)
    │
    └── disk_cache: TokenDiskCache
            ├── Директория: {cache_dir}/token_cache/
            ├── Файлы: {token_id}.bin (pickle)
            └── Индекс: disk_cache_index.json
`

### Инициализация (cache_core.py:131)

`python
self.disk_cache = TokenDiskCache(self.disk_cache_dir, max_disk_cache_gb)
`

### Использование в cache_eviction.py

`python
def _get_token_impl(cache, token_id: str) -> Optional[Dict]:
    # Сначала VRAM
    if cache.gpu_available and token_id in cache.vram_cache:
        return cache.vram_cache.get(token_id)
    
    # Потом RAM  
    if token_id in cache.ram_cache:
        return cache.ram_cache.get(token_id)
    
    # Наконец DISK ← TokenDiskCache
    disk_token = cache._load_token_from_disk(token_id)
    if disk_token:
        #-promotion to memory if hot
        if metadata.get("access_count", 0) > cache.hot_threshold:
            cache._move_token_to_memory(token_id, disk_token)
        return disk_token
`

### Методы TokenDiskCache, используемые HybridTokenCache

| Метод | Использование | Вызов |
|-------|---------------|-------|
| get() | Загрузка токена с диска | _load_token_from_disk() |
| put() | Сохранение на диск | _save_token_to_disk() |
| clear() | Полная очистка | HybridTokenCache.clear() |
| len() | Статистика | get_cache_stats() |

### Проблемы интеграции

1. **Нет TTL** - TokenDiskCache не использует cache_settings['cache_ttl']
2. **Неполная абстракция** - HybridTokenCache не скрывает внутренности TokenDiskCache
3. **Синхронный I/O** - операции с диском блокируют основной поток

---

## Дополнительные находки

### 1. Неиспользуемый token_disk_cache.py

`python
# eva_ai/memory/token_disk_cache.py - НЕ ИСПОЛЬЗУЕТСЯ!
class TokenDiskCache:
    """Простой дисковый кэш для токенов с JSON файлами"""
`

**Проблемы:**
- Не импортируется нигде кроме hybrid_token_cache.py (реэкспорт)
- Нет thread safety
- Нет интеграции с HybridTokenCache
- Не синхронизирован с основной реализацией

### 2. Потенциальные deadlock в search()

`python
def search(self, query: str, limit: int = 10) -> List[Dict]:
    # ...
    with self._lock:
        for token_id, meta in self.file_index.items():
            # ...
            data = self.get(token_id)  # ← Вложенный RLock!
`

Если тот же поток уже держит _lock, вложенный get() вызовет deadlock (RLock поддерживает рекурсию, но семантика может быть нарушена).

### 3. Странное поведение get_recent()

`python
def get_recent(self, limit: int = 20) -> List[Dict]:
    with self._lock:
        # Сортируем...
        for token_id, meta in sorted_items[:limit]:
            data = self.get(token_id)  # ← Вложенный RLock!
`

Вызывает get() (с блокировкой) внутри get_recent() (с блокировкой). Это работает с RLock, но неэффективно.

### 4. Нет batch операций

TokenDiskCache не поддерживает:
- get_many() -批量 получение
- put_many() -批量 запись
- clear_prefix() - очистка по префиксу

---

## Рекомендации

### КРИТИЧЕСКИЕ (немедленно)

1. **Заменить pickle на JSON или msgpack**

`python
# Вместо:
data = pickle.dumps(token_data)

# Использовать:
import json
# или
import msgpack

# JSON (медленнее, но безопасно):
data = json.dumps(token_data).encode('utf-8')

# msgpack (быстрее, компактнее):
import msgpack
data = msgpack.packb(token_data)
`

2. **Удалить неиспользуемый token_disk_cache.py**

`ash
rm eva_ai/memory/token_disk_cache.py
`

### ВЫСОКИЕ (скоро)

3. **Добавить TTL в TokenDiskCache**

`python
class TokenDiskCache:
    def __init__(self, cache_dir: str, max_size_gb: float = 50.0, default_ttl: int = 86400):
        self.default_ttl = default_ttl
        # ...
    
    def put(self, token_id: str, token_data: Dict, ttl: Optional[int] = None) -> bool:
        # ...
        self.file_index[token_id] = {
            'size': data_size,
            'created': time.time(),
            'last_access': time.time(),
            'access_count': 1,
            'ttl': ttl or self.default_ttl  # ← TTL
        }
    
    def _is_expired(self, token_meta: Dict) -> bool:
        age = time.time() - token_meta['created']
        return age > token_meta.get('ttl', self.default_ttl)
`

4. **Активировать cache_ttl из HybridTokenCache**

`python
# cache_core.py
self.disk_cache = TokenDiskCache(
    self.disk_cache_dir, 
    max_disk_cache_gb,
    default_ttl=self.cache_settings['cache_ttl']  # ← Передавать TTL
)
`

### СРЕДНИЕ

5. **Исправить deadlock в search() и get_recent()**

`python
def get_recent(self, limit: int = 20) -> List[Dict]:
    with self._lock:
        # Получить данные без вложенной блокировки
        sorted_items = sorted(...)
        results = []
        for token_id, meta in sorted_items[:limit]:
            # Читаем напрямую из файла, без get()
            file_path = self._get_file_path(token_id)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    # Декодируем без блокировки (файл уже эксклюзивный)
                    results.append(self._decode_data(f.read()))
`

6. **Добавить batch операции**

`python
def get_many(self, token_ids: List[str]) -> Dict[str, Optional[Dict]]:
    results = {}
    with self._lock:
        for token_id in token_ids:
            results[token_id] = self._read_token_unsafe(token_id)
    return results
`

---

## Оценка по 10-балльной шкале

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| **Безопасность (Pickle)** | 2/10 | Критическая уязвимость, удалённое выполнение кода возможно |
| **Thread Safety** | 8/10 | Хорошо, RLock используется, но есть проблемы с search() |
| **TTL система** | 2/10 | Полностью отсутствует в TokenDiskCache, настраивается но не работает |
| **Интеграция HybridCache** | 7/10 | Хорошая интеграция, но не использует все возможности |
| **Производительность** | 7/10 | LRU работает, нет batch операций |
| **Код качество** | 5/10 | Есть дублирование (token_disk_cache.py), нет документации |
| **Архитектура** | 6/10 | Понятная иерархия, но много неиспользуемого кода |
| **Поддержка** | 4/10 | Нет тестов, нет monitoring |
| **Согласованность** | 3/10 | cache_ttl настраивается но не применяется |
| **Общая оценка** | **5/10** | Требует серьёзных доработок |

---

## Выводы

TokenDiskCache является ключевым компонентом системы кэширования EVA AI, обеспечивая персистентность данных на диске. Однако система имеет серьёзные недостатки:

1. **Использование pickle** представляет критическую угрозу безопасности
2. **Отсутствие TTL** приводит к накоплению устаревших данных
3. **Дублирующийся код** создаёт путаницу и поддерживает два несовместимых интерфейса

**Рекомендуемый приоритет действий:**
1. Немедленно заменить pickle на JSON/msgpack
2. Удалить token_disk_cache.py
3. Добавить TTL в TokenDiskCache
4. Активировать существующий cache_ttl

---

*Отчёт подготовлен EVA AI Analysis System*
*Дата: 14.04.2026*
