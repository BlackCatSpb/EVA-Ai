# Анализ Web Server & GUI EVA

**Дата анализа:** 2026-04-27  
**Версия системы:** 3.1 (CogniFlex / EVA-Ai)

---

## 1. Архитектура сервера

### 1.1 Фреймворк

Система использует **Flask** в качестве веб-фреймворка.

| Компонент | Файл | Описание |
|-----------|------|----------|
| Flask App | server_main.py | Создание Flask приложения |
| WebGUI Class | server_main.py:62 | Главный класс |
| Entry Point | server.py | Реэкспорт модулей |

**Ключевые параметры:**
- Host: 127.0.0.1
- Port: 5555
- Secret Key: из eva_config.json

---

## 2. Модульная структура маршрутов

| Модуль | Файл | Основные функции |
|--------|------|------------------|
| Core Routes | server_routes_core.py | /api/system, /api/status, /api/shutdown, /api/health |
| Chat Routes | server_routes_chat.py | /api/chat, /api/chat/stream, /api/sessions |
| Auth Routes | server_routes_auth.py | /api/login |
| Knowledge Routes | server_routes_knowledge.py | /api/documents, /api/knowledge-graph |
| Analytics Routes | server_routes_analytics.py | /api/analytics, /api/learning |
| Graph Routes | server_routes_graph.py | /api/contradictions, /api/concepts |

---

## 3. Endpoints - основные группы

### 3.1 Core Endpoints
- GET / - Главная страница
- GET /api/system - Информация о системе
- GET /api/status - Статус сервера
- GET /api/health - Проверка здоровья
- POST /api/shutdown - Выключение системы
- GET /api/metrics - Метрики системы

### 3.2 Chat Endpoints
- POST /api/chat - Основной чат
- POST /api/chat/stream - SSE стриминг
- GET/POST/DELETE /api/sessions - Управление сессиями

### 3.3 Authentication
- POST /api/login - Аутентификация (PBKDF2-HMAC-SHA256)

### 3.4 Knowledge & Graph
- GET/POST /api/knowledge-graph - Операции с графом
- GET /api/memory-graph - Фрактальный граф
- GET /api/contradictions - Противоречия
- GET /api/concepts - Концепты

### 3.5 Analytics & Learning
- GET /api/analytics - Дашборд
- GET /api/learning - Статистика обучения
- GET /api/self-dialog - Управление самодиалогом
- GET /api/events/stream - SSE события

---

## 4. Интеграция с Brain

### 4.1 Обработка сообщений (process_message)

1. Ethics Check
2. Entity Extraction
3. Session Update
4. Brain обработка или Document Mode
5. Reasoning Extraction
6. Knowledge Conversion > fractal_graph_v2

### 4.2 GUIBridge (bridge.py)
- Двусторонняя коммуникация Core - GUI
- Подписка на EventBus события
- Кэширование данных для GUI

---

## 5. Выявленные проблемы

### 5.1 Критические
- P1: Дублирование маршрутов (server_routes.py vs модули)
- P2: /api/chat определен в двух файлах
- P3: /api/shutdown без проверки прав

### 5.2 Средние
- M1: Поиск eva_config.json в 6 местах
- M2: 3 уровня fallback для pipeline
- M3: Нет унифицированного интерфейса brain
- M4: Нет TTL для bridge кэша

### 5.3 Мелкие
- S1: Tesseract OCR зависимость
- S2: Нет унифицированной обработки ошибок

---

## 6. Безопасность

- Аутентификация: PBKDF2-HMAC-SHA256 (100000 итераций)
- Уязвимости:
  - Нет CSRF защиты
  - Admin пароль в plaintext
  - Нет rate limiting

---

## 7. Рекомендации

1. Удалить дублирующие маршруты
2. Создать PipelineSelector класс
3. Унифицировать доступ к brain
4. Добавить CSRF и rate limiting

---

## 8. Сводка

| Метрика | Значение |
|---------|----------|
| Всего endpoints | 60+ |
| Модулей маршрутов | 10 |
| Brain интеграций | 12+ |
| SSE streams | 2 |
| Порт по умолчанию | 5555 |

**Оценка:** Модульная архитектура, требует рефакторинга маршрутов и улучшения безопасности.

