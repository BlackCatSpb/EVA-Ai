# Аудит EmbeddingsManager системы EVA AI

**Дата аудита:** Tue Apr 14 2026

## Резюме

EmbeddingsManager имеет критические проблемы:

1. **Device:** init_factories.py использует CPU по умолчанию вместо CUDA
2. **Fallback:** Нет retry на CPU при ошибке CUDA - возвращает случайные векторы
3. **Кэш:** EmbeddingCache не подключен к EmbeddingsManager
4. **Hash:** Использует нестабильный hash() вместо SHA256

**Итоговая оценка: 3.7/10**

## Файлы

- embeddings.py - основная реализация
- embedding_cache.py - persistent cache (не используется)
- init_factories.py - проблема с device
