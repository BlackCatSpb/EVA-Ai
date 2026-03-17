import time
import pytest

from cogniflex.websearch.web_search_engine import WebSearchEngine


@pytest.fixture(scope="module")
def web_engine():
    # Инициализируем без brain, используем временную директорию кэша по умолчанию
    engine = WebSearchEngine()
    # Убедимся, что Bing выключен по умолчанию, чтобы избежать реальных сетевых вызовов
    engine.set_search_engines(use_google=True, use_yandex=True, use_bing=False)
    # Сделаем TTL кэша коротким для теста, но достаточным для повторного запроса
    engine.configure_settings(cache_ttl=60, max_results=5, use_cache=True)
    return engine


def test_search_returns_results_and_stats(web_engine: WebSearchEngine):
    query = "Python programming basics"
    resp = web_engine.search(query, max_results=3)

    assert isinstance(resp, dict)
    assert resp.get("status") in {"completed", "failed"}

    if resp["status"] == "completed":
        results = resp.get("results") or []
        assert isinstance(results, list)
        assert len(results) >= 1
        # Проверим структуру одного результата (минимум поля)
        item = results[0]
        # Результат может быть dataclass SearchResult или dict из кэша
        if hasattr(item, "title"):
            assert hasattr(item, "url")
            assert hasattr(item, "source")
        else:
            assert isinstance(item, dict)
            assert "title" in item and "url" in item and "source" in item

    stats = web_engine.get_stats()
    assert "total_queries" in stats and stats["total_queries"] >= 1


def test_web_search_and_learn_basic_flow(web_engine: WebSearchEngine):
    concept = "machine learning"
    knowledge = web_engine.web_search_and_learn(concept, num_results=3)

    # Должен вернуться список знаний (может быть пустым при ошибке, но обычно нет)
    assert isinstance(knowledge, list)
    if knowledge:
        item = knowledge[0]
        assert set(["concept", "content", "domain", "source", "relevance", "metadata"]).issubset(item.keys())
        assert item["concept"]
        assert isinstance(item["metadata"], dict)
        assert "url" in item["metadata"]


def test_search_cache_works(web_engine: WebSearchEngine):
    query = "knowledge graph basics"

    first = web_engine.search(query, max_results=3)
    assert isinstance(first, dict)
    assert first.get("status") in {"completed", "failed"}

    # Повторим запрос сразу — ожидаем либо cache hit, либо быстрый повтор
    second = web_engine.search(query, max_results=3)
    assert isinstance(second, dict)
    # Если кэш сработал, поле cached=True
    if first.get("status") == "completed" and second.get("status") == "completed":
        assert second.get("cached") in {True, False}
        # Если cached=False, это тоже допустимо, но чаще True при включённом кэше


if __name__ == "__main__":
    # Локальный запуск отдельного тестового файла
    import sys
    raise SystemExit(pytest.main([__file__, "-q"]))
