import pytest

from eva.core.core_brain import CoreBrain


@pytest.fixture(scope="module")
def brain():
    # Инициализация ядра с полным инициализатором компонентов
    b = CoreBrain()
    # Полная инициализация компонентов (включая web_search_engine через ComponentInitializer)
    initialized = b.initialize()
    assert initialized, "CoreBrain.initialize() должен возвращать True"
    # Убедимся, что web search компонент доступен
    assert "web_search_engine" in b.components, "web_search_engine должен быть зарегистрирован в components"
    assert hasattr(b, "web_search_engine"), "web_search_engine должен быть атрибутом CoreBrain"
    assert b.web_search_engine is not None
    return b


def test_web_search_engine_is_initialized(brain: CoreBrain):
    # Проверка, что компонент доступен через components и напрямую
    comp = brain.components.get("web_search_engine")
    assert comp is brain.web_search_engine


def test_web_search_engine_basic_search(brain: CoreBrain):
    engine = brain.web_search_engine

    # Базовый синхронный поиск
    resp = engine.search("test query", max_results=2)
    assert isinstance(resp, dict)
    assert resp.get("status") in {"completed", "failed"}


def test_web_search_and_learn_integration(brain: CoreBrain):
    engine = brain.web_search_engine

    knowledge = engine.web_search_and_learn("artificial intelligence", num_results=2)
    assert isinstance(knowledge, list)
    # Элементы знаний могут быть пустыми при ошибках сети, но структура тестируется, если они есть
    if knowledge:
        item = knowledge[0]
        assert set(["concept", "content", "domain", "source", "relevance", "metadata"]).issubset(item.keys())
        assert isinstance(item["metadata"], dict)
