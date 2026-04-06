#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Скрипт самопроверки ЕВА — тестирует основные модули системы."""

import os
import sys
import logging
import time
from datetime import datetime

# === Настройка логгера ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(PROJECT_ROOT, "system_selftest.log")

logger = logging.getLogger("ЕВАSelfTest")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(sh)

# === Импорт ядра ===
try:
    from eva.core.core_brain import ЕВАBrain
except Exception as e:
    logger.error(f"Не удалось импортировать ядро ЕВА: {e}")
    sys.exit(1)

def test_component(name: str, func, *args, **kwargs):
    """Выполняет тест функции/метода с логированием."""
    try:
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"[OK] {name} — выполнено за {elapsed:.2f} сек. Результат: {str(result)[:200]}")
        return True, result
    except Exception as e:
        logger.error(f"[FAIL] {name} — ошибка: {e}")
        return False, None

def main():
    logger.info("=" * 60)
    logger.info(f"Запуск самопроверки ЕВА — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    brain = ЕВАBrain()

    # Инициализация
    ok, _ = test_component("Инициализация ядра", brain.initialize)
    if not ok:
        logger.error("❌ Не удалось инициализировать ядро. Прерывание теста.")
        sys.exit(1)

    # Проверка наличия компонентов
    for comp in [
        "ml_unit",
        "knowledge_graph",
        "memory_manager",
        "web_search_engine",
        "ethics_framework",
        "contradiction_resolver",
    ]:
        status = "OK" if brain.components.get(comp) else "MISSING"
        logger.info(f"Компонент {comp}: {status}")

    # Базовые тесты каждого компонента
    if brain.components.get("ml_unit") and hasattr(brain.components["ml_unit"], "process_text"):
        test_component("MLUnit — NLP обработка", brain.components["ml_unit"].process_text, "Тестовая проверка системы")

    if brain.components.get("knowledge_graph"):
        for method in ["search_nodes", "search", "find_nodes", "query_nodes"]:
            if hasattr(brain.components["knowledge_graph"], method):
                test_component(f"KnowledgeGraph — {method}", getattr(brain.components["knowledge_graph"], method), "ЕВА", 1)
                break

    if brain.components.get("memory_manager") and hasattr(brain.components["memory_manager"], "search"):
        test_component("MemoryManager — поиск", brain.components["memory_manager"].search, "ЕВА")

    if brain.components.get("web_search_engine") and hasattr(brain.components["web_search_engine"], "search"):
        test_component("WebSearchEngine — поиск", brain.components["web_search_engine"].search, "ЕВА", max_results=1)

    if brain.components.get("ethics_framework") and hasattr(brain.components["ethics_framework"], "analyze_content"):
        test_component("EthicsFramework — проверка", brain.components["ethics_framework"].analyze_content, "Тестовый текст", context={})

    if brain.components.get("contradiction_resolver"):
        if hasattr(brain.components["contradiction_resolver"], "get_active_contradictions"):
            test_component("ContradictionResolver — активные противоречия", brain.components["contradiction_resolver"].get_active_contradictions)

    # Тест полного конвейера обработки запроса
    if hasattr(brain, "process_query"):
        test_component("Полный pipeline — process_query", brain.process_query, "Привет! Расскажи о системе ЕВА.")

    logger.info("=" * 60)
    logger.info("Самопроверка завершена. Подробности в system_selftest.log")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
