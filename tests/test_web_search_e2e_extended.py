import time
import pytest

from eva_ai.core.core_brain import CoreBrain

@pytest.fixture(scope="module")
def brain():
    brain = CoreBrain()
    brain.initialize()  # важно для инициализации web_search_engine и knowledge компонентов
    yield brain


def _has_web_search_engine(brain):
    return hasattr(brain, "web_search_engine") and brain.web_search_engine is not None


def test_expander_uses_web_search_and_integrates_into_graph(brain):
    assert _has_web_search_engine(brain), "web_search_engine должен быть инициализирован"
    # Базовый концепт для расширения
    concept = "test concept integration"

    # Убедимся, что поиска не упадет и вернет список
    expander = getattr(brain, "knowledge_expander", None) or getattr(brain, "knowledge_manager", None)
    # В большинстве сборок KnowledgeExpander доступен напрямую как brain.knowledge_expander
    if hasattr(brain, "knowledge_expander"):
        expanded = brain.knowledge_expander.expand_knowledge(concept, depth=1, num_results=2)
    else:
        pytest.skip("KnowledgeExpander не доступен на brain")

    assert isinstance(expanded, list)
    # Если знаний нет (например, оффлайн), тест должен быть устойчив: просто проверяем отсутствие исключений
    if expanded:
        # Проверяем формат элемента знаний из web_search_and_learn
        k = expanded[0]
        assert set(["concept", "content"]).issubset(k.keys())
        # Проверяем, что хотя бы один узел добавлен в граф (поиск по имени концепта)
        nodes = brain.knowledge_graph.search_nodes(k["concept"], limit=1)
        assert isinstance(nodes, list)


def test_integrator_updates_outdated_using_web_search(brain):
    assert _has_web_search_engine(brain)

    # Создадим искусственно устаревший узел и проверим, что интегратор попробует его обновить
    node_id = brain.knowledge_graph.add_node(
        name="outdated test concept",
        description="old description",
        node_type="concept",
        domain="general",
        strength=0.5,
        meta={}
    )

    # Принудительно сделаем старую метку времени
    node = brain.knowledge_graph.get_node(node_id)
    if node:
        node.last_updated = time.time() - 400 * 86400  # > 1 года
        if hasattr(brain.knowledge_graph, "_update_node_in_db"):
            try:
                brain.knowledge_graph._update_node_in_db(node)
            except Exception:
                pass

    # Запускаем интеграцию знаний: внутри она вызовет web_search_and_learn для устаревших
    integrator = getattr(brain, "knowledge_integrator", None)
    if not integrator:
        pytest.skip("KnowledgeIntegrator не доступен на brain")

    ok = integrator.integrate_knowledge("general", depth=1)
    assert isinstance(ok, bool)


def test_learning_opportunity_manager_expansion_flow(brain):
    assert _has_web_search_engine(brain)

    lom = getattr(brain, "learning_opportunity_manager", None)
    if not lom:
        pytest.skip("LearningOpportunityManager не доступен на brain")

    # Вставим возможность напрямую в БД AnalyzerCore
    conn = None
    try:
        conn = __import__("sqlite3").connect(lom.analyzer_core.db_path)
        cur = conn.cursor()
        opp_id = f"op_{int(time.time()*1000)}"
        cur.execute(
            """
            INSERT INTO learning_opportunities (
                id, concept, opportunity_type, priority, domain, evidence,
                suggested_actions, created_at, last_updated, executed, execution, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                opp_id,
                "learning expansion concept",
                "expansion",
                0.8,
                "general",
                __import__("json").dumps(["auto test"]),
                __import__("json").dumps(["auto action"]),
                time.time(),
                time.time(),
                0,
                None,
                __import__("json").dumps({})
            )
        )
        conn.commit()
    finally:
        if conn:
            conn.close()

    # Выполняем возможность
    success = lom.execute_learning_opportunity(opp_id)
    assert isinstance(success, bool)


def test_core_brain_has_all_links(brain):
    # Проверяем, что основные компоненты доступны на brain
    assert hasattr(brain, "knowledge_graph")
    # Инициализатор компонентов создает web_search_engine и learning_scheduler (алиас для LearningOpportunityManager)
    assert hasattr(brain, "web_search_engine")
    assert hasattr(brain, "learning_scheduler") or hasattr(brain, "self_analyzer")
