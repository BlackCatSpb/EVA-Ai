# Детальный анализ интеграции KCA и SRG в FCP Pipeline
## Дата: 2026-05-02
---## 1. Точная причина проблемы

### 1.1 Проблема KCA: Неверный формат данных

**Место:** cp_gnn/graph_encoder.py:156-175
**Текущий код:** etrieve_subgraph() возвращает ключ "x" вместо "embeddings"

KCA.forward() ожидает в cp_core/__init__.py:120-125:
`python
def forward(self, X_initial: np.ndarray, subgraph: dict):
    H = subgraph["embeddings"]  # Ожидает ключ "embeddings"
`
Но graph_encoder. retrieve_subgraph() возвращает:
`python
return {
    "x": embeddings,  # <-- НЕПРАВИЛЬНО
    ...
}
`

**Корневая причина:** В graph_encoder. retrieve_subgraph() используется ключ "x" для эмбеддингов, а KCA ожидает "embeddings".

---

### 1.2 Проблема SRG: Заглушка logits

**Место:** cp_gnn/hybrid_integration.zy:425
Текущий код:
`python
logits=np.zeros(100)  # ЗАГЛУШКА
`

**Проблема:**
- softmax([0]*100) = [0.01]*100 (равномерное)
- entropy = ~6.64 (максимум)
- is_confident = entropy <= threshold = False
- SRG ВСЕГДА возвращает "reasoning" mode

**Реальные logits должны быть:**
- После генерации: logits размера vocab_size (обычно 100K+)
- Показывают распределение вероятностей следующе��о токена

---

### 1.3 Проблема Query Embedding

**Место:** cp_gnn/... (некорректное использование tokenizer)

Проблема: Используется input_ids вместо эмбеддингов.

---## 2. Как компоненты ДОЛЖНЫ взаимодействовать

`
User Query
    [1. SRG Pre-Evaluation] - logits=zeros(hidden_dim) для pre-generation
    [2. retrieve_subgraph()] -> Subgraph
    [3. subgraph.to_dict()] -> {"embeddings": [...], ...}
    [4. KCA.forward()] - corrected_states
    [5. SRG Post-Evaluation] - real_Logits после генерации
    [_generate] -> OpenVINO
`

---

## 3. Конкретный код для исправления

### 3.1 graph_encoder.py - ключ "embeddings"

**БЫЛО:**
`python
return {
    "x": embeddings,  # НЕПРАВИЛЬНО
    "node_ids": node_ids,
}
`

**ДОЛЖНО БЫТЬ:**
`python
return {
    "embeddings": embeddings,  # ПРАВИЛЬНО - для KCA
    "node_ids": node_ids,
}
`

---

### 3.2 hybrid_integration.py - KCA forward

**БЫЛО:**
`python
self.kca.forward(hidden_states, self.state.current_subgraph)
`

**ДОЛЖНО БЫТЬ:**
`python
subgraph_data = self.state.current_subgraph
if hasattr(subgraph_data, "to_dict"):
    subgraph_data = subgraph_data.to_dict()
# Проверяем ключи и для "embeddings", и для "x"
embeddings = subgraph_data.get("embeddings") or subgraph_data.get("x")
if embeddings is not None and len(embeddings) > 0:
    self.kca.forward(hidden_states, subgraph_data)
`

---

### 3.3 SRG evaluate - правильная размерность

**БЫЛО:**
`python
logits=np.zeros(100)  # Размер 100
`

**ДОЛЖНО БЫТЬ:**
`python
logits=np.zeros(config.hidden_dim)  # hidden_dim = 25
`

---

## 4. Тесты для проверки

`python
def test_kca_forward_with_subgraph():
    from eva_ai.fcp_core import KnowledgeConsciousAttention
    from eva_ai.memory.fractal_graph_v2.types import Subgraph
    
    kca = KnowledgeConsciousAttention(config)
    subgraph = Subgraph(
        node_ids=["n1", "n2"],
        node_embeddings=np.random.rand(2, 25).astype(np.float32),
        node_contents=["концепт 1", "концепт 2"]
    )
    
    initial_states = np.random.rand(4, 25).astype(np.float32)
    
    # KCA ожидает dict с ключом "embeddings"
    subgraph_dict = subgraph.to_dict()
    assert "embeddings" in subgraph_dict
    
    corrected, info = kca.forward(initial_states, subgraph_dict)
    assert corrected.shape == initial_states.shape

def test_srg_entropy():
    from eva_ai.fcp_core import SemanticRelevanceGate
    
    srg = SemanticRelevanceGate(config)
    query = np.random.rand(25)
    
    # Uniform logits - максимальная энтропия
    uniform = np.zeros(25)
    mode1, metrics1 = srg.evaluate(query, query, uniform)
    assert metrics1["ent"] > 5.0
    
    # Peaked logits - низкая энтропия
    peaked = np.zeros(25)
    peaked[42] = 10.0
    mode2, metrics2 = srg.evaluate(query, query, peaked)
    assert metrics2["ent"] < 2.0
`

---

## 5. Резюме изменений

| Файл | Строка | Изменение |
|------|--------|------------|
| cp_gnn/graph_encoder.zy | 156-175 | Ключ "x" -> "embeddings" |
| cp_gnn/hybrid_integration.zy | 430-436 | Конвертация Subgraph -> dict |
| cp_gnn/hybrid_integration.zy | 425 | zeros(100) -> zeros(config.hidden_dim) |

---

## 6. Дополнительные замечания

### 6.1 Почему zeros(100) - проблема

При Logits = np.zeros(100):
- softmax(logits) = [0.01, 0.01, ..., 0.01]
- entropy = -sum(0.01 * log2(0.01)) = 100 * 0.01 * 6.64 = 6.64

Это максимальная энтропия (максимальная неопределённость).

### 6.2 Размерность logits

SRG. evaluate() получает query и response размера hidden_dim=2560.
Правильный размер logits: config.hidden_dim или размер vocab.

---

*Отчёт: 2026-05-02*
*Автор: EVA-Ai Code Analysis Agent*
