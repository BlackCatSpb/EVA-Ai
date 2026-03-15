# long_term_memory.py
import logging
import time
from typing import Dict, Any, Optional
from cogniflex.memory.memory_core import MemoryNeuron, MemoryField, MemoryDatabase
from collections import defaultdict

logger = logging.getLogger("cogniflex.memory.long_term")

class LongTermMemory:
    """Управляет долгосрочной памятью (семантической и эпизодической)."""
    
    def __init__(self, brain=None, memory_core=None):
        self.brain = brain
        self.memory_core = memory_core
        self.neurons: Dict[str, MemoryNeuron] = {}
        self.fields: Dict[str, MemoryField] = {}
        self.knowledge_graph: Dict[str, list] = defaultdict(list)
        self._init_fields()
    
    def _init_fields(self):
        """Инициализирует поля долгосрочной памяти."""
        self.fields = {
            "semantic": MemoryField(
                name="semantic",
                description="Семантическая память для общих знаний",
                capacity=5000,
                metadata={"type": "long_term"}
            ),
            "episodic": MemoryField(
                name="episodic",
                description="Эпизодическая память для личного опыта",
                capacity=3000,
                metadata={"type": "long_term"}
            )
        }
    
    def add_knowledge(self, concept: str, fact: Any, source: str = "unknown"):
        """Добавляет знание в долгосрочную память."""
        neuron_id = f"knowledge_{hash(concept) % 1000000}_{int(time.time())}"
        neuron = MemoryNeuron(
            id=neuron_id,
            content_type="knowledge",
            content={"concept": concept, "fact": fact},
            metadata={"source": source, "field": "semantic"}
        )
        self.neurons[neuron_id] = neuron
        # Обновляем граф знаний
        self.knowledge_graph[concept].append((neuron_id, fact, source))
        logger.debug(f"Знание добавлено в долгосрочную память: {concept}")
        return neuron_id