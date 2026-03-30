# working_memory.py
import logging
import time
from typing import Dict, Any, Optional
from eva.memory.memory_core import MemoryNeuron, MemoryField, MemoryDatabase

logger = logging.getLogger("eva.memory.working")

class WorkingMemory:
    """Управляет рабочей (краткосрочной) памятью."""
    
    def __init__(self, brain=None, memory_core=None):
        self.brain = brain
        self.memory_core = memory_core
        self.neurons: Dict[str, MemoryNeuron] = {}
        self.fields: Dict[str, MemoryField] = {}
        self._init_fields()
    
    def _init_fields(self):
        """Инициализирует поля рабочей памяти."""
        self.fields = {
            "general": MemoryField(
                name="general",
                description="Общая рабочая память",
                capacity=100,
                metadata={"type": "active"}
            ),
            "user_interactions": MemoryField(
                name="user_interactions",
                description="Память для текущих взаимодействий с пользователем",
                capacity=50,
                metadata={"type": "active"}
            )
        }
    
    def store_fact(self, fact: Any, user_id: str = "unknown"):
        """Сохраняет факт в рабочую память."""
        neuron_id = f"fact_{hash(str(fact)) % 1000000}_{int(time.time())}"
        neuron = MemoryNeuron(
            id=neuron_id,
            content_type="fact",
            content=fact,
            metadata={"user_id": user_id, "field": "general"}
        )
        self.neurons[neuron_id] = neuron
        field = self.fields["general"]
        field.current_size += 1
        field.last_updated = time.time()
        logger.debug(f"Факт сохранен в рабочую память: {fact}")
        return neuron_id
    
    def get_recent_interactions(self, limit: int = 10) -> list:
        """Возвращает недавние взаимодействия."""
        interactions = [
            neuron for neuron in self.neurons.values() 
            if neuron.metadata.get("field") == "user_interactions"
        ]
        return sorted(interactions, key=lambda x: x.timestamp, reverse=True)[:limit]