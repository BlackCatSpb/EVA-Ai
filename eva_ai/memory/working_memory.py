# working_memory.py
import logging
import time
from typing import Dict, Any, Optional
from eva_ai.memory.memory_core import MemoryNeuron, MemoryField, MemoryDatabase

logger = logging.getLogger("eva_ai.memory.working")

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
        field_name = "general"
        if field_name not in self.fields:
            logger.warning(f"Поле {field_name} не существует, создаем")
            self.fields[field_name] = MemoryField(
                name=field_name,
                description="Общая рабочая память",
                capacity=100,
                metadata={"type": "active"}
            )
        
        field = self.fields[field_name]
        if field.current_size >= field.capacity:
            self._evict_oldest(field_name)
        
        neuron_id = f"fact_{hash(str(fact)) % 1000000}_{int(time.time())}"
        neuron = MemoryNeuron(
            id=neuron_id,
            content_type="fact",
            content=fact,
            metadata={"user_id": user_id, "field": field_name}
        )
        self.neurons[neuron_id] = neuron
        field.current_size += 1
        field.last_updated = time.time()
        logger.debug(f"Факт сохранен в рабочую память: {fact}")
        return neuron_id
    
    def _evict_oldest(self, field_name: str):
        """Удаляет самый старый нейрон из указанного поля."""
        field = self.fields.get(field_name)
        if not field:
            return
        
        oldest_key = None
        oldest_time = float('inf')
        for key, neuron in self.neurons.items():
            if neuron.metadata.get("field") == field_name:
                if neuron.timestamp < oldest_time:
                    oldest_time = neuron.timestamp
                    oldest_key = key
        
        if oldest_key:
            del self.neurons[oldest_key]
            field.current_size = max(0, field.current_size - 1)
            logger.debug(f"Вытеснен нейрон {oldest_key} из поля {field_name}")
    
    def get_recent_interactions(self, limit: int = 10) -> list:
        """Возвращает недавние взаимодействия."""
        interactions = [
            neuron for neuron in self.neurons.values() 
            if neuron.metadata.get("field") == "user_interactions"
        ]
        return sorted(interactions, key=lambda x: x.timestamp, reverse=True)[:limit]