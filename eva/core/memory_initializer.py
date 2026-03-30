"""Инициализатор менеджера памяти для ЕВА."""
import os
import logging
from typing import Optional, Dict, Any
from ..memory.memory_manager import MemoryManager

logger = logging.getLogger("eva.core.memory_initializer")

def initialize_memory_manager(core_brain, config: Optional[Dict[str, Any]] = None) -> Optional[MemoryManager]:
    """
    Инициализирует менеджер памяти.
    
    Args:
        core_brain: Экземпляр CoreBrain
        config: Конфигурация менеджера памяти
        
    Returns:
        Initialized MemoryManager instance or None if initialization fails
    """
    config = config or {}
    
    try:
        # Создаем директорию для кэша, если она не указана
        cache_dir = config.get('cache_dir', os.path.join(core_brain.cache_dir, 'memory'))
        os.makedirs(cache_dir, exist_ok=True)
        
        # Инициализируем менеджер памяти
        memory_manager = MemoryManager(
            cache_dir=cache_dir,
            brain=core_brain,
            knowledge_graph=getattr(core_brain, 'knowledge_graph', None)
        )
        
        # Инициализируем гибридный кэш
        try:
            memory_manager.get_hybrid_cache()
        except Exception as e:
            logger.error(f"Ошибка инициализации гибридного кэша: {e}")
        
        return memory_manager
        
    except Exception as e:
        logger.error(f"Ошибка инициализации менеджера памяти: {e}", exc_info=True)
        return None
