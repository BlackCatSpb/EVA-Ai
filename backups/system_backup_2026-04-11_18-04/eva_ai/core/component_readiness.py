"""
Утилиты для проверки готовности компонентов ЕВА
"""
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("eva_ai.component_readiness")

def check_component_readiness(component: Any, component_name: str) -> bool:
    """
    Проверяет готовность компонента к работе.
    
    Args:
        component: Экземпляр компонента
        component_name: Имя компонента для логирования
        
    Returns:
        bool: True если компонент готов, False иначе
    """
    if not component:
        logger.debug(f"Компонент {component_name} не инициализирован")
        return False
    
    # Проверяем метод is_ready()
    if hasattr(component, 'is_ready'):
        try:
            ready = component.is_ready()
            if not ready:
                logger.debug(f"Компонент {component_name} не готов (is_ready() = False)")
            return ready
        except Exception as e:
            logger.warning(f"Ошибка проверки готовности {component_name}: {e}")
            return False
    
    # Проверяем флаг initialized
    if hasattr(component, 'initialized'):
        ready = component.initialized
        if not ready:
            logger.debug(f"Компонент {component_name} не инициализирован")
        return ready
    
    # Если нет специальных методов, считаем готовым
    return True

def check_brain_readiness(brain: Any) -> Dict[str, bool]:
    """
    Проверяет готовность всех компонентов brain.
    
    Args:
        brain: Экземпляр CoreBrain
        
    Returns:
        Dict[str, bool]: Статус готовности каждого компонента
    """
    readiness = {}
    
    # Список ключевых компонентов для проверки
    components_to_check = [
        'model_manager',
        'text_processor', 
        'memory_manager',
        'knowledge_graph',
        'ethics_framework',
        'ml_unit'
    ]
    
    for component_name in components_to_check:
        component = getattr(brain, component_name, None)
        readiness[component_name] = check_component_readiness(component, component_name)
    
    return readiness

def get_readiness_report(brain: Any) -> str:
    """
    Генерирует отчет о готовности компонентов.
    
    Args:
        brain: Экземпляр CoreBrain
        
    Returns:
        str: Текстовый отчет о готовности
    """
    readiness = check_brain_readiness(brain)
    
    ready_count = sum(readiness.values())
    total_count = len(readiness)
    
    report = f"Готовность компонентов: {ready_count}/{total_count}\n"
    
    for component_name, is_ready in readiness.items():
        status = "✓ Готов" if is_ready else "✗ Не готов"
        report += f"  {component_name}: {status}\n"
    
    return report

def wait_for_component_readiness(component: Any, component_name: str, 
                                timeout: float = 10.0, check_interval: float = 0.5) -> bool:
    """
    Ожидает готовности компонента с таймаутом.
    
    Args:
        component: Экземпляр компонента
        component_name: Имя компонента
        timeout: Максимальное время ожидания в секундах
        check_interval: Интервал проверки в секундах
        
    Returns:
        bool: True если компонент стал готов, False при таймауте
    """
    import time
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if check_component_readiness(component, component_name):
            logger.info(f"Компонент {component_name} готов к работе")
            return True
        
        time.sleep(check_interval)
    
    logger.warning(f"Таймаут ожидания готовности компонента {component_name}")
    return False