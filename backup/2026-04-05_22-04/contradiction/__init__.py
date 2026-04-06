"""
Модуль обнаружения и разрешения противоречий в системе ЕВА
"""

# Условные импорты для избежания циклических зависимостей
def __getattr__(name):
    """Lazy loading of contradiction classes to avoid circular imports."""
    imports = {
        'Contradiction': ('contradiction_core', 'Contradiction'),
        'OptimizedContradictionDetector': ('contradiction_core', 'OptimizedContradictionDetector'),
        'ContradictionManager': ('contradiction_manager', 'ContradictionManager'),
        'ContradictionResolver': ('contradiction_resolver', 'ContradictionResolver'),
        'ContradictionDetector': ('contradiction_detection', 'ContradictionDetector'),
        'ContradictionResolution': ('contradiction_resolution', 'ContradictionResolution'),
        'ContradictionResolutionStrategy': ('contradiction_strategies', 'ContradictionResolutionStrategy'),
        'ContradictionResponseGenerator': ('contradiction_responses', 'ContradictionResponseGenerator'),
        'SourceReputationSystem': ('contradiction_reputation', 'SourceReputationSystem'),
        'ContradictionLearning': ('contradiction_learning', 'ContradictionLearning'),
        'ContradictionAnalyzer': ('contradiction_analysis', 'ContradictionAnalyzer')
    }
    
    if name in imports:
        module_name, class_name = imports[name]
        try:
            module = __import__(f'eva.contradiction.{module_name}', fromlist=[class_name])
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Cannot import {name} from {module_name}: {e}")
    
    raise AttributeError(f"module 'eva.contradiction' has no attribute '{name}'")

__all__ = [
    'Contradiction',
    'OptimizedContradictionDetector',
    'ContradictionManager',
    'ContradictionResolver',
    'ContradictionDetector',
    'ContradictionResolution',
    'ContradictionResolutionStrategy',
    'ContradictionResponseGenerator',
    'SourceReputationSystem',
    'ContradictionLearning',
    'ContradictionAnalyzer'
]