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
        'ContradictionAnalyzer': ('contradiction_analysis', 'ContradictionAnalyzer'),
        'ContradictionGenerator': ('contradiction_generator', 'ContradictionGenerator'),
        'GeneratedContradiction': ('contradiction_generator', 'GeneratedContradiction'),
        'create_contradiction_generator': ('contradiction_generator', 'create_contradiction_generator'),
        'ContradictionMiner': ('contradiction_miner', 'ContradictionMiner'),
        'ContradictionStatus': ('contradiction_miner', 'ContradictionStatus'),
        'ContradictionCandidate': ('contradiction_miner', 'ContradictionCandidate'),
        'create_contradiction_miner': ('contradiction_miner', 'create_contradiction_miner')
    }
    
    if name in imports:
        module_name, class_name = imports[name]
        try:
            module = __import__(f'eva_ai.contradiction.{module_name}', fromlist=[class_name])
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Cannot import {name} from {module_name}: {e}")
    
    raise AttributeError(f"module 'eva_ai.contradiction' has no attribute '{name}'")

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
    'ContradictionAnalyzer',
    'ContradictionGenerator',
    'GeneratedContradiction',
    'create_contradiction_generator',
    'ContradictionMiner',
    'ContradictionStatus',
    'ContradictionCandidate',
    'create_contradiction_miner'
]