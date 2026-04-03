"""
Phase 1: Dead code cleanup
Fixes: unused imports, dead methods, dead fields, duplicate components
"""
import os

# ============================================================
# P1.1: Remove dead imports from core_brain.py
# ============================================================
filepath = r'C:\Users\black\OneDrive\Desktop\CogniFlex\eva\core\core_brain.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Remove unused imports
removals = [
    'from typing import TYPE_CHECKING\n',
    'import datetime\n',
    'from .opportunities.learning_detector import LearningOpportunityDetector\n',
    'from .autopilot_cache import AutopilotCache\n',
    'from .opportunities.web_discovery_detector import WebDiscoveryDetector\n',
    'from .opportunities.recovery_detector import ModuleRecoveryDetector\n',
    '    get_generation_coordinator,\n',
]

for removal in removals:
    if removal in content:
        content = content.replace(removal, '')
        print(f'core_brain.py: Removed import: {removal.strip()[:60]}')
    else:
        print(f'core_brain.py: NOT FOUND: {removal.strip()[:60]}')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

# ============================================================
# P1.2: Remove dead imports from other modules
# ============================================================

# recursive_model_pipeline.py — remove Tuple import
filepath = r'C:\Users\black\OneDrive\Desktop\CogniFlex\eva\core\recursive_model_pipeline.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('from typing import Dict, Any, List, Optional, Tuple', 
                          'from typing import Dict, Any, List, Optional')
with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print('recursive_model_pipeline.py: Removed Tuple import')

# component_initializer.py — remove threading import
filepath = r'C:\Users\black\OneDrive\Desktop\CogniFlex\eva\core\component_initializer.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('import threading\n', '')
with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print('component_initializer.py: Removed threading import')

# unified_fractal_memory.py — remove Tuple import
filepath = r'C:\Users\black\OneDrive\Desktop\CogniFlex\eva\memory\unified_fractal_memory.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('from typing import Dict, Any, List, Optional, Tuple',
                          'from typing import Dict, Any, List, Optional')
with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print('unified_fractal_memory.py: Removed Tuple import')

# gguf_fractal_exporter.py — remove hashlib and Optional
filepath = r'C:\Users\black\OneDrive\Desktop\CogniFlex\eva\memory\gguf_fractal_exporter.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('import hashlib\n', '')
content = content.replace('from typing import Dict, Any, List, Optional',
                          'from typing import Dict, Any, List')
with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print('gguf_fractal_exporter.py: Removed hashlib and Optional imports')

# graph_learning.py — remove Optional and Tuple
filepath = r'C:\Users\black\OneDrive\Desktop\CogniFlex\eva\memory\graph_learning.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('from typing import Dict, Any, List, Optional, Tuple',
                          'from typing import Dict, Any, List')
with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print('graph_learning.py: Removed Optional and Tuple imports')

# ============================================================
# P1.3: Remove dead methods
# ============================================================

# core_brain.py — remove _check_system_ready_for_training
filepath = r'C:\Users\black\OneDrive\Desktop\CogniFlex\eva\core\core_brain.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Remove _check_system_ready_for_training method
old_method = '''    def _check_system_ready_for_training(self):
        """Проверяет, готова ли система к обучению."""
        try:
            cpu_usage = self.resource_manager.get_cpu_usage()
            memory_usage = self.resource_manager.get_memory_usage()
            if cpu_usage > 90 or memory_usage > 90:
                return False
            return True
        except Exception as e:
            logger.debug(f"Ошибка проверки готовности к обучению: {e}")
            return True

'''

if old_method in content:
    content = content.replace(old_method, '')
    print('core_brain.py: Removed _check_system_ready_for_training')
else:
    # Try to find it with different formatting
    for line in content.split('\n'):
        if '_check_system_ready_for_training' in line:
            print(f'core_brain.py: Found reference: {line.strip()}')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

# Remove setup_smart_cache_eviction (defined but never called)
old_setup = '''    def setup_smart_cache_eviction(self):
        """Настраивает умную эвикцию кэша"""
        try:
            if hasattr(self, 'token_cache') and self.token_cache:
                self._setup_cache_monitoring()
                logger.info("Smart cache eviction настроен")
        except Exception as e:
            logger.warning(f"Ошибка настройки smart cache eviction: {e}")

'''

if old_setup in content:
    content = content.replace(old_setup, '')
    print('core_brain.py: Removed setup_smart_cache_eviction')
else:
    print('core_brain.py: setup_smart_cache_eviction not found (may already be removed)')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

# ============================================================
# P1.4: Remove dead fields from unified_fractal_memory.py
# ============================================================
filepath = r'C:\Users\black\OneDrive\Desktop\CogniFlex\eva\memory\unified_fractal_memory.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Remove unused NodeType enum values
old_enum = '''class NodeType(Enum):
    # Статичные узлы моделей (НИКОГДА не удаляются)
    MODEL_A = "model_a"           # Qwen 2.5 3B - логика
    MODEL_B = "model_b"           # Qwen 2.5 3B - развитие
    MODEL_C = "model_c"           # Qwen 2.5 Coder 1.5B - код
    
    # Фрактальные уровни знаний
    ROOT = "root"                 # L0 - корень
    CONCEPT = "concept"           # L1 - концепты
    FACT = "fact"                 # L2 - факты
    DETAIL = "detail"             # L3 - детали
    QUERY = "query"               # Узлы запросов
    RESPONSE = "response"         # Узлы ответов
    REASONING = "reasoning"       # Узлы рассуждений
    CONTEXT = "context"           # Контекстные узлы'''

new_enum = '''class NodeType(Enum):
    # Статичные узлы моделей (НИКОГДА не удаляются)
    MODEL_A = "model_a"           # Qwen 2.5 3B - логика
    MODEL_B = "model_b"           # Qwen 2.5 3B - развитие
    MODEL_C = "model_c"           # Qwen 2.5 Coder 1.5B - код
    
    # Фрактальные уровни знаний
    CONCEPT = "concept"           # L1 - концепты
    FACT = "fact"                 # L2 - факты'''

if old_enum in content:
    content = content.replace(old_enum, new_enum)
    print('unified_fractal_memory.py: Removed unused NodeType enum values')
else:
    print('unified_fractal_memory.py: NodeType enum pattern not found')

# Remove unused class constants
old_constants = '''    MAX_LEVELS = 4
    BRANCHING_FACTOR = 16
    HOT_NODE_LIMIT = 1000
    WARM_NODE_LIMIT = 5000
    AUTO_SAVE_INTERVAL = 10'''

new_constants = '''    HOT_NODE_LIMIT = 1000
    WARM_NODE_LIMIT = 5000
    AUTO_SAVE_INTERVAL = 10'''

if old_constants in content:
    content = content.replace(old_constants, new_constants)
    print('unified_fractal_memory.py: Removed unused constants MAX_LEVELS, BRANCHING_FACTOR')
else:
    print('unified_fractal_memory.py: Constants pattern not found')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

# ============================================================
# P1.5: Fix duplicate component creation — remove duplicate ReasoningIntegration
# ============================================================
filepath = r'C:\Users\black\OneDrive\Desktop\CogniFlex\eva\core\core_brain.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Remove duplicate ReasoningIntegration creation in core_brain.initialize()
# ComponentInitializer already creates it, so we skip it here
old_reasoning = '''        # Инициализация ReasoningIntegration
        try:
            from eva.reasoning.integration import ReasoningIntegration
            self.reasoning_integration = ReasoningIntegration(self)
            self.query_logger.info("ReasoningIntegration инициализирована")
        except Exception as e:
            self.query_logger.warning(f"ReasoningIntegration не инициализирована: {e}")
            self.reasoning_integration = None
        
'''

if old_reasoning in content:
    content = content.replace(old_reasoning, '')
    print('core_brain.py: Removed duplicate ReasoningIntegration creation')
else:
    print('core_brain.py: Duplicate ReasoningIntegration pattern not found')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

# ============================================================
# Remove duplicate _register_deferred_system_handlers call
# ============================================================
filepath = r'C:\Users\black\OneDrive\Desktop\CogniFlex\eva\core\core_brain.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Find and remove the second (unconditional) call to _register_deferred_system_handlers
# The first call is inside try/except, the second is unconditional
# We remove the unconditional one

old_dup = '''        # Регистрация обработчиков для отложенной системы
        self._register_deferred_system_handlers()
        
        # Инициализация компонентов через ComponentInitializer'''

new_dup = '''        # Инициализация компонентов через ComponentInitializer'''

if old_dup in content:
    content = content.replace(old_dup, new_dup)
    print('core_brain.py: Removed duplicate _register_deferred_system_handlers call')
else:
    print('core_brain.py: Duplicate call pattern not found')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print('\n=== Phase 1 cleanup complete ===')
