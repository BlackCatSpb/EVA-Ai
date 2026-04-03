"""
Phase 1 continued: Remove remaining dead code from core_brain.py
"""
filepath = r'C:\Users\black\OneDrive\Desktop\CogniFlex\eva\core\core_brain.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove get_generation_coordinator from import
old = 'from .generation_coordinator import initialize_generation_coordinator, get_generation_coordinator'
new = 'from .generation_coordinator import initialize_generation_coordinator'
if old in content:
    content = content.replace(old, new)
    print('Removed get_generation_coordinator import')

# Also remove the fallback
old2 = '    get_generation_coordinator = None\n'
if old2 in content:
    content = content.replace(old2, '')
    print('Removed get_generation_coordinator fallback')

# 2. Remove duplicate _register_deferred_system_handlers call at line ~244
# Find the pattern: after the try/except block, there's a second unconditional call
old3 = '''        # Регистрация обработчиков для отложенной системы
        self._register_deferred_system_handlers()
        
        # Настройка директории кэша'''

new3 = '''        # Настройка директории кэша'''

if old3 in content:
    content = content.replace(old3, new3)
    print('Removed duplicate _register_deferred_system_handlers call')
else:
    print('Duplicate call pattern not found, searching...')
    # Try alternate pattern
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if '_register_deferred_system_handlers()' in line and i > 220:
            print(f'  Found at line {i+1}: {line.strip()}')

# 3. Remove _check_system_ready_for_training method
# Find the method and remove it
start_marker = '    def _check_system_ready_for_training(self)'
end_marker = '\n    def '

start_idx = content.find(start_marker)
if start_idx != -1:
    # Find the next method definition
    search_start = start_idx + len(start_marker)
    end_idx = content.find(end_marker, search_start)
    if end_idx != -1:
        # Include the newline before the next method
        method_text = content[start_idx:end_idx]
        content = content[:start_idx] + content[end_idx:]
        print(f'Removed _check_system_ready_for_training ({len(method_text)} chars)')
    else:
        print('Could not find end of _check_system_ready_for_training')
else:
    print('_check_system_ready_for_training not found')

# 4. Remove setup_smart_cache_eviction method
start_marker = '    def setup_smart_cache_eviction(self)'
start_idx = content.find(start_marker)
if start_idx != -1:
    search_start = start_idx + len(start_marker)
    end_idx = content.find('\n    def ', search_start)
    if end_idx != -1:
        method_text = content[start_idx:end_idx]
        content = content[:start_idx] + content[end_idx:]
        print(f'Removed setup_smart_cache_eviction ({len(method_text)} chars)')
    else:
        print('Could not find end of setup_smart_cache_eviction')
else:
    print('setup_smart_cache_eviction not found')

# 5. Remove ReasoningIntegration creation (component_initializer creates it)
old_reasoning = '''        # Инициализация ReasoningIntegration
        try:
            from eva.reasoning.integration import ReasoningIntegration
            reasoning_integration = ReasoningIntegration(self)
            self.reasoning_integration = reasoning_integration
            self.query_logger.info("ReasoningIntegration инициализирована")
        except Exception as e:
            self.query_logger.debug(f"ReasoningIntegration недоступен: {e}")
            self.reasoning_integration = None
        
'''

if old_reasoning in content:
    content = content.replace(old_reasoning, '')
    print('Removed duplicate ReasoningIntegration creation')
else:
    print('ReasoningIntegration pattern not found, searching...')
    for line in content.split('\n'):
        if 'ReasoningIntegration' in line:
            print(f'  Found: {line.strip()[:80]}')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print('\nDone')
