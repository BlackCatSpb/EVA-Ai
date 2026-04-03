"""
Phase 1c: Remove remaining dead code
"""
filepath = r'C:\Users\black\OneDrive\Desktop\CogniFlex\eva\core\core_brain.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove duplicate _register_deferred_system_handlers call at line ~243
old = '''        # Регистрация health checks и recovery strategies для deferred system
        self._register_deferred_system_handlers()
        
        # Инициализация новых менеджеров'''

new = '''        # Инициализация новых менеджеров'''

if old in content:
    content = content.replace(old, new)
    print('Removed duplicate _register_deferred_system_handlers call')
else:
    print('Pattern not found')

# 2. Remove ReasoningIntegration creation (component_initializer creates it)
old2 = '''        # Инициализация ReasoningIntegration
        try:
            from eva.reasoning.integration import ReasoningIntegration
            reasoning_integration = ReasoningIntegration(self)
            self.reasoning_integration = reasoning_integration
            self.query_logger.debug(f"ReasoningIntegration инициализирована")
        except Exception as e:
            self.query_logger.debug(f"ReasoningIntegration недоступен: {e}")
            self.reasoning_integration = None
        
'''

if old2 in content:
    content = content.replace(old2, '')
    print('Removed duplicate ReasoningIntegration creation')
else:
    print('ReasoningIntegration pattern not found')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

# ============================================================
# Clean up component_initializer.py - remove dead factory functions
# ============================================================
filepath = r'C:\Users\black\OneDrive\Desktop\CogniFlex\eva\core\component_initializer.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Remove dead factory functions that are never registered
# create_training_orchestrator, create_learning_manager, create_learning_scheduler

# Find and remove create_training_orchestrator
start = content.find('        def create_training_orchestrator():')
if start != -1:
    end = content.find('\n        def ', start + 10)
    if end != -1:
        content = content[:start] + content[end:]
        print('Removed create_training_orchestrator')

# Find and remove create_learning_manager
start = content.find('        def create_learning_manager():')
if start != -1:
    end = content.find('\n        def ', start + 10)
    if end != -1:
        content = content[:start] + content[end:]
        print('Removed create_learning_manager')

# Find and remove create_learning_scheduler
start = content.find('        def create_learning_scheduler():')
if start != -1:
    end = content.find('\n        #', start + 10)
    if end != -1:
        content = content[:start] + content[end:]
        print('Removed create_learning_scheduler')

# Remove gui factory (gui is not in COMPONENT_LIST)
start = content.find('        def create_gui():')
if start != -1:
    end = content.find('\n        def ', start + 10)
    if end != -1:
        content = content[:start] + content[end:]
        print('Removed create_gui factory')

# Remove gui from component_dependencies
old_deps = '''            'gui': [],
            'web_search_engine': [],'''
new_deps = '''            'web_search_engine': [],'''
if old_deps in content:
    content = content.replace(old_deps, new_deps)
    print('Removed gui from component_dependencies')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

# ============================================================
# Clean up graph_learning.py - remove unused fields
# ============================================================
filepath = r'C:\Users\black\OneDrive\Desktop\CogniFlex\eva\memory\graph_learning.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Remove unused fields from ExperienceNode
old_exp = '''    embedding: List[float] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    usage_count: int = 0
    related_experiences: List[str] = field(default_factory=list)
    hemisphere: str = ""'''

new_exp = '''    embedding: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    usage_count: int = 0
    hemisphere: str = ""'''

if old_exp in content:
    content = content.replace(old_exp, new_exp)
    print('Removed unused ExperienceNode fields (tags, related_experiences)')

# Remove unused fields from ConceptNode
old_concept = '''    embedding: List[float] = field(default_factory=list)
    confidence: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    generation: int = 1
    parent_concepts: List[str] = field(default_factory=list)'''

new_concept = '''    embedding: List[float] = field(default_factory=list)
    confidence: float = 0.0
    created_at: float = field(default_factory=time.time)'''

if old_concept in content:
    content = content.replace(old_concept, new_concept)
    print('Removed unused ConceptNode fields (updated_at, generation, parent_concepts)')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

# ============================================================
# Clean up unified_fractal_memory.py - remove unused MemoryNode fields
# ============================================================
filepath = r'C:\Users\black\OneDrive\Desktop\CogniFlex\eva\memory\unified_fractal_memory.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Remove unused MemoryNode fields
old_node = '''    model_path: str = ""
    model_config: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    version: int = 1
    context: Dict[str, Any] = field(default_factory=dict)
    parent_id: str = ""
    child_ids: List[str] = field(default_factory=list)
    relations: Dict[str, Any] = field(default_factory=dict)
    is_static: bool = False'''

new_node = '''    model_path: str = ""
    model_config: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    parent_id: str = ""
    child_ids: List[str] = field(default_factory=list)
    is_static: bool = False'''

if old_node in content:
    content = content.replace(old_node, new_node)
    print('Removed unused MemoryNode fields (version, relations)')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

# ============================================================
# Clean up gguf_fractal_exporter.py - remove unused self.tensors
# ============================================================
filepath = r'C:\Users\black\OneDrive\Desktop\CogniFlex\eva\memory\gguf_fractal_exporter.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

old = '''        self.metadata = {}
        self.tensors = []
        self.file_size = 0'''

new = '''        self.metadata = {}
        self.file_size = 0'''

if old in content:
    content = content.replace(old, new)
    print('Removed unused self.tensors from GGUFFractalExporter')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print('\n=== Phase 1c cleanup complete ===')
