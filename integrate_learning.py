filepath = r'C:\Users\black\OneDrive\Desktop\CogniFlex\eva\memory\unified_fractal_memory.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Add graph learning imports and initialization
old_init_end = '''        # Загрузка
        self._load()
        
        # Создание статичных узлов моделей (если ещё нет)
        self._ensure_model_nodes()
        
        logger.info(f"UnifiedFractalMemory: {len(self.nodes)} узлов, {len(self.edges)} связей, "
                    f"моделей: {len(self.model_instances)}")'''

new_init_end = '''        # Загрузка
        self._load()
        
        # Создание статичных узлов моделей (если ещё нет)
        self._ensure_model_nodes()
        
        # Graph Learning — обучение через граф опыта
        self.context_builder = None
        self.learning_loop = None
        self.snapshot_manager = None
        self._init_graph_learning()
        
        logger.info(f"UnifiedFractalMemory: {len(self.nodes)} узлов, {len(self.edges)} связей, "
                    f"моделей: {len(self.model_instances)}")'''

if old_init_end in content:
    content = content.replace(old_init_end, new_init_end)
    print('Fixed init')
else:
    print('ERROR: init not found')

# Add _init_graph_learning method before _load
old_load = '''    def _load(self):'''

new_load = '''    def _init_graph_learning(self):
        """Инициализирует систему обучения через граф"""
        try:
            from eva.memory.graph_learning import DynamicContextBuilder, GraphLearningLoop, SnapshotManager
            
            self.context_builder = DynamicContextBuilder(self, max_experiences=5, max_concepts=3)
            self.learning_loop = GraphLearningLoop(self, self.context_builder, min_quality=0.7, cluster_interval=300)
            self.snapshot_manager = SnapshotManager(self, self.context_builder)
            
            self.learning_loop.start()
            logger.info("Graph Learning инициализирован")
        except Exception as e:
            logger.warning(f"Graph Learning не инициализирован: {e}")
            self.context_builder = None
            self.learning_loop = None
            self.snapshot_manager = None

    def save_experience(self, query: str, response: str, model_used: str, quality_score: float) -> str:
        """Сохраняет опыт Q&A для обучения"""
        if self.learning_loop:
            return self.learning_loop.add_experience(query, response, model_used, quality_score)
        return ""

    def get_context_for_query(self, query: str) -> str:
        """Получает контекст из графа для запроса"""
        if self.context_builder:
            return self.context_builder.build_context(query)
        return ""

    def _load(self):'''

if old_load in content:
    content = content.replace(old_load, new_load)
    print('Added graph learning methods')
else:
    print('ERROR: _load not found')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print('Done')