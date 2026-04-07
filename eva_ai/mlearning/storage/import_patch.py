"""
Патч для обновления импортов fractal_store
"""

# Старые импорты (нужно заменить):
# from .fractal_store import FractalWeightStore

# Новые импорты (замена):
# from .fractal_store_new import FractalWeightStore

# Файлы для обновления:
files_to_update = [
    "mlearning/storage/__init__.py",
    "mlearning/storage/unified_fractal_store.py", 
    "mlearning/storage/memory_graph_store.py",
    "mlearning/storage/fractal_weight_store.py",
    "mlearning/storage/fractal_model_loader.py",
    "mlearning/model_manager.py"
]

# Инструкции по замене:
"""
1. В файле mlearning/storage/__init__.py:
   Заменить: from .fractal_store import FractalWeightStore
   На:      from .fractal_store_new import FractalWeightStore

2. В файле mlearning/storage/unified_fractal_store.py:
   Заменить: from .fractal_store import FractalWeightStore, KnowledgeGraphProxy
   На:      from .fractal_store_new import FractalWeightStore, KnowledgeGraphProxy

3. В файле mlearning/storage/memory_graph_store.py:
   Заменить: from .fractal_store import FractalWeightStore
   На:      from .fractal_store_new import FractalWeightStore

4. В файле mlearning/storage/fractal_weight_store.py:
   Заменить: from .fractal_store import FractalContainer
   На:      from .fractal_store_core import FractalContainer

5. В файле mlearning/storage/fractal_model_loader.py:
   Заменить: from .fractal_store import FractalWeightStore
   На:      from .fractal_store_new import FractalWeightStore

6. В файле mlearning/model_manager.py:
   Заменить: from .storage.fractal_store import FractalWeightStore
   На:      from .storage.fractal_store_new import FractalWeightStore
"""

print("Патч для обновления импортов создан")
print("Необходимо вручную обновить импорты в указанных файлах")
