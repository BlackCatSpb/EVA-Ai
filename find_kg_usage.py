import os
import re

# Find which KG files are actually imported
kg_imports = {}

for root, dirs, files in os.walk('eva_ai'):
    # Skip backup directories
    if 'backup' in root:
        continue
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                for kg_file in ['knowledge_graph.py', 'knowledge_graph_core.py', 
                               'knowledge_graph_integrated.py', 'knowledge_graph_query.py',
                               'knowledge_graph_search.py', 'knowledge_graph_traversal.py',
                               'knowledge_graph_types.py', 'knowledge_graph_module.py']:
                    if kg_file in content and ('import' in content or 'from' in content):
                        if kg_file not in kg_imports:
                            kg_imports[kg_file] = set()
                        kg_imports[kg_file].add(path)

print('=== KG Files Usage ===')
for kg, files in sorted(kg_imports.items(), key=lambda x: -len(x[1])):
    print(f'\n{kg} ({len(files)} files):')
    for f in sorted(files):
        print(f'  - {f}')
