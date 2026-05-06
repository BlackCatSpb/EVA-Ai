import sys
sys.path.insert(0, 'C:/Users/black/OneDrive/Desktop/EVA-Ai')

checks = []

# 1. init_factories.py
with open('eva_ai/core/init_factories.py', 'r', encoding='utf-8') as f:
    content = f.read()
    if 'def create_closed_cognitive_loop' in content:
        checks.append('[OK] create_closed_cognitive_loop function exists')
    if "'closed_cognitive_loop'" in content:
        checks.append('[OK] closed_cognitive_loop registered in factories')

# 2. init_core.py
with open('eva_ai/core/init_core.py', 'r', encoding='utf-8') as f:
    content = f.read()
    if "'closed_cognitive_loop'" in content:
        checks.append('[OK] closed_cognitive_loop in COMPONENT_LIST')

# 3. init_connections.py
with open('eva_ai/core/init_connections.py', 'r', encoding='utf-8') as f:
    content = f.read()
    if "'closed_cognitive_loop':" in content:
        checks.append('[OK] closed_cognitive_loop dependency defined')

# 4. brain_query.py
with open('eva_ai/core/brain_query.py', 'r', encoding='utf-8') as f:
    content = f.read()
    if 'closed_cognitive_loop' in content:
        checks.append('[OK] brain_query has ClosedCognitiveLoop integration')

# 5. Files exist
import os
for f in ['eva_ai/core/split_model_runner.py', 'eva_ai/core/kca_integration.py', 'eva_ai/core/closed_cognitive_loop.py']:
    if os.path.exists(f):
        checks.append(f'[OK] {f} exists')
    else:
        checks.append(f'[FAIL] {f} MISSING')

for c in checks:
    print(c)