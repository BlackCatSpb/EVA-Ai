"""
Check if ClosedCognitiveLoop is initialized in EVA brain.
"""
import sys
sys.path.insert(0, 'C:/Users/black/OneDrive/Desktop/EVA-Ai')
sys.path.insert(0, 'C:/Users/black/OneDrive/Desktop/EVA-Ai/eva_ai')
sys.path.insert(0, 'C:/Users/black/OneDrive/Desktop/EVA-Ai/eva_ai/core')

print("=" * 60)
print("CHECKING CLOSED COGNITIVE LOOP STATUS")
print("=" * 60)

# Check the files
import os

print("\n[1] File existence check:")
files = [
    'eva_ai/core/split_model_runner.py',
    'eva_ai/core/kca_integration.py',
    'eva_ai/core/closed_cognitive_loop.py'
]
for f in files:
    exists = os.path.exists(f)
    print(f"  {f}: {'EXISTS' if exists else 'MISSING'}")

# Check integration points
print("\n[2] Integration points check:")

with open('eva_ai/core/init_core.py', 'r', encoding='utf-8') as f:
    content = f.read()
    if "'closed_cognitive_loop'" in content:
        print("  [OK] init_core.py has closed_cognitive_loop in COMPONENT_LIST")
    else:
        print("  [FAIL] init_core.py missing closed_cognitive_loop")

with open('eva_ai/core/init_factories.py', 'r', encoding='utf-8') as f:
    content = f.read()
    if 'def create_closed_cognitive_loop' in content:
        print("  [OK] init_factories.py has create_closed_cognitive_loop")
    else:
        print("  [FAIL] init_factories.py missing create_closed_cognitive_loop")

with open('eva_ai/core/init_connections.py', 'r', encoding='utf-8') as f:
    content = f.read()
    if "'closed_cognitive_loop':" in content:
        print("  [OK] init_connections.py has dependency definition")
    else:
        print("  [FAIL] init_connections.py missing dependency")

# Try to import the components directly
print("\n[3] Direct import test:")
try:
    from split_model_runner import SplitModelRunner
    print("  [OK] SplitModelRunner imported")
except Exception as e:
    print(f"  [FAIL] SplitModelRunner import: {e}")

try:
    from kca_integration import KCAModule, KCAIntegration
    print("  [OK] KCAModule, KCAIntegration imported")
except Exception as e:
    print(f"  [FAIL] kca_integration import: {e}")

try:
    from closed_cognitive_loop import ClosedCognitiveLoop
    print("  [OK] ClosedCognitiveLoop imported")
except Exception as e:
    print(f"  [FAIL] closed_cognitive_loop import: {e}")

# Check brain_query integration
print("\n[4] brain_query integration:")
with open('eva_ai/core/brain_query.py', 'r', encoding='utf-8') as f:
    content = f.read()
    if 'closed_cognitive_loop' in content:
        lines = [l.strip() for l in content.split('\n') if 'closed_cognitive_loop' in l]
        print(f"  [OK] brain_query.py has {len(lines)} references to closed_cognitive_loop")
        for l in lines[:3]:
            print(f"      {l[:80]}")
    else:
        print("  [FAIL] brain_query.py missing closed_cognitive_loop")

print("\n" + "=" * 60)
print("CHECK COMPLETE")
print("=" * 60)