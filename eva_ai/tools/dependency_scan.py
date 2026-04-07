import os, sys, ast, json, datetime
from typing import Dict, Set, List, Optional

# Detect project root (two levels up from this file: eva/tools/ -> project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

# Primary package root to scan
PKG_ROOT = os.path.join(PROJECT_ROOT, 'eva')
if not os.path.isdir(PKG_ROOT):
    # Fallback: if script moved, try current parent as package root
    PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Report outputs at project root
REPORT_PATH = os.path.join(PROJECT_ROOT, 'dependency_report.log')
GRAPH_JSON = os.path.join(PROJECT_ROOT, 'dependency_graph.json')
DOT_PATH = os.path.join(PROJECT_ROOT, 'dependency_graph.dot')

EXCLUDE_DIRS = {
    '.git', '__pycache__', 'venv', '.venv', 'Lib', 'site-packages',
    '.vscode', '.vs', '.pytest_cache', 'Помойка', 'eva_models'
}

modules: Dict[str, str] = {}
package_modules: Set[str] = set()
files: List[str] = []

for dirpath, dirnames, filenames in os.walk(PKG_ROOT):
    dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
    for fn in filenames:
        if fn.endswith('.py'):
            fp = os.path.join(dirpath, fn)
            files.append(fp)
            rel = os.path.relpath(fp, PKG_ROOT).replace(os.sep, '/')
            parts = rel.split('/')
            is_pkg = parts[-1] == '__init__.py'
            if is_pkg:
                parts = parts[:-1]
            else:
                parts[-1] = parts[-1][:-3]
            mod = 'eva' + ('.' + '.'.join([p for p in parts if p]) if parts else '')
            modules[fp] = mod
            if is_pkg:
                package_modules.add(mod)

present_modules: Set[str] = set(modules.values())

def resolve_from(current_mod: str, module: Optional[str], level: int) -> Optional[str]:
    base_parts = current_mod.split('.')
    # If current module is a package (__init__.py), do not drop the last segment
    if current_mod not in package_modules and base_parts:
        # drop the module name segment for non-packages
        base_parts = base_parts[:-1]
    # In Python, ImportFrom with level=N means:
    #   N=1 -> current package (no upward movement)
    #   N>1 -> go up (N-1) packages
    if level and level > 0:
        up = max(level - 1, 0)
        if up:
            if len(base_parts) - up < 1:
                return None
            base_parts = base_parts[:len(base_parts) - up]
    if module:
        target = '.'.join(base_parts + module.split('.'))
    else:
        target = '.'.join(base_parts)
    return target if target else None

edges: Dict[str, Set[str]] = {m: set() for m in present_modules}
external_imports: Dict[str, Set[str]] = {m: set() for m in present_modules}
missing_internal: Dict[str, Set[str]] = {m: set() for m in present_modules}

for fp, mod in modules.items():
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            src = f.read()
        tree = ast.parse(src, filename=fp)
    except Exception:
        continue

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name.startswith('eva'):
                    target = name
                    edges[mod].add(target)
                    if target not in present_modules:
                        missing_internal[mod].add(target)
                else:
                    external_imports[mod].add(name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            try:
                target_mod = resolve_from(mod, node.module, node.level or 0) if node.level else node.module
                if target_mod:
                    if target_mod.startswith('eva'):
                        edges[mod].add(target_mod)
                        if target_mod not in present_modules:
                            missing_internal[mod].add(target_mod)
                    else:
                        external_imports[mod].add(target_mod.split('.')[0])
            except Exception:
                pass

# Detect cycles
WHITE, GRAY, BLACK = 0, 1, 2
color: Dict[str, int] = {m: WHITE for m in present_modules}
cycle_paths: List[List[str]] = []
stack: List[str] = []

sys.setrecursionlimit(10000)

def dfs(u: str):
    color[u] = GRAY
    stack.append(u)
    for v in edges.get(u, ()):
        if not v.startswith('eva'):
            continue
        if v not in present_modules:
            continue
        if color[v] == WHITE:
            dfs(v)
        elif color[v] == GRAY:
            if v in stack:
                idx = stack.index(v)
                cycle_paths.append(stack[idx:] + [v])
    stack.pop()
    color[u] = BLACK

for m in sorted(present_modules):
    if color[m] == WHITE:
        dfs(m)

if __name__ == '__main__':
    # In-degree
    indeg: Dict[str, int] = {m: 0 for m in present_modules}
    for u, vs in edges.items():
        for v in vs:
            if v in indeg:
                indeg[v] += 1

    # Write report
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('Dependency Analysis Report\n')
        f.write(f'Timestamp: {datetime.datetime.now().isoformat()}\n')
        f.write(f'Root: {PROJECT_ROOT}\n')
        f.write(f'Total modules: {len(present_modules)}\n')
        f.write(f'Total edges: {sum(len(v) for v in edges.values())}\n\n')
        missing_total = sum(len(s) for s in missing_internal.values())
        f.write(f'Missing internal targets: {missing_total}\n')
        if missing_total:
            for m, miss in sorted(missing_internal.items()):
                if miss:
                    f.write(f'  - {m}:\n')
                    for t in sorted(miss):
                        f.write(f'      * {t}\n')
            f.write('\n')
        f.write(f'Cycles detected: {len(cycle_paths)}\n')
        if cycle_paths:
            for i, cyc in enumerate(cycle_paths, 1):
                f.write(f'  [{i}] ' + ' -> '.join(cyc) + '\n')
            f.write('\n')
        top = sorted(indeg.items(), key=lambda x: x[1], reverse=True)[:10]
        f.write('Top modules by in-degree (most depended-on):\n')
        for m, deg in top:
            f.write(f'  - {m}: {deg}\n')
        f.write('\n')
        f.write('Note: CoreBrain expected as orchestrator: eva_ai.core.core_brain\n')

    # Write JSON graph
    graph = {k: sorted(list(v)) for k, v in edges.items()}
    with open(GRAPH_JSON, 'w', encoding='utf-8') as jf:
        json.dump({
            'generated_at': datetime.datetime.now().isoformat(),
            'modules': sorted(list(present_modules)),
            'edges': graph,
            'indegree': indeg,
            'cycles': cycle_paths,
            'missing_internal': {k: sorted(list(v)) for k, v in missing_internal.items() if v},
        }, jf, ensure_ascii=False, indent=2)

    # Optional GraphViz .dot
    with open(DOT_PATH, 'w', encoding='utf-8') as df:
        df.write('digraph eva_deps {\n  rankdir=LR;\n  node [shape=box, fontsize=10];\n')
        for u, vs in graph.items():
            u2 = u.replace('.', '_')
            df.write(f'  {u2} [label="{u}"];\n')
            for v in vs:
                if v.startswith('eva'):
                    v2 = v.replace('.', '_')
                    df.write(f'  {u2} -> {v2};\n')
        df.write('}\n')

    print(f'Wrote {REPORT_PATH}, {GRAPH_JSON}, {DOT_PATH}')