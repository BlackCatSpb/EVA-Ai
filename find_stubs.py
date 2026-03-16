#!/usr/bin/env python3
"""Скрипт для поиска заглушек (stub) функций в проекте"""

import ast
import os
import re
from pathlib import Path

STUB_PATTERNS = [
    r'^\s*pass\s*$',
    r'^\s*\.\.\.\s*$',
    r'^\s*return\s+None\s*$',
    r'^\s*raise\s+NotImplementedError',
    r'#\s*TODO|FIXME|XXX',
    r'pass\s+#\s*placeholder|stub|TODO',
]

def find_stub_functions(filepath):
    """Находит функции с заглушками в файле"""
    stubs = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
            lines = source.split('\n')
    except Exception as e:
        return [(f"ERROR reading: {e}", 0)]
    
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Get function body
            if len(node.body) == 1:
                # Single statement function - likely a stub
                stmt = node.body[0]
                if isinstance(stmt, ast.Pass):
                    stubs.append((f"{node.name}() - пустая (pass)", node.lineno))
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    if stmt.value.value == Ellipsis:
                        stubs.append((f"{node.name}() - заглушка (...)", node.lineno))
                elif isinstance(stmt, ast.Return) and stmt.value is None:
                    stubs.append((f"{node.name}() - возвращает None", node.lineno))
            elif len(node.body) == 2:
                # Check for docstring + pass pattern
                first = node.body[0]
                second = node.body[1]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                    if isinstance(first.value.value, str):
                        if isinstance(second, ast.Pass):
                            stubs.append((f"{node.name}() - докстринг + pass", node.lineno))
                        elif isinstance(second, ast.Return) and second.value is None:
                            stubs.append((f"{node.name}() - докстринг + return None", node.lineno))
    
    return stubs

def main():
    root_dir = Path(r'c:\Users\black\OneDrive\Desktop\CogniFlex\cogniflex')
    all_stubs = []
    
    print("Поиск заглушек в проекте CogniFlex...")
    
    for py_file in root_dir.rglob('*.py'):
        # Skip special directories
        if any(x in str(py_file) for x in ['__pycache__', '.git', '.venv']):
            continue
            
        stubs = find_stub_functions(str(py_file))
        if stubs:
            rel_path = py_file.relative_to(root_dir.parent)
            all_stubs.append((str(rel_path), stubs))
    
    # Sort by number of stubs (most problematic first)
    all_stubs.sort(key=lambda x: len(x[1]), reverse=True)
    
    print(f"\nНайдено {len(all_stubs)} файлов с заглушками:\n")
    
    for filepath, stubs in all_stubs[:30]:  # Show top 30
        print(f"\n{filepath} ({len(stubs)} заглушек):")
        for stub, lineno in stubs[:10]:  # Show first 10 stubs per file
            print(f"  строка {lineno}: {stub}")
        if len(stubs) > 10:
            print(f"  ... и еще {len(stubs) - 10} заглушек")
    
    if len(all_stubs) > 30:
        print(f"\n... и еще {len(all_stubs) - 30} файлов с заглушками")
    
    # Summary
    total_stubs = sum(len(s[1]) for s in all_stubs)
    print(f"\n\nИТОГО: {total_stubs} заглушек в {len(all_stubs)} файлах")
    
    return all_stubs

if __name__ == '__main__':
    main()
