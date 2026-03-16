#!/usr/bin/env python3
"""Скрипт для копирования файлов из worktree в основную директорию"""

import os
import shutil
from pathlib import Path

# Пути
worktree_root = Path(r'c:\Users\black\.windsurf\worktrees\CogniFlex\CogniFlex-506e2973')
main_root = Path(r'c:\Users\black\OneDrive\Desktop\CogniFlex')

def copy_tree(src, dst):
    """Рекурсивное копирование директории с заменой"""
    copied = 0
    skipped = 0
    errors = []
    
    for item in src.rglob('*'):
        if item.is_file():
            # Пропускаем .git и __pycache__
            if '.git' in str(item) or '__pycache__' in str(item):
                continue
                
            # Вычисляем относительный путь
            rel_path = item.relative_to(src)
            dst_path = dst / rel_path
            
            try:
                # Создаем директории если нужно
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Копируем файл
                shutil.copy2(item, dst_path)
                copied += 1
                print(f"  Copied: {rel_path}")
            except Exception as e:
                errors.append(f"{rel_path}: {e}")
                skipped += 1
    
    return copied, skipped, errors

if __name__ == '__main__':
    print(f"Копирование из:\n  {worktree_root}\nв:\n  {main_root}\n")
    
    copied, skipped, errors = copy_tree(worktree_root, main_root)
    
    print(f"\nГотово!")
    print(f"  Скопировано: {copied} файлов")
    print(f"  Пропущено с ошибками: {skipped} файлов")
    
    if errors:
        print(f"\nОшибки:")
        for err in errors[:10]:  # Показываем первые 10 ошибок
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... и еще {len(errors) - 10} ошибок")
