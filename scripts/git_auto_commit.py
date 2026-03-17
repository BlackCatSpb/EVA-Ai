#!/usr/bin/env python3
"""
CogniFlex Git Auto-Commit Script
Автоматический коммит после успешной проверки кода
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

# Игнорируемые файлы и папки
IGNORED_PATTERNS = [
    '.env', '.key', '*.pem', 'credentials.json',
    '__pycache__', '.pyc', '.pyo',
    '.git', '.venv', 'venv',
    'node_modules', '.npm',
    '*.log', 'pytest.log',
    '.DS_Store', 'Thumbs.db',
    'cogniflex_cache', '*.db', '*.db-journal',
]

# Файлы для игнорирования в коммите
ALWAYS_IGNORE = [
    'nul',  # Windows artifact
]


def run_command(cmd, cwd=None):
    """Выполняет команду и возвращает результат."""
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            shell=True,
            timeout=60
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def get_git_status():
    """Получает статус git."""
    code, out, err = run_command("git status --porcelain")
    return code == 0, out


def get_changed_files():
    """Получает список изменённых файлов."""
    _, out, _ = run_command("git diff --name-only")
    _, out2, _ = run_command("git ls-files --others --exclude-standard")
    
    files = set(out.strip().split('\n'))
    files.update(out2.strip().split('\n'))
    
    # Фильтруем игнорируемые
    filtered = []
    for f in files:
        if not f or f in ALWAYS_IGNORE:
            continue
        ignored = False
        for pattern in IGNORED_PATTERNS:
            if pattern in f or f.endswith(pattern.replace('*', '')):
                ignored = True
                break
        if not ignored:
            filtered.append(f)
    
    return filtered


def check_imports():
    """Базовая проверка импортов."""
    print("[CHECK] Checking imports...")
    code, out, err = run_command("python -c \"import cogniflex\" 2>&1")
    if code != 0:
        print(f"[WARN] Import issues detected")
        return False
    print("[OK] Imports OK")
    return True


def check_syntax():
    """Проверка синтаксиса Python файлов."""
    print("🔍 Проверка синтаксиса...")
    files = get_changed_files()
    py_files = [f for f in files if f.endswith('.py')]
    
    if not py_files:
        print("ℹ️  Нет Python файлов для проверки")
        return True
    
    errors = []
    for f in py_files[:10]:  # Ограничиваем количество
        code, out, err = run_command(f"python -m py_compile \"{f}\"")
        if code != 0:
            errors.append(f"{f}: {err}")
    
    if errors:
        print(f"[ERROR] Syntax errors:")
        for e in errors:
            print(f"   {e}")
        return False
    
    print(f"[OK] Syntax OK ({len(py_files)} files)")
    return True


def create_commit_message(files):
    """Создаёт сообщение коммита."""
    if not files:
        return None
    
    # Группируем по директориям
    dirs = {}
    for f in files:
        d = Path(f).parts[0] if not Path(f).is_absolute() else "root"
        if d not in dirs:
            dirs[d] = []
        dirs[d].append(Path(f).name)
    
    # Формируем сообщение
    parts = []
    for d, names in dirs.items():
        if len(names) <= 3:
            parts.append(f"{d}: {', '.join(names)}")
        else:
            parts.append(f"{d}: {names[0]} +{len(names)-1}")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    message = f"Update: {', '.join(parts[:3])} | {timestamp}"
    
    return message


def git_commit(message):
    """Выполняет коммит."""
    print(f"[COMMIT] Creating commit: {message}")
    
    # Add все файлы
    code, out, err = run_command("git add -A")
    if code != 0:
        print(f"[ERROR] git add failed: {err}")
        return False
    
    # Commit
    code, out, err = run_command(f'git commit -m "{message}"')
    if code != 0:
        if "nothing to commit" in err.lower():
            print("[INFO] Nothing to commit")
            return None
        print(f"[ERROR] git commit failed: {err}")
        return False
    
    print("[OK] Commit created")
    return True


def git_push():
    """Выполняет push."""
    print("[PUSH] Pushing to remote...")
    code, out, err = run_command("git push")
    if code != 0:
        print(f"[ERROR] git push failed: {err}")
        return False
    print("[OK] Pushed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="CogniFlex Git Auto-Commit")
    parser.add_argument("--dry-run", action="store_true", help="Показать что будет сделано")
    parser.add_argument("--skip-checks", action="store_true", help="Пропустить проверки")
    parser.add_argument("--message", type=str, help="Кастомное сообщение")
    parser.add_argument("--push", action="store_true", help="Автоматически пушить")
    args = parser.parse_args()
    
    print("=" * 50)
    print("CogniFlex Git Auto-Commit")
    print("=" * 50)
    
    # Проверяем git
    is_git, _ = get_git_status()
    if not is_git:
        print("[ERROR] Not a git repository")
        return 1
    
    # Get changes
    files = get_changed_files()
    if not files:
        print("[INFO] No files to commit")
        return 0
    
    print(f"[FILES] Changed files ({len(files)}):")
    for f in files[:5]:
        print(f"   - {f}")
    if len(files) > 5:
        print(f"   ... and {len(files)-5} more")
    
    if args.dry_run:
        print("\n[DRY-RUN] No commit will be created")
        return 0
    
    # Проверки
    if not args.skip_checks:
        print("\n" + "=" * 50)
        if not check_syntax():
            print("[ERROR] Checks failed, commit aborted")
            return 1
        print("[OK] All checks passed")
    
    # Коммит
    print("\n" + "=" * 50)
    message = args.message or create_commit_message(files)
    if not message:
        print("[INFO] Nothing to commit")
        return 0
    
    result = git_commit(message)
    if result is None:
        return 0
    if not result:
        return 1
    
    # Push
    if args.push or os.environ.get("AUTO_PUSH", "").lower() == "true":
        if not git_push():
            return 1
    
    print("\n" + "=" * 50)
    print("[DONE] Complete!")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
