import os
import sys
import py_compile
import datetime
from typing import List, Dict, Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
LOG_PATH = os.path.join(ROOT, "pu_check.log")

# Exclude noisy or third-party dirs
EXCLUDE_DIRS = {
    '.git', '__pycache__', 'venv', '.venv', 'Lib', 'site-packages',
    '.vscode', 'Помойка', '.pytest_cache', '.vs', 'cogniflex_models',
}

INCLUDE_ROOTS = [
    ROOT,
    os.path.join(ROOT, 'cogniflex'),
]

def iter_py_files() -> List[str]:
    files: List[str] = []
    for base in INCLUDE_ROOTS:
        if not os.path.exists(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for fn in filenames:
                if fn.endswith('.py'):
                    files.append(os.path.join(dirpath, fn))
    # De-duplicate while preserving order
    seen = set()
    unique: List[str] = []
    for p in files:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def main() -> int:
    py_files = iter_py_files()
    errors: List[Dict[str, Any]] = []

    for fp in py_files:
        try:
            py_compile.compile(fp, doraise=True)
        except SyntaxError as e:
            rel = os.path.relpath(fp, ROOT)
            errors.append({
                'type': 'SyntaxError',
                'file': rel,
                'line': e.lineno,
                'offset': e.offset,
                'msg': e.msg,
                'text': (e.text or '').strip(),
            })
        except Exception as e:
            rel = os.path.relpath(fp, ROOT)
            errors.append({
                'type': type(e).__name__,
                'file': rel,
                'line': getattr(e, 'lineno', None),
                'msg': str(e),
            })

    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write("PU Analysis Log (Python files)\n")
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Root: {ROOT}\n")
        f.write(f"Files checked: {len(py_files)}\n")
        f.write(f"Errors found: {len(errors)}\n\n")
        for i, err in enumerate(errors, 1):
            f.write(f"[{i}] {err['type']} in {err['file']}")
            if err.get('line') is not None:
                f.write(f":{err.get('line')}")
            f.write("\n")
            if err.get('msg'):
                f.write(f"    Message: {err['msg']}\n")
            if err.get('offset'):
                f.write(f"    Offset: {err['offset']}\n")
            if err.get('text'):
                f.write(f"    Code: {err['text']}\n")
            f.write("\n")

    print(f"Wrote log to {LOG_PATH}. Files checked: {len(py_files)}. Errors: {len(errors)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
