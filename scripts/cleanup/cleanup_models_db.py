import argparse
import os
import sqlite3
import sys
from typing import List, Tuple, Optional

DB_DEFAULT = os.path.join("core", "cogniflex_cache", "models", "models.db")
MODELS_DIR_DEFAULT = os.path.join("cogniflex", "mlearning", "cogniflex_models")


def normpath(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    try:
        ap = os.path.abspath(p)
        ap = os.path.normpath(ap)
        # На Windows сравниваем без учета регистра
        if os.name == 'nt':
            ap = ap.lower()
        return ap
    except Exception:
        return p


def list_local_models(models_dir: str) -> List[str]:
    """Вернёт список абсолютных путей к локальным моделям (подпапки с config.json)."""
    res: List[str] = []
    base = os.path.abspath(models_dir)
    if not os.path.isdir(base):
        return res
    try:
        for entry in os.listdir(base):
            p = os.path.join(base, entry)
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json")):
                res.append(os.path.normpath(p))
    except Exception:
        pass
    return res


def fetch_models(conn: sqlite3.Connection) -> List[Tuple[str, str, str, int]]:
    cur = conn.cursor()
    cur.execute("SELECT id, model_path, model_type, priority FROM models")
    return [(r[0], r[1] or "", r[2] or "", int(r[3] or 0)) for r in cur.fetchall()]


def main():
    parser = argparse.ArgumentParser(description="Cleanup CogniFlex models DB: remove nonexistent models and keep only Small")
    parser.add_argument("--db", default=DB_DEFAULT, help="Path to models.db (default: core/cogniflex_cache/models/models.db)")
    parser.add_argument("--models-dir", default=MODELS_DIR_DEFAULT, help="Path to local models root (default: cogniflex/mlearning/cogniflex_models)")
    parser.add_argument("--keep-path", default=None, help="Absolute path to the ONLY model to keep (besides alias). If not set, auto-detect single local model.")
    parser.add_argument("--apply", action="store_true", help="Apply changes (otherwise dry-run)")
    args = parser.parse_args()

    db_path = args.db
    if not os.path.exists(db_path):
        print(f"[ERR] DB not found: {db_path}")
        sys.exit(1)

    # Определяем путь, который нужно оставить
    keep_path_abs: Optional[str] = None
    if args.keep_path:
        keep_path_abs = normpath(args.keep_path)
    else:
        locals_list = list_local_models(args.models_dir)
        if len(locals_list) == 1:
            keep_path_abs = normpath(locals_list[0])
        elif len(locals_list) > 1:
            print("[ERR] Multiple local models found. Specify --keep-path explicitly:")
            for p in locals_list:
                print("   ", p)
            sys.exit(2)
        else:
            print("[ERR] No local models detected.")
            sys.exit(3)

    print("[INFO] Will keep only models pointing to:")
    print("       ", keep_path_abs)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    rows = fetch_models(conn)
    to_delete: List[str] = []
    to_fix_type: List[str] = []

    # Соберём нормализованные пути -> мульти-ids
    for mid, mpath, mtype, prio in rows:
        mpath_norm = normpath(mpath)
        exists_local = bool(mpath and (os.path.isdir(mpath) or os.path.isabs(mpath)))
        keep = False
        reason = ""

        # Разрешаем alias default_text_gen, но только если он указывает на keep_path
        if mid == "default_text_gen":
            if mpath_norm == keep_path_abs:
                keep = True
                # Проверим тип
                if mtype.lower() != "rugpt3":
                    to_fix_type.append(mid)
            else:
                reason = "alias points elsewhere"
        else:
            # Прочие записи храним только если их путь совпадает с keep_path
            if mpath_norm == keep_path_abs:
                keep = True
                if mtype.lower() != "rugpt3":
                    to_fix_type.append(mid)
            else:
                # Если путь локальный, но директории не существует — удаляем
                if exists_local and not os.path.exists(mpath):
                    reason = "missing directory"
                else:
                    reason = "not the selected model"

        if not keep:
            to_delete.append(mid)
            print(f"[DEL] {mid} -> {mpath} ({reason})")
        else:
            print(f"[KEEP] {mid} -> {mpath} (type={mtype}, prio={prio})")

    if not args.apply:
        print("[DRY-RUN] No changes applied. Use --apply to modify DB.")
        conn.close()
        return

    # Применяем изменения
    if to_delete:
        qmarks = ",".join(["?"] * len(to_delete))
        cur.execute(f"DELETE FROM models WHERE id IN ({qmarks})", to_delete)
        print(f"[APPLY] Deleted {cur.rowcount} records")

    # Исправляем тип у сохраняемых записей
    for mid in set(to_fix_type):
        cur.execute("UPDATE models SET model_type = 'rugpt3' WHERE id = ?", (mid,))
    if to_fix_type:
        print(f"[APPLY] Fixed model_type='rugpt3' for: {sorted(set(to_fix_type))}")

    # Ставим высокий приоритет alias и умеренный приоритет для keep записи
    try:
        cur.execute("UPDATE models SET priority = 100 WHERE id = 'default_text_gen'")
    except Exception:
        pass
    try:
        # Найдем любую не-alias запись, которая указывает на keep_path
        cur.execute("UPDATE models SET priority = 95 WHERE id != 'default_text_gen' AND lower(replace(model_path,'\\\\','/')) = lower(replace(?, '\\\\','/'))", (keep_path_abs,))
    except Exception:
        pass

    conn.commit()
    try:
        cur.execute("VACUUM")
    except Exception:
        pass
    conn.close()
    print("[DONE] DB cleaned")


if __name__ == "__main__":
    main()
