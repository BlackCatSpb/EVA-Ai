import sqlite3, os

paths = [
    r"c:\\Users\\black\\OneDrive\\Desktop\\CogniFlex\\ml_cache\\models\\models.db",
    r"c:\\Users\\black\\OneDrive\\Desktop\\CogniFlex\\core\\cogniflex_cache\\models\\models.db",
]

for p in paths:
    print("DB:", p, "exists:", os.path.exists(p))
    if not os.path.exists(p):
        continue
    try:
        conn = sqlite3.connect(p)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cur.fetchall()}
        if "models" not in tables:
            print("  No 'models' table\n")
            conn.close()
            continue
        cur.execute("SELECT COUNT(*) FROM models")
        count = cur.fetchone()[0]
        print("  rows:", count)
        cur.execute(
            "SELECT id, name, model_path, model_type, priority FROM models ORDER BY priority DESC, last_updated DESC LIMIT 5"
        )
        for r in cur.fetchall():
            print("   ", r)
        conn.close()
    except Exception as e:
        print("  Error:", e)
print("DONE")
