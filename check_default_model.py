import sqlite3, os
db = r"core/cogniflex_cache/models/models.db"
print("DB exists:", os.path.exists(db))
if os.path.exists(db):
    con = sqlite3.connect(db)
    cur = con.cursor()
    row = cur.execute("SELECT id,name,model_path,model_type,priority,tags FROM models WHERE id='default_text_gen'").fetchone()
    print("default_text_gen:", row)
    total = cur.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    print("Total models:", total)
    con.close()
