import os
import sqlite3

db_path = 'eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db'
print(f"DB exists: {os.path.exists(db_path)}")

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get table list
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"Tables: {tables}")

    # Count rows in each table
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"{table_name}: {count} rows")

    # Sample some data
    if ('nodes',) in tables or ('node',) in tables:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        for t in cursor.fetchall():
            tname = t[0]
            cursor.execute(f"SELECT * FROM {tname} LIMIT 2")
            rows = cursor.fetchall()
            print(f"\n{tname} sample:")
            for r in rows:
                print(f"  {str(r)[:200]}")

    conn.close()
else:
    print("DB not found")