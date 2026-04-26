import sqlite3
import os
import json

db_path = 'eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM nodes LIMIT 10")
    rows = cursor.fetchall()

    print('=== Nodes ===')
    for row in rows:
        print(row)

    cursor.execute("PRAGMA table_info(nodes)")
    columns = cursor.fetchall()
    print('\n=== Node Columns ===')
    for col in columns:
        print(col)

    conn.close()
else:
    print('No graph DB found')