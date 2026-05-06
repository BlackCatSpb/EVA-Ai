import sqlite3
import os

db_path = "eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db"
if os.path.exists(db_path):
    c = sqlite3.connect(db_path)
    cur = c.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    print("Tables:", [t[0] for t in cur.fetchall()])
    cur.execute("SELECT COUNT(*) FROM nodes")
    print("Nodes count:", cur.fetchone()[0])
    c.close()
else:
    print("No graph found")