import os
import sqlite3

# Read FMF_EVA fractal graph
db_path = r"C:\Users\black\OneDrive\Desktop\FMF_EVA\eva_ai\memory\fractal_graph_v2\fractal_graph_v2_data\fractal_graph.db"
print(f"Reading: {db_path}")
print(f"Exists: {os.path.exists(db_path)}")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all nodes
cursor.execute("SELECT id, node_type, content FROM nodes")
rows = cursor.fetchall()
print(f"Total rows: {len(rows)}")

for r in rows[:5]:
    print(f"  {r[0][:30]}... | {r[1]} | {str(r[2])[:60]}")

conn.close()