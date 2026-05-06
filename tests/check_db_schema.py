import sqlite3

# Check our current DB schema
db_path = r'C:\Users\black\OneDrive\Desktop\EVA-Ai\eva_ai\memory\fractal_graph_v2\fractal_graph_v2_data\fractal_graph.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("Current schema:")
cursor.execute("PRAGMA table_info(nodes)")
columns = cursor.fetchall()
for col in columns:
    print(f"  {col[1]}: {col[2]}")

conn.close()