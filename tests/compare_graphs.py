import os
import sqlite3

# Check FMF_EVA fractal graph
db_path = r"C:\Users\black\OneDrive\Desktop\FMF_EVA\eva_ai\memory\fractal_graph_v2\fractal_graph_v2_data\fractal_graph.db"
print(f"FMF_EVA DB exists: {os.path.exists(db_path)}")

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"Tables: {[t[0] for t in tables]}")
    
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"  {table_name}: {count} rows")
    
    # Sample nodes
    cursor.execute("SELECT id, node_type, content FROM nodes LIMIT 10")
    rows = cursor.fetchall()
    print("\nNodes sample:")
    for r in rows:
        content = r[2][:100] if r[2] else ""
        print(f"  {r[0][:20]}... | {r[1]} | {content}")
    
    conn.close()

print("\n" + "="*60 + "\n")

# Check our fractal graph
our_db = r"C:\Users\black\OneDrive\Desktop\EVA-Ai\eva_ai\memory\fractal_graph_v2\fractal_graph_v2_data\fractal_graph.db"
print(f"Our DB exists: {os.path.exists(our_db)}")
if os.path.exists(our_db):
    conn = sqlite3.connect(our_db)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"Tables: {[t[0] for t in tables]}")
    
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"  {table_name}: {count} rows")
    
    conn.close()