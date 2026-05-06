import sqlite3
import struct

# Check what data is in FMF graph
db_path = r'C:\Users\black\OneDrive\Desktop\FMF_EVA\eva_ai\memory\fractal_graph_v2\fractal_graph_v2_data\fractal_graph.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all tables info
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [t[0] for t in cursor.fetchall()]
print(f"Tables: {tables}")

for table in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"  {table}: {count} rows")

# Check nodes table structure
print("\nNodes table schema:")
cursor.execute("PRAGMA table_info(nodes)")
columns = cursor.fetchall()
for col in columns:
    print(f"  {col[1]}: {col[2]}")

# Check first few nodes
print("\nFirst 5 nodes (all columns):")
cursor.execute("SELECT * FROM nodes LIMIT 5")
rows = cursor.fetchall()
for r in rows:
    # Print id, type, content, and a sample of what each field looks like
    print(f"\nNode:")
    for i, col in enumerate(columns):
        val = r[i]
        col_name = col[1]
        if col_name == 'embedding' and val:
            print(f"  {col_name}: {len(val)} bytes")
        elif col_name == 'metadata' and val:
            print(f"  {col_name}: {str(val)[:100]}")
        else:
            print(f"  {col_name}: {str(val)[:80]}")

conn.close()