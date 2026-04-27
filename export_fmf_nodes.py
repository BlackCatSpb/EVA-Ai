import sqlite3
import numpy as np
import struct

# Export nodes from FMF SQLite to numpy format
db_path = r'C:\Users\black\OneDrive\Desktop\FMF_EVA\eva_ai\memory\fractal_graph_v2\fractal_graph_v2_data\fractal_graph.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all concept nodes with embeddings
cursor.execute('SELECT id, content, embedding FROM nodes WHERE node_type=? LIMIT 200', ('concept',))
rows = cursor.fetchall()

nodes_list = []
metadata_list = []

for r in rows:
    if r[2]:  # has embedding
        try:
            emb = np.array(struct.unpack('f' * (len(r[2]) // 4), r[2]), dtype=np.float32)
            nodes_list.append(emb)
            metadata_list.append({
                'id': r[0],
                'content': r[1]
            })
        except Exception as e:
            print(f"Error parsing {r[0]}: {e}")

conn.close()

if nodes_list:
    nodes = np.vstack(nodes_list)
    output_dir = r'C:\Users\black\OneDrive\Desktop\EVA-Ai\eva_ai\memory\fractal_graph_v2\fractal_graph_v2_data'
    np.save(f'{output_dir}/concept_nodes.npy', nodes)
    print(f'Exported {len(nodes_list)} concept nodes with shape {nodes.shape}')

    # Also save metadata as JSON
    import json
    with open(f'{output_dir}/concept_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False)
    print(f'Saved metadata for {len(metadata_list)} nodes')
else:
    print('No concept nodes with embeddings found')