from eva_ai.memory.fractal_graph_v2 import FractalMemoryGraph

graph = FractalMemoryGraph()

print(f"FractalMemoryGraph loaded:")
print(f"  nodes: {len(graph.storage.nodes)}")
print(f"  edges: {len(graph.storage.edges)}")

# Sample some nodes
print("\nSample nodes:")
for i, (node_id, node) in enumerate(list(graph.storage.nodes.items())[:10]):
    emb_size = len(node.embedding) if node.embedding else 0
    print(f"  {node_id[:40]} | {node.node_type} | emb={emb_size} | {str(node.content)[:40]}")