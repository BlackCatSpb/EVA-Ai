from eva_ai.fcp_core.fractal_graph import FractalGraphV2

fg = FractalGraphV2()
print(f"Graph node_count: {fg.node_count}")
print(f"Graph has storage: {hasattr(fg, 'storage')}")
if hasattr(fg, 'storage') and hasattr(fg.storage, 'nodes'):
    print(f"Storage nodes: {len(fg.storage.nodes)}")
if hasattr(fg, 'nodes'):
    print(f"Direct nodes: {len(fg.nodes)}")