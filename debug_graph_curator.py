import traceback

from eva_ai.knowledge.graph_curator import GraphCurator

# Monkey patch to add debug
original_do_curation = GraphCurator._do_curation

def patched_do_curation(self):
    try:
        return original_do_curation(self)
    except AttributeError as e:
        traceback.print_exc()
        raise

GraphCurator._do_curation = patched_do_curation

print("Patched GraphCurator._do_curation")