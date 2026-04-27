"""
Full Knowledge Integration Script
Loads all knowledge sources and integrates into EVA's FractalGraph
"""
import os
import sys

sys.path.insert(0, '.')

from eva_ai.memory.fractal_graph_v2 import FractalMemoryGraph
from eva_ai.knowledge import KnowledgeIntegrator

def integrate_all_knowledge():
    """Load all knowledge sources and integrate into graph"""
    print("=" * 60)
    print("EVA AI - Full Knowledge Integration")
    print("=" * 60)
    
    # Initialize graph
    print("\n1. Loading FractalGraph...")
    graph = FractalMemoryGraph()
    print(f"   Current nodes: {len(graph.storage.nodes)}")
    
    class MockBrain:
        def __init__(self):
            self.fractal_graph_v2 = graph
    
    brain = MockBrain()
    
    # Create integrator
    print("\n2. Creating KnowledgeIntegrator...")
    integrator = KnowledgeIntegrator()
    
    # Initialize (loads ConceptNet, Wikidata, NEREL)
    print("\n3. Initializing knowledge sources...")
    results = integrator.initialize()
    for source, success in results.items():
        print(f"   {source}: {'OK' if success else 'FAILED'}")
    
    # Show statistics
    print("\n4. Knowledge Statistics:")
    stats = integrator.get_statistics()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.2f}" if v > 0 else f"   {k}: 0")
        else:
            print(f"   {k}: {v}")
    
    # Get existing concept nodes
    existing_concepts = []
    for node_id, node in graph.storage.nodes.items():
        if getattr(node, 'node_type', '') == 'concept':
            content = getattr(node, 'content', '')
            if content:
                existing_concepts.append(content)
    
    print(f"\n5. Found {len(existing_concepts)} existing concepts in graph")
    print(f"   Sample: {existing_concepts[:10]}")
    
    # Enrich existing concepts
    print("\n6. Enriching existing concepts with external knowledge...")
    enriched_count = 0
    for node_id, node in list(graph.storage.nodes.items()):
        if getattr(node, 'node_type', '') == 'concept':
            content = getattr(node, 'content', '')
            if content and len(content) > 2:
                ctx = integrator.get_concept_context(content)
                if ctx['wikidata_triplets'] or ctx['conceptnet_relations']:
                    enriched_count += 1
                    if hasattr(node, 'metadata') and isinstance(node.metadata, dict):
                        node.metadata['synonyms'] = ctx.get('synonyms', [])[:5]
                        node.metadata['hypernyms'] = ctx.get('hypernyms', [])[:5]
                        node.metadata['related'] = ctx.get('related_concepts', [])[:10]
    
    print(f"   Enriched {enriched_count} nodes")
    
    # Add key concepts from knowledge sources
    print("\n7. Adding key concepts from knowledge sources...")
    
    # Get key entities from Wikidata
    key_concepts = set()
    for triplet in integrator.wikidata.triplets:
        key_concepts.add(triplet.get('subject', ''))
        key_concepts.add(triplet.get('object', ''))
    
    # Get entities from NEREL
    for doc in integrator.nerel.documents:
        for entity in doc.get('entities', []):
            key_concepts.add(entity.get('text', ''))
    
    # Filter and add to graph
    added_count = 0
    for concept in list(key_concepts)[:50]:  # Limit to 50
        if concept and len(concept) > 2:
            # Check if already exists
            exists = any(concept in getattr(graph.storage.nodes[nid], 'content', '') 
                        for nid in graph.storage.nodes)
            if not exists:
                try:
                    graph.add_node(
                        content=concept,
                        node_type='concept',
                        level=2,
                        metadata={
                            'source': 'knowledge_integrator',
                            'wikidata_triplets': len(integrator.wikidata.get_outgoing_triplets(concept)),
                            'conceptnet_relations': len(integrator.conceptnet.get_concept_info(concept).get('edges', [])) if integrator._conceptnet_available else 0,
                        }
                    )
                    added_count += 1
                except Exception as e:
                    pass
    
    print(f"   Added {added_count} new concepts to graph")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(f"Total nodes: {len(graph.storage.nodes)}")
    
    type_counts = {}
    for node in graph.storage.nodes.values():
        nt = getattr(node, 'node_type', 'unknown')
        type_counts[nt] = type_counts.get(nt, 0) + 1
    print("Nodes by type:")
    for nt, count in sorted(type_counts.items()):
        print(f"   {nt}: {count}")
    
    # Save updated graph
    print("\n8. Graph updated successfully!")
    
    return graph

if __name__ == '__main__':
    print("Starting knowledge integration...\n")
    graph = integrate_all_knowledge()
    print("\nKnowledge integration complete!")