"""
Import ConceptNet knowledge (11 GB) into FractalGraphV2.
Run: python import_conceptnet_to_graph.py
"""
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("import_conceptnet")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eva_ai.knowledge.conceptnet_integration import ConceptNetConnector
# Use the ONLY correct FractalGraphV2 with storage attribute
from eva_ai.memory.fractal_graph_v2.storage import FractalGraphV2

# Also import needed types
from eva_ai.memory.fractal_graph_v2.types import FractalNode, NodeType

def import_conceptnet_to_graph(
    conceptnet_db_path: str = "conceptnet.db",
    graph_db_path: str = "eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db",
    max_concepts: int = 5000
):
    """
    Import ConceptNet concepts into FractalGraphV2.
    
    Args:
        conceptnet_db_path: Path to conceptnet.db (11 GB)
        graph_db_path: Path to FractalGraphV2 database
        max_concepts: Maximum number of concepts to import (safety limit)
    """
    logger.info("=== ConceptNet to FractalGraphV2 Import ===")
    logger.info(f"ConceptNet DB: {conceptnet_db_path}")
    logger.info(f"FractalGraph DB: {graph_db_path}")
    logger.info(f"Max concepts: {max_concepts}")
    
    # Check files exist
    if not os.path.exists(conceptnet_db_path):
        logger.error(f"ConceptNet DB not found: {conceptnet_db_path}")
        return False
    
    if not os.path.exists(graph_db_path):
        logger.warning(f"Graph DB not found, will create: {graph_db_path}")
    
    # Connect to ConceptNet
    logger.info("Connecting to ConceptNet...")
    cn = ConceptNetConnector(db_path=conceptnet_db_path)
    if not cn.connect():
        logger.error("Failed to connect to ConceptNet!")
        return False
    
    logger.info("Connected to ConceptNet successfully")
    
    # Load FractalGraphV2 - правильный метод загрузки
    logger.info("Loading FractalGraphV2...")
    try:
        # memory.fractal_graph_v2.FractalGraphV2 имеет метод load()
        from eva_ai.memory.fractal_graph_v2 import FractalGraphV2 as FGClass
        fg = FGClass.load(graph_db_path)
        logger.info(f"Graph loaded: {len(fg.storage.nodes) if hasattr(fg, 'storage') else 0} nodes")
    except Exception as e:
        logger.warning(f"Could not load graph: {e}")
        logger.info("Creating new FractalGraphV2...")
        fg = FractalGraphV2()
        logger.info(f"New graph created")
    
    # Starting concepts for import (core concepts)
    seed_concepts = [
        # Russian concepts
        "человек", "жизнь", "время", "пространство", "материя", "энергия",
        "информация", "знание", "истина", "разум", "сознание",
        "искусственный интеллект", "машина", "робот", "программа",
        "наука", "технология", "физика", "химия", "биология",
        "математика", "алгоритм", "данные", "компьютер", "интернет",
        # English concepts (ConceptNet uses English as primary)
        "person", "life", "time", "space", "matter", "energy",
        "information", "knowledge", "truth", "mind", "consciousness",
        "artificial intelligence", "machine", "robot", "program",
        "science", "technology", "physics", "chemistry", "biology",
        "mathematics", "algorithm", "data", "computer", "internet",
        # Additional concepts
        "language", "communication", "learning", "memory", "thought",
        "universe", "planet", "earth", "sun", "star",
        "atom", "molecule", "cell", "organism", "evolution",
        "язык", "общение", "обучение", "память", "мышление",
        "вселенная", "планета", "земля", "солнце", "звезда",
        "атом", "молекула", "клетка", "организм", "эволюция"
    ]
    
    imported = 0
    skipped = 0
    
    logger.info(f"Starting import of {len(seed_concepts)} seed concepts...")
    
    for concept in seed_concepts:
        if imported >= max_concepts:
            logger.info(f"Reached max_concepts limit: {max_concepts}")
            break
        
        try:
            # Get concept info from ConceptNet
            lang = 'ru' if any('\u0400' <= c <= '\u04FF' for c in concept) else 'en'
            info = cn.get_concept_info(concept, lang=lang)
            
            if not info or not info.get('edges'):
                logger.debug(f"No ConceptNet data for '{concept}'")
                skipped += 1
                continue
            
            # Check if concept already exists in graph
            existing = [nid for nid, node in fg.nodes.items() 
                       if hasattr(node, 'content') and node.content == concept]
            
            if existing:
                logger.debug(f"Concept '{concept}' already in graph")
                skipped += 1
                continue
            
            # Add concept node
            node_id = fg.add_node(
                content=concept,
                node_type='concept',
                level=2,
                metadata={
                    'source': 'conceptnet',
                    'edges_count': len(info.get('edges', [])),
                    'relations': dict(info.get('relations', {})),
                    'imported_from': 'conceptnet.db'
                }
            )
            
            if node_id:
                imported += 1
                if imported % 100 == 0:
                    logger.info(f"Imported {imported} concepts so far...")
                
                # Add related concepts (limited depth)
                related = cn.find_related_concepts(concept, limit=5)
                for rel in related[:3]:  # Only first 3 to avoid explosion
                    rel_concept = rel.get('concept', '')
                    if rel_concept and len(rel_concept) > 2:
                        # Check if related concept exists
                        rel_existing = [nid for nid, node in fg.nodes.items() 
                                       if hasattr(node, 'content') and node.content == rel_concept]
                        
                        if not rel_existing and imported < max_concepts:
                            fg.add_node(
                                content=rel_concept,
                                node_type='concept',
                                level=1,
                                metadata={
                                    'source': 'conceptnet',
                                    'relation_to': concept,
                                    'relation_type': rel.get('relation', 'unknown')
                                }
                            )
                            imported += 1
            else:
                skipped += 1
                
        except Exception as e:
            logger.error(f"Error importing concept '{concept}': {e}")
            skipped += 1
            continue
    
    # Save the graph
    logger.info(f"Import complete. Imported: {imported}, Skipped: {skipped}")
    logger.info(f"Saving graph with {fg.node_count} nodes...")
    
    try:
        fg.save(graph_db_path)
        logger.info(f"Graph saved successfully to {graph_db_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save graph: {e}")
        return False

if __name__ == '__main__':
    import_conceptnet_to_graph(max_concepts=3000)
