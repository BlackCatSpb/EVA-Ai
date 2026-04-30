"""
Clean import from ConceptNet to FractalGraphV2.
Properly uses fg.nodes (not fg.storage.nodes).
Curator will organize the graph after import.
"""
import sys
import os
import logging
import sqlite3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("import_conceptnet")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eva_ai.knowledge.conceptnet_integration import ConceptNetConnector
from eva_ai.memory.fractal_graph_v2 import FractalMemoryGraph

def import_conceptnet_clean(
    conceptnet_db_path: str = "conceptnet.db",
    graph_db_path: str = "eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db",
    max_concepts: int = 500000
):
    """
    Import ConceptNet concepts into FractalGraphV2.
    Uses direct SQL access to ConceptNet for reliability.
    Curator will handle graph organization after import.
    """
    logger.info("=== ConceptNet Clean Import ===")
    logger.info(f"ConceptNet DB: {conceptnet_db_path}")
    logger.info(f"FractalGraph DB: {graph_db_path}")
    logger.info(f"Max concepts: {max_concepts}")
    
    # Check files exist
    if not os.path.exists(conceptnet_db_path):
        logger.error(f"ConceptNet DB not found: {conceptnet_db_path}")
        return False
    
    # Connect to ConceptNet SQLite directly
    logger.info("Connecting to ConceptNet DB...")
    cn_conn = sqlite3.connect(conceptnet_db_path)
    cn_conn.row_factory = sqlite3.Row
    cn_cursor = cn_conn.cursor()
    
    # Check if ConceptNet has data
    try:
        cn_cursor.execute("SELECT COUNT(*) FROM concept")
        concept_count = cn_cursor.fetchone()[0]
        logger.info(f"ConceptNet concepts available: {concept_count}")
    except Exception as e:
        logger.warning(f"Could not count concepts: {e}")
    
    # Load FractalGraphV2
    logger.info("Loading FractalMemoryGraph...")
    try:
        storage_dir = os.path.dirname(graph_db_path)
        fg = FractalMemoryGraph(storage_dir=storage_dir)
        logger.info(f"Graph loaded: {len(fg.storage.nodes)} nodes, {len(fg.storage.edges)} edges")
    except Exception as e:
        logger.error(f"Failed to load graph: {e}")
        cn_conn.close()
        return False
    
    # Import concepts using direct SQL
    logger.info("Starting import using direct SQL...")
    
    # Get concept URIs (English concepts from ConceptNet)
    try:
        cn_cursor.execute("""
            SELECT c.id, l.text, c.sense_label
            FROM concept c
            JOIN label l ON c.label_id = l.id
            WHERE l.language_id = (SELECT id FROM language WHERE name = 'en')
            LIMIT ?
        """, (max_concepts,))
        
        rows = cn_cursor.fetchall()
        logger.info(f"Retrieved {len(rows)} concept URIs from ConceptNet")
        
    except Exception as e:
        logger.error(f"SQL query failed: {e}")
        cn_conn.close()
        return False
    
    imported = 0
    skipped = 0
    
    for idx, row in enumerate(rows):
        concept_id = row['id']
        concept_label = row['text'] or ''
        
        if not concept_label or len(concept_label) < 2:
            skipped += 1
            continue
        
        # Check if concept already exists in graph (using fg.storage.nodes)
        existing = [nid for nid, node in fg.storage.nodes.items() 
                       if hasattr(node, 'content') and node.content == concept_label]
        
        if existing:
            skipped += 1
            if idx < 3:
                logger.info(f"Skipping existing: {concept_label}")
            continue
        
        # Debug first few
        if idx < 3:
            logger.info(f"Adding: {concept_label}")
        
        # Add concept node
        try:
            node = fg.add_node(
                content=concept_label,
                node_type='concept',
                level=2,
                metadata={
                    'source': 'conceptnet',
                    'concept_id': concept_id,
                    'imported_from': 'conceptnet.db'
                }
            )
            
            if node:
                imported += 1
                if imported % 500 == 0:
                    logger.info(f"Imported {imported} concepts so far...")
            else:
                skipped += 1
                
        except Exception as e:
            logger.debug(f"Error adding node '{concept_label}': {e}")
            skipped += 1
    
    # Save graph
    logger.info(f"Import complete. Imported: {imported}, Skipped: {skipped}")
    logger.info(f"Saving graph with {len(fg.storage.nodes)} nodes...")
    
    try:
        fg.save()
        logger.info(f"Graph saved successfully")
        
        # Curator will organize the graph after import
        logger.info("Graph ready for curator to organize")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save graph: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
        
    finally:
        cn_conn.close()

if __name__ == '__main__':
    import_conceptnet_clean()
