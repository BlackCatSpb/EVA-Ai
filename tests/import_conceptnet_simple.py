"""
Simple import from ConceptNet to FractalGraphV2.
Uses direct SQL access to ConceptNet DB for reliability.
"""
import sys
import os
import logging
import sqlite3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("import_conceptnet")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eva_ai.memory.fractal_graph_v2.storage import FractalGraphV2

def import_from_conceptnet(
    conceptnet_db: str = "conceptnet.db",
    graph_db_path: str = "eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db",
    max_nodes: int = 5000
):
    """
    Import concepts from ConceptNet SQLite DB directly.
    """
    logger.info("=== ConceptNet Simple Import ===")
    
    # Connect to ConceptNet SQLite directly
    if not os.path.exists(conceptnet_db):
        logger.error(f"ConceptNet DB not found: {conceptnet_db}")
        return False
    
    logger.info(f"Connecting to ConceptNet DB: {conceptnet_db}")
    cn_conn = sqlite3.connect(conceptnet_db)
    cn_cursor = cn_conn.cursor()
    
    # Get some concepts directly via SQL
    logger.info("Fetching concepts from ConceptNet via SQL...")
    try:
        # Query edges table for concepts (subjects)
        cn_cursor.execute("""
            SELECT DISTINCT subject FROM edges 
            WHERE subject LIKE '/c/en/%' 
            LIMIT ?
        """, (max_nodes,))
        
        rows = cn_cursor.fetchall()
        logger.info(f"Found {len(rows)} concepts in ConceptNet")
        
    except Exception as e:
        logger.error(f"SQL query failed: {e}")
        cn_conn.close()
        return False
    
    # Load FractalGraphV2
    logger.info("Loading FractalGraphV2...")
    try:
        storage_dir = os.path.dirname(graph_db_path)
        fg = FractalGraphV2(storage_dir=storage_dir)
        logger.info(f"Graph loaded: {len(fg.nodes)} nodes, {len(fg.edges)} edges")
    except Exception as e:
        logger.error(f"Failed to load graph: {e}")
        cn_conn.close()
        return False
    
    # Import concepts
    imported = 0
    skipped = 0
    
    for row in rows:
        concept_uri = row[0]  # e.g., /c/en/person
        # Extract concept name from URI
        parts = concept_uri.split('/')
        if len(parts) < 4:
            skipped += 1
            continue
        concept_name = parts[3].replace('_', ' ')
        
        # Check if already exists
        existing = [nid for nid, node in fg.nodes.items() 
                    if hasattr(node, 'content') and node.content == concept_name]
        
        if existing:
            skipped += 1
            continue
        
        # Add node to graph
        try:
            node_id = fg.add_node(
                content=concept_name,
                node_type='concept',
                level=2,
                metadata={
                    'source': 'conceptnet',
                    'uri': concept_uri,
                    'imported_from': 'conceptnet.db'
                }
            )
            
            if node_id:
                imported += 1
                if imported % 100 == 0:
                    logger.info(f"Imported {imported} concepts so far...")
            else:
                skipped += 1
        except Exception as e:
            logger.debug(f"Error adding node '{concept_name}': {e}")
            skipped += 1
    
    # Save graph
    logger.info(f"Import complete. Imported: {imported}, Skipped: {skipped}")
    logger.info(f"Saving graph with {len(fg.nodes)} nodes...")
    
    try:
        fg._save_data()  # Save nodes to DB
        logger.info(f"Graph saved successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to save graph: {e}")
        return False
    finally:
        cn_conn.close()

if __name__ == '__main__':
    import_from_conceptnet(max_nodes=3000)
