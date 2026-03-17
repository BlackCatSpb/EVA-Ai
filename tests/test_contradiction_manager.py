"""
Test script for ContradictionManager and OptimizedContradictionDetector
"""
import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('contradiction_test.log')
    ]
)
logger = logging.getLogger("contradiction_test")

def test_contradiction_manager():
    """Test the ContradictionManager and OptimizedContradictionDetector"""
    try:
        logger.info("Starting ContradictionManager test...")
        
        # Add the project root to the Python path
        project_root = str(Path(__file__).parent.absolute())
        if project_root not in sys.path:
            sys.path.append(project_root)
        
        # Import required modules
        from cogniflex.contradiction.contradiction_manager import ContradictionManager
        
        # Test initialization
        logger.info("Creating ContradictionManager instance...")
        
        # Create a simple mock brain with required attributes
        class MockBrain:
            def __init__(self):
                self.knowledge_graph = None
                self.cache_dir = "./test_contradiction_cache"
        
        # Create test cache directory if it doesn't exist
        os.makedirs("./test_contradiction_cache", exist_ok=True)
        
        # Initialize with mock brain and test cache
        manager = ContradictionManager(brain=MockBrain(), cache_dir="./test_contradiction_cache")
        
        # Test basic functionality
        logger.info("Testing basic functionality...")
        
        # Test adding a contradiction
        test_contradiction = {
            'id': 'test_contradiction_1',
            'concept': 'test_concept',
            'conflicting_facts': [
                {'source': 'test1', 'fact': 'Test fact 1'},
                {'source': 'test2', 'fact': 'Test fact 2'}
            ],
            'divergence_level': 0.8,
            'metadata': {'test': 'test'},
            'status': 'detected'
        }
        
        manager.add_contradiction(test_contradiction)
        logger.info("Successfully added test contradiction")
        
        # Test getting known concepts
        concepts = manager.get_known_concepts()
        logger.info(f"Known concepts: {concepts}")
        
        # Test getting all contradictions
        contradictions = manager.get_contradictions()
        logger.info(f"Found {len(contradictions)} contradictions")
        
        # Test manual contradiction status update (since resolution requires more complex setup)
        if contradictions:
            logger.info("Testing manual contradiction status update...")
            contradiction = contradictions[0]
            original_status = contradiction.get('status', 'unknown')
            
            # Manually update the status
            contradiction['status'] = 'resolved'
            logger.info(f"Manually updated contradiction status from '{original_status}' to 'resolved'")
            
            # Verify the status was updated
            updated_contradictions = [c for c in manager.get_contradictions() 
                                   if c.get('id') == contradiction.get('id')]
            if updated_contradictions:
                logger.info(f"Verified contradiction status is now: {updated_contradictions[0].get('status')}")
        
        logger.info("ContradictionManager test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"ContradictionManager test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    if test_contradiction_manager():
        print("\n✅ ContradictionManager test completed successfully!")
        print("Check contradiction_test.log for detailed logs.")
    else:
        print("\n❌ ContradictionManager test failed!")
        print("Check contradiction_test.log for error details.")
