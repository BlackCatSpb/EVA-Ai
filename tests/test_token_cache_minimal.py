import os
import sys
import logging
import torch
from cogniflex.memory.hybrid_token_cache import HybridTokenCache

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MockBrain:
    def __init__(self):
        self.config = {
            'hybrid_cache': {
                'target_memory_gb': 2.0,
                'dynamic_memory_limit': True,
                'vram_threshold': 0.2,
                'ram_threshold': 0.15,
                'disk_cache_dir': './test_cache'
            }
        }
        self.cache_dir = './test_cache'
        self.resource_queue = None

def test_token_cache():
    print("\n=== Starting Minimal Token Cache Test ===")
    
    # Create mock brain
    brain = MockBrain()
    
    try:
        # Initialize cache with minimal settings
        print("\n1. Initializing HybridTokenCache...")
        cache = HybridTokenCache(
            brain=brain,
            max_memory_tokens=1000,
            disk_cache_dir='hybrid_cache',
            target_memory_gb=2.0,
            vram_threshold=0.2,
            ram_threshold=0.15,
            disk_write_mb_s=10.0,
            disk_read_mb_s=20.0,
            disk_burst_factor=1.5
        )
        print("✅ HybridTokenCache initialized successfully!")
        
        # Test basic operations
        print("\n2. Testing cache operations...")
        test_key = "test_key_123"
        test_value = torch.randn(128, 128)  # 128x128 tensor
        
        # Set value
        print(f"Setting value for key: {test_key}")
        cache.set(test_key, test_value)
        print("✅ Value set in cache")
        
        # Get value
        print(f"\nGetting value for key: {test_key}")
        result = cache.get(test_key)
        if result is not None:
            print(f"✅ Value retrieved successfully. Shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
        else:
            print("❌ Failed to retrieve value from cache")
        
        # Get stats
        print("\n3. Cache statistics:")
        stats = cache.get_cache_stats()
        for k, v in stats.items():
            print(f"{k}: {v}")
        
        # Cleanup
        print("\n4. Cleaning up...")
        cache.cleanup()
        print("✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        print("\n=== Test completed ===\n")

if __name__ == "__main__":
    success = test_token_cache()
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
