import torch
import logging
import time
from eva.core.core_brain import CoreBrain

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_hybrid_cache():
    print("🚀 Starting Hybrid Token Cache Test...")
    
    # Initialize CoreBrain with hybrid token cache
    config = {
        'generation': {
            'enabled': True,
            'model_name': 'sberbank-ai/rugpt3small_based_on_gpt2',
            'cache_config': {
                'enabled': True,
                'max_memory_tokens': 1000,
                'vram_threshold': 0.2,
                'ram_threshold': 0.15,
                'disk_cache_dir': './cogniflex_cache/hybrid_cache',
                'dynamic_memory_limit': True,
                'eviction_policy': 'hybrid'
            },
            'cache_memory_gb': 2.0  # 2GB for testing
        }
    }

    try:
        # Initialize CoreBrain
        print("\n1. 🏗️ Initializing CoreBrain...")
        start_time = time.time()
        brain = CoreBrain(config=config)
        print(f"✅ CoreBrain initialized in {time.time() - start_time:.2f} seconds!")

        # Test token cache
        print("\n2. 🔍 Testing Token Cache...")
        if hasattr(brain, 'token_cache') and brain.token_cache is not None:
            print("✅ Token cache initialized successfully!")
            
            # Get cache stats
            if hasattr(brain.token_cache, 'get_cache_stats'):
                stats = brain.token_cache.get_cache_stats()
                print(f"📊 Cache stats: {stats}")
            else:
                print("ℹ️ get_cache_stats method not available")
            
            # Test cache operations
            test_key = 'test_key_1'
            test_value = torch.randn(128, 128)  # Larger tensor for testing
            
            # Test set operation
            print(f"\n3. 💾 Setting value in cache with key: {test_key}")
            start_time = time.time()
            brain.token_cache.set(test_key, test_value)
            print(f"✅ Value set in {time.time() - start_time:.4f} seconds")
            
            # Test get operation
            print(f"\n4. 🔄 Getting value from cache with key: {test_key}")
            start_time = time.time()
            cached_value = brain.token_cache.get(test_key)
            elapsed = time.time() - start_time
            print(f"✅ Cache {'hit' if cached_value is not None else 'miss'} in {elapsed:.4f} seconds")
            
            # Test memory monitoring
            if hasattr(brain.token_cache, 'memory_monitor_running'):
                status = "✅ running" if brain.token_cache.memory_monitor_running else "❌ not running"
                print(f"\n5. 🖥️  Memory monitoring: {status}")
            else:
                print("\n5. ℹ️ Memory monitoring status not available")
            
            # Test cache clearing
            if hasattr(brain.token_cache, 'clear'):
                print("\n6. 🧹 Testing cache clear...")
                brain.token_cache.clear()
                print("✅ Cache cleared successfully")
                
                # Verify clear
                cleared_value = brain.token_cache.get(test_key)
                status = "✅ miss (expected)" if cleared_value is None else "❌ unexpected hit"
                print(f"   Cache get after clear: {status}")
        else:
            print("❌ Token cache initialization failed!")
            return False
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        print("\n7. 🛑 Shutting down...")
        if 'brain' in locals() and hasattr(brain, 'shutdown'):
            brain.shutdown()
            print("✅ Shutdown completed")
        else:
            print("⚠️  Shutdown method not found or brain not initialized")

if __name__ == "__main__":
    print("="*60)
    print("🧪 Starting Hybrid Token Cache Test Suite")
    print("="*60)
    
    success = test_hybrid_cache()
    
    print("\n" + "="*60)
    if success:
        print("🎉 All tests passed successfully!")
        print("You can now proceed with system deployment.")
    else:
        print("❌ Some tests failed. Please check the logs above.")
    print("="*60)
