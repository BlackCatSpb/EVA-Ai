import torch
from safetensors.torch import save_file, load_file
import os

def test_safetensors():
    print("=== SafeTensors Basic Test ===\n")
    
    try:
        # Create a test tensor
        test_tensor = torch.randn(3, 3)
        print("Created test tensor:")
        print(test_tensor)
        
        # Save to safetensors
        save_file({"weight": test_tensor}, "test_safetensors.safetensors")
        print("\nSaved tensor to test_safetensors.safetensors")
        print(f"File exists: {os.path.exists('test_safetensors.safetensors')}")
        print(f"File size: {os.path.getsize('test_safetensors.safetensors')} bytes")
        
        # Load back
        loaded = load_file("test_safetensors.safetensors")
        print("\nLoaded tensor:")
        print(loaded["weight"])
        
        # Verify
        if torch.allclose(test_tensor, loaded["weight"]):
            print("\nTest passed: Original and loaded tensors match!")
        else:
            print("\nTest failed: Tensors don't match!")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_safetensors()
