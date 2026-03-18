import os
import sys
import torch
import safetensors

def test_environment():
    print("=== Python Environment Test ===\n")
    
    # Python info
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    
    # Library versions
    print("\n=== Library Versions ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Safetensors version: {safetensors.__version__}")
    
    # CUDA availability
    print("\n=== CUDA Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Simple tensor operation
    print("\n=== Tensor Test ===")
    x = torch.rand(2, 3)
    print(f"Random tensor:\n{x}")
    print(f"Tensor multiplication:\n{x @ x.T}")
    
    print("\nEnvironment test completed successfully!")

if __name__ == "__main__":
    import os
    test_environment()
