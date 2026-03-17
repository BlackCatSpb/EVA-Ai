import sys
import torch

def test_environment():
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    
    # Test basic tensor operations
    x = torch.rand(2, 2)
    print("\nTest tensor:", x)
    print("Test multiplication:", x @ x.t())

if __name__ == "__main__":
    test_environment()
