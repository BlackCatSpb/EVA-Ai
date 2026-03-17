import torch
from safetensors.torch import load_file

def test_model_loading():
    print("Testing model loading...")
    
    # Path to the model
    model_path = "out/fractal_rugpt_full.safetensors"
    
    try:
        # Try loading the model
        print(f"Loading model from {model_path}...")
        state_dict = load_file(model_path, device="cpu")
        
        # Print some basic info
        print("\nModel loaded successfully!")
        print(f"Number of parameters: {len(state_dict)}")
        print("\nParameter shapes:")
        for i, (name, param) in enumerate(state_dict.items()):
            print(f"{name}: {tuple(param.shape)}")
            if i >= 4:  # Print first 5 parameters
                print("...")
                break
                
        # Test a simple operation
        print("\nTesting tensor operations...")
        test_tensor = next(iter(state_dict.values()))
        print(f"Tensor shape: {test_tensor.shape}")
        print(f"Tensor mean: {test_tensor.float().mean().item():.4f}")
        print(f"Tensor std: {test_tensor.float().std().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_loading()
