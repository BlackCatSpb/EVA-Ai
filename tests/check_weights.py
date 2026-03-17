import torch
from safetensors.torch import load_file

def check_weights():
    print("=== Model Weights Check ===\n")
    
    try:
        # Try to load the safetensors file
        print("Loading safetensors file...")
        state_dict = load_file("out/fractal_rugpt_full.safetensors")
        print("Successfully loaded safetensors file")
        
        # Print basic info
        print(f"Number of parameters: {len(state_dict)}")
        
        # Print first 10 keys
        print("\nFirst 10 parameter keys:")
        for i, key in enumerate(state_dict.keys()):
            if i >= 10:
                break
            print(f"{i+1}. {key}")
            
        # Check tensor shapes and values
        print("\nChecking tensor shapes and values...")
        for name, tensor in list(state_dict.items())[:5]:  # Check first 5 tensors
            print(f"\n{name}:")
            print(f"  Shape: {tuple(tensor.shape)}")
            print(f"  Dtype: {tensor.dtype}")
            print(f"  Min: {tensor.min().item():.4f}")
            print(f"  Max: {tensor.max().item():.4f}")
            print(f"  Mean: {tensor.float().mean().item():.4f}")
            print(f"  Std: {tensor.float().std().item():.4f}")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_weights()
