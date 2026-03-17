import torch
from safetensors.torch import load_file
import os

def check_safetensors():
    print("=== SafeTensors File Check ===\n")
    
    file_path = "out/fractal_rugpt_full.safetensors"
    print(f"Checking file: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    print(f"File size: {os.path.getsize(file_path) / (1024 * 1024):.2f} MB")
    
    try:
        # Try to load the file
        print("\nLoading SafeTensors file...")
        state_dict = load_file(file_path)
        
        # Basic info
        print(f"\nNumber of parameters: {len(state_dict)}")
        
        # Print first 10 keys and their shapes
        print("\nFirst 10 parameters:")
        for i, (k, v) in enumerate(state_dict.items()):
            if i >= 10:
                break
            print(f"{k}: {tuple(v.shape)}, {v.dtype}, "
                  f"mean={v.float().mean().item():.4f}, "
                  f"std={v.float().std().item():.4f}")
        
        # Check for NaN/Inf values
        print("\nChecking for NaN/Inf values...")
        has_nan = any(torch.isnan(v).any() for v in state_dict.values())
        has_inf = any(torch.isinf(v).any() for v in state_dict.values())
        print(f"Contains NaN: {has_nan}")
        print(f"Contains Inf: {has_inf}")
        
        # Check parameter statistics
        print("\nParameter statistics:")
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"Total parameters: {total_params:,}")
        
        # Group by layer type
        layer_types = {}
        for k, v in state_dict.items():
            layer = k.split('.')[1] if '.' in k else 'other'
            if layer not in layer_types:
                layer_types[layer] = []
            layer_types[layer].append((k, v))
        
        print("\nParameters by layer type:")
        for layer, params in layer_types.items():
            num_params = sum(p[1].numel() for p in params)
            print(f"{layer}: {len(params)} tensors, {num_params:,} parameters")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_safetensors()
