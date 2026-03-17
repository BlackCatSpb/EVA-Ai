import torch
from safetensors.torch import load_file
from transformers import GPT2LMHeadModel, GPT2Config

def minimal_test():
    print("Starting minimal test...")
    
    # Basic PyTorch test
    x = torch.rand(2, 2)
    print(f"PyTorch test - Random tensor: {x}")
    
    # Test safetensors loading
    try:
        state_dict = load_file("out/fractal_rugpt_full.safetensors")
        print(f"Successfully loaded safetensors file with {len(state_dict)} parameters")
        
        # Test model initialization
        config = GPT2Config(
            vocab_size=50264,
            n_ctx=2048,
            n_embd=768,
            n_layer=12,
            n_head=12,
            n_inner=3072
        )
        model = GPT2LMHeadModel(config)
        print("Successfully initialized model")
        
        # Test model loading
        model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded weights into model")
        
    except Exception as e:
        print(f"Error in minimal test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    minimal_test()
