import torch
from safetensors.torch import load_file
from transformers import GPT2LMHeadModel, GPT2Config

def test_forward_pass():
    print("Testing forward pass...")
    
    try:
        # 1. Load model config
        print("Loading model config...")
        config = GPT2Config(
            vocab_size=50264,
            n_positions=2048,
            n_ctx=2048,
            n_embd=768,
            n_layer=12,
            n_head=12,
            n_inner=3072,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            summary_type="cls_index",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=0.1,
            use_cache=True,
            bos_token_id=0,
            eos_token_id=2,
            return_dict=True
        )
        
        # 2. Initialize model
        print("Initializing model...")
        model = GPT2LMHeadModel(config)
        
        # 3. Load weights
        print("Loading weights...")
        state_dict = load_file("out/fractal_rugpt_full.safetensors")
        
        # Convert to float32 for better compatibility
        state_dict = {k: v.float() for k, v in state_dict.items()}
        
        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        
        # 4. Test forward pass
        print("Testing forward pass...")
        input_ids = torch.tensor([[0, 1, 2, 3, 4]])
        with torch.no_grad():
            outputs = model(input_ids)
            
        print("Forward pass successful!")
        print(f"Output shape: {outputs.logits.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_forward_pass()
