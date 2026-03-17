import torch
from safetensors.torch import load_file
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import numpy as np

def check_model_weights():
    print("=== Model Weights Diagnostics ===\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # 1. Load tokenizer
        print("\n1. Loading tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        
        # 2. Load fractal model weights
        print("\n2. Loading fractal model weights...")
        state_dict = load_file("out/fractal_rugpt_full.safetensors", device="cpu")
        print(f"Loaded state dict with {len(state_dict)} parameters")
        
        # 3. Create model with correct config
        print("\n3. Creating model with config...")
        config = GPT2Config(
            vocab_size=50264,  # For RuGPT3Small
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
            summary_first_dropout=0.1,
            use_cache=True,
            bos_token_id=0,
            eos_token_id=2,
            return_dict=True
        )
        
        model = GPT2LMHeadModel(config).to(device)
        
        # 4. Check parameter shapes
        print("\n4. Checking parameter shapes...")
        for name, param in model.named_parameters():
            if name in state_dict:
                if param.shape != state_dict[name].shape:
                    print(f"Shape mismatch for {name}: "
                          f"expected {param.shape}, got {state_dict[name].shape}")
        
        # 5. Load weights with strict=False
        print("\n5. Loading weights into model...")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        print(f"\nMissing keys: {len(missing)}")
        if missing:
            print("First 5 missing keys:", missing[:5])
            
        print(f"\nUnexpected keys: {len(unexpected)}")
        if unexpected:
            print("First 5 unexpected keys:", unexpected[:5])
        
        # 6. Check weight statistics
        print("\n6. Weight statistics:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: shape={tuple(param.shape)}, "
                      f"mean={param.data.mean().item():.6f}, "
                      f"std={param.data.std().item():.6f}, "
                      f"min={param.data.min().item():.6f}, "
                      f"max={param.data.max().item():.6f}")
        
        # 7. Test forward pass with simple input
        print("\n7. Testing forward pass...")
        input_ids = torch.tensor([[0, 1, 2, 3, 4]]).to(device)  # Simple input
        with torch.no_grad():
            outputs = model(input_ids)
            print(f"Output logits shape: {outputs.logits.shape}")
            print(f"First 5 logits: {outputs.logits[0, -1, :5].tolist()}")
        
        # 8. Check for NaN/Inf values
        print("\n8. Checking for NaN/Inf values...")
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"Found NaN in {name}")
            if torch.isinf(param).any():
                print(f"Found Inf in {name}")
        
        print("\nDiagnostics completed!")
        
    except Exception as e:
        print(f"\nError in diagnostics: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    check_model_weights()
