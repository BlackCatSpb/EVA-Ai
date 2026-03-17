import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

def minimal_test():
    print("=== Minimal Model Test ===\n")
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # 1. Create a small test model
        print("\nCreating test model...")
        config = GPT2Config(
            vocab_size=50264,
            n_embd=64,  # Smaller size for testing
            n_layer=2,   # Fewer layers for testing
            n_head=2,
            n_ctx=128
        )
        model = GPT2LMHeadModel(config).to(device)
        
        # 2. Create test input
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)  # Simple input
        
        # 3. Test forward pass
        print("\nTesting forward pass...")
        with torch.no_grad():
            outputs = model(input_ids)
            print(f"Output shape: {outputs.logits.shape}")
            print("First 5 logits:", outputs.logits[0, -1, :5].tolist())
        
        # 4. Test generation
        print("\nTesting generation...")
        with torch.no_grad():
            gen = model.generate(
                input_ids,
                max_length=10,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                pad_token_id=0
            )
            print(f"Generated token IDs: {gen[0].tolist()}")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nError in minimal_test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    minimal_test()
