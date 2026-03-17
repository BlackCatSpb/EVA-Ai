import torch
from transformers import GPT2LMHeadModel, GPT2Config

def test_model_loading():
    print("=== Direct Model Loading Test ===\n")
    
    # Create a small test model
    config = GPT2Config(
        vocab_size=50264,
        n_embd=64,  # Smaller size for testing
        n_layer=2,   # Fewer layers for testing
        n_head=2,
        n_ctx=128
    )
    
    # Create and save a small test model
    print("Creating test model...")
    model = GPT2LMHeadModel(config)
    torch.save(model.state_dict(), "test_model.pt")
    print("Test model created and saved successfully")
    
    # Try loading it back
    print("\nLoading test model...")
    loaded_state_dict = torch.load("test_model.pt")
    new_model = GPT2LMHeadModel(config)
    new_model.load_state_dict(loaded_state_dict)
    print("Test model loaded successfully")
    
    # Test forward pass
    print("\nTesting forward pass...")
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    outputs = new_model(input_ids)
    print(f"Output shape: {outputs.logits.shape}")
    print("Test completed successfully!")

if __name__ == "__main__":
    test_model_loading()
