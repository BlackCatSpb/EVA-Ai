import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from safetensors.torch import load_file
import os

def test_model_generation():
    print("=== Model Generation Test ===\n")
    
    # Configuration
    model_path = "out/fractal_rugpt_full.safetensors"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # 1. Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        print(f"Model file found: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        # 2. Load tokenizer
        print("\nLoading tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded successfully")
        
        # 3. Create model with config
        print("\nCreating model with config...")
        config = {
            "vocab_size": 50264,
            "n_positions": 2048,
            "n_ctx": 2048,
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
            "n_inner": 3072,
            "activation_function": "gelu_new",
            "resid_pdrop": 0.1,
            "embd_pdrop": 0.1,
            "attn_pdrop": 0.1,
            "layer_norm_epsilon": 1e-5,
            "initializer_range": 0.02,
            "bos_token_id": 0,
            "eos_token_id": 2,
        }
        model = GPT2LMHeadModel(GPT2Config(**config)).to(device)
        print("Model created successfully")
        
        # 4. Load weights
        print("\nLoading model weights...")
        state_dict = load_file(model_path, device=device)
        
        # Convert to float32 for compatibility
        state_dict = {k: v.float() for k, v in state_dict.items()}
        
        # Load state dict
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Weights loaded with:")
        print(f"- Missing keys: {len(missing)}")
        print(f"- Unexpected keys: {len(unexpected)}")
        
        if missing:
            print("\nFirst 5 missing keys:")
            for k in list(missing)[:5]:
                print(f"  - {k}")
        
        if unexpected:
            print("\nFirst 5 unexpected keys:")
            for k in list(unexpected)[:5]:
                print(f"  - {k}")
        
        # 5. Test generation
        print("\n=== Testing Generation ===")
        prompts = [
            "Привет, как дела?",
            "Сегодня я хочу рассказать о",
            "Искусственный интеллект - это"
        ]
        
        model.eval()
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            
            # Encode input
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs.input_ids
            
            print(f"Input shape: {input_ids.shape}")
            
            # Generate with different settings
            gen_params = [
                {"do_sample": False, "max_length": 30, "name": "Greedy"},
                {"do_sample": True, "temperature": 0.7, "top_k": 50, "max_length": 30, "name": "Sampling (temp=0.7)"},
                {"do_sample": True, "temperature": 0.3, "top_p": 0.9, "max_length": 30, "name": "Nucleus (p=0.9)"}
            ]
            
            for params in gen_params:
                try:
                    print(f"\n{params['name']}:")
                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids=input_ids,
                            do_sample=params.get("do_sample", False),
                            temperature=params.get("temperature", 1.0),
                            top_k=params.get("top_k", 50),
                            top_p=params.get("top_p", 1.0),
                            max_length=params["max_length"],
                            pad_token_id=tokenizer.eos_token_id,
                            num_return_sequences=1
                        )
                    
                    # Decode and print
                    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"Generated: {generated}")
                    
                except Exception as e:
                    print(f"Error during generation: {e}")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nError in test_model_generation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    test_model_generation()
