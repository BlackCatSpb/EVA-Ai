import torch
from safetensors.torch import load_file
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

def compare_models():
    print("=== Model Comparison Test ===\n")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # 1. Load tokenizer
        print("\n1. Loading tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded successfully")
        
        # Test text
        test_text = "Привет, как дела?"
        inputs = tokenizer(test_text, return_tensors="pt").to(device)
        print(f"Test text: {test_text}")
        print(f"Token IDs: {inputs.input_ids.tolist()}")
        
        # 2. Test original model
        print("\n2. Testing original model...")
        try:
            original_model = GPT2LMHeadModel.from_pretrained(
                "sberbank-ai/rugpt3small_based_on_gpt2"
            ).to(device)
            print("Original model loaded successfully")
            
            with torch.no_grad():
                outputs = original_model(**inputs)
                print(f"Original model logits shape: {outputs.logits.shape}")
                print(f"First 5 logits: {outputs.logits[0, -1, :5].tolist()}")
                
                # Generate text with original model
                print("\nOriginal model generation:")
                gen = original_model.generate(
                    **inputs,
                    max_length=20,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
                print(f"Generated: {tokenizer.decode(gen[0], skip_special_tokens=True)}")
                
        except Exception as e:
            print(f"Error with original model: {e}")
        
        # 3. Test fractal model
        print("\n3. Testing fractal model...")
        try:
            # Create model with same config as original
            config = GPT2Config.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
            fractal_model = GPT2LMHeadModel(config).to(device)
            
            # Load fractal weights
            print("Loading fractal weights...")
            state_dict = load_file("out/fractal_rugpt_full.safetensors", device=device)
            
            # Convert to float32 for better compatibility
            state_dict = {k: v.float() for k, v in state_dict.items()}
            
            # Load state dict
            fractal_model.load_state_dict(state_dict, strict=False)
            print("Fractal model loaded with weights")
            
            # Test forward pass
            with torch.no_grad():
                outputs = fractal_model(**inputs)
                print(f"Fractal model logits shape: {outputs.logits.shape}")
                print(f"First 5 logits: {outputs.logits[0, -1, :5].tolist()}")
                
                # Generate text with fractal model
                print("\nFractal model generation (with temperature=0.7):")
                try:
                    gen = fractal_model.generate(
                        **inputs,
                        max_length=20,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    print(f"Generated: {tokenizer.decode(gen[0], skip_special_tokens=True)}")
                except Exception as e:
                    print(f"Error during fractal model generation: {e}")
                
                # Try with lower temperature
                print("\nFractal model generation (with temperature=0.1):")
                try:
                    gen = fractal_model.generate(
                        **inputs,
                        max_length=20,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.1,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    print(f"Generated: {tokenizer.decode(gen[0], skip_special_tokens=True)}")
                except Exception as e:
                    print(f"Error during fractal model generation: {e}")
                
                # Try greedy decoding
                print("\nFractal model generation (greedy search):")
                try:
                    gen = fractal_model.generate(
                        **inputs,
                        max_length=20,
                        num_return_sequences=1,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    print(f"Generated: {tokenizer.decode(gen[0], skip_special_tokens=True)}")
                except Exception as e:
                    print(f"Error during fractal model greedy generation: {e}")
            
        except Exception as e:
            print(f"Error with fractal model: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"\nError in compare_models: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    compare_models()
