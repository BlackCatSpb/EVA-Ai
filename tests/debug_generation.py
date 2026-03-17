import torch
from safetensors.torch import load_file
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
import numpy as np

def debug_generation():
    print("Starting debug generation...")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # 1. Load the tokenizer first
        print("\n1. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded successfully")
        
        # 2. Test tokenizer
        print("\n2. Testing tokenizer...")
        test_text = "Привет, как дела?"
        tokens = tokenizer.encode(test_text, return_tensors="pt").to(device)
        print(f"Input text: {test_text}")
        print(f"Token IDs: {tokens.tolist()}")
        print(f"Decoded back: {tokenizer.decode(tokens[0])}")
        
        # 3. Load model config
        print("\n3. Loading model config...")
        config = GPT2Config.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
        model = GPT2LMHeadModel(config).to(device)
        print("Model initialized with config")
        
        # 4. Load weights
        print("\n4. Loading model weights...")
        try:
            state_dict = load_file("out/fractal_rugpt_full.safetensors", device=device)
            print(f"Loaded state dict with {len(state_dict)} parameters")
            
            # Convert to float32 for better compatibility
            state_dict = {k: v.float() for k, v in state_dict.items()}
            
            # Load state dict with strict=False to handle potential mismatches
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Missing keys: {len(missing)}")
            print(f"Unexpected keys: {len(unexpected)}")
            
            if missing:
                print("First few missing keys:", missing[:5])
            if unexpected:
                print("First few unexpected keys:", unexpected[:5])
                
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Trying to load partial weights...")
            model.load_state_dict(state_dict, strict=False)
        
        # 5. Test model forward pass
        print("\n5. Testing model forward pass...")
        model.eval()
        with torch.no_grad():
            outputs = model(tokens)
            print(f"Output logits shape: {outputs.logits.shape}")
            print(f"Output logits sample: {outputs.logits[0, :3, :3].tolist()}")
        
        # 6. Test generation with different parameters
        print("\n6. Testing generation with different parameters...")
        prompts = [
            "Привет, как дела?",
            "Сегодня я хочу рассказать о",
            "Искусственный интеллект - это"
        ]
        
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            
            # Encode input
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_length = inputs.input_ids.shape[1]
            
            # Try different generation parameters
            generation_params = [
                {"do_sample": False, "num_beams": 1},  # Greedy search
                {"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.9},
                {"do_sample": True, "temperature": 0.3, "top_k": 0, "top_p": 0.9},
            ]
            
            for i, params in enumerate(generation_params, 1):
                print(f"\nGeneration {i} with {params}:")
                try:
                    outputs = model.generate(
                        **inputs,
                        max_length=input_length + 20,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id,
                        **params
                    )
                    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"Generated: {generated}")
                except Exception as e:
                    print(f"Error during generation: {e}")
        
        # 7. Check model parameters
        print("\n7. Checking model parameters...")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Check first layer weights
        first_weight = next(model.parameters())
        print(f"First parameter shape: {first_weight.shape}")
        print(f"First parameter stats - Mean: {first_weight.mean().item():.4f}, "
              f"Std: {first_weight.std().item():.4f}, "
              f"Min: {first_weight.min().item():.4f}, "
              f"Max: {first_weight.max().item():.4f}")
        
    except Exception as e:
        print(f"\nError in debug_generation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    debug_generation()
