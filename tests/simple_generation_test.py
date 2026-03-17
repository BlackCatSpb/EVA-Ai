import torch
from safetensors.torch import load_file
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer

def test_simple_generation():
    print("Testing simple text generation...")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # 1. Load the model architecture
        print("Loading model architecture...")
        config = GPT2Config.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
        model = GPT2LMHeadModel(config).to(device)
        
        # 2. Load the weights
        print("Loading weights...")
        state_dict = load_file("out/fractal_rugpt_full.safetensors", device=device)
        
        # Convert all tensors to float32 for better compatibility
        state_dict = {k: v.float() for k, v in state_dict.items()}
        
        # Load state dict with strict=False to handle missing keys
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        
        # 3. Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 4. Test generation
        print("\nTesting generation...")
        prompts = [
            "Привет, как дела?",
            "Сегодня я хочу рассказать о",
            "Искусственный интеллект - это"
        ]
        
        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                print(f"\nPrompt: {prompt}")
                
                # Encode input
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                # Generate
                outputs = model.generate(
                    **inputs,
                    max_length=50,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=0.7
                )
                
                # Decode and print
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Generated: {generated}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    test_simple_generation()
