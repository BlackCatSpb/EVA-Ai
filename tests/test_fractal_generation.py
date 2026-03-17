import torch
from safetensors.torch import load_file
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
import numpy as np
from typing import List, Union, Optional

class FractalTextGenerator:
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Initializing model on {self.device}...")
        
        # Load ruGPT3 small config
        self.config = GPT2Config.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
        
        # Initialize model with config
        self.model = GPT2LMHeadModel(self.config).to(self.device)
        
        # Load fractal weights
        self.load_fractal_weights(model_path)
        
        # Initialize tokenizer
        print("Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
            # Ensure pad token is set
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise
        
        # Set model to evaluation mode
        self.model.eval()
    
    def load_fractal_weights(self, model_path: str):
        """Load weights from fractal storage into the model"""
        print(f"Loading weights from {model_path}...")
        try:
            # Load state dict
            state_dict = load_file(model_path, device=self.device)
            
            # Convert all tensors to the correct device and dtype
            for key in state_dict:
                state_dict[key] = state_dict[key].to(device=self.device, dtype=torch.float32)
            
            # Load state dict into model with strict=False to handle missing keys
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            print(f"Successfully loaded {len(state_dict)} parameters")
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
                
            # Clean up
            del state_dict
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error loading weights: {e}")
            raise
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        device: Optional[str] = None
    ) -> Union[str, List[str]]:
        """Generate text from prompt"""
        device = device or self.device
        self.model.to(device)
        
        try:
            # Encode the prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            print(f"Input shape: {input_ids.shape}")
            print(f"Generating text with max_length={max_length}...")
            
            # Generate text
            output_sequences = self.model.generate(
                input_ids=input_ids,
                max_length=min(max_length, input_ids.shape[1] + 100),  # Limit to max 100 new tokens
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                attention_mask=torch.ones_like(input_ids)  # Add attention mask
            )
            
            # Decode and return the generated text
            generated_sequences = []
            for i, generated_sequence in enumerate(output_sequences):
                # Skip the prompt
                gen_seq = generated_sequence[input_ids.shape[1]:].tolist()
                text = self.tokenizer.decode(gen_seq, skip_special_tokens=True)
                print(f"Generated {i+1}: {text[:100]}...")
                generated_sequences.append(text)
            
            return generated_sequences[0] if num_return_sequences == 1 else generated_sequences
            
        except Exception as e:
            print(f"Error during generation: {e}")
            raise
        finally:
            # Clean up
            torch.cuda.empty_cache()

def main():
    # Initialize the generator with your fractal model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        generator = FractalTextGenerator(
            model_path="out/fractal_rugpt_full.safetensors",
            device=device
        )
        
        # Test prompts
        test_prompts = [
            "Привет, как дела?",
            "Сегодня я хочу рассказать о",
            "Искусственный интеллект - это",
        ]
        
        for prompt in test_prompts:
            print(f"\n{'='*50}")
            print(f"PROMPT: {prompt}")
            print(f"{'='*50}")
            
            try:
                generated = generator.generate(
                    prompt,
                    max_length=100,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    do_sample=True,
                    num_return_sequences=1
                )
                
                print(f"\nRESULT: {generated}")
                print(f"{'='*50}\n")
                
            except Exception as e:
                print(f"Error generating text: {e}")
                continue
                
    except Exception as e:
        print(f"Initialization error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
