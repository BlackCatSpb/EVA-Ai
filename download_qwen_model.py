#!/usr/bin/env python3
"""
Script to download Qwen3.5-2B model for CogniFlex
"""
import os
import sys

def download_qwen_model():
    model_name = "Qwen/Qwen3.5-2B"
    target_dir = "cogniflex/mlearning/cogniflex_models/qwen3.5-2b"
    
    print(f"Downloading {model_name} to {target_dir}")
    print("This may take 10-30 minutes depending on your internet speed...")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(target_dir)
        print("Tokenizer downloaded!")
        
        print("Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="float16"
        )
        model.save_pretrained(target_dir)
        print("Model downloaded!")
        
        print(f"\n✓ Model saved to: {os.path.abspath(target_dir)}")
        print("You can now run CogniFlex with Qwen3.5-2B!")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\nAlternative: Download manually from HuggingFace:")
        print(f"https://huggingface.co/{model_name}")
        return False
    
    return True

if __name__ == "__main__":
    success = download_qwen_model()
    sys.exit(0 if success else 1)
