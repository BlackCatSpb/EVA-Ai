"""
Create the missing fractal_rugpt_full.safetensors file
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

def create_model_file():
    # Load the model
    model_name = "sberbank-ai/rugpt3large_based_on_gpt2"
    print(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    # Save as safetensors
    state_dict = model.state_dict()
    save_file(state_dict, "out/fractal_rugpt_full.safetensors")

    print("Model saved to out/fractal_rugpt_full.safetensors")

if __name__ == "__main__":
    create_model_file()
