"""
Download EVA-Ai models (minimal)
Only qwen3 4b and coder
"""

import os

def download():
    from huggingface_hub import hf_hub_download
    
    base = r"C:\Users\black\OneDrive\Desktop\EVA-Ai\eva_pie_architecture\models\gguf_models"
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join(base, "qwen2.5-coder-1.5b-instruct"), exist_ok=True)
    
    # 1. RuadaptQwen3-4B (~2.4 GB)
    print("Downloading RuadaptQwen3-4B...")
    hf_hub_download(
        repo_id="RefalMachine/RuadaptQwen3-4B-Instruct-GGUF",
        filename="Q4_K_M.gguf",
        local_dir=base,
        local_dir_use_symlinks=False
    )
    
    # 2. Qwen Coder 1.5B (~1 GB)
    print("Downloading Qwen Coder 1.5B...")
    hf_hub_download(
        repo_id="Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", 
        filename="qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        local_dir=os.path.join(base, "qwen2.5-coder-1.5b-instruct"),
        local_dir_use_symlinks=False
    )
    
    print("Done!")

if __name__ == "__main__":
    download()
