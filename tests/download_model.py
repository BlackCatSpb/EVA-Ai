# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import requests
from huggingface_hub import get_token
from tqdm import tqdm

local_dir = r'C:\Users\black\OneDrive\Desktop\EVA-Ai\eva_pie_architecture\models\gguf_models\RefalMachine_RuadaptQwen3-4B-Hybrid'
os.makedirs(local_dir, exist_ok=True)

# HuggingFace API
API_URL = "https://huggingface.co/api/models/RefalMachine/RuadaptQwen3-4B-Hybrid"

print("Downloading RefalMachine/RuadaptQwen3-4B-Hybrid (PyTorch)...")
print()

files_to_download = [
    ('model-00001-of-00002.safetensors', 3800000000),  # ~3.8 GB
    ('model-00002-of-00002.safetensors', 4200000000),  # ~4.2 GB
]

for filename, expected_size in files_to_download:
    dst = os.path.join(local_dir, filename)
    
    if os.path.exists(dst) and os.path.getsize(dst) > expected_size * 0.9:
        size = os.path.getsize(dst)
        print(f"Already exists: {filename} ({size / 1024 / 1024:.1f} MB)")
        continue
    
    print(f"Downloading {filename}...")
    print(f"Size: ~{expected_size / 1024 / 1024:.0f} MB")
    
    # Construct direct URL
    url = f"https://huggingface.co/RefalMachine/RuadaptQwen3-4B-Hybrid/resolve/main/{filename}"
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', expected_size))
        
        with open(dst + '.tmp', 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (100 * 1024 * 1024) < 8192:  # Print every ~100MB
                        print(f"  Downloaded: {downloaded / 1024 / 1024:.0f} MB / {total_size / 1024 / 1024:.0f} MB")
        
        os.rename(dst + '.tmp', dst)
        size = os.path.getsize(dst)
        print(f"Complete: {filename} ({size / 1024 / 1024:.1f} MB)")
        
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        if os.path.exists(dst + '.tmp'):
            print("Partial download saved. Run script again to continue.")
        break

print()
print("Done!")
