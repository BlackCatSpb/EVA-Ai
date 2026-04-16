# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from huggingface_hub import hf_hub_download
import os

local_dir = r'C:\Users\black\OneDrive\Desktop\EVA-Ai\eva_pie_architecture\models\gguf_models\RefalMachine_RuadaptQwen3-4B-Hybrid'

print("Downloading safetensors files...")
print("This may take a while due to large file size (~4GB each)")
print()

files = [
    'model-00001-of-00002.safetensors',
    'model-00002-of-00002.safetensors'
]

for f in files:
    dst = os.path.join(local_dir, f)
    if os.path.exists(dst) and os.path.getsize(dst) > 2000000000:
        size = os.path.getsize(dst)
        print(f"Already exists: {f} ({size / 1024 / 1024:.1f} MB)")
    else:
        print(f"Downloading {f}...")
        print(f"Target: {dst}")
        try:
            path = hf_hub_download(
                'RefalMachine/RuadaptQwen3-4B-Hybrid',
                f,
                local_dir=local_dir
            )
            size = os.path.getsize(path)
            print(f"Downloaded: {size / 1024 / 1024:.1f} MB")
        except Exception as e:
            print(f"Error: {e}")
            print("File will need to be downloaded manually or with longer timeout")

print()
print("Download complete. Check if both safetensors files are present.")
