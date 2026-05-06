# -*- coding: utf-8 -*-
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import requests
import time

local_dir = r'C:\Users\black\OneDrive\Desktop\EVA-Ai\eva_pie_architecture\models\gguf_models\RefalMachine_RuadaptQwen3-4B-Hybrid'
os.makedirs(local_dir, exist_ok=True)

parts = [
    ('model-00001-of-00002.safetensors', 'https://huggingface.co/RefalMachine/RuadaptQwen3-4B-Hybrid/resolve/main/model-00001-of-00002.safetensors'),
    ('model-00002-of-00002.safetensors', 'https://huggingface.co/RefalMachine/RuadaptQwen3-4B-Hybrid/resolve/main/model-00002-of-00002.safetensors'),
]

for filename, url in parts:
    filename = os.path.join(local_dir, filename)
    print(f'\n{"="*60}')
    print(f'Downloading: {os.path.basename(filename)}')
    print(f'{"="*60}')
    
    if os.path.exists(filename) and os.path.getsize(filename) > 3000000000:
        print(f'Already exists: {os.path.getsize(filename) / 1024 / 1024:.0f} MB')
        continue
    
    print(f'URL: {url}')
    start = time.time()
    
    response = requests.get(url, stream=True, timeout=120)
    print(f'Status: {response.status_code}')
    
    total = int(response.headers.get('content-length', 0))
    print(f'Size: {total / 1024 / 1024:.0f} MB')
    
    downloaded = 0
    last_print = 0
    
    with open(filename + '.tmp', 'wb') as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                
                if time.time() - last_print > 2:
                    pct = (downloaded / total * 100) if total else 0
                    speed = downloaded / (time.time() - start) / 1024 / 1024
                    eta = (total - downloaded) / (downloaded / (time.time() - start)) if downloaded > 0 else 0
                    print(f'  {downloaded / 1024 / 1024:.0f} / {total / 1024 / 1024:.0f} MB ({pct:.1f}%) | {speed:.1f} MB/s | ETA: {eta:.0f}s')
                    last_print = time.time()
    
    os.rename(filename + '.tmp', filename)
    elapsed = time.time() - start
    print(f'Complete: {os.path.getsize(filename) / 1024 / 1024:.0f} MB in {elapsed:.0f}s')

print('\nAll downloads complete!')
