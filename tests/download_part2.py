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

filename = 'model-00002-of-00002.safetensors'
url = f'https://huggingface.co/RefalMachine/RuadaptQwen3-4B-Hybrid/resolve/main/{filename}'
filepath = os.path.join(local_dir, filename)

print(f'Downloading: {filename}')
print(f'URL: {url}')
print()

start = time.time()
last_print = 0

try:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    
    total = int(response.headers.get('content-length', 0))
    print(f'Size: {total / 1024 / 1024:.0f} MB')
    
    downloaded = 0
    with open(filepath + '.tmp', 'wb') as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                
                if time.time() - last_print > 3:
                    pct = (downloaded / total * 100) if total else 0
                    speed = downloaded / (time.time() - start) / 1024 / 1024
                    eta = (total - downloaded) / (downloaded / (time.time() - start)) if downloaded > 0 else 0
                    print(f'  {downloaded / 1024 / 1024:.0f} / {total / 1024 / 1024:.0f} MB ({pct:.1f}%) | {speed:.1f} MB/s | ETA: {eta:.0f}s')
                    last_print = time.time()
    
    os.rename(filepath + '.tmp', filepath)
    elapsed = time.time() - start
    print(f'\nComplete! {os.path.getsize(filepath) / 1024 / 1024:.0f} MB in {elapsed:.0f}s')

except KeyboardInterrupt:
    print('\nDownload interrupted by user')
except Exception as e:
    print(f'\nError: {e}')
