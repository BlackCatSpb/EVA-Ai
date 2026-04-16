# -*- coding: utf-8 -*-
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import time

local_dir = r'C:\Users\black\OneDrive\Desktop\EVA-Ai\eva_pie_architecture\models\gguf_models\RefalMachine_RuadaptQwen3-4B-Hybrid'
os.makedirs(local_dir, exist_ok=True)

url = 'https://huggingface.co/RefalMachine/RuadaptQwen3-4B-Hybrid/resolve/main/model-00001-of-00002.safetensors'
filename = os.path.join(local_dir, 'model-00001-of-00002.safetensors')

print(f'Downloading...')
print(f'URL: {url}')

start = time.time()
response = requests.get(url, stream=True, timeout=120)
print(f'Status: {response.status_code}')

content_len = response.headers.get('content-length', '0')
total = int(content_len)
print(f'Content-Length: {total / 1024 / 1024:.0f} MB')

downloaded = 0

with open(filename + '.tmp', 'wb') as f:
    for chunk in response.iter_content(chunk_size=65536):
        if chunk:
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded % (50 * 1024 * 1024) < 65536:
                pct = (downloaded / total * 100) if total else 0
                speed = downloaded / (time.time() - start) / 1024 / 1024
                print(f'Downloaded: {downloaded / 1024 / 1024:.0f} / {total / 1024 / 1024:.0f} MB ({pct:.1f}%) - {speed:.1f} MB/s')

os.rename(filename + '.tmp', filename)
elapsed = time.time() - start
print(f'Complete!')
print(f'Size: {os.path.getsize(filename) / 1024 / 1024:.0f} MB')
print(f'Time: {elapsed:.0f}s')
print(f'Speed: {os.path.getsize(filename) / elapsed / 1024 / 1024:.1f} MB/s')
