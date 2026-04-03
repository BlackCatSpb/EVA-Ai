import os

gguf_dir = r'C:/Users/black/OneDrive/Desktop/CogniFlex/eva/memory/fractal_torch_storage/gguf_models'
files = [f for f in os.listdir(gguf_dir) if f.endswith('.gguf')]

print("GGUF files in fractal_torch_storage:")
for f in files:
    size_mb = os.path.getsize(os.path.join(gguf_dir, f)) / 1024 / 1024
    print(f"  {f}: {size_mb:.1f} MB")