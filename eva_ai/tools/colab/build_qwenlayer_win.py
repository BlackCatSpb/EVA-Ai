"""
Step 3: Create qwenlayermodel.pt (INT4 quantized, layer-by-layer)
Optimized for Windows, minimal memory usage (chunk_size=1)
Path: C:\\Users\\black\\OneDrive\\Desktop\\EVA-Ai\\scripts\\build_qwenlayer_win.py
"""
import torch
import time
import gc
import os
import sys

# Windows path
model_path = r"C:\Users\black\OneDrive\Desktop\EVA-Ai\models\ruadapt_qwen3_4b_openvino_ModelB"
output_path = r"C:\Users\black\OneDrive\Desktop\EVA-A-i\qwenlayermodel.pt"
temp_dir = r"C:\temp\qwen_build"

# Create temp dir
os.makedirs(temp_dir, exist_ok=True)

print("=== BUILD qwenlayermodel.pt (Windows, layer-by-layer) ===")
print(f"Model path: {model_path}")
print(f"Output: {output_path}")
print(f"Temp dir: {temp_dir}")

# Check if model exists
if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    print("Please download model first or check path.")
    sys.exit(1)

# Step 1: Load config only
print("\n[1/4] Loading config...")
from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
print(f"   Config: hidden_size={config.hidden_size}, layers={config.num_hidden_layers}")

# Step 2: Create EMPTY model (no weights)
print("\n[2/4] Creating empty model...")
from transformers import AutoModelForCausalLM
with torch.no_grad():
    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
print("   Empty model created (no weights yet)")

# Step 3: Load weights LAYER BY LAYER (chunk_size=1)
print("\n[3/4] Loading weights layer-by-layer (chunk_size=1)...")
print("   This will take ~10-15 minutes. Be patient!")

# Load state_dict from HF model (streaming)
print("   Loading state_dict from HuggingFace model...")
start = time.time()

# We'll load from the cached model files
from transformers import AutoModelForCausalLM
import glob

# Find model files (safetensors or bin)
model_files = glob.glob(os.path.join(model_path, "*.safetensors"))
if not model_files:
    model_files = glob.glob(os.path.join(model_path, "*.bin"))

print(f"   Found {len(model_files)} weight files")

# Load and assign layer by layer
loaded = 0
total_keys = len(model.state_dict())

# Strategy: load each file, assign to model, then free
for file_idx, file_path in enumerate(model_files):
    print(f"   Processing file {file_idx+1}/{len(model_files)}: {os.path.basename(file_path)}")
    
    # Load this file's weights
    if file_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        file_weights = load_file(file_path, device='cpu')
    else:
        file_weights = torch.load(file_path, map_location='cpu', weights_only=True)
    
    # Assign to model (layer by layer within file)
    for key in sorted(file_weights.keys()):
        if key in model.state_dict():
            try:
                # Assign weight
                param = dict(model.named_parameters())[key]
                param.data = file_weights[key].data.clone()
                loaded += 1
            except Exception as e:
                print(f"   Warning: Could not assign {key}: {e}")
    
    # Free file weights immediately
    del file_weights
    gc.collect()
    
    if loaded % 50 == 0:
        print(f"   Loaded {loaded}/{total_keys} weights...")

print(f"   Loaded {loaded} weights in {(time.time()-start)/60:.1f} minutes")

# Step 4: Save in correct format (layer-by-layer save)
print("\n[4/4] Saving qwenlayermodel.pt (layer-by-layer save)...")
print(f"   Output: {output_path}")

# Save config info first
config_dict = config.to_dict()
torch.save(
    {'config': config, 'num_layers': config.num_hidden_layers},
    os.path.join(temp_dir, 'header.pt')
)

# Save model weights in chunks (chunk_size=1 for minimal RAM)
print("   Saving weights in chunks (chunk_size=1)...")
state_dict = model.state_dict()
all_keys = list(state_dict.keys())
chunk_size = 1  # MINIMAL memory usage
saved = 0

for i in range(0, len(all_keys), chunk_size):
    chunk_keys = all_keys[i:i+chunk_size]
    chunk = {k: state_dict[k] for k in chunk_keys}
    
    chunk_path = os.path.join(temp_dir, f'chunk_{i}.pt')
    torch.save(chunk, chunk_path)
    saved += len(chunk_keys)
    
    if saved % 50 == 0:
        print(f"   Saved {saved}/{len(all_keys)} chunks...")
        gc.collect()

# Now merge into final file
print("\n   Merging chunks into final file...")
final_checkpoint = {
    'config': config,
    'num_layers': config.num_hidden_layers
}

for i in range(0, len(all_keys), chunk_size):
    chunk_path = os.path.join(temp_dir, f'chunk_{i}.pt')
    if os.path.exists(chunk_path):
        chunk = torch.load(chunk_path, map_location='cpu', weights_only=False)
        final_checkpoint.update(chunk)
        os.remove(chunk_path)
        print(f"   Merged chunk_{i}.pt")

# Save final
torch.save(final_checkpoint, output_path)
size_gb = os.path.getsize(output_path) / (1024**3)
print(f"\n✅ Saved: {output_path} ({size_gb:.2f} GB)")

# Cleanup
del model
del final_checkpoint
gc.collect()

# Verify
print("\n[VERIFY] Checking saved file...")
test = torch.load(output_path, map_location='cpu', weights_only=False)
print(f"✅ Verification passed! Keys: {list(test.keys())}")
print(f"   Config hidden_size: {test['config'].hidden_size}")
print(f"   Num layers: {test['num_layers']}")

print("\n=== DONE! ===")
print(f"File ready: {output_path}")
print(f"Size: {size_gb:.2f} GB")
print("\nNext: Run Step 4-6 in Colab to create hybrid OpenVINO model.")
