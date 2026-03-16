#!/usr/bin/env python3

from cogniflex.mlearning.storage.fractal_store import FractalWeightStore
from transformers import AutoConfig, AutoModelForCausalLM

# Загружаем фрактальное хранилище
task_dir = r'C:\Users\black\OneDrive\Desktop\CogniFlex\cache\ml_unit\fractal_storage\models\text-generation'

fs = FractalWeightStore(block_size=8192, fractal_levels=3)
fs.load_from_disk(task_dir, lazy=True)

print('Loading state_dict...')
state_dict = fs.reconstruct_state_dict(output_dtype='float32', device='cpu')

print('State dict keys for ln_1.weight:')
for key in state_dict.keys():
    if 'ln_1.weight' in key:
        print(f'  {key}: {state_dict[key].shape}')

print('\nLoading config...')
cfg = AutoConfig.from_pretrained(task_dir, local_files_only=True)
print(f'Config n_embd: {cfg.n_embd}')

print('\nCreating model from config...')
model = AutoModelForCausalLM.from_config(cfg)
model_state = model.state_dict()
print(f'Model ln_1.weight shape: {model_state["transformer.h.0.ln_1.weight"].shape}')

# Сравниваем
if 'transformer.h.0.ln_1.weight' in state_dict:
    fractal_shape = state_dict['transformer.h.0.ln_1.weight'].shape
    model_shape = model_state['transformer.h.0.ln_1.weight'].shape
    print(f'\nShape comparison:')
    print(f'  Fractal: {fractal_shape}')
    print(f'  Model:   {model_shape}')
    print(f'  Match:   {fractal_shape == model_shape}')
