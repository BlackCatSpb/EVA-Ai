#!/usr/bin/env python3

from transformers import AutoModelForCausalLM
import torch

# Загружаем модель
model = AutoModelForCausalLM.from_pretrained(
    r'C:\Users\black\OneDrive\Desktop\CogniFlex\text-generation',
    local_files_only=True,
    low_cpu_mem_usage=True,
    torch_dtype="float32",
)

print('Model config n_embd:', model.config.n_embd)
print('Model type:', type(model).__name__)

# Проверяем state_dict
state_dict = model.state_dict()
ln_1_weight = state_dict['transformer.h.0.ln_1.weight']
print('ln_1.weight shape:', ln_1_weight.shape)
print('ln_1.weight dtype:', ln_1_weight.dtype)

# Проверим, есть ли какие-то трансформации
print('ln_1.weight is_contiguous:', ln_1_weight.is_contiguous())
print('ln_1.weight storage size:', ln_1_weight.storage().size() if ln_1_weight.storage() else 'No storage')

# Проверим другие параметры
c_attn_weight = state_dict['transformer.h.0.attn.c_attn.weight']
print('c_attn.weight shape:', c_attn_weight.shape)
print('c_attn.weight is_contiguous:', c_attn_weight.is_contiguous())

# Попробуем создать новый state_dict с копиями тензоров
new_state_dict = {}
for key, tensor in state_dict.items():
    if 'transformer.h.0.ln_1.weight' in key or 'transformer.h.0.attn.c_attn.weight' in key:
        new_state_dict[key] = tensor.clone().contiguous()
        print(f'{key} cloned shape: {new_state_dict[key].shape}')
