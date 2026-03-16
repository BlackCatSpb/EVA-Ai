#!/usr/bin/env python3

from transformers import AutoModelForCausalLM
from cogniflex.mlearning.storage.fractal_store import export_hf_model_to_fractal

# Загружаем модель напрямую и проверяем размеры
model = AutoModelForCausalLM.from_pretrained(
    r'C:\Users\black\OneDrive\Desktop\CogniFlex\text-generation',
    local_files_only=True,
    low_cpu_mem_usage=True,
    torch_dtype="float32",
)

print('Model config n_embd:', model.config.n_embd)
state_dict = model.state_dict()
print('ln_1.weight shape:', state_dict['transformer.h.0.ln_1.weight'].shape)
print('c_attn.weight shape:', state_dict['transformer.h.0.attn.c_attn.weight'].shape)

# Теперь экспортируем
ok = export_hf_model_to_fractal(
    hf_model_dir_or_id=r'C:\Users\black\OneDrive\Desktop\CogniFlex\text-generation',
    output_path=r'C:\Users\black\OneDrive\Desktop\CogniFlex\debug_export_output',
    model_id="debug-model",
    device="cpu",
    fractal_levels=3,
    block_size=8192,
    local_files_only=True,
)

print('Export result:', ok)
