#!/usr/bin/env python3

import torch
import numpy as np
from cogniflex.mlearning.storage.fractal_store import FractalWeightStore

# Создадим тестовый state_dict с правильными размерами
test_state_dict = {
    'transformer.h.0.ln_1.weight': torch.randn(768),
    'transformer.h.0.attn.c_attn.weight': torch.randn(768, 2304),
}

# Создаём хранилище
store = FractalWeightStore(block_size=8192, fractal_levels=3)

# Упаковываем
result = store.pack_state_dict(test_state_dict, model_id="test-model")

print('Pack result:', result)

# Проверяем метаданные
for cid, container in store.containers.items():
    if container.metadata.get('tensor_path') == 'transformer.h.0.ln_1.weight':
        print(f'Container {cid}:')
        print(f'  original_shape: {container.metadata.get("original_shape")}')
        print(f'  data shape: {container.data.shape}')
        break
