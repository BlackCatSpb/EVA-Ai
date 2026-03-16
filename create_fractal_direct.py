#!/usr/bin/env python3

import os
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM
from cogniflex.mlearning.storage.fractal_store import FractalWeightStore

def create_fractal_from_model():
    """Создаёт фрактальное хранилище напрямую из модели с правильными метаданными."""
    
    # Загружаем модель
    model_path = r'C:\Users\black\OneDrive\Desktop\CogniFlex\text-generation'
    output_path = r'C:\Users\black\OneDrive\Desktop\CogniFlex\cache\ml_unit\fractal_storage\models\text-generation'
    
    print(f"Загрузка модели из {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype="float32",
    )
    
    print(f"Model config n_embd: {model.config.n_embd}")
    
    # Создаём фрактальное хранилище напрямую
    block_size = 8192
    fractal_levels = 3
    
    fs = FractalWeightStore(block_size=block_size, fractal_levels=fractal_levels)
    
    # Получаем state_dict
    state_dict = model.state_dict()
    
    print(f"Упаковка {len(state_dict)} тензоров...")
    
    # Упаковываем каждый тензор напрямую
    for tensor_path, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
            
        print(f"  {tensor_path}: {tensor.shape}")
        
        # Конвертируем в numpy
        arr = tensor.detach().cpu().numpy()
        flat = arr.reshape(-1)
        total_elements = int(flat.size)
        original_shape = tuple(int(x) for x in arr.shape)
        
        # Определяем тип хранения
        is_critical = any(critical in tensor_path for critical in ["wte", "wpe", "ln_f", "lm_head"])
        storage_dtype = "float64" if is_critical else "float32"
        
        # Получаем layer_name и param_name
        try:
            layer_name, param_name = tensor_path.rsplit('.', 1)
        except ValueError:
            layer_name, param_name = tensor_path, "weight"
        
        layer_key = f"{layer_name}.{param_name}" if layer_name else tensor_path
        
        # Создаём контейнеры
        for i in range(0, total_elements, block_size):
            block = flat[i:i + block_size]
            
            if storage_dtype == "float64":
                block = block.astype(np.float64, copy=False)
            else:
                block = block.astype(np.float32, copy=False)
            
            position = (i // block_size,)
            cid = fs._generate_container_id(0, position, layer_key, "text-generation")
            
            # Создаём метаданные с правильной формой
            meta = {
                "layer_name": layer_name,
                "model_id": "text-generation",
                "original_shape": original_shape,  # Правильная форма!
                "block_start": i,
                "block_end": min(i + block_size, total_elements),
                "is_critical": is_critical,
                "storage_dtype": storage_dtype,
                "param_name": param_name,
                "tensor_path": tensor_path,
            }
            
            # Создаём контейнер напрямую
            from cogniflex.mlearning.storage.fractal_store import FractalContainer
            container = FractalContainer(
                id=cid,
                level=0,
                position=position,
                data=block,
                shape=(int(block.size),),
                dtype=storage_dtype,
                metadata=meta,
            )
            
            fs.containers[cid] = container
            if 0 not in fs.fractal_tree:
                fs.fractal_tree[0] = []
            fs.fractal_tree[0].append(cid)
            fs.total_memory += container.get_memory_size()
    
    # Устанавливаем model_id
    fs.model_id = "text-generation"
    
    print(f"Создано {len(fs.containers)} контейнеров уровня 0")
    
    # Строим иерархию
    print("Построение фрактальной иерархии...")
    fs._build_fractal_hierarchy()
    fs._initialize_hot_window()
    
    # Сохраняем без оптимизации (чтобы не испортить метаданные)
    print("Сохранение на диск...")
    os.makedirs(output_path, exist_ok=True)
    
    # Сохраняем конфиг
    model.config.to_json_file(os.path.join(output_path, "config.json"))
    
    # Сохраняем токенизатор
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer_dir = os.path.join(output_path, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    
    # Сохраняем фрактальное хранилище
    success = fs.save_to_disk_sharded(
        output_path,
        knowledge_graph=None,
        shard_size=10000,
        by_level=True,
        compress=True,
    )
    
    print(f"Результат: {success}")
    
    # Проверяем метаданные
    print("\nПроверка метаданных:")
    for cid, cont in fs.containers.items():
        if cont.metadata.get('tensor_path') == 'transformer.h.0.ln_1.weight':
            print(f"  {cid}: original_shape = {cont.metadata.get('original_shape')}")
            break

if __name__ == "__main__":
    create_fractal_from_model()
