#!/usr/bin/env python3

import sys
from cogniflex.mlearning.storage.fractal_store import FractalWeightStore

def debug_fractal():
    store_path = r'C:\Users\black\OneDrive\Desktop\CogniFlex\cache\ml_unit\fractal_storage\models\text-generation'
    fs = FractalWeightStore(block_size=64, fractal_levels=5)
    fs.load_from_disk(store_path)
    
    # Проверить записи для ln_1.weight
    recs = [e for e in fs.lazy_index.values() 
            if int(e.get('level', -1)) == 0 and e.get('tensor_path') == 'transformer.h.0.ln_1.weight']
    
    print(f"Records for ln_1.weight: {len(recs)}")
    if recs:
        print(f"original_shape: {recs[0].get('original_shape')}")
        print(f"block_start: {recs[0].get('block_start')}")
        print(f"block_end: {recs[0].get('block_end')}")
        data = recs[0].get('data')
        if hasattr(data, 'shape'):
            print(f"first data shape: {data.shape}")
        else:
            print(f"first data type: {type(data)}")
    
    # Восстановить state_dict для проверки
    try:
        state_dict = fs.reconstruct_state_dict()
        if 'transformer.h.0.ln_1.weight' in state_dict:
            print(f"Reconstructed ln_1.weight shape: {state_dict['transformer.h.0.ln_1.weight'].shape}")
        else:
            print("ln_1.weight not found in reconstructed state_dict")
    except Exception as e:
        print(f"Error reconstructing: {e}")

if __name__ == "__main__":
    debug_fractal()
