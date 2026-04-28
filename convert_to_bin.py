"""
Convert hybrid_weights.npz to proper OpenVINO .bin format.

OpenVINO .bin format: raw float32 weights concatenated in order matching model.xml
"""
import os
import numpy as np
import logging
import struct

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("convert_to_bin")

def convert_npz_to_bin():
    model_dir = r"C:\Users\black\OneDrive\Desktop\EVA-Ai\models\hybrid_openvino"
    npz_path = os.path.join(model_dir, "hybrid_weights.npz")
    bin_path = os.path.join(model_dir, "model.bin")
    xml_path = os.path.join(model_dir, "model.xml")
    
    logger.info(f"Loading {npz_path}")
    
    try:
        # Load .npz
        data = np.load(npz_path, allow_pickle=False)
        logger.info(f"Loaded {len(data.keys())} arrays")
        
        # Get keys in sorted order
        keys = sorted(data.keys())
        logger.info(f"Keys: {keys[:10]}...")
        
        # Write to .bin in sorted order
        total_bytes = 0
        with open(bin_path, 'wb') as f:
            for key in keys:
                arr = data[key]
                # Convert to float32 if needed
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                
                raw = arr.tobytes()
                f.write(raw)
                total_bytes += len(raw)
                logger.debug(f"  {key}: shape={arr.shape}, {len(raw)} bytes")
        
        logger.info(f"SUCCESS! Written {total_bytes} bytes ({total_bytes/1024/1024:.2f} MB) to {bin_path}")
        
        # Verify
        file_size = os.path.getsize(bin_path)
        logger.info(f"File size verification: {file_size} bytes")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    convert_npz_to_bin()
