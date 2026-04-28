"""
Convert hybrid_weights.npz to OpenVINO .bin format.

OpenVINO expects:
- model.xml (network structure)  
- model.bin (weights in raw binary format)

This script converts the numpy arrays in .npz to a single .bin file.
"""
import os
import sys
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("convert_hybrid_bin")

def convert_npz_to_bin(npz_path: str, bin_path: str):
    """
    Convert .npz (numpy zip) to .bin (raw binary).
    
    OpenVINO .bin format:
    - Weights are stored as consecutive raw bytes
    - Each weight array is stored with:
      1. Shape as int32 array (2 values for 2D, 1 for 1D)
      2. Data as float32 raw bytes
    """
    logger.info(f"Loading .npz from {npz_path}")
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        logger.info(f"Loaded {len(data.keys())} tensors")
        
        # Create .bin file
        logger.info(f"Creating .bin at {bin_path}")
        
        with open(bin_path, 'wb') as f:
            # Write number of tensors
            num_tensors = len(data.keys())
            f.write(np.array([num_tensors], dtype=np.int32).tobytes())
            
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    # Convert to float32 if needed
                    if value.dtype == np.float64:
                        value = value.astype(np.float32)
                    elif value.dtype == np.float16:
                        # Convert float16 to float32 for compatibility
                        value = value.astype(np.float32)
                    
                    # Write key length + key
                    key_bytes = key.encode('utf-8')
                    f.write(np.array([len(key_bytes)], dtype=np.int32).tobytes())
                    f.write(key_bytes)
                    
                    # Write shape
                    shape = np.array(value.shape, dtype=np.int32)
                    f.write(np.array([len(shape)], dtype=np.int32).tobytes())
                    f.write(shape.tobytes())
                    
                    # Write data as float32
                    if value.dtype != np.float32:
                        value = value.astype(np.float32)
                    f.write(value.tobytes())
                    
                    logger.debug(f"Written {key}: shape={value.shape}, dtype={value.dtype}")
        
        logger.info(f"SUCCESS: Converted to {bin_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to convert: {e}")
        import traceback
        traceback.print_exc()
        return False

def rename_for_openvino(model_dir: str):
    """Rename files to what OpenVINO expects."""
    xml_path = os.path.join(model_dir, 'hybrid_layer.xml')
    bin_path = os.path.join(model_dir, 'hybrid_weights.bin')
    npz_path = os.path.join(model_dir, 'hybrid_weights.npz')
    
    # Rename .npz to .bin
    if os.path.exists(npz_path) and not os.path.exists(bin_path):
        logger.info(f"Renaming {npz_path} to {bin_path}")
        os.rename(npz_path, bin_path)
    
    # Rename .xml to model.xml
    model_xml = os.path.join(model_dir, 'model.xml')
    if os.path.exists(xml_path) and not os.path.exists(model_xml):
        logger.info(f"Renaming {xml_path} to {model_xml}")
        os.rename(xml_path, model_xml)
    
    return bin_path.replace('hybrid_weights.bin', 'model.bin')

def main():
    """Main conversion function."""
    model_dir = r"C:\Users\black\OneDrive\Desktop\EVA-Ai\models\hybrid_openvino"
    npz_path = os.path.join(model_dir, 'hybrid_weights.npz')
    bin_path = os.path.join(model_dir, 'model.bin')
    xml_path = os.path.join(model_dir, 'hybrid_layer.xml')
    model_xml = os.path.join(model_dir, 'model.xml')
    
    logger.info("=" * 60)
    logger.info("CONVERSION TO OPENVINO FORMAT")
    logger.info("=" * 60)
    
    # Step 1: Convert .npz to .bin
    if os.path.exists(npz_path):
        success = convert_npz_to_bin(npz_path, bin_path)
        if not success:
            logger.error("Conversion failed!")
            return False
    else:
        logger.warning(f".npz not found: {npz_path}")
        # Maybe it's already converted
        if not os.path.exists(bin_path):
            logger.error("No .bin file found either!")
            return False
    
    # Step 2: Rename .xml to model.xml
    if os.path.exists(xml_path) and not os.path.exists(model_xml):
        logger.info(f"Renaming {xml_path} -> {model_xml}")
        os.rename(xml_path, model_xml)
    
    # Step 3: Verify
    files = os.listdir(model_dir)
    logger.info(f"Files in {model_dir}: {files}")
    
    has_xml = any(f.endswith('.xml') for f in files)
    has_bin = any(f.endswith('.bin') for f in files)
    
    logger.info("=" * 60)
    if has_xml and has_bin:
        logger.info("CONVERSION SUCCESSFUL!")
        logger.info(f"model.xml: {os.path.exists(model_xml)}")
        logger.info(f"model.bin: {os.path.exists(bin_path)}")
    else:
        logger.error("CONVERSION FAILED - missing files!")
    logger.info("=" * 60)
    
    return has_xml and has_bin

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
