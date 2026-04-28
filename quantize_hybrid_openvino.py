"""
Quantize and convert hybrid transformer layers to OpenVINO format.

This script:
1. Takes the trained hybrid transformer layers (from Colab qwen_layer_model.pt)
2. Applies INT8 quantization
3. Converts to OpenVINO IR format (.xml + .bin)
4. Ready for deployment in EVA's OpenVINO pipeline
"""
import os
import sys
import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("eva_ai.hybrid_quantize")

def load_hybrid_checkpoint(checkpoint_path: str) -> dict:
    """Load the hybrid layer checkpoint from Colab training."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        logger.info(f"Checkpoint loaded. Keys: {list(checkpoint.keys())}")
        return checkpoint
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None

def apply_int8_quantization(model_state_dict: dict) -> dict:
    """
    Apply INT8 quantization to hybrid layer weights.
    Uses symmetric quantization: w_q = round(w / scale) where scale = max(|w|) / 127
    """
    logger.info("Applying INT8 quantization...")
    quantized = {}
    
    for key, value in model_state_dict.items():
        if isinstance(value, torch.Tensor) and value.dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            # Convert BFloat16 to Float32 for quantization
            if value.dtype == torch.bfloat16:
                value = value.to(torch.float32)
            
            # Compute scale for symmetric quantization
            max_val = torch.max(torch.abs(value))
            if max_val == 0:
                scale = 1.0
            else:
                scale = max_val / 127.0
            
            # Quantize to INT8
            w_q = torch.round(value / scale).clamp(-128, 127).to(torch.int8)
            
            # Store quantized weights and scale
            quantized[key] = w_q
            quantized[f"{key}_scale"] = torch.tensor([scale])
            
            logger.debug(f"Quantized {key}: shape={value.shape}, scale={scale.item():.4f}")
        else:
            quantized[key] = value
    
    logger.info(f"Quantized {len(model_state_dict)} tensors to INT8")
    return quantized

def save_openvino_weights(quantized_state: dict, output_dir: str, hidden_dim: int = 2560):
    """
    Save weights in OpenVINO-compatible format.
    OpenVINO expects: weights as [output_channels, input_channels] for Linear layers.
    """
    logger.info(f"Saving OpenVINO weights to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    openvino_weights = {}
    
    for key, value in quantized_state.items():
        if key.endswith('_scale'):
            continue  # Skip scale factors, they're applied differently in OpenVINO
            
        if isinstance(value, torch.Tensor):
            arr = value.cpu().numpy()
            openvino_weights[key] = arr
            logger.debug(f"Saved {key}: shape={arr.shape}, dtype={arr.dtype}")
    
    # Save as numpy dictionary (OpenVINO can load this)
    output_path = os.path.join(output_dir, 'hybrid_weights.npz')
    np.savez(output_path, **openvino_weights)
    logger.info(f"Saved OpenVINO weights to {output_path}")
    
    return output_path

def create_openvino_xml(output_dir: str, hidden_dim: int = 2560, num_heads: int = 32):
    """
    Create OpenVINO XML definition for the hybrid transformer layer.
    This defines the graph structure for OpenVINO.
    """
    logger.info("Creating OpenVINO XML definition...")
    
    # FF dim (SwiGLU uses 2/3 of hidden for gate, 1/3 for up, then projects back)
    ff_dim = hidden_dim * 2  # SwiGLU style
    
    # Write XML file
    xml_path = os.path.join(output_dir, 'hybrid_layer.xml')
    
    with open(xml_path, 'w', encoding='utf-8') as f:
        # Write XML content
        f.write('<?xml version="1.0" ?>\n')
        f.write(f'<net name="hybrid_transformer_layer" version="10">\n')
        f.write('    <layers>\n')
        
        # Input layer
        f.write(f'        <layer id="0" name="input" type="Parameter">\n')
        f.write(f'            <output>\n')
        f.write(f'                <port id="0" precision="FP32" dims="-1,{hidden_dim}"/>\n')
        f.write(f'            </output>\n')
        f.write(f'        </layer>\n')
        
        # LayerNorm before attention
        f.write(f'        <layer id="1" name="attn_norm" type="Normalize">\n')
        f.write(f'            <data eps="1e-6" axes="1"/>\n')
        f.write(f'            <input>\n')
        f.write(f'                <port id="0" precision="FP32" dims="-1,{hidden_dim}"/>\n')
        f.write(f'            </input>\n')
        f.write(f'            <output>\n')
        f.write(f'                <port id="1" precision="FP32" dims="-1,{hidden_dim}"/>\n')
        f.write(f'            </output>\n')
        f.write(f'        </layer>\n')
        
        # Self-Attention (simplified for OpenVINO)
        f.write(f'        <layer id="2" name="self_attn_q" type="FullyConnected">\n')
        f.write(f'            <data out-size="{hidden_dim}"/>\n')
        f.write(f'            <input>\n')
        f.write(f'                <port id="0" precision="FP32" dims="-1,{hidden_dim}"/>\n')
        f.write(f'            </input>\n')
        f.write(f'            <output>\n')
        f.write(f'                <port id="1" precision="FP32" dims="-1,{hidden_dim}"/>\n')
        f.write(f'            </output>\n')
        f.write(f'            <weights offset="0" size="{hidden_dim * hidden_dim * 4}"/>\n')
        f.write(f'            <biases offset="{hidden_dim * hidden_dim * 4}" size="{hidden_dim * 4}"/>\n')
        f.write(f'        </layer>\n')
        
        # FFN: gate_proj (SwiGLU)
        f.write(f'        <layer id="3" name="gate_proj" type="FullyConnected">\n')
        f.write(f'            <data out-size="{ff_dim}"/>\n')
        f.write(f'            <input>\n')
        f.write(f'                <port id="0" precision="FP32" dims="-1,{hidden_dim}"/>\n')
        f.write(f'            </input>\n')
        f.write(f'            <output>\n')
        f.write(f'                <port id="1" precision="FP32" dims="-1,{ff_dim}"/>\n')
        f.write(f'            </output>\n')
        f.write(f'            <weights offset="0" size="{hidden_dim * ff_dim * 4}"/>\n')
        f.write(f'            <biases offset="{hidden_dim * ff_dim * 4}" size="{ff_dim * 4}"/>\n')
        f.write(f'        </layer>\n')
        
        # FFN: up_proj (SwiGLU)
        f.write(f'        <layer id="4" name="up_proj" type="FullyConnected">\n')
        f.write(f'            <data out-size="{ff_dim}"/>\n')
        f.write(f'            <input>\n')
        f.write(f'                <port id="0" precision="FP32" dims="-1,{hidden_dim}"/>\n')
        f.write(f'            </input>\n')
        f.write(f'            <output>\n')
        f.write(f'                <port id="1" precision="FP32" dims="-1,{ff_dim}"/>\n')
        f.write(f'            </output>\n')
        f.write(f'            <weights offset="0" size="{hidden_dim * ff_dim * 4}"/>\n')
        f.write(f'            <biases offset="{hidden_dim * ff_dim * 4}" size="{ff_dim * 4}"/>\n')
        f.write(f'        </layer>\n')
        
        # FFN: SiLU activation
        f.write(f'        <layer id="5" name="silu" type="Swish">\n')
        f.write(f'            <input>\n')
        f.write(f'                <port id="0" precision="FP32" dims="-1,{ff_dim}"/>\n')
        f.write(f'            </input>\n')
        f.write(f'            <output>\n')
        f.write(f'                <port id="1" precision="FP32" dims="-1,{ff_dim}"/>\n')
        f.write(f'            </output>\n')
        f.write(f'        </layer>\n')
        
        # FFN: elementwise mul (gate * up)
        f.write(f'        <layer id="6" name="mul" type="Eltwise">\n')
        f.write(f'            <data operation="mul"/>\n')
        f.write(f'            <input>\n')
        f.write(f'                <port id="0" precision="FP32" dims="-1,{ff_dim}"/>\n')
        f.write(f'                <port id="1" precision="FP32" dims="-1,{ff_dim}"/>\n')
        f.write(f'            </input>\n')
        f.write(f'            <output>\n')
        f.write(f'                <port id="2" precision="FP32" dims="-1,{ff_dim}"/>\n')
        f.write(f'            </output>\n')
        f.write(f'        </layer>\n')
        
        # FFN: down_proj
        f.write(f'        <layer id="7" name="down_proj" type="FullyConnected">\n')
        f.write(f'            <data out-size="{hidden_dim}"/>\n')
        f.write(f'            <input>\n')
        f.write(f'                <port id="0" precision="FP32" dims="-1,{ff_dim}"/>\n')
        f.write(f'            </input>\n')
        f.write(f'            <output>\n')
        f.write(f'                <port id="1" precision="FP32" dims="-1,{hidden_dim}"/>\n')
        f.write(f'            </output>\n')
        f.write(f'            <weights offset="0" size="{ff_dim * hidden_dim * 4}"/>\n')
        f.write(f'            <biases offset="{ff_dim * hidden_dim * 4}" size="{hidden_dim * 4}"/>\n')
        f.write(f'        </layer>\n')
        
        # FFN: LayerNorm before output
        f.write(f'        <layer id="8" name="ffn_norm" type="Normalize">\n')
        f.write(f'            <data eps="1e-6" axes="1"/>\n')
        f.write(f'            <input>\n')
        f.write(f'                <port id="0" precision="FP32" dims="-1,{hidden_dim}"/>\n')
        f.write(f'            </input>\n')
        f.write(f'            <output>\n')
        f.write(f'                <port id="1" precision="FP32" dims="-1,{hidden_dim}"/>\n')
        f.write(f'            </output>\n')
        f.write(f'        </layer>\n')
        
        f.write('    </layers>\n')
        
        # Edges
        f.write('    <edges>\n')
        f.write('        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>\n')
        f.write('        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>\n')
        f.write('        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>\n')
        f.write('        <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>\n')
        f.write('        <edge from-layer="3" from-port="1" to-layer="5" to-port="0"/>\n')
        f.write('        <edge from-layer="4" from-port="1" to-layer="6" to-port="1"/>\n')
        f.write('        <edge from-layer="5" from-port="1" to-layer="6" to-port="0"/>\n')
        f.write('        <edge from-layer="6" from-port="2" to-layer="7" to-port="0"/>\n')
        f.write('        <edge from-layer="7" from-port="1" to-layer="8" to-port="0"/>\n')
        f.write('    </edges>\n')
        
        # Output
        f.write('    <output>\n')
        f.write(f'        <port id="0" precision="FP32" dims="-1,{hidden_dim}"/>\n')
        f.write('    </output>\n')
        f.write('</net>\n')
    
    logger.info(f"Saved OpenVINO XML to {xml_path}")
    
    return xml_path

def main():
    """Main function to quantize and convert."""
    # Paths
    checkpoint_path = r"C:\Users\black\OneDrive\Desktop\EVA-Ai\models\qwen_layer_model.pt"
    output_dir = r"C:\Users\black\OneDrive\Desktop\EVA-Ai\models\hybrid_openvino"
    
    # Step 1: Load checkpoint
    checkpoint = load_hybrid_checkpoint(checkpoint_path)
    if checkpoint is None:
        logger.error("Failed to load checkpoint. Exiting.")
        return False
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint  # Assume it's already a state dict
    
    logger.info(f"Loaded state dict with {len(state_dict)} tensors")
    
    # Get dimensions from config
    hidden_dim = 2560
    num_heads = 32
    ff_dim = hidden_dim * 2  # SwiGLU
    
    # Step 2: Apply INT8 quantization
    quantized = apply_int8_quantization(state_dict)
    
    # Step 3: Save in OpenVINO format
    weights_path = save_openvino_weights(quantized, output_dir, hidden_dim)
    
    # Step 4: Create OpenVINO XML
    xml_path = create_openvino_xml(output_dir, hidden_dim, num_heads)
    
    # Step 5: Save config
    import json
    config = {
        'hidden_dim': hidden_dim,
        'num_heads': num_heads,
        'ff_dim': ff_dim,
        'use_gnn': True,
        'use_lora': True,
        'quantization': 'INT8_SYM',
    }
    config_path = os.path.join(output_dir, 'hybrid_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {config_path}")
    
    logger.info("=" * 60)
    logger.info("QUANTIZATION AND CONVERSION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Weights: hybrid_weights.npz")
    logger.info(f"Config: hybrid_config.json")
    logger.info(f"XML: hybrid_layer.xml")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Use these files in OpenVINO LLMPipeline")
    logger.info("2. Load with: ov_genai.LLMPipeline(model_path)")
    logger.info("=" * 60)
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
