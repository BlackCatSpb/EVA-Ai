"""
Create OpenVINO XML for quantized hybrid transformer layer.
This defines the graph structure for OpenVINO.
"""
import os

def create_openvino_xml(output_dir: str, hidden_dim: int = 2560, num_heads: int = 32):
    """
    Create OpenVINO XML definition for the hybrid transformer layer.
    """
    print(f"Creating OpenVINO XML with hidden_dim={hidden_dim}, num_heads={num_heads}...")
    
    # FF dim (SwiGLU uses 2x hidden for gate, 1x hidden for up, then projects back)
    ff_dim = hidden_dim * 2  # SwiGLU: gate_proj and up_proj use 2x hidden
    
    # Write XML file
    xml_path = os.path.join(output_dir, 'hybrid_layer.xml')
    
    with open(xml_path, 'w', encoding='utf-8') as f:
        # Write XML content
        f.write('<?xml version="1.0" ?>\n')
        f.write(f'<net name="hybrid_transformer_layer" version="10">\n')
        f.write('    <layers>\n')
        
        # Input layer
        f.write(f'        <!-- Input -->\n')
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
        f.write(f'        <!-- Self-Attention -->\n')
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
        f.write(f'        <!-- FFN: gate_proj (SwiGLU) -->\n')
        f.write(f'        <layer id="3" name="gate_proj" type="FullyConnected">\n')
        f.write(f'            <data out-size="{ff_dim}"/>\n')
        f.write(f'            <input>\n')
        f.write(f'                <port id="0" precision="FP32" dims="-1,{hidden_dim}"/>\n')
        f.write(f'            </input>\n')
        f.write(f'            <output>\n')
        f.write(f'                <port id="1" precision="FP32" dims="-1,{ff_dim}"/>\n')
        f.write(f'            </output>\n')
        f.write(f'            <weights offset="0" size="{ff_dim * hidden_dim * 4}"/>\n')
        f.write(f'            <biases offset="{ff_dim * hidden_dim * 4}" size="{ff_dim * 4}"/>\n')
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
        f.write(f'            <weights offset="0" size="{ff_dim * hidden_dim * 4}"/>\n')
        f.write(f'            <biases offset="{ff_dim * hidden_dim * 4}" size="{ff_dim * 4}"/>\n')
        f.write(f'        </layer>\n')
        
        # FFN: SiLU activation
        f.write(f'        <!-- FFN: SiLU activation -->\n')
        f.write(f'        <layer id="5" name="silu" type="Swish">\n')
        f.write(f'            <input>\n')
        f.write(f'                <port id="0" precision="FP32" dims="-1,{ff_dim}"/>\n')
        f.write(f'            </input>\n')
        f.write(f'            <output>\n')
        f.write(f'                <port id="1" precision="FP32" dims="-1,{ff_dim}"/>\n')
        f.write(f'            </output>\n')
        f.write(f'        </layer>\n')
        
        # FFN: elementwise mul (gate * up)
        f.write(f'        <!-- FFN: elementwise mul (gate * up) -->\n')
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
        f.write(f'        <!-- FFN: down_proj -->\n')
        f.write(f'        <layer id="7" name="down_proj" type="FullyConnected">\n')
        f.write(f'            <data out-size="{hidden_dim}"/>\n')
        f.write(f'            <input>\n')
        f.write(f'                <port id="0" precision="FP32" dims="-1,{ff_dim}"/>\n')
        f.write(f'            </input>\n')
        f.write(f'            <output>\n')
        f.write(f'                <port id="1" precision="FP32" dims="-1,{hidden_dim}"/>\n')
        f.write(f'            </output>\n')
        f.write(f'            <weights offset="0" size="{hidden_dim * ff_dim * 4}"/>\n')
        f.write(f'            <biases offset="{hidden_dim * ff_dim * 4}" size="{hidden_dim * 4}"/>\n')
        f.write(f'        </layer>\n')
        
        # LayerNorm before output
        f.write(f'        <!-- LayerNorm before output -->\n')
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
        f.write('        <edge from-layer="4" from-port="1" to-layer="5" to-port="0"/>\n')
        f.write('        <edge from-layer="3" from-port="1" to-layer="6" to-port="0"/>\n')
        f.write('        <edge from-layer="5" from-port="1" to-layer="6" to-port="1"/>\n')
        f.write('        <edge from-layer="6" from-port="2" to-layer="7" to-port="0"/>\n')
        f.write('        <edge from-layer="0" from-port="0" to-layer="8" to-port="0"/>\n')
        f.write('        <edge from-layer="7" from-port="1" to-layer="8" to-port="0"/>\n')
        f.write('    </edges>\n')
        
        # Output
        f.write('    <output>\n')
        f.write(f'        <port id="0" precision="FP32" dims="-1,{hidden_dim}"/>\n')
        f.write('    </output>\n')
        
        f.write('</net>\n')
    
    print(f"Saved OpenVINO XML to {xml_path}")
    return xml_path

def main():
    """Main function to create OpenVINO XML."""
    output_dir = r"C:\Users\black\OneDrive\Desktop\EVA-Ai\models\hybrid_openvino"
    
    xml_path = create_openvino_xml(output_dir)
    
    print("=" * 60)
    print("OPENVINO XML CREATED!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"XML: hybrid_layer.xml")
    print("")
    print("Next steps:")
    print("1. Use these files in OpenVINO LLMPipeline")
    print("2. Load with: ov_genai.LLMPipeline(model_path)")
    print("=" * 60)

if __name__ == '__main__':
    main()
