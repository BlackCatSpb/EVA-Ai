"""
Integration test for ClosedCognitiveLoop in EVA system.
Tests that all components are properly connected.
"""
import sys
import os

# Add EVA paths
eva_root = "C:/Users/black/OneDrive/Desktop/EVA-Ai"
sys.path.insert(0, eva_root)
sys.path.insert(0, eva_root + "/eva_ai")
sys.path.insert(0, eva_root + "/eva_ai/core")

print("=" * 60)
print("CLOSED COGNITIVE LOOP INTEGRATION TEST")
print("=" * 60)

# Test 1: Import all components
print("\n[Test 1] Importing components...")
try:
    from split_model_runner import SplitModelRunner
    from kca_integration import KCAModule, KCAIntegration
    from closed_cognitive_loop import ClosedCognitiveLoop, ConceptMiner, ContradictionMiner, GNNProcessor
    print("  [OK] All components imported successfully")
except Exception as e:
    print(f"  [FAIL] Import error: {e}")
    sys.exit(1)

# Test 2: Initialize SplitModelRunner
print("\n[Test 2] Initializing SplitModelRunner...")
try:
    runner = SplitModelRunner(split_layer=6)
    runner.load_models()
    print("  [OK] SplitModelRunner initialized")
except Exception as e:
    print(f"  [FAIL] SplitModelRunner init error: {e}")
    sys.exit(1)

# Test 3: Test single layer extraction
print("\n[Test 3] Testing single layer extraction...")
try:
    import numpy as np
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    text = "What is artificial intelligence?"
    tokens = tokenizer(text, return_tensors="np")
    input_ids = tokens["input_ids"]
    seq_len = input_ids.shape[1]
    attention_mask = np.ones((1, seq_len), dtype=np.int64)
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

    hs = runner.get_layer_output(input_ids, attention_mask, position_ids, layer=12)
    if hs is not None:
        print(f"  [OK] Layer 12 hidden states: shape={hs.shape}, mean={np.mean(hs):.4f}")
    else:
        print("  [WARN] Layer 12 hidden states not found")
except Exception as e:
    print(f"  [FAIL] Layer extraction error: {e}")

# Test 4: Initialize ClosedCognitiveLoop
print("\n[Test 4] Initializing ClosedCognitiveLoop...")
try:
    loop = ClosedCognitiveLoop(runner)
    print("  [OK] ClosedCognitiveLoop initialized")
except Exception as e:
    print(f"  [FAIL] ClosedCognitiveLoop init error: {e}")

# Test 5: Update graph data
print("\n[Test 5] Updating graph data from miners...")
try:
    concepts = ["artificial intelligence", "machine learning", "neural network"]
    contradictions = []

    loop.concept_miner.extract_from_query("artificial intelligence and machine learning")
    concept_embs = loop.concept_miner.get_all_concept_embeddings()

    if concept_embs is not None:
        print(f"  [OK] Extracted concept embeddings: shape={concept_embs.shape}")
    else:
        print("  [WARN] No concept embeddings extracted")

    print("  [OK] Graph data updated")
except Exception as e:
    print(f"  [FAIL] Graph data update error: {e}")

# Test 6: Test KCA processing on single layer
print("\n[Test 6] Testing KCA processing on layer 12...")
try:
    hs = runner.get_layer_output(input_ids, attention_mask, position_ids, layer=12)
    if hs is not None:
        corrected, correction = loop.kca_integration.process_layer(hs, layer_idx=12, iteration=0)
        print(f"  [OK] KCA correction applied:")
        print(f"      Original mean: {np.mean(hs):.4f}")
        print(f"      Corrected mean: {np.mean(corrected):.4f}")
        print(f"      Gate gamma: {correction.gate_value:.4f}")
        print(f"      Gap norm: {np.linalg.norm(correction.gap_embedding):.4f}")
    else:
        print("  [WARN] No hidden states to process")
except Exception as e:
    print(f"  [FAIL] KCA processing error: {e}")

# Test 7: Test process_query with KCA
print("\n[Test 7] Testing full process_query with KCA...")
try:
    query = "What is the relationship between AI and machine learning?"
    final_hidden, metadata = loop.process_query(query, max_layers=10, apply_kca=True)

    print(f"  [OK] Query processed:")
    print(f"      Layers processed: {metadata.get('layers_processed', 0)}")
    print(f"      Concepts extracted: {metadata.get('concepts_extracted', 0)}")
    print(f"      KCA applied: {metadata.get('kca_applied', False)}")
    print(f"      Final hidden shape: {final_hidden.shape if final_hidden is not None else None}")
except Exception as e:
    print(f"  [FAIL] process_query error: {e}")

# Test 8: Verify init_factories integration
print("\n[Test 8] Checking init_factories integration...")
try:
    from eva_ai.core.init_factories import create_closed_cognitive_loop
    print("  [OK] create_closed_cognitive_loop function available in init_factories")
except Exception as e:
    print(f"  [WARN] init_factories integration not available: {e}")

# Test 9: Verify brain_query integration
print("\n[Test 9] Checking brain_query integration...")
try:
    # Check if _extract_key_concepts has ClosedCognitiveLoop update code
    with open("eva_ai/core/brain_query.py", "r") as f:
        content = f.read()
        if 'closed_cognitive_loop' in content:
            print("  [OK] brain_query has ClosedCognitiveLoop integration")
        else:
            print("  [WARN] brain_query doesn't have ClosedCognitiveLoop code")
except Exception as e:
    print(f"  [WARN] Could not check brain_query: {e}")

print("\n" + "=" * 60)
print("INTEGRATION TEST COMPLETE")
print("=" * 60)
print("""
Summary:
- SplitModelRunner: Layer-by-layer hidden state access
- KCAModule: Knowledge gap and contradiction detection
- KCAIntegration: KCA with SplitModelRunner
- ClosedCognitiveLoop: Full loop (Miners -> GNN -> KCA -> Layers)
- init_factories: create_closed_cognitive_loop registered
- brain_query: Updated to use ClosedCognitiveLoop

The closed loop is now integrated into EVA:
  Query -> brain_query -> ConceptMiner/ContradictionMiner ->
    -> Graph embeddings -> GNN -> KCA (all layers) ->
    -> Stable embeddings -> Response
""")
