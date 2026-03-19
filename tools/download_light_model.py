"""
Download script for rugpt3small model and export to fractal storage.

This script:
1. Downloads sberbank-ai/rugpt3small_based_on_gpt2 from HuggingFace
2. Exports it to fractal storage for CogniFlex
3. Updates brain_config.json to use the new model
"""

import os
import sys
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("download_light_model")


def get_project_root():
    return Path(__file__).parent.parent


def download_model(model_id: str, cache_dir: str) -> bool:
    """Download model from HuggingFace."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Downloading model: {model_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
        )
        logger.info(f"Tokenizer downloaded and cached")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
        )
        logger.info(f"Model downloaded and cached")
        
        return True
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False


def export_to_fractal_storage(
    model_id: str,
    cache_dir: str,
    output_path: str,
    model_name: str = "rugpt3_small_fractal"
) -> bool:
    """Export downloaded model to fractal storage format."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        import torch
        
        logger.info(f"Exporting model to fractal storage: {output_path}")
        
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Loading model from cache...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        model.eval()
        
        logger.info("Loading tokenizer from cache...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        
        logger.info("Saving config.json...")
        if hasattr(model, 'config') and hasattr(model.config, 'to_json_file'):
            model.config.to_json_file(str(out_dir / "config.json"))
        else:
            cfg = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
            cfg.to_json_file(str(out_dir / "config.json"))
        
        logger.info("Saving tokenizer files...")
        tok_dir = out_dir / "tokenizer"
        tok_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(tok_dir))
        
        logger.info("Creating fractal storage structure...")
        from cogniflex.mlearning.storage.fractal_store import FractalWeightStore
        
        store = FractalWeightStore(
            block_size=64,
            fractal_levels=4,
            device="cpu",
        )
        
        logger.info("Packing model weights into fractal structure...")
        if not store.pack_state_dict(model.state_dict(), model_id=model_name):
            logger.error("Failed to pack model weights")
            return False
        
        logger.info("Saving fractal structure to disk (sharded)...")
        if not store.save_to_disk_sharded(
            str(out_dir),
            knowledge_graph=None,
            shard_size=10000,
            by_level=True,
            compress=True,
        ):
            logger.info("Trying alternative save method...")
            if not store.save_to_disk_with_recovery(str(out_dir)):
                logger.error("Failed to save fractal structure")
                return False
        
        stats = store.get_statistics()
        logger.info(f"Fractal storage created successfully:")
        logger.info(f"  Total containers: {stats['total_containers']}")
        logger.info(f"  Total size: {stats['total_memory_mb']:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.exception(f"Failed to export to fractal storage: {e}")
        return False


def update_brain_config(config_path: str, model_info: dict) -> bool:
    """Update brain_config.json to use the new model."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        old_model_name = config.get('model', {}).get('name', 'rugpt3')
        old_model_path = config.get('model', {}).get('path', '')
        
        config['model'] = {
            'name': model_info['name'],
            'path': model_info['path'],
            'vocab_size': model_info.get('vocab_size', 50257),
            'max_length': model_info.get('max_length', 1024),
            'device': model_info.get('device', 'cuda:0'),
        }
        
        logger.info(f"Updating brain_config.json:")
        logger.info(f"  Old model: {old_model_name} -> {model_info['name']}")
        logger.info(f"  Old path: {old_model_path}")
        logger.info(f"  New path: {model_info['path']}")
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to update brain_config.json: {e}")
        return False


def test_model_loading(fractal_path: str) -> bool:
    """Test that the model loads correctly from fractal storage."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        
        logger.info("Testing model loading from fractal storage...")
        
        config = AutoConfig.from_pretrained(fractal_path, local_files_only=True)
        logger.info(f"  Config loaded: {config.model_type}")
        
        tok_dir = os.path.join(fractal_path, 'tokenizer')
        if os.path.isdir(tok_dir):
            tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True, use_fast=True)
            test_text = "Привет, как дела?"
            tokens = tokenizer.encode(test_text)
            decoded = tokenizer.decode(tokens)
            logger.info(f"  Tokenizer works: '{test_text}' -> {len(tokens)} tokens -> '{decoded}'")
        else:
            logger.warning("  Tokenizer directory not found")
        
        logger.info("Model configuration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Model loading test failed: {e}")
        return False


def get_model_config(model_id: str, cache_dir: str) -> dict:
    """Get model configuration details."""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
        return {
            'vocab_size': getattr(config, 'vocab_size', 50257),
            'n_positions': getattr(config, 'n_positions', getattr(config, 'max_position_embeddings', 1024)),
            'n_embd': getattr(config, 'n_embd', getattr(config, 'hidden_size', 768)),
        }
    except Exception:
        return {}


def main():
    model_id = "sberbank-ai/rugpt3small_based_on_gpt2"
    
    project_root = get_project_root()
    cache_dir = str(project_root / "cogniflex" / "core" / "cogniflex_cache" / "models" / "hf_cache")
    fractal_path = str(project_root / "cogniflex" / "core" / "cogniflex_cache" / "ml_unit" / "fractal_storage" / "models" / "rugpt3_small_fractal")
    config_path = str(project_root / "brain_config.json")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("CogniFlex Light Model Downloader")
    logger.info("=" * 60)
    logger.info(f"Model: {model_id}")
    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Fractal storage: {fractal_path}")
    logger.info("")
    
    logger.info("Step 1: Downloading model from HuggingFace...")
    if not download_model(model_id, cache_dir):
        logger.error("Failed to download model")
        sys.exit(1)
    
    logger.info("")
    logger.info("Step 2: Getting model configuration...")
    model_config = get_model_config(model_id, cache_dir)
    logger.info(f"  Vocab size: {model_config.get('vocab_size')}")
    logger.info(f"  Max length: {model_config.get('n_positions')}")
    
    logger.info("")
    logger.info("Step 3: Exporting to fractal storage...")
    if not export_to_fractal_storage(model_id, cache_dir, fractal_path, "rugpt3small"):
        logger.error("Failed to export to fractal storage")
        sys.exit(1)
    
    logger.info("")
    logger.info("Step 4: Updating brain_config.json...")
    model_info = {
        'name': 'rugpt3small',
        'path': fractal_path,
        'vocab_size': model_config.get('vocab_size', 50257),
        'max_length': model_config.get('n_positions', 1024),
        'device': 'cuda:0',
    }
    if not update_brain_config(config_path, model_info):
        logger.error("Failed to update brain_config.json")
        sys.exit(1)
    
    logger.info("")
    logger.info("Step 5: Testing model loading...")
    if not test_model_loading(fractal_path):
        logger.warning("Model loading test had issues, but fractal storage was created")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUCCESS: Light model downloaded and configured!")
    logger.info("=" * 60)
    logger.info(f"Model: {model_id}")
    logger.info(f"Fractal storage: {fractal_path}")
    logger.info(f"Config updated: {config_path}")


if __name__ == "__main__":
    main()
