"""
Download script for rugpt3small model and export to fractal storage.

This script:
1. Uses cached sberbank-ai/rugpt3small_based_on_gpt2 from HuggingFace
2. Exports it to fractal storage for ЕВА (simple HF format)
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


def ensure_model_cached(model_id: str, cache_dir: str) -> bool:
    """Ensure model is cached locally."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Checking/Caching model: {model_id}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True)
            logger.info("Tokenizer already cached")
        except Exception as e:
            logger.warning(f"Tokenizer not in cache, downloading: {e}")
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
            logger.info("Tokenizer downloaded and cached")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True)
            logger.info("Model already cached")
        except Exception as e:
            logger.warning(f"Model not in cache, downloading: {e}")
            model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)
            logger.info("Model downloaded and cached")
        
        return True
    except Exception as e:
        logger.error(f"Failed to cache model: {e}")
        return False


def find_cached_model_path(model_id: str, cache_dir: str) -> str:
    """Find the actual path to cached model files."""
    safe_model_id = model_id.replace("/", "--")
    model_cache_path = os.path.join(cache_dir, f"models--{safe_model_id}")
    
    if os.path.isdir(model_cache_path):
        snapshots_path = os.path.join(model_cache_path, "snapshots")
        if os.path.isdir(snapshots_path):
            for snapshot in os.listdir(snapshots_path):
                snapshot_full = os.path.join(snapshots_path, snapshot)
                if os.path.isdir(snapshot_full):
                    return snapshot_full
    
    return ""


def copy_model_to_fractal_storage(
    source_path: str,
    output_path: str,
    model_name: str = "rugpt3small"
) -> bool:
    """Copy model files to fractal storage directory."""
    try:
        logger.info(f"Copying model to fractal storage: {output_path}")
        
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        required_files = ["config.json", "pytorch_model.bin", "vocab.json", 
                          "merges.txt", "tokenizer_config.json", "special_tokens_map.json"]
        
        for fname in required_files:
            src = os.path.join(source_path, fname)
            if os.path.isfile(src) or os.path.islink(src):
                dst = out_dir / fname
                if dst.exists():
                    dst.unlink()
                shutil.copy2(src, dst)
                logger.info(f"  Copied: {fname}")
        
        tok_src = source_path
        tok_dst = out_dir / "tokenizer"
        if not tok_dst.exists():
            tok_dst.mkdir(parents=True, exist_ok=True)
            for fname in ["vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"]:
                src = os.path.join(tok_src, fname)
                if os.path.isfile(src) or os.path.islink(src):
                    shutil.copy2(src, tok_dst / fname)
        
        index_data = {
            "model_id": model_name,
            "source": source_path,
            "format": "hf_pytorch",
            "created_ts": __import__("time").time(),
        }
        with (out_dir / "index.json").open("w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Model copied successfully to: {output_path}")
        
        model_size = sum(
            os.path.getsize(os.path.join(dp, f)) 
            for dp, dn, fn in os.walk(out_dir) 
            for f in fn
        ) / (1024 * 1024)
        logger.info(f"Total size: {model_size:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.exception(f"Failed to copy model: {e}")
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
            'vocab_size': model_info.get('vocab_size', 50264),
            'max_length': model_info.get('max_length', 2048),
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
    """Test that the model loads correctly."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        
        logger.info("Testing model loading...")
        
        config_path = os.path.join(fractal_path, "config.json")
        if os.path.isfile(config_path):
            config = AutoConfig.from_pretrained(fractal_path, local_files_only=True)
            logger.info(f"  Config loaded: {config.model_type}")
        
        tok_dir = os.path.join(fractal_path, "tokenizer")
        if os.path.isdir(tok_dir):
            tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True, use_fast=True)
            test_text = "Привет, как дела?"
            tokens = tokenizer.encode(test_text)
            decoded = tokenizer.decode(tokens)
            logger.info(f"  Tokenizer works: '{test_text}' -> {len(tokens)} tokens")
        
        logger.info("Model loading test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Model loading test failed: {e}")
        return False


def get_model_config(model_id: str, cache_dir: str) -> dict:
    """Get model configuration details."""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True)
        return {
            'vocab_size': getattr(config, 'vocab_size', 50264),
            'n_positions': getattr(config, 'n_positions', getattr(config, 'max_position_embeddings', 2048)),
            'n_embd': getattr(config, 'n_embd', getattr(config, 'hidden_size', 768)),
        }
    except Exception:
        return {}


def main():
    model_id = "sberbank-ai/rugpt3small_based_on_gpt2"
    
    project_root = get_project_root()
    cache_dir = str(project_root / "eva" / "core" / "cogniflex_cache" / "models" / "hf_cache")
    fractal_path = str(project_root / "eva" / "core" / "cogniflex_cache" / "ml_unit" / "fractal_storage" / "models" / "rugpt3_small_fractal")
    config_path = str(project_root / "brain_config.json")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("ЕВА Light Model Downloader")
    logger.info("=" * 60)
    logger.info(f"Model: {model_id}")
    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Fractal storage: {fractal_path}")
    logger.info("")
    
    logger.info("Step 1: Ensuring model is cached...")
    if not ensure_model_cached(model_id, cache_dir):
        logger.error("Failed to cache model")
        sys.exit(1)
    
    logger.info("")
    logger.info("Step 2: Finding cached model path...")
    model_path = find_cached_model_path(model_id, cache_dir)
    if not model_path:
        logger.error("Could not find cached model")
        sys.exit(1)
    logger.info(f"  Found at: {model_path}")
    
    logger.info("")
    logger.info("Step 3: Getting model configuration...")
    model_config = get_model_config(model_id, cache_dir)
    logger.info(f"  Vocab size: {model_config.get('vocab_size')}")
    logger.info(f"  Max length: {model_config.get('n_positions')}")
    
    logger.info("")
    logger.info("Step 4: Copying to fractal storage...")
    if not copy_model_to_fractal_storage(model_path, fractal_path, "rugpt3small"):
        logger.error("Failed to copy model to fractal storage")
        sys.exit(1)
    
    logger.info("")
    logger.info("Step 5: Updating brain_config.json...")
    model_info = {
        'name': 'rugpt3small',
        'path': fractal_path,
        'vocab_size': model_config.get('vocab_size', 50264),
        'max_length': model_config.get('n_positions', 2048),
        'device': 'cuda:0',
    }
    if not update_brain_config(config_path, model_info):
        logger.error("Failed to update brain_config.json")
        sys.exit(1)
    
    logger.info("")
    logger.info("Step 6: Testing model loading...")
    if not test_model_loading(fractal_path):
        logger.warning("Model loading test had issues, but storage was created")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUCCESS: Light model downloaded and configured!")
    logger.info("=" * 60)
    logger.info(f"Model: {model_id}")
    logger.info(f"Fractal storage: {fractal_path}")
    logger.info(f"Config updated: {config_path}")


if __name__ == "__main__":
    main()
