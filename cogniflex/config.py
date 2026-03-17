# CogniFlex Configuration
# Configuration flags for disabling features

import os

# Model loading flags
# Set to True to disable all models except RUGPT3
DISABLE_ALL_MODELS = os.environ.get('COGNIFLEX_DISABLE_MODELS', 'true').lower() == 'true'
DISABLE_EMBEDDINGS = os.environ.get('COGNIFLEX_DISABLE_EMBEDDINGS', 'false').lower() == 'true'  # Включено для работы MemoryGraphML
DISABLE_TOKENIZERS = os.environ.get('COGNIFLEX_DISABLE_TOKENIZERS', 'false').lower() == 'true'
DISABLE_TRAINING = os.environ.get('COGNIFLEX_DISABLE_TRAINING', 'false').lower() == 'true'

# Allowed models (only RUGPT3 by default - others disabled)
ALLOWED_MODELS = ['rugpt3', 'rugpt3large', 'sberbank-ai/rugpt3large_based_on_gpt2']

# Feature flags
ENABLE_SELF_LEARNING = os.environ.get('COGNIFLEX_ENABLE_SELF_LEARNING', 'false').lower() == 'true'
ENABLE_NEUROMORPHIC = os.environ.get('COGNIFLEX_ENABLE_NEUROMORPHIC', 'false').lower() == 'true'

def is_model_allowed(model_name: str) -> bool:
    """Check if model is allowed to load"""
    if not DISABLE_ALL_MODELS:
        return True
    model_lower = model_name.lower() if model_name else ''
    for allowed in ALLOWED_MODELS:
        if allowed.lower() in model_lower:
            return True
    return False

def is_model_loading_disabled():
    """Check if model loading is disabled"""
    return DISABLE_ALL_MODELS

def is_embedding_loading_disabled():
    """Check if embedding model loading is disabled"""
    return DISABLE_ALL_MODELS or DISABLE_EMBEDDINGS

def is_tokenizer_loading_disabled():
    """Check if tokenizer loading is disabled"""
    return DISABLE_ALL_MODELS or DISABLE_TOKENIZERS
