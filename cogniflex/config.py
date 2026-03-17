# CogniFlex Configuration
# Configuration flags for disabling features

import os

# Model loading flags
DISABLE_ALL_MODELS = os.environ.get('COGNIFLEX_DISABLE_MODELS', 'false').lower() == 'true'
DISABLE_EMBEDDINGS = os.environ.get('COGNIFLEX_DISABLE_EMBEDDINGS', 'false').lower() == 'true'
DISABLE_TOKENIZERS = os.environ.get('COGNIFLEX_DISABLE_TOKENIZERS', 'false').lower() == 'true'
DISABLE_TRAINING = os.environ.get('COGNIFLEX_DISABLE_TRAINING', 'false').lower() == 'true'

# Feature flags
ENABLE_SELF_LEARNING = os.environ.get('COGNIFLEX_ENABLE_SELF_LEARNING', 'true').lower() == 'true'
ENABLE_NEUROMORPHIC = os.environ.get('COGNIFLEX_ENABLE_NEUROMORPHIC', 'false').lower() == 'true'

# To disable models, set environment variable:
# COGNIFLEX_DISABLE_MODELS=true
# Or use this file to toggle flags

def is_model_loading_disabled():
    """Check if model loading is disabled"""
    return DISABLE_ALL_MODELS

def is_embedding_loading_disabled():
    """Check if embedding model loading is disabled"""
    return DISABLE_ALL_MODELS or DISABLE_EMBEDDINGS

def is_tokenizer_loading_disabled():
    """Check if tokenizer loading is disabled"""
    return DISABLE_ALL_MODELS or DISABLE_TOKENIZERS
