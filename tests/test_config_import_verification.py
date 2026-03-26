"""
Config Import Verification Test
Run: python -m pytest tests/test_config_import_verification.py -v
"""
import pytest
from cogniflex.config import (
    is_model_allowed,
    is_model_loading_disabled,
    is_embedding_loading_disabled,
    is_tokenizer_loading_disabled,
    DISABLE_ALL_MODELS,
    DISABLE_EMBEDDINGS,
    DISABLE_TOKENIZERS,
    DISABLE_TRAINING,
    ALLOWED_MODELS,
    ENABLE_SELF_LEARNING,
    ENABLE_NEUROMORPHIC,
)


def test_imports_work():
    """Test that all config imports work correctly."""
    assert is_model_allowed is not None
    assert is_model_loading_disabled is not None
    assert is_embedding_loading_disabled is not None
    assert is_tokenizer_loading_disabled is not None
    assert isinstance(DISABLE_ALL_MODELS, bool)
    assert isinstance(DISABLE_EMBEDDINGS, bool)
    assert isinstance(DISABLE_TOKENIZERS, bool)
    assert isinstance(DISABLE_TRAINING, bool)
    assert isinstance(ALLOWED_MODELS, list)
    assert isinstance(ENABLE_SELF_LEARNING, bool)
    assert isinstance(ENABLE_NEUROMORPHIC, bool)


def test_is_model_allowed_rugpt3():
    """Test that RUGPT3 models are allowed."""
    assert is_model_allowed("rugpt3") is True
    assert is_model_allowed("rugpt3large") is True
    assert is_model_allowed("sberbank-ai/rugpt3large_based_on_gpt2") is True


def test_is_model_allowed_other():
    """Test that other models are blocked when DISABLE_ALL_MODELS is True."""
    assert is_model_allowed("gpt2") is False
    assert is_model_allowed("gpt-4") is False


def test_is_model_loading_disabled():
    """Test model loading disabled check."""
    assert is_model_loading_disabled() == DISABLE_ALL_MODELS


def test_is_embedding_loading_disabled():
    """Test embedding loading disabled check."""
    expected = DISABLE_ALL_MODELS or DISABLE_EMBEDDINGS
    assert is_embedding_loading_disabled() == expected


def test_is_tokenizer_loading_disabled():
    """Test tokenizer loading disabled check."""
    expected = DISABLE_ALL_MODELS or DISABLE_TOKENIZERS
    assert is_tokenizer_loading_disabled() == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
