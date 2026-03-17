"""
Tests for the unified ModelManager implementation
"""

import os
import pytest
from pathlib import Path
import torch
from cogniflex.mlearning.model_manager import ModelManager
from cogniflex.mlearning.storage.unified_graph_store import UnifiedMemoryGraph

@pytest.fixture
def model_manager():
    """Create a model manager instance for testing"""
    return ModelManager(
        cache_dir="test_cache",
        model_dir="test_models",
        use_gpu=False,
        safe_test_mode=True
    )

def test_model_manager_init(model_manager):
    """Test basic initialization"""
    assert model_manager is not None
    assert model_manager.safe_test_mode is True
    assert isinstance(model_manager.storage, UnifiedMemoryGraph)

def test_model_loading(model_manager):
    """Test model loading functionality"""
    # Create a simple test model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 2)
            
        def forward(self, x):
            return self.linear(x)
    
    test_model = SimpleModel()
    try:
        # Register the model first
        model_manager.register_model(
            "simple_model",
            test_model,
            model_type="test"
        )
        
        # Test loading states
        assert "simple_model" not in model_manager.model_states
        
        # Test loading
        loaded_model = model_manager.load_model("simple_model")
        assert loaded_model is not None
        assert model_manager.model_states["simple_model"] == "loaded"
        
        # Test model functionality
        test_input = torch.randn(1, 10)
        output = loaded_model(test_input)
        assert output.shape == (1, 2)
        
        # Test error handling for non-existent model
        with pytest.raises(RuntimeError) as excinfo:
            model_manager.load_model("nonexistent_model")
        assert "Модель не найдена" in str(excinfo.value)
        
        # Test unloading
        model_manager.unload_model("simple_model")
        assert "simple_model" not in model_manager.loaded_models
        assert "simple_model" not in model_manager.model_states
        assert "simple_model" not in model_manager.loading_errors
    finally:
        # Cleanup
        if "simple_model" in model_manager.loaded_models:
            model_manager.unload_model("simple_model")

def test_model_graph_storage(model_manager):
    """Test unified graph storage integration"""
    test_data = {"test_key": "test_value"}
    
    # Store data
    model_manager.storage.store("test_node", test_data)
    
    # Retrieve data
    retrieved = model_manager.storage.get("test_node")
    assert retrieved == test_data
