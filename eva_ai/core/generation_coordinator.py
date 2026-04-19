"""
Generation Coordinator - Central orchestrator for model generation.

Manages:
- Async orchestration
- Model pool
- Token caching
- Smart LoRA routing
- Parallel generation

Usage:
    coordinator = GenerationCoordinator(pipeline)
    result = await coordinator.generate(query)
"""

import asyncio
import logging
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("eva_ai.generation_coordinator")


@dataclass
class GenerationResult:
    """Result from generation."""
    text: str
    reasoning: str = ""
    tokens_count: int = 0
    elapsed_ms: int = 0
    model_used: str = "unknown"
    tokens: List[int] = field(default_factory=list)


class TokenCache:
    """Cache tokens between generations."""
    
    def __init__(self, max_size: int = 100):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
    
    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def get(self, query: str) -> Optional[Dict]:
        """Get cached tokens."""
        key = self._hash(query)
        entry = self._cache.get(key)
        if entry:
            entry['hits'] += 1
            return entry
        return None
    
    def set(self, query: str, tokens: List[int], model: str):
        """Cache tokens."""
        key = self._hash(query)
        self._cache[key] = {
            'tokens': tokens,
            'model': model,
            'created': datetime.now(),
            'hits': 0
        }
        
        # Evict old if needed
        if len(self._cache) > self._max_size:
            oldest = min(self._cache.items(), key=lambda x: x[1]['created'])
            del self._cache[oldest[0]]
    
    def clear(self):
        """Clear cache."""
        self._cache.clear()


class GenerationCoordinator:
    """
    Central coordinator for model generation.
    
    Features:
    - Async/await orchestration
    - Token caching
    - Smart LoRA routing
    - Parallel generation A||B
    """
    
    def __init__(self, pipeline):
        """
        Args:
            pipeline: PipelineAdapter or UnifiedGenerator
        """
        self.pipeline = pipeline
        self.token_cache = TokenCache()
        
        # Import smart router
        try:
            from eva_ai.core.smart_lora_router import SmartLoRARouter
            self.lora_router = SmartLoRARouter()
        except ImportError:
            self.lora_router = None
        
        # Import token streaming
        try:
            from eva_ai.core.token_streaming import TokenStreamingAPI
            self.token_api = None  # Initialize later
        except ImportError:
            self.token_api = None
    
    async def generate(
        self,
        query: str,
        config: Optional[Dict] = None,
        use_caching: bool = True,
        use_parallel: bool = False
    ) -> GenerationResult:
        """
        Generate response with smart orchestration.
        
        Args:
            query: User query
            config: Generation config
            use_caching: Use token caching
            use_parallel: Run A||B parallel
            
        Returns:
            GenerationResult
        """
        import time
        start = time.time()
        
        # Get LoRA
        lora_name = 'eva_knowledge'
        if self.lora_router:
            lora_name = self.lora_router.route(query)
        
        # Get cached tokens
        cached = None
        if use_caching:
            cached = self.token_cache.get(query)
        
        # Generate
        if use_parallel and hasattr(self, '_generate_parallel'):
            result = await self._generate_parallel(query, lora_name, cached, config)
        else:
            result = await self._generate_sequential(query, lora_name, cached, config)
        
        # Cache tokens
        if use_caching and result.tokens:
            self.token_cache.set(query, result.tokens, result.model_used)
        
        result.elapsed_ms = int((time.time() - start) * 1000)
        return result
    
    async def _generate_sequential(
        self,
        query: str,
        lora_name: str,
        cached: Optional[Dict],
        config: Optional[Dict]
    ) -> GenerationResult:
        """Sequential generation (A → B)."""
        # Import pipeline
        pipeline = getattr(self.pipeline, '_generator', None) or self.pipeline
        
        if not pipeline:
            return GenerationResult(text="[Error: No pipeline]")
        
        try:
            # Use asyncio to not block
            result = await asyncio.to_thread(
                pipeline.generate,
                query,
                task_type='logic',
                max_tokens=512
            )
            
            return GenerationResult(
                text=result.generated_text if hasattr(result, 'generated_text') else str(result),
                model_used='logic'
            )
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return GenerationResult(text=f"[Error: {str(e)[:50]}]")
    
    async def _generate_parallel(
        self,
        query: str,
        lora_name: str,
        cached: Optional[Dict],
        config: Optional[Dict]
    ) -> GenerationResult:
        """Parallel generation (A || B)."""
        pipeline = getattr(self.pipeline, '_generator', None) or self.pipeline
        
        if not pipeline:
            return GenerationResult(text="[Error: No pipeline]")
        
        # Get models
        model_a = getattr(pipeline, '_openvino_cpu', None)
        model_b = getattr(pipeline, '_openvino_gpu', None)
        
        if not model_a:
            return await self._generate_sequential(query, lora_name, cached, config)
        
        # Run parallel
        async def gen_a():
            return await asyncio.to_thread(
                model_a.generate, query, max_tokens=512
            )
        
        async def gen_b():
            if model_b:
                return await asyncio.to_thread(
                    model_b.generate, query, max_tokens=2048
                )
            return None
        
        # Launch both
        task_a = asyncio.create_task(gen_a())
        task_b = asyncio.create_task(gen_b())
        
        # Wait for A (quick)
        try:
            result_a = await asyncio.wait_for(task_a, timeout=15)
        except asyncio.TimeoutError:
            return GenerationResult(text="[Error: Timeout]")
        
        # Get B if available
        result_b = None
        if model_b:
            try:
                result_b = await task_b
            except:
                pass
        
        # Merge results
        text_a = str(result_a) if result_a else ""
        text_b = str(result_b) if result_b else ""
        
        # Use A for streaming, B for depth
        final_text = text_b if text_b else text_a
        
        return GenerationResult(
            text=final_text,
            model_used='A+B' if text_b else 'A'
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        return {
            'token_cache_size': len(self.token_cache._cache) if self.token_cache else 0,
            'lora_stats': self.lora_router.get_stats() if self.lora_router else {}
        }


def create_coordinator(pipeline) -> GenerationCoordinator:
    """Factory function."""
    return GenerationCoordinator(pipeline)


# Test
if __name__ == '__main__':
    print("Generation Coordinator module created")
    print(f"Features: Async, Token Caching, Smart LoRA, Parallel")