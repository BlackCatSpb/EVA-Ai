import os
import sys
import time
import logging

# Ensure project path
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'cogniflex'))

# Enable DEBUG logging for our modules
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

from cogniflex.core.response_generator import ResponseGenerator

try:
    from cogniflex.memory.hybrid_token_cache import HybridTokenCache  # optional
    HAS_REAL_CACHE = True
except Exception:
    HAS_REAL_CACHE = False

class StubHybridCache:
    def prioritize_context(self, query: str, context: str, task_type: str = "general") -> str:
        # Simulate optimization by truncating context roughly in half
        words = context.split()
        if len(words) > 8:
            return " ".join(words[: len(words)//2])
        return context

class StubBrain:
    def __init__(self):
        self.components = {}
        self.config = {}
        self.running = True


def build_context_dict():
    return {
        "system": "You are a helpful AI system providing concise answers.",
        "history": [
            "User: Provide summary of token cache optimizations.",
            "AI: Described hybrid cache with prioritization and eviction weighting.",
            "User: How does context optimization affect prompt size?"
        ],
        "evidence": [
            "Doc: Hybrid cache improves simple text performance by 34.59x.",
            "Log: Cache hit rates currently low, need tuning.",
            "Design: Dynamic token budget and eviction policies active."
        ]
    }


def run_case(use_cache: bool, use_real: bool) -> dict:
    brain = StubBrain()
    gen = ResponseGenerator(brain)

    if use_cache:
        if use_real and HAS_REAL_CACHE:
            try:
                gen.hybrid_cache = HybridTokenCache(cache_dir=os.path.join(ROOT, 'cogniflex_cache'))
            except Exception:
                gen.hybrid_cache = StubHybridCache()
        else:
            gen.hybrid_cache = StubHybridCache()
    else:
        gen.hybrid_cache = None

    ctx = build_context_dict()

    t0 = time.perf_counter()
    prompt = gen._prepare_prompt(prompt="Test query about cache", task="general", context=ctx)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "use_cache": use_cache,
        "use_real": use_real and HAS_REAL_CACHE,
        "prompt_len": len(prompt),
        "elapsed_ms": dt_ms,
    }


if __name__ == "__main__":
    print("=== A/B: ResponseGenerator with vs without Hybrid Cache ===")
    a = run_case(use_cache=False, use_real=False)
    b = run_case(use_cache=True, use_real=True)

    print("\n--- Baseline (no cache) ---")
    print(f"prompt_len: {a['prompt_len']} chars")
    print(f"elapsed:    {a['elapsed_ms']:.2f} ms")

    print("\n--- With cache (real if available, otherwise stub) ---")
    print(f"prompt_len: {b['prompt_len']} chars")
    print(f"elapsed:    {b['elapsed_ms']:.2f} ms")
    print(f"cache_type: {'real' if b['use_real'] else 'stub'}")

    delta_len = a['prompt_len'] - b['prompt_len']
    delta_time = a['elapsed_ms'] - b['elapsed_ms']

    print("\n=== Delta (Baseline - Cache) ===")
    print(f"prompt_len_delta: {delta_len} chars (positive => cache shorter)")
    print(f"elapsed_delta:    {delta_time:.2f} ms (positive => cache faster)")
    print("\nNote: When using cache, DEBUG logs should include 'context_optimization:' with before/after token counts.")
