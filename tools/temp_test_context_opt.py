import os
import sys
import logging

# Ensure project path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cogniflex'))

# Enable DEBUG logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

from cogniflex.core.response_generator import ResponseGenerator

class StubHybridCache:
    def prioritize_context(self, query: str, context: str, task_type: str = "general") -> str:
        # Simulate optimization by truncating context roughly in half
        words = context.split()
        if len(words) > 4:
            return " ".join(words[: len(words)//2])
        return context

class StubBrain:
    def __init__(self):
        self.components = {}
        self.config = {}
        self.running = True

if __name__ == "__main__":
    brain = StubBrain()
    gen = ResponseGenerator(brain)
    # Inject stub hybrid cache to ensure the code path executes
    gen.hybrid_cache = StubHybridCache()

    context = {
        "system": "You are a helpful AI system providing concise answers.",
        "history": [
            "User: Provide summary of token cache optimizations.",
            "AI: Described hybrid cache with prioritization and eviction weighting."
        ],
        "evidence": [
            "Doc: Hybrid cache improves simple text performance by 34.59x.",
            "Log: Cache hit rates currently low, need tuning."
        ]
    }

    prompt = gen._prepare_prompt(prompt="Test query about cache", task="general", context=context)
    print("Prepared prompt length:", len(prompt))
