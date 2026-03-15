import asyncio
import sys
import os

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cogniflex.mlearning.model_manager import ModelManager


async def main():
    m = ModelManager(autoload=False)
    # Register tiny model to avoid large downloads
    m.register_model("tiny_test", "gpt2", "gpt2", priority=1, name="tiny gpt2")

    prompt = "Проверь потоковую генерацию с tiny моделью."
    sampling = {"max_new_tokens": 16, "temperature": 0.9, "top_p": 0.95, "stream_interval": 1}
    cache = {"enable_prompt_cache": True, "enable_kv_cache": True}

    print("[stream] tiny_test start", flush=True)
    chunks = []
    async for chunk in m.generate_stream("tiny_test", prompt, sampling=sampling, cache=cache):
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    print("\n[done]", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
