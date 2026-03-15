import asyncio
import sys
import os

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cogniflex.mlearning.model_manager import ModelManager


async def main():
    prompt = "Напиши два коротких стиха про летний дождь."
    sampling = {"max_new_tokens": 64, "temperature": 0.8, "top_p": 0.95, "stream_interval": 1}
    cache = {
        "enable_prompt_cache": True,
        "enable_disk_prompt_cache": True,
        "enable_kv_cache": True,
        # Use project-local cache dir to avoid clutter
        "disk_cache_dir": os.path.join("token_cache", "disk_storage"),
        "disk_max_gb": 10.0,
    }

    m = ModelManager(autoload=False)
    print("[stream] model=default_text_gen, sampling=", sampling)
    sys.stdout.flush()

    try:
        async for chunk in m.generate_stream("default_text_gen", prompt, sampling=sampling, cache=cache):
            print(chunk, end="", flush=True)
    except KeyboardInterrupt:
        print("\n[stream] interrupted", flush=True)
    except Exception as e:
        print(f"\n[stream] error: {e}", flush=True)
    finally:
        print("\n[stream] done", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
