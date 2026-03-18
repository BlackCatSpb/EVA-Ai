# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application with GUI
python cogniflex/run.py
# or
python run_gui.py

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_e2e_chat_query.py

# Run tests with coverage
pytest tests/ --cov=cogniflex --cov-report=html

# Quick smoke test (no model weights needed)
python minimal_test.py

# Initialize ModelManager without downloading (offline/safe mode)
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python -c \
  "from cogniflex.mlearning.model_manager import ModelManager; \
   mm=ModelManager(brain=None,use_gpu=False,max_workers=1,autoload=False); print('OK')"

# Inspect models database
python -c "import sqlite3, json; db='core/cogniflex_cache/models/models.db'; conn=sqlite3.connect(db); cur=conn.cursor(); cur.execute('select id,name,model_path,model_type,priority from models order by priority desc'); print(json.dumps([dict(zip(['id','name','path','type','priority'],r)) for r in cur.fetchall()], ensure_ascii=False, indent=2))"
```

## Key Environment Variables

| Variable | Purpose |
|---|---|
| `TRANSFORMERS_OFFLINE=1` | Strict offline mode — no HF Hub downloads |
| `HF_HUB_OFFLINE=1` | Same as above, HF Hub level |
| `COGNIFLEX_DEFAULT_TEXT_GEN` | Override default text generation model alias (HF repo ID, absolute path, or subfolder name in `cogniflex_models/`) |
| `COGNIFLEX_FORCE_MODEL_REFRESH=1` | Force re-download of ruGPT3 snapshot even if local copy exists |
| `COGNIFLEX_RUGPT3_LOCAL_NAME` | Override local subfolder name for ruGPT3 (default: `rugpt3_large`) |

## Architecture

### System Initialization Flow

`CoreBrain` (`cogniflex/core/core_brain.py`) is the central coordinator. At startup it progresses through states: `INITIALIZING → LOADING_MODELS → INITIALIZING_COMPONENTS → READY`. The `ComponentInitializer` manages dependency-ordered startup of all subsystems.

### Query Processing Pipeline (8 stages)

User input flows through `QueryProcessor` (`cogniflex/core/query_processor.py`):
1. Query reception
2. Attention focus (fractal hot window)
3. Upward processing (abstraction levels 0→4)
4. Horizontal connectivity (knowledge graph)
5. Downward processing (refinement)
6. Contradiction check (`ContradictionManager`)
7. Ethics validation (`EthicsManager`, 7 principles)
8. Response generation (`ResponseGenerator`)

### Fractal Knowledge Representation

Knowledge is structured at 5 abstraction levels (0: tokens → 4: concepts) with a "hot window" split: 40% core context, 40% extended context, 20% details. The hybrid cache mirrors this: VRAM (hot window) → RAM buffer → SSD persistent cache.

### Model Management

`ModelManager` (`cogniflex/mlearning/model_manager.py`) stores model metadata in SQLite at `cogniflex/core/cogniflex_cache/models/models.db`. At first run it provisions ruGPT3 (`sberbank-ai/rugpt3large_based_on_gpt2`) into `cogniflex/mlearning/cogniflex_models/rugpt3_large/`. A local copy is considered valid if it contains `config.json` + weights (`pytorch_model.bin` or `model.safetensors`) + tokenizer files. **Currently only ruGPT3 is enabled** (see `brain_config.json` and recent commits).

### Component Subsystems

- **`cogniflex/core/`** — CoreBrain, QueryProcessor, ResponseGenerator, EventSystem, BackgroundCoordinator, TokenProcessor, ResourceManager, ConfigManager, HybridCache
- **`cogniflex/mlearning/`** — ModelManager, ML inference/training pipeline, custom tokenizer
- **`cogniflex/memory/`** — MemoryManager, HybridTokenCache, MemoryGraph (NetworkX), FractalStore
- **`cogniflex/knowledge/`** — KnowledgeGraph, SemanticEngine (sentence-transformers), ConceptExtractor
- **`cogniflex/ethics/`** — EthicsManager with 7 principle categories in `principles/`
- **`cogniflex/contradiction/`** — ContradictionManager, detection, resolution strategies
- **`cogniflex/learning/`** — Background learning jobs (disabled by default in `brain_config.json`)
- **`cogniflex/gui/`** — tkinter GUI: chat, memory visualization, system dashboard
- **`cogniflex/websearch/`** — Multi-engine web search with SQLAlchemy result caching

### Event-Driven Communication

`EventSystem` (`cogniflex/core/event_system.py`) provides pub/sub coupling between all components. Key events include `text_processor_ready` (triggers `UnifiedTextProcessor` registration in ModelManager), metrics events, and state transitions.

### Configuration

`brain_config.json` in the repo root is the main system config. **Note: this file currently has a git merge conflict** — resolve before using. Key sections: `hybrid_cache` (VRAM/RAM/SSD tiers), `model` (active model path), `generation` (sampling parameters), `learning` (disabled by default), `system` (background threads disabled).

### Test Structure

Tests in `tests/` use pytest with fixtures from `conftest.py` (`temp_cache_dir`, `mock_logger`, `sample_text`). Notable test files:
- `test_e2e_chat_query.py` — end-to-end query pipeline
- `test_cogniflex_full_system.py` — full system integration
- `test_hybrid_cache_async.py` — cache behavior
- `comprehensive_tests.py` — component-level
- `test_autopilot_system_health.py` — health monitoring

### Performance Notes

Async tokenization + hybrid cache benchmarks: **9.54× average speedup** (34.59× on simple texts, ~1× on large contexts). Cache hit rate on complex scenarios is a known area for improvement.
