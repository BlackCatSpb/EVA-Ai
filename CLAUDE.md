# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working on code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application with GUI
python -m cogniflex.run
# or
python cogniflex/run.py

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
python -c "import sqlite3, json; db='cogniflex/core/cogniflex_cache/models/models.db'; conn=sqlite3.connect(db); cur=conn.cursor(); cur.execute('select id,name,model_path,model_type,priority from models order by priority desc'); print(json.dumps([dict(zip(['id','name','path','type','priority'],r)) for r in cur.fetchall()], ensure_ascii=False, indent=2))"
```

## Key Environment Variables

| Variable | Purpose |
|---|---|
| `TRANSFORMERS_OFFLINE=1` | Strict offline mode — no HF Hub downloads |
| `HF_HUB_OFFLINE=1` | Same as above, HF Hub level |
| `COGNIFLEX_DEFAULT_TEXT_GEN` | Override default text generation model alias (HF repo ID, absolute path, or subfolder name in `cogniflex_models/`) |
| `COGNIFLEX_DISABLE_MODELS` | Set to 'false' to enable all models (default: 'false') |

## System Status (March 2026)

### Active Model
**Qwen3.5-0.8b** - Primary local model (from brain_config.json)
- Path: `cogniflex/mlearning/cogniflex_models/qwen3.5-0.8b`
- Type: qwen
- max_length: 32768
- max_new_tokens: 2048

### Known Issues (Fixed)
1. ✅ Tokenizer loading with Windows paths - Fixed with fallback logic
2. ✅ GUI "ML недоступна" - Fixed with ml_unit availability check
3. ✅ FractalStorage parameter mismatch (storage_path vs storage_dir) - Fixed
4. ✅ Config import shadowing (config.py vs config/) - Fixed
5. ✅ max_new_tokens default values - Updated to 2048
6. ✅ max_length default values - Updated to 32768

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

`ModelManager` (`cogniflex/mlearning/model_manager.py`) stores model metadata in SQLite at `cogniflex/core/cogniflex_cache/models/models.db`. **Currently only Qwen3.5-0.8b is enabled** (see `brain_config.json`).

### Component Subsystems

- **`cogniflex/core/`** — CoreBrain, QueryProcessor, ResponseGenerator, EventSystem, BackgroundCoordinator, TokenProcessor, ResourceManager, ConfigManager, HybridCache, ComponentInitializer
- **`cogniflex/mlearning/`** — ModelManager, MLUnit, HybridModelManager, QwenModelManager, UnifiedTextProcessor, FractalModelManager
- **`cogniflex/memory/`** — MemoryManager, HybridTokenCache, MemoryGraph (NetworkX), FractalStore
- **`cogniflex/knowledge/`** — KnowledgeGraph, SemanticEngine (sentence-transformers), ConceptExtractor, QwenAPIEnhancer, WikipediaSearch
- **`cogniflex/ethics/`** — EthicsManager with 7 principle categories in `principles/`
- **`cogniflex/contradiction/`** — ContradictionManager, detection, resolution strategies
- **`cogniflex/learning/`** — SelfLearningSystem, SelfDialogLearningSystem, SelfAnalyzer, MemoryGraphTrainer
- **`cogniflex/gui/`** — tkinter GUI: chat, memory visualization, system dashboard, analytics
- **`cogniflex/websearch/`** — Multi-engine web search with SQLAlchemy result caching

### Event-Driven Communication

`EventSystem` (`cogniflex/core/event_system.py`) provides pub/sub coupling between all components. Key events include `text_processor_ready`, `model_ready`, metrics events, and state transitions.

### Configuration

`brain_config.json` in the repo root is the main system config. Key sections:
- `hybrid_cache` (VRAM/RAM/SSD tiers)
- `model` (active model: qwen3.5-0.8b)
- `generation` (sampling parameters: temperature 0.7, top_p 0.9, max_new_tokens 2048)
- `learning` (disabled by default)
- `system` (background threads, logging)

## Test Structure

Tests in `tests/` use pytest with fixtures from `conftest.py` (`temp_cache_dir`, `mock_logger`, `sample_text`). Notable test files:
- `test_e2e_chat_query.py` — end-to-end query pipeline
- `test_cogniflex_full_system.py` — full system integration
- `test_hybrid_cache_async.py` — cache behavior

## Performance Notes

Async tokenization + hybrid cache benchmarks: **9.54× average speedup** (34.59× on simple texts, ~1× on large contexts). Cache hit rate on complex scenarios is a known area for improvement.

## Git Structure

- **Main branch**: `C:/Users/black/OneDrive/Desktop/CogniFlex` - all active development
- **Worktrees**: Located in `C:/Users/black/.windsurf/worktrees/CogniFlex/` - older snapshots, not used for development