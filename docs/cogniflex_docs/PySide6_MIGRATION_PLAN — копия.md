# PySide6 Migration Plan (CogniFlex)

## Goals
- Provide a responsive desktop dashboard to monitor and control GlobalResourceQueue (GRQ) and subsystems.
- Decouple UI from core logic; GUI acts as a thin client over `CoreBrain` APIs.
- Enable packaging to a single Windows executable for users without Python.

## Scope (Phase 1: PoC)
- One-window dashboard with dockable panels:
  - Metrics (RAM/CPU/IO) with live charts.
  - Queues table (waiting requests with priority/age).
  - Controls (simulate load, reset stats, priority selection).
- Data pulled from `CoreBrain` at 200–500 ms intervals.

## Architecture
- UI Project: `tools/gui_pyside6/`
  - `main.py` — app bootstrap, `QMainWindow`, theme.
  - `data_provider.py` — wrapper over `CoreBrain` with `QTimer` polling.
  - `views/metrics_view.py` — charts (PyQtGraph or QtCharts).
  - `views/queues_view.py` — tables for RAM/CPU queues.
  - `views/controls_view.py` — buttons, priority selectors.
  - `widgets/` — reusable components.
  - `theme/` — qss (light/dark).
- Threading
  - UI stays in main thread.
  - Polling via `QTimer`. Heavy ops (if appear) moved to `QRunnable` + `QThreadPool`.

## CoreBrain API (additions)
File: `cogniflex/core/core_brain.py`
- `def get_resource_queue_snapshot(self) -> dict:`
  - Returns aggregated metrics from GRQ: totals, per-queue items, rates.
- `def reset_resource_stats(self) -> None:` (optional)
  - Resets GRQ statistics.
- `def simulate_load(self, kind: str, duration_s: float = 1.0, intensity: float = 1.0, priority: int = 0) -> None:` (optional)
  - Generates short, safe synthetic CPU/RAM/IO load for demo.

## Snapshot Schema (draft)
```json
{
  "timestamp": 1724000000,
  "ram": {"reserved_bytes": 0, "limit_bytes": 0, "available_bytes": 0},
  "cpu": {"available_tokens": 0, "limit_tokens": 0},
  "io": {"rate_current": 0.0, "rate_limit": 0.0, "bucket_fill": 0.0},
  "queues": {
    "ram": [{"id": "req-1", "priority": 100, "age_ms": 120, "size_bytes": 4096}],
    "cpu": [{"id": "req-2", "priority": 50, "age_ms": 45, "tokens": 2}],
    "io":  [{"id": "req-3", "priority": 20, "age_ms": 60, "bytes": 8192}]
  },
  "aging": {"enabled": true, "policy": "linear", "max_boost": 50}
}
```

## UI Update Loop
- `QTimer(interval=300 ms)` calls `DataProvider.fetch_snapshot()`.
- Emitted signal updates charts and tables.
- Controls call `CoreBrain` methods (simulate, reset, priority set).

## Technology Choices
- Python 3.11+ (as in project).
- PySide6 (Qt6).
- Charts: PyQtGraph for performance (fallback to QtCharts if preferred look).
- Packaging: PyInstaller.

## Packaging (Windows)
- PyInstaller `.spec` that includes Qt plugins:
  - `platforms`, `styles`, `imageformats` (ensure ANGLE/OpenGL as needed).
- Smoke test on clean Windows VM (no Python installed).

## Milestones & Effort
1. CoreBrain snapshot API — 0.5 day
2. PoC UI (metrics + queues + controls) — 1 day
3. Integrate simulate/reset actions — 0.5 day
4. Packaging and smoke test — 0.5–1 day
Total: ~2–3 days for PoC + 1 day packaging.

## Acceptance Criteria (PoC)
- Realtime charts (RAM/CPU/IO) refresh smoothly without UI freezes.
- Queues table shows items with priority and age (aging visible under load).
- Controls operate without blocking; simulate load affects metrics.
- Standalone EXE launches and works on a clean Windows VM.

## Risks & Mitigations
- UI freeze due to blocking calls → use `QTimer`/`QThreadPool`.
- Rendering performance → prefer PyQtGraph, reduce refresh rate to 300–500 ms.
- Qt plugin issues in packaging → curated `.spec` and checklist.
- OpenGL problems → force ANGLE/Software rendering via env or Qt args if needed.

## Next Steps
- Implement `get_resource_queue_snapshot()` in `CoreBrain`.
- Scaffold `tools/gui_pyside6/` with `main.py` and placeholder views.
- Hook DataProvider polling and render mocked charts.
- Add simulate/reset controls and connect to `CoreBrain`.
- Prepare PyInstaller spec and run smoke test.
