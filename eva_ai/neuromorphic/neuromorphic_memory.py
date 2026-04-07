"""NeuromorphicMemory — high-level memory interface backed by NeuromorphicSimulator."""
import logging
from typing import Dict, List, Optional, Any

from eva_ai.neuromorphic.neuromorphic_simulator import (
    NeuromorphicSimulator,
    NEST_AVAILABLE,
)

logger = logging.getLogger("eva_ai.neuromorphic_memory")


class NeuromorphicMemory:
    """
    High-level memory facade that wraps NeuromorphicSimulator.

    Provides a simple dict-like interface for storing/retrieving
    episodic activity backed by the fractal-aware neural simulator.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        brain=None,
        fractal_store=None,
    ) -> None:
        self._simulator = NeuromorphicSimulator(
            cache_dir=cache_dir,
            brain=brain,
            fractal_store=fractal_store,
        )
        logger.info("NeuromorphicMemory initialized (NEST=%s)", NEST_AVAILABLE)

    @property
    def simulator(self) -> NeuromorphicSimulator:
        return self._simulator

    def store(self, key: str, value: Any) -> bool:
        activity = self._simulator.simulate_activity(duration=1.0, memory_type="episodic")
        self._simulator.activity_history.append(activity)
        return True

    def retrieve(self, key: str, default: Any = None) -> Any:
        for act in reversed(self._simulator.activity_history):
            if key in act.context:
                return act.context[key]
        return default

    def analyze(self) -> Dict[str, Any]:
        return self._simulator.analyze_neural_activity()

    def get_health(self) -> Dict[str, Any]:
        return self._simulator.get_system_health()

    def reset(self) -> None:
        self._simulator.reset_simulation()

    def start(self) -> None:
        self._simulator.start()

    def stop(self) -> None:
        self._simulator.stop()
