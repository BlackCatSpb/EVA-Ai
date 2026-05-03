import time
import numpy as np
from typing import Callable, Dict, Any

import logging
logger = logging.getLogger("FCP.UES")


class PGOAutoTuner:
    """
    Выполняет автоматический подбор параметров OpenVINO с помощью Optuna.
    Минимизирует медианную задержку инференса.
    """
    
    def __init__(self, benchmark_fn: Callable[[Dict[str, int]], float], n_trials: int = 50):
        """
        benchmark_fn: функция, принимающая словарь параметров (num_streams, num_threads)
                      и возвращающая среднюю задержку в миллисекундах.
        """
        self.benchmark_fn = benchmark_fn
        self.n_trials = n_trials
        self.best_params: Dict[str, int] = {}
        self.best_latency: float = float('inf')
    
    def tune(self) -> Dict[str, int]:
        try:
            import optuna
        except ImportError:
            logger.error("Optuna не установлена. Выполните: pip install optuna")
            return {"num_streams": 1, "num_threads": 4}
        
        def objective(trial):
            num_streams = trial.suggest_int("num_streams", 1, 8)
            num_threads = trial.suggest_int("num_threads", 1, 16)
            params = {"num_streams": num_streams, "num_threads": num_threads}
            latency = self.benchmark_fn(params)
            return latency
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = study.best_params
        self.best_latency = study.best_value
        logger.info(f"Auto-tune completed: {self.best_params}, latency={self.best_latency:.2f} ms")
        return self.best_params
    
    def get_optimal_config(self) -> Dict[str, Any]:
        if not self.best_params:
            return {"PERFORMANCE_HINT": "LATENCY"}
        return {
            "NUM_STREAMS": str(self.best_params.get("num_streams", 1)),
            "INFERENCE_NUM_THREADS": str(self.best_params.get("num_threads", 4)),
            "PERFORMANCE_HINT": "LATENCY",
            "ENABLE_CPU_PINNING": "YES"
        }
