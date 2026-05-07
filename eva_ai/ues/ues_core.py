"""
Universal Execution Environment (UES) - EVA.txt Section 8

UES provides a universal execution environment with Optuna-based hyperparameter tuning.
"""
import os
import json
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field


@dataclass
class ExecutionConfig:
    """Configuration for execution environment."""
    max_trials: int = 100
    timeout: Optional[int] = None
    study_name: str = "eva_optimization"
    storage: Optional[str] = None  # e.g., "sqlite:///optuna.db"
    direction: str = "minimize"  # or "maximize"
    
    
class UniversalExecutionEnvironment:
    """
    Universal Execution Environment with Optuna tuning.
    
    Features:
    - Hyperparameter optimization via Optuna
    - Execution environment management
    - Resource monitoring
    """
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.study = None
        self._best_params = {}
        self._execution_history = []
        
    def init_optuna(self, study_name: Optional[str] = None):
        """Initialize Optuna study."""
        try:
            import optuna
            
            study_name = study_name or self.config.study_name
            storage = self.config.storage
            
            self.study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction=self.config.direction,
                load_if_exists=True
            )
            return True
        except ImportError:
            print("[UES] Optuna not installed. Install with: pip install optuna")
            return False
        except Exception as e:
            print(f"[UES] Failed to init Optuna: {e}")
            return False
    
    def optimize(
        self,
        objective_fn: Callable,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None
    ):
        """
        Run Optuna optimization.
        
        Args:
            objective_fn: Function that takes (trial) and returns score
            n_trials: Number of trials (overrides config)
            timeout: Timeout in seconds (overrides config)
        """
        if not self.study:
            if not self.init_optuna():
                return None
        
        try:
            import optuna
            
            n_trials = n_trials or self.config.max_trials
            timeout = timeout or self.config.timeout
            
            self.study.optimize(
                objective_fn,
                n_trials=n_trials,
                timeout=timeout
            )
            
            self._best_params = self.study.best_params
            print(f"[UES] Best params: {self._best_params}")
            print(f"[UES] Best value: {self.study.best_value}")
            
            return self.study.best_params
        except Exception as e:
            print(f"[UES] Optimization failed: {e}")
            return None
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters from optimization."""
        return self._best_params.copy()
    
    def suggest_hyperparams(
        self,
        trial,
        param_space: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Suggest hyperparameters for Optuna trial.
        
        Args:
            trial: Optuna trial object
            param_space: Dict of {param_name: config} where config is:
                {"type": "float"/"int"/"categorical", "low": ..., "high": ..., "choices": [...]}
        
        Returns:
            Dict of suggested parameters
        """
        params = {}
        for name, config in param_space.items():
            ptype = config["type"]
            if ptype == "float":
                params[name] = trial.suggest_float(
                    name, config["low"], config["high"],
                    log=config.get("log", False)
                )
            elif ptype == "int":
                params[name] = trial.suggest_int(
                    name, config["low"], config["high"]
                )
            elif ptype == "categorical":
                params[name] = trial.suggest_categorical(
                    name, config["choices"]
                )
        return params
    
    def execute_with_monitoring(self, fn: Callable, *args, **kwargs):
        """
        Execute a function with resource monitoring.
        
        Returns:
            Tuple of (result, execution_time, resource_usage)
        """
        start_time = time.time()
        start_resources = self._get_resource_usage()
        
        try:
            result = fn(*args, **kwargs)
            success = True
        except Exception as e:
            result = e
            success = False
        
        end_time = time.time()
        end_resources = self._get_resource_usage()
        
        execution_time = end_time - start_time
        resource_diff = {
            k: end_resources.get(k, 0) - start_resources.get(k, 0)
            for k in set(list(start_resources.keys()) + list(end_resources.keys()))
        }
        
        execution_record = {
            "timestamp": time.time(),
            "execution_time": execution_time,
            "success": success,
            "resource_diff": resource_diff
        }
        self._execution_history.append(execution_record)
        
        if not success:
            raise result
        
        return result, execution_time, resource_diff
    
    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        usage = {}
        try:
            import psutil
            # CPU
            usage["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            # Memory
            mem = psutil.virtual_memory()
            usage["memory_percent"] = mem.percent
            usage["memory_used_mb"] = mem.used / (1024 * 1024)
            # Disk
            disk = psutil.disk_usage('/')
            usage["disk_percent"] = disk.percent
        except ImportError:
            pass
        return usage
    
    def save_study(self, path: str):
        """Save study results to file."""
        if not self.study:
            print("[UES] No study to save")
            return
        
        data = {
            "best_params": self.study.best_params,
            "best_value": self.study.best_value,
            "n_trials": len(self.study.trials),
            "study_name": self.study.study_name
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[UES] Study saved to {path}")
    
    def load_study(self, path: str):
        """Load study results from file."""
        if not os.path.exists(path):
            print(f"[UES] File not found: {path}")
            return False
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self._best_params = data.get("best_params", {})
        print(f"[UES] Loaded params: {self._best_params}")
        return True
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self._execution_history:
            return {}
        
        total_time = sum(r["execution_time"] for r in self._execution_history)
        avg_time = total_time / len(self._execution_history)
        success_rate = sum(1 for r in self._execution_history if r["success"]) / len(self._execution_history)
        
        return {
            "total_executions": len(self._execution_history),
            "total_time": total_time,
            "avg_execution_time": avg_time,
            "success_rate": success_rate
        }
