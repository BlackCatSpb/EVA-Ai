"""
Universal Execution Environment (UES) - EVA.txt Section 8

UES-100: Полная реализация универсальной执行环境.

Features:
- TopologyDiscoverer integration
- PGO-style autotuning with Optuna
- Resource monitoring (CPU, Memory, GPU)
- Double buffering for LoRA adapters
- QATTrainer integration for model quantization
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from threading import Lock

logger = logging.getLogger("eva_ai.ues")


@dataclass
class ExecutionConfig:
    """Configuration for execution environment."""
    max_trials: int = 100
    timeout: Optional[int] = None
    study_name: str = "eva_optimization"
    storage: Optional[str] = None
    direction: str = "minimize"


class UniversalExecutionEnvironment:
    """
    UES-100: Universal Execution Environment with full feature set.
    
    Integrates:
    - TopologyDiscoverer for hardware-aware optimization
    - PGOAutoTuner for hyperparameter optimization
    - ResourcePinner for CPU affinity
    - DoubleBufferPipeline for seamless LoRA switching
    - QATTrainer for model quantization
    """
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.study = None
        self._best_params = {}
        self._execution_history = []
        self._topology = None
        self._lora_double_buffer = None
        self._qat_trainer = None
        self._lock = Lock()
        
        self._init_components()
    
    def _init_components(self):
        """UES-100: Инициализация всех компонентов."""
        self._init_topology_discoverer()
        self._init_pgo_autotuner()
        self._init_resource_pinner()
        self._init_double_buffer()
        self._init_qat_trainer()
    
    def _init_topology_discoverer(self):
        """Инициализация TopologyDiscoverer."""
        try:
            from eva_ai.fcp_ues.topology import TopologyDiscoverer
            self._topology_discoverer = TopologyDiscoverer
        except ImportError:
            self._topology_discoverer = None
    
    def _init_pgo_autotuner(self):
        """Инициализация PGO AutoTuner."""
        try:
            from eva_ai.fcp_ues.pgo_autotuner import PGOAutoTuner
            self._pgo_autotuner = PGOAutoTuner()
        except ImportError:
            self._pgo_autotuner = None
    
    def _init_resource_pinner(self):
        """Инициализация ResourcePinner."""
        try:
            from eva_ai.fcp_ues.resource_pinner import ResourcePinner
            self._resource_pinner = ResourcePinner()
        except ImportError:
            self._resource_pinner = None
    
    def _init_double_buffer(self):
        """Инициализация DoubleBufferPipeline для LoRA."""
        try:
            from eva_ai.ues.double_buffer import DoubleBufferPipeline
            self._lora_double_buffer = DoubleBufferPipeline()
        except ImportError:
            self._lora_double_buffer = None
    
    def _init_qat_trainer(self):
        """UES-3: Инициализация QATTrainer."""
        try:
            from eva_ai.fcp_ues.qat_trainer import QATTrainer, QATConfig
            self._qat_trainer = QATTrainer(QATConfig(
                precision="int8",
                calibration_samples=100,
                finetune_epochs=3
            ))
            logger.info("QATTrainer initialized in UES")
        except ImportError:
            self._qat_trainer = None
    
    def discover_topology(self):
        """Discover hardware topology and cache the result."""
        if self._topology_discoverer is None:
            return self._discover_topology_fallback()
        
        try:
            self._topology = self._topology_discoverer.discover()
            return self._topology
        except Exception as e:
            logger.warning(f"Topology discovery failed: {e}")
            return self._discover_topology_fallback()
    
    def _discover_topology_fallback(self) -> Dict:
        """Fallback топология для систем без TopologyDiscoverer."""
        return {
            "units": [
                {"type": "CPU", "id": "cpu_0", "cores": os.cpu_count() or 4}
            ],
            "total_memory_gb": self._get_system_memory() // (1024**3) if hasattr(os, 'sysconf') else 8,
            "gpu_available": False
        }
    
    def _get_system_memory(self) -> int:
        """Получить объём памяти в байтах."""
        try:
            import psutil
            return psutil.virtual_memory().total
        except:
            return 8 * (1024**3)
    
    def get_topology(self):
        """Get cached topology or discover if not cached."""
        if self._topology is None:
            return self.discover_topology()
        return self._topology
    
    def get_optimal_device(self) -> str:
        """Get optimal device for execution based on topology."""
        topology = self.get_topology()
        if topology is None:
            return "CPU"
        
        for unit in topology.get("units", []):
            if unit.get("type") == "GPU":
                return unit.get("id", "GPU")
            elif unit.get("type") == "NPU":
                return unit.get("id", "NPU")
        
        return "CPU"
    
    def pin_to_cores(self, core_ids: List[int]):
        """UES-4: Привязка к конкретным ядрам CPU."""
        if self._resource_pinner:
            self._resource_pinner.pin_to_cores(core_ids)
    
    def get_pinned_cores(self) -> List[int]:
        """Получить текущие привязанные ядра."""
        if self._resource_pinner:
            return self._resource_pinner.get_pinned_cores()
        return list(range(os.cpu_count() or 4))
    
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
            logger.warning("Optuna not installed. Install with: pip install optuna")
            return False
        except Exception as e:
            logger.error(f"Failed to init Optuna: {e}")
            return False
    
    def optimize(
        self,
        objective_fn: Callable,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None
    ):
        """
        Run Optuna optimization or PGO auto-tuning.
        """
        if self._pgo_autotuner and hasattr(self._pgo_autotuner, 'optimize'):
            try:
                result = self._pgo_autotuner.optimize(objective_fn, n_trials, timeout)
                return result
            except:
                pass
        
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
            logger.info(f"Best params: {self._best_params}")
            logger.info(f"Best value: {self.study.best_value}")
            
            return self.study.best_params
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return None
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters from optimization."""
        return self._best_params.copy()
    
    def suggest_hyperparams(self, trial, param_space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Suggest hyperparameters for Optuna trial.
        """
        params = {}
        for name, config in param_space.items():
            ptype = config.get("type", "float")
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
    
    def quantize_model(self, model, calibration_data=None):
        """
        UES-3: Применить QAT к модели через интегрированный QATTrainer.
        """
        if self._qat_trainer is None:
            logger.warning("QATTrainer not available")
            return model
        
        return self._qat_trainer.quantize_model(model, calibration_data)
    
    def finetune_quantized(
        self,
        model,
        train_loader,
        epochs: int = 3
    ):
        """
        UES-3: Дообучить квантованную модель.
        """
        if self._qat_trainer is None:
            return model
        
        return self._qat_trainer.finetune(model, train_loader, epochs=epochs)
    
    def execute_with_monitoring(self, fn: Callable, *args, **kwargs):
        """
        Execute a function with resource monitoring.
        
        Returns:
            Tuple of (result, execution_time, resource_usage)
        """
        start_time = time.time()
        start_resources = self._get_resource_usage()
        
        with self._lock:
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
        
        if len(self._execution_history) > 1000:
            self._execution_history = self._execution_history[-500:]
        
        if not success:
            raise result
        
        return result, execution_time, resource_diff
    
    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        usage = {}
        try:
            import psutil
            usage["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            usage["memory_percent"] = mem.percent
            usage["memory_used_mb"] = mem.used / (1024 * 1024)
            
            try:
                disk = psutil.disk_usage('/')
                usage["disk_percent"] = disk.percent
            except:
                pass
            
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    usage["cpu_temp"] = temps.get('coretemp', [{}])[0].get('current', 0)
            except:
                pass
            
        except ImportError:
            pass
        
        return usage
    
    def get_execution_history(self, limit: int = 100) -> List[Dict]:
        """Получить историю выполнений."""
        return self._execution_history[-limit:]
    
    def get_resource_stats(self) -> Dict:
        """Статистика использования ресурсов."""
        if not self._execution_history:
            return {}
        
        successful = [e for e in self._execution_history if e.get("success")]
        
        if not successful:
            return {}
        
        avg_time = sum(e.get("execution_time", 0) for e in successful) / len(successful)
        
        return {
            "total_executions": len(self._execution_history),
            "successful": len(successful),
            "failed": len(self._execution_history) - len(successful),
            "avg_execution_time": avg_time,
            "current_resources": self._get_resource_usage()
        }
    
    def save_study(self, path: str):
        """Save study results to file."""
        if not self.study:
            logger.warning("No study to save")
            return
        
        data = {
            "best_params": self.study.best_params,
            "best_value": self.study.best_value,
            "n_trials": len(self.study.trials),
            "study_name": self.study.study_name
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_study(self, path: str):
        """Load study from file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self._best_params = data.get("best_params", {})
            return self._best_params
        except Exception as e:
            logger.error(f"Failed to load study: {e}")
            return {}
    
    def get_optimization_report(self) -> Dict:
        """Получить отчёт по оптимизации."""
        report = {
            "best_params": self._best_params,
            "execution_history_size": len(self._execution_history),
            "topology_cached": self._topology is not None,
            "qat_available": self._qat_trainer is not None,
            "double_buffer_ready": self._lora_double_buffer is not None
        }
        
        if self.study:
            report["optuna_study"] = {
                "name": self.study.study_name,
                "n_trials": len(self.study.trials),
                "best_value": self.study.best_value
            }
        
        return report
    
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
    
    def create_lora_double_buffer(self, adapter_paths: List[str]) -> "DoubleBufferPipeline":
        """
        Create double buffer pipeline for LoRA adapters.
        
        Args:
            adapter_paths: List of paths to LoRA adapter files
        
        Returns:
            DoubleBufferPipeline for seamless adapter switching
        """
        return DoubleBufferPipeline(adapter_paths)


class DoubleBufferPipeline:
    """
    Double buffering for LoRA adapters.
    
    Allows seamless switching between LoRA adapters without blocking:
    - Buffer A: currently in use
    - Buffer B: being prepared (preloaded)
    - Swap when ready
    """
    
    def __init__(self, adapter_paths: List[str]):
        self.adapter_paths = adapter_paths
        self._buffer_a = None
        self._buffer_b = None
        self._active_buffer = "A"
        self._loaded_paths = {"A": None, "B": None}
    
    def load_adapter(self, path: str, target_buffer: str = "B"):
        """Load adapter into specified buffer."""
        try:
            from safetensors import safe_open
            tensors = {}
            with safe_open(path, framework="pt") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
            
            if target_buffer == "A":
                self._buffer_a = tensors
            else:
                self._buffer_b = tensors
            
            self._loaded_paths[target_buffer] = path
            return True
        except Exception as e:
            print(f"[DoubleBuffer] Failed to load {path}: {e}")
            return False
    
    def preload_next(self, path: str):
        """Preload adapter into inactive buffer."""
        target = "B" if self._active_buffer == "A" else "A"
        return self.load_adapter(path, target)
    
    def swap(self) -> bool:
        """Swap active buffer (non-blocking)."""
        if self._active_buffer == "A" and self._buffer_b is not None:
            self._active_buffer = "B"
            return True
        elif self._active_buffer == "B" and self._buffer_a is not None:
            self._active_buffer = "A"
            return True
        return False
    
    def get_active_adapter(self) -> Optional[Dict]:
        """Get currently active adapter tensors."""
        if self._active_buffer == "A":
            return self._buffer_a
        return self._buffer_b
    
    def is_ready(self, buffer: str = None) -> bool:
        """Check if specified buffer is loaded."""
        if buffer is None:
            buffer = self._active_buffer
        return self._loaded_paths[buffer] is not None
