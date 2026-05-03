import os
import platform
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import logging
logger = logging.getLogger("FCP.UES")


@dataclass
class ComputeUnit:
    id: str
    type: str          # "CPU", "GPU", "NPU"
    vendor: str        # "Intel", "AMD", "ARM"
    cores: int
    threads: int
    vector_extensions: List[str] = field(default_factory=list)
    cache_sizes: Dict[str, int] = field(default_factory=dict)  # L1/L2/L3 в байтах
    frequency_mhz: float = 0.0
    memory_bandwidth_gb_s: float = 0.0


@dataclass
class ComputeTopology:
    units: List[ComputeUnit]
    total_memory_gb: float
    numa_nodes: int


class TopologyDiscoverer:
    """Зондирует аппаратную платформу и строит модель ресурсов."""
    
    @staticmethod
    def discover() -> ComputeTopology:
        units = []
        
        # --- CPU ---
        cpu_info = TopologyDiscoverer._probe_cpu()
        if cpu_info:
            units.append(cpu_info)
        
        # --- GPU (через OpenVINO) ---
        try:
            import openvino as ov
            core = ov.Core()
            available = core.available_devices
            for dev in available:
                if "GPU" in dev:
                    units.append(ComputeUnit(
                        id=dev, type="GPU", vendor="Intel",
                        cores=0, threads=0
                    ))
        except Exception:
            pass
        
        # --- NPU ---
        if os.path.exists("/dev/accel") or os.path.exists("/dev/dri/renderD128"):
            units.append(ComputeUnit(
                id="NPU", type="NPU", vendor="Intel",
                cores=0, threads=0
            ))
        
        # Общая память
        try:
            import psutil
            total_memory = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            total_memory = 0.0
        
        # NUMA-узлы
        numa_nodes = 1
        if os.path.exists("/sys/devices/system/node"):
            numa_nodes = len([d for d in os.listdir("/sys/devices/system/node") if d.startswith("node")])
        
        return ComputeTopology(units=units, total_memory_gb=total_memory, numa_nodes=numa_nodes)
    
    @staticmethod
    def _probe_cpu() -> Optional[ComputeUnit]:
        system = platform.system()
        cores = os.cpu_count() or 1
        threads = cores  # упрощение
        
        # Векторные расширения
        vec = []
        if system == "Windows":
            try:
                out = subprocess.check_output("wmic cpu get caption", shell=True).decode()
                if "AVX512" in out: vec.append("AVX512")
                if "AVX2" in out: vec.append("AVX2")
            except Exception:
                pass
        else:
            try:
                with open("/proc/cpuinfo") as f:
                    flags = f.read()
                if "avx512f" in flags: vec.append("AVX512")
                if "avx2" in flags: vec.append("AVX2")
                if "neon" in flags: vec.append("NEON")
            except Exception:
                pass
        
        return ComputeUnit(
            id="CPU", type="CPU", vendor="Intel",
            cores=cores, threads=threads, vector_extensions=vec
        )
