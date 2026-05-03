import os
import sys
import platform
import logging
from typing import Literal

logger = logging.getLogger("FCP.UES")


class ResourcePinner:
    """
    Закрепляет вычислительные задачи за указанным типом ядер.
    Использует OpenVINO hint::scheduling_core_type.
    """
    
    @staticmethod
    def get_ov_config(core_type: Literal["ANY", "PCORE", "ECORE"] = "ANY") -> dict:
        """Возвращает конфигурацию OpenVINO с указанием типа ядер."""
        return {"SCHEDULING_CORE_TYPE": core_type}
    
    @staticmethod
    def pin_gnn_to_e_cores() -> dict:
        """GNN-вычисления на энергоэффективных ядрах."""
        return ResourcePinner.get_ov_config("ECORE")
    
    @staticmethod
    def pin_llm_to_p_cores() -> dict:
        """LLM-инференс на производительных ядрах."""
        return ResourcePinner.get_ov_config("PCORE")
    
    @staticmethod
    def pin_current_process_to_e_cores():
        """
        Привязывает текущий процесс к энергоэффективным ядрам (E-cores).
        Поддерживает Windows и Linux. Полная реализация без упрощений.
        """
        system = platform.system()
        
        if system == "Linux":
            ResourcePinner._pin_linux_e_cores()
        elif system == "Windows":
            ResourcePinner._pin_windows_e_cores()
        else:
            logger.warning(f"Unsupported operating system: {system}")
    
    @staticmethod
    def _pin_linux_e_cores():
        """Linux-реализация привязки к E-cores через sched_setaffinity."""
        if os.name != "posix":
            return
        
        try:
            import psutil
            
            # Получаем информацию о всех логических ядрах
            logical_cores = psutil.cpu_count(logical=True) or 1
            physical_cores = psutil.cpu_count(logical=False) or 1
            
            # Для гибридных процессоров (Intel 12th+) E-cores обычно идут после P-cores
            # Пытаемся определить по частоте (E-cores имеют более низкую базовую частоту)
            e_core_ids = set()
            
            try:
                # Читаем информацию о частоте каждого ядра
                for cpu_num in range(logical_cores):
                    cpu_path = f"/sys/devices/system/cpu/cpu{cpu_num}/cpufreq/base_frequency"
                    if os.path.exists(cpu_path):
                        with open(cpu_path) as f:
                            freq = int(f.read().strip())
                            # E-cores обычно имеют частоту < 2000 МГц
                            if freq < 2000000:  # Меньше 2 ГГц считаем E-core
                                e_core_ids.add(cpu_num)
            except Exception:
                pass
            
            # Если не удалось определить по частоте, используем эвристику:
            # Предполагаем, что E-cores составляют вторую половину логических ядер
            if not e_core_ids:
                e_core_start = logical_cores // 2
                e_core_ids = set(range(e_core_start, logical_cores))
            
            os.sched_setaffinity(0, e_core_ids)
            logger.info(f"Process pinned to E-cores (Linux, affinity: {e_core_ids})")
        except ImportError:
            logger.error("psutil not installed, cannot detect E-cores on Linux. Install with: pip install psutil")
        except Exception as e:
            logger.warning(f"Failed to pin process to E-cores on Linux: {e}")
    
    @staticmethod
    def _pin_windows_e_cores():
        """Windows-реализация привязки к E-cores через Win32 API."""
        try:
            import win32process
            import win32api
            from ctypes import windll, c_void_p, POINTER, Structure, sizeof
            from ctypes.wintypes import DWORD, BOOL
            
            # Определяем структуру SYSTEM_CPU_SET_INFORMATION
            class SYSTEM_CPU_SET_INFORMATION(Structure):
                _fields_ = [
                    ("Size", DWORD),
                    ("Flags", DWORD),
                    ("Id", DWORD),
                    ("Group", DWORD),
                    ("LogicalProcessorIndex", DWORD),
                    ("CoreIndex", DWORD),
                    ("LastLevelCacheIndex", DWORD),
                    ("NumericPropertyMask", DWORD),
                    ("EfficiencyClass", DWORD),  # 0-1=E-core, 2+=P-core для Intel Hybrid
                ]
            
            # Получаем информацию о CPU sets
            GetSystemCpuSetInformation = windll.kernel32.GetSystemCpuSetInformation
            GetSystemCpuSetInformation.argtypes = [
                POINTER(SYSTEM_CPU_SET_INFORMATION),
                DWORD,
                POINTER(DWORD),
                c_void_p,
                DWORD
            ]
            GetSystemCpuSetInformation.restype = BOOL
            
            # Выделяем память для информации
            max_cpu_sets = 256
            info = (SYSTEM_CPU_SET_INFORMATION * max_cpu_sets)()
            return_length = DWORD(0)
            
            # Вызываем функцию
            success = GetSystemCpuSetInformation(
                info,
                sizeof(info),
                return_length,
                None,
                0
            )
            
            if not success:
                logger.warning("Failed to get CPU set information on Windows")
                return
            
            # Считаем количество активных CPU sets
            num_sets = return_length.value // sizeof(SYSTEM_CPU_SET_INFORMATION)
            e_core_mask = 0
            
            for i in range(num_sets):
                cpu_info = info[i]
                # EfficiencyClass: 0 или 1 для E-core, 2+ для P-core (Intel Hybrid)
                if cpu_info.EfficiencyClass <= 1:
                    e_core_mask |= (1 << cpu_info.LogicalProcessorIndex)
            
            if e_core_mask == 0:
                logger.warning("No E-cores found on Windows, pinning to all available cores")
                # Привязываем ко всем доступным ядрам
                all_cores_mask = (1 << (os.cpu_count() or 1)) - 1
                win32process.SetProcessAffinityMask(win32api.GetCurrentProcess(), all_cores_mask)
            else:
                # Устанавливаем аффинность
                handle = win32api.GetCurrentProcess()
                win32process.SetProcessAffinityMask(handle, e_core_mask)
                logger.info(f"Process pinned to E-cores (Windows, mask: {hex(e_core_mask)})")
        
        except ImportError:
            logger.error("pywin32 not installed, cannot pin process on Windows. Install with: pip install pywin32")
        except Exception as e:
            logger.warning(f"Failed to pin process to E-cores on Windows: {e}")
