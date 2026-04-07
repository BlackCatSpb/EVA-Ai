"""
Аппаратные оптимизации для ЕВА.

Модуль содержит функции для оптимизации работы с GPU и CPU,
автоматической настройки параметров PyTorch для максимальной производительности.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def apply_hardware_optimizations(device, config: Optional[dict] = None, deferred_system=None) -> None:
    """
    Применяет аппаратные оптимизации для текущего устройства.

    Args:
        device: Устройство PyTorch (CPU/CUDA)
        config: Конфигурация системы
        deferred_system: Система отложенных команд (опционально)
    """
    def _apply_cpu_optimizations():
        try:
            import torch
            # Оптимизации для CPU
            threads = max(1, (os.cpu_count() or 2) // 2)
            torch.set_num_threads(threads)
            torch.set_num_interop_threads(1)
            logger.info(f"CPU оптимизации применены (потоков: {threads})")
        except Exception as e:
            logger.warning(f"Не удалось применить CPU оптимизации: {e}")
    
    def _apply_gpu_optimizations():
        try:
            import torch
            if not torch.cuda.is_available():
                return
                
            # Очищаем кэш GPU
            torch.cuda.empty_cache()
            
            # Устанавливаем оптимальное количество потоков
            torch.set_num_threads(min(4, os.cpu_count() or 2))
            
            # Настраиваем лимит памяти GPU (если доступно)
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                try:
                    memory_fraction = float(config.get("gpu_memory_fraction", 0.8)) if config else 0.8
                    torch.cuda.set_per_process_memory_fraction(memory_fraction)
                    logger.debug(f"Установлен лимит GPU памяти: {memory_fraction*100:.0f}%")
                except Exception as e:
                    logger.warning(f"Не удалось установить лимит GPU памяти: {e}")
            
            # Применяем оптимизации CUDA
            optimize_cuda_settings(device, config, deferred_system=deferred_system)
            
        except Exception as e:
            logger.warning(f"Не удалось применить GPU оптимизации: {e}")
    
    try:
        if not hasattr(device, 'type'):
            logger.warning(f"Некорректное устройство: {device}")
            return
            
        if device.type == 'cuda':
            _apply_gpu_optimizations()
        elif device.type == 'cpu':
            _apply_cpu_optimizations()
        else:
            logger.info(f"Оптимизации для устройства {device.type} не поддерживаются")
            
    except Exception as e:
        logger.warning(f"Не удалось применить аппаратные оптимизации: {e}", exc_info=True)


def get_runtime_diagnostics(device, precision, pin_memory_default) -> dict:
    """
    Возвращает диагностическую информацию по настройкам исполнения.

    Args:
        device: Устройство PyTorch
        precision: Точность вычислений
        pin_memory_default: Флаг закрепления памяти

    Returns:
        dict: Диагностическая информация
    """
    try:
        import torch
        from ..utils.memory_info import memory_info

        di = memory_info()
        diag = {
            "device": f"{device.type}",
            "precision": f"{precision}",
            "pin_memory_default": bool(pin_memory_default),
            "torch_threads": int(getattr(torch, 'get_num_threads', lambda: None)() or 0),
            "interop_threads": int(getattr(torch, 'get_num_interop_threads', lambda: None)() or 0),
            "cuda": di,
            "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM", "false"),
        }
        return diag
    except Exception:
        return {
            "device": getattr(device, 'type', 'cpu'),
            "precision": str(precision),
            "pin_memory_default": bool(pin_memory_default)
        }


def setup_tokenizer_parallelism(config: Optional[dict] = None) -> None:
    """
    Настраивает параллелизм токенизатора.

    Args:
        config: Конфигурация системы
    """
    try:
        tok_par = str(config.get("tokenizers_parallelism", "false")).lower() if config else "false"
        os.environ["TOKENIZERS_PARALLELISM"] = "true" if tok_par in ("1", "true", "yes") else "false"
        logger.debug(f"Tokenizers parallelism: {os.environ.get('TOKENIZERS_PARALLELISM')}")
    except Exception as e:
        logger.warning(f"Не удалось настроить параллелизм токенизатора: {e}")


def optimize_cuda_settings(device, config: Optional[dict] = None, deferred_system=None) -> None:
    """
    Оптимизирует настройки CUDA для лучшей производительности.

    Args:
        device: CUDA устройство
        config: Конфигурация системы
        deferred_system: Система отложенных команд (опционально)
    """
    def _apply_cuda_optimizations():
        try:
            import torch
            
            if not torch.cuda.is_available():
                return
                
            # Включаем TF32 для матричных операций (Ampere+)
            if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = True
                logger.debug("TF32 активирован для матричных операций")
                
            # Включаем TF32 для сверточных операций
            if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
                logger.debug("TF32 активирован для сверточных операций")
                
            # Включаем бенчмаркинг для cudnn
            if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'benchmark'):
                torch.backends.cudnn.benchmark = True
                logger.debug("Бенчмаркинг cudnn активирован")
                
            logger.info("CUDA оптимизации применены")
            
        except Exception as e:
            logger.warning(f"Не удалось применить оптимизации CUDA: {e}")
    
    # Если передан deferred_system, откладываем выполнение, иначе выполняем сразу
    if deferred_system is not None:
        deferred_system.defer_command(
            _apply_cuda_optimizations,
            priority='high',
            condition=lambda: torch.cuda.is_available() if 'torch' in globals() else False
        )
        logger.debug("Оптимизации CUDA запланированы к выполнению")
    else:
        _apply_cuda_optimizations()


def optimize_torch_precision(config: Optional[dict] = None) -> None:
    """
    Оптимизирует точность вычислений PyTorch.

    Args:
        config: Конфигурация системы
    """
    try:
        import torch

        # Точность умножений FP32 (PyTorch 2.x)
        level = "high"  # По умолчанию высокая производительность

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(level)

        logger.debug(f"Точность PyTorch установлена: {level}")

    except Exception as e:
        logger.warning(f"Не удалось оптимизировать точность PyTorch: {e}")
