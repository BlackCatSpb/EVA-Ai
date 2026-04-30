"""
Memory Management mixin for CoreBrain.
Управление памятью моделей: активная выгрузка, мониторинг, автовыгрузка при простое.
"""
import time
import gc
import logging
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger("eva_ai.memory_manager")


class MemoryManagerMixin:
    """Миксин для управления памятью моделей в CoreBrain."""

    def _init_memory_manager(self):
        """Инициализация менеджера памяти."""
        self._last_query_time = time.time()
        self._idle_unload_timer = None
        self._idle_unload_threshold = float(
            self.config.get('memory', {}).get('idle_unload_seconds', 300)
        )
        self._auto_unload_enabled = bool(
            self.config.get('memory', {}).get('auto_unload_enabled', True)
        )
        self._unload_in_progress = False
        self._unload_lock = threading.Lock()
        
        logger.info(
            f"MemoryManager инициализирован (idle_unload={self._idle_unload_threshold}s, "
            f"auto_unload={self._auto_unload_enabled})"
        )

    def record_query_activity(self):
        """Зафиксировать активность запроса (сбрасывает таймер автовыгрузки)."""
        self._last_query_time = time.time()
        
        # Отменяем запланированную выгрузку
        if self._idle_unload_timer is not None:
            self._idle_unload_timer.cancel()
            self._idle_unload_timer = None

    def unload_all_models(self) -> Dict[str, bool]:
        """
        Активная выгрузка ВСЕХ моделей из памяти.
        
        Returns:
            Dict[str, bool]: результат выгрузки по каждому компоненту
        """
        with self._unload_lock:
            if self._unload_in_progress:
                logger.warning("Выгрузка уже выполняется")
                return {'status': 'already_in_progress'}
            
            self._unload_in_progress = True
        
        results = {}
        
        try:
            # 1. Two-model pipeline (GGUF Model A, B, C)
            if hasattr(self, 'two_model_pipeline') and self.two_model_pipeline:
                try:
                    self.two_model_pipeline.unload_models()
                    results['two_model_pipeline'] = True
                    logger.info("Two-model pipeline выгружен")
                except Exception as e:
                    results['two_model_pipeline'] = False
                    logger.error(f"Ошибка выгрузки pipeline: {e}")
            
            # 3. FractalModelManager (GGUF + PyTorch)
            if hasattr(self, 'fractal_model_manager') and self.fractal_model_manager:
                try:
                    if hasattr(self.fractal_model_manager, 'unload'):
                        self.fractal_model_manager.unload()
                        results['fractal_model_manager'] = True
                        logger.info("FractalModelManager выгружен")
                    else:
                        results['fractal_model_manager'] = False
                except Exception as e:
                    results['fractal_model_manager'] = False
                    logger.error(f"Ошибка выгрузки FractalModelManager: {e}")
            
            # 4. LlamaCppHotDeployment
            if hasattr(self, 'llama_cpp_deployment') and self.llama_cpp_deployment:
                try:
                    if hasattr(self.llama_cpp_deployment, 'unload'):
                        self.llama_cpp_deployment.unload()
                        results['llama_cpp_deployment'] = True
                        logger.info("LlamaCppHotDeployment выгружен")
                    else:
                        results['llama_cpp_deployment'] = False
                except Exception as e:
                    results['llama_cpp_deployment'] = False
                    logger.error(f"Ошибка выгрузки LlamaCppHotDeployment: {e}")
            
            # 5. ResponseGenerator
            if hasattr(self, 'response_generator') and self.response_generator:
                try:
                    if hasattr(self.response_generator, 'shutdown'):
                        self.response_generator.shutdown()
                        results['response_generator'] = True
                        logger.info("ResponseGenerator остановлен")
                    else:
                        results['response_generator'] = False
                except Exception as e:
                    results['response_generator'] = False
                    logger.error(f"Ошибка остановки ResponseGenerator: {e}")
            
            # 6. QwenModelManager (если есть отдельно)
            if hasattr(self, 'qwen_model_manager') and self.qwen_model_manager:
                try:
                    if hasattr(self.qwen_model_manager, 'unload'):
                        self.qwen_model_manager.unload()
                        results['qwen_model_manager'] = True
                        logger.info("QwenModelManager выгружен")
                    else:
                        results['qwen_model_manager'] = False
                except Exception as e:
                    results['qwen_model_manager'] = False
                    logger.error(f"Ошибка выгрузки QwenModelManager: {e}")
            
            # 7. ModelManager (общий кэш моделей)
            if hasattr(self, 'model_manager') and self.model_manager:
                try:
                    if hasattr(self.model_manager, 'cleanup'):
                        self.model_manager.cleanup()
                        results['model_manager'] = True
                        logger.info("ModelManager очищен")
                    else:
                        results['model_manager'] = False
                except Exception as e:
                    results['model_manager'] = False
                    logger.error(f"Ошибка очистки ModelManager: {e}")
            
            # 8. Глобальная очистка
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("VRAM очищен (torch.cuda.empty_cache)")
            except Exception:
                pass
            
            collected = gc.collect()
            logger.info(f"Сборка мусора: {collected} объектов")
            
            results['gc_collected'] = collected
            results['status'] = 'success'
            
            # Обновляем время последней активности
            self._last_query_time = time.time()
            
        finally:
            with self._unload_lock:
                self._unload_in_progress = False
        
        return results
 
    def reload_models(self) -> Dict[str, bool]:
        """Перезагрузить все модели после выгрузки."""
        results = {}
        
        try:
            # Переинициализация пайплайна
            if hasattr(self, 'two_model_pipeline') and self.two_model_pipeline:
                try:
                    self.two_model_pipeline.load_models()
                    results['two_model_pipeline'] = True
                    logger.info("Two-model pipeline перезагружен")
                except Exception as e:
                    results['two_model_pipeline'] = False
                    logger.error(f"Ошибка перезагрузки pipeline: {e}")
            
            # Переинициализация FractalModelManager
            if hasattr(self, 'fractal_model_manager') and self.fractal_model_manager:
                try:
                    self.fractal_model_manager._initialize_llama_cpp()
                    results['fractal_model_manager'] = True
                    logger.info("FractalModelManager перезагружен")
                except Exception as e:
                    results['fractal_model_manager'] = False
                    logger.error(f"Ошибка перезагрузки FractalModelManager: {e}")
            
            results['status'] = 'success'
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results

    def get_memory_usage(self) -> Dict[str, Any]:
        """Получить текущее потребление памяти моделями."""
        usage = {
            'models_loaded': [],
            'total_estimated_mb': 0,
        }
        
        # Проверяем каждую модель
        if hasattr(self, 'two_model_pipeline') and self.two_model_pipeline:
            pipeline = self.two_model_pipeline
            if getattr(pipeline, 'model_a', None):
                usage['models_loaded'].append('Model A (GGUF)')
                usage['total_estimated_mb'] += 500  # ~500MB для 3B Q4
            if getattr(pipeline, 'model_b', None):
                usage['models_loaded'].append('Model B (GGUF)')
                usage['total_estimated_mb'] += 500
            
        if hasattr(self, 'fractal_model_manager') and self.fractal_model_manager:
            fmm = self.fractal_model_manager
            if getattr(fmm, 'llama_cpp_ready', False):
                usage['models_loaded'].append('Fractal GGUF')
                usage['total_estimated_mb'] += 300
            if getattr(fmm, 'model', None):
                usage['models_loaded'].append('Fractal PyTorch')
                usage['total_estimated_mb'] += 1000  # ~1GB для PyTorch
        
        if hasattr(self, 'llama_cpp_deployment') and self.llama_cpp_deployment:
            if getattr(self.llama_cpp_deployment, 'ready', False):
                usage['models_loaded'].append('LlamaCpp Deployment')
                usage['total_estimated_mb'] += 300
        
        # Системная память
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            usage['process_rss_mb'] = mem_info.rss / (1024 * 1024)
            usage['process_vms_mb'] = mem_info.vms / (1024 * 1024)
        except Exception:
            pass
        
        # VRAM
        try:
            import torch
            if torch.cuda.is_available():
                usage['vram_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
                usage['vram_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
        except Exception:
            pass
        
        return usage

    def schedule_idle_unload(self, delay: float = None):
        """Запланировать автовыгрузку моделей при простое."""
        if not self._auto_unload_enabled:
            return
        
        delay = delay or self._idle_unload_threshold
        
        # Отменяем предыдущий таймер
        if self._idle_unload_timer is not None:
            self._idle_unload_timer.cancel()
        
        def _idle_unload_callback():
            idle_time = time.time() - self._last_query_time
            if idle_time >= delay:
                logger.info(f"Автовыгрузка после {idle_time:.0f}s простоя")
                self.unload_all_models()
            else:
                logger.debug(f"Прерываем автовыгрузку: активность {idle_time:.0f}s назад")
        
        self._idle_unload_timer = threading.Timer(delay, _idle_unload_callback)
        self._idle_unload_timer.daemon = True
        self._idle_unload_timer.start()
        logger.debug(f"Запланирована автовыгрузка через {delay}s")

    def cancel_idle_unload(self):
        """Отменить запланированную автовыгрузку."""
        if self._idle_unload_timer is not None:
            self._idle_unload_timer.cancel()
            self._idle_unload_timer = None
            logger.debug("Автовыгрузка отменена")
