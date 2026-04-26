"""
FCPPipelineV15 - Основной FCP Pipeline для EVA-Ai

Простой и рабочий пайплайн генерации на базе ruadapt_qwen3_4b OpenVINO.
"""
import os
import time
from typing import Optional, Dict, Any, Callable, Generator
import numpy as np


# Системный промпт - ВСЕГДА рассуждать перед ответом
SYSTEM_PROMPT = """Ты - интеллектуальный помощник EVA. ВСЕГДА перед ответом выполняй глубокое обдумывание и анализ. Показывай свои рассуждения в тегах <think>...</think>. ОБЯЗАТЕЛЬНО закрой тег </think> после завершения рассуждений, затем давай окончательный ответ. Рассуждения должны быть подробными, логичными и полезными."""


class SimpleStreamer:
    """Streamer для OpenVINO GenAI"""
    
    def __init__(self, callback: Callable[[str], None] = None):
        self.callback = callback
        self.generated_text = ""
    
    def __call__(self, token_text: str) -> bool:
        self.generated_text += token_text
        if self.callback:
            self.callback(token_text)
        return False  # Не останавливать генерацию

try:
    import openvino_genai as ov_genai
    HAS_OV_GENAI = True
except ImportError:
    HAS_OV_GENAI = False

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class FCPPipelineV15:
    """Основной FCP Pipeline - простая и рабочая версия"""
    
    def __init__(
        self,
        model_path: str,
        graph_path: str = None,
        gnn_ov_path: Optional[str] = None,
        lora_dir: Optional[str] = None,
        draft_model_path: Optional[str] = None
    ):
        self.model_path = model_path
        self.graph_path = graph_path
        self.gnn_ov_path = gnn_ov_path
        self.lora_dir = lora_dir or "C:/Users/black/OneDrive/Desktop/FCP/lora_adapters"
        
        self.stats = {"queries": 0, "injections": 0}
        
        # Инициализация
        self._init_tokenizer()
        self._init_pipeline(draft_model_path)
        self._init_lora_manager()
        
        print(f"[FCP] FCPPipelineV15 created: model={model_path}")
    
    def _init_tokenizer(self):
        if HAS_TRANSFORMERS and os.path.exists(self.model_path):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            except:
                self.tokenizer = None
        else:
            self.tokenizer = None
    
    def _init_pipeline(self, draft_model_path=None):
        import os
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        print(f"[FCP] CPU cores available: {cpu_count}")
        print(f"[FCP] _init_pipeline: model_path={self.model_path}")
        print(f"[FCP] _init_pipeline: exists={os.path.exists(self.model_path)}")
        print(f"[FCP] _init_pipeline: HAS_OV_GENAI={HAS_OV_GENAI}")
        if not HAS_OV_GENAI:
            self.pipeline = None
            print("[FCP] OpenVINO GenAI not available")
            return
        
        try:
            # Максимальная производительность CPU
            os.environ['PERFORMANCE_HINT'] = 'LATENCY'
            os.environ['INFERENCE_NUM_THREADS'] = str(cpu_count)
            os.environ['NUM_STREAMS'] = '1'
            os.environ['ENABLE_HYPER_THREADING'] = 'YES'
            os.environ['ENABLE_CPU_PINNING'] = 'YES'
            os.environ['CPU_DENORMALS_OPTIMIZATION'] = 'YES'
            print(f"[FCP] CPU optimization enabled: {cpu_count} threads")
            
            # SchedulerConfig - оптимизировано для одного запроса
            scheduler = ov_genai.SchedulerConfig()
            scheduler.cache_size = 4
            scheduler.max_num_seqs = 1
            scheduler.max_num_batched_tokens = 4096
            scheduler.enable_prefix_caching = True
            scheduler.use_cache_eviction = True
            
            # GenerationConfig
            gen_config = ov_genai.GenerationConfig()
            gen_config.max_new_tokens = 4096
            gen_config.temperature = 0.2
            gen_config.top_p = 0.9
            gen_config.top_k = 40
            gen_config.repetition_penalty = 1.1
            gen_config.no_repeat_ngram_size = 5
            gen_config.do_sample = True
            
            # Draft model для спекулятивного декодирования
            draft_model = None
            if draft_model_path and os.path.exists(draft_model_path):
                try:
                    draft_model = ov_genai.LLMPipeline(draft_model_path, "CPU",
                                                       config={"scheduler_config": scheduler})
                    print(f"[FCP] Draft model loaded")
                except Exception as e:
                    print(f"[FCP] Draft model load failed: {e}")
                    draft_model = None
            
            print(f"[FCP] Loading LLMPipeline from {self.model_path}...")
            # Передаём параметры правильно для openvino_genai
            pipeline_kwargs = {
                "models_path": self.model_path,
                "device": "CPU",
                "config": {"scheduler_config": scheduler}
            }
            if draft_model is not None:
                pipeline_kwargs["draft_model"] = draft_model
            
            self.pipeline = ov_genai.LLMPipeline(**pipeline_kwargs)
            
            # Применяем GenerationConfig
            self.pipeline.set_generation_config(gen_config)
            
            print(f"[FCP] Pipeline initialized successfully: {self.pipeline is not None}")
        except Exception as e:
            print(f"[FCP] Pipeline init error: {e}")
            import traceback
            traceback.print_exc()
            self.pipeline = None
            return
    
    def generate_streaming(self, prompt, max_new_tokens=4096, enable_thinking=True, callback=None, **kwargs):
        """Streaming с парсингом тегов размышления в процессе генерации"""
        if not self.pipeline:
            yield {"type": "error", "text": "[No pipeline]"}
            return
        
        chat_prompt = self._build_prompt(prompt, enable_thinking)
        
        yield {"type": "start", "timestamp": time.time()}
        
        # СРАЗУ отправляем reasoning_start - начинаем с рассуждений
        if enable_thinking:
            yield {"type": "reasoning_start"}
        
        try:
            import queue
            import threading
            
            event_queue = queue.Queue()
            
            # Состояние парсера
            buffer = ""
            in_thinking = False  # Начинаем с обычного режима, ждём <think>
            partial_tag = ""  # Для накопления частичных тегов
            
            def token_callback(token_text: str):
                nonlocal buffer, in_thinking, partial_tag
                buffer += token_text
                
                while True:
                    if in_thinking:
                        # Режим рассуждений - ищем конец </think>
                        # Ищем как подстроку
                        idx = buffer.find("</think>")
                        if idx != -1:
                            print(f"[FCP STREAM] Found </think> tag, switching to answer mode")
                            thinking = buffer[:idx]
                            if thinking.strip():
                                print(f"[FCP STREAM] Sending reasoning_text, length: {len(thinking)}")
                                event_queue.put({"type": "reasoning_text", "text": thinking})
                            in_thinking = False
                            partial_tag = ""
                            buffer = buffer[idx + len("</think>"):]
                            event_queue.put({"type": "reasoning_end"})
                        else:
                            # Тег не найден, отправляем как есть
                            if buffer:
                                event_queue.put({"type": "reasoning_text", "text": buffer})
                            buffer = ""
                            break
                    else:
                        # Режим основного ответа - ищем <think>
                        idx = buffer.find("<think>")
                        if idx != -1:
                            print(f"[FCP STREAM] Found <think> tag, switching to thinking mode")
                            # Текст до <think> - отправляем как chunk
                            if idx > 0:
                                event_queue.put({"type": "chunk", "text": buffer[:idx]})
                            in_thinking = True
                            buffer = buffer[idx + len("<think>"):]
                            event_queue.put({"type": "reasoning_start"})
                        else:
                            # Тега нет, отправляем буфер как чанк
                            if buffer:
                                event_queue.put({"type": "chunk", "text": buffer})
                                buffer = ""
                            break
                return False
            
            def generate():
                """Генерация в отдельном потоке"""
                nonlocal buffer, in_thinking, partial_tag
                try:
                    gen_cfg = self.pipeline.get_generation_config()
                    gen_cfg.max_new_tokens = max_new_tokens
                    gen_cfg.temperature = 0.2
                    gen_cfg.top_p = 0.9
                    gen_cfg.top_k = 40
                    gen_cfg.repetition_penalty = 1.1
                    gen_cfg.do_sample = True
                    
                    self.pipeline.generate(chat_prompt, generation_config=gen_cfg, streamer=token_callback)
                    
                    # После завершения обрабатываем остаток буфера
                    if buffer:
                        if in_thinking:
                            if buffer.strip():
                                print(f"[FCP STREAM] Final reasoning_text, length: {len(buffer)}")
                                event_queue.put({"type": "reasoning_text", "text": buffer})
                            event_queue.put({"type": "reasoning_end"})
                        else:
                            if buffer.strip():
                                event_queue.put({"type": "chunk", "text": buffer})
                except Exception as e:
                    event_queue.put({"type": "error", "text": str(e)})
                finally:
                    event_queue.put({"type": "done", "timestamp": time.time()})
            
            # Запускаем генерацию
            gen_thread = threading.Thread(target=generate)
            gen_thread.start()
            
            # Читаем события из очереди
            while True:
                try:
                    event = event_queue.get(timeout=0.1)
                    yield event
                    if event['type'] == 'done':
                        break
                except queue.Empty:
                    if not gen_thread.is_alive():
                        break
            
            gen_thread.join()
            
        except Exception as e:
            yield {"type": "error", "text": str(e)}
    
    def _init_lora_manager(self):
        self.current_adapter = None
        if self.lora_dir and os.path.exists(self.lora_dir):
            default_adapter = "fcp_finetuned"
            adapter_path = os.path.join(self.lora_dir, default_adapter)
            if os.path.exists(adapter_path):
                self.current_adapter = default_adapter
                print(f"[FCP] LoRA adapter ready: {default_adapter}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        enable_injection: bool = False,
        use_lora: bool = True,
        return_metadata: bool = False,
        **kwargs
    ) -> str:
        """Основной метод генерации"""
        self.stats["queries"] += 1
        
        # Подготовка промпта
        chat_prompt = self._build_prompt(prompt, enable_thinking)
        
        # Генерация
        response = self._generate(chat_prompt, max_new_tokens, **kwargs)
        
        self.stats["injections"] += 1
        
        if return_metadata:
            return response, {"query_count": self.stats["queries"]}
        return response
    
    def _build_prompt(self, prompt: str, enable_thinking: bool) -> str:
        """Формирование промпта"""
        if enable_thinking:
            # Модель сама генерирует теги <think>...
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def _generate(self, prompt: str, max_new_tokens: int = 1024, **kwargs) -> str:
        """Генерация ответа"""
        if not self.pipeline:
            return "[No pipeline]"
        
        try:
            gen_cfg = ov_genai.GenerationConfig()
            gen_cfg.max_new_tokens = max_new_tokens
            gen_cfg.temperature = 0.2
            gen_cfg.top_p = 0.9
            gen_cfg.top_k = 40
            gen_cfg.repetition_penalty = 1.1
            
            result = self.pipeline.generate(prompt, generation_config=gen_cfg, **kwargs)
            return result
        except Exception as e:
            return f"Generation error: {e}"
    
    def load_lora_adapter(self, adapter_name: str = "fcp_finetuned", alpha: float = 0.8):
        """Загрузить LoRA адаптер"""
        if not self.lora_dir:
            return False
        
        adapter_path = os.path.join(self.lora_dir, adapter_name)
        if os.path.exists(adapter_path):
            self.current_adapter = adapter_name
            return True
        return False
    
    def get_statistics(self) -> Dict:
        return self.stats.copy()


def create_fcp_pipeline(model_path: str, graph_path: str = None, **kwargs):
    """Factory function"""
    return FCPPipelineV15(model_path, graph_path, **kwargs)