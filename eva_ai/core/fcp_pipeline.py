"""
FCPPipelineV15 - Основной FCP Pipeline для EVA-Ai

Простой и рабочий пайплайн генерации на базе ruadapt_qwen3_4b OpenVINO.
"""
import os
from typing import Optional, Dict, Any, Callable, Generator
import numpy as np


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
        if not HAS_OV_GENAI:
            self.pipeline = None
            print("[FCP] OpenVINO GenAI not available")
            return
        
        try:
            # Настройки из FMF - оптимальные
            import os
            os.environ['PERFORMANCE_HINT'] = 'LATENCY'
            os.environ['INFERENCE_NUM_THREADS'] = '8'
            
            # SchedulerConfig
            scheduler = ov_genai.SchedulerConfig()
            scheduler.max_num_batched_tokens = 2048
            scheduler.max_num_seqs = 8
            scheduler.enable_prefix_caching = True
            
            # GenerationConfig - установим по умолчанию
            gen_config = ov_genai.GenerationConfig()
            gen_config.max_new_tokens = 2048
            gen_config.temperature = 0.2
            gen_config.top_p = 0.9
            gen_config.top_k = 40
            gen_config.repetition_penalty = 1.1
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
            
            self.pipeline = ov_genai.LLMPipeline(
                self.model_path, "CPU",
                kv_cache_precision="u8",
                draft_model=draft_model,
                config={"scheduler_config": scheduler}
            )
            
            # Применяем GenerationConfig
            self.pipeline.set_generation_config(gen_config)
            
            print(f"[FCP] Pipeline initialized with FMF config")
        except Exception as e:
            print(f"[FCP] Pipeline init error: {e}")
            self.pipeline = None
            print("[FCP] OpenVINO GenAI not available")
            return
        
        try:
            # Настройки из FMF - оптимальные
            import os
            os.environ['PERFORMANCE_HINT'] = 'LATENCY'
            os.environ['INFERENCE_NUM_THREADS'] = '8'
            
            # SchedulerConfig
            scheduler = ov_genai.SchedulerConfig()
            scheduler.max_num_batched_tokens = 2048
            scheduler.max_num_seqs = 8
            scheduler.enable_prefix_caching = True
            
            # GenerationConfig - установим по умолчанию
            gen_config = ov_genai.GenerationConfig()
            gen_config.max_new_tokens = 2048
            gen_config.temperature = 0.2
            gen_config.top_p = 0.9
            gen_config.top_k = 40
            gen_config.repetition_penalty = 1.1
            gen_config.do_sample = True
            
             self.pipeline = ov_genai.LLMPipeline(
                self.model_path, 
                "CPU",
                kv_cache_precision="u8",
                draft_model=draft_model,
                config={"scheduler_config": scheduler}
            )
            
            # Применяем GenerationConfig
            self.pipeline.set_generation_config(gen_config)
            
            print(f"[FCP] Pipeline initialized with FMF config: {self.model_path}")
            if draft_model:
                print(f"[FCP] Draft model enabled for speculative decoding")
        except Exception as e:
            print(f"[FCP] Pipeline init error: {e}")
            self.pipeline = None
    
    def generate_streaming(self, prompt, max_new_tokens=1024, enable_thinking=True, callback=None, **kwargs):
        """Генерация со стримингом - токены отправляются через callback"""
        if not self.pipeline:
            yield "[No pipeline]"
            return
        
        chat_prompt = self._build_prompt(prompt, enable_thinking)
        
        try:
            streamer = SimpleStreamer(callback)
            # Используем streamer - должен возвращать токены по одному
            result = self.pipeline.generate(chat_prompt, max_new_tokens=max_new_tokens, streamer=streamer, **kwargs)
            # Если результат пришёл весь сразу, стриминг не работает - возвращаем целиком
            yield result
        except Exception as e:
            yield f"Error: {e}"
    
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
    ) -> str:
        enable_injection: bool = False,
        use_lora: bool = True,
        return_metadata: bool = False,
        **kwargs
    ) -> str:
        """Основной метод генерации"""
        **kwargs
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
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
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