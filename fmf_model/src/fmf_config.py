"""
FMF Optimized Configuration
Полная конфигурация OpenVINO для FMF Eva
"""
import openvino_genai as ov_genai

def create_fmf_config() -> dict:
    """Создать оптимальную конфигурацию для FMF"""
    
    # === GenerationConfig ===
    generation_config = ov_genai.GenerationConfig()
    generation_config.max_new_tokens = 2048
    generation_config.temperature = 0.2  # Точные ответы
    generation_config.top_p = 0.9        # Nucleus sampling
    generation_config.top_k = 40         # Top-k sampling
    generation_config.repetition_penalty = 1.1  # Уменьшение повторов
    generation_config.do_sample = True
    
    # === SchedulerConfig (continuous batching) ===
    scheduler_config = ov_genai.SchedulerConfig()
    scheduler_config.max_num_batched_tokens = 2048
    scheduler_config.max_num_seqs = 8
    scheduler_config.enable_prefix_caching = True
    
    # === SparseAttentionConfig (для длинных контекстов > 8K) ===
    # Note: CacheEvictionConfig параметры недоступны в текущей версии
    # sparse_attention_config = ov_genai.SparseAttentionConfig(
    #     mode=ov_genai.SparseAttentionMode.TRISHAPE,
    #     num_last_dense_tokens_in_prefill=100,
    #     num_retained_start_tokens_in_cache=128,
    #     num_retained_recent_tokens_in_cache=1920
    # )
    
    # === Pipeline Config ===
    config = {
        "generation_config": generation_config,
        "scheduler_config": scheduler_config,
        "sparse_attention_config": sparse_attention_config,
        # Performance hints
        "PERFORMANCE_HINT": "LATENCY",
        "NUM_STREAMS": "AUTO",
        "INFERENCE_NUM_THREADS": 8,
        "ENABLE_HYPER_THREADING": "AUTO"
    }
    
    return config


def create_fmf_pipeline(model_path: str, device: str = "CPU"):
    """Создать оптимизированный FMF пайплайн"""
    
    config_dict = create_fmf_config()
    
    pipe = ov_genai.LLMPipeline(
        model_path=model_path,
        device=device,
        config=config_dict
    )
    
    return pipe


class FMFGeneratorOptimized:
    """Оптимизированный FMF генератор"""
    
    def __init__(self, model_path: str, graph_path: str = None, device: str = "CPU"):
        self.model_path = model_path
        self.graph_path = graph_path
        self.device = device
        self.config = create_fmf_config()
        
        # Создаём пайплайн с настройками
        self.pipe = ov_genai.LLMPipeline(model_path, device, config=self.config)
        
        # Токенизатор
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, fix_mistral_regex=True
        )
        
        # Загружаем граф
        from .fmf_interactive import FractalGraphV2ThreadSafe
        self.graph = FractalGraphV2ThreadSafe(graph_path, enable_embeddings=True) if graph_path else None
    
    def generate(self, prompt: str, max_new_tokens: int = None, temperature: float = None) -> str:
        """Генерация с переопределением параметров"""
        
        config = self.config.get("generation_config")
        
        if max_new_tokens:
            config.max_new_tokens = max_new_tokens
        if temperature:
            config.temperature = temperature
            
        return self.pipe.generate(prompt, config)
    
    def generate_streaming(self, prompt: str, max_new_tokens: int = 2048):
        """Streaming генерация"""
        config = self.config.get("generation_config")
        config.max_new_tokens = max_new_tokens
        
        # Токенизация
        tokens = self.tokenizer.encode(prompt)
        
        # Генерация чанками
        self.pipe._tokenizer.start_chat()
        for token in self.pipe.generate_iter(prompt, config):
            yield token


if __name__ == "__main__":
    # Тест
    config = create_fmf_config()
    print("=== FMF Config ===")
    print(f"Generation: max_new_tokens={config['generation_config'].max_new_tokens}")
    print(f"  temperature={config['generation_config'].temperature}")
    print(f"  top_p={config['generation_config'].top_p}")
    print(f"  repetition_penalty={config['generation_config'].repetition_penalty}")
    print(f"Scheduler: max_num_seqs={config['scheduler_config'].max_num_seqs}")
    print(f"  enable_prefix_caching={config['scheduler_config'].enable_prefix_caching}")
    print(f"Performance: {config['PERFORMANCE_HINT']}")