"""
FCPPipelineV15 - Основной FCP Pipeline для EVA-Ai

Простой и рабочий пайплайн генерации на базе ruadapt_qwen3_4b OpenVINO.
С KCA (Knowledge Conscious Attention) и SRG (Semantic Relevance Gate).
"""
import os
import time
from typing import Optional, Dict, Any, Callable, Generator, Tuple, List
import numpy as np

# FCP Core Components
from eva_ai.fcp_core import (
    FCPConfig,
    ConvergenceController,
    KnowledgeConsciousAttention,
    SemanticRelevanceGate,
    FractalGraphV2
)

# FCP GNN Components
from eva_ai.fcp_gnn import (
    HybridLayerProcessor,
    HybridLayerManager,
    HybridLayerConfig,
    FractalGraphEncoder,
    AdaptiveFusionInjector,
    TextFusionInjector,
    HybridFusionInjector
)


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
    """Основной FCP Pipeline с KCA и SRG"""

    def __init__(
        self,
        model_path: str,
        graph_path: str = None,
        gnn_ov_path: Optional[str] = None,
        lora_dir: Optional[str] = None,
        draft_model_path: Optional[str] = None,
        max_history: int = 10
    ):
        self.model_path = model_path
        self.graph_path = graph_path
        self.gnn_ov_path = gnn_ov_path
        self.lora_dir = lora_dir or "C:/Users/black/OneDrive/Desktop/FCP/lora_adapters"
        self.max_history = max_history

        self.stats = {"queries": 0, "injections": 0}

        # Conversation history for context
        self.conversation_history: List[Dict[str, str]] = []

        # FCP Core Components
        self.fcp_config = FCPConfig()
        self.fractal_graph = None
        self.kca = None
        self.srg = None
        self.convergence_controller = None

        # FCP Hybrid Layer Components (LLM + GNN + LoRA + KCA + SRG)
        self.hybrid_layer_config = HybridLayerConfig(
            hidden_dim=2560,
            num_layers=32,
            use_gnn=True,
            use_lora=True,
            use_kca=True,
            use_srg=True,
            injection_scale=0.1,
            lora_rank=8
        )
        self.hybrid_layer_manager = None
        self.hybrid_processor = None
        self.memory_snapshot = None

        # Инициализация
        self._init_tokenizer()
        self._init_fcp_components()
        self._init_hybrid_layers()
        self._init_memory_snapshot()
        self._init_pipeline(draft_model_path)
        self._init_lora_manager()

        print(f"[FCP] FCPPipelineV15 created: model={model_path}")
    
    def _init_fcp_components(self):
        """Инициализация FCP компонентов: KCA, SRG, Graph"""
        print("[FCP] Initializing FCP components...")
        
        # SRG (Semantic Relevance Gate)
        self.srg = SemanticRelevanceGate(self.fcp_config)
        print("[FCP] SRG initialized")
        
        # KCA (Knowledge Conscious Attention)
        self.kca = KnowledgeConsciousAttention(self.fcp_config)
        print("[FCP] KCA initialized")
        
        # Convergence Controller
        self.convergence_controller = ConvergenceController(self.fcp_config)
        print("[FCP] ConvergenceController initialized")
        
        # FractalGraphV2
        if self.graph_path and os.path.exists(self.graph_path):
            try:
                self.fractal_graph = FractalGraphV2.load(self.graph_path)
                print(f"[FCP] FractalGraphV2 loaded: {self.fractal_graph.node_count} nodes")
            except Exception as e:
                print(f"[FCP] Graph load failed: {e}, creating empty graph")
                self.fractal_graph = FractalGraphV2(config=self.fcp_config)
        else:
            self.fractal_graph = FractalGraphV2(config=self.fcp_config)
            print("[FCP] FractalGraphV2 created (empty)")
        
        print("[FCP] All FCP components initialized")

    def _init_hybrid_layers(self):
        """Инициализация гибридных слоёв (LLM + GNN + LoRA + KCA + SRG)"""
        print("[FCP] Initializing Hybrid Layers...")

        # HybridLayerProcessor для обработки запросов
        self.hybrid_processor = HybridLayerProcessor(self.hybrid_layer_config)
        print("[FCP] HybridLayerProcessor initialized")

        # HybridLayerManager для управления состоянием на каждом слое
        self.hybrid_layer_manager = HybridLayerManager(self.hybrid_layer_config)
        print("[FCP] HybridLayerManager initialized")

        # Загружаем обученный GNN энкодер если есть
        gnn_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'models', 'graph_encoder.pt'
        )
        if os.path.exists(gnn_path):
            success = self.hybrid_processor.load_trained_encoder(gnn_path)
            if success:
                print(f"[FCP] Loaded trained GNN encoder into HybridLayerProcessor")
                self.gnn_encoder = self.hybrid_processor.graph_encoder
            else:
                print(f"[FCP] Failed to load GNN encoder")
                self.gnn_encoder = None
        else:
            print(f"[FCP] GNN encoder not found at {gnn_path}")
            self.gnn_encoder = None

        # Добавляем FractalGraphV2 в гибридный менеджер
        if self.fractal_graph and self.fractal_graph.node_count > 0:
            nodes = []
            for i in range(self.fractal_graph.node_count):
                node_emb = self.fractal_graph.get_node(i)
                if node_emb is not None:
                    nodes.append({
                        'id': str(i),
                        'embedding': node_emb,
                        'content': f'Node {i}',
                        'metadata': {}
                    })
            if nodes:
                self.hybrid_layer_manager.set_global_graph(nodes)
                print(f"[FCP] Hybrid layer graph populated: {len(nodes)} nodes")

        print("[FCP] Hybrid layers initialized")

    def _init_memory_snapshot(self):
        """Инициализация MemorySnapshotIntegration - сохранение состояний всех слоёв в граф."""
        print("[FCP] Initializing MemorySnapshot...")

        try:
            from eva_ai.core.memory_snapshot_integration import MemorySnapshotIntegration

            self.memory_snapshot = MemorySnapshotIntegration(
                brain=self,
                fractal_graph=self.fractal_graph,
                config={
                    'enabled': True,
                    'snapshot_all_layers': True,
                    'num_layers': 32,
                    'save_to_graph': True
                }
            )

            print("[FCP] MemorySnapshotIntegration initialized (all layers)")
        except Exception as e:
            print(f"[FCP] MemorySnapshot init failed: {e}")
            self.memory_snapshot = None

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
            
            # SchedulerConfig - оптимизировано для одного запроса с сохранением контекста
            scheduler = ov_genai.SchedulerConfig()
            scheduler.cache_size = 16  # GB - больше кэш для длинных диалогов
            scheduler.max_num_seqs = 1
            scheduler.max_num_batched_tokens = 8192  # больше для длинных контекстов
            scheduler.enable_prefix_caching = True  # кэшируем префиксы (системный промпт)
            scheduler.use_cache_eviction = True

            # CacheEvictionConfig - сохраняем начало (системный промпт) и конец (последние токены)
            try:
                cache_eviction = ov_genai.CacheEvictionConfig(
                    start_size=256,      # системный промпт никогда не вытесняется
                    recent_size=1024,    # текущий контекст тоже
                    max_cache_size=4096, # общий размер кэша
                    aggregation_mode=ov_genai.AggregationMode.MEAN
                )
                scheduler.cache_eviction_config = cache_eviction
                print("[FCP] CacheEvictionConfig enabled: start=256, recent=1024")
            except Exception as e:
                print(f"[FCP] CacheEvictionConfig not available: {e}")
            
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
    
    def generate_streaming(self, prompt, max_new_tokens=4096, enable_thinking=True, callback=None, add_to_history=True, **kwargs):
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

            full_response_parts = []

            # Читаем события из очереди
            while True:
                try:
                    event = event_queue.get(timeout=0.1)
                    # Накапливаем текст ответа для истории
                    if event['type'] == 'chunk':
                        full_response_parts.append(event.get('text', ''))
                    yield event
                    if event['type'] == 'done':
                        break
                except queue.Empty:
                    if not gen_thread.is_alive():
                        break

            gen_thread.join()

            # Сохраняем в историю разговора
            full_response = ''.join(full_response_parts)
            if full_response and add_to_history:
                self.conversation_history.append({
                    "user": prompt,
                    "assistant": full_response
                })
                if len(self.conversation_history) > self.max_history:
                    self.conversation_history = self.conversation_history[-self.max_history:]
                self.stats["queries"] += 1
            
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
        enable_thinking: bool = True,
        return_metadata: bool = False,
        add_to_history: bool = True,
        **kwargs
    ) -> str:
        """Основной метод генерации"""
        self.stats["queries"] += 1

        # Подготовка промпта с историей
        chat_prompt = self._build_prompt(prompt, enable_thinking)

        # Генерация
        response = self._generate(chat_prompt, max_new_tokens, **kwargs)

        self.stats["injections"] += 1

        # Сохраняем в историю
        if add_to_history and response:
            self.conversation_history.append({
                "user": prompt,
                "assistant": response
            })
            # Ограничиваем историю
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]

        if return_metadata:
            return response, {"query_count": self.stats["queries"]}
        return response
    
    def _build_prompt(self, prompt: str, enable_thinking: bool) -> str:
        """Формирование промпта с историей разговора"""
        history_text = ""
        if self.conversation_history:
            for entry in self.conversation_history[-self.max_history:]:
                history_text += f"<|im_start|>user\n{entry['user']}<|im_end|>\n"
                history_text += f"<|im_start|>assistant\n{entry['assistant']}<|im_end|>\n"
        return f"{history_text}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

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
    
    def get_fcp_status(self) -> Dict:
        """Получить статус FCP компонентов"""
        status = {
            "kca_ready": self.kca is not None,
            "srg_ready": self.srg is not None,
            "graph_nodes": self.fractal_graph.node_count if self.fractal_graph else 0,
            "config": {
                "kca_max_cycles": self.fcp_config.kca_max_cycles,
                "kca_rho": self.fcp_config.kca_rho,
                "srg_cosine_threshold": self.fcp_config.srg_cosine_threshold,
                "srg_entropy_threshold": self.fcp_config.srg_entropy_threshold,
            }
        }

        # Hybrid Layer status
        if self.hybrid_layer_manager:
            hlm_status = self.hybrid_layer_manager.get_statistics()
            status["hybrid_layers"] = {
                "manager_ready": True,
                "total_layers": hlm_status["total_layers"],
                "graph_nodes": hlm_status.get("graph_nodes", 0)
            }
        else:
            status["hybrid_layers"] = {"manager_ready": False}

        if self.hybrid_processor:
            proc_status = self.hybrid_processor.get_status()
            status["hybrid_processor"] = proc_status
        else:
            status["hybrid_processor"] = {"initialized": False}

        return status
    
    def enrich_with_kca(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        initial_hidden_state: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        KCA обогащение скрытого состояния через граф знаний.
        
        Args:
            query_text: текст запроса
            query_embedding: [D] вектор запроса
            initial_hidden_state: [T, D] начальное скрытое состояние
            
        Returns:
            Tuple[np.ndarray, dict]: (обогащённое состояние, метаинформация)
        """
        if self.fractal_graph is None or self.fractal_graph.node_count == 0:
            return initial_hidden_state, {"status": "NO_GRAPH", "cycles": 0}
        
        if self.kca is None:
            return initial_hidden_state, {"status": "NO_KCA", "cycles": 0}
        
        # 1. Retrieve subgraph
        subgraph = self.fractal_graph.retrieve_subgraph(
            query_embedding,
            top_k=self.fcp_config.graph_top_k
        )
        
        if subgraph["embeddings"].shape[0] == 0:
            return initial_hidden_state, {"status": "NO_SUBGRAPH", "cycles": 0}
        
        # 2. SRG Evaluation
        response_vec = initial_hidden_state.mean(axis=0)
        mode, metrics = self.srg.evaluate(query_embedding, response_vec, np.zeros(100))
        
        if mode == "direct":
            return initial_hidden_state, {"status": "DIRECT", "mode": mode, "metrics": metrics}
        
        # 3. KCA cycles
        corrected_state, kca_info = self.kca.forward(initial_hidden_state, subgraph)
        
        return corrected_state, {
            "status": "KCA_COMPLETE",
            "mode": mode,
            "metrics": metrics,
            "kca": kca_info
        }
    
    def add_knowledge_node(self, text: str, embedding: np.ndarray, metadata: dict = None) -> int:
        """Добавить узел знаний в граф"""
        if self.fractal_graph is None:
            self.fractal_graph = FractalGraphV2(config=self.fcp_config)
        
        return self.fractal_graph.add_node(embedding, metadata)
    
    def retrieve_relevant_knowledge(self, query_embedding: np.ndarray, top_k: int = 5) -> dict:
        """Получить релевантные знания из графа"""
        if self.fractal_graph is None or self.fractal_graph.node_count == 0:
            return {"indices": [], "embeddings": [], "scores": []}
        
        return self.fractal_graph.retrieve_subgraph(query_embedding, top_k=top_k)


def create_fcp_pipeline(model_path: str, graph_path: str = None, **kwargs):
    """Factory function"""
    return FCPPipelineV15(model_path, graph_path, **kwargs)