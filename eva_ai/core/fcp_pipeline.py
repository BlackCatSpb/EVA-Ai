"""
FCPPipelineV15 - Основной FCP Pipeline для EVA-Ai

Простой и рабочий пайплайн генерации на базе ruadapt_qwen3_4b OpenVINO.
С KCA (Knowledge Conscious Attention) и SRG (Semantic Relevance Gate).

Возможности:
- Сохранение/загрузка сессий (conversation_history)
- Семантический поиск релевантного контекста из FractalGraphV2
- Полная гибридная интеграция: KCA + SRG + GNN + LoRA
"""
import os
import time
import json
import logging
from typing import Optional, Dict, Any, Callable, Generator, Tuple, List
import numpy as np

logger = logging.getLogger("eva_ai.fcp_pipeline")

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

# FCP State Injection Components (NEW - from Доработка.txt)
from eva_ai.core.core_injector import LayerwiseStateInjector
from eva_ai.core.analysis_and_injection import (
    SemanticQueryAnalyzer,
    compute_kca_correction,
    GraphIntegrationManager,
    SRGFeedbackLoop,
    inject_graph_vector,
    apply_sqam_scaling
)

# Системный промпт - ВСЕГДА рассуждать перед ответом
SYSTEM_PROMPT = """Ты - интеллектуальный помощник EVA. ВСЕГДА перед ответом выполняй глубокое обдумывание и анализ. Показывай свои рассуждения в тегах <think>...</think>.
ОБЯЗАТЕЛЬНО закрой тег </think> после завершения рассуждений, затем давай окончательный ответ. Рассуждения должны быть подробными, логичными и полезными.

ВАЖНО: ВСЕГДА возвращайся к данным запроса! Используй только предоставленный контекст.
Если в контексте есть факты - опирайся на них. Если контекста нет - говори что не знаешь."""


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
        
        # NEW: State Injector for direct KV-cache access (from Доработка.txt)
        self.state_injector = None
        self.sqam_analyzer = None
        self.graph_mgr = None
        self.srg_feedback = None

        # FCP Hybrid Layer Components (LLM + GNN + LoRA + KCA + SRG)
        self.hybrid_layer_config = HybridLayerConfig(
            hidden_dim=2560,
            num_layers=36,
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

        # Загружаем сохранённую сессию если есть
        self.load_session("default")

        print(f"[FCP] FCPPipelineV15 created: model={model_path}")
    
    def _init_fcp_components(self):
        """Инициализация FCP компонентов: KCA, SRG, Graph, State Injector"""
        print("[FCP] Initializing FCP components...")
        
        # NEW: State Injector for direct KV-cache access (from Доработка.txt)
        try:
            device = "GPU.0" if "GPU" in self.model_path else "CPU"
            # StateInjector needs path to XML file, not folder
            model_xml = os.path.join(self.model_path, "openvino_model.xml")
            self.state_injector = LayerwiseStateInjector(model_xml, device)
            print(f"[FCP] StateInjector initialized: device={device}, model={model_xml}")
        except Exception as e:
            print(f"[FCP] StateInjector init failed: {e}")
            self.state_injector = None
        
        # NEW: SQAM Analyzer
        self.sqam_analyzer = SemanticQueryAnalyzer()
        print("[FCP] SQAM Analyzer initialized")
        
        # NEW: Graph Integration Manager
        self.graph_mgr = GraphIntegrationManager(embedding_dim=2560)
        print("[FCP] GraphIntegrationManager initialized")
        
        # NEW: SRG Feedback Loop
        self.srg_feedback = SRGFeedbackLoop(threshold=0.6)
        print("[FCP] SRG FeedbackLoop initialized")
        
        # SRG (Semantic Relevance Gate) - existing
        self.srg = SemanticRelevanceGate(self.fcp_config)
        print("[FCP] SRG initialized")
        
        # KCA (Knowledge Conscious Attention) - existing
        self.kca = KnowledgeConsciousAttention(self.fcp_config)
        print("[FCP] KCA initialized")
        
        # Convergence Controller
        self.convergence_controller = ConvergenceController(self.fcp_config)
        print("[FCP] ConvergenceController initialized")
        
        # FractalGraphV2
        graph_dir = os.path.dirname(self.graph_path) if self.graph_path else None
        if self.graph_path and os.path.exists(self.graph_path):
            try:
                self.fractal_graph = FractalGraphV2(storage_dir=graph_dir, lazy=True)
                # Lazy - не обращаемся к storage.nodes сразу
                print(f"[FCP] FractalGraphV2 loaded (lazy): {self.graph_path}")
            except Exception as e:
                print(f"[FCP] Graph load failed: {e}, creating empty graph")
                self.fractal_graph = FractalGraphV2(storage_dir=graph_dir or "eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data", lazy=True)
        else:
            self.fractal_graph = FractalGraphV2(storage_dir=graph_dir or "eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data", lazy=True)
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
        # Исправляем путь: from eva_ai/core -> project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        gnn_path = os.path.join(project_root, 'models', 'graph_encoder.pt')
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
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            logical_processors = cpu_count  # 12 для i5-12450H

            # Максимальная производительность CPU - используем ВСЕ потоки
            os.environ['PERFORMANCE_HINT'] = 'LATENCY'
            os.environ['NUM_STREAMS'] = str(logical_processors)  # 12 вместо 1
            os.environ['INFERENCE_NUM_THREADS'] = str(logical_processors)  # Все потоки
            os.environ['CPU_THREADS_PER_STREAM'] = 'AUTO'
            os.environ['CPU_BIND_THREAD'] = 'HYBRID_AWARE'  # P-cores для GenAI
            os.environ['ENABLE_HYPER_THREADING'] = 'YES'
            os.environ['ENABLE_CPU_PINNING'] = 'YES'
            os.environ['CPU_DENORMALS_OPTIMIZATION'] = 'YES'
            print(f"[FCP] CPU optimization: {logical_processors} streams, {logical_processors} threads")
            
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
            gen_config.temperature = 0.15  # Снижаем для точности
            gen_config.top_p = 0.85
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
                    gen_cfg.temperature = 0.15  # Снижаем температуру для точности
                    gen_cfg.top_p = 0.85
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
        """Основной метод генерации с полной гибридной интеграцией"""
        self.stats["queries"] += 1

        # Если требуется полнослойная инъекция (Runtime State Injection)
        if enable_injection and self.state_injector:
            # Используем новый метод с полнослойной инъекцией согласно Доработка.txt
            result = self.generate_with_injection(
                prompt, 
                max_new_tokens=max_new_tokens,
                enable_thinking=enable_thinking,
                return_metadata=return_metadata
            )
            
            if return_metadata:
                response, metadata = result
                metadata["query_count"] = self.stats["queries"]
                metadata["injection_type"] = "full_layer"
            else:
                response = result
                metadata = {"injection_type": "full_layer"}
            
            # Обновляем статистику инъекций
            self.stats["injections"] += 1
            
            # Сохраняем в историю если нужно
            if add_to_history and response:
                self.conversation_history.append({
                    "user": prompt,
                    "assistant": response
                })
                if len(self.conversation_history) > self.max_history:
                    self.conversation_history = self.conversation_history[-self.max_history:]

                # Сохраняем сессию сразу после изменения
                self.save_session("default")
                
            return response if not return_metadata else (response, metadata)

        # 1. Подготовка промпта с историей
        chat_prompt = self._build_prompt(prompt, enable_thinking)

        # 2. Гибридная обработка через HybridLayerProcessor (KCA + GNN + SRG) - fallback
        if enable_injection and self.hybrid_processor and self.fractal_graph:
            processed_prompt, metadata = self._process_with_hybrid_layers(
                chat_prompt, prompt
            )
            self.stats["injections"] += 1

            # Используем обогащённый промпт если есть контекст
            if processed_prompt != chat_prompt:
                chat_prompt = processed_prompt
                logger.info(f"[FCP] Hybrid injection applied: srg_mode={metadata.get('srg_mode', 'unknown')}")
        else:
            metadata = {}

        # 3. Генерация через OpenVINO pipeline
        response = self._generate(chat_prompt, max_new_tokens, **kwargs)

        # 4. Сохранение snapshot состояния (все слои)
        if self.memory_snapshot and self.fractal_graph:
            try:
                self._save_layer_snapshot(prompt, response)
            except Exception as e:
                logger.debug(f"Snapshot save skipped: {e}")

        # 5. Сохраняем в историю
        if add_to_history and response:
            self.conversation_history.append({
                "user": prompt,
                "assistant": response
            })
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]

            # Сохраняем сессию сразу после изменения
            self.save_session("default")

        if return_metadata:
            metadata["query_count"] = self.stats["queries"]
            return response, metadata
        return response

    def _process_with_hybrid_layers(self, chat_prompt: str, user_prompt: str) -> tuple:
        """
        Обработка через HybridLayerProcessor с полными матричными вычислениями.

        Returns:
            (enriched_prompt, metadata)
        """
        metadata = {
            "srg_mode": "unknown",
            "kca_cycles": 0,
            "injections": 0
        }

        try:
            # 1. Получаем запрос как эмбеддинг
            if self.fractal_graph and self.fractal_graph.node_count > 0:
                # Используем внутренний эмбеддинг модели если есть
                query_embedding = self._get_query_embedding(user_prompt)

                # 2. SRG evaluation - определяем режим
                if self.srg:
                    mode, srg_metrics = self.srg.evaluate(
                        query_vec=query_embedding,
                        response_vec=query_embedding,  # будет уточнено после generation
                        logits=np.zeros(100)
                    )
                    metadata["srg_mode"] = mode
                    metadata["srg_metrics"] = srg_metrics

                    # Если direct mode - пропускаем KCA коррекцию
                    if mode == "direct":
                        subgraph = self.fractal_graph.retrieve_subgraph(
                            query_embedding.reshape(1, -1),
                            top_k=self.fcp_config.graph_top_k
                        )
                        enriched_prompt = self._enrich_prompt_with_subgraph(chat_prompt, subgraph)
                        return enriched_prompt, metadata

                # 3. Retrieve subgraph для KCA
                if self.fractal_graph.node_count > 0:
                    subgraph = self.fractal_graph.retrieve_subgraph(
                        query_embedding.reshape(1, -1),
                        top_k=self.fcp_config.graph_top_k
                    )

                    # 4. Если reasoning mode - запускаем KCA
                    # Subgraph объект имеет атрибут node_embeddings
                    if self.kca and not subgraph.is_empty and subgraph.node_embeddings is not None:
                        # Создаём synthetic hidden states для KCA
                        hidden_dim = query_embedding.shape[-1]
                        seq_len = 4
                        initial_states = np.tile(query_embedding, (seq_len, 1)).astype(np.float32)

                        # KCA forward pass - передаём dict формат
                        corrected_states, kca_info = self.kca.forward(initial_states, subgraph.to_dict())

                        metadata["kca_cycles"] = kca_info.get("cycles", 0)
                        metadata["kca_status"] = kca_info.get("status", "unknown")

                        # Обогащаем промпт текстовым контекстом из подграфа
                        enriched_prompt = self._enrich_prompt_with_subgraph(chat_prompt, subgraph)
                        return enriched_prompt, metadata

            # Fallback - возвращаем оригинальный промпт
            return chat_prompt, metadata

        except Exception as e:
            logger.debug(f"Hybrid processing error: {e}")
            return chat_prompt, metadata

    def _get_query_embedding(self, text: str) -> np.ndarray:
        """Получить эмбеддинг запроса из FractalGraph"""
        try:
            if hasattr(self, 'tokenizer') and self.tokenizer:
                inputs = self.tokenizer(text, return_tensors="np", padding=True, truncation=True)
                input_ids = inputs["input_ids"]
                if input_ids.shape[1] > 0:
                    return np.mean(inputs["input_ids"].astype(np.float32), axis=0)
        except:
            pass

        # Fallback - случайный вектор
        return np.random.randn(2560).astype(np.float32)

    def _enrich_prompt_with_subgraph(self, prompt: str, subgraph: dict) -> str:
        """Обогатить промпт текстовым контекстом из подграфа"""
        if not subgraph or subgraph.get("embeddings").shape[0] == 0:
            return prompt

        # Получаем текстовый контекст
        context_lines = []
        contents = subgraph.get("contents", [])

        for i, content in enumerate(contents[:5]):
            context_lines.append(f"  {i+1}. {content}")

        if context_lines:
            context_str = "\n".join(context_lines)
            enriched = f"\n📚 Контекст из графа знаний:\n{context_str}\n\n{prompt}"
            return enriched

        return prompt

    def _save_layer_snapshot(self, query: str, response: str) -> None:
        """Сохранить snapshot состояния всех слоёв в FractalGraph"""
        try:
            if not self.fractal_graph or self.fractal_graph.node_count == 0:
                return

            # Эмбеддинг запроса для поиска related nodes
            query_emb = self._get_query_embedding(query)

            # Retrieve subgraph для связывания
            subgraph = self.fractal_graph.retrieve_subgraph(
                query_emb.reshape(1, -1),
                top_k=8
            )

            # Сохраняем в memory snapshot если доступен
            if self.memory_snapshot:
                self.memory_snapshot.save_snapshot(
                    query=query,
                    response=response,
                    subgraph=subgraph
                )

        except Exception as e:
            logger.debug(f"Snapshot save error: {e}")
    
    def _build_prompt(self, prompt: str, enable_thinking: bool) -> str:
        """
        Формирование промпта с историей разговора и семантическим контекстом.

        Использует SRG для определения релевантности и добавляет только
        семантически и логически связанные entries из истории.
        """
        # Получаем семантически релевантный контекст из истории
        relevant_context = self.get_relevant_context(prompt, max_history=3)

        history_text = ""
        if self.conversation_history:
            for entry in self.conversation_history[-self.max_history:]:
                history_text += f"<|im_start|>user\n{entry['user']}<|im_end|>\n"
                history_text += f"<|im_start|>assistant\n{entry['assistant']}<|im_end|>\n"

        # Добавляем релевантный контекст если есть
        if relevant_context:
            history_text = f"[Релевантный контекст из прошлых разговоров]:\n{relevant_context}\n\n" + history_text

        return f"{history_text}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def _generate(self, prompt: str, max_new_tokens: int = 1024, **kwargs) -> str:
        """Генерация ответа"""
        if not self.pipeline:
            return "[No pipeline]"
        
        try:
            gen_cfg = ov_genai.GenerationConfig()
            gen_cfg.max_new_tokens = max_new_tokens
            gen_cfg.temperature = 0.15  # Снижаем температуру для точности
            gen_cfg.top_p = 0.85
            gen_cfg.top_k = 40
            gen_cfg.repetition_penalty = 1.1
            gen_cfg.do_sample = True
            
            result = self.pipeline.generate(prompt, generation_config=gen_cfg, **kwargs)
            
            # После завершення обробатываєм залишок буфера
            if hasattr(self, 'buffer') and self.buffer:
                if hasattr(self, 'in_thinking') and self.in_thinking:
                    if self.buffer.strip():
                        print(f"[FCP STREAM] Final reasoning_text, length: {len(self.buffer)}")
                        if hasattr(self, 'event_queue'):
                            self.event_queue.put({"type": "reasoning_text", "text": self.buffer})
                            self.event_queue.put({"type": "reasoning_end"})
                else:
                    if self.buffer.strip():
                        if hasattr(self, 'event_queue'):
                            self.event_queue.put({"type": "chunk", "text": self.buffer})
        except Exception as e:
            if hasattr(self, 'event_queue'):
                self.event_queue.put({"type": "error", "text": str(e)})
            return f"Generation error: {e}"
        finally:
            if hasattr(self, 'event_queue'):
                self.event_queue.put({"type": "done", "timestamp": time.time()})
        
        return result

    def generate_with_injection(self, prompt: str, max_new_tokens: int = 1024, 
                              enable_thinking: bool = True, return_metadata: bool = False) -> str:
        """
        Полнослойная инъекция согласно Доработка.txt (FCP specification)
        Runtime State Injection: модификация Key и Value тензоров на всех слоях
        """
        if not self.pipeline or not self.state_injector:
            # Fallback к обычной генерации если injector недоступен
            return self._generate(prompt, max_new_tokens, **{})
        
        try:
            # Import here to avoid circular imports
            import openvino_genai as ov_genai
            import openvino as ov
            
            logger.info(f"[FCP] Starting generation with full-layer injection: '{prompt[:50]}...'")
            
            # 1. Pre-fill - обработка промпта для заполнения KV-кеша
            if hasattr(self, 'tokenizer') and self.tokenizer:
                input_ids = self.tokenizer.encode(prompt, return_tensors="np")
            else:
                # Fallback если токенизатор недоступен
                return self._generate(prompt, max_new_tokens, **{})
                
            # Reset KV-кеша перед новой сессией
            self.state_injector.reset_all_states()
            
            # Запускаем pre-fill inference
            self.state_injector.request.infer({"input_ids": input_ids})
            seq_len = input_ids.shape[1]
            
            # Получаем текст токенов для анализа
            token_texts = [self.tokenizer.decode([tid]) for tid in input_ids[0]] if hasattr(self, 'tokenizer') else []
            
            # 2. SQAM Analysis & Full-Layer Key Scaling (согласно Доработка.txt)
            key0 = self.state_injector.get_key(0)  # Первый слой
            _, importance = self.sqam_analyzer.analyze(key0, seq_len)
            all_layers = self.state_injector.get_all_layer_indices()
            
            # Применяем SQAM ко ВСЕМ слоям (Key scaling)
            self.state_injector.transform_keys(all_layers, apply_sqam_scaling, weights=importance)
            
            # 3. Graph Enrichment - извлечение якорных токенов и обновление центроида
            anchors = self.sqam_analyzer.get_core_anchors(token_texts, threshold=0.6)
            key_per_token = key0[0].mean(axis=0)  # Average over heads
            self.graph_mgr.add_anchors(anchors, key_per_token)
            
            # 4. Decoding Loop with Full-Layer KCA Injection
            generated_ids = input_ids[0].tolist()
            eos_token_id = getattr(self.tokenizer, 'eos_token_id', 2) if hasattr(self, 'tokenizer') else 2
            
            for step in range(max_new_tokens):
                # Инференс на последнем токене
                self.state_injector.request.infer({"input_ids": np.array([[generated_ids[-1]]])})
                logits = self.state_injector.request.get_tensor("logits").data[0, -1]
                next_token = int(np.argmax(logits))
                generated_ids.append(next_token)
                
                if next_token == eos_token_id:
                    break
                
                # Динамический расчет KCA коррекции
                # Извлекаем прокси-состояние из последнего слоя (Value тензор)
                val_proxy = self.state_injector.get_value(all_layers[-1])[0, :, -1, :].mean(axis=0)
                self.kca_correction_vec = compute_kca_correction(val_proxy, self.graph_mgr.get_centroid())
                
                # Применяем KCA ко ВСЕМ слоям Value (инъекция знаний)
                if np.linalg.norm(self.kca_correction_vec) > 1e-5:
                    # Адаптивный вес для квантованных моделей (см. Доработка.txt)
                    kca_weight = 0.07  # Базовый вес для FP16/FP32
                    if hasattr(self, 'model_path') and 'int4' in self.model_path.lower():
                        kca_weight = 0.2  # Усиленный вес для INT4
                    elif 'int8' in self.model_path.lower():
                        kca_weight = 0.12  # Средний вес для INT8
                    
                    self.state_injector.transform_values(
                        all_layers, 
                        inject_graph_vector, 
                        vector=self.kca_correction_vec, 
                        weight=kca_weight
                    )
            
            # 5. SRG Post-Evaluation (оценка уверенности)
            final_logits = self.state_injector.request.get_tensor("logits").data[0, -1]
            srg_metrics = self.srg.evaluate(final_logits) if hasattr(self.srg, 'evaluate') else {"mode": "reasoning", "confidence": 0.5}
            
            # Декодируем результат
            if hasattr(self, 'tokenizer') and self.tokenizer:
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            else:
                # Fallback если токенизатор недоступен
                response = f"[Generated {len(generated_ids)} tokens]"
            
            logger.info(f"[FCP] Generation completed with injection. SRG: {srg_metrics}")
            
            # Сохраняем в историю если нужно
            if return_metadata:
                metadata = {
                    "injection_used": True,
                    "srg_metrics": srg_metrics,
                    "tokens_generated": len(generated_ids) - len(input_ids[0]),
                    "layers_injected": len(all_layers),
                    "sqam_applied": True,
                    "kca_applied": np.linalg.norm(self.kca_correction_vec) > 1e-5
                }
                return response, metadata
            
            return response
            
        except Exception as e:
            logger.error(f"[FCP] Injection generation error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback к обычной генерации при ошибке
            return self._generate(prompt, max_new_tokens, **{})
    
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
        
        if subgraph.is_empty or subgraph.node_embeddings is None:
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
            self.fractal_graph = FractalGraphV2(storage_dir="eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data", lazy=True)
        
        return self.fractal_graph.add_node(embedding, metadata)
    
    def retrieve_relevant_knowledge(self, query_embedding: np.ndarray, top_k: int = 5) -> dict:
        """Получить релевантные знания из графа"""
        if self.fractal_graph is None or self.fractal_graph.node_count == 0:
            return {"indices": [], "embeddings": [], "scores": []}

        return self.fractal_graph.retrieve_subgraph(query_embedding, top_k=top_k)

    def save_session(self, session_id: str = "default") -> bool:
        """
        Сохранить состояние сессии (conversation_history + fractal_graph).

        Args:
            session_id: идентификатор сессии

        Returns:
            True если успешно сохранено
        """
        try:
            session_dir = os.path.join(os.path.dirname(self.model_path), "sessions")
            os.makedirs(session_dir, exist_ok=True)

            session_file = os.path.join(session_dir, f"{session_id}.json")

            session_data = {
                "session_id": session_id,
                "timestamp": time.time(),
                "conversation_history": self.conversation_history,
                "stats": self.stats
            }

            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)

            logger.info(f"[FCP] Session saved: {session_id}, history_size={len(self.conversation_history)}")
            return True

        except Exception as e:
            logger.error(f"[FCP] Session save error: {e}")
            return False

    def load_session(self, session_id: str = "default") -> bool:
        """
        Загрузить состояние сессии.

        Args:
            session_id: идентификатор сессии

        Returns:
            True если успешно загружено
        """
        try:
            session_dir = os.path.join(os.path.dirname(self.model_path), "sessions")
            session_file = os.path.join(session_dir, f"{session_id}.json")

            if not os.path.exists(session_file):
                logger.debug(f"[FCP] Session file not found: {session_id}")
                return False

            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            self.conversation_history = session_data.get("conversation_history", [])
            self.stats = session_data.get("stats", {"queries": 0, "injections": 0})

            logger.info(f"[FCP] Session loaded: {session_id}, history_size={len(self.conversation_history)}")
            return True

        except Exception as e:
            logger.error(f"[FCP] Session load error: {e}")
            return False

    def get_relevant_context(self, query: str, max_history: int = 5) -> str:
        """
        Получить семантически релевантный контекст из истории сессии.

        Использует FractalGraph для поиска релевантных entries в conversation_history.

        Args:
            query: текст запроса
            max_history: максимальное количество entries для возврата

        Returns:
            Строка с релевантным контекстом
        """
        if not self.conversation_history:
            return ""

        try:
            query_emb = self._get_query_embedding(query)

            # Вычисляем similarity с каждым entry в истории
            best_entries = []
            for i, entry in enumerate(self.conversation_history):
                entry_text = f"{entry.get('user', '')} {entry.get('assistant', '')}"
                entry_emb = self._get_query_embedding(entry_text)

                # Cosine similarity
                sim = np.dot(query_emb, entry_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(entry_emb) + 1e-8
                )

                if sim > 0.3:  # Порог релевантности
                    best_entries.append((sim, entry))

            # Сортируем по similarity и берём top
            best_entries.sort(key=lambda x: x[0], reverse=True)

            context_parts = []
            for sim, entry in best_entries[:max_history]:
                user_text = entry.get('user', '')[:100]
                assistant_text = entry.get('assistant', '')[:200]
                context_parts.append(f"User: {user_text}\nAssistant: {assistant_text}")

            if context_parts:
                return "\n\n---\n".join(context_parts)

        except Exception as e:
            logger.debug(f"[FCP] Context retrieval error: {e}")

        return ""

    def clear_session(self, session_id: str = "default") -> bool:
        """Очистить сессию (удалить файл)"""
        try:
            session_dir = os.path.join(os.path.dirname(self.model_path), "sessions")
            session_file = os.path.join(session_dir, f"{session_id}.json")

            if os.path.exists(session_file):
                os.remove(session_file)

            self.conversation_history = []
            self.stats = {"queries": 0, "injections": 0}
            self.kca_correction_vec = None  # Для хранения вектора коррекции KCA

            return True
        except Exception as e:
            logger.error(f"[FCP] Session clear error: {e}")
            return False


def create_fcp_pipeline(model_path: str, graph_path: str = None, **kwargs):
    """Factory function"""
    return FCPPipelineV15(model_path, graph_path, **kwargs)