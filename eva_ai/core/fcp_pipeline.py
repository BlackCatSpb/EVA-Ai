"""
FCPPipelineV15 - Основной FCP Pipeline для EVA-Ai

Простой и рабочий пайплайн генерации на базе ruadapt_qwen3_4b OpenVINO.
С KCA (Knowledge Conscious Attention) и SRG (Semantic Relevance Gate).

Возможности:
- Сохранение/загрузка сессий (conversation_history)
- Семантический поиск релевантного контекста из FractalGraphV2
- Полная гибридная интеграция: KCA + SRG + GNN + LoRA

Соответствует спецификации: EVA.txt (разделы 2.1-2.3, 3, 4, 5, 8)
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
    LearningGraphManager,
    ShadowLoRAManager
)

# FractalGraphV2 Singleton
from eva_ai.memory.fractal_graph_v2 import get_fractal_graph

# Reasoning Chain (NEW - для накопления цепочки рассуждений)
try:
    from eva_ai.fcp_core.reasoning_chain import ReasoningChain, ReasoningChainManager
except ImportError:
    ReasoningChain = None
    ReasoningChainManager = None

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

# Self-Evaluation (SRG-like)
from eva_ai.core.self_evaluation import SelfEvaluation, EvaluationResult

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

# NEW: KCA and Graph Injection (EVA.txt sections 8.2, 9.3)
from eva_ai.fcp_core.kca_detector import KCADetector
from eva_ai.fcp_core.graph_injection import GraphStateInjector, InjectionConfig

# Системный промпт - ВСЕГДА рассуждать перед ответом
SYSTEM_PROMPT = r"""Ты - интеллектуальный помощник EVA. ВСЕГДА перед ответом выполняй глубокое обдумывание и анализ. Показывай свои рассуждения в тегах <think>...
ОБЯЗАТЕЛЬНО закрой тег  после завершения рассуждений, затем давай окончательный ответ. Рассуждения должны быть подробными, логичными и полезными.

ВАЖНО: ВСЕГДА возвращайся к данным запроса! Используй только предоставленный контекст.
Если в контексте есть факты - опирайся на них. Если контекста нет - говори что не знаешь.

МАТЕМАТИКА: Для формул используй LaTeX нотацию с $...$ для inline и $$...$$ для block формул.
Пример: $a^2 + b^2 = c^2$ или $$\int_0^1 x^2 dx$$
Если LaTeX недоступен - используй Unicode символы: ÷ × √ ∞ ∑ ∫ ∂"""


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


class FCPipeline:
    """Основной FCP Pipeline с KCA и SRG"""

    def __init__(
        self,
        model_path: str,
        graph_path: str = None,
        gnn_ov_path: Optional[str] = None,
        lora_dir: Optional[str] = None,
        draft_model_path: Optional[str] = None,
        max_history: int = 10,
        brain: Any = None  # Ссылка на brain для доступа к hybrid_cache
    ):
        self.model_path = model_path
        self.graph_path = graph_path
        self.gnn_ov_path = gnn_ov_path
        self.lora_dir = lora_dir or "C:/Users/black/OneDrive/Desktop/FCP/lora_adapters"
        self.max_history = max_history
        self.brain = brain  # Ссылка на brain для интеграции с HybridTokenCache

        # === Единая конфигурация генерации (с гибридным кэшем) ===
        self.generation_config = {
            "temperature": 0.15,
            "top_p": 0.85,
            "top_k": 40,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "max_new_tokens": 4096,  # Максимум для баланса
            "min_new_tokens": 1
        }

        # === Единая конфигурация KV кэша (интегрирована с HybridTokenCache) ===
        self.hybrid_cache = None
        hybrid_cache_config = self._get_hybrid_cache_config()
        
        self.kv_cache_config = {
            "cache_size_gb": hybrid_cache_config.get("cache_size_gb", 8),  # 8GB - безопасное
            "max_num_seqs": 1,
            "max_num_batched_tokens": 4096,  # 4K batch
            "enable_prefix_caching": True,
            "use_cache_eviction": True,
            "cache_eviction_start_size": 256,
            "cache_eviction_recent_size": 512,
            "cache_eviction_max_size": 1024,
            "linked_hybrid_cache": hybrid_cache_config.get("enabled", False)
        }

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

        # LoRA Hot-Reload tracking
        self._lora_adapter_path = None
        self._lora_file_mtime = None
        self._pending_adapter = None
        
        # GNN Hot-Reload tracking
        self._gnn_hybrid_path = None
        self._gnn_file_mtime = None
        
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

        # ONLINE TRAINER - Фоновое обучение GNN и LoRA
        self.online_trainer = None
        self._init_online_trainer()

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
    
    def _init_online_trainer(self):
        """Инициализация онлайн-обучения GNN и LoRA. НЕ запускаем - только создаём."""
        try:
            from eva_ai.fcp_core.online_trainer import OnlineTrainerManager
            
            self.online_trainer = OnlineTrainerManager({
                "enabled": True,
                "gnn_step_interval": 15,
                "gnn_save_interval": 200,
                "lora_step_interval": 30,
                "lora_save_interval": 100,
                "use_gpu": True  # GPU-only training (no CPU fallback per EVA.txt)
            }, brain=self)
            
            # Connect GNN encoder to GraphStateInjector if available
            if hasattr(self.online_trainer, 'gnn_trainer') and self.online_trainer.gnn_trainer:
                gnn_encoder = self.online_trainer.gnn_trainer.get_encoder()
                if gnn_encoder and hasattr(self, 'graph_state_injector') and self.graph_state_injector:
                    self.graph_state_injector.gnn_encoder = gnn_encoder
                    logger.info("[FCP] GNN encoder from OnlineTrainer connected to GraphStateInjector")
            
            # НЕ запускаем автоматически - запустим после старта сервера
            print("[FCP] Online Trainer initialized (will start after server startup)")
            
        except Exception as e:
            print(f"[FCP] Online Trainer SKIPPED: {e}")
            self.online_trainer = None
    
    def start_online_training(self):
        """Запустить онлайн-обучение после старта сервера."""
        print(f"[FCP] start_online_training called, online_trainer={self.online_trainer}")
        if self.online_trainer:
            self.online_trainer.start()
            print("[FCP] Online Trainer started: GNN + LoRA training")
            # Логируем статус
            status = self.online_trainer.get_status()
            print(f"[FCP] GNN: {status['gnn']['ready']}, LoRA: {status['lora']['ready']}")
        else:
            print("[FCP] online_trainer is None - cannot start!")
    
    def _init_fcp_components(self):
        """Инициализация FCP компонентов: KCA, SRG, Graph, State Injector"""
        print("[FCP] Initializing FCP components...")
        
        # NEW: State Injector for direct KV-cache access (from Доработка.txt)
        self.state_injector = None
        try:
            device = "GPU.0" if "GPU" in self.model_path else "CPU"
            # StateInjector needs path to XML file, not folder
            # Ищем openvino_model.xml в различных возможных locations
            possible_paths = [
                os.path.join(self.model_path, "openvino_model.xml"),
                os.path.join(self.model_path, "model.ov", "openvino_model.xml"),
                os.path.join(self.model_path, "openvino_model", "openvino_model.xml"),
                os.path.join(os.path.dirname(self.model_path), "fmf_model", "model.ov", "openvino_model.xml"),
            ]
            
            model_xml = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_xml = path
                    print(f"[FCP] StateInjector: found model.xml at {path}")
                    break
            
            if not model_xml:
                print(f"[FCP] StateInjector SKIPPED: model.xml not found in any of: {possible_paths}")
            else:
                self.state_injector = LayerwiseStateInjector(model_xml, device)
                print(f"[FCP] StateInjector initialized: device={device}, model={model_xml}")
                
                # Проверяем что инжектор работает
                if hasattr(self.state_injector, '_layer_indices'):
                    print(f"[FCP] StateInjector has {len(self.state_injector._layer_indices)} layers")
                    
                    # NEW: Initialize GraphStateInjector with state_injector
                    # Full-layer graph injection (EVA.txt section 8.2)
                    try:
                        gnn_encoder = getattr(self, 'gnn_encoder', None) or getattr(self, 'hybrid_processor', None)
                        if gnn_encoder and hasattr(gnn_encoder, 'graph_encoder'):
                            gnn_encoder = gnn_encoder.graph_encoder
                        
                        self.graph_state_injector = GraphStateInjector(
                            state_injector=self.state_injector,
                            gnn_encoder=gnn_encoder,
                            kca_detector=self.kca_detector,
                            config=InjectionConfig(
                                graph_correction_strength=0.15,
                                key_scaling_strength=0.10,
                                activation_gate_threshold=0.85,
                                use_gate_weights=True,
                                num_layers=len(self.state_injector._layer_indices)
                            )
                        )
                        print(f"[FCP] GraphStateInjector initialized with {len(self.state_injector._layer_indices)} layers")
                    except Exception as e:
                        print(f"[FCP] GraphStateInjector init failed: {e}")
                        self.graph_state_injector = None
        except Exception as e:
            print(f"[FCP] StateInjector FAILED: {type(e).__name__}: {e}")
            self.state_injector = None
        
        # NEW: FCP Inference API
        self.fcp_api = None
        try:
            from eva_ai.fcp_core.fcp_inference_api import FCPInferenceAPI
            self.fcp_api = FCPInferenceAPI(model_path=self.model_path, device=device, max_seq_len=2048)
            if self.fcp_api.is_initialized():
                logger.info("[FCP] FCP Inference API initialized")
            else:
                self.fcp_api = None
        except Exception as e:
            logger.info(f"[FCP] FCP Inference API not available: {e}")
        
        # NEW: SQAM Analyzer
        self.sqam_analyzer = SemanticQueryAnalyzer()
        print("[FCP] SQAM Analyzer initialized")
        
        # NEW: Graph Integration Manager (связываем с fractal_graph после его создания)
        # Пока создаём без ссылки, позже обновим
        self.graph_mgr = GraphIntegrationManager(embedding_dim=2560, fractal_graph=None)
        print("[FCP] GraphIntegrationManager initialized")
        
        # NEW: SRG Feedback Loop
        self.srg_feedback = SRGFeedbackLoop(threshold=0.6)
        print("[FCP] SRG FeedbackLoop initialized")
        
        # NEW: KCA Detector (EVA.txt раздел 9.3)
        # Knowledge Cognitive Analyzer для обнаружения лакун и противоречий
        self.kca_detector = KCADetector(
            embedding_dim=256,
            lacuna_threshold=0.3,
            contradiction_threshold=0.4,
            correction_strength=0.15
        )
        print("[FCP] KCADetector initialized")
        
        # NEW: GraphStateInjector (EVA.txt раздел 8.2)
        # Полнослойная инъекция графа в KV-кеш
        # Инициализируется позже когда появится state_injector
        self.graph_state_injector = None
        print("[FCP] GraphStateInjector (lazy init)")
        
        # === Activation Gate (ранний выход) согласно EVA.txt раздел 2.1 ===
        # Если накопленная уверенность > порог → пропуск последующих слоёв
        # Даёт до 85% ускорения на простых запросах
        self.activation_gate_config = {
            'early_exit_threshold': 0.85,  # Порог уверенности для early exit
            'min_tokens_for_check': 5,      # Минимум токенов перед проверкой
            'confidence_window': 3,          # Окно для сглаживания уверенности
            'accumulated_confidence': 0.0,   # Накопленная уверенность
            'early_exits_count': 0,          # Счётчик ранних выходов
        }
        print(f"[FCP] Activation Gate initialized: threshold={self.activation_gate_config['early_exit_threshold']}")
        
        # === KCA Gate (γ) согласно EVA.txt раздел 3.2-3.3 ===
        # Монитор насыщения: если γ < 0.05 за 2 итерации → завершение KCA
        self.kca_gate_config = {
            'gate_threshold': 0.05,          # Порог отклонения коррекции
            'min_iterations': 2,             # Минимум итераций для проверки
            'damping_factor': 0.85,          # ρ^t для экспоненциального снижения
            'gate_history': [],              # История значений γ
            'gamma': 0.5,                    # Текущее значение гейта
            'kca_iterations': 0,             # Счётчик итераций KCA
            'kca_rejected': False,           # Флаг отклонения коррекции
            # Для обнаружения осцилляций (EVA.txt раздел 3.3)
            'state_history': [],             # История состояний для обнаружения осцилляций
            'state_change_history': [],      # История изменений состояний
        }
        print(f"[FCP] KCA Gate initialized: threshold={self.kca_gate_config['gate_threshold']}, damping={self.kca_gate_config['damping_factor']}")
        
        # SRG (Semantic Relevance Gate) - existing
        self.srg = SemanticRelevanceGate(self.fcp_config)
        print("[FCP] SRG initialized")
        
        # KCA (Knowledge Conscious Attention) - existing
        self.kca = KnowledgeConsciousAttention(self.fcp_config)
        print("[FCP] KCA initialized")
        
        # Convergence Controller
        self.convergence_controller = ConvergenceController(self.fcp_config)
        print("[FCP] ConvergenceController initialized")
        
        # FractalGraphV2 (use singleton to avoid duplicate initialization)
        graph_dir = os.path.dirname(self.graph_path) if self.graph_path else None
        graph_dir = graph_dir or "eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data"
        try:
            self.fractal_graph = get_fractal_graph(storage_dir=graph_dir, lazy=True)
            print(f"[FCP] FractalGraphV2 singleton loaded: {self.fractal_graph.node_count} nodes")
        except Exception as e:
            print(f"[FCP] FractalGraphV2 singleton init failed: {e}")
            self.fractal_graph = None
        
        # Связываем GraphIntegrationManager с FractalGraphV2
        if self.graph_mgr:
            self.graph_mgr.fractal_graph = self.fractal_graph
            print("[FCP] GraphIntegrationManager linked to FractalGraphV2")
        
        # === ScenarioTCM (согласно EVA.txt раздел 6.3) ===
        # Сохраняет цепочки диалогов как сценарии в графе
        from eva_ai.memory.scenario_tcm import ScenarioTCM
        self.scenario_tcm = ScenarioTCM(graph=self.fractal_graph)
        print("[FCP] ScenarioTCM initialized")
        
        # === ConceptMiner (согласно EVA.txt раздел 7.1) ===
        # Автономный концептуальный вывод, обнаружение семантических лакун
        try:
            from eva_ai.knowledge.concept_miner import ConceptMiner
            self.concept_miner = ConceptMiner(brain=self)
            print("[FCP] ConceptMiner initialized")
        except Exception as e:
            print(f"[FCP] ConceptMiner init failed: {e}")
            self.concept_miner = None
        
        # === ContradictionDetector (согласно EVA.txt раздел 7.2) ===
        # Обнаружение противоречий в графе знаний
        self.contradiction_detector = None
        try:
            from eva_ai.contradiction.detect_core import ContradictionDetector
            self.contradiction_detector = ContradictionDetector(
                knowledge_graph=self.fractal_graph if hasattr(self, 'fractal_graph') else None,
                detection_threshold=0.65
            )
            print("[FCP] ContradictionDetector initialized")
        except Exception as e:
            print(f"[FCP] ContradictionDetector init failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        
        # === LearningOrchestrator (согласно EVA.txt раздел 7.3) ===
        # Оркестратор обучения для LoRA адаптеров
        self.learning_orchestrator = None
        lora_mgr = getattr(self, 'lora_manager', None)
        try:
            from eva_ai.fcp_core.learning_orchestrator import LearningOrchestrator
            from eva_ai.knowledge.fcp_learning_manager import LearningGraphManager
            learning_manager = LearningGraphManager()
            self.learning_orchestrator = LearningOrchestrator(
                learning_manager=learning_manager,
                lora_manager=lora_mgr
            )
            print("[FCP] LearningOrchestrator initialized")
        except Exception as e:
            print(f"[FCP] LearningOrchestrator init failed: {type(e).__name__}: {e}")
        
        # === UES (Universal Execution Subsystem) (согласно EVA.txt раздел 8.3) ===
        # Оптимизация и контроль исполнения на произвольном оборудовании
        try:
            from eva_ai.fcp_ues import UES
            self.ues = UES(model_path=self.model_path)
            print(f"[FCP] UES initialized: {len(self.ues.topology.units)} compute units")
        except Exception as e:
            print(f"[FCP] UES init failed: {e}")
            self.ues = None
        
        # === ContextualTokenizer (согласно EVA.txt раздел 2.1) ===
        # Контекстная токенизация с учётом графа знаний
        try:
            from eva_ai.fcp_core.contextual_tokenizer import ContextualTokenizer
            # Получаем базовый токенизатор из конфигурации
            base_tokenizer = self._get_base_tokenizer()
            if base_tokenizer:
                self.contextual_tokenizer = ContextualTokenizer(
                    base_tokenizer=base_tokenizer,
                    fractal_graph=self.fractal_graph,
                    embedding_model=self._get_embedding_model()
                )
                print(f"[FCP] ContextualTokenizer initialized: vocab_size={self.contextual_tokenizer.get_vocab_size()}")
            else:
                self.contextual_tokenizer = None
                print("[FCP] ContextualTokenizer skipped: no base tokenizer")
        except Exception as e:
            print(f"[FCP] ContextualTokenizer init failed: {e}")
            self.contextual_tokenizer = None
        
        # === CrossAttentionFusion (согласно EVA.txt раздел 2.1) ===
        # Слияние через cross-attention между моделью и графом
        try:
            from eva_ai.fcp_core.cross_attention import CrossAttentionFusion
            self.cross_attention = CrossAttentionFusion(
                hidden_dim=2560,  # Размерность скрытых состояний
                graph_dim=384,     # Размерность эмбеддингов графа
                num_heads=8
            )
            print(f"[FCP] CrossAttentionFusion initialized: heads=8")
        except Exception as e:
            print(f"[FCP] CrossAttentionFusion init failed: {e}")
            self.cross_attention = None
        
        # === TrainableGate (согласно EVA.txt раздел 2.1) ===
        # Обучаемый гейт для слияния источников
        try:
            from eva_ai.fcp_core.trainable_gate import TrainableGate
            self.trainable_gate = TrainableGate(
                input_dim=2560,
                hidden_dim=128,
                num_sources=3  # модель, граф, KCA
            )
            print(f"[FCP] TrainableGate initialized: sources=3")
        except Exception as e:
            print(f"[FCP] TrainableGate init failed: {e}")
            self.trainable_gate = None
        
        # === ExpertSystem (согласно EVA.txt) ===
        # Мультиагентная система обсуждения
        try:
            from eva_ai.tools.fcp.expert_system import ExpertSystem, Expert
            # Создаем экспертов (используем текущий pipeline как базу)
            experts = [
                Expert("base", self._generate, None),
                Expert("creative", self._generate, "creative_lora"),
                Expert("factual", self._generate, "factual_lora")
            ]
            self.expert_system = ExpertSystem(
                experts=experts,
                critic=self.contradiction_detector
            )
            print(f"[FCP] ExpertSystem initialized: {len(experts)} experts")
        except Exception as e:
            print(f"[FCP] ExpertSystem init failed: {e}")
            self.expert_system = None
        
        # === ThinkingController (согласно EVA.txt) ===
        try:
            from eva_ai.tools.fcp.thinking_controller import ThinkingController
            self.thinking_controller = ThinkingController()
            print("[FCP] ThinkingController initialized")
        except Exception as e:
            print(f"[FCP] ThinkingController init failed: {e}")
            self.thinking_controller = None
        
        # === ToolOrchestrator (согласно EVA.txt) ===
        try:
            from eva_ai.tools.fcp.orchestrator import ToolOrchestrator
            self.tool_orchestrator = ToolOrchestrator()
            print("[FCP] ToolOrchestrator initialized")
        except Exception as e:
            print(f"[FCP] ToolOrchestrator init failed: {e}")
            self.tool_orchestrator = None
        
        # === ReasoningChain (согласно EVA.txt - накопление цепочки рассуждений) ===
        try:
            if ReasoningChain:
                self.reasoning_chain = ReasoningChain(
                    max_steps=20,
                    similarity_threshold=0.7,
                    fractal_graph=getattr(self, 'fractal_graph', None),
                    memory_snapshot=getattr(self, 'memory_snapshot', None),
                    scenario_tcm=getattr(self, 'scenario_tcm', None)
                )
                print("[FCP] ReasoningChain initialized")
            else:
                self.reasoning_chain = None
        except Exception as e:
            print(f"[FCP] ReasoningChain init failed: {e}")
            self.reasoning_chain = None
        
        # === ReasoningChainManager для параллельных задач ===
        try:
            if ReasoningChainManager:
                self.reasoning_manager = ReasoningChainManager(
                    default_config={"max_steps": 20, "similarity_threshold": 0.7}
                )
                print("[FCP] ReasoningChainManager initialized")
            else:
                self.reasoning_manager = None
        except Exception as e:
            print(f"[FCP] ReasoningChainManager init failed: {e}")
            self.reasoning_manager = None
        
        # === ClarificationGenerator (согласно EVA.txt) ===
        try:
            from eva_ai.tools.fcp.clarification import ClarificationGenerator
            self.clarification_generator = ClarificationGenerator()
            print("[FCP] ClarificationGenerator initialized")
        except Exception as e:
            print(f"[FCP] ClarificationGenerator init failed: {e}")
            self.clarification_generator = None
        
        # === AttributionReport (согласно EVA.txt) ===
        try:
            from eva_ai.tools.fcp.attribution import AttributionReport
            self.attribution_report = AttributionReport()
            print("[FCP] AttributionReport initialized")
        except Exception as e:
            print(f"[FCP] AttributionReport init failed: {e}")
            self.attribution_report = None
        
        # === SemanticCacheEvictor (согласно EVA.txt) ===
        try:
            from eva_ai.tools.fcp.semantic_cache_evictor import SemanticCacheEvictor
            self.semantic_cache_evictor = SemanticCacheEvictor()
            print("[FCP] SemanticCacheEvictor initialized")
        except Exception as e:
            print(f"[FCP] SemanticCacheEvictor init failed: {e}")
            self.semantic_cache_evictor = None
        
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
        
        # 1. Приоритет: GNN веса от OnlineTrainer (после обучения)
        gnn_hybrid_path = os.path.join(project_root, 'eva_ai', 'training', 'checkpoints', 'gnn', 'gnn_for_hybrid.pt')
        if os.path.exists(gnn_hybrid_path):
            success = self.hybrid_processor.load_trained_encoder(gnn_hybrid_path)
            if success:
                print(f"[FCP] Loaded trained GNN from OnlineTrainer: {gnn_hybrid_path}")
                self.gnn_encoder = self.hybrid_processor.graph_encoder
                self._gnn_hybrid_path = gnn_hybrid_path
                self._gnn_file_mtime = os.path.getmtime(gnn_hybrid_path)
            else:
                self._try_load_pretrained_gnn(project_root)
        else:
            # 2. Fallback: pretrained GNN encoder
            self._try_load_pretrained_gnn(project_root)

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
        if HAS_OV_GENAI and os.path.exists(self.model_path):
            try:
                from openvino_genai import Tokenizer
                self.tokenizer = Tokenizer(self.model_path)
                print(f"[FCP] Tokenizer loaded via OpenVINO GenAI: {self.model_path}")
            except Exception as e:
                print(f"[FCP] OpenVINO Tokenizer load failed: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None
        
        # HuggingFace tokenizer for decoding with fix_mistral_regex
        try:
            from transformers import AutoTokenizer
            self.hf_tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True, 
                fix_mistral_regex=True
            )
            print(f"[FCP] HF Tokenizer loaded with fix_mistral_regex")
        except Exception as e:
            print(f"[FCP] HF Tokenizer load failed: {e}")
            self.hf_tokenizer = None
    
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
            
            # SchedulerConfig - используем единую конфигурацию KV кэша
            scheduler = ov_genai.SchedulerConfig()
            scheduler.cache_size = self.kv_cache_config["cache_size_gb"]
            scheduler.max_num_seqs = self.kv_cache_config["max_num_seqs"]
            scheduler.max_num_batched_tokens = self.kv_cache_config["max_num_batched_tokens"]
            scheduler.enable_prefix_caching = self.kv_cache_config["enable_prefix_caching"]
            scheduler.use_cache_eviction = self.kv_cache_config["use_cache_eviction"]

            # CacheEvictionConfig - сохраняем начало (системный промпт) и конец (последние токены)
            try:
                cache_eviction = ov_genai.CacheEvictionConfig(
                    start_size=self.kv_cache_config["cache_eviction_start_size"],
                    recent_size=self.kv_cache_config["cache_eviction_recent_size"],
                    max_cache_size=self.kv_cache_config["cache_eviction_max_size"],
                    aggregation_mode=ov_genai.AggregationMode.SUM
                )
                scheduler.cache_eviction_config = cache_eviction
                print(f"[FCP] CacheEvictionConfig enabled: start={self.kv_cache_config['cache_eviction_start_size']}, recent={self.kv_cache_config['cache_eviction_recent_size']}")
            except Exception as e:
                print(f"[FCP] CacheEvictionConfig not available: {e}")
            
            # GenerationConfig - используем единую конфигурацию
            gen_config = self.get_generation_config()
            gen_config.no_repeat_ngram_size = 5
            
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
        
        import queue
        import threading
        
        event_queue = queue.Queue()
        
        buffer = ""
        in_thinking = False
        partial_tag = ""
        
        def token_callback(token_text: str):
            nonlocal buffer, in_thinking, partial_tag
            buffer += token_text
            
            while True:
                if in_thinking:
                    idx = buffer.find("</think>")
                    if idx != -1:
                        thinking = buffer[:idx]
                        if thinking.strip():
                            event_queue.put({"type": "reasoning_text", "text": thinking})
                        in_thinking = False
                        buffer = buffer[idx + len("</think>"):]
                    else:
                        if len(buffer) > 20:
                            event_queue.put({"type": "reasoning_text", "text": buffer})
                            buffer = ""
                        break
                else:
                    idx = buffer.find("<think>")
                    if idx != -1:
                        if idx > 0:
                            event_queue.put({"type": "chunk", "text": buffer[:idx]})
                        in_thinking = True
                        buffer = buffer[idx + len("<think>"):]
                    else:
                        if buffer:
                            event_queue.put({"type": "chunk", "text": buffer})
                            buffer = ""
                        break
            return False
        
        def generate():
            nonlocal buffer, in_thinking, partial_tag, chat_prompt
            try:
                # 1. Гибридная обработка (KCA + GNN + SRG) согласно EVA.txt
                logger.info(f"[HYBRID_CHECK] hybrid_processor={self.hybrid_processor is not None}, fractal_graph={self.fractal_graph is not None if hasattr(self, 'fractal_graph') else 'N/A'}")
                if self.hybrid_processor and self.fractal_graph:
                    processed_prompt, metadata = self._process_with_hybrid_layers(chat_prompt, prompt)
                    if processed_prompt != chat_prompt:
                        chat_prompt = processed_prompt
                        logger.info(f"[HYBRID] srg_mode={metadata.get('srg_mode')}, kca_cycles={metadata.get('kca_cycles')}, kca_delta={metadata.get('kca_delta_norm', 0):.4f}")
                
                # 2. Генерация с поддержкой потока (всегда используем pipeline с гибридными слоями)
                gen_cfg = self.get_generation_config(max_new_tokens)
                self.pipeline.generate(chat_prompt, generation_config=gen_cfg, streamer=token_callback)
            
            except Exception as e:
                event_queue.put({"type": "error", "text": str(e)})
            finally:
                if buffer:
                    if in_thinking:
                        if buffer.strip():
                            event_queue.put({"type": "reasoning_text", "text": buffer})
                        event_queue.put({"type": "reasoning_end"})
                    else:
                        if buffer.strip():
                            event_queue.put({"type": "chunk", "text": buffer})
                event_queue.put({"type": "done", "timestamp": time.time()})
        
        gen_thread = threading.Thread(target=generate)
        gen_thread.start()

        full_response_parts = []

        while True:
            try:
                event = event_queue.get(timeout=0.1)
                if event['type'] == 'chunk':
                    full_response_parts.append(event.get('text', ''))
                yield event
                if event['type'] == 'done':
                    break
            except queue.Empty:
                if not gen_thread.is_alive():
                    break

        gen_thread.join()

        full_response = ''.join(full_response_parts)
        if full_response and add_to_history:
            self.conversation_history.append({
                "user": prompt,
                "assistant": full_response,
                "timestamp": time.time()
            })
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]
            self.stats["queries"] += 1
            self.add_dialog_turn("user", prompt)
            self.add_dialog_turn("assistant", full_response)
    
    def _init_lora_manager(self):
        self.current_adapter = None
        if self.lora_dir and os.path.exists(self.lora_dir):
            # Search for adapter files in order of preference
            candidates = [
                "lora_model.safetensors",  # New format for OpenVINO GenAI
                "lora_model.pt",           # Old PyTorch format
                "fcp_finetuned",           # Legacy name
                "fcp_finetuned.safetensors",
                "fcp_finetuned.pt"
            ]
            for candidate in candidates:
                candidate_path = os.path.join(self.lora_dir, candidate)
                if os.path.exists(candidate_path):
                    self.current_adapter = candidate
                    print(f"[FCP] LoRA adapter ready: {candidate}")
                    break
            if not self.current_adapter:
                print(f"[FCP] LoRA adapter not found in {self.lora_dir}")
        
        # Try to apply adapter to pipeline if available
        if self.current_adapter and self.pipeline:
            self._apply_lora_adapter(self.current_adapter)
    
    def _apply_lora_adapter(self, adapter_name: str):
        """Apply LoRA adapter to pipeline if supported."""
        if not self.pipeline:
            print("[FCP] No pipeline to apply LoRA adapter")
            return False
        adapter_path = os.path.join(self.lora_dir, adapter_name)
        if not os.path.exists(adapter_path):
            print(f"[FCP] Adapter file not found: {adapter_path}")
            return False
        
        try:
            import openvino_genai as ov_genai
            
            # Check if pipeline supports adapters
            if hasattr(self.pipeline, 'set_adapters') and callable(getattr(self.pipeline, 'set_adapters')):
                adapter = ov_genai.Adapter(adapter_path)
                config = ov_genai.AdapterConfig()
                config.add(adapter, alpha=0.7)
                self.pipeline.set_adapters(config)
                print(f"[FCP] LoRA adapter applied to pipeline: {adapter_name}")
                return True
            else:
                # Try alternative: use OpenVINOGenerator wrapper
                print(f"[FCP] Pipeline does not support set_adapters. Trying alternative...")
                # The adapter file exists but pipeline doesn't support direct loading
                # Store adapter path for later use in generation
                self._pending_adapter = adapter_path
                print(f"[FCP] Adapter stored for later use: {adapter_path}")
                return True
        except ImportError:
            print("[FCP] openvino_genai not available")
            return False
        except Exception as e:
            print(f"[FCP] Failed to apply LoRA adapter: {e}")
            return False
    
    def _try_load_pretrained_gnn(self, project_root: str):
        """Загрузить pretrained GNN encoder (fallback если нет обученных весов)."""
        gnn_path = os.path.join(project_root, 'models', 'graph_encoder.pt')
        if os.path.exists(gnn_path):
            success = self.hybrid_processor.load_trained_encoder(gnn_path)
            if success:
                print(f"[FCP] Loaded pretrained GNN encoder")
                self.gnn_encoder = self.hybrid_processor.graph_encoder
            else:
                print(f"[FCP] Failed to load pretrained GNN encoder")
                self.gnn_encoder = None
        else:
            print(f"[FCP] No pretrained GNN encoder found")
            self.gnn_encoder = None
    
    def _check_and_reload_gnn(self):
        """
        Проверить и перезагрузить GNN веса из OnlineTrainer если обновились.
        Вызывается перед каждой генерацией для hot-reload обученных весов.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        gnn_hybrid_path = os.path.join(project_root, 'eva_ai', 'training', 'checkpoints', 'gnn', 'gnn_for_hybrid.pt')
        
        if not os.path.exists(gnn_hybrid_path):
            return
        
        try:
            current_mtime = os.path.getmtime(gnn_hybrid_path)
            
            # Перезагрузить если файл обновился или ещё не загружен
            if self._gnn_file_mtime is None or current_mtime > self._gnn_file_mtime:
                if self._gnn_file_mtime is not None:
                    logger.info(f"[FCP] GNN weights updated, reloading from OnlineTrainer")
                
                success = self.hybrid_processor.load_trained_encoder(gnn_hybrid_path)
                if success:
                    self.gnn_encoder = self.hybrid_processor.graph_encoder
                    self._gnn_hybrid_path = gnn_hybrid_path
                    self._gnn_file_mtime = current_mtime
                    logger.info(f"[FCP] GNN hot-reload complete")
                else:
                    logger.warning(f"[FCP] GNN hot-reload failed")
        except Exception as e:
            logger.debug(f"[FCP] GNN file check error: {e}")
    
    def _check_and_reload_lora(self):
        """
        Проверить и перезагрузить LoRA адаптер если веса обновились.
        Вызывается перед каждой генерацией для hot-reload обученных весов.
        """
        if not self.lora_dir or not os.path.exists(self.lora_dir):
            return
        
        # Определить текущий адаптер
        adapter_name = None
        for candidate in ["lora_model.safetensors", "lora_model.pt", "fcp_finetuned.safetensors", "fcp_finetuned.pt"]:
            candidate_path = os.path.join(self.lora_dir, candidate)
            if os.path.exists(candidate_path):
                adapter_name = candidate
                break
        
        if not adapter_name:
            return
        
        adapter_path = os.path.join(self.lora_dir, adapter_name)
        
        # Проверить timestamp файла
        try:
            current_mtime = os.path.getmtime(adapter_path)
            
            # Перезагрузить если файл обновился или ещё не загружен
            if self._lora_file_mtime is None or current_mtime > self._lora_file_mtime:
                if self._lora_file_mtime is not None:
                    logger.info(f"[FCP] LoRA weights updated, reloading: {adapter_name}")
                
                # Применить адаптер к pipeline
                success = self._apply_lora_adapter(adapter_name)
                if success:
                    self._lora_adapter_path = adapter_path
                    self._lora_file_mtime = current_mtime
                    logger.info(f"[FCP] LoRA hot-reload complete: {adapter_name}")
                else:
                    logger.warning(f"[FCP] LoRA hot-reload failed for: {adapter_name}")
        except Exception as e:
            logger.debug(f"[FCP] LoRA file check error: {e}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 4096,
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
            # Приостановить обучение перед генерацией
            if self.online_trainer:
                self.online_trainer.on_generation_start()
            
            # HOT-RELOAD: проверить обновлённые веса перед инъекцией
            self._check_and_reload_gnn()
            self._check_and_reload_lora()
            
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
                    "assistant": response,
                    "timestamp": time.time()
                })
                if len(self.conversation_history) > self.max_history:
                    self.conversation_history = self.conversation_history[-self.max_history:]
                
                # Сохраняем сессию сразу после изменения
                self.save_session("default")
                
                # Сохраняем в сценарий (EVA.txt 6.3)
                self.add_dialog_turn("user", prompt)
                self.add_dialog_turn("assistant", response)
                
                # SYNC с HybridTokenCache
                self._sync_cache_with_history(prompt, response)
            
            # Возобновить обучение после генерации
            if self.online_trainer:
                self.online_trainer.on_generation_end()
                
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

        # ONLINE TRAINER: приостановить обучение перед генерацией
        if self.online_trainer and self.online_trainer.resource_manager:
            self.online_trainer.on_generation_start()
        
        # HOT-RELOAD GNN: проверить и перезагрузить веса из OnlineTrainer
        self._check_and_reload_gnn()
        
        # HOT-RELOAD LoRA: проверить и перезагрузить адаптер если веса обновились
        self._check_and_reload_lora()
        
        # 3. Генерация через OpenVINO pipeline
        response = self._generate(chat_prompt, max_new_tokens, **kwargs)

        # 4. Сохранение snapshot состояния (все слои)
        if self.memory_snapshot and self.fractal_graph:
            try:
                self._save_layer_snapshot(prompt, response)
            except Exception as e:
                logger.debug(f"Snapshot save skipped: {e}")

        # 5. Сохраняем в историю с timestamp
        if add_to_history and response:
            self.conversation_history.append({
                "user": prompt,
                "assistant": response,
                "timestamp": time.time()
            })
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]
            
            # Сохраняем в сценарий (EVA.txt 6.3)
            self.add_dialog_turn("user", prompt)
            self.add_dialog_turn("assistant", response)

            # Сохраняем сессию сразу после изменения
            self.save_session("default")
            
            # SYNC с HybridTokenCache: добавляем контекст после каждого диалога
            self._sync_cache_with_history(prompt, response)

        # ONLINE TRAINER: возобновить обучение после генерации + обновить историю
        if self.online_trainer:
            # Запустить обучение после ПЕРВОГО ответа
            if not self._training_started:
                self._training_started = True
                self.online_trainer.start()
                print("[FCP] Training started after first response")
            self.online_trainer.on_generation_end()
            self.online_trainer.update_conversation_history(self.conversation_history)

        if return_metadata:
            metadata["query_count"] = self.stats["queries"]
            # Добавляем информацию о гибридной обработке
            if 'srg_mode' in metadata:
                logger.info(f"[FCP] Generate complete: srg_mode={metadata['srg_mode']}, "
                           f"kca_cycles={metadata.get('kca_cycles', 0)}, "
                           f"injections={self.stats['injections']}")
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
                self.current_query_embedding = query_embedding  # Сохраняем для SRG оценки

                # 2. SRG evaluation - определяем режим
                if self.srg:
                    # Оцениваем logits на основе контекста промпта
                    estimated_logits = self._estimate_logits_from_prompt(chat_prompt)
                    mode, srg_metrics = self.srg.evaluate(
                        query_vec=query_embedding,
                        response_vec=query_embedding,
                        logits=estimated_logits
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
                        metadata["kca_delta_norm"] = float(np.linalg.norm(corrected_states - initial_states))

                        # Обогащаем промпт текстовым контекстом из подграфа + KCA
                        enriched_prompt = self._enrich_prompt_with_subgraph(
                            chat_prompt, 
                            subgraph,
                            kca_info=kca_info  # Передаём KCA info для дополнительного контекста
                        )
                        return enriched_prompt, metadata

            # Fallback - возвращаем оригинальный промпт
            return chat_prompt, metadata

        except Exception as e:
            logger.debug(f"Hybrid processing error: {e}")
            return chat_prompt, metadata

    def _get_query_embedding(self, text: str) -> np.ndarray:
        """Получить эмбеддинг запроса - детерминированный по хешу текста"""
        try:
            if hasattr(self, 'tokenizer') and self.tokenizer:
                inputs = self.tokenizer(text, return_tensors="np", padding=True, truncation=True)
                input_ids = inputs["input_ids"]
                if input_ids.shape[1] > 0:
                    return np.mean(inputs["input_ids"].astype(np.float32), axis=0)
        except Exception:
            pass

        # Fallback - детерминированный вектор по хешу текста (вместо random)
        text_hash = hash(text) % (2**31)
        np.random.seed(text_hash)
        return np.random.randn(2560).astype(np.float32) * 0.1
    
    def _get_base_tokenizer(self):
        """Получить базовый токенизатор для ContextualTokenizer."""
        try:
            # Пытаемся получить токенизатор из конфигурации или существующих компонентов
            if hasattr(self, 'tokenizer') and self.tokenizer:
                return self.tokenizer
            # Пытаемся загрузить из конфигурации
            tokenizer_path = self.fcp_config.get('tokenizer_path', '')
            if tokenizer_path and os.path.exists(tokenizer_path):
                from transformers import AutoTokenizer
                return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, fix_mistral_regex=True)
            # Дефолтный токенизатор
            from transformers import GPT2Tokenizer
            return GPT2Tokenizer.from_pretrained('gpt2')
        except Exception as e:
            logger.debug(f"Failed to load base tokenizer: {e}")
            return None
    
    def _get_embedding_model(self):
        """Получить модель эмбеддингов для ContextualTokenizer."""
        try:
            from sentence_transformers import SentenceTransformer
            model_name = self.fcp_config.get('embedding_model', 'intfloat/multilingual-e5-small')
            return SentenceTransformer(model_name, device='cpu')
        except Exception as e:
            logger.debug(f"Failed to load embedding model: {e}")
            return None
    
    def _convert_tokenized_to_numpy(self, tokenized):
        """Convert tokenized output to numpy array for OpenVINO inference."""
        import numpy as np
        if tokenized is None:
            return None
        if hasattr(tokenized, 'input_ids'):
            tokenized = tokenized.input_ids
        if hasattr(tokenized, 'data'):
            tokenized = tokenized.data
        if hasattr(tokenized, 'tolist'):
            return np.array(tokenized.tolist(), dtype=np.int64)
        if isinstance(tokenized, (list, tuple)):
            return np.array(tokenized, dtype=np.int64)
        if hasattr(tokenized, 'shape'):
            return np.array(tokenized, dtype=np.int64)
        return np.array(tokenized, dtype=np.int64)
    
    def _estimate_logits_from_prompt(self, prompt: str, hidden_dim: int = 2560) -> np.ndarray:
        """
        Оценить псевдо-logs для SRG на основе контекста промпта.
        
        Логика:
        - Вопросы (?) → выше энтропия (неопределённость)
        - Утверждения → ниже энтропия (уверенность)
        - Длинные промпты → выше энтропия (сложность)
        - Ключевые слова неопределённости → выше энтропия
        - Ключевые слова уверенности → ниже энтропия
        """
        # Базовое значение - средняя неопределённость
        base_entropy = 0.5
        prompt_lower = prompt.lower()
        
        # 1. Проверка на вопрос
        if '?' in prompt:
            base_entropy += 0.2
        
        # 2. Проверка на ключевые слова неопределённости
        uncertainty_words = ['может', 'возможно', 'вероятно', 'думаю', 'кажется', 
                            'perhaps', 'maybe', 'probably', 'might', 'think', 'seems']
        for word in uncertainty_words:
            if word in prompt_lower:
                base_entropy += 0.15
                break
        
        # 3. Проверка на ключевые слова уверенности
        certainty_words = ['определённо', 'конечно', 'точно', 'известно', 'точно знаю',
                         'definitely', 'certainly', 'know', 'sure', 'clearly']
        for word in certainty_words:
            if word in prompt_lower:
                base_entropy -= 0.2
                break
        
        # 4. Длина промпта (длиннее = сложнее = выше неопределённость)
        word_count = len(prompt.split())
        if word_count > 50:
            base_entropy += 0.1
        elif word_count > 100:
            base_entropy += 0.2
        
        # 5. Наличие контекста в промпте (history) - больше контекста = увереннее
        if '[Релевантный контекст' in prompt or '<|im_start|>' in prompt:
            base_entropy -= 0.15
        
        # Ограничиваем в диапазон [0.1, 0.9]
        base_entropy = max(0.1, min(0.9, base_entropy))
        
        # Конвертируем в pseudo-logits
        # entropy_ratio = 0.1 → peaked (уверенный) → logits с одним пиком
        # entropy_ratio = 0.9 → uniform (неопределённый) → logits равномерные
        
        entropy_ratio = base_entropy
        vocab_size = 100  # фиксированный размер для SRG
        
        logits = np.zeros(vocab_size, dtype=np.float32)
        
        # Чем выше entropy_ratio, тем более равномерное распределение
        num_peaks = max(1, int(vocab_size * (1 - entropy_ratio * 0.8)))
        
        # Распределяем пики по словарю
        peak_positions = np.linspace(0, vocab_size - 1, num_peaks, dtype=int)
        for pos in peak_positions:
            logits[pos] = 5.0 * (1 - entropy_ratio * 0.5)
        
        return logits

    def _enrich_prompt_with_subgraph(self, prompt: str, subgraph: dict, kca_info: dict = None) -> str:
        """Обогатить промпт текстовым контекстом из подграфа + KCA"""
        # Добавляем KCA информацию если есть
        if kca_info:
            cycles = kca_info.get("cycles", 0)
            status = kca_info.get("status", "unknown")
            # Добавляем подсказку для модели о том, что использовался KCA
            kca_hint = f"[Когнитивная обработка: {cycles} циклов, статус: {status}]"
        
        # Получаем текстовый контекст из графа
        if not subgraph or not isinstance(subgraph, dict):
            return prompt
        embeddings = subgraph.get("embeddings")
        if embeddings is None or (hasattr(embeddings, 'shape') and embeddings.shape[0] == 0):
            return prompt

        context_lines = []
        contents = subgraph.get("contents", []) or subgraph.get("node_contents", [])

        for i, content in enumerate(contents[:5]):
            context_lines.append(f"  {i+1}. {content}")

        if context_lines:
            context_str = "\n".join(context_lines)
            
            # Формируем обогащённый промпт
            if kca_info:
                kca_hint = f"[KCA: {kca_info.get('cycles', 0)} циклов, {kca_info.get('status', 'unknown')}]"
                enriched = f"\n📚 Контекст из графа знаний:\n{context_str}\n{kca_hint}\n\n{prompt}"
            else:
                enriched = f"\n📚 Контекст из графа знаний:\n{context_str}\n\n{prompt}"
            return enriched

        # Если есть KCA но нет контента - всё равно добавляем KCA подсказку
        if kca_info:
            kca_hint = f"[KCA: {kca_info.get('cycles', 0)} циклов, {kca_info.get('status', 'unknown')}]"
            return f"\n{kca_hint}\n\n{prompt}"
        
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
        Использует ContextualTokenizer для обогащения контекста (EVA.txt 2.1).
        Использует ThinkingController для управления режимом рассуждений (EVA.txt).
        """
        # === ThinkingController: определение режима рассуждений ===
        if hasattr(self, 'thinking_controller') and self.thinking_controller:
            try:
                enable_thinking = self.thinking_controller.should_enable_thinking(prompt)
                logger.info(f"[FCP] ThinkingController: enable_thinking={enable_thinking}")
            except Exception as e:
                logger.debug(f"ThinkingController error: {e}")
        
        # Используем ContextualTokenizer если доступен
        context_nodes = []
        if hasattr(self, 'contextual_tokenizer') and self.contextual_tokenizer:
            try:
                # Получаем релевантные узлы графа для контекста
                if self.fractal_graph and self.fractal_graph.node_count > 0:
                    query_emb = self._get_query_embedding(prompt)
                    subgraph = self.fractal_graph.retrieve_subgraph(
                        query_emb.reshape(1, -1), top_k=5
                    )
                    if subgraph and 'nodes' in subgraph:
                        context_nodes = [n.get('id', '') for n in subgraph['nodes'] if 'id' in n]
                
                # Токенизируем с контекстом
                tokenized = self.contextual_tokenizer.tokenize_with_context(prompt, context_nodes)
                
                # Можно использовать tokenized['graph_anchors'] для обогащения промпта
                if tokenized['graph_anchors']:
                    anchor_info = " ".join([a['label'] for a in tokenized['graph_anchors']])
                    prompt = f"[Контекстные якоря: {anchor_info}]\n{prompt}"
            except Exception as e:
                logger.debug(f"ContextualTokenizer error: {e}")
        
        # === ExpertSystem: мультиагентное обсуждение (EVA.txt) ===
        if hasattr(self, 'expert_system') and self.expert_system:
            try:
                # Определяем сложность запроса по длине и ключевым словам
                complex_keywords = ['почему', 'как именно', 'объясни', 'докажи', 'сравни',
                                'why', 'how exactly', 'explain', 'prove', 'compare']
                is_complex = len(prompt.split()) > 30 or any(kw in prompt.lower() for kw in complex_keywords)
                
                if is_complex:
                    print(f"[FCP] Using ExpertSystem for complex query")
                    expert_response = self.expert_system.discuss(prompt)
                    if expert_response and expert_response != "No experts configured":
                        # Сохраняем в историю с timestamp
                        if add_to_history:
                            self.conversation_history.append({
                                "user": prompt,
                                "assistant": expert_response,
                                "timestamp": time.time()
                            })
                            if len(self.conversation_history) > self.max_history:
                                self.conversation_history = self.conversation_history[-self.max_history:]
                        
                        return expert_response
            except Exception as e:
                print(f"[FCP] ExpertSystem error: {e}")
        
        # Получаем семантически релевантный контекст из истории
        relevant_context = self.get_relevant_context(prompt, max_history=3)
        
        # Получаем похожие сценарии из ScenarioTCM (EVA.txt 6.3)
        scenario_context = self.get_similar_scenarios(prompt, max_scenarios=3)
        
        # === ReasoningChain: контекст предыдущих рассуждений ===
        reasoning_context = ""
        if hasattr(self, 'reasoning_chain') and self.reasoning_chain:
            # Проверяем, является ли это многошаговой задачей
            multi_step_keywords = ['решить', 'загадка', 'логика', 'задача', 'рассуждать', 
                                  'шаг', 'этап', 'следовательно', 'значит', 'далее',
                                  'solve', 'riddle', 'logic', 'task', 'reason', 'step']
            is_multi_step = any(kw in prompt.lower() for kw in multi_step_keywords)
            
            if is_multi_step or self.reasoning_chain.steps:
                # Получаем контекст рассуждений для сложных задач
                reasoning_context = self.reasoning_chain.get_context(
                    include_reasoning=True,
                    max_steps=5
                )
                logger.debug(f"[FCP] ReasoningChain context added: {len(self.reasoning_chain.steps)} steps")
        
        # Формируем историю с приоритетом по времени
        history_text = ""
        if self.conversation_history:
            # Сортируем по timestamp (новые first) и берём последние max_history
            sorted_history = sorted(
                self.conversation_history,
                key=lambda x: x.get('timestamp', 0),
                reverse=True
            )[:self.max_history]
            
            # Добавляем маркер времени для контекста
            for i, entry in enumerate(sorted_history):
                entry_time = entry.get('timestamp', 0)
                time_marker = ""
                if entry_time > 0:
                    # Добавляем относительное время
                    age_seconds = time.time() - entry_time
                    if age_seconds < 60:
                        time_marker = "[только что] "
                    elif age_seconds < 3600:
                        time_marker = f"[{int(age_seconds/60)} мин назад] "
                    elif age_seconds < 86400:
                        time_marker = f"[{int(age_seconds/3600)} ч назад] "
                
                history_text += f"<|im_start|>user\n{time_marker}{entry['user']}<|im_end|>\n"
                history_text += f"<|im_start|>assistant\n{entry['assistant']}<|im_end|>\n"
        
        # Добавляем найденные сценарии
        if scenario_context:
            history_text = f"[Похожие сценарии]:\n{scenario_context}\n\n" + history_text
        
        # Добавляем релевантный контекст если есть
        if relevant_context:
            history_text = f"[Релевантный контекст из прошлых разговоров]:\n{relevant_context}\n\n" + history_text
        
        # Добавляем контекст рассуждений (приоритет - в начало)
        if reasoning_context:
            history_text = f"{reasoning_context}\n\n" + history_text
        
        return f"{history_text}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def _generate(self, prompt: str, max_new_tokens: int = 4096, **kwargs) -> str:
        """Генерация ответа (non-streaming, возвращает полный результат)"""
        if not self.pipeline:
            return "[No pipeline]"
        
        try:
            gen_cfg = self.get_generation_config(max_new_tokens)
            result = self.pipeline.generate(prompt, generation_config=gen_cfg, **kwargs)
        except Exception as e:
            return f"Generation error: {e}"
        
        return result

    def generate_with_injection(self, prompt: str, max_new_tokens: int = 4096, 
                              enable_thinking: bool = True, return_metadata: bool = False) -> str:
        """
        Полнослойная инъекция согласно Доработка.txt (FCP specification)
        Runtime State Injection: модификация Key и Value тензоров на всех слоях
        """
        if not self.pipeline or not self.state_injector:
            # Fallback к обычной генерации если injector недоступен
            return self._generate(prompt, max_new_tokens, **{})
        
        # Track if StateInjector is working
        injector_working = True
        
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
            try:
                self.state_injector.reset_all_states()
            except Exception as e:
                logger.warning(f"[FCP] StateInjector reset failed: {e}")
                injector_working = False
            
            if not injector_working:
                return self._generate(prompt, max_new_tokens, **{})
            
            # Запускаем pre-fill inference
            try:
                self.state_injector.request.infer({"input_ids": input_ids})
            except Exception as e:
                logger.warning(f"[FCP] StateInjector pre-fill failed: {e}, falling back")
                return self._generate(prompt, max_new_tokens, **{})
            seq_len = input_ids.shape[1]
            
            # Получаем текст токенов для анализа
            token_texts = [self.tokenizer.decode([tid]) for tid in input_ids[0]] if hasattr(self, 'tokenizer') else []
            
            # 2. SQAM Analysis & Full-Layer Key Scaling (согласно Доработка.txt)
            key0 = self.state_injector.get_key(0)  # Первый слой
            _, importance = self.sqam_analyzer.analyze(key0, seq_len)
            all_layers = self.state_injector.get_all_layer_indices()
            
            # Применяем SQAM ко ВСЕМ слоям (Key scaling)
            self.state_injector.transform_keys(all_layers, apply_sqam_scaling, weights=importance)
            
            # NEW: GraphStateInjector - Full-layer graph injection (EVA.txt section 8.2)
            # После prefill, инжектируем graph_vector во все Value тензоры
            if hasattr(self, 'graph_state_injector') and self.graph_state_injector and self.fractal_graph:
                try:
                    # Get subgraph embeddings from fractal_graph
                    query_emb = self._get_query_embedding(prompt)
                    subgraph = self.fractal_graph.retrieve_subgraph(query_emb.reshape(1, -1), top_k=8)
                    
                    subgraph_embeddings = None
                    if subgraph and 'nodes' in subgraph:
                        node_embs = []
                        for node in subgraph['nodes']:
                            if 'emb' in node:
                                emb = node['emb']
                                if isinstance(emb, list):
                                    node_embs.append(emb)
                                elif hasattr(emb, 'tolist'):
                                    node_embs.append(emb.tolist())
                        
                        if node_embs:
                            import numpy as np
                            subgraph_embeddings = np.array(node_embs, dtype=np.float32)
                    
                    if subgraph_embeddings is not None and len(subgraph_embeddings) > 0:
                        inj_result = self.graph_state_injector.inject_graph(
                            subgraph_embeddings=subgraph_embeddings,
                            apply_kca=True
                        )
                        logger.info(f"[FCP] Graph injection: {inj_result['layers_processed']} layers, "
                                   f"lacuna={inj_result['kca_results']['lacuna_detected']}, "
                                   f"early_exit={inj_result['early_exit_triggered']}")
                        metadata["graph_injection_applied"] = True
                        metadata["graph_injection_layers"] = inj_result['layers_processed']
                except Exception as e:
                    logger.debug(f"[FCP] GraphStateInjector failed: {e}")
            
            # 3. Graph Enrichment - извлечение якорных токенов и обновление центроида
            anchors = self.sqam_analyzer.get_core_anchors(token_texts, threshold=0.6)
            key_per_token = key0[0].mean(axis=0)  # Average over heads
            self.graph_mgr.add_anchors(anchors, key_per_token)
            
            # 4. Decoding Loop with Full-Layer KCA Injection + Activation Gate + KCA Gate
            generated_ids = input_ids[0].tolist()
            eos_token_id = getattr(self.tokenizer, 'eos_token_id', 2) if hasattr(self, 'tokenizer') else 2
            
            # Инициализация для раннего выхода
            self.activation_gate_config['accumulated_confidence'] = 0.0
            early_exit_triggered = False
            early_exit_at_step = -1
            
            for step in range(max_new_tokens):
                # Инференс на последнем токене
                self.state_injector.request.infer({"input_ids": np.array([[generated_ids[-1]]])})
                logits = self.state_injector.request.get_tensor("logits").data[0, -1]
                next_token = int(np.argmax(logits))
                
                # === Activation Gate: вычисление уверенности (EVA.txt раздел 2.1) ===
                # Уверенность = softmax max probability
                probs = np.exp(logits) / np.sum(np.exp(logits))
                token_confidence = float(np.max(probs))
                
                # Сглаживание уверенности (окно из предыдущих токенов)
                acc_conf = self.activation_gate_config['accumulated_confidence']
                window = self.activation_gate_config['confidence_window']
                if step > 0:
                    smoothed_conf = (acc_conf * (step - 1) + token_confidence) / step
                else:
                    smoothed_conf = token_confidence
                self.activation_gate_config['accumulated_confidence'] = smoothed_conf
                
                # === Early Exit проверка ===
                # Если накопленная уверенность > порог и прошло достаточно токенов
                if (step >= self.activation_gate_config['min_tokens_for_check'] and 
                    smoothed_conf > self.activation_gate_config['early_exit_threshold'] and 
                    not early_exit_triggered):
                    early_exit_triggered = True
                    early_exit_at_step = step
                    self.activation_gate_config['early_exits_count'] += 1
                    logger.info(f"[FCP] Activation Gate TRIGGERED at step {step}, confidence={smoothed_conf:.3f}")
                    # Полный Early Exit: прерываем генерацию
                    metadata["early_exit_triggered"] = True
                    metadata["early_exit_at_step"] = step
                    # Прерываем цикл генерации
                    break  # <-- ПОЛНЫЙ EARLY EXIT
                
                # === SemanticCacheEvictor: управление кэшем (EVA.txt) ===
                if hasattr(self, 'semantic_cache_evictor') and self.semantic_cache_evictor:
                    try:
                        # Проверяем, нужно ли вытеснение
                        current_size = len(generated_ids)
                        max_cache_size = 100  # Можно взять из конфигурации
                        
                        if self.semantic_cache_evictor.should_evict(
                            current_size=current_size,
                            max_size=max_cache_size,
                            threshold=0.7
                        ):
                            # Вытесняем блоки (упрощение: просто очищаем часть истории)
                            if len(self.conversation_history) > 3:
                                self.conversation_history = self.conversation_history[-3:]
                                metadata["cache_evicted"] = True
                                logger.info(f"[FCP] SemanticCacheEvictor: cache evicted")
                    except Exception as e:
                        logger.debug(f"SemanticCacheEvictor error: {e}")
                
                # === AttributionReport: финализация (EVA.txt) ===
                if hasattr(self, 'attribution_report') and self.attribution_report:
                    try:
                        if hasattr(self, 'attribution_tracker'):
                            self.attribution_tracker.finalize(self.attribution_report)
                            metadata["attribution_report"] = self.attribution_report.explain()
                    except Exception as e:
                        logger.debug(f"AttributionReport finalize error: {e}")
                
                generated_ids.append(next_token)
                
                if next_token == eos_token_id:
                    break
                
                # === KCA Gate: мониторинг коррекции (EVA.txt раздел 3.3) ===
                self.kca_gate_config['kca_iterations'] += 1
                
                # Динамический расчет KCA коррекции
                val_proxy = self.state_injector.get_value(all_layers[-1])[0, :, -1, :].mean(axis=0)

                # === Oscillation detection (EVA.txt раздел 3.3) ===
                current_state = val_proxy.copy()
                state_history = self.kca_gate_config['state_history']
                state_history.append(current_state)
                if len(state_history) > 3:
                    state_history.pop(0)
                # Detect oscillation if we have at least 3 states
                if len(state_history) >= 3:
                    h0, h1, h2 = state_history[-3], state_history[-2], state_history[-1]
                    delta1 = h1 - h0
                    delta2 = h2 - h1
                    norm1 = np.linalg.norm(delta1)
                    norm2 = np.linalg.norm(delta2)
                    if norm1 > 1e-8 and norm2 > 1e-8:
                        cos_sim = np.dot(delta1, delta2) / (norm1 * norm2)
                        if cos_sim < -0.5:
                            # Average the last three states to damp oscillations
                            avg_state = (h0 + h1 + h2) / 3.0
                            val_proxy = avg_state
                            self.kca_gate_config['state_history'] = []  # reset to avoid repeated oscillation
                            logger.info(f"[FCP] Oscillation detected (cos_sim={cos_sim:.3f}), averaged last three states.")
                # === Cross-Attention Слияние (EVA.txt раздел 2.1) ===
                if hasattr(self, 'cross_attention') and self.cross_attention:
                    try:
                        # Получаем эмбеддинги узлов графа для cross-attention
                        if self.fractal_graph and self.fractal_graph.node_count > 0:
                            # Получаем подграф для cross-attention
                            query_emb = self._get_query_embedding(prompt)
                            subgraph = self.fractal_graph.retrieve_subgraph(
                                query_emb.reshape(1, -1), top_k=10
                            )
                            
                            graph_embeddings = []
                            if subgraph and 'nodes' in subgraph:
                                for node in subgraph['nodes']:
                                    if 'emb' in node:
                                        graph_embeddings.append(node['emb'])
                            
                            if len(graph_embeddings) > 0:
                                graph_emb_array = np.array(graph_embeddings)
                                # Вычисляем cross-attention между моделью и графом
                                ca_output, ca_weights = self.cross_attention.compute_cross_attention(
                                    val_proxy, graph_emb_array
                                )
                                # Сохраняем результат cross-attention
                                metadata["cross_attention_applied"] = True
                                metadata["cross_attention_weights"] = ca_weights.tolist() if hasattr(ca_weights, 'tolist') else list(ca_weights)
                            else:
                                ca_output = val_proxy
                        else:
                            ca_output = val_proxy
                    except Exception as e:
                        logger.debug(f"Cross-attention failed: {e}")
                        ca_output = val_proxy
                else:
                    ca_output = val_proxy
                
                # NEW: Use KCADetector for advanced lacuna/contradiction detection
                if hasattr(self, 'kca_detector') and self.kca_detector and hasattr(self, 'state_injector') and self.state_injector:
                    try:
                        # Get layer indices from state_injector
                        layer_indices = self.state_injector.get_all_layer_indices()
                        
                        # Generate gate weights if not available
                        # For simplicity, use a fixed pattern based on position
                        gate_weights = np.array([
                            0.5 + 0.3 * np.sin(i * np.pi / 18) for i in range(len(layer_indices))
                        ], dtype=np.float32)
                        
                        # Use graph vector from graph_mgr as proxy
                        graph_vec = self.graph_mgr.get_centroid()
                        if len(graph_vec) >= 256:
                            graph_vec = graph_vec[:256]
                        else:
                            graph_vec = np.pad(graph_vec, (0, 256 - len(graph_vec)))
                        
                        # Run KCA detection
                        kca_result = self.kca_detector.detect(
                            graph_vector=graph_vec,
                            layer_indices=layer_indices,
                            gate_weights=gate_weights
                        )
                        
                        # If KCA detected lacunas or contradictions, adjust correction
                        if kca_result['lacuna_detected'] or kca_result['contradiction_detected']:
                            # Use KCA correction embedding
                            corrections = kca_result.get('corrections', {})
                            if corrections:
                                # Average all correction embeddings
                                correction_embs = [c['embedding'] for c in corrections.values() if 'embedding' in c]
                                if correction_embs:
                                    kca_correction = np.mean(correction_embs, axis=0)
                                    self.kca_correction_vec = kca_correction * kca_result['confidence']
                                    metadata["kca_lacunas"] = len(kca_result['lacuna_layers'])
                                    metadata["kca_contradictions"] = len(kca_result['contradiction_layers'])
                                    metadata["kca_confidence"] = kca_result['confidence']
                        else:
                            self.kca_correction_vec = compute_kca_correction(val_proxy, self.graph_mgr.get_centroid())
                    except Exception as e:
                        logger.debug(f"KCADetector failed: {e}")
                        self.kca_correction_vec = compute_kca_correction(val_proxy, self.graph_mgr.get_centroid())
                else:
                    self.kca_correction_vec = compute_kca_correction(val_proxy, self.graph_mgr.get_centroid())

                # Apply adaptive damping to correction vector (EVA.txt раздел 3.3)
                rho = self.kca_gate_config['damping_factor']
                t = self.kca_gate_config['kca_iterations']
                damping = rho ** t
                self.kca_correction_vec = self.kca_correction_vec * damping


                # === TrainableGate: Обучаемый гейт слияния (EVA.txt раздел 2.1) ===
                if hasattr(self, 'trainable_gate') and self.trainable_gate:
                    try:
                        # Комбинируем источники: модель, cross-attention, KCA
                        source_vectors = [val_proxy, ca_output, self.kca_correction_vec]
                        combined_output, gate_weights = self.trainable_gate.forward(source_vectors)
                        
                        # Используем комбинированный результат для инъекции
                        injection_vec = combined_output
                        metadata["trainable_gate_applied"] = True
                        metadata["trainable_gate_weights"] = gate_weights.tolist() if hasattr(gate_weights, 'tolist') else list(gate_weights)
                    except Exception as e:
                        logger.debug(f"TrainableGate failed: {e}")
                        injection_vec = self.kca_correction_vec
                else:
                    injection_vec = self.kca_correction_vec
                
                # Используем KCA ко ВСЕМ слоям Value (инъекция знаний)
                if np.linalg.norm(injection_vec) > 1e-5:
                    # Адаптивный вес для квантованных моделей
                    kca_weight = 0.07
                    if hasattr(self, 'model_path') and 'int4' in self.model_path.lower():
                        kca_weight = 0.2
                    elif 'int8' in self.model_path.lower():
                        kca_weight = 0.12
                    
                    # === ToolOrchestrator: Toolformer интеграция (EVA.txt) ===
                    if hasattr(self, 'tool_orchestrator') and self.tool_orchestrator:
                        try:
                            # Проверяем, есть ли вызовы инструментов в ответе
                            if self.tool_orchestrator._has_tool_call(token_text):
                                # Выполняем инструменты
                                processed = self.tool_orchestrator.process_response(token_text)
                                if processed != token_text:
                                    # Инструмент был вызван, обновляем ответ
                                    token_text = processed
                                    metadata["tool_used"] = True
                        except Exception as e:
                            logger.debug(f"ToolOrchestrator error: {e}")
                    
                    # Применяем коррекцию ко всем слоям
                    
                # === KCA Gate: вычисление γ (gamma) ===
                # Gamma зависит от norm(correction) / norm(hidden_state)
                correction_norm = np.linalg.norm(self.kca_correction_vec)
                hidden_norm = np.linalg.norm(val_proxy) + 1e-8
                gamma = min(1.0, correction_norm / hidden_norm)  # 0 ≤ γ ≤ 1

                # Сохраняем в историю
                self.kca_gate_config['gate_history'].append(gamma)
                self.kca_gate_config['gamma'] = gamma

                # === KCA Gate монитор насыщения ===
                # Если среднее γ < threshold за последние min_iterations → отклонение коррекции
                if len(self.kca_gate_config['gate_history']) >= self.kca_gate_config['min_iterations']:
                    recent_gamma = self.kca_gate_config['gate_history'][-self.kca_gate_config['min_iterations']:]
                    avg_gamma = np.mean(recent_gamma)
                    if avg_gamma < self.kca_gate_config['gate_threshold']:
                        self.kca_gate_config['kca_rejected'] = True
                        logger.info(f"[FCP] KCA Gate REJECTED: avg_gamma={avg_gamma:.4f} < {self.kca_gate_config["gate_threshold"]}")

                # Применяем KCA только если не отклонено
                if not self.kca_gate_config['kca_rejected']:
                    self.state_injector.transform_values(
                        all_layers, 
                        inject_graph_vector, 
                        vector=self.kca_correction_vec, 
                        weight=kca_weight * gamma  # Применяем γ к весу
                    )
                else:
                    logger.debug(f"[FCP] KCA skipped: gamma={gamma:.4f}")
            # Логирование статистики gate
            logger.info(f"[FCP] Generation complete: {len(generated_ids)} tokens, "
                       f"early_exits={self.activation_gate_config['early_exits_count']}, "
                       f"kca_iterations={self.kca_gate_config['kca_iterations']}, "
                       f"kca_rejected={self.kca_gate_config['kca_rejected']}")
            
            # 5. SRG Post-Evaluation (оценка уверенности)
            final_logits = self.state_injector.request.get_tensor("logits").data[0, -1]
            # Простая оценка на основе энтропии
            probs = np.exp(final_logits) / np.sum(np.exp(final_logits))
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            # Используем пороги из конфигурации SRG
            is_confident = entropy <= self.srg.config.srg_entropy_threshold
            if is_confident:
                mode = "direct"
            else:
                mode = "reasoning"
            srg_metrics = {"mode": mode, "entropy": entropy, "confidence": 1.0 - entropy / 10.0}
            
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
                    "kca_applied": np.linalg.norm(self.kca_correction_vec) > 1e-5,
                    # Activation Gate stats
                    "activation_gate": {
                        "early_exit_triggered": self.activation_gate_config['early_exits_count'] > 0,
                        "early_exits_count": self.activation_gate_config['early_exits_count'],
                        "final_confidence": self.activation_gate_config['accumulated_confidence'],
                    },
                    # KCA Gate stats
                    "kca_gate": {
                        "gamma": self.kca_gate_config['gamma'],
                        "kca_iterations": self.kca_gate_config['kca_iterations'],
                        "kca_rejected": self.kca_gate_config['kca_rejected'],
                    }
                }
                return response, metadata
            
            return response
            
        except Exception as e:
            logger.error(f"[FCP] Injection generation FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            
            # Пробуем partial fallback - генерация с базовым KCA (без StateInjector)
            try:
                logger.info("[FCP] Trying partial fallback: generation with graph context only")
                # Используем базовый generate с включённой гибридной обработкой
                return self._generate(prompt, max_new_tokens, **{})
            except Exception as fallback_error:
                logger.error(f"[FCP] Partial fallback also failed: {fallback_error}")
                # Полный fallback
                return self._generate(prompt, max_new_tokens, **{})
    
    def load_lora_adapter(self, adapter_name: str = "lora_model.pt", alpha: float = 0.8):
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
            self._training_started = False  # Обучение начнётся после первого ответа
            self.kca_correction_vec = None  # Для хранения вектора коррекции KCA

            return True
        except Exception as e:
            logger.error(f"[FCP] Session clear error: {e}")
            return False
    
    def get_similar_scenarios(self, query: str, max_scenarios: int = 3) -> str:
        """
        Поиск похожих сценариев из истории диалогов (EVA.txt раздел 6.3).
        
        Args:
            query: текст запроса
            max_scenarios: максимум сценариев для возврата
            
        Returns:
            Строка с контекстом из похожих сценариев
        """
        if not hasattr(self, 'scenario_tcm') or not self.scenario_tcm:
            return ""
        
        try:
            # Получаем эмбеддинг запроса
            query_emb = self._get_query_embedding(query)
            
            # Ищем похожие сценарии
            similar = self.scenario_tcm.find_similar(query_emb, max_results=max_scenarios)
            
            if not similar:
                return ""
            
            # Форматируем контекст из сценариев
            context_parts = []
            for scenario in similar:
                title = scenario.get('title', 'Сценарий')
                summary = scenario.get('summary', '')
                if summary:
                    context_parts.append(f"[{title}]: {summary}")
            
            if context_parts:
                return "\n\n".join(context_parts)
            
            return ""
            
        except Exception as e:
            logger.debug(f"[FCP] Scenario search error: {e}")
            return ""
    
    def add_dialog_turn(self, role: str, text: str, embedding: np.ndarray = None):
        """
        Добавить ход диалога в текущую цепочку сценария (EVA.txt раздел 6.3).
        
        Args:
            role: 'user' или 'assistant'
            text: текст сообщения
            embedding: эмбеддинг (если None - создастся из текста)
        """
        if not hasattr(self, 'scenario_tcm') or not self.scenario_tcm:
            return
        
        try:
            if embedding is None:
                embedding = self._get_query_embedding(text)
            
            self.scenario_tcm.add_turn(role, text, embedding)
        except Exception as e:
            logger.debug(f"[FCP] Add dialog turn error: {e}")
    
    # === ConceptMiner методы (EVA.txt раздел 7.1) ===
    def start_concept_mining(self):
        """Запуск фонового поиска концептов"""
        if hasattr(self, 'concept_miner') and self.concept_miner:
            try:
                self.concept_miner.start()
                print("[FCP] ConceptMiner started")
                return True
            except Exception as e:
                print(f"[FCP] ConceptMiner start error: {e}")
        return False
    
    def stop_concept_mining(self):
        """Остановка поиска концептов"""
        if hasattr(self, 'concept_miner') and self.concept_miner:
            try:
                self.concept_miner.stop()
                print("[FCP] ConceptMiner stopped")
                return True
            except Exception as e:
                print(f"[FCP] ConceptMiner stop error: {e}")
        return False
    
    def _get_hybrid_cache_config(self) -> Dict[str, Any]:
        """
        Получить конфигурацию из HybridTokenCache если доступен.
        Интеграция KV кэша OpenVINO с гибридным кэшем EVA.
        """
        config = {"enabled": False}
        
        try:
            # Пробуем найти HybridTokenCache
            if hasattr(self, 'hybrid_cache') and self.hybrid_cache:
                hc = self.hybrid_cache
                config["enabled"] = True
                
                # Получаем параметры из HybridTokenCache
                if hasattr(hc, 'target_memory_bytes'):
                    config["cache_size_gb"] = hc.target_memory_bytes / (1024**3)
                if hasattr(hc, 'max_memory_tokens'):
                    config["max_tokens"] = hc.max_memory_tokens * 2048  # примерный размер
                if hasattr(hc, 'disk_cache_dir'):
                    config["disk_cache_dir"] = hc.disk_cache_dir
                    
                logger.info(f"[FCP] Linked with HybridTokenCache: {config.get('cache_size_gb', 0):.1f}GB")
            
            # Альтернативно - пробуем из brain
            elif hasattr(self, 'brain') and self.brain:
                if hasattr(self.brain, 'hybrid_cache') and self.brain.hybrid_cache:
                    hc = self.brain.hybrid_cache
                    config["enabled"] = True
                    if hasattr(hc, 'target_memory_bytes'):
                        config["cache_size_gb"] = hc.target_memory_bytes / (1024**3)
                    if hasattr(hc, 'max_memory_tokens'):
                        config["max_tokens"] = hc.max_memory_tokens * 2048
                    logger.info(f"[FCP] Linked with brain.hybrid_cache: {config.get('cache_size_gb', 0):.1f}GB")
                    
        except Exception as e:
            logger.debug(f"[FCP] HybridCache config read error: {e}")
            
        return config
    
    def get_mined_concepts(self) -> List[Dict]:
        """Получить список найденных концептов-кандидатов"""
        if hasattr(self, 'concept_miner') and self.concept_miner:
            try:
                return list(self.concept_miner._candidates.values())
            except Exception:
                pass
        return []
    
    # === ContradictionDetector методы (EVA.txt раздел 7.2) ===
    def detect_contradictions(self, concept: str = None) -> List[Dict]:
        """Обнаружить противоречия в графе знаний"""
        if hasattr(self, 'contradiction_detector') and self.contradiction_detector:
            try:
                return self.contradiction_detector.detect_contradictions(concept=concept)
            except Exception as e:
                print(f"[FCP] Contradiction detection error: {e}")
        return []
    
    def get_contradiction_stats(self) -> Dict:
        """Получить статистику обнаруженных противоречий"""
        if hasattr(self, 'contradiction_detector') and self.contradiction_detector:
            try:
                return {
                    "total": len(self.contradiction_detector.detected_contradictions),
                    "last_detection": self.contradiction_detector.last_detection_time,
                    "history_len": len(self.contradiction_detector.detection_history)
                }
            except Exception:
                pass
        return {}
    
    # === UES методы (EVA.txt раздел 8.3) ===
    def optimize_with_ues(self, benchmark_fn=None) -> Dict:
        """Запуск оптимизации через UES"""
        if hasattr(self, 'ues') and self.ues:
            try:
                if benchmark_fn is None:
                    benchmark_fn = self._default_benchmark
                optimal_config = self.ues.optimize_pipeline(benchmark_fn)
                print(f"[FCP] UES optimization completed: {optimal_config}")
                return optimal_config
            except Exception as e:
                print(f"[FCP] UES optimization error: {e}")
        return {}
    
    def _default_benchmark(self, params: Dict[str, int]) -> float:
        """Бенчмарк по умолчанию для UES"""
        try:
            import time
            start = time.time()
            # Простой тест генерации
            test_prompt = "Test"
            config = {"NUM_STREAMS": str(params.get("num_streams", 1)),
                     "INFERENCE_NUM_THREADS": str(params.get("num_threads", 4))}
            # Возвращаем задержку в мс
            return (time.time() - start) * 1000
        except Exception:
            return 100.0
    
    def get_ues_topology(self) -> Dict:
        """Получить топологию вычислительных ресурсов"""
        if hasattr(self, 'ues') and self.ues:
            try:
                return {
                    "units": [{"id": u.id, "type": u.type, "cores": u.cores} 
                             for u in self.ues.topology.units],
                    "total_memory_gb": self.ues.topology.total_memory_gb,
                    "numa_nodes": self.ues.topology.numa_nodes
                }
            except Exception:
                pass
        return {}
    
    def pin_gnn_to_e_cores(self) -> Dict:
        """Привязать GNN к энергоэффективным ядрам"""
        if hasattr(self, 'ues') and self.ues:
            try:
                return self.ues.pin_gnn_to_e_cores()
            except Exception as e:
                print(f"[FCP] GNN pinning error: {e}")
        return {}
    
    def pin_llm_to_p_cores(self) -> Dict:
        """Привязать LLM к производительным ядрам"""
        if hasattr(self, 'ues') and self.ues:
            try:
                return self.ues.pin_llm_to_p_cores()
            except Exception as e:
                print(f"[FCP] LLM pinning error: {e}")
        return {}
    
    # === ReasoningChain методы (накопление цепочки рассуждений) ===
    def start_reasoning_session(self, session_id: str = None) -> str:
        """Начать новую сессию рассуждений"""
        if hasattr(self, 'reasoning_chain') and self.reasoning_chain:
            return self.reasoning_chain.start_session(session_id)
        return None
    
    def end_reasoning_session(self, save_to_tcm: bool = True) -> Dict:
        """Завершить текущую сессию рассуждений"""
        if hasattr(self, 'reasoning_chain') and self.reasoning_chain:
            return self.reasoning_chain.end_session(save_to_tcm)
        return {}
    
    def add_reasoning_step(self, 
                           prompt: str,
                           reasoning: str,
                           conclusion: str,
                           intermediate_claims: List[str] = None,
                           confidence: float = 0.5) -> int:
        """
        Добавить шаг рассуждения в цепочку.
        
        Используется для накопления выводов при многошаговых задачах.
        
        Args:
            prompt: вопрос/задача на этом шаге
            reasoning: текст рассуждения
            conclusion: вывод из рассуждения
            intermediate_claims: промежуточные утверждения
            confidence: уверенность в выводе (0-1)
            
        Returns:
            step_id или -1 если ReasoningChain не инициализирован
        """
        if hasattr(self, 'reasoning_chain') and self.reasoning_chain:
            return self.reasoning_chain.add_step(
                prompt=prompt,
                reasoning=reasoning,
                conclusion=conclusion,
                intermediate_claims=intermediate_claims,
                confidence=confidence
            )
        return -1
    
    def get_reasoning_context(self, max_steps: int = 5) -> str:
        """Получить контекст рассуждений для включения в промпт"""
        if hasattr(self, 'reasoning_chain') and self.reasoning_chain:
            return self.reasoning_chain.get_context(max_steps=max_steps)
        return ""
    
    def get_reasoning_summary(self) -> str:
        """Получить краткую сводку выводов"""
        if hasattr(self, 'reasoning_chain') and self.reasoning_chain:
            return self.reasoning_chain.get_conclusions_summary()
        return ""
    
    def analyze_reasoning_consistency(self) -> Dict:
        """Проанализировать согласованность цепочки рассуждений"""
        if hasattr(self, 'reasoning_chain') and self.reasoning_chain:
            return self.reasoning_chain.analyze_consistency()
        return {}
    
    def clear_reasoning_chain(self):
        """Очистить текущую цепочку рассуждений"""
        if hasattr(self, 'reasoning_chain') and self.reasoning_chain:
            self.reasoning_chain.clear()
    
    def get_reasoning_state(self) -> Dict:
        """Получить состояние цепочки рассуждений"""
        if hasattr(self, 'reasoning_chain') and self.reasoning_chain:
            return self.reasoning_chain.get_state()
        return {}
    
    def restore_reasoning_state(self, state: Dict):
        """Восстановить состояние цепочки рассуждений"""
        if hasattr(self, 'reasoning_chain') and self.reasoning_chain:
            self.reasoning_chain.restore_state(state)
    
    # === Единая конфигурация генерации ===
    def get_generation_config(self, max_new_tokens: int = None) -> ov_genai.GenerationConfig:
        """
        Получить унифицированную конфигурацию генерации.
        Все методы генерации используют этот метод для согласованности.
        
        Args:
            max_new_tokens: максимальное количество токенов (по умолчанию из self.generation_config)
            
        Returns:
            Настроенный ov_genai.GenerationConfig
        """
        gen_cfg = ov_genai.GenerationConfig()
        
        # Применяем настройки из единого конфига
        gen_cfg.temperature = self.generation_config.get("temperature", 0.15)
        gen_cfg.top_p = self.generation_config.get("top_p", 0.85)
        gen_cfg.top_k = self.generation_config.get("top_k", 40)
        gen_cfg.repetition_penalty = self.generation_config.get("repetition_penalty", 1.1)
        gen_cfg.do_sample = self.generation_config.get("do_sample", True)
        
        # max_tokens - может быть переопределен при вызове
        if max_new_tokens is not None:
            gen_cfg.max_new_tokens = max_new_tokens
        else:
            gen_cfg.max_new_tokens = self.generation_config.get("max_new_tokens", 2048)
        
        return gen_cfg
    
    def update_generation_config(self, **kwargs):
        """
        Обновить параметры генерации.
        Пример: pipeline.update_generation_config(temperature=0.2, top_p=0.9)
        """
        self.generation_config.update(kwargs)
        logger.info(f"[FCP] Generation config updated: {kwargs}")
    
    def get_generation_config_summary(self) -> Dict:
        """Получить текущую сводку конфигурации генерации"""
        return self.generation_config.copy()
    
    # === Единая конфигурация KV кэша ===
    def get_kv_cache_config(self) -> Dict:
        """Получить текущую конфигурацию KV кэша"""
        return self.kv_cache_config.copy()
    
    def update_kv_cache_config(self, **kwargs):
        """
        Обновить параметры KV кэша.
        Пример: pipeline.update_kv_cache_config(cache_size_gb=8, enable_prefix_caching=False)
        Примечание: Изменения вступят в силу при следующей инициализации pipeline.
        """
        self.kv_cache_config.update(kwargs)
        logger.info(f"[FCP] KV cache config updated: {kwargs}")
    
    def get_kv_cache_stats(self) -> Dict:
        """Получить статистику использования KV кэша (если доступно)"""
        stats = {"kv_cache": {}, "hybrid_cache": {}}
        
        if hasattr(self, 'pipeline') and self.pipeline:
            try:
                if hasattr(self.pipeline, 'get_cache_stats'):
                    stats["kv_cache"] = self.pipeline.get_cache_stats()
            except Exception:
                pass
        
        # Добавляем статистику из HybridTokenCache
        if hasattr(self, 'hybrid_cache') and self.hybrid_cache:
            try:
                if hasattr(self.hybrid_cache, 'get_stats'):
                    stats["hybrid_cache"] = self.hybrid_cache.get_stats()
                stats["hybrid_cache"]["linked"] = True
                stats["hybrid_cache"]["config"] = self.kv_cache_config.get("linked_hybrid_cache", False)
            except Exception:
                pass
        elif hasattr(self, 'brain') and self.brain:
            if hasattr(self.brain, 'hybrid_cache') and self.brain.hybrid_cache:
                try:
                    if hasattr(self.brain.hybrid_cache, 'get_stats'):
                        stats["hybrid_cache"] = self.brain.hybrid_cache.get_stats()
                    stats["hybrid_cache"]["linked"] = True
                except Exception:
                    pass
        
        return stats
    
    def _sync_cache_with_history(self, user_query: str, assistant_response: str):
        """
        Синхронизировать HybridTokenCache с историей диалога.
        
        После каждого диалогового turn добавляем контекст в кэш
        для улучшения последующих ответов.
        """
        cache = None
        if hasattr(self, 'hybrid_cache') and self.hybrid_cache:
            cache = self.hybrid_cache
        elif hasattr(self, 'brain') and self.brain and hasattr(self.brain, 'hybrid_cache'):
            cache = self.brain.hybrid_cache
        
        if cache is None:
            return
        
        try:
            import re
            words = re.findall(r'\b[A-ZА-Я][a-zа-я]+|\b\d+(?:\.\d+)*\b', user_query + " " + assistant_response)
            entities = list(set(words))[:20]
            
            if hasattr(cache, 'add_context'):
                cache.add_context(
                    session_id="fcp_default",
                    query=user_query,
                    entities=entities,
                    raw_text=f"Q: {user_query}\nA: {assistant_response}",
                    ttl=3600
                )
                logger.debug(f"[FCP] Synced {len(entities)} entities to HybridTokenCache")
        except Exception as e:
            logger.debug(f"[FCP] Cache sync error: {e}")
    
    def link_brain(self, brain):
        """
        Связать FCP pipeline с brain для доступа к HybridTokenCache.
        Вызывается после инициализации brain.
        """
        self.brain = brain
        # Обновляем конфигурацию KV кэша с учётом HybridTokenCache
        if hasattr(brain, 'hybrid_cache') and brain.hybrid_cache:
            self.hybrid_cache = brain.hybrid_cache
            hybrid_config = self._get_hybrid_cache_config()
            if hybrid_config.get("enabled"):
                logger.info(f"[FCP] KV cache integrated with HybridTokenCache: {hybrid_config.get('cache_size_gb', 0):.1f}GB")


def create_fcp_pipeline(model_path: str, graph_path: str = None, **kwargs):
    """Factory function"""
    return FCPPipelineV15(model_path, graph_path, **kwargs)

    def _sync_cache_with_history(self, user_query: str, assistant_response: str):
        """
        Синхронизировать HybridTokenCache с историей диалога.
        
        После каждого диалогового turn добавляем контекст в кэш
        для улучшения последующих ответов.
        """
        # Определяем какой кэш использовать
        cache = None
        if hasattr(self, 'hybrid_cache') and self.hybrid_cache:
            cache = self.hybrid_cache
        elif hasattr(self, 'brain') and self.brain and hasattr(self.brain, 'hybrid_cache'):
            cache = self.brain.hybrid_cache
        
        if cache is None:
            return
        
        try:
            # Извлекаем сущности из запроса и ответа
            entities = []
            
            # Простой извлекатель сущностей: ищем capitalized words и numbers
            import re
            words = re.findall(r'\b[A-ZА-Я][a-zа-я]+|\b\d+(?:\.\d+)*\b', user_query + " " + assistant_response)
            entities = list(set(words))[:20]  # Ограничиваем 20 сущностями
            
            # Добавляем контекст в кэш
            if hasattr(cache, 'add_context'):
                cache.add_context(
                    session_id="fcp_default",
                    query=user_query,
                    entities=entities,
                    raw_text=f"Q: {user_query}\nA: {assistant_response}",
                    ttl=3600  # 1 час
                )
                logger.debug(f"[FCP] Synced {len(entities)} entities to HybridTokenCache")
        except Exception as e:
            logger.debug(f"[FCP] Cache sync error: {e}")
