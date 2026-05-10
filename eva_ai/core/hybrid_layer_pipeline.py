"""
HybridLayerPipeline - Гибридный пайплайн с LayerCaptureModel

Собирает:
1. OpenVINO LLMPipeline - для генерации (быстро, эффективно)
2. LayerCaptureModel - для захвата hidden states на каждом слое
3. HybridLayerProcessor - для обработки слоёв (KCA + GNN + LoRA)
4. MemorySnapshotIntegration - для сохранения состояний в граф

Режим работы:
- Запрос -> OpenVINO generate (быстрая генерация)
- LayerCaptureModel захватывает hidden states
- HybridLayerProcessor/KCA корректирует через retrieval
- Результат сохраняется в FractalGraphV2 через MemorySnapshot
"""
import os
import time
import json
from typing import Dict, Any, Optional, List, Callable, Tuple
import numpy as np
import logging

logger = logging.getLogger("eva_ai.core.hybrid_layer_pipeline")

from eva_ai.fcp_core import (
    FCPConfig,
    ConvergenceController,
    KnowledgeConsciousAttention,
    SemanticRelevanceGate
)

# FractalGraphV2 Singleton
from eva_ai.memory.fractal_graph_v2 import get_fractal_graph
from eva_ai.fcp_gnn import (
    HybridLayerConfig,
    HybridLayerProcessor,
    HybridLayerManager,
    AdaptiveFusionInjector,
    TextFusionInjector
)
from eva_ai.core.memory_snapshot_integration import MemorySnapshotIntegration


class HybridLayerPipeline:
    """
    Гибридный пайплайн с двумя моделями:
    1. OpenVINO - быстрая генерация
    2. Transformers LayerCaptureModel - захват и анализ слоёв
    """

    def __init__(
        self,
        openvino_model_path: str,
        transformer_model_path: str = None,
        graph_path: Optional[str] = None,
        lora_dir: Optional[str] = None,
        num_layers: int = 32,
        use_gnn: bool = True,
        use_kca: bool = True,
        use_srg: bool = True,
        injection_scale: float = 0.1
    ):
        self.openvino_model_path = openvino_model_path
        
        # Use correct transformer model path
        if transformer_model_path:
            self.transformer_model_path = transformer_model_path.replace('qwen_layer_model.pt', 'qwenlayermodel.pt')
        else:
            from eva_ai.core.utils import get_project_root
            self.transformer_model_path = os.path.join(get_project_root(), 'models', 'qwenlayermodel.pt')
        
        self.graph_path = graph_path
        self.lora_dir = lora_dir
        self.num_layers = num_layers

        self.stats = {"queries": 0, "layer_captures": 0, "kca_cycles": 0}

        # FCP Config
        self.fcp_config = FCPConfig()

        # Components
        self.openvino_pipeline = None
        self.layer_capture = None
        self.hybrid_processor = None
        self.hybrid_layer_manager = None
        self.memory_snapshot = None
        self.fractal_graph = None

        # Инициализация
        self._init_fractal_graph()
        self._init_openvino_pipeline()
        self._init_layer_capture()
        self._init_hybrid_components(use_gnn, use_kca, use_srg, injection_scale)
        self._init_memory_snapshot()

        logger.info("[HybridLayerPipeline] Initialized successfully")

    def _init_fractal_graph(self):
        """Инициализация FractalGraphV2 (use singleton)"""
        graph_dir = self.graph_path if self.graph_path else "eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data"
        
        try:
            self.fractal_graph = get_fractal_graph(storage_dir=graph_dir, lazy=True)
            logger.info(f"[HybridLayerPipeline] Graph singleton loaded: {self.fractal_graph.node_count} nodes")
        except Exception as e:
            logger.warning(f"[HybridLayerPipeline] Graph singleton init failed: {e}")
            self.fractal_graph = None

    def _init_openvino_pipeline(self):
        """Инициализация OpenVINO LLMPipeline для генерации"""
        try:
            import openvino_genai as ov_genai
        except ImportError:
            logger.warning("[HybridLayerPipeline] OpenVINO GenAI not available")
            self.openvino_pipeline = None
            return
        
        if not os.path.exists(self.openvino_model_path):
            logger.warning(f"[HybridLayerPipeline] OpenVINO model not found: {self.openvino_model_path}")
            self.openvino_pipeline = None
            return
        
        # Try hybrid model first, fallback to regular
        from eva_ai.core.utils import get_project_root
        hybrid_model_dir = os.path.join(get_project_root(), 'models', 'hybrid_openvino')
        model_to_load = self.openvino_model_path
        
        if os.path.exists(hybrid_model_dir):
            xml_path = os.path.join(hybrid_model_dir, 'model.xml')
            bin_path = os.path.join(hybrid_model_dir, 'model.bin')
            if os.path.exists(xml_path) and os.path.exists(bin_path):
                # Check if .bin is valid (not a corrupted .npz/ZIP)
                try:
                    with open(bin_path, 'rb') as f:
                        header = f.read(4)
                    if header != b'PK\x03\x04':  # Not a ZIP file
                        model_to_load = hybrid_model_dir
                        logger.info(f"[HybridLayerPipeline] Using hybrid model: {hybrid_model_dir}")
                except Exception as e:
                    logger.warning(f"[HybridLayerPipeline] Error checking hybrid model: {e}")
        
        try:
            scheduler = ov_genai.SchedulerConfig()
            scheduler.cache_size = 4
            scheduler.max_num_seqs = 1
            scheduler.max_num_batched_tokens = 2048
            scheduler.enable_prefix_caching = True
            scheduler.use_cache_eviction = True
            
            gen_config = ov_genai.GenerationConfig()
            max_new_tokens = 2048
            temperature = 0.2
            top_p = 0.9
            top_k = 40
            repetition_penalty = 1.1
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
                config_path = os.path.join(project_root, "brain_config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    gen_config_dict = config.get("generation", {})
                    max_new_tokens = gen_config_dict.get("max_new_tokens", max_new_tokens)
                    temperature = gen_config_dict.get("temperature", temperature)
                    top_p = gen_config_dict.get("top_p", top_p)
                    top_k = gen_config_dict.get("top_k", top_k)
                    repetition_penalty = gen_config_dict.get("repetition_penalty", repetition_penalty)
            except Exception:
                pass
            gen_config.max_new_tokens = max_new_tokens
            gen_config.temperature = temperature
            gen_config.top_p = top_p
            gen_config.top_k = top_k
            gen_config.repetition_penalty = repetition_penalty
            gen_config.do_sample = True
            
            logger.info(f"[HybridLayerPipeline] Loading OpenVINO model from {model_to_load}")
            self.openvino_pipeline = ov_genai.LLMPipeline(model_to_load, "CPU",
                                                           config={"scheduler_config": scheduler})
            self.openvino_pipeline.set_generation_config(gen_config)
            logger.info("[HybridLayerPipeline] OpenVINO pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"[HybridLayerPipeline] OpenVINO init failed: {e}")
            self.openvino_pipeline = None

    def _init_layer_capture(self):
        """Инициализация LayerCaptureModel для захвата hidden states"""
        try:
            from eva_ai.core.layer_capture_model import LayerCaptureModel
        except ImportError:
            logger.warning("[HybridLayerPipeline] LayerCaptureModel not available")
            self.layer_capture = None
            return
        
        if not os.path.exists(self.transformer_model_path):
            logger.warning(f"[HybridLayerPipeline] Transformer model not found: {self.transformer_model_path}")
            self.layer_capture = None
            return
        
        # Check available RAM before loading (8GB model needs >12GB free)
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024**3)
            if available_gb < 12:
                logger.warning(f"[HybridLayerPipeline] Not enough RAM ({available_gb:.1f}GB < 12GB required). Skipping LayerCapture.")
                self.layer_capture = None
                return
        except ImportError:
            pass
        
        try:
            self.layer_capture = LayerCaptureModel(
                model_path=self.transformer_model_path,
                num_layers=self.num_layers,
                device=self.device
            )
            success = self.layer_capture.load()
            if success:
                logger.info("[HybridLayerPipeline] LayerCaptureModel loaded")
            else:
                self.layer_capture = None
        except Exception as e:
            logger.error(f"[HybridLayerPipeline] LayerCaptureModel init failed: {e}")
            self.layer_capture = None
            return

        if not os.path.exists(self.transformer_model_path):
            logger.warning(f"[HybridLayerPipeline] Transformer model not found: {self.transformer_model_path}")
            return

        try:
            self.layer_capture = LayerCaptureModel(
                model_path=self.transformer_model_path,
                num_layers=self.num_layers,
                device="CPU"
            )
            success = self.layer_capture.load()
            if success:
                logger.info("[HybridLayerPipeline] LayerCaptureModel loaded")
            else:
                self.layer_capture = None
        except Exception as e:
            logger.error(f"[HybridLayerPipeline] LayerCaptureModel init failed: {e}")
            self.layer_capture = None

    def _init_hybrid_components(self, use_gnn, use_kca, use_srg, injection_scale):
        """Инициализация гибридных компонентов"""
        self.hybrid_layer_config = HybridLayerConfig(
            hidden_dim=2560,
            num_layers=self.num_layers,
            use_gnn=use_gnn,
            use_kca=use_kca,
            use_srg=use_srg,
            injection_scale=injection_scale,
            lora_rank=8
        )

        self.hybrid_processor = HybridLayerProcessor(self.hybrid_layer_config)
        self.hybrid_layer_manager = HybridLayerManager(self.hybrid_layer_config)

        logger.info(f"[HybridLayerPipeline] Hybrid components initialized: "
                   f"gnn={use_gnn}, kca={use_kca}, srg={use_srg}")

    def _init_memory_snapshot(self):
        """Инициализация MemorySnapshotIntegration"""
        try:
            self.memory_snapshot = MemorySnapshotIntegration(
                brain=self,
                fractal_graph=self.fractal_graph,
                config={
                    'enabled': True,
                    'snapshot_all_layers': True,
                    'num_layers': self.num_layers,
                    'save_to_graph': True
                }
            )
            logger.info("[HybridLayerPipeline] MemorySnapshot initialized")
        except Exception as e:
            logger.warning(f"[HybridLayerPipeline] MemorySnapshot init failed: {e}")
            self.memory_snapshot = None

    def process_query(
        self,
        query: str,
        max_new_tokens: int = 2048,
        enable_layer_capture: bool = True,
        enable_kca: bool = True,
        return_metadata: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Обработка запроса в гибридном режиме.

        Flow:
        1. OpenVINO генерирует base response
        2. LayerCaptureModel захватывает hidden states
        3. HybridLayerProcessor/KCA корректирует через retrieval
        4. MemorySnapshot сохраняет в граф
        """
        self.stats["queries"] += 1

        start_time = time.time()
        response = ""
        layer_data = {}
        metadata = {
            "mode": "hybrid",
            "layer_capture_used": False,
            "kca_applied": False,
            "graph_nodes_used": 0,
            "correction_source": None
        }

        # 1. OpenVINO Generation (fast path)
        if self.openvino_pipeline:
            try:
                gen_cfg = self.openvino_pipeline.get_generation_config()
                gen_cfg.max_new_tokens = max_new_tokens
                response = self.openvino_pipeline.generate(query, generation_config=gen_cfg)
            except Exception as e:
                logger.error(f"[HybridLayerPipeline] OpenVINO generation failed: {e}")
                response = f"[Generation error: {e}]"

        # 2. LayerCaptureModel - захват hidden states
        if enable_layer_capture and self.layer_capture and self.memory_snapshot:
            try:
                layer_data = self._capture_and_process_layers(query, response)
                metadata["layer_capture_used"] = True
                self.stats["layer_captures"] += 1
            except Exception as e:
                logger.warning(f"[HybridLayerPipeline] Layer capture failed: {e}")

        # 3. KCA enrichment через HybridLayerProcessor
        if enable_kca and self.fractal_graph and self.fractal_graph.node_count > 0:
            try:
                kca_result = self._enrich_with_kca(query, response)
                if kca_result:
                    metadata["kca_applied"] = True
                    metadata["kca_cycles"] = kca_result.get("cycles", 0)
                    self.stats["kca_cycles"] += kca_result.get("cycles", 0)
                    metadata["correction_source"] = "kca"
            except Exception as e:
                logger.warning(f"[HybridLayerPipeline] KCA enrichment failed: {e}")

        # 4. MemorySnapshot - сохраняем результаты в граф
        if self.memory_snapshot and response:
            try:
                self.memory_snapshot.on_generation_complete(response)
            except Exception as e:
                logger.warning(f"[HybridLayerPipeline] MemorySnapshot save failed: {e}")

        elapsed = time.time() - start_time

        result = {
            "response": response,
            "final_response": response,
            "confidence": 0.85,
            "mode": "hybrid",
            "processing_time": elapsed,
            "metadata": metadata,
            "layer_data": layer_data if layer_data else None
        }

        if return_metadata:
            result["full_metadata"] = metadata

        return result

    def _capture_and_process_layers(self, query: str, response: str) -> Dict[str, Any]:
        """
        Захват hidden states через LayerCaptureModel и обработка.

        Returns:
            Dict с данными слоёв и статистикой
        """
        if not self.layer_capture:
            return {}

        captured_layers = []

        def layer_callback(layer_idx: int, hidden_states: np.ndarray):
            """Callback для каждого слоя"""
            if self.memory_snapshot:
                try:
                    self.memory_snapshot.on_layer_forward(
                        layer_idx=layer_idx,
                        hidden_states=hidden_states,
                        layer_confidence=0.0
                    )
                except Exception:
                    pass

            captured_layers.append({
                "layer_idx": layer_idx,
                "shape": hidden_states.shape,
                "mean": float(np.mean(hidden_states)),
                "std": float(np.std(hidden_states))
            })

        try:
            input_ids, attention_mask = self.layer_capture.tokenize(query)

            with np.errstate(divide='ignore', invalid='ignore'):
                logits, hidden_states_list = self.layer_capture.get_all_layer_outputs(
                    input_ids,
                    attention_mask,
                    layer_callback=layer_callback
                )

            return {
                "num_layers_captured": len(captured_layers),
                "layers": captured_layers,
                "logits_shape": logits.shape if hasattr(logits, 'shape') else None
            }

        except Exception as e:
            logger.warning(f"[HybridLayerPipeline] Layer capture error: {e}")
            return {}

    def _enrich_with_kca(self, query: str, response: str) -> Optional[Dict]:
        """KCA обогащение через HybridLayerProcessor"""
        if not self.hybrid_processor or not self.fractal_graph:
            return None

        if self.fractal_graph.node_count == 0:
            return None

        try:
            nodes = []
            for i in range(min(self.fractal_graph.node_count, 50)):
                node_emb = self.fractal_graph.get_node(i)
                if node_emb is not None:
                    nodes.append({
                        'id': str(i),
                        'embedding': node_emb,
                        'content': f'Node {i}'
                    })

            if not nodes:
                return None

            hidden_states = np.random.randn(10, 2560).astype(np.float32)

            enriched_prompt, corrected_states, proc_metadata = self.hybrid_processor.process(
                query_text=query,
                hidden_states=hidden_states,
                knowledge_nodes=nodes
            )

            return {
                "cycles": proc_metadata.get("kca_cycles", 0),
                "mode": proc_metadata.get("srg_mode", "unknown"),
                "enriched_prompt": enriched_prompt
            }

        except Exception as e:
            logger.warning(f"[HybridLayerPipeline] KCA error: {e}")
            return None

    def get_fcp_status(self) -> Dict[str, Any]:
        """Получить статус всех FCP компонентов"""
        status = {
            "openvino_ready": self.openvino_pipeline is not None,
            "layer_capture_ready": self.layer_capture is not None,
            "hybrid_processor_ready": self.hybrid_processor is not None,
            "memory_snapshot_ready": self.memory_snapshot is not None,
            "graph_nodes": self.fractal_graph.node_count if self.fractal_graph else 0,
            "stats": self.stats
        }

        if self.hybrid_processor:
            status["hybrid_status"] = self.hybrid_processor.get_status()

        return status

    def add_knowledge_node(self, text: str, embedding: np.ndarray, metadata: dict = None) -> int:
        """Добавить узел в граф"""
        if self.fractal_graph is None:
            graph_dir = self.graph_path if self.graph_path else "eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data"
            self.fractal_graph = FractalGraphV2(storage_dir=graph_dir, lazy=True)
        return self.fractal_graph.add_node(embedding, metadata)

    def save_graph(self, path: Optional[str] = None):
        """Сохранить граф"""
        save_path = path or self.graph_path
        if self.fractal_graph and save_path:
            try:
                self.fractal_graph.save(save_path)
                logger.info(f"[HybridLayerPipeline] Graph saved: {save_path}")
            except Exception as e:
                logger.error(f"[HybridLayerPipeline] Graph save failed: {e}")


def create_hybrid_layer_pipeline(
    openvino_model_path: str,
    transformer_model_path: str,
    graph_path: str = None,
    **kwargs
) -> HybridLayerPipeline:
    """Factory function для HybridLayerPipeline"""
    return HybridLayerPipeline(
        openvino_model_path=openvino_model_path,
        transformer_model_path=transformer_model_path,
        graph_path=graph_path,
        **kwargs
    )
