"""
FCP Core Components - KCA, SRG, ConvergenceController, Types

Реализация на основе Update.txt и FCP:
- Knowledge Conscious Attention (KCA)
- Semantic Relevance Gate (SRG)
- Convergence Controller
- FractalGraphV2
- FCP Types (Subgraph, LayerState, HaltDecision, etc.)
- ReasoningChain (накопление цепочки рассуждений)
"""
import numpy as np
from scipy.special import softmax
import logging

logger = logging.getLogger("eva_ai.fcp_core")

from eva_ai.fcp_core.config import FCPConfig, StackConfig
from eva_ai.memory.fractal_graph_v2 import FractalMemoryGraph, create_fractal_memory_graph
FractalGraphV2 = FractalMemoryGraph
def create_fractal_graph_from_texts(texts):
    return create_fractal_memory_graph()
from eva_ai.fcp_core.hybrid_layer import FractalGatedHybridLayer
from eva_ai.fcp_core.hybrid_stack import HybridStack
from eva_ai.fcp_core.input_layer import InputLayer, LayerState, GraphContext, LayerOutput
from eva_ai.fcp_core.output_layer import OutputLayer, SamplingResult, FCPPipeline
from eva_ai.fcp_core.adaptive_lora import AdaLoRALayer, AdaLoRALinear
from eva_ai.fcp_core.learning_orchestrator import LearningGraphManager
from eva_ai.fcp_core.shadow_lora import ShadowLoRAManager
from eva_ai.fcp_core.reasoning_chain import ReasoningChain, ReasoningChainManager
from eva_ai.fcp_core.types import (
    Subgraph,
    MemorySegment,
    Concept,
    Fact,
    Contradiction,
    ResolutionResult,
    ComputeTopology,
    RequestMetrics,
    LayerState,
    TransformerBlockOutput,
    HaltDecision,
    FusionOutput,
    ExecutionPlan,
    CompiledKernel,
)


class ConvergenceController:
    """
    Детерминированный контроллер сходимости.
    Анализирует историю изменений состояний.
    """
    
    def __init__(self, config: FCPConfig):
        self.max_cycles = config.kca_max_cycles
        self.rho = config.kca_rho
        self.osc_threshold = config.kca_osc_threshold
        self.gate_threshold = config.kca_gate_threshold
        
        self.history_states = []
        self.history_deltas = []
        self.history_gates = []
    
    def check(self, X_current, X_prev, gamma_mean, step_idx):
        """Проверка сходимости"""
        self.history_states.append(X_current.copy())
        self.history_gates.append(gamma_mean)
        
        current_delta = X_current - X_prev
        self.history_deltas.append(current_delta)
        
        # 1. Проверка насыщения гейта (Модель сама отказывается от коррекции)
        if len(self.history_gates) >= 2:
            if all(g < self.gate_threshold for g in self.history_gates[-2:]):
                return "SATURATED", X_current
        
        # 2. Детектор осцилляции (Векторное изменение развернулось на 180 градусов)
        if len(self.history_deltas) >= 2:
            d_curr = self.history_deltas[-1].flatten()
            d_prev = self.history_deltas[-2].flatten()
            
            norm_c = np.linalg.norm(d_curr) + 1e-8
            norm_p = np.linalg.norm(d_prev) + 1e-8
            
            cos_sim = np.dot(d_curr, d_prev) / (norm_c * norm_p)
            
            if cos_sim < self.osc_threshold:
                logger.warning(f"[KCA] Oscillation detected (cos={cos_sim:.2f}). Stabilizing...")
                stable_X = np.mean(self.history_states[-3:] + [X_current, X_prev], axis=0)
                return "OSCILLATION_DETECTED", stable_X
        
        # 3. Жесткий лимит
        if step_idx >= self.max_cycles - 1:
            return "MAX_CYCLES", X_current
        
        return "CONTINUE", X_current
    
    def reset(self):
        """Сброс истории"""
        self.history_states = []
        self.history_deltas = []
        self.history_gates = []


class KnowledgeConsciousAttention:
    """
    KCA: Внимание к графу -> Поиск лакун/противоречий -> Инъекция.
    """
    
    def __init__(self, config: FCPConfig):
        self.config = config
        d = config.hidden_dim
        
        # Веса KCA (инициализация - в проде загружаются из чекпоинта)
        self.W_Qk = np.eye(d) * 0.5
        self.W_Kk = np.eye(d) * 0.5
        self.W_Vk = np.eye(d) * 0.5
        self.W_g = np.eye(2 * d, d) * 0.5
        self.b_g = np.zeros(d)
    
    def forward(self, X_initial: np.ndarray, subgraph: dict):
        """
        X_initial: [T, D] - скрытые состояния модели
        subgraph: dict с ключом 'embeddings' [N, D]
        """
        H = subgraph["embeddings"]
        logger.info(f"[KCA] Processing. Tokens: {X_initial.shape[0]}, Nodes: {H.shape[0]}")
        
        # Subgraph Freezing: K и V фиксируются для всего цикла
        K_k = H @ self.W_Kk
        V_k = H @ self.W_Vk
        
        X_prev = X_initial
        controller = ConvergenceController(self.config)
        
        history_log = {"status": "UNKNOWN", "cycles": 0}
        
        for t in range(self.config.kca_max_cycles):
            # 1. Внимание Токен -> Узел
            Q_k = X_prev @ self.W_Qk
            scores = (Q_k @ K_k.T) / np.sqrt(self.config.hidden_dim)
            A = softmax(scores, axis=-1)  # [T, N]
            
            # 2. Вычисление вектора коррекции E_corr
            
            # A) Lacuna (Лакуны): где внимание МАЛО (1 - A)
            E_lacuna = (1.0 - A) @ V_k
            
            # Б) Contradiction (Противоречия)
            T_dim = A.shape[0]
            E_contra = np.zeros_like(X_prev)
            
            # Попарное сходство узлов в подграфе
            H_norm = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
            node_sims = H_norm @ H_norm.T
            
            for i in range(T_dim):
                # Топ-2 узла для токена i
                top_indices = np.argsort(A[i])[-2:][::-1]
                if len(top_indices) < 2:
                    continue
                u, v = top_indices[0], top_indices[1]
                
                if node_sims[u, v] < self.config.contradiction_sim_threshold:
                    E_contra[i] = H[u] - H[v]
            
            # Итоговая коррекция
            E_corr = self.config.lambda_l * E_lacuna + self.config.lambda_c * E_contra
            
            # 3. Адаптивное затухание (Damping)
            damping = self.config.kca_rho ** t
            E_corr *= damping
            
            # 4. Инъекция через гейт
            concat = np.concatenate([X_prev, E_corr], axis=-1)
            gate_logits = concat @ self.W_g + self.b_g
            gamma = 1.0 / (1.0 + np.exp(-gate_logits))  # Sigmoid
            
            X_new = X_prev + gamma * E_corr
            gamma_mean = np.mean(gamma)
            
            logger.info(f"[KCA] Cycle {t+1}: Gamma={gamma_mean:.4f}, Delta_Norm={np.linalg.norm(X_new-X_prev):.4f}")
            
            # 5. Проверка сходимости
            status, X_out = controller.check(X_new, X_prev, gamma_mean, t)
            
            if status != "CONTINUE":
                history_log["status"] = status
                history_log["cycles"] = t + 1
                logger.info(f"[KCA] Finished. Status: {status}")
                return X_out, history_log
            
            X_prev = X_new
        
        history_log["status"] = "MAX_CYCLES"
        history_log["cycles"] = self.config.kca_max_cycles
        return X_prev, history_log


class SemanticRelevanceGate:
    """
    SRG: Оценка релевантности через косинусное сходство и энтропию.
    """
    
    def __init__(self, config: FCPConfig):
        self.config = config
    
    def evaluate(self, query_vec, response_vec, logits):
        """
        Оценка релевантности.

        Returns:
            tuple: (mode, metrics)
                mode: "direct", "reasoning", или "variational"
                metrics: dict с 'sim' и 'ent'
        """
        # Similarity
        sim = np.dot(query_vec, response_vec) / (
            np.linalg.norm(query_vec) * np.linalg.norm(response_vec) + 1e-8
        )
        
        # Entropy
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Logic
        is_coherent = sim >= self.config.srg_cosine_threshold
        is_confident = entropy <= self.config.srg_entropy_threshold
        
        if is_coherent and is_confident:
            return "direct", {"sim": sim, "ent": entropy}
        elif not is_coherent:
            return "reasoning", {"sim": sim, "ent": entropy}
        else:
            return "variational", {"sim": sim, "ent": entropy}
    
    # evaluate_from_probs removed: mathematically incorrect (treated probabilities as embeddings)
    # If needed, implement using token embeddings to compute expected response embedding