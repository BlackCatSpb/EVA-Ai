"""
OnlineTrainer - Фоновое обучение GNN и LoRA во время работы EVA
Непрерывное обучение с паузой при генерации текста моделью.
"""
import os
import time
import threading
import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from enum import Enum

logger = logging.getLogger("eva_ai.online_trainer")

# Event types for EventBus integration
from eva_ai.core.event_bus import Event, EventTypes


class TrainerState(Enum):
    """Состояние тренера - для управления ресурсами"""
    IDLE = "idle"           # Обучение активно
    BUSY = "busy"           # Модель генерирует - обучение пауза
    TRAINING = "training"   # Идёт обучение
    COMPLETED = "completed"  # Обучение завершено


class ResourceManager:
    """
    Управление ресурсами - определяет когда можно обучаться.
    """
    
    def __init__(self):
        self._state = TrainerState.IDLE
        self._lock = threading.Lock()
        self._generation_count = 0
        self._last_generation_time = 0
    
    def set_generating(self):
        """Модель начинает генерацию - запретить обучение"""
        with self._lock:
            self._state = TrainerState.BUSY
            self._generation_count += 1
            self._last_generation_time = time.time()
            logger.debug("[ResourceManager] State: BUSY (generation)")
    
    def set_idle(self):
        """Модель закончила генерацию - разрешить обучение"""
        with self._lock:
            self._state = TrainerState.IDLE
            logger.debug("[ResourceManager] State: IDLE")
    
    def can_train(self) -> bool:
        """Можно ли обучаться"""
        with self._lock:
            return self._state == TrainerState.IDLE
    
    def get_state(self) -> TrainerState:
        """Получить текущее состояние"""
        with self._lock:
            return self._state


class GPUManager:
    """
    Управление GPU для обучения.
    Автоматически использует GPU если доступен.
    ПОЛНОСТЬЮ ПЕРЕВЕДЕНО НА GPU - обучение только на GPU!
    """
    
    def __init__(self, min_memory_mb: int = 128):
        self.min_memory_mb = min_memory_mb
        self.device = None
        self.available = False
        self._init_gpu()
    
    def _init_gpu(self):
        """Инициализировать GPU - ТОЛЬКО GPU, CPU не используется!"""
        try:
            import torch
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                self.available = True
                logger.info(f"[GPUManager] GPU initialized: cuda:0 - Training ONLY on GPU!")
            else:
                self.device = None
                self.available = False
                logger.warning("[GPUManager] CUDA not available - Training DISABLED!")
        except Exception as e:
            logger.error(f"[GPUManager] GPU init failed: {e}")
            self.device = None
    
    def is_available(self) -> bool:
        """Доступен ли GPU - проверяем динамически при каждом запросе."""
        if not self.available:
            return False
        
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            
            props = torch.cuda.get_device_properties(0)
            free_mem = props.total_memory - torch.cuda.memory_allocated()
            free_mb = free_mem / (1024 * 1024)
            
            # GPU доступен если свободно больше 64MB
            available = free_mb >= 64
            if available:
                logger.debug(f"[GPUManager] GPU: {free_mb:.0f}MB free, available={available}")
            return available
        except Exception as e:
            logger.debug(f"[GPUManager] Check failed: {e}")
            return False
    
    def get_device(self):
        """Получить устройство - ТОЛЬКО GPU!"""
        if self.is_available():
            return self.device
        logger.error("[GPUManager] No GPU available - training not possible!")
        return None
    
    def get_memory_info(self) -> Dict[str, float]:
        """Получить информацию о памяти GPU."""
        try:
            import torch
            if not torch.cuda.is_available():
                return {"total_mb": 0, "free_mb": 0, "used_mb": 0}
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory / (1024 * 1024)
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            free = total - allocated
            return {"total_mb": total, "free_mb": free, "used_mb": allocated}
        except Exception:
            return {"total_mb": 0, "free_mb": 0, "used_mb": 0}


class BackgroundTrainer:
    """
    Базовый класс для непрерывного фонового обучения.
    ПОЛНОСТЬЮ НА GPU - обучение выполняется ТОЛЬКО на GPU!
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "eva_ai/training/checkpoints",
        save_interval: int = 100,
        resource_manager: Optional[ResourceManager] = None,
        use_gpu: bool = True,
        brain: Any = None
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_interval = save_interval
        self.resource_manager = resource_manager
        self._brain = brain
        
        # GPU менеджер - ПОЛНОСТЬЮ GPU
        self.gpu_manager = GPUManager() if use_gpu else None
        
        # Проверяем доступность GPU
        if self.gpu_manager and not self.gpu_manager.is_available():
            logger.warning("[BackgroundTrainer] GPU not available! Training will be PAUSED!")
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._step_count = 0
        self._total_steps = None
        self._losses: List[float] = []
        
        self._lock = threading.Lock()
        self._ready = False
        self._state = TrainerState.IDLE
    
    def start(self):
        """Запустить непрерывное обучение."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._continuous_training_loop, daemon=True, name=f"{self.__class__.__name__}_thread")
        self._thread.start()
        logger.info(f"[{self.__class__.__name__}] Started continuous training")
    
    def stop(self):
        """Остановить обучение."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=30)
        logger.info(f"[{self.__class__.__name__}] Stopped")
    
    def _continuous_training_loop(self):
        """Непрерывный цикл обучения - работает пока running=True"""
        logger.info(f"[{self.__class__.__name__}] Training loop started")
        
        while self._running:
            try:
                # Проверяем завершение
                if self._state == TrainerState.COMPLETED:
                    logger.info(f"[{self.__class__.__name__}] Already completed, exiting loop")
                    break
                
                # Проверяем состояние ресурсов
                if self.resource_manager and not self.resource_manager.can_train():
                    time.sleep(0.5)
                    continue
                
                # Один шаг обучения
                success = self._do_training_step()
                
                if success:
                    with self._lock:
                        self._step_count += 1
                        
                        # Проверяем прогресс
                        if self._step_count % 10 == 0:
                            avg_loss = sum(self._losses[-10:]) / min(10, len(self._losses))
                            msg = f"[{self.__class__.__name__}] Step {self._step_count}, loss={avg_loss:.6f}"
                            logger.info(msg)
                            print(msg)  # Принудительный вывод
                        
                        # Сохраняем чекпоинт
                        if self._step_count % self.save_interval == 0:
                            self.save_checkpoint()
                            
                        # Периодическая пересборка индекса графа (для новых узлов)
                        if hasattr(self, '_reload_interval') and self._reload_interval and self._step_count % self._reload_interval == 0:
                            if hasattr(self, '_init_graph_indexer'):
                                self._init_graph_indexer()
                                logger.info(f"[{self.__class__.__name__}] Graph index rebuilt, vectors: {len(self._graph_indexer._hnsw_index) if self._graph_indexer and self._graph_indexer._index_built else 0}")
                            
                        # Проверяем завершение
                        if self._total_steps is not None and self._step_count >= self._total_steps:
                            self._state = TrainerState.COMPLETED
                            logger.info(f"[{self.__class__.__name__}] Training COMPLETED!")
                            break
                
                # Минимальная задержка между шагами
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"[{self.__class__.__name__}] Step error: {e}")
                time.sleep(1)  # Пауза при ошибке
        
        logger.info(f"[{self.__class__.__name__}] Training loop exited")
    
    def _do_training_step(self) -> bool:
        """Один шаг обучения (переопределить в подклассе)."""
        return False
    
    def save_checkpoint(self, suffix: str = ""):
        """Сохранить чекпоинт - должен быть реализован в подклассах."""
        raise NotImplementedError("save_checkpoint must be implemented by subclass")
    
    def load_latest_checkpoint(self) -> bool:
        """Загрузить последний чекпоинт (поддерживает .safetensors и .pt)."""
        import torch
        
        # Сначала ищем .safetensors
        safetensors_checkpoints = list(self.checkpoint_dir.glob("lora_model*.safetensors"))
        pt_checkpoints = list(self.checkpoint_dir.glob("lora_model*.pt"))
        
        latest = None
        if safetensors_checkpoints:
            latest = max(safetensors_checkpoints, key=os.path.getmtime)
            return self._load_safetensors_checkpoint(latest)
        elif pt_checkpoints:
            latest = max(pt_checkpoints, key=os.path.getmtime)
            return self._load_pt_checkpoint(latest)
        else:
            return False
    
    def _load_safetensors_checkpoint(self, path):
        """Загрузить чекпоинт из .safetensors."""
        try:
            from safetensors.torch import load_file
            state = load_file(path)
            # Reconstruct per-layer state dicts
            layer_states = {}
            for key, tensor in state.items():
                # key format: "layer_name.weight", etc.
                if '.' in key:
                    layer_name, param_name = key.split('.', 1)
                    if layer_name not in layer_states:
                        layer_states[layer_name] = {}
                    layer_states[layer_name][param_name] = tensor
            for name, state_dict in layer_states.items():
                if name in self._lora_layers:
                    self._lora_layers[name].load_state_dict(state_dict)
            # Load metadata from .pt file if exists
            meta_path = path.with_suffix('').with_suffix('.pt')  # remove .safetensors, add .pt? Actually meta file is lora_model_meta.pt
            meta_path = path.parent / (path.stem + "_meta.pt")
            if meta_path.exists():
                meta = torch.load(meta_path, weights_only=True)
                self._optimizer.load_state_dict(meta['optimizer'])
                self._step_count = meta.get('step_count', 0)
                self._losses = meta.get('losses', [])
            logger.info(f"[LoRATrainer] Loaded (safetensors): {path}, step={self._step_count}")
            return True
        except Exception as e:
            logger.warning(f"[LoRATrainer] Load failed (safetensors): {e}")
            return False
    
    def _load_pt_checkpoint(self, path):
        """Загрузить чекпоинт из .pt (старый формат)."""
        try:
            checkpoint = torch.load(path, weights_only=True)
            for name, state in checkpoint['layers'].items():
                if name in self._lora_layers:
                    self._lora_layers[name].load_state_dict(state)
            self._optimizer.load_state_dict(checkpoint['optimizer'])
            self._step_count = checkpoint.get('step_count',0)
            self._losses = checkpoint.get('losses', [])
            logger.info(f"[LoRATrainer] Loaded (pt): {path}, step={self._step_count}")
            return True
        except Exception as e:
            logger.warning(f"[LoRATrainer] Load failed (pt): {e}")
            return False
    
    def is_ready(self) -> bool:
        """Готов ли к использованию."""
        return self._ready
    
    def get_progress(self) -> float:
        """Получить прогресс обучения (0.0 - 1.0)"""
        with self._lock:
            return min(1.0, self._step_count / self._total_steps)
    
    def is_completed(self) -> bool:
        """Завершено ли обучение"""
        return self._state == TrainerState.COMPLETED
    
    def get_avg_loss(self) -> float:
        """Средний loss"""
        with self._lock:
            if not self._losses:
                return 0.0
            return sum(self._losses[-100:]) / min(100, len(self._losses))


class GNNTrainer(BackgroundTrainer):
    """
    Непрерывное обучение GNN на эмбеддингах FractalGraphV2.
    Использует индекс графа (GraphIndexer) для интеллектуального выбора данных:
    - Не повторяет одни и те же данные циклически
    - Приоритизирует узлы с низким количеством шагов обучения
    - Динамически подхватывает новые узлы графа
    """
    
    # Антонимный словарь (русский) — копия из contradiction_miner.py
    ANTONYM_MAP = {
        'быстрый': ['медленный', 'медленнее', 'медленно'],
        'медленный': ['быстрый', 'быстрее', 'быстро'],
        'хороший': ['плохой', 'худший'],
        'плохой': ['хороший', 'лучший'],
        'высокий': ['низкий', 'ниже', 'низко'],
        'низкий': ['высокий', 'выше', 'высоко'],
        'большой': ['маленький', 'меньше', 'мало'],
        'маленький': ['большой', 'больше', 'много'],
        'да': ['нет', 'не', 'никогда'],
        'нет': ['да', 'всегда'],
        'всегда': ['никогда', 'редко'],
        'никогда': ['всегда', 'часто'],
        'правда': ['ложь', 'враньё', 'неправда'],
        'ложь': ['правда', 'истина'],
        'истина': ['ложь', 'враньё'],
        'важно': ['неважно', 'второстепенно'],
        'неважно': ['важно', 'главное'],
        'нужно': ['нельзя', 'запрещено'],
        'можно': ['нельзя', 'запрещено'],
        'верно': ['неверно', 'ошибочно'],
        'неверно': ['верно', 'правильно'],
    }

    DEFAULT_QWEN_DB = "eva_ai/fcp_core/data/qwen_knowledge.db"

    def __init__(
        self,
        graph_db_path: str = "eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db",
        qwen_db_path: str = None,
        input_dim: int = 768,
        total_steps: int = None,
        **kwargs
    ):
        super().__init__(
            checkpoint_dir="eva_ai/training/checkpoints/gnn",
            save_interval=200,
            **kwargs
        )
        self.graph_db_path = graph_db_path
        self.input_dim = input_dim
        self._total_steps = total_steps
        
        self._model = None
        self._optimizer = None
        self._loss_fn = None

        # Индекс графа для интеллектуального выбора данных
        self._graph_indexer = None
        self._node_training_counts = {}
        self._reload_interval = 1000

        # Rare concept boost (minesweeper inversion)
        self._node_gap_history = {}
        self._rare_concept_threshold = 5
        self._rare_boost_factor = 2.0

        # Cluster centroid pull
        self._cluster_pull_interval = 500
        self._cluster_pull_strength = 0.05

        # Qwen knowledge distillation
        self.qwen_db_path = qwen_db_path or self.DEFAULT_QWEN_DB
        self._qwen_available = os.path.exists(self.qwen_db_path)
        if self._qwen_available:
            logger.info(f"[GNNTrainer] Qwen knowledge DB found: {self.qwen_db_path}")
        else:
            logger.info(f"[GNNTrainer] No Qwen knowledge DB at {self.qwen_db_path}, using default graph")

        self._init_model()
        self._init_graph_indexer()
    
    def _init_model(self):
        """Инициализировать GNN модель - ТОЛЬКО GPU!"""
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            
            # Определить устройство - ТОЛЬКО GPU
            if self.gpu_manager and self.gpu_manager.is_available():
                self.device = self.gpu_manager.get_device()
            else:
                logger.error("[GNNTrainer] GPU not available! Cannot initialize training!")
                self._ready = False
                return
            
            class MiniGNN(nn.Module):
                """
                Graph Neural Network Encoder (EVA.txt раздел 9.1).
                Преобразует подграф (список эмбеддингов узлов) в:
                - graph_vector: объединённый вектор графа
                - gate_weights: веса для модуляции внимания (36 слоёв)
                """
                def __init__(self, input_dim=768, hidden=256, output=512, num_layers=36, device=None):
                    super().__init__()
                    self._device = device
                    self.input_dim = input_dim
                    self.hidden = hidden
                    self.output = output
                    self.num_layers = num_layers
                    
                    # Encoder: [batch, input_dim] -> [batch, output]
                    self.encoder1 = nn.Linear(input_dim, hidden)
                    self.encoder2 = nn.Linear(hidden, output)
                    
                    # Decoder: для реконструкции (self-supervised learning)
                    self.decoder1 = nn.Linear(output, hidden)
                    self.decoder2 = nn.Linear(hidden, input_dim)
                    
                    # Graph vector projection: output -> graph_vector_dim
                    self.graph_proj = nn.Linear(output, 256)
                    
                    # Gate weights generator: graph_vector -> 36 layer gates
                    # Каждый слой получает свой gate weight [0, 1]
                    self.gate_mlp = nn.Sequential(
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, num_layers)
                    )
                    
                    # Attention projection (для subgraph pooling)
                    self.attention = nn.Linear(output, 1)
                    
                    # Projection layer (для self-supervised обучения)
                    self.proj = nn.Linear(input_dim, input_dim)
                    
                    if device:
                        self.to(device)
                
                def encode(self, node_embeddings: torch.Tensor) -> tuple:
                    """
                    Кодирует подграф в graph_vector и gate_weights.
                    
                    Args:
                        node_embeddings: [batch, input_dim] - эмбеддинги узлов графа
                        
                    Returns:
                        (graph_vector, gate_weights)
                        - graph_vector: [256] - объединённый вектор графа
                        - gate_weights: [36] - веса гейтов для каждого слоя модели
                    """
                    # Encode: input -> hidden -> output
                    h = F.relu(self.encoder1(node_embeddings))
                    encoded = F.relu(self.encoder2(h))  # [batch, output]
                    
                    # Graph-level pooling (attention-based pooling)
                    attn_scores = self.attention(encoded)  # [batch, 1]
                    attn_weights = F.softmax(attn_scores, dim=0)
                    graph_vector = (encoded * attn_weights).sum(dim=0)  # [output]
                    
                    # Project to graph_vector_dim
                    graph_vector = self.graph_proj(graph_vector)  # [256]
                    
                    # Generate gate weights for 36 layers
                    gate_logits = self.gate_mlp(graph_vector)  # [num_layers]
                    gate_weights = torch.sigmoid(gate_logits)  # [num_layers], values in [0, 1]
                    
                    return graph_vector, gate_weights
                
                def forward(self, x):
                    """Forward для self-supervised обучения (реконструкция)."""
                    h = F.relu(self.encoder1(x))
                    encoded = F.relu(self.encoder2(h))
                    h = F.relu(self.decoder1(encoded))
                    reconstructed = self.decoder2(h)
                    projected = self.proj(reconstructed)
                    return projected
            
            self._model = MiniGNN(input_dim=self.input_dim, device=self.device)
            
            # Только proj обучаем (минимальная нагрузка)
            for param in self._model.parameters():
                param.requires_grad = False
            for param in self._model.proj.parameters():
                param.requires_grad = True
            
            self._optimizer = torch.optim.SGD(
                [p for p in self._model.parameters() if p.requires_grad],
                lr=0.01
            )
            
            # Все на GPU - оптимизация CUDA
            torch.backends.cudnn.benchmark = True
            
            self._ready = True
            logger.info(f"[GNNTrainer] Model initialized on GPU: {self.input_dim}d input, device={self.device}")
            
        except Exception as e:
            logger.error(f"[GNNTrainer] Init failed: {e}")
            self._ready = False
    
    def _init_graph_indexer(self):
        """Инициализировать индекс графа для интеллектуального выбора данных.
        
        Индекс НЕ строится при инициализации - только при первом использовании
        (ленивая инициализация), чтобы не сканировать 1.9GB БД при старте.
        """
        try:
            from eva_ai.memory.fractal_graph_v2.graph_indexer import GraphIndexer
            if not os.path.exists(self.graph_db_path):
                logger.warning(f"[GNNTrainer] Graph DB not found, using synthetic fallback")
                self._graph_indexer = None
                return
            self._graph_indexer = GraphIndexer(self.graph_db_path, embedding_dim=self.input_dim)
            logger.info(f"[GNNTrainer] GraphIndexer created (lazy init, will build on first use)")
        except Exception as e:
            logger.warning(f"[GNNTrainer] GraphIndexer init failed: {e}")
            self._graph_indexer = None
    
    def _add_synthetic_fallback(self):
        """Минимальный синтетический fallback (только если нет данных графа)."""
        import numpy as np
        self._synthetic_batch = [np.random.randn(self.input_dim).astype(np.float32) * 0.1 for _ in range(8)]
        logger.warning("[GNNTrainer] Using synthetic fallback data (no graph data)")
    
    def _load_batch(self):
        """Загрузить батч. Возвращает (tensor, metadata_list) или (None, [])."""
        import torch
        import numpy as np
        import sqlite3
        
        # 1. Попробовать через GraphIndexer (HNSW)
        if self._graph_indexer and self._graph_indexer._index_built:
            try:
                query_emb = np.random.randn(self.input_dim).astype(np.float32)
                results = self._graph_indexer.search(query_emb.tolist(), top_k=8, min_similarity=0.0)
                if results:
                    batch = []
                    metadata = []
                    for res in results:
                        node_id = res.get("id")
                        emb = res.get("embedding")
                        content = res.get("content", "")
                        if emb and len(emb) >= self.input_dim:
                            arr = np.array(emb[:self.input_dim], dtype=np.float32)
                            batch.append(arr)
                            metadata.append({"id": node_id, "content": str(content)})
                            self._node_training_counts[node_id] = self._node_training_counts.get(node_id, 0) + 1
                    if batch:
                        tensor = torch.tensor(np.array(batch), dtype=torch.float32).to(self.device)
                        return tensor, metadata
            except Exception as e:
                logger.debug(f"[GNNTrainer] Index search failed: {e}")
        
        # 2. Fallback: SQL запрос с content
        try:
            if not os.path.exists(self.graph_db_path):
                if not hasattr(self, '_synthetic_batch'):
                    self._add_synthetic_fallback()
                batch = self._synthetic_batch[:8]
                dummy_meta = [{"id": f"synth_{i}", "content": ""} for i in range(len(batch))]
                return torch.tensor(np.array(batch), dtype=torch.float32).to(self.device), dummy_meta
            
            conn = sqlite3.connect(self.graph_db_path)
            cur = conn.cursor()
            cur.execute("""
                SELECT id, content, embedding FROM nodes 
                WHERE embedding IS NOT NULL 
                ORDER BY RANDOM() 
                LIMIT 8
            """)
            rows = cur.fetchall()
            conn.close()
            
            batch = []
            metadata = []
            for row in rows:
                node_id, content, emb = row
                if not emb:
                    continue
                try:
                    if isinstance(emb, (bytes, bytearray)):
                        arr = np.frombuffer(bytes(emb), dtype=np.float32).copy()
                    elif isinstance(emb, str):
                        arr = np.array([float(x) for x in emb.split(',')], dtype=np.float32)
                    else:
                        arr = np.array(emb, dtype=np.float32)
                    if len(arr) >= self.input_dim:
                        arr = arr[:self.input_dim]
                    elif len(arr) > 0:
                        arr = np.pad(arr, (0, self.input_dim - len(arr)))
                    else:
                        continue
                    batch.append(arr)
                    metadata.append({"id": node_id, "content": str(content)})
                    self._node_training_counts[node_id] = self._node_training_counts.get(node_id, 0) + 1
                except Exception as e:
                    logger.debug(f"[GNNTrainer] Parse error: {e}")
            if batch:
                tensor = torch.tensor(np.array(batch), dtype=torch.float32).to(self.device)
                return tensor, metadata
        except Exception as e:
            logger.debug(f"[GNNTrainer] SQL fallback failed: {e}")
        
        # 3. Синтетический fallback
        if not hasattr(self, '_synthetic_batch'):
            self._add_synthetic_fallback()
        batch = self._synthetic_batch[:8]
        dummy_meta = [{"id": f"synth_{i}", "content": ""} for i in range(len(batch))]
        return torch.tensor(np.array(batch), dtype=torch.float32).to(self.device), dummy_meta
    
    def _load_qwen_batch(self, batch_size: int = 8):
        """Загрузить батч пар (src_emb, tgt_emb, weight) из Qwen knowledge DB.

        Returns:
            (src_tensor, tgt_tensor, weight_tensor) или (None, None, None)
        """
        import sqlite3
        import numpy as np
        try:
            conn = sqlite3.connect(self.qwen_db_path)
            cur = conn.cursor()
            cur.execute("""
                SELECT n1.embedding, n2.embedding, e.weight
                FROM edges e
                JOIN nodes n1 ON e.source_id = n1.id
                JOIN nodes n2 ON e.target_id = n2.id
                WHERE n1.embedding IS NOT NULL
                  AND n2.embedding IS NOT NULL
                  AND ABS(e.weight - 0.5) < 0.45
                ORDER BY RANDOM()
                LIMIT ?
            """, (batch_size,))
            rows = cur.fetchall()
            conn.close()

            if len(rows) < 2:
                return None, None, None

            src_list, tgt_list, w_list = [], [], []
            for row in rows:
                src_emb = np.frombuffer(bytes(row[0]), dtype=np.float32).copy()
                tgt_emb = np.frombuffer(bytes(row[1]), dtype=np.float32).copy()
                if len(src_emb) >= self.input_dim:
                    src_emb = src_emb[:self.input_dim]
                    tgt_emb = tgt_emb[:self.input_dim]
                else:
                    continue
                src_list.append(src_emb)
                tgt_list.append(tgt_emb)
                w_list.append(float(row[2]))

            if not src_list:
                return None, None, None

            src_t = torch.tensor(np.array(src_list), dtype=torch.float32).to(self.device)
            tgt_t = torch.tensor(np.array(tgt_list), dtype=torch.float32).to(self.device)
            w_t = torch.tensor(w_list, dtype=torch.float32).to(self.device)
            return src_t, tgt_t, w_t

        except Exception as e:
            logger.debug(f"[GNNTrainer] Qwen batch error: {e}")
            return None, None, None

    def _check_antonym_pair(self, content1: str, content2: str) -> bool:
        """Проверить, являются ли два текста антонимами по словарю."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        for word in words1:
            if word in self.ANTONYM_MAP:
                if any(ant in words2 for ant in self.ANTONYM_MAP[word]):
                    return True
        return False

    def _do_training_step(self) -> bool:
        """Один шаг обучения GNN."""
        if not self._ready or self._model is None:
            return False
        
        try:
            import torch
            import torch.nn.functional as F
            
            # Проверить GPU доступность
            if self.device.type == "cuda" and self.gpu_manager:
                if not self.gpu_manager.is_available():
                    return False
            
            # Загрузить батч + метаданные
            x, metadata = self._load_batch()
            if x is None:
                return False
            if isinstance(x, torch.Tensor):
                x = x.to(self.device)
            else:
                return False
            
            self._model.train()
            
            # MiniGNN.encode(): subgraph embeddings -> (graph_vector, gate_weights)
            graph_vector, gate_weights = self._model.encode(x)
            
            # Forward через decoder для self-supervised learning
            h = F.relu(self._model.encoder1(x))
            encoded = F.relu(self._model.encoder2(h))
            h = F.relu(self._model.decoder1(encoded))
            reconstructed = F.relu(self._model.decoder2(h))
            graph_emb = self._model.proj(reconstructed)  # [batch, 768]
            
            # TASK 1: Contradiction Detection + Antonym Repel
            batch_size = graph_emb.size(0)
            contradiction_scores = []
            for i in range(batch_size - 1):
                emb_i = graph_emb[i]
                emb_j = graph_emb[i + 1]
                cos_sim = F.cosine_similarity(emb_i.unsqueeze(0), emb_j.unsqueeze(0))
                
                # Антонимный repel: если пара — антонимы, усиливаем отталкивание
                antonym_boost = 1.0
                if i < len(metadata) and (i + 1) < len(metadata):
                    c1 = metadata[i].get("content", "")
                    c2 = metadata[i + 1].get("content", "")
                    if c1 and c2 and self._check_antonym_pair(c1, c2):
                        antonym_boost = 2.0  # удвоенный сигнал противоречия
                
                score = (-cos_sim).clamp(min=0) * antonym_boost
                contradiction_scores.append(score)
            
            if contradiction_scores:
                contradiction_tensor = torch.stack(contradiction_scores)
                target_contradiction = (contradiction_tensor > 0.3).float()
            else:
                target_contradiction = torch.zeros(1, device=graph_emb.device)
            
            pred_contradiction = self._model.proj(graph_emb[:max(1, len(contradiction_scores))])
            if pred_contradiction.numel() > 0:
                pred_contr_mean = pred_contradiction.mean(dim=1, keepdim=True)
                target_contr = target_contradiction.view(-1, 1) if target_contradiction.dim() == 1 else target_contradiction
                loss_contradiction = F.binary_cross_entropy_with_logits(
                    pred_contr_mean,
                    target_contr
                )
            else:
                loss_contradiction = torch.tensor(0.0, device=self.device)
            
            # TASK 2: Knowledge Gap Detection + Rare Concept Boost (minesweeper inversion)
            emb_variance = graph_emb.var(dim=1)
            emb_sparsity = (graph_emb.abs() < 0.1).float().mean(dim=1)
            
            gap_score = (1 - emb_variance / (emb_variance.max() + 1e-8)) * emb_sparsity
            
            # Minesweeper inversion: редкие концепты с высокой ошибкой получают буст
            gap_targets = []
            for i in range(batch_size):
                base_gap = gap_score[i].item()
                multiplier = 1.0
                
                # Сохраняем gap_score в историю узла
                if i < len(metadata):
                    nid = metadata[i].get("id", "")
                    if nid:
                        if nid not in self._node_gap_history:
                            self._node_gap_history[nid] = []
                        self._node_gap_history[nid].append(base_gap)
                        if len(self._node_gap_history[nid]) > 20:
                            self._node_gap_history[nid] = self._node_gap_history[nid][-20:]
                        
                        # Если узел редкий (мало шагов обучения) и ошибка высокая — бустим
                        train_count = self._node_training_counts.get(nid, 0)
                        if train_count < self._rare_concept_threshold and base_gap > 0.3:
                            multiplier = self._rare_boost_factor
                
                adjusted_gap = min(1.0, base_gap * multiplier)
                gap_targets.append(adjusted_gap)
            
            gap_target_tensor = torch.tensor(gap_targets, device=graph_emb.device)
            gap_target_binary = (gap_target_tensor > 0.5).float()
            
            pred_gap = self._model.proj(graph_emb)
            pred_gap_mean = pred_gap.mean(dim=1, keepdim=True)
            target_gap = gap_target_binary.view(-1, 1) if gap_target_binary.dim() == 1 else gap_target_binary
            loss_gap = F.binary_cross_entropy_with_logits(
                pred_gap_mean,
                target_gap
            )
            
            # Combined loss: KCA tasks
            loss = loss_contradiction + loss_gap
            
            # Backward
            self._optimizer.zero_grad()
            if loss.requires_grad:
                loss.backward()
                self._optimizer.step()
            
            self._model.eval()
            
            # Периодический cluster centroid pull
            if self._step_count > 0 and self._step_count % self._cluster_pull_interval == 0:
                try:
                    self._cluster_centroid_pull()
                except Exception as pull_e:
                    logger.debug(f"[GNNTrainer] Cluster pull error: {pull_e}")
            
            # Очистить GPU кэш
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            # Сохранить loss
            with self._lock:
                self._losses.append(loss.item())
            
            return True
            
        except Exception as e:
            logger.debug(f"[GNNTrainer] Step error: {e}")
            return False

    def _cluster_centroid_pull(self):
        """Cluster centroid pull: дотягивание эмбеддингов к центроиду кластера.

        Предотвращает разреженность в пределах одной семантической группы.
        """
        import sqlite3
        import numpy as np
        try:
            conn = sqlite3.connect(self.graph_db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # Группируем узлы по parent_group_id
            cur.execute("""
                SELECT id, parent_group_id, embedding FROM nodes
                WHERE embedding IS NOT NULL AND parent_group_id IS NOT NULL
            """)
            rows = cur.fetchall()

            groups = {}
            for row in rows:
                gid = row["parent_group_id"]
                emb_bytes = row["embedding"]
                if not gid or not emb_bytes:
                    continue
                emb = np.frombuffer(bytes(emb_bytes), dtype=np.float32)
                if len(emb) < 8:
                    continue
                groups.setdefault(gid, []).append((row["id"], emb))

            updates = []
            for gid, members in groups.items():
                if len(members) < 2:
                    continue
                centroid = np.mean([m[1] for m in members], axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                for node_id, emb in members:
                    pulled = emb + self._cluster_pull_strength * (centroid - emb)
                    pulled = pulled / (np.linalg.norm(pulled) + 1e-8)
                    updates.append((pulled.tobytes(), node_id))

            if updates:
                cur.executemany(
                    "UPDATE nodes SET embedding = ? WHERE id = ?",
                    updates
                )
                conn.commit()
                logger.debug(f"[GNNTrainer] Cluster pull: {len(updates)} nodes in {len(groups)} groups")
            conn.close()
        except Exception as e:
            logger.debug(f"[GNNTrainer] Cluster pull error: {e}")

    def save_checkpoint(self, suffix: str = ""):
        """Сохранить веса GNN."""
        if self._model is None:
            return
        
        import torch
        path = self.checkpoint_dir / f"gnn_model{suffix}.pt"
        torch.save({
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'step_count': self._step_count,
            'losses': self._losses[-100:]
        }, path)
        logger.info(f"[GNNTrainer] Saved: {path}")
        
        # Bridge to HybridLayerProcessor: сохранить веса в numpy формате
        self._save_for_hybrid_processor(path, suffix)
    
    def load_latest_checkpoint(self) -> bool:
        """Загрузить последний чекпоинт."""
        import torch
        import glob
        
        checkpoints = list(self.checkpoint_dir.glob("gnn_model*.pt"))
        if not checkpoints:
            return False
        
        latest = max(checkpoints, key=os.path.getmtime)
        try:
            checkpoint = torch.load(latest, weights_only=True)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self._step_count = checkpoint.get('step_count', 0)
            self._losses = checkpoint.get('losses', [])
            logger.info(f"[GNNTrainer] Loaded: {latest}, step={self._step_count}")
            return True
        except Exception as e:
            logger.warning(f"[GNNTrainer] Load failed: {e}")
            return False
    
    def get_encoder(self):
        """Получить обученный энкодер."""
        return self._model
    
    def get_graph_output(self, node_embeddings) -> Tuple[np.ndarray, np.ndarray]:
        """
        Получить graph_vector и gate_weights для инъекции в модель.
        
        Args:
            node_embeddings: torch.Tensor [batch, 768] или np.ndarray
        
        Returns:
            (graph_vector, gate_weights)
            - graph_vector: [256] numpy array
            - gate_weights: [36] numpy array
        """
        import numpy as np
        import torch
        if self._model is None:
            return np.zeros(256, dtype=np.float32), np.zeros(36, dtype=np.float32)
        
        # Convert to tensor if needed
        if isinstance(node_embeddings, np.ndarray):
            node_embeddings = torch.from_numpy(node_embeddings).float()
        
        self._model.eval()
        with torch.no_grad():
            graph_vec, gates = self._model.encode(node_embeddings.to(self.device))
            return graph_vec.cpu().numpy(), gates.cpu().numpy()
    
    def get_layer_count(self) -> int:
        """Количество слоёв модели (36 для Qwen)."""
        return 36
    
    def _save_for_hybrid_processor(self, torch_checkpoint_path: Path, suffix: str = ""):
        """
        Конвертировать PyTorch веса GNN в numpy формат для HybridLayerProcessor.
        
        MiniGNN (768 -> 256 -> 512) -> FractalGraphEncoderLocal (384 -> 512 -> 2560)
        
        Конвертация:
        1. encoder1.weight (768, 256) -> W_agg (384, 512) через transpose + проекцию
        2. encoder2.weight (256, 512) -> W_update (512, 2560) через проекцию
        3. proj.weight (768, 768) -> пропускается (не используется в encode)
        """
        try:
            import torch
            import numpy as np
            
            # Загрузить PyTorch чекпоинт
            checkpoint = torch.load(torch_checkpoint_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('model_state_dict', {})
            
            # Создать numpy версию в формате HybridLayerProcessor
            # FractalGraphEncoderLocal ожидает: W_agg, W_update, W_gate, W_proj, lora_A, lora_B
            
            np_weights = {}
            
            # 1. W_agg: проекция encoder1 весов
            if 'encoder1.weight' in state_dict:
                enc1 = state_dict['encoder1.weight'].numpy()  # (256, 768)
                # Транспонируем: (768, 256) -> проекция на (384, 512)
                enc1_t = enc1.T  # (768, 256)
                # Берём первые 384 входных измерений и 512 выходных
                np_weights['W_agg'] = enc1_t[:384, :256].astype(np.float32) * 0.02
                
            if 'encoder1.bias' in state_dict:
                np_weights['b_agg'] = state_dict['encoder1.bias'].numpy()[:256].astype(np.float32)
            
            # 2. W_update: проекция encoder2 весов
            if 'encoder2.weight' in state_dict:
                enc2 = state_dict['encoder2.weight'].numpy()  # (512, 256)
                # Транспонируем: (256, 512) -> проекция на (512, 2560)
                enc2_t = enc2.T  # (256, 512)
                # Выход 512 -> 2560
                np_weights['W_update'] = enc2_t.astype(np.float32)
                # Масштабируем для большего выходного измерения
                np_weights['W_update'] = np_weights['W_update'] * 2.0
                
            if 'encoder2.bias' in state_dict:
                np_weights['b_update'] = state_dict['encoder2.bias'].numpy().astype(np.float32)
            
            # 3. W_gate: инициализировать случайно (GNNTrainer не обучает)
            np_weights['W_gate'] = np.random.randn(2 * 2560, 2560).astype(np.float32) * 0.02
            
            # 4. W_proj: инициализировать случайно
            np_weights['W_proj'] = np.random.randn(2560, 2560).astype(np.float32) * 0.02
            
            # 5. LoRA адаптер
            np_weights['lora_A'] = np.random.randn(2560, 8).astype(np.float32) * 0.02
            np_weights['lora_B'] = np.random.randn(8, 2560).astype(np.float32) * 0.02
            
            # Сохранить в формате graph_encoder.pt (HybridLayerProcessor ожидает torch)
            np_weights_torch = {k: torch.from_numpy(v) for k, v in np_weights.items()}
            
            hybrid_save_path = self.checkpoint_dir / f"gnn_for_hybrid{suffix}.pt"
            torch.save(np_weights_torch, hybrid_save_path)
            logger.info(f"[GNNTrainer] Saved hybrid weights to: {hybrid_save_path}")
            
        except Exception as e:
            logger.warning(f"[GNNTrainer] Failed to save hybrid weights: {e}")


class LoRATrainer(BackgroundTrainer):
    """
    Непрерывное обучение LoRA на данных графа.
    Использует индекс графа (GraphIndexer) для интеллектуального выбора данных:
    - Не повторяет одни и те же данные циклически
    - Приоритизирует узлы с низким количеством шагов обучения
    - Динамически подхватывает новые узлы графа
    """
    
    def __init__(
        self,
        conversation_history: List[Dict] = None,
        total_steps: int = None,
        graph_db_path: str = "eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db",
        brain: Any = None,
        active_lora_dir: str = "eva_ai/models/lora",
        **kwargs
    ):
        self._conversation_history = conversation_history or []
        self._total_steps = total_steps
        self.graph_db_path = graph_db_path
        self._lora_layers = {}
        self._optimizer = None
        
        # Индекс графа для интеллектуального выбора данных
        self._graph_indexer = None
        self._node_training_counts = {}
        self._reload_interval = 1000
        
        super().__init__(
            checkpoint_dir="eva_ai/training/checkpoints/lora",
            save_interval=100,
            brain=brain,
            **kwargs
        )
        
        self._init_model()
        self._init_graph_indexer()
        
        # Set active LoRA dir for FCPipeline sync
        self.set_active_lora_dir(active_lora_dir)
    
    def _init_model(self):
        """Инициализировать LoRA слои - ТОЛЬКО GPU!"""
        try:
            import torch
            import torch.nn as nn
            
            # Определить устройство - ТОЛЬКО GPU
            if self.gpu_manager and self.gpu_manager.is_available():
                self.device = self.gpu_manager.get_device()
            else:
                logger.error("[LoRATrainer] GPU not available! Cannot initialize training!")
                self._ready = False
                return
            
            class LoRALayer(nn.Module):
                def __init__(self, in_features, out_features, rank=4, device=None):
                    super().__init__()
                    self.rank = rank
                    self.lora_a = nn.Parameter(torch.randn(rank, in_features) * 0.01)
                    self.lora_b = nn.Parameter(torch.randn(out_features, rank) * 0.01)
                    self.scaling = 1.0
                    if device:
                        self.to(device)
                
                def forward(self, x):
                    return x + x @ self.lora_a.T @ self.lora_b.T * self.scaling
            
            # Используем 768-мерные эмбеддинги (как в GNN)
            # Иерархические ранги LoRA согласно EVA.txt
            num_layers = 36  # предполагаемое количество слоев модели
            self._lora_layers = {}
            for i in range(num_layers):
                if i < 8:
                    rank = 4
                elif i < 16:
                    rank = 8
                else:
                    rank = 16
                self._lora_layers[f"layer_{i}_q_proj"] = LoRALayer(768, 768, rank=rank, device=self.device)
                self._lora_layers[f"layer_{i}_v_proj"] = LoRALayer(768, 768, rank=rank, device=self.device)
            
            # Переместить на устройство
            for layer in self._lora_layers.values():
                layer.to(self.device)
            
            params = []
            for layer in self._lora_layers.values():
                params.extend(list(layer.parameters()))
            
            self._optimizer = torch.optim.AdamW(params, lr=0.001)
            
            # Все на GPU - оптимизация CUDA
            torch.backends.cudnn.benchmark = True
            
            self._ready = True
            logger.info(f"[LoRATrainer] LoRA layers initialized on GPU: {self.device}")
            
        except Exception as e:
            logger.error(f"[LoRATrainer] Init failed: {e}")
            self._ready = False
    
    def _init_graph_indexer(self):
        """Инициализировать индекс графа для интеллектуального выбора данных.
        
        Индекс НЕ строится при инициализации - только при первом использовании
        (ленивая инициализация), чтобы не сканировать 1.9GB БД при старте.
        """
        try:
            from eva_ai.memory.fractal_graph_v2.graph_indexer import GraphIndexer
            if not os.path.exists(self.graph_db_path):
                logger.warning(f"[LoRATrainer] Graph DB not found, using synthetic fallback")
                self._graph_indexer = None
                return
            self._graph_indexer = GraphIndexer(self.graph_db_path, embedding_dim=768)
            logger.info(f"[LoRATrainer] GraphIndexer created (lazy init, will build on first use)")
        except Exception as e:
            logger.warning(f"[LoRATrainer] GraphIndexer init failed: {e}")
            self._graph_indexer = None
    
    def _add_synthetic_fallback(self):
        """Минимальный синтетический fallback."""
        import numpy as np
        self._synthetic_batch = [np.random.randn(768).astype(np.float32) * 0.1 for _ in range(8)]
        logger.warning("[LoRATrainer] Using synthetic fallback data")
    
    def update_history(self, history: List[Dict]):
        """Обновить историю диалогов."""
        self._conversation_history = history
    
    def _load_batch(self):
        """Загрузить батч через индекс графа (интеллектуальный выбор, без повторов)."""
        import torch
        import numpy as np
        import sqlite3
        
        # 1. Попробовать через GraphIndexer (HNSW)
        if self._graph_indexer and self._graph_indexer._index_built:
            try:
                query_emb = np.random.randn(768).astype(np.float32)
                results = self._graph_indexer.search(query_emb.tolist(), top_k=8, min_similarity=0.0)
                if results:
                    batch = []
                    for res in results:
                        node_id = res.get("id")
                        emb = res.get("embedding")
                        if emb and len(emb) >= 768:
                            arr = np.array(emb[:768], dtype=np.float32)
                            batch.append(arr)
                            self._node_training_counts[node_id] = self._node_training_counts.get(node_id, 0) + 1
                    if batch:
                        return torch.tensor(np.array(batch), dtype=torch.float32).to(self.device)
            except Exception as e:
                logger.debug(f"[LoRATrainer] Index search failed: {e}")
        
        # 2. Fallback: SQL запрос
        try:
            if not os.path.exists(self.graph_db_path):
                if not hasattr(self, '_synthetic_batch'):
                    self._add_synthetic_fallback()
                batch = self._synthetic_batch[:8]
                return torch.tensor(np.array(batch), dtype=torch.float32).to(self.device)
            
            conn = sqlite3.connect(self.graph_db_path)
            cur = conn.cursor()
            cur.execute("""
                SELECT id, embedding FROM nodes 
                WHERE embedding IS NOT NULL 
                ORDER BY RANDOM() 
                LIMIT 8
            """)
            rows = cur.fetchall()
            conn.close()
            
            batch = []
            for row in rows:
                node_id, emb = row
                if not emb:
                    continue
                try:
                    if isinstance(emb, (bytes, bytearray)):
                        arr = np.frombuffer(bytes(emb), dtype=np.float32).copy()
                    elif isinstance(emb, str):
                        arr = np.array([float(x) for x in emb.split(',')], dtype=np.float32)
                    else:
                        arr = np.array(emb, dtype=np.float32)
                    if len(arr) >= 768:
                        arr = arr[:768]
                    elif len(arr) > 0:
                        arr = np.pad(arr, (0, 768 - len(arr)))
                    else:
                        continue
                    batch.append(arr)
                    self._node_training_counts[node_id] = self._node_training_counts.get(node_id, 0) + 1
                except Exception as e:
                    logger.debug(f"[LoRATrainer] Parse error: {e}")
            if batch:
                return torch.tensor(np.array(batch), dtype=torch.float32).to(self.device)
        except Exception as e:
            logger.debug(f"[LoRATrainer] SQL fallback failed: {e}")
        
        # 3. Синтетический fallback
        if not hasattr(self, '_synthetic_batch'):
            self._add_synthetic_fallback()
        batch = self._synthetic_batch[:8]
        return torch.tensor(np.array(batch), dtype=torch.float32).to(self.device)
    
    def _do_training_step(self) -> bool:
        """Один шаг обучения LoRA на данных графа."""
        if not self._ready:
            return False
        
        try:
            import torch
            import torch.nn.functional as F
            
            # Проверить GPU доступность
            if self.device.type == "cuda" and self.gpu_manager:
                if not self.gpu_manager.is_available():
                    return False
            
            # Загрузить батч из графа (вместо dummy input)
            batch = self._load_batch()
            if batch is None:
                return False
            
            # Verify batch quality
            batch_mean = batch.mean().item()
            batch_std = batch.std().item()
            logger.info(f"[LoRATrainer] Batch stats: shape={batch.shape}, mean={batch_mean:.4f}, std={batch_std:.4f}")
            
            # Check if batch appears synthetic (low variance)
            if batch_std < 1e-6:
                logger.warning("[LoRATrainer] Batch appears synthetic (low variance). Check graph data loading!")
            
            # Log whether we're using graph indexer or fallback
            if self._graph_indexer and hasattr(self._graph_indexer, '_index_built') and self._graph_indexer._index_built:
                logger.info("[LoRATrainer] Using GraphIndexer for batch selection")
            else:
                logger.warning("[LoRATrainer] GraphIndexer not available, using SQL or synthetic fallback")
            
            for layer in self._lora_layers.values():
                layer.train()
            
            # LoRA Training: Full Integration with LearningOrchestrator
            # Per EVA.txt:
            # - LoRA rank distribution: 1-8 → rank 4 (grammar/style), 
            #                          9-16 → rank 8 (domain-specific), 
            #                          17-36 → rank 16 (reasoning)
            # - Training triggered when LearningOrchestrator detects low success_rate
            # - Uses LayerSensitivity and LearningSignal feedback
            
            # Try to get brain's LearningGraphManager for real feedback signals
            learning_manager = None
            
            # Check brain reference
            if self._brain is None:
                logger.warning("[LoRATrainer] No brain reference - cannot access LearningGraphManager")
            else:
                logger.info(f"[LoRATrainer] Brain reference present: {type(self._brain).__name__}")
            
            if self._brain and hasattr(self._brain, 'learning_graph_manager'):
                learning_manager = self._brain.learning_graph_manager
            
            if learning_manager and hasattr(learning_manager, 'signals') and learning_manager.signals:
                # Use real LearningSignal data from conversations
                recent_signals = learning_manager.signals[-100:] if len(learning_manager.signals) >= 100 else learning_manager.signals
                
                # Extract training targets from feedback signals
                signal_features = []
                signal_targets = []
                for sig in recent_signals:
                    # Feature: domain embedding (one-hot encoded)
                    domain_idx = learning_manager.domains.index(sig.domain) if sig.domain in learning_manager.domains else 0
                    domain_emb = torch.zeros(len(learning_manager.domains), device=self.device)
                    domain_emb[domain_idx] = 1.0
                    
                    # Feature: confidence score
                    conf_emb = torch.tensor([sig.confidence], device=self.device)
                    
                    # Target: success (1) or failure (0)
                    signal_features.append(torch.cat([domain_emb, conf_emb]))
                    signal_targets.append(1.0 if sig.success else 0.0)
                
                if signal_features:
                    features = torch.stack(signal_features)  # [N, domains + 1]
                    targets = torch.tensor(signal_targets, device=self.device)
                    
                    # Forward through LoRA: adapt features
                    adapted = batch[:len(features)].clone()
                    for layer in self._lora_layers.values():
                        adapted = layer(adapted)
                    
                    # Predict success probability from adapted features
                    pred_success = adapted.mean(dim=1)
                    loss = F.binary_cross_entropy_with_logits(pred_success, targets)
                else:
                    loss = torch.tensor(0.0, device=self.device)
            else:
                # Fallback: learn from LayerSensitivity patterns (success rate by domain)
                # Simulate learning which embeddings improve low-success layers
                
                # Extract embedding statistics that correlate with success
                emb_norm = batch.norm(dim=1)
                emb_kurtosis = ((batch - batch.mean(dim=0)).pow(4)).mean(dim=1) / (batch.var(dim=1).pow(2) + 1e-8)
                
                # Target: high success for embeddings with balanced statistics
                success_proxy = torch.sigmoid(emb_norm * 0.1 + emb_kurtosis * 0.01)
                
                # Forward through LoRA
                transformed = batch.clone()
                for layer in self._lora_layers.values():
                    transformed = layer(transformed)
                
                pred_success = transformed.mean(dim=1)
                loss = F.binary_cross_entropy_with_logits(pred_success, success_proxy.detach())
            
            if loss.requires_grad:
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                
                # Log gradient norm
                total_norm = 0.0
                for layer in self._lora_layers.values():
                    for p in layer.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                logger.info(f"[LoRATrainer] Loss: {loss.item():.6f}, Gradient norm: {total_norm:.6f}")
            
            for layer in self._lora_layers.values():
                layer.eval()
            
            # Очистить GPU кэш
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            with self._lock:
                self._losses.append(loss.item())
            
            return True
            
        except Exception as e:
            logger.debug(f"[LoRATrainer] Step error: {e}")
            return False
    
    def save_checkpoint(self, suffix: str = ""):
        """Сохранить LoRA веса в формате safetensors для OpenVINO GenAI."""
        import torch
        try:
            from safetensors.torch import save_file
            # Flatten state dict for safetensors
            flat_state = {}
            for name, layer in self._lora_layers.items():
                layer_state = layer.state_dict()
                for k, v in layer_state.items():
                    flat_state[f"{name}.{k}"] = v
            # Save only LoRA layers in safetensors
            path = self.checkpoint_dir / f"lora_model{suffix}.safetensors"
            save_file(flat_state, path)
            # Save optimizer and metadata separately in .pt file
            meta_path = self.checkpoint_dir / f"lora_model{suffix}_meta.pt"
            torch.save({
                'optimizer': self._optimizer.state_dict(),
                'step_count': self._step_count,
                'losses': self._losses[-100:]
            }, meta_path)
            logger.info(f"[LoRATrainer] Saved: {path}")

            # Sync to active LoRA directory for FCPipeline
            self._sync_to_active_lora_dir(path)

        except ImportError:
            # Fallback to .pt if safetensors not available
            state = {
                name: layer.state_dict()
                for name, layer in self._lora_layers.items()
            }
            path = self.checkpoint_dir / f"lora_model{suffix}.pt"
            torch.save({
                'layers': state,
                'optimizer': self._optimizer.state_dict(),
                'step_count': self._step_count,
                'losses': self._losses[-100:]
            }, path)
            logger.info(f"[LoRATrainer] Saved (fallback .pt): {path}")

            # Sync .pt fallback too (FCPipeline checks for .safetensors first)
            self._sync_pt_to_active_lora_dir(path)
    
    def load_latest_checkpoint(self) -> bool:
        """Загрузить последний чекпоинт (поддерживает .safetensors и .pt)."""
        import torch
        
        # Сначала ищем .safetensors
        safetensors_checkpoints = list(self.checkpoint_dir.glob("lora_model*.safetensors"))
        pt_checkpoints = list(self.checkpoint_dir.glob("lora_model*.pt"))
        
        latest = None
        if safetensors_checkpoints:
            latest = max(safetensors_checkpoints, key=os.path.getmtime)
            return self._load_safetensors_checkpoint(latest)
        elif pt_checkpoints:
            latest = max(pt_checkpoints, key=os.path.getmtime)
            return self._load_pt_checkpoint(latest)
        else:
            return False
        
        latest = max(checkpoints, key=os.path.getmtime)
        try:
            checkpoint = torch.load(latest, weights_only=True)
            for name, state in checkpoint['layers'].items():
                if name in self._lora_layers:
                    self._lora_layers[name].load_state_dict(state)
            self._optimizer.load_state_dict(checkpoint['optimizer'])
            self._step_count = checkpoint.get('step_count', 0)
            self._losses = checkpoint.get('losses', [])
            logger.info(f"[LoRATrainer] Loaded: {latest}, step={self._step_count}")
            return True
        except Exception as e:
            logger.warning(f"[LoRATrainer] Load failed: {e}")
            return False
    
    def get_lora_layers(self):
        """Получить обученные LoRA слои."""
        return self._lora_layers

    def set_active_lora_dir(self, path: str):
        """Установить директорию для активных LoRA адаптеров (FCPipeline)."""
        self._active_lora_dir = Path(path)
        self._active_lora_dir.mkdir(parents=True, exist_ok=True)

    def _sync_to_active_lora_dir(self, checkpoint_path: Path):
        """Синхронизировать чекпоинт в активную LoRA директорию."""
        if not hasattr(self, '_active_lora_dir') or not self._active_lora_dir:
            # Default: sync to eva_ai/models/lora
            active_dir = Path("eva_ai/models/lora")
            active_dir.mkdir(parents=True, exist_ok=True)
            self._active_lora_dir = active_dir
        try:
            from safetensors.torch import load_file, save_file
            # Load checkpoint weights
            weights = load_file(checkpoint_path)
            # Save to active lora dir
            active_path = self._active_lora_dir / "lora_model.safetensors"
            save_file(weights, active_path)
            logger.info(f"[LoRATrainer] Synced to active LoRA dir: {active_path}")
        except Exception as e:
            logger.warning(f"[LoRATrainer] Sync to active LoRA dir failed: {e}")

    def _sync_pt_to_active_lora_dir(self, pt_path: Path):
        """Синхронизировать .pt чекпоинт в активную LoRA директорию."""
        if not hasattr(self, '_active_lora_dir') or not self._active_lora_dir:
            active_dir = Path("eva_ai/models/lora")
            active_dir.mkdir(parents=True, exist_ok=True)
            self._active_lora_dir = active_dir
        try:
            import torch
            checkpoint = torch.load(pt_path, weights_only=True)
            layers = checkpoint.get('layers', {})
            flat_state = {}
            for name, state_dict in layers.items():
                for k, v in state_dict.items():
                    flat_state[f"{name}.{k}"] = v
            from safetensors.torch import save_file
            active_path = self._active_lora_dir / "lora_model.safetensors"
            save_file(flat_state, active_path)
            logger.info(f"[LoRATrainer] Synced .pt to active LoRA dir: {active_path}")
        except Exception as e:
            logger.warning(f"[LoRATrainer] Sync .pt to active LoRA dir failed: {e}")


class HotSwapManager:
    """
    Горячая замена весов без перезагрузки.
    """
    
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self._gnn_trainer: Optional[GNNTrainer] = None
        self._lora_trainer: Optional[LoRATrainer] = None
        
        self._active_gnn = None
        self._active_lora = None
        self._update_counter = 0
    
    def register_gnn(self, trainer: GNNTrainer):
        """Зарегистрировать GNN тренер."""
        self._gnn_trainer = trainer
    
    def register_lora(self, trainer: LoRATrainer):
        """Зарегистрировать LoRA тренер."""
        self._lora_trainer = trainer
    
    def update(self):
        """Обновить активные веса (периодически)."""
        self._update_counter += 1
        
        if self._update_counter % 50 == 0:
            # GNN swap
            if self._gnn_trainer and self._gnn_trainer.is_ready():
                encoder = self._gnn_trainer.get_encoder()
                if encoder is not None:
                    self._active_gnn = encoder
                    progress = self._gnn_trainer.get_progress()
                    logger.info(f"[HotSwap] GNN updated, progress={progress:.1%}")
            
            # LoRA swap
            if self._lora_trainer and self._lora_trainer.is_ready():
                layers = self._lora_trainer.get_lora_layers()
                if layers:
                    self._active_lora = layers
                    progress = self._lora_trainer.get_progress()
                    logger.info(f"[HotSwap] LoRA updated, progress={progress:.1%}")
    
    def get_active_gnn(self):
        """Получить активный GNN энкодер."""
        return self._active_gnn
    
    def get_active_lora(self):
        """Получить активные LoRA слои."""
        return self._active_lora
    
    def is_gnn_ready(self) -> bool:
        """GNN готов к использованию"""
        return self._active_gnn is not None and self._gnn_trainer is not None and self._gnn_trainer.is_completed() is False
    
    def is_lora_ready(self) -> bool:
        """LoRA готов к использованию"""
        return self._active_lora is not None and self._lora_trainer is not None and self._lora_trainer.is_completed() is False


class OnlineTrainerManager:
    """
    Главный менеджер онлайн обучения.
    Интегрируется в CoreBrain/FCP Pipeline.
    """
    
    def __init__(self, config: Optional[Dict] = None, brain: Any = None):
        config = config or {}
        self._brain = brain
        
        # GPU настройка
        use_gpu = config.get("use_gpu", True)  # По умолчанию использовать GPU
        
        # Resource Manager - контролирует когда можно обучаться
        self.resource_manager = ResourceManager()
        
        # Тренеры с GPU поддержкой и доступом к brain для LearningOrchestrator
        self.gnn_trainer = GNNTrainer(
            resource_manager=self.resource_manager,
            total_steps=config.get("gnn_total_steps", 5000),
            use_gpu=use_gpu,
            brain=brain
        )
        
        self.lora_trainer = LoRATrainer(
            resource_manager=self.resource_manager,
            total_steps=config.get("lora_total_steps", 100000),
            use_gpu=use_gpu,
            brain=brain,
            active_lora_dir=config.get("lora", {}).get("adapters_dir", "eva_ai/models/lora")
        )
        
        self.hot_swap = HotSwapManager(self.resource_manager)
        self.hot_swap.register_gnn(self.gnn_trainer)
        self.hot_swap.register_lora(self.lora_trainer)
        
        self._enabled = config.get("enabled", True)
        
        # Подписка на EventBus для управления через события
        self._event_bus = None
        if brain and hasattr(brain, '_new_event_bus'):
            self._event_bus = brain._new_event_bus
            self._subscribe_to_events()
    
    def _subscribe_to_events(self):
        """Подписаться на события EventBus"""
        if self._event_bus is None:
            return
        from eva_ai.core.event_bus import EventTypes, Event
        self._event_bus.subscribe(EventTypes.SYSTEM_STOP, self._on_system_stop, priority=1)
        self._event_bus.subscribe(EventTypes.LEARNING_STARTED, self._on_learning_started, priority=1)
        self._event_bus.subscribe(EventTypes.LEARNING_COMPLETED, self._on_learning_completed, priority=1)
        logger.info("[OnlineTrainer] Subscribed to EventBus")
    
    def _on_system_stop(self, event: Event):
        """Обработка события остановки системы"""
        logger.info("[OnlineTrainer] Received SYSTEM_STOP event")
        self.stop()
    
    def _on_learning_started(self, event: Event):
        logger.debug(f"[OnlineTrainer] Learning started: {event.data}")
    
    def _on_learning_completed(self, event: Event):
        logger.debug(f"[OnlineTrainer] Learning completed: {event.data}")
    
    def start(self):
        """Запустить всех тренеров."""
        if not self._enabled:
            logger.info("[OnlineTrainer] Disabled in config")
            return
        
        logger.info(f"[OnlineTrainer] GNN ready: {self.gnn_trainer._ready}, LoRA ready: {self.lora_trainer._ready}")
        
        # Загрузить последние чекпоинты
        gnn_loaded = self.gnn_trainer.load_latest_checkpoint()
        lora_loaded = self.lora_trainer.load_latest_checkpoint()
        logger.info(f"[OnlineTrainer] Checkpoints loaded: GNN={gnn_loaded}, LoRA={lora_loaded}")
        
        # Запустить непрерывное обучение
        self.gnn_trainer.start()
        self.lora_trainer.start()
        
        logger.info("[OnlineTrainer] All trainers started (continuous mode)")
    
    def stop(self):
        """Остановить всех тренеров."""
        self.gnn_trainer.stop()
        self.lora_trainer.stop()
        logger.info("[OnlineTrainer] All trainers stopped")
    
    # === Resource Management - вызывается из FCP Pipeline ===
    
    def on_generation_start(self):
        """Модель начинает генерацию текста"""
        self.resource_manager.set_generating()
        logger.debug("[OnlineTrainer] Generation START - training paused")
    
    def on_generation_end(self):
        """Модель закончила генерацию текста"""
        self.resource_manager.set_idle()
        # Обновить активные веса
        self.hot_swap.update()
        logger.debug("[OnlineTrainer] Generation END - training resumed")
    
    # === Legacy методы для совместимости ===
    
    def on_query(self):
        """Вызывается после запроса (для совместимости).
        
        Непрерывное обучение уже работает в фоне,
        этот метод оставлен для совместимости с внешним API.
        """
        logger.debug("[OnlineTrainerManager] Query hook called - training runs continuously in background")
    
    def update_conversation_history(self, history):
        """Обновить историю для LoRA."""
        self.lora_trainer.update_history(history)
    
    def get_gnn_encoder(self):
        """Получить активный GNN энкодер."""
        return self.hot_swap.get_active_gnn()
    
    def get_lora_layers(self):
        """Получить активные LoRA слои."""
        return self.hot_swap.get_active_lora()
    
    def get_status(self) -> Dict[str, Any]:
        """Получить статус обучения."""
        return {
            "gnn": {
                "ready": self.gnn_trainer.is_ready(),
                "progress": self.gnn_trainer.get_progress(),
                "step_count": self.gnn_trainer._step_count,
                "avg_loss": self.gnn_trainer.get_avg_loss(),
                "completed": self.gnn_trainer.is_completed()
            },
            "lora": {
                "ready": self.lora_trainer.is_ready(),
                "progress": self.lora_trainer.get_progress(),
                "step_count": self.lora_trainer._step_count,
                "avg_loss": self.lora_trainer.get_avg_loss(),
                "completed": self.lora_trainer.is_completed()
            },
            "resource_state": self.resource_manager.get_state().value
        }


def integrate_online_trainer(brain, config: Optional[Dict] = None):
    """
    Интегрировать OnlineTrainer в CoreBrain.
    """
    manager = OnlineTrainerManager(config, brain=brain)
    manager.start()
    
    # Добавить методы в brain
    brain.online_trainer = manager
    brain.get_gnn_encoder = manager.get_gnn_encoder
    brain.get_lora_layers = manager.get_lora_layers
    brain.update_conversation_history = manager.update_conversation_history
    brain.get_training_status = manager.get_status
    
    # Опубликовать событие о запуске обучения
    event_bus = getattr(brain, '_new_event_bus', None)
    if event_bus:
        from eva_ai.core.event_bus import Event, EventTypes, EventPriority
        event = Event(
            event_type=EventTypes.LEARNING_STARTED,
            source="integrate_online_trainer",
            data={"trainer": "OnlineTrainerManager", "status": "running"},
            priority=EventPriority.NORMAL
        )
        event_bus.publish(event)
        logger.info("[OnlineTrainer] Published LEARNING_STARTED event")
    
    return manager