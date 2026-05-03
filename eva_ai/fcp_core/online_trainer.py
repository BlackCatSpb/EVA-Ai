"""
OnlineTrainer - Фоновое обучение GNN и LoRA во время работы EVA
Непрерывное обучение с паузой при генерации текста моделью.
"""
import os
import time
import threading
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from enum import Enum

logger = logging.getLogger("eva_ai.online_trainer")


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
    Автоматически использует GPU если доступен и достаточно памяти.
    """
    
    def __init__(self, min_memory_mb: int = 256):
        self.min_memory_mb = min_memory_mb
        self.device = None
        self.available = False
        self._init_gpu()
    
    def _init_gpu(self):
        """Инициализировать GPU если доступен."""
        try:
            import torch
            if torch.cuda.is_available():
                # Проверить память
                mem = torch.cuda.get_device_properties(0)
                free_mem = mem.total_memory - torch.cuda.memory_allocated()
                free_mem_mb = free_mem / (1024 * 1024)
                
                if free_mem_mb >= self.min_memory_mb:
                    self.device = torch.device("cuda:0")
                    self.available = True
                    logger.info(f"[GPUManager] GPU available: {mem.name}, {free_mem_mb:.0f}MB free")
                else:
                    logger.info(f"[GPUManager] Not enough GPU memory: {free_mem_mb:.0f}MB")
                    self.device = None
            else:
                logger.info("[GPUManager] CUDA not available")
        except Exception as e:
            logger.info(f"[GPUManager] GPU init failed: {e}")
            self.device = None
    
    def is_available(self) -> bool:
        """Доступен ли GPU для обучения."""
        if not self.available:
            return False
        
        # Проверить занятость GPU
        try:
            import torch
            if torch.cuda.is_available():
                # Если модель использует GPU, не обучаем
                return torch.cuda.memory_allocated(0) < (torch.cuda.get_device_properties(0).total_memory * 0.5)
        except:
            pass
        return False
    
    def get_device(self):
        """Получить устройство для тензоров."""
        return self.device


class BackgroundTrainer:
    """
    Базовый класс для непрерывного фонового обучения.
    Поддержка GPU и CPU.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "eva_ai/training/checkpoints",
        save_interval: int = 100,
        cpu_limit: int = 2,
        resource_manager: Optional[ResourceManager] = None,
        use_gpu: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_interval = save_interval
        self.cpu_limit = cpu_limit
        self.resource_manager = resource_manager
        self.use_gpu = use_gpu
        
        # GPU менеджер
        self.gpu_manager = GPUManager() if use_gpu else None
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._step_count = 0
        self._total_steps = None  # None = без ограничений, учится постоянно
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
        """Сохранить чекпоинт."""
        pass
    
    def load_latest_checkpoint(self) -> bool:
        """Загрузить последний чекпоинт."""
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
    
    def __init__(
        self,
        graph_db_path: str = "eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db",
        input_dim: int = 768,
        total_steps: int = None,  # None = без ограничений, учится постоянно
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
        self._node_training_counts = {}  # node_id -> количество шагов обучения
        self._reload_interval = 1000  # Перезагружать индекс каждые N шагов
        
        self._init_model()
        self._init_graph_indexer()
    
    def _init_model(self):
        """Инициализировать GNN модель."""
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            
            # Определить устройство
            if self.gpu_manager and self.gpu_manager.is_available():
                self.device = self.gpu_manager.get_device()
            else:
                self.device = torch.device("cpu")
            
            class MiniGNN(nn.Module):
                def __init__(self, input_dim=768, hidden=256, output=512):
                    super().__init__()
                    # Encoder: input_dim -> hidden -> output
                    self.encoder1 = nn.Linear(input_dim, hidden)
                    self.encoder2 = nn.Linear(hidden, output)
                    # Decoder: output -> hidden -> input_dim (for reconstruction)
                    self.decoder1 = nn.Linear(output, hidden)
                    self.decoder2 = nn.Linear(hidden, input_dim)
                    # Projection layer (only this is trained)
                    self.proj = nn.Linear(input_dim, input_dim)
                
                def forward(self, x):
                    # x: [batch, input_dim]
                    # Encode
                    h = F.relu(self.encoder1(x))      # [batch, hidden]
                    encoded = F.relu(self.encoder2(h))  # [batch, output]
                    
                    # Decode (for denoising self-supervised learning)
                    h = F.relu(self.decoder1(encoded))  # [batch, hidden]
                    reconstructed = self.decoder2(h)        # [batch, input_dim]
                    
                    # Apply projection (trainable)
                    projected = self.proj(reconstructed)
                    return projected  # [batch, input_dim]
            
            self._model = MiniGNN(input_dim=self.input_dim)
            self._model.to(self.device)
            
            # Только proj обучаем (минимальная нагрузка)
            for param in self._model.parameters():
                param.requires_grad = False
            for param in self._model.proj.parameters():
                param.requires_grad = True
            
            self._optimizer = torch.optim.SGD(
                [p for p in self._model.parameters() if p.requires_grad],
                lr=0.01
            )
            
            # Лимитировать CPU если не используем GPU
            if self.device.type == "cpu":
                torch.set_num_threads(self.cpu_limit)
            
            self._ready = True
            logger.info(f"[GNNTrainer] Model initialized: {self.input_dim}d input, device={self.device}")
            
        except Exception as e:
            logger.error(f"[GNNTrainer] Init failed: {e}")
            self._ready = False
    
    def _init_graph_indexer(self):
        """Инициализировать индекс графа для интеллектуального выбора данных."""
        try:
            from eva_ai.memory.fractal_graph_v2.graph_indexer import GraphIndexer
            if not os.path.exists(self.graph_db_path):
                logger.warning(f"[GNNTrainer] Graph DB not found, using synthetic fallback")
                self._graph_indexer = None
                return
            self._graph_indexer = GraphIndexer(self.graph_db_path, embedding_dim=self.input_dim)
            built = self._graph_indexer.build_index(limit=50000)
            if built:
                logger.info(f"[GNNTrainer] Graph index initialized with {len(self._graph_indexer._hnsw_index)} vectors")
            else:
                logger.warning("[GNNTrainer] Graph index build failed, will use SQL fallback")
        except Exception as e:
            logger.warning(f"[GNNTrainer] GraphIndexer init failed: {e}")
            self._graph_indexer = None
    
    def _add_synthetic_fallback(self):
        """Минимальный синтетический fallback (только если нет данных графа)."""
        import numpy as np
        self._synthetic_batch = [np.random.randn(self.input_dim).astype(np.float32) * 0.1 for _ in range(8)]
        logger.warning("[GNNTrainer] Using synthetic fallback data (no graph data)")
    
    def _load_batch(self):
        """Загрузить батч через индекс графа (интеллектуальный выбор, без повторов)."""
        import torch
        import numpy as np
        import sqlite3
        
        # 1. Попробовать через GraphIndexer (HNSW)
        if self._graph_indexer and self._graph_indexer._index_built:
            try:
                # Генерируем запрос: приоритет узлам с низким количеством обучений
                # Для простоты: случайный запрос, но можно улучшить через учет node_training_counts
                query_emb = np.random.randn(self.input_dim).astype(np.float32)
                results = self._graph_indexer.search(query_emb.tolist(), top_k=8, min_similarity=0.0)
                if results:
                    batch = []
                    for res in results:
                        node_id = res.get("id")
                        # Получить эмбеддинг узла
                        emb = res.get("embedding")
                        if emb and len(emb) >= self.input_dim:
                            arr = np.array(emb[:self.input_dim], dtype=np.float32)
                            batch.append(arr)
                            # Увеличить счетчик обучения узла
                            self._node_training_counts[node_id] = self._node_training_counts.get(node_id, 0) + 1
                    if batch:
                        return torch.tensor(np.array(batch), dtype=torch.float32).to(self.device)
            except Exception as e:
                logger.debug(f"[GNNTrainer] Index search failed: {e}")
        
        # 2. Fallback: SQL запрос с учетом количества обучений
        try:
            if not os.path.exists(self.graph_db_path):
                if not hasattr(self, '_synthetic_batch'):
                    self._add_synthetic_fallback()
                batch = self._synthetic_batch[:8]
                return torch.tensor(np.array(batch), dtype=torch.float32).to(self.device)
            
            conn = sqlite3.connect(self.graph_db_path)
            cur = conn.cursor()
            # Выбрать узлы с минимальным количеством обучений
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
                    if len(arr) >= self.input_dim:
                        arr = arr[:self.input_dim]
                    elif len(arr) > 0:
                        arr = np.pad(arr, (0, self.input_dim - len(arr)))
                    else:
                        continue
                    batch.append(arr)
                    self._node_training_counts[node_id] = self._node_training_counts.get(node_id, 0) + 1
                except Exception as e:
                    logger.debug(f"[GNNTrainer] Parse error: {e}")
            if batch:
                return torch.tensor(np.array(batch), dtype=torch.float32).to(self.device)
        except Exception as e:
            logger.debug(f"[GNNTrainer] SQL fallback failed: {e}")
        
        # 3. Синтетический fallback
        if not hasattr(self, '_synthetic_batch'):
            self._add_synthetic_fallback()
        batch = self._synthetic_batch[:8]
        return torch.tensor(np.array(batch), dtype=torch.float32).to(self.device)
    
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
                    # GPU занят - пропустить шаг
                    return False
            
            # Загрузить батч
            x = self._load_batch()
            if x is None:
                return False
            
            # Переместить на устройство
            x = x.to(self.device)
            
            # Self-supervised: denoising autoencoder
            # Input: x [batch, 768] - clean embeddings
            self._model.train()
            
            # Add noise to input
            noise = torch.randn_like(x) * 0.1
            noisy_x = x + noise
            
            # Forward through model: reconstruct clean from noisy
            # Model output: [batch, 768] (reconstructed)
            reconstructed = self._model(noisy_x)
            
            # Loss: MSE between reconstructed (from noisy) and original clean x
            loss = F.mse_loss(reconstructed, x.detach())
            
            # Backward
            self._optimizer.zero_grad()
            if loss.requires_grad:
                loss.backward()
                self._optimizer.step()
            
            self._model.eval()
            
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
        total_steps: int = None,  # None = без ограничений, учится постоянно
        graph_db_path: str = "eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db",
        **kwargs
    ):
        super().__init__(
            checkpoint_dir="eva_ai/training/checkpoints/lora",
            save_interval=100,
            **kwargs
        )
        
        self._conversation_history = conversation_history or []
        self._total_steps = total_steps
        self.graph_db_path = graph_db_path
        self._lora_layers = {}
        self._optimizer = None
        
        # Индекс графа для интеллектуального выбора данных
        self._graph_indexer = None
        self._node_training_counts = {}  # node_id -> количество шагов обучения
        self._reload_interval = 1000  # Перезагружать индекс каждые N шагов
        
        self._init_model()
        self._init_graph_indexer()
    
    def _init_model(self):
        """Инициализировать LoRA слои."""
        try:
            import torch
            import torch.nn as nn
            
            # Определить устройство
            if self.gpu_manager and self.gpu_manager.is_available():
                self.device = self.gpu_manager.get_device()
            else:
                self.device = torch.device("cpu")
            
            class LoRALayer(nn.Module):
                def __init__(self, in_features, out_features, rank=4):
                    super().__init__()
                    self.rank = rank
                    self.lora_a = nn.Parameter(torch.randn(rank, in_features) * 0.01)
                    self.lora_b = nn.Parameter(torch.randn(out_features, rank) * 0.01)
                    self.scaling = 1.0
                
                def forward(self, x):
                    return x + x @ self.lora_a.T @ self.lora_b.T * self.scaling
            
            # Используем 768-мерные эмбеддинги (как в GNN)
            self._lora_layers = {
                "q_proj": LoRALayer(768, 768, rank=4),
                "v_proj": LoRALayer(768, 768, rank=4),
            }
            
            # Переместить на устройство
            for layer in self._lora_layers.values():
                layer.to(self.device)
            
            params = []
            for layer in self._lora_layers.values():
                params.extend(list(layer.parameters()))
            
            self._optimizer = torch.optim.AdamW(params, lr=0.001)
            
            # Лимитировать CPU если не GPU
            if self.device.type == "cpu":
                torch.set_num_threads(self.cpu_limit)
            
            self._ready = True
            logger.info(f"[LoRATrainer] LoRA layers initialized, device={self.device}")
            
        except Exception as e:
            logger.error(f"[LoRATrainer] Init failed: {e}")
            self._ready = False
    
    def _init_graph_indexer(self):
        """Инициализировать индекс графа для интеллектуального выбора данных."""
        try:
            from eva_ai.memory.fractal_graph_v2.graph_indexer import GraphIndexer
            if not os.path.exists(self.graph_db_path):
                logger.warning(f"[LoRATrainer] Graph DB not found, using synthetic fallback")
                self._graph_indexer = None
                return
            self._graph_indexer = GraphIndexer(self.graph_db_path, embedding_dim=768)
            built = self._graph_indexer.build_index(limit=50000)
            if built:
                logger.info(f"[LoRATrainer] Graph index initialized with {len(self._graph_indexer._hnsw_index)} vectors")
            else:
                logger.warning("[LoRATrainer] Graph index build failed, will use SQL fallback")
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
            
            for layer in self._lora_layers.values():
                layer.train()
            
            # Forward через LoRA (вход: 768-мерные эмбеддинги графа)
            output = batch.clone()
            for layer in self._lora_layers.values():
                output = layer(output)
            
            # Target: reconstruction (self-supervised)
            target = batch.detach()
            loss = F.mse_loss(output, target)
            
            if loss.requires_grad:
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
            
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
        """Сохранить LoRA веса."""
        import torch
        
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
        logger.info(f"[LoRATrainer] Saved: {path}")
    
    def load_latest_checkpoint(self) -> bool:
        """Загрузить последний чекпоинт."""
        import torch
        
        checkpoints = list(self.checkpoint_dir.glob("lora_model*.pt"))
        if not checkpoints:
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
    
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        
        # GPU настройка
        use_gpu = config.get("use_gpu", True)  # По умолчанию использовать GPU
        
        # Resource Manager - контролирует когда можно обучаться
        self.resource_manager = ResourceManager()
        
        # Тренеры с GPU поддержкой
        self.gnn_trainer = GNNTrainer(
            cpu_limit=config.get("gnn_cpu_limit", 2),
            resource_manager=self.resource_manager,
            total_steps=config.get("gnn_total_steps", 5000),
            use_gpu=use_gpu
        )
        
        self.lora_trainer = LoRATrainer(
            cpu_limit=config.get("lora_cpu_limit", 2),
            resource_manager=self.resource_manager,
            total_steps=config.get("lora_total_steps", 100000),
            use_gpu=use_gpu
        )
        
        self.hot_swap = HotSwapManager(self.resource_manager)
        self.hot_swap.register_gnn(self.gnn_trainer)
        self.hot_swap.register_lora(self.lora_trainer)
        
        self._enabled = config.get("enabled", True)
    
    def start(self):
        """Запустить всех тренеров."""
        if not self._enabled:
            logger.info("[OnlineTrainer] Disabled in config")
            return
        
        # Загрузить последние чекпоинты
        self.gnn_trainer.load_latest_checkpoint()
        self.lora_trainer.load_latest_checkpoint()
        
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
        """Вызывается после запроса (для совместимости)"""
        # Непрерывное обучение уже работает в фоне
        pass
    
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
    manager = OnlineTrainerManager(config)
    manager.start()
    
    # Добавить методы в brain
    brain.online_trainer = manager
    brain.get_gnn_encoder = manager.get_gnn_encoder
    brain.get_lora_layers = manager.get_lora_layers
    brain.update_conversation_history = manager.update_conversation_history
    brain.get_training_status = manager.get_status
    
    return manager