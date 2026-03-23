
"""
Модуль обучения графа памяти для CogniFlex - тренировка на предзагруженных моделях
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from datetime import datetime
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger("cogniflex.learning.memory_graph_trainer")

@dataclass
class TrainingConfig:
    """Конфигурация для обучения графа памяти."""
    learning_rate: float = 0.001
    batch_size: int = 8
    epochs: int = 5
    embedding_dim: int = 128
    hidden_dim: int = 64
    dropout_rate: float = 0.1
    weight_decay: float = 1e-5
    patience: int = 3
    min_delta: float = 1e-4

class MemoryGraphNetwork(nn.Module):
    """Нейронная сеть для обучения графа памяти."""
    
    def __init__(self, config: TrainingConfig, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.config = config
        
        # Энкодер для узлов графа
        self.node_encoder = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # Энкодер для связей
        self.edge_encoder = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU()
        )
        
        # Предсказатель связей
        self.link_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim + config.hidden_dim // 2, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Классификатор типов узлов
        self.node_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, 5)  # 5 типов узлов
        )

        # Переносим всю модель на целевое устройство и dtype единообразно
        try:
            self.to(device=device, dtype=dtype)
        except Exception:
            # В редких случаях to(dtype=half) может не поддерживаться CPU — fallback: только устройство
            try:
                self.to(device=device)
            except Exception:
                pass
        
    def forward(self, node_features, edge_features=None, predict_links=True, classify_nodes=True):
        """Прямой проход сети."""
        # Кодируем узлы
        node_encoded = self.node_encoder(node_features)
        
        results = {"node_encoded": node_encoded}
        
        if classify_nodes:
            # Классифицируем типы узлов
            node_types = self.node_classifier(node_encoded)
            results["node_types"] = node_types
        
        if predict_links and edge_features is not None:
            # Кодируем связи
            edge_encoded = self.edge_encoder(edge_features)
            
            # Объединяем признаки узлов и связей
            combined_features = torch.cat([node_encoded, edge_encoded], dim=-1)
            
            # Предсказываем вероятность связи
            link_probs = self.link_predictor(combined_features)
            results["link_probs"] = link_probs
        
        return results

class MemoryGraphTrainer:
    """Тренер для обучения графа памяти на предзагруженных моделях."""
    
    def __init__(self, brain=None, config: Optional[TrainingConfig] = None):
        """
        Инициализирует тренер графа памяти.
        
        Args:
            brain: Ссылка на ядро CogniFlex
            config: Конфигурация обучения
        """
        self.brain = brain
        self.config = config or TrainingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Согласованный dtype для вычислений
        self.compute_dtype = torch.float16 if (self.device.type == "cuda") else torch.float32
        
        # Инициализируем модель сразу на целевом устройстве и с нужным dtype (без промежуточного meta/CPU)
        self.model = MemoryGraphNetwork(self.config, device=self.device, dtype=self.compute_dtype)
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Критерии потерь
        self.link_criterion = nn.BCELoss()
        self.node_criterion = nn.CrossEntropyLoss()
        
        # Статистика обучения
        self.training_stats = {
            "epoch": 0,
            "total_loss": 0.0,
            "link_loss": 0.0,
            "node_loss": 0.0,
            "accuracy": 0.0,
            "best_loss": float('inf'),
            "patience_counter": 0,
            "training_history": []
        }
        
        # Состояние обучения
        self.is_training = False
        self.training_thread = None
        
        logger.info(f"MemoryGraphTrainer инициализирован на устройстве: {self.device}")
    
    def prepare_training_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Подготавливает данные для обучения из предзагруженных моделей.
        
        Returns:
            Tuple: (node_features, edge_features, link_labels, node_labels)
        """
        try:
            # Получаем граф памяти из системы
            memory_graph = None
            if self.brain and hasattr(self.brain, 'knowledge_graph'):
                memory_graph = self.brain.knowledge_graph
            elif self.brain and hasattr(self.brain, 'components'):
                memory_graph = self.brain.components.get('knowledge_graph')
            
            if not memory_graph:
                logger.warning("Граф памяти недоступен, создаем синтетические данные")
                return self._create_synthetic_data()
            
            # Извлекаем узлы и связи
            nodes = memory_graph.get_all_nodes() if hasattr(memory_graph, 'get_all_nodes') else []
            edges = memory_graph.get_all_edges() if hasattr(memory_graph, 'get_all_edges') else []
            
            if not nodes:
                logger.warning("Узлы графа памяти не найдены, создаем синтетические данные")
                return self._create_synthetic_data()
            
            # Подготавливаем признаки узлов
            node_features = []
            node_labels = []
            
            for node in nodes:
                # Получаем эмбеддинг узла
                embedding = self._get_node_embedding(node)
                if embedding is not None:
                    node_features.append(embedding)
                    # Определяем тип узла (0-4)
                    node_type = self._classify_node_type(node)
                    node_labels.append(node_type)
            
            if not node_features:
                logger.warning("Не удалось извлечь признаки узлов, создаем синтетические данные")
                return self._create_synthetic_data()
            
            # Подготавливаем признаки связей
            edge_features = []
            link_labels = []
            
            for edge in edges:
                # Получаем признаки связи
                edge_feature = self._get_edge_features(edge, nodes)
                if edge_feature is not None:
                    edge_features.append(edge_feature)
                    link_labels.append(1.0)  # Существующие связи помечаем как положительные
            
            # Добавляем отрицательные примеры (несуществующие связи)
            negative_edges = self._generate_negative_edges(nodes, edges)
            for neg_edge in negative_edges[:len(edge_features)]:  # Балансируем классы
                edge_feature = self._get_edge_features(neg_edge, nodes)
                if edge_feature is not None:
                    edge_features.append(edge_feature)
                    link_labels.append(0.0)  # Несуществующие связи помечаем как отрицательные
            
            # Конвертируем в тензоры
            node_features = torch.tensor(np.array(node_features), dtype=torch.float32)
            edge_features = torch.tensor(np.array(edge_features), dtype=torch.float32) if edge_features else None
            link_labels = torch.tensor(link_labels, dtype=torch.float32) if link_labels else None
            node_labels = torch.tensor(node_labels, dtype=torch.long)
            
            logger.info(f"Подготовлены данные: {len(node_features)} узлов, {len(edge_features) if edge_features is not None else 0} связей")
            
            return node_features, edge_features, link_labels, node_labels
            
        except Exception as e:
            logger.error(f"Ошибка подготовки данных для обучения: {e}")
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Создает синтетические данные для обучения."""
        logger.info("Создание синтетических данных для обучения")
        
        # Создаем случайные признаки узлов
        num_nodes = 100
        node_features = torch.randn(
            num_nodes, self.config.embedding_dim, device=self.device, dtype=self.compute_dtype
        )
        node_labels = torch.randint(0, 5, (num_nodes,), device=self.device, dtype=torch.long)
        
        # Создаем случайные признаки связей
        num_edges = 200
        edge_features = torch.randn(
            num_edges, self.config.embedding_dim * 2, device=self.device, dtype=self.compute_dtype
        )
        link_labels = torch.randint(0, 2, (num_edges,), device=self.device, dtype=torch.int64).to(self.compute_dtype)
        
        return node_features, edge_features, link_labels, node_labels
    
    def _get_node_embedding(self, node) -> Optional[np.ndarray]:
        """Получает эмбеддинг узла."""
        try:
            # Пытаемся получить эмбеддинг из различных источников
            if hasattr(node, 'embedding') and node.embedding is not None:
                return self._ensure_dim(np.array(node.embedding, dtype=np.float32))
            
            if hasattr(node, 'vector') and node.vector is not None:
                return self._ensure_dim(np.array(node.vector, dtype=np.float32))
            
            # Если эмбеддинга нет, создаем его из текста
            if hasattr(node, 'content') or hasattr(node, 'text'):
                text = getattr(node, 'content', None) or getattr(node, 'text', '')
                return self._text_to_embedding(text)
            
            # Если ничего нет, возвращаем случайный вектор
            return np.random.randn(self.config.embedding_dim)
            
        except Exception as e:
            logger.error(f"Ошибка получения эмбеддинга узла: {e}")
            return None
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Конвертирует текст в эмбеддинг."""
        try:
            # Пытаемся использовать текстовый процессор из системы
            if self.brain and hasattr(self.brain, 'ml_unit') and self.brain.ml_unit:
                text_processor = getattr(self.brain.ml_unit, 'text_processor', None)
                if text_processor and hasattr(text_processor, 'get_embeddings'):
                    emb = text_processor.get_embeddings(text)
                    # ожидаем np.ndarray; приводим к (D,)
                    if isinstance(emb, np.ndarray):
                        if emb.ndim == 2 and emb.shape[0] >= 1:
                            emb = emb[0]
                        emb = emb.astype(np.float32, copy=False)
                        return self._ensure_dim(emb)
            
            # Если не получилось, создаем простой хэш-эмбеддинг
            return self._hash_to_embedding(text)
            
        except Exception as e:
            logger.error(f"Ошибка конвертации текста в эмбеддинг: {e}")
            # Возвращаем стабильный случайный вектор нужной размерности
            return np.random.randn(self.config.embedding_dim).astype(np.float32)
    
    def _hash_to_embedding(self, text: str) -> np.ndarray:
        """Создает эмбеддинг на основе хэша текста."""
        import hashlib
        
        # Создаем хэш текста
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Конвертируем в числовой вектор
        embedding = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)
        
        # Нормализуем и расширяем до нужной размерности
        embedding = embedding / 255.0  # Нормализация к [0, 1]
        
        # Повторяем и обрезаем до нужной размерности
        while len(embedding) < self.config.embedding_dim:
            embedding = np.concatenate([embedding, embedding])
        
        return embedding[:self.config.embedding_dim]

    def _ensure_dim(self, vec: np.ndarray) -> np.ndarray:
        """Гарантирует размерность (D,) = config.embedding_dim: обрезает или дополняет нулями."""
        try:
            if vec.ndim > 1:
                vec = vec.reshape(-1)
            D = int(self.config.embedding_dim)
            if vec.size == D:
                return vec
            if vec.size > D:
                return vec[:D]
            # pad with zeros
            out = np.zeros(D, dtype=vec.dtype)
            out[:vec.size] = vec
            return out
        except Exception:
            return np.zeros(int(self.config.embedding_dim), dtype=np.float32)

    def _resolve_edge_nodes(self, edge, nodes: List[Any]) -> Optional[Tuple[Any, Any]]:
        """Пытается извлечь пару узлов (src, dst) из произвольного представления ребра.
        Поддерживаются форматы:
        - кортеж индексов (i, j)
        - кортеж узлов (node_i, node_j)
        - объект с атрибутами/ключами source/target
        """
        try:
            # Кортеж или список
            if isinstance(edge, (tuple, list)) and len(edge) >= 2:
                a, b = edge[0], edge[1]
                # индексы
                if isinstance(a, int) and isinstance(b, int) and 0 <= a < len(nodes) and 0 <= b < len(nodes):
                    return nodes[a], nodes[b]
                # непосредственно узлы
                return a, b

            # Объект с атрибутами
            if hasattr(edge, 'source') and hasattr(edge, 'target'):
                src = getattr(edge, 'source')
                dst = getattr(edge, 'target')
                # индексы -> разрешаем в узлы
                if isinstance(src, int) and isinstance(dst, int):
                    if 0 <= src < len(nodes) and 0 <= dst < len(nodes):
                        return nodes[src], nodes[dst]
                return src, dst

            # Словарное представление
            if isinstance(edge, dict):
                src = edge.get('source') or edge.get('src')
                dst = edge.get('target') or edge.get('dst')
                if src is not None and dst is not None:
                    if isinstance(src, int) and isinstance(dst, int):
                        if 0 <= src < len(nodes) and 0 <= dst < len(nodes):
                            return nodes[src], nodes[dst]
                    return src, dst
        except Exception:
            pass
        return None

    def _get_edge_features(self, edge, nodes: List[Any]) -> Optional[np.ndarray]:
        """Формирует признаки ребра как конкатенацию эмбеддингов его концов."""
        try:
            resolved = self._resolve_edge_nodes(edge, nodes)
            if not resolved:
                return None
            n1, n2 = resolved
            emb1 = self._get_node_embedding(n1)
            emb2 = self._get_node_embedding(n2)
            if emb1 is None or emb2 is None:
                return None
            return np.concatenate([emb1, emb2], axis=-1)
        except Exception as e:
            logger.error(f"Ошибка формирования признаков ребра: {e}")
            return None

    def _generate_negative_edges(self, nodes: List[Any], edges: List[Any], max_samples: Optional[int] = None) -> List[Tuple[int, int]]:
        """Генерирует отрицательные (несуществующие) ребра как пары индексов узлов."""
        try:
            n = len(nodes)
            if n < 2:
                return []
            # Собираем множество существующих пар индексов
            existing: set[Tuple[int, int]] = set()

            def node_index(node_obj) -> Optional[int]:
                try:
                    return nodes.index(node_obj)
                except ValueError:
                    return None

            for e in edges or []:
                pair = None
                if isinstance(e, (tuple, list)) and len(e) >= 2:
                    a, b = e[0], e[1]
                    if isinstance(a, int) and isinstance(b, int):
                        pair = (a, b)
                    else:
                        ia = node_index(a)
                        ib = node_index(b)
                        if ia is not None and ib is not None:
                            pair = (ia, ib)
                elif hasattr(e, 'source') and hasattr(e, 'target'):
                    a = getattr(e, 'source')
                    b = getattr(e, 'target')
                    if isinstance(a, int) and isinstance(b, int):
                        pair = (a, b)
                    else:
                        ia = node_index(a)
                        ib = node_index(b)
                        if ia is not None and ib is not None:
                            pair = (ia, ib)
                elif isinstance(e, dict):
                    a = e.get('source') or e.get('src')
                    b = e.get('target') or e.get('dst')
                    if isinstance(a, int) and isinstance(b, int):
                        pair = (a, b)
                    else:
                        ia = node_index(a)
                        ib = node_index(b)
                        if ia is not None and ib is not None:
                            pair = (ia, ib)
                if pair and 0 <= pair[0] < n and 0 <= pair[1] < n:
                    existing.add(pair)

            import random
            random_pairs: List[Tuple[int, int]] = []
            target = max_samples if max_samples is not None else max(1, len(edges) or n)
            attempts = 0
            max_attempts = target * 10
            while len(random_pairs) < target and attempts < max_attempts:
                i = random.randrange(n)
                j = random.randrange(n)
                if i == j:
                    attempts += 1
                    continue
                pair = (i, j)
                if pair in existing:
                    attempts += 1
                    continue
                existing.add(pair)
                random_pairs.append(pair)
            return random_pairs
        except Exception as e:
            logger.error(f"Ошибка генерации отрицательных ребер: {e}")
            return []
    
    def _classify_node_type(self, node) -> int:
        """Классифицирует тип узла (0-4)."""
        try:
            # Определяем тип узла на основе его свойств
            if hasattr(node, 'type'):
                node_type = getattr(node, 'type', '').lower()
                if 'concept' in node_type:
                    return 0
                elif 'entity' in node_type:
                    return 1
                elif 'relation' in node_type:
                    return 2
                elif 'attribute' in node_type:
                    return 3
                else:
                    return 4
            
            # Если типа нет, определяем по содержимому
            content = getattr(node, 'content', '') or getattr(node, 'text', '')
            if len(content) > 100:
                return 0  # Концепт
            elif len(content) > 50:
                return 1  # Сущность
            else:
                return 2  # Отношение
                
        except Exception:
            return 4  # Неизвестный тип
    
    def start_learning_process(self):
        """Запускает процесс обучения (для совместимости с GUI)."""
        return self.train_async()
    
    def pause_learning_process(self):
        """Приостанавливает процесс обучения (для совместимости с GUI)."""
        return self.stop_training()
    
    def train_async(self, epochs: Optional[int] = None) -> bool:
        """
        Запускает асинхронное обучение модели.
        
        Args:
            epochs: Количество эпох обучения
            
        Returns:
            bool: Успешно ли запущено обучение
        """
        if self.is_training:
            logger.warning("Обучение уже запущено")
            return False
        
        try:
            self.training_thread = threading.Thread(
                target=self._train_worker,
                args=(epochs,),
                daemon=True
            )
            self.training_thread.start()
            logger.info("Асинхронное обучение запущено")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка запуска асинхронного обучения: {e}")
            return False
    
    def _train_worker(self, epochs: Optional[int] = None):
        """Рабочий метод для обучения в отдельном потоке."""
        try:
            self.is_training = True
            epochs = epochs or self.config.epochs
            
            logger.info(f"Начинаем обучение графа памяти на {epochs} эпох")
            
            # Подготавливаем данные
            node_features, edge_features, link_labels, node_labels = self.prepare_training_data()
            
            if node_features is None or len(node_features) == 0:
                logger.error("Не удалось подготовить данные для обучения")
                return
            
            # Сбрасываем счетчики
            self.training_stats["patience_counter"] = 0
            self.training_stats["best_loss"] = float('inf')
            
            # Обучаем модель
            for epoch in range(epochs):
                if not self.is_training:  # Проверяем, не остановили ли обучение
                    logger.info("Обучение остановлено пользователем")
                    break
                
                # Обучаем эпоху
                epoch_stats = self._train_epoch(node_features, edge_features, link_labels, node_labels)
                
                # Обновляем статистику
                self.training_stats["epoch"] = epoch + 1
                self.training_stats.update(epoch_stats)
                
                # Сохраняем историю
                epoch_record = {
                    "epoch": epoch + 1,
                    "timestamp": datetime.now().isoformat(),
                    **epoch_stats
                }
                self.training_stats["training_history"].append(epoch_record)
                
                # Проверяем улучшение
                current_loss = epoch_stats["total_loss"]
                if current_loss < self.training_stats["best_loss"] - self.config.min_delta:
                    self.training_stats["best_loss"] = current_loss
                    self.training_stats["patience_counter"] = 0
                    # Сохраняем лучшую модель
                    self._save_best_model()
                else:
                    self.training_stats["patience_counter"] += 1
                
                # Логируем прогресс
                logger.info(
                    f"Эпоха {epoch + 1}/{epochs}: "
                    f"Loss={current_loss:.4f}, "
                    f"Accuracy={epoch_stats['accuracy']:.4f}, "
                    f"Patience={self.training_stats['patience_counter']}"
                )
                
                # Проверяем раннюю остановку
                if self.training_stats["patience_counter"] >= self.config.patience:
                    logger.info(f"Ранняя остановка на эпохе {epoch + 1}")
                    break
            
            logger.info("Обучение графа памяти завершено")
            
        except Exception as e:
            logger.error(f"Ошибка во время обучения: {e}")
        finally:
            self.is_training = False
    
    def _train_epoch(self, node_features, edge_features, link_labels, node_labels) -> Dict[str, float]:
        """Обучает модель на одной эпохе."""
        self.model.train()
        model_dtype = next(self.model.parameters()).dtype
        
        total_loss = 0.0
        link_loss_total = 0.0
        node_loss_total = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Создаем батчи
        batch_size = self.config.batch_size
        num_batches = max(1, len(node_features) // batch_size)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(node_features))
            
            # Получаем батч данных
            batch_nodes = node_features[start_idx:end_idx].to(self.device, dtype=model_dtype, non_blocking=True)
            batch_node_labels = node_labels[start_idx:end_idx].to(self.device, dtype=torch.long, non_blocking=True)
            
            batch_edges = None
            batch_link_labels = None
            if edge_features is not None and link_labels is not None:
                if end_idx <= len(edge_features):
                    batch_edges = edge_features[start_idx:end_idx].to(self.device, dtype=model_dtype, non_blocking=True)
                    # Лейблы линков должны совпадать по dtype с выходами сигмоида
                    batch_link_labels = link_labels[start_idx:end_idx].to(self.device, dtype=model_dtype, non_blocking=True)
            
            # Обнуляем градиенты
            self.optimizer.zero_grad()
            
            # Прямой проход
            outputs = self.model(
                batch_nodes, 
                batch_edges, 
                predict_links=(batch_edges is not None),
                classify_nodes=True
            )
            
            # Вычисляем потери
            node_loss = self.node_criterion(outputs["node_types"], batch_node_labels)
            total_batch_loss = node_loss
            
            link_loss = torch.zeros((), device=self.device, dtype=model_dtype)
            if batch_edges is not None and "link_probs" in outputs:
                link_loss = self.link_criterion(outputs["link_probs"].squeeze(), batch_link_labels)
                total_batch_loss += link_loss
            
            # Обратное распространение
            total_batch_loss.backward()
            self.optimizer.step()
            
            # Обновляем статистику
            total_loss += total_batch_loss.item()
            node_loss_total += node_loss.item()
            link_loss_total += link_loss.item()
            
            # Вычисляем точность для классификации узлов
            _, predicted = torch.max(outputs["node_types"], 1)
            correct_predictions += (predicted == batch_node_labels).sum().item()
            total_predictions += batch_node_labels.size(0)
        
        # Вычисляем средние метрики
        avg_total_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_node_loss = node_loss_total / num_batches if num_batches > 0 else 0.0
        avg_link_loss = link_loss_total / num_batches if num_batches > 0 else 0.0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            "total_loss": avg_total_loss,
            "node_loss": avg_node_loss,
            "link_loss": avg_link_loss,
            "accuracy": accuracy
        }
    
    def stop_training(self) -> bool:
        """
        Останавливает обучение.
        
        Returns:
            bool: Успешно ли остановлено обучение
        """
        if not self.is_training:
            logger.warning("Обучение не запущено")
            return False
        
        try:
            self.is_training = False
            
            if self.training_thread and self.training_thread.is_alive():
                self.training_thread.join(timeout=5.0)
            
            logger.info("Обучение остановлено")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка остановки обучения: {e}")
            return False
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Возвращает статистику обучения."""
        return self.training_stats.copy()
    
    def _save_best_model(self):
        """Сохраняет лучшую модель."""
        try:
            if not hasattr(self, 'best_model_path'):
                cache_dir = getattr(self.brain, 'cache_dir', './cache') if self.brain else './cache'
                os.makedirs(cache_dir, exist_ok=True)
                self.best_model_path = os.path.join(cache_dir, 'best_memory_graph_model.pth')
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_stats': self.training_stats,
                'config': self.config
            }, self.best_model_path)
            
            logger.debug(f"Лучшая модель сохранена в {self.best_model_path}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения лучшей модели: {e}")

"""
Модуль обучения графа памяти для CogniFlex - тренировка на предзагруженных моделях
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from datetime import datetime
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger("cogniflex.learning.memory_graph_trainer")

@dataclass
class TrainingConfig:
    """Конфигурация для обучения графа памяти."""
    learning_rate: float = 0.001
    batch_size: int = 8
    epochs: int = 10
    embedding_dim: int = 128
    hidden_dim: int = 128
    dropout_rate: float = 0.1
    weight_decay: float = 1e-5
    patience: int = 5
    min_delta: float = 1e-4

class MemoryGraphNetwork(nn.Module):
    """Нейронная сеть для обучения графа памяти."""
    
    def __init__(self, config: TrainingConfig, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.config = config
        
        # Энкодер для узлов графа
        self.node_encoder = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # Энкодер для связей
        self.edge_encoder = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU()
        )
        
        # Предсказатель связей
        self.link_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim + config.hidden_dim // 2, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Классификатор типов узлов
        self.node_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, 5)  # 5 типов узлов
        )

        # Переносим всю модель на целевое устройство и dtype единообразно
        try:
            self.to(device=device, dtype=dtype)
        except Exception:
            # В редких случаях to(dtype=half) может не поддерживаться CPU — fallback: только устройство
            try:
                self.to(device=device)
            except Exception:
                pass
        
    def forward(self, node_features, edge_features=None, predict_links=True, classify_nodes=True):
        """Прямой проход сети."""
        # Кодируем узлы
        node_encoded = self.node_encoder(node_features)
        
        results = {"node_encoded": node_encoded}
        
        if classify_nodes:
            # Классифицируем типы узлов
            node_types = self.node_classifier(node_encoded)
            results["node_types"] = node_types
        
        if predict_links and edge_features is not None:
            # Кодируем связи
            edge_encoded = self.edge_encoder(edge_features)
            
            # Объединяем признаки узлов и связей
            combined_features = torch.cat([node_encoded, edge_encoded], dim=-1)
            
            # Предсказываем вероятность связи
            link_probs = self.link_predictor(combined_features)
            results["link_probs"] = link_probs
        
        return results

class MemoryGraphTrainer:
    """Тренер для обучения графа памяти на предзагруженных моделях."""
    
    def __init__(self, brain=None, config: Optional[TrainingConfig] = None):
        """
        Инициализирует тренер графа памяти.
        
        Args:
            brain: Ссылка на ядро CogniFlex
            config: Конфигурация обучения
        """
        # Проверяем конфигурацию на отключение обучения
        training_disabled = False
        if brain and hasattr(brain, 'config'):
            learning_config = brain.config.get('learning', {})
            training_disabled = learning_config.get('enable_training', True) == False
            training_disabled = training_disabled or learning_config.get('training_disabled', False)
        
        # Проверяем системную конфигурацию
        if brain and hasattr(brain, 'config'):
            system_config = brain.config.get('system', {})
            training_disabled = training_disabled or system_config.get('disable_learning_threads', False)
            training_disabled = training_disabled or system_config.get('disable_background_training', False)
        
        if training_disabled:
            logger.info("🚀 Обучение графа памяти отключено через конфигурацию")
            self.disabled = True
            self.brain = brain
            self.config = config or TrainingConfig()
            self.device = torch.device("cpu")  # Принудительно CPU
            self.compute_dtype = torch.float32
            self.model = None
            self.optimizer = None
            self.training_stats = {"disabled": True}
            return
        
        self.brain = brain
        self.config = config or TrainingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Согласованный dtype для вычислений
        self.compute_dtype = torch.float16 if (self.device.type == "cuda") else torch.float32
        
        # Инициализируем модель сразу на целевом устройстве и с нужным dtype (без промежуточного meta/CPU)
        self.model = MemoryGraphNetwork(self.config, device=self.device, dtype=self.compute_dtype)
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Критерии потерь
        self.link_criterion = nn.BCELoss()
        self.node_criterion = nn.CrossEntropyLoss()
        
        # Статистика обучения
        self.training_stats = {
            "epoch": 0,
            "total_loss": 0.0,
            "link_loss": 0.0,
            "node_loss": 0.0,
            "accuracy": 0.0,
            "best_loss": float('inf'),
            "patience_counter": 0,
            "training_history": []
        }
        
        self.disabled = False
        
        # Состояние обучения
        self.is_training = False
        self.training_thread = None
        
        logger.info(f"MemoryGraphTrainer инициализирован на устройстве: {self.device}")
    
    def prepare_training_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Подготавливает данные для обучения из предзагруженных моделей.
        
        Returns:
            Tuple: (node_features, edge_features, link_labels, node_labels)
        """
        try:
            # Получаем граф памяти из системы
            memory_graph = None
            if self.brain and hasattr(self.brain, 'knowledge_graph'):
                memory_graph = self.brain.knowledge_graph
            elif self.brain and hasattr(self.brain, 'components'):
                memory_graph = self.brain.components.get('knowledge_graph')
            
            if not memory_graph:
                logger.warning("Граф памяти недоступен, создаем синтетические данные")
                return self._create_synthetic_data()
            
            # Извлекаем узлы и связи
            nodes = memory_graph.get_all_nodes() if hasattr(memory_graph, 'get_all_nodes') else []
            edges = memory_graph.get_all_edges() if hasattr(memory_graph, 'get_all_edges') else []
            
            if not nodes:
                logger.warning("Узлы графа памяти не найдены, создаем синтетические данные")
                return self._create_synthetic_data()
            
            # Подготавливаем признаки узлов
            node_features = []
            node_labels = []
            
            for node in nodes:
                # Получаем эмбеддинг узла
                embedding = self._get_node_embedding(node)
                if embedding is not None:
                    node_features.append(embedding)
                    # Определяем тип узла (0-4)
                    node_type = self._classify_node_type(node)
                    node_labels.append(node_type)
            
            if not node_features:
                logger.warning("Не удалось извлечь признаки узлов, создаем синтетические данные")
                return self._create_synthetic_data()
            
            # Подготавливаем признаки связей
            edge_features = []
            link_labels = []
            
            for edge in edges:
                # Получаем признаки связи
                edge_feature = self._get_edge_features(edge, nodes)
                if edge_feature is not None:
                    edge_features.append(edge_feature)
                    link_labels.append(1.0)  # Существующие связи помечаем как положительные
            
            # Добавляем отрицательные примеры (несуществующие связи)
            negative_edges = self._generate_negative_edges(nodes, edges)
            for neg_edge in negative_edges[:len(edge_features)]:  # Балансируем классы
                edge_feature = self._get_edge_features(neg_edge, nodes)
                if edge_feature is not None:
                    edge_features.append(edge_feature)
                    link_labels.append(0.0)  # Несуществующие связи помечаем как отрицательные
            
            # Конвертируем в тензоры
            node_features = torch.tensor(np.array(node_features), dtype=torch.float32)
            edge_features = torch.tensor(np.array(edge_features), dtype=torch.float32) if edge_features else None
            link_labels = torch.tensor(link_labels, dtype=torch.float32) if link_labels else None
            node_labels = torch.tensor(node_labels, dtype=torch.long)
            
            logger.info(f"Подготовлены данные: {len(node_features)} узлов, {len(edge_features) if edge_features is not None else 0} связей")
            
            return node_features, edge_features, link_labels, node_labels
            
        except Exception as e:
            logger.error(f"Ошибка подготовки данных для обучения: {e}")
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Создает синтетические данные для обучения."""
        logger.info("Создание синтетических данных для обучения")
        
        # Создаем случайные признаки узлов
        num_nodes = 100
        node_features = torch.randn(
            num_nodes, self.config.embedding_dim, device=self.device, dtype=self.compute_dtype
        )
        node_labels = torch.randint(0, 5, (num_nodes,), device=self.device, dtype=torch.long)
        
        # Создаем случайные признаки связей
        num_edges = 200
        edge_features = torch.randn(
            num_edges, self.config.embedding_dim * 2, device=self.device, dtype=self.compute_dtype
        )
        link_labels = torch.randint(0, 2, (num_edges,), device=self.device, dtype=torch.int64).to(self.compute_dtype)
        
        return node_features, edge_features, link_labels, node_labels
    
    def _get_node_embedding(self, node) -> Optional[np.ndarray]:
        """Получает эмбеддинг узла."""
        try:
            # Пытаемся получить эмбеддинг из различных источников
            if hasattr(node, 'embedding') and node.embedding is not None:
                return self._ensure_dim(np.array(node.embedding, dtype=np.float32))
            
            if hasattr(node, 'vector') and node.vector is not None:
                return self._ensure_dim(np.array(node.vector, dtype=np.float32))
            
            # Если эмбеддинга нет, создаем его из текста
            if hasattr(node, 'content') or hasattr(node, 'text'):
                text = getattr(node, 'content', None) or getattr(node, 'text', '')
                return self._text_to_embedding(text)
            
            # Если ничего нет, возвращаем случайный вектор
            return np.random.randn(self.config.embedding_dim)
            
        except Exception as e:
            logger.error(f"Ошибка получения эмбеддинга узла: {e}")
            return None
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Конвертирует текст в эмбеддинг."""
        try:
            # Пытаемся использовать текстовый процессор из системы
            if self.brain and hasattr(self.brain, 'ml_unit') and self.brain.ml_unit:
                text_processor = getattr(self.brain.ml_unit, 'text_processor', None)
                if text_processor and hasattr(text_processor, 'get_embeddings'):
                    emb = text_processor.get_embeddings(text)
                    # ожидаем np.ndarray; приводим к (D,)
                    if isinstance(emb, np.ndarray):
                        if emb.ndim == 2 and emb.shape[0] >= 1:
                            emb = emb[0]
                        emb = emb.astype(np.float32, copy=False)
                        return self._ensure_dim(emb)
            
            # Если не получилось, создаем простой хэш-эмбеддинг
            return self._hash_to_embedding(text)
            
        except Exception as e:
            logger.error(f"Ошибка конвертации текста в эмбеддинг: {e}")
            # Возвращаем стабильный случайный вектор нужной размерности
            return np.random.randn(self.config.embedding_dim).astype(np.float32)
    
    def _hash_to_embedding(self, text: str) -> np.ndarray:
        """Создает эмбеддинг на основе хэша текста."""
        import hashlib
        
        # Создаем хэш текста
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Конвертируем в числовой вектор
        embedding = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)
        
        # Нормализуем и расширяем до нужной размерности
        embedding = embedding / 255.0  # Нормализация к [0, 1]
        
        # Повторяем и обрезаем до нужной размерности
        while len(embedding) < self.config.embedding_dim:
            embedding = np.concatenate([embedding, embedding])
        
        return embedding[:self.config.embedding_dim]

    def _ensure_dim(self, vec: np.ndarray) -> np.ndarray:
        """Гарантирует размерность (D,) = config.embedding_dim: обрезает или дополняет нулями."""
        try:
            if vec.ndim > 1:
                vec = vec.reshape(-1)
            D = int(self.config.embedding_dim)
            if vec.size == D:
                return vec
            if vec.size > D:
                return vec[:D]
            # pad with zeros
            out = np.zeros(D, dtype=vec.dtype)
            out[:vec.size] = vec
            return out
        except Exception:
            return np.zeros(int(self.config.embedding_dim), dtype=np.float32)

    def _resolve_edge_nodes(self, edge, nodes: List[Any]) -> Optional[Tuple[Any, Any]]:
        """Пытается извлечь пару узлов (src, dst) из произвольного представления ребра.
        Поддерживаются форматы:
        - кортеж индексов (i, j)
        - кортеж узлов (node_i, node_j)
        - объект с атрибутами/ключами source/target
        """
        try:
            # Кортеж или список
            if isinstance(edge, (tuple, list)) and len(edge) >= 2:
                a, b = edge[0], edge[1]
                # индексы
                if isinstance(a, int) and isinstance(b, int) and 0 <= a < len(nodes) and 0 <= b < len(nodes):
                    return nodes[a], nodes[b]
                # непосредственно узлы
                return a, b

            # Объект с атрибутами
            if hasattr(edge, 'source') and hasattr(edge, 'target'):
                src = getattr(edge, 'source')
                dst = getattr(edge, 'target')
                # индексы -> разрешаем в узлы
                if isinstance(src, int) and isinstance(dst, int):
                    if 0 <= src < len(nodes) and 0 <= dst < len(nodes):
                        return nodes[src], nodes[dst]
                return src, dst

            # Словарное представление
            if isinstance(edge, dict):
                src = edge.get('source') or edge.get('src')
                dst = edge.get('target') or edge.get('dst')
                if src is not None and dst is not None:
                    if isinstance(src, int) and isinstance(dst, int):
                        if 0 <= src < len(nodes) and 0 <= dst < len(nodes):
                            return nodes[src], nodes[dst]
                    return src, dst
        except Exception:
            pass
        return None

    def _get_edge_features(self, edge, nodes: List[Any]) -> Optional[np.ndarray]:
        """Формирует признаки ребра как конкатенацию эмбеддингов его концов."""
        try:
            resolved = self._resolve_edge_nodes(edge, nodes)
            if not resolved:
                return None
            n1, n2 = resolved
            emb1 = self._get_node_embedding(n1)
            emb2 = self._get_node_embedding(n2)
            if emb1 is None or emb2 is None:
                return None
            return np.concatenate([emb1, emb2], axis=-1)
        except Exception as e:
            logger.error(f"Ошибка формирования признаков ребра: {e}")
            return None

    def _generate_negative_edges(self, nodes: List[Any], edges: List[Any], max_samples: Optional[int] = None) -> List[Tuple[int, int]]:
        """Генерирует отрицательные (несуществующие) ребра как пары индексов узлов."""
        try:
            n = len(nodes)
            if n < 2:
                return []
            # Собираем множество существующих пар индексов
            existing: set[Tuple[int, int]] = set()

            def node_index(node_obj) -> Optional[int]:
                try:
                    return nodes.index(node_obj)
                except ValueError:
                    return None

            for e in edges or []:
                pair = None
                if isinstance(e, (tuple, list)) and len(e) >= 2:
                    a, b = e[0], e[1]
                    if isinstance(a, int) and isinstance(b, int):
                        pair = (a, b)
                    else:
                        ia = node_index(a)
                        ib = node_index(b)
                        if ia is not None and ib is not None:
                            pair = (ia, ib)
                elif hasattr(e, 'source') and hasattr(e, 'target'):
                    a = getattr(e, 'source')
                    b = getattr(e, 'target')
                    if isinstance(a, int) and isinstance(b, int):
                        pair = (a, b)
                    else:
                        ia = node_index(a)
                        ib = node_index(b)
                        if ia is not None and ib is not None:
                            pair = (ia, ib)
                elif isinstance(e, dict):
                    a = e.get('source') or e.get('src')
                    b = e.get('target') or e.get('dst')
                    if isinstance(a, int) and isinstance(b, int):
                        pair = (a, b)
                    else:
                        ia = node_index(a)
                        ib = node_index(b)
                        if ia is not None and ib is not None:
                            pair = (ia, ib)
                if pair and 0 <= pair[0] < n and 0 <= pair[1] < n:
                    existing.add(pair)

            import random
            random_pairs: List[Tuple[int, int]] = []
            target = max_samples if max_samples is not None else max(1, len(edges) or n)
            attempts = 0
            max_attempts = target * 10
            while len(random_pairs) < target and attempts < max_attempts:
                i = random.randrange(n)
                j = random.randrange(n)
                if i == j:
                    attempts += 1
                    continue
                pair = (i, j)
                if pair in existing:
                    attempts += 1
                    continue
                existing.add(pair)
                random_pairs.append(pair)
            return random_pairs
        except Exception as e:
            logger.error(f"Ошибка генерации отрицательных ребер: {e}")
            return []
    
    def _classify_node_type(self, node) -> int:
        """Классифицирует тип узла (0-4)."""
        try:
            # Определяем тип узла на основе его свойств
            if hasattr(node, 'type'):
                node_type = getattr(node, 'type', '').lower()
                if 'concept' in node_type:
                    return 0
                elif 'entity' in node_type:
                    return 1
                elif 'relation' in node_type:
                    return 2
                elif 'attribute' in node_type:
                    return 3
                else:
                    return 4
            
            # Если типа нет, определяем по содержимому
            content = getattr(node, 'content', '') or getattr(node, 'text', '')
            if len(content) > 100:
                return 0  # Концепт
            elif len(content) > 50:
                return 1  # Сущность
            else:
                return 2  # Отношение
                
        except Exception:
            return 4  # Неизвестный тип
    
    def start_learning_process(self):
        """Запускает процесс обучения (для совместимости с GUI)."""
        return self.train_async()
    
    def pause_learning_process(self):
        """Приостанавливает процесс обучения (для совместимости с GUI)."""
        return self.stop_training()
    
    def start_learning_process(self) -> bool:
        """
        Запускает процесс обучения графа памяти.
        
        Returns:
            bool: Успешно ли запущено обучение
        """
        if self.disabled:
            logger.info("🚀 Обучение графа памяти отключено - запуск пропущен")
            return False
        
        if self.is_training:
            logger.warning("Обучение уже запущено")
            return False
        
        return self.train_async()
    
    def train_async(self, epochs: Optional[int] = None) -> bool:
        """
        Запускает асинхронное обучение модели.
        
        Args:
            epochs: Количество эпох обучения
            
        Returns:
            bool: Успешно ли запущено обучение
        """
        if self.disabled:
            logger.info("🚀 Обучение графа памяти отключено - асинхронный запуск пропущен")
            return False
            
        if self.is_training:
            logger.warning("Обучение уже запущено")
            return False
        
        try:
            self.training_thread = threading.Thread(
                target=self._train_worker,
                args=(epochs,),
                daemon=True
            )
            self.training_thread.start()
            logger.info("Асинхронное обучение запущено")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка запуска асинхронного обучения: {e}")
            return False
    
    def _train_worker(self, epochs: Optional[int] = None):
        """Рабочий метод для обучения в отдельном потоке."""
        if self.disabled:
            logger.info("🚀 Обучение графа памяти отключено - worker пропущен")
            return
            
        try:
            self.is_training = True
            epochs = epochs or self.config.epochs
            
            logger.info(f"Начинаем обучение графа памяти на {epochs} эпох")
            
            # Подготавливаем данные
            node_features, edge_features, link_labels, node_labels = self.prepare_training_data()
            
            if node_features is None or len(node_features) == 0:
                logger.error("Не удалось подготовить данные для обучения")
                return
            
            # Сбрасываем счетчики
            self.training_stats["patience_counter"] = 0
            self.training_stats["best_loss"] = float('inf')
            
            # Обучаем модель
            for epoch in range(epochs):
                if not self.is_training:  # Проверяем, не остановили ли обучение
                    logger.info("Обучение остановлено пользователем")
                    break
                
                # Обучаем эпоху
                epoch_stats = self._train_epoch(node_features, edge_features, link_labels, node_labels)
                
                # Обновляем статистику
                self.training_stats["epoch"] = epoch + 1
                self.training_stats.update(epoch_stats)
                
                # Сохраняем историю
                epoch_record = {
                    "epoch": epoch + 1,
                    "timestamp": datetime.now().isoformat(),
                    **epoch_stats
                }
                self.training_stats["training_history"].append(epoch_record)
                
                # Проверяем улучшение
                current_loss = epoch_stats["total_loss"]
                if current_loss < self.training_stats["best_loss"] - self.config.min_delta:
                    self.training_stats["best_loss"] = current_loss
                    self.training_stats["patience_counter"] = 0
                    # Сохраняем лучшую модель
                    self._save_best_model()
                else:
                    self.training_stats["patience_counter"] += 1
                
                # Логируем прогресс
                logger.info(
                    f"Эпоха {epoch + 1}/{epochs}: "
                    f"Loss={current_loss:.4f}, "
                    f"Accuracy={epoch_stats['accuracy']:.4f}, "
                    f"Patience={self.training_stats['patience_counter']}"
                )
                
                # Проверяем раннюю остановку
                if self.training_stats["patience_counter"] >= self.config.patience:
                    logger.info(f"Ранняя остановка на эпохе {epoch + 1}")
                    break
            
            logger.info("Обучение графа памяти завершено")
            
        except Exception as e:
            logger.error(f"Ошибка во время обучения: {e}")
        finally:
            self.is_training = False
    
    def _train_epoch(self, node_features, edge_features, link_labels, node_labels) -> Dict[str, float]:
        """Обучает модель на одной эпохе."""
        self.model.train()
        model_dtype = next(self.model.parameters()).dtype
        
        total_loss = 0.0
        link_loss_total = 0.0
        node_loss_total = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Создаем батчи
        batch_size = self.config.batch_size
        num_batches = max(1, len(node_features) // batch_size)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(node_features))
            
            # Получаем батч данных
            batch_nodes = node_features[start_idx:end_idx].to(self.device, dtype=model_dtype, non_blocking=True)
            batch_node_labels = node_labels[start_idx:end_idx].to(self.device, dtype=torch.long, non_blocking=True)
            
            batch_edges = None
            batch_link_labels = None
            if edge_features is not None and link_labels is not None:
                if end_idx <= len(edge_features):
                    batch_edges = edge_features[start_idx:end_idx].to(self.device, dtype=model_dtype, non_blocking=True)
                    # Лейблы линков должны совпадать по dtype с выходами сигмоида
                    batch_link_labels = link_labels[start_idx:end_idx].to(self.device, dtype=model_dtype, non_blocking=True)
            
            # Обнуляем градиенты
            self.optimizer.zero_grad()
            
            # Прямой проход
            outputs = self.model(
                batch_nodes, 
                batch_edges, 
                predict_links=(batch_edges is not None),
                classify_nodes=True
            )
            
            # Вычисляем потери
            node_loss = self.node_criterion(outputs["node_types"], batch_node_labels)
            total_batch_loss = node_loss
            
            link_loss = torch.zeros((), device=self.device, dtype=model_dtype)
            if batch_edges is not None and "link_probs" in outputs:
                link_loss = self.link_criterion(outputs["link_probs"].squeeze(), batch_link_labels)
                total_batch_loss += link_loss
            
            # Обратное распространение
            total_batch_loss.backward()
            self.optimizer.step()
            
            # Обновляем статистику
            total_loss += total_batch_loss.item()
            node_loss_total += node_loss.item()
            link_loss_total += link_loss.item()
            
            # Вычисляем точность для классификации узлов
            _, predicted = torch.max(outputs["node_types"], 1)
            correct_predictions += (predicted == batch_node_labels).sum().item()
            total_predictions += batch_node_labels.size(0)
        
        # Вычисляем средние метрики
        avg_total_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_node_loss = node_loss_total / num_batches if num_batches > 0 else 0.0
        avg_link_loss = link_loss_total / num_batches if num_batches > 0 else 0.0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            "total_loss": avg_total_loss,
            "node_loss": avg_node_loss,
            "link_loss": avg_link_loss,
            "accuracy": accuracy
        }
    
    def stop_training(self) -> bool:
        """
        Останавливает обучение.
        
        Returns:
            bool: Успешно ли остановлено обучение
        """
        if not self.is_training:
            logger.warning("Обучение не запущено")
            return False
        
        try:
            self.is_training = False
            
            if self.training_thread and self.training_thread.is_alive():
                self.training_thread.join(timeout=5.0)
            
            logger.info("Обучение остановлено")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка остановки обучения: {e}")
            return False
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Возвращает статистику обучения."""
        return self.training_stats.copy()
    
    def _save_best_model(self):
        """Сохраняет лучшую модель."""
        try:
            if not hasattr(self, 'best_model_path'):
                cache_dir = getattr(self.brain, 'cache_dir', './cache') if self.brain else './cache'
                os.makedirs(cache_dir, exist_ok=True)
                self.best_model_path = os.path.join(cache_dir, 'best_memory_graph_model.pth')
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_stats': self.training_stats,
                'config': self.config
            }, self.best_model_path)
            
            logger.debug(f"Лучшая модель сохранена в {self.best_model_path}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения лучшей модели: {e}")
