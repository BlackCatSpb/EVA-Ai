"""
Hot Deployment Module для EVA
Модульная архитектура с постоянной загрузкой модели в "горячем" состоянии.

Особенности:
- Графоподобная структура узлов с индексами
- Интеграция с гибридным кэшем
- Постоянная готовность модели к генерации
- Модульная архитектура - не изменяет текущую систему
"""
import os
import json
import time
import threading
import hashlib
import logging
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("eva_ai.mlearning.hot_deployment")


class NodeState(Enum):
    """Состояния узла в графе"""
    INACTIVE = "inactive"       # Не загружен
    LOADING = "loading"         # Загружается
    HOT = "hot"                 # Готов к использованию
    WARMING = "warming"         # Разогревается
    COOLING = "cooling"         # Остывает


@dataclass
class NodeIndex:
    """Индекс узла для быстрого поиска"""
    node_id: str
    depth: int                  # Глубина в графе
    parent_id: Optional[str]    # Родительский узел
    child_ids: List[str]       # Дочерние узлы
    address: str               # Уникальный адрес (например: 0.1.2.3)
    state: NodeState = NodeState.INACTIVE
    last_access: float = 0.0
    access_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "address": self.address,
            "state": self.state.value,
            "last_access": self.last_access,
            "access_count": self.access_count
        }


@dataclass
class NodeMetadata:
    """Метаданные узла"""
    node_id: str
    description: str
    purpose: str               # Назначение узла
    model_layer: Optional[int] # Какие слои модели загружены
    memory_footprint: int      # Оценочный размер в байтах
    created_at: float = field(default_factory=time.time)


class GraphNode:
    """
    Узел графа с моделью в горячем состоянии.
    Каждый узел представляет собой контекст/состояние генерации.
    """
    
    _node_counter = 0
    _counter_lock = threading.Lock()
    
    def __init__(
        self,
        node_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        address: str = "0",
        depth: int = 0
    ):
        # Генерация ID если не передан
        if node_id is None:
            with GraphNode._counter_lock:
                GraphNode._node_counter += 1
                node_id = f"node_{GraphNode._node_counter}"
        
        self.node_id = node_id
        self.parent_id = parent_id
        self.address = address
        self.depth = depth
        self.state = NodeState.INACTIVE
        
        # Модель и токенизатор
        self.model = None
        self.tokenizer = None
        
        # Индекс
        self.index = NodeIndex(
            node_id=node_id,
            depth=depth,
            parent_id=parent_id,
            child_ids=[],
            address=address,
            state=NodeState.INACTIVE
        )
        
        # Метаданные
        self.metadata = NodeMetadata(
            node_id=node_id,
            description="",
            purpose="general",
            model_layer=None,
            memory_footprint=0
        )
        
        # Контекст генерации
        self.context = ""
        self.last_response = ""
        self.generation_params = {}
        
        # Блокировка для потокобезопасности
        self._lock = threading.RLock()
        
        logger.debug(f"Создан узел: {node_id} с адресом {address}")
    
    def activate(
        self,
        model,
        tokenizer,
        description: str = "",
        purpose: str = "general"
    ) -> bool:
        """Активирует узел - загружает модель в горячее состояние"""
        with self._lock:
            if self.state == NodeState.HOT:
                logger.debug(f"Узел {self.node_id} уже в горячем состоянии")
                return True
            
            self.state = NodeState.LOADING
            self.model = model
            self.tokenizer = tokenizer
            
            # Обновляем индекс
            self.index.state = NodeState.HOT
            self.index.last_access = time.time()
            self.index.access_count += 1
            
            # Метаданные
            self.metadata.description = description
            self.metadata.purpose = purpose
            
            # Оценочный размер (для 0.8B модели ~1.6GB в float16)
            self.metadata.memory_footprint = 1_600_000_000
            self.metadata.model_layer = None  # Вся модель
            
            self.state = NodeState.HOT
            logger.info(f"Узел {self.node_id} активирован (HOT), адрес: {self.address}")
            return True
    
    def deactivate(self) -> bool:
        """Деактивирует узел - выгружает модель"""
        with self._lock:
            if self.state == NodeState.INACTIVE:
                return True
            
            # Очищаем модель
            if self.model is not None:
                del self.model
                self.model = None
            
            self.tokenizer = None
            
            # Обновляем состояние
            self.state = NodeState.INACTIVE
            self.index.state = NodeState.INACTIVE
            
            logger.info(f"Узел {self.node_id} деактивирован")
            return True
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs
    ) -> Optional[str]:
        """Генерирует текст если узел в горячем состоянии"""
        with self._lock:
            if self.state != NodeState.HOT:
                logger.warning(f"Узел {self.node_id} не готов к генерации: {self.state}")
                return None
            
            if self.model is None or self.tokenizer is None:
                logger.error(f"Узел {self.node_id}: модель или токенизатор None")
                return None
            
            try:
                # Токенизация
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
                
                # Генерация
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self.tokenizer.eos_token_id,
                        **kwargs
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Убираем промпт из ответа
                if prompt in response:
                    response = response.replace(prompt, "").strip()
                
                # Сохраняем контекст
                self.context = prompt
                self.last_response = response
                self.index.last_access = time.time()
                
                logger.debug(f"Узел {self.node_id} сгенерировал {len(response)} символов")
                return response
                
            except Exception as e:
                logger.error(f"Ошибка генерации в узле {self.node_id}: {e}")
                return None
    
    def update_context(self, context: str):
        """Обновляет контекст узла"""
        with self._lock:
            self.context = context
    
    def get_info(self) -> Dict:
        """Возвращает информацию об узле"""
        with self._lock:
            return {
                "node_id": self.node_id,
                "address": self.address,
                "depth": self.depth,
                "state": self.state.value,
                "purpose": self.metadata.purpose,
                "access_count": self.index.access_count,
                "has_model": self.model is not None,
                "has_tokenizer": self.tokenizer is not None,
                "last_access": self.index.last_access,
                "memory_mb": self.metadata.memory_footprint / (1024 * 1024)
            }


class FractalGraph:
    """
    Графоподобная структура для горячего развертывания.
    Использует фрактальную адресацию (类似 IP адресам).
    
    Пример адресации:
    - 0 - корневой узел
    - 0.1 - первый дочерний
    - 0.1.2 - второй уровень
    - 0.1.2.3 - третий уровень
    """
    
    def __init__(self, max_depth: int = 4, max_children: int = 10):
        self.max_depth = max_depth
        self.max_children = max_children
        
        # Хранилище узлов
        self._nodes: Dict[str, GraphNode] = {}
        
        # Индекс для быстрого поиска
        self._address_index: Dict[str, str] = {}  # address -> node_id
        self._state_index: Dict[NodeState, List[str]] = {
            NodeState.HOT: [],
            NodeState.INACTIVE: [],
            NodeState.LOADING: [],
            NodeState.WARMING: [],
            NodeState.COOLING: []
        }
        
        # Блокировка (до создания корневого узла!)
        self._lock = threading.RLock()
        
        # Корневой узел
        self._root = GraphNode(node_id="root", address="0", depth=0)
        self._add_node(self._root)
        
        logger.info(f"FractalGraph инициализирован: max_depth={max_depth}, max_children={max_children}")
    
    def _add_node(self, node: GraphNode):
        """Добавляет узел в граф"""
        with self._lock:
            self._nodes[node.node_id] = node
            self._address_index[node.address] = node.node_id
            
            # Обновляем родительский узел
            if node.parent_id and node.parent_id in self._nodes:
                parent = self._nodes[node.parent_id]
                parent.index.child_ids.append(node.node_id)
    
    def create_child_node(
        self,
        parent_address: str,
        description: str = "",
        purpose: str = "refinement"
    ) -> Optional[GraphNode]:
        """
        Создает дочерний узел.
        Возвращает None если превышен лимит глубины или детей.
        """
        with self._lock:
            # Находим родительский узел
            parent_id = self._address_index.get(parent_address)
            if not parent_id:
                logger.error(f"Родительский узел не найден: {parent_address}")
                return None
            
            parent = self._nodes[parent_id]
            
            # Проверяем глубину
            if parent.depth >= self.max_depth:
                logger.warning(f"Достигнута максимальная глубина: {self.max_depth}")
                return None
            
            # Проверяем количество детей
            if len(parent.index.child_ids) >= self.max_children:
                logger.warning(f"Достигнуто максимальное количество детей: {self.max_children}")
                return None
            
            # Генерируем адрес
            child_num = len(parent.index.child_ids) + 1
            child_address = f"{parent_address}.{child_num}"
            
            # Создаем узел
            child = GraphNode(
                node_id=f"node_{child_address.replace('.', '_')}",
                parent_id=parent_id,
                address=child_address,
                depth=parent.depth + 1
            )
            child.metadata.description = description
            child.metadata.purpose = purpose
            
            self._add_node(child)
            logger.info(f"Создан дочерний узел: {child_address}, родитель: {parent_address}")
            
            return child
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Получает узел по ID"""
        with self._lock:
            return self._nodes.get(node_id)
    
    def get_node_by_address(self, address: str) -> Optional[GraphNode]:
        """Получает узел по адресу"""
        with self._lock:
            node_id = self._address_index.get(address)
            if node_id:
                return self._nodes.get(node_id)
            return None
    
    def get_hot_nodes(self) -> List[GraphNode]:
        """Возвращает все горячие узлы"""
        with self._lock:
            return [
                node for node in self._nodes.values()
                if node.state == NodeState.HOT
            ]
    
    def get_best_node(self) -> Optional[GraphNode]:
        """
        Возвращает лучший узел для генерации.
        Приоритет: HOT узлы с наибольшим access_count.
        """
        with self._lock:
            hot_nodes = self.get_hot_nodes()
            if not hot_nodes:
                return None
            
            # Сортируем по access_count (DESC)
            hot_nodes.sort(key=lambda n: n.index.access_count, reverse=True)
            return hot_nodes[0]
    
    def get_stats(self) -> Dict:
        """Возвращает статистику графа"""
        with self._lock:
            return {
                "total_nodes": len(self._nodes),
                "hot_nodes": len(self.get_hot_nodes()),
                "max_depth_reached": max((n.depth for n in self._nodes.values()), default=0),
                "nodes_by_state": {
                    state.value: len(nodes)
                    for state, nodes in self._state_index.items()
                }
            }


class HotDeploymentManager:
    """
    Главный менеджер горячего развертывания.
    Управляет графом узлов и взаимодействует с гибридным кэшем.
    """
    
    def __init__(
        self,
        model_path: str,
        hybrid_cache=None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16
    ):
        self.model_path = model_path
        self.hybrid_cache = hybrid_cache
        self.device = device
        self.dtype = dtype
        
        # Модель и токенизатор (загружаются один раз)
        self.model = None
        self.tokenizer = None
        
        # Граф узлов
        self.graph = FractalGraph(max_depth=4, max_children=10)
        
        # Конфигурация генерации по умолчанию
        self.default_params = {
            "max_new_tokens": 100,
            "do_sample": False,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        # Флаг готовности
        self.ready = False
        self._lock = threading.RLock()
        
        logger.info(f"HotDeploymentManager инициализирован: model_path={model_path}, device={device}")
    
    def initialize(self, preload_root: bool = True) -> bool:
        """
        Инициализирует менеджер - загружает модель.
        """
        try:
            logger.info("Загрузка модели для горячего развертывания...")
            
            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Загружаем модель
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            
            logger.info(f"Модель загружена: {self.model_path}")
            
            # Активируем корневой узел
            if preload_root:
                root = self.graph._root
                root.activate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    description="Корневой узел",
                    purpose="root"
                )
                self.ready = True
                logger.info("Горячее развертывание готово!")
            
            return self.ready
            
        except Exception as e:
            logger.error(f"Ошибка инициализации HotDeploymentManager: {e}")
            return False
    
    def activate_node(
        self,
        address: str,
        description: str = "",
        purpose: str = "refinement"
    ) -> bool:
        """
        Активирует узел по адресу.
        Если узел не существует - создает его.
        """
        with self._lock:
            # Пробуем найти существующий узел
            node = self.graph.get_node_by_address(address)
            
            if node is None:
                # Создаем новый узел (родитель - корень)
                parts = address.split(".")
                if len(parts) > 1:
                    parent_address = ".".join(parts[:-1])
                else:
                    parent_address = "0"
                
                node = self.graph.create_child_node(
                    parent_address=parent_address,
                    description=description,
                    purpose=purpose
                )
                
                if node is None:
                    return False
            
            # Активируем узел
            return node.activate(
                model=self.model,
                tokenizer=self.tokenizer,
                description=description,
                purpose=purpose
            )
    
    def generate(
        self,
        prompt: str,
        node_address: Optional[str] = None,
        use_best_node: bool = True,
        **kwargs
    ) -> Optional[str]:
        """
        Генерирует текст через горячий узел.
        
        Args:
            prompt: Промпт для генерации
            node_address: Конкретный адрес узла (опционально)
            use_best_node: Использовать лучший доступный узел
            **kwargs: Дополнительные параметры генерации
            
        Returns:
            str: Сгенерированный текст или None
        """
        with self._lock:
            if not self.ready:
                logger.error("HotDeploymentManager не готов")
                return None
            
            # Определяем узел
            node = None
            
            if node_address:
                node = self.graph.get_node_by_address(node_address)
            elif use_best_node:
                node = self.graph.get_best_node()
            
            if node is None:
                logger.warning("Нет доступных горячих узлов")
                return None
            
            #合并 параметры
            params = {**self.default_params, **kwargs}
            
            # Генерация
            return node.generate(prompt, **params)
    
    def generate_with_refinement(
        self,
        query: str,
        feedback: Dict[str, str],
        max_iterations: int = 3
    ) -> Tuple[Optional[str], List[str]]:
        """
        Генерирует с итеративным уточнением через разные узлы.
        
        Returns:
            Tuple[final_response, list_of_thoughts]
        """
        thoughts = []
        
        # Начальная генерация
        initial_response = self.generate(
            prompt=f"Ты - ЕВА. Ответь на вопрос: {query}",
            node_address="0"  # Корневой узел
        )
        
        if not initial_response:
            return None, ["Ошибка начальной генерации"]
        
        thoughts.append(f"Начальный ответ: {initial_response[:100]}...")
        
        # Итерации уточнения
        for i in range(max_iterations):
            # Создаем узел для уточнения
            node_address = f"0.{i+1}"
            self.activate_node(
                address=node_address,
                description=f"Уточнение итерация {i+1}",
                purpose="refinement"
            )
            
            # Генерируем уточненный ответ
            refined_response = self.generate(
                prompt=f"Улучши этот ответ: {initial_response}\n\nУчти обратную связь: {feedback}",
                node_address=node_address,
                max_new_tokens=150
            )
            
            if refined_response:
                initial_response = refined_response
                thoughts.append(f"Итерация {i+1}: {refined_response[:100]}...")
        
        return initial_response, thoughts
    
    def get_status(self) -> Dict:
        """Возвращает статус системы горячего развертывания"""
        with self._lock:
            return {
                "ready": self.ready,
                "model_loaded": self.model is not None,
                "tokenizer_loaded": self.tokenizer is not None,
                "device": self.device,
                "dtype": str(self.dtype),
                "graph_stats": self.graph.get_stats(),
                "hot_nodes": [
                    node.get_info() for node in self.graph.get_hot_nodes()
                ]
            }
    
    def deactivate_all(self):
        """Деактивирует все узлы (кроме корневого)"""
        with self._lock:
            for node in list(self.graph._nodes.values()):
                if node.node_id != "root":
                    node.deactivate()
            logger.info("Все узлы деактивированы")


# ============================================================================
# Глобальный экземпляр и функции доступа
# ============================================================================

_hot_deployment_instance: Optional[HotDeploymentManager] = None
_init_lock = threading.Lock()


def get_hot_deployment_manager(
    model_path: Optional[str] = None,
    force_reload: bool = False,
    hybrid_cache=None
) -> HotDeploymentManager:
    """
    Возвращает синглтон экземпляр HotDeploymentManager.
    """
    global _hot_deployment_instance
    
    with _init_lock:
        if _hot_deployment_instance is None or force_reload:
            # Определяем путь к модели
            if model_path is None:
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                model_path = os.path.join(
                    project_root, "mlearning", "eva_models", "qwen3.5-0.8b"
                )
            
            _hot_deployment_instance = HotDeploymentManager(
                model_path=model_path,
                hybrid_cache=hybrid_cache,
                device="cpu",
                dtype=torch.float16
            )
        
        return _hot_deployment_instance


def initialize_hot_deployment(
    model_path: Optional[str] = None,
    preload_root: bool = True
) -> bool:
    """
    Инициализирует горячее развертывание.
    """
    manager = get_hot_deployment_manager(model_path=model_path)
    return manager.initialize(preload_root=preload_root)