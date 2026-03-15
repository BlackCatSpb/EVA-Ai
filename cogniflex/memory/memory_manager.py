"""Модуль управления памятью для CogniFlex"""
import os
import logging
import json
import time
import threading
import shutil
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Пытаемся импортировать psutil для получения реальных системных метрик памяти
try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # graceful fallback


logger = logging.getLogger("cogniflex.memory.manager")

class MemoryManager:
    """Менеджер памяти для CogniFlex, управляющий различными типами памяти и кэшированием."""
    
    def __init__(self, cache_dir: str, brain=None, knowledge_graph=None):
        """
        Инициализирует менеджер памяти.
        
        Args:
            cache_dir: Путь к директории кэша
            brain: Ссылка на ядро CogniFlex
            knowledge_graph: Ссылка на граф знаний
        """
        self.brain = brain
        self.knowledge_graph = knowledge_graph
        self.cache_dir = cache_dir
        self.initialized = False
        self.running = False
        self.stop_event = threading.Event()
        
        # Создаем директорию кэша если не существует
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Пути к файлам
        self.working_memory_file = os.path.join(self.cache_dir, "working_memory.json")
        self.semantic_memory_file = os.path.join(self.cache_dir, "semantic_memory.json")
        self.episodic_memory_file = os.path.join(self.cache_dir, "episodic_memory.json")
        self.user_profiles_file = os.path.join(self.cache_dir, "user_profiles.json")
        
        # Типы памяти
        self.working_memory = []
        self.semantic_memory = []
        self.episodic_memory = []
        self.user_profiles = {}
        
        # Блокировки для многопоточного доступа
        self.memory_locks = {
            "working": threading.RLock(),
            "semantic": threading.RLock(),
            "episodic": threading.RLock(),
            "user_profiles": threading.RLock()
        }
        
        # Инициализируем гибридный кэш ДОЛЖЕН БЫТЬ ЗДЕСЬ
        self.hybrid_cache = None
        self._init_hybrid_cache()
        
        # Инициализируем остальные компоненты
        self._initialize()
        logger.info("Менеджер памяти инициализирован")

    def _init_hybrid_cache(self):
        """Инициализирует гибридный кэш для токенизации и других операций."""
        try:
            # Импортируем HybridTokenCache с обработкой ошибок
            try:
                from cogniflex.memory.hybrid_token_cache import HybridTokenCache
                logger.debug("HybridTokenCache импортирован успешно")
            except ImportError:
                logger.warning("HybridTokenCache недоступен, кэширование будет ограничено")
                return

            # Инициализируем гибридный кэш
            # Если brain отсутствует или не имеет cache_dir, используем shim с cache_dir менеджера
            brain_obj = self.brain
            if brain_obj is None or not hasattr(brain_obj, "cache_dir") or not brain_obj.cache_dir:
                class _BrainShim:
                    def __init__(self, cache_dir: str):
                        self.cache_dir = cache_dir
                # Используем локальную директорию кэша менеджера как основу
                safe_cache_dir = self.cache_dir if os.path.isabs(self.cache_dir) else os.path.join(os.getcwd(), self.cache_dir)
                os.makedirs(safe_cache_dir, exist_ok=True)
                brain_obj = _BrainShim(safe_cache_dir)

            self.hybrid_cache = HybridTokenCache(
                brain=brain_obj,
                max_memory_tokens=10000,
                disk_cache_dir="hybrid_cache"
            )
            logger.info(f"Гибридный кэш инициализирован с размером памяти {10000} токенов.")

        except Exception as e:
            logger.error(f"Ошибка инициализации гибридного кэша: {e}", exc_info=True)
    
    def _initialize(self):
        """Инициализирует компоненты менеджера памяти."""
        try:
            # Загружаем сохраненные данные
            self._load_working_memory()
            self._load_semantic_memory()
            self._load_episodic_memory()
            self._load_user_profiles()
            
            self.initialized = True
        except Exception as e:
            logger.error(f"Ошибка инициализации менеджера памяти: {e}", exc_info=True)
    
    def get_hybrid_cache(self):
        """Возвращает экземпляр гибридного кэша."""
        return self.hybrid_cache

    # ===== GUI-facing compatibility API =====
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Возвращает агрегированную статистику памяти для GUI.

        По возможности использует реальные системные метрики через psutil.
        При недоступности psutil/ошибках — безопасный откат к прежним приближенным значениям.
        """
        try:
            total_nodes = sum([
                len(getattr(self, 'working_memory', [])),
                len(getattr(self, 'semantic_memory', [])),
                len(getattr(self, 'episodic_memory', []))
            ])

            # Попытка получить реальные метрики памяти
            total_gb: float
            used_gb: float
            free_gb: float
            cache_gb: float = 0.0

            if psutil is not None:
                try:
                    vm = psutil.virtual_memory()
                    gb = 1024.0 ** 3
                    # Используем total и available: used = total - available, free = available
                    total_gb = float(vm.total) / gb
                    free_gb = float(vm.available) / gb
                    used_gb = max(0.0, total_gb - free_gb)
                    cache_gb = float(getattr(vm, 'cached', 0) or 0) / gb
                except Exception:
                    # fallback к приближенным значениям на основе количества узлов
                    total_gb = 2.0
                    used_gb = min(2.0, 0.5 + total_nodes * 0.001)
                    free_gb = max(0.0, total_gb - used_gb)
                    cache_gb = 0.2
            else:
                # psutil недоступен — прежняя логика
                total_gb = 2.0
                used_gb = min(2.0, 0.5 + total_nodes * 0.001)
                free_gb = max(0.0, total_gb - used_gb)
                cache_gb = 0.2

            stats = {
                "total_memory": round(total_gb, 3),
                "used_memory": round(used_gb, 3),
                "free_memory": round(free_gb, 3),
                "cache_memory": round(cache_gb, 3),
                "total_nodes": total_nodes,
                "active_nodes": total_nodes,
                "cached_nodes": 0,
                "memory_efficiency": 0.8,
                "cache_hits": 0,
                "cache_hit_ratio": 0.0,
                "last_update": time.time()
            }
            return stats
        except Exception as e:
            logger.error(f"Ошибка формирования статистики памяти: {e}", exc_info=True)
            return {
                "total_memory": 2.0,
                "used_memory": 1.2,
                "free_memory": 0.8,
                "cache_memory": 0.2,
                "total_nodes": 0,
                "active_nodes": 0,
                "cached_nodes": 0,
                "memory_efficiency": 0.0,
                "cache_hits": 0,
                "cache_hit_ratio": 0.0,
                "last_update": time.time()
            }

    def analyze_memory_usage(self) -> Dict[str, Any]:
        """Возвращает простой анализ использования памяти для GUI."""
        try:
            domain_distribution: Dict[str, int] = {}
            for entry in getattr(self, 'semantic_memory', []):
                domain = entry.get('metadata', {}).get('domain', 'unknown') if isinstance(entry, dict) else 'unknown'
                domain_distribution[domain] = domain_distribution.get(domain, 0) + 1

            return {
                "efficiency_score": 0.8,
                "fragmentation_level": 0.2,
                "cache_hit_rate": 0.0,
                "recommendations": [
                    "Оптимизируйте сохранение повторяющейся информации",
                    "Рассмотрите возможность увеличения кэширования"
                ],
                "memory_trends": {
                    "usage_trend": "stable",
                    "efficiency_trend": "stable"
                },
                "domain_distribution": domain_distribution
            }
        except Exception as e:
            logger.error(f"Ошибка анализа памяти: {e}", exc_info=True)
            return {
                "efficiency_score": 0.0,
                "fragmentation_level": 0.0,
                "cache_hit_rate": 0.0,
                "recommendations": [],
                "memory_trends": {"usage_trend": "unknown", "efficiency_trend": "unknown"},
                "domain_distribution": {}
            }

    class _MemoryNodeShim:
        """Простой адаптер для представления записи памяти как узла для GUI."""
        def __init__(self, entry: Dict[str, Any]):
            self._e = entry
            self.id = entry.get("id")
            self.content = entry.get("content")
            self.node_type = entry.get("metadata", {}).get("type", "fact")
            self.domain = entry.get("metadata", {}).get("domain", "unknown")
            ts = entry.get("timestamp", time.time())
            self.created_at = ts
            self.last_updated = ts
            self.timestamp = ts  # совместимость со старыми вызовами GUI
            self.meta = entry.get("metadata", {})
            self.edges: list = []  # GUI экспорт ожидает поле edges

        def get_strength_factor(self) -> float:
            strength = self._e.get("metadata", {}).get("strength") if isinstance(self._e, dict) else None
            try:
                return float(strength) if strength is not None else 1.0
            except Exception:
                return 1.0

    def get_all_nodes(self) -> List[Any]:
        """Возвращает список узлов памяти для GUI (может быть пустым)."""
        nodes: List[MemoryManager._MemoryNodeShim] = []
        try:
            for mem_type in ("working_memory", "semantic_memory", "episodic_memory"):
                for entry in getattr(self, mem_type, []):
                    if isinstance(entry, dict) and "id" in entry:
                        nodes.append(MemoryManager._MemoryNodeShim(entry))
            return nodes
        except Exception as e:
            logger.error(f"Ошибка получения узлов памяти: {e}")
            return []

    def get_all_edges(self) -> List[Any]:
        """Возвращает список связей памяти (GUI ожидает метод) — по умолчанию пусто."""
        return []

    def get_node(self, node_id: str) -> Optional[Any]:
        """Возвращает узел по ID из различных типов памяти."""
        try:
            for mem_type in ("working_memory", "semantic_memory", "episodic_memory"):
                for entry in getattr(self, mem_type, []):
                    if isinstance(entry, dict) and entry.get("id") == node_id:
                        return MemoryManager._MemoryNodeShim(entry)
            return None
        except Exception as e:
            logger.error(f"Ошибка получения узла {node_id}: {e}")
            return None

    def remove_node(self, node_id: str) -> bool:
        """Удаляет запись памяти с заданным ID из всех типов памяти."""
        removed = False
        try:
            for mem_type in ("working_memory", "semantic_memory", "episodic_memory"):
                mem_list = getattr(self, mem_type, [])
                for i, entry in enumerate(list(mem_list)):
                    if isinstance(entry, dict) and entry.get("id") == node_id:
                        del mem_list[i]
                        removed = True
                # сохраняем при изменении
                if removed:
                    self._save_memory(mem_type.replace("_memory", ""))
            return removed
        except Exception as e:
            logger.error(f"Ошибка удаления узла {node_id}: {e}")
            return False

    def clear_cache(self):
        """Очищает кэш памяти, если доступен."""
        try:
            if self.hybrid_cache:
                if hasattr(self.hybrid_cache, "clear_all"):
                    self.hybrid_cache.clear_all()
                else:
                    if hasattr(self.hybrid_cache, "clear_memory"):
                        self.hybrid_cache.clear_memory()
                    if hasattr(self.hybrid_cache, "clear_disk"):
                        self.hybrid_cache.clear_disk()
            # удаляем директорию дискового кэша, если существует
            cache_dir = os.path.join(os.getcwd(), "hybrid_cache")
            if os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
            logger.info("Кэш памяти очищен")
        except Exception as e:
            logger.error(f"Ошибка очистки кэша памяти: {e}")
            raise

    def optimize_cache(self):
        """Оптимизация кэша (заглушка)."""
        logger.info("Оптимизация кэша выполнена")
    
    def _initialize(self):
        """Инициализирует внутренние компоненты менеджера памяти."""
        logger.info("Инициализация менеджера памяти...")
        try:
            # Загружаем данные
            self._load_working_memory()
            self._load_semantic_memory()
            self._load_episodic_memory()
            self._load_user_profiles()
            
            self.initialized = True
            logger.info("Менеджер памяти полностью инициализирован")
        except Exception as e:
            logger.error(f"Ошибка полной инициализации менеджера памяти: {e}", exc_info=True)
            self.initialized = False
    
    def start(self):
        """Запускает фоновые процессы менеджера памяти."""
        if not self.initialized:
            logger.error("Невозможно запустить неинициализированный менеджер памяти")
            return False
        
        self.running = True
        self.stop_event.clear()
        logger.info("Менеджер памяти запущен")
        return True
    
    def stop(self):
        """Останавливает фоновые процессы менеджера памяти."""
        self.running = False
        self.stop_event.set()
        logger.info("Менеджер памяти остановлен")
    
    def add_memory(self, memory_type: str, content: Any, metadata: Optional[Dict] = None, user_id: Optional[str] = None) -> str:
        """
        Добавляет информацию в указанную память.
        
        Args:
            memory_type: Тип памяти (working, semantic, episodic)
            content: Содержимое для сохранения
            metadata: Метаданные
            user_id: ID пользователя (для персональной информации)
            
        Returns:
            str: ID добавленной информации
        """
        memory_id = f"mem_{int(time.time())}_{os.urandom(4).hex()}"
        timestamp = time.time()
        
        memory_entry = {
            "id": memory_id,
            "content": content,
            "timestamp": timestamp,
            "metadata": metadata or {},
            "user_id": user_id
        }
        
        with self.memory_locks[memory_type]:
            if memory_type == "working":
                self.working_memory.append(memory_entry)
            elif memory_type == "semantic":
                self.semantic_memory.append(memory_entry)
            elif memory_type == "episodic":
                self.episodic_memory.append(memory_entry)
            else:
                raise ValueError(f"Неизвестный тип памяти: {memory_type}")
        
        # Сохраняем изменения
        self._save_memory(memory_type)
        logger.debug(f"Добавлена информация в {memory_type} память: {memory_id}")
        
        return memory_id
    
    def get_memory(self, memory_id: str) -> Optional[Dict]:
        """
        Получает информацию по ID.
        
        Args:
            memory_id: ID информации
            
        Returns:
            Optional[Dict]: Информация или None
        """
        for memory_type in ["working", "semantic", "episodic"]:
            with self.memory_locks[memory_type]:
                for entry in getattr(self, f"{memory_type}_memory"):
                    if entry["id"] == memory_id:
                        return entry.copy()
        return None
    
    def delete_memory(self, memory_id: str, memory_type: Optional[str] = None) -> bool:
        """
        Удаляет информацию.
        
        Args:
            memory_id: ID информации
            memory_type: Тип памяти (опционально)
            
        Returns:
            bool: Успешно ли удалено
        """
        if memory_type:
            with self.memory_locks[memory_type]:
                memory_list = getattr(self, f"{memory_type}_memory")
                for i, entry in enumerate(memory_list):
                    if entry["id"] == memory_id:
                        del memory_list[i]
                        self._save_memory(memory_type)
                        logger.debug(f"Удалена информация из {memory_type} памяти: {memory_id}")
                        return True
                return False
        else:
            for memory_type in ["working", "semantic", "episodic"]:
                if self.delete_memory(memory_id, memory_type):
                    return True
            return False
    
    def get_user_profile(self, user_id: str) -> Dict:
        """
        Получает профиль пользователя.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Dict: Профиль пользователя
        """
        with self.memory_locks["user_profiles"]:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = {
                    "id": user_id,
                    "preferences": {},
                    "interaction_history": [],
                    "created_at": time.time(),
                    "last_active": time.time()
                }
            return self.user_profiles[user_id].copy()
    
    def update_user_profile(self, user_id: str, updates: Dict) -> bool:
        """
        Обновляет профиль пользователя.
        
        Args:
            user_id: ID пользователя
            updates: Обновления
            
        Returns:
            bool: Успешно ли обновлено
        """
        with self.memory_locks["user_profiles"]:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = {
                    "id": user_id,
                    "preferences": {},
                    "interaction_history": [],
                    "created_at": time.time(),
                    "last_active": time.time()
                }
            
            # Обновляем поля
            for key, value in updates.items():
                if key == "preferences":
                    self.user_profiles[user_id]["preferences"].update(value)
                elif key == "interaction_history":
                    self.user_profiles[user_id]["interaction_history"].extend(value)
                else:
                    self.user_profiles[user_id][key] = value
            
            self.user_profiles[user_id]["last_active"] = time.time()
            self._save_user_profiles()
            logger.debug(f"Профиль пользователя {user_id} обновлен")
            return True
    
    def add_interaction(self, user_id: str, query: str, response: str, context: Optional[Dict] = None) -> str:
        """
        Добавляет взаимодействие в историю.
        
        Args:
            user_id: ID пользователя
            query: Запрос пользователя
            response: Ответ системы
            context: Контекст
            
        Returns:
            str: ID взаимодействия
        """
        interaction_id = f"inter_{int(time.time())}_{os.urandom(4).hex()}"
        
        interaction = {
            "id": interaction_id,
            "user_id": user_id,
            "query": query,
            "response": response,
            "timestamp": time.time(),
            "context": context or {}
        }
        
        # Добавляем во временную память
        self.add_memory("working", interaction, {"type": "interaction"}, user_id)
        
        # Добавляем в профиль пользователя
        with self.memory_locks["user_profiles"]:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = {
                    "id": user_id,
                    "preferences": {},
                    "interaction_history": [],
                    "created_at": time.time(),
                    "last_active": time.time()
                }
            
            self.user_profiles[user_id]["interaction_history"].append(interaction)
            self.user_profiles[user_id]["last_active"] = time.time()
        
        # Сохраняем изменения
        self._save_user_profiles()
        logger.debug(f"Взаимодействие добавлено: {interaction_id}")
        
        return interaction_id
    
    def update_interaction_response(self, interaction_id: str, response: str) -> bool:
        """
        Обновляет ответ в истории взаимодействий.
        
        Args:
            interaction_id: ID взаимодействия
            response: Новый ответ
            
        Returns:
            bool: Успешно ли обновлено
        """
        # Ищем во временной памяти
        for memory_type in ["working", "semantic", "episodic"]:
            with self.memory_locks[memory_type]:
                for entry in getattr(self, f"{memory_type}_memory"):
                    if entry["id"] == interaction_id and "type" in entry["metadata"] and entry["metadata"]["type"] == "interaction":
                        entry["content"]["response"] = response
                        self._save_memory(memory_type)
                        logger.debug(f"Ответ в истории обновлен: {interaction_id}")
                        return True
        
        # Ищем в профилях пользователей
        with self.memory_locks["user_profiles"]:
            for user_id, profile in self.user_profiles.items():
                for i, interaction in enumerate(profile["interaction_history"]):
                    if interaction["id"] == interaction_id:
                        profile["interaction_history"][i]["response"] = response
                        self._save_user_profiles()
                        logger.debug(f"Ответ в профиле пользователя обновлен: {interaction_id}")
                        return True
        
        return False
    
    def get_recent_actions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Возвращает недавние действия системы.
        
        Args:
            limit: Максимальное количество действий
            
        Returns:
            List[Dict[str, Any]]: Список действий
        """
        actions = []
        
        # Получаем из временной памяти
        with self.memory_locks["working"]:
            for entry in self.working_memory:
                if "type" in entry["metadata"] and entry["metadata"]["type"] == "action":
                    actions.append({
                        "id": entry["id"],
                        "type": entry["metadata"].get("action_type", "unknown"),
                        "description": entry["content"],
                        "timestamp": entry["timestamp"],
                        "system": True
                    })
        
        # Получаем из профилей пользователей (взаимодействия как действия)
        with self.memory_locks["user_profiles"]:
            for user_id, profile in self.user_profiles.items():
                for interaction in profile["interaction_history"]:
                    actions.append({
                        "id": interaction["id"],
                        "type": "user_interaction",
                        "description": f"Пользователь {user_id}: {interaction['query']}",
                        "timestamp": interaction["timestamp"],
                        "user_id": user_id
                    })
        
        # Сортируем по времени и ограничиваем
        actions.sort(key=lambda x: x["timestamp"], reverse=True)
        return actions[:limit]
    
    def get_recent_interactions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Возвращает недавние взаимодействия с пользователями.
        
        Args:
            limit: Максимальное количество взаимодействий
            
        Returns:
            List[Dict[str, Any]]: Список взаимодействий
        """
        interactions = []
        
        # Получаем из профилей пользователей
        with self.memory_locks["user_profiles"]:
            for user_id, profile in self.user_profiles.items():
                interactions.extend(profile["interaction_history"])
        
        # Сортируем по времени и ограничиваем
        interactions.sort(key=lambda x: x["timestamp"], reverse=True)
        return interactions[:limit]
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """
        Возвращает всех пользователей.
        
        Returns:
            List[Dict[str, Any]]: Список пользователей
        """
        with self.memory_locks["user_profiles"]:
            return [{"id": user_id, "last_active": profile["last_active"]} 
                    for user_id, profile in self.user_profiles.items()]
    
    def _load_working_memory(self):
        """Загружает рабочую память из файла."""
        try:
            if os.path.exists(self.working_memory_file):
                with open(self.working_memory_file, 'r', encoding='utf-8') as f:
                    self.working_memory = json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки рабочей памяти: {e}")
    
    def _save_working_memory(self):
        """Сохраняет рабочую память в файл."""
        try:
            with open(self.working_memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.working_memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения рабочей памяти: {e}")
    
    def _load_semantic_memory(self):
        """Загружает семантическую память из файла."""
        try:
            if os.path.exists(self.semantic_memory_file):
                with open(self.semantic_memory_file, 'r', encoding='utf-8') as f:
                    self.semantic_memory = json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки семантической памяти: {e}")
    
    def _save_semantic_memory(self):
        """Сохраняет семантическую память в файл."""
        try:
            with open(self.semantic_memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.semantic_memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения семантической памяти: {e}")
    
    def _load_episodic_memory(self):
        """Загружает эпизодическую память из файла."""
        try:
            if os.path.exists(self.episodic_memory_file):
                with open(self.episodic_memory_file, 'r', encoding='utf-8') as f:
                    self.episodic_memory = json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки эпизодической памяти: {e}")
    
    def _save_episodic_memory(self):
        """Сохраняет эпизодическую память в файл."""
        try:
            with open(self.episodic_memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.episodic_memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения эпизодической памяти: {e}")
    
    def _load_user_profiles(self):
        """Загружает профили пользователей из файла."""
        try:
            if os.path.exists(self.user_profiles_file):
                with open(self.user_profiles_file, 'r', encoding='utf-8') as f:
                    self.user_profiles = json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки профилей пользователей: {e}")
    
    def _save_user_profiles(self):
        """Сохраняет профили пользователей в файл."""
        try:
            with open(self.user_profiles_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_profiles, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения профилей пользователей: {e}")
    
    def _save_memory(self, memory_type: str):
        """Сохраняет указанный тип памяти."""
        if memory_type == "working":
            self._save_working_memory()
        elif memory_type == "semantic":
            self._save_semantic_memory()
        elif memory_type == "episodic":
            self._save_episodic_memory()
    
    def set_cache_size(self, cache_size: int):
        """
        Устанавливает размер кэша.
        
        Args:
            cache_size: Новый размер кэша
        """
        self.cache_size = cache_size
        logger.info(f"Установлен размер кэша: {cache_size}")
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Возвращает статус памяти."""
        return {
            "working_memory_size": len(self.working_memory),
            "semantic_memory_size": len(self.semantic_memory),
            "episodic_memory_size": len(self.episodic_memory),
            "user_profiles_count": len(self.user_profiles),
            "memory_usage": (len(self.working_memory) + len(self.semantic_memory) + len(self.episodic_memory)) / 1000.0,
            "initialized": self.initialized,
            "running": self.running
        }