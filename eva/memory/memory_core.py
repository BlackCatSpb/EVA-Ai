# memory_core.py
import os
import logging
import sqlite3
import json
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

logger = logging.getLogger("eva.memory.core")

class MemoryNeuron:
    """Представляет нейрон в системе памяти."""
    def __init__(self, id: str, content_type: str, content: Any, strength: float = 1.0,
                 importance: float = 0.5, timestamp: float = 0.0, last_accessed: float = 0.0,
                 access_count: int = 0, metadata: Optional[Dict] = None, connections: Optional[Dict] = None):
        self.id = id
        self.content_type = content_type
        self.content = content
        self.strength = max(0.0, min(1.0, strength))
        self.importance = max(0.0, min(1.0, importance))
        self.timestamp = timestamp or time.time()
        self.last_accessed = last_accessed or self.timestamp
        self.access_count = access_count
        self.metadata = metadata or {}
        self.connections = connections or {}

class MemoryField:
    """Представляет поле (область) в системе памяти."""
    def __init__(self, name: str, description: str, capacity: int, current_size: int = 0,
                 last_updated: float = 0.0, metadata: Optional[Dict] = None, access_patterns: Optional[List] = None):
        self.name = name
        self.description = description
        self.capacity = capacity
        self.current_size = current_size
        self.last_updated = last_updated or time.time()
        self.metadata = metadata or {}
        self.access_patterns = access_patterns or []

class MemoryDatabase:
    """Управляет базой данных для хранения памяти."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db = self._init_database()
    
    def _init_database(self) -> sqlite3.Connection:
        """Инициализирует структуру базы данных."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Таблица для нейронов памяти
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS active_memory (
                    id TEXT PRIMARY KEY,
                    content_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    importance REAL DEFAULT 0.5,
                    timestamp REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    metadata TEXT,
                    connections TEXT
                )
            ''')
            
            # Таблица для долгосрочной памяти
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id TEXT PRIMARY KEY,
                    content_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    importance REAL DEFAULT 0.5,
                    timestamp REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    metadata TEXT,
                    connections TEXT
                )
            ''')
            
            # Таблица для полей памяти
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_fields (
                    name TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    capacity INTEGER NOT NULL,
                    current_size INTEGER DEFAULT 0,
                    last_updated REAL NOT NULL,
                    metadata TEXT,
                    access_patterns TEXT
                )
            ''')
            
            # Таблица для метрик памяти
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_metrics (
                    timestamp REAL NOT NULL,
                    active_memory_usage REAL NOT NULL,
                    long_term_memory_usage REAL NOT NULL,
                    total_neurons INTEGER NOT NULL,
                    active_neurons INTEGER NOT NULL,
                    long_term_neurons INTEGER NOT NULL
                )
            ''')
            
            conn.commit()
            logger.debug("Структура базы данных памяти инициализирована")
            return conn
        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных памяти: {e}")
            raise
    
    def save_neuron(self, neuron: MemoryNeuron, memory_type: str):
        """Сохраняет нейрон в базу данных."""
        try:
            cursor = self.db.cursor()
            table = "active_memory" if memory_type == "active" else "long_term_memory"
            cursor.execute(f'''
                INSERT OR REPLACE INTO {table}
                (id, content_type, content, strength, importance, timestamp, last_accessed, access_count, metadata, connections)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                neuron.id,
                neuron.content_type,
                json.dumps(neuron.content),
                neuron.strength,
                neuron.importance,
                neuron.timestamp,
                neuron.last_accessed,
                neuron.access_count,
                json.dumps(neuron.metadata),
                json.dumps(neuron.connections)
            ))
            self.db.commit()
        except Exception as e:
            logger.error(f"Ошибка сохранения нейрона {neuron.id}: {e}")
    
    def load_neuron(self, neuron_id: str, memory_type: str) -> Optional[MemoryNeuron]:
        """Загружает нейрон из базы данных."""
        try:
            cursor = self.db.cursor()
            table = "active_memory" if memory_type == "active" else "long_term_memory"
            cursor.execute(f'''
                SELECT id, content_type, content, strength, importance, timestamp, last_accessed, access_count, metadata, connections
                FROM {table}
                WHERE id = ?
            ''', (neuron_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return MemoryNeuron(
                id=row[0],
                content_type=row[1],
                content=json.loads(row[2]),
                strength=row[3],
                importance=row[4],
                timestamp=row[5],
                last_accessed=row[6],
                access_count=row[7],
                metadata=json.loads(row[8]),
                connections=json.loads(row[9])
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки нейрона {neuron_id}: {e}")
            return None
    
    def close(self):
        """Закрывает соединение с базой данных."""
        if hasattr(self, 'db'):
            self.db.close()
            logger.debug("Соединение с базой данных памяти закрыто")

class MemoryCore:
    """Ядро системы памяти ЕВА."""
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None):
        """
        Инициализирует ядро памяти.
        
        Args:
            brain: Ссылка на ядро ЕВА
            cache_dir: Путь к директории кэша
        """
        self.brain = brain
        self.cache_dir = cache_dir or "memory_cache"
        self.initialized = False
        
        # Создаем директорию кэша
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Инициализируем базу данных
        db_path = os.path.join(self.cache_dir, "memory_core.db")
        self.database = MemoryDatabase(db_path)
        
        # Внутренние структуры
        self.active_neurons: Dict[str, MemoryNeuron] = {}
        self.memory_fields: Dict[str, MemoryField] = {}
        
        # Инициализируем поля памяти по умолчанию
        self._init_default_fields()
        
        self.initialized = True
        logger.info("MemoryCore инициализирован")
    
    def _init_default_fields(self):
        """Инициализирует поля памяти по умолчанию."""
        default_fields = [
            MemoryField("working", "Рабочая память", 1000),
            MemoryField("semantic", "Семантическая память", 10000),
            MemoryField("episodic", "Эпизодическая память", 5000)
        ]
        
        for field in default_fields:
            self.memory_fields[field.name] = field
    
    def is_ready(self) -> bool:
        """Проверяет готовность ядра памяти."""
        return self.initialized
    
    def store_memory(self, content: Any, content_type: str, importance: float = 0.5, 
                    field_name: str = "working") -> str:
        """Сохраняет память в указанном поле."""
        try:
            # Генерируем ID
            memory_id = hashlib.md5(f"{content_type}_{str(content)}_{time.time()}".encode()).hexdigest()
            
            # Создаем нейрон
            neuron = MemoryNeuron(
                id=memory_id,
                content_type=content_type,
                content=content,
                importance=importance
            )
            
            # Сохраняем в активную память
            self.active_neurons[memory_id] = neuron
            self.database.save_neuron(neuron, "active")
            
            logger.debug(f"Память сохранена: {memory_id}")
            return memory_id
        except Exception as e:
            logger.error(f"Ошибка сохранения памяти: {e}")
            return ""
    
    def retrieve_memory(self, memory_id: str) -> Optional[MemoryNeuron]:
        """Извлекает память по ID."""
        try:
            # Сначала ищем в активной памяти
            if memory_id in self.active_neurons:
                neuron = self.active_neurons[memory_id]
                neuron.access_count += 1
                neuron.last_accessed = time.time()
                return neuron
            
            # Затем в базе данных
            neuron = self.database.load_neuron(memory_id, "active")
            if not neuron:
                neuron = self.database.load_neuron(memory_id, "long_term")
            
            if neuron:
                neuron.access_count += 1
                neuron.last_accessed = time.time()
                self.active_neurons[memory_id] = neuron
            
            return neuron
        except Exception as e:
            logger.error(f"Ошибка извлечения памяти {memory_id}: {e}")
            return None
    
    def get_system_health(self) -> Dict[str, Any]:
        """Возвращает информацию о здоровье системы памяти."""
        try:
            active_count = len(self.active_neurons)
            total_importance = sum(n.importance for n in self.active_neurons.values())
            avg_importance = total_importance / active_count if active_count > 0 else 0
            
            health_score = 100.0
            if active_count > 10000:
                health_score -= 20
            elif active_count > 5000:
                health_score -= 10
            
            status = "healthy" if health_score > 80 else "warning" if health_score > 50 else "critical"
            
            return {
                "health_score": health_score,
                "status": status,
                "active_neurons": active_count,
                "avg_importance": avg_importance,
                "memory_fields": len(self.memory_fields),
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Ошибка получения здоровья системы памяти: {e}")
            return {
                "health_score": 0,
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def close(self):
        """Закрывает ядро памяти."""
        if hasattr(self, 'database'):
            self.database.close()
        logger.info("MemoryCore закрыт")