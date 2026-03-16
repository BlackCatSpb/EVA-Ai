"""Основной модуль распределенной системы для CogniFlex"""
import os
import logging
import time
import threading
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
import sqlite3
import requests
from collections import defaultdict

logger = logging.getLogger("cogniflex.distributed.core")

class DistributedSystem:
    """Распределенная система для CogniFlex, обеспечивающая масштабируемость и отказоустойчивость."""
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None, cluster_manager=None):
        """
        Инициализирует распределенную систему.
        
        Args:
            brain: Ссылка на ядро CogniFlex
            cache_dir: Путь к директории кэша
            cluster_manager: Ссылка на менеджер кластера
        """
        self.brain = brain
        self.cache_dir = cache_dir
        self.cluster_manager = cluster_manager
        self.initialized = False
        self.running = False
        self.stop_event = threading.Event()
        
        # Создаем директорию кэша
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Путь к базе данных
        self.db_path = os.path.join(self.cache_dir, "distributed_system.db") if self.cache_dir else None
        
        # Загружаем конфигурацию
        self.config = self._load_config()
        
        # Инициализируем компоненты
        self.task_scheduler = None
        self.knowledge_sync = None
        self.fault_manager = None
        
        # Создаем соединение с БД
        self._init_database()
        
        # Инициализируем дополнительные компоненты
        self._init_components()
        
        self.initialized = True
        logger.info("DistributedSystem инициализирован")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Возвращает соединение с базой данных для текущего потока."""
        if not hasattr(threading.current_thread(), "distributed_system_connection"):
            # Создаем новое соединение для этого потока
            threading.current_thread().distributed_system_connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False  # Разрешаем использование в разных потоках
            )
        return threading.current_thread().distributed_system_connection
    
    def _init_database(self):
        """Инициализирует базу данных для распределенной системы."""
        if not self.db_path:
            return
            
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Таблица узлов
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                address TEXT NOT NULL,
                port INTEGER NOT NULL,
                node_type TEXT NOT NULL,
                status TEXT NOT NULL,
                last_heartbeat REAL NOT NULL,
                load REAL DEFAULT 0.0,
                capabilities TEXT DEFAULT '{}'
            )
            """)
            
            # Таблица задач
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                priority REAL NOT NULL,
                timeout REAL NOT NULL,
                created_at REAL NOT NULL,
                status TEXT NOT NULL,
                result TEXT,
                error TEXT,
                metadata TEXT,
                assigned_node TEXT,
                started_at REAL,
                completed_at REAL
            )
            """)
            
            # Таблица статистики
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_stats (
                id INTEGER PRIMARY KEY,
                total_tasks INTEGER DEFAULT 0,
                completed_tasks INTEGER DEFAULT 0,
                failed_tasks INTEGER DEFAULT 0,
                avg_processing_time REAL DEFAULT 0.0,
                last_update DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Инициализируем статистику, если таблица пуста
            cursor.execute("SELECT COUNT(*) FROM system_stats")
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                INSERT INTO system_stats (total_tasks, completed_tasks, failed_tasks, avg_processing_time)
                VALUES (0, 0, 0, 0.0)
                """)
            
            conn.commit()
        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных распределенной системы: {e}", exc_info=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию распределенной системы."""
        config_path = os.path.join(self.cache_dir, "distributed_config.json") if self.cache_dir else None
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Ошибка загрузки конфигурации: {e}")
        
        # Конфигурация по умолчанию
        return {
            "node_id": f"node_{int(time.time())}",
            "cluster_size": 1,
            "is_coordinator": True,
            "heartbeat_interval": 5.0,
            "timeout": 30.0,
            "max_retries": 3
        }
    
    def _init_components(self):
        """Инициализирует компоненты распределенной системы."""
        try:
            # Импортируем компоненты из текущего пакета
            from .distributed_task_scheduler import TaskScheduler
            self.task_scheduler = TaskScheduler(brain=self.brain, cache_dir=self.cache_dir)
            logger.info("TaskScheduler инициализирован")
        except ImportError as e:
            logger.warning(f"TaskScheduler недоступен: {e}")
        
        try:
            from .knowledge_sync import KnowledgeSync
            self.knowledge_sync = KnowledgeSync(brain=self.brain, cache_dir=self.cache_dir)
            logger.info("KnowledgeSync инициализирован")
        except ImportError as e:
            logger.warning(f"KnowledgeSync недоступен: {e}")
        
        try:
            from .distributed_recovery_manager import RecoveryManager
            self.fault_manager = RecoveryManager(brain=self.brain, cache_dir=self.cache_dir)
            logger.info("RecoveryManager инициализирован")
        except ImportError as e:
            logger.warning(f"RecoveryManager недоступен: {e}")
    
    def start(self):
        """Запускает распределенную систему."""
        if self.running:
            logger.warning("DistributedSystem уже запущен")
            return
            
        self.running = True
        self.stop_event.clear()
        
        # Запускаем фоновые процессы
        self._start_background_processes()
        
        # Запускаем компоненты
        if self.task_scheduler:
            self.task_scheduler.start()
        
        if self.knowledge_sync:
            self.knowledge_sync.start()
        
        if self.fault_manager:
            self.fault_manager.start()
        
        logger.info("DistributedSystem запущен")
    
    def _start_background_processes(self):
        """Запускает фоновые процессы распределенной системы."""
        try:
            import threading
            
            # Запускаем поток мониторинга статуса
            self._status_monitor_thread = threading.Thread(
                target=self._monitor_system_status,
                daemon=True,
                name="DistributedSystemStatusMonitor"
            )
            self._status_monitor_thread.start()
            logger.debug("Фоновые процессы DistributedSystem запущены")
        except Exception as e:
            logger.error(f"Ошибка запуска фоновых процессов: {e}")
    
    def _monitor_system_status(self):
        """Мониторинг статуса системы в фоне."""
        while self.running and not self.stop_event.is_set():
            try:
                # Обновляем статистику каждые 30 секунд
                self.stop_event.wait(30)
                if not self.stop_event.is_set():
                    self._update_system_stats()
            except Exception as e:
                logger.error(f"Ошибка мониторинга статуса: {e}")
    
    def _update_system_stats(self):
        """Обновляет статистику системы."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO system_stats (timestamp, active_nodes, task_count, completion_rate)
                VALUES (?, ?, ?, ?)
            ''', (time.time(), len(self.get_active_nodes()), 
                  self.task_scheduler.get_pending_count() if self.task_scheduler else 0, 0.0))
            conn.commit()
        except Exception as e:
            logger.error(f"Ошибка обновления статистики: {e}")
    
    def stop(self):
        """Останавливает распределенную систему."""
        if not self.running:
            logger.warning("DistributedSystem уже остановлен")
            return
            
        self.running = False
        self.stop_event.set()
        
        # Останавливаем компоненты
        if self.task_scheduler:
            self.task_scheduler.stop()
        
        if self.knowledge_sync:
            self.knowledge_sync.stop()
        
        if self.fault_manager:
            self.fault_manager.stop()
        
        # Останавливаем фоновые процессы
        self._stop_background_processes()
        
        # Закрываем соединение с БД
        if hasattr(threading.current_thread(), "distributed_system_connection"):
            try:
                threading.current_thread().distributed_system_connection.close()
                delattr(threading.current_thread(), "distributed_system_connection")
            except Exception as e:
                logger.error(f"Ошибка закрытия соединения с БД: {e}")
        
        logger.info("DistributedSystem остановлен")
    
    def _stop_background_processes(self):
        """Останавливает фоновые процессы распределенной системы."""
        try:
            self.stop_event.set()
            
            # Ожидаем завершения потоков
            if hasattr(self, '_status_monitor_thread'):
                self._status_monitor_thread.join(timeout=2.0)
                
            logger.debug("Фоновые процессы DistributedSystem остановлены")
        except Exception as e:
            logger.error(f"Ошибка остановки фоновых процессов: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Возвращает статус распределенной системы."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Получаем статистику
            cursor.execute("SELECT * FROM system_stats ORDER BY id DESC LIMIT 1")
            stats = cursor.fetchone()
            
            system_status = {
                "status": "running" if self.running else "stopped",
                "node_id": self.config.get("node_id", "unknown"),
                "is_coordinator": self.config.get("is_coordinator", False),
                "cluster_size": self.config.get("cluster_size", 1),
                "timestamp": datetime.now().isoformat()
            }
            
            if stats:
                system_status.update({
                    "total_tasks": stats[1],
                    "completed_tasks": stats[2],
                    "failed_tasks": stats[3],
                    "avg_processing_time": stats[4],
                    "last_update": stats[5]
                })
            
            # Добавляем статус компонентов
            system_status["task_scheduler"] = self.task_scheduler.get_scheduler_health_report() if self.task_scheduler else None
            system_status["knowledge_sync"] = self.knowledge_sync.get_system_health() if self.knowledge_sync else None
            system_status["fault_manager"] = self.fault_manager.get_system_health() if self.fault_manager else None
            
            return system_status
        except Exception as e:
            logger.error(f"Ошибка получения статуса системы: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def is_coordinator(self) -> bool:
        """Проверяет, является ли этот узел координатором."""
        return self.config.get("is_coordinator", False)
    
    def join_cluster(self, coordinator_address: str) -> bool:
        """
        Присоединяется к существующему кластеру.
        
        Args:
            coordinator_address: Адрес координатора
            
        Returns:
            bool: Успешно ли присоединение
        """
        try:
            # Отправляем запрос координатору
            response = requests.post(
                f"http://{coordinator_address}/api/cluster/join",
                json={
                    "node_id": self.config["node_id"],
                    "address": self.config.get("address", "localhost"),
                    "port": self.config.get("port", 8000),
                    "capabilities": self.config.get("capabilities", {})
                },
                timeout=10.0
            )
            
            if response.status_code == 200:
                # Обновляем конфигурацию
                cluster_info = response.json()
                self.config.update({
                    "cluster_size": cluster_info["cluster_size"],
                    "is_coordinator": False
                })
                logger.info(f"Присоединение к кластеру успешно. Размер кластера: {cluster_info['cluster_size']}")
                return True
            else:
                logger.error(f"Ошибка присоединения к кластеру: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Ошибка присоединения к кластеру: {e}")
            return False
    
    def leave_cluster(self) -> bool:
        """
        Покидает кластер.
        
        Returns:
            bool: Успешно ли покинут кластер
        """
        if not self.cluster_manager or not self.is_coordinator:
            return False
            
        try:
            # Отправляем запрос координатору
            coordinator = self.cluster_manager.get_coordinator_node()
            if not coordinator:
                return False
                
            response = requests.post(
                f"http://{coordinator.address}:{coordinator.port}/api/cluster/leave",
                json={"node_id": self.config["node_id"]},
                timeout=10.0
            )
            
            if response.status_code == 200:
                # Обновляем конфигурацию
                self.config["cluster_size"] = max(1, self.config["cluster_size"] - 1)
                self.config["is_coordinator"] = self.config["cluster_size"] == 1
                logger.info("Покидание кластера успешно")
                return True
            else:
                logger.error(f"Ошибка покидания кластера: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Ошибка покидания кластера: {e}")
            return False
    
    def get_node_status(self) -> Dict[str, Any]:
        """Возвращает статус текущего узла."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Получаем информацию об этом узле
            cursor.execute("""
            SELECT node_id, address, port, node_type, status, last_heartbeat, load, capabilities
            FROM nodes
            WHERE node_id = ?
            """, (self.config["node_id"],))
            
            node_data = cursor.fetchone()
            if node_data:
                return {
                    "node_id": node_data[0],
                    "address": node_data[1],
                    "port": node_data[2],
                    "node_type": node_data[3],
                    "status": node_data[4],
                    "last_heartbeat": node_data[5],
                    "load": node_data[6],
                    "capabilities": json.loads(node_data[7]) if node_data[7] else {},
                    "is_coordinator": self.is_coordinator()
                }
            else:
                return {
                    "node_id": self.config["node_id"],
                    "status": "unknown",
                    "is_coordinator": self.is_coordinator()
                }
        except Exception as e:
            logger.error(f"Ошибка получения статуса узла: {e}")
            return {
                "node_id": self.config["node_id"],
                "status": "error",
                "error": str(e),
                "is_coordinator": self.is_coordinator()
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Возвращает отчет о здоровье системы."""
        try:
            # Получаем статус кластера
            cluster_health = self.get_cluster_health()
            
            # Получаем статус компонентов
            task_scheduler_health = self.task_scheduler.get_scheduler_health_report() if self.task_scheduler else None
            knowledge_sync_health = self.knowledge_sync.get_system_health() if self.knowledge_sync else None
            fault_manager_health = self.fault_manager.get_system_health() if self.fault_manager else None
            
            # Рассчитываем общий показатель здоровья
            health_score = 100.0
            
            # Учитываем здоровье кластера
            health_score = min(health_score, cluster_health.get("health_score", 100))
            
            # Учитываем здоровье компонентов
            if task_scheduler_health:
                health_score = min(health_score, task_scheduler_health.get("health_score", 100))
            
            if knowledge_sync_health:
                health_score = min(health_score, knowledge_sync_health.get("health_score", 100))
            
            if fault_manager_health:
                health_score = min(health_score, fault_manager_health.get("health_score", 100))
            
            # Формируем список проблем
            problem_areas = []
            if cluster_health.get("health_score", 100) < 80:
                problem_areas.append("Проблемы с кластером")
            
            if task_scheduler_health and task_scheduler_health.get("health_score", 100) < 80:
                problem_areas.append("Проблемы с планировщиком задач")
            
            if knowledge_sync_health and knowledge_sync_health.get("health_score", 100) < 80:
                problem_areas.append("Проблемы с синхронизацией знаний")
            
            # Формируем рекомендации
            recommendations = []
            if not problem_areas:
                recommendations.append("Система работает стабильно")
            else:
                recommendations.append("Проверьте компоненты с низким уровнем здоровья")
            
            return {
                "health_score": max(0, min(100, health_score)),
                "cluster_health": cluster_health,
                "task_scheduler_health": task_scheduler_health,
                "knowledge_sync_health": knowledge_sync_health,
                "fault_manager_health": fault_manager_health,
                "problem_areas": problem_areas,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Ошибка получения информации о здоровье системы: {e}")
            return {
                "health_score": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_cluster_health(self) -> Dict[str, Any]:
        """Возвращает информацию о здоровье кластера."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Получаем все узлы
            cursor.execute("SELECT status, load FROM nodes")
            nodes = cursor.fetchall()
            
            # Анализируем здоровье
            total_nodes = len(nodes)
            online_nodes = sum(1 for node in nodes if node[0] == "online")
            avg_load = sum(node[1] for node in nodes) / total_nodes if total_nodes > 0 else 0
            
            # Рассчитываем общий показатель здоровья
            health_score = 100.0
            
            # Учитываем долю онлайн-узлов
            online_ratio = online_nodes / total_nodes if total_nodes > 0 else 0
            if online_ratio < 0.7:
                health_score -= min(40, (0.7 - online_ratio) * 100)
            elif online_ratio < 0.9:
                health_score -= min(15, (0.9 - online_ratio) * 50)
            
            # Учитываем нагрузку
            if avg_load > 0.7:
                health_score -= min(20, (avg_load - 0.7) * 50)
            
            return {
                "health_score": max(0, min(100, health_score)),
                "total_nodes": total_nodes,
                "online_nodes": online_nodes,
                "offline_nodes": total_nodes - online_nodes,
                "avg_load": avg_load,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Ошибка получения информации о здоровье кластера: {e}")
            return {
                "health_score": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def close(self):
        """Закрывает распределенную систему и освобождает ресурсы."""
        self.stop()
        logger.info("DistributedSystem закрыт")