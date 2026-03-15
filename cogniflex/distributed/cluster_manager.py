"""Модуль управления кластером для CogniFlex - обнаружение и управление узлами"""
import os
import logging
import time
import threading
import json
import socket
import requests
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import random
import hashlib
import sqlite3

logger = logging.getLogger("cogniflex.distributed.cluster")

class ClusterNode:
    """Представляет узел в кластере CogniFlex."""
    
    def __init__(self, node_id: str, address: str, port: int,
                 node_type: str = "worker", status: str = "online",
                 last_heartbeat: float = time.time(),
                 load: float = 0.0, capabilities: Dict[str, bool] = None):
        """
        Инициализирует узел кластера.
        
        Args:
            node_id: Уникальный идентификатор узла
            address: Адрес узла
            port: Порт узла
            node_type: Тип узла (worker, coordinator)
            status: Статус узла (online, offline, degraded)
            last_heartbeat: Время последнего сигнала
            load: Текущая нагрузка узла (0.0-1.0)
            capabilities: Возможности узла
        """
        self.node_id = node_id
        self.address = address
        self.port = port
        self.node_type = node_type
        self.status = status
        self.last_heartbeat = last_heartbeat
        self.load = load
        self.capabilities = capabilities or {}
        self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует узел в словарь."""
        return {
            "node_id": self.node_id,
            "address": self.address,
            "port": self.port,
            "node_type": self.node_type,
            "status": self.status,
            "last_heartbeat": self.last_heartbeat,
            "load": self.load,
            "capabilities": self.capabilities,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusterNode':
        """Создает узел из словаря."""
        return cls(
            node_id=data["node_id"],
            address=data["address"],
            port=data["port"],
            node_type=data["node_type"],
            status=data["status"],
            last_heartbeat=data["last_heartbeat"],
            load=data["load"],
            capabilities=data["capabilities"],
            metadata=data.get("metadata", {})
        )


class ClusterManager:
    """Менеджер кластера для CogniFlex."""
    
    def __init__(self, distributed_system=None, cache_dir: Optional[str] = None):
        """
        Инициализирует менеджер кластера.
        
        Args:
            distributed_system: Ссылка на распределенную систему
            cache_dir: Путь к директории кэша
        """
        self.distributed_system = distributed_system
        self.cache_dir = cache_dir
        self.initialized = False
        self.running = False
        self.stop_event = threading.Event()
        self.min_cluster_size = 1  # Уменьшаем минимальный размер кластера
        self.check_interval = 30  # Увеличиваем интервал проверки (сек)
        
        # Добавляем переменные для отслеживания времени последнего предупреждения
        self.last_scaling_warning = 0
        self.warning_interval = 1800  # Интервал предупреждений в секундах (30 минут)
        self.warning_suppressed = False
        
        # Создаем директорию кэша
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Путь к базе данных
        self.db_path = os.path.join(self.cache_dir, "cluster_manager.db") if self.cache_dir else None
        
        # Инициализируем компоненты
        self.nodes: Dict[str, ClusterNode] = {}
        self.node_lock = threading.RLock()
        
        # Настройки кластера
        self.cluster_settings = {
            "heartbeat_interval": 30.0,  # Интервал отправки сигнала (сек)
            "timeout": 30.0,            # Таймаут ожидания сигнала (сек)
            "coordinator_election": True,  # Разрешено ли выбор координатора
            "min_cluster_size": 1,      # Минимальный размер кластера
            "max_cluster_size": 10      # Максимальный размер кластера
        }
        
        # Инициализируем базу данных
        self._init_database()
        
        # Загружаем узлы
        self._load_nodes()
        
        self.initialized = True
        logger.info("ClusterManager инициализирован")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Возвращает соединение с базой данных для текущего потока."""
        if not hasattr(threading.current_thread(), "cluster_manager_connection"):
            # Создаем новое соединение для этого потока
            threading.current_thread().cluster_manager_connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False  # Разрешаем использование в разных потоках
            )
        return threading.current_thread().cluster_manager_connection
    
    def _init_database(self):
        """Инициализирует базу данных для менеджера кластера."""
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
            
            conn.commit()
        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных менеджера кластера: {e}", exc_info=True)
    
    def _load_nodes(self):
        """Загружает узлы из базы данных."""
        if not self.db_path:
            return
            
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM nodes")
            for row in cursor.fetchall():
                node = ClusterNode(
                    node_id=row[0],
                    address=row[1],
                    port=row[2],
                    node_type=row[3],
                    status=row[4],
                    last_heartbeat=row[5],
                    load=row[6],
                    capabilities=json.loads(row[7]) if row[7] else {}
                )
                self.nodes[node.node_id] = node
            
            logger.info(f"Загружено {len(self.nodes)} узлов кластера")
        except Exception as e:
            logger.error(f"Ошибка загрузки узлов: {e}")
    
    def _save_nodes(self):
        """Сохраняет узлы в базу данных."""
        if not self.db_path:
            return
            
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Очищаем таблицу
            cursor.execute("DELETE FROM nodes")
            
            # Сохраняем узлы
            for node in self.nodes.values():
                cursor.execute("""
                INSERT INTO nodes (node_id, address, port, node_type, status, last_heartbeat, load, capabilities)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    node.node_id,
                    node.address,
                    node.port,
                    node.node_type,
                    node.status,
                    node.last_heartbeat,
                    node.load,
                    json.dumps(node.capabilities)
                ))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Ошибка сохранения узлов: {e}")
    
    def start(self):
        """Запускает фоновые процессы менеджера кластера."""
        if self.running:
            return
            
        self.running = True
        self.stop_event.clear()
        
        # Запускаем фоновый процесс
        self.cluster_thread = threading.Thread(
            target=self._cluster_loop,
            name="ClusterManager",
            daemon=True
        )
        self.cluster_thread.start()
        
        logger.info("Фоновый процесс менеджера кластера запущен")
    
    def _cluster_loop(self):
        """Цикл обработки кластера."""
        # Увеличиваем интервал проверки до 5 секунд
        check_interval = 5.0
        
        while self.running and not self.stop_event.is_set():
            try:
                # Проверяем здоровье узлов
                self._check_node_health()
                
                # Проверяем необходимость выбора координатора
                self._check_coordinator()
                
                # Проверяем необходимость масштабирования
                self._check_scaling()
                
                # Небольшая пауза чтобы не перегружать CPU
                self.stop_event.wait(timeout=check_interval)
                
            except Exception as e:
                logger.error(f"Ошибка в фоновом цикле кластера: {e}", exc_info=True)
                time.sleep(5.0)
    
    def _check_node_health(self):
        """Проверяет здоровье узлов кластера."""
        current_time = time.time()
        timeout = self.cluster_settings["timeout"]
        
        with self.node_lock:
            for node_id, node in list(self.nodes.items()):
                # Проверяем таймаут
                if current_time - node.last_heartbeat > timeout:
                    if node.status != "offline":
                        logger.warning(f"Узел {node_id} перешел в статус offline")
                        node.status = "offline"
                
                # Проверяем нагрузку
                if node.load > 0.9:
                    if node.status != "degraded":
                        logger.warning(f"Узел {node_id} перешел в статус degraded (нагрузка: {node.load:.2f})")
                        node.status = "degraded"
                elif node.status == "degraded" and node.load < 0.7:
                    logger.info(f"Узел {node_id} восстановил работоспособность")
                    node.status = "online"
    
    def _check_coordinator(self):
        """Проверяет необходимость выбора координатора."""
        if not self.cluster_settings["coordinator_election"]:
            return
            
        # Проверяем, есть ли координатор
        coordinator = self.get_coordinator_node()
        if coordinator:
            return
            
        # Выбираем новый координатор
        online_nodes = self.get_online_nodes()
        if online_nodes:
            # Выбираем узел с наименьшей нагрузкой
            online_nodes.sort(key=lambda x: x.load)
            new_coordinator = online_nodes[0]
            
            # Обновляем статус
            new_coordinator.node_type = "coordinator"
            logger.info(f"Новый координатор выбран: {new_coordinator.node_id}")
            
            # Уведомляем другие узлы
            self._notify_nodes_of_coordinator_change(new_coordinator.node_id)
    
    def _notify_nodes_of_coordinator_change(self, coordinator_id: str):
        """Уведомляет узлы о смене координатора."""
        for node in self.get_online_nodes():
            try:
                url = f"http://{node.address}:{node.port}/api/cluster/coordinator_changed"
                requests.post(url, json={"new_coordinator": coordinator_id}, timeout=5.0)
            except Exception as e:
                logger.debug(f"Не удалось уведомить узел {node.node_id}: {e}")
    
    def _check_scaling(self):
        """Проверяет необходимость масштабирования кластера."""
        current_time = time.time()
        current_size = len(self.get_online_nodes())
        min_size = self.cluster_settings["min_cluster_size"]
        max_size = self.cluster_settings["max_cluster_size"]
        
        # Если кластер слишком мал
        if current_size < min_size:
            # Проверяем, прошло ли достаточно времени с последнего предупреждения
            if current_time - self.last_scaling_warning >= self.warning_interval:
                logger.info(f"Кластер слишком мал ({current_size}/{min_size}), требуется добавление узлов")
                self.last_scaling_warning = current_time
                self.warning_suppressed = False
            elif not self.warning_suppressed:
                logger.debug("Подавлено повторное предупреждение о недостатке узлов")
                self.warning_suppressed = True
        
        # Если кластер слишком велик
        elif current_size > max_size:
            # Аналогично для случая слишком большого кластера
            if current_time - self.last_scaling_warning >= self.warning_interval:
                logger.info(f"Кластер слишком велик ({current_size}/{max_size}), требуется удаление узлов")
                self.last_scaling_warning = current_time
                self.warning_suppressed = False
            elif not self.warning_suppressed:
                logger.debug("Подавлено повторное предупреждение о избытке узлов")
                self.warning_suppressed = True
    
    def stop(self):
        """Останавливает фоновые процессы менеджера кластера."""
        if not self.running:
            return
            
        self.running = False
        self.stop_event.set()
        
        # Ожидаем завершения фонового потока
        if hasattr(self, 'cluster_thread') and self.cluster_thread.is_alive():
            self.cluster_thread.join(timeout=5.0)
        
        # Сохраняем узлы
        self._save_nodes()
        
        logger.info("Фоновый процесс менеджера кластера остановлен")
    
    def register_node(self, node: ClusterNode) -> bool:
        """
        Регистрирует новый узел в кластере.
        
        Args:
            node: Узел для регистрации
            
        Returns:
            bool: Успешно ли зарегистрирован
        """
        with self.node_lock:
            if node.node_id in self.nodes:
                # Обновляем существующий узел
                existing_node = self.nodes[node.node_id]
                existing_node.address = node.address
                existing_node.port = node.port
                existing_node.node_type = node.node_type
                existing_node.status = node.status
                existing_node.last_heartbeat = node.last_heartbeat
                existing_node.load = node.load
                existing_node.capabilities = node.capabilities
                logger.debug(f"Обновлен узел: {node.node_id}")
            else:
                # Добавляем новый узел
                self.nodes[node.node_id] = node
                logger.info(f"Добавлен новый узел: {node.node_id}")
                
                # Проверяем, нужно ли выбрать координатора
                if self.cluster_settings["coordinator_election"] and node.node_type == "coordinator":
                    self._check_coordinator()
            
            # Сохраняем в базу данных
            self._save_nodes()
        
        return True
    
    def heartbeat(self, node_id: str) -> bool:
        """
        Обрабатывает сигнал от узла.
        
        Args:
            node_id: Идентификатор узла
            
        Returns:
            bool: Успешно ли обработан
        """
        with self.node_lock:
            if node_id not in self.nodes:
                logger.warning(f"Сигнал от неизвестного узла: {node_id}")
                return False
            
            node = self.nodes[node_id]
            node.last_heartbeat = time.time()
            
            if node.status == "offline":
                node.status = "online"
                logger.info(f"Узел {node_id} восстановил связь")
            
            # Сохраняем в базу данных
            self._save_nodes()
        
        return True
    
    def get_node(self, node_id: str) -> Optional[ClusterNode]:
        """
        Возвращает узел по идентификатору.
        
        Args:
            node_id: Идентификатор узла
            
        Returns:
            Optional[ClusterNode]: Узел или None
        """
        with self.node_lock:
            return self.nodes.get(node_id)
    
    def get_online_nodes(self) -> List[ClusterNode]:
        """
        Возвращает список онлайн-узлов.
        
        Returns:
            List[ClusterNode]: Список онлайн-узлов
        """
        with self.node_lock:
            return [node for node in self.nodes.values() if node.status == "online"]
    
    def get_coordinator_node(self) -> Optional[ClusterNode]:
        """
        Возвращает координатора кластера.
        
        Returns:
            Optional[ClusterNode]: Координатор или None
        """
        with self.node_lock:
            for node in self.nodes.values():
                if node.node_type == "coordinator" and node.status == "online":
                    return node
            return None
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """
        Возвращает статус кластера.
        
        Returns:
            Dict[str, Any]: Статус кластера
        """
        with self.node_lock:
            online_nodes = self.get_online_nodes()
            coordinator = self.get_coordinator_node()
            
            return {
                "status": "healthy" if coordinator and len(online_nodes) >= self.cluster_settings["min_cluster_size"] else "degraded",
                "node_count": len(self.nodes),
                "online_count": len(online_nodes),
                "coordinator_id": coordinator.node_id if coordinator else None,
                "is_coordinator": self.distributed_system.is_coordinator() if self.distributed_system else False,
                "timestamp": time.time()
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Возвращает отчет о здоровье системы.
        
        Returns:
            Dict: Отчет о здоровье
        """
        cluster_status = self.get_cluster_status()
        
        # Рассчитываем общий показатель здоровья
        health_score = 100.0
        
        # Учитываем количество онлайн-узлов
        online_ratio = cluster_status["online_count"] / cluster_status["node_count"] if cluster_status["node_count"] > 0 else 0
        if online_ratio < 0.7:
            health_score -= min(40, (0.7 - online_ratio) * 100)
        elif online_ratio < 0.9:
            health_score -= min(15, (0.9 - online_ratio) * 50)
        
        # Учитываем наличие координатора
        if not cluster_status["coordinator_id"]:
            health_score -= 20
        
        # Анализируем проблемы
        problem_areas = []
        if online_ratio < 0.7:
            problem_areas.append("Низкое количество активных узлов")
        
        if not cluster_status["coordinator_id"]:
            problem_areas.append("Отсутствует координатор кластера")
        
        # Формируем рекомендации
        recommendations = []
        if online_ratio < 0.7:
            recommendations.append(
                f"Только {cluster_status['online_count']}/{cluster_status['node_count']} узлов активны. "
                "Проверьте состояние узлов и сеть."
            )
        
        if not cluster_status["coordinator_id"]:
            recommendations.append(
                "Координатор кластера отсутствует. Проверьте работу узлов и запустите выбор координатора."
            )
        
        if not recommendations:
            recommendations.append(
                "Кластер работает стабильно. Продолжайте мониторинг для "
                "раннего выявления потенциальных проблем."
            )
        
        return {
            "health_score": max(0, min(100, health_score)),
            "cluster_status": cluster_status,
            "problem_areas": problem_areas,
            "recommendations": recommendations,
            "timestamp": time.time()
        }
    
    def close(self):
        """Закрывает менеджер кластера и освобождает ресурсы."""
        self.stop()
        
        # Закрываем соединение с БД
        if hasattr(threading.current_thread(), "cluster_manager_connection"):
            try:
                threading.current_thread().cluster_manager_connection.close()
                delattr(threading.current_thread(), "cluster_manager_connection")
            except Exception as e:
                logger.error(f"Ошибка закрытия соединения с БД: {e}")
        
        logger.info("ClusterManager закрыт")