"""Модуль синхронизации знаний для CogniFlex - синхронизация графа знаний между узлами"""
import os
import logging
import time
import threading
import json
import requests
import random
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import sqlite3
from collections import defaultdict

logger = logging.getLogger("cogniflex.distributed.knowledge")

class KnowledgeSync:
    """Синхронизирует знания между узлами распределенной системы."""
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None):
        """
        Инициализирует синхронизацию знаний.
        
        Args:
            brain: Ссылка на ядро CogniFlex
            cache_dir: Путь к директории кэша
        """
        self.brain = brain
        self.cache_dir = cache_dir
        self.active = False
        self.sync_thread = None
        self.stop_event = threading.Event()
        
        # Получаем доступ к распределенной системе и менеджеру кластера
        self.distributed_system = brain.distributed_system if brain else None
        self.cluster_manager = brain.cluster_manager if brain else None
        
        # Создаем директорию кэша если нужно
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Настройки синхронизации
        self.sync_settings = {
            "sync_interval": 300.0,  # Интервал синхронизации (сек) - 5 минут
            "warning_interval": 1800.0,  # Интервал для повторных предупреждений (сек) - 30 минут
            "sync_mode": "pull",      # pull, push, hybrid
            "batch_size": 100,        # Размер пакета для синхронизации
            "max_retries": 3,         # Максимальное количество попыток
            "retry_delay": 5.0,       # Задержка между попытками (сек)
            "timeout": 30.0,          # Таймаут синхронизации (сек)
            "enabled": True           # Включена ли синхронизация
        }
        
        # Статистика
        self.sync_stats = {
            "total_syncs": 0,
            "successful_syncs": 0,
            "failed_syncs": 0,
            "last_sync": 0.0,
            "last_warning": 0.0,
            "avg_duration": 0.0
        }
        
        # Флаг для предотвращения параллельных синхронизаций
        self.is_syncing = False
        self.sync_lock = threading.Lock()
        
        # Отслеживание времени последней проверки
        self.last_check_time = 0
        self.check_interval = 10.0  # Интервал проверки в секундах
        
        logger.info("Синхронизация знаний инициализирована")
    
    def start(self):
        """Запускает фоновые процессы синхронизации знаний."""
        if self.active or not self.sync_settings["enabled"]:
            return
        
        self.active = True
        self.stop_event.clear()
        self.sync_thread = threading.Thread(
            target=self._sync_loop,
            name="KnowledgeSync",
            daemon=True
        )
        self.sync_thread.start()
        logger.info("Фоновый процесс синхронизации знаний запущен")
    
    def stop(self):
        """Остановить фоновые процессы синхронизации знаний."""
        if not self.active:
            return
        
        self.active = False
        self.stop_event.set()
        
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=5.0)
        
        logger.info("Фоновый процесс синхронизации знаний остановлен")
    
    def _sync_loop(self):
        """Цикл синхронизации знаний."""
        while self.active and not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # Проверяем, нужно ли синхронизировать (с учетом интервала проверки)
                if current_time - self.last_check_time >= self.check_interval:
                    self.last_check_time = current_time
                    
                    # Проверяем, нужно ли синхронизировать
                    if current_time - self.sync_stats["last_sync"] >= self.sync_settings["sync_interval"]:
                        self.sync_knowledge()
                
                # Проверяем здоровье системы
                self._check_system_health()
                
                # Ждем перед следующей проверкой
                self.stop_event.wait(timeout=1.0)
                
            except Exception as e:
                logger.error(f"Ошибка в цикле синхронизации знаний: {e}", exc_info=True)
                time.sleep(5)
    
    def sync_knowledge(self) -> bool:
        """
        Синхронизирует знания с другими узлами кластера.
        
        Returns:
            bool: Успешно ли синхронизировано
        """
        # Проверяем, не выполняется ли уже синхронизация
        with self.sync_lock:
            if self.is_syncing:
                logger.debug("Синхронизация уже выполняется, пропускаем новый запуск")
                return False
            self.is_syncing = True
        
        try:
            start_time = time.time()
            
            # Проверяем наличие графа знаний
            if not hasattr(self.brain, 'knowledge_graph'):
                logger.warning("Граф знаний недоступен для синхронизации")
                return False
            
            # Получаем активные узлы
            if not self.cluster_manager:
                logger.warning("ClusterManager недоступен для синхронизации знаний")
                return False
                
            active_nodes = self.cluster_manager.get_online_nodes()
            current_cluster_size = len(active_nodes)
            
            # Проверяем, нужно ли выводить предупреждение
            if not active_nodes:
                # Проверяем, прошло ли достаточно времени с последнего предупреждения
                if time.time() - self.sync_stats["last_warning"] >= self.sync_settings["warning_interval"]:
                    logger.warning("Нет активных узлов для синхронизации знаний")
                    self.sync_stats["last_warning"] = time.time()
                return False
            
            # Выбираем режим синхронизации
            if self.sync_settings["sync_mode"] == "pull":
                success = self._pull_sync(active_nodes)
            elif self.sync_settings["sync_mode"] == "push":
                success = self._push_sync(active_nodes)
            else:  # hybrid
                success = self._hybrid_sync(active_nodes)
            
            # Обновляем статистику
            self.sync_stats["total_syncs"] += 1
            if success:
                self.sync_stats["successful_syncs"] += 1
                self.sync_stats["last_sync"] = time.time()
            else:
                self.sync_stats["failed_syncs"] += 1
            
            # Обновляем среднее время
            duration = time.time() - start_time
            if self.sync_stats["successful_syncs"] > 0:
                self.sync_stats["avg_duration"] = (
                    (self.sync_stats["avg_duration"] * (self.sync_stats["successful_syncs"] - 1) + duration) / 
                    self.sync_stats["successful_syncs"]
                )
            else:
                self.sync_stats["avg_duration"] = duration
            
            status = "успешно" if success else "не успешно"
            logger.info(f"Синхронизация знаний завершена {status} за {duration:.2f} сек")
            return success
            
        except Exception as e:
            logger.error(f"Ошибка синхронизации знаний: {e}", exc_info=True)
            self.sync_stats["failed_syncs"] += 1
            return False
        finally:
            # Сбрасываем флаг синхронизации
            with self.sync_lock:
                self.is_syncing = False
    
    def _pull_sync(self, active_nodes: List) -> bool:
        """
        Выполняет синхронизацию в режиме pull (локальный узел запрашивает данные).
        
        Args:
            active_nodes: Список активных узлов
            
        Returns:
            bool: Успешно ли синхронизировано
        """
        # Выбираем случайный узел для синхронизации
        target_node = random.choice(active_nodes)
        logger.debug(f"Синхронизация знаний в режиме pull с узлом {target_node.node_id}")
        
        try:
            # Получаем данные для синхронизации
            sync_data = self._get_sync_data()
            
            # Формируем запрос
            url = f"http://{target_node.address}:{target_node.port}/api/knowledge/sync"
            headers = {"Content-Type": "application/json"}
            payload = {
                "source_node": self.cluster_manager.get_node_id() if self.cluster_manager else "unknown",
                "last_sync": target_node.last_heartbeat if hasattr(target_node, 'last_heartbeat') else time.time(),
                "capabilities": self.cluster_manager.get_node_capabilities() if self.cluster_manager else {}
            }
            
            # Отправляем запрос
            response = requests.post(
                url, 
                json=payload, 
                headers=headers, 
                timeout=self.sync_settings["timeout"]
            )
            
            if response.status_code == 200:
                update_data = response.json()
                
                # Применяем обновления
                if "updates" in update_data:
                    self._apply_updates(update_data["updates"])
                
                # Обновляем время последней синхронизации
                if self.cluster_manager:
                    self.cluster_manager.update_node_heartbeat(target_node.node_id)
                
                return True
            else:
                logger.error(f"Ошибка синхронизации: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка pull-синхронизации: {e}", exc_info=True)
            if self.cluster_manager:
                self.cluster_manager.remove_node(target_node.node_id)
            return False
    
    def _push_sync(self, active_nodes: List) -> bool:
        """
        Выполняет синхронизацию в режиме push (локальный узел отправляет данные).
        
        Args:
            active_nodes: Список активных узлов
            
        Returns:
            bool: Успешно ли синхронизировано
        """
        success_count = 0
        total_count = 0
        
        for node in active_nodes:
            if node.node_id == (self.cluster_manager.get_node_id() if self.cluster_manager else ""):
                continue
            
            total_count += 1
            logger.debug(f"Синхронизация знаний в режиме push с узлом {node.node_id}")
            
            try:
                # Получаем данные для синхронизации
                sync_data = self._get_sync_data()
                
                # Формируем запрос
                url = f"http://{node.address}:{node.port}/api/knowledge/update"
                headers = {"Content-Type": "application/json"}
                payload = {
                    "source_node": self.cluster_manager.get_node_id() if self.cluster_manager else "unknown",
                    "updates": sync_data["updates"]
                }
                
                # Отправляем запрос
                response = requests.post(
                    url, 
                    json=payload, 
                    headers=headers, 
                    timeout=self.sync_settings["timeout"]
                )
                
                if response.status_code == 200:
                    # Обновляем время последней синхронизации
                    if self.cluster_manager:
                        self.cluster_manager.update_node_heartbeat(node.node_id)
                    success_count += 1
                else:
                    logger.error(f"Ошибка синхронизации с {node.node_id}: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Ошибка push-синхронизации с {node.node_id}: {e}", exc_info=True)
                if self.cluster_manager:
                    self.cluster_manager.remove_node(node.node_id)
        
        # Возвращаем успех, если синхронизация с большинством узлов прошла успешно
        return success_count >= max(1, total_count // 2)
    
    def _hybrid_sync(self, active_nodes: List) -> bool:
        """
        Выполняет синхронизацию в гибридном режиме.
        
        Args:
            active_nodes: Список активных узлов
            
        Returns:
            bool: Успешно ли синхронизировано
        """
        # Если мы координатор, используем push
        if self.cluster_manager and self.cluster_manager.get_coordinator_node() and \
           self.cluster_manager.get_coordinator_node().node_id == (self.cluster_manager.get_node_id() if self.cluster_manager else ""):
            return self._push_sync(active_nodes)
        # Иначе используем pull
        else:
            return self._pull_sync(active_nodes)
    
    def _get_sync_data(self) -> Dict[str, Any]:
        """
        Получает данные для синхронизации.
        
        Returns:
            Dict[str, Any]: Данные для синхронизации
        """
        # Получаем обновления из графа знаний
        updates = []
        if hasattr(self.brain, 'knowledge_graph') and hasattr(self.brain.knowledge_graph, 'get_recent_updates'):
            updates = self.brain.knowledge_graph.get_recent_updates()
        
        return {
            "node_id": self.cluster_manager.get_node_id() if self.cluster_manager else "unknown",
            "timestamp": time.time(),
            "updates": updates,
            "stats": self.brain.knowledge_graph.get_statistics() if hasattr(self.brain, 'knowledge_graph') else {}
        }
    
    def _apply_updates(self, updates: List[Dict[str, Any]]):
        """
        Применяет обновления к локальному графу знаний.
        
        Args:
            updates: Список обновлений
        """
        if not updates or not hasattr(self.brain, 'knowledge_graph'):
            return
        
        logger.debug(f"Применение {len(updates)} обновлений к графу знаний")
        
        # Группируем обновления по типу
        node_updates = []
        edge_updates = []
        
        for update in updates:
            if update.get("type") == "node":
                node_updates.append(update)
            elif update.get("type") == "edge":
                edge_updates.append(update)
        
        # Применяем обновления узлов
        if node_updates and hasattr(self.brain.knowledge_graph, 'batch_update_nodes'):
            self.brain.knowledge_graph.batch_update_nodes(node_updates)
        
        # Применяем обновления связей
        if edge_updates and hasattr(self.brain.knowledge_graph, 'batch_update_edges'):
            self.brain.knowledge_graph.batch_update_edges(edge_updates)
    
    def _check_system_health(self):
        """Проверяет здоровье системы синхронизации."""
        # Проверяем размер кластера
        if self.cluster_manager:
            try:
                online_nodes = self.cluster_manager.get_online_nodes()
                if len(online_nodes) == 0:
                    # Проверяем, нужно ли вывести предупреждение
                    if time.time() - self.sync_stats["last_warning"] >= self.sync_settings["warning_interval"]:
                        logger.warning("Кластер слишком мал (0/1), требуется добавление узлов")
                        self.sync_stats["last_warning"] = time.time()
                elif len(online_nodes) < self.cluster_manager.min_cluster_size:
                    # Проверяем, нужно ли вывести предупреждение
                    if time.time() - self.sync_stats["last_warning"] >= self.sync_settings["warning_interval"]:
                        logger.warning(f"Кластер слишком мал ({len(online_nodes)}/{self.cluster_manager.min_cluster_size}), требуется добавление узлов")
                        self.sync_stats["last_warning"] = time.time()
            except Exception as e:
                logger.debug(f"Не удалось проверить размер кластера: {e}")
        
        # Логируем состояние системы раз в 5 минут
        if time.time() - self.sync_stats.get("last_health_check", 0) >= 300.0:
            health = self.get_system_health()
            
            # Логируем состояние системы
            logger.info(
                f"Состояние синхронизации знаний: всего {health['stats']['total_syncs']}, "
                f"успешных {health['stats']['successful_syncs']}, "
                f"неудачных {health['stats']['failed_syncs']}, "
                f"здоровье: {health['health_score']:.2f}/100"
            )
            
            self.sync_stats["last_health_check"] = time.time()
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Возвращает отчет о здоровье системы синхронизации.
        
        Returns:
            Dict: Отчет о здоровье
        """
        # Рассчитываем общий показатель здоровья
        health_score = 100.0
        
        # Учитываем успешность синхронизации
        success_rate = (
            self.sync_stats["successful_syncs"] / self.sync_stats["total_syncs"] 
            if self.sync_stats["total_syncs"] > 0 else 0.0
        )
        if success_rate < 0.7:
            health_score -= min(40, (0.7 - success_rate) * 100)
        elif success_rate < 0.85:
            health_score -= min(15, (0.85 - success_rate) * 50)
        
        # Учитываем время последней синхронизации
        time_since_last = time.time() - self.sync_stats["last_sync"] if self.sync_stats["last_sync"] > 0 else float('inf')
        if time_since_last > self.sync_settings["sync_interval"] * 2:
            health_score -= min(20, (time_since_last - self.sync_settings["sync_interval"]) / self.sync_settings["sync_interval"] * 10)
        
        # Анализируем проблемы
        problem_areas = []
        if success_rate < 0.7:
            problem_areas.append("Низкий процент успешных синхронизаций")
        
        if time_since_last > self.sync_settings["sync_interval"] * 1.5:
            problem_areas.append("Длительный период без синхронизации")
        
        # Формируем рекомендации
        recommendations = []
        if success_rate < 0.7:
            recommendations.append(
                "Низкий процент успешных синхронизаций. Проверьте соединение с другими узлами "
                "и работоспособность API синхронизации."
            )
        
        if time_since_last > self.sync_settings["sync_interval"] * 1.5:
            recommendations.append(
                f"Синхронизация не выполнялась более {time_since_last:.2f} секунд. "
                "Проверьте настройки интервала синхронизации."
            )
        
        if not recommendations:
            recommendations.append(
                "Система синхронизации знаний работает стабильно. Продолжайте мониторинг для "
                "раннего выявления потенциальных проблем."
            )
        
        return {
            "health_score": max(0, min(100, health_score)),
            "stats": self.sync_stats.copy(),
            "problem_areas": problem_areas,
            "recommendations": recommendations,
            "timestamp": time.time()
        }
    
    def get_system_summary(self) -> str:
        """
        Возвращает краткую сводку о системе синхронизации.
        
        Returns:
            str: Сводка о состоянии
        """
        health = self.get_system_health()
        
        summary = (
            f"Синхронизация знаний\n"
            f"{'=' * 30}\n\n"
            f"Синхронизации: всего {health['stats']['total_syncs']}, "
            f"успешных {health['stats']['successful_syncs']}, "
            f"неудачных {health['stats']['failed_syncs']}\n"
            f"Последняя: {datetime.fromtimestamp(health['stats']['last_sync']).strftime('%Y-%m-%d %H:%M:%S') if health['stats']['last_sync'] > 0 else 'никогда'}\n"
            f"Здоровье: {health['health_score']:.2f}/100\n"
        )
        
        if health["health_score"] < 70:
            summary += "\nРекомендации:\n"
            for i, rec in enumerate(health["recommendations"][:2], 1):
                summary += f"{i}. {rec}\n"
        
        return summary