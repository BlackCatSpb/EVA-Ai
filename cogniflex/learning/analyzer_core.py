"""Основной модуль самоанализа для CogniFlex - ядро системы"""
import sqlite3
import os
import logging
import time
import threading
import json
import queue
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger("cogniflex.analyzer_core")

@dataclass
class LearningOpportunity:
    """Представляет возможность для обучения системы."""
    concept: str
    opportunity_type: str  # expansion, refinement, updating, integration
    priority: float  # 0.0-1.0
    domain: str
    evidence: List[str]
    suggested_actions: List[str]
    created_at: float
    last_updated: float
    executed: bool = False
    execution: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует возможность для обучения в словарь."""
        return {
            "concept": self.concept,
            "opportunity_type": self.opportunity_type,
            "priority": self.priority,
            "domain": self.domain,
            "evidence": self.evidence,
            "suggested_actions": self.suggested_actions,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "executed": self.executed,
            "execution": self.execution,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningOpportunity':
        """Создает возможность для обучения из словаря."""
        return cls(
            concept=data["concept"],
            opportunity_type=data["opportunity_type"],
            priority=data["priority"],
            domain=data["domain"],
            evidence=data["evidence"],
            suggested_actions=data["suggested_actions"],
            created_at=data["created_at"],
            last_updated=data["last_updated"],
            executed=data.get("executed", False),
            execution=data.get("execution"),
            metadata=data.get("metadata", {})
        )

class AnalyzerCore:
    """Основной интерфейс модуля самоанализа для CogniFlex."""
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None):
        """
        Инициализирует ядро модуля самоанализа.
        
        Args:
            brain: Ссылка на ядро CogniFlex (опционально)
            cache_dir: Путь к директории кэша (опционально)
        """
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "cogniflex_self_analyzer_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Путь к базе данных
        self.db_path = os.path.join(self.cache_dir, "self_analyzer.db")
        
        # Инициализируем базу данных
        self._init_db()
        
        # Конфигурация
        self.min_severity = 0.3
        self.min_priority = 0.3
        self.analysis_interval = 300  # 5 минут
        self.max_feedback_buffer = 100
        
        # Статистика
        self.analysis_stats = {
            "total_analyses": 0,
            "last_analysis": 0,
            "total_gaps": 0,
            "resolved_gaps": 0,
            "learning_opportunities": 0
        }
        
        # Очередь анализа
        self.analysis_queue = queue.Queue()
        
        # Состояние
        self.running = False
        self.stop_event = threading.Event()
        
        # Загружаем данные
        self._load_data()
        
        logger.info("AnalyzerCore инициализирован")
    
    def _init_db(self):
        """Инициализирует базу данных для хранения данных AnalyzerCore."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Таблица для возможностей обучения
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_opportunities (
                id TEXT PRIMARY KEY,
                concept TEXT NOT NULL,
                opportunity_type TEXT NOT NULL,
                priority REAL NOT NULL,
                domain TEXT NOT NULL,
                evidence TEXT NOT NULL,
                suggested_actions TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_updated REAL NOT NULL,
                executed BOOLEAN NOT NULL,
                execution TEXT,
                metadata TEXT NOT NULL
            )
            ''')
            
            # Таблица для истории анализа
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                findings TEXT NOT NULL,
                recommendations TEXT NOT NULL,
                metrics TEXT NOT NULL
            )
            ''')
            
            # Таблица для конфигурации
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.debug("База данных AnalyzerCore инициализирована")
        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных AnalyzerCore: {e}")
    
    def _load_data(self):
        """Загружает данные из базы данных."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Загружаем конфигурацию
            cursor.execute("SELECT key, value FROM config")
            for row in cursor.fetchall():
                if row[0] == "min_severity":
                    self.min_severity = float(row[1])
                elif row[0] == "min_priority":
                    self.min_priority = float(row[1])
                elif row[0] == "analysis_interval":
                    self.analysis_interval = int(row[1])
                elif row[0] == "max_feedback_buffer":
                    self.max_feedback_buffer = int(row[1])
            
            conn.close()
            
            logger.debug("Конфигурация AnalyzerCore загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки данных AnalyzerCore: {e}")
    
    def save_config(self):
        """Сохраняет конфигурацию модуля самоанализа."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Сохраняем конфигурацию
            config = {
                "min_severity": self.min_severity,
                "min_priority": self.min_priority,
                "analysis_interval": self.analysis_interval,
                "max_feedback_buffer": self.max_feedback_buffer
            }
            
            for key, value in config.items():
                cursor.execute('''
                INSERT OR REPLACE INTO config (key, value)
                VALUES (?, ?)
                ''', (key, str(value)))
            
            conn.commit()
            conn.close()
            
            logger.debug("Конфигурация AnalyzerCore сохранена")
        except Exception as e:
            logger.error(f"Ошибка сохранения конфигурации AnalyzerCore: {e}")
    
    def get_learning_opportunities(self, min_priority: float = 0.0, 
                                 executed: Optional[bool] = None,
                                 limit: int = 100,
                                 domain: Optional[str] = None,
                                 resolved: Optional[bool] = None) -> List[LearningOpportunity]:
        """
        Возвращает список возможностей для обучения с фильтрацией.
        
        Args:
            min_priority: Минимальный приоритет (0.0-1.0)
            executed: Фильтр по выполненным возможностям (True, False, None)
            limit: Максимальное количество результатов
            domain: Фильтр по домену
            resolved: Фильтр по разрешенным возможностям
            
        Returns:
            List[LearningOpportunity]: Список возможностей для обучения
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Формируем запрос
            query = "SELECT * FROM learning_opportunities WHERE priority >= ?"
            params = [min_priority]
            
            if executed is not None:
                query += " AND executed = ?"
                params.append(1 if executed else 0)
                
            if domain:
                query += " AND domain = ?"
                params.append(domain)
                
            # Для resolved используем executed
            if resolved is not None:
                query += " AND executed = ?"
                params.append(1 if resolved else 0)
                
            query += " ORDER BY priority DESC LIMIT ?"
            params.append(limit)
            
            # Выполняем запрос
            cursor.execute(query, params)
            
            # Преобразуем результаты
            opportunities = []
            for row in cursor.fetchall():
                opportunity = LearningOpportunity(
                    concept=row[1],
                    opportunity_type=row[2],
                    priority=row[3],
                    domain=row[4],
                    evidence=json.loads(row[5]),
                    suggested_actions=json.loads(row[6]),
                    created_at=row[7],
                    last_updated=row[8],
                    executed=bool(row[9]),
                    execution=json.loads(row[10]) if row[10] else None,
                    metadata=json.loads(row[11])
                )
                opportunities.append(opportunity)
            
            conn.close()
            return opportunities
            
        except Exception as e:
            logger.error(f"Ошибка получения возможностей для обучения: {e}")
            return []
    
    def add_learning_opportunity(self, concept: str, opportunity_type: str, 
                               priority: float, domain: str, 
                               evidence: List[str], suggested_actions: List[str],
                               callback: Optional[Callable] = None) -> bool:
        """
        Добавляет новую возможность для обучения.
        
        Args:
            concept: Концепт, связанный с возможностью
            opportunity_type: Тип возможности (expansion, refinement, updating, integration)
            priority: Приоритет (0.0-1.0)
            domain: Домен знаний
            evidence: Доказательства необходимости
            suggested_actions: Предлагаемые действия
            callback: Функция обратного вызова
            
        Returns:
            bool: Успешно ли добавлено
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Проверяем на дубликат
            cursor.execute("SELECT id FROM learning_opportunities WHERE concept = ? AND opportunity_type = ?", 
                          (concept, opportunity_type))
            existing = cursor.fetchone()
            
            if existing:
                # Обновляем существующую возможность, если новый приоритет выше
                cursor.execute("SELECT priority FROM learning_opportunities WHERE id = ?", (existing[0],))
                old_priority = cursor.fetchone()[0]
                
                if priority > old_priority:
                    cursor.execute('''
                    UPDATE learning_opportunities SET
                        priority = ?,
                        evidence = ?,
                        suggested_actions = ?,
                        last_updated = ?,
                        metadata = ?
                    WHERE id = ?
                    ''', (
                        priority,
                        json.dumps(evidence),
                        json.dumps(suggested_actions),
                        time.time(),
                        json.dumps({"updated": True}),
                        existing[0]
                    ))
                
                conn.commit()
                conn.close()
                logger.info(f"Обновлена возможность для обучения: {concept} ({opportunity_type}, {priority:.2f})")
                return True
            
            # Создаем новую возможность
            opportunity_id = f"{concept}_{opportunity_type}_{int(time.time())}"
            cursor.execute('''
            INSERT INTO learning_opportunities
            (id, concept, opportunity_type, priority, domain, evidence, 
            suggested_actions, created_at, last_updated, executed, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                opportunity_id,
                concept,
                opportunity_type,
                priority,
                domain,
                json.dumps(evidence),
                json.dumps(suggested_actions),
                time.time(),
                time.time(),
                False,
                json.dumps({})
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Добавлена возможность для обучения: {concept} ({opportunity_type}, {priority:.2f})")
            
            # Вызываем callback, если он предоставлен
            if callback:
                callback(True, "Возможность успешно добавлена")
                
            return True
            
        except Exception as e:
            logger.error(f"Ошибка добавления возможности для обучения: {e}")
            if callback:
                callback(False, str(e))
            return False
    
    def start_background_analysis(self, interval: Optional[int] = None):
        """Запускает фоновые процессы анализа."""
        if self.running:
            logger.warning("Попытка запуска уже активного процесса анализа")
            return
            
        self.running = True
        self.stop_event.clear()
        
        # Устанавливаем интервал, если он предоставлен
        if interval is not None:
            self.analysis_interval = interval
            
        # Запускаем рабочий поток
        self.analysis_thread = threading.Thread(
            target=self._analysis_worker,
            daemon=True,
            name="SelfAnalyzer-Background"
        )
        self.analysis_thread.start()
        
        # Запускаем регулярный анализ
        self.schedule_periodic_analysis()
        
        logger.info(f"Фоновый анализ системы запущен (интервал: {self.analysis_interval} сек)")
    
    def stop_background_analysis(self):
        """Останавливает фоновый анализ системы."""
        self.running = False
        self.stop_event.set()
        
        if hasattr(self, 'analysis_thread') and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=5.0)
        
        logger.info("Фоновый анализ системы остановлен")
    
    def schedule_periodic_analysis(self):
        """Запускает регулярный анализ с заданным интервалом."""
        if not self.running:
            return
            
        def put_task():
            if self.running:
                self.analysis_queue.put({"type": "periodic_analysis"})
        
        timer = threading.Timer(self.analysis_interval, put_task)
        timer.start()
    
    def _analysis_worker(self):
        """Рабочий процесс для обработки очереди анализа."""
        while not self.stop_event.is_set():
            try:
                task = self.analysis_queue.get(timeout=1)
                
                task_type = task.get("type")
                if task_type == "periodic_analysis":
                    # Выполняем регулярный анализ
                    self._perform_periodic_analysis()
                    # Планируем следующий анализ
                    self.schedule_periodic_analysis()
                    
                elif task_type == "analyze_feedback":
                    self._analyze_feedback_task()
                    
                elif task_type == "analyze_knowledge":
                    self._analyze_knowledge_task()
                    
                elif task_type == "analyze_performance":
                    self._analyze_performance_task()
                    
                self.analysis_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Ошибка в рабочем процессе самоанализа: {e}")
                time.sleep(5)
    
    def _perform_periodic_analysis(self):
        """Выполняет регулярный анализ."""
        logger.info("Выполнение регулярного анализа...")
        try:
            # Анализируем фидбэк
            self._analyze_feedback_task()
            
            # Анализируем знания
            self._analyze_knowledge_task()
            
            # Анализируем производительность
            self._analyze_performance_task()
            
            logger.info("Регулярный анализ завершен")
        except Exception as e:
            logger.error(f"Ошибка регулярного анализа: {e}")
    
    def _analyze_feedback_task(self):
        """Анализирует фидбэк."""
        logger.info("Анализ фидбэка...")
        # TODO: Реализовать анализ фидбэка
        pass
    
    def _analyze_knowledge_task(self):
        """Анализирует знания."""
        logger.info("Анализ знаний...")
        # TODO: Реализовать анализ знаний
        pass
    
    def _analyze_performance_task(self):
        """Анализирует производительность."""
        logger.info("Анализ производительности...")
        # TODO: Реализовать анализ производительности
        pass
    
    def _update_stats(self):
        """Обновляет статистику модуля."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Получаем количество возможностей для обучения
            cursor.execute("SELECT COUNT(*) FROM learning_opportunities")
            learning_opportunities = cursor.fetchone()[0]
            
            conn.close()
            
            self.analysis_stats["learning_opportunities"] = learning_opportunities
            
        except Exception as e:
            logger.error(f"Ошибка обновления статистики: {e}")
    
    def close(self):
        """Закрывает AnalyzerCore и освобождает ресурсы."""
        logger.info("Закрытие AnalyzerCore...")
        
        # Останавливаем фоновый анализ
        self.stop_background_analysis()
        
        # Сохраняем данные
        self._save_data()
        
        logger.info("AnalyzerCore закрыт")
    
    def _save_data(self):
        """Сохраняет данные в базу данных."""
        # В реальной системе здесь будут операции сохранения
        pass