"""Модуль управления этическими принципами для CogniFlex"""
import os
import logging
import json
import sqlite3
import time
import threading
import base64
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from io import BytesIO
import hashlib
from cogniflex.ethics.ethics_framework import EthicalPrinciple

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
except ImportError:
    # Matplotlib не установлен, визуализация будет недоступна
    Figure = None
    FigureCanvasAgg = None
    plt = None

logger = logging.getLogger("cogniflex.ethics.principles")

class PrinciplesManager:
    """Управляет этическими принципами и их состоянием."""
    
    def __init__(self, ethics_core, cache_dir: Optional[str] = None):
        """
        Инициализирует менеджер этических принципов.
        
        Args:
            ethics_core: Ссылка на основной модуль этической рамки
            cache_dir: Путь к директории кэша
        """
        self.ethics_core = ethics_core
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "cogniflex_ethics_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Путь к базе данных
        self.db_path = os.path.join(self.cache_dir, "ethics_principles.db")
        
        # Инициализируем базу данных
        self._init_db()
        
        # Блокировка ресурсов
        self.lock = threading.Lock()
        
        # Загружаем принципы
        self.principles = {}
        self.load_principles()
        
        logger.info(f"Менеджер этических принципов инициализирован с {len(self.principles)} принципами")
    
    def _init_db(self):
        """Инициализирует базу данных для хранения этических принципов."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Таблица для хранения принципов
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS principles (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                weight REAL NOT NULL,
                threshold REAL NOT NULL,
                category TEXT NOT NULL,
                last_updated REAL NOT NULL,
                active BOOLEAN NOT NULL
            )
            ''')
            
            # Таблица для хранения истории изменений
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS principles_history (
                id TEXT PRIMARY KEY,
                principle_id TEXT NOT NULL,
                action TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                timestamp REAL NOT NULL,
                user_id TEXT
            )
            ''')
            
            # Таблица для хранения оценок принципов
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS principles_assessments (
                id TEXT PRIMARY KEY,
                principle_id TEXT NOT NULL,
                score REAL NOT NULL,
                confidence REAL NOT NULL,
                context TEXT NOT NULL,
                timestamp REAL NOT NULL,
                scenario_id TEXT
            )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.debug("База данных менеджера принципов инициализирована")
        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных менеджера принципов: {e}")
    
    def load_principles(self):
        """Загружает этические принципы из базы данных."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Загружаем активные принципы
            cursor.execute("SELECT * FROM principles WHERE active = 1")
            for row in cursor.fetchall():
                principle = EthicalPrinciple(
                    name=row[1],
                    description=row[2],
                    weight=row[3],
                    threshold=row[4],
                    category=row[5],
                    last_updated=row[6],
                    active=bool(row[7])
                )
                self.principles[row[0]] = principle
            
            conn.close()
            
            # Если принципы не загружены, используем стандартные
            if not self.principles:
                self._load_default_principles()
                logger.info("Используются стандартные этические принципы")
            else:
                logger.info(f"Загружено {len(self.principles)} этических принципов")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки этических принципов: {e}")
            self._load_default_principles()
    
    def _load_default_principles(self):
        """Загружает стандартные этические принципы."""
        default_principles = [
            EthicalPrinciple(
                name="non_harm",
                description="Не наносить вред людям и обществу",
                weight=1.0,
                threshold=0.85,
                category="safety"
            ),
            EthicalPrinciple(
                name="beneficence",
                description="Принести пользу и улучшить благополучие",
                weight=0.9,
                threshold=0.8,
                category="benefit"
            ),
            EthicalPrinciple(
                name="autonomy",
                description="Уважать автономию и свободу выбора людей",
                weight=0.85,
                threshold=0.75,
                category="rights"
            ),
            EthicalPrinciple(
                name="justice",
                description="Обеспечивать справедливое распределение выгод и бремени",
                weight=0.8,
                threshold=0.7,
                category="fairness"
            ),
            EthicalPrinciple(
                name="transparency",
                description="Быть прозрачным в действиях и решениях",
                weight=0.75,
                threshold=0.65,
                category="trust"
            ),
            EthicalPrinciple(
                name="accountability",
                description="Брать ответственность за свои действия",
                weight=0.95,
                threshold=0.9,
                category="responsibility"
            )
        ]
        
        # Добавляем принципы в систему
        for principle in default_principles:
            self.add_principle(principle)
    
    def add_principle(self, principle: EthicalPrinciple, user_id: Optional[str] = None) -> str:
        """
        Добавляет новый этический принцип.
        
        Args:
            principle: Экземпляр EthicalPrinciple
            user_id: ID пользователя, который добавляет принцип
            
        Returns:
            str: ID добавленного принципа
        """
        try:
            with self.lock:
                # Генерируем уникальный ID
                principle_id = f"principle_{hash(principle.name + str(time.time())) % 1000000}"
                
                # Сохраняем в базу данных
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT OR REPLACE INTO principles
                (id, name, description, weight, threshold, category, last_updated, active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    principle_id,
                    principle.name,
                    principle.description,
                    principle.weight,
                    principle.threshold,
                    principle.category,
                    principle.last_updated,
                    principle.active
                ))
                
                # Сохраняем историю
                cursor.execute('''
                INSERT INTO principles_history
                (id, principle_id, action, old_value, new_value, timestamp, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    f"history_{hash(principle_id + 'add') % 1000000}",
                    principle_id,
                    "add",
                    None,
                    json.dumps(principle.__dict__),
                    time.time(),
                    user_id
                ))
                
                conn.commit()
                conn.close()
                
                # Добавляем в память
                self.principles[principle_id] = principle
                
                logger.info(f"Добавлен новый этический принцип: {principle.name} (ID: {principle_id})")
                return principle_id
                
        except Exception as e:
            logger.error(f"Ошибка добавления этического принципа: {e}")
            raise
    
    def update_principle(self, principle_id: str, updates: Dict[str, Any], user_id: Optional[str] = None) -> bool:
        """
        Обновляет существующий этический принцип.
        
        Args:
            principle_id: ID принципа
            updates: Словарь с обновлениями
            user_id: ID пользователя, который обновляет принцип
            
        Returns:
            bool: Успешно ли обновлено
        """
        try:
            with self.lock:
                if principle_id not in self.principles:
                    logger.warning(f"Попытка обновить несуществующий принцип: {principle_id}")
                    return False
                
                # Получаем текущий принцип
                old_principle = self.principles[principle_id]
                old_data = old_principle.__dict__.copy()
                
                # Создаем обновленный принцип
                updated_principle = EthicalPrinciple(
                    name=updates.get("name", old_principle.name),
                    description=updates.get("description", old_principle.description),
                    weight=updates.get("weight", old_principle.weight),
                    threshold=updates.get("threshold", old_principle.threshold),
                    category=updates.get("category", old_principle.category),
                    last_updated=time.time(),
                    active=updates.get("active", old_principle.active)
                )
                
                # Сохраняем в базу данных
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                UPDATE principles SET
                    name = ?,
                    description = ?,
                    weight = ?,
                    threshold = ?,
                    category = ?,
                    last_updated = ?,
                    active = ?
                WHERE id = ?
                ''', (
                    updated_principle.name,
                    updated_principle.description,
                    updated_principle.weight,
                    updated_principle.threshold,
                    updated_principle.category,
                    updated_principle.last_updated,
                    updated_principle.active,
                    principle_id
                ))
                
                # Сохраняем историю
                cursor.execute('''
                INSERT INTO principles_history
                (id, principle_id, action, old_value, new_value, timestamp, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    f"history_{hash(principle_id + 'update') % 1000000}",
                    principle_id,
                    "update",
                    json.dumps(old_data),
                    json.dumps(updated_principle.__dict__),
                    time.time(),
                    user_id
                ))
                
                conn.commit()
                conn.close()
                
                # Обновляем в памяти
                self.principles[principle_id] = updated_principle
                
                logger.info(f"Обновлен этический принцип: {updated_principle.name} (ID: {principle_id})")
                return True
                
        except Exception as e:
            logger.error(f"Ошибка обновления этического принципа: {e}")
            return False
    
    def get_principle(self, principle_id: str) -> Optional[EthicalPrinciple]:
        """
        Получает этический принцип по ID.
        
        Args:
            principle_id: ID принципа
            
        Returns:
            Optional[EthicalPrinciple]: Принцип или None
        """
        return self.principles.get(principle_id)
    
    def get_principles_by_category(self, category: str) -> List[Tuple[str, EthicalPrinciple]]:
        """
        Получает принципы по категории.
        
        Args:
            category: Категория принципов
            
        Returns:
            List[Tuple[str, EthicalPrinciple]]: Список ID и принципов
        """
        return [(pid, p) for pid, p in self.principles.items() if p.category == category]
    
    def get_all_principles(self) -> Dict[str, EthicalPrinciple]:
        """
        Возвращает все этические принципы.
        
        Returns:
            Dict[str, EthicalPrinciple]: Словарь принципов
        """
        return self.principles.copy()
    
    def record_assessment(self, principle_id: str, score: float, confidence: float, 
                          context: Dict[str, Any], scenario_id: Optional[str] = None):
        """
        Записывает оценку применения принципа.
        
        Args:
            principle_id: ID принципа
            score: Оценка применения (0.0-1.0)
            confidence: Уверенность в оценке (0.0-1.0)
            context: Контекст оценки
            scenario_id: ID сценария (опционально)
        """
        try:
            # Генерируем уникальный ID
            assessment_id = f"assessment_{hash(principle_id + str(time.time())) % 1000000}"
            
            # Сохраняем в базу данных
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO principles_assessments
            (id, principle_id, score, confidence, context, timestamp, scenario_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                assessment_id,
                principle_id,
                score,
                confidence,
                json.dumps(context),
                time.time(),
                scenario_id
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Записана оценка для принципа {principle_id}: {score:.2f}")
            
        except Exception as e:
            logger.error(f"Ошибка записи оценки принципа: {e}")
    
    def get_assessment_history(self, principle_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Получает историю оценок для принципа.
        
        Args:
            principle_id: ID принципа
            days: Количество дней для анализа
            
        Returns:
            List[Dict[str, Any]]: История оценок
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = time.time() - (days * 86400)
            cursor.execute(
                "SELECT * FROM principles_assessments WHERE principle_id = ? AND timestamp >= ? ORDER BY timestamp DESC",
                (principle_id, cutoff_time)
            )
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    "id": row[0],
                    "principle_id": row[1],
                    "score": row[2],
                    "confidence": row[3],
                    "context": json.loads(row[4]),
                    "timestamp": row[5],
                    "scenario_id": row[6],
                    "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row[5]))
                })
            
            conn.close()
            return history
            
        except Exception as e:
            logger.error(f"Ошибка получения истории оценок: {e}")
            return []
    
    def get_principles_dashboard_data(self) -> Dict[str, Any]:
        """
        Возвращает данные для дашборда этических принципов.
        
        Returns:
            Dict[str, Any]: Данные для дашборда
        """
        # Получаем данные за последние 7 дней
        end_time = time.time()
        start_time = end_time - (7 * 86400)  # 7 дней назад
        
        # Формируем данные по категориям
        category_data = defaultdict(lambda: {"count": 0, "avg_score": 0.0, "total_score": 0.0})
        
        for principle_id, principle in self.principles.items():
            # Получаем историю оценок
            history = self.get_assessment_history(principle_id, days=7)
            
            if history:
                total_score = sum(item["score"] for item in history)
                avg_score = total_score / len(history) if history else 0.0
                
                category = category_data[principle.category]
                category["count"] += 1
                category["total_score"] += avg_score
                category["avg_score"] = category["total_score"] / category["count"]
        
        # Формируем данные для категорий
        categories = []
        for category, data in category_data.items():
            categories.append({
                "category": category,
                "principles_count": data["count"],
                "avg_compliance": data["avg_score"]
            })
        
        # Формируем данные для проблемных принципов
        problematic_principles = []
        for principle_id, principle in self.principles.items():
            history = self.get_assessment_history(principle_id, days=7)
            
            if history:
                avg_score = sum(item["score"] for item in history) / len(history)
                if avg_score < principle.threshold * 0.9:  # Принцип часто нарушается
                    problematic_principles.append({
                        "id": principle_id,
                        "name": principle.name,
                        "current_compliance": avg_score,
                        "threshold": principle.threshold,
                        "category": principle.category
                    })
        
        # Формируем временные ряды
        daily_scores = defaultdict(lambda: defaultdict(float))
        daily_counts = defaultdict(lambda: defaultdict(int))
        
        for principle_id, principle in self.principles.items():
            history = self.get_assessment_history(principle_id, days=7)
            
            for item in history:
                date = time.strftime("%Y-%m-%d", time.localtime(item["timestamp"]))
                daily_scores[date][principle.category] += item["score"]
                daily_counts[date][principle.category] += 1
        
        # Вычисляем средние значения
        trends = []
        for date, categories in daily_scores.items():
            for category, total_score in categories.items():
                count = daily_counts[date][category]
                if count > 0:
                    trends.append({
                        "date": date,
                        "category": category,
                        "compliance": total_score / count
                    })
        
        return {
            "principles_count": len(self.principles),
            "categories": categories,
            "problematic_principles": problematic_principles,
            "trends": trends,
            "timestamp": time.time()
        }
    
    def generate_ethical_visualization(self, view_type: str = "compliance") -> str:
        """
        Генерирует визуализацию этических данных.
        
        Args:
            view_type: Тип визуализации
            
        Returns:
            str: Изображение в формате base64
        """
        try:
            # Получаем данные для визуализации
            dashboard_data = self.get_principles_dashboard_data()
            
            # Создаем фигуру
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            if view_type == "compliance":
                # Визуализация соблюдения принципов
                categories = [c["category"] for c in dashboard_data["categories"]]
                compliance = [c["avg_compliance"] for c in dashboard_data["categories"]]
                
                ax.bar(categories, compliance, color='skyblue')
                ax.axhline(y=0.8, color='r', linestyle='--', label='Порог')
                ax.set_ylabel('Соблюдение (0-1)')
                ax.set_title('Соблюдение этических принципов по категориям')
                ax.legend()
                
            elif view_type == "trends":
                # Визуализация временных трендов
                trends = defaultdict(list)
                dates = set()
                
                for trend in dashboard_data["trends"]:
                    trends[trend["category"]].append((trend["date"], trend["compliance"]))
                    dates.add(trend["date"])
                
                dates = sorted(dates)
                for category, values in trends.items():
                    values_dict = {date: compliance for date, compliance in values}
                    compliance = [values_dict.get(date, 0) for date in dates]
                    ax.plot(dates, compliance, marker='o', label=category)
                
                ax.set_xlabel('Дата')
                ax.set_ylabel('Соблюдение (0-1)')
                ax.set_title('Тренды соблюдения этических принципов')
                ax.legend()
                plt.xticks(rotation=45)
            
            elif view_type == "problematic":
                # Визуализация проблемных принципов
                principles = [p["name"] for p in dashboard_data["problematic_principles"]]
                compliance = [p["current_compliance"] for p in dashboard_data["problematic_principles"]]
                thresholds = [p["threshold"] for p in dashboard_data["problematic_principles"]]
                
                x = range(len(principles))
                ax.bar(x, compliance, color='salmon', label='Текущее')
                ax.bar(x, thresholds, color='lightgray', alpha=0.3, label='Порог')
                ax.set_xticks(x)
                ax.set_xticklabels(principles, rotation=45, ha='right')
                ax.set_ylabel('Значение')
                ax.set_title('Проблемные этические принципы')
                ax.legend()
            
            # Сохраняем в буфер
            buf = BytesIO()
            fig.tight_layout()
            canvas = FigureCanvasAgg(fig)
            canvas.print_png(buf)
            
            # Преобразуем в base64
            buf.seek(0)
            img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{img_data}"
            
        except Exception as e:
            logger.error(f"Ошибка генерации визуализации этических данных: {e}")
            return ""
    
    def close(self):
        """Закрывает менеджер принципов и освобождает ресурсы."""
        logger.info("Закрытие менеджера этических принципов...")
        # Здесь можно добавить дополнительные действия при закрытии
        logger.info("Менеджер этических принципов закрыт")
    
    def close(self):
        """Закрывает менеджер принципов и освобождает ресурсы."""
        logger.info("Закрытие менеджера этических принципов...")
        # Здесь можно добавить дополнительные действия при закрытии
        logger.info("Менеджер этических принципов закрыт")