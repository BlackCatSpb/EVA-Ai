"""Модуль для анализа репутации источников в системе ЕВА"""
import os
import logging
import time
import json
import re
import sqlite3
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
import tldextract
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logger = logging.getLogger("eva.contradiction.reputation")

# Инициализация NLP-ресурсов (offline-safe)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception:
        pass

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except Exception:
        pass

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    try:
        nltk.download('vader_lexicon', quiet=True)
    except Exception:
        pass

class SourceReputationSystem:
    """Система анализа и управления репутацией источников информации."""
    
    def __init__(self, db_path: Optional[str] = None, 
                 update_interval: int = 86400,  # 24 часа
                 decay_rate: float = 0.95):
        """
        Инициализирует систему репутации источников.
        
        Args:
            db_path: Путь к базе данных
            update_interval: Интервал обновления репутации в секундах
            decay_rate: Коэффициент затухания репутации со временем
        """
        self.update_interval = update_interval
        self.decay_rate = decay_rate
        self.reputation_cache = {}
        self.domain_reputation_cache = {}
        self.last_update = {}
        
        # Настройка базы данных
        self.db_path = db_path or os.path.join(os.path.dirname(__file__), "source_reputation.db")
        self._init_database()
        
        # Загружаем репутацию из базы данных
        self._load_reputation_data()
        
        # Регистрируем систему в логгере
        logger.info("Система репутации источников инициализирована")
    
    def _init_database(self):
        """Инициализирует базу данных для хранения репутации источников."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            # Создаем таблицу для репутации источников
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS source_reputation (
                source TEXT PRIMARY KEY,
                reputation REAL DEFAULT 0.5,
                domain TEXT,
                last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_ratings INTEGER DEFAULT 0,
                positive_ratings INTEGER DEFAULT 0,
                negative_ratings INTEGER DEFAULT 0,
                source_type TEXT,
                domain_reputation REAL DEFAULT 0.5
            )
            ''')
            
            # Создаем таблицу для метрик источников
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS source_metrics (
                source TEXT,
                metric_name TEXT,
                metric_value REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(source) REFERENCES source_reputation(source)
            )
            ''')
            
            # Создаем таблицу для белого списка авторитетных источников
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS trusted_sources (
                domain TEXT PRIMARY KEY,
                category TEXT,
                trust_score REAL DEFAULT 0.9,
                added_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Создаем таблицу для черного списка ненадежных источников
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS untrusted_sources (
                domain TEXT PRIMARY KEY,
                reason TEXT,
                trust_score REAL DEFAULT 0.1,
                added_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            self.conn.commit()
            
            # Добавляем известные авторитетные источники в белый список
            self._populate_trusted_sources()
            
            logger.info(f"База данных репутации источников инициализирована: {self.db_path}")
        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных репутации: {e}")
            raise
    
    def _populate_trusted_sources(self):
        """Заполняет белый список известными авторитетными источниками."""
        cursor = self.conn.cursor()
        
        trusted_sources = [
            ("wikipedia.org", "encyclopedia", 0.85),
            ("nih.gov", "medical", 0.95),
            ("nasa.gov", "scientific", 0.92),
            ("who.int", "medical", 0.93),
            ("nature.com", "scientific", 0.88),
            ("science.org", "scientific", 0.87),
            ("britannica.com", "encyclopedia", 0.82),
            ("mit.edu", "educational", 0.90),
            ("stanford.edu", "educational", 0.89),
            ("harvard.edu", "educational", 0.91),
            ("ox.ac.uk", "educational", 0.88),
            ("cam.ac.uk", "educational", 0.87),
            ("nih.gov", "medical", 0.95),
            ("cdc.gov", "medical", 0.93),
            ("nih.gov", "medical", 0.95),
            ("pubmed.ncbi.nlm.nih.gov", "medical", 0.94),
            ("arxiv.org", "scientific", 0.85),
            ("ieee.org", "technical", 0.86),
            ("acm.org", "technical", 0.84)
        ]
        
        for domain, category, trust_score in trusted_sources:
            try:
                cursor.execute('''
                INSERT OR IGNORE INTO trusted_sources (domain, category, trust_score)
                VALUES (?, ?, ?)
                ''', (domain, category, trust_score))
            except Exception as e:
                logger.warning(f"Не удалось добавить доверенный источник {domain}: {e}")
        
        self.conn.commit()
        logger.debug(f"Добавлено {len(trusted_sources)} доверенных источников в белый список")
    
    def _load_reputation_data(self):
        """Загружает данные о репутации из базы данных в кэш."""
        try:
            cursor = self.conn.cursor()
            
            # Загружаем репутацию источников
            cursor.execute("SELECT source, reputation, last_update FROM source_reputation")
            for source, reputation, last_update in cursor.fetchall():
                self.reputation_cache[source] = reputation
                self.last_update[source] = datetime.strptime(last_update, "%Y-%m-%d %H:%M:%S.%f") if isinstance(last_update, str) else last_update
            
            # Загружаем репутацию доменов
            cursor.execute('''
            SELECT domain, AVG(reputation) as avg_reputation 
            FROM source_reputation 
            GROUP BY domain
            ''')
            for domain, reputation in cursor.fetchall():
                self.domain_reputation_cache[domain] = reputation
            
            logger.info("Данные о репутации источников загружены в кэш")
        except Exception as e:
            logger.error(f"Ошибка загрузки данных о репутации: {e}")
    
    def get_source_reputation(self, source: str) -> float:
        """
        Возвращает репутацию источника.
        
        Args:
            source: URL или название источника
            
        Returns:
            float: Репутация источника (0.0-1.0)
        """
        # Проверяем кэш
        if source in self.reputation_cache:
            # Проверяем, нужно ли обновить репутацию
            if time.time() - self.last_update.get(source, 0) > self.update_interval:
                self._update_source_reputation(source)
            return self.reputation_cache[source]
        
        # Определяем домен
        domain = self._extract_domain(source)
        
        # Проверяем белый список
        if self._is_trusted_domain(domain):
            return 0.85  # Высокая репутация для доверенных доменов
        
        # Проверяем черный список
        if self._is_untrusted_domain(domain):
            return 0.15  # Низкая репутация для недоверенных доменов
        
        # Оцениваем репутацию на основе домена
        domain_reputation = self._get_domain_reputation(domain)
        
        # Вычисляем базовую репутацию
        base_reputation = self._calculate_base_reputation(source, domain, domain_reputation)
        
        # Сохраняем в кэш
        self.reputation_cache[source] = base_reputation
        self.last_update[source] = time.time()
        
        # Сохраняем в базу данных
        self._save_source_reputation(source, base_reputation, domain)
        
        return base_reputation
    
    def _extract_domain(self, source: str) -> str:
        """
        Извлекает домен из URL источника.
        
        Args:
            source: URL или название источника
            
        Returns:
            str: Домен источника
        """
        try:
            # Проверяем, является ли источник URL
            if "://" in source or "." in source:
                extracted = tldextract.extract(source)
                return f"{extracted.domain}.{extracted.suffix}".lower()
            else:
                # Если источник не URL, возвращаем его как есть
                return source.lower()
        except Exception as e:
            logger.warning(f"Ошибка извлечения домена из {source}: {e}")
            return source.lower()
    
    def _is_trusted_domain(self, domain: str) -> bool:
        """
        Проверяет, находится ли домен в белом списке доверенных источников.
        
        Args:
            domain: Домен для проверки
            
        Returns:
            bool: Является ли домен доверенным
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM trusted_sources WHERE domain = ?", (domain,))
        return cursor.fetchone() is not None
    
    def _is_untrusted_domain(self, domain: str) -> bool:
        """
        Проверяет, находится ли домен в черном списке недоверенных источников.
        
        Args:
            domain: Домен для проверки
            
        Returns:
            bool: Является ли домен недоверенным
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM untrusted_sources WHERE domain = ?", (domain,))
        return cursor.fetchone() is not None
    
    def _get_domain_reputation(self, domain: str) -> float:
        """
        Возвращает репутацию домена на основе всех источников с этим доменом.
        
        Args:
            domain: Домен
            
        Returns:
            float: Репутация домена (0.0-1.0)
        """
        if domain in self.domain_reputation_cache:
            return self.domain_reputation_cache[domain]
        
        # Проверяем белый список
        cursor = self.conn.cursor()
        cursor.execute("SELECT trust_score FROM trusted_sources WHERE domain = ?", (domain,))
        result = cursor.fetchone()
        if result:
            self.domain_reputation_cache[domain] = result[0]
            return result[0]
        
        # Проверяем черный список
        cursor.execute("SELECT trust_score FROM untrusted_sources WHERE domain = ?", (domain,))
        result = cursor.fetchone()
        if result:
            self.domain_reputation_cache[domain] = result[0]
            return result[0]
        
        # Вычисляем среднюю репутацию для домена
        cursor.execute('''
        SELECT AVG(reputation) 
        FROM source_reputation 
        WHERE domain = ?
        ''', (domain,))
        
        result = cursor.fetchone()
        if result and result[0] is not None:
            domain_reputation = result[0]
        else:
            # Базовая репутация для неизвестных доменов
            domain_reputation = 0.5
            
            # Проверяем, содержит ли домен подозрительные ключевые слова
            suspicious_keywords = ["fake", "hoax", "conspiracy", "clickbait", "scam", "spam"]
            if any(keyword in domain for keyword in suspicious_keywords):
                domain_reputation = 0.3
            
            # Проверяем TLD (верхнеуровневый домен)
            tld = domain.split('.')[-1]
            low_trust_tlds = ["xyz", "info", "top", "club", "online", "site", "click"]
            if tld in low_trust_tlds:
                domain_reputation *= 0.7
        
        self.domain_reputation_cache[domain] = domain_reputation
        return domain_reputation
    
    def _calculate_base_reputation(self, source: str, domain: str, domain_reputation: float) -> float:
        """
        Вычисляет базовую репутацию источника.
        
        Args:
            source: Источник
            domain: Домен источника
            domain_reputation: Репутация домена
            
        Returns:
            float: Базовая репутация
        """
        # Начинаем с репутации домена
        reputation = domain_reputation
        
        # Учитываем тип источника
        source_type = self._determine_source_type(source)
        if source_type == "academic":
            reputation = min(1.0, reputation * 1.15)
        elif source_type == "news":
            reputation = min(1.0, reputation * 1.05)
        elif source_type == "blog":
            reputation = max(0.0, reputation * 0.9)
        elif source_type == "social_media":
            reputation = max(0.0, reputation * 0.75)
        
        # Учитываем SSL
        if "https://" in source.lower():
            reputation = min(1.0, reputation * 1.05)
        
        # Учитываем возраст источника (в реальной системе здесь будет проверка через WHOIS)
        # Для демонстрации используем случайное значение
        age_factor = 0.9 + 0.1 * np.random.random()
        reputation = min(1.0, reputation * age_factor)
        
        return max(0.0, min(1.0, reputation))
    
    def _determine_source_type(self, source: str) -> str:
        """
        Определяет тип источника.
        
        Args:
            source: Источник
            
        Returns:
            str: Тип источника
        """
        source_lower = source.lower()
        
        # Академические источники
        if any(domain in source_lower for domain in [
            "edu", "ac.uk", "researchgate.net", "arxiv.org", "springer.com", "ieee.org"
        ]):
            return "academic"
        
        # Новостные источники
        if any(domain in source_lower for domain in [
            "bbc.co.uk", "nytimes.com", "reuters.com", "apnews.com", "cnn.com", "theguardian.com"
        ]):
            return "news"
        
        # Блоги
        if any(domain in source_lower for domain in [
            "wordpress.com", "blogspot.com", "medium.com", "tumblr.com"
        ]):
            return "blog"
        
        # Социальные сети
        if any(domain in source_lower for domain in [
            "facebook.com", "twitter.com", "instagram.com", "reddit.com", "tiktok.com"
        ]):
            return "social_media"
        
        # Вики
        if "wikipedia.org" in source_lower:
            return "wiki"
        
        return "other"
    
    def _update_source_reputation(self, source: str):
        """
        Обновляет репутацию источника на основе новых данных.
        
        Args:
            source: Источник для обновления
        """
        try:
            # Получаем текущую репутацию
            current_reputation = self.reputation_cache.get(source, 0.5)
            
            # Получаем домен
            domain = self._extract_domain(source)
            
            # Получаем репутацию домена
            domain_reputation = self._get_domain_reputation(domain)
            
            # Вычисляем новую репутацию с учетом затухания
            time_factor = self.decay_rate ** ((time.time() - self.last_update.get(source, time.time())) / self.update_interval)
            new_reputation = current_reputation * time_factor + domain_reputation * (1 - time_factor)
            
            # Сохраняем обновленную репутацию
            self.reputation_cache[source] = new_reputation
            self.last_update[source] = time.time()
            
            # Обновляем в базе данных
            self._save_source_reputation(source, new_reputation, domain)
            
            logger.debug(f"Репутация источника {source} обновлена: {new_reputation:.4f}")
        except Exception as e:
            logger.error(f"Ошибка обновления репутации источника {source}: {e}")
    
    def _save_source_reputation(self, source: str, reputation: float, domain: str):
        """
        Сохраняет репутацию источника в базу данных.
        
        Args:
            source: Источник
            reputation: Репутация
            domain: Домен
        """
        try:
            cursor = self.conn.cursor()
            
            # Проверяем, существует ли запись
            cursor.execute("SELECT 1 FROM source_reputation WHERE source = ?", (source,))
            exists = cursor.fetchone() is not None
            
            if exists:
                # Обновляем существующую запись
                cursor.execute('''
                UPDATE source_reputation 
                SET reputation = ?, last_update = CURRENT_TIMESTAMP, domain = ?
                WHERE source = ?
                ''', (reputation, domain, source))
            else:
                # Создаем новую запись
                cursor.execute('''
                INSERT INTO source_reputation (source, reputation, domain)
                VALUES (?, ?, ?)
                ''', (source, reputation, domain))
            
            # Обновляем кэш домена
            self.domain_reputation_cache[domain] = self._get_domain_reputation(domain)
            
            self.conn.commit()
        except Exception as e:
            logger.error(f"Ошибка сохранения репутации источника {source}: {e}")
    
    def update_reputation_from_feedback(self, source: str, feedback: Dict[str, Any]):
        """
        Обновляет репутацию источника на основе пользовательской обратной связи.
        
        Args:
            source: Источник
            feedback: Обратная связь от пользователя
        """
        try:
            # Получаем текущую репутацию
            current_reputation = self.get_source_reputation(source)
            
            # Определяем влияние обратной связи
            feedback_impact = 0.0
            
            if "rating" in feedback:
                # Нормализуем рейтинг в диапазон 0.0-1.0
                normalized_rating = max(0.0, min(1.0, feedback["rating"]))
                
                # Вычисляем отклонение от текущей репутации
                deviation = normalized_rating - current_reputation
                
                # Определяем силу воздействия (меньше для экстремальных репутаций)
                impact_strength = 0.2 * (1.0 - abs(deviation))
                
                # Учитываем количество голосов
                vote_weight = 1.0
                if "vote_count" in feedback:
                    vote_weight = min(1.0, feedback["vote_count"] * 0.1)
                
                feedback_impact += deviation * impact_strength * vote_weight
            
            # Учитываем комментарий
            if "comment" in feedback and feedback["comment"]:
                comment = feedback["comment"].lower()
                
                # Ключевые слова, указывающие на низкую достоверность
                negative_keywords = ["недостоверно", "ложь", "ошибка", "неправда", "некорректно", 
                                    "неверно", "фейк", "манипуляция", "предвзято"]
                
                # Ключевые слова, указывающие на высокую достоверность
                positive_keywords = ["точно", "верно", "достоверно", "правильно", "академический",
                                    "научный", "авторитетный", "проверенный"]
                
                # Подсчитываем вхождения
                negative_count = sum(1 for kw in negative_keywords if kw in comment)
                positive_count = sum(1 for kw in positive_keywords if kw in comment)
                
                # Вычисляем чистый эффект
                net_effect = (positive_count - negative_count) * 0.05
                feedback_impact += net_effect
            
            # Обновляем репутацию
            if feedback_impact != 0:
                new_reputation = current_reputation + feedback_impact
                new_reputation = max(0.0, min(1.0, new_reputation))
                
                # Сохраняем обновленную репутацию
                self.reputation_cache[source] = new_reputation
                self.last_update[source] = time.time()
                
                # Сохраняем в базу данных
                self._save_source_reputation(source, new_reputation, self._extract_domain(source))
                
                logger.info(f"Репутация источника {source} обновлена на основе обратной связи: {new_reputation:.4f}")
        except Exception as e:
            logger.error(f"Ошибка обновления репутации из обратной связи: {e}")
    
    def add_to_trusted_sources(self, domain: str, category: str, trust_score: float = 0.9):
        """
        Добавляет домен в белый список доверенных источников.
        
        Args:
            domain: Домен
            category: Категория источника
            trust_score: Уровень доверия (0.0-1.0)
        """
        try:
            cursor = self.conn.cursor()
            
            # Проверяем, существует ли домен в белом списке
            cursor.execute("SELECT 1 FROM trusted_sources WHERE domain = ?", (domain,))
            exists = cursor.fetchone() is not None
            
            if exists:
                # Обновляем существующую запись
                cursor.execute('''
                UPDATE trusted_sources 
                SET category = ?, trust_score = ?, added_timestamp = CURRENT_TIMESTAMP
                WHERE domain = ?
                ''', (category, trust_score, domain))
            else:
                # Добавляем новый домен
                cursor.execute('''
                INSERT INTO trusted_sources (domain, category, trust_score)
                VALUES (?, ?, ?)
                ''', (domain, category, trust_score))
            
            # Удаляем домен из черного списка, если он там есть
            cursor.execute("DELETE FROM untrusted_sources WHERE domain = ?", (domain,))
            
            self.conn.commit()
            
            # Обновляем кэш
            self.domain_reputation_cache[domain] = trust_score
            
            logger.info(f"Домен {domain} добавлен в белый список доверенных источников")
        except Exception as e:
            logger.error(f"Ошибка добавления домена в белый список: {e}")
    
    def add_to_untrusted_sources(self, domain: str, reason: str, trust_score: float = 0.1):
        """
        Добавляет домен в черный список недоверенных источников.
        
        Args:
            domain: Домен
            reason: Причина добавления
            trust_score: Уровень доверия (0.0-1.0)
        """
        try:
            cursor = self.conn.cursor()
            
            # Проверяем, существует ли домен в черном списке
            cursor.execute("SELECT 1 FROM untrusted_sources WHERE domain = ?", (domain,))
            exists = cursor.fetchone() is not None
            
            if exists:
                # Обновляем существующую запись
                cursor.execute('''
                UPDATE untrusted_sources 
                SET reason = ?, trust_score = ?, added_timestamp = CURRENT_TIMESTAMP
                WHERE domain = ?
                ''', (reason, trust_score, domain))
            else:
                # Добавляем новый домен
                cursor.execute('''
                INSERT INTO untrusted_sources (domain, reason, trust_score)
                VALUES (?, ?, ?)
                ''', (domain, reason, trust_score))
            
            # Удаляем домен из белого списка, если он там есть
            cursor.execute("DELETE FROM trusted_sources WHERE domain = ?", (domain,))
            
            self.conn.commit()
            
            # Обновляем кэш
            self.domain_reputation_cache[domain] = trust_score
            
            logger.warning(f"Домен {domain} добавлен в черный список недоверенных источников: {reason}")
        except Exception as e:
            logger.error(f"Ошибка добавления домена в черный список: {e}")
    
    def get_source_metrics(self, source: str) -> Dict[str, Any]:
        """
        Возвращает метрики источника.
        
        Args:
            source: Источник
            
        Returns:
            Dict: Метрики источника
        """
        try:
            cursor = self.conn.cursor()
            
            # Получаем основные метрики
            cursor.execute('''
            SELECT reputation, total_ratings, positive_ratings, negative_ratings, source_type
            FROM source_reputation
            WHERE source = ?
            ''', (source,))
            
            result = cursor.fetchone()
            if not result:
                return {
                    "reputation": self.get_source_reputation(source),
                    "total_ratings": 0,
                    "positive_ratings": 0,
                    "negative_ratings": 0,
                    "source_type": self._determine_source_type(source)
                }
            
            reputation, total_ratings, positive_ratings, negative_ratings, source_type = result
            
            # Получаем дополнительные метрики
            cursor.execute('''
            SELECT metric_name, metric_value
            FROM source_metrics
            WHERE source = ?
            ''', (source,))
            
            metrics = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                "reputation": reputation,
                "total_ratings": total_ratings,
                "positive_ratings": positive_ratings,
                "negative_ratings": negative_ratings,
                "source_type": source_type or self._determine_source_type(source),
                **metrics
            }
        except Exception as e:
            logger.error(f"Ошибка получения метрик источника {source}: {e}")
            return {
                "reputation": self.get_source_reputation(source),
                "total_ratings": 0,
                "positive_ratings": 0,
                "negative_ratings": 0,
                "source_type": self._determine_source_type(source)
            }
    
    def record_source_metric(self, source: str, metric_name: str, metric_value: float):
        """
        Записывает метрику источника.
        
        Args:
            source: Источник
            metric_name: Название метрики
            metric_value: Значение метрики
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
            INSERT INTO source_metrics (source, metric_name, metric_value)
            VALUES (?, ?, ?)
            ''', (source, metric_name, metric_value))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Ошибка записи метрики источника {source}: {e}")
    
    def close(self):
        """Закрывает соединение с базой данных."""
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
                logger.info("Соединение с базой данных репутации источников закрыто")
        except Exception as e:
            logger.error(f"Ошибка закрытия соединения с базой данных: {e}")
    
    def analyze_source_credibility(self, source: str, content: str) -> Dict[str, Any]:
        """
        Проводит комплексный анализ достоверности источника и его контента.
        
        Args:
            source: Источник
            content: Контент для анализа
            
        Returns:
            Dict: Результаты анализа
        """
        try:
            # Получаем репутацию источника
            reputation = self.get_source_reputation(source)
            
            # Анализируем контент
            content_analysis = self._analyze_content(content)
            
            # Вычисляем общую достоверность
            credibility = (
                reputation * 0.6 +
                content_analysis["factual_accuracy"] * 0.2 +
                content_analysis["bias_level"] * 0.1 +
                content_analysis["consistency"] * 0.1
            )
            
            # Генерируем предупреждения
            warnings = self._generate_credibility_warnings(
                reputation,
                content_analysis["sentiment"],
                content_analysis["fact_errors"],
                content_analysis["originality"],
                content_analysis["argument_quality"]
            )
            
            return {
                "source": source,
                "reputation": reputation,
                "credibility": credibility,
                "content_analysis": content_analysis,
                "warnings": warnings,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Ошибка анализа достоверности источника: {e}")
            return {
                "source": source,
                "reputation": 0.5,
                "credibility": 0.5,
                "content_analysis": {},
                "warnings": ["Ошибка анализа достоверности"],
                "timestamp": time.time()
            }
    
    def _analyze_content(self, content: str) -> Dict[str, Any]:
        """
        Анализирует контент на предмет достоверности.
        
        Args:
            content: Контент для анализа
            
        Returns:
            Dict: Результаты анализа контента
        """
        # Инициализируем анализатор тональности
        sentiment_analyzer = SentimentIntensityAnalyzer()
        stop_words = set(stopwords.words('english') + stopwords.words('russian'))
        
        # Анализ тональности
        sentiment = sentiment_analyzer.polarity_scores(content)
        
        # Анализ структуры текста
        words = word_tokenize(content)
        words = [word for word in words if word.isalnum() and word not in stop_words]
        unique_words = len(set(words))
        total_words = len(words)
        
        # Оценка оригинальности (в реальной системе здесь будет проверка через поисковик)
        originality = 0.7 + 0.3 * np.random.random()  # Имитация
        
        # Оценка качества аргументации
        argument_quality = self._evaluate_argument_quality(content)
        
        # Оценка фактологических ошибок (в реальной системе здесь будет проверка через базу знаний)
        fact_errors = 0.2 * np.random.random()  # Имитация
        
        # Оценка согласованности с другими источниками (в реальной системе здесь будет сравнение)
        consistency = 0.7 + 0.3 * np.random.random()  # Имитация
        
        # Оценка фактологической точности
        factual_accuracy = max(0.0, min(1.0, 1.0 - fact_errors))
        
        # Оценка предвзятости
        bias_level = abs(sentiment['compound']) * 0.5
        
        return {
            "sentiment": sentiment,
            "word_count": total_words,
            "unique_words": unique_words,
            "vocabulary_diversity": unique_words / total_words if total_words > 0 else 0.0,
            "originality": originality,
            "argument_quality": argument_quality,
            "fact_errors": fact_errors,
            "consistency": consistency,
            "factual_accuracy": factual_accuracy,
            "bias_level": bias_level
        }
    
    def _evaluate_argument_quality(self, content: str) -> float:
        """
        Оценивает качество аргументации в контенте.
        
        Args:
            content: Контент для анализа
            
        Returns:
            float: Качество аргументации (0.0-1.0)
        """
        # Ищем ключевые слова, указывающие на логическую структуру
        logical_indicators = [
            "следовательно", "поэтому", "таким образом", "однако", "несмотря на это",
            "с одной стороны", "с другой стороны", "при условии что", "если... то",
            "следует из", "основываясь на", "в свете", "принимая во внимание"
        ]
        
        # Подсчитываем вхождения
        logical_count = sum(1 for indicator in logical_indicators if indicator in content.lower())
        
        # Ищем ключевые слова, указывающие на эмоциональную окраску
        emotional_indicators = [
            "очевидно", "несомненно", "бесспорно", "абсолютно", "никогда", "всегда",
            "ужасно", "прекрасно", "невероятно", "фантастически", "катастрофически"
        ]
        
        # Подсчитываем вхождения
        emotional_count = sum(1 for indicator in emotional_indicators if indicator in content.lower())
        
        # Вычисляем базовую оценку
        base_score = 0.5 + 0.3 * min(1.0, logical_count / 5) - 0.2 * min(1.0, emotional_count / 3)
        
        # Нормализуем
        return max(0.0, min(1.0, base_score))
    
    def _generate_credibility_warnings(self, source_rep: float, sentiment: Dict, 
                                     fact_errors: float, originality: float, 
                                     argument_quality: float) -> List[str]:
        """
        Генерирует предупреждения о проблемах с достоверностью.
        
        Args:
            source_rep: Репутация источника
            sentiment: Анализ тональности
            fact_errors: Уровень фактологических ошибок
            originality: Оценка оригинальности
            argument_quality: Качество аргументации
            
        Returns:
            List[str]: Список предупреждений
        """
        warnings = []
        
        # Анализируем репутацию источника
        if source_rep < 0.4:
            warnings.append("Источник имеет низкую репутацию. Рекомендуется проверить информацию из других источников.")
        elif source_rep < 0.6:
            warnings.append("Источник имеет среднюю репутацию. Рекомендуется дополнительная проверка информации.")
        
        # Анализируем тональность
        neutrality = 1.0 - abs(sentiment['compound'])
        if neutrality < 0.3:
            warnings.append("Высокая поляризация текста. Возможно, информация содержит предвзятость.")
        
        # Анализируем фактологические ошибки
        if fact_errors > 0.5:
            warnings.append("Высокий уровень фактологических ошибок. Требуется тщательная проверка информации.")
        elif fact_errors > 0.3:
            warnings.append("Умеренный уровень фактологических ошибок. Рекомендуется проверка ключевых утверждений.")
        
        # Анализируем оригинальность
        if originality < 0.4:
            warnings.append("Низкая оригинальность контента. Возможно, информация скопирована или автоматически сгенерирована.")
        
        # Анализируем качество аргументации
        if argument_quality < 0.4:
            warnings.append("Низкое качество аргументации. Требуется дополнительная проверка логической структуры.")
        
        return warnings