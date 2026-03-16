"""
Модуль управления противоречиями для системы CogniFlex.
Улучшенная и более робастная версия: учитывает возможную несовместимость схемы БД
(например, отсутствие колонки `data`) и корректно загружает/сохраняет данные.
"""
import logging
import time
import os
import json
import nltk
import sqlite3
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from datetime import datetime

try:
    from cogniflex.distributed.database_utils import get_connection, execute_query
except ImportError:
    get_connection = None
    execute_query = None

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logger = logging.getLogger("cogniflex.contradiction")

# Инициализация NLP-ресурсов (неблокирующая — только если не загружены)
def _safe_nltk_download(resource, name):
    """Безопасная загрузка NLTK ресурса без блокировки."""
    try:
        nltk.data.find(resource)
        return True
    except LookupError:
        try:
            # Попытка загрузки с таймаутом
            import threading
            import time

            result = [False]
            def download_worker():
                try:
                    nltk.download(name, quiet=True)
                    result[0] = True
                except Exception:
                    pass

            thread = threading.Thread(target=download_worker, daemon=True)
            thread.start()
            thread.join(timeout=5.0)  # Таймаут 5 секунд
            return result[0]
        except Exception:
            return False

# Попытка загрузки ресурсов NLTK (неблокирующая)
_safe_nltk_download('tokenizers/punkt', 'punkt')
_safe_nltk_download('corpora/stopwords', 'stopwords')
_safe_nltk_download('sentiment/vader_lexicon', 'vader_lexicon')

# Подготовленные стоп-слова (кешируем)
try:
    _STOPWORDS = set(stopwords.words('english') + stopwords.words('russian'))
except Exception:
    _STOPWORDS = set()


# ============================================================================
# Классы данных
# ============================================================================

@dataclass
class Contradiction:
    """Представляет обнаруженное противоречие в знаниях системы CogniFlex."""
    contradiction_id: str
    concept: str
    conflicting_facts: List[Dict[str, Any]]
    divergence_level: float
    timestamp: float = field(default_factory=time.time)
    status: str = "detected"
    resolution: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    assigned_to: Optional[str] = None
    resolved_at: Optional[float] = None
    confidence: float = 0.0
    resolution_history: List[Dict[str, Any]] = field(default_factory=list)
    resolution_confidence: float = 0.0
    impact_score: float = 0.0
    resolution_priority: float = 0.0
    source_analysis: Dict[str, Any] = field(default_factory=dict)
    nlp_metrics: Dict[str, Any] = field(default_factory=dict)
    severity: str = "low"
    
    def __post_init__(self):
        """Инициализация после создания объекта."""
        # Пополняем nlp-метрики при создании
        self._analyze_facts()
        # Вычисляем severity
        self.severity = self._calculate_severity()
        logger.debug(f"Создано противоречие: ID={self.contradiction_id}, concept={self.concept}, "
                    f"divergence={self.divergence_level}, severity={self.severity}")
    
    def _calculate_severity(self) -> str:
        """Вычисляет уровень серьезности противоречия на основе divergence_level и impact_score."""
        combined_score = (self.divergence_level + (self.impact_score / 10.0)) / 2.0
        if combined_score >= 0.7:
            return "critical"
        elif combined_score >= 0.5:
            return "high"
        elif combined_score >= 0.3:
            return "medium"
        else:
            return "low"
    
    def _analyze_facts(self):
        """Простейший NLP-анализ фактов для дополнительной метрики."""
        try:
            sia = SentimentIntensityAnalyzer()
            for idx, fact in enumerate(self.conflicting_facts):
                text = fact.get("text", fact.get("value", ""))
                if isinstance(text, (int, float, bool)):
                    text = str(text)
                if isinstance(text, str) and text.strip():
                    sentiment = sia.polarity_scores(text)
                    tokens = word_tokenize(text.lower())
                    keywords = [word for word in tokens if word.isalnum() and word not in _STOPWORDS]
                    self.nlp_metrics[f"fact_{idx}"] = {
                        "sentiment": sentiment,
                        "keywords": keywords[:5]
                    }
        except Exception as e:
            logger.warning(f"Ошибка NLP-анализа фактов: {e}")
            self.nlp_metrics["error"] = str(e)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для сериализации."""
        result = asdict(self)
        result["type"] = self.get_contradiction_type()
        result["summary"] = self.get_resolution_summary()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Contradiction':
        """Создание из словаря."""
        try:
            if not isinstance(data, dict):
                raise ValueError("Данные должны быть словарем")
            
            required_fields = ["contradiction_id", "concept", "conflicting_facts"]
            for field_name in required_fields:
                if field_name not in data:
                    raise ValueError(f"Отсутствует ключ '{field_name}' в данных")
            
            if not isinstance(data["conflicting_facts"], list) or len(data["conflicting_facts"]) < 2:
                raise ValueError("conflicting_facts должен содержать не менее двух фактов")
            
            divergence = data.get("divergence_level", 0.0)
            if not isinstance(divergence, (int, float)) or not 0.0 <= divergence <= 1.0:
                divergence = 0.0
            
            instance = cls(
                contradiction_id=data["contradiction_id"],
                concept=data["concept"],
                conflicting_facts=data["conflicting_facts"],
                divergence_level=divergence,
                timestamp=data.get("timestamp", time.time()),
                status=data.get("status", "detected"),
                resolution=data.get("resolution", {}),
                metadata=data.get("metadata", {})
            )
            
            # Восстанавливаем дополнительные атрибуты
            for attr in ["assigned_to", "resolved_at", "confidence", "resolution_confidence", 
                        "impact_score", "resolution_priority"]:
                if attr in data:
                    setattr(instance, attr, data[attr])
            
            for attr in ["resolution_history", "source_analysis", "nlp_metrics"]:
                if attr in data and isinstance(data[attr], dict):
                    setattr(instance, attr, data[attr])
            
            # Устанавливаем severity из данных, если он есть
            if "severity" in data:
                instance.severity = data["severity"]
            else:
                instance.severity = instance._calculate_severity()
            
            return instance
        except Exception as e:
            logger.error(f"Ошибка создания Contradiction из словаря: {e}", exc_info=True)
            raise
    
    def update_status(self, new_status: str, resolution: Optional[Dict[str, Any]] = None):
        """Обновление статуса противоречия."""
        valid_statuses = {"detected", "pending", "resolved"}
        if new_status not in valid_statuses:
            logger.warning(f"Некорректный статус: {new_status}. Допустимые: {valid_statuses}")
            return
        self.status = new_status
        if resolution:
            self.resolution = resolution
            self.resolved_at = time.time()
        logger.info(f"Противоречие {self.contradiction_id} обновлено: статус={new_status}")
    
    def add_resolution_history(self, resolver: str, resolution: Dict[str, Any],
                              confidence: float, nlp_metrics: Optional[Dict] = None):
        """Добавление записи в историю разрешения."""
        if not 0.0 <= confidence <= 1.0:
            logger.warning(f"Некорректная уверенность: {confidence}. Должна быть в [0.0, 1.0]")
            return
        entry = {
            "resolver": resolver,
            "timestamp": time.time(),
            "resolution": resolution,
            "confidence": confidence,
            "nlp_metrics": nlp_metrics or {}
        }
        self.resolution_history.append(entry)
        self.resolution_confidence = self.calculate_resolution_confidence()
        logger.debug(f"Добавлена запись в историю разрешения для {self.contradiction_id}: {entry}")
    
    def get_resolution_history(self) -> List[Dict[str, Any]]:
        """Получение истории разрешения."""
        return self.resolution_history
    
    def calculate_resolution_confidence(self) -> float:
        """Расчет уверенности разрешения на основе истории."""
        if not self.resolution_history:
            return 0.0
        total_weight = 0.0
        weighted_confidence = 0.0
        now = time.time()
        for entry in self.resolution_history:
            time_diff = now - entry["timestamp"]
            weight = max(0.0, 1.0 - time_diff / (86400 * 7))  # 7 дней
            weighted_confidence += entry["confidence"] * weight
            total_weight += weight
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def is_resolved(self) -> bool:
        """Проверка, разрешено ли противоречие."""
        return self.status == "resolved"
    
    def get_resolution_summary(self) -> Dict[str, Any]:
        """Получение краткой сводки разрешения."""
        summary = {
            "contradiction_id": self.contradiction_id,
            "concept": self.concept,
            "status": self.status,
            "resolution_steps": len(self.resolution_history),
            "divergence_level": self.divergence_level,
            "last_attempt": max((entry["timestamp"] for entry in self.resolution_history), default=None),
            "resolution_confidence": self.resolution_confidence,
            "severity": self.severity
        }
        if self.is_resolved():
            summary.update({
                "resolved_at": self.resolved_at,
                "final_resolution": self.resolution
            })
        return summary
    
    def get_contradiction_type(self) -> str:
        """Определение типа противоречия."""
        if "type" in self.metadata:
            return self.metadata["type"]
        if len(self.conflicting_facts) >= 2:
            fact1 = self.conflicting_facts[0]
            fact2 = self.conflicting_facts[1]
            if isinstance(fact1.get("value"), (int, float)) and isinstance(fact2.get("value"), (int, float)):
                return "numeric_conflict"
            if isinstance(fact1.get("value"), bool) and isinstance(fact2.get("value"), bool):
                return "boolean_conflict"
            if "response" in fact1.get("relation", "") and "response" in fact2.get("relation", ""):
                return "response_conflict"
        if "relation_type" in self.metadata:
            rt = self.metadata["relation_type"]
            if rt.startswith("only_") or rt.startswith("not_only_"):
                return "exclusivity_conflict"
            if rt in ["is_a", "part_of", "member_of"]:
                return "hierarchy_conflict"
        return "unknown"
    
    def update_impact_score(self, new_score: float):
        """Обновление оценки влияния противоречия."""
        if 0.0 <= new_score <= 10.0:
            self.impact_score = new_score
            self.severity = self._calculate_severity()
            logger.debug(f"Обновлен impact_score для противоречия {self.contradiction_id}: {new_score}")
        else:
            logger.warning(f"Некорректный impact_score: {new_score}. Должен быть в [0.0, 10.0]")


# ============================================================================
# Детектор противоречий
# ============================================================================

class OptimizedContradictionDetector:
    """Класс для обнаружения и управления противоречиями в знаниях."""
    
    def __init__(self, knowledge_graph=None, brain=None, cache_dir: Optional[str] = None):
        self.knowledge_graph = knowledge_graph
        self.brain = brain
        self.cache_dir = cache_dir
        self.initialized = False
        self.contradictions: Dict[str, Contradiction] = {}
        self.storage_path: Optional[str] = None
        self._init_storage()
        self.initialized = True
        logger.info("OptimizedContradictionDetector инициализирован")
    
    def _init_storage(self):
        """Инициализация хранилища противоречий."""
        try:
            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                self.storage_path = os.path.join(self.cache_dir, "contradictions.db")
                
                if get_connection:
                    conn = get_connection(self.storage_path)
                    try:
                        conn.row_factory = sqlite3.Row
                    except Exception:
                        pass
                    
                    execute_query(conn, """
                        CREATE TABLE IF NOT EXISTS contradictions (
                            id TEXT PRIMARY KEY,
                            data TEXT
                        )
                    """)
                    
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA table_info(contradictions)")
                    cols_info = cursor.fetchall()
                    cols = [row[1] if isinstance(row, (list, tuple)) else row["name"] for row in cols_info]
                    logger.debug(f"Колонки в таблице contradictions: {cols}")
                    
                    if "data" not in cols:
                        try:
                            logger.info("Колонка 'data' отсутствует — пытаемся добавить")
                            cursor.execute("ALTER TABLE contradictions ADD COLUMN data TEXT")
                            conn.commit()
                            logger.info("Колонка 'data' успешно добавлена")
                        except sqlite3.OperationalError as e:
                            logger.warning(f"Не удалось добавить колонку 'data': {e}")
                    
                    cursor.close()
                    conn.close()
                else:
                    logger.warning("database_utils недоступен — хранилище будет работать в памяти")
                    self.storage_path = None
                
                self._load_contradictions()
            else:
                self.storage_path = None
        except Exception as e:
            logger.error(f"Ошибка инициализации хранилища противоречий: {e}", exc_info=True)
            self.storage_path = None
    
    def _row_to_mapping(self, cursor: sqlite3.Cursor, row: Tuple) -> Dict[str, Any]:
        """Преобразует sqlite row/tuple в словарь с именами колонок."""
        try:
            colnames = [d[0] for d in cursor.description]
            return {colnames[i]: row[i] for i in range(len(colnames))}
        except Exception:
            return {}
    
    def _load_contradictions(self):
        """Загружает противоречия из БД с учётом разных схем."""
        if not self.storage_path or not os.path.exists(self.storage_path):
            logger.debug("Хранилище противоречий не существует")
            return
        
        try:
            if not get_connection:
                logger.warning("database_utils недоступен — пропускаем загрузку из БД")
                return
            
            conn = get_connection(self.storage_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            
            cur.execute("PRAGMA table_info(contradictions)")
            cols_info = cur.fetchall()
            cols = []
            for r in cols_info:
                if isinstance(r, (list, tuple)):
                    cols.append(r[1])
                elif hasattr(r, "keys"):
                    try:
                        cols.append(r["name"])
                    except Exception:
                        try:
                            cols.append(r[1])
                        except Exception:
                            pass
            
            logger.debug(f"Прочитаны колонки contradictions: {cols}")
            loaded = 0
            
            if "data" in cols:
                try:
                    cur.execute("SELECT id, data FROM contradictions")
                    rows = cur.fetchall()
                    for row in rows:
                        try:
                            raw = row["data"]
                        except Exception:
                            try:
                                raw = row[1]
                            except Exception:
                                raw = None
                        
                        if not raw:
                            continue
                        
                        try:
                            contradiction_data = json.loads(raw)
                        except Exception as e:
                            logger.warning(f"Не удалось распарсить JSON: {e}")
                            continue
                        
                        try:
                            contradiction = Contradiction.from_dict(contradiction_data)
                            self.contradictions[contradiction.contradiction_id] = contradiction
                            loaded += 1
                        except Exception as e:
                            logger.warning(f"Не удалось создать Contradiction: {e}")
                    
                    logger.info(f"Загружено {loaded} противоречий из {self.storage_path} (поле data)")
                except Exception as e:
                    logger.warning(f"Ошибка при SELECT id,data: {e} — попробуем SELECT *", exc_info=True)
            
            if loaded == 0:
                try:
                    cur.execute("SELECT * FROM contradictions")
                    rows = cur.fetchall()
                    loaded = 0
                    for row in rows:
                        if isinstance(row, sqlite3.Row):
                            mapping = dict(row)
                        else:
                            mapping = self._row_to_mapping(cur, row)
                        
                        if "data" in mapping and mapping["data"]:
                            try:
                                contradiction_data = json.loads(mapping["data"])
                            except Exception:
                                contradiction_data = None
                        else:
                            contradiction_data = {}
                            if "contradiction_id" in mapping:
                                contradiction_data["contradiction_id"] = mapping["contradiction_id"]
                            elif "id" in mapping:
                                contradiction_data["contradiction_id"] = mapping["id"]
                            if "concept" in mapping:
                                contradiction_data["concept"] = mapping["concept"]
                            if "conflicting_facts" in mapping and mapping["conflicting_facts"]:
                                try:
                                    contradiction_data["conflicting_facts"] = json.loads(mapping["conflicting_facts"])
                                except Exception:
                                    contradiction_data["conflicting_facts"] = mapping["conflicting_facts"]
                        
                        if contradiction_data and "contradiction_id" in contradiction_data and \
                           "concept" in contradiction_data and "conflicting_facts" in contradiction_data:
                            try:
                                contradiction = Contradiction.from_dict(contradiction_data)
                                self.contradictions[contradiction.contradiction_id] = contradiction
                                loaded += 1
                            except Exception as e:
                                logger.warning(f"Не удалось создать Contradiction: {e}")
                    
                    logger.info(f"Загружено {loaded} противоречий из {self.storage_path} (SELECT *)")
                except Exception as e:
                    logger.error(f"Ошибка при загрузке противоречий: {e}", exc_info=True)
            
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"Ошибка загрузки противоречий: {e}", exc_info=True)
    
    def _save_contradictions(self):
        """Сохраняет все текущие противоречия в БД."""
        if not self.storage_path or not get_connection:
            return
        
        try:
            conn = get_connection(self.storage_path)
            cur = conn.cursor()
            
            cur.execute("PRAGMA table_info(contradictions)")
            cols_info = cur.fetchall()
            cols = [row[1] if isinstance(row, (list, tuple)) else row["name"] for row in cols_info]
            
            if "data" not in cols:
                try:
                    cur.execute("ALTER TABLE contradictions ADD COLUMN data TEXT")
                    conn.commit()
                    logger.info("Добавлена колонка 'data' в таблицу contradictions")
                except Exception as e:
                    logger.warning(f"Не удалось добавить колонку data: {e}")
            
            for contradiction in self.contradictions.values():
                try:
                    data = json.dumps(contradiction.to_dict(), ensure_ascii=False)
                    cur.execute("INSERT OR REPLACE INTO contradictions (id, data) VALUES (?, ?)",
                               (contradiction.contradiction_id, data))
                except Exception as e:
                    logger.error(f"Ошибка при сохранении противоречия {contradiction.contradiction_id}: {e}", exc_info=True)
                    continue
            
            conn.commit()
            cur.close()
            conn.close()
            logger.debug(f"Противоречия сохранены в {self.storage_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения противоречий: {e}", exc_info=True)
    
    def start(self):
        """Запуск детектора."""
        if not self.initialized:
            logger.error("Детектор не инициализирован")
            return
        logger.info("OptimizedContradictionDetector запущен")
    
    def stop(self):
        """Остановка детектора."""
        self._save_contradictions()
        logger.info("OptimizedContradictionDetector остановлен")
    
    def clear_all_contradictions(self) -> Dict[str, Any]:
        """Полностью очищает противоречия в памяти и в БД."""
        report: Dict[str, Any] = {
            "ok": True,
            "cleared": 0,
            "db_path": self.storage_path,
            "error": None,
        }
        try:
            count_before = len(self.contradictions)
            self.contradictions.clear()
            
            if self.storage_path and get_connection:
                try:
                    conn = get_connection(self.storage_path)
                    cur = conn.cursor()
                    cur.execute("CREATE TABLE IF NOT EXISTS contradictions (id TEXT PRIMARY KEY, data TEXT)")
                    cur.execute("DELETE FROM contradictions")
                    conn.commit()
                    cur.close()
                    conn.close()
                except Exception as e:
                    logger.warning(f"Ошибка очистки БД противоречий: {e}", exc_info=True)
                    report["ok"] = False
                    report["error"] = str(e)
            
            report["cleared"] = count_before
            logger.info(f"Очищено противоречий: {count_before}; storage={self.storage_path}")
            return report
        except Exception as e:
            logger.error(f"Сбой при очистке противоречий: {e}", exc_info=True)
            report["ok"] = False
            report["error"] = str(e)
            return report
    
    def detect_contradiction(self, concept: str, facts: List[Dict[str, Any]],
                            metadata: Optional[Dict] = None) -> Optional[Contradiction]:
        """Простейшая детекция: числовой конфликт или булев конфликт."""
        try:
            if len(facts) < 2:
                logger.debug(f"Недостаточно фактов для обнаружения противоречия: {len(facts)}")
                return None
            
            values = [fact["value"] for fact in facts if "value" in fact and isinstance(fact.get("value"), (int, float))]
            
            if len(set(values)) > 1:
                contradiction_id = f"contradiction_{int(time.time())}_{hash(concept) & 0xFFFFFFFF}"
                divergence = max(values) - min(values)
                max_abs = max(abs(v) for v in values)
                divergence_level = min(1.0, divergence / max(max_abs, 1.0))
                contradiction = Contradiction(
                    contradiction_id=contradiction_id,
                    concept=concept,
                    conflicting_facts=facts,
                    divergence_level=divergence_level,
                    metadata=metadata or {}
                )
                self.contradictions[contradiction_id] = contradiction
                self._save_contradictions()
                logger.info(f"Обнаружено противоречие: {contradiction_id} для концепта {concept}")
                return contradiction
            
            if all(isinstance(fact.get("value"), bool) for fact in facts) and \
               len(set(fact.get("value") for fact in facts)) > 1:
                contradiction_id = f"contradiction_{int(time.time())}_{hash(concept) & 0xFFFFFFFF}"
                contradiction = Contradiction(
                    contradiction_id=contradiction_id,
                    concept=concept,
                    conflicting_facts=facts,
                    divergence_level=1.0,
                    metadata=metadata or {}
                )
                self.contradictions[contradiction_id] = contradiction
                self._save_contradictions()
                logger.info(f"Обнаружено булево противоречие: {contradiction_id} для концепта {concept}")
                return contradiction
            
            return None
        except Exception as e:
            logger.error(f"Ошибка обнаружения противоречия для концепта {concept}: {e}", exc_info=True)
            return None
    
    def get_active_contradictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Возвращает активные (неразрешенные) противоречия."""
        active = [c.to_dict() for c in self.contradictions.values() if not c.is_resolved()]
        return sorted(active, key=lambda x: x.get("resolution_priority", 0.0), reverse=True)[:limit]
    
    def get_detected_contradictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Совместимый метод для старого API - возвращает активные противоречия."""
        return self.get_active_contradictions(limit)
    
    def get_contradiction_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по противоречиям."""
        total = len(self.contradictions)
        resolved = sum(1 for c in self.contradictions.values() if c.is_resolved())
        active = total - resolved
        
        by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        by_domain = {}
        
        for c in self.contradictions.values():
            domain = c.metadata.get("domain", "general") if c.metadata else "general"
            if domain not in by_domain:
                by_domain[domain] = 0
            by_domain[domain] += 1
            
            severity = c.severity
            if severity in by_severity:
                by_severity[severity] += 1
        
        return {
            "total": total,
            "resolved": resolved,
            "active": active,
            "by_severity": by_severity,
            "by_domain": by_domain,
            "contradictions": [c.to_dict() for c in self.contradictions.values()]
        }
    
    def get_contradiction_summary(self) -> Dict[str, int]:
        """Возвращает краткую сводку по противоречиям."""
        total = len(self.contradictions)
        resolved = sum(1 for c in self.contradictions.values() if c.is_resolved())
        return {
            "total": total,
            "resolved": resolved,
            "active": total - resolved
        }
    
    def resolve_contradiction(self, contradiction_id: str, resolution: Dict[str, Any],
                             resolver: str, confidence: float) -> bool:
        """Разрешает противоречие с указанным ID."""
        try:
            if contradiction_id not in self.contradictions:
                logger.error(f"Противоречие {contradiction_id} не найдено")
                return False
            
            contradiction = self.contradictions[contradiction_id]
            contradiction.add_resolution_history(resolver, resolution, confidence)
            contradiction.update_status("resolved", resolution)
            self._save_contradictions()
            logger.info(f"Противоречие {contradiction_id} разрешено: {resolution}")
            return True
        except Exception as e:
            logger.error(f"Ошибка разрешения противоречия {contradiction_id}: {e}", exc_info=True)
            return False
    
    def get_all_contradictions(self) -> List[Dict[str, Any]]:
        """Возвращает все противоречия (активные и разрешенные)."""
        return [c.to_dict() for c in self.contradictions.values()]


# ============================================================================
# Экспорт для совместимости
# ============================================================================

__all__ = [
    'Contradiction',
    'OptimizedContradictionDetector'
]