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
from cogniflex.distributed.database_utils import get_connection, execute_query  # предполагается, что get_connection возвращает sqlite3.Connection
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logger = logging.getLogger("cogniflex.contradiction")

# Инициализация NLP-ресурсов (безопасно — только если не загружены)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Подготовленные стоп-слова (кешируем)
try:
    _STOPWORDS = set(stopwords.words('english') + stopwords.words('russian'))
except Exception:
    _STOPWORDS = set()


class Contradiction:
    """Представляет обнаруженное противоречие в знаниях системы CogniFlex."""
    
    def __init__(self, contradiction_id: str, concept: str, 
                 conflicting_facts: List[Dict[str, Any]], 
                 divergence_level: float, timestamp: Optional[float] = None,
                 status: str = "detected", resolution: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        # Валидация входных данных
        if not contradiction_id or not isinstance(contradiction_id, str):
            raise ValueError("contradiction_id должен быть непустой строкой")
        if not concept or not isinstance(concept, str):
            raise ValueError("concept должен быть непустой строкой")
        if not conflicting_facts or len(conflicting_facts) < 2:
            raise ValueError("conflicting_facts должен содержать не менее двух фактов")
        if not isinstance(divergence_level, (int, float)) or not 0.0 <= divergence_level <= 1.0:
            raise ValueError("divergence_level должен быть числом в диапазоне [0.0, 1.0]")
        
        self.contradiction_id = contradiction_id
        self.concept = concept
        self.conflicting_facts = conflicting_facts
        self.divergence_level = float(divergence_level)
        self.timestamp = timestamp or time.time()
        self.status = status
        self.resolution = resolution or {}
        self.metadata = metadata or {}
        self.assigned_to = None
        self.resolved_at = None
        self.confidence = 0.0
        self.resolution_history = []
        self.resolution_confidence = 0.0
        self.impact_score = 0.0
        self.resolution_priority = 0.0
        self.source_analysis = {}
        self.nlp_metrics = {}
        
        # Пополняем nlp-метрики при создании
        self._analyze_facts()
        
        # Добавлено для совместимости с GUI
        self.severity = self._calculate_severity()
        
        logger.debug(f"Создано противоречие: ID={contradiction_id}, concept={concept}, divergence={divergence_level}, severity={self.severity}")

    def _calculate_severity(self) -> str:
        """Вычисляет уровень серьезности противоречия на основе divergence_level и impact_score."""
        # Определяем серьезность на основе divergence_level и impact_score
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
        result = {
            "contradiction_id": self.contradiction_id,
            "concept": self.concept,
            "conflicting_facts": self.conflicting_facts,
            "divergence_level": self.divergence_level,
            "timestamp": self.timestamp,
            "status": self.status,
            "resolution": self.resolution,
            "metadata": self.metadata,
            "assigned_to": self.assigned_to,
            "resolved_at": self.resolved_at,
            "confidence": self.confidence,
            "resolution_history": self.resolution_history,
            "resolution_confidence": self.resolution_confidence,
            "impact_score": self.impact_score,
            "resolution_priority": self.resolution_priority,
            "source_analysis": self.source_analysis,
            "nlp_metrics": self.nlp_metrics,
            # Добавлено для совместимости с GUI
            "severity": self.severity
        }
        
        # Добавляем вычисляемые поля для удобства
        result["type"] = self.get_contradiction_type()
        result["summary"] = self.get_resolution_summary()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Contradiction':
        try:
            if not isinstance(data, dict):
                raise ValueError("Данные должны быть словарем")
            if "contradiction_id" not in data:
                raise ValueError("Отсутствует ключ 'contradiction_id' в данных")
            if "concept" not in data:
                raise ValueError("Отсутствует ключ 'concept' в данных")
            if "conflicting_facts" not in data or not isinstance(data["conflicting_facts"], list):
                raise ValueError("Некорректный формат 'conflicting_facts'")

            # Извлекаем severity, если он есть, или вычисляем
            severity = data.get("severity", "low")
            
            instance = cls(
                contradiction_id=data["contradiction_id"],
                concept=data["concept"],
                conflicting_facts=data["conflicting_facts"],
                divergence_level=data.get("divergence_level", 0.0),
                timestamp=data.get("timestamp", time.time()),
                status=data.get("status", "detected"),
                resolution=data.get("resolution", {}),
                metadata=data.get("metadata", {})
            )
            
            # Восстанавливаем дополнительные атрибуты
            instance.assigned_to = data.get("assigned_to")
            instance.resolved_at = data.get("resolved_at")
            instance.confidence = data.get("confidence", 0.0)
            instance.resolution_history = data.get("resolution_history", [])
            instance.resolution_confidence = data.get("resolution_confidence", 0.0)
            instance.impact_score = data.get("impact_score", 0.0)
            instance.resolution_priority = data.get("resolution_priority", 0.0)
            instance.source_analysis = data.get("source_analysis", {})
            instance.nlp_metrics = data.get("nlp_metrics", {})
            
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
        return self.resolution_history

    def calculate_resolution_confidence(self) -> float:
        if not self.resolution_history:
            return 0.0
        total_weight = 0.0
        weighted_confidence = 0.0
        now = time.time()
        for entry in self.resolution_history:
            time_diff = now - entry["timestamp"]
            weight = max(0.0, 1.0 - time_diff / (86400 * 7))  # большая важность для свежих записей (7 дней)
            weighted_confidence += entry["confidence"] * weight
            total_weight += weight
        return weighted_confidence / total_weight if total_weight > 0 else 0.0

    def is_resolved(self) -> bool:
        return self.status == "resolved"

    def get_resolution_summary(self) -> Dict[str, Any]:
        summary = {
            "contradiction_id": self.contradiction_id,
            "concept": self.concept,
            "status": self.status,
            "resolution_steps": len(self.resolution_history),
            "divergence_level": self.divergence_level,
            "last_attempt": max((entry["timestamp"] for entry in self.resolution_history), default=None),
            "resolution_confidence": self.resolution_confidence,
            # Добавлено для совместимости
            "severity": self.severity
        }
        if self.is_resolved():
            summary.update({
                "resolved_at": self.resolved_at,
                "final_resolution": self.resolution
            })
        return summary

    def get_contradiction_type(self) -> str:
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
            if self.metadata["relation_type"].startswith("only_") or self.metadata["relation_type"].startswith("not_only_"):
                return "exclusivity_conflict"
            if self.metadata["relation_type"] in ["is_a", "part_of", "member_of"]:
                return "hierarchy_conflict"
        return "unknown"

    def update_impact_score(self, new_score: float):
        """Обновляет оценку влияния противоречия."""
        if 0.0 <= new_score <= 10.0:
            self.impact_score = new_score
            self.severity = self._calculate_severity()
            logger.debug(f"Обновлен impact_score для противоречия {self.contradiction_id}: {new_score}")
        else:
            logger.warning(f"Некорректный impact_score: {new_score}. Должен быть в [0.0, 10.0]")


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
        """Инициализация хранилища: создаёт файл БД (если указан cache_dir) и таблицу (если нужно).
        Также — делает попытку добавить колонку `data`, если её нет (безопасный ALTER)."""
        try:
            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                self.storage_path = os.path.join(self.cache_dir, "contradictions.db")
                conn = get_connection(self.storage_path)
                # Убедимся, что режим row_factory установлен для удобства
                try:
                    conn.row_factory = sqlite3.Row
                except Exception:
                    pass

                # Создаём таблицу, если её нет (минимальная схема)
                execute_query(conn, """
                    CREATE TABLE IF NOT EXISTS contradictions (
                        id TEXT PRIMARY KEY,
                        data TEXT
                    )
                """)

                # Проверим схему: есть ли колонка 'data'; если нет — попробуем добавить
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(contradictions)")
                cols_info = cursor.fetchall()
                cols = [row[1] if isinstance(row, (list, tuple)) else row["name"] for row in cols_info]
                logger.debug(f"Колонки в таблице contradictions: {cols}")

                if "data" not in cols:
                    # Попытка добавить колонку data (ADD COLUMN безопасен — добавляет NULL для старых строк)
                    try:
                        logger.info("Колонка 'data' отсутствует в таблице contradictions — пытаемся добавить")
                        cursor.execute("ALTER TABLE contradictions ADD COLUMN data TEXT")
                        conn.commit()
                        logger.info("Колонка 'data' успешно добавлена в таблицу contradictions")
                    except sqlite3.OperationalError as e:
                        logger.warning(f"Не удалось добавить колонку 'data' (ALTER TABLE): {e}")

                cursor.close()
                conn.close()
            else:
                self.storage_path = None
            # Загружаем существующие противоречия (если есть)
            self._load_contradictions()
        except Exception as e:
            logger.error(f"Ошибка инициализации хранилища противоречий: {e}", exc_info=True)
            self.storage_path = None

    def _row_to_mapping(self, cursor: sqlite3.Cursor, row: Tuple) -> Dict[str, Any]:
        """Преобразует sqlite row/tuple в словарь с именами колонок (если доступно)."""
        try:
            colnames = [d[0] for d in cursor.description]
            return {colnames[i]: row[i] for i in range(len(colnames))}
        except Exception:
            # Фоллбек — если нет description, вернём пустой словарь
            return {}

    def _load_contradictions(self):
        """Загружает противоречия из БД с учётом разных схем."""
        if not self.storage_path or not os.path.exists(self.storage_path):
            logger.debug("Хранилище противоречий не существует")
            return
        try:
            conn = get_connection(self.storage_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # Проверим, какие колонки есть
            cur.execute("PRAGMA table_info(contradictions)")
            cols_info = cur.fetchall()
            cols = []
            for r in cols_info:
                # r может быть sqlite3.Row или tuple
                if isinstance(r, (list, tuple)):
                    cols.append(r[1])
                elif isinstance(r, dict) or hasattr(r, "keys"):
                    # sqlite3.Row поддерживает индекс и ключи
                    try:
                        cols.append(r["name"])
                    except Exception:
                        # fallback: try common positions
                        try:
                            cols.append(r[1])
                        except Exception:
                            pass
            logger.debug(f"Прочитаны колонки contradictions: {cols}")

            # Попробуем выбрать id и data (если есть)
            if "data" in cols:
                try:
                    cur.execute("SELECT id, data FROM contradictions")
                    rows = cur.fetchall()
                    loaded = 0
                    for row in rows:
                        # row может быть sqlite3.Row (доступ по ключу) или tuple
                        try:
                            raw = row["data"]
                        except Exception:
                            try:
                                raw = row[1]
                            except Exception:
                                raw = None
                        if not raw:
                            logger.debug(f"Строка id={row['id'] if 'id' in row.keys() else (row[0] if isinstance(row, (list, tuple)) else 'unknown')} имеет пустое поле data — пропускаем")
                            continue
                        try:
                            contradiction_data = json.loads(raw)
                        except Exception as e:
                            logger.warning(f"Не удалось распарсить JSON в поле data для id={row['id'] if 'id' in row.keys() else 'unknown'}: {e}")
                            continue
                        # Создаем объект Contradiction
                        try:
                            contradiction = Contradiction.from_dict(contradiction_data)
                            self.contradictions[contradiction.contradiction_id] = contradiction
                            loaded += 1
                        except Exception as e:
                            logger.warning(f"Не удалось создать Contradiction из данных для id {row.get('id', 'unknown')}: {e}")
                    logger.info(f"Загружено {loaded} противоречий из {self.storage_path} (поле data)")
                    cur.close()
                    conn.close()
                    return
                except Exception as e:
                    logger.warning(f"Ошибка при SELECT id,data: {e} — попробуем общий SELECT *", exc_info=True)

            # Если мы дошли сюда — либо data нет, либо SELECT id,data упал => делаем общий SELECT *
            try:
                cur.execute("SELECT * FROM contradictions")
                rows = cur.fetchall()
                loaded = 0
                for row in rows:
                    # Преобразуем row в mapping
                    if isinstance(row, sqlite3.Row):
                        mapping = dict(row)
                    else:
                        mapping = self._row_to_mapping(cur, row)

                    # Если есть поле 'data' и оно непустое, парсим его
                    if "data" in mapping and mapping["data"]:
                        try:
                            contradiction_data = json.loads(mapping["data"])
                        except Exception:
                            contradiction_data = None
                    else:
                        # Попробуем реконструировать структуру: возьмём все колонки кроме id и сложим их в словарь
                        # Ожидаем, что некоторые поля (conflicting_facts) могут быть представлены как JSON в отдельных колонках
                        contradiction_data = {}
                        # если в таблице есть колонка 'contradiction_id' или 'id' используем её
                        if "contradiction_id" in mapping:
                            contradiction_data["contradiction_id"] = mapping["contradiction_id"]
                        elif "id" in mapping:
                            # предполагаем, что id совпадает с contradiction_id
                            contradiction_data["contradiction_id"] = mapping["id"]
                        # Попробуем найти концепт
                        if "concept" in mapping:
                            contradiction_data["concept"] = mapping["concept"]
                        # Попробуем найти conflicting_facts — если есть колонка, попробуем распарсить JSON или сделать список из value/text колонок
                        if "conflicting_facts" in mapping and mapping["conflicting_facts"]:
                            try:
                                contradiction_data["conflicting_facts"] = json.loads(mapping["conflicting_facts"])
                            except Exception:
                                contradiction_data["conflicting_facts"] = mapping["conflicting_facts"]
                        else:
                            # Ищем колонки, которые выглядят как факт (например fact_1, fact_2, или value_1)
                            # В общем случае — если нет явных полей, пропускаем такую строку
                            possible_facts = []
                            for k, v in mapping.items():
                                if k in ("id", "contradiction_id", "data"):
                                    continue
                                # если значение похоже на JSON-структуру списка/словаря — попробуем распарсить
                                if isinstance(v, str) and (v.strip().startswith("[") or v.strip().startswith("{")):
                                    try:
                                        parsed = json.loads(v)
                                        # Если это список фактов — используем
                                        if isinstance(parsed, list):
                                            possible_facts.extend(parsed)
                                        elif isinstance(parsed, dict):
                                            possible_facts.append(parsed)
                                        continue
                                    except Exception:
                                        pass
                            if possible_facts:
                                contradiction_data.setdefault("conflicting_facts", possible_facts)

                        # Попробуем взять divergence_level, timestamp, status, metadata
                        for field in ("divergence_level", "timestamp", "status", "resolution", "metadata"):
                            if field in mapping and mapping[field] is not None:
                                val = mapping[field]
                                # Если val - строка, попробуем распарсить JSON
                                if isinstance(val, str):
                                    try:
                                        parsed = json.loads(val)
                                        contradiction_data[field] = parsed
                                    except Exception:
                                        # Для divergence_level/timestamp — попробуем привести к числу
                                        if field in ("divergence_level", "timestamp"):
                                            try:
                                                contradiction_data[field] = float(val)
                                            except Exception:
                                                contradiction_data[field] = val
                                        else:
                                            contradiction_data[field] = val
                                else:
                                    contradiction_data[field] = val

                    # Если у нас есть как минимум contradiction_id, concept и conflicting_facts — пытаемся создать
                    if contradiction_data and "contradiction_id" in contradiction_data and "concept" in contradiction_data and "conflicting_facts" in contradiction_data:
                        try:
                            contradiction = Contradiction.from_dict(contradiction_data)
                            self.contradictions[contradiction.contradiction_id] = contradiction
                            loaded += 1
                        except Exception as e:
                            logger.warning(f"Не удалось создать Contradiction из реконструированных данных (row id={mapping.get('id','?')}): {e}")
                    else:
                        logger.debug(f"Пропущена строка contradictions (не удалось извлечь поля): {mapping.keys()}")
                logger.info(f"Загружено {loaded} противоречий из {self.storage_path} (SELECT *)")
                cur.close()
                conn.close()
                return
            except Exception as e:
                logger.error(f"Ошибка при SELECT * FROM contradictions: {e}", exc_info=True)
                try:
                    cur.close()
                    conn.close()
                except Exception:
                    pass
                return

        except Exception as e:
            logger.error(f"Ошибка загрузки противоречий: {e}", exc_info=True)
            return

    def _save_contradictions(self):
        """Сохраняет все текущие противоречия в БД. Работает аккуратно с схемой."""
        if not self.storage_path:
            return
        try:
            conn = get_connection(self.storage_path)
            cur = conn.cursor()
            # Убедимся, что таблица существует (с колонкой data)
            cur.execute("PRAGMA table_info(contradictions)")
            cols_info = cur.fetchall()
            cols = [row[1] if isinstance(row, (list, tuple)) else row["name"] for row in cols_info]
            if "data" not in cols:
                # Попытка добавить колонку, если это возможно
                try:
                    cur.execute("ALTER TABLE contradictions ADD COLUMN data TEXT")
                    conn.commit()
                    logger.info("Добавлена колонка 'data' в таблицу contradictions при сохранении")
                except Exception as e:
                    logger.warning(f"Не удалось добавить колонку data при сохранении: {e}")

            # Сохранение через INSERT OR REPLACE
            for contradiction in self.contradictions.values():
                try:
                    data = json.dumps(contradiction.to_dict(), ensure_ascii=False)
                    # Используем параметры запроса
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
        if not self.initialized:
            logger.error("Детектор не инициализирован")
            return
        logger.info("OptimizedContradictionDetector запущен")

    def stop(self):
        # перед остановкой сохраним всё, что есть
        self._save_contradictions()
        logger.info("OptimizedContradictionDetector остановлен")

    def detect_contradiction(self, concept: str, facts: List[Dict[str, Any]], 
                            metadata: Optional[Dict] = None) -> Optional[Contradiction]:
        """Простейшая детекция: числовой конфликт или булев конфликт."""
        try:
            if len(facts) < 2:
                logger.debug(f"Недостаточно фактов для обнаружения противоречия: {len(facts)}")
                return None

            # Собираем числовые значения фактов
            values = [fact["value"] for fact in facts if "value" in fact and isinstance(fact.get("value"), (int, float))]

            # Числовой конфликт
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

            # Булев конфликт
            if all(isinstance(fact.get("value"), bool) for fact in facts) and len(set(fact.get("value") for fact in facts)) > 1:
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

            # Можно добавить дополнительные правила
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
        
        # Собираем статистику по серьезности
        by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        by_domain = {}
        for c in self.contradictions.values():
            # Определяем домен
            domain = c.metadata.get("domain", "general") if c.metadata else "general"
            if domain not in by_domain:
                by_domain[domain] = 0
            by_domain[domain] += 1
            
            # Считаем по серьезности
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


if __name__ == "__main__":
    # Короткий самотест
    detector = OptimizedContradictionDetector(cache_dir="cogniflex_cache")
    contradiction = detector.detect_contradiction(
        concept="test_concept",
        facts=[
            {"value": 10, "text": "Значение равно 10"},
            {"value": 20, "text": "Значение равно 20"}
        ],
        metadata={"type": "numeric_conflict"}
    )
    if contradiction:
        print(contradiction.to_dict())
        detector.resolve_contradiction(
            contradiction_id=contradiction.contradiction_id,
            resolution={"accepted_value": 15, "reason": "Среднее значение"},
            resolver="test_resolver",
            confidence=0.8
        )
        print(detector.get_contradiction_summary())
        print(detector.get_contradiction_statistics())