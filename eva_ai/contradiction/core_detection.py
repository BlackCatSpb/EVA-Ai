"""Core detection algorithms, rule-based checks."""
import logging
import time
import os
import json
import sqlite3
import nltk
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from datetime import datetime

try:
    from eva_ai.distributed.database_utils import get_connection, execute_query
except ImportError:
    get_connection = None
    execute_query = None

try:
    from eva_ai.knowledge.context_entity import EntityExtractor, AmbiguousEntity, AmbiguityType
except ImportError:
    EntityExtractor = None
    AmbiguousEntity = None
    AmbiguityType = None

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from .core_resolution import ResolutionMixin
from .core_tracking import TrackingMixin

logger = logging.getLogger("eva_ai.contradiction.core")


def _safe_nltk_download(resource, name):
    """Безопасная загрузка NLTK ресурса без блокировки."""
    try:
        nltk.data.find(resource)
        return True
    except LookupError:
        try:
            import threading
            result = [False]
            def download_worker():
                try:
                    nltk.download(name, quiet=True)
                    result[0] = True
                except Exception:
                    pass
            thread = threading.Thread(target=download_worker, daemon=True)
            thread.start()
            thread.join(timeout=5.0)
            return result[0]
        except Exception:
            return False


_safe_nltk_download('tokenizers/punkt', 'punkt')
_safe_nltk_download('corpora/stopwords', 'stopwords')
_safe_nltk_download('sentiment/vader_lexicon', 'vader_lexicon')

try:
    _STOPWORDS = set(stopwords.words('english') + stopwords.words('russian'))
except Exception:
    _STOPWORDS = set()


@dataclass
class Contradiction:
    """Представляет обнаруженное противоречие в знаниях системы ЕВА."""
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
        self._analyze_facts()
        self.severity = self._calculate_severity()
        logger.debug(f"Создано противоречие: ID={self.contradiction_id}, concept={self.concept}, "
                    f"divergence={self.divergence_level}, severity={self.severity}")
    
    def _calculate_severity(self) -> str:
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
        result = asdict(self)
        result["type"] = self.get_contradiction_type()
        result["summary"] = self.get_resolution_summary()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Contradiction':
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
            for attr in ["assigned_to", "resolved_at", "confidence", "resolution_confidence",
                        "impact_score", "resolution_priority"]:
                if attr in data:
                    setattr(instance, attr, data[attr])
            for attr in ["resolution_history", "source_analysis", "nlp_metrics"]:
                if attr in data and isinstance(data[attr], dict):
                    setattr(instance, attr, data[attr])
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
            "resolver": resolver, "timestamp": time.time(),
            "resolution": resolution, "confidence": confidence,
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
            weight = max(0.0, 1.0 - time_diff / (86400 * 7))
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
            "severity": self.severity
        }
        if self.is_resolved():
            summary.update({
                "resolved_at": self.resolved_at,
                "final_resolution": self.resolution
            })
        return summary
    
    def get_contradiction_type(self) -> str:
        metadata = getattr(self, 'metadata', None)
        if isinstance(metadata, dict) and "type" in metadata:
            return metadata["type"]
        if len(self.conflicting_facts) >= 2:
            fact1 = self.conflicting_facts[0]
            fact2 = self.conflicting_facts[1]
            if isinstance(fact1.get("value"), (int, float)) and isinstance(fact2.get("value"), (int, float)):
                return "numeric_conflict"
            if isinstance(fact1.get("value"), bool) and isinstance(fact2.get("value"), bool):
                return "boolean_conflict"
            if "response" in fact1.get("relation", "") and "response" in fact2.get("relation", ""):
                return "response_conflict"
        if isinstance(metadata, dict) and "relation_type" in metadata:
            rt = metadata["relation_type"]
            if rt.startswith("only_") or rt.startswith("not_only_"):
                return "exclusivity_conflict"
            if rt in ["is_a", "part_of", "member_of"]:
                return "hierarchy_conflict"
        return "unknown"
    
    def update_impact_score(self, new_score: float):
        if 0.0 <= new_score <= 10.0:
            self.impact_score = new_score
            self.severity = self._calculate_severity()
            logger.debug(f"Обновлен impact_score для противоречия {self.contradiction_id}: {new_score}")
        else:
            logger.warning(f"Некорректный impact_score: {new_score}. Должен быть в [0.0, 10.0]")


class CoreDetectionMixin:
    """Mixin providing core detection algorithms and rule-based checks."""
    
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
                    contradiction_id=contradiction_id, concept=concept,
                    conflicting_facts=facts, divergence_level=divergence_level,
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
                    contradiction_id=contradiction_id, concept=concept,
                    conflicting_facts=facts, divergence_level=1.0,
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
    
    def detect_with_context(self, query: str, response: str, context: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
        """Enhanced contradiction detection with context extraction."""
        contradictions = []
        clarifications_needed = []
        ambiguities = []
        
        if not self._entity_extractor:
            logger.warning("EntityExtractor not available, falling back to basic detection")
            return contradictions, clarifications_needed, ambiguities
        
        try:
            query_entities = self._entity_extractor.extract_ambiguous_terms(query)
            response_entities = self._entity_extractor.extract_ambiguous_terms(response)
            
            for entity in query_entities:
                if entity.needs_clarification:
                    resolved_in_response = False
                    for resp_entity in response_entities:
                        if resp_entity.text.lower() == entity.text.lower():
                            resolved_in_response = True
                            break
                    if not resolved_in_response:
                        ambiguities.append({
                            "term": entity.text,
                            "type": entity.ambiguity_type.value if hasattr(entity.ambiguity_type, 'value') else str(entity.ambiguity_type),
                            "context": entity.context,
                            "possible_meanings": entity.possible_meanings,
                            "confidence": entity.confidence,
                            "refinement_suggestion": entity.refinement_suggestion
                        })
                        clarification = self.generate_clarification_for_contradiction({
                            "term": entity.text,
                            "type": entity.ambiguity_type.value if hasattr(entity.ambiguity_type, 'value') else str(entity.ambiguity_type),
                            "context": entity.context
                        })
                        if clarification:
                            clarifications_needed.append(clarification)
            
            logger.debug(f"detect_with_context: found {len(ambiguities)} ambiguities, {len(clarifications_needed)} clarifications")
        except Exception as e:
            logger.error(f"Error in detect_with_context: {e}", exc_info=True)
        
        return contradictions, clarifications_needed, ambiguities
    
    def detect_ambiguity_conflicts(self, query: str, response: str) -> List[Dict[str, Any]]:
        """Extract ambiguous terms from query and response."""
        unresolved = []
        if not self._entity_extractor:
            logger.warning("EntityExtractor not available")
            return unresolved
        
        try:
            query_entities = self._entity_extractor.extract_ambiguous_terms(query)
            response_entities = self._entity_extractor.extract_ambiguous_terms(response)
            response_terms = {e.text.lower() for e in response_entities}
            
            for entity in query_entities:
                if entity.needs_clarification and entity.text.lower() not in response_terms:
                    unresolved.append({
                        "term": entity.text,
                        "ambiguity_type": entity.ambiguity_type.value if hasattr(entity.ambiguity_type, 'value') else str(entity.ambiguity_type),
                        "context": entity.context,
                        "possible_meanings": entity.possible_meanings,
                        "refinement_suggestion": entity.refinement_suggestion,
                        "confidence": entity.confidence
                    })
        except Exception as e:
            logger.error(f"Error in detect_ambiguity_conflicts: {e}", exc_info=True)
        
        return unresolved
    
    def generate_clarification_for_contradiction(self, contradiction: Dict[str, Any]) -> str:
        """Generate specific clarification questions for contradictions."""
        term = contradiction.get("term", "")
        term_type = contradiction.get("type", "unknown")
        context = contradiction.get("context", "")
        
        templates = {
            "vague_adjective": f"Вы использовали '{term}', но не уточнили степень. Что именно вы имеете в виду?",
            "vague_quantifier": f"Вы сказали '{term}', но не указали точное количество. Сколько именно?",
            "pronoun_reference": f"На что ссылается '{term}' в контексте '{context}'?",
            "demonstrative_reference": f"Что именно означает '{term}' в '{context}'?",
            "comparative_term": f"Вы упомянули '{term}', но не указали с чем сравниваете. По отношению к чему?",
            "implicit_subject": f"Что именно вы имеете в виду под '{term}' в '{context}'?",
            "temporal_vagueness": f"Когда именно происходит '{term}' в '{context}'?",
            "spatial_vagueness": f"Где именно находится '{term}' в '{context}'?"
        }
        
        return templates.get(term_type, f"Пожалуйста, уточните что вы имеете в виду под '{term}'")
    
    def find_contradictions(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Find contradictions in text with ambiguity detection."""
        contradictions_found = []
        if not self._entity_extractor:
            logger.warning("EntityExtractor not available for find_contradictions")
        return contradictions_found


class StorageMixin:
    """Mixin providing SQLite storage for contradictions."""
    
    def _init_storage(self):
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
        try:
            colnames = [d[0] for d in cursor.description]
            return {colnames[i]: row[i] for i in range(len(colnames))}
        except Exception:
            return {}
    
    def _load_contradictions(self):
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
    
    def clear_all_contradictions(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "ok": True, "cleared": 0, "db_path": self.storage_path, "error": None,
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


class ContradictionCore(StorageMixin, CoreDetectionMixin, ResolutionMixin, TrackingMixin):
    """Main class for contradiction management in EVA system."""
    
    def __init__(self, knowledge_graph=None, brain=None, cache_dir: Optional[str] = None, fractal_graph_v2=None):
        self.knowledge_graph = knowledge_graph
        self.brain = brain
        self.cache_dir = cache_dir
        self.fractal_graph_v2 = fractal_graph_v2
        self.initialized = False
        self.contradictions: Dict[str, Contradiction] = {}
        self.storage_path: Optional[str] = None
        self._entity_extractor = EntityExtractor() if EntityExtractor else None
        self._init_storage()
        
        if self.fractal_graph_v2 is not None:
            logger.info("ContradictionCore: FractalGraph v2 подключён")
        else:
            logger.warning("ContradictionCore: FractalGraph v2 не подключён")
        
        self.initialized = True
        logger.info("ContradictionCore инициализирован")
    
    def start(self):
        """Запуск детектора."""
        if not self.initialized:
            logger.error("Детектор не инициализирован")
            return
        logger.info("ContradictionCore запущен")
    
    def stop(self):
        """Остановка детектора."""
        self._save_contradictions()
        logger.info("ContradictionCore остановлен")
        
        try:
            ambiguous_entities = self._entity_extractor.extract_ambiguous_terms(text)
            for entity in ambiguous_entities:
                if entity.needs_clarification and entity.confidence > 0.5:
                    contradictions_found.append({
                        "type": "ambiguity_conflict",
                        "term": entity.text,
                        "ambiguity_type": entity.ambiguity_type.value if hasattr(entity.ambiguity_type, 'value') else str(entity.ambiguity_type),
                        "context": entity.context,
                        "severity": "medium" if entity.confidence > 0.7 else "low",
                        "possible_meanings": entity.possible_meanings
                    })
        except Exception as e:
            logger.error(f"Error in find_contradictions: {e}", exc_info=True)
        
        return contradictions_found
    
    def check_fractal_graph_contradiction(self, knowledge: str) -> Dict[str, Any]:
        """
        Проверяет знание на противоречия через FractalGraph v2.
        
        Использует встроенные методы FG для семантической проверки противоречий.
        
        Args:
            knowledge: Знание для проверки
            
        Returns:
            {is_contradiction, distance, group_id, confirmed, action, reasoning, new_nodes}
        """
        if self.fractal_graph_v2 is None:
            return {"error": "fractal_graph_v2 not available", "is_contradiction": False}
        
        try:
            check_result = self.fractal_graph_v2.check_contradiction(knowledge)
            is_contr = check_result.get("is_contradiction", False)
            
            if is_contr:
                logger.debug(f"Противоречие через FG: distance={check_result.get('distance')}")
            else:
                logger.debug(f"FG: противоречие не обнаружено")
            
            return {
                "is_contradiction": is_contr,
                "distance": check_result.get("distance"),
                "group_id": check_result.get("group_id"),
            }
        except Exception as e:
            logger.error(f"Ошибка проверки FG противоречий: {e}")
            return {"error": str(e), "is_contradiction": False}
    
    def verify_with_self_dialogue(self, knowledge: str) -> Dict[str, Any]:
        """
        Верифицирует знание через самодиалог FractalGraph.
        
        Args:
            knowledge: Знание для проверки
            
        Returns:
            {confirmed, action, reasoning, new_nodes}
        """
        if self.fractal_graph_v2 is None:
            return {"error": "fractal_graph_v2 not available"}
        
        try:
            return self.fractal_graph_v2.self_dialogue(knowledge)
        except Exception as e:
            logger.error(f"Ошибка FG self_dialogue: {e}")
            return {"error": str(e)}
    
    def add_concept_to_graph(self, concept: str, description: str = "", 
                            node_type: str = "concept", 
                            confidence: float = 0.7) -> Optional[str]:
        """
        Добавляет концепт напрямую в FractalGraph v2.
        
        Args:
            concept: Текст концепта
            description: Описание
            node_type: Тип узла
            confidence: Уверенность
            
        Returns:
            ID узла или None
        """
        if self.fractal_graph_v2 is None:
            logger.warning("FractalGraph v2 не доступен для добавления концепта")
            return None
        
        try:
            node = self.fractal_graph_v2.add_node(
                content=concept,
                node_type=node_type,
                confidence=confidence,
                metadata={"description": description, "source": "contradiction_manager"}
            )
            logger.debug(f"Концепт добавлен в FG: {concept[:30]}... -> {node.id}")
            return node.id
        except Exception as e:
            logger.error(f"Ошибка добавления концепта в FG: {e}")
            return None
