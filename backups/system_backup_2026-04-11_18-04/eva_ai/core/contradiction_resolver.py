# contradiction_resolver.py
"""Решатель противоречий для ЕВА."""

import os
import sys
import time
import json
import logging
import hashlib
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger("eva_ai.contradiction_resolver")


class ContradictionResolver:
    """Система обнаружения и разрешения противоречий в знаниях."""

    def __init__(self, attention_system):
        self.attention_system = attention_system
        self.active_contradictions = []
        self.resolution_history = []
        self.logger = logging.getLogger("eva_ai.contradiction_resolver")
        self.contradiction_threshold = 0.7
        self.semantic_threshold = 0.3
        self._embedding_cache = {}
        self._embedding_model = None

    def check_response(self, question: str, response: str):
        """Проверяет ответ на наличие противоречий."""
        try:
            # Анализируем ответ на наличие противоречий
            contradictions = self._detect_contradictions(response)

            # Добавляем обнаруженные противоречия
            for contradiction in contradictions:
                self._add_contradiction(contradiction)

            # Пытаемся разрешить противоречия
            self._attempt_resolution()
        except Exception as e:
            self.logger.error(f"Ошибка проверки ответа на противоречия: {e}")

    def _detect_contradictions(self, text: str) -> List[Dict[str, Any]]:
        """Обнаруживает потенциальные противоречия в тексте."""
        contradictions = []

        # Простые правила для обнаружения противоречий
        contradiction_indicators = [
            ("но", "however"),
            ("однако", "nevertheless"),
            ("противоречит", "contradicts"),
            ("несоответствие", "inconsistency"),
            ("несогласованность", "discrepancy")
        ]

        text_lower = text.lower()

        for indicator, _ in contradiction_indicators:
            if indicator in text_lower:
                # Простая заглушка для обнаружения противоречий
                contradictions.append({
                    "text": text,
                    "indicator": indicator,
                    "confidence": 0.8,
                    "detected_at": time.time()
                })
                break  # Достаточно одного индикатора для обнаружения

        return contradictions

    def _add_contradiction(self, contradiction: Dict[str, Any]):
        """Добавляет обнаруженное противоречие в список активных."""
        # Проверяем, не дублируется ли противоречие
        if not self._is_duplicate(contradiction):
            self.active_contradictions.append(contradiction)
            self.logger.info(f"Обнаружено новое противоречие: {contradiction['indicator']}")

    def _is_duplicate(self, new_contradiction: Dict[str, Any]) -> bool:
        """Проверяет, является ли противоречие дубликатом."""
        for existing in self.active_contradictions:
            # Простая проверка на дубликаты
            if new_contradiction['indicator'] == existing['indicator']:
                time_diff = abs(new_contradiction['detected_at'] - existing['detected_at'])
                if time_diff < 300:  # 5 минут
                    return True
        return False

    def _attempt_resolution(self):
        """Пытается разрешить активные противоречия."""
        if not self.active_contradictions:
            return

        for contradiction in self.active_contradictions[:]:
            try:
                # Проверяем, можно ли разрешить противоречие
                if self._can_resolve(contradiction):
                    resolution = self._resolve_contradiction(contradiction)
                    self._record_resolution(contradiction, resolution)
                    self.active_contradictions.remove(contradiction)
            except Exception as e:
                self.logger.error(f"Ошибка при попытке разрешения противоречия: {e}")

    def _can_resolve(self, contradiction: Dict[str, Any]) -> bool:
        """Проверяет, можно ли разрешить противоречие."""
        # Простая проверка - если противоречие достаточно уверенное
        return contradiction['confidence'] >= self.contradiction_threshold

    def _resolve_contradiction(self, contradiction: Dict[str, Any]) -> Dict[str, Any]:
        """Разрешает противоречие, используя доступные знания."""
        try:
            # Получаем контекст из горячего окна
            hot_window = self._get_hot_window_data()

            # Формируем запрос для разрешения противоречия
            resolution_prompt = self._create_resolution_prompt(contradiction, hot_window)

            # Получаем ответ от системы
            if hasattr(self.attention_system.core_brain, 'generation_coordinator'):
                resolution = self.attention_system.core_brain.generation_coordinator.generate_response(
                    resolution_prompt
                )
            else:
                resolution = f"Противоречие '{contradiction['indicator']}' может быть разрешено путем анализа контекста."

            return {
                "resolution": resolution,
                "confidence": 0.75,
                "resolved_at": time.time()
            }
        except Exception as e:
            self.logger.error(f"Ошибка разрешения противоречия: {e}")
            return {
                "resolution": "Не удалось автоматически разрешить противоречие.",
                "confidence": 0.0,
                "resolved_at": time.time()
            }

    def _get_hot_window_data(self) -> Dict:
        """Получает данные из горячего окна."""
        try:
            if hasattr(self.attention_system.core_brain, 'memory_manager') and \
               self.attention_system.core_brain.memory_manager:
                return self.attention_system.core_brain.memory_manager.get_hot_window_data()
            return {}
        except Exception as e:
            logger.debug(f"Ошибка получения данных горячего окна: {e}")
            return {}

    def _create_resolution_prompt(self, contradiction: Dict[str, Any], hot_window: Dict) -> str:
        """Создает промпт для разрешения противоречия."""
        context = "Контекст: " + ", ".join(hot_window.keys()[:3]) if hot_window else "Нет контекста"

        return (
            f"Разреши следующее противоречие в знаниях:\n\n"
            f"Обнаружено противоречие: {contradiction['text']}\n"
            f"Индикатор: {contradiction['indicator']}\n"
            f"{context}\n\n"
            f"Предоставь логическое объяснение и укажи, какая информация более достоверна. "
            f"Предложи способ интеграции этих знаний в согласованную модель."
        )

    def _record_resolution(self, contradiction: Dict[str, Any], resolution: Dict[str, Any]):
        """Записывает разрешение противоречия в историю."""
        self.resolution_history.append({
            "contradiction": contradiction,
            "resolution": resolution,
            "timestamp": time.time()
        })
        self.logger.info(f"Противоречие разрешено: {contradiction['indicator']}")

    def has_active_contradictions(self) -> bool:
        """Проверяет, есть ли активные противоречия."""
        return len(self.active_contradictions) > 0

    def get_active_contradictions(self) -> List[Dict[str, Any]]:
        """Возвращает активные противоречия (совместимость с core.query_processor)."""
        return list(self.active_contradictions)

    def check_response_contradictions(self, query: str, response: str) -> List[Dict[str, Any]]:
        """Проверяет ответ на наличие противоречий и возвращает найденные (совместимость)."""
        before = len(self.active_contradictions)
        try:
            self.check_response(query, response)
        except Exception as e:
            logger.error(f"Ошибка проверки противоречий в ответе: {e}")
        after = len(self.active_contradictions)
        try:
            return self.active_contradictions[before:after] if after >= before else list(self.active_contradictions)
        except Exception:
            return list(self.active_contradictions)

    def get_resolution_history(self) -> List[Dict[str, Any]]:
        """Возвращает историю разрешенных противоречий."""
        return self.resolution_history

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Получает embedding текста через доступные методы."""
        cache_key = hash(text[:100])
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        embedding = None
        
        try:
            if hasattr(self.attention_system, 'core_brain'):
                core_brain = self.attention_system.core_brain
                
                if hasattr(core_brain, 'embedding_model') and core_brain.embedding_model:
                    emb_result = core_brain.embedding_model.encode([text])
                    if hasattr(emb_result, 'tolist'):
                        embedding = np.array(emb_result.tolist()[0])
                    else:
                        embedding = np.array(emb_result[0])
                elif hasattr(core_brain, 'fractal_memory') and core_brain.fractal_memory:
                    fractal = core_brain.fractal_memory
                    if hasattr(fractal, 'learning_loop') and fractal.learning_loop:
                        if hasattr(fractal.learning_loop, '_compute_embedding'):
                            embedding = fractal.learning_loop._compute_embedding(text)
        except Exception as e:
            logger.debug(f"Embedding не получен: {e}")
        
        if embedding is None:
            embedding = self._simple_embedding(text)
        
        if embedding is not None:
            self._embedding_cache[cache_key] = embedding
            if len(self._embedding_cache) > 500:
                self._embedding_cache.pop(next(iter(self._embedding_cache)))
        
        return embedding
    
    def _simple_embedding(self, text: str) -> Optional[np.ndarray]:
        """Простой fallback embedding на основе хеша и случайности."""
        try:
            import hashlib
            hash_bytes = hashlib.md5(text.encode()).digest()
            np.random.seed(int.from_bytes(hash_bytes[:4], 'little'))
            return np.random.randn(128)
        except Exception:
            return None
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Вычисляет косинусное сходство между двумя embeddings."""
        try:
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(emb1, emb2) / (norm1 * norm2))
        except Exception:
            return 0.0
    
    def _semantic_contradiction_detection(self, text1: str, text2: str) -> float:
        """Обнаружение противоречий через семантическое сравнение."""
        if not text1 or not text2:
            return 0.0
        
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        similarity = self._cosine_similarity(emb1, emb2)
        return 1.0 - max(0.0, similarity)
    
    def check_graph_contradictions(self, experience_id: str) -> List[Dict]:
        """Проверить противоречия между связанными опытами в графе."""
        try:
            if not hasattr(self.attention_system, 'core_brain'):
                return []
            
            core_brain = self.attention_system.core_brain
            if not hasattr(core_brain, 'fractal_memory'):
                return []
            
            fractal = core_brain.fractal_memory
            if not fractal or not hasattr(fractal, 'learning_loop'):
                return []
            
            learning_loop = fractal.learning_loop
            if not hasattr(learning_loop, 'get_experience'):
                return []
            
            experience = learning_loop.get_experience(experience_id)
            if not experience:
                return []
            
            related = experience.get('related_experiences', [])
            if not related:
                return []
            
            contradictions = []
            for rel_id in related:
                rel_exp = learning_loop.get_experience(rel_id)
                if not rel_exp:
                    continue
                
                score = self._semantic_contradiction_detection(
                    experience.get('response', ''),
                    rel_exp.get('response', '')
                )
                
                if score > self.semantic_threshold:
                    contradictions.append({
                        'experience_id': rel_id,
                        'contradiction_score': score
                    })
            
            return contradictions
        except Exception as e:
            logger.debug(f"Ошибка проверки противоречий в графе: {e}")
            return []
