"""
Модуль для токенизации и обработки токенов в CogniFlex.
"""

import hashlib
import re
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TokenProcessor:
    """Обработчик токенов для системы CogniFlex."""
    
    def __init__(self, brain):
        self.brain = brain
        self.token_cache = getattr(brain, 'token_cache', None)
    
    def tokenize_query(self, query: str, context: Optional[Dict] = None) -> List[Dict]:
        """Токенизирует запрос и возвращает список токенов."""
        tokens = []
        
        # Простая токенизация по словам
        words = re.findall(r'\b\w+\b', query.lower())
        
        for word in words:
            token_id = self._generate_token_id(word)
            
            # Проверяем кэш
            cached_token = None
            if self.token_cache:
                cached_token = self.token_cache.get_token(token_id)
            
            if cached_token:
                tokens.append(cached_token)
            else:
                # Создаем новый токен
                token_data = {
                    "token": word,
                    "token_id": token_id,
                    "source": "user_query",
                    "priority": self._calculate_token_priority(word, context),
                    "context": context or {}
                }
                
                # Добавляем в кэш
                if self.token_cache:
                    self.token_cache.add_token(token_id, token_data)
                
                tokens.append(token_data)
        
        return tokens
    
    def _generate_token_id(self, token: str) -> str:
        """Генерирует уникальный ID для токена."""
        return hashlib.md5(token.encode('utf-8')).hexdigest()
    
    def _calculate_token_priority(self, token: str, context: Optional[Dict]) -> float:
        """Рассчитывает приоритет токена."""
        priority = 0.5  # Базовый приоритет
        
        # Увеличиваем приоритет для важных слов
        important_words = ['что', 'как', 'где', 'когда', 'почему', 'кто']
        if token in important_words:
            priority += 0.3
        
        # Учитываем контекст
        if context and 'priority_boost' in context:
            priority += context['priority_boost']
        
        return min(1.0, priority)
    
    def get_token_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по токенам."""
        if not self.token_cache:
            return {"error": "Token cache not available"}
        
        return self.token_cache.get_cache_stats()

    # --------------------
    # Интеграция с отложенными командами и восстановлением модулей
    # --------------------
    def prewarm_tokens_async(self, texts: List[str], priority: int = 5, batch_size: int = 100) -> bool:
        """Запускает прогрев токен-кэша как отложенную команду, если доступна система отложенных команд.

        Args:
            texts: Список текстов для предварительной токенизации.
            priority: Приоритет отложенной команды (меньше — выше приоритет).
            batch_size: Размер батча при прогреве.

        Returns:
            True если задача поставлена в очередь, иначе False.
        """
        try:
            deferred = getattr(self.brain, 'deferred_system', None)
            if not deferred:
                logger.debug("DeferredCommandSystem недоступна — прогрев будет выполнен синхронно")
                # Fallback: синхронный прогрев (может быть тяжелым)
                self._prewarm_tokens(texts, batch_size=batch_size)
                return True

            # Оборачиваем вызов в безопасную функцию без захвата self по ссылке замыкания
            def run_prewarm(tp: 'TokenProcessor', data: List[str], bs: int):
                tp._prewarm_tokens(data, batch_size=bs)

            deferred.add_command(
                command=run_prewarm,
                args=(self, texts, batch_size),
                kwargs={},
                priority=priority,
                description="Token cache prewarm",
                max_retries=2,
                timeout=60.0
            )
            logger.info(f"Отложенная команда прогрева токенов поставлена в очередь: {len(texts)} текстов")
            return True
        except Exception as e:
            logger.error(f"Не удалось запланировать прогрев токенов: {e}")
            return False

    def _prewarm_tokens(self, texts: List[str], batch_size: int = 100):
        """Прогревает кэш токенов, предварительно токенизируя тексты и сохраняя результаты в кэше."""
        if not texts:
            return
        try:
            count = 0
            for idx, text in enumerate(texts):
                # Простая токенизация, затем добавление в кэш
                tokens = self.tokenize_query(text, context={"source": "prewarm"})
                # При агрессивном дисковом кэше это наполнит метаданные и файлы
                count += len(tokens)
                if batch_size and (idx + 1) % batch_size == 0:
                    logger.debug(f"Prewarm progress: {idx + 1}/{len(texts)} items, {count} tokens cached")
            logger.info(f"Завершен прогрев токен-кэша: обработано {len(texts)} текстов, создано {count} токенов")
        except Exception as e:
            logger.error(f"Ошибка при прогреве токен-кэша: {e}")

    # Методы здоровья/восстановления для стратегии восстановления модулей
    def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья TokenProcessor и доступности кэша."""
        status = {
            "component": "token_processor",
            "healthy": True,
            "issues": []
        }
        try:
            if not self.token_cache:
                status["healthy"] = False
                status["issues"].append("token_cache_missing")
            else:
                # Быстрая проверка статистики кэша
                stats = self.get_token_statistics()
                if isinstance(stats, dict) and stats.get("error"):
                    status["healthy"] = False
                    status["issues"].append("token_cache_unavailable")
        except Exception as e:
            status["healthy"] = False
            status["issues"].append(f"exception:{e}")
        return status

    def recover(self) -> bool:
        """Пытается восстановить TokenProcessor через компонент-инициализатор ядра."""
        try:
            initializer = getattr(self.brain, 'component_initializer', None)
            if initializer and hasattr(initializer, 'initialize_token_processor'):
                ok = initializer.initialize_token_processor(self.brain)
                if ok:
                    # обновляем ссылку на кэш после реинициализации
                    self.token_cache = getattr(self.brain, 'token_cache', None)
                return bool(ok)
            # Fallback: проверить наличие кэша напрямую из brain
            self.token_cache = getattr(self.brain, 'token_cache', None)
            return self.token_cache is not None
        except Exception as e:
            logger.error(f"Ошибка восстановления TokenProcessor: {e}")
            return False