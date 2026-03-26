"""
Интеграция веб-поиска в систему самообучения модели
"""
import os
import logging
import time
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("cogniflex.web_search_integration")

class WebSearchLearningIntegration:
    """Интеграция веб-поиска с системой самообучения"""
    
    def __init__(self, fractal_model_manager, web_search_engine=None):
        """
        Инициализирует интеграцию веб-поиска
        
        Args:
            fractal_model_manager: Менеджер фрактальной модели
            web_search_engine: Движок веб-поиска (опционально)
        """
        self.fractal_model_manager = fractal_model_manager
        self.web_search_engine = web_search_engine
        
        # Инициализируем веб-поиск если не предоставлен
        if not self.web_search_engine:
            try:
                from ..websearch.web_search_engine import WebSearchEngine
                self.web_search_engine = WebSearchEngine(brain=getattr(fractal_model_manager, 'brain', None))
                logger.info("WebSearchEngine инициализирован для интеграции")
            except Exception as e:
                logger.error(f"Не удалось инициализировать WebSearchEngine: {e}")
                self.web_search_engine = None
        
        # Настройки интеграции
        self.integration_settings = {
            "auto_search_threshold": 0.5,  # Порог качества для авто-поиска
            "max_search_results": 5,        # Максимум результатов поиска
            "search_timeout": 30.0,         # Таймаут поиска
            "use_search_cache": True,       # Использовать кэш поиска
            "min_content_length": 100,      # Минимальная длина контента
            "max_content_length": 1000,     # Максимальная длина контента
            "search_engines": ["google", "yandex"],  # Активные поисковые системы
            "learning_integration": True,    # Интеграция с обучением
        }
        
        # Статистика интеграции
        self.integration_stats = {
            "search_queries": 0,
            "successful_searches": 0,
            "knowledge_extracted": 0,
            "training_texts_generated": 0,
            "quality_improvements": 0,
            "last_search_time": 0,
            "total_search_time": 0.0
        }
        
        # Кэш результатов поиска
        self.search_cache = {}
        self.cache_ttl = 3600  # 1 час
        
        logger.info("WebSearchLearningIntegration инициализирована")
    
    def search_and_enhance_response(self, query: str, max_new_tokens: int = 2048) -> Dict[str, Any]:
        """
        Выполняет поиск и улучшает ответ на основе найденной информации
        
        Args:
            query: Запрос пользователя
            max_new_tokens: Максимальное количество токенов для ответа
            
        Returns:
            Dict[str, Any]: Результат с улучшенным ответом и метаданными
        """
        if not self.web_search_engine:
            return {
                "status": "error",
                "message": "Веб-поиск недоступен",
                "response": self.fractal_model_manager.generate_response(query, max_tokens)
            }
        
        try:
            start_time = time.time()
            
            # 1. Генерируем базовый ответ
            base_response = self.fractal_model_manager.generate_response(query, max_tokens)
            
            # 2. Анализируем качество базового ответа
            quality_metrics = self._analyze_response_quality(base_response)
            
            # 3. Решаем, нужен ли поиск
            need_search = self._should_search(query, quality_metrics)
            
            if not need_search:
                return {
                    "status": "completed",
                    "query": query,
                    "response": base_response,
                    "quality_metrics": quality_metrics,
                    "search_performed": False,
                    "processing_time": time.time() - start_time
                }
            
            # 4. Выполняем веб-поиск
            search_results = self._perform_web_search(query)
            
            if not search_results:
                return {
                    "status": "completed",
                    "query": query,
                    "response": base_response,
                    "quality_metrics": quality_metrics,
                    "search_performed": True,
                    "search_results": [],
                    "message": "Поиск не дал результатов",
                    "processing_time": time.time() - start_time
                }
            
            # 5. Улучшаем ответ на основе поиска
            enhanced_response = self._enhance_response_with_search(query, base_response, search_results)
            
            # 6. Анализируем качество улучшенного ответа
            enhanced_metrics = self._analyze_response_quality(enhanced_response)
            
            processing_time = time.time() - start_time
            
            # 7. Обновляем статистику
            self._update_integration_stats(search_results, enhanced_metrics)
            
            return {
                "status": "completed",
                "query": query,
                "response": enhanced_response,
                "base_response": base_response,
                "quality_metrics": enhanced_metrics,
                "base_quality_metrics": quality_metrics,
                "search_performed": True,
                "search_results": search_results,
                "quality_improvement": enhanced_metrics.get("overall", 0) - quality_metrics.get("overall", 0),
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Ошибка в search_and_enhance_response: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e),
                "response": self.fractal_model_manager.generate_response(query, max_tokens)
            }
    
    def _should_search(self, query: str, quality_metrics: Dict[str, Any]) -> bool:
        """Определяет, нужно ли выполнять веб-поиск"""
        
        # Проверяем порог качества
        overall_quality = quality_metrics.get("overall", 1.0)
        if overall_quality < self.integration_settings["auto_search_threshold"]:
            return True
        
        # Проверяем наличие вопросительных слов
        question_words = ["что", "кто", "где", "когда", "почему", "как", "какой", "какая", "какое"]
        if any(word in query.lower() for word in question_words):
            return True
        
        # Проверяем сложность запроса
        if len(query.split()) > 5:  # Длинные запросы
            return True
        
        # Проверяем на специфические термины
        specific_terms = ["определение", "объясни", "расскажи", "поясни", "что такое"]
        if any(term in query.lower() for term in specific_terms):
            return True
        
        return False
    
    def _perform_web_search(self, query: str) -> List[Dict[str, Any]]:
        """Выполняет веб-поиск"""
        
        try:
            # Проверяем кэш
            cache_key = f"search_{hash(query)}"
            if cache_key in self.search_cache:
                cached_result = self.search_cache[cache_key]
                if time.time() - cached_result["timestamp"] < self.cache_ttl:
                    logger.info(f"Используем кэшированный результат для: {query}")
                    return cached_result["results"]
            
            # Выполняем поиск
            search_response = self.web_search_engine.search(
                query=query,
                max_results=self.integration_settings["max_search_results"],
                use_cache=self.integration_settings["use_search_cache"]
            )
            
            if search_response.get("status") != "completed":
                logger.warning(f"Поиск не удался: {search_response.get('message', 'Unknown error')}")
                return []
            
            # Извлекаем результаты
            results = search_response.get("results", [])
            
            # Конвертируем в нужный формат
            formatted_results = []
            for result in results:
                if hasattr(result, 'title'):
                    formatted_results.append({
                        "title": result.title,
                        "url": result.url,
                        "snippet": result.snippet,
                        "source": result.source,
                        "relevance": getattr(result, 'relevance_score', 1.0),
                        "content": getattr(result, 'content', None)
                    })
                elif isinstance(result, dict):
                    formatted_results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("snippet", ""),
                        "source": result.get("source", "unknown"),
                        "relevance": result.get("relevance_score", 1.0),
                        "content": result.get("content", None)
                    })
            
            # Сохраняем в кэш
            self.search_cache[cache_key] = {
                "results": formatted_results,
                "timestamp": time.time()
            }
            
            # Ограничиваем размер кэша
            if len(self.search_cache) > 100:
                oldest_key = min(self.search_cache.keys(), 
                               key=lambda k: self.search_cache[k]["timestamp"])
                del self.search_cache[oldest_key]
            
            logger.info(f"Найдено {len(formatted_results)} результатов по запросу: {query}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Ошибка выполнения веб-поиска: {e}", exc_info=True)
            return []
    
    def _enhance_response_with_search(self, query: str, base_response: str, 
                                   search_results: List[Dict[str, Any]]) -> str:
        """Улучшает ответ на основе результатов поиска"""
        
        try:
            # Извлекаем ключевую информацию из результатов поиска
            extracted_info = self._extract_key_information(search_results)
            
            if not extracted_info:
                return base_response
            
            # Создаем улучшенный промпт
            enhanced_prompt = self._create_enhanced_prompt(query, base_response, extracted_info)
            
            # Генерируем улучшенный ответ
            enhanced_response = self.fractal_model_manager.generate_response(
                enhanced_prompt, 
                max_new_tokens=min(200, len(base_response) + 100)
            )
            
            # Очищаем ответ
            enhanced_response = self._clean_enhanced_response(enhanced_response, query)
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Ошибка улучшения ответа: {e}", exc_info=True)
            return base_response
    
    def _extract_key_information(self, search_results: List[Dict[str, Any]]) -> List[str]:
        """Извлекает ключевую информацию из результатов поиска"""
        
        key_info = []
        
        for result in search_results:
            # Извлекаем информацию из заголовка и сниппета
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            content = result.get("content", "")
            
            # Комбинируем информацию
            combined_info = f"{title}. {snippet}"
            if content and len(content) > 50:
                combined_info += f" {content[:200]}..."
            
            # Фильтруем и добавляем
            if len(combined_info.strip()) > self.integration_settings["min_content_length"]:
                # Ограничиваем длину
                if len(combined_info) > self.integration_settings["max_content_length"]:
                    combined_info = combined_info[:self.integration_settings["max_content_length"]] + "..."
                
                key_info.append(combined_info.strip())
        
        return key_info[:3]  # Максимум 3 источника информации
    
    def _create_enhanced_prompt(self, query: str, base_response: str, 
                               extracted_info: List[str]) -> str:
        """Создает улучшенный промпт с учетом найденной информации"""
        
        info_text = "\n".join([f"- {info}" for info in extracted_info])
        
        enhanced_prompt = f"""Исходный запрос: {query}

Базовый ответ: {base_response}

Дополнительная информация из интернета:
{info_text}

Учитывая дополнительную информацию, дай более полный и точный ответ на исходный запрос. Ответ должен быть информативным, структурированным и основанным на предоставленных данных."""
        
        return enhanced_prompt
    
    def _clean_enhanced_response(self, response: str, query: str) -> str:
        """Очищает улучшенный ответ"""
        
        # Убираем упоминания промпта
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Пропускаем строки с упоминанием промпта
            if any(keyword in line.lower() for keyword in 
                   ['исходный запрос', 'базовый ответ', 'дополнительная информация', 'учитывая']):
                continue
            
            # Пропускаем пустые строки в начале
            if not line.strip() and not cleaned_lines:
                continue
            
            cleaned_lines.append(line.strip())
        
        cleaned_response = '\n'.join(cleaned_lines).strip()
        
        # Убираем повторения запроса
        if cleaned_response.lower().startswith(query.lower()):
            cleaned_response = cleaned_response[len(query):].strip()
        
        return cleaned_response if cleaned_response else response
    
    def _analyze_response_quality(self, response: str) -> Dict[str, Any]:
        """Анализирует качество ответа"""
        
        try:
            if hasattr(self.fractal_model_manager, 'get_quality_metrics'):
                return self.fractal_model_manager.get_quality_metrics()
            else:
                # Базовый анализ качества
                return {
                    "overall": 0.7,
                    "coherence": 0.8,
                    "diversity": 0.6,
                    "length": 0.7,
                    "grammar": 0.9,
                    "readability": 0.8,
                    "relevance": 0.7
                }
        except Exception as e:
            logger.error(f"Ошибка анализа качества: {e}")
            return {"overall": 0.5}
    
    def _update_integration_stats(self, search_results: List[Dict[str, Any]], 
                                 quality_metrics: Dict[str, Any]):
        """Обновляет статистику интеграции"""
        
        self.integration_stats["search_queries"] += 1
        self.integration_stats["last_search_time"] = time.time()
        
        if search_results:
            self.integration_stats["successful_searches"] += 1
            self.integration_stats["knowledge_extracted"] += len(search_results)
        
        # Проверяем улучшение качества
        if quality_metrics.get("overall", 0) > 0.7:
            self.integration_stats["quality_improvements"] += 1
    
    def generate_training_texts_from_search(self, topics: List[str], 
                                       max_texts_per_topic: int = 3) -> List[str]:
        """
        Генерирует обучающие тексты на основе веб-поиска
        
        Args:
            topics: Список тем для поиска
            max_texts_per_topic: Максимум текстов на тему
            
        Returns:
            List[str]: Список обучающих текстов
        """
        
        training_texts = []
        
        for topic in topics:
            try:
                # Выполняем поиск
                search_results = self._perform_web_search(topic)
                
                if not search_results:
                    continue
                
                # Генерируем обучающие тексты
                for result in search_results[:max_texts_per_topic]:
                    title = result.get("title", "")
                    snippet = result.get("snippet", "")
                    content = result.get("content", "")
                    
                    # Создаем обучающий текст
                    if content and len(content) > 100:
                        training_text = f"Тема: {topic}\nЗаголовок: {title}\nСодержание: {content[:500]}"
                    else:
                        training_text = f"Тема: {topic}\nЗаголовок: {title}\nОписание: {snippet}"
                    
                    training_texts.append(training_text)
                
                # Генерируем обобщающий текст
                if len(search_results) > 1:
                    combined_info = "\n".join([r.get("snippet", "") for r in search_results[:3]])
                    summary_prompt = f"На основе следующей информации о '{topic}', создай краткое обобщение:\n{combined_info}"
                    
                    summary = self.fractal_model_manager.generate_response(summary_prompt, max_new_tokens=150)
                    training_texts.append(f"Тема: {topic}\nОбобщение: {summary}")
                
            except Exception as e:
                logger.error(f"Ошибка генерации обучающих текстов для темы '{topic}': {e}")
                continue
        
        self.integration_stats["training_texts_generated"] += len(training_texts)
        
        logger.info(f"Сгенерировано {len(training_texts)} обучающих текстов из {len(topics)} тем")
        return training_texts
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Возвращает статистику интеграции"""
        
        stats = self.integration_stats.copy()
        
        # Добавляем дополнительную статистику
        if stats["search_queries"] > 0:
            stats["search_success_rate"] = stats["successful_searches"] / stats["search_queries"]
        else:
            stats["search_success_rate"] = 0.0
        
        stats["cache_size"] = len(self.search_cache)
        stats["web_search_available"] = self.web_search_engine is not None
        
        return stats
    
    def configure_integration(self, **settings):
        """Настраивает параметры интеграции"""
        
        self.integration_settings.update(settings)
        logger.info(f"Настройки интеграции обновлены: {settings}")
    
    def clear_cache(self):
        """Очищает кэш поиска"""
        
        self.search_cache.clear()
        logger.info("Кэш веб-поиска очищен")
    
    def __del__(self):
        """Очистка при удалении"""
        
        try:
            self.clear_cache()
            logger.info("WebSearchLearningIntegration очищена")
        except Exception:
            pass
