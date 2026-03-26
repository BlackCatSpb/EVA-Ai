"""
Qwen API Enhancer - дополняет генерацию через облачный API.
Работает ВСЕГДА с fallback цепочкой: Qwen API → Wikipedia → Web Search → None
НИКОГДА не вызывает ошибок генерации.
"""
import os
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("cogniflex.knowledge.qwen_api_enhancer")


class QwenAPIEnhancer:
    """
    Knowledge enhancer using Qwen API with fallback chain.
    
    Priority:
    1. Qwen API (OpenRouter) - лучший результат
    2. Wikipedia - fallback если нет API
    3. Web Search (DuckDuckGo) - последний fallback
    4. None - если ничего не доступно (НЕ ошибка!)
    """
    
    def __init__(
        self, 
        api_key: str = None, 
        enable_fallbacks: bool = True
    ):
        self.api_key = api_key or os.environ.get('OPENROUTER_API_KEY', '')
        self.enable_fallbacks = enable_fallbacks
        
        # Компоненты
        self.qwen_client = None
        self.wiki_search = None
        self.web_search = None
        
        # Статусы
        self.status = {
            'qwen_api': False,
            'wikipedia': False,
            'websearch': False,
            'enabled': False,
            'total_enhancements': 0,
            'successful_enhancements': 0,
            'fallback_used': 0,
            'last_source': None
        }
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Инициализирует компоненты"""
        
        # 1. Qwen API (приоритет 1)
        if self.api_key:
            try:
                from cogniflex.mlearning.qwen_api_client import QwenAPIClient
                self.qwen_client = QwenAPIClient(
                    api_key=self.api_key,
                    base_url="https://openrouter.ai/api/v1",
                    model="qwen/qwen3-30b-a3b"
                )
                self.status['qwen_api'] = True
                logger.info("QwenAPIEnhancer: Qwen API подключен")
            except Exception as e:
                logger.warning(f"QwenAPIEnhancer: Qwen API недоступен: {e}")
        
        # 2. Wikipedia (fallback)
        if self.enable_fallbacks:
            try:
                from cogniflex.websearch.search_engines import SearchEngines
                self.wiki_search = SearchEngines()
                self.status['wikipedia'] = True
                logger.info("QwenAPIEnhancer: Wikipedia подключена")
            except Exception as e:
                logger.warning(f"QwenAPIEnhancer: Wikipedia недоступна: {e}")
        
        # 3. Web Search (fallback)
        if self.enable_fallbacks:
            try:
                from cogniflex.websearch.web_search_engine import WebSearchEngine
                self.web_search = WebSearchEngine(cache_dir='./cache')
                self.status['websearch'] = True
                logger.info("QwenAPIEnhancer: Web Search подключен")
            except Exception as e:
                logger.warning(f"QwenAPIEnhancer: Web Search недоступен: {e}")
        
        # Общий статус
        self.status['enabled'] = (
            self.status['qwen_api'] or 
            self.status['wikipedia'] or 
            self.status['websearch']
        )
        
        logger.info(f"QwenAPIEnhancer: Статус - {self.status}")
    
    def enhance(
        self, 
        query: str, 
        local_knowledge: Dict = None
    ) -> Optional[Dict]:
        """
        Основной метод - обогащает запрос.
        НИКОГДА не вызывает ошибок - всегда возвращает None если недоступен.
        
        Args:
            query: Запрос пользователя
            local_knowledge: Локальные знания из KnowledgeGraph
            
        Returns:
            Dict с обогащением или None если недоступно
        """
        local_knowledge = local_knowledge or {}
        self.status['total_enhancements'] += 1
        
        # 1. Пробуем Qwen API (лучший результат)
        if self.status['qwen_api']:
            try:
                result = self._enhance_with_qwen(query, local_knowledge)
                if result:
                    self.status['successful_enhancements'] += 1
                    self.status['last_source'] = 'qwen_api'
                    return result
            except Exception as e:
                logger.warning(f"Qwen API enhance failed: {e}")
        
        # 2. Fallback на Wikipedia
        if self.status['wikipedia']:
            try:
                result = self._enhance_with_wikipedia(query)
                if result:
                    self.status['successful_enhancements'] += 1
                    self.status['fallback_used'] += 1
                    self.status['last_source'] = 'wikipedia'
                    return result
            except Exception as e:
                logger.warning(f"Wikipedia enhance failed: {e}")
        
        # 3. Fallback на Web Search
        if self.status['websearch']:
            try:
                result = self._enhance_with_websearch(query)
                if result:
                    self.status['successful_enhancements'] += 1
                    self.status['fallback_used'] += 1
                    self.status['last_source'] = 'websearch'
                    return result
            except Exception as e:
                logger.warning(f"Web search enhance failed: {e}")
        
        # 4. Ничего не доступно - это НЕ ошибка!
        self.status['last_source'] = None
        return None
    
    def _enhance_with_qwen(
        self, 
        query: str, 
        local_knowledge: Dict
    ) -> Optional[Dict]:
        """Использует Qwen API для обогащения"""
        if not self.qwen_client:
            return None
        
        # Формируем промпт
        context = local_knowledge.get('summary', 'Нет данных')
        prompt = f"""Ты - экспертный ассистент CogniFlex.
Пользователь спрашивает: {query}

Текущие знания системы:
{context}

Дополни свои знания если нужно. Отвечай кратко (2-3 предложения):"""
        
        try:
            result = self.qwen_client.generate(
                prompt=prompt,
                max_new_tokens=512,
                temperature=0.7
            )
            
            if result and result.get('text'):
                return {
                    'source': 'qwen_api',
                    'text': result['text'],
                    'model': result.get('model', 'qwen3-235b-a22b'),
                    'usage': result.get('usage', {})
                }
        except Exception as e:
            logger.warning(f"Qwen API call failed: {e}")
        
        return None
    
    def _enhance_with_wikipedia(self, query: str) -> Optional[Dict]:
        """Fallback: Поиск в Wikipedia"""
        if not self.wiki_search:
            return None
        
        try:
            results = self.wiki_search.search_wikipedia(query, max_results=3)
            
            if results:
                snippets = []
                for r in results:
                    snippets.append({
                        'title': r.title,
                        'url': r.url,
                        'snippet': r.snippet[:200] if r.snippet else ''
                    })
                
                return {
                    'source': 'wikipedia',
                    'results': snippets,
                    'count': len(snippets)
                }
        except Exception as e:
            logger.warning(f"Wikipedia search failed: {e}")
        
        return None
    
    def _enhance_with_websearch(self, query: str) -> Optional[Dict]:
        """Fallback: Web Search (DuckDuckGo)"""
        if not self.web_search:
            return None
        
        try:
            results = self.web_search.search(query, max_results=3)
            
            if results and results.get('results'):
                snippets = []
                for r in results['results']:
                    snippets.append({
                        'title': r.title,
                        'url': r.url,
                        'snippet': r.snippet[:200] if r.snippet else ''
                    })
                
                return {
                    'source': 'websearch',
                    'results': snippets,
                    'count': len(snippets)
                }
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
        
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Возвращает статус для GUI"""
        return {
            'enabled': self.status['enabled'],
            'qwen_api': self.status['qwen_api'],
            'wikipedia': self.status['wikipedia'],
            'websearch': self.status['websearch'],
            'total_enhancements': self.status['total_enhancements'],
            'successful_enhancements': self.status['successful_enhancements'],
            'fallback_used': self.status['fallback_used'],
            'last_source': self.status['last_source'],
            'current_source': self._get_current_source()
        }
    
    def _get_current_source(self) -> str:
        """Определяет текущий источник для отображения"""
        if self.status['qwen_api']:
            return 'qwen_api'
        elif self.status['wikipedia']:
            return 'wikipedia'
        elif self.status['websearch']:
            return 'websearch'
        else:
            return 'local_only'
    
    def is_available(self) -> bool:
        """Проверяет доступность любого источника"""
        return self.status['enabled']


__all__ = ['QwenAPIEnhancer']
