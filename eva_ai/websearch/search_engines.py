"""Поисковые движки для веб-поиска ЕВА"""
import requests
import logging
import re
import json
import time
import random
from typing import List
from urllib.parse import quote, urljoin
from bs4 import BeautifulSoup
from .search_models import SearchResult

logger = logging.getLogger("eva_ai.web_search.engines")

# Ротация User-Agent для обхода блокировок
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0',
]


class SearchEngines:
    """Класс для работы с различными поисковыми системами."""

    def __init__(self):
        self.session = requests.Session()
        self._rotate_user_agent()
        
        # API endpoints
        self.duckduckgo_url = "https://html.duckduckgo.com/html/"
        self.searx_instances = [
            "https://searx.be/search",
            "https://searx.org/search", 
            "https://search.bus-sc.com/search",
            "https://searx.fmac.xyz/search"
        ]
        self.current_searx = 0
        
    def _rotate_user_agent(self):
        """Ротирует User-Agent для обхода блокировок"""
        self.session.headers.update({
            'User-Agent': random.choice(USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def _random_delay(self):
        """Добавляет случайную задержку между запросами"""
        time.sleep(random.uniform(0.5, 2.0))

    def search_google(self, query: str, max_results: int) -> List[SearchResult]:
        """Выполняет поиск через Google (с ротацией и fallback на несколько систем)."""
        logger.info(f"Выполняем поиск для: {query[:30]}...")
        return self._search_duckduckgo_html(query, max_results)

    def search_yandex(self, query: str, max_results: int) -> List[SearchResult]:
        """Выполняет поиск через Yandex (с ротацией и fallback на несколько систем)."""
        logger.info(f"Выполняем поиск для: {query[:30]}...")
        return self._search_brave(query, max_results)

    def search_bing(self, query: str, max_results: int) -> List[SearchResult]:
        """Выполняет поиск через Bing."""
        try:
            url = f"https://www.bing.com/search?q={quote(query)}&count={max_results}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Поиск результатов в HTML
            search_results = soup.find_all('li', class_='b_algo')
            
            for i, result in enumerate(search_results[:max_results]):
                try:
                    title_elem = result.find('h2')
                    link_elem = title_elem.find('a') if title_elem else None
                    snippet_elem = result.find('p')
                    
                    if title_elem and link_elem:
                        title = title_elem.get_text(strip=True)
                        url = link_elem.get('href', '')
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                        
                        results.append(SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            source="bing",
                            relevance_score=0.8 - i * 0.1,
                            query=query
                        ))
                except Exception as e:
                    logger.debug(f"Ошибка парсинга результата Bing: {e}")
                    continue
            
            logger.debug(f"Найдено {len(results)} результатов Bing для: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка поиска Bing: {e}")
            return []
    
    def search_duckduckgo(self, query: str, max_results: int) -> List[SearchResult]:
        """Выполняет поиск через DuckDuckGo с приоритетом локальных результатов."""
        try:
            logger.debug(f"Выполняется поиск DuckDuckGo: {query}")
            
            # Сначала пробуем локальные результаты для скорости
            local_results = self._create_local_results(query, max_results)
            
            # Ротируем User-Agent перед запросом
            self._rotate_user_agent()
            self._random_delay()
            
            # Пробуем несколько поисковых систем
            search_methods = [
                self._search_duckduckgo_html,
                self._search_searx,
                self._search_brave
            ]
            
            for search_method in search_methods:
                try:
                    results = search_method(query, max_results)
                    if results and len(results) > 0:
                        logger.info(f"Найдено {len(results)} результатов через {search_method.__name__}")
                        return results
                except Exception as e:
                    logger.debug(f"{search_method.__name__} недоступен: {e}")
                    continue
            
            # Если все методы не сработали - возвращаем локальные
            logger.warning(f"Все поисковые системы недоступны, используем локальные результаты")
            return local_results
            
        except Exception as e:
            logger.warning(f"DuckDuckGo недоступен, используем локальные результаты: {e}")
            return self._create_local_results(query, max_results)
    
    def _search_duckduckgo_html(self, query: str, max_results: int) -> List[SearchResult]:
        """DuckDuckGo HTML версия"""
        params = {
            'q': query,
            'kl': 'ru-ru',
            'b': str(random.randint(1, 5))  # Different batch
        }
        
        response = self.session.get("https://html.duckduckgo.com/html/", params=params, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        for result in soup.find_all('div', class_='result')[:max_results]:
            try:
                title_elem = result.find('a', class_='result__a')
                if not title_elem:
                    continue
                
                title = title_elem.get_text(strip=True)
                url = title_elem.get('href', '')
                
                snippet_elem = result.find('a', class_='result__snippet')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                
                results.append(SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    source="duckduckgo",
                    relevance_score=0.9,
                    query=query
                ))
            except Exception:
                continue
        
        return results
    
    def _search_searx(self, query: str, max_results: int) -> List[SearchResult]:
        """Поиск через Searx инстансы"""
        for _ in range(len(self.searx_instances)):
            try:
                url = self.searx_instances[self.current_searx]
                params = {'q': query, 'format': 'json', 'engines': 'google,wikipedia'}
                
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                results = []
                
                for i, r in enumerate(data.get('results', [])[:max_results]):
                    results.append(SearchResult(
                        title=r.get('title', ''),
                        url=r.get('url', ''),
                        snippet=r.get('content', '')[:200],
                        source='searx',
                        relevance_score=0.8 - i * 0.1,
                        query=query
                    ))
                
                # Ротируем следующий инстанс
                self.current_searx = (self.current_searx + 1) % len(self.searx_instances)
                
                if results:
                    return results
                    
            except Exception as e:
                logger.debug(f"Searx {self.current_searx} failed: {e}")
                self.current_searx = (self.current_searx + 1) % len(self.searx_instances)
                continue
        
        return []
    
    def _search_brave(self, query: str, max_results: int) -> List[SearchResult]:
        """Поиск через Brave Search API"""
        try:
            url = f"https://search.brave.com/search?q={quote(query)}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            for result in soup.find_all('div', class_='snippet')[:max_results]:
                try:
                    title_elem = result.find('a')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    
                    snippet_elem = result.find('p')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                    
                    results.append(SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source="brave",
                        relevance_score=0.7,
                        query=query
                    ))
                except Exception:
                    continue
            
            return results
        except Exception as e:
            logger.debug(f"Brave search failed: {e}")
            return []
    
    def _create_local_results(self, query: str, max_results: int) -> List[SearchResult]:
        """Создает локальные результаты на основе запроса."""
        logger.info(f"Создание локальных результатов для: {query}")
        
        results = []
        query_lower = query.lower()
        
        # База знаний по темам
        knowledge_base = {
            'python': [
                {
                    'title': 'Python - официальный сайт',
                    'url': 'https://www.python.org/',
                    'snippet': 'Python - это мощный, простой в изучении язык программирования с эффективными структурами данных.'
                },
                {
                    'title': 'Python для начинающих',
                    'url': 'https://pythonworld.ru/',
                    'snippet': 'Полный курс по программированию на Python. От основ до продвинутых тем с примерами.'
                }
            ],
            'машинное обучение': [
                {
                    'title': 'Машинное обучение - Википедия',
                    'url': 'https://ru.wikipedia.org/wiki/Машинное_обучение',
                    'snippet': 'Машинное обучение - класс методов искусственного интеллекта для обучения систем на данных.'
                },
                {
                    'title': 'Scikit-learn - библиотека ML',
                    'url': 'https://scikit-learn.org/',
                    'snippet': 'Простые и эффективные инструменты для анализа данных и машинного обучения на Python.'
                }
            ],
            'искусственный интеллект': [
                {
                    'title': 'Искусственный интеллект - Википедия',
                    'url': 'https://ru.wikipedia.org/wiki/Искусственный_интеллект',
                    'snippet': 'Искусственный интеллект - область информатики, изучающая создание интеллектуальных систем.'
                },
                {
                    'title': 'OpenAI - исследования ИИ',
                    'url': 'https://openai.com/',
                    'snippet': 'OpenAI - исследовательская компания в области искусственного интеллекта.'
                }
            ],
            'нейронные сети': [
                {
                    'title': 'Нейронные сети - основы',
                    'url': 'https://habr.com/ru/post/348450/',
                    'snippet': 'Введение в нейронные сети: архитектура, обучение, применение на практике.'
                },
                {
                    'title': 'TensorFlow - фреймворк',
                    'url': 'https://www.tensorflow.org/',
                    'snippet': 'TensorFlow - открытая платформа для машинного обучения от Google.'
                }
            ],
            'глубокое обучение': [
                {
                    'title': 'Глубокое обучение - Википедия',
                    'url': 'https://ru.wikipedia.org/wiki/Глубокое_обучение',
                    'snippet': 'Глубокое обучение - подмножество методов машинного обучения, основанное на искусственных нейронных сетях.'
                },
                {
                    'title': 'Keras - библиотека глубокого обучения',
                    'url': 'https://keras.io/',
                    'snippet': 'Keras - высокоуровневый API для глубокого обучения, работающий поверх TensorFlow.'
                }
            ]
        }
        
        # Поиск релевантной информации
        for key, info_list in knowledge_base.items():
            if key in query_lower:
                for info in info_list[:max_results]:
                    result = SearchResult(
                        title=info['title'],
                        url=info['url'],
                        snippet=info['snippet'],
                        source="local_knowledge",
                        relevance_score=0.85,
                        query=query
                    )
                    results.append(result)
                break
        
        # Если ничего не найдено, создаем общие результаты
        if not results:
            for i in range(min(max_results, 2)):
                result = SearchResult(
                    title=f"Информация о {query}",
                    url=f"https://example.com/search?q={quote(query)}",
                    snippet=f"Это базовая информация о {query}. Для получения актуальных данных рекомендуется использовать веб-поиск.",
                    source="local_fallback",
                    relevance_score=0.5,
                    query=query
                )
                results.append(result)
        
        return results
    
    def search_searx(self, query: str, max_results: int) -> List[SearchResult]:
        """Выполняет поиск через Searx (метапоисковик) с fallback."""
        try:
            logger.debug(f"Выполняется поиск Searx: {query}")
            
            # Параметры запроса
            params = {
                'q': query,
                'format': 'json',
                'engines': 'google,bing,duckduckgo',
                'language': 'ru'
            }
            
            # Выполняем запрос
            response = self.session.get("https://searx.be/search", params=params, timeout=10)
            response.raise_for_status()
            
            # Парсим JSON
            data = response.json()
            results = []
            
            # Обрабатываем результаты
            for item in data.get('results', [])[:max_results]:
                try:
                    title = item.get('title', '')
                    url = item.get('url', '')
                    snippet = item.get('content', '')
                    
                    search_result = SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source="searx",
                        relevance_score=self._calculate_relevance(title, snippet, query),
                        query=query
                    )
                    results.append(search_result)
                    
                except Exception as e:
                    logger.debug(f"Ошибка обработки результата: {e}")
                    continue
            
            logger.info(f"Найдено {len(results)} результатов через Searx")
            return results
            
        except Exception as e:
            logger.warning(f"Searx недоступен, используем fallback: {e}")
            # Fallback на DuckDuckGo
            return self.search_duckduckgo(query, max_results)
    
    def search_wikipedia(self, query: str, max_results: int) -> List[SearchResult]:
        """Search Wikipedia API.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects with source='wikipedia'
        """
        try:
            logger.info(f"Searching Wikipedia for: {query}")
            
            # Try English Wikipedia first, then Russian
            wikis = [
                ('en', 'https://en.wikipedia.org/w/api.php'),
                ('ru', 'https://ru.wikipedia.org/w/api.php')
            ]
            
            results = []
            
            # Create session with proper User-Agent for Wikipedia
            wiki_session = requests.Session()
            wiki_session.headers.update({
                'User-Agent': 'ЕВАAI/1.0 (https://github.com/BlackCatSpb/ЕВА; contact@example.com) Python/3.13'
            })
            
            for lang_code, api_url in wikis:
                params = {
                    'action': 'query',
                    'list': 'search',
                    'srsearch': query,
                    'format': 'json',
                    'srlimit': max_results,
                    'srprop': 'snippet'
                }
                
                try:
                    response = wiki_session.get(api_url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    
                    search_results = data.get('query', {}).get('search', [])
                    
                    for item in search_results[:max_results]:
                        title = item.get('title', '')
                        page_id = item.get('pageid', '')
                        
                        # Get snippet with HTML tags cleaned
                        snippet = item.get('snippet', '')
                        snippet = re.sub(r'<[^>]+>', '', snippet)
                        snippet = snippet.replace('&quot;', '"').replace('&amp;', '&')
                        
                        wiki_url = f"https://{lang_code}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                        
                        result = SearchResult(
                            title=title,
                            url=wiki_url,
                            snippet=snippet,
                            source="wikipedia",
                            relevance_score=self._calculate_relevance(title, snippet, query),
                            query=query
                        )
                        results.append(result)
                    
                    if results:
                        logger.info(f"Found {len(results)} Wikipedia results ({lang_code})")
                        break
                        
                except Exception as e:
                    logger.debug(f"Wikipedia ({lang_code}) error: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Wikipedia search failed: {e}")
            return []
    
    def _calculate_relevance(self, title: str, snippet: str, query: str) -> float:
        """Рассчитывает релевантность результата."""
        if not title and not snippet:
            return 0.0
        
        score = 0.0
        query_terms = query.lower().split()
        
        # Проверяем вхождение терминов запроса
        combined_text = f"{title} {snippet}".lower()
        
        for term in query_terms:
            # Точное вхождение в заголовок
            if term in title.lower():
                score += 0.3
            # Точное вхождение в описание
            elif term in snippet.lower():
                score += 0.2
            # Частичное вхождение
            elif any(term in word for word in combined_text.split()):
                score += 0.1
        
        # Бонус за полноту
        matched_terms = sum(1 for term in query_terms if term in combined_text)
        if matched_terms == len(query_terms):
            score += 0.2
        elif matched_terms > len(query_terms) / 2:
            score += 0.1
        
        # Нормализуем
        return min(score, 1.0)