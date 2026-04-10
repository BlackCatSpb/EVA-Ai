"""
Wikipedia Loader для EVA
Загрузка статей из русскоязычной Википедии через API.
Поддерживает пакетную загрузку, категории и поиск.
"""
import os
import json
import time
import logging
import threading
from typing import Optional, List, Dict, Any
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import quote

logger = logging.getLogger("eva_ai.wikipedia_loader")

WIKIPEDIA_API_URL = "https://ru.wikipedia.org/w/api.php"

class WikipediaLoader:
    """
    Загрузчик статей из русскоязычной Википедии.
    
    Использует MediaWiki API для получения статей.
    """
    
    def __init__(self, kb=None, batch_size: int = 10, delay: float = 0.5):
        """
        Args:
            kb: Экземпляр WikipediaKnowledgeBase
            batch_size: Количество статей за один запрос
            delay: Задержка между запросами (секунды)
        """
        self.kb = kb
        self.batch_size = batch_size
        self.delay = delay
        self._running = False
        self._stop_event = threading.Event()
        self._stats = {
            'articles_loaded': 0,
            'articles_skipped': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None,
        }
    
    def _api_request(self, params: Dict) -> Optional[Dict]:
        """Выполняет запрос к Wikipedia API."""
        param_str = "&".join(f"{k}={quote(str(v))}" for k, v in params.items())
        url = f"{WIKIPEDIA_API_URL}?{param_str}&format=json"
        
        try:
            req = Request(url, headers={'User-Agent': 'EVA-Ai/1.0 (educational bot; https://github.com/BlackCatSpb/EVA-Ai)'})
            with urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            logger.warning(f"Ошибка запроса к Wikipedia API: {e}")
            return None
    
    def get_article(self, title: str) -> Optional[Dict[str, str]]:
        """
        Получает статью по заголовку.
        
        Returns:
            {'title': str, 'text': str, 'url': str} или None
        """
        params = {
            'action': 'query',
            'prop': 'extracts',
            'exlimit': '1',
            'explaintext': '1',
            'titles': title,
        }
        
        result = self._api_request(params)
        if not result or 'query' not in result:
            return None
        
        pages = result['query'].get('pages', {})
        for page_id, page in pages.items():
            if page_id == '-1':  # Страница не найдена
                return None
            
            extract = page.get('extract', '')
            if not extract.strip():
                return None
            
            return {
                'title': page.get('title', title),
                'text': extract,
                'url': f"https://ru.wikipedia.org/wiki/{quote(page.get('title', ''))}",
            }
        
        return None
    
    def search_articles(self, query: str, limit: int = 10) -> List[str]:
        """
        Ищет статьи по запросу.
        
        Returns:
            Список заголовков статей
        """
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'srlimit': str(limit),
            'srprop': 'title',
        }
        
        result = self._api_request(params)
        if not result or 'query' not in result:
            return []
        
        return [item['title'] for item in result['query'].get('search', [])]
    
    def get_category_articles(self, category: str, limit: int = 50) -> List[str]:
        """
        Получает статьи из категории.
        
        Args:
            category: Название категории (без "Категория:")
            limit: Максимальное количество
        
        Returns:
            Список заголовков статей
        """
        params = {
            'action': 'query',
            'list': 'categorymembers',
            'cmtitle': f"Категория:{category}",
            'cmlimit': str(min(limit, 500)),
            'cmtype': 'page',
        }
        
        result = self._api_request(params)
        if not result or 'query' not in result:
            return []
        
        return [item['title'] for item in result['query'].get('categorymembers', [])]
    
    def get_random_articles(self, limit: int = 10) -> List[str]:
        """
        Получает случайные статьи.
        
        Returns:
            Список заголовков
        """
        params = {
            'action': 'query',
            'list': 'random',
            'rnlimit': str(limit),
            'rnnamespace': '0',  # Только статьи
        }
        
        result = self._api_request(params)
        if not result or 'query' not in result:
            return []
        
        return [item['title'] for item in result['query'].get('random', [])]
    
    def load_article(self, title: str, category: str = None) -> bool:
        """
        Загружает одну статью и сохраняет в базу знаний.
        
        Returns:
            True если успешно загружена
        """
        if self.kb is None:
            logger.error("WikipediaKnowledgeBase не инициализирован")
            return False
        
        article = self.get_article(title)
        if not article:
            self._stats['articles_skipped'] += 1
            return False
        
        self.kb.add_article(
            title=article['title'],
            text=article['text'],
            url=article['url'],
            category=category,
        )
        self._stats['articles_loaded'] += 1
        return True
    
    def load_batch(self, titles: List[str], category: str = None) -> Dict[str, int]:
        """
        Загружает пакет статей.
        
        Returns:
            Статистика загрузки
        """
        loaded = 0
        skipped = 0
        
        for i, title in enumerate(titles):
            if self._stop_event.is_set():
                break
            
            if self.load_article(title, category):
                loaded += 1
            else:
                skipped += 1
            
            if i < len(titles) - 1:
                time.sleep(self.delay)
        
        return {'loaded': loaded, 'skipped': skipped}
    
    def load_category(self, category: str, limit: int = 50) -> Dict[str, int]:
        """
        Загружает все статьи из категории.
        
        Returns:
            Статистика загрузки
        """
        logger.info(f"Загрузка категории: {category} (до {limit} статей)")
        titles = self.get_category_articles(category, limit)
        logger.info(f"Найдено {len(titles)} статей в категории '{category}'")
        return self.load_batch(titles, category=category)
    
    def load_random(self, limit: int = 20) -> Dict[str, int]:
        """
        Загружает случайные статьи.
        
        Returns:
            Статистика загрузки
        """
        logger.info(f"Загрузка {limit} случайных статей")
        titles = self.get_random_articles(limit)
        return self.load_batch(titles, category='random')
    
    def load_topic(self, topic: str, limit: int = 20) -> Dict[str, int]:
        """
        Загружает статьи по теме через поиск.
        
        Returns:
            Статистика загрузки
        """
        logger.info(f"Загрузка статей по теме: {topic}")
        titles = self.search_articles(topic, limit)
        return self.load_batch(titles, category=topic)
    
    def start_auto_learning(self, categories: List[str] = None, 
                           articles_per_category: int = 20,
                           interval_hours: float = 24,
                           include_random: int = 10):
        """
        Запускает автоматическое изучение Википедии в фоновом потоке.
        
        Args:
            categories: Список категорий для изучения
            articles_per_category: Количество статей из каждой категории
            interval_hours: Интервал между циклами (часы)
            include_random: Количество случайных статей за цикл
        """
        if self._running:
            logger.warning("Автообучение уже запущено")
            return
        
        self._running = True
        self._stop_event.clear()
        self._stats['start_time'] = datetime.now().isoformat()
        
        default_categories = categories or [
            'Наука',
            'Математика',
            'Физика',
            'Химия',
            'Биология',
            'Информатика',
            'История',
            'География',
            'Литература',
            'Философия',
        ]
        
        def _learning_loop():
            logger.info(f"Автообучение Википедии запущено: {len(default_categories)} категорий, "
                       f"{articles_per_category} статей/категория, интервал {interval_hours}ч")
            
            while not self._stop_event.is_set():
                cycle_start = datetime.now().isoformat()
                logger.info(f"Начало цикла обучения: {cycle_start}")
                
                cycle_stats = {'loaded': 0, 'skipped': 0, 'errors': 0}
                
                # Загружаем статьи из категорий
                for category in default_categories:
                    if self._stop_event.is_set():
                        break
                    try:
                        result = self.load_category(category, articles_per_category)
                        cycle_stats['loaded'] += result.get('loaded', 0)
                        cycle_stats['skipped'] += result.get('skipped', 0)
                    except Exception as e:
                        logger.error(f"Ошибка загрузки категории '{category}': {e}")
                        cycle_stats['errors'] += 1
                
                # Случайные статьи
                if include_random > 0:
                    try:
                        result = self.load_random(include_random)
                        cycle_stats['loaded'] += result.get('loaded', 0)
                        cycle_stats['skipped'] += result.get('skipped', 0)
                    except Exception as e:
                        logger.error(f"Ошибка загрузки случайных статей: {e}")
                        cycle_stats['errors'] += 1
                
                self._stats['end_time'] = datetime.now().isoformat()
                logger.info(f"Цикл обучения завершён: загружено {cycle_stats['loaded']}, "
                           f"пропущено {cycle_stats['skipped']}, ошибок {cycle_stats['errors']}")
                
                # Ждём следующий цикл
                self._stop_event.wait(timeout=interval_hours * 3600)
            
            self._running = False
            logger.info("Автообучение Википедии остановлено")
        
        thread = threading.Thread(target=_learning_loop, daemon=True, name="wikipedia-auto-learn")
        thread.start()
    
    def stop_auto_learning(self):
        """Останавливает автоматическое изучение."""
        self._stop_event.set()
        logger.info("Остановка автообучения Википедии...")
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика загрузчика."""
        stats = dict(self._stats)
        stats['running'] = self._running
        if self.kb:
            stats['kb_stats'] = self.kb.get_stats()
        return stats


# Singleton
_loader = None
_loader_lock = threading.Lock()


def get_wikipedia_loader(kb=None) -> WikipediaLoader:
    """Возвращает singleton загрузчик."""
    global _loader
    with _loader_lock:
        if _loader is None:
            from eva_ai.knowledge.wikipedia_kb import get_wikipedia_kb
            _kb = kb or get_wikipedia_kb()
            _loader = WikipediaLoader(kb=_kb)
        return _loader
