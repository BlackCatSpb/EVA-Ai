"""Куратор Графа Знаний для ЕВА.

Этот модуль отвечает за:
1. Контроль качества связей в графе
2. Автоматическое создание семантических связей
3. Оптимизацию структуры графа
4. Использование fractal_model для генерации промтов

Пример: снег -> белый, искрящийся, холодный, зимний, первый
"""
import logging
import time
import threading
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("eva.graph_curator")

@dataclass
class SemanticLink:
    """Семантическая связь между понятиями."""
    source: str
    target: str
    relation: str  # is, has, can, related, etc.
    weight: float = 1.0
    context: str = ""

class GraphCurator:
    """
    Куратор графа знаний.
    
    Обеспечивает:
    - Правильную адресацию узлов
    - Логические связи между понятиями
    - Автоматическое улучшение графа
    """
    
    # Словарь семантических ассоциаций
    SEMANTIC_ASSOCIATIONS = {
        # Природа
        'снег': ['белый', 'холодный', 'зимний', 'искрящийся', 'пушистый', 'первый', 'сугроб', 'февраль'],
        'дождь': ['мокрый', 'холодный', 'тёплый', 'ливень', 'капли', 'зонт', 'облака'],
        'солнце': ['яркое', 'тёплое', 'жёлтое', 'лето', 'утро', 'рассвет', 'закат'],
        'море': ['синее', 'голубое', 'тёплое', 'солёное', 'волны', 'пляж', 'отпуск'],
        'лес': ['зелёный', 'густой', 'тихий', 'таинственный', 'деревья', 'грибы', 'тропа'],
        
        # Люди
        'человек': ['мужчина', 'женщина', 'ребёнок', 'взрослый', 'друг', 'враг'],
        'друг': ['близкий', 'верный', 'надёжный', 'поддержка', 'доверие'],
        'мама': ['забота', 'любовь', 'тёплая', 'добрая', 'семья'],
        
        # Эмоции
        'счастье': ['радость', 'удовольствие', 'удовольствие', 'улыбка', 'смех'],
        'грусть': ['печаль', 'тоска', 'одиночество', 'слёзы', 'потеря'],
        'страх': ['ужас', 'тревога', 'беспокойство', 'опасность'],
        
        # Время
        'утро': ['рассвет', 'свежесть', 'кофе', 'начало', 'свет'],
        'вечер': ['закат', 'отдых', 'ужин', 'тишина', 'темнота'],
        'ночь': ['темнота', 'звёзды', 'луна', 'сон', 'покой'],
        
        # Еда
        'еда': ['вкусная', 'горячая', 'холодная', 'полезная', 'сытная'],
        'кофе': ['горячий', 'ароматный', 'утренний', 'крепкий', 'бодрящий'],
        
        # Технологии
        'компьютер': ['быстрый', 'умный', 'технологичный', 'современный'],
        'программирование': ['код', 'алгоритм', 'логика', 'создание'],
        
        # Животные
        'кот': ['мягкий', 'пушистый', 'ласковый', 'игривый', 'домашний'],
        'собака': ['верный', 'преданный', 'дружелюбный', 'энергичный', 'умный'],
        
        # Музыка
        'музыка': ['мелодичная', 'ритмичная', 'грустная', 'весёлая', 'классическая'],
        
        # Искусство
        'картина': ['красивая', 'яркая', 'выразительная', 'живописная'],
        'книга': ['интересная', 'захватывающая', 'познавательная', 'художественная'],
    }
    
    def __init__(self, brain=None, config: Optional[Dict] = None):
        """
        Инициализация куратора графа.
        
        Args:
            brain: Ссылка на ядро ЕВА
            config: Конфигурация
        """
        self.brain = brain
        self.config = config or {}
        
        # Настройки
        self.enabled = self.config.get('enabled', True)
        self.curation_interval = self.config.get('curation_interval', 300)  # 5 минут
        self.min_link_weight = self.config.get('min_link_weight', 0.3)
        self.max_links_per_node = self.config.get('max_links_per_node', 20)
        
        # Статистика
        self.stats = {
            'nodes_curated': 0,
            'links_created': 0,
            'links_removed': 0,
            'semantic_associations_found': 0
        }
        
        self.running = False
        self.stop_event = threading.Event()
        self.curator_thread = None
        
        logger.info("GraphCurator инициализирован")
    
    def start(self):
        """Запускает куратора в фоновом режиме."""
        if not self.enabled:
            logger.info("GraphCurator отключен в конфигурации")
            return False
        
        if self.running:
            logger.warning("GraphCurator уже запущен")
            return False
        
        self.running = True
        self.stop_event.clear()
        
        self.curator_thread = threading.Thread(target=self._curation_loop, daemon=True)
        self.curator_thread.start()
        
        logger.info("GraphCurator запущен")
        return True
    
    def stop(self):
        """Останавливает куратора."""
        if not self.running:
            return
        
        self.stop_event.set()
        if self.curator_thread and self.curator_thread.is_alive():
            self.curator_thread.join(timeout=5)
        
        self.running = False
        logger.info("GraphCurator остановлен")
    
    def _curation_loop(self):
        """Основной цикл куратора."""
        logger.info("Запущен цикл куратора графа")
        
        while not self.stop_event.is_set():
            try:
                self._curate_graph()
            except Exception as e:
                logger.error(f"Ошибка куратора: {e}", exc_info=True)
            
            # Ждем следующий цикл
            self.stop_event.wait(timeout=self.curation_interval)
    
    def _curate_graph(self):
        """Выполняет кураторскую работу по улучшению графа."""
        knowledge_graph = self._get_knowledge_graph()
        if not knowledge_graph:
            return
        
        logger.info("Начинаю кураторскую обработку графа")
        
        try:
            # Получаем все узлы
            nodes = self._get_all_nodes(knowledge_graph)
            if not nodes:
                return
            
            new_links = 0
            
            for node in nodes:
                node_name = getattr(node, 'name', '') or ''
                if not node_name:
                    continue
                
                # Ищем семантические ассоциации
                links = self._find_semantic_links(node_name)
                
                for link in links:
                    try:
                        # Проверяем, существует ли уже связь
                        if not self._link_exists(knowledge_graph, node_name, link.target):
                            # Создаем новую связь
                            self._create_link(knowledge_graph, node_name, link.target, link.relation)
                            new_links += 1
                    except Exception as e:
                        logger.debug(f"Ошибка создания связи: {e}")
            
            if new_links > 0:
                self.stats['links_created'] += new_links
                logger.info(f"Куратор: создано {new_links} новых связей")
                
        except Exception as e:
            logger.error(f"Ошибка кураторской обработки: {e}")
    
    def _get_knowledge_graph(self):
        """Получает ссылку на граф знаний."""
        if not self.brain:
            return None
        return getattr(self.brain, 'knowledge_graph', None)
    
    def _get_all_nodes(self, kg) -> List:
        """Получает все узлы из графа."""
        try:
            if hasattr(kg, 'get_all_nodes'):
                return kg.get_all_nodes()
            elif hasattr(kg, 'nodes'):
                return list(kg.nodes.values()) if hasattr(kg.nodes, 'values') else []
            return []
        except Exception as e:
            logger.debug(f"Ошибка получения узлов: {e}")
            return []
    
    def _find_semantic_links(self, node_name: str) -> List[SemanticLink]:
        """Ищет семантические связи для узла."""
        links = []
        node_lower = node_name.lower().strip()
        
        # Проверяем словарь ассоциаций
        if node_lower in self.SEMANTIC_ASSOCIATIONS:
            for assoc in self.SEMANTIC_ASSOCIATIONS[node_lower]:
                links.append(SemanticLink(
                    source=node_name,
                    target=assoc,
                    relation='related',
                    weight=0.8,
                    context=f"Семантическая ассоциация: {node_name} -> {assoc}"
                ))
        
        # Проверяем обратные связи
        for key, values in self.SEMANTIC_ASSOCIATIONS.items():
            if node_lower in values:
                links.append(SemanticLink(
                    source=node_name,
                    target=key,
                    relation='related',
                    weight=0.7,
                    context=f"Семантическая ассоциация: {node_name} -> {key}"
                ))
        
        self.stats['semantic_associations_found'] += len(links)
        return links
    
    def _link_exists(self, kg, source: str, target: str) -> bool:
        """Проверяет, существует ли связь."""
        try:
            if hasattr(kg, 'edge_exists'):
                return kg.edge_exists(source, target)
            elif hasattr(kg, 'get_edges'):
                edges = kg.get_edges(source)
                return any(getattr(e, 'target', '') == target for e in edges)
            return False
        except Exception:
            return False
    
    def _create_link(self, kg, source: str, target: str, relation: str = 'related'):
        """Создает связь в графе."""
        try:
            if hasattr(kg, 'add_edge'):
                kg.add_edge(source, target, relation=relation, weight=0.8)
            elif hasattr(kg, 'connect'):
                kg.connect(source, target, relation=relation)
        except Exception as e:
            logger.debug(f"Ошибка создания связи {source}->{target}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику куратора."""
        return {
            'running': self.running,
            **self.stats
        }
