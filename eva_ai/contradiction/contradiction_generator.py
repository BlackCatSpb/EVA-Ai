"""
ContradictionGenerator - генерация противоречий через самодиалог
Создаёт противоречия путём генерации различных точек зрения на концепт
"""
import time
import logging
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("eva_ai.contradiction.generator")


@dataclass
class GeneratedContradiction:
    """Сгенерированное противоречие с обеими точками зрения."""
    concept: str
    viewpoint_a: str  # Первая точка зрения
    viewpoint_b: str  # Противоположная точка зрения
    divergence_level: float
    reasoning_a: str  # Обоснование точки зрения A
    reasoning_b: str  # Обоснование точки зрения B
    resolution: Optional[str] = None  # Разрешение противоречия
    source_dialog: Optional[str] = None  # ID диалога, создавшего противоречие


class ContradictionGenerator:
    """
    Генерирует противоречия путём создания различных точек зрения на концепт.
    
    Флоу:
    1. Берёт концепт из графа или создаёт новый
    2. Генерирует две противоположные точки зрения через "диалог" (симуляцию)
    3. Формулирует их как противоречивые факты
    4. Сохраняет противоречие
    
    Примеры точек зрения:
    - "AI будет полезен человечеству" vs "AI опасен для человечества"
    - "Свобода важнее безопасности" vs "Безопасность важнее свободы"
    """
    
    def __init__(self, brain=None, fractal_graph=None):
        self.brain = brain
        self._fg = fractal_graph
        self._viewpoint_templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, List[tuple]]:
        """
        Загружает шаблоны для генерации противоположных точек зрения.
        
        Каждый шаблон - пара (утверждение, анти-утверждение)
        """
        return {
            'general': [
                ('{concept} является положительным явлением', 
                 '{concept} несёт негативные последствия'),
                ('{concept} приносит пользу обществу',
                 '{concept} создаёт проблемы для общества'),
                ('{concept} следует развивать',
                 '{concept} требует ограничений'),
                ('{concept} важен для прогресса',
                 '{concept} тормозит развитие'),
            ],
            'technology': [
                ('{concept} делает жизнь лучше',
                 '{concept} создаёт новые проблемы'),
                ('{concept} повышает эффективность',
                 '{concept} снижает качество'),
                ('{concept} автоматизирует рутину',
                 '{concept} лишает работы'),
                ('{concept} доступен каждому',
                 '{concept} увеличивает неравенство'),
            ],
            'science': [
                ('{concept} объясняет мир',
                 '{concept} ограничивает понимание'),
                ('{concept} даёт объективные знания',
                 '{concept} зависит от интерпретации'),
                ('{concept} доказан наукой',
                 '{concept} требует дальнейшего изучения'),
            ],
            'philosophy': [
                ('{concept} имеет объективную природу',
                 '{concept} является субъективным конструктом'),
                ('{concept} универсален',
                 '{concept} культурно-специфичен'),
                ('{concept} можно понять разумом',
                 '{concept} выходит за рамки рационального'),
            ]
        }
    
    def generate_contradiction(self, concept_name: str, 
                              domain: str = 'general') -> Optional[GeneratedContradiction]:
        """
        Генерирует противоречие для концепта.
        
        Args:
            concept_name: Имя концепта
            domain: Домен для выбора шаблонов
            
        Returns:
            Сгенерированное противоречие или None
        """
        # Выбираем шаблоны для домена
        templates = self._viewpoint_templates.get(domain, self._viewpoint_templates['general'])
        
        # Выбираем случайную пару
        template_pair = random.choice(templates)
        
        # Формируем точки зрения
        viewpoint_a = template_pair[0].format(concept=concept_name)
        viewpoint_b = template_pair[1].format(concept=concept_name)
        
        # Генерируем обоснование для каждой точки зрения
        reasoning_a = self._generate_reasoning(concept_name, viewpoint_a, 'positive')
        reasoning_b = self._generate_reasoning(concept_name, viewpoint_b, 'negative')
        
        # Вычисляем уровень расхождения (0.5-1.0 для сгенерированных)
        divergence = random.uniform(0.6, 0.95)
        
        contradiction = GeneratedContradiction(
            concept=concept_name,
            viewpoint_a=viewpoint_a,
            viewpoint_b=viewpoint_b,
            divergence_level=divergence,
            reasoning_a=reasoning_a,
            reasoning_b=reasoning_b,
            resolution=None,
            source_dialog=None
        )
        
        logger.info(f"Сгенерировано противоречие для '{concept_name}' (divergence: {divergence:.2f})")
        return contradiction
    
    def _generate_reasoning(self, concept: str, viewpoint: str, 
                           perspective: str) -> str:
        """
        Генерирует обоснование для точки зрения.
        
        Args:
            concept: Концепт
            viewpoint: Точка зрения
            perspective: 'positive' или 'negative'
            
        Returns:
            Обоснование
        """
        reasoning_templates = {
            'positive': [
                'Потому что {concept} создаёт возможности для развития',
                'Это подтверждается опытом использования {concept}',
                'Анализ показывает преимущества {concept}',
                'Исторически {concept} вёл к прогрессу',
            ],
            'negative': [
                'Потому что {concept} создаёт риски и неопределённость',
                'Это видно на примерах негативного влияния {concept}',
                'Анализ показывает проблемы {concept}',
                'Исторически подобное вело к кризисам',
            ]
        }
        
        templates = reasoning_templates.get(perspective, reasoning_templates['positive'])
        reasoning = random.choice(templates).format(concept=concept)
        
        return reasoning
    
    def save_contradiction(self, contradiction: GeneratedContradiction) -> Optional[str]:
        """
        Сохраняет противоречие в систему.
        
        Args:
            contradiction: Противоречие для сохранения
            
        Returns:
            ID противоречия или None
        """
        if not self.brain:
            logger.warning("Brain не доступен для сохранения противоречия")
            return None
        
        try:
            # Получаем contradiction_manager
            cm = getattr(self.brain, 'contradiction_manager', None)
            if not cm:
                logger.warning("ContradictionManager не доступен")
                return None
            
            # Формируем факты для противоречия
            conflicting_facts = [
                {
                    'value': contradiction.viewpoint_a,
                    'relation_type': 'viewpoint',
                    'source': 'generated',
                    'reasoning': contradiction.reasoning_a,
                    'timestamp': time.time()
                },
                {
                    'value': contradiction.viewpoint_b,
                    'relation_type': 'viewpoint',
                    'source': 'generated', 
                    'reasoning': contradiction.reasoning_b,
                    'timestamp': time.time()
                }
            ]
            
            # Создаём противоречие через manager
            contr_dict = {
                'id': f"contr_gen_{int(time.time())}_{hash(contradiction.concept) & 0xFFFF}",
                'concept': contradiction.concept,
                'conflicting_facts': conflicting_facts,
                'divergence_level': contradiction.divergence_level,
                'status': 'detected',
                'metadata': {
                    'type': 'generated',
                    'source_dialog': contradiction.source_dialog,
                    'auto_generated': True
                }
            }
            
            cm.add_contradiction(contr_dict)
            
            logger.info(f"Противоречие сохранено: {contr_dict['id']}")
            return contr_dict['id']
            
        except Exception as e:
            logger.error(f"Ошибка сохранения противоречия: {e}")
            return None
    
    def format_for_dialog(self, contradiction: GeneratedContradiction) -> str:
        """
        Форматирует противоречие в текст для самодиалога.
        
        Returns:
            Текст с формулировкой противоречия
        """
        return f"""Противоречие по концепту: {contradiction.concept}

Точка зрения A:
{contradiction.viewpoint_a}
Обоснование: {contradiction.reasoning_a}

Точка зрения B:
{contradiction.viewpoint_b}
Обоснование: {contradiction.reasoning_b}

Уровень расхождения: {contradiction.divergence_level:.2f}

Задача: проанализировать обе точки зрения и найти синтез или разрешение."""
    
    def generate_batch(self, concept_names: List[str], 
                      domain_map: Optional[Dict[str, str]] = None) -> List[GeneratedContradiction]:
        """
        Генерирует противоречия для нескольких концептов.
        
        Args:
            concept_names: Список имён концептов
            domain_map: Маппинг concept -> domain
            
        Returns:
            Список сгенерированных противоречий
        """
        contradictions = []
        
        for concept in concept_names:
            domain = domain_map.get(concept, 'general') if domain_map else 'general'
            
            contr = self.generate_contradiction(concept, domain)
            if contr:
                contradictions.append(contr)
        
        logger.info(f"Сгенерировано {len(contradictions)} противоречий для {len(concept_names)} концептов")
        return contradictions
    
    def auto_generate_for_unknown_concepts(self, min_concepts: int = 3) -> List[GeneratedContradiction]:
        """
        Автоматически генерирует противоречия для концептов без противоречий.
        
        Args:
            min_concepts: Минимальное количество концептов для генерации
            
        Returns:
            Список сгенерированных противоречий
        """
        if not self._fg:
            logger.warning("FractalGraph не доступен")
            return []
        
        try:
            # Находим концепты без противоречий
            concepts_without_contr = []
            
            for node_id, node in self._fg.storage.nodes.items():
                if hasattr(node, 'node_type') and node.node_type == 'concept':
                    concept_name = getattr(node, 'content', '')
                    if concept_name:
                        # Проверяем, есть ли уже противоречия
                        if not self._has_contradiction(concept_name):
                            concepts_without_contr.append(concept_name)
            
            if len(concepts_without_contr) < min_concepts:
                logger.info(f"Недостаточно концептов без противоречий ({len(concepts_without_contr)} < {min_concepts})")
                return []
            
            # Генерируем противоречия для случайных концептов
            selected = random.sample(concepts_without_contr, min(min_concepts, len(concepts_without_contr)))
            return self.generate_batch(selected)
            
        except Exception as e:
            logger.error(f"Ошибка авто-генерации противоречий: {e}")
            return []
    
    def _has_contradiction(self, concept_name: str) -> bool:
        """Проверяет, есть ли уже противоречия для концепта."""
        if not self.brain:
            return False
        
        try:
            cm = getattr(self.brain, 'contradiction_manager', None)
            if cm and hasattr(cm, 'contradictions'):
                for c in cm.contradictions:
                    if isinstance(c, dict) and c.get('concept') == concept_name:
                        return True
        except:
            pass
        
        return False
    
    def get_contradictions_for_prompt(self, concept_name: str) -> str:
        """
        Получает противоречия для концепта и формирует промпт.
        
        Сначала ищет в ContradictionManager, затем в hybrid_cache.
        
        Args:
            concept_name: Имя концепта
            
        Returns:
            Строка с контекстом противоречий для промпта
        """
        if not self.brain:
            return ""
        
        try:
            # 1. СНАЧАЛА ищем в ContradictionManager
            cm = getattr(self.brain, 'contradiction_manager', None)
            if cm and hasattr(cm, 'contradictions'):
                relevant_contradictions = []
                for c in cm.contradictions:
                    if isinstance(c, dict) and c.get('concept') == concept_name:
                        if c.get('status') == 'detected':
                            facts = c.get('conflicting_facts', [])
                            if len(facts) >= 2:
                                relevant_contradictions.append({
                                    'view_a': facts[0].get('value', ''),
                                    'view_b': facts[1].get('value', '')
                                })
                
                if relevant_contradictions:
                    return self._format_contradiction_prompt(concept_name, relevant_contradictions)
            
            # 2. Если не нашли - ищем в hybrid_cache
            cache_result = self._get_contradictions_from_cache(concept_name)
            if cache_result:
                return cache_result
            
            return ""
            
        except Exception as e:
            logger.debug(f"Ошибка формирования промпта противоречий: {e}")
            return ""
    
    def _get_contradictions_from_cache(self, concept_name: str) -> str:
        """Получает противоречия из hybrid_cache."""
        try:
            hybrid_cache = getattr(self.brain, 'hybrid_cache', None)
            if not hybrid_cache or not hasattr(hybrid_cache, 'get_contradictions_for_prompt'):
                return ""
            
            cache_result = hybrid_cache.get_contradictions_for_prompt(concept_name, limit=3)
            if cache_result:
                logger.debug(f"Противоречия из кеша для '{concept_name}'")
                return cache_result
            
        except Exception as e:
            logger.debug(f"Ошибка получения противоречий из кеша: {e}")
        
        return ""
    
    def _format_contradiction_prompt(self, concept_name: str, contradictions: list) -> str:
        """Форматирует противоречия в промпт."""
        prompt_parts = [f"Известные противоречивые точки зрения по '{concept_name}':"]
        for i, contr in enumerate(contradictions[:2], 1):
            prompt_parts.append(f"  {i}A. {contr['view_a'][:100]}")
            prompt_parts.append(f"  {i}B. {contr['view_b'][:100]}")
        
        prompt_parts.append("При ответе учитывай обе точки зрения и предложи сбалансированный подход.")
        
        return "\n".join(prompt_parts)
        
        try:
            cm = getattr(self.brain, 'contradiction_manager', None)
            if not cm or not hasattr(cm, 'contradictions'):
                return ""
            
            # Ищем противоречия для концепта
            relevant_contradictions = []
            for c in cm.contradictions:
                if isinstance(c, dict) and c.get('concept') == concept_name:
                    if c.get('status') == 'detected':
                        facts = c.get('conflicting_facts', [])
                        if len(facts) >= 2:
                            relevant_contradictions.append({
                                'view_a': facts[0].get('value', ''),
                                'view_b': facts[1].get('value', '')
                            })
            
            if not relevant_contradictions:
                return ""
            
            # Формируем промпт
            prompt_parts = [f"Известные противоречивые точки зрения по '{concept_name}':"]
            for i, contr in enumerate(relevant_contradictions[:2], 1):
                prompt_parts.append(f"  {i}A. {contr['view_a'][:100]}")
                prompt_parts.append(f"  {i}B. {contr['view_b'][:100]}")
            
            prompt_parts.append("При ответе учитывай обе точки зрения и предложь сбалансированный подход.")
            
            return "\n".join(prompt_parts)
            
        except Exception as e:
            logger.debug(f"Ошибка формирования промпта противоречий: {e}")
            return ""
    
    def save_resolution(self, concept_name: str, resolution_text: str, source: str = "dialog"):
        """
        Сохраняет разрешение противоречия.
        
        Args:
            concept_name: Имя концепта
            resolution_text: Текст разрешения
            source: Источник разрешения
        """
        if not self.brain:
            return
        
        try:
            cm = getattr(self.brain, 'contradiction_manager', None)
            if not cm:
                return
            
            # Ищем противоречие
            for c in cm.contradictions:
                if isinstance(c, dict) and c.get('concept') == concept_name:
                    if c.get('status') == 'detected':
                        # Обновляем статус
                        c['status'] = 'resolved'
                        c['resolution'] = resolution_text
                        c['resolved_at'] = time.time()
                        c['resolution_source'] = source
                        
                        logger.info(f"Противоречие для '{concept_name}' разрешено")
                        break
            
            # Сохраняем разрешение как факт в концепт
            if self.brain and hasattr(self.brain, 'concept_extractor'):
                self.brain.concept_extractor.save_learned_concept(
                    concept_name=concept_name,
                    new_facts=[f"Разрешение противоречия: {resolution_text}"],
                    source=source
                )
            
        except Exception as e:
            logger.error(f"Ошибка сохранения разрешения: {e}")


def create_contradiction_generator(brain=None, fractal_graph=None) -> ContradictionGenerator:
    """Factory function для создания ContradictionGenerator."""
    return ContradictionGenerator(brain=brain, fractal_graph=fractal_graph)
