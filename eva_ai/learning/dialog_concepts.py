"""
Dialog Concepts Integration - интеграция концептов и противоречий в самодиалог
Позволяет самодиалогам обсуждать концепты и разрешать противоречия
"""
import time
import logging
from typing import Dict, List, Any, Optional

from eva_ai.learning.dialog_types import DialogRole, DialogTurn, SelfDialog

logger = logging.getLogger("eva_ai.self_dialog_learning")


class DialogConceptsMixin:
    """
    Mixin для интеграции концептов и противоречий в самодиалог.
    
    Добавляет:
    1. Генерацию тем из концептов
    2. Обсуждение противоречий в диалоге
    3. Сохранение результатов в кеш контекста
    """
    
    MAX_RESOLVED_KNOWLEDGE = 200
    MAX_CONCEPT_QUEUE = 100
    MAX_CONTRADICTION_TOPICS = 100
    
    def __init__(self, *args, **kwargs):
        self._concept_queue = []  # Очередь концептов для обсуждения
        self._contradiction_topics = []  # Противоречия для разрешения
        self._resolved_knowledge = []  # Разрешённые знания
        
    def queue_concept_for_dialog(self, concept_name: str, 
                                 priority: float = 0.5):
        """
        Добавляет концепт в очередь для обсуждения в самодиалоге.
        
        Args:
            concept_name: Имя концепта
            priority: Приоритет (0.0-1.0)
        """
        self._concept_queue.append({
            'name': concept_name,
            'priority': priority,
            'queued_at': time.time()
        })
        # Сортируем по приоритету
        self._concept_queue.sort(key=lambda x: x['priority'], reverse=True)
        
        # Ограничиваем размер очереди
        if len(self._concept_queue) > self.MAX_CONCEPT_QUEUE:
            self._concept_queue = self._concept_queue[:self.MAX_CONCEPT_QUEUE]
        
        logger.debug(f"Концепт '{concept_name}' добавлен в очередь (priority: {priority})")
    
    def queue_contradiction_for_resolution(self, contradiction_id: str,
                                          concept: str,
                                          priority: float = 0.7):
        """
        Добавляет противоречие в очередь для разрешения.
        
        Args:
            contradiction_id: ID противоречия
            concept: Имя концепта
            priority: Приоритет разрешения
        """
        self._contradiction_topics.append({
            'contradiction_id': contradiction_id,
            'concept': concept,
            'priority': priority,
            'queued_at': time.time()
        })
        
        # Ограничиваем размер очереди
        if len(self._contradiction_topics) > self.MAX_CONTRADICTION_TOPICS:
            self._contradiction_topics = self._contradiction_topics[:self.MAX_CONTRADICTION_TOPICS]
        
        logger.debug(f"Противоречие '{contradiction_id}' добавлено в очередь")
    
    def _get_next_dialog_topic(self) -> Optional[Dict[str, Any]]:
        """
        Получает следующую тему для самодиалога.
        
        Приоритет:
        1. Противоречия для разрешения
        2. Концепты из очереди
        3. Темы из истории разговоров (fallback)
        """
        # Сначала проверяем противоречия
        if self._contradiction_topics:
            contr = self._contradiction_topics.pop(0)
            return {
                'type': 'contradiction',
                'title': f"Разрешение противоречия: {contr['concept']}",
                'data': contr
            }
        
        # Затем концепты
        if self._concept_queue:
            concept = self._concept_queue.pop(0)
            return {
                'type': 'concept',
                'title': f"Исследование концепта: {concept['name']}",
                'data': concept
            }
        
        return None
    
    def _get_unified_generator(self):
        """Получить UnifiedGenerator из brain если доступен."""
        try:
            if self.brain and hasattr(self.brain, 'two_model_pipeline'):
                from eva_ai.core.pipeline_adapter import PipelineAdapter
                if isinstance(self.brain.two_model_pipeline, PipelineAdapter):
                    return self.brain.two_model_pipeline._generator
        except Exception as e:
            logger.debug(f"Could not get unified generator: {e}")
        return None
    
    def _run_concept_dialog(self, dialog: SelfDialog, concept_data: Dict[str, Any]):
        """
        Выполняет самодиалог для исследования концепта.
        
        Использует UnifiedGenerator (LOGIC -> CONTEXT) для генерации ответов.
        
        Роли:
        - ASSISTANT: Представляет базовое определение концепта
        - CRITIC: Выявляет противоречивые аспекты
        - LEARNER: Предлагает направления для углубления
        - TEACHER: Даёт рекомендации по изучению
        """
        concept_name = concept_data['name']
        
        # Получаем информацию о концепте из графа
        concept_info = self._get_concept_info(concept_name)
        
        # Используем UnifiedGenerator если доступен
        generator = self._get_unified_generator()
        
        if generator:
            # Turn 1: ASSISTANT - базовое определение через LLM
            prompt_intro = f"""Ты исследуешь концепт "{concept_name}". 

Домен: {concept_info.get('domain', 'общий')}
Связанные понятия: {concept_info.get('related_concepts', [])}

Дай краткое, но информативное определение этого концепта. Включи ключевые характеристики и применение."""
            
            result = generator.generate_iterative(
                query=prompt_intro,
                max_tokens_logic=128,
                max_tokens_context=256,
                temperature=0.7
            )
            assistant_content = result.text if result else self._generate_concept_intro(concept_name, concept_info)
        else:
            assistant_content = self._generate_concept_intro(concept_name, concept_info)
        
        dialog.turns.append(DialogTurn(
            role=DialogRole.ASSISTANT,
            content=assistant_content,
            timestamp=time.time()
        ))
        
        # Turn 2: CRITIC - поиск противоречий через LLM
        if generator:
            prompt_critic = f"""Ты критически анализируешь концепт "{concept_name}".

Предыдущий ответ: {assistant_content[:200]}

Выяви возможные проблемы, противоречия или неполноту в определении. Какие аспекты требуют уточнения?"""
            
            result = generator.generate_iterative(
                query=prompt_critic,
                max_tokens_logic=128,
                max_tokens_context=256,
                temperature=0.7
            )
            critic_content = result.text if result else self._generate_concept_criticism(concept_name, concept_info)
        else:
            critic_content = self._generate_concept_criticism(concept_name, concept_info)
        
        dialog.turns.append(DialogTurn(
            role=DialogRole.CRITIC,
            content=critic_content,
            timestamp=time.time()
        ))
        
        # Turn 3: LEARNER - направления через LLM
        if generator:
            prompt_learner = f"""Ты предлагаешь направления для углублённого изучения концепта "{concept_name}".

Основные вопросы: {critic_content[:200]}

Какие шаги нужно предпринять для полного понимания этого концепта?"""
            
            result = generator.generate_iterative(
                query=prompt_learner,
                max_tokens_logic=128,
                max_tokens_context=256,
                temperature=0.7
            )
            learner_content = result.text if result else self._generate_learning_directions(concept_name, concept_info)
        else:
            learner_content = self._generate_learning_directions(concept_name, concept_info)
        
        dialog.turns.append(DialogTurn(
            role=DialogRole.LEARNER,
            content=learner_content,
            timestamp=time.time()
        ))
        
        # Turn 4: TEACHER - рекомендации через LLM
        if generator:
            prompt_teacher = f"""Ты даёшь финальные рекомендации по изучению концепта "{concept_name}".

Изучено: {concept_name}
Направления: {learner_content[:200]}

Сформулируй практические рекомендации для использования этого знания."""
            
            result = generator.generate_iterative(
                query=prompt_teacher,
                max_tokens_logic=128,
                max_tokens_context=256,
                temperature=0.7
            )
            teacher_content = result.text if result else self._generate_teaching_recommendations(concept_name, concept_info)
        else:
            teacher_content = self._generate_teaching_recommendations(concept_name, concept_info)
        
        dialog.turns.append(DialogTurn(
            role=DialogRole.TEACHER,
            content=teacher_content,
            timestamp=time.time()
        ))
        
        # Сохраняем результаты
        self._save_concept_dialog_results(dialog, concept_name)
    
    def _run_contradiction_dialog(self, dialog: SelfDialog, 
                                 contradiction_data: Dict[str, Any]):
        """
        Выполняет самодиалог для разрешения противоречия.
        
        Роли:
        - ASSISTANT: Представляет обе точки зрения
        - CRITIC: Анализирует сильные и слабые стороны
        - LEARNER: Ищет синтез или компромисс
        - TEACHER: Формулирует разрешение
        """
        concept = contradiction_data['concept']
        contr_id = contradiction_data['contradiction_id']
        
        # Получаем детали противоречия
        contr_info = self._get_contradiction_info(contr_id)
        
        # Получаем генератор
        generator = self._get_unified_generator()
        
        # Turn 1: Assistant представляет противоречие
        if generator:
            facts = contr_info.get('conflicting_facts', [])
            fact_a = facts[0].get('value', 'Точка зрения A') if len(facts) >= 1 else 'Точка зрения A'
            fact_b = facts[1].get('value', 'Точка зрения B') if len(facts) >= 2 else 'Противоположная точка зрения'
            
            prompt_assistant = f"""Представь противоречие по концепту "{concept}":

Точка зрения A: {fact_a}
Точка зрения B: {fact_b}

Сформулируй это противоречие чётко и объективно."""
            
            result = generator.generate_iterative(
                query=prompt_assistant,
                max_tokens_logic=128,
                max_tokens_context=256,
                temperature=0.7
            )
            assistant_content = result.text if result else self._present_contradiction(concept, contr_info)
        else:
            assistant_content = self._present_contradiction(concept, contr_info)
        
        dialog.turns.append(DialogTurn(
            role=DialogRole.ASSISTANT,
            content=assistant_content,
            timestamp=time.time()
        ))
        
        # Turn 2: Critic анализирует
        if generator:
            prompt_critic = f"""Проанализируй противоречие по концепту "{concept}":

{assistant_content[:300]}

Какие сильные и слабые стороны у каждой точки зрения?"""
            
            result = generator.generate_iterative(
                query=prompt_critic,
                max_tokens_logic=128,
                max_tokens_context=256,
                temperature=0.7
            )
            critic_content = result.text if result else self._analyze_contradiction_sides(concept, contr_info)
        else:
            critic_content = self._analyze_contradiction_sides(concept, contr_info)
        
        dialog.turns.append(DialogTurn(
            role=DialogRole.CRITIC,
            content=critic_content,
            timestamp=time.time()
        ))
        
        # Turn 3: Learner ищет синтез
        if generator:
            prompt_learner = f"""Найди синтез для противоречия "{concept}":

Точка A: {critic_content[:150]}
Точка B: {critic_content[150:300] if len(critic_content) > 150 else ''}

Предложи третий путь или компромисс."""
            
            result = generator.generate_iterative(
                query=prompt_learner,
                max_tokens_logic=128,
                max_tokens_context=256,
                temperature=0.7
            )
            learner_content = result.text if result else self._synthesize_contradiction(concept, contr_info)
        else:
            learner_content = self._synthesize_contradiction(concept, contr_info)
        
        dialog.turns.append(DialogTurn(
            role=DialogRole.LEARNER,
            content=learner_content,
            timestamp=time.time()
        ))
        
        # Turn 4: Teacher формулирует разрешение
        if generator:
            prompt_teacher = f"""Сформулируй финальное разрешение противоречия "{concept}":

Анализ: {critic_content[:200]}
Синтез: {learner_content[:200]}

Дай чёткую резолюцию, которая объединяет обе точки зрения."""
            
            result = generator.generate_iterative(
                query=prompt_teacher,
                max_tokens_logic=128,
                max_tokens_context=256,
                temperature=0.7
            )
            resolution = result.text if result else self._formulate_resolution(concept, contr_info, dialog.turns)
        else:
            resolution = self._formulate_resolution(concept, contr_info, dialog.turns)
        
        dialog.turns.append(DialogTurn(
            role=DialogRole.TEACHER,
            content=resolution,
            timestamp=time.time()
        ))
        
        # Сохраняем разрешение
        self._save_contradiction_resolution(contr_id, resolution, dialog)
    
    def _get_concept_info(self, concept_name: str) -> Dict[str, Any]:
        """Получает информацию о концепте из графа."""
        if not self.brain or not hasattr(self.brain, 'knowledge_graph'):
            return {}
        
        try:
            kg = self.brain.knowledge_graph
            if hasattr(kg, 'search_nodes'):
                results = kg.search_nodes(concept_name, limit=1)
                if results:
                    return results[0]
        except Exception as e:
            logger.debug(f"Ошибка получения инфо о концепте: {e}")
        
        return {}
    
    def _get_contradiction_info(self, contradiction_id: str) -> Dict[str, Any]:
        """Получает информацию о противоречии."""
        if not self.brain:
            return {}
        
        try:
            cm = getattr(self.brain, 'contradiction_manager', None)
            if cm and hasattr(cm, 'contradictions'):
                for c in cm.contradictions:
                    if isinstance(c, dict) and c.get('id') == contradiction_id:
                        return c
        except Exception as e:
            logger.debug(f"Ошибка получения инфо о противоречии: {e}")
        
        return {}
    
    def _generate_concept_intro(self, concept: str, info: Dict) -> str:
        """Генерирует введение о концепте."""
        domain = info.get('domain', 'общий')
        return f"""Изучаем концепт: {concept}

Домен: {domain}

Базовое определение: {concept} - это ключевое понятие требующее понимания. 
Нужно рассмотреть его свойства, связи с другими понятиями и практическое применение."""
    
    def _generate_concept_criticism(self, concept: str, info: Dict) -> str:
        """Генерирует критический анализ концепта."""
        return f"""Критический анализ концепта '{concept}':

Возможные проблемы:
1. Определение может быть неполным или двусмысленным
2. Связи с другими понятиями не ясны
3. Могут существовать противоречивые интерпретации
4. Недостаточно примеров использования

Требуется углублённое исследование для выявления всех аспектов."""
    
    def _generate_learning_directions(self, concept: str, info: Dict) -> str:
        """Генерирует направления для изучения."""
        return f"""Направления для изучения '{concept}':

1. Найти формальное определение и источники
2. Изучить связанные понятия и контекст
3. Найти примеры применения
4. Выявить противоречивые точки зрения
5. Проверить актуальность информации

Приоритет: создание полной картины понятия."""
    
    def _generate_teaching_recommendations(self, concept: str, info: Dict) -> str:
        """Генерирует рекомендации по изучению."""
        return f"""Рекомендации по изучению '{concept}':

- Начать с базового определения
- Изучить историю и эволюцию понятия
- Рассмотреть различные подходы
- Найти практические примеры
- Сформулировать собственное понимание

Результат: углубленное понимание концепта для использования в ответах."""
    
    def _present_contradiction(self, concept: str, info: Dict) -> str:
        """Представляет противоречие для обсуждения."""
        facts = info.get('conflicting_facts', [])
        
        if len(facts) >= 2:
            fact_a = facts[0].get('value', 'Точка зрения A')
            fact_b = facts[1].get('value', 'Точка зрения B')
        else:
            fact_a = 'Первая точка зрения'
            fact_b = 'Противоположная точка зрения'
        
        return f"""Противоречие по концепту: {concept}

Точка зрения A: {fact_a}

Точка зрения B: {fact_b}

Задача: проанализировать обе позиции и найти разрешение."""
    
    def _analyze_contradiction_sides(self, concept: str, info: Dict) -> str:
        """Анализирует стороны противоречия."""
        return f"""Анализ противоречия по '{concept}':

Сильные стороны точки A:
- Логическая последовательность
- Эмпирическая поддержка
- Практическая применимость

Сильные стороны точки B:
- Альтернативный взгляд
- Учёт других факторов
- Критика ограничений A

Противоречие требует синтеза для полноты понимания."""
    
    def _synthesize_contradiction(self, concept: str, info: Dict) -> str:
        """Ищет синтез противоречия."""
        return f"""Поиск синтеза для '{concept}':

Возможные подходы:
1. Обе точки зрения частично верны в разных контекстах
2. Нужно уточнение условий применимости каждой
3. Существует третья, более полная точка зрения
4. Противоречие - следствие неполной информации

Синтез: рассматривать концепт в контексте, учитывая оба аспекта."""
    
    def _formulate_resolution(self, concept: str, info: Dict, 
                             turns: List[DialogTurn]) -> str:
        """Формулирует разрешение противоречия."""
        # Собираем анализ из предыдущих ходов
        analysis = "\n".join([t.content[:100] + "..." for t in turns[-2:]])
        
        resolution = f"""Разрешение противоречия по '{concept}':

На основе анализа:
{analysis}

Резолюция: 
Концепт '{concept}' имеет многоаспектную природу. 
Различные точки зрения отражают разные контексты применения.
Для полного понимания необходимо учитывать оба аспекта.

Итог: противоречие разрешено через контекстуализацию."""
        
        return resolution
    
    def _save_concept_dialog_results(self, dialog: SelfDialog, concept: str):
        """Сохраняет результаты диалога о концепте в кеш."""
        # Формируем сводку
        summary = {
            'type': 'concept_research',
            'concept': concept,
            'dialog_id': dialog.id,
            'turns': len(dialog.turns),
            'summary': f"Исследование концепта '{concept}' завершено",
            'timestamp': time.time(),
            'dialog_content': '\n\n'.join([f"{t.role.value}: {t.content[:200]}" for t in dialog.turns])
        }
        
        self._resolved_knowledge.append(summary)
        # Ограничиваем размер списка
        if len(self._resolved_knowledge) > self.MAX_RESOLVED_KNOWLEDGE:
            self._resolved_knowledge = self._resolved_knowledge[-self.MAX_RESOLVED_KNOWLEDGE:]
        
        self._save_to_context_cache(f"resolved_concept_{concept}", summary)
        
        logger.info(f"Результаты исследования '{concept}' сохранены")
    
    def _save_contradiction_resolution(self, contr_id: str, resolution: str,
                                      dialog: SelfDialog):
        """Сохраняет разрешение противоречия."""
        summary = {
            'type': 'contradiction_resolution',
            'contradiction_id': contr_id,
            'dialog_id': dialog.id,
            'resolution': resolution,
            'timestamp': time.time(),
            'dialog_summary': '\n'.join([f"{t.role.value}: {t.content[:150]}..." for t in dialog.turns])
        }
        
        self._resolved_knowledge.append(summary)
        # Ограничиваем размер списка
        if len(self._resolved_knowledge) > self.MAX_RESOLVED_KNOWLEDGE:
            self._resolved_knowledge = self._resolved_knowledge[-self.MAX_RESOLVED_KNOWLEDGE:]
        
        self._save_to_context_cache(f"resolved_contr_{contr_id}", summary)
        
        # Обновляем статус противоречия
        self._update_contradiction_status(contr_id, 'resolved', resolution)
        
        logger.info(f"Разрешение противоречия '{contr_id}' сохранено")
        
        # Сохраняем факты в FG
        self._save_learned_facts_to_fg(contr_id, resolution)
    
    def _save_learned_facts_to_fg(self, concept_or_id: str, learned_text: str):
        """
        Сохраняет полученные знания в FractalGraph v2.
        Создает узел типа 'fact' с результатами диалога.
        """
        if not self.brain:
            return
        
        try:
            fg = getattr(self.brain, 'fractal_graph_v2', None)
            if not fg or not hasattr(fg, 'add_node'):
                return
            
            # Создаем узел с результатами
            node = fg.add_node(
                content=learned_text[:500],  # Ограничиваем размер
                node_type='fact',
                level=2,  # Факты на уровне 2
                confidence=0.7,  # Диалог дает среднюю уверенность
                metadata={
                    'source': 'self_dialog',
                    'source_concept': concept_or_id,
                    'learned_at': time.time(),
                    'is_learned': True
                }
            )
            
            if node:
                logger.debug(f"Learned fact saved to FG: {node.id}")
                
        except Exception as e:
            logger.error(f"Error saving learned facts to FG: {e}")
    
    def _save_to_context_cache(self, key: str, data: Dict):
        """Сохраняет данные в кеш контекста."""
        if not self.brain or not hasattr(self.brain, 'hybrid_cache'):
            return
        
        try:
            cache = self.brain.hybrid_cache
            cache_key = f"self_dialog:{key}"
            cache.put(cache_key, data, ttl=86400 * 7)  # 7 дней
            logger.debug(f"Сохранено в кеш: {cache_key}")
        except Exception as e:
            logger.debug(f"Ошибка сохранения в кеш: {e}")
    
    def _update_contradiction_status(self, contr_id: str, status: str, 
                                    resolution: str):
        """Обновляет статус противоречия."""
        if not self.brain:
            return
        
        try:
            cm = getattr(self.brain, 'contradiction_manager', None)
            if cm and hasattr(cm, 'resolve_contradiction'):
                cm.resolve_contradiction(contr_id, {
                    'status': status,
                    'resolution_text': resolution[:500],
                    'resolved_by': 'self_dialog',
                    'timestamp': time.time()
                })
        except Exception as e:
            logger.debug(f"Ошибка обновления статуса противоречия: {e}")
    
    def get_resolved_knowledge(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Возвращает разрешённые знания из кеша."""
        resolved = []
        
        if not self.brain or not hasattr(self.brain, 'hybrid_cache'):
            return self._resolved_knowledge[-limit:]
        
        try:
            cache = self.brain.hybrid_cache
            # Используем get_recent вместо search_keys
            if hasattr(cache, 'get_recent'):
                recent = cache.get_recent(limit=limit * 2)
                for item in recent:
                    if isinstance(item, dict) and 'self_dialog' in str(item.get('key', '')):
                        resolved.append(item)
            elif hasattr(cache, 'disk_cache') and hasattr(cache.disk_cache, 'get_recent'):
                recent = cache.disk_cache.get_recent(limit=limit * 2)
                resolved.extend(recent)
            
        except Exception as e:
            logger.debug(f"Ошибка получения знаний из кеша: {e}")
        
        # Сортируем по времени
        resolved.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        return resolved[:limit]
    
    def extract_knowledge_from_cache(self, concept: str = None) -> List[Dict[str, Any]]:
        """
        Извлекает знания из кеша контекста.
        
        Args:
            concept: Опционально - фильтр по концепту
            
        Returns:
            Список извлечённых знаний
        """
        knowledge = []
        
        if not self.brain or not hasattr(self.brain, 'hybrid_cache'):
            return knowledge
        
        try:
            cache = self.brain.hybrid_cache
            
            # Используем get_recent вместо search_keys
            recent_items = []
            if hasattr(cache, 'get_recent'):
                recent_items = cache.get_recent(limit=50)
            elif hasattr(cache, 'disk_cache') and hasattr(cache.disk_cache, 'get_recent'):
                recent_items = cache.disk_cache.get_recent(limit=50)
            
            for item in recent_items:
                try:
                    data = item if isinstance(item, dict) else {}
                    if not data or not isinstance(data, dict):
                        continue
                    
                    # Фильтр по концепту
                    if concept:
                        data_concept = data.get('concept', '')
                        if concept.lower() not in data_concept.lower():
                            continue
                    
                    # Преобразуем в формат факта
                    fact = self._convert_to_fact(data)
                    if fact:
                        knowledge.append(fact)
                    
                except Exception as e:
                    logger.debug(f"Ошибка обработки кеша: {e}")
                    continue
            
            logger.info(f"Извлечено {len(knowledge)} знаний из кеша" + 
                       (f" для '{concept}'" if concept else ""))
            
        except Exception as e:
            logger.error(f"Ошибка извлечения знаний: {e}")
        
        return knowledge
    
    def _convert_to_fact(self, data: Dict) -> Optional[Dict[str, Any]]:
        """Преобразует данные из кеша в формат факта."""
        data_type = data.get('type', '')
        
        if data_type == 'concept_research':
            return {
                'relation_type': 'is_a',
                'value': data.get('summary', ''),
                'source': 'self_dialog',
                'confidence': 0.7,
                'concept': data.get('concept', ''),
                'context': data.get('dialog_content', '')[:500]
            }
        
        elif data_type == 'contradiction_resolution':
            return {
                'relation_type': 'resolved_contradiction',
                'value': data.get('resolution', ''),
                'source': 'self_dialog',
                'confidence': 0.8,
                'contradiction_id': data.get('contradiction_id', ''),
                'context': data.get('dialog_summary', '')[:500]
            }
        
        return None

    def get_context_for_generation(self, query: str) -> str:
        """
        Получает контекст из концептов и противоречий для генерации ответа.
        Вызывается в цикле генерации.
        
        Args:
            query: Запрос пользователя
            
        Returns:
            Строка с контекстом для добавления в промпт
        """
        context_parts = []
        
        try:
            # 1. Получаем концепты из запроса
            if self.brain and hasattr(self.brain, 'concept_extractor'):
                concepts_context = self.brain.concept_extractor.get_concepts_for_prompt(query)
                if concepts_context:
                    context_parts.append(concepts_context)
            
            # 2. Проверяем противоречия для извлечённых концептов
            if self.brain and hasattr(self.brain, 'contradiction_generator'):
                # Извлекаем термины из запроса
                import re
                terms = re.findall(r'\b[а-яёa-z]{4,}\b', query.lower())
                
                for term in terms[:3]:  # Проверяем топ-3 термина
                    contr_context = self.brain.contradiction_generator.get_contradictions_for_prompt(term)
                    if contr_context:
                        context_parts.append(contr_context)
                        break  # Достаточно одного противоречия
            
            # 3. Добавляем разрешённые знания из кеша
            resolved = self.extract_knowledge_from_cache()
            if resolved:
                context_parts.append("Ранее разрешённые знания:")
                for r in resolved[:3]:
                    fact = self._convert_to_fact(r)
                    if fact:
                        context_parts.append(f"  - {fact['value'][:150]}")
            
            if context_parts:
                return "\n\n".join(["[Контекст из базы знаний]:", *context_parts, ""])
            
        except Exception as e:
            logger.debug(f"Ошибка получения контекста: {e}")
        
        return ""
