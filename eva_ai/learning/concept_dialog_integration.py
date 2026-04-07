"""
Интеграция SelfDialogLearning и ConceptMiner
Связывает методы самодиалога с поиском концептов и семантической конвергенцией
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("eva_ai.learning.concept_dialog_integration")


class ConceptDialogIntegrator:
    """
    Интегратор самодиалога и поиска концептов.
    
    Связи:
    1. Knowledge gaps (SDL) → Trigger Concept Mining
    2. Concept candidates → Verify via Self-Dialog
    3. Semantic convergence → Topic selection for dialogs
    4. Dialog results → Feed back to ConceptMiner
    """
    
    def __init__(
        self,
        self_dialog_learning=None,
        concept_miner=None,
        brain=None,
        config: Dict = None
    ):
        self.sdl = self_dialog_learning
        self.concept_miner = concept_miner
        self.brain = brain
        self.config = config or {}
        
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ConceptDialog")
        
        self._last_concept_check = 0
        self._concept_check_interval = self.config.get("concept_check_interval", 300)
        
        self._linked_topics: Dict[str, str] = {}
        
        self._metrics = {
            "concepts_to_dialogs": 0,
            "dialogs_to_concepts": 0,
            "semantic_convergence_dialogs": 0,
            "knowledge_gaps_mined": 0
        }
        
        logger.info("ConceptDialogIntegrator инициализирован")
    
    def start(self):
        """Запуск интегратора"""
        if self._running:
            return
        
        self._running = True
        
        if self.sdl and hasattr(self.sdl, '_setup_curator_events'):
            original_setup = self.sdl._setup_curator_events
            
            def extended_setup():
                original_setup()
                self._subscribe_to_concept_events()
            
            if hasattr(self.sdl, '_setup_curator_events'):
                self.sdl._setup_curator_events = extended_setup
        
        self._subscribe_to_sdl_events()
        
        logger.info("ConceptDialogIntegrator запущен")
    
    def stop(self):
        """Остановка интегратора"""
        self._running = False
        self._executor.shutdown(wait=False)
        logger.info("ConceptDialogIntegrator остановлен")
    
    def _subscribe_to_concept_events(self):
        """Подписка на события ConceptMiner"""
        if not self.concept_miner or not self.concept_miner.event_bus:
            return
        
        try:
            from eva_ai.core.event_bus import EventTypes
            
            self.concept_miner.event_bus.subscribe(
                EventTypes.CONCEPT_CANDIDATE_GENERATED,
                self._on_concept_candidate_generated,
                priority=5
            )
            
            self.concept_miner.event_bus.subscribe(
                EventTypes.CONCEPT_VALIDATION_COMPLETE,
                self._on_concept_validation_complete,
                priority=5
            )
            
            logger.debug("Подписка на события ConceptMiner установлена")
        except Exception as e:
            logger.warning(f"Ошибка подписки на ConceptMiner: {e}")
    
    def _subscribe_to_sdl_events(self):
        """Подписка на события SelfDialogLearning"""
        if not self.sdl or not hasattr(self.sdl, 'learning_callbacks'):
            return
        
        self.sdl.learning_callbacks.append(self._on_learning_opportunity)
    
    def _on_concept_candidate_generated(self, event):
        """При генерации кандидата - создание верификационного диалога"""
        data = event.data if hasattr(event, 'data') else {}
        candidate_id = data.get('candidate_id')
        title = data.get('title', '')
        
        if not candidate_id or not self.sdl or not self._running:
            return
        
        self._executor.submit(self._verify_concept_via_dialog, candidate_id, title)
    
    def _verify_concept_via_dialog(self, candidate_id: str, title: str):
        """Верификация концепта через самодиалог"""
        try:
            topic = f"Концепт: {title}"
            
            context = {
                "source": "concept_miner",
                "candidate_id": candidate_id,
                "type": "concept_verification"
            }
            
            if hasattr(self.sdl, 'create_dialog'):
                dialog = self.sdl.create_dialog(topic=topic, context=context)
                
                if dialog:
                    self._linked_topics[candidate_id] = topic
                    self._metrics["concepts_to_dialogs"] += 1
                    logger.info(f"Создан диалог верификации для концепта: {title}")
            
        except Exception as e:
            logger.warning(f"Ошибка верификации концепта через диалог: {e}")
    
    def _on_concept_validation_complete(self, event):
        """При завершении валидации концепта"""
        data = event.data if hasattr(event, 'data') else {}
        candidate_id = data.get('candidate_id')
        status = data.get('status')
        
        if candidate_id in self._linked_topics:
            topic = self._linked_topics[candidate_id]
            
            if status == "confirmed":
                self._trigger_convergence_dialog(topic)
            else:
                del self._linked_topics[candidate_id]
    
    def _trigger_convergence_dialog(self, topic: str):
        """Триггер диалога семантической конвергенции после подтверждения концепта"""
        try:
            context = {
                "source": "concept_miner_confirmed",
                "type": "semantic_convergence",
                "topic": topic
            }
            
            if hasattr(self.sdl, 'create_dialog'):
                self.sdl.create_dialog(topic=topic, context=context)
                self._metrics["semantic_convergence_dialogs"] += 1
                logger.info(f"Создан диалог конвергенции для: {topic}")
        except Exception as e:
            logger.warning(f"Ошибка создания диалога конвергенции: {e}")
    
    def _on_learning_opportunity(self, opportunity: Dict):
        """Обработка возможности обучения - запуск майнинга концептов"""
        if not self.concept_miner or not self._running:
            return
        
        opportunity_type = opportunity.get('type', '')
        priority = opportunity.get('priority', 0)
        
        if opportunity_type in ['expansion', 'refinement'] and priority > 0.3:
            current_time = time.time()
            if current_time - self._last_concept_check < self._concept_check_interval:
                return
            
            self._last_concept_check = current_time
            
            self._executor.submit(self._trigger_concept_mining_from_gap, opportunity)
    
    def _trigger_concept_mining_from_gap(self, opportunity: Dict):
        """Триггер майнинга концептов из пробела знаний"""
        try:
            concept = opportunity.get('concept', '')
            domain = opportunity.get('domain', '')
            
            logger.info(f"Триггер майнинга из пробела: {concept} ({domain})")
            
            if hasattr(self.concept_miner, 'force_mining_cycle'):
                self.concept_miner.force_mining_cycle()
                self._metrics["knowledge_gaps_mined"] += 1
            
        except Exception as e:
            logger.warning(f"Ошибка триггера майнинга: {e}")
    
    def link_semantic_convergence_to_topics(self, clusters: Dict) -> List[str]:
        """
        Связь семантической конвергенции с выбором тем для диалогов
        
        Использует данные кластеров для выбора тем, которые:
        - Имеют высокую семантическую связность
        - Содержат мало связанных концептов
        - Требуют укрупнения связей
        """
        topics = []
        
        try:
            if not clusters:
                return topics
            
            cluster_items = list(clusters.items())
            
            cluster_items.sort(key=lambda x: len(x[1]), reverse=True)
            
            for cluster_id, node_ids in cluster_items[:5]:
                if len(node_ids) >= 3:
                    topic = f"Кластер {cluster_id}: {len(node_ids)} элементов"
                    topics.append(topic)
            
            topics.extend(self._get_orphan_topic_suggestions(clusters))
            
        except Exception as e:
            logger.warning(f"Ошибка связи конвергенции с темами: {e}")
        
        return topics
    
    def _get_orphan_topic_suggestions(self, clusters: Dict) -> List[str]:
        """Получение предложений тем для сиротских узлов"""
        suggestions = []
        
        try:
            if not self.brain:
                return suggestions
            
            if hasattr(self.brain, 'memory_manager'):
                mm = self.brain.memory_manager
                
                if hasattr(mm, 'get_orphans'):
                    orphans = mm.get_orphans()
                    
                    for orphan in orphans[:3]:
                        content = getattr(orphan, 'content', '')[:50]
                        if content:
                            suggestions.append(f"Сиротский узел: {content}...")
        
        except Exception:
            pass
        
        return suggestions
    
    def get_concept_dialog_pairs(self) -> List[Dict]:
        """Получение связей концепт-диалог"""
        return [
            {"candidate_id": k, "dialog_topic": v}
            for k, v in self._linked_topics.items()
        ]
    
    def get_metrics(self) -> Dict:
        """Получение метрик интеграции"""
        return {
            **self._metrics,
            "active_links": len(self._linked_topics),
            "status": "running" if self._running else "stopped"
        }
    
    def force_concept_dialog_cycle(self):
        """Принудительный цикл концепт-диалог"""
        if self.concept_miner and hasattr(self.concept_miner, 'force_mining_cycle'):
            self.concept_miner.force_mining_cycle()


def create_concept_dialog_integrator(
    self_dialog_learning=None,
    concept_miner=None,
    brain=None,
    config: Dict = None
) -> ConceptDialogIntegrator:
    """Фабрика создания интегратора"""
    return ConceptDialogIntegrator(
        self_dialog_learning=self_dialog_learning,
        concept_miner=concept_miner,
        brain=brain,
        config=config
    )


__all__ = [
    'ConceptDialogIntegrator',
    'create_concept_dialog_integrator'
]
