import logging
from typing import List, Dict, Any, Optional
try:
    from ..core.base_component import BaseComponent
except ImportError:
    class BaseComponent:
        def __init__(self, brain=None):
            self.brain = brain
        
        def _setup_component(self):
            pass
        
        def initialize(self):
            pass
        
        def start(self):
            pass
        
        def stop(self):
            pass
from .contradiction_core import OptimizedContradictionDetector

logger = logging.getLogger(__name__)

class ContradictionManager(BaseComponent):
    """Менеджер противоречий - отвечает за обнаружение и разрешение противоречий в знаниях системы"""
    
    def __init__(self, brain=None, cache_dir="./cache"):
        """
        Инициализация менеджера противоречий
        
        Args:
            brain: Основной объект мозга системы
            cache_dir: Директория для кэширования
        """
        super().__init__(brain)
        self.cache_dir = cache_dir
        self.contradictions = []
        self.known_concepts = set()
        self.detector = None
        self._initialize_components()
    
    def _setup_component(self) -> None:
        """Настраивает компонент после проверки зависимостей."""
        self._initialize_components()

    def _initialize_components(self):
        """Инициализация базовых компонентов"""
        if self.detector is not None:
            logger.debug("Детектор уже инициализирован")
            return
        try:
            self.detector = OptimizedContradictionDetector(
                knowledge_graph=self.brain.knowledge_graph if hasattr(self.brain, 'knowledge_graph') else None,
                brain=self.brain,
                cache_dir=self.cache_dir
            )
            logger.info("Детектор противоречий успешно инициализирован")
        except Exception as e:
            logger.error(f"Ошибка при инициализации детектора противоречий: {e}")
            self.detector = None

    def get_known_concepts(self) -> List[str]:
        """Возвращает список известных концепций"""
        return list(self.known_concepts)

    def add_contradiction(self, contradiction: Dict[str, Any]):
        """
        Добавляет новое противоречие в список
        
        Args:
            contradiction: Словарь с данными о противоречии
        """
        if 'contradiction_id' in contradiction and 'id' not in contradiction:
            contradiction['id'] = contradiction['contradiction_id']
        contradiction_id = contradiction.get('id')
        existing_ids = {c.get('id') for c in self.contradictions}
        if contradiction_id and contradiction_id not in existing_ids:
            self.contradictions.append(contradiction)
        elif not contradiction_id:
            self.contradictions.append(contradiction)
        # Добавляем концепции в known_concepts
        if 'concepts' in contradiction:
            self.known_concepts.update(contradiction['concepts'])

    def get_contradictions(self) -> List[Dict[str, Any]]:
        """Возвращает список всех обнаруженных противоречий"""
        return self.contradictions

    def detect_contradictions(self, text: Optional[str] = None) -> Dict[str, Any]:
        """
        Запускает поиск противоречий в тексте или в имеющихся знаниях

        Args:
            text: Опциональный текст для анализа

        Returns:
            Dict с ключом 'contradictions' содержащий список обнаруженных противоречий
        """
        if self.detector is None:
            logger.warning("Детектор противоречий не инициализирован")
            return {'contradictions': []}

        logger.debug("Начинаем поиск противоречий...")

        try:
            if text:
                result = self.detector.detect_contradiction("text_analysis", [{'text': text}])
                new_contradictions = [result] if result else []
            else:
                new_contradictions = self.detector.get_active_contradictions()

            formatted_contradictions = []
            for contradiction in new_contradictions:
                if isinstance(contradiction, dict):
                    formatted = contradiction
                else:
                    formatted = {
                        'id': getattr(contradiction, 'id', str(id(contradiction))),
                        'concept': getattr(contradiction, 'concept', 'unknown'),
                        'conflicting_facts': getattr(contradiction, 'conflicting_facts', []),
                        'divergence_level': getattr(contradiction, 'divergence_level', 0.0),
                        'status': getattr(contradiction, 'status', 'detected'),
                        'metadata': getattr(contradiction, 'metadata', {})
                    }

                self.add_contradiction(formatted)
                formatted_contradictions.append(formatted)

            return {'contradictions': formatted_contradictions}

        except Exception as e:
            logger.error(f"Ошибка при поиске противоречий: {e}")
            return {'contradictions': []}

    def resolve_contradiction(self, contradiction_id: str, resolution: Optional[Dict[str, Any]] = None) -> bool:
        """
        Пытается разрешить конкретное противоречие
        
        Args:
            contradiction_id: Идентификатор противоречия
            resolution: Дополнительные данные для разрешения противоречия
            
        Returns:
            True если противоречие успешно разрешено
        """
        if self.detector is None:
            logger.warning("Детектор противоречий не инициализирован")
            return False
            
        try:
            # Находим противоречие по ID
            if not isinstance(self.contradictions, list):
                logger.error("Список противоречий имеет неверный тип")
                return False
                
            contradiction = next((c for c in self.contradictions 
                               if isinstance(c, dict) and (c.get('id') == contradiction_id or c.get('contradiction_id') == contradiction_id)), None)
            if not contradiction:
                logger.warning(f"Противоречие с ID {contradiction_id} не найдено")
                return False
            
            # Подготавливаем данные для разрешения
            resolution = resolution or {}
            if 'resolver' not in resolution:
                resolution['resolver'] = 'system'
            if 'confidence' not in resolution:
                resolution['confidence'] = 1.0
                
            # Пытаемся разрешить противоречие
            if self.detector.resolve_contradiction(
                contradiction_id=contradiction_id,
                resolution=resolution,
                resolver=resolution['resolver'],
                confidence=resolution['confidence']
            ):
                # Обновляем статус противоречия
                contradiction['status'] = 'resolved'
                if resolution:
                    contradiction['resolution'] = resolution
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Ошибка при разрешении противоречия: {e}", exc_info=True)
            return False

    def get_contradiction_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику по противоречиям
        
        Returns:
            Словарь со статистикой
        """
        return {
            'total': len(self.contradictions),
            'resolved': sum(1 for c in self.contradictions if c.get('status') == 'resolved'),
            'unresolved': sum(1 for c in self.contradictions if c.get('status') != 'resolved'),
            'by_type': self._get_contradictions_by_type(),
        }
    
    def _get_contradictions_by_type(self) -> Dict[str, int]:
        """
        Группирует противоречия по типам
        
        Returns:
            Словарь с количеством противоречий каждого типа
        """
        type_counts = {}
        for c in self.contradictions:
            c_type = c.get('type', 'unknown')
            type_counts[c_type] = type_counts.get(c_type, 0) + 1
        return type_counts

    def check_with_context(
        self,
        text: str,
        query: str = "",
        conversation_history: Optional[List[Dict]] = None,
        knowledge_context: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Проверяет текст на противоречия с учётом контекста
        
        Args:
            text: Текст для проверки
            query: Оригинальный запрос
            conversation_history: История диалогов
            knowledge_context: Контекст из knowledge graph
            
        Returns:
            Dict с результатами проверки и весами противоречий
        """
        logger.info("Проверка противоречий с контекстом...")
        
        # Базовая проверка
        base_result = self.detect_contradictions(text)
        contradictions = base_result.get('contradictions', [])
        
        # Рассчитываем веса противоречий
        weighted_contradictions = []
        for contr in contradictions:
            weight = self._calculate_contradiction_weight(contr, query, knowledge_context)
            contr_with_weight = contr.copy()
            contr_with_weight['weight'] = weight
            contr_with_weight['is_significant'] = weight >= 0.3
            weighted_contradictions.append(contr_with_weight)
        
        # Отделяем значимые противоречия
        significant = [c for c in weighted_contradictions if c.get('is_significant', False)]
        minor = [c for c in weighted_contradictions if not c.get('is_significant', True)]
        
        return {
            'contradictions': weighted_contradictions,
            'significant_contradictions': significant,
            'minor_contradictions': minor,
            'total_count': len(weighted_contradictions),
            'significant_count': len(significant),
            'has_conflicts': len(significant) > 0
        }
    
    def _calculate_contradiction_weight(
        self,
        contradiction: Dict[str, Any],
        query: str,
        knowledge_context: Optional[List[str]] = None
    ) -> float:
        """
        Рассчитывает вес противоречия (0.0-1.0)
        
        Чем выше вес - тем важнее противоречие для регенерации
        """
        weight = 0.5  # Базовый вес
        
        # Увеличиваем вес если противоречие связано с запросом
        if query:
            concept = contradiction.get('concept', '').lower()
            query_lower = query.lower()
            if concept and concept in query_lower:
                weight += 0.3
        
        # Увеличиваем вес если есть противоречие с известными фактами
        divergence = contradiction.get('divergence_level', 0.0)
        if isinstance(divergence, (int, float)):
            weight += divergence * 0.2
        
        # Проверяем контекст знаний
        if knowledge_context:
            conflicting_facts = contradiction.get('conflicting_facts', [])
            for fact in conflicting_facts:
                fact_str = str(fact).lower()
                for ctx in knowledge_context:
                    if any(w in ctx.lower() for w in ['не', 'нет', 'нельзя']) and any(
                        w in fact_str for w in ctx.lower().split()
                    ):
                        weight += 0.1
        
        return min(1.0, max(0.0, weight))
    
    def generate_refinement_prompt(
        self,
        contradiction_result: Dict[str, Any],
        query: str = "",
        response: str = ""
    ) -> str:
        """
        Генерирует промпт для регенерации на основе противоречий
        
        Args:
            contradiction_result: Результат check_with_context()
            query: Оригинальный запрос
            response: Текущий ответ
            
        Returns:
            str: Промпт для Qwen
        """
        significant = contradiction_result.get('significant_contradictions', [])
        
        if not significant:
            return ""  # Нет значимых противоречий
        
        parts = []
        
        for i, contr in enumerate(significant[:3], 1):
            concept = contr.get('concept', 'unknown')
            facts = contr.get('conflicting_facts', [])
            
            parts.append(f"{i}. Противоречие в концепции '{concept}':")
            
            if facts:
                for fact in facts[:2]:
                    if isinstance(fact, dict):
                        parts.append(f"   - Факт: {fact.get('fact', str(fact))}")
                    else:
                        parts.append(f"   - Факт: {fact}")
            
            parts.append(f"   Важность: {contr.get('weight', 0.5):.2f}")
        
        prompt = """Обнаружены противоречия в ответе:
"
        prompt += "\n".join(parts)
        prompt += "\n\nПереформулируй ответ, устранив противоречия. Сохрани основную информацию."""

        return prompt
