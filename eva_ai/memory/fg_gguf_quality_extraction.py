"""
Quality-controlled GGUF to FractalGraph extraction.
Контроль качества при извлечении знаний из GGUF модели.
"""
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("eva_ai.fg_extraction_quality")

@dataclass
class ExtractionQuality:
    """Результат проверки качества извлечения."""
    is_valid: bool
    confidence: float
    issues: List[str] = field(default_factory=list)
    cleaned_content: str = ""

class KnowledgeQualityFilter:
    """
    Фильтр качества для извлекаемых из GGUF знаний.
    
    Уровень детализации: фразы (2-4 ключевых слова)
    """
    
    # Мусорные паттерны (Generic responses)
    GARBAGE_PATTERNS = [
        r'продолжим разговор',
        r'перспективы развития',
        r'давайте обсудим',
        r'интересный вопрос',
        r'как я уже упоминал',
        r'в контексте нашего',
        r'基于.*内容',  # китайский мусор
        r'以下.*回答',  # китайский мусор
        r'###',
        r'^q:',
        r'^a:',
        r'^пример:',
        r'^особенности',
    ]
    
    # Минимальная длина (символы)
    MIN_LENGTH = 3
    MAX_LENGTH = 100
    
    # Минимальное количество слов для осмысленной фразы
    MIN_WORDS = 2
    MAX_WORDS = 6
    
    # Паттерны бессвязного текста
    NONSENSE_PATTERNS = [
        r'^(\w)\1{3,}',  # повторяющийся символ >3 раз
        r'^[а-яёa-z]{1,2}\s[а-яёa-z]{1,2}\s[а-яёa-z]{1,2}\s[а-яёa-z]{1,2}\s[а-яёa-z]{1,2}$',  # все короткие слова
        r'[a-zA-Z]{30,}',  # слишком длинное английское слово
    ]
    
    # Обязательные части речи (должно быть существительное или глагол)
    REQUIRED_POS = ['NOUN', 'VERB', 'ADJS', 'ADVB']
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.min_words = self.config.get('min_words', self.MIN_WORDS)
        self.max_words = self.config.get('max_words', self.MAX_WORDS)
        
    def validate(self, text: str, model_confidence: float = 0.8) -> ExtractionQuality:
        """
        Проверить качество извлечённого текста.
        
        Args:
            text: Текст для проверки
            model_confidence: Confidence от модели (0-1)
            
        Returns:
            ExtractionQuality с результатом проверки
        """
        issues = []
        cleaned = text.strip()
        
        # 1. Базовая проверка длины
        if len(cleaned) < self.MIN_LENGTH:
            issues.append("Слишком короткий текст")
            return ExtractionQuality(False, 0.0, issues, "")
            
        if len(cleaned) > self.MAX_LENGTH:
            issues.append("Слишком длинный текст")
            cleaned = cleaned[:self.MAX_LENGTH]
        
        # 2. Проверка на мусорные паттерны
        for pattern in self.GARBAGE_PATTERNS:
            if re.search(pattern, cleaned.lower()):
                issues.append(f"Мусорный паттерн: {pattern}")
                return ExtractionQuality(False, 0.0, issues, "")
        
        # 3. Проверка на бессмыслицу
        for pattern in self.NONSENSE_PATTERNS:
            if re.search(pattern, cleaned):
                issues.append("Бессвязный текст")
                return ExtractionQuality(False, 0.0, issues, "")
        
        # 4. Проверка количества слов
        words = cleaned.split()
        word_count = len(words)
        
        if word_count < self.min_words:
            issues.append(f"Слишком мало слов: {word_count}")
            return ExtractionQuality(False, 0.0, issues, "")
            
        if word_count > self.max_words:
            # Обрезаем до max_words
            words = words[:self.max_words]
            cleaned = ' '.join(words)
        
        # 5. Проверка на наличие ключевых слов (буквы)
        letter_count = sum(1 for c in cleaned if c.isalpha())
        if letter_count < word_count * 0.5:
            issues.append("Мало букв относительно слов")
            return ExtractionQuality(False, 0.0, issues, "")
        
        # 6. Проверка confidence модели
        if model_confidence < self.min_confidence:
            issues.append(f"Низкий confidence модели: {model_confidence:.2f}")
            # Не отклоняем, но снижаем общий score
        
        # 7. Проверка что текст начинается с буквы (не спецсимвол)
        if cleaned and not cleaned[0].isalnum():
            # Убираем начальные спецсимволы
            cleaned = re.sub(r'^[\s\-\.\,\:\;]+', '', cleaned)
            if not cleaned:
                issues.append("Только спецсимволы")
                return ExtractionQuality(False, 0.0, issues, "")
        
        # Рассчитываем итоговый confidence
        final_confidence = model_confidence
        if issues:
            final_confidence *= 0.5  # Штраф за issues
        
        # Нормализуем
        final_confidence = min(1.0, max(0.0, final_confidence))
        
        return ExtractionQuality(
            is_valid=True,
            confidence=final_confidence,
            issues=issues,
            cleaned_content=cleaned
        )
    
    def extract_phrases(self, model_response: str, min_phrases: int = 1, 
                       max_phrases: int = 5) -> List[str]:
        """
        Извлечь осмысленные фразы из ответа модели.
        
        Args:
            model_response: Ответ модели
            min_phrases: Минимум фраз для извлечения
            max_phrases: Максимум фраз
            
        Returns:
            List валидных фраз
        """
        valid_phrases = []
        
        # Разбиваем по предложениям и пунктуации
        # Но не слишком дробно - сохраняем фразы
        segments = re.split(r'[\.\!\?\n]+', model_response)
        
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
                
            # Проверяем каждую фразу
            quality = self.validate(segment)
            
            if quality.is_valid and quality.confidence >= self.min_confidence:
                valid_phrases.append(quality.cleaned_content)
                
                if len(valid_phrases) >= max_phrases:
                    break
        
        return valid_phrases


class GGUFToFGIntegrator:
    """
    Интегратор GGUF -> FractalGraph с контролем качества.
    
    Поток:
    1. Загрузить GGUF модель (LlamaCpp)
    2. Извлечь vocab (через GGUFModelParser) - сохранить в FG как метаданные
    3. Для каждого запроса:
       a. Контекст из FG
       b. Генерация через GGUF (родной токенизатор)
       c. Контроль качества ответа
       d. Извлечение фраз (2-4 слова) с валидацией
       e. Сохранение в FG через стандартные методы FG (add_node, add_knowledge)
    """
    
    def __init__(self, brain=None, fractal_graph=None):
        self.brain = brain
        self.fg = fractal_graph
        self.quality_filter = KnowledgeQualityFilter()
        
        # GGUF модель
        self.llama_cpp = getattr(brain, 'llama_cpp_deployment', None)
        
        # Токенизатор модели (родной)
        self.model_tokenizer = None
        self._load_model_tokenizer()
        
    def _load_model_tokenizer(self):
        """Загрузить родной токенизатор модели."""
        # Попробовать получить из llama_cpp
        if self.llama_cpp and hasattr(self.llama_cpp, 'tokenizer'):
            self.model_tokenizer = self.llama_cpp.tokenizer
            return
            
        # Попробовать из конфига
        try:
            import json
            config_path = os.path.join(
                os.path.dirname(__file__), 
                "..", "mlearning", "eva_models", "qwen3.5-0.8b", 
                "tokenizer_config.json"
            )
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.tokenizer_config = json.load(f)
                logger.info("Tokenizer config loaded")
        except Exception as e:
            logger.debug(f"Tokenizer config load error: {e}")
    
    def generate_with_extraction(self, query: str, context: str = "",
                                  max_new_tokens: int = 512) -> Dict[str, Any]:
        """
        Генерация + извлечение знаний в FG.
        
        Returns:
            {
                'response': str,
                'extracted_phrases': List[str],
                'quality': float
            }
        """
        # 1. Формируем промт
        prompt = self._build_prompt(query, context)
        
        # 2. Генерируем через GGUF (родной токенизатор)
        response = ""
        if self.llama_cpp:
            try:
                response = self.llama_cpp.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7
                )
            except Exception as e:
                logger.error(f"GGUF generation error: {e}")
                return {'response': '', 'extracted_phrases': [], 'quality': 0}
        
        if not response:
            return {'response': '', 'extracted_phrases': [], 'quality': 0}
        
        # 3. Контроль качества ответа
        response_quality = self.quality_filter.validate(response)
        
        # 4. Извлекаем фразы из ответа
        phrases = []
        if response_quality.is_valid:
            phrases = self.quality_filter.extract_phrases(response)
        
        # 5. Сохраняем в FG если есть валидные фразы
        saved_count = 0
        if phrases and self.fg:
            for phrase in phrases:
                try:
                    self.fg.add_node(
                        content=phrase,
                        node_type='extracted_knowledge',
                        level=2,
                        metadata={
                            'source': 'gguf_generation',
                            'query': query[:100],
                            'confidence': response_quality.confidence,
                            'extraction_method': 'phrase_extraction'
                        }
                    )
                    saved_count += 1
                except Exception as e:
                    logger.debug(f"Save phrase error: {e}")
        
        return {
            'response': response,
            'extracted_phrases': phrases,
            'saved_count': saved_count,
            'quality': response_quality.confidence
        }
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Сформировать промт с контекстом из FG."""
        prompt = query
        if context:
            prompt = f"Контекст: {context}\n\nВопрос: {query}\nОтвет:"
        return prompt
    
    def load_model_metadata_to_fg(self, model_path: str) -> bool:
        """
        Загрузить metadata GGUF модели в FG.
        
        Парсит GGUF, извлекает vocab, конфиг, сохраняет как узлы.
        """
        if not self.fg:
            return False
            
        try:
            from eva_ai.memory.fractal_graph_v2.gguf_parser import GGUFModelParser
            
            parser = GGUFModelParser()
            model_info = parser.parse(model_path)
            
            # Сохраняем конфиг модели как специальный узел
            config_json = json.dumps(model_info.__dict__, ensure_ascii=False)
            
            self.fg.add_node(
                content=f"Model: {model_info.model_type}, Vocab: {model_info.vocab_size}, Layers: {model_info.num_layers}",
                node_type='model_metadata',
                level=0,  # Самый верхний уровень - системный
                metadata={
                    'source': 'gguf_model',
                    'model_path': model_path,
                    'architecture': model_info.architecture,
                    'vocab_size': model_info.vocab_size,
                    'hidden_size': model_info.hidden_size,
                    'num_layers': model_info.num_layers,
                    'config_json': config_json,
                    'is_static': True  # Не удалять
                }
            )
            
            logger.info(f"Model metadata saved to FG: {model_info.model_type}")
            return True
            
        except Exception as e:
            logger.error(f"Model metadata load error: {e}")
            return False


def create_gguf_fg_integrator(brain, fractal_graph) -> GGUFToFGIntegrator:
    """Создать интегратор."""
    return GGUFToFGIntegrator(brain, fractal_graph)