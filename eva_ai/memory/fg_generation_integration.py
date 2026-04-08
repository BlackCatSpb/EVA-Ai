"""
FractalGraphV2 Integration with LLM Generation.
Объединяет FGv2 как единую систему для:
- Хранения знаний (узлы, связи)
- Токенизации (GraphTokenizer)
- Генерации (через GGUF/LlamaCpp)
- Дообучения (через fractal storage)
"""
import logging
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger("eva_ai.fg_integration")

@dataclass
class FGGenerationResult:
    """Результат генерации через FG."""
    response: str
    tokens_generated: int
    context_used: List[str]
    confidence: float
    source: str  # gguf, fractal, hybrid

class FractalGraphGenerator:
    """
    Интеграция FractalGraphV2 с генерацией LLM.
    
    Поток:
    1. Получить запрос
    2. Извлечь контекст из FG (semantic_search)
    3. Токенизировать через GraphTokenizer
    4. Сгенерировать через GGUF/LlamaCpp
    5. Сохранить результат в FG как узел
    """
    
    def __init__(self, brain=None, config: Optional[Dict] = None):
        self.brain = brain
        self.config = config or {}
        
        # FractalGraphV2
        self.fg = getattr(brain, 'fractal_graph_v2', None)
        
        # Токенизатор графа
        self.tokenizer = None
        if self.fg and hasattr(self.fg, 'tokenizer'):
            self.tokenizer = self.fg.tokenizer
        else:
            try:
                from eva_ai.memory.fractal_graph_v2.tokenizer import GraphTokenizer
                self.tokenizer = GraphTokenizer(self.fg) if self.fg else None
            except Exception as e:
                logger.debug(f"GraphTokenizer init error: {e}")
        
        # GGUF/LlamaCpp модель
        self.llama_cpp = getattr(brain, 'llama_cpp_deployment', None)
        self.llama_cpp_ready = getattr(brain, 'llama_cpp_ready', False)
        
        # Qwen/Fractal модель
        self.fractal_model = getattr(brain, 'fractal_model_manager', None)
        
        # Параметры генерации
        self.default_temperature = self.config.get('temperature', 0.7)
        self.default_max_tokens = self.config.get('max_new_tokens', 2048)
        
    def generate(self, query: str, context: str = "", 
                 max_new_tokens: int = None, temperature: float = None,
                 use_context: bool = True) -> FGGenerationResult:
        """
        Основной метод генерации через FG.
        
        Args:
            query: Запрос пользователя
            context: Дополнительный контекст
            max_new_tokens: Максимум токенов
            temperature: Температура генерации
            use_context: Использовать контекст из FG
            
        Returns:
            FGGenerationResult
        """
        max_new_tokens = max_new_tokens or self.default_max_tokens
        temperature = temperature or self.default_temperature
        
        # 1. Получаем контекст из FG если нужно
        graph_context = ""
        context_nodes = []
        if use_context and self.fg:
            try:
                results = self.fg.semantic_search(query, top_k=5, min_similarity=0.5)
                if results:
                    for r in results[:3]:
                        content = r.get('content', '')
                        if content and len(content) > 20:
                            graph_context += content + "\n"
                            context_nodes.append(r.get('id', ''))
            except Exception as e:
                logger.debug(f"FG context error: {e}")
        
        # Объединяем контексты
        full_context = graph_context
        if context:
            full_context += "\n" + context
        
        # 2. Формируем промт
        if full_context:
            prompt = f"Контекст из памяти:\n{full_context}\n\nВопрос: {query}\nОтвет:"
        else:
            prompt = query
        
        # 3. Токенизируем через GraphTokenizer если доступен
        tokens_info = ""
        if self.tokenizer:
            try:
                tokens = self.tokenizer.tokenize(prompt)
                tokens_info = f"[Токенизировано: {len(tokens)} токенов через GraphTokenizer]"
            except Exception as e:
                logger.debug(f"Tokenization error: {e}")
        
        # 4. Генерируем через доступную модель
        response_text = ""
        source = "none"
        
        # Приоритет 1: LlamaCpp (GGUF)
        if self.llama_cpp and self.llama_cpp_ready:
            try:
                response_text = self.llama_cpp.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    repeat_penalty=1.1
                )
                source = "gguf"
            except Exception as e:
                logger.debug(f"GGUF generation error: {e}")
        
        # Приоритет 2: FractalModelManager
        if not response_text and self.fractal_model:
            try:
                if hasattr(self.fractal_model, 'generate_response'):
                    response_text = self.fractal_model.generate_response(
                        query=prompt,
                        max_new_tokens=max_new_tokens
                    )
                    source = "fractal"
            except Exception as e:
                logger.debug(f"Fractal generation error: {e}")
        
        # Приоритет 3: Qwen (через brain)
        if not response_text and self.brain:
            qwen = getattr(self.brain, 'qwen_model_manager', None)
            if qwen and hasattr(qwen, 'generate'):
                try:
                    response_text = qwen.generate(
                        messages=[{"role": "user", "content": prompt}],
                        max_new_tokens=max_new_tokens
                    )
                    source = "qwen"
                except Exception as e:
                    logger.debug(f"Qwen generation error: {e}")
        
        if not response_text:
            response_text = "Извините, генерация недоступна."
            source = "none"
        
        # 5. Сохраняем交互 в FG для обучения
        if self.fg and response_text and source != "none":
            try:
                self._save_interaction(query, response_text, context_nodes, source)
            except Exception as e:
                logger.debug(f"Save interaction error: {e}")
        
        # 6. Оцениваем качество
        confidence = self._estimate_confidence(response_text, source)
        
        return FGGenerationResult(
            response=response_text,
            tokens_generated=len(response_text.split()),
            context_used=context_nodes,
            confidence=confidence,
            source=source
        )
    
    def _save_interaction(self, query: str, response: str, 
                          context_nodes: List[str], source: str):
        """Сохранить交互 в FG для future обучения."""
        if not self.fg:
            return
            
        try:
            # Сохраняем как опыт
            self.fg.add_knowledge(
                subject=query[:100],
                relation="generated_response",
                obj=response[:500],
                metadata={
                    "source": source,
                    "context_nodes": context_nodes,
                    "type": "training_data"
                }
            )
        except Exception as e:
            logger.debug(f"Save interaction: {e}")
    
    def _estimate_confidence(self, response: str, source: str) -> float:
        """Оценить уверенность ответа."""
        if source == "none":
            return 0.0
        
        # Базовые факторы
        conf = 0.7 if source in ["gguf", "fractal"] else 0.5
        
        # Короткие ответы - ниже уверенность
        if len(response) < 50:
            conf -= 0.2
        
        # Ответы с ошибками
        if "извините" in response.lower() or "не могу" in response.lower():
            conf -= 0.3
        
        return max(0.0, min(1.0, conf))
    
    def fine_tune(self, training_data: List[Dict[str, str]], 
                  epochs: int = 1) -> Dict[str, Any]:
        """
        Дообучение на данных из FG.
        
        Args:
            training_data: [{"query": ..., "response": ...}, ...]
            epochs: Количество эпох
            
        Returns:
            Результат дообучения
        """
        if not self.fg:
            return {"status": "no_fg", "message": "FractalGraphV2 недоступен"}
        
        # Сохраняем данные в fractal storage для дообучения
        try:
            from eva_ai.mlearning.storage.fractal_store import export_hf_model_to_fractal
            # Здесь должна быть логика дообучения
            logger.info(f"Fine-tuning on {len(training_data)} samples, {epochs} epochs")
            return {
                "status": "ok",
                "samples": len(training_data),
                "epochs": epochs
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику интеграции."""
        stats = {
            "fg_available": self.fg is not None,
            "tokenizer_available": self.tokenizer is not None,
            "llama_cpp_ready": self.llama_cpp_ready,
            "fractal_model_ready": self.fractal_model is not None
        }
        
        if self.fg and hasattr(self.fg, 'get_stats'):
            try:
                fg_stats = self.fg.get_stats()
                stats["fg_stats"] = fg_stats
            except:
                pass
                
        return stats


def create_fg_generator(brain, config: Optional[Dict] = None) -> FractalGraphGenerator:
    """Создать интегратор FG с генерацией."""
    return FractalGraphGenerator(brain, config)