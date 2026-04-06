"""
Optimized Inference для EVA.
BetterTransformer + KV-Cache + batch processing.
"""
import os
import sys
import time
import logging
from typing import Optional, List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("eva.mlearning.hot_deployment.optimized_inference")


class OptimizedQwenGenerator:
    """
    Оптимизированный генератор Qwen.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        use_cache: bool = True,
        use_compile: bool = False
    ):
        self.model_path = model_path
        self.device = device
        self.use_cache = use_cache
        self.use_compile = use_compile
        
        self.model = None
        self.tokenizer = None
        self._initialized = False
        
    def initialize(self) -> bool:
        """Инициализирует оптимизированный генератор"""
        try:
            logger.info(f"Загрузка токенизатора...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Загрузка модели...")
            
            # Загружаем модель
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            
            # Применяем оптимизации
            self._apply_optimizations()
            
            self._initialized = True
            logger.info("Оптимизированный генератор готов")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации: {e}")
            return False
    
    def _apply_optimizations(self):
        """Применяет оптимизации для CPU"""
        self.model.eval()
        
        # torch.compile - может не работать на всех моделях
        if self.use_compile and hasattr(torch, 'compile'):
            try:
                logger.info("Применяем torch.compile...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("torch.compile применен")
            except Exception as e:
                logger.warning(f"torch.compile не применим: {e}")
        
        # Enable KV cache
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'use_cache'):
                logger.info(f"KV cache в конфиге: {self.model.config.use_cache}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> Optional[str]:
        """Генерирует текст"""
        if not self._initialized:
            logger.error("Генератор не инициализирован")
            return None
        
        try:
            # Токенизация
            start_tok = time.time()
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
            tok_time = time.time() - start_tok
            
            logger.info(f"Токенизация: {tok_time:.3f}s, {input_ids.shape[1]} токенов")
            
            # Генерация
            start_gen = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **kwargs
                )
            
            gen_time = time.time() - start_gen
            
            # Декодирование
            response = self.tokenizer.decode(
                outputs.sequences[0],
                skip_special_tokens=True
            )
            
            # Убираем промпт
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            tokens_generated = outputs.sequences[0].shape[0] - input_ids.shape[1]
            speed = tokens_generated / gen_time if gen_time > 0 else 0
            
            logger.info(f"Сгенерировано: {tokens_generated} токенов за {gen_time:.1f}s ({speed:.2f} tok/s)")
            
            return response
            
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_streaming(self, prompt: str, max_new_tokens: int = 50):
        """Генерация с потоковым выводом (эмуляция)"""
        if not self._initialized:
            return
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        # Генерируем токен за токеном
        self.model.eval()
        
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(generated)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply temperature and top-p
                if hasattr(self, 'temperature') and self.temperature > 0:
                    next_token_logits = next_token_logits / self.temperature
                
                # Greedy или sampling
                if hasattr(self, 'do_sample') and self.do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Декодируем и выводим
            text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            yield text
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break


class MultiBatchGenerator:
    """
    Генератор с батчевой обработкой для ускорения.
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._initialized = False
        
    def initialize(self) -> bool:
        """Инициализация"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        self.model.eval()
        self._initialized = True
        return True
    
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 50
    ) -> List[str]:
        """Генерирует для батча запросов"""
        if not self._initialized:
            return []
        
        # Токенизация батча
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Генерация
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )
        
        # Декодирование
        results = []
        for i, output in enumerate(outputs):
            # Убираем input из output
            input_len = inputs["input_ids"].shape[1]
            generated = output[input_len:]
            
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            results.append(text)
        
        return results


# ============================================================================
# Тесты
# ============================================================================

def test_optimized_generation():
    """Тест оптимизированной генерации"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(project_root, "mlearning", "eva_models", "qwen3.5-0.8b")
    
    logger.info(f"Путь: {model_path}")
    
    generator = OptimizedQwenGenerator(
        model_path=model_path,
        use_compile=False  # torch.compile требует много памяти
    )
    
    if not generator.initialize():
        logger.error("Ошибка инициализации")
        return
    
    logger.info("\n=== Тест генерации ===")
    
    # Прогрев - первая генерация (медленная)
    logger.info("Первая генерация (прогрев)...")
    start = time.time()
    resp1 = generator.generate("Привет!", max_new_tokens=10)
    t1 = time.time() - start
    logger.info(f"Ответ: {resp1[:100]}")
    logger.info(f"Время: {t1:.1f}s")
    
    # Вторая генерация
    logger.info("\nВторая генерация...")
    start = time.time()
    resp2 = generator.generate("Как дела?", max_new_tokens=10)
    t2 = time.time() - start
    logger.info(f"Ответ: {resp2[:100]}")
    logger.info(f"Время: {t2:.1f}s")
    
    # Третья генерация - чуть длиннее
    logger.info("\nТретья генерация (30 токенов)...")
    start = time.time()
    resp3 = generator.generate("Расскажи анекдот", max_new_tokens=30)
    t3 = time.time() - start
    logger.info(f"Ответ: {resp3[:200]}")
    logger.info(f"Время: {t3:.1f}s")
    
    # Сравнение
    logger.info("\n=== Результаты ===")
    logger.info(f"10 токенов: {t1:.1f}s")
    logger.info(f"10 токенов (2й): {t2:.1f}s")
    logger.info(f"30 токенов: {t3:.1f}s")
    
    avg_speed = 50 / ((t1 + t2 + t3) / 3)
    logger.info(f"Средняя скорость: ~{avg_speed:.2f} tok/s")


def test_batch_generation():
    """Тест батчевой генерации"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(project_root, "mlearning", "eva_models", "qwen3.5-0.8b")
    
    logger.info("=== Тест батчевой генерации ===")
    
    generator = MultiBatchGenerator(model_path)
    generator.initialize()
    
    prompts = [
        "Привет",
        "Как дела",
        "Что такое AI",
        "Расскажи анекдот"
    ]
    
    start = time.time()
    results = generator.generate_batch(prompts, max_new_tokens=20)
    elapsed = time.time() - start
    
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        logger.info(f"{i+1}. {prompt} -> {result[:50]}...")
    
    logger.info(f"\nБатч из 4x: {elapsed:.1f}s ({elapsed/4:.1f}s на запрос)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    test_optimized_generation()
    # test_batch_generation()  # опционально