import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoConfig
from typing import Dict, List, Optional, Tuple
import re
from dataclasses import dataclass

@dataclass
class TokenCache:
    """Класс для кеширования токенов"""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None

class TextGenerator:
    def __init__(self, model_name: str = "sberbank-ai/rugpt3small_based_on_gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.cache = None
        
    def initialize(self):
        """Инициализация модели и токенизатора"""
        print("Инициализация модели и токенизатора...")
        
        # Загрузка конфигурации с увеличенным контекстом
        config = AutoConfig.from_pretrained(
            self.model_name,
            max_position_embeddings=2048,
            pad_token_id=50256  # eos_token_id для GPT2
        )
        
        # Загрузка токенизатора
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            self.model_name,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Загрузка модели
        self.model = GPT2LMHeadModel.from_pretrained(
            self.model_name,
            config=config
        ).to(self.device)
        self.model.eval()
        
        print(f"Модель загружена на устройство: {self.device}")
    
    def prepare_inputs(self, text: str, max_length: int = 1024) -> Dict:
        """Подготовка входных данных с учетом максимальной длины"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding='max_length' if self.cache is None else 'do_not_pad'
        ).to(self.device)
        
        if self.cache is not None:
            # Объединяем кешированные токены с новыми
            input_ids = torch.cat([self.cache.input_ids, inputs.input_ids[:, -1:]], dim=-1)
            attention_mask = torch.cat([self.cache.attention_mask, inputs.attention_mask[:, -1:]], dim=-1)
            
            # Обрезаем до максимальной длины, если необходимо
            if input_ids.size(1) > max_length:
                input_ids = input_ids[:, -max_length:]
                attention_mask = attention_mask[:, -max_length:]
                
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'past_key_values': self.cache.past_key_values
            }
            
        return inputs
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        max_context_length: int = 1536,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        num_return_sequences: int = 1
    ) -> str:
        """Генерация текста с кешированием"""
        if self.model is None or self.tokenizer is None:
            self.initialize()
            
        # Подготавливаем входные данные
        inputs = self.prepare_inputs(prompt, max_context_length)
        
        # Параметры генерации
        generation_params = {
            'max_new_tokens': max_new_tokens,
            'do_sample': True,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'repetition_penalty': repetition_penalty,
            'pad_token_id': self.tokenizer.eos_token_id,
            'num_return_sequences': num_return_sequences,
            'use_cache': True
        }
        
        # Генерация текста
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_params
            )
            
            # Обновляем кеш для следующего вызова
            self.cache = TokenCache(
                input_ids=outputs,
                attention_mask=torch.ones_like(outputs, device=self.device),
                past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None
            )
            
        # Декодируем сгенерированный текст
        generated_text = self.tokenizer.decode(
            outputs[0] if num_return_sequences == 1 else outputs,
            skip_special_tokens=True
        )
        
        return generated_text

def improved_generation():
    """Улучшенная генерация текста с кешированием"""
    print("=== Улучшенная генерация текста RuGPT3Small с кешированием ===\n")
    
    try:
        # Инициализация генератора
        generator = TextGenerator("sberbank-ai/rugpt3small_based_on_gpt2")
        
        # Промпты для тестирования
        prompts = [
            "Привет! Как твои дела?",
            "Расскажи подробнее о машинном обучении и искусственном интеллекте.",
            "Напиши небольшой рассказ о путешествии во времени."
        ]
        
        for prompt in prompts:
            print(f"\n{'='*80}")
            print(f"ПРОМПТ: {prompt}")
            print(f"{'='*80}")
            
            # Генерация текста
            generated_text = generator.generate(
                prompt=prompt,
                max_new_tokens=300,  # Увеличиваем количество новых токенов
                max_context_length=1536,  # Максимальная длина контекста
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2
            )
            
            # Разбиваем текст на предложения для лучшей читаемости
            sentences = re.split(r'(?<=[.!?])\s+', generated_text)
            formatted_text = '\n- ' + '\n- '.join(sentences)
            
            print(f"СГЕНЕРИРОВАННЫЙ ТЕКСТ:{formatted_text}")
            print(f"\nДлина текста: {len(generated_text)} символов")
            
            # Очищаем кеш между разными промптами
            generator.cache = None
        
        print("\n=== Улучшенная генерация с кешированием завершена успешно! ===")
        
    except Exception as e:
        print(f"\nОшибка при генерации: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    improved_generation()
