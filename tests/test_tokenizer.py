"""
Модуль для тестирования функциональности токенизатора в CogniFlex.
"""
import unittest
from unittest.mock import MagicMock, patch
import torch
from transformers import AutoTokenizer

from cogniflex.core.response_generator import ResponseGenerator
from cogniflex.core.core_brain import CoreBrain


class TestTokenizer(unittest.TestCase):
    """Тесты для функциональности токенизатора."""
    
    def setUp(self):
        """Настройка тестового окружения."""
        # Создаем мок для CoreBrain
        self.brain_mock = MagicMock(spec=CoreBrain)
        
        # Инициализируем тестовый токенизатор
        self.test_tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
        self.brain_mock.tokenizer = self.test_tokenizer
        
        # Инициализируем ResponseGenerator с тестовыми параметрами
        self.response_generator = ResponseGenerator(
            brain=self.brain_mock,
            tokenizer_config={
                'max_length': 512,
                'truncation': True,
                'padding': 'max_length',
                'return_tensors': 'pt',
                'add_special_tokens': True
            }
        )
    
    def test_safe_tokenize_basic(self):
        """Тест базовой функциональности _safe_tokenize."""
        # Подготовка
        test_text = "Привет, как дела?"
        
        # Действие
        result = self.response_generator._safe_tokenize(
            tokenizer=self.test_tokenizer,
            text=test_text,
            return_tokens=True
        )
        
        # Проверка
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(token, str) for token in result))
    
    def test_safe_tokenize_with_encoding(self):
        """Тест _safe_tokenize с return_tokens=False."""
        # Подготовка
        test_text = "Тестовый текст для токенизации"
        
        # Создаем мок токенизатора, который возвращает ожидаемую структуру
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode_plus.return_value = {
            'input_ids': [1, 2, 3, 4, 5],
            'attention_mask': [1, 1, 1, 1, 1]
        }
        mock_tokenizer.convert_ids_to_tokens.return_value = ['test', '##ing', 'text']
        mock_tokenizer.word_ids.return_value = [0, 1, 2, 2, 3]
        
        # Действие
        result = self.response_generator._safe_tokenize(
            tokenizer=mock_tokenizer,
            text=test_text,
            return_tokens=False
        )
        
        # Проверка
        self.assertIsInstance(result, dict)
        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)
        self.assertIn('tokens', result)
        self.assertIn('word_ids', result)
    
    def test_safe_tokenize_empty_text(self):
        """Тест _safe_tokenize с пустым текстом."""
        # Действие
        result = self.response_generator._safe_tokenize(
            tokenizer=self.test_tokenizer,
            text="",
            return_tokens=True
        )
        
        # Проверка
        self.assertEqual(result, [])
    
    def test_safe_tokenize_fallback(self):
        """Тест отката на split() при ошибке токенизации."""
        # Подготовка
        test_text = "Текст с ошибкой токенизации"
        
        # Создаем мок токенизатора, который вызывает исключение
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.side_effect = Exception("Test error")
        
        # Действие
        result = self.response_generator._safe_tokenize(
            tokenizer=mock_tokenizer,
            text=test_text,
            return_tokens=True
        )
        
        # Проверка
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(test_text.split()))
        self.assertEqual(" ".join(result), test_text)
    
    def test_tokenizer_validation(self):
        """Тест валидации токенизатора."""
        # Подготовка валидного токенизатора
        class ValidTokenizer:
            def tokenize(self, text):
                return text.split()
                
            def encode(self, text, **kwargs):
                return [1, 2, 3]
                
            def decode(self, tokens, **kwargs):
                return "decoded text"
                
        # Действие и проверка
        self.assertTrue(self.response_generator._validate_tokenizer(ValidTokenizer()))
        
        # Невалидный токенизатор без обязательных методов
        class InvalidTokenizer:
            pass
            
        self.assertFalse(self.response_generator._validate_tokenizer(InvalidTokenizer()))
        
        # Проверка с токенизатором, у которого vocab_size <= 0
        class TokenizerWithVocab:
            def __init__(self, vocab_size=0):
                self.vocab_size = vocab_size
                
            def tokenize(self, text):
                return text.split()
                
            def encode(self, text, **kwargs):
                return [1, 2, 3]
                
            def decode(self, tokens, **kwargs):
                return "decoded text"
        
        # Проверка с нулевым словарем (должно быть предупреждение, но не ошибка)
        tokenizer = TokenizerWithVocab(vocab_size=0)
        with self.assertLogs(level='WARNING') as cm:
            self.assertTrue(self.response_generator._validate_tokenizer(tokenizer))
        self.assertIn('Подозрительный размер словаря', cm.output[0])
    
    def test_tokenizer_config_override(self):
        """Тест переопределения конфигурации токенизатора."""
        # Подготовка
        custom_config = {
            'max_length': 256,
            'truncation': False,
            'custom_param': 'test'
        }
        
        # Действие
        rg = ResponseGenerator(
            brain=self.brain_mock,
            tokenizer_config=custom_config
        )
        
        # Проверка
        self.assertEqual(rg.tokenizer_config['max_length'], 256)
        self.assertFalse(rg.tokenizer_config['truncation'])
        self.assertEqual(rg.tokenizer_config['custom_param'], 'test')
        
        # Проверка, что значения по умолчанию остались без изменений
        self.assertEqual(rg.tokenizer_config['padding'], 'max_length')


if __name__ == '__main__':
    unittest.main()
