"""
Система обучения для улучшения генерации текста CogniFlex
"""
from __future__ import annotations

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import numpy as np

logger = logging.getLogger("cogniflex.text_trainer")


@dataclass
class TrainingConfig:
    """Конфигурация обучения"""
    learning_rate: float = 5e-5
    batch_size: int = 4
    num_epochs: int = 3
    max_length: int = 32768
    warmup_steps: int = 100
    weight_decay: float = 0.01
    save_steps: int = 500
    eval_steps: int = 100
    gradient_accumulation_steps: int = 4


class TextDataset(Dataset):
    """Датасет для обучения текстовой модели"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 32768):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Токенизируем текст
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }


class TextQualityTrainer:
    """Тренер для улучшения качества генерации текста"""
    
    def __init__(self, model, tokenizer, config: TrainingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Перемещаем модель на устройство
        self.model.to(self.device)
        
        # Создаем оптимизатор
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Счетчики шагов
        self.global_step = 0
        self.epoch = 0
        
        # История обучения
        self.training_history = []
        
        logger.info(f"TextQualityTrainer инициализирован на {self.device}")
    
    def prepare_training_data(self, additional_texts: Optional[List[str]] = None) -> List[str]:
        """Готовит данные для обучения
        
        Args:
            additional_texts: Дополнительные тексты для обучения
            
        Returns:
            List[str]: Список текстов для обучения
        """
        # Базовые примеры качественных русских текстов
        training_texts = [
            # Приветствия
            "Здравствуйте! Рад помочь вам с вашим вопросом.",
            "Добрый день! Чем я могу быть полезен?",
            "Приветствую! Я готов ответить на ваши вопросы.",
            
            # Ответы на вопросы
            "Машинное обучение - это область искусственного интеллекта.",
            "Нейронные сети имитируют работу человеческого мозга.",
            "Фрактальные структуры обладают самоподобием на разных масштабах.",
            
            # Развернутые ответы
            "Искусственный интеллект развивается очень быстро в последние годы.",
            "Современные технологии позволяют решать сложные задачи.",
            "Научные открытия меняют наше представление о мире.",
            
            # Короткие ответы
            "Да, это возможно.",
            "Интересный вопрос!",
            "Давайте разберем это подробнее.",
            
            # Технические объяснения
            "Алгоритм работает по шагам для достижения цели.",
            "Система обрабатывает данные и возвращает результат.",
            
            # Объяснения концепций
            "Концепция включает в себя несколько ключевых аспектов.",
            "Процесс состоит из нескольких последовательных этапов.",
            "Методология основана на проверенных принципах.",
            
            # Аналитические ответы
            "Анализ показывает интересные закономерности.",
            "Результаты исследования подтверждают гипотезу.",
            "Данные указывают на важные тенденции.",
            
            # Описательные тексты
            "Система характеризуется высокой производительностью.",
            "Процесс отличается эффективностью и надежностью.",
            "Технология обеспечивает превосходные результаты."
        ]
        
        # Добавляем дополнительные тексты если предоставлены
        if additional_texts:
            training_texts.extend(additional_texts)
        
        # Удаляем дубликаты и пустые строки
        unique_texts = []
        seen = set()
        for text in training_texts:
            if text and text.strip() and text not in seen:
                unique_texts.append(text)
                seen.add(text)
        
        logger.info(f"Подготовлено {len(unique_texts)} текстов для обучения")
        return unique_texts
    
    def _generate_quality_examples(self) -> List[str]:
        """Генерирует примеры качественных текстов"""
        examples = [
            # Вопрос-ответ пары
            "Вопрос: Что такое искусственный интеллект? Ответ: Искусственный интеллект - это технология, которая позволяет машинам думать и учиться.",
            "Вопрос: Как работают нейронные сети? Ответ: Нейронные сети работают подобно человеческому мозгу, обрабатывая информацию через слои нейронов.",
            "Вопрос: Что такое фрактал? Ответ: Фрактал - это геометрическая фигура, которая повторяет себя при изменении масштаба.",
            
            # Описательные тексты
            "Современные технологии стремительно меняют наш мир. Каждое новое открытие открывает новые возможности для человечества.",
            "Наука и техника развиваются в тесной связи. Успехи в одной области часто приводят к прорывам в другой.",
            
            # Объяснения
            "Машинное обучение использует алгоритмы для анализа данных. Система учится на примерах и улучшает свои результаты со временем.",
            
            # Диалоги
            "Пользователь: Помоги мне понять эту тему. Система: Конечно, я с удовольствием объясню все детали.",
        ]
        
        return examples
    
    def train(self, training_texts: List[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Запускает обучение модели
        
        Args:
            training_texts: Тексты для обучения
            config: Дополнительная конфигурация обучения
            
        Returns:
            Dict[str, Any]: Результат обучения
        """
        if training_texts is None:
            training_texts = self.prepare_training_data()
        
        # Применяем дополнительную конфигурацию если предоставлена
        if config:
            if 'epochs' in config:
                self.config.num_epochs = config['epochs']
            if 'batch_size' in config:
                self.config.batch_size = config['batch_size']
            if 'learning_rate' in config:
                self.config.learning_rate = config['learning_rate']
            if 'max_length' in config:
                self.config.max_length = config['max_length']
        
        if len(training_texts) < 10:
            logger.warning("Слишком мало данных для обучения, требуется минимум 10 примеров")
            return {"status": "error", "message": "Недостаточно данных"}
        
        logger.info(f"Начало обучения на {len(training_texts)} примеров")
        
        # Создаем датасет и загрузчик
        dataset = TextDataset(training_texts, self.tokenizer, self.config.max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Переводим модель в режим обучения
        self.model.train()
        
        # Основной цикл обучения
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0
            
            logger.info(f"Эпоха {epoch + 1}/{self.config.num_epochs}")
            
            for batch_idx, batch in enumerate(dataloader):
                # Перемещаем данные на устройство
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Прямой проход
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                
                # Обратный проход
                loss.backward()
                
                # Накопление градиентов
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Логирование
                if self.global_step % self.config.eval_steps == 0:
                    avg_loss = epoch_loss / max(num_batches, 1)
                    logger.info(f"Step {self.global_step}, Loss: {avg_loss:.4f}")
                    
                    # Сохраняем историю
                    self.training_history.append({
                        'step': self.global_step,
                        'epoch': epoch,
                        'loss': avg_loss,
                        'timestamp': time.time()
                    })
            
            # Средняя потеря за эпоху
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"Эпоха {epoch + 1} завершена. Средняя потеря: {avg_epoch_loss:.4f}")
        
        # Переводим модель в режим оценки
        self.model.eval()
        
        logger.info("Обучение завершено успешно")
        
        return {
            "status": "success",
            "epochs_trained": self.config.num_epochs,
            "final_loss": avg_epoch_loss,
            "steps_trained": self.global_step,
            "training_history": self.training_history
        }
    
    def evaluate_quality(self, test_queries: List[str]) -> Dict[str, Any]:
        """Оценивает качество генерации после обучения"""
        logger.info("Оценка качества генерации...")
        
        results = []
        self.model.eval()
        
        with torch.no_grad():
            for query in test_queries:
                try:
                    # Токенизируем запрос
                    inputs = self.tokenizer(
                        query,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_length
                    )
                    
                    input_ids = inputs['input_ids'].to(self.device)
                    attention_mask = inputs['attention_mask'].to(self.device)
                    
                    # Генерируем ответ
                    output = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=input_ids.shape[1] + 50,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7,
                        top_k=40,
                        top_p=0.85,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        early_stopping=True
                    )
                    
                    # Декодируем
                    response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    
                    # Очищаем ответ
                    response = self._clean_response(response, query)
                    
                    results.append({
                        'query': query,
                        'response': response,
                        'length': len(response),
                        'has_russian': self._has_russian_text(response)
                    })
                    
                except Exception as e:
                    logger.error(f"Ошибка при оценке запроса '{query}': {e}")
                    results.append({
                        'query': query,
                        'response': f"Ошибка: {str(e)}",
                        'length': 0,
                        'has_russian': False
                    })
        
        # Анализируем результаты
        total_queries = len(results)
        successful_responses = len([r for r in results if r['has_russian'] and r['length'] > 10])
        avg_length = np.mean([r['length'] for r in results])
        
        quality_score = successful_responses / total_queries if total_queries > 0 else 0
        
        evaluation = {
            'total_queries': total_queries,
            'successful_responses': successful_responses,
            'quality_score': quality_score,
            'avg_response_length': avg_length,
            'results': results
        }
        
        logger.info(f"Оценка завершена. Качество: {quality_score:.2f}")
        
        return evaluation
    
    def _clean_response(self, response: str, query: str) -> str:
        """Очищает сгенерированный ответ"""
        # Убираем запрос из ответа
        if response.lower().startswith(query.lower()):
            response = response[len(query):].strip()
        
        # Убираем артефакты
        import re
        response = re.sub(r'[^\w\s\.\,\!\?\;\:\-\—\nа-яА-ЯёЁ]', '', response)
        response = re.sub(r'\s+', ' ', response)
        
        return response.strip()
    
    def _has_russian_text(self, text: str) -> bool:
        """Проверяет наличие русского текста"""
        import re
        russian_chars = len(re.findall(r'[а-яА-ЯёЁ]', text))
        return russian_chars > len(text) * 0.3
    
    def save_model(self, save_path: str):
        """Сохраняет обученную модель"""
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # Сохраняем модель
            torch.save(self.model.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))
            
            # Сохраняем токенизатор
            self.tokenizer.save_pretrained(save_path)
            
            # Сохраняем конфигурацию обучения
            config_data = {
                'training_config': self.config.__dict__,
                'training_history': self.training_history,
                'final_step': self.global_step,
                'final_epoch': self.epoch,
                'timestamp': time.time()
            }
            
            with open(os.path.join(save_path, 'training_config.json'), 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Модель сохранена в: {save_path}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}")
            raise
    
    def train_async(self, training_texts: List[str] = None, config: Optional[Dict[str, Any]] = None) -> bool:
        """Запускает обучение модели в фоновом потоке
        
        Args:
            training_texts: Тексты для обучения
            config: Конфигурация обучения
            
        Returns:
            True если обучение запущено успешно
        """
        try:
            import threading
            
            def train_in_background():
                try:
                    self.train(training_texts, config)
                    logger.info("Фоновое обучение завершено")
                except Exception as e:
                    logger.error(f"Ошибка фонового обучения: {e}")
            
            # Запускаем обучение в фоновом потоке
            thread = threading.Thread(target=train_in_background, daemon=True)
            thread.start()
            
            logger.info("Фоновое обучение запущено")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка запуска фонового обучения: {e}")
            return False
