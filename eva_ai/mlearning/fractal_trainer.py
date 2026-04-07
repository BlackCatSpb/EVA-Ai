"""
Тренировочный конвейер для Fractal Transformer.
"""
import os
import torch
import logging
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup

from .fractal_transformer import FractalTransformer, FractalConfig
from .tokenization_fractal import ExtendedFractalTokenizer
from .neuromorphic_memory import NeuromorphicMemoryLayer

logger = logging.getLogger("eva_ai.trainer")

class FractalKnowledgeTrainer:
    """
    Класс для обучения и адаптации Fractal Transformer.
    """
    
    def __init__(
        self,
        model: Optional[FractalTransformer] = None,
        tokenizer: Optional[ExtendedFractalTokenizer] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """
        Инициализация тренера.
        
        Args:
            model: Экземпляр FractalTransformer
            tokenizer: Токенизатор
            config: Конфигурация обучения
            device: Устройство для обучения (CPU/GPU)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = self._get_default_config()
        
        if config:
            self.config.update(config)
        
        # Устройство
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Оптимизатор и планировщик
        self.optimizer = None
        self.scheduler = None
        
        # Инициализация модели, если не передана
        if self.model is None:
            self._init_model()
        
        # Перенос модели на устройство
        self.model = self.model.to(self.device)
        
        # Инициализация памяти, если используется
        self.memory = None
        if self.config.get('use_memory', True):
            self.memory = NeuromorphicMemoryLayer(
                hidden_size=self.model.config.hidden_size,
                memory_slots=self.config.get('memory_slots', 32),
                memory_size=self.config.get('memory_size', 512),
                num_heads=self.model.config.num_attention_heads,
                dropout=self.model.config.hidden_dropout_prob
            ).to(self.device)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Возвращает конфигурацию по умолчанию."""
        return {
            'batch_size': 8,
            'learning_rate': 5e-5,
            'weight_decay': 0.01,
            'num_train_epochs': 3,
            'warmup_steps': 0,
            'max_grad_norm': 1.0,
            'gradient_accumulation_steps': 1,
            'logging_steps': 10,
            'save_steps': 1000,
            'save_total_limit': 5,
            'fp16': False,
            'fp16_opt_level': 'O1',
            'use_memory': True,
            'memory_slots': 32,
            'memory_size': 512,
            'fractal_levels': 4,
        }
    
    def _init_model(self):
        """Инициализация модели, если она не была передана."""
        if self.tokenizer is None:
            raise ValueError("Токенизатор должен быть передан или инициализирован до модели")
        
        # Создаем конфигурацию модели
        model_config = {
            'vocab_size': self.tokenizer.vocab_size,
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'max_position_embeddings': 512,
            'fractal_levels': self.config.get('fractal_levels', 4),
        }
        
        # Создаем модель
        self.model = FractalTransformer(FractalConfig(**model_config))
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: Optional[Union[str, os.PathLike]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Обучение модели.
        
        Args:
            train_dataset: Набор данных для обучения
            eval_dataset: Набор данных для валидации
            output_dir: Директория для сохранения чекпоинтов
            **kwargs: Дополнительные аргументы
            
        Returns:
            Словарь с метриками обучения
        """
        # Обновляем конфигурацию переданными аргументами
        self.config.update(kwargs)
        
        # Создаем DataLoader для обучения
        train_dataloader = self._get_dataloader(train_dataset, is_training=True)
        
        # Вычисляем общее количество шагов
        num_update_steps_per_epoch = len(train_dataloader) // self.config['gradient_accumulation_steps']
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        
        total_train_steps = int(num_update_steps_per_epoch * self.config['num_train_epochs'])
        
        # Инициализация оптимизатора и планировщика
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler(total_train_steps)
        
        # Подготовка к обучению с mixed precision, если нужно
        scaler = torch.cuda.amp.GradScaler() if self.config.get('fp16', False) else None
        
        # Обучение
        logger.info("***** Запуск обучения *****")
        logger.info(f"  Размер батча = {self.config['batch_size']}")
        logger.info(f"  Всего шагов = {total_train_steps}")
        logger.info(f"  Количество эпох = {self.config['num_train_epochs']}")
        
        self.model.zero_grad()
        self.model.train()
        
        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        
        # Основной цикл обучения
        for epoch in range(epochs_trained, int(self.config['num_train_epochs'])):
            epoch_iterator = tqdm(
                train_dataloader,
                desc=f"Эпоха {epoch + 1}/{int(self.config['num_train_epochs'])}",
                mininterval=10
            )
            
            for step, batch in enumerate(epoch_iterator):
                # Пропускаем уже обработанные шаги, если возобновляем обучение
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                
                # Обработка батча
                batch = self._prepare_inputs(batch)
                
                with torch.cuda.amp.autocast(enabled=self.config.get('fp16', False)):
                    outputs = self._training_step(batch)
                    loss = outputs['loss']
                    
                    if self.config['gradient_accumulation_steps'] > 1:
                        loss = loss / self.config['gradient_accumulation_steps']
                
                # Обратное распространение с масштабированием градиентов
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Обновление весов
                if (step + 1) % self.config['gradient_accumulation_steps'] == 0 or step == len(epoch_iterator) - 1:
                    # Обрезка градиентов
                    if self.config['max_grad_norm'] > 0:
                        if scaler is not None:
                            scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['max_grad_norm']
                        )
                    
                    # Шаг оптимизатора
                    if scaler is not None:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    
                    # Обновление планировщика
                    if self.scheduler is not None:
                        self.scheduler.step()
                    
                    # Обнуление градиентов
                    self.model.zero_grad()
                    global_step += 1
                    
                    # Логирование и сохранение
                    if global_step % self.config['logging_steps'] == 0:
                        logs = {
                            'loss': loss.item(),
                            'learning_rate': self._get_learning_rate(),
                            'step': global_step,
                        }
                        
                        if self.config.get('use_memory', False) and self.memory is not None:
                            # Добавляем метрики памяти, если используется
                            memory_metrics = self._get_memory_metrics()
                            logs.update(memory_metrics)
                        
                        logger.info(f"Шаг {global_step}: {logs}")
                    
                    # Сохранение модели
                    if self.config['save_steps'] > 0 and global_step % self.config['save_steps'] == 0:
                        self._save_checkpoint(output_dir, global_step)
            
            # Валидация после эпохи
            if eval_dataset is not None:
                eval_results = self.evaluate(eval_dataset)
                logger.info(f"Результаты после эпохи {epoch + 1}: {eval_results}")
        
        # Сохранение финальной модели
        if output_dir is not None:
            self._save_checkpoint(output_dir, global_step)
        
        return {}
    
    def evaluate(
        self,
        eval_dataset: Dataset,
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        """
        Оценка модели на наборе данных.
        
        Args:
            eval_dataset: Набор данных для оценки
            metric_key_prefix: Префикс для ключей метрик
            
        Returns:
            Словарь с метриками
        """
        eval_dataloader = self._get_dataloader(eval_dataset, is_training=False)
        
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        for batch in tqdm(eval_dataloader, desc="Оценка"):
            batch = self._prepare_inputs(batch)
            
            with torch.no_grad():
                outputs = self._evaluation_step(batch)
                total_loss += outputs['loss'].item() * len(batch['input_ids'])
                total_samples += len(batch['input_ids'])
        
        avg_loss = total_loss / total_samples
        perplexity = math.exp(avg_loss)
        
        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_perplexity": perplexity,
        }
        
        return metrics
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Один шаг обучения."""
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'labels': batch['labels']
        }
        
        # Добавляем память, если используется
        if self.config.get('use_memory', False) and self.memory is not None:
            memory_outputs = self.memory(
                hidden_states=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            inputs['memory'] = memory_outputs[0]
        
        # Прямой проход
        outputs = self.model(**inputs)
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states
        }
    
    def _evaluation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Один шаг оценки."""
        return self._training_step(batch)
    
    def _prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Подготовка входных данных для модели."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    
    def _get_dataloader(
        self,
        dataset: Dataset,
        is_training: bool = False
    ) -> DataLoader:
        """Создает DataLoader для набора данных."""
        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=is_training,
            num_workers=self.config.get('num_workers', 0),
            pin_memory=self.config.get('pin_memory', True)
        )
    
    def _create_optimizer(self):
        """Создает оптимизатор."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config['weight_decay'],
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config['learning_rate'],
            eps=self.config.get('adam_epsilon', 1e-8),
            weight_decay=self.config['weight_decay']
        )
    
    def _create_scheduler(self, num_training_steps: int):
        """Создает планировщик скорости обучения."""
        warmup_steps = self.config.get('warmup_steps', 0)
        
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def _get_learning_rate(self) -> float:
        """Возвращает текущую скорость обучения."""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0
    
    def _get_memory_metrics(self) -> Dict[str, float]:
        """Возвращает метрики памяти."""
        if self.memory is None:
            return {}
        
        # Пример метрик памяти (можно расширить)
        memory_tensor = self.memory.memory.data
        return {
            'memory_mean': memory_tensor.mean().item(),
            'memory_std': memory_tensor.std().item(),
            'memory_max': memory_tensor.max().item(),
            'memory_min': memory_tensor.min().item(),
        }
    
    def _save_checkpoint(self, output_dir: Union[str, os.PathLike], step: int):
        """Сохраняет чекпоинт модели."""
        if output_dir is None:
            return
        
        output_dir = Path(output_dir) / f"checkpoint-{step}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем модель и токенизатор
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        
        # Сохраняем состояние оптимизатора и планировщика
        torch.save(
            {
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
                'step': step,
                'config': self.config,
            },
            output_dir / 'trainer_state.pt'
        )
        
        # Сохраняем состояние памяти, если используется
        if self.config.get('use_memory', False) and self.memory is not None:
            torch.save(
                self.memory.state_dict(),
                output_dir / 'memory_state.pt'
            )
        
        logger.info(f"Чекпоинт сохранен в {output_dir}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, os.PathLike],
        **kwargs
    ) -> 'FractalKnowledgeTrainer':
        """Загружает предобученный тренер."""
        model_name_or_path = Path(model_name_or_path)
        
        # Загружаем конфигурацию
        config_path = model_name_or_path / 'trainer_config.json'
        if config_path.exists():
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Обновляем конфигурацию переданными аргументами
        config.update(kwargs)
        
        # Загружаем модель и токенизатор
        model = FractalTransformer.from_pretrained(model_name_or_path)
        tokenizer = ExtendedFractalTokenizer.from_pretrained(model_name_or_path)
        
        # Создаем экземпляр тренера
        trainer = cls(model=model, tokenizer=tokenizer, config=config)
        
        # Загружаем состояние тренера, если есть
        trainer_state_path = model_name_or_path / 'trainer_state.pt'
        if trainer_state_path.exists():
            trainer_state = torch.load(trainer_state_path, map_location='cpu', weights_only=False)
            trainer.optimizer.load_state_dict(trainer_state['optimizer'])
            
            if trainer.scheduler is not None and 'scheduler' in trainer_state:
                trainer.scheduler.load_state_dict(trainer_state['scheduler'])
            
            # Загружаем состояние памяти, если используется
            if trainer.config.get('use_memory', False) and trainer.memory is not None:
                memory_state_path = model_name_or_path / 'memory_state.pt'
                if memory_state_path.exists():
                    trainer.memory.load_state_dict(torch.load(memory_state_path, weights_only=False))
        
        return trainer
