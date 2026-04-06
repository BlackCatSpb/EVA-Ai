"""
OpenVINO инференс для EVA.
Оптимизирован для CPU Intel с ускорением в 2-3x.
"""
import os
import sys
import time
import logging
import numpy as np
from typing import Optional, List, Dict, Any

logger = logging.getLogger("eva.mlearning.hot_deployment.openvino_inference")

try:
    import openvino as ov
    from openvino_tokenizers import convert_tokenizer
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False
    logger.error("OpenVINO не установлен: pip install openvino openvino-tokenizers")


class OpenVINOGenerator:
    """
    Генератор через OpenVINO.
    Оптимизирован для Intel CPU (AVX2, AMX).
    """
    
    def __init__(
        self,
        model_path: str,
        ir_path: Optional[str] = None,
        num_threads: int = 8
    ):
        self.model_path = model_path
        self.ir_path = ir_path  # Путь к OpenVINO IR (xml/bin)
        self.num_threads = num_threads
        
        self.core = None
        self.model = None
        self.compiled_model = None
        self.tokenizer = None
        self._initialized = False
        
    def convert_to_ir(
        self,
        output_dir: str,
        precision: str = "fp16"
    ) -> bool:
        """
        Конвертирует модель в OpenVINO IR формат.
        """
        if not HAS_OPENVINO:
            return False
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Конвертация модели в OpenVINO IR...")
            
            # Загружаем модель
            logger.info("Загрузка модели...")
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype="float32",
                device_map="cpu"
            )
            model.eval()
            
            # Создаём директорию
            os.makedirs(output_dir, exist_ok=True)
            
            # Используем ov.convert_model для конвертации
            logger.info("Конвертация в OpenVINO...")
            
            # Подготавливаем вход
            dummy_input = tokenizer("test", return_tensors="pt")
            input_ids = dummy_input["input_ids"]
            
            # Конвертация через torch.onnx -> OpenVINO
            # Сначала создаём ONNX
            import torch
            
            onnx_path = os.path.join(output_dir, "model.onnx")
            
            # Пробуем с упрощённой моделью (без cache)
            logger.info("Экспорт в ONNX...")
            
            try:
                # Пробуем через HF официальный конвертер
                from optimum.exporters.openvino import export
                
                export(
                    self.model_path,
                    output_dir=output_dir,
                    task="text-generation",
                    batch_size=1,
                    sequence_length=256,
                    precision=precision
                )
                
                logger.info(f"OpenVINO модель сохранена: {output_dir}")
                self.ir_path = output_dir
                return True
                
            except Exception as e:
                logger.warning(f"optimum export failed: {e}")
                logger.info("Пробуем альтернативный метод...")
                
                # Пробуем через ov.convert_model
                try:
                    from optimum.utils import OvModel
                    
                    ov_model = OvModel.from_pretrained(self.model_path)
                    ov_model.save_pretrained(output_dir)
                    
                    logger.info(f"Сохранено в: {output_dir}")
                    self.ir_path = output_dir
                    return True
                    
                except Exception as e2:
                    logger.error(f"Альтернатива не работает: {e2}")
                    return False
            
        except Exception as e:
            logger.error(f"Ошибка конвертации: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load(self) -> bool:
        """Загружает OpenVINO модель"""
        if not HAS_OPENVINO:
            logger.error("OpenVINO недоступен")
            return False
        
        # Проверяем IR файлы
        if self.ir_path is None:
            # Ищем в директории модели
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            ir_dir = os.path.join(project_root, "models", "qwen3.5-0.8b-openvino")
            
            if os.path.exists(ir_dir):
                # Ищем .xml файл
                xml_files = [f for f in os.listdir(ir_dir) if f.endswith('.xml')]
                if xml_files:
                    self.ir_path = os.path.join(ir_dir, xml_files[0])
        
        if self.ir_path is None or not os.path.exists(self.ir_path):
            logger.error(f"OpenVINO IR не найден: {self.ir_path}")
            return False
        
        try:
            logger.info(f"Загрузка OpenVINO модели: {self.ir_path}")
            
            self.core = ov.Core()
            
            # Читаем модель
            self.model = self.core.read_model(self.ir_path)
            
            # Компилируем для CPU
            logger.info(f"Компиляция для CPU (threads={self.num_threads})...")
            
            self.compiled_model = self.core.compile_model(
                self.model,
                "CPU",
                properties={
                    "PERF_COUNT": "YES",
                    "INFERENCE_PRECISION_HINT": "f32",
                    "NUM_STREAMS": "1",
                    "AFFINITY": "NUMA",
                    "INFERENCE_NUM_THREADS": str(self.num_threads)
                }
            )
            
            # Получаем входы/выходы
            self.input_names = [inp.get_any_name() for inp in self.compiled_model.inputs]
            self.output_names = [out.get_any_name() for out in self.compiled_model.outputs]
            
            logger.info(f"Входы: {self.input_names}")
            logger.info(f"Выходы: {self.output_names}")
            
            logger.info("OpenVINO модель загружена!")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки OpenVINO: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_tokenizer(self, tokenizer_path: str):
        """Загружает токенизатор"""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Токенизатор загружен")
        except Exception as e:
            logger.error(f"Ошибка токенизатора: {e}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.1,
        top_p: float = 0.9,
        **kwargs
    ) -> Optional[str]:
        """Генерирует текст через OpenVINO"""
        if self.compiled_model is None or self.tokenizer is None:
            logger.error("Модель или токенизатор не загружены")
            return None
        
        try:
            start = time.time()
            
            # Токенизация
            input_ids = self.tokenizer.encode(prompt, return_tensors="np")
            
            # Генерация токен за токеном
            generated = input_ids.tolist()[0]
            
            for step in range(max_new_tokens):
                # Подготовка входа
                input_tensor = ov.Tensor(
                    name=self.input_names[0],
                    tensor=ov.Tensor.from_numpy(input_ids.astype(np.int64))
                )
                
                # Предсказание
                results = self.compiled_model([input_tensor])
                
                # Получаем логиты
                logits = results[self.output_names[0]]
                next_token_logits = logits[0, -1, :]
                
                # Применяем temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Top-k
                top_k = kwargs.get('top_k', 40)
                if top_k > 0:
                    top_indices = np.argpartition(next_token_logits, -top_k)[-top_k:]
                    mask = np.ones_like(next_token_logits) * (-np.inf)
                    mask[top_indices] = next_token_logits[top_indices]
                    next_token_logits = mask
                
                # Top-p
                if top_p < 1.0:
                    sorted_indices = np.argsort(next_token_logits)[::-1]
                    sorted_probs = np.exp(next_token_logits[sorted_indices])
                    sorted_probs = sorted_probs / sorted_probs.sum()
                    
                    cumsum = np.cumsum(sorted_probs)
                    cutoff_idx = np.searchsorted(cumsum, top_p)
                    
                    mask = np.ones_like(next_token_logits) * (-np.inf)
                    mask[sorted_indices[:cutoff_idx+1]] = next_token_logits[sorted_indices[:cutoff_idx+1]]
                    next_token_logits = mask
                
                # Выбор токена
                next_token = int(np.argmax(next_token_logits))
                
                # Добавляем
                generated.append(next_token)
                input_ids = np.array([generated])
                
                # Остановка
                if next_token == self.tokenizer.eos_token_id:
                    break
                
                # Прогресс каждые 10 токенов
                if (step + 1) % 10 == 0:
                    elapsed = time.time() - start
                    current_speed = (step + 1) / elapsed
                    logger.info(f"Шаг {step+1}/{max_new_tokens}: {current_speed:.2f} tok/s")
            
            # Декодирование
            response = self.tokenizer.decode(generated, skip_special_tokens=True)
            
            elapsed = time.time() - start
            tokens_generated = len(generated) - len(input_ids[0])
            speed = tokens_generated / elapsed if elapsed > 0 else 0
            
            logger.info(f"Сгенерировано {tokens_generated} токенов за {elapsed:.1f}s ({speed:.2f} tok/s)")
            
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Ошибка генерации OpenVINO: {e}")
            import traceback
            traceback.print_exc()
            return None


def convert_and_test():
    """Конвертирует и тестирует OpenVINO"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(project_root, "mlearning", "eva_models", "qwen3.5-0.8b")
    output_dir = os.path.join(project_root, "models", "qwen3.5-0.8b-openvino")
    
    logger.info(f"Модель: {model_path}")
    logger.info(f"Выход: {output_dir}")
    
    # Проверяем, есть ли уже сконвертированная модель
    ir_files = []
    if os.path.exists(output_dir):
        ir_files = [f for f in os.listdir(output_dir) if f.endswith('.xml')]
    
    generator = OpenVINOGenerator(model_path, ir_path=output_dir if ir_files else None)
    
    if not ir_files:
        # Конвертируем
        logger.info("=== Конвертация в OpenVINO ===")
        success = generator.convert_to_ir(output_dir, precision="fp16")
        
        if not success:
            logger.error("Конвертация не удалась")
            return
    else:
        logger.info("Модель уже сконвертирована")
    
    # Загружаем
    logger.info("\n=== Загрузка модели ===")
    
    if not generator.load():
        logger.error("Не удалось загрузить модель")
        return
    
    # Токенизатор
    generator.load_tokenizer(model_path)
    
    # Тест генерации
    logger.info("\n=== Тест генерации ===")
    
    logger.info("Первая генерация (прогрев)...")
    start = time.time()
    resp1 = generator.generate("Привет!", max_new_tokens=10)
    t1 = time.time() - start
    
    logger.info(f"Ответ: {resp1[:100] if resp1 else 'None'}")
    logger.info(f"Время: {t1:.1f}s")
    
    # Вторая генерация
    logger.info("\nВторая генерация...")
    start = time.time()
    resp2 = generator.generate("Как дела?", max_new_tokens=20)
    t2 = time.time() - start
    
    logger.info(f"Ответ: {resp2[:150] if resp2 else 'None'}")
    logger.info(f"Время: {t2:.1f}s")
    
    # Сравнение
    logger.info("\n=== Итоги ===")
    logger.info(f"10 токенов: {t1:.1f}s")
    logger.info(f"20 токенов: {t2:.1f}s")
    
    # Рассчитываем скорость
    if t1 > 0:
        speed1 = 10 / t1
        logger.info(f"Скорость (10 токенов): {speed1:.2f} tok/s")
    if t2 > 0:
        speed2 = 20 / t2
        logger.info(f"Скорость (20 токенов): {speed2:.2f} tok/s")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    convert_and_test()