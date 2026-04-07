"""
ONNX Runtime инференс для eva_ai.
Оптимизирован для CPU с Intel OpenVINO.
"""
import os
import sys
import time
import logging
import numpy as np
from typing import Optional, List, Dict, Any
import torch

logger = logging.getLogger("eva_ai.mlearning.hot_deployment.onnx_runtime")

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    logger.error("ONNX Runtime не установлен: pip install onnx onnxruntime")


class OnnxRuntimeGenerator:
    """
    Генератор через ONNX Runtime.
    Значительно быстрее чем PyTorch на CPU.
    """
    
    def __init__(
        self,
        model_path: str,
        intra_threads: int = 8,
        inter_threads: int = 2,
        execution_provider: str = "CPUExecutionProvider"
    ):
        self.model_path = model_path
        self.intra_threads = intra_threads
        self.inter_threads = inter_threads
        self.execution_provider = execution_provider
        
        self.session = None
        self.tokenizer = None
        self._initialized = False
        
    def load(self) -> bool:
        """Загружает ONNX модель"""
        if not HAS_ONNX:
            logger.error("ONNX Runtime недоступен")
            return False
        
        if not os.path.exists(self.model_path):
            logger.error(f"ONNX модель не найдена: {self.model_path}")
            return False
        
        try:
            logger.info(f"Загрузка ONNX модели: {self.model_path}")
            
            # Настройки сессии для оптимизации CPU
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.intra_threads
            sess_options.inter_op_num_threads = self.inter_threads
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            # Провайдеры
            providers = [self.execution_provider]
            if self.execution_provider == "CPUExecutionProvider":
                providers = ["CPUExecutionProvider"]
            
            # Создаём сессию
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            # Получаем входные/выходные имена
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            
            logger.info(f"Входы: {self.input_names}")
            logger.info(f"Выходы: {self.output_names}")
            logger.info("ONNX модель загружена")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки ONNX: {e}")
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
            logger.error(f"Ошибка загрузки токенизатора: {e}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.1,
        top_p: float = 0.9,
        **kwargs
    ) -> Optional[str]:
        """Генерирует текст через ONNX"""
        if self.session is None or self.tokenizer is None:
            logger.error("Модель или токенизатор не загружены")
            return None
        
        try:
            start = time.time()
            
            # Токенизация
            input_ids = self.tokenizer.encode(prompt, return_tensors="np")
            
            # Генерация токен за токеном
            generated = input_ids.tolist()[0]
            
            for _ in range(max_new_tokens):
                # Подготовка входа
                feed = {self.input_names[0]: input_ids}
                
                # Предсказание
                outputs = self.session.run(self.output_names, feed)
                logits = outputs[0]
                
                # Получаем следующий токен
                next_token_logits = logits[0, -1, :]
                
                # Применяем temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Top-p sampling
                if top_p < 1.0:
                    sorted_indices = np.argsort(next_token_logits)[::-1]
                    sorted_probs = np.exp(next_token_logits[sorted_indices])
                    sorted_probs = sorted_probs / sorted_probs.sum()
                    
                    cumsum = np.cumsum(sorted_probs)
                    cutoff = sorted_probs[cumsum > top_p][0]
                    
                    # Обрезаем
                    next_token_logits[sorted_indices[cumsum > top_p]] = -np.inf
                
                # Greedy выбор
                next_token = np.argmax(next_token_logits)
                
                # Добавляем
                generated.append(int(next_token))
                input_ids = np.array([generated])
                
                # Остановка по EOS
                if next_token == self.tokenizer.eos_token_id:
                    break
            
            # Декодирование
            response = self.tokenizer.decode(generated, skip_special_tokens=True)
            
            elapsed = time.time() - start
            tokens_generated = len(generated) - len(input_ids[0])
            speed = tokens_generated / elapsed if elapsed > 0 else 0
            
            logger.info(f"Сгенерировано {tokens_generated} токенов за {elapsed:.1f}s ({speed:.2f} tok/s)")
            
            # Убираем промпт
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Ошибка генерации ONNX: {e}")
            return None
    
    def get_status(self) -> Dict:
        """Статус"""
        return {
            "initialized": self.session is not None,
            "model_path": self.model_path,
            "input_names": self.input_names,
            "output_names": self.output_names
        }


class OnnxConverter:
    """
    Конвертер модели Transformers в ONNX.
    """
    
    def __init__(self, model_path: str, onnx_path: str):
        self.model_path = model_path
        self.onnx_path = onnx_path
        self.model = None
        self.tokenizer = None
        
    def convert(
        self,
        max_seq_length: int = 512,
        dynamic_axes: bool = True
    ) -> bool:
        """
        Конвертирует модель в ONNX.
        """
        if not HAS_ONNX:
            logger.error("ONNX недоступен")
            return False
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Загрузка модели: {self.model_path}")
            
            # Загружаем
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # ONNX лучше с fp32
                device_map="cpu"
            )
            
            self.model.eval()
            
            # Создаём фиктивный вход
            dummy_input = self.tokenizer(
                "test",
                return_tensors="pt"
            )
            
            input_ids = dummy_input["input_ids"]
            attention_mask = dummy_input["attention_mask"]
            
            logger.info("Экспорт в ONNX...")
            
            # Экспорт через torch.onnx
            torch.onnx.export(
                self.model,
                (input_ids, attention_mask),
                self.onnx_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch", 1: "sequence"}
                } if dynamic_axes else None,
                opset_version=14,
                do_constant_folding=True,
                export_params=True
            )
            
            logger.info(f"ONNX модель сохранена: {self.onnx_path}")
            
            # Сохраняем токенизатор
            tokenizer_json = os.path.join(
                os.path.dirname(self.onnx_path),
                "tokenizer.json"
            )
            self.tokenizer.save_pretrained(os.path.dirname(self.onnx_path))
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка конвертации: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def quantize(self, quantized_path: Optional[str] = None) -> bool:
        """
        Квантизует ONNX модель.
        """
        if not HAS_ONNX:
            return False
        
        if quantized_path is None:
            quantized_path = self.onnx_path.replace(".onnx", "_quant.onnx")
        
        try:
            from onnxruntime.quantization import quantize_dynamic
            
            logger.info(f"Квантизация модели...")
            
            quantize_dynamic(
                self.onnx_path,
                quantized_path,
                weight_type=ort.QuantType.QUInt8
            )
            
            logger.info(f"Квантизованная модель: {quantized_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка квантизации: {e}")
            return False


# ============================================================================
# Тесты
# ============================================================================

def test_onnx_conversion():
    """Тест конвертации в ONNX"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(project_root, "mlearning", "eva_models", "qwen3.5-0.8b")
    onnx_path = os.path.join(project_root, "models", "qwen3.5-0.8b.onnx")
    
    logger.info(f"Модель: {model_path}")
    logger.info(f"ONNX выход: {onnx_path}")
    
    # Создаём директорию
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    # Конвертация
    converter = OnnxConverter(model_path, onnx_path)
    
    logger.info("Начало конвертации...")
    success = converter.convert(max_seq_length=256)
    
    if success:
        logger.info("Конвертация завершена!")
        
        # Тест инференса
        logger.info("\nТест инференса...")
        
        generator = OnnxRuntimeGenerator(onnx_path)
        
        if generator.load():
            generator.load_tokenizer(model_path)
            
            # Генерация
            response = generator.generate("Привет!", max_new_tokens=10)
            logger.info(f"Ответ: {response}")
    else:
        logger.error("Конвертация не удалась")


def test_existing_onnx():
    """Тест существующей ONNX модели"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    onnx_path = os.path.join(project_root, "models", "qwen3.5-0.8b.onnx")
    model_path = os.path.join(project_root, "mlearning", "eva_models", "qwen3.5-0.8b")
    
    if not os.path.exists(onnx_path):
        logger.error(f"ONNX модель не найдена: {onnx_path}")
        logger.info("Сначала нужно сконвертировать модель")
        return
    
    logger.info(f"Загрузка ONNX: {onnx_path}")
    
    generator = OnnxRuntimeGenerator(onnx_path, intra_threads=8)
    
    if generator.load():
        generator.load_tokenizer(model_path)
        
        # Генерация
        logger.info("Генерация...")
        start = time.time()
        response = generator.generate("Привет! Как дела?", max_new_tokens=20)
        elapsed = time.time() - start
        
        logger.info(f"Ответ: {response[:200] if response else 'None'}")
        logger.info(f"Время: {elapsed:.1f}s")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Тест конвертации
    test_onnx_conversion()
    
    # Или тест существующей
    # test_existing_onnx()