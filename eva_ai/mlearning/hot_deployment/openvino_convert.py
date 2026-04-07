"""
OpenVINO конвертация через optimum-cli.
"""
import os
import sys
import subprocess
import logging

logger = logging.getLogger("eva_ai.mlearning.hot_deployment.openvino_convert")


def convert_with_cli():
    """Конвертирует через optimum-cli"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(project_root, "mlearning", "eva_models", "qwen3.5-0.8b")
    output_dir = os.path.join(project_root, "models", "qwen3.5-0.8b-openvino")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Модель: {model_path}")
    logger.info(f"Выход: {output_dir}")
    
    # Проверяем уже сконвертированное
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        if any(f.endswith('.xml') for f in files):
            logger.info("Модель уже сконвертирована")
            return output_dir
    
    # Пробуем через optimum-cli
    cmd = [
        sys.executable, "-m",
        "optimum.exporters.openvio",
        model_path,
        "--task", "text-generation",
        "--output", output_dir,
        "--weight-format", "fp16"
    ]
    
    logger.info(f"Команда: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode == 0:
            logger.info("Конвертация успешна!")
            return output_dir
        else:
            logger.error(f"Ошибка: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Исключение: {e}")
    
    # Альтернатива - пробуем напрямую через openvino
    logger.info("Пробуем альтернативный метод...")
    
    return None


def convert_direct():
    """Прямая конвертация через openvino"""
    import openvino as ov
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(project_root, "mlearning", "eva_models", "qwen3.5-0.8b")
    output_dir = os.path.join(project_root, "models", "qwen3.5-0.8b-openvino")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Загрузка модели...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="float32",
        device_map="cpu"
    )
    model.eval()
    
    # Пробуем использовать ov.convert_model
    logger.info("Конвертация через openvino...")
    
    try:
        # Создаём пример ввода
        input_ids = tokenizer("test", return_tensors="pt")
        
        # Пробуем конвертировать
        ov_model = ov.convert_model(model, example_inputs=(input_ids["input_ids"], input_ids["attention_mask"]))
        
        # Сохраняем
        xml_path = os.path.join(output_dir, "model.xml")
        ov.save_model(ov_model, xml_path)
        
        logger.info(f"Сохранено: {xml_path}")
        
        # Сохраняем токенизатор
        tokenizer.save_pretrained(output_dir)
        
        return output_dir
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        
        # Пробуем ещё проще - через trace
        logger.info("Пробуем через torch.jit.trace...")
        
        try:
            import torch
            
            # Создаём обёртку без cache
            class SimpleModel(torch.nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.model = base_model
                
                def forward(self, input_ids, attention_mask):
                    # Без use_cache
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                    return outputs.logits
            
            simple_model = SimpleModel(model)
            simple_model.eval()
            
            # Трейс
            input_ids = tokenizer("test", return_tensors="pt")["input_ids"]
            traced = torch.jit.trace(simple_model, (input_ids, torch.ones_like(input_ids)))
            
            # Сохраняем
            traced.save(os.path.join(output_dir, "model.pt"))
            
            logger.info("Сохранено через torch.jit")
            return output_dir
            
        except Exception as e2:
            logger.error(f"torch.jit тоже не работает: {e2}")
            return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    convert_direct()