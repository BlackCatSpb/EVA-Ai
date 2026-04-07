"""
ONNX Export через optimum (официальный инструмент HuggingFace).
"""
import os
import sys
import subprocess
import logging

logger = logging.getLogger("eva_ai.mlearning.hot_deployment.onnx_export")

def export_with_optimum(
    model_path: str,
    output_dir: str,
    task: str = "text-generation"
) -> bool:
    """
    Экспортирует модель в ONNX через optimum.
    """
    try:
        logger.info(f"Экспорт модели {model_path} в {output_dir}")
        
        # Команда optimum-cli
        cmd = [
            sys.executable, "-m", "optimum.exporters.onnx",
            model_path,
            "--task", task,
            "--output", output_dir,
            "--batch-size", "1",
            "--sequence-length", "256"
        ]
        
        logger.info(f"Команда: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode == 0:
            logger.info("Экспорт успешен!")
            logger.info(result.stdout)
            return True
        else:
            logger.error(f"Ошибка экспорта: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return False


def export_with_cli():
    """Экспорт через командную строку"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(project_root, "mlearning", "eva_models", "qwen3.5-0.8b")
    output_dir = os.path.join(project_root, "models", "qwen3.5-0.8b-onnx")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Model: {model_path}")
    logger.info(f"Output: {output_dir}")
    
    # Пробуем через torch.onnx с отключенным кэшированием
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info("Загрузка модели...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Отключаем use_cache для экспорта
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        model.eval()
        
        # Фиктивный вход
        dummy_input = tokenizer("test", return_tensors="pt")
        input_ids = dummy_input["input_ids"]
        attention_mask = dummy_input["attention_mask"]
        
        onnx_path = os.path.join(output_dir, "model.onnx")
        
        logger.info("Экспорт в ONNX...")
        
        # Экспорт без использования кэша
        with torch.no_grad():
            torch.onnx.export(
                model,
                (input_ids, attention_mask),
                onnx_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch", 1: "sequence", 2: "vocab"}
                },
                opset_version=17,
                do_constant_folding=True,
                export_params=True
            )
        
        logger.info(f"ONNX сохранён: {onnx_path}")
        
        # Сохраняем токенизатор
        tokenizer.save_pretrained(output_dir)
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    export_with_cli()