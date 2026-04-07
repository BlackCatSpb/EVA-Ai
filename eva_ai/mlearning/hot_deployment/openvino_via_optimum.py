"""
OpenVINO через optimum-intel (официальный метод).
"""
import os
import sys
import logging

logger = logging.getLogger("eva_ai.mlearning.hot_deployment.openvino_via_optimum")

def convert_via_optimum():
    """Конвертирует через optimum-intel"""
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
    
    try:
        from optimum.intel.openvino import OVModelForCausalLM
        
        logger.info("Загрузка модели...")
        
        # Загружаем и конвертируем
        model = OVModelForCausalLM.from_pretrained(
            model_path,
            export=True,
            ov_config={
                "PERFORMANCE_HINT": "LATENCY",
                "NUM_STREAMS": "1",
                "INFERENCE_NUM_THREADS": "8"
            }
        )
        
        logger.info("Сохранение...")
        model.save_pretrained(output_dir)
        
        logger.info(f"Сохранено: {output_dir}")
        return output_dir
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_ov_model():
    """Тестирует уже сконвертированную модель"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(project_root, "mlearning", "eva_models", "qwen3.5-0.8b")
    ov_dir = os.path.join(project_root, "models", "qwen3.5-0.8b-openvino")
    
    # Проверяем наличие модели
    if not os.path.exists(ov_dir):
        logger.error(f"Директория не существует: {ov_dir}")
        return
    
    xml_files = [f for f in os.listdir(ov_dir) if f.endswith('.xml')]
    if not xml_files:
        logger.error("IR модель не найдена")
        return
    
    logger.info(f"Тестируем OpenVINO модель...")
    
    try:
        import openvino as ov
        from transformers import AutoTokenizer
        
        # Загружаем
        core = ov.Core()
        model = core.read_model(os.path.join(ov_dir, xml_files[0]))
        
        # Компилируем
        compiled = core.compile_model(model, "CPU", {
            "INFERENCE_NUM_THREADS": "8"
        })
        
        # Токенизатор
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Модель загружена!")
        
        # Тест генерации
        import time
        import numpy as np
        
        prompt = "Привет!"
        input_ids = tokenizer.encode(prompt, return_tensors="np")
        
        logger.info("Генерация...")
        start = time.time()
        
        generated = input_ids.tolist()[0]
        
        for step in range(20):
            input_tensor = ov.Tensor.from_numpy(input_ids.astype(np.int64))
            
            results = compiled([input_tensor])
            logits = results[0]
            next_token = int(np.argmax(logits[0, -1, :]))
            
            generated.append(next_token)
            input_ids = np.array([generated])
            
            if next_token == tokenizer.eos_token_id:
                break
        
        elapsed = time.time() - start
        
        response = tokenizer.decode(generated, skip_special_tokens=True)
        
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        tokens_generated = len(generated) - len(input_ids[0])
        speed = tokens_generated / elapsed if elapsed > 0 else 0
        
        logger.info(f"Ответ: {response[:100]}")
        logger.info(f"Токенов: {tokens_generated}, время: {elapsed:.1f}s, скорость: {speed:.2f} tok/s")
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Пробуем конвертировать
    result = convert_via_optimum()
    
    if result:
        # Тестируем
        test_ov_model()