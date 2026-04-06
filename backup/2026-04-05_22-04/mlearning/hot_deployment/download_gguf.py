"""
Скачивание GGUF модели напрямую.
Использует публичные ссылки без аутентификаии.
"""
import os
import sys
import urllib.request
import ssl
import logging

logger = logging.getLogger("eva.mlearning.hot_deployment.download_gguf")

def download_with_progress(url: str, output_path: str) -> bool:
    """Скачивает с отображением прогресса"""
    try:
        # Создаём SSL контекст без проверки
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        logger.info(f"Скачивание: {url}")
        
        # Получаем размер файла
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=30, context=ctx)
        
        total_size = int(response.headers.get('Content-Length', 0))
        logger.info(f"Размер: {total_size / (1024*1024):.1f} MB")
        
        # Скачиваем батчами
        downloaded = 0
        chunk_size = 64 * 1024  # 64KB
        
        with open(output_path, 'wb') as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                
                # Прогресс каждые 5MB
                if downloaded % (5 * 1024 * 1024) == 0:
                    percent = (downloaded / total_size * 100) if total_size > 0 else 0
                    logger.info(f"Скачано: {downloaded / (1024*1024):.1f} MB ({percent:.1f}%)")
        
        logger.info(f"Сохранено: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return False


def try_download_gguf():
    """Пробует скачать GGUF модель"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Список моделей для попытки (публичные, без авторизации)
    # Пробуем разные варианты
    
    candidates = [
        # Qwen 0.5B - меньше размером
        {
            "name": "qwen2-0.5b-instruct-q5_k_m.gguf",
            "url": "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0.5b-instruct-q5_k_m.gguf"
        },
        # TinyLlama
        {
            "name": "tinyllama-1.1b-chat-v1.0.q5_k_m.gguf", 
            "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.q5_k_m.gguf"
        },
        # Qwen 2.5 0.5B
        {
            "name": "qwen2.5-0.5b-instruct-q4_0.gguf",
            "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf"
        },
    ]
    
    for candidate in candidates:
        output_path = os.path.join(models_dir, candidate["name"])
        
        if os.path.exists(output_path):
            logger.info(f"Файл уже существует: {output_path}")
            return output_path
        
        logger.info(f"\nПопытка: {candidate['name']}")
        
        if download_with_progress(candidate["url"], output_path):
            logger.info(f"Успешно скачано!")
            return output_path
        else:
            logger.warning(f"Не удалось скачать, пробуем следующую...")
    
    return None


def try_civitai():
    """Пробуем скачать с CivitAI (не требует аутентификации для некоторых моделей)"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(project_root, "models")
    
    # CivitAI API для получения прямых ссылок
    # Это пример - нужно использовать реальный API key
    
    logger.info("CivitAI требует API ключ, пропускаем...")
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("=== Скачивание GGUF модели ===")
    
    result = try_download_gguf()
    
    if result:
        logger.info(f"\nМодель скачана: {result}")
        
        # Тест
        logger.info("\nТест генерации...")
        
        from llama_cpp import Llama
        
        ll = Llama(
            model_path=result,
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=0
        )
        
        import time
        start = time.time()
        
        response = ll.create_chat_completion(
            messages=[{"role": "user", "content": "Привет! Как дела?"}],
            max_tokens=20,
            temperature=0.1
        )
        
        elapsed = time.time() - start
        
        text = response['choices'][0]['message']['content']
        logger.info(f"Ответ: {text[:200]}")
        logger.info(f"Время: {elapsed:.1f}s")
        
    else:
        logger.error("\nНе удалось скачать GGUF модель")