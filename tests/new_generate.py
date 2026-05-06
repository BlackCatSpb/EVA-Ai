import re
import queue
import threading
import time

def generate_streaming_replacement():
    return '''
    def generate_streaming(self, prompt, max_new_tokens=1024, enable_thinking=True, callback=None, conversation_history=None, **kwargs):
        """Streaming с парсингом тегов размышления в процессе генерации"""
        if not self.pipeline:
            yield {"type": "error", "text": "[No pipeline]"}
            return
        
        chat_prompt = self._build_prompt(prompt, enable_thinking=enable_thinking, conversation_history=conversation_history)
        
        yield {"type": "start", "timestamp": time.time()}
        
        try:
            import queue
            import threading
            import re
            
            event_queue = queue.Queue()
            
            # Состояние парсера
            buffer = ""
            in_thinking = False
            thinking_text_accum = ""
            
            def token_callback(token_text: str):
                nonlocal buffer, in_thinking, thinking_text_accum
                buffer += token_text
                
                if not in_thinking:
                    # Ищем начало размышления
                    start_idx = buffer.find("<think>")
                    if start_idx != -1:
                        # Текст до <think> - это основной ответ
                        if start_idx > 0:
                            event_queue.put({"type": "chunk", "text": buffer[:start_idx]})
                        in_thinking = True
                        buffer = buffer[start_idx + len("<think>"):]
                        event_queue.put({"type": "reasoning_start"})
                        # Сбрасываем накопитель рассуждений
                        thinking_text_accum = ""
                    else:
                        # Тега нет, весь буфер - это основной ответ
                        if buffer:
                            event_queue.put({"type": "chunk", "text": buffer})
                            buffer = ""
                else:
                    # Внутри размышления, ищем конец
                    end_idx = buffer.find("</think>")
                    if end_idx != -1:
                        # Текст до </think> - это рассуждение
                        thinking_text_accum += buffer[:end_idx]
                        if thinking_text_accum.strip():
                            event_queue.put({"type": "reasoning_text", "text": thinking_text_accum})
                        thinking_text_accum = ""
                        in_thinking = False
                        buffer = buffer[end_idx + len("</think>"):]
                        event_queue.put({"type": "reasoning_end"})
                        # После конца размышления, возможно, сразу пойдет ответ
                        # Буфер может содержать начало ответа, обработаем в следующей итерации
                    else:
                        # Тег конца не найден, весь буфер - это рассуждение
                        thinking_text_accum += buffer
                        buffer = ""
                        # Отправляем накопленное рассуждение (потоково)
                        if thinking_text_accum.strip():
                            event_queue.put({"type": "reasoning_text", "text": thinking_text_accum})
                            thinking_text_accum = ""
                
                return False  # Продолжить генерацию
            
            def generate():
                """Генерация в отдельном потоке"""
                try:
                    gen_cfg = ov_genai.GenerationConfig()
                    gen_cfg.max_new_tokens = max_new_tokens
                    gen_cfg.temperature = 0.2
                    gen_cfg.top_p = 0.9
                    gen_cfg.top_k = 40
                    gen_cfg.repetition_penalty = 1.1
                    gen_cfg.no_repeat_ngram_size = 5
                    gen_cfg.do_sample = True
                    
                    # Запускаем генерацию с callback
                    self.pipeline.generate(chat_prompt, generation_config=gen_cfg, streamer=token_callback)
                    
                    # После завершения обрабатываем остаток буфера
                    if buffer:
                        if in_thinking:
                            thinking_text_accum += buffer
                            if thinking_text_accum.strip():
                                event_queue.put({"type": "reasoning_text", "text": thinking_text_accum})
                            event_queue.put({"type": "reasoning_end"})
                        else:
                            if buffer.strip():
                                event_queue.put({"type": "chunk", "text": buffer})
                    
                except Exception as e:
                    event_queue.put({"type": "error", "text": str(e)})
                finally:
                    event_queue.put({"type": "done", "timestamp": time.time()})
            
            # Запускаем генерацию в отдельном потоке
            gen_thread = threading.Thread(target=generate)
            gen_thread.start()
            
            # Читаем события из очереди и отправляем клиенту
            while True:
                try:
                    event = event_queue.get(timeout=0.1)
                    yield event
                    if event['type'] == 'done':
                        break
                except queue.Empty:
                    if not gen_thread.is_alive():
                        break
            
            gen_thread.join()
            
        except Exception as e:
            yield {"type": "error", "text": str(e)}
'''
