import re

path = 'C:/Users/black/OneDrive/Desktop/EVA-Ai/eva_ai/core/fcp_pipeline.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find start of generate_streaming method
start_marker = '    def generate_streaming('
start_pos = content.find(start_marker)
if start_pos == -1:
    print("Start not found")
    exit(1)

# Find end of method: next method at same indentation level (4 spaces)
# We'll look for '\n    def ' after start_pos+1, but not inside the method.
# Simple: find the next '\n    def ' that is not part of the method body.
# Since method body is indented by 8 spaces, lines with 4 spaces are at method level.
# We'll search for pattern that matches newline + 4 spaces + 'def '
import re
pattern = r'\n    def '
matches = [m for m in re.finditer(pattern, content[start_pos+1:])]
if matches:
    # first match is the next method
    end_pos = start_pos + 1 + matches[0].start()
else:
    # maybe end of file
    end_pos = len(content)

print(f"Replacing from {start_pos} to {end_pos}")

# New method
new_method = '''    def generate_streaming(self, prompt, max_new_tokens=1024, enable_thinking=True, callback=None, conversation_history=None, **kwargs):
        """Streaming с парсингом тегов размышления в процессе генерации"""
        if not self.pipeline:
            yield {"type": "error", "text": "[No pipeline]"}
            return
        
        chat_prompt = self._build_prompt(prompt, enable_thinking=enable_thinking, conversation_history=conversation_history)
        
        yield {"type": "start", "timestamp": time.time()}
        
        try:
            import queue
            import threading
            
            event_queue = queue.Queue()
            
            # Состояние парсера
            buffer = ""
            in_thinking = False
            thinking_accum = ""
            
            def token_callback(token_text: str):
                nonlocal buffer, in_thinking, thinking_accum
                buffer += token_text
                
                while True:
                    if not in_thinking:
                        idx = buffer.find("<think>")
                        if idx != -1:
                            # Текст до <think> - это основной ответ
                            if idx > 0:
                                event_queue.put({"type": "chunk", "text": buffer[:idx]})
                            in_thinking = True
                            buffer = buffer[idx + len("<think>"):]
                            event_queue.put({"type": "reasoning_start"})
                        else:
                            # Тега нет, отправляем буфер как чанк
                            if buffer:
                                event_queue.put({"type": "chunk", "text": buffer})
                                buffer = ""
                            break
                    else:
                        idx = buffer.find("</think>")
                        if idx != -1:
                            # Текст до </think> - это рассуждение
                            thinking_accum += buffer[:idx]
                            if thinking_accum.strip():
                                event_queue.put({"type": "reasoning_text", "text": thinking_accum})
                            thinking_accum = ""
                            in_thinking = False
                            buffer = buffer[idx + len("</think>"):]
                            event_queue.put({"type": "reasoning_end"})
                        else:
                            # Тег конца не найден, накапливаем рассуждение
                            thinking_accum += buffer
                            buffer = ""
                            break
                return False
            
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
                    
                    self.pipeline.generate(chat_prompt, generation_config=gen_cfg, streamer=token_callback)
                    
                    # После завершения обрабатываем остаток буфера
                    if buffer:
                        if in_thinking:
                            thinking_accum += buffer
                            if thinking_accum.strip():
                                event_queue.put({"type": "reasoning_text", "text": thinking_accum})
                            event_queue.put({"type": "reasoning_end"})
                        else:
                            if buffer.strip():
                                event_queue.put({"type": "chunk", "text": buffer})
                except Exception as e:
                    event_queue.put({"type": "error", "text": str(e)})
                finally:
                    event_queue.put({"type": "done", "timestamp": time.time()})
            
            # Запускаем генерацию
            gen_thread = threading.Thread(target=generate)
            gen_thread.start()
            
            # Читаем события из очереди
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

# Replace
new_content = content[:start_pos] + new_method + content[end_pos:]

with open(path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Replacement done")
