import sys, os
sys.path.insert(0, 'C:/Users/black/OneDrive/Desktop/CogniFlex')

from llama_cpp import Llama

# Test 0.5B model
model_05b = r'C:\Users\black\OneDrive\Desktop\CogniFlex\eva\memory\fractal_torch_storage\gguf_models\qwen2.5-0.5b-instruct-q4_0.gguf'
model_3b = r'C:\Users\black\OneDrive\Desktop\CogniFlex\eva\memory\fractal_torch_storage\gguf_models\qwen2.5-3b-instruct\qwen2.5-3b-instruct-q4_k_m.gguf'

print('=== Testing 0.5B model ===')
m = Llama(model_path=model_05b, chat_format='qwen', n_ctx=2048, n_threads=8, verbose=False)

tests = [
    'Привет!',
    'Что такое искусственный интеллект?',
    'Какие языки программирования самые популярные?',
]

for t in tests:
    print(f'\nQ: {t}')
    out = m.create_chat_completion(
        messages=[{'role': 'user', 'content': t}],
        max_tokens=128,
        temperature=0.3,
        repeat_penalty=1.2,
    )
    resp = out['choices'][0]['message']['content'].strip()
    chinese = sum(1 for c in resp if '\u4e00' <= c <= '\u9fff')
    korean = sum(1 for c in resp if '\uac00' <= c <= '\ud7a3')
    print(f'Response: {resp[:200]}')
    print(f'Chinese: {chinese}, Korean: {korean}')

print('\n\n=== Testing 3B model ===')
m = Llama(model_path=model_3b, chat_format='qwen', n_ctx=2048, n_threads=8, verbose=False)

for t in tests:
    print(f'\nQ: {t}')
    out = m.create_chat_completion(
        messages=[{'role': 'user', 'content': t}],
        max_tokens=128,
        temperature=0.3,
        repeat_penalty=1.2,
    )
    resp = out['choices'][0]['message']['content'].strip()
    chinese = sum(1 for c in resp if '\u4e00' <= c <= '\u9fff')
    korean = sum(1 for c in resp if '\uac00' <= c <= '\ud7a3')
    print(f'Response: {resp[:200]}')
    print(f'Chinese: {chinese}, Korean: {korean}')
