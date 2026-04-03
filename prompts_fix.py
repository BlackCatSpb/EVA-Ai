import sys

filepath = r'C:\Users\black\OneDrive\Desktop\CogniFlex\eva\core\recursive_model_pipeline.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix Model A system prompt - stronger Russian enforcement
old_a = '''        messages = [
            {"role": "system", "content": "Ты - ЕВА, искусственный интеллект женского рода. Отвечай ТОЛЬКО на русском языке. Отвечай по существу, без воды. Не используй китайские, английские или другие иностранные слова. Не повторяй вопрос в ответе."},
            {"role": "user", "content": query}
        ]
        
        logger.info(f"Model A query: {query[:100]}...")'''

new_a = '''        messages = [
            {"role": "system", "content": "Ты - ЕВА, русскоязычный ИИ. ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ. Никаких китайских, английских или иных иностранных слов. Только русский."},
            {"role": "user", "content": f"[ОТВЕЧАЙ НА РУССКОМ] {query}"}
        ]
        
        logger.info(f"Model A query: {query[:100]}...")'''

if old_a in content:
    content = content.replace(old_a, new_a)
    print('Fixed Model A prompt')
else:
    print('ERROR: Model A prompt not found')

# Fix Model B system prompt
old_b = '''        messages = [
            {"role": "system", "content": "Ты - ЕВА, искусственный интеллект женского рода. Отвечай ТОЛЬКО на русском языке. Развей мысль подробнее, добавь детали и объяснения. Не используй китайские, английские или другие иностранные слова. Не повторяй предыдущий текст."},
            {"role": "user", "content": f"Вот что я думаю по этому вопросу: {truncated_previous}\\n\\nРазвей эту мысль подробнее, добавь детали и объяснения."}
        ]'''

new_b = '''        messages = [
            {"role": "system", "content": "Ты - ЕВА, русскоязычный ИИ. ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ. Никаких китайских, английских или иных иностранных слов. Развей мысль подробнее."},
            {"role": "user", "content": f"[НА РУССКОМ] Вот что я думаю: {truncated_previous}\\n\\nРазвей эту мысль подробнее на русском."}
        ]'''

if old_b in content:
    content = content.replace(old_b, new_b)
    print('Fixed Model B prompt')
else:
    print('ERROR: Model B prompt not found')

# Fix Model C system prompt
old_c = '''        messages = [
            {"role": "system", "content": "Ты - ЕВА, помощница-программист. Пиши чистый, рабочий код с комментариями на русском. Отвечай только кодом, без лишних объяснений."},
            {"role": "user", "content": f"Контекст: {context}\\n\\nЗапрос: {query}\\n\\nНапиши код для этого запроса."}
        ]'''

new_c = '''        messages = [
            {"role": "system", "content": "Ты - ЕВА, помощница-программист. Пиши чистый, рабочий код. Комментарии на русском языке."},
            {"role": "user", "content": f"[НА РУССКОМ] Контекст: {context}\\n\\nЗапрос: {query}\\n\\nНапиши код."}
        ]'''

if old_c in content:
    content = content.replace(old_c, new_c)
    print('Fixed Model C prompt')
else:
    print('ERROR: Model C prompt not found')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print('All prompts updated')