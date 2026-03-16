# Исправление lm_head.weight во втором вхождении с уникальным контекстом
import re

with open('cogniflex/mlearning/fractal_model_manager.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Ищем уникальный паттерн для второго вхождения (с resid_pdrop=0.1)
pattern = r'(resid_pdrop=0\.1,\s+embd_pdrop=0\.1,\s+attn_pdrop=0\.1,\s+# Важно: устанавливаем правильные параметры для bias\s+use_cache=True,\s+\)\s+\s+# Создаем модель\s+self\.model = GPT2LMHeadModel\(self\.config\)\s+)(\s+# Загружаем веса с strict=False для пропуска отсутствующих bias\s+missing_keys, unexpected_keys = self\.model\.load_state_dict\(self\.state_dict, strict=False\))'

replacement = r'\1# Специальная обработка для lm_head.weight\n                    if "transformer.wte.weight" in self.state_dict:\n                        logger.info("Используем общие веса для lm_head и wte")\n                        # Сначала загружаем без lm_head\n                        temp_state = {k: v for k, v in self.state_dict.items() if k != "lm_head.weight"}\n                        temp_missing, temp_unexpected = self.model.load_state_dict(temp_state, strict=False)\n                        # Затем устанавливаем lm_head.weight равным wte.weight\n                        self.model.lm_head.weight = self.model.transformer.wte.weight\n                        missing_keys = [k for k in temp_missing if k != "lm_head.weight"]\n                        unexpected_keys = temp_unexpected\n                    else:\n                        # Обычная загрузка если нет wte\n                        missing_keys, unexpected_keys = self.model.load_state_dict(self.state_dict, strict=False)\n\n                    \2'

new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

if new_content != content:
    with open('cogniflex/mlearning/fractal_model_manager.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Исправление применено ко второму вхождению")
else:
    print("Паттерн не найден, пробуем более простой вариант")
    
    # Пробуем более простой паттерн
    simple_pattern = r'(\s+# Создаем модель\s+self\.model = GPT2LMHeadModel\(self\.config\)\s+)(\s+# Загружаем веса с strict=False)'
    simple_replacement = r'\1# Специальная обработка для lm_head.weight\n                    if "transformer.wte.weight" in self.state_dict:\n                        logger.info("Используем общие веса для lm_head и wte")\n                        # Сначала загружаем без lm_head\n                        temp_state = {k: v for k, v in self.state_dict.items() if k != "lm_head.weight"}\n                        temp_missing, temp_unexpected = self.model.load_state_dict(temp_state, strict=False)\n                        # Затем устанавливаем lm_head.weight равным wte.weight\n                        self.model.lm_head.weight = self.model.transformer.wte.weight\n                        missing_keys = [k for k in temp_missing if k != "lm_head.weight"]\n                        unexpected_keys = temp_unexpected\n                    else:\n                        # Обычная загрузка если нет wte\n                        missing_keys, unexpected_keys = self.model.load_state_dict(self.state_dict, strict=False)\n\n                    \2'
    
    # Находим все вхождения
    matches = list(re.finditer(simple_pattern, content))
    if len(matches) >= 2:
        # Заменяем только второе вхождение
        second_match = matches[1]
        new_content = content[:second_match.start()] + simple_replacement + content[second_match.end():]
        
        with open('cogniflex/mlearning/fractal_model_manager.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("Простое исправление применено ко второму вхождению")
    else:
        print(f"Найдено только {len(matches)} вхождений простого паттерна")
