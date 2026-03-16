# Удаление дублирующей строки загрузки state_dict
import re

with open('cogniflex/mlearning/fractal_model_manager.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Находим и удаляем дублирующую строку
pattern = r'(\s+# Загружаем веса с strict=False для пропуска отсутствующих bias\s+)missing_keys, unexpected_keys = self\.model\.load_state_dict\(self\.state_dict, strict=False\)\s+'

replacement = r'\1'

new_content = re.sub(pattern, replacement, content)

if new_content != content:
    with open('cogniflex/mlearning/fractal_model_manager.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Дублирующая строка удалена")
else:
    print("Дублирующая строка не найдена")
