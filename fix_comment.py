# Удаление лишнего комментария во втором вхождении
import re

with open('cogniflex/mlearning/fractal_model_manager.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Разделяем содержимое на строки для точного редактирования
lines = content.split('\n')

# Находим второе вхождение комментария и удаляем его
count = 0
for i, line in enumerate(lines):
    if '# Загружаем веса с strict=False для пропуска отсутствующих bias' in line:
        count += 1
        if count == 2:
            # Удаляем эту строку
            lines.pop(i)
            break

# Соединяем обратно
new_content = '\n'.join(lines)

with open('cogniflex/mlearning/fractal_model_manager.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Лишний комментарий во втором вхождении удален")
