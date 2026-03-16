# Исправление синтаксической ошибки в fractal_model_manager.py
file_path = r"C:\Users\black\.windsurf\worktrees\CogniFlex\CogniFlex-81c8d36b\cogniflex\mlearning\fractal_model_manager.py"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Исправляем строки с неправильным экранированием
for i, line in enumerate(lines):
    if "critical_missing = [k for k in missing_keys if \\'weight\\'" in line:
        lines[i] = line.replace("\\'weight\\'", "'weight'").replace("\\'lm_head\\'", "'lm_head'")
    elif "other_missing = [k for k in missing_keys if \\'bias\\'" in line:
        lines[i] = line.replace("\\'bias\\'", "'bias'").replace("\\'weight\\'", "'weight'").replace("\\'lm_head\\'", "'lm_head'")

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Синтаксическая ошибка исправлена")
