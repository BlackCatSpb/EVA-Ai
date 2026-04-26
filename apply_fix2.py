filepath = r'C:\Users\black\OneDrive\Desktop\EVA-Ai\eva_ai\core\fcp_pipeline.py'

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_lines.append(line)
    if 'chat_prompt = self._build_prompt(prompt, enable_thinking)' in line:
        indent = '        '
        new_lines.append(indent + 'if enable_injection:\n')
        new_lines.append(indent + '    prompt = self._enrich_prompt_with_graph(prompt)\n')

with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('Done')