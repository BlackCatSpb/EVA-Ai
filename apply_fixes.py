import sys
import os

path = r'C:\Users\black\OneDrive\Desktop\EVA-Ai\eva_ai\core\fcp_pipeline.py'

with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix indentation line 146
lines[145] = '            self.pipeline = ov_genai.LLMPipeline(\n'

# Add **kwargs after use_lora
for i, line in enumerate(lines):
    if 'use_lora: bool = True' in line:
        if i+1 >= len(lines) or 'return_metadata' in lines[i+1]:
            lines.insert(i+1, '        **kwargs\n')
        else:
            lines.insert(i+1, '        **kwargs\n')
        break

# Fix generate call to _generate
for i, line in enumerate(lines):
    if 'response = self._generate(chat_prompt, max_new_tokens)' in line:
        lines[i] = '        response = self._generate(chat_prompt, max_new_tokens, **kwargs)\n'
        break

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Fixed fcp_pipeline.py')

# Verify compile
import py_compile
try:
    py_compile.compile(path, doraise=True)
    print('Compilation OK')
except Exception as e:
    print(f'Compile error: {e}')
    sys.exit(1)