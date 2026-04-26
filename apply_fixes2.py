import sys
import os

path = r'C:\Users\black\OneDrive\Desktop\EVA-Ai\eva_ai\core\fcp_pipeline.py'
print(f'Reading {path}...')

with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f'File has {len(lines)} lines')
print(f'Line 145 (146): {repr(lines[145][:80])}')

# Fix indentation line 146
lines[145] = '            self.pipeline = ov_genai.LLMPipeline(\n'
print('Fixed indentation')

# Add **kwargs after use_lora
for i, line in enumerate(lines):
    if 'use_lora: bool = True' in line:
        print(f'Found use_lora at line {i+1}')
        insert_pos = i + 1
        if i+1 < len(lines) and 'return_metadata' in lines[i+1]:
            insert_pos = i + 2
        lines.insert(insert_pos, '        **kwargs\n')
        print(f'Inserted **kwargs at line {insert_pos+1}')
        break

# Fix generate call to _generate
for i, line in enumerate(lines):
    if 'response = self._generate(chat_prompt, max_new_tokens)' in line:
        lines[i] = '        response = self._generate(chat_prompt, max_new_tokens, **kwargs)\n'
        print(f'Fixed generate call at line {i+1}')
        break

print(f'Writing {len(lines)} lines back...')
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
    import traceback
    traceback.print_exc()
    sys.exit(1)