#!/usr/bin/env python3
import sys
import os

filepath = r'C:\Users\black\OneDrive\Desktop\EVA-Ai\eva_ai\core\fcp_pipeline.py'

with open(filepath, 'rb') as f:
    content = f.read().decode('utf-8')

old = '        chat_prompt = self._build_prompt(prompt, enable_thinking)'
new = '        if enable_injection:\n            prompt = self._enrich_prompt_with_graph(prompt)\n        chat_prompt = self._build_prompt(prompt, enable_thinking)'

count = content.count(old)
print(f'Found {count} occurrences')
if count > 0:
    content = content.replace(old, new, 1)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print('Modified file')
else:
    print('Pattern not found')