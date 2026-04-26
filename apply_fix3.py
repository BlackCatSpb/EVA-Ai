import fileinput
import sys

filepath = r'C:\Users\black\OneDrive\Desktop\EVA-Ai\eva_ai\core\fcp_pipeline.py'

for line in fileinput.input(filepath, inplace=True, backup='.bak'):
    sys.stdout.write(line)
    if 'chat_prompt = self._build_prompt(prompt, enable_thinking)' in line:
        indent = '        '
        sys.stdout.write(indent + 'if enable_injection:\n')
        sys.stdout.write(indent + '    prompt = self._enrich_prompt_with_graph(prompt)\n')