import re
f=open(r'C:\Users\black\OneDrive\Desktop\EVA-Ai\eva_ai\core\fcp_pipeline.py','r',encoding='utf-8')
c=f.read()

# Find _build_prompt
match = re.search(r'def _build_prompt\(self.*?\):(.*?)(?=\n    def |\Z)', c, re.DOTALL)
if match:
    print('_build_prompt:')
    print(match.group(1)[:500])
