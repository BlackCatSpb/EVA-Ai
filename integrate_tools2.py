import re

filepath = r'C:\Users\black\OneDrive\Desktop\EVA-Ai\eva_ai\core\fcp_pipeline.py'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add import for ToolOrchestrator after other optional imports
# Find the section with optional imports (around line 37)
imports_end = content.find('HAS_TRANSFORMERS = False')
if imports_end == -1:
    print("Could not find imports section")
    exit()

# Insert after the HAS_TRANSFORMERS block
insert_point = content.find('\n\n', imports_end) + 2
if insert_point < 2:
    insert_point = len(content)

# Add the ToolOrchestrator import
tool_import = '''\ntry:\n    from eva_ai.fcp_migration.fcp_tools.orchestrator import ToolOrchestrator\n    HAS_FCP_TOOLS = True\nexcept ImportError:\n    HAS_FCP_TOOLS = False\n'''

content = content[:insert_point] + tool_import + content[insert_point:]

# 2. Initialize tool_orchestrator in __init__
# Find the end of __init__ (after the last assignment)
init_end = content.find('        print(f"[FCP] FCPPipelineV15 created: model={model_path}")')
if init_end == -1:
    print("Could not find end of __init__")
    exit()

# Find the end of that line (end of line)
line_end = content.find('\n', init_end)
if line_end == -1:
    line_end = len(content)

# Insert after that line
insert_point2 = line_end + 1
init_code = '''\n        # Initialize tool orchestrator\n        self.tool_orchestrator = ToolOrchestrator() if HAS_FCP_TOOLS else None\n'''
content = content[:insert_point2] + init_code + content[insert_point2:]

# 3. Modify generate method to add use_tools parameter and apply orchestrator
# Find the generate method signature
gen_sig = content.find('    def generate(')
if gen_sig == -1:
    print("Could not find generate method signature")
    exit()

# Find the **kwargs part
kwargs_pos = content.find('**kwargs', gen_sig)
if kwargs_pos == -1:
    print("Could not find **kwargs in generate method")
    exit()

# Find the use_lora parameter line
use_lora_pos = content.find('use_lora: bool = True', gen_sig)
if use_lora_pos == -1:
    print("Could not find use_lora parameter")
    exit()

# Find the end of that line (look for comma)
line_end2 = content.find('\n', use_lora_pos)
if line_end2 == -1:
    line_end2 = len(content)

# Check if there's a comma at the end of the line
if line_end2 < len(content) and content[line_end2:line_end2+1] == ',':
    # There is a comma, we can insert after the comma and newline
    insert_point3 = line_end2 + 1
else:
    # No comma, we need to add one
    insert_point3 = line_end2
    # We'll add a comma and newline later

# Now insert the new parameter
new_param = 'use_tools: bool = True, '
content = content[:insert_point3] + new_param + content[insert_point3:]

# 4. Add the tool orchestration logic in the generate method body
# Find where we return the response (before the return statement)
# We'll look for the line: "return response"
return_line = content.find('return response', gen_sig)
if return_line == -1:
    print("Could not find return statement in generate method")
    exit()

# We want to insert before the return statement, but after the response is generated
# Find the beginning of the line with return response
line_start = content.rfind('\n', 0, return_line) + 1
# Insert before that line
# We'll add the tool orchestration code
tool_code = '''\n        # Apply tool orchestration if enabled\n        if use_tools and self.tool_orchestrator is not None:\n            response = self.tool_orchestrator.process_response(response)\n'''
content = content[:line_start] + tool_code + content[line_start:]

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Successfully integrated ToolOrchestrator into FCPPipelineV15")