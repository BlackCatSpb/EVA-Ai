import re

path = 'C:/Users/black/OneDrive/Desktop/EVA-Ai/eva_ai/core/fcp_pipeline.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Read new method from new_generate.py
import os
new_method_path = 'C:/Users/black/OneDrive/Desktop/EVA-Ai/new_generate.py'
with open(new_method_path, 'r', encoding='utf-8') as f:
    new_method = f.read()

# Extract the actual code from the triple-quoted string
# The file contains a function returning a string with triple quotes.
# We'll just take the content between the first ''' and last '''
import re
match = re.search(r"'''(.*?)'''", new_method, re.DOTALL)
if match:
    new_method_code = match.group(1)
else:
    # Maybe the file is just the code
    new_method_code = new_method

print("New method code length:", len(new_method_code))
print("First 100 chars:", new_method_code[:100])

# Now replace in original content
# Find start of generate_streaming method
start_marker = '    def generate_streaming('
end_marker = '    def generate('  # next method

start_pos = content.find(start_marker)
if start_pos == -1:
    print("Start marker not found")
    exit()

# Find the end marker after start_pos
end_pos = content.find(end_marker, start_pos + len(start_marker))
if end_pos == -1:
    # maybe end of file
    end_pos = len(content)

print(f"Replacing from {start_pos} to {end_pos}")

new_content = content[:start_pos] + new_method_code + '\n' + content[end_pos:]

with open(path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Replacement done")
