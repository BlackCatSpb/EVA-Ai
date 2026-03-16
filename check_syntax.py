import os
import py_compile
import sys

def check_syntax(directory):
    errors = []
    success_count = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    py_compile.compile(filepath, doraise=True)
                    print(f"✓ {filepath}")
                    success_count += 1
                except py_compile.PyCompileError as e:
                    print(f"✗ {filepath}")
                    print(f"  Error: {e}")
                    errors.append((filepath, str(e)))
                except Exception as e:
                    print(f"✗ {filepath}")
                    print(f"  Unexpected error: {e}")
                    errors.append((filepath, str(e)))
    
    print(f"\n{'='*50}")
    print(f"Syntax check complete!")
    print(f"Successfully compiled: {success_count} files")
    print(f"Errors found: {len(errors)} files")
    
    if errors:
        print("\nFiles with errors:")
        for filepath, error in errors:
            print(f"  - {filepath}: {error}")
    
    return len(errors) == 0

if __name__ == "__main__":
    directory = r"C:\Users\black\.windsurf\worktrees\CogniFlex\CogniFlex-81c8d36b\cogniflex"
    success = check_syntax(directory)
    sys.exit(0 if success else 1)
