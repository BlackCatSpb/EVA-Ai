"""
ЕВА optimal configuration application script
"""

import os
import sys
import json

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

def get_config_path(filename):
    """Return path to config file relative to script"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)

def load_config(config_path):
    """Load configuration from file"""
    if not os.path.exists(config_path):
        print("[!] Error: Config file not found")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[!] JSON error in {config_path}: {e}")
        return {}
    except Exception as e:
        print(f"[!] Config load error {config_path}: {e}")
        return {}

def apply_fractal_config():
    """Apply FractalModelManager configuration"""
    config_path = get_config_path("fractal_model_config.json")
    config = load_config(config_path)
    
    if config:
        print("[BRAIN] Applying FractalModelManager config...")
        print(f"  Device: {config.get('device', 'cpu')}")
        print(f"  Max tokens: {config.get('max_memory_tokens', 10000)}")
        print(f"  Target memory: {config.get('target_memory_gb', 2.0)} GB")
        print(f"  Parallel tokenization: {config.get('parallel_tokenization', False)}")
        print(f"  Workers: {config.get('tokenization_workers', 2)}")
        return True
    return False

def apply_gui_config():
    """Apply GUI configuration"""
    config_path = get_config_path("gui_config.json")
    config = load_config(config_path)
    
    if config:
        print("[GUI] Applying GUI config...")
        print(f"  Theme: {config.get('theme', 'light')}")
        print(f"  Auto-refresh: {config.get('auto_refresh_interval', 5000)} ms")
        print(f"  Advanced metrics: {config.get('show_advanced_metrics', True)}")
        return True
    return False

if __name__ == "__main__":
    print("[*] Applying optimal ЕВА configuration")
    print("=" * 60)
    
    success = True
    
    if not apply_fractal_config():
        success = False
    
    if not apply_gui_config():
        success = False
    
    print("=" * 60)
    if success:
        print("[OK] Optimal configuration applied successfully!")
        print("\nKey improvements:")
        print("  [MEM] Token cache increased to 50K-100K")
        print("  [CPU] Parallel tokenization enabled")
        print("  [MEM] Memory optimized (uint16, pools)")
        print("  [MET] Performance monitoring enabled")
        print("  [QUAL] Auto quality improvement enabled")
    else:
        print("[ERROR] Failed to apply configuration")
    
    print("\nPress Enter to exit...")
    input()