import sys
import os
import logging

# Setup file logging immediately
log_file = os.path.join(os.path.dirname(__file__), 'import_debug.log')
if os.path.exists(log_file):
    os.remove(log_file)
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s - %(message)s')

def log_and_print(message):
    print(message, flush=True)
    logging.debug(message)

log_and_print("--- Starting Import Debug ---")

try:
    log_and_print("Importing standard libraries...")
    import time
    import threading
    import queue
    import datetime
    import random
    log_and_print("✅ Standard libraries imported.")

    log_and_print("Importing psutil...")
    import psutil
    log_and_print("✅ psutil imported.")

    log_and_print("Importing torch...")
    import torch
    log_and_print("✅ torch imported.")

    log_and_print("Adding project path...")
    project_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, project_dir)
    log_and_print(f"✅ Project path added: {project_dir}")

    log_and_print("Importing CoreBrain...")
    from cogniflex.core.core_brain import CoreBrain
    log_and_print("✅ CoreBrain imported.")

    log_and_print("--- Import Debug Finished Successfully ---")

except Exception as e:
    log_and_print(f"❌ An error occurred: {e}")
    logging.error("Exception occurred", exc_info=True)
    import traceback
    traceback.print_exc()
