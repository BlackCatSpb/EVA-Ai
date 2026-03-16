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

    # Probe torch import in a subprocess to avoid hanging this process
    log_and_print("Probing torch import in subprocess (timeout 30s)...")
    import subprocess
    probe_code = (
        "import os, sys, time\n"
        "os.environ.setdefault('OMP_NUM_THREADS','1')\n"
        "os.environ.setdefault('MKL_NUM_THREADS','1')\n"
        "os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')\n"
        "print('Starting torch import...', flush=True)\n"
        "t=time.time()\n"
        "import torch\n"
        "print('Imported in', round(time.time()-t,2),'s', flush=True)\n"
        "print('torch', getattr(torch,'__version__','?'), flush=True)\n"
        "try:\n"
        "    x=torch.tensor([1.0,2.0]).sum().item()\n"
        "    print('tensor sum', x, flush=True)\n"
        "except Exception as e:\n"
        "    print('tensor op error', e, flush=True)\n"
    )
    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')
    env.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    try:
        result = subprocess.run(
            [sys.executable, "-u", "-c", probe_code],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        if result.returncode == 0:
            log_and_print("✅ Torch probe succeeded.")
            if result.stdout:
                for line in result.stdout.strip().splitlines():
                    logging.debug(f"[torch-stdout] {line}")
            if result.stderr:
                for line in result.stderr.strip().splitlines():
                    logging.debug(f"[torch-stderr] {line}")
        else:
            log_and_print(f"❌ Torch probe failed with code {result.returncode}.")
            if result.stdout:
                for line in result.stdout.strip().splitlines():
                    logging.debug(f"[torch-stdout] {line}")
            if result.stderr:
                for line in result.stderr.strip().splitlines():
                    logging.debug(f"[torch-stderr] {line}")
            raise RuntimeError("Torch import probe failed. See import_debug.log for details.")
    except subprocess.TimeoutExpired as te:
        log_and_print("❌ Torch probe timed out after 30s.")
        logging.error("Torch subprocess timed out", exc_info=True)
        raise

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
