"""System monitor: GPU, CPU, RAM, Disk I/O."""
import os, sys, time, subprocess, shutil, traceback

try:
    import psutil
except ImportError:
    print('Installing psutil...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'psutil', '-q'])
    import psutil

NV_SMI = shutil.which('nvidia-smi') or r'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe'

try:
    proc = psutil.Process()
    old_io = psutil.disk_io_counters()
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # GPU
        try:
            out = subprocess.check_output(
                [NV_SMI, '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'],
                timeout=2, text=True)
            gpu_util, mem_used, mem_total, temp, power = out.strip().split(', ')
            gpu_util = int(gpu_util)
            bar = '█'*(gpu_util//5) + '░'*(20-gpu_util//5)
            print(f'  GPU: {gpu_util:3d}% | Mem: {mem_used}/{mem_total} MB | {power}W | {temp}°C')
            print(f' CUDA: {gpu_util:3d}% | {bar}')
        except:
            print('  GPU: N/A')
        
        # CPU
        cpu = int(psutil.cpu_percent(interval=0.5))
        bar = '█'*(cpu//5) + '░'*(20-cpu//5)
        print(f'  CPU: {cpu:3d}% | {bar}')
        
        # RAM
        ram = psutil.virtual_memory()
        ram_pct = int(ram.percent)
        bar = '█'*(ram_pct//5) + '░'*(20-ram_pct//5)
        print(f'  RAM: {ram.percent:3d}% | {ram.used/1024**3:.1f}/{ram.total/1024**3:.1f} GB | {bar}')
        
        # Disk I/O
        try:
            io = psutil.disk_io_counters()
            read_spd = (io.read_bytes - old_io.read_bytes) / 1024 / 1024
            write_spd = (io.write_bytes - old_io.write_bytes) / 1024 / 1024
            old_io = io
        except:
            read_spd = write_spd = 0
        print(f' Disk: R:{read_spd:.1f} W:{write_spd:.1f} MB/s')
        
        # Python process
        try:
            p = proc.memory_info()
            print(f'\n Python: {proc.memory_percent():.1f}% RAM | {p.rss/1024**2:.0f} MB | {proc.num_threads()} threads')
        except:
            pass
        
        time.sleep(1)

except Exception as e:
    print(f'\nError: {e}')
    traceback.print_exc()
    input('\nPress Enter to exit...')
