"""System monitor: GPU, CPU, RAM, Disk I/O. No extra deps needed."""
import os, sys, time, subprocess, shutil
import psutil

# Find nvidia-smi
NV_SMI = shutil.which('nvidia-smi') or r'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe'

proc = psutil.Process()
old_io = psutil.disk_io_counters()

while True:
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # GPU via nvidia-smi
    try:
        out = subprocess.check_output(
            [NV_SMI, '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            timeout=2, text=True)
        gpu_util, mem_used, mem_total, temp, power = out.strip().split(', ')
        gpu_util = int(gpu_util)
        mem_pct = int(mem_used) / int(mem_total) * 100
        print(f'  GPU: {gpu_util:3d}% | Mem: {mem_used}/{mem_total} MB | {power}W | {temp}°C')
        print(f' CUDA: {gpu_util:3d}% | {"█"*(gpu_util//5)}{"░"*(20-gpu_util//5)}')
    except:
        print('  GPU: N/A')
    
    # CPU
    cpu = psutil.cpu_percent(interval=0.5)
    print(f'  CPU: {cpu:3d}% | {"█"*(cpu//5)}{"░"*(20-cpu//5)}')
    
    # RAM
    ram = psutil.virtual_memory()
    print(f'  RAM: {ram.percent:3d}% | {ram.used/1024**3:.1f}/{ram.total/1024**3:.1f} GB')
    
    # Disk I/O
    io = psutil.disk_io_counters()
    read_spd = (io.read_bytes - old_io.read_bytes) / 1024 / 1024
    write_spd = (io.write_bytes - old_io.write_bytes) / 1024 / 1024
    old_io = io
    print(f' Disk: R:{read_spd:.1f} W:{write_spd:.1f} MB/s')
    
    # Process
    p = proc.memory_info()
    print(f'\n Python: {proc.memory_percent():.1f}% RAM | {p.rss/1024**2:.0f} MB | {proc.num_threads()} threads')
    
    time.sleep(1)
