#!/usr/bin/env python3
"""
Мониторинг метрик CogniFlex Core Brain
Каждые 30 секунд проверяет логи и сохраняет метрики
"""
import time
import re
from datetime import datetime
from pathlib import Path

def parse_metrics(log_line):
    """Извлекает метрики из строки лога"""
    if "Статистика противоречий получена за" in log_line:
        match = re.search(r'получена за (\d+\.\d+) сек', log_line)
        return float(match.group(1)) if match else None
    elif "Данные дашборда сформированы за" in log_line:
        match = re.search(r'сформированы за (\d+\.\d+) сек', log_line)
        return float(match.group(1)) if match else None
    return None

def monitor_core_brain(duration_hours=5):
    """Мониторинг Core Brain"""
    log_file = Path("current_run.log")
    start_time = datetime.now()
    end_time = start_time.timestamp() + (duration_hours * 3600)
    
    print(f"Начало мониторинга: {start_time.strftime('%H:%M:%S')}")
    print(f"Окончание: {datetime.fromtimestamp(end_time).strftime('%H:%M:%S')}")
    print("-" * 60)
    
    metrics = []
    last_position = 0
    
    while time.time() < end_time:
        current_time = datetime.now()
        
        # Читаем новые строки из лога
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(last_position)
                new_lines = f.readlines()
                last_position = f.tell()
                
                for line in new_lines:
                    if "core_brain.query_processing" in line:
                        metric = parse_metrics(line)
                        if metric is not None:
                            timestamp = re.search(r'(\d{2}:\d{2}:\d{2})', line)
                            if timestamp:
                                metrics.append({
                                    'time': timestamp.group(1),
                                    'metric': metric,
                                    'type': 'stats' if 'Статистика' in line else 'dashboard'
                                })
                                print(f"[{timestamp.group(1)}] {'Статистика' if 'Статистика' in line else 'Дашборд'}: {metric:.4f} сек")
        
        # Проверка каждые 30 секунд
        time.sleep(30)
        
        # Ежечасный отчет
        if int((time.time() - start_time.timestamp()) / 3600) > len(metrics) // 240:
            hour = int((time.time() - start_time.timestamp()) / 3600)
            print(f"\n=== ЧАС {hour} ===")
            print(f"Всего замеров: {len(metrics)}")
            if metrics:
                avg_stats = sum(m['metric'] for m in metrics if m['type'] == 'stats') / len([m for m in metrics if m['type'] == 'stats'])
                avg_dash = sum(m['metric'] for m in metrics if m['type'] == 'dashboard') / len([m for m in metrics if m['type'] == 'dashboard'])
                print(f"Среднее время статистики: {avg_stats:.4f} сек")
                print(f"Среднее время дашборда: {avg_dash:.4f} сек")
            print("-" * 60)
    
    print(f"\nМониторинг завершен: {datetime.now().strftime('%H:%M:%S')}")
    return metrics

if __name__ == "__main__":
    monitor_core_brain()
