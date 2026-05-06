#!/usr/bin/env python3
"""
Анализатор eva_gui.log - показывает что работает, что нет
"""
import re
from collections import defaultdict
from datetime import datetime

LOG_FILE = 'eva_gui.log'

def parse_log():
    """Парсит лог и возвращает статистику."""
    results = defaultdict(lambda: {'ok': 0, 'fail': 0, 'errors': []})
    metrics_data = defaultdict(lambda: {'values': []})
    metrics_history = []
    current_iteration = 0
    
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                # Итерация
                iter_match = re.search(r'ITERATION (\d+)', line)
                if iter_match:
                    current_iteration = int(iter_match.group(1))
                
                # Успешный endpoint
                ok_match = re.search(r'/api/(\S+):\s+OK', line)
                if ok_match:
                    endpoint = ok_match.group(1)
                    results[endpoint]['ok'] += 1
                
                # Ошибка endpoint
                fail_match = re.search(r'/api/(\S+):\s+(\d+|ERROR|CONN)', line)
                if fail_match:
                    endpoint = fail_match.group(1)
                    error = fail_match.group(2)
                    results[endpoint]['fail'] += 1
                    if error not in results[endpoint]['errors']:
                        results[endpoint]['errors'].append(error)
                
                # Метрика
                metric_match = re.search(r'METRIC:\s+(\S+)=([\d.]+)', line)
                if metric_match:
                    metric = metric_match.group(1)
                    value = float(metric_match.group(2))
                    metrics_data[metric]['values'].append((current_iteration, value))
                
                # Итоговая строка
                summary_match = re.search(r'SUMMARY:\s+(.*)$', line)
                if summary_match:
                    metrics_history.append({
                        'iteration': current_iteration,
                        'summary': summary_match.group(1)
                    })
    
    except FileNotFoundError:
        print(f"ERROR: Log file '{LOG_FILE}' not found!")
        return None
    
    return results, metrics_data, metrics_history

def main():
    print("=" * 70)
    print("EVA GUI LOG ANALYZER")
    print("=" * 70)
    
    data = parse_log()
    if not data:
        return
    
    results, metrics_data, metrics_history = data
    
    # === ENDPOINTS ===
    print("\n### ENDPOINTS STATUS ###\n")
    
    endpoints = {k: v for k, v in results.items() if not k.startswith('__')}
    if not endpoints:
        print("No endpoint data found in log")
        return
    
    print(f"{'Endpoint':<30} {'OK':>8} {'FAIL':>8} {'Status':>10}")
    print("-" * 60)
    
    for name, data in sorted(endpoints.items()):
        ok = data['ok']
        fail = data['fail']
        total = ok + fail
        
        if total == 0:
            status = "NEVER"
        elif fail == 0:
            status = "OK"
        elif ok == 0:
            status = "FAIL"
        else:
            pct = (ok / total) * 100
            status = f"WARN ({pct:.0f}%)"
        
        print(f"/api/{name:<23} {ok:>8} {fail:>8} {status:>10}")
        
        if data['errors']:
            for err in data['errors'][:3]:
                print(f"    Error: {err}")
    
    # === METRICS ===
    print("\n### METRICS STATUS ###\n")
    
    if not metrics_data:
        print("No metrics found in log")
    else:
        print(f"{'Metric':<40} {'Min':>10} {'Max':>10} {'Avg':>10} {'Current':>10}")
        print("-" * 80)
        
        for name, data in sorted(metrics_data.items()):
            values = [v for _, v in data['values']]
            if values:
                print(f"{name:<40} {min(values):>10.2f} {max(values):>10.2f} "
                      f"{sum(values)/len(values):>10.2f} {values[-1]:>10.2f}")
    
    # === SUMMARY HISTORY ===
    print("\n### SUMMARY HISTORY (last 5) ###\n")
    
    for entry in metrics_history[-5:]:
        print(f"Iter {entry['iteration']:>5}: {entry['summary']}")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
