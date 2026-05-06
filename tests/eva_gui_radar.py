#!/usr/bin/env python3
"""
EVA GUI Radar - мониторинг всех метрик каждую секунду
Записывает всё в eva_gui.log для анализа
"""
import requests
from requests.auth import HTTPBasicAuth
import json
import time
import os
from datetime import datetime

AUTH = HTTPBasicAuth('admin', 'cogniflex')
BASE = 'http://127.0.0.1:5555'
LOG_FILE = 'eva_gui.log'

ENDPOINTS = [
    ('/', ''),
    ('/api/health', '/api/health'),
    ('/api/analytics', '/api/analytics'),
    ('/api/metrics', '/api/metrics'),
    ('/api/stats', '/api/stats'),
    ('/api/memory-graph', '/api/memory-graph'),
    ('/api/websearch_stats', '/api/websearch_stats'),
    ('/api/graph/stats', '/api/graph/stats'),
    ('/api/nodes', '/api/nodes'),
    ('/api/edges', '/api/edges'),
    ('/api/contradictions', '/api/contradictions'),
    ('/api/concepts', '/api/concepts'),
    ('/api/sessions', '/api/sessions'),
    ('/api/events/stream', '/api/events/stream'),
]

def log(msg):
    """Записать в лог с timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def check_endpoint(name, url):
    """Проверить один endpoint."""
    try:
        r = requests.get(f'{BASE}{url}', auth=AUTH, timeout=5)
        status = r.status_code
        
        if status == 200:
            try:
                data = r.json()
                keys = list(data.keys()) if isinstance(data, dict) else f"list({len(data)})"
                return status, 'OK', keys, data
            except:
                return status, 'OK (not JSON)', None, r.text[:100]
        else:
            return status, 'FAIL', None, r.text[:100]
    except requests.exceptions.ConnectionError:
        return 'CONN_ERR', 'NO_CONNECTION', None, None
    except requests.exceptions.Timeout:
        return 'TIMEOUT', 'TIMEOUT', None, None
    except Exception as e:
        return 'ERROR', str(e)[:50], None, None

def extract_metrics(data, prefix=''):
    """Извлечь метрики из ответа."""
    if not isinstance(data, dict):
        return {}
    
    metrics = {}
    for key, value in data.items():
        if isinstance(value, (int, float, str, bool)):
            metrics[f"{prefix}{key}"] = value
        elif isinstance(value, dict):
            metrics.update(extract_metrics(value, f"{prefix}{key}."))
        elif isinstance(value, list):
            metrics[f"{prefix}{key}_count"] = len(value)
    return metrics

def main():
    log("=" * 80)
    log("EVA GUI RADAR - STARTED")
    log(f"Base URL: {BASE}")
    log(f"Log file: {LOG_FILE}")
    log("=" * 80)
    
    iteration = 0
    consecutive_errors = 0
    
    while True:
        iteration += 1
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        
        log(f"--- ITERATION {iteration} at {timestamp} ---")
        
        all_ok = True
        all_metrics = {}
        
        for name, url in ENDPOINTS:
            status, result, keys, data = check_endpoint(name, url)
            
            if status == 200:
                # Извлекаем метрики
                metrics = extract_metrics(data)
                all_metrics.update(metrics)
                
                # Логируем ключевые метрики
                key_metrics = []
                for key, value in list(metrics.items())[:10]:
                    if isinstance(value, (int, float)) and value != 0:
                        key_metrics.append(f"{key}={value}")
                
                if key_metrics:
                    log(f"  {name}: OK, keys={len(data) if isinstance(data, dict) else 'N/A'}")
                    for m in key_metrics[:5]:
                        log(f"    METRIC: {m}")
                else:
                    log(f"  {name}: OK, keys={len(data) if isinstance(data, dict) else 'N/A'}")
                
                consecutive_errors = 0
            else:
                log(f"  {name}: {status} - {result}")
                all_ok = False
                consecutive_errors += 1
        
        # Итоговая строка с ключевыми метриками
        summary_metrics = []
        for key in ['fractal_nodes', 'total_nodes', 'cpu', 'memory', 'queries', 'dialogs']:
            if key in all_metrics and all_metrics[key]:
                summary_metrics.append(f"{key}={all_metrics[key]}")
        
        if summary_metrics:
            log(f"  SUMMARY: {', '.join(summary_metrics)}")
        
        # Проверка соединения
        if consecutive_errors >= 3:
            log("ERROR: 3 consecutive failures - server might be down")
            break
        
        log("")  # Пустая строка между итерациями
        time.sleep(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log("=" * 80)
        log("EVA GUI RADAR - STOPPED by user")
        log("=" * 80)
    except Exception as e:
        log(f"ERROR: {e}")
        raise
