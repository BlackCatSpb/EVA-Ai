"""
Metrics Monitor - мониторинг метрик системы в реальном времени
Запускать в отдельном терминале параллельно с EVA
"""
import requests
import time
import sys
from datetime import datetime

EVA_URL = "http://127.0.0.1:5555"
REFRESH_INTERVAL = 1  # секунды


def get_metrics():
    """Получить все метрики от EVA"""
    try:
        # Analytics
        analytics = requests.get(f"{EVA_URL}/api/analytics", timeout=2).json()
        
        # Metrics
        metrics = requests.get(f"{EVA_URL}/api/metrics", timeout=2).json()
        
        # WebSearch stats
        websearch = requests.get(f"{EVA_URL}/api/websearch_stats", timeout=2).json()
        
        return analytics, metrics, websearch
    except Exception as e:
        return None, None, None


def print_metrics(analytics, metrics, websearch):
    """Вывести метрики в консоль"""
    print("\n" + "=" * 70)
    print(f"  EVA Metrics Monitor - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)
    
    # Basic Analytics
    print("\n[ОСНОВНЫЕ МЕТРИКИ]")
    print(f"  Запросов:      {analytics.get('queries', 'N/A')}")
    print(f"  Среднее время:  {analytics.get('avg_time', 0):.0f}ms")
    print(f"  Успешность:    {analytics.get('success_rate', 0)*100:.1f}%")
    
    # Resources
    print("\n[РЕСУРСЫ]")
    print(f"  CPU:           {analytics.get('cpu', 0):.1f}%")
    print(f"  RAM:           {analytics.get('memory', 0):.1f}%")
    print(f"  VRAM:          {analytics.get('vram', 0):.1f}%")
    
    # Learning
    print("\n[ОБУЧЕНИЕ]")
    print(f"  Диалогов:      {analytics.get('dialogs', 0)}")
    print(f"  Пробелов:      {analytics.get('gaps', 0)}")
    print(f"  Изучено:       {analytics.get('learned', 0)}")
    
    # Cache
    print("\n[КЭШ]")
    print(f"  Hit Rate:      {analytics.get('cache_hit_rate', 0)*100:.1f}%")
    print(f"  Utilization:   {analytics.get('cache_utilization', 0)*100:.1f}%")
    
    # FractalGraph
    print("\n[FRACTAL GRAPH v2]")
    fg_stats = metrics.get('graph', {}).get('fractal_graph_v2', {}) if isinstance(metrics.get('graph'), dict) else {}
    print(f"  Узлы:         {analytics.get('fractal_nodes', fg_stats.get('total_nodes', 'N/A'))}")
    print(f"  Связи:         {analytics.get('fractal_edges', fg_stats.get('total_edges', 'N/A'))}")
    print(f"  Группы:        {analytics.get('fractal_groups', fg_stats.get('total_groups', 'N/A'))}")
    
    # Knowledge Graph (deprecated)
    kg_stats = metrics.get('graph', {}).get('knowledge_graph', {}) if isinstance(metrics.get('graph'), dict) else {}
    print("\n[KNOWLEDGE GRAPH]")
    print(f"  Узлы:         {kg_stats.get('total_nodes', 'N/A')} {'(adapter)' if kg_stats.get('total_nodes') else '(deprecated)'}")
    
    # Web Search / Tavily
    print("\n[ВЕБ-ПОИСК (TAVILY)]")
    if websearch and websearch.get('stats'):
        ws = websearch['stats']
        print(f"  Tavily запросов:   {ws.get('tavily_requests', 0)}")
        print(f"  Tavily ответов:   {ws.get('tavily_responses', 0)}")
        print(f"  Активных:         {ws.get('active_requests', 0)}")
        print(f"  Ошибок:           {ws.get('tavily_errors', 0)}")
        print(f"  Cache хитов:       {ws.get('cache_hits', 0)}")
    else:
        print("  (нет данных)")
    
    # Analytics from API
    print("\n[ANALYTICS API]")
    print(f"  Tavily запр.:    {analytics.get('tavily_requests', 'N/A')}")
    print(f"  Tavily отв.:     {analytics.get('tavily_responses', 'N/A')}")
    print(f"  Всего поисков:   {analytics.get('web_searches', 'N/A')}")
    
    # Curator
    print("\n[ГРАФ КУРАТОР]")
    print(f"  Циклов:         {analytics.get('curator_cycles', 0)}")
    print(f"  Состояние:       {analytics.get('curator_state', 'N/A')}")
    next_run = analytics.get('curator_next_run', 0)
    if next_run and next_run > 0:
        from datetime import datetime
        print(f"  След. запуск:    {datetime.fromtimestamp(next_run).strftime('%H:%M:%S')}")
    
    # Contradictions
    print("\n[ПРОТИВОРЕЧИЯ]")
    contrad = metrics.get('contradictions', {}) if metrics else {}
    print(f"  Всего:           {contrad.get('total', 'N/A')}")
    print(f"  Активных:        {contrad.get('active', 'N/A')}")
    
    # Concepts
    print("\n[КОНЦЕПТЫ]")
    concepts = metrics.get('concepts', {}) if metrics else {}
    print(f"  Временных:        {concepts.get('provisional', 'N/A')}")
    print(f"  Подтверждённых:  {concepts.get('confirmed', 'N/A')}")
    print(f"  Архивных:        {concepts.get('archived', 'N/A')}")


def main():
    print("EVA Metrics Monitor")
    print(f"Target: {EVA_URL}")
    print(f"Refresh: {REFRESH_INTERVAL}s")
    print("Press Ctrl+C to stop\n")
    
    consecutive_errors = 0
    
    while True:
        try:
            analytics, metrics, websearch = get_metrics()
            
            if analytics is None:
                consecutive_errors += 1
                if consecutive_errors <= 3:
                    print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Ожидание EVA... ", end="", flush=True)
                continue
            
            consecutive_errors = 0
            print_metrics(analytics, metrics, websearch)
            
            time.sleep(REFRESH_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n\nОстановлен пользователем")
            sys.exit(0)
        except Exception as e:
            print(f"\nОшибка: {e}")
            time.sleep(REFRESH_INTERVAL)


if __name__ == "__main__":
    main()
