"""
EVA Metrics Monitor - мониторинг метрик системы в реальном времени
Запускать в отдельном терминале параллельно с EVA
"""
import requests
import time
import sys
from datetime import datetime

EVA_URL = "http://127.0.0.1:5555"
REFRESH_INTERVAL = 1  # секунды


def get_all_metrics():
    """Получить все метрики от EVA"""
    try:
        analytics = requests.get(f"{EVA_URL}/api/analytics", timeout=2).json()
        metrics = requests.get(f"{EVA_URL}/api/metrics", timeout=2).json()
        return analytics, metrics
    except Exception as e:
        return None, None


def print_metrics(analytics, metrics):
    """Вывести метрики в консоль"""
    print("\n" + "=" * 80)
    print(f"  EVA METRICS MONITOR - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 80)
    
    if not analytics:
        print("\n[ОШИБКА] Нет данных от EVA")
        return
    
    # === ОСНОВНЫЕ МЕТРИКИ ===
    print("\n+==================================================================+")
    print("|  ОСНОВНЫЕ МЕТРИКИ                                                |")
    print("+------------------------------------------------------------------+")
    print("|  Запросов:     {:>8}   |  Среднее время: {:>8.0f}ms  |".format(analytics.get('queries', 0), analytics.get('avg_time', 0)))
    print("|  Успешность:   {:>8.1f}%   |  Изучено:     {:>8}      |".format(analytics.get('success_rate', 0)*100, analytics.get('learned', 0)))
    
    # === РЕСУРСЫ ===
    print("\n+==================================================================+")
    print("|  РЕСУРСЫ                                                        |")
    print("+------------------------------------------------------------------+")
    cpu = analytics.get('cpu', 0)
    ram = analytics.get('memory', 0)
    vram = analytics.get('vram', 0)
    print("|  CPU: {:>6.1f}%  |  RAM: {:>6.1f}%  |  VRAM: {:>6.1f}%                   |".format(cpu, ram, vram))
    
    # === FRACTAL GRAPH v2 ===
    print("\n+==================================================================+")
    print("|  FRACTAL GRAPH v2                                                |")
    print("+------------------------------------------------------------------+")
    print("|  Узлы: {:>6}   |  Связи: {:>6}   |  Группы: {:>6}    |".format(
        analytics.get('fractal_nodes', 0),
        analytics.get('fractal_edges', 0),
        analytics.get('fractal_groups', 0)
    ))
    
    # Детали из metrics
    if metrics and isinstance(metrics.get('graph'), dict):
        fg = metrics['graph'].get('fractal_graph_v2', {})
        if fg:
            nodes_by_level = fg.get('nodes_by_level', {})
            nodes_by_type = fg.get('nodes_by_type', {})
            total_emb = fg.get('nodes_with_embeddings', 0)
            
            # Уровни
            levels = ""
            for lvl, cnt in sorted(nodes_by_level.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
                levels += "L{}:{} ".format(lvl, cnt)
            print("|  По уровням: {:<50}    |".format(levels))
            print("|  С эмбеддингами: {:<40}    |".format(total_emb))
            
            # Типы узлов (топ-5)
            if nodes_by_type:
                top_types = sorted(nodes_by_type.items(), key=lambda x: x[1], reverse=True)[:5]
                types_str = ", ".join(["{}:{}".format(t, c) for t, c in top_types])
                print("|  Типы: {:<60}    |".format(types_str[:60]))
    
    # === КУРАТОР ГРАФА ===
    print("\n+==================================================================+")
    print("|  КУРАТОР ГРАФА                                                   |")
    print("+------------------------------------------------------------------+")
    print("|  Циклов:   {:>6}   |  Состояние: {:<12}        |".format(
        analytics.get('curator_cycles', 0),
        str(analytics.get('curator_state', 'N/A'))
    ))
    next_run = analytics.get('curator_next_run', 0)
    if next_run and next_run > 0:
        next_time = datetime.fromtimestamp(next_run).strftime('%H:%M:%S')
        print("|  След. запуск: {:<46}    |".format(next_time))
    
    # Детали куратора из metrics
    if metrics and isinstance(metrics.get('graph'), dict):
        curator = metrics['graph'].get('curator', {})
        if curator:
            print("|  Обработано узлов: {:>6}   |  Связей создано: {:>6}     |".format(
                curator.get('nodes_curated', 0),
                curator.get('links_created', 0)
            ))
            print("|  Удалено связей: {:>6}   |  Ошибка: {:<12}    |".format(
                curator.get('links_removed', 0),
                str(curator.get('last_error', 'None'))
            ))
    
    # === ВЕБ-ПОИСК (TAVILY) ===
    print("\n+==================================================================+")
    print("|  ВЕБ-ПОИСК (TAVILY)                                              |")
    print("+------------------------------------------------------------------+")
    print("|  Tavily запросов: {:>6}   |  Tavily ответов: {:>6}     |".format(
        analytics.get('tavily_requests', 0),
        analytics.get('tavily_responses', 0)
    ))
    print("|  Всего поисков:   {:>6}   |  Cache хит: {:>6}         |".format(
        analytics.get('web_searches', 0),
        analytics.get('web_cache_hits', 0)
    ))
    
    # === ПРОТИВОРЕЧИЯ ===
    print("\n+==================================================================+")
    print("|  ПРОТИВОРЕЧИЯ                                                     |")
    print("+------------------------------------------------------------------+")
    contrad = metrics.get('contradictions', {}) if metrics else {}
    print("|  Всего: {:>6}   |  Активных: {:>6}   |  Разрешено: {:>6}     |".format(
        contrad.get('total', 'N/A'),
        contrad.get('active', 'N/A'),
        contrad.get('total', 0) - contrad.get('active', 0)
    ))
    if contrad:
        runtime = contrad.get('runtime', 0)
        print("|  Время работы: {:.0f}s   |  Операций: {:>6}   |  Ошибок: {:>6}      |".format(
            runtime,
            contrad.get('operations_count', 0),
            contrad.get('error_count', 0)
        ))
    
    # === ЗДОРОВЬЕ СИСТЕМЫ ===
    print("\n+==================================================================+")
    print("|  ЗДОРОВЬЕ СИСТЕМЫ                                                 |")
    print("+------------------------------------------------------------------+")
    health = metrics.get('health', {}) if metrics else {}
    health_status = health.get('status', 'unknown')
    status_icon = "[OK]" if health_status == "healthy" else ("[WARN]" if health_status == "degraded" else "[FAIL]")
    print("|  Статус: {} {:<55}    |".format(status_icon, health_status.upper()))
    issues = health.get('issues', [])
    if issues:
        for issue in issues[:3]:
            print("|  [!] {:<70}    |".format(issue[:70]))
    else:
        print("|  [OK] Проблем не обнаружено{:50}    |".format(""))
    
    # === КЭШ ===
    print("\n+==================================================================+")
    print("|  КЭШ                                                             |")
    print("+------------------------------------------------------------------+")
    print("|  Hit Rate: {:>6.1f}%   |  Utilization: {:>6.1f}%              |".format(
        analytics.get('cache_hit_rate', 0)*100,
        analytics.get('cache_utilization', 0)*100
    ))
    
    # === АКТИВНОСТЬ ===
    print("\n+==================================================================+")
    print("|  ПОСЛЕДНЯЯ АКТИВНОСТЬ                                            |")
    print("+------------------------------------------------------------------+")
    activities = analytics.get('activities', [])
    if activities:
        for act in activities[:5]:
            title = act.get('title', 'N/A')[:50]
            print("|  * {:<70}    |".format(title))
    else:
        print("|  Нет активности{:58}    |".format(""))
    
    print("\n" + "=" * 80)


def main():
    print("+==================================================================+")
    print("|                    EVA METRICS MONITOR                            |")
    print("+------------------------------------------------------------------+")
    print("|  Target: {:<55}        |".format(EVA_URL))
    print("|  Refresh: {}s{:51}        |".format(REFRESH_INTERVAL, ""))
    print("+==================================================================+")
    print("Press Ctrl+C to stop\n")
    
    consecutive_errors = 0
    
    while True:
        try:
            analytics, metrics = get_all_metrics()
            
            if analytics is None:
                consecutive_errors += 1
                if consecutive_errors <= 3:
                    print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Ожидание EVA... ", end="", flush=True)
                continue
            
            consecutive_errors = 0
            print_metrics(analytics, metrics)
            
            time.sleep(REFRESH_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n\nОстановлен пользователем")
            sys.exit(0)
        except Exception as e:
            print(f"\nОшибка: {e}")
            time.sleep(REFRESH_INTERVAL)


if __name__ == "__main__":
    main()
