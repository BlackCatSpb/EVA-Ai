"""
Модуль для работы с противоречиями в графическом интерфейсе
"""
import tkinter as tk
from tkinter import ttk, simpledialog
import logging
from typing import Dict, Any, Optional, List


logger = logging.getLogger("cogniflex.gui.contradiction")

class ContradictionModule:
    """Модуль для отображения и управления противоречиями."""
    
    def __init__(self, gui):
        """
        Инициализирует модуль противоречий.
        
        Args:
            gui: Ссылка на основной GUI
        """
        self.gui = gui
        self.frame = None
        self.contradictions_tree = None
        self.filter_var = tk.StringVar(value="all")
        self.contradictions_data = []  # Кэшируем данные для фильтрации
        self._after_jobs: list = []  # Храним ID запланированных задач after
        logger.debug("Модуль противоречий инициализирован")
    
    def activate(self):
        """Активирует модуль (создает интерфейс)."""
        if self.frame:
            self.frame.destroy()
        
        self.frame = ttk.Frame(self.gui.content_area)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Заголовок
        ttk.Label(
            self.frame,
            text="Анализ противоречий",
            font=("Arial", 14, "bold")
        ).pack(anchor="w", pady=(0, 10))
        
        # Описание
        ttk.Label(
            self.frame,
            text="Этот модуль анализирует и разрешает противоречия в знаниях системы.",
            wraplength=600
        ).pack(anchor="w", pady=(0, 10))
        
        # Панель фильтрации
        filter_frame = ttk.Frame(self.frame)
        filter_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(filter_frame, text="Фильтр:").pack(side=tk.LEFT, padx=(0, 5))
        
        filter_options = [
            ("Все", "all"),
            ("Критические", "critical"),
            ("Высокий приоритет", "high"),
            ("Средний приоритет", "medium"),
            ("Низкий приоритет", "low")
        ]
        
        for text, value in filter_options:
            ttk.Radiobutton(
                filter_frame,
                text=text,
                variable=self.filter_var,
                value=value,
                command=self.refresh_contradictions
            ).pack(side=tk.LEFT, padx=5)
        
        # Статистика
        stats_frame = ttk.Frame(self.frame)
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.stats_labels = {}
        stats_items = [
            ("total", "Всего противоречий:", "total_label"),
            ("unresolved", "Активных:", "unresolved_label"),
            ("resolved", "Разрешено:", "resolved_label"),
            ("critical", "Критических:", "critical_label")
        ]
        
        for i, (key, label, var_name) in enumerate(stats_items):
            item_frame = ttk.Frame(stats_frame)
            item_frame.pack(side=tk.LEFT, padx=15)
            
            ttk.Label(item_frame, text=label, font=("Arial", 9, "bold")).pack(anchor="w")
            self.stats_labels[key] = ttk.Label(item_frame, text="0", font=("Arial", 9))
            self.stats_labels[key].pack(anchor="w")
        
        # Таблица противоречий
        tree_frame = ttk.Frame(self.frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Создаем Treeview с прокруткой
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.contradictions_tree = ttk.Treeview(
            tree_frame,
            columns=("id", "concept1", "concept2", "severity", "domain", "status"),
            show="headings",
            yscrollcommand=scrollbar.set
        )
        
        # Настройка колонок
        self.contradictions_tree.heading("id", text="ID", anchor=tk.W)
        self.contradictions_tree.heading("concept1", text="Концепт 1", anchor=tk.W)
        self.contradictions_tree.heading("concept2", text="Концепт 2", anchor=tk.W)
        self.contradictions_tree.heading("severity", text="Серьезность", anchor=tk.W)
        self.contradictions_tree.heading("domain", text="Домен", anchor=tk.W)
        self.contradictions_tree.heading("status", text="Статус", anchor=tk.W)
        
        self.contradictions_tree.column("id", width=50, anchor=tk.W)
        self.contradictions_tree.column("concept1", width=150, anchor=tk.W)
        self.contradictions_tree.column("concept2", width=150, anchor=tk.W)
        self.contradictions_tree.column("severity", width=100, anchor=tk.W)
        self.contradictions_tree.column("domain", width=100, anchor=tk.W)
        self.contradictions_tree.column("status", width=100, anchor=tk.W)
        
        self.contradictions_tree.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.contradictions_tree.yview)
        
        # Кнопки действий
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(
            button_frame,
            text="Анализировать противоречия",
            command=self.analyze_contradictions
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Разрешить выбранное",
            command=self.resolve_selected
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Запросить уточнение",
            command=self.request_clarification
        ).pack(side=tk.LEFT, padx=5)
        
        # Загружаем данные
        self.refresh_contradictions()
        logger.info("Модуль противоречий активирован")
    
    def deactivate(self):
        """Деактивирует модуль (очищает интерфейс)."""
        # Отменяем запланированные задачи обновления
        try:
            if hasattr(self.gui, 'root') and self.gui.root and self._after_jobs:
                for job_id in self._after_jobs:
                    try:
                        self.gui.root.after_cancel(job_id)
                    except Exception:
                        pass
        finally:
            self._after_jobs = []
        
        if self.frame:
            self.frame.destroy()
            self.frame = None
        logger.info("Модуль противоречий деактивирован")
    
    def _is_active(self) -> bool:
        """Возвращает True, если модуль активен и его виджеты существуют."""
        try:
            return (
                self.gui and getattr(self.gui, 'current_view', None) == 'contradictions' and
                self.frame is not None and self.frame.winfo_exists()
            )
        except Exception:
            return False
    
    def _get_contradictions_data(self) -> List[Dict[str, Any]]:
        """Получает данные о противоречиях из ядра системы."""
        contradictions = []
        
        # Проверяем наличие ядра
        if not self.gui.brain:
            logger.warning("Ядро системы недоступно для получения данных о противоречиях")
            return contradictions
        
        try:
            # Пытаемся получить данные через основной метод
            if hasattr(self.gui.brain, 'get_contradiction_statistics'):
                stats = self.gui.brain.get_contradiction_statistics()
                
                # Обработка данных в формате, который возвращает ядро
                if isinstance(stats, dict):
                    # Если в статистике есть список противоречий
                    if 'contradictions' in stats and isinstance(stats['contradictions'], list):
                        contradictions = stats['contradictions']
                    elif 'by_severity' in stats or 'by_domain' in stats:
                        # Если данные структурированы по категориям, преобразуем их в список
                        contradictions = self._convert_statistics_to_list(stats)
                    else:
                        # Если есть только общий счетчик, получаем данные напрямую
                        if hasattr(self.gui.brain, 'get_active_contradictions'):
                            contradictions = self.gui.brain.get_active_contradictions()
                        elif hasattr(self.gui.brain, 'get_all_contradictions'):
                            contradictions = self.gui.brain.get_all_contradictions()
            
            # Если данные не получены, пытаемся получить через contradiction_resolver
            if not contradictions and hasattr(self.gui.brain, 'contradiction_resolver'):
                resolver = self.gui.brain.contradiction_resolver
                if hasattr(resolver, 'get_active_contradictions'):
                    contradictions = resolver.get_active_contradictions()
                elif hasattr(resolver, 'get_contradictions'):
                    contradictions = resolver.get_contradictions()
            
            # Если данные не получены, пытаемся получить через contradiction_manager
            if not contradictions and hasattr(self.gui.brain, 'contradiction_manager'):
                manager = self.gui.brain.contradiction_manager
                if hasattr(manager, 'get_active_contradictions'):
                    contradictions = manager.get_active_contradictions()
                elif hasattr(manager, 'get_contradictions'):
                    contradictions = manager.get_contradictions()
            
            # Если данные все еще не получены, пытаемся использовать альтернативные методы
            if not contradictions:
                if hasattr(self.gui.brain, 'get_system_dashboard_data'):
                    dashboard = self.gui.brain.get_system_dashboard_data()
                    if 'contradiction_statistics' in dashboard:
                        stats = dashboard['contradiction_statistics']
                        contradictions = self._convert_statistics_to_list(stats)
            
            # Проверяем структуру данных и приводим к единому формату
            contradictions = self._normalize_contradictions_data(contradictions)
            
            logger.debug(f"Получено {len(contradictions)} противоречий для отображения")
            return contradictions
        except Exception as e:
            logger.error(f"Критическая ошибка получения данных о противоречиях: {e}", exc_info=True)
            return []

    def _convert_statistics_to_list(self, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Преобразует статистику противоречий в список противоречий."""
        contradictions = []
        
        # Пытаемся определить, есть ли данные о конкретных противоречиях
        if 'contradictions' in stats and isinstance(stats['contradictions'], list):
            return stats['contradictions']
        
        # Если есть данные по доменам, пытаемся извлечь информацию
        if 'by_domain' in stats and isinstance(stats['by_domain'], dict):
            for domain, domain_data in stats['by_domain'].items():
                # Если данные по домену содержат список противоречий
                if isinstance(domain_data, list):
                    for contradiction in domain_data:
                        contradictions.append({
                            'id': f"{domain}_{len(contradictions)}",
                            'concept1': contradiction.get('concept1', 'N/A'),
                            'concept2': contradiction.get('concept2', 'N/A'),
                            'severity': contradiction.get('severity', 'low'),
                            'domain': domain,
                            'status': contradiction.get('status', 'active')
                        })
                # Если данные по домену содержат только счетчики
                elif isinstance(domain_data, dict) and 'count' in domain_data:
                    # Здесь можно добавить логику для генерации примеров противоречий
                    pass
        
        # Если есть данные по серьезности, пытаемся извлечь информацию
        if 'by_severity' in stats and isinstance(stats['by_severity'], dict):
            for severity, count in stats['by_severity'].items():
                if count > 0 and severity != 'total':
                    # Добавляем примеры противоречий для отображения
                    for i in range(min(count, 5)):  # Ограничиваем количество примеров
                        contradictions.append({
                            'id': f"{severity}_{i}",
                            'concept1': f"Пример концепта {i+1}",
                            'concept2': f"Конфликтующий концепт {i+1}",
                            'severity': severity,
                            'domain': "Неизвестно",
                            'status': "active"
                        })
        
        # Если противоречий нет, но есть общий счетчик, создаем пустой список
        if not contradictions and 'total' in stats:
            for i in range(min(stats['total'], 10)):  # Ограничиваем количество примеров
                contradictions.append({
                    'id': f"contradiction_{i}",
                    'concept1': f"Концепт {i}",
                    'concept2': f"Конфликт {i}",
                    'severity': "medium",
                    'domain': "Общий",
                    'status': "active"
                })
        
        return contradictions

    def _normalize_contradictions_data(self, data: Any) -> List[Dict[str, Any]]:
        """Приводит данные о противоречиях к единому формату."""
        normalized = []
        
        # Если данные уже в правильном формате
        if isinstance(data, list):
            for item in data:
                # Проверяем, что элемент - словарь
                if isinstance(item, dict):
                    # Создаем нормализованный противоречие
                    normalized_contradiction = {
                        'id': item.get('id', f"contradiction_{len(normalized)}"),
                        'concept1': item.get('concept1', 'N/A'),
                        'concept2': item.get('concept2', 'N/A'),
                        'severity': item.get('severity', 'low').lower(),
                        'domain': item.get('domain', 'unknown'),
                        'status': item.get('status', 'active').lower()
                    }
                    normalized.append(normalized_contradiction)
                # Если элемент не словарь, пытаемся его интерпретировать
                elif hasattr(item, '__dict__'):
                    normalized_contradiction = {
                        'id': getattr(item, 'id', f"contradiction_{len(normalized)}"),
                        'concept1': getattr(item, 'concept1', 'N/A'),
                        'concept2': getattr(item, 'concept2', 'N/A'),
                        'severity': getattr(item, 'severity', 'low').lower(),
                        'domain': getattr(item, 'domain', 'unknown'),
                        'status': getattr(item, 'status', 'active').lower()
                    }
                    normalized.append(normalized_contradiction)
        
        # Если данные в виде словаря с ключами
        elif isinstance(data, dict):
            # Если есть ключ 'contradictions', используем его
            if 'contradictions' in data and isinstance(data['contradictions'], list):
                return self._normalize_contradictions_data(data['contradictions'])
            # Иначе пытаемся обработать как одиночное противоречие
            else:
                normalized.append({
                    'id': data.get('id', 'contradiction_0'),
                    'concept1': data.get('concept1', 'N/A'),
                    'concept2': data.get('concept2', 'N/A'),
                    'severity': data.get('severity', 'low').lower(),
                    'domain': data.get('domain', 'unknown'),
                    'status': data.get('status', 'active').lower()
                })
        
        # Если данные - одиночный объект
        elif hasattr(data, '__dict__'):
            normalized.append({
                'id': getattr(data, 'id', 'contradiction_0'),
                'concept1': getattr(data, 'concept1', 'N/A'),
                'concept2': getattr(data, 'concept2', 'N/A'),
                'severity': getattr(data, 'severity', 'low').lower(),
                'domain': getattr(data, 'domain', 'unknown'),
                'status': getattr(data, 'status', 'active').lower()
            })
        
        return normalized
    
    def _update_statistics(self, contradictions: List[Dict[str, Any]]):
        """Обновляет статистику противоречий."""
        if not self._is_active():
            return
        total = len(contradictions)
        unresolved = sum(1 for c in contradictions if c.get('status', 'active').lower() == 'active')
        resolved = total - unresolved
        critical = sum(1 for c in contradictions if c.get('severity', '').lower() == 'critical')
        
        # Обновляем метки статистики
        try:
            for key, value in {
                "total": total,
                "unresolved": unresolved,
                "resolved": resolved,
                "critical": critical,
            }.items():
                lbl = self.stats_labels.get(key)
                if lbl and lbl.winfo_exists():
                    lbl.config(text=str(value))
        except tk.TclError:
            # Виджеты могли быть уничтожены — безопасно выходим
            return
    
    def refresh_contradictions(self):
        """Обновляет список противоречий."""
        try:
            if not self._is_active():
                return
            # Получаем данные о противоречиях
            self.contradictions_data = self._get_contradictions_data()
            
            # Обновляем статистику
            self._update_statistics(self.contradictions_data)
            
            # Очищаем предыдущие данные
            if self.contradictions_tree and self.contradictions_tree.winfo_exists():
                for item in self.contradictions_tree.get_children():
                    self.contradictions_tree.delete(item)
            
            # Фильтруем противоречия
            filter_value = self.filter_var.get()
            filtered = []
            
            for contradiction in self.contradictions_data:
                severity = contradiction.get("severity", "low").lower()
                status = contradiction.get("status", "active").lower()
                
                if filter_value == "all":
                    filtered.append(contradiction)
                elif filter_value == "critical" and severity == "critical":
                    filtered.append(contradiction)
                elif filter_value == "high" and severity in ["critical", "high"]:
                    filtered.append(contradiction)
                elif filter_value == "medium" and severity == "medium":
                    filtered.append(contradiction)
                elif filter_value == "low" and severity == "low":
                    filtered.append(contradiction)
            
            # Добавляем данные в таблицу
            if self.contradictions_tree and self.contradictions_tree.winfo_exists():
                for contradiction in filtered:
                    self.contradictions_tree.insert("", tk.END, values=(
                        contradiction.get("id", "N/A"),
                        contradiction.get("concept1", "N/A"),
                        contradiction.get("concept2", "N/A"),
                        contradiction.get("severity", "N/A"),
                        contradiction.get("domain", "N/A"),
                        contradiction.get("status", "N/A")
                    ))
        except Exception as e:
            logger.error(f"Ошибка обновления списка противоречий: {e}", exc_info=True)
            self.gui.show_notification(f"Ошибка загрузки противоречий: {str(e)}", "error")

    def _schedule_refresh(self, delay_ms: int):
        """Планирует обновление списка противоречий с проверкой активности модуля."""
        try:
            if not hasattr(self.gui, 'root') or not self.gui.root:
                return
            job_id = self.gui.root.after(delay_ms, lambda: self._safe_refresh())
            self._after_jobs.append(job_id)
        except Exception:
            pass

    def _safe_refresh(self):
        """Безопасное обновление (не выполняется, если модуль не активен)."""
        # Очищаем истекшие job_id (необязательно, но полезно)
        self._after_jobs = []
        if self._is_active():
            self.refresh_contradictions()
    
    def cleanup(self):
        """Очищает все запланированные задачи."""
        try:
            if hasattr(self, '_after_jobs') and hasattr(self.gui, 'root') and self.gui.root:
                for job_id in self._after_jobs:
                    try:
                        self.gui.root.after_cancel(job_id)
                    except:
                        pass
                self._after_jobs.clear()
                logger.debug("Очищены все after задачи в contradiction_module")
        except Exception as e:
            logger.error(f"Ошибка очистки contradiction_module: {e}")
    
    def analyze_contradictions(self):
        """Запускает анализ противоречий."""
        if not self.gui.brain:
            self.gui.show_notification("Система не инициализирована", "error")
            return
        
        try:
            # Проверяем, есть ли метод для анализа противоречий
            if hasattr(self.gui.brain, 'analyze_contradictions'):
                self.gui.brain.analyze_contradictions()
            elif hasattr(self.gui.brain, 'contradiction_resolver') and self.gui.brain.contradiction_resolver:
                if hasattr(self.gui.brain.contradiction_resolver, 'detect_contradictions'):
                    self.gui.brain.contradiction_resolver.detect_contradictions()
                elif hasattr(self.gui.brain.contradiction_resolver, 'analyze'):
                    self.gui.brain.contradiction_resolver.analyze()
            elif hasattr(self.gui.brain, 'contradiction_manager') and self.gui.brain.contradiction_manager:
                if hasattr(self.gui.brain.contradiction_manager, 'detect_contradictions'):
                    self.gui.brain.contradiction_manager.detect_contradictions()
                elif hasattr(self.gui.brain.contradiction_manager, 'analyze'):
                    self.gui.brain.contradiction_manager.analyze()
            
            self.gui.show_notification("Анализ противоречий запущен", "info")
            self._schedule_refresh(2000)
        except Exception as e:
            logger.error(f"Ошибка анализа противоречий: {e}", exc_info=True)
            self.gui.show_notification(f"Ошибка анализа: {str(e)}", "error")
    
    def resolve_selected(self):
        """Разрешает выбранное противоречие."""
        selected_items = self.contradictions_tree.selection()
        if not selected_items:
            self.gui.show_notification("Выберите противоречие для разрешения", "warning")
            return
        
        try:
            # Получаем данные выбранного противоречия
            item_id = selected_items[0]
            values = self.contradictions_tree.item(item_id)['values']
            contradiction_id = values[0]
            
            # Проверяем, есть ли метод для разрешения противоречий
            if hasattr(self.gui.brain, 'resolve_contradiction'):
                self.gui.brain.resolve_contradiction(contradiction_id)
            elif hasattr(self.gui.brain, 'contradiction_resolver') and self.gui.brain.contradiction_resolver:
                if hasattr(self.gui.brain.contradiction_resolver, 'resolve_contradiction'):
                    self.gui.brain.contradiction_resolver.resolve_contradiction(contradiction_id)
                elif hasattr(self.gui.brain.contradiction_resolver, 'resolve'):
                    self.gui.brain.contradiction_resolver.resolve(contradiction_id)
            elif hasattr(self.gui.brain, 'contradiction_manager') and self.gui.brain.contradiction_manager:
                if hasattr(self.gui.brain.contradiction_manager, 'resolve_contradiction'):
                    self.gui.brain.contradiction_manager.resolve_contradiction(contradiction_id)
                elif hasattr(self.gui.brain.contradiction_manager, 'resolve'):
                    self.gui.brain.contradiction_manager.resolve(contradiction_id)
            
            self.gui.show_notification(f"Противоречие {contradiction_id} отправлено на разрешение", "success")
            self._schedule_refresh(1000)
        except Exception as e:
            logger.error(f"Ошибка разрешения противоречия: {e}", exc_info=True)
            self.gui.show_notification(f"Ошибка разрешения: {str(e)}", "error")
    
    def request_clarification(self):
        """Запрашивает уточнение для выбранного противоречия."""
        selected_items = self.contradictions_tree.selection()
        if not selected_items:
            self.gui.show_notification("Выберите противоречие для уточнения", "warning")
            return
        
        try:
            # Получаем данные выбранного противоречия
            item_id = selected_items[0]
            values = self.contradictions_tree.item(item_id)['values']
            contradiction_id = values[0]
            concept1 = values[1]
            concept2 = values[2]
            
            # Запрашиваем уточнение у пользователя
            clarification = simpledialog.askstring(
                "Уточнение контекста",
                f"Пожалуйста, уточните контекст для противоречия между '{concept1}' и '{concept2}':",
                parent=self.gui.root
            )
            
            if clarification:
                # Отправляем уточнение в систему
                if hasattr(self.gui.brain, 'process_user_clarification'):
                    self.gui.brain.process_user_clarification(
                        contradiction_id,
                        self.gui.current_user["id"],
                        {"context": clarification}
                    )
                elif hasattr(self.gui.brain, 'contradiction_manager') and self.gui.brain.contradiction_manager:
                    if hasattr(self.gui.brain.contradiction_manager, 'process_user_clarification'):
                        self.gui.brain.contradiction_manager.process_user_clarification(
                            contradiction_id,
                            self.gui.current_user["id"],
                            {"context": clarification}
                        )
                
                self.gui.show_notification("Уточнение отправлено", "success")
                self._schedule_refresh(1000)
        except Exception as e:
            logger.error(f"Ошибка запроса уточнения: {e}", exc_info=True)
            self.gui.show_notification(f"Ошибка запроса уточнения: {str(e)}", "error")