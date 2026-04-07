"""
Status bar, system indicators, and health display for ЕВА GUI.
"""
import logging
import time
from typing import Dict, Any

import tkinter as tk
from tkinter import ttk

logger = logging.getLogger("eva_ai.gui.core")


class StatusBarMixin:
    """Mixin for status bar creation and system metrics display."""

    def _create_status_bar(self):
        self.status_bar = ttk.Frame(self.root, height=30)
        self.status_bar.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.status_indicator = tk.Canvas(self.status_bar, width=15, height=15, highlightthickness=0)
        self.status_indicator.pack(side=tk.LEFT, padx=(0, 5))
        self.status_indicator.create_oval(2, 2, 13, 13, fill=self.colors['warning'], tags="indicator")
        self.connection_status = ttk.Label(self.status_bar, text="Соединение: инициализация...")
        self.connection_status.pack(side=tk.LEFT)
        self.model_load_label = ttk.Label(self.status_bar, text="")
        self.model_load_label.pack(side=tk.LEFT, padx=(10, 0))
        self.timestamp_label = ttk.Label(self.status_bar, text="--:--:--")
        self.timestamp_label.pack(side=tk.RIGHT, padx=10)
        metrics_frame = ttk.Frame(self.status_bar)
        metrics_frame.pack(side=tk.RIGHT, padx=10)
        ttk.Label(metrics_frame, text="CPU:").pack(side=tk.LEFT)
        self.cpu_label = ttk.Label(metrics_frame, text="0%")
        self.cpu_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(metrics_frame, text="RAM:").pack(side=tk.LEFT)
        self.memory_label = ttk.Label(metrics_frame, text="0%")
        self.memory_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(metrics_frame, text="Противоречия:").pack(side=tk.LEFT)
        self.contradictions_label = ttk.Label(metrics_frame, text="0")
        self.contradictions_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(metrics_frame, text="HitRate:").pack(side=tk.LEFT)
        self.hit_rate_label = ttk.Label(metrics_frame, text="0.0%")
        self.hit_rate_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(metrics_frame, text="CacheUtil:").pack(side=tk.LEFT)
        self.cache_util_label = ttk.Label(metrics_frame, text="0.0%")
        self.cache_util_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(metrics_frame, text="DiskEntries:").pack(side=tk.LEFT)
        self.disk_entries_label = ttk.Label(metrics_frame, text="0")
        self.disk_entries_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(metrics_frame, text="IOtokens:").pack(side=tk.LEFT)
        self.io_tokens_label = ttk.Label(metrics_frame, text="0")
        self.io_tokens_label.pack(side=tk.LEFT)

    def _update_interface(self):
        if not self.root or not self.running:
            return
        
        try:
            if self.integrator and hasattr(self.integrator, 'get_system_stats'):
                self.dashboard_data = self.integrator.get_system_stats() or {}
            elif self.brain and hasattr(self.brain, 'get_system_dashboard_data'):
                self.dashboard_data = self.brain.get_system_dashboard_data() or {}
            else:
                self.dashboard_data = {}
            
            self.resource_snapshot = {}
            self.cache_stats = {}
            try:
                if self.brain and hasattr(self.brain, 'get_resource_snapshot'):
                    self.resource_snapshot = self.brain.get_resource_snapshot() or {}
            except Exception:
                self.resource_snapshot = {}
            try:
                if self.brain and hasattr(self.brain, 'get_cache_stats'):
                    self.cache_stats = self.brain.get_cache_stats() or {}
            except Exception:
                self.cache_stats = {}
            
            if hasattr(self, 'timestamp_label') and self.timestamp_label:
                from datetime import datetime
                self.timestamp_label.config(text=datetime.now().strftime("%H:%M:%S"))
            self._update_status_indicator()
            self._update_system_metrics()
            self._handle_notifications()

            active_module = getattr(self, f"{self.current_view}_module", None)
            if active_module and hasattr(active_module, 'update'):
                active_module.update()

        except tk.TclError:
            pass
        except Exception as e:
            logger.error(f"Ошибка обновления интерфейса: {e}", exc_info=True)
        
        return None

    def _update_system_metrics(self):
        dashboard = getattr(self, 'dashboard_data', None) or {}
        metrics = dashboard.get('metrics', {})
        cpu = metrics.get('cpu_usage', 0.0)
        mem = metrics.get('memory_usage', 0.0)
        try:
            cpu_val = float(cpu)
            if cpu_val <= 1.5:
                cpu_val *= 100.0
        except Exception:
            cpu_val = 0.0
        try:
            mem_val = float(mem)
            if mem_val <= 1.5:
                mem_val *= 100.0
        except Exception:
            mem_val = 0.0
        if hasattr(self, 'cpu_label') and self.cpu_label:
            self.cpu_label.config(text=f"{cpu_val:.1f}%")
        if hasattr(self, 'memory_label') and self.memory_label:
            self.memory_label.config(text=f"{mem_val:.1f}%")
        contradiction_stats = self.dashboard_data.get('contradiction_stats', {})
        if hasattr(self, 'contradictions_label') and self.contradictions_label:
            self.contradictions_label.config(text=str(contradiction_stats.get('total', 0)))
        try:
            hit_rate = float(self.cache_stats.get('hit_rate', 0.0)) * (100.0 if self.cache_stats.get('hit_rate', 0.0) <= 1.5 else 1.0)
        except Exception:
            hit_rate = 0.0
        try:
            util = float(self.cache_stats.get('cache_utilization_percent', 0.0))
            if util <= 1.5:
                util *= 100.0
        except Exception:
            util = 0.0
        try:
            disk_entries = int(self.cache_stats.get('disk_stats', {}).get('entries', 0))
        except Exception:
            disk_entries = 0
        try:
            io_tokens = float(self.resource_snapshot.get('io_tokens', 0.0))
        except Exception:
            io_tokens = 0.0
        if hasattr(self, 'hit_rate_label'):
            self.hit_rate_label.config(text=f"{hit_rate:.1f}%")
        if hasattr(self, 'cache_util_label'):
            self.cache_util_label.config(text=f"{util:.1f}%")
        if hasattr(self, 'disk_entries_label'):
            self.disk_entries_label.config(text=str(disk_entries))
        if hasattr(self, 'io_tokens_label'):
            try:
                if io_tokens >= 1_000_000_000:
                    io_disp = f"{io_tokens/1_000_000_000:.1f}G"
                elif io_tokens >= 1_000_000:
                    io_disp = f"{io_tokens/1_000_000:.1f}M"
                elif io_tokens >= 1_000:
                    io_disp = f"{io_tokens/1_000:.1f}K"
                else:
                    io_disp = f"{int(io_tokens)}"
            except Exception:
                io_disp = "0"
            self.io_tokens_label.config(text=io_disp)

    def _update_status_indicator(self):
        """Обновляет индикатор статуса соединения."""
        try:
            brain_active = False
            if self.brain:
                brain_active = (
                    hasattr(self.brain, 'running') and self.brain.running
                ) or (
                    hasattr(self.brain, 'components') and self.brain.components
                ) or (
                    hasattr(self.brain, 'get_system_status')
                )

            status = "active" if brain_active else "disconnected"
            color = self.colors['success'] if status == "active" else self.colors['danger']
            status_text = f"Соединение: {'активно' if status == 'active' else 'отключено'}"

            if hasattr(self, '_last_status') and self._last_status != status:
                self.chat_logger.info(f"Статус соединения изменился: {self._last_status} -> {status}")
            self._last_status = status

            if hasattr(self, 'status_indicator') and self.status_indicator:
                self.status_indicator.itemconfig("indicator", fill=color)
            if hasattr(self, 'connection_status') and self.connection_status:
                self.connection_status.config(text=status_text)

            if not brain_active and self.brain:
                self.chat_logger.debug(f"Brain найден, но не активен. Свойства brain: {dir(self.brain)[:10]}...")
            elif brain_active:
                self.chat_logger.debug("Brain активен и работает нормально")

        except Exception as e:
            if hasattr(self, 'status_indicator') and self.status_indicator:
                self.status_indicator.itemconfig("indicator", fill=self.colors['danger'])
            if hasattr(self, 'connection_status') and self.connection_status:
                self.connection_status.config(text="Соединение: ошибка")
            self.chat_logger.error(f"Ошибка проверки статуса соединения: {e}")

        self._update_model_loading_indicator()

    def _update_model_loading_indicator(self):
        try:
            if not hasattr(self, 'model_load_label') or not self.model_load_label:
                return
            st = self.model_loading_state
            if not st.get("active"):
                self.model_load_label.config(text="")
                return
            name = st.get("name") or st.get("model_id") or "модель"
            prog = int(st.get("progress") or 0)
            action = st.get("action") or "load"
            if st.get("error"):
                if action == "unload":
                    self.model_load_label.config(text=f"Выгрузка модели '{name}': ошибка")
                else:
                    self.model_load_label.config(text=f"Загрузка модели '{name}': ошибка")
            elif prog >= 100 and action != "unload":
                self.model_load_label.config(text=f"Загрузка модели '{name}': завершено")
            else:
                if action == "unload":
                    self.model_load_label.config(text=f"Выгрузка модели '{name}'...")
                else:
                    self.model_load_label.config(text=f"Загрузка модели '{name}': {prog}%")
        except Exception:
            pass

    def _handle_model_load_event(self, data: Dict[str, Any]):
        """Обрабатывает события загрузки модели из ядра."""
        try:
            event = data.get('event') if isinstance(data, dict) else None
            if not event:
                return
            def apply_update():
                try:
                    if event == 'model_load_start':
                        self.model_loading_state.update({
                            "active": True,
                            "model_id": data.get('model_id'),
                            "name": data.get('name'),
                            "progress": 0,
                            "error": None,
                            "action": "load",
                        })
                    elif event == 'model_load_progress':
                        self.model_loading_state.update({
                            "active": True,
                            "model_id": data.get('model_id'),
                            "progress": max(0, min(100, int(data.get('progress', 0)))),
                            "action": "load",
                        })
                    elif event == 'model_load_complete':
                        self.model_loading_state.update({
                            "active": False,
                            "progress": 100,
                            "error": None,
                            "action": "load",
                        })
                    elif event == 'model_load_error':
                        self.model_loading_state.update({
                            "active": False,
                            "progress": 0,
                            "error": data.get('error') or 'unknown',
                            "action": "load",
                        })
                    elif event == 'model_unload_start':
                        self.model_loading_state.update({
                            "active": True,
                            "model_id": data.get('model_id'),
                            "name": data.get('name'),
                            "progress": 0,
                            "error": None,
                            "action": "unload",
                        })
                    elif event == 'model_unload_complete':
                        self.model_loading_state.update({
                            "active": False,
                            "progress": 0,
                            "error": None,
                            "action": "unload",
                        })
                    elif event == 'model_unload_error':
                        self.model_loading_state.update({
                            "active": False,
                            "error": data.get('error') or 'unknown',
                            "action": "unload",
                        })
                    self._update_model_loading_indicator()
                except tk.TclError:
                    pass
                except Exception:
                    pass
            self.gui_queue.put(apply_update)
        except Exception as e:
            logger.error(f"Error scheduling model ready update: {e}")

    def _handle_models_ready_event(self):
        """Обработка события глобальной готовности моделей."""
        try:
            def apply_ready():
                try:
                    self.model_loading_state.update({
                        "active": False,
                        "progress": 100,
                        "error": None,
                        "action": "load",
                    })
                    self._update_model_loading_indicator()
                    self.show_toast("Модели готовы к работе", "info")
                except Exception:
                    pass
            self.gui_queue.put(apply_ready)
        except Exception:
            pass

    def _handle_notifications(self):
        contradiction_stats = self.dashboard_data.get('contradiction_stats', {})
        high_severity_count = contradiction_stats.get('by_severity', {}).get('high', 0)
        critical_count = contradiction_stats.get('by_severity', {}).get('critical', 0)
        serious_contradictions = high_severity_count + critical_count

        if serious_contradictions > 0:
            self.show_toast(f"Обнаружено {serious_contradictions} серьезных противоречий!", "warning", key="serious_contradictions")

    def update_status(self, status: str, details: Dict[str, Any] = None):
        """Обновляет статус системы в GUI."""
        self.system_status = status
        if details:
            self.dashboard_data.update(details)
        logger.debug(f"Статус обновлен: {status}")

    def show_error(self, title: str, message: str):
        """Показывает сообщение об ошибке."""
        logger.error(f"GUI Error - {title}: {message}")
        if self.root:
            from tkinter import messagebox
            messagebox.showerror(title, message)

    def show_message(self, title: str, message: str, msg_type: str = "info"):
        """Показывает информационное сообщение."""
        logger.info(f"GUI Message - {title}: {message}")
        if self.root:
            from tkinter import messagebox
            if msg_type == "info":
                messagebox.showinfo(title, message)
            elif msg_type == "warning":
                messagebox.showwarning(title, message)
            elif msg_type == "error":
                messagebox.showerror(title, message)

    def show_notification(self, message: str, msg_type: str = "info", actions=None):
        """Показывает уведомление пользователю."""
        logger.info(f"GUI Notification - {msg_type}: {message}")
        if self.root:
            if actions:
                dialog = tk.Toplevel(self.root)
                dialog.title("Уведомление")
                dialog.geometry("400x150")
                dialog.transient(self.root)
                dialog.grab_set()
                
                ttk.Label(dialog, text=message, wraplength=350).pack(pady=20)
                
                btn_frame = ttk.Frame(dialog)
                btn_frame.pack(pady=10)
                
                for action in actions:
                    btn = ttk.Button(btn_frame, text=action.get("text", "OK"), 
                                   command=lambda a=action: [dialog.destroy(), a.get("command", lambda: None)()])
                    btn.pack(side=tk.LEFT, padx=5)
                
                ttk.Button(btn_frame, text="Закрыть", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
            else:
                from tkinter import messagebox
                if msg_type == "info":
                    messagebox.showinfo("Уведомление", message)
                elif msg_type == "warning":
                    messagebox.showwarning("Предупреждение", message)
                elif msg_type == "error":
                    messagebox.showerror("Ошибка", message)
