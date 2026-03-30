"""Neuromorphic Simulator GUI module for ЕВА"""
import tkinter as tk
from tkinter import ttk
import logging

import time
from typing import Any, Dict, Optional, List

logger = logging.getLogger("eva.gui.neuromorphic")

class NeuromorphicModule:
    """GUI module to visualize and control the Neuromorphic Simulator."""

    def __init__(self, gui):
        self.gui = gui
        self.frame: Optional[ttk.Frame] = None
        self.status_labels: Dict[str, ttk.Label] = {}
        self.controls_frame: Optional[ttk.Frame] = None
        self.activity_canvas: Optional[tk.Canvas] = None
        self._update_job_id: Optional[str] = None
        self._last_strengths: List[float] = []
        logger.info("Модуль нейроморфики инициализирован")

    def _safe_brain_call(self, method_name: str, *args, **kwargs):
        """Безопасный вызов метода brain с логированием"""
        if hasattr(self, 'gui') and hasattr(self.gui, 'brain') and self.gui.brain:
            try:
                method = getattr(self.gui.brain, method_name, None)
                if method:
                    result = method(*args, **kwargs)
                    logger.debug(f"[{self.__class__.__name__}] Успешный вызов brain.{method_name}()")
                    return result
                else:
                    logger.warning(f"[{self.__class__.__name__}] Метод brain.{method_name} не найден")
                    return None
            except Exception as e:
                logger.error(f"[{self.__class__.__name__}] Ошибка вызова brain.{method_name}: {e}")
                return None
        else:
            logger.warning(f"[{self.__class__.__name__}] Brain недоступен")
            return None

    # Lifecycle
    def activate(self):
        for w in self.gui.content_area.winfo_children():
            w.destroy()
        self._build_ui()
        self._schedule_update()
        logger.info("Модуль нейроморфики активирован")

    def deactivate(self):
        try:
            if self._update_job_id and self.gui.root:
                self.gui.root.after_cancel(self._update_job_id)
        except Exception:
            pass
        finally:
            self._update_job_id = None
        logger.info("Модуль нейроморфики деактивирован")

    def update(self):
        # Called from core GUI periodic update; we do fine-grained updates via after as well
        self._refresh_metrics()

    # UI construction
    def _build_ui(self):
        self.frame = ttk.Frame(self.gui.content_area)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Frame(self.frame)
        header.pack(fill=tk.X, padx=10, pady=(10, 5))
        ttk.Label(header, text="Нейроморфный симулятор", font=("Segoe UI", 14, "bold"),
                  foreground=self.gui.colors.get("primary", "#0078d7")).pack(side=tk.LEFT)

        # Controls
        self.controls_frame = ttk.Frame(self.frame)
        self.controls_frame.pack(fill=tk.X, padx=10, pady=5)
        self.start_btn = ttk.Button(self.controls_frame, text="Запустить", command=self._on_start)
        self.stop_btn = ttk.Button(self.controls_frame, text="Остановить", command=self._on_stop)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Metrics Grid
        metrics_frame = ttk.LabelFrame(self.frame, text="Статус и метрики")
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)
        rows = [
            ("Доступен", "available"),
            ("Запущен", "running"),
            ("Использует NEST", "use_nest"),
            ("Оценка здоровья", "health_score"),
            ("Состояние здоровья", "health_status"),
            ("Сила взаимодействия", "interaction_strength"),
            ("Всего активностей", "total_activities"),
            ("Время", "timestamp"),
        ]
        for i, (label, key) in enumerate(rows):
            ttk.Label(metrics_frame, text=label + ":").grid(row=i, column=0, sticky=tk.W, padx=8, pady=3)
            val = ttk.Label(metrics_frame, text="--")
            val.grid(row=i, column=1, sticky=tk.W, padx=8, pady=3)
            self.status_labels[key] = val

        # Simple activity visualization
        viz_frame = ttk.LabelFrame(self.frame, text="Последняя активность (средняя сила)")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.activity_canvas = tk.Canvas(viz_frame, height=120, background=self.gui.colors.get("card-bg", "white"),
                                         highlightthickness=1, highlightbackground=self.gui.colors.get("border", "#cccccc"))
        self.activity_canvas.pack(fill=tk.BOTH, expand=True)

        self._sync_controls()

    # Actions
    def _on_start(self):
        sim = getattr(self.gui.brain, 'neuromorphic_simulator', None) if self.gui and self.gui.brain else None
        try:
            if sim and hasattr(sim, 'start'):
                sim.start()
                self.gui.show_notification("Нейроморфный симулятор запущен", "info")
            else:
                self.gui.show_notification("Симулятор недоступен", "error")
        except Exception as e:
            logger.error(f"Ошибка запуска симулятора: {e}", exc_info=True)
            self.gui.show_notification(f"Ошибка запуска: {e}", "error")
        finally:
            self._sync_controls()

    def _on_stop(self):
        sim = getattr(self.gui.brain, 'neuromorphic_simulator', None) if self.gui and self.gui.brain else None
        try:
            if sim and hasattr(sim, 'stop'):
                sim.stop()
                self.gui.show_notification("Нейроморфный симулятор остановлен", "info")
            else:
                self.gui.show_notification("Симулятор недоступен", "error")
        except Exception as e:
            logger.error(f"Ошибка остановки симулятора: {e}", exc_info=True)
            self.gui.show_notification(f"Ошибка остановки: {e}", "error")
        finally:
            self._sync_controls()

    # Data/updates
    def _schedule_update(self, interval_ms: int = 2000):
        try:
            if not self.gui or not self.gui.root or not self.frame or not self.frame.winfo_exists():
                return
            self._refresh_metrics()
            self._update_job_id = self.gui.root.after(interval_ms, lambda: self._schedule_update(interval_ms))
        except Exception:
            self._update_job_id = None
    
    def cleanup(self):
        """Очищает запланированные задачи."""
        try:
            if hasattr(self, '_update_job_id') and self._update_job_id and hasattr(self.gui, 'root') and self.gui.root:
                self.gui.root.after_cancel(self._update_job_id)
                self._update_job_id = None
                logger.debug("Очищена after задача в neuromorphic_module")
        except Exception as e:
            logger.error(f"Ошибка очистки neuromorphic_module: {e}")

    def _refresh_metrics(self):
        metrics = {}
        try:
            if self.gui and self.gui.brain and hasattr(self.gui.brain, 'get_system_metrics'):
                metrics = self.gui.brain.get_system_metrics() or {}
        except Exception as e:
            logger.debug(f"Не удалось получить системные метрики: {e}")

        nm = metrics.get("neuromorphic", {}) if isinstance(metrics, dict) else {}
        # Fallback: if not present, probe simulator
        sim = getattr(self.gui.brain, 'neuromorphic_simulator', None) if self.gui and self.gui.brain else None
        if sim and not nm:
            try:
                nm = {
                    "available": True,
                    "running": bool(getattr(sim, 'running', False)),
                    "use_nest": bool(getattr(sim, 'use_nest', False)),
                }
                if hasattr(sim, 'get_system_health'):
                    h = sim.get_system_health()
                    nm.update({
                        "health_status": h.get("status"),
                        "health_score": h.get("health_score"),
                        "interaction_strength": (h.get("analysis", {}) or {}).get("interaction_strength"),
                        "total_activities": (h.get("analysis", {}) or {}).get("total_activities"),
                        "timestamp": h.get("timestamp"),
                    })
            except Exception:
                pass

        # Update labels
        def fmt_bool(v):
            return "да" if bool(v) else "нет"
        def fmt_num(v):
            try:
                return f"{float(v):.3f}"
            except Exception:
                return "--"

        self._set_label("available", fmt_bool(nm.get("available", False)))
        self._set_label("running", fmt_bool(nm.get("running", False)))
        self._set_label("use_nest", fmt_bool(nm.get("use_nest", False)))
        self._set_label("health_status", str(nm.get("health_status", "--")))
        self._set_label("health_score", fmt_num(nm.get("health_score")) if nm.get("health_score") is not None else "--")
        self._set_label("interaction_strength", fmt_num(nm.get("interaction_strength")) if nm.get("interaction_strength") is not None else "--")
        self._set_label("total_activities", str(nm.get("total_activities", "--")))
        ts = nm.get("timestamp")
        self._set_label("timestamp", time.strftime("%H:%M:%S", time.localtime(ts)) if isinstance(ts, (int, float)) else "--")

        # Update simple activity visualization using recent strengths
        try:
            strengths = self._collect_recent_strengths(sim)
            if strengths:
                self._last_strengths = strengths[-100:]
            self._draw_strengths(self._last_strengths)
        except Exception:
            pass

        self._sync_controls()

    def _set_label(self, key: str, value: str):
        lbl = self.status_labels.get(key)
        if lbl and lbl.winfo_exists():
            lbl.config(text=value)

    def _sync_controls(self):
        sim = getattr(self.gui.brain, 'neuromorphic_simulator', None) if self.gui and self.gui.brain else None
        running = bool(getattr(sim, 'running', False)) if sim else False
        try:
            if self.start_btn:
                self.start_btn.configure(state=(tk.DISABLED if running else tk.NORMAL))
            if self.stop_btn:
                self.stop_btn.configure(state=(tk.NORMAL if running else tk.DISABLED))
        except Exception:
            pass

    def _collect_recent_strengths(self, sim) -> List[float]:
        strengths: List[float] = []
        if not sim:
            return strengths
        try:
            hist = getattr(sim, 'activity_history', [])
            # Average strength per recorded activity
            for act in hist[-100:]:
                s = getattr(act, 'strength', None)
                if s is None:
                    # Derive from activity pattern if available
                    patt = getattr(act, 'activity_pattern', None)
                    if isinstance(patt, list) and patt:
                        try:
                            s = sum(patt) / max(1, len(patt))
                        except Exception:
                            s = None
                if isinstance(s, (int, float)):
                    strengths.append(float(s))
        except Exception:
            pass
        return strengths

    def _draw_strengths(self, values: List[float]):
        canvas = self.activity_canvas
        if not canvas or not canvas.winfo_exists():
            return
        try:
            canvas.delete("all")
            w = canvas.winfo_width() or 200
            h = canvas.winfo_height() or 120
            if not values:
                # Draw placeholder grid
                canvas.create_text(w//2, h//2, text="Нет данных", fill=self.gui.colors.get("text-muted", "#666"))
                return
            n = len(values)
            max_v = max(max(values), 1e-6)
            min_v = min(min(values), 0.0)
            rng = max(max_v - min_v, 1e-6)
            # Padding
            px = 8
            py = 6
            prev = None
            for i, v in enumerate(values):
                x = px + (w - 2*px) * (i / max(1, n-1))
                y = h - py - (h - 2*py) * ((v - min_v) / rng)
                if prev is not None:
                    canvas.create_line(prev[0], prev[1], x, y, fill=self.gui.colors.get("primary", "#0078d7"), width=2)
                prev = (x, y)
            # Draw axes
            canvas.create_line(px, py, px, h - py, fill=self.gui.colors.get("border", "#ccc"))
            canvas.create_line(px, h - py, w - px, h - py, fill=self.gui.colors.get("border", "#ccc"))
        except Exception:
            pass
