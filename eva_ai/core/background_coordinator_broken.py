
"""
BackgroundCoordinator: Автопилот для CoreBrain
- Планировщик фоновых задач с учётом простоя пользователя и ресурсов
- Поддержка политик (пороги CPU/RAM, кулдауны), детекторов возможностей и job-плагинов
- Интеграция с deferred_command_system при наличии
"""
from __future__ import annotations
import threading
import time
import logging
import os
from collections import deque
from typing import Dict, List, Optional, Type, Callable, Any, Deque

try:
    import psutil  # для оценки ресурсов при отсутствии ResourceManager
except Exception:  # noqa
    psutil = None  # type: ignore

logger = logging.getLogger("eva_ai.core.autopilot")
timeline_logger = logging.getLogger("eva_ai.core.autopilot.timeline")


class Policies:
    def __init__(
        self,
        idle_threshold_s: float = 10.0,
        cpu_threshold_soft: float = 0.80,
        cpu_threshold_hard: float = 0.90,
        ram_threshold_soft: float = 0.90,
        ram_threshold_hard: float = 0.95,
        tick_interval_s: float = 3.0,
        concurrency: Optional[Dict[str, int]] = None,
        cooldowns: Optional[Dict[str, float]] = None,
    ) -> None:
        self.idle_threshold_s = idle_threshold_s
        self.cpu_threshold_soft = cpu_threshold_soft
        self.cpu_threshold_hard = cpu_threshold_hard
        self.ram_threshold_soft = ram_threshold_soft
        self.ram_threshold_hard = ram_threshold_hard
        self.tick_interval_s = tick_interval_s
        self.concurrency = concurrency or {"CPU": 2, "GPU": 1, "IO": 4}
        self.cooldowns = cooldowns or {
            "training": 60.0,
            "import": 30.0,
            "consolidation": 120.0,
            "contradiction": 120.0,
            "cache_gc": 300.0,
        }


class BackgroundCoordinator:
    """Координатор фоновой активности CoreBrain."""

    def __init__(
        self,
        brain: Any,
        deferred_system: Optional[Any] = None,
        resource_manager: Optional[Any] = None,
        metrics_manager: Optional[Any] = None,
        state_manager: Optional[Any] = None,
        policies: Optional[Policies] = None,
    ) -> None:
        self.brain = brain
        self.deferred = deferred_system
        self.rm = resource_manager
        self.mm = metrics_manager
        self.sm = state_manager
        self.policies = policies or Policies()
        # попытка получить кэш автопилота у мозга (может быть установлен позже)
        self.cache = getattr(brain, 'autopilot_cache', None)

        # Файловый лог автопилота
        try:
            # База логов: сначала brain.cache_dir, иначе корень проекта + eva_cache
            cache_dir = getattr(brain, 'cache_dir', None)
            if not cache_dir:
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
                cache_dir = os.path.join(project_root, 'eva_cache')
            ap_dir = os.path.join(cache_dir, 'autopilot')
            os.makedirs(ap_dir, exist_ok=True)
            log_path = os.path.join(ap_dir, 'autopilot.log')
            # не дублируем хендлеры между экземплярами
            if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '').endswith('autopilot.log') for h in logger.handlers):
                fh = logging.FileHandler(log_path, encoding='utf-8')
                fh.setLevel(logging.DEBUG)
                fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
                fh.setFormatter(fmt)
                logger.addHandler(fh)
                logger.debug(f"Autopilot file logging initialized at {log_path}")
        except Exception:
            # не мешаем работе при ошибке логирования
            pass

        self._detectors: List[Any] = []
        self._job_types: Dict[str, Type[Any]] = {}
        self._last_user_activity_ts: float = time.time()
        self._running = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        # Таймлайн планировщика: фиксируем последовательность событий для отладки
        self._timeline: Deque[Dict[str, Any]] = deque(maxlen=2000)
        self._seq: int = 0

        # учёт активных задач по ресурсным классам
        self._active_counts: Dict[str, int] = {k: 0 for k in self.policies.concurrency.keys()}

        # Кулдауны по типам задач (секунды). Можно переопределить через policies.job_cooldowns_s
        default_cooldowns = {
            'TrainingJob': 1200,      # 20 минут
            'WebIndexJob': 300,       # 5 минут
            'ModuleRecoveryJob': 600, # 10 минут
        }
        self.job_cooldowns_s: Dict[str, int] = getattr(self.policies, 'job_cooldowns_s', default_cooldowns)
        self._last_run_ts: Dict[str, float] = {}

        # Crash-loop защита: счётчик неудач и окно бэкоффа на тип задачи
        self._job_failures: Dict[str, int] = {}
        self._job_next_allowed_ts: Dict[str, float] = {}
        # Гард на повторное планирование одного и того же типа, пока он в работе
        self._job_pending: Dict[str, bool] = {}
        # Дополнительно: запрет на планирование до определённого времени (после фейла)
        self._job_pending_until_ts: Dict[str, float] = {}
        # Явный флаг бэкоффа, блокирующий планирование до истечения окна
        self._job_backing_off: Dict[str, bool] = {}

    # ---- Публичный API ----
    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._paused = False
            self._thread = threading.Thread(target=self._run_loop, name="BackgroundCoordinator", daemon=True)
            self._thread.start()
            logger.info("BackgroundCoordinator started")

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        logger.info("BackgroundCoordinator stopped")

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    def signal_user_activity(self, ts: Optional[float] = None) -> None:
        self._last_user_activity_ts = ts or time.time()

    def register_detector(self, detector: Any) -> None:
        self._detectors.append(detector)

    def register_job_type(self, job_type: Type[Any]) -> None:
        self._job_types[getattr(job_type, "job_type", job_type.__name__)] = job_type

    # ---- Основной цикл ----
    def _run_loop(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    break
                paused = self._paused
            if not paused:
                try:
                    self._tick()
                except Exception as e:
                    logger.error(f"Background tick error: {e}", exc_info=True)
            time.sleep(self.policies.tick_interval_s)

    # ---- Один тик планировщика ----
    def _tick(self) -> None:
        if not self._can_run_background():
            logger.debug("Tick skipped: background not allowed by policies/idle/resources")
            return
        context = self._build_context()
        # Логируем тик в таймлайн
        try:
            with self._lock:
                self._seq += 1
                seq = self._seq
            rec = {"type": "tick", "seq": seq, "ts": time.time()}
            self._timeline.append(rec)
            try:
                timeline_logger.debug(rec)
            except Exception:
                pass
        except Exception:
            pass
        if self.cache:
            try:
                self.cache.append_event("tick", {"idle_for": time.time() - self._last_user_activity_ts})
            except Exception:
                pass
        logger.debug("Tick: probing detectors (%d)" % len(self._detectors))
        for det in list(self._detectors):
            try:
                job_requests = det.probe(context) or []
                if job_requests:
                    logger.debug(f"Detector {getattr(det, 'name', det)} produced {len(job_requests)} job(s)")
            except Exception as e:
                logger.warning(f"Detector {getattr(det, 'name', det)} failed: {e}")
                if self.mm and hasattr(self.mm, 'record_error'):
                    try:
                        self.mm.record_error("autopilot_detector_error")
                    except Exception:
                        pass
                continue
            for req in job_requests:
                # помечаем источник детектирования для логов
                try:
                    if isinstance(req, dict) and 'source' not in req:
                        req['source'] = getattr(det, 'name', det.__class__.__name__)
                except Exception:
                    pass
                self._schedule_job(req, context)

    # ---- Проверки условий ----
    def _can_run_background(self) -> bool:
        # idle user check
        idle_for = time.time() - self._last_user_activity_ts
        if idle_for < self.policies.idle_threshold_s:
            logger.debug(f"Background blocked: user not idle (idle_for={idle_for:.2f}s < {self.policies.idle_threshold_s}s)")
            return False
        # resource check
        cpu, ram = self._get_resource_usage()
        # жёсткие пороги
        if cpu is not None and cpu >= self.policies.cpu_threshold_hard:
            logger.debug(f"Background blocked: CPU {cpu:.2f} >= hard {self.policies.cpu_threshold_hard:.2f}")
            return False
        if ram is not None and ram >= self.policies.ram_threshold_hard:
            logger.debug(f"Background blocked: RAM {ram:.2f} >= hard {self.policies.ram_threshold_hard:.2f}")
            return False
        return True

    def _get_resource_usage(self) -> (Optional[float], Optional[float]):
        # Пытаемся использовать ResourceManager/metrics, иначе psutil
        try:
            if self.rm and hasattr(self.rm, 'get_cpu_usage') and hasattr(self.rm, 'get_memory_usage'):
                cpu = float(self.rm.get_cpu_usage())  # 0..1
                mem = float(self.rm.get_memory_usage())  # 0..1
                return cpu, mem
        except Exception:
            pass
        if psutil:
            try:
                cpu = psutil.cpu_percent(interval=0.0) / 100.0
                mem = psutil.virtual_memory().percent / 100.0
                return cpu, mem
            except Exception:
                return None, None
        return None, None

    def _build_context(self) -> Dict[str, Any]:
        def should_pause() -> bool:
            # мягкие пороги: дросселируем
            cpu, ram = self._get_resource_usage()
            if cpu is not None and cpu >= self.policies.cpu_threshold_soft:
                return True
            if ram is not None and ram >= self.policies.ram_threshold_soft:
                return True
            # активность пользователя
            if (time.time() - self._last_user_activity_ts) < self.policies.idle_threshold_s:
                return True
            return False

        return {
            "brain": self.brain,
            "policies": self.policies,
            "should_pause": should_pause,
            "now": time.time(),
        }

    # ---- Планирование задач ----
    def _schedule_job(self, req: Dict[str, Any], context: Dict[str, Any]) -> None:
        job_type_name: str = req.get("job_type")
        if not job_type_name or job_type_name not in self._job_types:
            logger.debug(f"Unknown job type: {job_type_name}")
            return
        job_cls = self._job_types[job_type_name]
        resource_class: str = getattr(job_cls, 'resource_class', 'CPU')
        # Зафиксируем попытку планирования
        try:
            with self._lock:
                self._seq += 1
                seq = self._seq
            rec = {
                "type": "schedule_attempt",
                "seq": seq,
                "ts": time.time(),
                "job": job_type_name,
                "resource": resource_class,
                "active": self._active_counts.get(resource_class, 0),
                "limit": self.policies.concurrency.get(resource_class, 1),
            }
            self._timeline.append(rec)
            try:
                timeline_logger.debug(rec)
            except Exception:
                pass
        except Exception:
            pass
        # ограничение конкуренции (под защитой lock)
        with self._lock:
            if self._active_counts.get(resource_class, 0) >= self.policies.concurrency.get(resource_class, 1):
                logger.debug(f"Concurrency limit reached for {resource_class}")
                # Логируем отказ в планировании по лимиту
                try:
                    with self._lock:
                        self._seq += 1
                        seq = self._seq
                    rec = {"type": "schedule_reject_limit", "seq": seq, "ts": time.time(),
                           "job": job_type_name, "resource": resource_class,
                           "active": self._active_counts.get(resource_class, 0),
                           "limit": self.policies.concurrency.get(resource_class, 1)}
                    self._timeline.append(rec)
                    timeline_logger.debug(rec)
                except Exception:
                    pass
                return
            # ранняя проверка явного бэкофф-флага
            if self._job_backing_off.get(job_type_name, False):
                logger.debug(f"Job {job_type_name} is in explicit backoff state; skip scheduling")
                return

        # мягкие пороги: если дросселируем — не планируем
        try:
            sp = context.get("should_pause")
            if callable(sp) and sp():
                logger.debug("Scheduling throttled by soft thresholds or user activity (should_pause=True)")
                # Логируем отказ по паузе
                try:
                    with self._lock:
                        self._seq += 1
                        seq = self._seq
                    rec = {"type": "schedule_reject_pause", "seq": seq, "ts": time.time(),
                           "job": job_type_name, "resource": resource_class}
                    self._timeline.append(rec)
                    timeline_logger.debug(rec)
                except Exception:
                    pass
                return
        except Exception:
            pass

        # Полная проверка условий и атомарная установка pending
        now = time.time()
        set_pending = False
        with self._lock:
            # повторная проверка конкуренции на случай параллельных обновлений
            if self._active_counts.get(resource_class, 0) >= self.policies.concurrency.get(resource_class, 1):
                logger.debug(f"Concurrency limit reached for {resource_class}")
                return
            if self._job_pending.get(job_type_name, False):
                logger.debug(f"Job type {job_type_name} already pending; skip reschedule")
                try:
                    with self._lock:
                        self._seq += 1
                        seq = self._seq
                    rec = {"type": "schedule_reject_pending", "seq": seq, "ts": time.time(),
                           "job": job_type_name, "resource": resource_class}
                    self._timeline.append(rec)
                    timeline_logger.debug(rec)
                except Exception:
                    pass
                return
            hold_until = float(self._job_pending_until_ts.get(job_type_name, 0.0))
            next_allowed = float(self._job_next_allowed_ts.get(job_type_name, 0.0))
            last = self._last_run_ts.get(job_type_name, 0.0)
            # hold/backoff/cooldown enforcement
            if time.time() < hold_until:
                remain = hold_until - time.time()
                logger.debug(f"Job type {job_type_name} held until backoff expires: {remain:.1f}s remaining")
                try:
                    with self._lock:
                        self._seq += 1
                        seq = self._seq
                    rec = {"type": "schedule_reject_hold", "seq": seq, "ts": time.time(),
                           "job": job_type_name, "resource": resource_class, "remain_s": round(remain, 3)}
                    self._timeline.append(rec)
                    timeline_logger.debug(rec)
                except Exception:
                    pass
                return
            if now < next_allowed:
                remain = next_allowed - now
                logger.debug(f"Backoff active for {job_type_name}: {remain:.1f}s remaining")
                try:
                    with self._lock:
                        self._seq += 1
                        seq = self._seq
                    rec = {"type": "schedule_reject_backoff", "seq": seq, "ts": time.time(),
                           "job": job_type_name, "resource": resource_class, "remain_s": round(remain, 3)}
                    self._timeline.append(rec)
                    timeline_logger.debug(rec)
                except Exception:
                    pass
                return
            cooldown = float(self.job_cooldowns_s.get(job_type_name, 0))
            if cooldown > 0 and (now - last) < cooldown:
                remain = cooldown - (now - last)
                logger.debug(f"Cooldown active for {job_type_name}: {remain:.1f}s remaining")
                try:
                    with self._lock:
                        self._seq += 1
                        seq = self._seq
                    rec = {"type": "schedule_reject_cooldown", "seq": seq, "ts": time.time(),
                           "job": job_type_name, "resource": resource_class, "remain_s": round(remain, 3)}
                    self._timeline.append(rec)
                    timeline_logger.debug(rec)
                except Exception:
                    pass
                return
            # Все проверки пройдены — помечаем pending атомарно с проверками
            self._job_pending[job_type_name] = True
            set_pending = True

        job = job_cls(self.brain, **req.get("params", {}))
        logger.debug(
            "Scheduling job %s (resource=%s) with params=%s source=%s active[%s]=%d",
            job_type_name, resource_class, req.get('params', {}), req.get('source'),
            resource_class, self._active_counts.get(resource_class, 0)
        )
        # Логируем успешное планирование
        try:
            with self._lock:
                self._seq += 1
                seq = self._seq
            rec = {"type": "scheduled", "seq": seq, "ts": time.time(),
                   "job": job_type_name, "resource": resource_class, "params": req.get('params', {}),
                   "active": self._active_counts.get(resource_class, 0)}
            self._timeline.append(rec)
            timeline_logger.debug(rec)
        except Exception:
            pass

        def _on_start():
            with self._lock:
                self._active_counts[resource_class] = self._active_counts.get(resource_class, 0) + 1

        def _on_done(failed: bool):
            # always decrement resource usage
            with self._lock:
                self._active_counts[resource_class] = max(0, self._active_counts.get(resource_class, 0) - 1)
                hold_until = float(self._job_pending_until_ts.get(job_type_name, 0.0))
                now_local = time.time()
                if failed and now_local < hold_until:
                    # keep pending during backoff and schedule a delayed clear
                    self._job_pending[job_type_name] = True
                    delay = max(0.0, hold_until - now_local)
                    def _clear_pending_later():
                        time.sleep(delay)
                        with self._lock:
                            # clear only if hold period actually elapsed and still pending
                            if time.time() >= float(self._job_pending_until_ts.get(job_type_name, 0.0)):
                                self._job_pending[job_type_name] = False
                    t_clear = threading.Thread(target=_clear_pending_later, daemon=True)
                    t_clear.start()
                else:
                    # success or no backoff window -> clear pending immediately
                    self._job_pending[job_type_name] = False

        def _runner():
            failed = False
            try:
                _on_start()
                # Зафиксировать старт
                t_start = time.perf_counter()
                try:
                    with self._lock:
                        self._seq += 1
                        seq = self._seq
                    rec = {"type": "job_start", "seq": seq, "ts": time.time(),
                           "job": job_type_name, "resource": resource_class,
                           "active_after_inc": self._active_counts.get(resource_class, 0)}
                    self._timeline.append(rec)
                    timeline_logger.debug(rec)
                except Exception:
                    pass
                # фиксируем время запуска для кулдауна
                try:
                    with self._lock:
                        self._last_run_ts[job_type_name] = time.time()
                except Exception:
                    pass
                if self.cache:
                    try:
                        self.cache.append_event("job_start", {"job": job_type_name, "resource": resource_class, "params": req.get('params', {})})
                    except Exception:
                        pass
                job.run(context)
            except Exception as e:
                logger.error(f"Job {job_type_name} failed: {e}", exc_info=True)
                failed = True
                if self.mm and hasattr(self.mm, 'record_error'):
                    try:
                        self.mm.record_error(f"autopilot_job_failed_{job_type_name}")
                    except Exception:
                        pass
                # crash-loop backoff: увеличиваем счётчик и задаём экспоненциальную задержку
                try:
                    with self._lock:
                        fails = int(self._job_failures.get(job_type_name, 0)) + 1
                        self._job_failures[job_type_name] = fails
                        backoff = min(300.0, 1.0 * (2 ** max(0, fails - 1)))
                        until = time.time() + backoff
                        self._job_next_allowed_ts[job_type_name] = until
                        # Удерживаем тип задачи в состоянии pending до истечения backoff
                        self._job_pending_until_ts[job_type_name] = until
                        # Включаем явный флаг бэкоффа и планируем его сброс
                        self._job_backing_off[job_type_name] = True
                    def _clear_backoff_later(delay: float, job=job_type_name):
                        time.sleep(delay)
                        with self._lock:
                            # снимаем флаг только если окно действительно истекло
                            if time.time() >= float(self._job_next_allowed_ts.get(job, 0.0)):
                                self._job_backing_off[job] = False
                    t_boff = threading.Thread(target=_clear_backoff_later, args=(backoff,), daemon=True)
                    t_boff.start()
                    logger.debug(f"Set backoff for {job_type_name}: {backoff:.1f}s after {fails} failures")
                except Exception:
                    pass
            finally:
                _on_done(failed)
                # Зафиксировать завершение + длительность
                try:
                    dur = time.perf_counter() - t_start
                except Exception:
                    dur = 0.0
                try:
                    with self._lock:
                        self._seq += 1
                        seq = self._seq
                    rec = {"type": "job_done", "seq": seq, "ts": time.time(),
                           "job": job_type_name, "resource": resource_class,
                           "failed": bool(failed), "duration_s": round(dur, 6),
                           "active_after_dec": self._active_counts.get(resource_class, 0)}
                    self._timeline.append(rec)
                    timeline_logger.debug(rec)
                except Exception:
                    pass
                if self.cache:
                    try:
                        self.cache.append_event("job_done", {"job": job_type_name, "resource": resource_class})
                    except Exception:
                        pass
                # сбрасываем состояния при успешном завершении
                try:
                    if not failed:
                        with self._lock:
                            self._job_failures[job_type_name] = 0
                            self._job_next_allowed_ts[job_type_name] = 0.0
                            self._job_pending_until_ts[job_type_name] = 0.0
                except Exception:
                    pass

        # если есть deferred_system — используем его, иначе поток
        try:
            if self.deferred and hasattr(self.deferred, 'add_command'):
                priority = getattr(job_cls, 'default_priority', None)
                self.deferred.add_command(command=lambda: _runner(), args=(), kwargs={}, priority=priority)
            else:
                t = threading.Thread(target=_runner, daemon=True)
                t.start()
        except Exception as e:
            logger.error(f"Failed to dispatch job {job_type_name}: {e}", exc_info=True)
            # откатываем pending, так как запуск не состоялся
            with self._lock:
                self._job_pending[job_type_name] = False
            return

    # ---- Публичный доступ к таймлайну ----
    def get_timeline(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            if not limit or limit <= 0:
                return list(self._timeline)
            return list(self._timeline)[-int(limit):]
        except Exception:
            return []

    # ---- Проверки условий ----
    def _can_run_background(self) -> bool:
        # idle user check
        idle_for = time.time() - self._last_user_activity_ts
        if idle_for < self.policies.idle_threshold_s:
            logger.debug(f"Background blocked: user not idle (idle_for={idle_for:.2f}s < {self.policies.idle_threshold_s}s)")
            return False
        # resource check
        cpu, ram = self._get_resource_usage()
        # жёсткие пороги
        if cpu is not None and cpu >= self.policies.cpu_threshold_hard:
            logger.debug(f"Background blocked: CPU {cpu:.2f} >= hard {self.policies.cpu_threshold_hard:.2f}")
            return False
        if ram is not None and ram >= self.policies.ram_threshold_hard:
            logger.debug(f"Background blocked: RAM {ram:.2f} >= hard {self.policies.ram_threshold_hard:.2f}")
            return False
        return True

    def _get_resource_usage(self) -> (Optional[float], Optional[float]):
        # Пытаемся использовать ResourceManager/metrics, иначе psutil
        try:
            if self.rm and hasattr(self.rm, 'get_cpu_usage') and hasattr(self.rm, 'get_memory_usage'):
                cpu = float(self.rm.get_cpu_usage())  # 0..1
                mem = float(self.rm.get_memory_usage())  # 0..1
                return cpu, mem
        except Exception as e:
            logger.debug(f"Ошибка получения ресурсов из ResourceManager: {e}")
        if psutil:
            try:
                cpu = psutil.cpu_percent(interval=0.0) / 100.0
                mem = psutil.virtual_memory().percent / 100.0
                return cpu, mem
            except Exception as e:
                logger.debug(f"Ошибка получения ресурсов из psutil: {e}")
                return None, None
        return None, None

    def _build_context(self) -> Dict[str, Any]:
        def should_pause() -> bool:
            # мягкие пороги: дросселируем
            cpu, ram = self._get_resource_usage()
            if cpu is not None and cpu >= self.policies.cpu_threshold_soft:
                return True
            if ram is not None and ram >= self.policies.ram_threshold_soft:
                return True
            # активность пользователя
            if (time.time() - self._last_user_activity_ts) < self.policies.idle_threshold_s:
                return True
            return False

        return {
            "brain": self.brain,
            "policies": self.policies,
            "should_pause": should_pause,
            "now": time.time(),
        }

    # ---- Планирование задач ----
    def _schedule_job(self, req: Dict[str, Any], context: Dict[str, Any]) -> None:
        job_type_name: str = req.get("job_type")
        if not job_type_name or job_type_name not in self._job_types:
            logger.debug(f"Unknown job type: {job_type_name}")
            return
        job_cls = self._job_types[job_type_name]
        resource_class: str = getattr(job_cls, 'resource_class', 'CPU')
        # Зафиксируем попытку планирования
        try:
            with self._lock:
                self._seq += 1
                seq = self._seq
            rec = {
                "type": "schedule_attempt",
                "seq": seq,
                "ts": time.time(),
                "job": job_type_name,
                "resource": resource_class,
                "active": self._active_counts.get(resource_class, 0),
                "limit": self.policies.concurrency.get(resource_class, 1),
            }
            self._timeline.append(rec)
            try:
                timeline_logger.debug(rec)
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Timeline logging error: {e}")
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.debug(f"Error in concurrency check: {e}")
        # ограничение конкуренции (под защитой lock)
        with self._lock:
            if self._active_counts.get(resource_class, 0) >= self.policies.concurrency.get(resource_class, 1):
                logger.debug(f"Concurrency limit reached for {resource_class}")
                # Логируем отказ в планировании по лимиту
                try:
                    with self._lock:
                        self._seq += 1
                        seq = self._seq
                    rec = {"type": "schedule_reject_limit", "seq": seq, "ts": time.time(),
                           "job": job_type_name, "resource": resource_class,
                           "active": self._active_counts.get(resource_class, 0),
                           "limit": self.policies.concurrency.get(resource_class, 1)}
                    self._timeline.append(rec)
                    timeline_logger.debug(rec)
                except (AttributeError, TypeError, ValueError) as e:
                    logger.debug(f"Timeline logging error in schedule_reject_limit: {e}")
                return
            # ранняя проверка явного бэкофф-флага
            if self._job_backing_off.get(job_type_name, False):
                logger.debug(f"Job {job_type_name} is in explicit backoff state; skip scheduling")
                return

        # мягкие пороги: если дросселируем — не планируем
        try:
            sp = context.get("should_pause")
            if callable(sp) and sp():
                logger.debug("Scheduling throttled by soft thresholds or user activity (should_pause=True)")
                # Логируем отказ по паузе
                try:
                    with self._lock:
                        self._seq += 1
                        seq = self._seq
                    rec = {"type": "schedule_reject_pause", "seq": seq, "ts": time.time(),
                           "job": job_type_name, "resource": resource_class}
                    self._timeline.append(rec)
                    timeline_logger.debug(rec)
                except (AttributeError, TypeError, ValueError) as e:
                    logger.debug(f"Timeline logging error in schedule_reject_pause: {e}")
                return
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.debug(f"Error in should_pause check: {e}")

        # Полная проверка условий и атомарная установка pending
        now = time.time()
        set_pending = False
        with self._lock:
            # повторная проверка конкуренции на случай параллельных обновлений
            if self._active_counts.get(resource_class, 0) >= self.policies.concurrency.get(resource_class, 1):
                logger.debug(f"Concurrency limit reached for {resource_class}")
                return
            if self._job_pending.get(job_type_name, False):
                logger.debug(f"Job type {job_type_name} already pending; skip reschedule")
                try:
                    with self._lock:
                        self._seq += 1
                        seq = self._seq
                    rec = {"type": "schedule_reject_pending", "seq": seq, "ts": time.time(),
                           "job": job_type_name, "resource": resource_class}
                    self._timeline.append(rec)
                    timeline_logger.debug(rec)
                except (AttributeError, TypeError, ValueError) as e:
                    logger.debug(f"Timeline logging error in schedule_reject_pending: {e}")
                return
            hold_until = float(self._job_pending_until_ts.get(job_type_name, 0.0))
            next_allowed = float(self._job_next_allowed_ts.get(job_type_name, 0.0))
            last = self._last_run_ts.get(job_type_name, 0.0)
            # hold/backoff/cooldown enforcement
            if time.time() < hold_until:
                remain = hold_until - time.time()
                logger.debug(f"Job type {job_type_name} held until backoff expires: {remain:.1f}s remaining")
                try:
                    with self._lock:
                        self._seq += 1
                        seq = self._seq
                    rec = {"type": "schedule_reject_hold", "seq": seq, "ts": time.time(),
                           "job": job_type_name, "resource": resource_class, "remain_s": round(remain, 3)}
                    self._timeline.append(rec)
                    timeline_logger.debug(rec)
                except (AttributeError, TypeError, ValueError) as e:
                    logger.debug(f"Timeline logging error in schedule_reject_hold: {e}")
                return
            if now < next_allowed:
                remain = next_allowed - now
                logger.debug(f"Backoff active for {job_type_name}: {remain:.1f}s remaining")
                try:
                    with self._lock:
                        self._seq += 1
                        seq = self._seq
                    rec = {"type": "schedule_reject_backoff", "seq": seq, "ts": time.time(),
                           "job": job_type_name, "resource": resource_class, "remain_s": round(remain, 3)}
                    self._timeline.append(rec)
                    timeline_logger.debug(rec)
                except (AttributeError, TypeError, ValueError) as e:
                    logger.debug(f"Timeline logging error in schedule_reject_backoff: {e}")
                return
            cooldown = float(self.job_cooldowns_s.get(job_type_name, 0))
            if cooldown > 0 and (now - last) < cooldown:
                remain = cooldown - (now - last)
                logger.debug(f"Cooldown active for {job_type_name}: {remain:.1f}s remaining")
                try:
                    with self._lock:
                        self._seq += 1
                        seq = self._seq
                    rec = {"type": "schedule_reject_cooldown", "seq": seq, "ts": time.time(),
                           "job": job_type_name, "resource": resource_class, "remain_s": round(remain, 3)}
                    self._timeline.append(rec)
                    timeline_logger.debug(rec)
                except (AttributeError, TypeError, ValueError) as e:
                    logger.debug(f"Timeline logging error in schedule_reject_cooldown: {e}")
                return
            # Все проверки пройдены — помечаем pending атомарно с проверками
            self._job_pending[job_type_name] = True
            set_pending = True

        job = job_cls(self.brain, **req.get("params", {}))
        logger.debug(
            "Scheduling job %s (resource=%s) with params=%s source=%s active[%s]=%d",
            job_type_name, resource_class, req.get('params', {}), req.get('source'),
            resource_class, self._active_counts.get(resource_class, 0)
        )
        # Логируем успешное планирование
        try:
            with self._lock:
                self._seq += 1
                seq = self._seq
            rec = {"type": "scheduled", "seq": seq, "ts": time.time(),
                   "job": job_type_name, "resource": resource_class, "params": req.get('params', {}),
                   "active": self._active_counts.get(resource_class, 0)}
            self._timeline.append(rec)
            timeline_logger.debug(rec)
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"Timeline logging error in scheduled: {e}")

        def _on_start():
            with self._lock:
                self._active_counts[resource_class] = self._active_counts.get(resource_class, 0) + 1

        def _on_done(failed: bool):
            # always decrement resource usage
            with self._lock:
                self._active_counts[resource_class] = max(0, self._active_counts.get(resource_class, 0) - 1)
                hold_until = float(self._job_pending_until_ts.get(job_type_name, 0.0))
                now_local = time.time()
                if failed and now_local < hold_until:
                    # keep pending during backoff and schedule a delayed clear
                    self._job_pending[job_type_name] = True
                    delay = max(0.0, hold_until - now_local)
                    def _clear_pending_later():
                        time.sleep(delay)
                        with self._lock:
                            # clear only if hold period actually elapsed and still pending
                            if time.time() >= float(self._job_pending_until_ts.get(job_type_name, 0.0)):
                                self._job_pending[job_type_name] = False
                    t_clear = threading.Thread(target=_clear_pending_later, daemon=True)
                    t_clear.start()
                else:
                    # success or no backoff window -> clear pending immediately
                    self._job_pending[job_type_name] = False

        def _runner():
            failed = False
            try:
                _on_start()
                # Зафиксировать старт
                t_start = time.perf_counter()
                try:
                    with self._lock:
                        self._seq += 1
                        seq = self._seq
                    rec = {"type": "job_start", "seq": seq, "ts": time.time(),
                           "job": job_type_name, "resource": resource_class,
                           "active_after_inc": self._active_counts.get(resource_class, 0)}
                    self._timeline.append(rec)
                    timeline_logger.debug(rec)
                except Exception:
                    pass
                # фиксируем время запуска для кулдауна
                try:
                    with self._lock:
                        self._last_run_ts[job_type_name] = time.time()
                except Exception:
                    pass
                if self.cache:
                    try:
                        self.cache.append_event("job_start", {"job": job_type_name, "resource": resource_class, "params": req.get('params', {})})
                    except Exception:
                        pass
                job.run(context)
            except Exception as e:
                logger.error(f"Job {job_type_name} failed: {e}", exc_info=True)
                failed = True
                if self.mm and hasattr(self.mm, 'record_error'):
                    try:
                        self.mm.record_error(f"autopilot_job_failed_{job_type_name}")
                    except Exception:
                        pass
                # crash-loop backoff: увеличиваем счётчик и задаём экспоненциальную задержку
                try:
                    with self._lock:
                        fails = int(self._job_failures.get(job_type_name, 0)) + 1
                        self._job_failures[job_type_name] = fails
                        backoff = min(300.0, 1.0 * (2 ** max(0, fails - 1)))
                        until = time.time() + backoff
                        self._job_next_allowed_ts[job_type_name] = until
                        # Удерживаем тип задачи в состоянии pending до истечения backoff
                        self._job_pending_until_ts[job_type_name] = until
                        # Включаем явный флаг бэкоффа и планируем его сброс
                        self._job_backing_off[job_type_name] = True
                    def _clear_backoff_later(delay: float, job=job_type_name):
                        time.sleep(delay)
                        with self._lock:
                            # снимаем флаг только если окно действительно истекло
                            if time.time() >= float(self._job_next_allowed_ts.get(job, 0.0)):
                                self._job_backing_off[job] = False
                    t_boff = threading.Thread(target=_clear_backoff_later, args=(backoff,), daemon=True)
                    t_boff.start()
                    logger.debug(f"Set backoff for {job_type_name}: {backoff:.1f}s after {fails} failures")
                except Exception:
                    pass
            finally:
                _on_done(failed)
                # Зафиксировать завершение + длительность
                try:
                    dur = time.perf_counter() - t_start
                except Exception:
                    dur = 0.0
                try:
                    with self._lock:
                        self._seq += 1
                        seq = self._seq
                    rec = {"type": "job_done", "seq": seq, "ts": time.time(),
                           "job": job_type_name, "resource": resource_class,
                           "failed": bool(failed), "duration_s": round(dur, 6),
                           "active_after_dec": self._active_counts.get(resource_class, 0)}
                    self._timeline.append(rec)
                    timeline_logger.debug(rec)
                except Exception:
                    pass
                if self.cache:
                    try:
                        self.cache.append_event("job_done", {"job": job_type_name, "resource": resource_class})
                    except Exception:
                        pass
                # сбрасываем состояния при успешном завершении
                try:
                    if not failed:
                        with self._lock:
                            self._job_failures[job_type_name] = 0
                            self._job_next_allowed_ts[job_type_name] = 0.0
                            self._job_pending_until_ts[job_type_name] = 0.0
                except Exception:
                    pass

        # если есть deferred_system — используем его, иначе поток
        try:
            if self.deferred and hasattr(self.deferred, 'add_command'):
                priority = getattr(job_cls, 'default_priority', None)
                self.deferred.add_command(command=lambda: _runner(), args=(), kwargs={}, priority=priority)
            else:
                t = threading.Thread(target=_runner, daemon=True)
                t.start()
        except Exception as e:
            logger.error(f"Failed to dispatch job {job_type_name}: {e}", exc_info=True)
            # откатываем pending, так как запуск не состоялся
            with self._lock:
                self._job_pending[job_type_name] = False
            return

    # ---- Публичный доступ к таймлайну ----
    def get_timeline(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            if not limit or limit <= 0:
                return list(self._timeline)
            return list(self._timeline)[-int(limit):]
        except Exception:
            return []
