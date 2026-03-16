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

logger = logging.getLogger("cogniflex.core.autopilot")
timeline_logger = logging.getLogger("cogniflex.core.autopilot.timeline")


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

        # Интеграция с системой событий
        self._setup_event_integration()

        # Файловый лог автопилота
        try:
            # База логов: сначала brain.cache_dir, иначе корень проекта + cogniflex_cache
            cache_dir = getattr(brain, 'cache_dir', None)
            if not cache_dir:
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                cache_dir = os.path.join(project_root, 'cogniflex_cache')
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
        # Явный флаг бэкоффа, блокирующий планирование до определённого времени (после фейла)
        self._job_backing_off: Dict[str, bool] = {}

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Публикует событие в систему событий."""
        try:
            if hasattr(self.brain, 'event_bus') and self.brain.event_bus:
                self.brain.event_bus.emit(event_type, data)
                logger.debug(f"Событие опубликовано: {event_type}")
        except Exception as e:
            logger.error(f"Ошибка публикации события {event_type}: {e}")

    # ---- Публичный API ----
    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            # Проверяем состояние инициализации системы
            if hasattr(self, 'brain') and self.brain:
                if not getattr(self.brain, 'initialized', False):
                    logger.info("BackgroundCoordinator: система еще не инициализирована, ожидаю...")
                    return
                if not getattr(self.brain, 'running', False):
                    logger.info("BackgroundCoordinator: система еще не запущена, ожидаю...")
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
        
        # Публикуем событие планирования задачи
        self._emit_event('job_scheduled', {
            'job_type': job_type_name,
            'request': req,
            'source': req.get('source', 'unknown'),
            'timestamp': time.time()
        })
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

        # Запускаем задачу в отдельном потоке
        t = threading.Thread(target=self._run_job, args=(job, job_type_name, resource_class, req, _on_start, _on_done), daemon=True)
        t.start()

    def _run_job(self, job, job_type_name: str, resource_class: str, req: Dict[str, Any], _on_start, _on_done):
        """Выполняет задачу с обработкой ошибок."""
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
            
            # Публикуем событие запуска задачи
            self._emit_event('job_started', {
                'job_type': job_type_name,
                'resource_class': resource_class,
                'request': req,
                'active_count': self._active_counts.get(resource_class, 0),
                'timestamp': time.time()
            })
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
            
            # Публикуем событие завершения задачи
            self._emit_event('job_completed', {
                'job_type': job_type_name,
                'resource_class': resource_class,
                'failed': bool(failed),
                'duration': round(dur, 6),
                'request': req,
                'timestamp': time.time()
            })
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
                        self._job_backing_off[job_type_name] = False
                else:
                    # increment failure count
                    with self._lock:
                        self._job_failures[job_type_name] = self._job_failures.get(job_type_name, 0) + 1
                        fails = self._job_failures[job_type_name]
                        # exponential backoff: 2^fails * base, max 1h
                        base = 30.0
                        backoff = min(3600.0, base * (2 ** fails))
                        hold_until = time.time() + backoff
                        self._job_pending_until_ts[job_type_name] = hold_until
                        self._job_backing_off[job_type_name] = True
                        logger.debug(f"Set backoff for {job_type_name}: {backoff:.1f}s after {fails} failures")
            except Exception:
                pass

    def _setup_event_integration(self) -> None:
        try:
            # Интеграция с системой событий
            if hasattr(self.brain, 'events') and self.brain.events:
                events = self.brain.events
                
                # Подписываемся на события системы
                events.subscribe('system_ready', self._handle_system_ready, priority=5)
                events.subscribe('system_shutdown', self._handle_system_shutdown, priority=5)
                events.subscribe('user_activity', self._handle_user_activity, priority=8)
                events.subscribe('component_health_change', self._handle_component_health_change, priority=7)
                events.subscribe('training_completed', self._handle_training_completed, priority=6)
                
                logger.info("BackgroundCoordinator подписан на события системы")
            
            # Интеграция с отложенными командами
            if self.deferred:
                # Регистрируем команды автопилота
                self.deferred.add_command('autopilot_start', self._deferred_start, priority=10)
                self.deferred.add_command('autopilot_stop', self._deferred_stop, priority=10)
                self.deferred.add_command('autopilot_pause', self._deferred_pause, priority=10)
                self.deferred.add_command('autopilot_resume', self._deferred_resume, priority=10)
                self.deferred.add_command('autopilot_status', self._deferred_status, priority=10)
                
                logger.info("BackgroundCoordinator зарегистрировал команды в DeferredCommandSystem")
                
        except Exception as e:
            logger.error(f"Ошибка интеграции с системой событий: {e}")

    def _handle_system_ready(self, data: Dict[str, Any]) -> None:
        """Обработчик события готовности системы."""
        try:
            logger.info("Система готова - BackgroundCoordinator может начинать планирование")
            # Система готова, можно запускать фоновые задачи
            if hasattr(self, '_running') and not self._running:
                self.start()
        except Exception as e:
            logger.error(f"Ошибка обработки system_ready: {e}")

    def _handle_system_shutdown(self, data: Dict[str, Any]) -> None:
        """Обработчик события выключения системы."""
        try:
            logger.info("Система выключается - BackgroundCoordinator останавливается")
            self.stop()
        except Exception as e:
            logger.error(f"Ошибка обработки system_shutdown: {e}")

    def _handle_user_activity(self, data: Dict[str, Any]) -> None:
        """Обработчик события активности пользователя."""
        try:
            self.signal_user_activity()
            logger.debug("Получен сигнал активности пользователя")
        except Exception as e:
            logger.error(f"Ошибка обработки user_activity: {e}")

    def _handle_component_health_change(self, data: Dict[str, Any]) -> None:
        """Обработчик события изменения здоровья компонента."""
        try:
            component_name = data.get('component', 'unknown')
            health_status = data.get('health', 'unknown')
            
            logger.debug(f"Изменение здоровья компонента {component_name}: {health_status}")
            
            # Если критичный компонент стал нездоров, приостанавливаем фоновые задачи
            if health_status in ['unhealthy', 'error', 'failed']:
                critical_components = ['model_manager', 'ml_unit', 'memory_manager']
                if component_name in critical_components:
                    logger.warning(f"Критичный компонент {component_name} нездоров - приостанавливаем фоновые задачи")
                    self.pause()
        except Exception as e:
            logger.error(f"Ошибка обработки component_health_change: {e}")

    def _handle_training_completed(self, data: Dict[str, Any]) -> None:
        """Обработчик события завершения обучения."""
        try:
            training_type = data.get('type', 'unknown')
            success = data.get('success', False)
            
            logger.info(f"Обучение {training_type} завершено: {'успешно' if success else 'с ошибкой'}")
            
            # Сбрасываем кулдаун для обучения при успешном завершении
            if success and training_type in self.job_cooldowns_s:
                with self._lock:
                    self._job_next_allowed_ts[training_type] = time.time()
                    
        except Exception as e:
            logger.error(f"Ошибка обработки training_completed: {e}")

    # ---- Отложенные команды ----
    def _deferred_start(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Отложенная команда запуска автопилота."""
        try:
            if not self._running:
                self.start()
                return {"status": "success", "message": "Autopilot started"}
            else:
                return {"status": "info", "message": "Autopilot already running"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _deferred_stop(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Отложенная команда остановки автопилота."""
        try:
            if self._running:
                self.stop()
                return {"status": "success", "message": "Autopilot stopped"}
            else:
                return {"status": "info", "message": "Autopilot already stopped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _deferred_pause(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Отложенная команда паузы автопилота."""
        try:
            self.pause()
            return {"status": "success", "message": "Autopilot paused"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _deferred_resume(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Отложенная команда возобновления автопилота."""
        try:
            self.resume()
            return {"status": "success", "message": "Autopilot resumed"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _deferred_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Отложенная команда получения статуса автопилота."""
        try:
            return {
                "status": "success",
                "data": {
                    "running": self._running,
                    "paused": self._paused,
                    "detectors": len(self._detectors),
                    "active_jobs": dict(self._active_counts),
                    "last_user_activity": self._last_user_activity_ts
                }
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Публикует событие в систему событий."""
        try:
            if hasattr(self.brain, 'events') and self.brain.events:
                self.brain.events.trigger(event_type, data)
                logger.debug(f"Событие опубликовано: {event_type}")
        except Exception as e:
            logger.error(f"Ошибка публикации события {event_type}: {e}")

    # ---- Публичный доступ к таймлайну ----
    def get_timeline(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            if not limit or limit <= 0:
                return list(self._timeline)
            return list(self._timeline)[-int(limit):]
        except Exception:
            return []
