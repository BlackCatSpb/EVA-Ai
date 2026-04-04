"""
Event handlers, callbacks, signal connections, and query processing for ЕВА GUI.
"""
import os
import logging
import threading
import queue
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional

import tkinter as tk
from tkinter import messagebox

logger = logging.getLogger("eva.gui.core")


class EventHandlerMixin:
    """Mixin for event handlers, callbacks, and signal connections."""

    def process_query_via_integrator(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Обработка запроса через интегратор системы."""
        if not self.integrator:
            logger.warning("Интегратор не доступен, использую прямую обработку через brain")
            return self._fallback_query_processing(query, context)

        try:
            logger.info(f"Отправка запроса через интегратор: '{query[:50]}...'")

            query_data = {
                'query': query,
                'context': context or {},
                'source': 'gui',
                'timestamp': time.time()
            }

            if hasattr(self.integrator, 'event_bus'):
                self.integrator.event_bus.trigger('query_received', query_data)
                return self._wait_for_response(query_data)
            else:
                return self.integrator.process_query(query, context)

        except Exception as e:
            logger.error(f"Ошибка обработки запроса через интегратор: {e}")
            return self._fallback_query_processing(query, context)

    def _wait_for_response(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ожидание ответа на запрос через событийную шину."""
        timeout = 30.0
        start_time = time.time()
        request_id = f"gui_{int(start_time)}"

        response_received = threading.Event()
        response_data = {}

        def on_response_received(data):
            nonlocal response_data
            if data.get('request_id') == request_id or 'response' in data:
                response_data = data
                response_received.set()

        if hasattr(self.integrator, 'event_bus'):
            try:
                self.integrator.event_bus.subscribe(
                    'response_generated',
                    on_response_received,
                    priority=10
                )
            except Exception as e:
                self.chat_logger.warning(f"Не удалось подписаться на событие response_generated: {e}")

        if response_received.wait(timeout):
            return response_data
        else:
            return {
                'status': 'timeout',
                'error': 'Превышено время ожидания ответа',
                'response': 'Извините, обработка запроса заняла слишком много времени.'
            }

    def _fallback_query_processing(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Запасной вариант обработки запроса напрямую через brain."""
        try:
            if not self.brain:
                return {
                    'status': 'error',
                    'error': 'Система недоступна',
                    'response': 'Извините, система временно недоступна.'
                }

            if hasattr(self.brain, 'process_query'):
                result = self.brain.process_query(query, context)
                if isinstance(result, dict):
                    response_text = result.get("text", result.get("response", "Ошибка обработки"))
                else:
                    response_text = str(result) if result else "Пустой ответ от системы"
                
                if self.chat_module and hasattr(self.chat_module, '_add_message'):
                    self.gui_queue.put(lambda: self.chat_module._add_message("ЕВА", response_text, "system"))
                
                return {
                    'status': 'ok',
                    'response': response_text
                }
            else:
                if self.chat_module and hasattr(self.chat_module, '_add_message'):
                    self.gui_queue.put(lambda: self.chat_module._add_message("ЕВА", "Система обработки запросов недоступна", "system"))
                return {
                    'status': 'error',
                    'error': 'No brain available',
                    'response': 'Система временно недоступна.'
                }

        except Exception as e:
            logger.error(f"Ошибка fallback обработки: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'response': 'Произошла ошибка при обработке запроса.'
            }

    def get_system_status_via_integrator(self) -> Dict[str, Any]:
        """Получение статуса системы через интегратор."""
        if self.integrator and hasattr(self.integrator, 'get_system_health'):
            return self.integrator.get_system_health()
        elif self.integrator and hasattr(self.integrator, 'get_system_stats'):
            return self.integrator.get_system_stats()
        else:
            return self._get_system_status_fallback()

    def _get_system_status_fallback(self) -> Dict[str, Any]:
        """Запасной вариант получения статуса системы."""
        try:
            if not self.brain:
                return {'status': 'disconnected'}

            status = {'status': 'unknown'}

            if hasattr(self.brain, 'running') and self.brain.running:
                status['status'] = 'active'
            elif hasattr(self.brain, 'components') and self.brain.components:
                status['status'] = 'active'
            else:
                status['status'] = 'disconnected'

            if hasattr(self.brain, 'components'):
                status['components_count'] = len(self.brain.components)

            return status

        except Exception as e:
            logger.error(f"Ошибка получения статуса системы: {e}")
            return {'status': 'error', 'error': str(e)}

    def start_self_dialog_via_integrator(self):
        """Запуск самодиалога через интегратор."""
        try:
            if self.integrator and hasattr(self.integrator, 'start_self_dialog'):
                self.integrator.start_self_dialog()
                self.show_toast("Самодиалог запущен", "info")
            else:
                self.show_toast("Самодиалог недоступен", "warning")
        except Exception as e:
            logger.error(f"Ошибка запуска самодиалога: {e}")
            self.show_toast(f"Ошибка самодиалога: {e}", "error")

    def optimize_system_via_integrator(self):
        """Оптимизация системы через интегратор."""
        try:
            if self.integrator and hasattr(self.integrator, 'optimize_system'):
                self.integrator.optimize_system()
                self.show_toast("Оптимизация системы запущена", "info")
            else:
                self.show_toast("Оптимизация недоступна", "warning")
        except Exception as e:
            logger.error(f"Ошибка оптимизации системы: {e}")
            self.show_toast(f"Ошибка оптимизации: {e}", "error")

    def _schedule_update(self):
        try:
            if not self.running:
                return
            if not self.root:
                return
            self._update_interface()
            interval = self.settings.get("gui", {}).get("auto_update_interval", 5000)
            self.update_job = self.root.after(interval, self._schedule_update)
        except Exception as e:
            try:
                self.chat_logger.debug(f"_schedule_update error: {e}")
            except Exception as e2:
                logger.warning(f"Error logging _schedule_update error: {e2}")

    def _start_background_services(self):
        self._schedule_update()
        self._process_gui_queue()
        try:
            if self.brain:
                handler = self._handle_model_load_event
                if hasattr(self.brain, 'events') and self.brain.events and hasattr(self.brain.events, 'subscribe'):
                    try:
                        self.brain.events.subscribe('model_load', handler)
                        self.chat_logger.info("GUI подписан на события model_load через EventSystem")
                        try:
                            self.brain.events.subscribe('models_ready', lambda data=None: self.gui_queue.put(self._handle_models_ready_event))
                            self.chat_logger.info("GUI подписан на событие models_ready")
                        except Exception:
                            pass
                        try:
                            self.brain.events.subscribe('request_gui_reload', lambda data=None: self.gui_queue.put(self.reload))
                            self.chat_logger.info("GUI подписан на событие request_gui_reload")
                        except Exception:
                            pass
                    except Exception:
                        if not hasattr(self.brain, 'on_model_load'):
                            setattr(self.brain, 'on_model_load', [])
                        self.brain.on_model_load.append(handler)
                        self.chat_logger.info("GUI подписан на события model_load через on_model_load")
                        try:
                            if not hasattr(self.brain, 'on_models_ready'):
                                setattr(self.brain, 'on_models_ready', [])
                            self.brain.on_models_ready.append(lambda data=None: self.gui_queue.put(self._handle_models_ready_event))
                        except Exception:
                            pass
                else:
                    if not hasattr(self.brain, 'on_model_load'):
                        setattr(self.brain, 'on_model_load', [])
                    self.brain.on_model_load.append(handler)
                    self.chat_logger.info("GUI подписан на события model_load (fallback)")
                    try:
                        if not hasattr(self.brain, 'on_models_ready'):
                            setattr(self.brain, 'on_models_ready', [])
                        self.brain.on_models_ready.append(lambda data=None: self.gui_queue.put(self._handle_models_ready_event))
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Не удалось подписаться на события model_load: {e}")

        try:
            if self.brain and not (hasattr(self.brain, 'events') and self.brain.events):
                setattr(self.brain, 'request_gui_reload', lambda reason=None: self.gui_queue.put(self.reload))
        except Exception:
            pass

    def _process_gui_queue(self):
        if not self.running or not self.root:
            return
        consecutive_errors = 0
        max_delay = 5000
        current_delay = 100

        while self.running:
            try:
                task = self.gui_queue.get(timeout=0.1)
                if callable(task):
                    task()
                    consecutive_errors = 0
            except queue.Empty:
                break
            except tk.TclError:
                return
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"GUI queue processing error {consecutive_errors}: {e}")
                if consecutive_errors >= 3:
                    current_delay = min(current_delay * 2, max_delay)
                    logger.warning(f"Too many errors, backing off to {current_delay}ms")
                    break
        try:
            if self.running and self.root and self.root.winfo_exists():
                self.gui_queue_job = self.root.after(current_delay, self._process_gui_queue)
        except tk.TclError:
            self.gui_queue_job = None

    def process_query(self, query: str) -> str:
        """Обрабатывает запрос пользователя с полным логгированием."""
        self.chat_logger.info(f"Получен запрос от пользователя: '{query}'")
        self.chat_logger.debug(f"Начало обработки запроса: '{query}'")
        start_time = time.time()
        
        if not self.brain:
            error_msg = "Ошибка: ядро системы недоступно. Пожалуйста, перезагрузите систему."
            self.chat_logger.error(error_msg)
            return error_msg
        
        try:
            self.chat_logger.info(f"Передача запроса в ядро системы: '{query}'")
            
            if hasattr(self.brain, 'get_system_status'):
                system_status = self.brain.get_system_status()
                self.chat_logger.debug(f"Состояние системы перед обработкой запроса: {system_status}")
            
            self.chat_logger.debug(f"Начало токенизации запроса: '{query}'")
            tokenization_start = time.time()
            
            if hasattr(self.brain, 'tokenize_query'):
                tokens = self.brain.tokenize_query(query)
                tokenization_time = time.time() - tokenization_start
                self.chat_logger.info(f"Токенизация завершена за {tokenization_time:.4f} сек. Количество токенов: {len(tokens)}")
                self.chat_logger.debug(f"Токены: {tokens[:10]}..." if len(tokens) > 10 else f"Токены: {tokens}")
            else:
                self.chat_logger.warning("Метод tokenize_query не найден в ядре. Используется базовая токенизация.")
                tokens = query.split()
                self.chat_logger.info(f"Базовая токенизация выполнена. Количество токенов: {len(tokens)}")
            
            processing_start = time.time()
            self.chat_logger.info(f"Начало обработки запроса в ядре. Токенов: {len(tokens)}")
            
            history_context = {}

            if hasattr(self, 'chat_module') and hasattr(self.chat_module, 'message_history'):
                recent_history = self.chat_module.message_history[-10:]
                if recent_history:
                    history_context["message_history"] = recent_history

            if hasattr(self.brain, 'memory_manager') and self.brain.memory_manager:
                try:
                    interactions = self.brain.memory_manager.get_recent_interactions(limit=10)
                    if interactions:
                        history_context["conversation_history"] = [
                            {"query": i.get("query", ""), "response": i.get("response", "")}
                            for i in interactions[-10:] if i
                        ]
                except Exception as e:
                    self.chat_logger.debug(f"Error getting conversation from memory: {e}")

            if not history_context:
                history_context = {}
            
            if not hasattr(self.brain, 'process_query'):
                self.chat_logger.error("Метод process_query недоступен в ядре системы")
                return "Система обработки запросов недоступна."
            
            response_obj = self.brain.process_query(query, context=history_context)
            
            if isinstance(response_obj, dict):
                response = response_obj.get('text') or response_obj.get('response') or str(response_obj)
                metadata = response_obj.get('metadata')
                if isinstance(metadata, dict):
                    raw_reasoning = response_obj.get('reasoning') or response_obj.get('thinking') or metadata.get('reasoning_steps', '')
                else:
                    raw_reasoning = response_obj.get('reasoning') or response_obj.get('thinking') or ''
                if isinstance(raw_reasoning, dict):
                    reasoning = self._format_reasoning_display(raw_reasoning)
                else:
                    reasoning = str(raw_reasoning) if raw_reasoning else ''
            else:
                response = str(response_obj) if response_obj else "нет ответа"
                reasoning = ""
            
            processing_time = time.time() - processing_start
            
            response_preview = response[:100] if response else ""
            self.chat_logger.info(f"Получен ответ от ядра: '{response_preview}...'")
            
            if not response or (isinstance(response, str) and response.strip().lower() == "нет ответа"):
                self.chat_logger.error(f"Ядро вернуло недопустимый ответ на запрос '{query}'")
                response = (
                    "Система временно не может предоставить полный ответ. "
                    "Попробуйте перефразировать запрос или задать другой вопрос."
                )
                self.chat_logger.warning("Сгенерирован альтернативный ответ из-за недопустимого ответа ядра")
            
            self.chat_logger.debug("Сохранение запроса и ответа в историю")
            self._save_to_history(query, response)
            
            if reasoning and hasattr(self, 'chat_module'):
                try:
                    self.gui_queue.put(lambda r=reasoning: self.chat_module._set_reasoning_content(r, auto_expand=True))
                except Exception as e:
                    self.chat_logger.debug(f"Error displaying reasoning: {e}")
            
            total_time = time.time() - start_time
            self.chat_logger.info(f"Общий процесс обработки запроса завершен за {total_time:.4f} сек")
            
            return response
        except Exception as e:
            self.chat_logger.exception(f"Критическая ошибка при обработке запроса: {e}")
            error_response = (
                "Произошла ошибка при обработке запроса. "
                "Пожалуйста, попробуйте повторить запрос позже."
            )
            self.chat_logger.error(f"Возвращен ошибочный ответ: '{error_response}'")
            return error_response

    def _save_to_history(self, query: str, response: str):
        """Сохраняет запрос и ответ в историю."""
        self.chat_logger.debug(f"Начало сохранения в историю: Q: {query[:50]}{'...' if len(query) > 50 else ''}")
        self.chat_logger.info(f"Сохранение в историю: длина запроса={len(query)}, длина ответа={len(response)}")
        
        if len(query) > 100:
            self.chat_logger.debug(f"Запрос (первые 100 символов): {query[:100]}...")
        else:
            self.chat_logger.debug(f"Запрос: {query}")
            
        if len(response) > 100:
            self.chat_logger.debug(f"Ответ (первые 100 символов): {response[:100]}...")
        else:
            self.chat_logger.debug(f"Ответ: {response}")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.chat_logger.info(f"Сохранение в историю с временной меткой: {timestamp}")
        
        try:
            history_file = os.path.join(self.cache_dir, "chat_history.json")
            history = []
            
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            
            history.append({
                "timestamp": timestamp,
                "query": query,
                "response": response,
                "query_length": len(query),
                "response_length": len(response)
            })
            
            if len(history) > 500:
                history = history[-500:]
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            
            self.chat_logger.info(f"История чата успешно сохранена в {history_file}. Текущая длина истории: {len(history)}")
        
        except Exception as e:
            self.chat_logger.error(f"Ошибка при сохранении истории чата: {str(e)}", exc_info=True)
        
        self.chat_logger.debug("Сохранение в историю завершено")

    def _format_reasoning_display(self, reasoning_dict: dict) -> str:
        """Форматирует словарь рассуждений для отображения в GUI."""
        if not reasoning_dict:
            return ""
        
        lines = []
        
        if 'steps' in reasoning_dict and reasoning_dict['steps']:
            lines.append("Этапы рассуждения:")
            for i, step in enumerate(reasoning_dict['steps'][:5], 1):
                if isinstance(step, dict):
                    phase = step.get('phase', step.get('thought', f'Шаг {i}'))
                    thought = step.get('thought', '')
                    lines.append(f"  {i}. {phase}")
                    if thought:
                        lines.append(f"     {thought}")
                else:
                    lines.append(f"  {i}. {step}")
        
        if 'iterations' in reasoning_dict:
            lines.append(f"Итераций: {reasoning_dict['iterations']}")
        
        if 'confidence' in reasoning_dict:
            lines.append(f"Уверенность: {reasoning_dict['confidence']:.2f}")
        
        if 'final_response' in reasoning_dict:
            response = reasoning_dict['final_response']
            if response and len(response) > 100:
                lines.append(f"\nОтвет: {response[:200]}...")
        
        return "\n".join(lines) if lines else str(reasoning_dict)

    def show_toast(self, message: str, level: str = "info", duration: int = 5000, key: str = None):
        if key:
            now = time.time()
            last_time = self.last_notification_times.get(key, 0)
            if now - last_time < self.notification_throttle_seconds:
                self.chat_logger.debug(f"Подавлено повторное уведомление с ключом '{key}'")
                return
            self.last_notification_times[key] = now
            self.chat_logger.info(f"Показ уведомления с ключом '{key}': {message}")
        
        self.gui_queue.put(lambda: self._create_toast_window(message, level, duration))

    def _create_toast_window(self, message, level, duration):
        """Создает всплывающее окно с уведомлением"""
        try:
            toast = tk.Toplevel(self.root)
            toast.overrideredirect(True)
            
            colors = {
                'info': ('#0078d7', 'white'),
                'success': ('#28a745', 'white'),
                'warning': ('#ffc107', 'black'),
                'error': ('#dc3545', 'white')
            }
            bg_color, fg_color = colors.get(level, ('#333333', 'white'))
            
            toast.configure(bg=bg_color)
            
            if not self.root:
                return
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            label = tk.Label(toast, text=message, bg=bg_color, fg=fg_color,
                           font=('Segoe UI', 10), wraplength=300, padx=15, pady=10)
            label.pack()
            
            toast.update_idletasks()
            width = toast.winfo_width()
            height = toast.winfo_height()
            
            x = screen_width - width - 20
            y = screen_height - height - 100
            toast.geometry(f"{width}x{height}+{x}+{y}")
            
            if self.root:
                self.root.after(int(duration * 1000), toast.destroy)
            
        except Exception as e:
            logger.debug(f"Ошибка создания toast окна: {e}")

    def on_close(self):
        if messagebox.askyesno("Подтверждение", "Вы действительно хотите выйти?"):
            self.stop()

    def _reboot_system(self):
        if not messagebox.askyesno("Подтверждение", "Вы действительно хотите перезагрузить систему?"):
            self.chat_logger.info("Отмена перезагрузки системы по запросу пользователя")
            return
            
        self.chat_logger.warning("Инициирована перезагрузка системы по запросу пользователя")
        self.show_toast("Перезагрузка системы...", "info")
        logger.info("Инициирована перезагрузка системы")
        
        if self.brain and hasattr(self.brain, 'reboot'):
            self.chat_logger.info("Запуск фонового потока для перезагрузки ядра")
            threading.Thread(target=self.brain.reboot, daemon=True).start()
        else:
            self.chat_logger.error("Попытка перезагрузки без поддержки функции reboot в ядре")
            self.show_toast("Функция перезагрузки не поддерживается", "error")

    def _soft_reload(self):
        """Инициирует soft-reload ядра с сохранением ML и обновлением GUI."""
        try:
            if not self.brain or not hasattr(self.brain, 'soft_reload'):
                self.show_toast("Soft-reload не поддерживается ядром", "error")
                return
            self.show_toast("Горячая перезагрузка...", "info")
            self.chat_logger.warning("Инициирован soft-reload ядра по запросу пользователя")
            def _do_reload():
                try:
                    ok = self.brain.soft_reload(reload_gui=True)
                    if not ok:
                        self.gui_queue.put(lambda: self.show_toast("Soft-reload завершился с ошибкой", "error"))
                except Exception as e:
                    self.gui_queue.put(lambda: self.show_toast(f"Soft-reload ошибка: {e}", "error"))
            threading.Thread(target=_do_reload, daemon=True).start()
        except Exception:
            pass
