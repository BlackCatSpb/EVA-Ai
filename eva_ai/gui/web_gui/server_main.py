"""
Flask app creation, WebGUI class, imports from other modules, re-exports
"""
import os
import time
import logging
import threading
import json
import uuid
import hashlib
import secrets
import socket
from datetime import datetime
from typing import Dict, Any, Optional, List

from flask import Flask

logger = logging.getLogger("eva_ai.webgui")

from eva_ai.gui.web_gui.server_auth import SessionManager, AuthManager, EntityExtractor, EthicsChecker
from eva_ai.gui.web_gui.server_api_wikipedia import register_routes as register_wikipedia_routes
from eva_ai.gui.web_gui.server_api_export import register_routes as register_export_routes
from eva_ai.gui.web_gui.server_models import register_routes as register_model_routes
from eva_ai.gui.web_gui.server_routes_graph import register_graph_routes

# Refactored route modules
from eva_ai.gui.web_gui.server_routes_core import register_core_routes
from eva_ai.gui.web_gui.server_routes_chat import register_chat_routes
from eva_ai.gui.web_gui.server_routes_auth import register_auth_routes
from eva_ai.gui.web_gui.server_routes_analytics import register_analytics_routes
from eva_ai.gui.web_gui.server_routes_knowledge import register_knowledge_routes
from eva_ai.gui.web_gui.server_routes_upload import register_upload_routes


def _get_secret_key():
    try:
        env_key = os.environ.get('COGNIFLEX_SECRET_KEY')
        if env_key:
            return env_key
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eva_config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if config.get('web_gui', {}).get('secret_key'):
                        return config['web_gui']['secret_key']
            except Exception:
                pass
        return os.urandom(32).hex()
    except Exception:
        return os.urandom(32).hex()


app = Flask(__name__,
            template_folder='templates',
            static_folder='static',
            static_url_path='/static')
app.config['SECRET_KEY'] = _get_secret_key()
app.config['JSON_AS_ASCII'] = False


class WebGUI:
    def __init__(self, brain=None, integrator=None, host='127.0.0.1', port=5555):
        self.brain = brain
        self.integrator = integrator
        self.host = host
        self.port = port
        self.running = False
        self.thread = None
        self.bridge = None
        
        self.auth_manager = AuthManager()
        self.session_manager = SessionManager()
        self.entity_extractor = EntityExtractor()
        self.ethics_checker = EthicsChecker()
        
        # Хранилище документов для сессий (session_id -> {document_id: metadata})
        self._session_documents: Dict[str, Dict[str, Any]] = {}
        # Порог для использования DocumentVirtualMemory (токенов)
        self._document_memory_threshold = 1000

        admin_user = os.environ.get('COGNIFLEX_ADMIN_USER', 'admin')
        admin_pass = os.environ.get('COGNIFLEX_ADMIN_PASS')
        config_needs_password = False
        
        # Try to load config from multiple possible locations
        possible_config_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eva_config.json'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'eva_config.json'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'gui', 'eva_config.json'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', '..', 'gui', 'eva_config.json'),
            'eva_config.json',
            'gui/eva_config.json',
        ]
        config_path = None
        for path in possible_config_paths:
            if os.path.exists(path):
                config_path = os.path.abspath(path)
                break
        
        admin_salt = None
        if not admin_pass and config_path:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        admin_pass = config.get('web_gui', {}).get('admin_password')
                        admin_salt = config.get('web_gui', {}).get('admin_salt')
                except Exception:
                    pass
        
        if not admin_pass:
            admin_pass = secrets.token_urlsafe(16)
            config_needs_password = True
            logger.warning("GENERATED DEFAULT ADMIN PASSWORD: {}".format(admin_pass))
        else:
            logger.info("Admin password loaded from config")
        
        # Set credentials with stored salt or generate new
        if admin_salt:
            password_hash = hashlib.pbkdf2_hmac('sha256', admin_pass.encode(), admin_salt.encode(), 100000).hex()
            with self.auth_manager._lock:
                self.auth_manager.users[admin_user] = {
                    'username': admin_user,
                    'password_hash': password_hash,
                    'salt': admin_salt,
                    'created_at': datetime.now().isoformat()
                }
        else:
            self.auth_manager.set_default_credentials(admin_user, admin_pass)
        
        if config_needs_password and config_path:
            try:
                config = {}
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                if 'web_gui' not in config:
                    config['web_gui'] = {}
                config['web_gui']['admin_password'] = admin_pass
                new_salt = secrets.token_hex(16)
                config['web_gui']['admin_salt'] = new_salt
                password_hash = hashlib.pbkdf2_hmac('sha256', admin_pass.encode(), new_salt.encode(), 100000).hex()
                with self.auth_manager._lock:
                    self.auth_manager.users[admin_user] = {
                        'username': admin_user,
                        'password_hash': password_hash,
                        'salt': new_salt,
                        'created_at': datetime.now().isoformat()
                    }
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                logger.info("Admin password saved to config")
            except Exception as e:
                logger.error("Failed to save admin password: {}".format(e))

        logger.info("WebGUI initialized on {}:{}".format(host, port))

    def _get_document_manager(self):
        """Получить доступ к DocumentVirtualMemory через dual_generator."""
        try:
            if (self.brain and 
                hasattr(self.brain, 'two_model_pipeline') and 
                self.brain.two_model_pipeline and
                hasattr(self.brain.two_model_pipeline, 'dual_generator') and
                self.brain.two_model_pipeline.dual_generator):
                
                dg = self.brain.two_model_pipeline.dual_generator
                if hasattr(dg, 'document_manager') and dg.document_manager:
                    return dg
        except Exception as e:
            logger.debug(f"DocumentManager недоступен: {e}")
        return None
    
    def _ingest_document_to_memory(self, content: str, title: str, session_id: str) -> Optional[str]:
        """Загрузить документ в DocumentVirtualMemory."""
        dg = self._get_document_manager()
        if not dg:
            return None
        
        try:
            doc_id = dg.load_document(content, title)
            if doc_id and session_id:
                # Сохраняем информацию о документе в сессии
                if session_id not in self._session_documents:
                    self._session_documents[session_id] = {}
                self._session_documents[session_id][doc_id] = {
                    'title': title,
                    'loaded_at': datetime.now().isoformat(),
                    'size': len(content)
                }
                logger.info(f"Документ '{title}' загружен в память (ID: {doc_id})")
            return doc_id
        except Exception as e:
            logger.error(f"Ошибка загрузки документа в память: {e}")
            return None
    
    def _estimate_tokens(self, text: str) -> int:
        """Оценка количества токенов в тексте."""
        # Простая эвристика: ~1.3 токена на слово
        return int(len(text.split()) * 1.3)

    def _prepare_file_context(self, query: str, file_data: Dict) -> tuple:
        """Подготовить контекст файла для запроса."""
        if not file_data or not file_data.get('extracted_text'):
            return query, None
            
        filename = file_data.get('filename', 'file')
        extracted = file_data['extracted_text']
        estimated_tokens = self._estimate_tokens(extracted)
        
        if estimated_tokens > self._document_memory_threshold:
            document_id = self._ingest_document_to_memory(extracted, filename, None)
            if document_id:
                enhanced_query = f"""Пользователь прикрепил документ "{filename}" ({estimated_tokens} токенов).
Документ загружен в виртуальную память (ID: {document_id}).

Запрос пользователя: {query}"""
                return enhanced_query, document_id
            else:
                logger.warning(f"Не удалось загрузить документ в память, используем обычный режим")
        
        enhanced_query = f"""Пользователь прикрепил файл "{filename}".
Содержимое файла:
---
{extracted[:8000] if estimated_tokens > self._document_memory_threshold else extracted}
---

Запрос пользователя: {query}"""
        return enhanced_query, None
    
    def _extract_reasoning_from_result(self, result: Dict, source: str, 
                                        brain_reasoning: Any, brain_reasoning_raw: Any,
                                        brain_ethics: Dict, brain_contradiction: Dict,
                                        confidence: float, search_results: List) -> tuple:
        """Извлечь reasoning_steps и reasoning_data из результата brain."""
        reasoning_steps = []
        reasoning_data = None
        
        if source == 'document_virtual_memory':
            doc_context = result.get('document_context', {})
            doc_title = doc_context.get('document_title', 'Unknown')
            relevant_pages = doc_context.get('relevant_pages', [])
            
            reasoning_steps = [
                {'step': 1, 'phase': 'document_analysis', 
                 'thought': f'Документ "{doc_title}" загружен в виртуальную память', 'confidence': 0.95},
                {'step': 2, 'phase': 'page_retrieval', 
                 'thought': f'Извлечены релевантные страницы: {relevant_pages}', 'confidence': 0.9},
                {'step': 3, 'phase': 'context_building', 
                 'thought': 'Построение контекста из страниц документа', 'confidence': 0.9},
                {'step': 4, 'phase': 'generation', 
                 'thought': 'Генерация ответа на основе контекста документа', 'confidence': 0.85},
                {'step': 5, 'phase': 'final_synthesis', 
                 'thought': 'Финальный ответ с учетом документа', 'confidence': result.get('confidence', 0.9)}
            ]
            
            reasoning_data = f"📚 Анализ документа:\n\n"
            reasoning_data += f"Документ: {doc_title}\n"
            reasoning_data += f"Использованы страницы: {', '.join(map(str, relevant_pages))}\n\n"
            reasoning_data += "Шаги:\n"
            for s in reasoning_steps:
                reasoning_data += f"{s['step']}. [{s['phase']}] {s['thought']} (conf: {s['confidence']:.2f})\n"
        
        elif source in ['llama_cpp_with_modules', 'self_reasoning_engine', 'hybrid_dialog_manager', 'unified_generator']:
            # Парсим рассуждения из текста результата
            import re
            think_pattern = r'<think>\s*(.*?)\s*</think>'
            
            # Сначала проверяем result.get('reasoning_steps')
            if result.get('reasoning_steps'):
                reasoning_steps = result.get('reasoning_steps', [])
            elif result.get('reasoning'):
                # Если есть reasoning в result
                reasoning_data = str(result.get('reasoning'))
            else:
                # Парсим из текста
                full_text = result.get('response', '') or str(result)
                matches = re.findall(think_pattern, full_text, re.DOTALL)
                if matches:
                    reasoning_data = '\n'.join(matches)
                    lines = [l.strip() for l in reasoning_data.split('\n') if l.strip()]
                    for i, step in enumerate(lines[:20]):
                        reasoning_steps.append({
                            'step': i + 1,
                            'phase': 'reasoning',
                            'thought': step[:500],
                            'confidence': 0.8
                        })
            if result.get('reasoning_steps'):
                reasoning_steps = result.get('reasoning_steps', [])
            elif brain_reasoning_raw and isinstance(brain_reasoning_raw, dict):
                steps = brain_reasoning_raw.get('steps', [])
                for i, step in enumerate(steps):
                    reasoning_steps.append({
                        'step': i + 1,
                        'phase': step.get('phase', 'unknown'),
                        'thought': step.get('thought', ''),
                        'confidence': step.get('confidence', 0)
                    })
            
            if brain_reasoning:
                reasoning_data = str(brain_reasoning)
        
        # Добавляем ethics и contradiction шаги
        if brain_contradiction:
            reasoning_steps.append({
                'step': len(reasoning_steps) + 1,
                'phase': 'contradiction_check',
                'thought': f'Проверка противоречий: найдено={brain_contradiction.get("significant_count", 0)}',
                'confidence': 1.0 - brain_contradiction.get('contradiction_level', 0.0)
            })
        
        if brain_ethics:
            reasoning_steps.append({
                'step': len(reasoning_steps) + 1,
                'phase': 'ethics_check',
                'thought': f'Этическая оценка: violations={brain_ethics.get("has_violations", False)}',
                'confidence': brain_ethics.get('is_ethical', 1.0)
            })
        
        if search_results and len(search_results) > 0:
            reasoning_steps.append({
                'step': len(reasoning_steps) + 1,
                'phase': 'web_search',
                'thought': f'Веб-поиск: найдено {len(search_results)} результатов',
                'confidence': 0.8
            })
        
        return reasoning_steps, reasoning_data

    def process_message(self, query: str, session_id: str, user_id: str, file_data: Dict = None) -> Dict[str, Any]:
        # Начинаем измерение времени
        request_start = time.time()
        
        # Проверяем, есть ли активные документы для этой сессии
        has_active_documents = (
            session_id and 
            session_id in self._session_documents and 
            self._session_documents[session_id]
        )
        
        # Собираем метрики в конце
        def record_metrics(success: bool = True, error: str = None):
            """Записать метрики запроса."""
            try:
                from eva_ai.core.metrics import get_eva_metrics
                eva_metrics = get_eva_metrics()
                
                duration = time.time() - request_start
                eva_metrics.record_request(duration, endpoint='process_message')
                
                if not success:
                    eva_metrics.record_error(error_type=error or 'unknown')
            except Exception as e:
                logger.debug(f"Ошибка записи метрик: {e}")
        
        # Подготавливаем контекст файла
        query, document_id = self._prepare_file_context(query, file_data or {})
        
        # Проверка этики
        ethics_result = self.ethics_checker.check_message(query)
        if not ethics_result['allowed']:
            record_metrics(success=False, error='ethics_blocked')
            return {
                'response': 'Извините, это сообщение заблокировано.',
                'status': 'blocked',
                'reason': ethics_result.get('reason')
            }
        
        # Извлечение сущностей и сохранение в сессию
        entities = self.entity_extractor.extract_entities(query)
        
        if session_id:
            self.session_manager.add_chat_message(session_id, 'user', query)
            context_node = {
                'id': str(uuid.uuid4()),
                'user_message': query,
                'timestamp': datetime.now().isoformat(),
                'entities': entities,
                'file_data': file_data
            }
            self.session_manager.add_context_node(session_id, context_node)
            
            for entity in entities:
                if not self.entity_extractor.is_personal_info(entity.get('context', '')):
                    sanitized = self.ethics_checker.sanitize_entity(entity)
                    self.session_manager.add_entity(session_id, sanitized)
        
        # Подготовка контекста пользователя
        conversation_history = []
        if session_id and isinstance(session_id, str) and session_id.strip():
            conversation_history = self.session_manager.get_chat_history(session_id, limit=20)
        
        user_context = {
            'session_id': session_id,
            'user_id': user_id,
            'conversation_history': conversation_history
        }
        
        # Обработка через DocumentVirtualMemory или brain
        result = None
        response_text = "Система обрабатывает запрос..."
        document_mode = False
        document_context_info = None
        
        if (has_active_documents or document_id) and self._get_document_manager():
            result = self._process_with_document(query, document_id, session_id)
            if result:
                response_text = result.get('response', response_text)
                document_mode = True
        
        # Используем HybridKnowledgeDialogManager если доступен
        if not document_mode and self.brain:
            # 1. Пробуем HybridKnowledgeDialogManager
            if (hasattr(self.brain, 'hybrid_dialog_manager') and 
                self.brain.hybrid_dialog_manager and 
                self.brain.hybrid_dialog_manager.initialized):
                try:
                    logger.info("Using HybridKnowledgeDialogManager for response")
                    dialog_result = self.brain.hybrid_dialog_manager.process(query)
                    if dialog_result:
                        response_text = dialog_result.get('response', '')
                        result = {
                            'response': response_text,
                            'source': 'hybrid_dialog_manager',
                            'reasoning': dialog_result.get('reasoning'),
                            'reasoning_steps': dialog_result.get('reasoning_steps')
                        }
                except Exception as e:
                    logger.warning(f"HybridKnowledgeDialogManager error: {e}")
            
            # 2. Fallback на process_query
            elif hasattr(self.brain, 'process_query'):
                result = self.brain.process_query(query, user_context)
                if result and isinstance(result, dict):
                    response_text = result.get('response', result.get('text', response_text))
        
        # Извлечение reasoning из результата
        reasoning_steps = []
        reasoning_data = None
        search_results = []
        brain_ethics = None
        brain_contradiction = None
        
        if result and isinstance(result, dict):
            source = result.get('source', '')
            confidence = result.get('confidence', 0)
            search_results = result.get('search_results', [])
            brain_ethics = result.get('ethics_result') or ethics_result
            brain_contradiction = result.get('contradiction_result')
            
            reasoning_steps, reasoning_data = self._extract_reasoning_from_result(
                result, source, result.get('reasoning'), result.get('reasoning_raw'),
                brain_ethics, brain_contradiction, confidence, search_results
            )
        
        # Сохранение ответа в сессию
        if session_id and response_text:
            self.session_manager.add_chat_message(session_id, 'assistant', response_text)
            response_node = {
                'id': str(uuid.uuid4()),
                'assistant_message': response_text,
                'timestamp': datetime.now().isoformat(),
                'reasoning': reasoning_data
            }
            self.session_manager.add_context_node(session_id, response_node)
            
            if self.brain and hasattr(self.brain, 'fractal_graph_v2') and self.brain.fractal_graph_v2:
                self.session_manager.convert_chat_to_knowledge(session_id, self.brain.fractal_graph_v2)
        
        # Формирование ответа
        return_data = {
            'response': response_text,
            'status': 'ok',
            'warnings': ethics_result.get('warnings', []) if ethics_result else [],
            'ethics_result': ethics_result,
            'contradiction_metrics': brain_contradiction,
            'reasoning': reasoning_data,
            'reasoning_steps': reasoning_steps,
            'search_results': search_results if search_results else None,
            'web_search_info': f"Найдено {len(search_results)} результатов" if search_results else None
        }
        
        record_metrics(success=True)
        return return_data
    
    def _process_with_document(self, query: str, document_id: str, session_id: str) -> Optional[Dict]:
        """Обработать запрос с использованием DocumentVirtualMemory."""
        dg = self._get_document_manager()
        if not dg:
            return None
        
        try:
            active_doc_id = document_id
            if not active_doc_id and session_id and session_id in self._session_documents:
                active_doc_id = max(
                    self._session_documents[session_id].keys(),
                    key=lambda k: self._session_documents[session_id][k].get('loaded_at', '')
                )
            
            if not active_doc_id:
                return None
            
            logger.info(f"Используем DocumentVirtualMemory для документа {active_doc_id}")
            gen_result = dg.generate_with_document(query=query, document_id=active_doc_id, mode="extended")
            
            if gen_result and isinstance(gen_result, dict):
                return {
                    'response': gen_result.get('response', gen_result.get('text', '')),
                    'source': 'document_virtual_memory',
                    'document_context': gen_result.get('document_context', {})
                }
        except Exception as e:
            logger.error(f"Ошибка при использовании DocumentVirtualMemory: {e}")
        
        return None

    def get_session_documents(self, session_id: str) -> Dict[str, Any]:
        """Получить список документов для сессии."""
        if not session_id or session_id not in self._session_documents:
            return {}
        return self._session_documents[session_id].copy()
    
    def get_document_stats(self, session_id: str, document_id: str) -> Optional[Dict[str, Any]]:
        """Получить статистику по документу."""
        dg = self._get_document_manager()
        if not dg:
            return None
        
        try:
            return dg.get_document_stats(document_id)
        except Exception as e:
            logger.error(f"Ошибка получения статистики документа: {e}")
            return None
    
    def clear_session_documents(self, session_id: str):
        """Очистить документы сессии."""
        if session_id and session_id in self._session_documents:
            logger.info(f"Очистка документов сессии {session_id}")
            del self._session_documents[session_id]

    def start(self):
        if self.running:
            return

        self.running = True
        self.shutting_down = False

        def run():
            import click
            click.echo = lambda *args, **kwargs: None
            self._server = None
            try:
                from werkzeug.serving import make_server
                self._server = make_server(self.host, self.port, app, threaded=True)
                logger.info(f"Flask server created on {self.host}:{self.port}")
                # Serve until shutdown with short timeout
                while self.running and not self.shutting_down:
                    try:
                        self._server.socket.settimeout(0.5)  # Короче timeout для быстрого shutdown
                        self._server.handle_request()
                    except socket.timeout:
                        continue
                    except OSError as e:
                        if not self.shutting_down and e.winerror != 10004:  # WSAEWOULDBLOCK
                            logger.error(f"Flask socket error: {e}")
                        break
                    except Exception as e:
                        if not self.shutting_down:
                            logger.error(f"Flask error: {e}")
                        break
                # Быстрый shutdown без ожидания
                if self._server:
                    try:
                        self._server.shutdown()
                    except:
                        pass
                logger.info("Flask server shut down")
            except Exception as e:
                if not getattr(self, 'shutting_down', False):
                    logger.error(f"Flask error: {e}")
            finally:
                # Явно сигнализируем о завершении
                self.running = False

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.daemon = True
        self.thread.start()

        logger.info(f"WebGUI сервер запущен на http://{self.host}:{self.port}")

    def stop(self):
        self.shutting_down = True
        self.running = False
        
        # Закрываем сокет принудительно
        if hasattr(self, '_server') and self._server:
            try:
                try:
                    self._server.shutdown()
                except:
                    pass
                # Закрываем сокет
                if hasattr(self._server, 'socket') and self._server.socket:
                    try:
                        self._server.socket.close()
                    except:
                        pass
            except:
                pass
        
        # Прерываем поток если он всё ещё работает
        if self.thread and self.thread.is_alive():
            logger.debug(f"Ожидание завершения потока WebGUI...")
            self.thread.join(timeout=3)  # Ждём максимум 3 секунды
        
        # Принудительно завершаем если поток всё ещё жив
        if self.thread and self.thread.is_alive():
            logger.warning("Поток WebGUI не завершился, продолжаем...")
        
        logger.info("WebGUI сервер остановлен")


# Don't create default instance at import time - let create_app handle it
web_gui_instance: Optional[WebGUI] = None


def create_app(brain=None, integrator=None, host='127.0.0.1', port=5555):
    global web_gui_instance
    logger.info("=== CREATE_APP CALLED ===")
    web_gui_instance = WebGUI(brain=brain, integrator=integrator, host=host, port=port)
    logger.info("=== WEBGUI INSTANCE CREATED ===")
    logger.info("web_gui_instance.auth_manager.users: {}".format(web_gui_instance.auth_manager.users))

    # Register all route modules (new refactored structure)
    register_core_routes(app, web_gui_instance)
    register_chat_routes(app, web_gui_instance)
    register_auth_routes(app, web_gui_instance)
    register_analytics_routes(app, web_gui_instance)
    register_knowledge_routes(app, web_gui_instance)
    register_upload_routes(app, web_gui_instance)
    
    # Additional routes from other modules
    # NOTE: server_routes.py conflicts with server_routes_core.py - NOT registering
    register_wikipedia_routes(app, web_gui_instance)
    register_export_routes(app, web_gui_instance)
    register_model_routes(app, web_gui_instance)
    register_graph_routes(app, web_gui_instance)

    web_gui_instance.start()
    return web_gui_instance


def get_app() -> Optional[WebGUI]:
    """
    Получить экземпляр WebGUI (singleton).
    Правильный способ доступа к инстансу - через эту функцию.
    """
    return web_gui_instance


def get_brain():
    """Получить brain из WebGUI инстанса."""
    if web_gui_instance and hasattr(web_gui_instance, 'brain'):
        return web_gui_instance.brain
    return None


# Direct execution - create app and run
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    create_app()
    app.run(host='127.0.0.1', port=5555, debug=False, use_reloader=False)
