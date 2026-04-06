"""
Flask app creation, WebGUI class, imports from other modules, re-exports
"""
import os
import logging
import threading
import json
import uuid
import hashlib
import secrets
from datetime import datetime
from typing import Dict, Any, Optional

from flask import Flask

logger = logging.getLogger("eva.webgui")

from eva.gui.web_gui.server_auth import SessionManager, AuthManager, EntityExtractor, EthicsChecker
from eva.gui.web_gui.server_routes import register_routes as register_basic_routes
from eva.gui.web_gui.server_routes import extract_text_from_file
from eva.gui.web_gui.server_api_wikipedia import register_routes as register_wikipedia_routes
from eva.gui.web_gui.server_api_knowledge import register_routes as register_knowledge_routes
from eva.gui.web_gui.server_api_export import register_routes as register_export_routes
from eva.gui.web_gui.server_models import register_routes as register_model_routes


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

        self.auth_manager = AuthManager()
        self.session_manager = SessionManager()
        self.entity_extractor = EntityExtractor()
        self.ethics_checker = EthicsChecker()

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

    def process_message(self, query: str, session_id: str, user_id: str, file_data: Dict = None) -> Dict[str, Any]:

        if file_data and file_data.get('extracted_text'):
            filename = file_data.get('filename', 'file')
            extracted = file_data['extracted_text']
            enhanced_query = f"""Пользователь прикрепил файл "{filename}".
Содержимое файла:
---
{extracted}
---

Запрос пользователя: {query}"""
            query = enhanced_query

        ethics_result = self.ethics_checker.check_message(query)
        if not ethics_result['allowed']:
            return {
                'response': 'Извините, это сообщение заблокировано.',
                'status': 'blocked',
                'reason': ethics_result.get('reason')
            }

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

        response_text = "Система обрабатывает запрос..."

        result = None
        debug_info = {"brain": self.brain is not None, "has_process_query": False}

        conversation_history = []
        if session_id and isinstance(session_id, str) and session_id.strip():
            conversation_history = self.session_manager.get_chat_history(session_id, limit=20)

        user_context = {
            'session_id': session_id,
            'user_id': user_id,
            'conversation_history': conversation_history
        }

        result = None
        if self.brain and hasattr(self.brain, 'process_query'):
            debug_info["has_process_query"] = True
            debug_info["brain_loaded"] = self.brain is not None and hasattr(self.brain, 'self_reasoning_engine') and self.brain.self_reasoning_engine is not None
            debug_info["enhanced_reasoning_loaded"] = self.brain is not None and hasattr(self.brain, 'enhanced_reasoning_engine') and self.brain.enhanced_reasoning_engine is not None
            result = self.brain.process_query(query, user_context)
            debug_info["result_keys"] = list(result.keys()) if result and isinstance(result, dict) else []
            debug_info["result_reasoning"] = str(result.get('reasoning'))[:100] if result and isinstance(result, dict) else None
            debug_info["result_source"] = result.get('source') if result and isinstance(result, dict) else None
            logger.debug(f"brain result: source={result.get('source') if result and isinstance(result, dict) else 'None'}")
            if result and isinstance(result, dict):
                response_text = result.get('response', result.get('text', response_text))
        else:
            debug_info["reason"] = "no brain or no process_query"

        brain_reasoning = None
        reasoning_data = None
        reasoning_steps = []
        search_results = []
        web_search_info = None

        # Извлекаем ethics и contradiction из brain result ДО обработки по источникам
        brain_ethics = None
        brain_contradiction = None
        if result and isinstance(result, dict):
            brain_reasoning = result.get('reasoning')
            brain_reasoning_raw = result.get('reasoning_raw')
            source = result.get('source', '')
            confidence = result.get('confidence', 0)
            search_results = result.get('search_results', [])

            if result.get('reasoning_steps'):
                reasoning_steps = result.get('reasoning_steps', [])
                logger.debug(f"Got {len(reasoning_steps)} reasoning steps from result")

            # Этическая оценка из brain
            if result.get('ethics_result') and not ethics_result:
                ethics_result = result.get('ethics_result')
            brain_ethics = ethics_result

            # Противоречия из brain
            brain_contradiction = result.get('contradiction_result')

            if search_results and len(search_results) > 0:
                web_search_info = f"Найдено {len(search_results)} результатов:"
                for i, sr in enumerate(search_results[:3]):
                    title = sr.get('title', 'No title')[:60]
                    url = sr.get('url', '')[:50]
                    web_search_info += f"\n{i+1}. {title}... ({url})"

            if source == 'llama_cpp_with_modules' or (file_data and file_data.get('extracted_text')):
                reasoning_steps = []

                if file_data and file_data.get('extracted_text'):
                    filename = file_data.get('filename', 'file')
                    text_len = len(file_data.get('extracted_text', ''))
                    reasoning_steps.append({
                        'step': 0,
                        'phase': 'document_analysis',
                        'thought': f'Анализ документа "{filename}" - извлечено {text_len} символов',
                        'confidence': 0.9
                    })

                reasoning_steps.append({
                    'step': len(reasoning_steps) + 1,
                    'phase': 'generation',
                    'thought': 'Первичная генерация ответа через LlamaCpp (GGUF)',
                    'confidence': 0.5
                })

                # Противоречия
                contr_count = 0
                if brain_contradiction:
                    contr_count = brain_contradiction.get('significant_count', 0)
                    contr_conf = 1.0 - brain_contradiction.get('contradiction_level', 0.0)
                    reasoning_steps.append({
                        'step': 2,
                        'phase': 'contradiction_check',
                        'thought': f'Проверка противоречий: {contr_count} найдено, уровень={brain_contradiction.get("contradiction_level", 0):.2f}',
                        'confidence': contr_conf
                    })

                # Этика
                if brain_ethics:
                    has_violations = brain_ethics.get('has_violations', False)
                    ethics_conf = brain_ethics.get('is_ethical', 1.0)
                    reasoning_steps.append({
                        'step': 3,
                        'phase': 'ethics_check',
                        'thought': f'Проверка этики: violations={has_violations}, score={ethics_conf:.2f}',
                        'confidence': ethics_conf
                    })

                if search_results and len(search_results) > 0:
                    reasoning_steps.append({
                        'step': 4,
                        'phase': 'web_search',
                        'thought': f'Веб-поиск: найдено {len(search_results)} результатов',
                        'confidence': 0.8
                    })

                    reasoning_steps.append({
                        'step': 5,
                        'phase': 'refinement',
                        'thought': 'Перегенерация с контекстом из веб-поиска',
                        'confidence': 0.9
                    })
                elif contr_count > 0 or (brain_ethics and brain_ethics.get('has_violations', False)):
                    reasoning_steps.append({
                        'step': 4,
                        'phase': 'refinement',
                        'thought': 'Перегенерация после исправления модулей',
                        'confidence': 0.7
                    })

                reasoning_steps.append({
                    'step': len(reasoning_steps) + 1,
                    'phase': 'final_synthesis',
                    'thought': 'Финальный ответ с учетом всех проверок',
                    'confidence': result.get('confidence', 0.9)
                })

                reasoning_data = "Рассуждения системы (qwen_only_mode):\n\n" + "\n".join([
                    f"{s['step']}. [{s['phase']}] {s['thought']} (conf: {s['confidence']:.2f})"
                    for s in reasoning_steps
                ])

            elif source == 'self_reasoning_engine':
                if result.get('reasoning_steps'):
                    reasoning_steps = result.get('reasoning_steps', [])
                    reasoning_text = "Рассуждения системы:\n\n"
                    for i, step in enumerate(reasoning_steps):
                        phase = step.get('phase', 'unknown')
                        thought = step.get('thought', '')
                        conf = step.get('confidence', 0)
                        model = step.get('model', '')
                        if i < 15:
                            reasoning_text += f"{i+1}. [{phase}] {thought} (conf: {conf:.2f})\n"
                    reasoning_data = reasoning_text
                elif brain_reasoning_raw and isinstance(brain_reasoning_raw, dict):
                    steps = brain_reasoning_raw.get('steps', [])
                    if steps:
                        reasoning_text = "Рассуждения системы:\n\n"
                        for i, step in enumerate(steps):
                            phase = step.get('phase', 'unknown')
                            thought = step.get('thought', '')
                            conf = step.get('confidence', 0)
                            reasoning_steps.append({
                                'step': i + 1,
                                'phase': phase,
                                'thought': thought,
                                'confidence': conf
                            })
                            if i < 10:
                                reasoning_text += f"{i+1}. [{phase}] {thought} (conf: {conf:.2f})\n"
                        reasoning_data = reasoning_text
                    elif brain_reasoning:
                        reasoning_data = str(brain_reasoning)
                elif brain_reasoning:
                    reasoning_data = str(brain_reasoning)

            elif source == 'enhanced_reasoning_engine':
                if brain_reasoning_raw and isinstance(brain_reasoning_raw, dict):
                    chain = brain_reasoning_raw.get('reasoning_chain', [])
                    if chain:
                        reasoning_text = "Регенерация ответа:\n\n"
                        for i, iteration in enumerate(chain):
                            resp_preview = iteration.get('response', '')[:80]
                            conf = iteration.get('confidence', 0)
                            has_contr = iteration.get('has_contradictions', False)
                            has_ethics = iteration.get('has_ethics_issues', False)
                            module_prompts = iteration.get('module_prompts', {})

                            prompts_text = ""
                            if module_prompts:
                                prompts_text = "\nПромты модулей:"
                                for mod, prompt in module_prompts.items():
                                    prompts_text += f"\n  [{mod.upper()}]: {prompt[:100]}..."

                            reasoning_steps.append({
                                'step': i + 1,
                                'phase': 'regeneration',
                                'thought': resp_preview,
                                'confidence': conf,
                                'has_contradictions': has_contr,
                                'has_ethics_issues': has_ethics,
                                'module_prompts': prompts_text
                            })
                            reasoning_text += f"{i+1}. [regeneration] {resp_preview} (conf: {conf:.2f})\n"
                            if has_contr:
                                reasoning_text += f"   ⚠️ Противоречия обнаружены\n"
                            if has_ethics:
                                reasoning_text += f"   ⚠️ Этические проблемы обнаружены\n"
                            if prompts_text:
                                reasoning_text += prompts_text + "\n"
                        reasoning_data = reasoning_text
                    elif brain_reasoning:
                        reasoning_data = str(brain_reasoning)
                elif brain_reasoning:
                    reasoning_data = str(brain_reasoning)

            else:
                if brain_reasoning:
                    reasoning_data = str(brain_reasoning)

            # Добавляем ethics и contradiction шаги для НЕ-llama_cpp источников
            # (для llama_cpp_with_modules они уже добавлены внутри блока выше)
            if source != 'llama_cpp_with_modules' and not (file_data and file_data.get('extracted_text')):
                # Противоречия
                if brain_contradiction:
                    contr_count = brain_contradiction.get('significant_count', 0)
                    contr_level = brain_contradiction.get('contradiction_level', 0.0)
                    contr_conf = 1.0 - contr_level
                    reasoning_steps.append({
                        'step': len(reasoning_steps) + 1,
                        'phase': 'contradiction_check',
                        'thought': f'Проверка противоречий: найдено={contr_count}, уровень={contr_level:.2f}',
                        'confidence': contr_conf
                    })

                # Этика
                if brain_ethics:
                    has_violations = brain_ethics.get('has_violations', False)
                    ethics_conf = brain_ethics.get('is_ethical', brain_ethics.get('confidence', 1.0))
                    reasoning_steps.append({
                        'step': len(reasoning_steps) + 1,
                        'phase': 'ethics_check',
                        'thought': f'Этическая оценка: violations={has_violations}, score={ethics_conf:.2f}',
                        'confidence': ethics_conf
                    })
                elif brain_reasoning:
                    reasoning_data = str(brain_reasoning)

            elif source == 'enhanced_reasoning_engine':
                if brain_reasoning_raw and isinstance(brain_reasoning_raw, dict):
                    chain = brain_reasoning_raw.get('reasoning_chain', [])
                    if chain:
                        reasoning_text = "Регенерация ответа:\n\n"
                        for i, iteration in enumerate(chain):
                            resp_preview = iteration.get('response', '')[:80]
                            conf = iteration.get('confidence', 0)
                            has_contr = iteration.get('has_contradictions', False)
                            has_ethics = iteration.get('has_ethics_issues', False)
                            module_prompts = iteration.get('module_prompts', {})

                            prompts_text = ""
                            if module_prompts:
                                prompts_text = "\nПромты модулей:"
                                for mod, prompt in module_prompts.items():
                                    prompts_text += f"\n  [{mod.upper()}]: {prompt[:100]}..."

                            reasoning_steps.append({
                                'step': i + 1,
                                'phase': 'regeneration',
                                'thought': resp_preview,
                                'confidence': conf,
                                'has_contradictions': has_contr,
                                'has_ethics_issues': has_ethics,
                                'module_prompts': module_prompts
                            })

                            status = ""
                            if has_contr:
                                status += " [противоречия]"
                            if has_ethics:
                                status += " [этика]"

                            reasoning_text += f"Итерация {i+1}: {resp_preview}...{status}\n"
                            reasoning_text += f"  Уверенность: {conf:.2f}\n"
                            if prompts_text:
                                reasoning_text += f"  {prompts_text}\n"
                            reasoning_text += "\n"

                        reasoning_data = reasoning_text
                    elif brain_reasoning:
                        reasoning_data = str(brain_reasoning)
                elif brain_reasoning:
                    reasoning_data = str(brain_reasoning)
            elif source == 'qwen_model' and confidence >= 0.85:
                reasoning_data = f"🤖 Qwen Model обработал запрос (уверенность: {confidence:.2f})"

        if session_id and response_text:
            self.session_manager.add_chat_message(session_id, 'assistant', response_text)

            response_node = {
                'id': str(uuid.uuid4()),
                'assistant_message': response_text,
                'timestamp': datetime.now().isoformat(),
                'reasoning': reasoning_data
            }
            self.session_manager.add_context_node(session_id, response_node)

            if self.brain and hasattr(self.brain, 'fractal_memory') and self.brain.fractal_memory:
                self.session_manager.convert_chat_to_knowledge(session_id, self.brain.fractal_memory)

        self_dialog_result = None
        # Самодиалог НЕ запускается при каждом запросе!
        # Он запускается автоматически в brain_query.py когда система не знает ответа (обнаружены unknown_patterns)
        # Во время генерации ответа — только рассуждения, самодиалог это фоновое обучение

        return_data = {
            'response': response_text,
            'status': 'ok',
            'warnings': ethics_result.get('warnings', []) if ethics_result else [],
            'ethics_result': ethics_result,
            'contradiction_metrics': brain_contradiction,
            'reasoning': reasoning_data,
            'reasoning_steps': reasoning_steps,
            'self_dialog': self_dialog_result,
            'search_results': search_results if search_results else None,
            'web_search_info': web_search_info
        }

        return return_data

    def start(self):
        if self.running:
            return

        self.running = True

        def run():
            import click
            click.echo = lambda *args, **kwargs: None
            app.run(host=self.host, port=self.port, debug=False, use_reloader=False, threaded=True)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

        logger.info(f"WebGUI сервер запущен на http://{self.host}:{self.port}")

    def stop(self):
        self.running = False
        logger.info("WebGUI сервер остановлен")


# Don't create default instance at import time - let create_app handle it
web_gui_instance: Optional[WebGUI] = None


def create_app(brain=None, integrator=None, host='127.0.0.1', port=5555):
    global web_gui_instance
    logger.info("=== CREATE_APP CALLED ===")
    web_gui_instance = WebGUI(brain=brain, integrator=integrator, host=host, port=port)
    logger.info("=== WEBGUI INSTANCE CREATED ===")
    logger.info("web_gui_instance.auth_manager.users: {}".format(web_gui_instance.auth_manager.users))

    register_basic_routes(app, web_gui_instance)
    register_wikipedia_routes(app, web_gui_instance)
    register_knowledge_routes(app, web_gui_instance)
    register_export_routes(app, web_gui_instance)
    register_model_routes(app, web_gui_instance)

    web_gui_instance.start()
    return web_gui_instance


def get_app() -> WebGUI:
    return web_gui_instance


# Direct execution - create app and run
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    create_app()
    app.run(host='127.0.0.1', port=5555, debug=False, use_reloader=False)
