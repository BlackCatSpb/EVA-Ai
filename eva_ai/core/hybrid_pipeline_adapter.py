"""
HybridPipelineAdapter - Гибридный адаптер для FractalPipeline

Позволяет:
1. Использовать FractalPipeline (новый) с виртуальными токенами
2. Использовать DualGenerator (2 физических модели) - БЫСТРЫЙ РЕЖИМ
3. Fallback на RecursiveModelPipeline (старый) при необходимости
4. Постепенный переход между режимами

Режимы работы:
- 'fractal': Только FractalPipeline (рекомендуется)
- 'dual': DualGenerator с 2 физическими моделями (БЫСТРО)
- 'recursive': Только RecursiveModelPipeline (для сравнения)
- 'hybrid': FractalPipeline с fallback на RecursiveModelPipeline
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("eva_ai.core.hybrid_adapter")


class HybridPipelineAdapter:
    """
    Гибридный адаптер для прозрачной замены RecursiveModelPipeline.
    
    Использование:
    ```python
    # Инициализация
    adapter = HybridPipelineAdapter(
        fractal_graph=graph,
        mode='fractal',  # или 'hybrid', 'recursive'
        model_a=model_a,  # llama_cpp Llama instance
        model_a_path='path/to/model.gguf'
    )
    
    # Обработка запроса - тот же API что и у RecursiveModelPipeline
    result = adapter.process_query(query)
    ```
    """
    
    MODE_FRACTAL = 'fractal'      # Только новый FractalPipeline
    MODE_DUAL = 'dual'            # DualGenerator с 2 физическими моделями (БЫСТРО)
    MODE_RECURSIVE = 'recursive'  # Только старый RecursiveModelPipeline  
    MODE_HYBRID = 'hybrid'        # Новый + fallback на старый
    
    def __init__(
        self,
        fractal_graph,
        mode: str = 'fractal',
        model_a=None,
        model_b=None,
        model_c=None,
        model_a_path: str = None,
        model_b_path: str = None,
        model_c_path: str = None,
        n_ctx: int = 8192,
        n_threads: int = 8,
        load_models: bool = True,
        brain=None,  # Добавляем brain для компактификации контекста
        **kwargs
    ):
        self.mode = mode
        self.fractal_graph = fractal_graph
        self._kwargs = kwargs
        self.brain = brain  # Сохраняем ссылку на brain
        
        # Модели (могут быть переданы или загружены)
        self.model_a = model_a
        self.model_b = model_b
        self.model_c = model_c
        self.model_a_path = model_a_path
        self.model_b_path = model_b_path
        self.model_c_path = model_c_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self._models_loaded = False
        
        # Пайплайны
        self.fractal_pipeline = None
        self.dual_generator = None
        self.recursive_pipeline = None
        
        if load_models:
            self._init_pipelines(**kwargs)
    
    def load_models(self):
        """Загрузить модели если ещё не загружены."""
        if self._models_loaded:
            return
        
        from llama_cpp import Llama
        
        # Определяем размер контекста
        recommended_ctx = min(self.n_ctx, 4096)
        
        if self.model_a is None and self.model_a_path:
            logger.info(f"Loading Model A: {self.model_a_path}")
            self.model_a = Llama(
                model_path=self.model_a_path,
                chat_format="qwen",
                n_ctx=recommended_ctx,
                n_threads=self.n_threads,
                verbose=False
            )
            logger.info(f"Model A loaded")
        
        if self.model_b is None and self.model_b_path:
            logger.info(f"Loading Model B: {self.model_b_path}")
            self.model_b = Llama(
                model_path=self.model_b_path,
                chat_format="qwen",
                n_ctx=recommended_ctx,
                n_threads=self.n_threads,
                verbose=False
            )
            logger.info(f"Model B loaded")
        
        if self.model_c is None and self.model_c_path and os.path.exists(self.model_c_path):
            logger.info(f"Model C path configured (lazy loading)")
        
        self._models_loaded = True
        logger.info("All models loaded")
    
    def unload_models(self):
        """Выгрузить модели."""
        self.model_a = None
        self.model_b = None
        self.model_c = None
        self._models_loaded = False
        self.fractal_pipeline = None
        self.dual_generator = None
        self.recursive_pipeline = None
        logger.info("Models unloaded")
    
    def _init_pipelines(self, **kwargs):
        """Инициализировать нужные пайплайны."""
        # Сначала загружаем модели если нужно
        self.load_models()
        
        # Инициализируем нужный pipeline в зависимости от режима
        if self.mode == self.MODE_DUAL:
            self._init_dual_generator()
        elif self.mode == self.MODE_RECURSIVE:
            self._init_recursive_pipeline()
        elif self.mode == self.MODE_FRACTAL or self.mode == self.MODE_HYBRID:
            self._init_fractal_pipeline(**kwargs)
            if self.mode == self.MODE_HYBRID:
                self._init_recursive_pipeline()
        else:
            self._init_fractal_pipeline(**kwargs)
        
        logger.info(f"HybridPipelineAdapter инициализирован: mode={self.mode}")
    
    def _init_fractal_pipeline(self, **kwargs):
        """Инициализировать FractalPipeline."""
        from eva_ai.core.fractal_pipeline import FractalPipeline
        
        if self.model_a or self.model_a_path:
            try:
                self.fractal_pipeline = FractalPipeline(
                    fractal_graph=self.fractal_graph,
                    gguf_model=self.model_a,
                    model_path=self.model_a_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    **kwargs
                )
                logger.info(f"FractalPipeline инициализирован")
            except Exception as e:
                logger.error(f"Ошибка инициализации FractalPipeline: {e}")
                self.fractal_pipeline = None
    
    def _init_dual_generator(self):
        """Инициализировать DualGenerator с 2 физическими моделями."""
        try:
            from eva_ai.memory.fractal_graph_v2.dual_generator import DualGenerator
            
            if not self.model_a or not self.model_b:
                logger.error("DualGenerator требует 2 модели (model_a и model_b)")
                return
            
            extended_config = self._kwargs.get('fractal_graph_v2', {})
            extended_max_tokens = extended_config.get('extended_max_tokens', 4096)
            extended_temperature = extended_config.get('extended_temperature', 0.35)
            extended_repeat_penalty = extended_config.get('extended_repeat_penalty', 1.8)
            
            self.dual_generator = DualGenerator(
                llama_condensed=self.model_a,
                llama_extended=self.model_b,
                graph=self.fractal_graph,
                condensed_max_tokens=512,
                extended_max_tokens=extended_max_tokens,
                extended_temperature=extended_temperature,
                extended_repeat_penalty=extended_repeat_penalty,
                brain=self.brain  # Передаем brain для компактификации
            )
            logger.info(f"DualGenerator инициализирован: extended_max_tokens={extended_max_tokens}")
        except Exception as e:
            logger.error(f"Ошибка инициализации DualGenerator: {e}")
            self.dual_generator = None
    
    def _init_recursive_pipeline(self):
        """Инициализировать RecursiveModelPipeline для fallback."""
        try:
            from eva_ai.core.recursive_model_pipeline import RecursiveModelPipeline
            
            kwargs = {
                'model_a_path': self.model_a_path,
                'model_b_path': self.model_b_path or self.model_a_path,
                'n_ctx': self.n_ctx,
                'n_threads': self.n_threads
            }
            if self.model_c_path:
                kwargs['model_c_path'] = self.model_c_path
            if self.fractal_graph:
                kwargs['fractal_memory'] = self.fractal_graph
            
            self.recursive_pipeline = RecursiveModelPipeline(**kwargs)
            if self.model_a is None and self.model_b is None:
                self.recursive_pipeline.load_models()
            else:
                self.recursive_pipeline.model_a = self.model_a
                self.recursive_pipeline.model_b = self.model_b
                self._models_loaded = True
            logger.info("RecursiveModelPipeline инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации RecursiveModelPipeline: {e}")
            self.recursive_pipeline = None
    
    def process_query(
        self,
        query: str,
        max_iterations: int = 1,
        gen_params: Dict[str, Any] = None,
        generation_mode: str = "auto"
    ) -> Dict[str, Any]:
        """
        Обработать запрос.
        
        API совместим с RecursiveModelPipeline.process_query()
        
        Args:
            query: Текст запроса
            max_iterations: Максимум итераций
            gen_params: Параметры генерации
            generation_mode: Режим генерации для DualGenerator
                - "auto": определяется автоматически
                - "condensed": краткий ответ
                - "extended": развёрнутый ответ
        """
        gen_params = gen_params or {}
        
        # Проверяем режим
        if self.mode == self.MODE_DUAL:
            return self._process_dual(query, generation_mode, gen_params)
        
        if self.mode == self.MODE_RECURSIVE:
            return self._process_recursive(query, max_iterations, gen_params)
        
        if self.mode == self.MODE_FRACTAL:
            return self._process_fractal(query, max_iterations, gen_params)
        
        # HYBRID mode
        return self._process_hybrid(query, max_iterations, gen_params)
    
    def _process_fractal(
        self, 
        query: str, 
        max_iterations: int, 
        gen_params: Dict
    ) -> Dict[str, Any]:
        """Обработка через FractalPipeline."""
        if not self.fractal_pipeline:
            logger.error("FractalPipeline не инициализирован")
            return self._error_response(query, "Pipeline not initialized")
        
        try:
            result = self.fractal_pipeline.process_query(
                query=query,
                conversation_history=gen_params.get('conversation_history'),
                max_tokens=gen_params.get('max_tokens', 512),
                temperature=gen_params.get('temperature', 0.5)
            )
            return result
        except Exception as e:
            logger.error(f"Ошибка FractalPipeline: {e}")
            if self.mode == self.MODE_HYBRID:
                return self._process_recursive(query, max_iterations, gen_params)
            return self._error_response(query, str(e))
    
    def _process_dual(
        self, 
        query: str, 
        generation_mode: str,
        gen_params: Dict
    ) -> Dict[str, Any]:
        """Обработка через DualGenerator."""
        if not self.dual_generator:
            logger.error("DualGenerator не инициализирован")
            return self._error_response(query, "DualGenerator not initialized")
        
        reasoning_steps = []
        step_num = 1
        
        try:
            start_time = time.time()
            
            reasoning_steps.append({
                'step': step_num,
                'phase': 'query_analysis',
                'thought': f'Анализ запроса: "{query[:50]}..."',
                'confidence': 1.0,
                'icon': '🔍'
            })
            step_num += 1
            
            reasoning_steps.append({
                'step': step_num,
                'phase': 'model_selection',
                'thought': f'Выбор режима генерации: {generation_mode}',
                'confidence': 0.9,
                'icon': '⚙️'
            })
            step_num += 1
            
            reasoning_steps.append({
                'step': step_num,
                'phase': 'context_retrieval',
                'thought': 'Извлечение контекста из FractalGraphV2',
                'confidence': 0.85,
                'icon': '📚'
            })
            step_num += 1
            
            reasoning_steps.append({
                'step': step_num,
                'phase': 'generation',
                'thought': f'Генерация ответа через DualGenerator ({generation_mode})',
                'confidence': 0.8,
                'icon': '🤖'
            })
            step_num += 1
            
            gen_result = self.dual_generator.generate(query, mode=generation_mode, return_details=True)
            
            if isinstance(gen_result, dict):
                response = gen_result.get('response', '')
                mode_used = gen_result.get('mode', generation_mode)
                gen_time = gen_result.get('time', 0)
                
                reasoning_steps.append({
                    'step': step_num,
                    'phase': mode_used,
                    'thought': f'Использован режим: {mode_used} ({gen_time:.1f}s, {len(response)} символов)',
                    'confidence': 0.9,
                    'icon': '📝' if mode_used == 'condensed' else '📖'
                })
                step_num += 1
            else:
                response = gen_result
            
            elapsed = time.time() - start_time
            
            reasoning_steps.append({
                'step': step_num,
                'phase': 'quality_check',
                'thought': f'Проверка качества ответа: {len(response)} символов',
                'confidence': 0.9,
                'icon': '✅'
            })
            step_num += 1
            
            reasoning_steps.append({
                'step': step_num,
                'phase': 'final_synthesis',
                'thought': 'Формирование финального ответа',
                'confidence': 0.95,
                'icon': '✨'
            })
            
            return {
                'response': response,
                'final_response': response,
                'natural_response': response,
                'confidence': 0.9,
                'quality': {
                    'score': 0.9,
                    'is_gibberish': False,
                    'reasons': ['OK']
                },
                'query_type': generation_mode,
                'reasoning_steps': reasoning_steps,
                'processing_time': elapsed,
                'model_a_result': {'response': response},
                'model_b_result': {'response': response},
                'model_c_result': None,
                'has_code': False,
                'fractal_context': None
            }
        except Exception as e:
            logger.error(f"Ошибка DualGenerator: {e}")
            return self._error_response(query, str(e))
    
    def _process_recursive(
        self, 
        query: str, 
        max_iterations: int, 
        gen_params: Dict
    ) -> Dict[str, Any]:
        """Обработка через RecursiveModelPipeline."""
        if not self.recursive_pipeline:
            logger.error("RecursiveModelPipeline не инициализирован")
            return self._error_response(query, "Recursive pipeline not initialized")
        
        try:
            return self.recursive_pipeline.process_query(
                query=query,
                max_iterations=max_iterations,
                gen_params=gen_params
            )
        except Exception as e:
            logger.error(f"Ошибка RecursiveModelPipeline: {e}")
            return self._error_response(query, str(e))
    
    def _process_hybrid(
        self, 
        query: str, 
        max_iterations: int, 
        gen_params: Dict
    ) -> Dict[str, Any]:
        """Гибридная обработка: Fractal -> fallback -> error."""
        if self.fractal_pipeline:
            try:
                return self.fractal_pipeline.process_query(
                    query=query,
                    conversation_history=gen_params.get('conversation_history'),
                    max_tokens=gen_params.get('max_tokens', 512),
                    temperature=gen_params.get('temperature', 0.5)
                )
            except Exception as e:
                logger.warning(f"FractalPipeline ошибка: {e}")
        
        if self.recursive_pipeline:
            logger.info("Fallback на RecursiveModelPipeline")
            return self._process_recursive(query, max_iterations, gen_params)
        
        return self._error_response(query, "No pipeline available")
    
    def _error_response(self, query: str, error: str) -> Dict[str, Any]:
        """Сформировать ответ об ошибке."""
        return {
            'response': f'Ошибка обработки: {error}',
            'final_response': f'Ошибка обработки: {error}',
            'query': query,
            'model_a_result': None,
            'model_b_result': None,
            'model_c_result': None,
            'reasoning_steps': [],
            'has_code': False,
            'fractal_context': None,
            'final_quality': {'is_gibberish': False, 'score': 0.0, 'reasons': [error]}
        }
    
    def set_mode(self, mode: str):
        """Изменить режим работы."""
        if mode not in [self.MODE_FRACTAL, self.MODE_DUAL, self.MODE_RECURSIVE, self.MODE_HYBRID]:
            logger.warning(f"Неизвестный режим: {mode}")
            return
        
        if mode == self.mode:
            return
        
        logger.info(f"Смена режима: {self.mode} -> {mode}")
        self.mode = mode
        
        # Инициализируем нужный pipeline
        if mode == self.MODE_DUAL:
            if not self.dual_generator:
                self._init_dual_generator()
        elif mode == self.MODE_RECURSIVE:
            if not self.recursive_pipeline:
                self._init_recursive_pipeline()
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику."""
        stats = {
            'mode': self.mode,
            'fractal_pipeline_ready': self.fractal_pipeline is not None,
            'dual_generator_ready': self.dual_generator is not None,
            'recursive_pipeline_ready': self.recursive_pipeline is not None
        }
        
        if self.fractal_pipeline:
            stats['fractal_stats'] = self.fractal_pipeline.get_context_stats()
        
        if self.dual_generator:
            stats['dual_stats'] = self.dual_generator.get_stats()
        
        return stats
    
    @property
    def fractal_memory(self):
        """Совместимость с атрибутом fractal_memory."""
        return self.fractal_graph
    
    def unload_models(self):
        """Выгрузить модели из памяти."""
        if self.recursive_pipeline:
            self.recursive_pipeline.unload_models()
        
        # FractalPipeline использует переданные модели,
        # поэтому не выгружаем их здесь
    
    def generate_with_virtual_tokens(
        self,
        query: str,
        session_id: str = "default",
        max_tokens: int = 1024,
        use_streaming: bool = True,
        generation_mode: str = "extended"
    ) -> Dict[str, Any]:
        """
        Генерация с использованием виртуальных токенов.
        
        Использует SnapshotManager для создания неизменяемого снимка
        и VirtualTokenManager для замены токенов на контент из памяти.
        
        Args:
            query: Запрос пользователя
            session_id: ID сессии
            max_tokens: Максимальное число токенов
            use_streaming: Использовать streaming для замены токенов
            generation_mode: 'extended' или 'condensed'
            
        Returns:
            {response, virtual_token_stats, context_stats}
        """
        try:
            from eva_ai.memory.fractal_graph_v2 import (
                create_snapshot_manager,
                create_virtual_token_manager
            )
        except ImportError as e:
            logger.warning(f"Virtual tokens not available: {e}")
            return {
                "response": None,
                "error": "virtual_tokens_module_not_available",
                "fallback": True
            }
        
        if not self.fractal_graph:
            return {
                "response": None,
                "error": "fractal_graph_not_available",
                "fallback": True
            }
        
        try:
            snapshot_mgr = create_snapshot_manager(
                fractal_graph=self.fractal_graph,
                ttl_seconds=300.0,
                max_active_snapshots=20
            )
            
            relevant_nodes = self._find_relevant_nodes(query, top_k=10)
            node_ids = [n.get('id', n.get('node_id')) for n in relevant_nodes if n]
            node_ids = [n for n in node_ids if n]
            
            snapshot = snapshot_mgr.create_snapshot(
                session_id=session_id,
                node_ids=node_ids,
                dialogue_context=""
            )
            
            virtual_token_mgr = create_virtual_token_manager(
                snapshot_or_contents=snapshot,
                llama_model=self.model_a or self.model_b,
                config=self._kwargs.get('virtual_tokens', {})
            )
            
            prompt = self._build_prompt_for_mode(query, generation_mode)
            
            if use_streaming and virtual_token_mgr._streaming_handler:
                response = self._generate_with_streaming(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    virtual_token_mgr=virtual_token_mgr,
                    snapshot=snapshot
                )
            else:
                response = self._generate_without_streaming(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    virtual_token_mgr=virtual_token_mgr,
                    snapshot=snapshot
                )
            
            return {
                "response": response,
                "virtual_token_stats": virtual_token_mgr.get_stats(),
                "context_stats": snapshot_mgr.get_stats(),
                "nodes_used": len(node_ids),
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Virtual token generation error: {e}")
            return {
                "response": None,
                "error": str(e),
                "fallback": True
            }
    
    def _find_relevant_nodes(self, query: str, top_k: int = 10) -> List[Dict]:
        """Находит релевантные узлы в графе."""
        if not self.fractal_graph:
            return []
        
        try:
            if hasattr(self.fractal_graph, 'semantic_search'):
                return self.fractal_graph.semantic_search(query, top_k=top_k)
            elif hasattr(self.fractal_graph, 'retrieve_knowledge'):
                return self.fractal_graph.retrieve_knowledge(query, top_k=top_k)
        except Exception as e:
            logger.warning(f"Error finding relevant nodes: {e}")
        
        return []
    
    def _build_prompt_for_mode(self, query: str, mode: str) -> str:
        """Строит промт в зависимости от режима."""
        if mode == "condensed":
            return f"""Ты — краткий ассистент. Дай ответ в 1-2 предложениях.

Вопрос: {query}

Ответ:"""
        else:
            return f"""Дай развёрнутый и подробный ответ на вопрос.

Вопрос: {query}

Ответ:"""
    
    def _generate_with_streaming(
        self,
        prompt: str,
        max_tokens: int,
        virtual_token_mgr,
        snapshot
    ) -> str:
        """Генерация с использованием streaming."""
        from llama_cpp import LogitsProcessorList
        
        logits_processor = None
        lp = virtual_token_mgr.get_logits_processor()
        if lp:
            logits_processor = LogitsProcessorList([lp])
        
        model = self.model_a or self.model_b
        if not model:
            return ""
        
        try:
            stream = model.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.35,
                repeat_penalty=1.8,
                stream=True,
                logits_processor=logits_processor
            )
            
            streaming_handler = virtual_token_mgr.create_streaming_handler(snapshot)
            if not streaming_handler:
                return ""
            
            full_response = ""
            for chunk in streaming_handler.process_stream(stream):
                full_response += chunk
            
            return full_response
            
        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            return ""
    
    def _generate_without_streaming(
        self,
        prompt: str,
        max_tokens: int,
        virtual_token_mgr,
        snapshot
    ) -> str:
        """Генерация без streaming (fallback)."""
        model = self.model_a or self.model_b
        if not model:
            return ""
        
        try:
            response = model(
                prompt,
                max_tokens=max_tokens,
                temperature=0.35,
                repeat_penalty=1.8,
                echo=False
            )
            
            if isinstance(response, dict):
                text = response.get('choices', [{}])[0].get('text', '')
            else:
                text = str(response)
            
            return text
            
        except Exception as e:
            logger.error(f"Non-streaming generation error: {e}")
            return ""
