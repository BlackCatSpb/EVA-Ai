# EVA AI - Функциональная карта системы

> Дата: 2026-04-13
> Файлов с кодом: 469
> Модулей: 38

---

## Содержание

| Модуль | Файлов | Описание |
|--------|--------|----------|
| [__main__](#__main__) | 1 | Точка входа |
| [adaptation](#adaptation) | 7 | Адаптация: Analytics, Core |
| [adapters](#adapters) | 1 | Адаптеры: Torch |
| [analytics](#analytics) | 4 | Аналитика: Manager, Integration |
| [backends](#backends) | 5 | Бэкенды: PIE, GGUF, ONNX |
| [config](#config) | 2 | Конфиг: Apply |
| [contradiction](#contradiction) | 22 | Противоречия: Generator, Miner, Manager |
| [core](#core) | 102 | Ядро: UnifiedGenerator, Brain Query, Pipeline, Event Bus |
| [distributed](#distributed) | 8 | Распределённость: Cluster, Tasks, Sync |
| [ethics](#ethics) | 13 | Этика: Framework, Checks, Violations |
| [fractal](#fractal) | 2 | Фрактал: Store |
| [generation](#generation) | 1 | Генерация: Coordinator |
| [gui](#gui) | 54 | GUI: Web Server, Chat, Widgets |
| [knowledge](#knowledge) | 10 | Знания: ConceptExtractor, ConceptMiner, KG |
| [learning](#learning) | 31 | Самообучение: Dialog, Scheduler, Opportunity |
| [memory](#memory) | 71 | Память: HybridCache, FractalGraph, DocumentManager |
| [mlearning](#mlearning) | 66 | ML: Models, Tokenizers, Training |
| [monitoring](#monitoring) | 1 | Мониторинг: System Monitor |
| [neuromorphic](#neuromorphic) | 6 | Нейроморфные: Memory, Sim |
| [nlp](#nlp) | 1 | NLP: Text Processor |
| [nlp_fallbacks](#nlp_fallbacks) | 1 |  |
| [preprocess](#preprocess) | 1 | Предобработка: Pipeline |
| [reasoning](#reasoning) | 23 | Reasoning: Engine, Types |
| [recovery](#recovery) | 1 | Восстановление: System |
| [run](#run) | 1 |  |
| [runtime](#runtime) | 2 | Рантайм: Worker Pool |
| [scripts](#scripts) | 8 | Скрипты: Загрузка, Миграция |
| [security](#security) | 1 | Безопасность: Framework |
| [server_handlers](#server_handlers) | 1 |  |
| [server_routes](#server_routes) | 1 |  |
| [storage](#storage) | 2 | Хранилище: Types |
| [system](#system) | 3 | Система: Health, Fault Tolerance |
| [system_selftest](#system_selftest) | 1 |  |
| [tests](#tests) | 1 |  |
| [tools](#tools) | 5 | Инструменты: Dependency Scan |
| [training](#training) | 1 | Обучение: GGUF Training |
| [utils](#utils) | 1 |  |
| [websearch](#websearch) | 7 | Поиск: Tavily, Search Engine |


---

## Детальная функциональная карта

## __MAIN__

### __main__

**Методы:** main

---

## ADAPTATION

### adaptation_analytics

**Методы:** get_adaptation_metrics, get_system_health, get_adaptation_dashboard_data, _get_analytics_data, get_adaptation_insights, apply_adaptation_insights

---

### adaptation_core

**Классы:** AdaptationManager

**Методы:** initialize, _init_db, _get_connection, _load_profiles, _save_profiles, _load_feedback, _save_feedback, _update_user_statistics, _update_feedback_statistics, _save_analytics_snapshot

---

### adaptation_integrated

**Классы:** IntegratedAdaptationManager

**Методы:** _do_initialize, _do_start, _do_stop, adapt_response, _basic_adaptation, create_user_profile, _create_basic_profile, process_feedback, _process_basic_feedback, get_statistics

---

### adaptation_integration

**Методы:** get_user_profile, update_user_profile, record_feedback, _update_profile_from_feedback, analyze_user_patterns, export_adaptation_data, get_feedback_history, import_adaptation_data, integrate_with_knowledge_graph, get_cultural_adaptation

---

### adaptation_manager

**Классы:** AdaptationManager

**Методы:** _load_data, _cleanup_profiles, _save_data, _start_background_analysis, _background_analysis_worker, _analyze_user_patterns, _update_statistics, _is_user_active, initialize, start

---

### adaptation_profiles

**Классы:** UserFeedback, UserProfile

**Методы:** to_dict, from_dict, to_dict, from_dict

---

### adaptation_types

**Классы:** AdaptationLevel, LearningStyle, UserPreferences, AdaptationProfile

**Методы:** to_dict, to_dict

---

## ADAPTERS

### torch_adapter

**Классы:** Meta, Batch, TorchBatchAdapter

**Методы:** pad_1d, default_collate, _now, push, _should_emit, try_pop_batch, flush

---

## ANALYTICS

### analytics_integrated

**Классы:** IntegratedAnalyticsManager

**Методы:** _do_initialize, _do_start, _do_stop, track_query, get_performance_metrics, get_system_health, _save_metrics, generate_report

---

### analytics_manager

**Классы:** AnalyticsManager

**Методы:** record, _init_components, start_monitoring, stop_monitoring, _monitoring_loop, _collect_performance_metrics, _collect_learning_metrics, _collect_system_metrics, _analyze_metrics, _analyze_performance_trends

---

### contradiction_analyzer

**Классы:** ContradictionAnalyzer, RelevanceCalculator

**Методы:** detect_contradictions, calculate_divergence, _check_negation_contradiction, _check_number_contradiction, _check_temporal_contradiction, _check_keyword_contradiction, _split_into_sentences, _categorize_contradiction, get_significant_contradictions, analyze_contradiction_patterns

---

### learning_integration

**Классы:** AnalyticsLearningIntegration

**Методы:** analyze_learning_effectiveness, _analyze_model_performance, _get_model_info, _analyze_knowledge_expansion, _analyze_contradiction_resolution, _get_contradiction_stats, _generate_learning_recommendations, execute_learning_recommendation, _retrain_models, _expand_knowledge

---

## BACKENDS

### base

**Классы:** GenerationResult, GenerationConfig, BaseBackend

**Методы:** load_model, generate, generate_stream, tokenize, detokenize, get_model_info, unload, get_memory_usage, is_model_loaded

---

### gguf_backend

**Классы:** GGUFBackend

**Методы:** load_model, generate, generate_stream, tokenize, detokenize, get_model_info, unload, get_memory_usage, _extract_model_info

---

### layer_wise

**Классы:** LayerConfig, LayerCache, EmbeddingLayer, LayerWiseEmbedder

**Методы:** get, put, clear, _load_model, embed, _quantize, get_stats, add_layer, remove_layer, embed_sequential

---

### onnx_backend

**Классы:** ONNXBackend

**Методы:** load_model, generate, generate_stream, tokenize, detokenize, get_model_info, unload

---

### transformers_backend

**Классы:** TransformersBackend

**Методы:** load_model, generate, generate_stream, tokenize, detokenize, get_model_info, unload

---

## CONFIG

### __init__

**Методы:** is_model_allowed, is_model_loading_disabled, is_embedding_loading_disabled, is_tokenizer_loading_disabled

---

### apply_optimal_config

**Методы:** get_config_path, load_config, apply_fractal_config, apply_gui_config

---

## CONTRADICTION

### contradiction_analysis

**Классы:** ContradictionAnalyzer

**Методы:** get_contradiction_type, get_severity, get_resolution_priority, _calculate_concept_importance, calculate_semantic_divergence, analyze_source_credibility, calculate_nlp_metrics, get_resolution_strategy, _get_numeric_conflict_strategy, _get_boolean_conflict_strategy

---

### contradiction_generator

**Классы:** GeneratedContradiction, ContradictionGenerator

**Методы:** _load_templates, generate_contradiction, _generate_reasoning, save_contradiction, format_for_dialog, generate_batch, auto_generate_for_unknown_concepts, _has_contradiction, get_contradictions_for_prompt, save_resolution

---

### contradiction_integrated

**Классы:** IntegratedContradictionManager

**Методы:** _do_initialize, _do_start, _do_stop, detect_contradiction, _basic_contradiction_detection, resolve_contradiction, _basic_contradiction_resolution, get_contradiction_statistics, _load_contradictions_db, _save_contradictions_db

---

### contradiction_learning_append

**Классы:** ContradictionLearning

**Методы:** create_learning_opportunity, get_learning_opportunities, execute_learning_cycle, get_learning_statistics

---

### contradiction_manager

**Классы:** ContradictionManager

**Методы:** _setup_component, _initialize_components, verify_fact_with_web_search, check_concept_with_web_search, get_known_concepts, add_contradiction, get_contradictions, detect_contradictions, resolve_contradiction, get_contradiction_stats

---

### contradiction_miner

**Классы:** ContradictionStatus, ContradictionCandidate, ContradictionMiner

**Методы:** _load_candidates, _save_candidates, _get_storage_dir, start, stop, _subscribe_to_events, _on_node_created, _on_graph_updated, _on_system_idle, _schedule_check

---

### contradiction_reputation

**Классы:** SourceReputationSystem

**Методы:** _init_database, _populate_trusted_sources, _load_reputation_data, get_source_reputation, _extract_domain, _is_trusted_domain, _is_untrusted_domain, _get_domain_reputation, _calculate_base_reputation, _determine_source_type

---

### contradiction_resolution

**Классы:** ContradictionResolution

**Методы:** generate_resolution_report, _generate_resolution_recommendations, calculate_contradiction_impact, get_learning_opportunity, update_confidence, get_contradiction_type, get_severity, get_resolution_priority, get_resolution_summary, calculate_resolution_confidence

---

### contradiction_resolver

**Классы:** ContradictionResolver

**Методы:** _emit_metrics, start, stop, detect_contradictions, _analyze_concept_contradictions, resolve_contradiction, _auto_resolve, _weighted_resolve, _manual_resolve, get_active_contradictions

---

### contradiction_responses

**Классы:** ContradictionResponseGenerator

**Методы:** generate_balanced_response, _get_contradiction_type, _is_numeric_conflict, _is_boolean_conflict, _is_response_conflict, _is_exclusivity_conflict, _is_hierarchy_conflict, _get_severity, _format_numeric_conflict_response, _format_boolean_conflict_response

---

### contradiction_strategies

**Классы:** ContradictionResolutionStrategy, ConservativeStrategy, MajorityVoteStrategy, ConfidenceBasedStrategy, LearningBasedStrategy

**Методы:** resolve, get_info, resolve, resolve, resolve, resolve, create_strategy, get_available_strategies

---

### contradiction_types

**Классы:** ContradictionType, ContradictionSeverity, Contradiction, ContradictionReport

**Методы:** to_dict, to_dict

---

### core_detection

**Классы:** Contradiction, CoreDetectionMixin, StorageMixin, ContradictionCore

**Методы:** _safe_nltk_download, _calculate_severity, _analyze_facts, to_dict, from_dict, update_status, add_resolution_history, get_resolution_history, calculate_resolution_confidence, is_resolved

---

### core_resolution

**Классы:** ResolutionMixin

**Методы:** resolve_contradiction, get_active_contradictions, get_detected_contradictions, get_all_contradictions, merge_contradictions, prioritize_contradictions

---

### core_tracking

**Классы:** TrackingMixin

**Методы:** get_contradiction_statistics, get_contradiction_summary, generate_report, get_history, export_contradictions

---

### detect_core

**Классы:** ContradictionDetector

**Методы:** _init_nlp_model, detect_contradictions, _detect_contradictions_for_concept, _find_potential_contradictions, _calculate_divergence, _create_contradiction, detect_contradictions_in_new_fact, _are_facts_equivalent, analyze_fact_consistency, get_detection_statistics

---

### detect_logical

**Классы:** LogicalDetectionMixin

**Методы:** detect_hierarchy_contradictions, _are_concepts_mutually_exclusive, _has_cyclic_dependency, _find_cyclic_dependency, detect_exclusivity_contradictions

---

### detect_semantic

**Классы:** SemanticDetectionMixin

**Методы:** _calculate_text_divergence, _calculate_lexical_divergence

---

### detect_temporal

**Классы:** TemporalDetectionMixin

**Методы:** detect_temporal_contradictions, detect_contextual_contradictions

---

### learn_core

**Классы:** ContradictionLearningOpportunity, ContradictionLearner

**Методы:** to_dict, from_dict, assess_impact, _calculate_concept_importance, generate_learning_opportunity, get_learning_report, create_opportunity, process_opportunity, apply_feedback, get_active_opportunities

---

### learn_feedback

**Классы:** FeedbackProcessingMixin

**Методы:** update_status, add_learning_task, complete_task, get_progress_report, get_learning_recommendations, is_high_priority, get_time_since_creation, requires_immediate_attention

---

### learn_patterns

**Классы:** PatternExtractionMixin

**Методы:** _determine_contradiction_type, _get_contradiction_type, _determine_domain, generate_learning_plan, _generate_numeric_conflict_learning_plan, _generate_boolean_conflict_learning_plan, _generate_exclusivity_conflict_learning_plan, _generate_hierarchy_conflict_learning_plan, _generate_response_conflict_learning_plan, _generate_general_learning_plan

---

## CORE

### api_compat

**Методы:** api_version, decorator, _is_compatible

---

### async_pipeline

**Классы:** TaskPriority, TaskStatus, GenerationTask, AsyncGenerationPipeline

**Методы:** to_dict, get_duration_ms, get_task, get_metrics, _generate_sync, get_pipeline

---

### autopilot_cache

**Классы:** AutopilotCache

**Методы:** put, get, append_event

---

### background_coordinator

**Классы:** Policies, BackgroundCoordinator

**Методы:** _emit_event, start, stop, pause, resume, signal_user_activity, register_detector, register_job_type, _run_loop, _tick

---

### base_job

**Классы:** BaseJob

**Методы:** cancel, run

---

### module_recovery_job

**Классы:** ModuleRecoveryJob

**Методы:** run

---

### training_job

**Классы:** TrainingJob

**Методы:** run

---

### web_index_job

**Классы:** WebIndexJob

**Методы:** run

---

### base_component

**Классы:** ComponentState, BaseComponent

**Методы:** initialize, start, stop, get_state, is_ready, is_running, get_stats, _check_dependencies, _setup_event_subscriptions, _subscribe

---

### batch_wrapper

**Классы:** WrapperMetadata, BatchEnvelope

**Методы:** wrap_for_transfer, unwrap_for_adapter, assert_clean_batch, emit_wrapper_event

---

### brain_components

**Классы:** ComponentMixin

**Методы:** _init_mode_controller, _init_managers, _init_fractal_model, _init_llama_cpp, _init_two_model_pipeline, _init_preprocessing, _init_qwen_config, _init_background, _init_mode_controller, _register_deferred_system_handlers

---

### brain_config

**Классы:** ConfigMixin

**Методы:** load_brain_config, mask_secrets, get_system_info, _load_brain_config, _mask_secrets, _get_system_info

---

### brain_coordination

**Классы:** EventSubscriptionMixin, CommandIssuerMixin, ProcessTrackerMixin

**Методы:** _to_priority, _make_event, _extract_data, _subscribe_to_system_events, _on_pipeline_start, _on_model_a_complete, _on_model_b_complete, _on_pipeline_complete, _on_pipeline_failed, _cmd_retry_pipeline

---

### brain_init

**Методы:** _init_fractal_final, _init_gen_coord, _init_wikipedia, _init_reasoning, _init_performance_monitor, _start_post_init_services, _connect_components, _start_components, _stop_components

---

### brain_memory

**Классы:** MemoryMixin

**Методы:** _check_memory_pressure, _handle_memory_pressure, _handle_cache_eviction, _perform_smart_eviction, _evict_vram_to_ram, _evict_ram_to_ssd, _perform_basic_eviction, _update_cache_metrics

---

### brain_memory_manager

**Классы:** MemoryManagerMixin

**Методы:** _init_memory_manager, record_query_activity, unload_all_models, unload_model_c_only, reload_models, get_memory_usage, schedule_idle_unload, cancel_idle_unload

---

### brain_monitoring

**Классы:** MonitoringMixin

**Методы:** _log_throttled, log_module_activity, get_module_activity, get_system_health, get_metrics, emit_metric, emit_metrics, flush_emitted_metrics, get_status, get_system_metrics

---

### brain_query

**Классы:** QueryMixin

**Методы:** needs_web_search, _get_cached_response, _cache_response, _is_greeting_query, _check_proactive_fallback, _update_fallback_state, process_query, _execute_query_strategy, _handle_gguf_pipeline, _handle_fg_only

---

### brain_state

**Классы:** StateMixin

**Методы:** _update_state, _get_current_state_value

---

### component_managers

**Классы:** SecurityManager, AuthManager, AlertManager, MonitoringManager, HealthChecker

**Методы:** authenticate_request, authorize_request, get_security_events, authenticate, authorize, get_active_alerts, get_system_status, register_component_check, register_check, check_all_components

---

### component_readiness

**Методы:** check_component_readiness, check_brain_readiness, get_readiness_report, wait_for_component_readiness

---

### component_types

**Классы:** ComponentType, ComponentState, ComponentInfo, Dependency

**Методы:** to_dict, to_dict

---

### config_manager

**Классы:** ConfigManager

**Методы:** _load_default_config, _load_config_file, _merge_config, get, set, get_section, save_config, reload_config, validate_config, get_all_config

---

### context_chunking

**Классы:** ContextChunk, ChunkedContextProcessor, StreamingGenerator

**Методы:** to_prompt, process, _semantic_split, _simple_split, _detect_chunk_type, _extract_keywords, _calculate_relevance, format_for_prompt, get_chunk_summary, generate_streaming

---

### context_first_policy

**Классы:** ContextFirstPolicy

**Методы:** _safe_set, _resource_hint, apply

---

### contradiction_resolver

**Классы:** ContradictionResolver

**Методы:** check_response, _detect_contradictions, _add_contradiction, _is_duplicate, _attempt_resolution, _can_resolve, _resolve_contradiction, _get_hot_window_data, _create_resolution_prompt, _record_resolution

---

### coordinator

**Классы:** Coordinator

**Методы:** initialize, process, generate

---

### core_brain

**Классы:** CoreBrain

**Методы:** _get_global_brain, _set_global_brain, event_bus, event_bus, initialize, start, stop, start_background_services, stop_background_services, signal_user_activity

---

### core_brain_types

**Классы:** SystemState, ComponentStatus

**Методы:** is_valid_state, get_all_states, is_operational_state, is_error_state, to_dict

---

### cot_logger

**Классы:** CoTStage, CoTLog, CoTLogger

**Методы:** to_dict, add_stage, finalize, to_json, to_dict, start_stage, add_hypothesis, select_hypothesis, set_metadata, end_stage

---

### deferred_command_system

**Классы:** CommandPriority, CommandStatus, DeferredCommand, DeferredCommandSystem

**Методы:** set_event_bus, get_event_bus, add_command, add_module_recovery_strategy, add_module_health_check, _process_commands, _execute_command, _schedule_retry, _monitor_modules, _recover_module

---

### deferred_commands

**Классы:** DeferredCommandSystem

**Методы:** add_module_health_check, add_module_recovery_strategy, add_command, _worker, start, stop, reduce_background_tasks, clear

---

### device_resolver

**Классы:** DeviceConfig

**Методы:** resolve_device, select_precision, autocast_context, memory_info, should_pin_memory

---

### engine_analysis

**Классы:** ReasoningAnalysisMixin

**Методы:** _ethics_check, _analytics_check, _performance_analysis, _generate_internal_questions, _ask_internal, _calculate_synthesis_confidence, _reflection, _is_gibberish, _calculate_overall_confidence

---

### engine_core

**Классы:** ReasoningEngine

**Методы:** reason, process_query, create_reasoning_engine

---

### engine_steps

**Классы:** ReasoningPhase, _ReasoningPhaseEnum, ReasoningStep, InternalDialogue, ReasoningStepsMixin

**Методы:** _initial_analysis, _classify_query, _estimate_complexity, _extract_entities, _extract_intent, _estimate_urgency, _memory_retrieval, _check_contradictions, _web_search_if_needed, _query_knowledge_graph

---

### engine_synthesis

**Классы:** ReasoningSynthesisMixin

**Методы:** _synthesize_answer, _direct_generate, _finalize_answer, _update_memory_graph, _add_to_history, _get_step_by_phase, get_reasoning_stats

---

### enhanced_self_learning

**Классы:** TrainingStatus, EpochMetrics, TrainingSession, EnhancedSelfLearningSystem

**Методы:** to_dict, to_dict, _initialize_system_integration, set_progress_callback, set_epoch_callback, start, stop, add_training_data, auto_collect_training_data, _extract_entities

---

### event_bus

**Классы:** EventPriority, Event, EventTypes, EventBus

**Методы:** subscribe, unsubscribe, _cleanup_dead_subscribers, publish, publish_sync, _process_event, _worker_loop, start, stop, get_stats

---

### event_bus_bridge

**Классы:** EventBusBridge

**Методы:** _setup_bridges, _bridge_old_to_new, _bridge_new_to_old, _on_new_event, _convert_to_old_format, _register_bridge, link, _forward_old_to_new, _forward_new_to_old_sync, map_event_type

---

### event_management

**Классы:** SimpleEventSystem

**Методы:** on, trigger, off, once

---

### event_system

**Классы:** _WrappedCallback, EventBus, ComponentInitializationManager, EventSystem

**Методы:** subscribe, unsubscribe, trigger, is_triggered, get_event_data, get_event_stats, clear_triggered_events, get_timeline, clear_timeline, enable_timeline

---

### feedback_processor

**Классы:** FeedbackData, FeedbackProcessor

**Методы:** process_feedback, _update_experience_from_feedback, _update_edge_weights, _get_recent_experiences, _save_experience, _validate_feedback, _to_feedback_data, get_feedback_stats

---

### fractal_attention_system

**Классы:** FractalAttentionSystem

**Методы:** _reset_focus, update_focus, _identify_domain, _determine_dialog_type, _contains_contradiction, _is_time_sensitive, _is_refinement, _extract_primary_concepts, _calculate_priority, _calculate_attention_span

---

### fractal_pipeline

**Классы:** PipelineResult, FractalPipeline, FractalPipelineAdapter

**Методы:** process_query, _detect_query_type, get_context_stats, process_query, _fallback_process, get_context_stats, fractal_memory

---

### generation_tracker

**Классы:** GenerationStatus, GenerationTracker

**Методы:** start_generation, update_progress, complete, fail, timeout, get_status, get_all_active, get_all, cleanup_completed, _publish

---

### global_resource_queue

**Классы:** GlobalResourceQueue

**Методы:** acquire_memory, release_memory, acquire_cpu, release_cpu, acquire_io, release_io, get_resource_usage, update_memory_limit, _monitor_resources, stop

---

### graph_ml_core

**Классы:** GraphEmbedding, AmbiguousEntity, ClarificationRequest, GraphPattern, MemoryGraphML

**Методы:** graph, _init_st_model, initialize, _load_graph_structure, _compute_embeddings, _compute_fallback_embedding, get_context_for_query, _extract_entities_from_query, _compute_query_embedding, _cosine_similarity

---

### graph_ml_inference

**Методы:** predict_relation, find_similar_nodes, classify_node, _setup_inference

---

### graph_ml_patterns

**Методы:** detect_clusters, find_frequent_patterns, analyze_graph_structure, get_pattern_insights, _setup_patterns

---

### graph_ml_training

**Методы:** generate_training_sample, _extract_patterns, _setup_training

---

### hardware_optimizations

**Методы:** apply_hardware_optimizations, _apply_cpu_optimizations, _apply_gpu_optimizations, get_runtime_diagnostics, setup_tokenizer_parallelism, optimize_cuda_settings, _apply_cuda_optimizations, optimize_torch_precision

---

### hybrid_pipeline_adapter

**Классы:** HybridPipelineAdapter

**Методы:** load_models, unload_models, _init_pipelines, _init_fractal_pipeline, _init_dual_generator, _init_recursive_pipeline, process_query, _process_fractal, _process_dual, _process_recursive

---

### hybrid_token_cache

**Классы:** HybridTokenCache

**Методы:** get, set, _can_add_to_vram, _can_add_to_ram, _can_move_to_vram, _move_to_vram, _update_access_time, _estimate_token_count, _save_to_disk, _load_from_disk

---

### init_connections

**Методы:** define_dependencies, validate_dependencies, check_dependencies, post_initialize_connections

---

### init_core

**Классы:** ComponentInitializer

**Методы:** _ensure_eva_path, initialize_components, post_initialize_connections, register_component, get_component, get_initialization_status, retry_failed_components, shutdown_components, get_component_health, get_all_component_health

---

### init_factories

**Методы:** _ensure_eva_path, create_event_bus, create_resource_manager, create_config_manager, create_memory_manager, create_hybrid_cache, create_qwen_api_enhancer, create_text_processor, create_ml_unit, create_model_manager

---

### init_validation

**Методы:** get_component_health, get_all_component_health, get_initialization_status, retry_failed_components

---

### integration_adapters

**Методы:** _handle_query_received, _handle_tokenize_request, _handle_tokens_ready, _handle_hot_window_ready, _handle_response_generated, _handle_contradiction_detected, _handle_learning_opportunity, _handle_self_dialog_request, _handle_ethical_check_request, _tokenize_text

---

### integration_core

**Классы:** ЕВАIntegrator

**Методы:** initialize, _initialize_components, _start_background_processes, process_query, _update_request_status, _update_metrics, get_system_health, get_system_stats, start_self_dialog, optimize_system

---

### integration_events

**Методы:** _setup_event_subscriptions, _handle_pipeline_start, _handle_pipeline_model_a_complete, _handle_pipeline_web_search_complete, _handle_pipeline_contradiction_check_complete, _handle_pipeline_ethics_check_complete, _handle_pipeline_model_b_complete, _handle_pipeline_relevance_check_complete, _handle_pipeline_refinement_needed, _handle_pipeline_refinement_attempt

---

### integration_sync

**Методы:** _learning_scheduler_worker, _system_optimizer_worker, _health_monitor_worker, _setup_sync

---

### integration_types

**Классы:** IntegrationType, ConnectionStatus, IntegrationEndpoint, IntegrationConfig

**Методы:** to_dict, to_dict

---

### knowledge_rollback

**Классы:** KnowledgeRollback

**Методы:** set_graph_learning, set_event_bus, rollback_knowledge, _publish_rollback_event, get_rollback_history, check_and_rollback

---

### learning_scheduler

**Классы:** LearningScheduler

**Методы:** schedule_learning, _identify_learning_opportunities, _has_low_confidence, _has_new_connections, _extract_topic, _add_learning_opportunity, _is_duplicate, _check_learning_trigger, _initiate_learning, _create_learning_plan

---

### memory_initializer

**Методы:** initialize_memory_manager

---

### metrics

**Классы:** MetricValue, Histogram, Counter, Gauge, MetricsRegistry

**Методы:** observe, get_stats, _percentile, get_buckets, inc, get, set, inc, dec, get

---

### metrics_collector

**Классы:** MetricsCollector

**Методы:** record_metric, record_error, get_component_stats, get_system_health, get_performance_metrics

---

### base_detector

**Классы:** BaseDetector

**Методы:** probe, _do_probe

---

### learning_detector

**Классы:** LearningOpportunityDetector

**Методы:** _do_probe, _do_probe

---

### recovery_detector

**Классы:** ModuleRecoveryDetector

**Методы:** _do_probe

---

### web_discovery_detector

**Классы:** WebDiscoveryDetector

**Методы:** _do_probe

---

### config

**Классы:** ModelConfig, BackendConfig, TokenizerConfig, QuantizationConfig, KnowledgeConfig

**Методы:** from_dict, from_json, from_yaml, to_dict, to_json, to_yaml, validate

---

### container

**Классы:** EvaStructure, EvaContainer

**Методы:** compute_file_hash, version, model_type, supports_virtual_tokens, mount, unmount, get_model_path, get_graph_path, get_tokenizer_config, get_backend_config

---

### model

**Классы:** ModelInfo, EvaModel

**Методы:** to_dict, from_eva, from_pretrained, _init_embedder, _init_pie_architecture, generate, generate_stream, embed, embed_layers_parallel, _dummy_embed

---

### pie_fallback

**Классы:** FallbackResult, PieFallbackPipeline

**Методы:** _init_pie_components, _get_db_path, generate, _determine_strategy, _execute_fallback_chain, _keyword_fallback, _cache_fallback, _graph_fallback, _minimal_fallback, get_stats

---

### pie_model_paths

**Методы:** get_pie_model_path, get_available_pie_models, verify_pie_models, map_to_legacy_paths

---

### pipeline_adapter

**Классы:** PipelineAdapter

**Методы:** process_query, generate, generate_streaming, generate_with_context, get_stats, unload_models, load_models, is_ready, create_pipeline_adapter

---

### pipeline_adaptive

**Классы:** AdaptiveParameterController

**Методы:** _get_embedder, _compute_embedding, _cosine_similarity, _are_embeddings_stuck, get_params_for_attempt, record_failure, record_success, reset, get_stats, cleanup

---

### pipeline_core

**Классы:** RecursiveModelPipeline

**Методы:** _init_ethics_framework, _check_ethics, _publish_event, _get_adaptive_generation_params, load_models, _is_code_request, _review_with_model_a, process_query, _process_query_impl, _process_query_impl_inner

---

### pipeline_models

**Методы:** _generate_response, generate_with_model_a, _adaptive_fallback_a, _generate_with_chunking_a, generate_with_model_b, _adaptive_fallback_b, _generate_with_chunking_b, _load_model_c, _unload_model_c, generate_with_model_c

---

### pipeline_quality

**Методы:** check_quality, _sanitize_response, save_code_block, fix_mixed, _clean_filler_start, _remove_looping_blocks, check_russian_quality, _fix_russian_punctuation, _generate_with_timeout, _generate

---

### proactive_fallback

**Классы:** GenerationMetrics, ProactiveDegradationMonitor, StatePreservingFallback, FallbackErrorMapper

**Методы:** should_degrade, get_degradation_reason, analyze_response, should_trigger_fallback, set_state, get_state, set_artifact, get_artifact, get_partial_context, clear

---

### processor_core

**Классы:** QueryProcessor

**Методы:** _initialize_model_components, _get_embedding, _set_embedding, process_query, close

---

### query_router

**Классы:** QueryIntent, QueryRouter

**Методы:** classify, _count_matches, _extract_keywords, get_routing_decision, create_query_router

---

### real_self_learning

**Классы:** RealTimeLearningIntegration, ЕВАSelfLearningManager

**Методы:** start, stop, add_training_sample, _assess_sample_quality, _learning_worker, _should_start_training, _perform_training_session, _select_training_samples, _train_on_samples, _notify_model_improvement

---

### reasoning_types

**Классы:** ReasoningType, ReasoningStatus, ReasoningContext, ReasoningChain

**Методы:** to_dict, to_dict

---

### resource_manager

**Классы:** ResourceManager

**Методы:** start_monitoring, stop_monitoring, _monitoring_loop, _update_metrics, _update_gpu_metrics, _add_to_history, _check_thresholds, get_current_metrics, get_cpu_usage, get_memory_usage

---

### response_generator

**Классы:** ResponseGenerator

**Методы:** _deferred_init_components, _detect_ambiguity_before_response, _generate_clarification_response, _init_components, _init_tokenizer, _init_tokenizer_internal, _init_hybrid_cache, _init_unified_bridge, _create_fallback_tokenizer, _validate_tokenizer

---

### response_types

**Классы:** ResponseStyle, ResponseType, ResponseContext, GeneratedResponse

**Методы:** to_dict, to_dict

---

### self_dialog_manager

**Классы:** SelfDialogManager

**Методы:** start_dialog, stop, _determine_dialog_type, _generate_dialog_questions, _generate_next_question, _get_system_response, _get_hot_window_data, _process_response, _update_hot_window, _extract_concepts

---

### self_evaluation

**Классы:** EvaluationResult, SelfEvaluationLoop

**Методы:** evaluate, _evaluate_completeness, _evaluate_accuracy, _evaluate_ethics, should_regenerate, create_self_evaluation

---

### self_learning_system

**Классы:** SelfLearningSystem, AutoLearningIntegration

**Методы:** start, stop, add_training_data, _learning_loop, _should_train, _perform_training_session, _prepare_training_data, _train_model_on_data, _expand_training_texts, _save_updated_model

---

### system_metrics

**Классы:** SystemMetricsManager

**Методы:** start_tracking, update_request_metrics, get_metrics, emit, emit_many, flush, get_quarantine, _normalize_metric, validate_metric_schema, validate_many

---

### system_optimizer

**Классы:** SystemOptimizer

**Методы:** start_optimization_monitor, stop_optimization_monitor, enter_power_saving_mode, exit_power_saving_mode, _check_optimization_needed, _get_system_health, _is_user_inactive, _optimize_for_memory, _optimize_for_cpu, _apply_power_saving_optimizations

---

### system_state

**Классы:** SystemState, ComponentStateInfo, SystemStateManager

**Методы:** to_dict, update_component_state, get_component_state, get_all_component_states, get_system_summary, get_state, set_state, _update_system_state, _setup_event_subscriptions, _subscribe

---

### task_types

**Классы:** TaskPriority, TaskStatus, BackgroundTask, TaskSchedule

**Методы:** to_dict, to_dict

---

### text_chunker

**Классы:** TextChunker, ChunkedGenerator

**Методы:** estimate_tokens, chunk_text_by_tokens, chunk_by_sentences, generate_with_chunking, merge_chunk_results, create_text_chunker, chunk_query_if_needed

---

### token_processor

**Классы:** TokenProcessor

**Методы:** tokenize_query, _generate_token_id, _calculate_token_priority, get_token_statistics, prewarm_tokens_async, _prewarm_tokens, health_check, recover

---

### unified_cache_bridge

**Классы:** UnifiedCacheBridge

**Методы:** set_token_cache, set_knowledge_graph, find_relevant_graph_nodes, _search_graph, _get_nodes_by_ids, preload_graph_context, _node_to_text, build_enriched_prompt, cache_generation_result, get_cached_generation

---

### unified_generator

**Классы:** ModelType, GenerationResult, SimpleRouter, UnifiedGenerator

**Методы:** route, _load_model_paths, _detect_optimal_threads, _load_model, generate, generate_dual, generate_iterative, _get_concepts_context, _get_contradictions_context, _get_web_search_context

---

### utils

**Классы:** EthicalDecision

**Методы:** setup_logging, handle_exception

---

## DISTRIBUTED

### cluster_manager

**Классы:** ClusterNode, ClusterManager

**Методы:** to_dict, from_dict, _get_connection, _init_database, _load_nodes, _save_nodes, start, _cluster_loop, _check_node_health, _check_coordinator

---

### database_utils

**Методы:** get_connection, execute_query

---

### distributed_recovery_manager

**Классы:** RecoveryManager

**Методы:** start, stop, _load_checkpoints, create_checkpoint, _get_working_memory_state, _get_semantic_memory_state, _get_episodic_memory_state, _get_user_profiles_state, _get_cluster_state, _get_system_status

---

### distributed_system

**Классы:** DistributedSystem

**Методы:** _get_connection, _init_database, _load_config, _init_components, start, _start_background_processes, _monitor_system_status, _update_system_stats, stop, _stop_background_processes

---

### distributed_task_scheduler

**Классы:** Task, TaskScheduler, SimpleTaskScheduler

**Методы:** schedule_task, start, stop, _worker_loop, get_task_status, get_task_result, get_scheduler_statistics, get_scheduler_health_report

---

### distributed_tasks

**Классы:** DistributedTask, TaskScheduler

**Методы:** to_dict, from_dict, start, stop, _scheduler_loop, _process_next_task, _assign_task_to_node, _get_suitable_nodes, _get_required_capabilities, _send_task_to_node

---

### distributed_types

**Классы:** NodeRole, TaskState, NodeInfo, DistributedTask

**Методы:** to_dict, to_dict

---

### knowledge_sync

**Классы:** KnowledgeSync

**Методы:** start, stop, _sync_loop, sync_knowledge, _pull_sync, _push_sync, _hybrid_sync, _get_sync_data, _apply_updates, _check_system_health

---

## ETHICS

### ethical_situations

**Классы:** EthicalSituationHandler

---

### ethics_core

**Классы:** EthicsFramework

**Методы:** start, stop, assess_ethics, _get_default_decision, needs_ethical_review, get_ethical_issues, add_ethical_issue, resolve_ethical_issue, get_system_health, get_dashboard_data

---

### ethics_integrated

**Классы:** IntegratedEthicsFramework

**Методы:** _do_initialize, _do_start, _do_stop, evaluate_action, _basic_ethical_evaluation, _check_principles_violations, get_ethical_guidance, get_ethical_statistics, _get_most_common_violations, _load_ethical_history

---

### framework_checks

**Классы:** EthicalAssessment, EthicsAnalysisResult, EthicsChecksMixin

**Методы:** analyze_content, analyze_response, analyze_request, _assess_request, _evaluate_principle, _evaluate_privacy, _evaluate_safety, _evaluate_fairness, _evaluate_transparency, _evaluate_autonomy

---

### framework_core

**Классы:** EthicsFramework

**Методы:** is_ready, _init_background_services, _monitor_violations, _check_resolved_violations, _periodic_principle_check, _update_principle_stats, start, stop, get_system_health, get_system_status

---

### framework_principles

**Классы:** EthicalPrinciple, EthicsPrinciplesMixin

**Методы:** _load_configuration, _load_configuration_v1, _load_configuration_v2, _init_default_configuration, _init_default_principles, add_ethical_principle, update_ethical_principle, get_principle, get_all_principles

---

### framework_violations

**Классы:** EthicalDecision, EthicalReview, EthicsViolationsMixin

**Методы:** _load_violations_and_stats, _save_violations, _save_principles, _save_stats, get_violation_history, resolve_violation, get_active_violations, get_ethics_statistics, export_ethics_data, import_ethics_data

---

### principles_manager

**Классы:** PrinciplesManager

**Методы:** _init_db, load_principles, _load_default_principles, add_principle, update_principle, get_principle, get_principle_by_name, get_principles_by_category, get_all_principles, record_assessment

---

### risk_assessment

**Классы:** RiskAssessor

**Методы:** load_reference_scenarios, assess_risk, _extract_text_from_context, _identify_scenario, _assess_principle_risk, _get_principle_keywords, analyze_ethical_gaps, get_risk_dashboard_data, generate_risk_visualization

---

### situations_db

**Классы:** EthicalIssue, SituationsDBMixin

**Методы:** _load_cache, _save_cache, _load_ethical_issues, _save_ethical_issues, get_ethical_issues, add_ethical_issue, resolve_ethical_issue

---

### situations_evaluation

**Классы:** SituationsEvaluationMixin

**Методы:** _calculate_confidence, get_situation_dashboard_data, generate_situation_visualization, export_ethics_data, import_ethics_data, get_system_health, close

---

### situations_scenarios

**Классы:** EthicalAssessment, EthicalPrinciple, EthicalDecision, SituationsScenariosMixin

**Методы:** handle_situation, _requires_human_review, _generate_decision, _get_principle_weight, _generate_alternatives, _cache_solution, _generate_cache_key, _summarize_context, _update_statistics, _get_default_decision

---

### violation_id_manager

**Классы:** ViolationIDComponents, ViolationIDManager

**Методы:** generate_id, _get_principle_shortcode, parse_id, is_valid_id, convert_legacy_id, get_timestamp, get_principle, get_hash, get_version, is_new_format

---

## FRACTAL

### entity_fractal_store

**Классы:** EntityLevelData, EntityFractalStore

**Методы:** store_entity, _store_at_level, _extract_tokens, _compute_level_embedding, _update_level_3, _generate_definition, _find_related_concepts, _update_level_4, _compute_understanding_level, _compute_confidence

---

### fractal_store

**Классы:** FractalContainer, FractalStore

**Методы:** update_priority, get_memory_size, pack_model_weights, pack_state_dict, get_container, get_container_data, get_statistics, clear, save_to_disk, load_from_disk

---

## GENERATION

### generation_coordinator

**Классы:** GenerationRequest, GenerationResponse, GenerationProvider, HybridModelProvider, FractalModelProvider

**Методы:** to_dict, is_available, generate, get_priority, is_available, generate, get_priority, is_available, generate, get_priority

---

## GUI

### __init__

**Методы:** create_gui

---

### analytics_module

**Классы:** AnalyticsModule

**Методы:** _log_brain_access, _safe_brain_call, activate, deactivate, _start_data_collection, _stop_data_collection, _data_collection_loop, _collect_system_data, _add_data_point, _create_analytics_interface

---

### analytics_types

**Классы:** MetricType, TimeRange, AnalyticsMetric, AnalyticsReport

**Методы:** to_dict, to_dict

---

### base_gui

**Классы:** BaseGUI

**Методы:** initialize, _setup_ui, _bind_events, show, hide, destroy, register_callback, trigger_callback, get_state, set_enabled

---

### chat_actions

**Классы:** ChatActionsMixin

**Методы:** _create_context_menu, _show_context_menu, _get_selected_chat_text, _quote_selection_to_input, _run_command_on_selection, _copy_selected, _copy_all, _clear_chat, _cut_text, _copy_text

---

### chat_history

**Классы:** ChatHistoryMixin

**Методы:** _load_chat_history, _save_history_incremental, _save_chat_history

---

### chat_input

**Классы:** ChatInputMixin

**Методы:** _create_input_area, _create_input_context_menu, _create_input_buttons, _bind_input_events, _create_status_bar, _create_typing_indicator, _show_typing, _hide_typing, _send_message, _on_enter_pressed

---

### chat_messages

**Классы:** ChatMessagesMixin

**Методы:** _add_message, _process_and_insert_formatted_message, _process_urls, _process_markdown_links, _process_emojis, _process_images, _configure_chat_tags, _remove_last_message, _show_welcome_message, _redraw_chat

---

### chat_module

**Классы:** ChatModule

**Методы:** activate, deactivate, _setup_event_subscriptions, _start_processing_thread, _stop_processing_thread, _cancel_status_update, _save_draft_text, _processing_loop, _process_request, _build_analytics_lines

---

### chat_reasoning

**Классы:** ChatReasoningMixin

**Методы:** _init_reasoning_panel, _toggle_reasoning_panel, _set_reasoning_content

---

### chat_text_utils

**Методы:** _to_display_str, _looks_mojibake, _fix_mojibake

---

### contradiction_module

**Классы:** ContradictionModule

**Методы:** activate, deactivate, _is_active, _get_contradictions_data, _convert_statistics_to_list, _normalize_contradictions_data, _update_statistics, refresh_contradictions, _schedule_refresh, _safe_refresh

---

### core_gui

**Классы:** ЕВАGUI

**Методы:** create_gui

---

### gui_events

**Классы:** EventHandlerMixin

**Методы:** process_query_via_integrator, _wait_for_response, _fallback_query_processing, get_system_status_via_integrator, _get_system_status_fallback, start_self_dialog_via_integrator, optimize_system_via_integrator, _schedule_update, _start_background_services, _process_gui_queue

---

### gui_graph_types

**Классы:** VisualizationType, NodeDisplayMode, GraphNode, GraphEdge, GraphLayout

**Методы:** to_dict, to_dict, to_dict

---

### gui_main

**Классы:** MainWindowMixin

**Методы:** _validate_module_path, create_main_window, _create_styles, _create_interface, _create_navbar, _create_notebook, start_gui, start, stop, _load_state

---

### gui_modules

**Методы:** init_modules, switch_view, update_nav_button_selection

---

### gui_status

**Классы:** StatusBarMixin

**Методы:** _create_status_bar, _update_interface, _update_system_metrics, _update_status_indicator, _update_model_loading_indicator, _handle_model_load_event, _handle_models_ready_event, _handle_notifications, update_status, show_error

---

### gui_tabs

**Классы:** TabManagerMixin, MemoryTab, SystemTab

**Методы:** _switch_view, _on_tab_changed, _update_nav_visual_state, _init_modules, _cleanup_modules, _validate_module_path, activate, deactivate, update, activate

---

### gui_themes

**Методы:** create_styles

---

### gui_types

**Классы:** MessageType, MessageRole, ChatMessage, ChatSession, ChatContext

**Методы:** to_dict, to_dict, to_dict

---

### gui_util

**Методы:** create_rounded_button, on_click

---

### gui_utils

**Методы:** load_settings, save_settings, process_gui_queue, _process, show_notification, hide_notification, show_learning_opportunities

---

### gui_widgets

**Методы:** create_main_interface, create_navbar, create_content_area, create_status_bar, create_notification_area

---

### kg_actions

**Методы:** refresh_graph, search_by_domain, export_node, export_graph, show_settings_dialog, apply_settings, on_tab_changed, on_domain_double_click, show_domain_on_graph, show_domain_details

---

### kg_nodes

**Методы:** show_node_info, show_edge_info, clear_info_panel, show_error_in_info_panel, show_node_details, show_related_nodes, resolve_contradiction

---

### kg_search

**Методы:** create_search_tab, _create_search_context_menu, _show_search_context_menu, show_search_dialog, perform_search, perform_search, update_search_results, on_search_result_double_click, highlight_search_result, show_search_result_details

---

### kg_stats

**Методы:** create_domains_tab, _create_domains_context_menu, _show_domains_context_menu, load_domains_data, create_stats_tab, load_statistics, update_statistics_charts

---

### kg_visualization

**Методы:** create_graph_tab, _create_graph_context_menu, _show_graph_context_menu, _create_info_panel, _create_node_types_chart, _create_domains_chart, initialize_graph, update_graph_visualization, on_graph_click, on_graph_hover

---

### knowledge_graph_module

**Классы:** KnowledgeGraphModule

**Методы:** _safe_brain_call, activate, deactivate, _log_brain_access, _start_update_thread, _stop_update_thread, _cancel_pending_after_events, _safe_after, _update_data_loop, _update_graph_data

---

### learning_module

**Классы:** LearningModule

**Методы:** safe_get, activate, deactivate, _safe_after, cleanup, _widget_exists, _start_auto_update, refresh_learning_data, _determine_learning_trend, _create_learning_interface

---

### memory_module

**Классы:** MemoryModule

**Методы:** _safe_brain_call, activate, deactivate, _start_auto_update, cleanup, _initialize_memory_data, refresh_memory_data, _get_fallback_memory_stats, _get_fallback_memory_analysis, _load_usage_history

---

### neuromorphic_module

**Классы:** NeuromorphicModule

**Методы:** _safe_brain_call, activate, deactivate, update, _build_ui, _on_start, _on_stop, _schedule_update, cleanup, _refresh_metrics

---

### settings

**Методы:** get_default_settings, load_settings, save_settings

---

### settings_module

**Классы:** SettingsModule

**Методы:** _safe_brain_call, activate, deactivate, _create_settings_interface, _create_general_settings_tab, _create_system_settings_tab, _browse_cache_dir, _create_adaptation_settings_tab, _create_backup_settings_tab, _browse_backup_dir

---

### bridge

**Классы:** GUIBridge, NetworkBridge

**Методы:** set_web_gui, _setup_core_subscriptions, _subscribe_to_new_event_bus, _setup_deferred_system_access, _on_new_system_error, _on_new_component_initialized, _on_new_learning_progress, _on_new_pipeline_complete, _on_new_contradiction_detected, _on_query_received

---

### server_api_export

**Методы:** register_routes, api_export, api_import

---

### server_api_knowledge

**Методы:** register_routes, api_knowledge

---

### server_api_wikipedia

**Методы:** load_brain_config, load_quota, save_quota, check_quota, increment_quota, tavily_search, register_routes, api_wikipedia

---

### server_auth

**Классы:** SessionManager, AuthManager, EntityExtractor, EthicsChecker

**Методы:** _ensure_storage_dir, _load_sessions, _save_sessions, create_session, get_session, get_user_sessions, update_session, delete_session, add_context_node, add_chat_message

---

### server_main

**Классы:** WebGUI

**Методы:** _get_secret_key, _get_document_manager, _ingest_document_to_memory, _estimate_tokens, process_message, get_session_documents, get_document_stats, clear_session_documents, start, stop

---

### server_models

**Методы:** register_routes, api_model_status, api_fractal_graph

---

### server_routes

**Методы:** register_routes, favicon, api_system, check_request_timeout, index, api_debug_test, api_status, api_debug_deferred, api_debug_events, api_debug_login

---

### server_routes_analytics

**Методы:** register_analytics_routes, api_memory_graph, api_analytics, api_learning, api_self_dialog, api_events_stream, api_generation_status, api_dashboard

---

### server_routes_auth

**Методы:** register_auth_routes, api_login, api_debug_auth

---

### server_routes_backup

**Методы:** register_routes, favicon, api_system, check_request_timeout, index, api_debug_test, api_status, api_debug_deferred, api_debug_events, api_debug_login

---

### server_routes_chat

**Методы:** register_chat_routes, api_chat, api_chat_v1, api_chat_stream, api_sessions, api_session

---

### server_routes_core

**Методы:** register_core_routes, favicon, api_system, index, api_debug_test, api_status, api_shutdown, api_debug_deferred, api_debug_events, api_health

---

### server_routes_graph

**Методы:** register_graph_routes, api_contradictions, api_concepts, api_graph_stats, api_nodes, api_edges

---

### server_routes_knowledge

**Методы:** register_knowledge_routes, api_documents, api_delete_document, api_documents_memory, api_document_memory_detail, api_knowledge_graph, api_cache_stats, api_settings, api_snapshots, api_stats

---

### server_routes_upload

**Методы:** extract_text_from_file, register_upload_routes, api_upload, api_entities, api_feedback

---

### server_routes_utils

**Методы:** api_version, extract_text_from_file, validate_json_request, check_brain_initialized, get_brain_components

---

### server_types

**Классы:** SessionStatus, WebSession, UserCredentials

**Методы:** to_dict, to_dict

---

### widgets

**Методы:** create_rounded_button, on_click, create_gradient_canvas, create_card_frame, create_header_label, create_secondary_label, _get_toast_root, _show_toast_window, show_notification, show_toast

---

## KNOWLEDGE

### ambiguity_resolver

**Классы:** AmbiguityResolver

**Методы:** resolve, find_disambiguations

---

### concept_extractor

**Классы:** Concept, ConceptExtractor

**Методы:** _load_stop_words, extract_concepts, _extract_terms, _create_concept, _detect_domain, _generate_facts, _find_related_terms, save_concept_to_graph, _save_concept_facts, get_concept_facts

---

### concept_miner

**Классы:** ConceptStatus, PhantomCandidate, ConceptMiner

**Методы:** _load_audit_log, _save_audit_log, _log_rejection, _load_candidates, _save_candidates, _get_storage_dir, start, stop, _subscribe_to_events, _on_memory_graph_updated

---

### context_entity

**Классы:** AmbiguityType, AmbiguousEntity, EntityExtractor

**Методы:** extract_entities, resolve_ambiguity, find_related_entities

---

### graph_curator

**Классы:** CuratorState, GraphCurator

**Методы:** _get_fractal_graph, _is_protected_node, start, stop, _run_loop, _do_curation, _cleanup_garbage, _process_level_promotions, _consolidate_nodes, _update_group

---

### kg_adapter

**Классы:** KnowledgeGraphAdapter

**Методы:** nodes, edges, get_recent_entities, add_entity, add_concept, add_relation, find_related, get_related_concepts, find_path_between_concepts, get_entity_facts

---

### knowledge_analytics

**Классы:** KnowledgeAnalytics

**Методы:** get_concept_stats, get_topic_distribution, get_learning_opportunities, analyze_knowledge_gaps, get_interaction_stats

---

### knowledge_graph

**Классы:** KnowledgeGraph

**Методы:** get_all, get_nodes, get_edges, add_node, add_edge, search, get_stats

---

### qwen_api_enhancer

**Классы:** QwenAPIEnhancer

**Методы:** is_available, generate, enhance_response

---

### wikipedia_kb

**Классы:** WikipediaKnowledgeBase, WikipediaLoader

**Методы:** _init_db, _get_embedder, _compute_embedding, add_article, _chunk_text, search, get_article, get_stats, clear, add_to_fractal_graph

---

## LEARNING

### analyzer_core

**Классы:** LearningOpportunity, AnalyzerCore

**Методы:** to_dict, from_dict, clear_learning_opportunities, _init_db, _load_data, save_config, get_learning_opportunities, add_learning_opportunity, start_background_analysis, stop_background_analysis

---

### analyzer_types

**Классы:** AnalysisType, OpportunityPriority, AnalysisResult, LearningOpportunity

**Методы:** to_dict, to_dict

---

### concept_dialog_integration

**Классы:** ConceptDialogIntegrator

**Методы:** start, stop, _subscribe_to_concept_events, _subscribe_to_sdl_events, _on_concept_candidate_generated, _verify_concept_via_dialog, _on_concept_validation_complete, _trigger_convergence_dialog, _on_learning_opportunity, _trigger_concept_mining_from_gap

---

### curiosity_engine

**Классы:** CuriosityType, CuriosityTrigger, CuriosityEngine

**Методы:** detect_curiosity_triggers, _extract_entities, _create_entity_trigger, _is_entity_known, _extract_topic_from_pattern, _generate_questions, assess_knowledge_gap, trigger_self_learning, get_curiosity_report

---

### data_processor

**Классы:** DataProcessor

**Методы:** initialize, start_collection, stop_collection, _collection_loop, collect_all_data, get_recent_data, get_data_statistics, _cleanup_old_data, _collect_adaptation_analytics, _collect_knowledge_analytics

---

### dialog_concepts

**Классы:** DialogConceptsMixin

**Методы:** queue_concept_for_dialog, queue_contradiction_for_resolution, _get_next_dialog_topic, _get_unified_generator, _run_concept_dialog, _run_contradiction_dialog, _get_concept_info, _get_contradiction_info, _generate_concept_intro, _generate_concept_criticism

---

### dialog_core

**Классы:** SelfDialogLearning

**Методы:** _setup_curator_events, _on_curator_knowledge_extracted, _on_curator_graph_optimized, _on_curator_cleanup_done, start, stop, _cleanup_low_priority_opportunities, _worker_loop, _process_pending_context_compactions, _generate_dialog_from_conversations

---

### dialog_generation

**Классы:** DialogGenerationMixin

**Методы:** _run_dialog, _generate_assistant_prompt, _get_conversation_context, _simulate_assistant_response, _simulate_critic_response, _simulate_learner_response, _simulate_teacher_response, _simulate_observer_response, _identify_knowledge_gaps, _assess_turn_quality

---

### dialog_learning

**Классы:** DialogLearningMixin

**Методы:** _get_fractal_graph, _add_to_fractal_graph, _check_contradiction_fractal, _check_and_execute_learning_opportunities, _get_learning_opportunities, _execute_learning_opportunity, _perform_learning, _learn_expansion, _learn_refinement, _learn_updating

---

### dialog_topics

**Классы:** DialogTopicsMixin

**Методы:** _generate_dialog_from_conversations, extract_key_concepts, analyze_unknown_concepts, search_and_learn_concepts

---

### dialog_types

**Классы:** DialogRole, LearningType, DialogTurn, SelfDialog

---

### fractal_store

**Классы:** FractalContainer, FractalWeightStore

**Методы:** get_memory_size, store, get, _cleanup, get_memory_usage

---

### integrated_learning_manager

**Классы:** LearningConfig, IntegratedLearningManager

**Методы:** initialize, add_learning_task, start_learning, stop_learning, _learning_worker, _get_next_task, _process_learning_task, _train_from_document, _train_knowledge_graph, _fine_tune_model

---

### integration_manager

**Классы:** IntegrationStrategy, IntegrationResult, LearningIntegrationManager

**Методы:** complete, fail, rollback, to_dict, initialize, start_integration_processing, stop_integration_processing, _integration_loop, queue_integration, process_next_integration

---

### interest_scorer

**Классы:** InterestScore, InterestScorer

**Методы:** score, _calculate_complexity, _calculate_novelty, _calculate_learning_potential, is_interesting

---

### knowledge_awareness

**Классы:** KnowledgeSource, KnowledgeEntry, KnowledgeAwareness

**Методы:** mark_verified, mark_generated, get_status, get_source_type, get_verified_sources, get_generated_confidence, get_knowledge_report, get_recent_knowledge, merge_knowledge, clear_generated

---

### learning_integrated

**Классы:** IntegratedLearningManager

**Методы:** _do_initialize, _do_start, _do_stop, start_learning_session, _basic_learning_session, train_model, _basic_model_training, adapt_behavior, _basic_behavior_adaptation, get_learning_statistics

---

### learning_manager

**Классы:** LearningManager

**Методы:** initialize, train_model, _get_training_orchestrator, _train_from_segments, _train_from_text, _create_error_result, evaluate_model, get_model_status, stop_training

---

### learning_opportunity

**Классы:** LearningOpportunity

---

### learning_opportunity_manager

**Классы:** LearningOpportunityManager

**Методы:** get_learning_opportunities, execute_learning_opportunity, _handle_expansion, _handle_refinement, _handle_updating, _handle_integration, get_fixes, _update_model, _optimize_response_time, _reduce_error_rate

---

### learning_processor

**Классы:** ProcessorStatus, LearningProcessor

**Методы:** initialize, _initialize_components, _start_processing_threads, _main_processing_loop, _execute_learning_cycle, _process_tasks, _simulate_learning_process, _integrate_results, _select_integration_strategy, _health_check_loop

---

### learning_scheduler

**Классы:** LearningScheduler

**Методы:** create_scheduler, create_learning_task

---

### learning_types

**Классы:** LearningTaskType, LearningTaskStatus, LearningTask, LearningPlan, ResourceAllocation

**Методы:** to_dict, to_dict, to_dict

---

### performance_analyzer

**Классы:** PerformanceAnalyzer

**Методы:** analyze_performance, _get_components, _analyze_component_performance, _analyze_ml_component, _analyze_knowledge_graph, _analyze_memory_manager, _add_learning_opportunities, analyze_user_feedback

---

### scheduler_core

**Классы:** ResourceAllocation, LearningTask, LearningSchedulerCore

**Методы:** acquire_slot, release_slot, get_slot_usage, get_active_tasks, to_dict, from_dict, get_duration, is_overdue, _load_tasks, _save_tasks

---

### scheduler_monitor

**Классы:** MonitorMixin

**Методы:** get_scheduler_statistics, _calculate_tasks_per_hour, _calculate_average_completion_time, _calculate_failure_rate, get_scheduler_health_report, get_scheduler_diagnostics, export_scheduler_diagnostics, get_system_summary

---

### scheduler_tasks

**Классы:** TaskManagerMixin

**Методы:** add_task, get_task, _update_task_status_internal, update_task_status, _process_dependencies_internal, _process_dependencies, start_task, complete_task, fail_task, clear_schedule

---

### scheduler_triggers

**Классы:** TriggerMixin

**Методы:** create_learning_plan, _assess_knowledge_state, _determine_learning_type, _create_learning_tasks, generate_learning_plan_report, get_concept_domain, create_adaptive_learning_plan, _get_user_profile, _determine_target_concepts, _adapt_learning_plan

---

### self_analyzer

**Классы:** SelfAnalyzer

**Методы:** _check_ready, start, stop, analyze_system, get_learning_opportunities, clear_learning_opportunities, add_learning_opportunity, execute_learning_opportunity, get_fixes, analyze_user_feedback

---

### self_dialog_new_methods

**Методы:** extract_key_concepts, analyze_unknown_concepts, search_and_learn_concepts

---

### task_generator

**Классы:** TaskType, TaskPriority, LearningTask, LearningTaskGenerator

**Методы:** to_dict, initialize, start_generation, stop_generation, _generation_loop, generate_tasks_from_data, _queue_tasks, get_next_task, complete_task, get_task_statistics

---

## MEMORY

### cache_core

**Классы:** HybridTokenCache

**Методы:** get_shared_cache, get_token, add_token, put, get, set, clear, _load_token_from_disk, _save_token_to_disk, _move_token_to_memory

---

### cache_disk

**Классы:** TokenDiskCache

**Методы:** _load_index, _save_index, _get_file_path, get, put, _evict_lru, _remove_file, remove, clear, get_stats

---

### cache_eviction

**Методы:** _get_token_impl, _add_token_impl, _move_token_to_memory_impl, _evict_one_lru_impl, _start_memory_pressure_worker_impl, worker, _offload_under_pressure_impl, _add_context_impl, _get_context_impl, _get_recent_contexts_impl

---

### cache_index

**Классы:** CacheIndex

**Методы:** _exec, _create_schema, upsert_batch, upsert_segment, add_token_nodes, link_nodes_to_kg, set_weight, rank_segments, address_of, get_segment_path

---

### cache_ram

**Классы:** LRUCache

**Методы:** get, put, remove, clear, keys

---

### cache_router

**Классы:** CacheRouter

**Методы:** make_segment_id, make_node_id, register_batch, register_segment, register_token_nodes, link_nodes_to_kg, set_weight, rank_segments, get_segment_bytes, address_of

---

### cache_types

**Классы:** CacheLevel, CacheStrategy, CacheEntry, CacheStats, CacheConfig

**Методы:** to_dict, to_dict, to_dict

---

### context_book

**Классы:** BookIndex, BookPage, ContextBook, ContextBookBuilder, ContextBookMixin

**Методы:** add_entry, search, get_toc, to_text, get_page, find_pages, get_table_of_contents, get_context_for_generation, build_from_graph, _collect_nodes

---

### disk_cache

**Классы:** DiskCache

**Методы:** _refill_tokens, _dynamic_backoff_factor, _throttle_write, _throttle_read, _execute, _create_tables, _calculate_current_size, _evict_old_entries, put, get

---

### document_manager

**Классы:** DocumentPage, DocumentMetadata, LazyLoadingCache, DocumentChunker, DocumentVirtualMemory

**Методы:** to_node_dict, get, put, _evict_oldest, get_stats, _calculate_hit_rate, split_document, _extract_structure, _generate_global_context, ingest_document

---

### embedding_cache

**Классы:** EmbeddingCache

**Методы:** _init_db, _hash_text, get, put, get_or_compute, _evict_if_needed, get_stats, _compute_hit_rate, clear, batch_get

---

### fg_generation_integration

**Классы:** FGGenerationResult, FractalGraphGenerator

**Методы:** generate, _save_interaction, _estimate_confidence, fine_tune, get_stats, create_fg_generator

---

### fg_gguf_architecture_mapper

**Классы:** LayerMapping, GGUFToFGArchitectureMapper

**Методы:** load_model_architecture, _create_model_config_node, _create_vocab_metadata_node, _create_layer_structure, create_aci_concept_from_context, _create_new_concept, _create_aci_concept, get_hot_window_stats, create_architecture_mapper

---

### fg_gguf_quality_extraction

**Классы:** ExtractionQuality, KnowledgeQualityFilter, GGUFToFGIntegrator

**Методы:** validate, extract_phrases, _load_model_tokenizer, generate_with_extraction, _build_prompt, load_model_metadata_to_fg, create_gguf_fg_integrator

---

### cache_manager

**Классы:** FractalCache

**Методы:** store, search, get_stats, _generate_key, clear

---

### eviction_policy

**Классы:** EvictionPolicy

**Методы:** size, should_evict, get_eviction_candidate, record_access, remove, clear, get_stats

---

### response_store

**Классы:** ResponseStore

**Методы:** _load_index, _save_index, save, load, delete, get_recent, clear_all, size

---

### semantic_embedder

**Классы:** SemanticEmbedder

**Методы:** encode, _tokenize, _hash_to_index, batch_encode

---

### similarity_engine

**Классы:** SimilarityEngine

**Методы:** compute, _cosine_similarity, _euclidean_similarity, _jaccard_similarity, _combined_similarity, find_most_similar

---

### __init__

**Классы:** LRUCacheWithTTL, FractalMemoryGraph

**Методы:** timed, decorator, get, put, clear, stats, add_node, add_nodes_batch, add_knowledge, add_edge

---

### dual_generator

**Классы:** GeneratorStats, CondensedGenerator, ExtendedGenerator, DualGenerator

**Методы:** generate, _clean_response, generate, generate_chunked, _deduplicate_chunk, _create_context_summary, _remove_repetitions, _get_context, _clean_response, generate_condensed

---

### dual_generator_pie

**Классы:** GenerationResult, DualGeneratorPie, PieEnabledDualGenerator

**Методы:** generate, _generate_with_dual, _generate_pie, _generate_fallback, _record_to_l1, feedback, get_stats, get_routing_params

---

### embeddings

**Классы:** EmbeddingsManager

**Методы:** _load_model, encode, encode_single, _normalize, _random_embeddings, compute_similarity, compute_similarities_batch, find_similar, clear_cache, get_cache_size

---

### eva_container

**Классы:** ContainerHeader, EVAContainer

**Методы:** to_bytes, from_bytes, create, _load_model_info, save, load, _serialize_graph, _deserialize_graph, _serialize_metadata, _deserialize_metadata

---

### eva_generator

**Классы:** GenerationRequest, GenerationResult, EVAGenerator

**Методы:** generate, _determine_query_type, _get_generation_params, _get_routing_for_entities, _merge_params, _build_prompt, _build_simple_prompt, _get_graph_context, _get_entity_context, _format_semantic_results

---

### gguf_extractor

**Классы:** KnowledgeEntry, GGUFKnowledgeExtractor

**Методы:** load_model, extract_vocab_ru, generate_knowledge, extract_spo, _infer_predicate, extract_knowledge_from_prompts, build_concept_hierarchy, save_to_graph_hierarchy, create_russian_corpus_training, create_extractor

---

### gguf_parser

**Классы:** GGUFModelInfo, GGUFModelParser

**Методы:** parse, _load_llama_model, _count_layers, extract_knowledge_for_graph, parse_gguf_model, extract_to_graph

---

### gguf_shadow

**Классы:** GGUFShadowProfiler

**Методы:** _load_model_metadata, register_model_root, _create_model_shadow_group, _get_node_id, _get_root_id, create_domain_profile, create_activation_fingerprint, _find_domain_profile, bind_routing_rule, _serialize_value

---

### hybrid_tokenizer

**Классы:** Token, AhoCorasick, HybridTokenizer

**Методы:** add, build, search, _build_entity_index, _get_virtual_id, encode, _encode_bpe, decode, get_virtual_token_info, extract_entities

---

### optimizations

**Классы:** HNSWIndex, NLIContradictionDetector, IncrementalClustering, PathCache, FractalGraphOptimized

**Методы:** _init_index, add_items, search, _load_model, check_contradiction, should_recluster, update_cluster, get, put, _evict_lru

---

### semantic_context_cache

**Классы:** ContextEntry, SemanticContextCache

**Методы:** _init_faiss, _get_embedding, _compute_embedding, add, search, _numpy_search, get_hot_window, get_session_contexts, clear_session, _rebuild_index

---

### snapshot_manager

**Классы:** MemorySnapshot, SnapshotManager

**Методы:** is_expired, get_content, get_confidence, get_metadata, get_all_nodes, to_dict, start, stop, _cleanup_loop, create_snapshot

---

### storage

**Классы:** FractalGraphV2

**Методы:** _init_database, _load_data, _build_indexes, add_node, _find_nearest_group, add_edge, create_semantic_group, semantic_search, keyword_search, get_group_members

---

### tokenizer

**Классы:** Token, GraphTokenizer

**Методы:** _build_ngram_indexes, _build_word_index, _tokenize_text, tokenize, decode, get_context_for_generation, enhance_prompt, get_vocab_size, get_token_id, create_graph_tokenizer

---

### types

**Классы:** NodeType, RelationType, MemoryTier, FractalNode, FractalEdge

**Методы:** get_effective_confidence, to_dict, from_dict, to_dict, from_dict, to_dict, from_dict, create_node_id, create_edge_id, create_group_id

---

### virtual_token_handler

**Классы:** VirtualTokenInfo, VirtualTokenLogitsProcessor, StreamingVirtualTokenHandler, VirtualTokenManager

**Методы:** mark_token_used, reset_used_tokens, process_stream, _extract_text, _process_accumulated, _extract_node_id, _get_content, get_stats, _setup_components, _extract_contents

---

### base_storage

**Классы:** FractalWeightStorage

**Методы:** store_weight, load_weight, get_layer_weights, promote_to_hot, get_stats, save_to_disk, load_from_disk

---

### compression

**Классы:** WeightCompressor

**Методы:** should_compress, compress, decompress, _compress_int8, _decompress_int8, _compress_sparse, _decompress_sparse, get_stats

---

### layer_manager

**Классы:** LayerManager

**Методы:** register_layer, promote_to_hot, promote_to_warm, record_access, _demote_from_hot, _demote_from_warm, get_zone, get_stats, to_dict, from_dict

---

### model_exporter

**Классы:** ModelExporter

**Методы:** export_model, _load_config, _load_model, _export_weights, _load_tokenizer, import_model, list_exported_models, get_export_stats

---

### weight_index

**Классы:** WeightIndex

**Методы:** register, get, get_layer_keys, get_component_keys, get_by_shape, _extract_component, layer_names, component_names, total_weights, total_bytes

---

### gguf_fractal_exporter

**Классы:** GGUFFractalExporter

**Методы:** load, get_architecture, get_block_count, get_embedding_length, get_context_length, get_feed_forward_length, get_attention_heads, get_attention_heads_kv, get_total_params, get_model_summary

---

### gguf_parser

**Классы:** TensorInfo, GGUFMetadata, GGUFParser

**Методы:** parse, _read_string, _read_value, _read_tensor_info, _extract_metadata, _build_summary, _get_quantization_summary, get_layer_tensors, get_component_tensors, get_fractal_hierarchy

---

### graph_learning

**Классы:** NeuromorphicRanker, ExperienceNode, ConceptNode, DynamicContextBuilder, GraphLearningLoop

**Методы:** rank_context_nodes, simple_wave_propagation, to_dict, from_dict, to_dict, from_dict, build_context, _find_relevant_experiences, _find_relevant_concepts, _load_experiences

---

### hotset

**Классы:** _Entry, HotSetManager

**Методы:** _vram_budget, _can_fit, _evict_until_can_fit, _total_bytes_locked, touch, get, put, promote_from_host_view, promote_from_tensor, evict

---

### long_term_memory

**Классы:** LongTermMemory

**Методы:** _init_fields, add_knowledge

---

### longterm_types

**Классы:** MemoryCategory, RetentionPolicy, SemanticEntry, EpisodicEntry

**Методы:** to_dict, to_dict

---

### ltm_consolidation

**Методы:** _consolidate_from_working_impl, _is_duplicate_impl, _generate_health_recommendations_impl, _consolidate_memory_impl, _check_for_contradictions_impl, _optimize_knowledge_graph_impl, _merge_similar_concepts_impl, _merge_concepts_impl, _cleanup_old_episodes_impl, _generate_episodic_health_recommendations_impl

---

### ltm_core

**Классы:** SemanticMemory, EpisodicMemory, LongTermMemory

**Методы:** _load_from_db, _update_knowledge_graph, consolidate_from_working, _is_duplicate, _calculate_text_similarity, retrieve_by_concept, retrieve_by_similarity, get_statistics, start, _consolidation_worker

---

### ltm_retrieval

**Методы:** _retrieve_by_concept_impl, _retrieve_by_similarity_impl, _retrieve_by_time_impl, _retrieve_episodic_by_similarity_impl, _get_user_history_impl

---

### ltm_storage

**Методы:** _load_semantic_from_db, _update_knowledge_graph_impl, _load_episodic_from_db, _store_episode_impl

---

### macro_archive

**Классы:** MacroArchiveState, MacroArchiveWriter, MacroArchive

**Методы:** _align_up, begin_superblock, add_extent, add_subblock, finalize, open_store

---

### macro_integration

**Классы:** MacroPrefetchConfig, MacroblockPrefetcher

**Методы:** _start_worker, _admit, _enqueue, _evict_ready_until, _touch_ready, _io_token_bucket, _worker_loop, _select_layers, _layer_ranges, build

---

### manager_cache

**Методы:** get_memory_statistics, analyze_memory_usage, set_cache_size, clear_cache, optimize_cache, _optimize_memory_lists, clear_inactive_caches, compress_data

---

### manager_core

**Классы:** MemoryManager

**Методы:** _setup_event_connections, _on_memory_optimized, _on_memory_warning, _on_system_state_changed, _deferred_optimize, _deferred_cleanup, get_hybrid_cache, get_state, _init_hybrid_cache, _initialize

---

### manager_gc

**Классы:** _MemoryNodeShim

**Методы:** get_strength_factor, get_all_nodes, get_all_edges, get_node, remove_node, export_memory_graph, add_node, import_memory_graph, save_memory_graph_manifest, load_memory_graph_manifest

---

### manager_operations

**Методы:** _load_working_memory, _save_working_memory, _load_semantic_memory, _save_semantic_memory, _load_episodic_memory, _save_episodic_memory, _load_user_profiles, _save_user_profiles, _save_memory, add_memory

---

### memory_cache

**Классы:** MemoryCache

**Методы:** get, put, clear, remove_expired, get_stats

---

### memory_consolidator

**Классы:** ConsolidationResult, MemoryConsolidator

**Методы:** start, stop, _consolidation_loop, _run_consolidation, _merge_duplicates, _create_semantic_groups, _remove_old_temp_nodes, trigger_now, create_memory_consolidator

---

### memory_core

**Классы:** MemoryNeuron, MemoryField, MemoryDatabase, MemoryCore

**Методы:** _init_database, save_neuron, load_neuron, close, _init_default_fields, is_ready, store_memory, retrieve_memory, get_system_health, close

---

### memory_types

**Классы:** MemoryType, MemoryEntry, UserProfile, EpisodicMemory

**Методы:** to_dict, to_dict, from_dict, to_dict

---

### memory_working

**Классы:** WorkingMemory

**Методы:** _load_from_db, store, _evict_least_important, retrieve, _calculate_similarity, update_importance, decay_memory, get_consolidation_candidates, get_statistics, start

---

### metadata_manager

**Классы:** MetadataManager

**Методы:** update_token_metadata, get_expired_tokens, remove_metadata, clear_all_metadata, save_metadata, _load_metadata, get_stats

---

### activation_profiler

**Классы:** ProfileStats, SimilarProfile, ActivationProfiler

**Методы:** _get_embedder, create_profile, get_or_create_profile, update_profile, record_generation, get_profile, find_similar_profiles, compute_fingerprint, _compute_embedding, _simple_hash_embedding

---

### fractal_graph_l1_l2

**Классы:** ActivationProfileData, RoutingRuleData, FractalGraphL1L2

**Методы:** _ensure_tables, create_activation_profile, get_activation_profile, update_activation_profile, find_similar_profiles, list_activation_profiles, create_routing_rule, get_routing_rule, update_routing_rule_stats, list_routing_rules

---

### pie_adapter

**Классы:** GenerationMetadata, PieIntegration

**Методы:** _get_db_path_from_graph, _init_l1_l2, get_generation_params, _get_default_params, suggest_domain, record_feedback, record_generation, get_profile_stats, before_generation, after_generation

---

### routing_engine

**Классы:** RoutingParams, RoutingEngine

**Методы:** to_dict, create_rule, get_rule, update_rule_stats, record_feedback, get_default_rule, list_rules, get_rule_for_generation, create_default_rules, _data_to_params

---

### semantic_cache

**Классы:** CachedEntry, SemanticCache

**Методы:** get, put, _cosine_similarity, _evict_oldest, clear, get_stats, create_semantic_cache

---

### token_disk_cache

**Классы:** TokenDiskCache

**Методы:** load_token, save_token, delete_token, clear_all, delete_multiple, has_token

---

### unified_fractal_memory

**Классы:** MemoryTier, NodeType, EventTypes, MemoryNode, MemoryEdge

**Методы:** to_dict, from_dict, to_dict, _init_graph_learning, save_experience, get_context_for_query, _publish_migration_event, _migrate_to_warm, _migrate_to_cold, _evict_hot_nodes

---

### working_memory

**Классы:** WorkingMemory

**Методы:** _init_fields, store_fact, _evict_oldest, get_recent_interactions

---

## MLEARNING

### __init__

**Методы:** get_fractal_manager

---

### async_text_generator

**Классы:** SamplingConfig, SimpleKVCache, AsyncTextGenerator, _maybe_async

**Методы:** get, put, _hash_text, _sample_token, _decode_chunk

---

### bitnet_model_manager

**Классы:** BitNetModelManager

**Методы:** _initialize_model, _fallback_load, generate, get_info, list_available_models, unload, install_bitnet_support

---

### comprehensive_learning_system

**Классы:** LearningSession, ComprehensiveLearningSystem

**Методы:** _load_config, _initialize_components, _start_background_processes, start_learning_session, _run_learning_session, _get_learning_topics, _generate_training_texts, _train_model, _assess_current_quality, _update_session_stats

---

### current_manager

**Классы:** OptimizedFractalModelManager

**Методы:** _load_optimal_config, _initialize_components, _initialize_model, _create_optimized_model, _load_optimized_tokenizer, _load_fallback_tokenizer, _find_tokenizer_in_fractal_storage, optimized_tokenize, generate_response_optimized, _clean_response

---

### enhanced_learning_integration

**Классы:** LearningSession, EnhancedLearningIntegration

**Методы:** _load_config, _check_existing_methods, _initialize_components, _start_monitoring, start_enhanced_learning_session, _run_enhanced_learning_session, _get_learning_topics, _generate_enhanced_training_texts, _train_model_enhanced, _assess_current_quality

---

### eva_tokenizer

**Классы:** DummyTokenizer, TokenizationConfig, ЕВАTokenizer

**Методы:** tokenize, encode, is_initialized, _create_fallback_tokenizer, _ensure_initialized, _load_inner_tokenizer, _initialize_fractal_components, _add_fractal_special_tokens, _initialize_fractal_metadata, _initialize_hybrid_cache

---

### fractal_model_manager

**Классы:** FractalModelManager

**Методы:** _get_project_root, _load_config, _initialize_llama_cpp, _initialize_model, _create_conversational_prompt, generate_response, _is_good_response, _get_fallback_response, _clean_response, generate

---

### fractal_qwen_manager

**Классы:** FractalQwenManager

**Методы:** initialize, _load_from_fractal_storage, _load_standard, generate_prompt, _build_refinement_prompt, _extract_prompt, _default_prompt, get_status, get_fractal_qwen

---

### fractal_trainer

**Классы:** FractalKnowledgeTrainer

**Методы:** _get_default_config, _init_model, train, evaluate, _training_step, _evaluation_step, _prepare_inputs, _get_dataloader, _create_optimizer, _create_scheduler

---

### fractal_transformer

**Классы:** FractalConfig, FractalAttention, FractalLayer, FractalTransformer

**Методы:** transpose_for_scores, forward, forward, init_weights, _init_weights, forward, from_pretrained, save_pretrained, get_input_embeddings, set_input_embeddings

---

### __init__

**Классы:** NodeState, NodeIndex, NodeMetadata, GraphNode, FractalGraph

**Методы:** to_dict, activate, deactivate, generate, update_context, get_info, _add_node, create_child_node, get_node, get_node_by_address

---

### convert_to_gguf

**Методы:** get_llama_cpp_dir, check_conda_environ, run_gguf_conversion, convert_via_llama_cpp_python, simple_quantization, check_gguf_available

---

### download_gguf

**Методы:** download_with_progress, try_download_gguf, try_civitai

---

### export_onnx

**Методы:** export_with_optimum, export_with_cli

---

### llama_cpp_hot

**Классы:** LlamaCppHotNode, LlamaCppHotDeployment

**Методы:** load_ethics_prompt, format_prompt_with_ethics, activate_with_llama, generate, initialize, generate, _remove_repetitions, get_status, unload, get_llama_cpp_deployment

---

### llama_cpp_wrapper

**Классы:** LlamaCppGenerator, HotDeploymentLlamaCpp

**Методы:** load, generate, tokenize, get_status, download_qwen_gguf, convert_to_gguf, initialize, generate, test_llama_cpp

---

### onnx_optimizer

**Классы:** OnnxOptimizer, OptimizedGenerator, HotDeploymentOnnx

**Методы:** convert_to_onnx, _export_simple, initialize, _apply_optimizations, generate, get_status, initialize, test_optimized_generation

---

### onnx_runtime

**Классы:** OnnxRuntimeGenerator, OnnxConverter

**Методы:** load, load_tokenizer, generate, get_status, convert, quantize, test_onnx_conversion, test_existing_onnx

---

### openvino_convert

**Методы:** convert_with_cli, convert_direct

---

### openvino_inference

**Классы:** OpenVINOGenerator

**Методы:** convert_to_ir, load, load_tokenizer, generate, convert_and_test

---

### openvino_via_optimum

**Методы:** convert_via_optimum, test_ov_model

---

### optimized_inference

**Классы:** OptimizedQwenGenerator, MultiBatchGenerator

**Методы:** initialize, _apply_optimizations, generate, generate_streaming, initialize, generate_batch, test_optimized_generation, test_batch_generation

---

### hybrid_model_manager

**Классы:** WindowType, ModelWindow, HybridModelManager

**Методы:** _get_project_root, initialize, _analyze_resources, register_model, _estimate_model_size, load_tokenizer, load_model, _find_window, _can_fit_in_vram, _load_model_to_window

---

### language_filter

**Классы:** LanguageFilterTokenizer, DynamicQuantizationManager, ModelModeController

**Методы:** _update_blocked_tokens, set_mode, encode, decode, _filter_input_text, _keep_only_russian, _remove_chinese, _remove_foreign_latin, filter_output, set_mode

---

### ml_core

**Классы:** ModelHealth, MLCore

**Методы:** from_dict, update_stats, get_statistics, enable_cache, set_cache_size, set_cache_ttl, cleanup_cache, get_cached_response, cache_response, get_model_health_report

---

### ml_types

**Классы:** ModelType, ProcessingMode, MLModelConfig, TrainingConfig, ModelStats

**Методы:** to_dict, to_dict, to_dict

---

### model_config

**Методы:** get_model_config, list_available_models, get_recommended_model

---

### model_manager

**Классы:** ModelManager

**Методы:** _init_fractal_store, _setup_component, get_model_for_task, export_task_model_to_fractal, scan_models_directory, _save_to_fractal_store, register_models_with_core_brain, cleanup

---

### model_selector

**Классы:** ModelSelector

**Методы:** get_current_model_info, list_models, switch_model, load_model, _load_qwen, _load_bitnet, unload_model, get_recommendation, print_model_comparison

---

### neuromorphic_memory

**Классы:** NeuromorphicMemoryLayer

**Методы:** _reset_parameters, _split_heads, _combine_heads, forward, _update_memory, get_initial_memory

---

### parallel_tokenization

**Классы:** ParallelTokenizer

**Методы:** start, stop, submit, _worker, _process_item, _persist_stub

---

### qwen_api_client

**Классы:** QwenAPIError, QwenAPIConnectionError, QwenAPIAuthenticationError, QwenAPIRateLimitError, QwenAPIClient

**Методы:** _initialize_client, generate, generate_sync, generate_stream, is_available, get_models

---

### qwen_model_manager

**Классы:** QwenModelManager

**Методы:** get_qwen_model_path, is_qwen_available, get_qwen_model_manager, reset_qwen_model_manager, _apply_compile_optimization, _get_device, _initialize_model, generate, chat, _format_chat

---

### sentence_transformers_cache

**Методы:** _detect_device, _encode_with_cache, get_sentence_transformer, encode_text, encode_batch, clear_sentence_transformer_cache, is_sentence_transformer_loaded

---

### fractal_model_loader

**Классы:** FractalModelLoader

**Методы:** has_model, list_models, load_model, save_model, _create_default_model

---

### fractal_store_core

**Классы:** FractalContainer, FractalWeightStore

**Методы:** get_memory_size, _load_legacy_format, get_container_data, reconstruct_state_dict, load_from_disk

---

### fractal_store_utils

**Классы:** FractalStoreUtils

**Методы:** optimize_memory_usage, get_memory_stats, validate_store_integrity, export_store, import_store

---

### fractal_weight_store

**Классы:** FractalWeightStore

**Методы:** store, save_to_disk, update_hot_window, get, get_tensor_by_key, store_tensor, update_hot_window, get

---

### legacy_format_patch

**Методы:** _load_legacy_format

---

### memory_graph_store

**Классы:** MemoryGraphStore

**Методы:** add_node, add_edge, get_node_vector, find_similar_nodes, get_connected_nodes, get_training_data, store_tensor, load_tensor, remove_tensor, get_tensor_metadata

---

### model_storage_adapter

**Классы:** ModelStorageAdapter

**Методы:** store_model, load_model, remove_model, save_to_disk, load_from_disk, get_model_info, get_stats

---

### model_storage_config

**Классы:** ModelStorageConfig

---

### opt_cache

**Методы:** optimize_memory, clear_gpu_cache

---

### opt_core

**Классы:** OptimizedFractalModelManager

**Методы:** _load_optimal_config

---

### opt_models

**Методы:** _initialize_components, _initialize_model, _create_optimized_model, _load_optimized_tokenizer, _load_fallback_tokenizer, _find_tokenizer_in_fractal_storage, optimized_tokenize, generate_response, generate_text, generate_response_optimized

---

### store_cache

**Методы:** _initialize_hot_window, _calculate_container_priority, _get_available_hot_window_space, _evict_lowest_priority_containers, _load_container_from_ssd, _optimize_fractal_structure, _reconfigure_fractal_structure, _determine_new_parameters, _extract_all_data, _update_metadata

---

### store_core

**Классы:** NodeProxy, EdgeProxy, KnowledgeGraphProxy, FractalContainer, FractalWeightStore

**Методы:** get_all_nodes, get_all_edges, update_priority, get_memory_size, clear, repack_model_to_fractal, _safe_load_model, _load_hf_model_dir, _build_knowledge_graph, export_hf_model_to_fractal

---

### store_operations

**Методы:** pack_model_weights, pack_state_dict, get_container, get_tensor, save_weights, load_weights, unpack_model_weights, reconstruct_state_dict, index, save_to_disk

---

### store_queries

**Методы:** get_statistics, get_similar_tensors, _analyze_container_usage, _are_containers_adjacent, _needs_reconfiguration, _calculate_fragmentation, _usage_patterns_changed, compute_checksum

---

### unified_fractal_store

**Классы:** UnifiedStorageConfig, UnifiedFractalStore

**Методы:** _vectorize_text, store_knowledge, query_knowledge, train_on_knowledge, _get_batches, _train_step, save_model_state, load_model_state

---

### unified_graph_store

**Классы:** UnifiedMemoryGraph

**Методы:** store, get, add_node, add_edge, get_node, get_node_vector, get_connected_nodes, find_similar_nodes, get_training_data, save_state

---

### unified_storage

**Классы:** StorageConfig, UnifiedFractalStorage

**Методы:** _text_to_vector_fallback, store_knowledge, connect_knowledge, query_knowledge, register_model, save_model, load_model, train_step, get_training_data, save_state

---

### text_quality_improver

**Классы:** GenerationMetrics, TextQualityImprover

**Методы:** _initialize_quality_checker, improve_generation_parameters, analyze_text_quality, _calculate_coherence, _calculate_diversity, _calculate_length_score, _calculate_grammar_score, post_process_response, _clean_text, _improve_low_quality_response

---

### text_quality_learning_integration

**Классы:** TextQualityLearningIntegration

**Методы:** activate_integration, _check_components, _ensure_components, check_components_availability, _extend_learning_module, _add_quality_controls, _add_model_improvement_section, _add_real_time_quality_monitoring, _setup_quality_monitoring, _setup_auto_improvement

---

### text_quality_trainer

**Классы:** TrainingConfig, TextDataset, TextQualityTrainer

**Методы:** prepare_training_data, _generate_quality_examples, train, evaluate_quality, _clean_response, _has_russian_text, save_model, train_async

---

### tokenization_fractal

**Классы:** ExtendedFractalTokenizer

**Методы:** encode_fractal_path, encode_metadata, prepare_fractal_input, batch_encode_plus_fractal, save_pretrained, from_pretrained

---

### tokenizer_registry

**Классы:** TokenizerRegistry, LlamaCppTokenizerAdapter

**Методы:** get_tokenizer, _find_model_path, _load_tokenizer, reset, get_stats, tokenize, encode, decode, batch_decode

---

### training_types

**Классы:** TrainingPhase, CheckpointStatus, TrainingCheckpoint, TrainingMetrics

**Методы:** to_dict, to_dict

---

### unified_fractal_manager

**Классы:** UnifiedFractalManager

**Методы:** _select_manager, _should_use_optimized, generate_response, get_quality_metrics, improve_quality, get_performance_stats, start_enhanced_learning_session, generate_enhanced_response, get_enhanced_system_status, add_enhanced_topics

---

### unified_text_processor

**Классы:** UnifiedTextProcessor

**Методы:** is_embedding_loading_disabled, _setup_component, _init_nlp_models, _load_embedding_models, process_text, tokenize, encode, lemmatize, extract_keywords, analyze_sentiment

---

### unit_components

**Методы:** _init_ml_core, _init_text_processor, _init_model_manager, _init_response_generator, _init_hybrid_cache, _link_components, _verify_basic_functionality, _init_training_orchestrator, _is_training_mode, get_system_health

---

### unit_core

**Классы:** MLUnit

**Методы:** _load_brain_config, _get_hybrid_cache_config, _get_project_root, _maybe_cleanup_memory

---

### unit_training

**Методы:** _init_training_orchestrator

---

### universal_model_manager

**Классы:** UniversalModelManager

**Методы:** _auto_select, _load_model, generate, get_status, list_models, get_recommendation

---

### web_search_learning_integration

**Классы:** WebSearchLearningIntegration

**Методы:** search_and_enhance_response, _should_search, _perform_web_search, _enhance_response_with_search, _extract_key_information, _create_enhanced_prompt, _clean_enhanced_response, _analyze_response_quality, _update_integration_stats, generate_training_texts_from_search

---

## MONITORING

### system_monitor

**Классы:** Metric, Alert, MetricsCollector, HealthChecker, AlertManager

**Методы:** record_metric, get_metrics, get_latest_metric, get_metric_stats, register_check, check_all_components, get_system_health, add_alert_rule, check_alerts, resolve_alert

---

## NEUROMORPHIC

### neuromorphic_memory

**Классы:** NeuromorphicMemory

**Методы:** simulator, store, retrieve, analyze, get_health, reset, start, stop

---

### sim_core

**Классы:** NeuromorphicSimulator

**Методы:** initialize, _register_fractal_store_handlers, _on_container_accessed, _on_hot_window_updated, _init_neural_networks, simulate_activity, _generate_input_stimulus, consolidate_activity, _analyze_activation_patterns, _update_fractal_structure

---

### sim_neurons

**Классы:** FallbackNeuralNetwork

**Методы:** simulate_step, get_activity_pattern, connect_neurons, get_network_info

---

### sim_plasticity

**Классы:** STDPPlasticity, AdaptiveThreshold, HomeostaticPlasticity

**Методы:** compute_weight_change, get_stats, update, get_threshold, reset, adjust_synaptic_scaling, get_stats

---

### sim_spikes

**Классы:** SpikeEvent, NeuralActivity, SpikeGenerator, SpikePropagator

**Методы:** to_dict, from_dict, to_dict, from_dict, get_analysis, generate_spikes, get_spike_rate, get_stats, propagate, get_propagation_stats

---

### sim_synapses

**Классы:** SynapseManager

**Методы:** update_weights, get_connection_strength, prune_weak_connections, get_stats

---

## NLP

### text_processor

**Классы:** TextProcessor

**Методы:** _get_project_root, tokenizer, _initialize_tokenizer, tokenize, _encode_cached, clear_cache, encode, decode, preprocess_text, process_text

---

## NLP_FALLBACKS

### nlp_fallbacks

**Классы:** BatchProcessor

**Методы:** _safe_import_sklearn, _safe_import_nltk_vader, _safe_import_spacy, clean_text, tokenize, jaccard_similarity, cosine_similarity_texts, get_sentiment_analyzer, polarity_scores, process_batch

---

## PREPROCESS

### preprocessing_pipeline

**Классы:** ExtractedEntity, PreprocessedQuery, GGUFEntityExtractor, PreprocessingPipeline

**Методы:** extract_entities, _parse_entities_response, _fallback_extraction, check_clarification_needed, _parse_clarification_response, process, _extract_keywords, _build_raw_context, _save_to_cache, get_cached_context

---

## REASONING

### analytics_module

**Классы:** SemanticEntity, LogicalBlock, AnalyticsResult, AnalyticsModule

**Методы:** extract_entities, decompose_into_logical_blocks, _split_into_sentences, _detect_block_type, _fallback_split, analyze_text, _calculate_coherence, get_entity_summary, format_for_prompt

---

### clarification_generator

**Классы:** ClarificationGenerator

**Методы:** generate_clarification, _generate_question_for_entity, _generate_entity_question, _generate_from_query_analysis, generate_simple_clarification, _extract_key_nouns

---

### combined_metric

**Классы:** ImprovementResult, CombinedMetricCalculator

**Методы:** calculate_improvement, _get_adaptive_weights, _calculate_ethics_improvement, _extract_ethics_score, _calculate_contradiction_improvement, _extract_contradiction_count, _calculate_knowledge_improvement, _extract_knowledge_score, should_continue, get_threshold

---

### confidence_scorer

**Методы:** get_adaptive_weights, get_adaptive_threshold, calculate_adaptive_confidence, calculate_ethics_score, calculate_contradiction_score, calculate_knowledge_score, calculate_overall_confidence, should_terminate, get_confidence_level

---

### correlation_calculator

**Классы:** CorrelationResult, CorrelationCalculator

**Методы:** check_correlation, _calculate_semantic_correlation, _calculate_knowledge_relevance, _calculate_web_relevance, check_multi_iteration, calculate_correlation_score

---

### enhanced_reasoning_engine

**Классы:** ReasoningIteration, EnhancedReasoningEngine

**Методы:** _init_modules_from_brain, _get_qwen, _get_fractal_qwen, process_query, _build_prompt, _combine_prompts_context, _fallback_response, _should_stop, _calculate_confidence, _perform_self_learning

---

### entity_extractor

**Классы:** ExtractedEntity, ExtractionResult, EntityExtractor

**Методы:** extract_from_query, extract_from_response, extract_from_contradiction, extract_all, save_to_knowledge_graph, format_for_self_learning

---

### fractal_address

**Классы:** FractalAddress

**Методы:** _compute_hash, distance_to, cosine_similarity, normalized, get_fractal_path, _get_branch_index, to_vector, from_vector, create_root, to_dict

---

### fractal_base

**Классы:** FractalNodeType, FractalRelationType, FractalNode, FractalEdge, FractalAddress

**Методы:** add_child, add_relation, to_dict, from_dict, to_dict, compute_address, get_parent_address, resolve_address, get_level_path, add_node

---

### fractal_embedder

**Классы:** FractalEmbedder

**Методы:** _init_model, generate_embedding, embed_text, _generate_hash_embedding, find_similar, embed_node, compute_similarity, create_address_from_text, create_address_from_query, batch_embed

---

### fractal_retriever

**Классы:** FractalRetriever

**Методы:** retrieve_with_depth, retrieve_path, retrieve_with_depth_from_node, _retrieve_recursive, retrieve_by_confidence, retrieve_session, retrieve_recent, semantic_search, search_by_type, get_subtree

---

### fractal_storage

**Классы:** FractalStorage

**Методы:** _load, _save, flush, add_node, get_node, get_children, get_path_to_root, search_by_content, get_nodes_by_level, add_reasoning_step

---

### fractal_tokenizer

**Классы:** FractalTokenizer, FractalTokenizerWrapper

**Методы:** _init_special_tokens, tokenize, _simple_tokenize, tokenize_to_ids, decode, train, pad_token_id, unk_token_id, bos_token_id, eos_token_id

---

### integration

**Классы:** ReasoningIntegration

**Методы:** _error_response, _load_config, integrate_with_brain, process_query, get_stats, enable, disable, get_status, integrate_reasoning

---

### prompt_composer

**Классы:** ModulePrompt, ComposedPrompt, PromptComposer

**Методы:** compose, _combine_module_feedback, _truncate_prompt, compose_from_results, _generate_contradiction_prompt, _generate_ethics_prompt, _generate_websearch_context, get_module_weights

---

### reasoning_nodes

**Классы:** ReasoningNodeType, ReasoningRelationType, ReasoningNode, ReasoningSession

**Методы:** to_knowledge_node_format, to_node

---

### reasoning_types

**Классы:** ReasoningPhase, ReasoningStep, ReasoningResult, AnalysisResult

**Методы:** to_dict, to_dict, reasoning_text, to_dict

---

### self_reasoning_engine

**Классы:** SelfReasoningEngine

**Методы:** _init_fractal_components, process_query, _analyze_logical_factors, _evaluate_ethics_factor, _evaluate_knowledge_factor, _evaluate_contradiction_factor, _evaluate_context_factor, _evaluate_logic_factor, analyze_response, _find_alternative_reasoning_branches

---

### semantic_stability

**Классы:** StabilityResult, SemanticStabilityChecker

**Методы:** compute_similarity, _normalize_text, _jaccard_similarity, _levenshtein_similarity, _levenshtein_distance, _sequence_similarity, _get_ngrams, _short_text_similarity, is_stable, analyze_changes

---

### sre_context

**Методы:** _build_contextual_query, _get_wikipedia_context, _search, _determine_query_type, _get_generation_params, _generate_with_qwen, _generate_simple_response, _get_knowledge_response

---

### sre_feedback

**Методы:** process_user_feedback, learn_from_outcome, _update_confidence_threshold, refine_reasoning_chain, self_correct, adaptive_recursion_depth, cross_session_learning, _trigger_self_learning, get_feedback_stats

---

### sre_quality

**Методы:** check_quality, _sanitize_response, _clean_filler_start, _remove_looping_blocks, _check_relevance

---

### sre_recursive

**Методы:** _recursive_process_query, _check_semantic_stability, _recursive_reasoning_step, _is_complex_query, decompose_query, retrieve_similar_reasoning, build_recursive_context, _synthesize_recursive_results, _linear_process_query, _init_retriever

---

## RECOVERY

### recovery_system

**Классы:** RecoveryCheckpoint, RecoveryPlan, ComponentStateManager, FailureDetector, RecoveryManager

**Методы:** save_component_state, load_component_state, _save_checkpoint_to_disk, _load_checkpoints_from_disk, _calculate_checksum, cleanup_old_checkpoints, register_failure_pattern, detect_failure, _matches_pattern, get_failure_statistics

---

## RUN

### run

**Методы:** _cleanup_pid, _cleanup_brain, _signal_handler, _check_singleton, launch_gui, main

---

## RUNTIME

### simple_model

**Методы:** example_model_fn

---

### worker_pool

**Классы:** InferenceWorkerPool

**Методы:** _resolve_callable, _sanitize_batch_inplace, _worker_entry, start, stop, running, submit, recv, infer_batches

---

## SCRIPTS

### activate_max_cache

**Методы:** activate_max_cache

---

### complete_fractal_solution

**Методы:** cleanup_old_models, create_unique_fractal_tokenizer, download_and_export_rugpt3_with_custom_tokenizer, create_fractal_integration, setup_fractal_integration, load_fractal_model_if_available, main

---

### export_qwen

**Методы:** export

---

### load_gguf_to_fg

**Методы:** main

---

### migrate_events

**Методы:** log_migration_event, run_migration

---

### migrate_kg_to_fg

**Методы:** main

---

### migrate_to_optimized

**Методы:** migrate_to_optimized

---

### simple_test

**Методы:** test_basic_generation

---

## SECURITY

### security_framework

**Классы:** User, SecurityEvent, RateLimiter, AuthenticationManager, AuthorizationManager

**Методы:** is_allowed, get_remaining_requests, _create_default_admin, authenticate, _verify_password, _create_session, validate_session, logout, create_user, check_permission

---

## SERVER_HANDLERS

### server_handlers

**Методы:** api_analytics, api_learning, api_settings, api_documents, api_knowledge_graph, api_cache_stats, api_system

---

## SERVER_ROUTES

### server_routes

**Методы:** index, api_login, api_sessions, api_session, api_upload, api_chat, api_entities, api_feedback, api_status, api_metrics

---

## STORAGE

### fractal_storage

**Классы:** FractalStorage

**Методы:** initialize, store, retrieve, delete, get_tokenizer, save_tokenizer, get_model

---

### storage_types

**Классы:** StorageType, AccessPattern, StorageMetrics, StorageEntry

**Методы:** to_dict, to_dict

---

## SYSTEM

### fault_tolerance

**Классы:** FaultTolerance

**Методы:** is_ready, start, stop, register_fault_handler, handle_fault, get_system_health

---

### health_monitor

**Классы:** HealthMonitor

**Методы:** analyze_system_health, _add_ml_issue, _generate_recommendations, analyze_evolution, _get_analysis_history, _analyze_trends, _calculate_historical_health, _calculate_improvement_rate, _identify_critical_events, _analyze_component_performance

---

### system_types

**Классы:** HealthStatus, AlertLevel, SystemHealth, SystemAlert

**Методы:** to_dict, to_dict

---

## SYSTEM_SELFTEST

### system_selftest

**Методы:** test_component, main

---

## TESTS

### test_pie_integration

**Методы:** test_imports, test_l1_l2_graph, test_pie_integration, test_fallback_pipeline, test_dual_generator_pie, main

---

## TOOLS

### dependency_scan

**Методы:** resolve_from, dfs

---

### document_reader

**Классы:** DocumentContent, DocumentTextReader

**Методы:** read, _read_file, _detect_encoding, read_as_messages, read_text_file_simple

---

### import_pipeline

**Классы:** ImportedDocument, ImportPipeline

**Методы:** _safe_import, iter_segments, import_path, _read_txt, _read_pdf, _read_epub, _normalize_text, _chunk_text, _approx_tokens, _derive_id

---

### layer_expertise_analysis

**Методы:** analyze_layer_expertise

---

### system_generation_analysis

**Классы:** SystemAnalyzer

**Методы:** analyze_module_structure, analyze_generation_flow, test_generation_pipeline, analyze_file_structure, create_analysis_report, main

---

## TRAINING

### gguf_training_system

**Классы:** TrainingStatus, TrainingMetrics, VerifiedKnowledge, GGUFTrainingSystem

**Методы:** deploy_training_model, verify_training_model, initialize_training_model, auto_start_if_ready, start, stop, _training_loop, _extract_verified_knowledge, _get_knowledge_graph, _get_all_nodes

---

## UTILS

### text_quality

**Классы:** TextQualityChecker, TextPostProcessor

**Методы:** check_quality, _is_gibberish_word, _calculate_score, process, _basic_clean, _remove_query, _aggressive_clean, check_and_fix_response

---

## WEBSEARCH

### cache_manager

**Классы:** CacheManager

**Методы:** _load_cache, _save_cache, generate_cache_key, get_cached_results, save_to_cache, clear_cache, get_cache_size

---

### database_manager

**Классы:** DatabaseManager

**Методы:** _get_connection, _init_database, save_query, get_stats, get_last_query, update_stats, get_recent_queries, close

---

### search_engines

**Классы:** SearchEngines

**Методы:** _rotate_user_agent, _random_delay, search_google, search_yandex, search_bing, search_duckduckgo, _search_duckduckgo_html, _search_searx, _search_brave, _create_local_results

---

### search_models

**Классы:** SearchResult, SearchQuery

---

### search_types

**Классы:** SearchEngine, ContentType, SearchResult, SearchQuery

**Методы:** to_dict, to_dict

---

### web_search_engine

**Классы:** WebSearchEngine

**Методы:** _get_connection, _init_database, _update_query_stats, _load_cache, _save_cache, _init_cache_cleanup, _cache_cleanup_worker, _cleanup_expired_cache, set_search_engines, get_active_search_engines

---

### web_search_integrated

**Классы:** AsyncWebSearchClient, IntegratedWebSearchEngine

**Методы:** load_brain_config, tavily_search, get_instance, _do_initialize, _do_start, _do_stop, search, _basic_web_search, _save_search_cache_async, search_with_filters

---

