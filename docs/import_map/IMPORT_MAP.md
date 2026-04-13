# EVA AI - Полная карта импортов

> Дата生成: 2026-04-13
> Всего файлов: 531
> Модулей с импортами: 515

---

## Структура проекта

- **root**: 515 файлов

---

## Детальная карта импортов

## ROOT

### __init__.py

**Internal (eva_ai):**
- import eva_ai.memory
- import eva_ai.core
- import eva_ai.mlearning

---

### __main__.py

**Stdlib:**
- import os
- import logging

**External:**
- import sys
- import io
- import ctypes

**Internal (eva_ai):**
- from eva_ai.run import main as run_main

---

### adaptation\__init__.py

**Stdlib:**
- import logging

**External:**
- from .adaptation_core import AdaptationManager
- from .adaptation_profiles import UserFeedback, UserProfile

---

### adaptation\adaptation_analytics.py

**Stdlib:**
- import os
- import logging
- import time
- import sqlite3

**External:**
- import hashlib
- from typing import Dict, List, Optional, Any
- from datetime import datetime, timedelta
- from .adaptation_core import AdaptationManager

---

### adaptation\adaptation_core.py

**Stdlib:**
- import os
- import logging
- import time
- import threading
- import sqlite3
- import numpy
- import json

**External:**
- from typing import Dict, List, Optional, Any, Tuple
- from datetime import datetime, timedelta
- import re
- import hashlib
- from .adaptation_profiles import UserFeedback, UserProfile
- import spacy

---

### adaptation\adaptation_integrated.py

**Stdlib:**
- import logging
- import os
- import time
- import json
- import json

**External:**
- from typing import Dict, List, Optional, Any, Tuple
- from datetime import datetime

**Internal (eva_ai):**
- from eva_ai.core.base_component import BaseComponent, ComponentState
- from eva_ai.core.event_bus import get_event_bus, Event, EventTypes
- from eva_ai.adaptation.adaptation_core import AdaptationManager

---

### adaptation\adaptation_integration.py

**Stdlib:**
- import os
- import logging
- import time
- import json

**External:**
- from typing import Dict, List, Optional, Any
- from datetime import datetime, timedelta
- import hashlib
- from .adaptation_core import AdaptationManager
- from .adaptation_profiles import UserProfile, UserFeedback
- from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
- import math
- import spacy

---

### adaptation\adaptation_manager.py

**Stdlib:**
- import logging
- import time
- import os
- import json
- import threading

**External:**
- from typing import Dict, List, Optional, Any, Tuple
- from datetime import datetime
- from dataclasses import dataclass, field, asdict, is_dataclass
- from collections import defaultdict, deque
- from .adaptation_profiles import UserProfile, UserFeedback

---

### adaptation\adaptation_profiles.py

**Stdlib:**
- import os
- import logging
- import time
- import json

**External:**
- from typing import Dict, List, Optional, Any
- from dataclasses import dataclass, field

---

### adaptation\adaptation_types.py

**External:**
- from dataclasses import dataclass, field
- from typing import List, Dict, Any, Optional
- from enum import Enum

---

### adapters\torch_adapter.py

**Stdlib:**
- import time
- import torch
- import logging

**External:**
- from __future__ import annotations
- import math
- from dataclasses import dataclass
- from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

**Internal (eva_ai):**
- from eva_ai.core.batch_wrapper import (
- from eva_ai.core.device_resolver import resolve_device, should_pin_memory

---

### analytics\__init__.py

**External:**
- from .analytics_manager import AnalyticsManager
- from .learning_integration import AnalyticsLearningIntegration
- from .contradiction_analyzer import ContradictionAnalyzer, RelevanceCalculator

---

### analytics\analytics_integrated.py

**Stdlib:**
- import logging
- import time
- import os
- import json

**External:**
- from typing import Dict, List, Any, Optional, Tuple
- from datetime import datetime, timedelta
- from collections import defaultdict, deque

**Internal (eva_ai):**
- from eva_ai.core.base_component import BaseComponent, ComponentState
- from eva_ai.core.event_bus import get_event_bus, Event, EventTypes
- from eva_ai.analytics.analytics_manager import AnalyticsManager

---

### analytics\analytics_manager.py

**Stdlib:**
- import os
- import logging
- import time
- import json
- import threading

**External:**
- from typing import Dict, List, Any, Optional, Tuple
- from collections import defaultdict, deque
- from datetime import datetime, timedelta

**Internal (eva_ai):**
- from eva_ai.learning.performance_analyzer import PerformanceAnalyzer
- from eva_ai.knowledge.knowledge_analytics import KnowledgeAnalytics
- from eva_ai.learning.learning_opportunity_manager import LearningOpportunityManager

---

### analytics\contradiction_analyzer.py

**Stdlib:**
- import os
- import logging
- import time
- import json
- import numpy

**External:**
- import re
- from typing import Dict, List, Any, Optional, Tuple
- from collections import defaultdict
- from datetime import datetime, timedelta
- from sentence_transformers import SentenceTransformer
- from sklearn.metrics.pairwise import cosine_similarity

---

### analytics\learning_integration.py

**Stdlib:**
- import os
- import logging
- import time
- import json

**External:**
- from typing import Dict, List, Any, Optional, Tuple
- from datetime import datetime, timedelta

---

### backends\pie\__init__.py

**External:**
- from .layer_wise import (

---

### backends\pie\base.py

**External:**
- from abc import ABC, abstractmethod
- from typing import Dict, Any, Optional, List, Iterator
- from dataclasses import dataclass
- from pathlib import Path

---

### backends\pie\gguf_backend.py

**Stdlib:**
- import logging
- import time
- import os
- import logging

**External:**
- from pathlib import Path
- from typing import Dict, Any, List, Optional, Iterator
- from llama_cpp import Llama
- from .base import BaseBackend, GenerationResult, GenerationConfig
- import psutil

---

### backends\pie\layer_wise.py

**Stdlib:**
- import logging
- import numpy
- import threading
- import time

**External:**
- from typing import Dict, List, Optional, Any, Tuple
- from dataclasses import dataclass
- from concurrent.futures import ThreadPoolExecutor, as_completed
- from pathlib import Path
- from sentence_transformers import SentenceTransformer

---

### backends\pie\onnx_backend.py

**Stdlib:**
- import logging

**External:**
- from typing import Dict, Any, List, Optional, Iterator
- from .base import BaseBackend, GenerationResult, GenerationConfig

---

### backends\pie\transformers_backend.py

**Stdlib:**
- import logging

**External:**
- from typing import Dict, Any, List, Optional, Iterator
- from .base import BaseBackend, GenerationResult, GenerationConfig

---

### cache\__init__.py

**Internal (eva_ai):**
- from eva_ai.memory.hybrid_token_cache import HybridTokenCache

---

### config\__init__.py

**Stdlib:**
- import os

---

### config\apply_optimal_config.py

**Stdlib:**
- import os
- import json

**External:**
- import sys
- import io

---

### contradiction\contradiction_analysis.py

**Stdlib:**
- import os
- import logging
- import time
- import json
- import numpy
- import os
- import logging
- import time
- import json
- import numpy

**External:**
- import re
- from collections import defaultdict
- from typing import Dict, List, Optional, Any, Tuple, Set, Union
- from datetime import datetime, timedelta
- import random
- import hashlib
- import re
- from collections import defaultdict
- from typing import Dict, List, Optional, Any, Tuple, Set, Union
- from datetime import datetime, timedelta
- ... и ещё 2

**Internal (eva_ai):**
- from eva_ai.nlp_fallbacks import (
- from eva_ai.nlp_fallbacks import (

---

### contradiction\contradiction_core.py

**Internal (eva_ai):**
- from eva_ai.contradiction.core_detection import Contradiction, ContradictionCore

---

### contradiction\contradiction_detection.py

**Internal (eva_ai):**
- from eva_ai.contradiction.detect_core import ContradictionDetector

---

### contradiction\contradiction_generator.py

**Stdlib:**
- import time
- import logging

**External:**
- import random
- from typing import Dict, List, Any, Optional
- from dataclasses import dataclass

---

### contradiction\contradiction_integrated.py

**Stdlib:**
- import logging
- import time
- import os
- import json
- import json

**External:**
- from typing import Dict, List, Optional, Any, Tuple
- from datetime import datetime

**Internal (eva_ai):**
- from eva_ai.core.base_component import BaseComponent, ComponentState
- from eva_ai.core.event_bus import get_event_bus, Event, EventTypes
- from eva_ai.contradiction.contradiction_manager import ContradictionManager
- from eva_ai.contradiction.contradiction_resolver import ContradictionResolver

---

### contradiction\contradiction_learning.py

**Internal (eva_ai):**
- from eva_ai.contradiction.learn_core import ContradictionLearner, ContradictionLearningOpportunity

---

### contradiction\contradiction_manager.py

**Stdlib:**
- import logging

**External:**
- from typing import List, Dict, Any, Optional
- from ..core.base_component import BaseComponent
- from .contradiction_core import OptimizedContradictionDetector

---

### contradiction\contradiction_miner.py

**Stdlib:**
- import os
- import time
- import json
- import logging
- import threading
- import numpy

**External:**
- from typing import Dict, List, Any, Optional, Set, Tuple
- from dataclasses import dataclass, field, asdict
- from enum import Enum
- from collections import defaultdict
- from concurrent.futures import ThreadPoolExecutor

**Internal (eva_ai):**
- from eva_ai.core.deferred_command_system import CommandPriority

---

### contradiction\contradiction_reputation.py

**Stdlib:**
- import os
- import logging
- import time
- import json
- import sqlite3
- import numpy

**External:**
- import re
- from collections import defaultdict
- from datetime import datetime, timedelta
- from typing import Dict, List, Optional, Any, Tuple, Set, Union
- import tldextract
- import nltk
- from nltk.sentiment import SentimentIntensityAnalyzer
- from nltk.corpus import stopwords
- from nltk.tokenize import word_tokenize

---

### contradiction\contradiction_resolution.py

**Stdlib:**
- import os
- import logging
- import time
- import json
- import numpy

**External:**
- import re
- from collections import defaultdict
- from typing import Dict, List, Optional, Any, Tuple, Set, Union
- from datetime import datetime, timedelta
- import random
- import hashlib
- import nltk
- from nltk.sentiment import SentimentIntensityAnalyzer
- from nltk.corpus import stopwords
- from nltk.tokenize import word_tokenize

---

### contradiction\contradiction_resolver.py

**Stdlib:**
- import logging
- import time

**External:**
- from typing import Dict, List, Optional, Any
- from .contradiction_core import OptimizedContradictionDetector, Contradiction
- from .contradiction_resolution import ContradictionResolution

---

### contradiction\contradiction_responses.py

**Stdlib:**
- import logging

**External:**
- import re
- from typing import Dict, Any, Optional

---

### contradiction\contradiction_strategies.py

**Stdlib:**
- import logging

**External:**
- from typing import Dict, List, Any, Optional
- from abc import ABC, abstractmethod

---

### contradiction\contradiction_types.py

**External:**
- from dataclasses import dataclass, field
- from typing import List, Dict, Any, Optional
- from enum import Enum
- from datetime import datetime

---

### contradiction\core_detection.py

**Stdlib:**
- import logging
- import time
- import os
- import json
- import sqlite3
- import threading

**External:**
- import nltk
- from typing import Dict, List, Optional, Any, Tuple, Union
- from dataclasses import dataclass, field, asdict
- from collections import defaultdict, deque
- from datetime import datetime
- from nltk.sentiment import SentimentIntensityAnalyzer
- from nltk.corpus import stopwords
- from nltk.tokenize import word_tokenize
- from .core_resolution import ResolutionMixin
- from .core_tracking import TrackingMixin

**Internal (eva_ai):**
- from eva_ai.distributed.database_utils import get_connection, execute_query
- from eva_ai.knowledge.context_entity import EntityExtractor, AmbiguousEntity, AmbiguityType

---

### contradiction\core_resolution.py

**Stdlib:**
- import logging
- import time

**External:**
- from typing import Dict, List, Optional, Any
- from collections import defaultdict
- from .core_detection import Contradiction

---

### contradiction\core_tracking.py

**Stdlib:**
- import logging
- import time
- import json

**External:**
- from typing import Dict, List, Optional, Any
- from collections import defaultdict

---

### contradiction\detect_core.py

**Stdlib:**
- import os
- import logging
- import time
- import json
- import threading
- import numpy
- import torch

**External:**
- import re
- import random
- import hashlib
- from collections import defaultdict
- from typing import Dict, List, Optional, Any, Tuple, Set, Union
- from datetime import datetime, timedelta
- from sentence_transformers import SentenceTransformer
- from .detect_semantic import SemanticDetectionMixin
- from .detect_logical import LogicalDetectionMixin
- from .detect_temporal import TemporalDetectionMixin

**Internal (eva_ai):**
- from eva_ai.nlp_fallbacks import (
- from eva_ai.mlearning.sentence_transformers_cache import get_sentence_transformer

---

### contradiction\detect_logical.py

**Stdlib:**
- import logging
- import time

**External:**
- import hashlib
- from typing import Dict, List, Any, Optional

---

### contradiction\detect_semantic.py

**Stdlib:**
- import logging
- import numpy

**External:**
- import re
- from typing import Dict, List, Any, Tuple

**Internal (eva_ai):**
- from eva_ai.nlp_fallbacks import compute_semantic_similarity, tokenize

---

### contradiction\detect_temporal.py

**Stdlib:**
- import logging
- import time

**External:**
- from typing import Dict, List, Any, Optional
- from collections import defaultdict

---

### contradiction\learn_core.py

**Stdlib:**
- import logging
- import time
- import json
- import numpy
- import torch

**External:**
- import re
- import random
- import hashlib
- from collections import defaultdict
- from typing import Dict, List, Optional, Any, Tuple, Set, Union
- from datetime import datetime, timedelta
- import nltk
- from nltk.sentiment import SentimentIntensityAnalyzer
- from nltk.corpus import stopwords
- from nltk.tokenize import word_tokenize
- ... и ещё 5

---

### contradiction\learn_feedback.py

**Stdlib:**
- import logging
- import time

**External:**
- from typing import Dict, List, Any, Optional

---

### contradiction\learn_patterns.py

**Stdlib:**
- import logging
- import time

**External:**
- import hashlib
- from typing import Dict, List, Any, Optional
- from collections import defaultdict

---

### core\__init__.py

**External:**
- from .core_brain import CoreBrain
- from .query_processor import QueryProcessor
- from .response_generator import ResponseGenerator
- from .event_system import EventSystem
- from .background_coordinator import BackgroundCoordinator
- from .token_processor import TokenProcessor
- from .resource_manager import ResourceManager
- from .config_manager import ConfigManager
- from .unified_generator import UnifiedGenerator, create_unified_generator, ModelType
- from .pipeline_adapter import PipelineAdapter, create_pipeline_adapter

---

### core\api_compat.py

**External:**
- from functools import wraps
- from flask import jsonify, request

---

### core\async_pipeline.py

**Stdlib:**
- import time
- import logging
- import threading
- import os

**External:**
- import asyncio
- import uuid
- from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
- from dataclasses import dataclass, field
- from enum import Enum
- from concurrent.futures import ThreadPoolExecutor

---

### core\autopilot_cache.py

**Stdlib:**
- import os
- import json
- import time
- import threading

**External:**
- from __future__ import annotations
- from typing import Any, Optional, Dict

---

### core\background_coordinator.py

**Stdlib:**
- import threading
- import time
- import logging
- import os

**External:**
- from __future__ import annotations
- from collections import deque
- from typing import Dict, List, Optional, Type, Callable, Any, Deque
- import psutil

**Internal (eva_ai):**
- from eva_ai.core.event_bus import EventTypes
- from eva_ai.core.deferred_command_system import CommandPriority

---

### core\background_jobs\base_job.py

**Stdlib:**
- import logging

**External:**
- from __future__ import annotations
- from typing import Any

**Internal (eva_ai):**
- from eva_ai.core.deferred_command_system import CommandPriority

---

### core\background_jobs\module_recovery_job.py

**Stdlib:**
- import logging

**External:**
- from __future__ import annotations
- from typing import Any, Dict
- from .base_job import BaseJob, CommandPriority

---

### core\background_jobs\training_job.py

**Stdlib:**
- import logging
- import time

**External:**
- from __future__ import annotations
- from typing import Any
- from .base_job import BaseJob, CommandPriority

---

### core\background_jobs\web_index_job.py

**Stdlib:**
- import logging

**External:**
- from __future__ import annotations
- from typing import Any, Dict, List
- from .base_job import BaseJob, CommandPriority

---

### core\base_component.py

**Stdlib:**
- import logging
- import time
- import threading

**External:**
- from typing import Optional, Any, Dict, List, Type, TypeVar, Generic, Callable, Set
- from abc import ABC, abstractmethod
- from enum import Enum
- from .event_bus import EventBus, Event, EventTypes, get_event_bus
- from ..security.security_framework import get_security_manager
- from .event_bus import Event

---

### core\batch_wrapper.py

**Stdlib:**
- import time

**External:**
- from __future__ import annotations
- from dataclasses import dataclass, field, asdict
- from typing import Any, Dict, Optional, Tuple

---

### core\brain_components.py

**Stdlib:**
- import os
- import logging
- import time
- import os
- import os
- import torch

**External:**
- from typing import Any, Dict, List, Optional
- import sys
- from .config_manager import ConfigManager
- from .system_state import SystemStateManager, SystemState
- from .resource_manager import ResourceManager
- from .system_metrics import SystemMetricsManager
- from .feedback_processor import FeedbackProcessor
- from .self_learning_system import initialize_self_learning
- from .query_processor import QueryProcessor
- from .component_initializer import ComponentInitializer
- ... и ещё 6

**Internal (eva_ai):**
- from eva_ai.mlearning.language_filter import ModelModeController
- from eva_ai.learning.self_analyzer import SelfAnalyzer
- from eva_ai.learning.self_dialog_learning import SelfDialogLearningSystem
- from eva_ai.learning.performance_analyzer import PerformanceAnalyzer
- from eva_ai.knowledge.online_knowledge import OnlineKnowledgeAccess
- from eva_ai.memory.hybrid_token_cache import get_shared_cache
- from eva_ai.memory import get_shared_cache
- from eva_ai.mlearning.fractal_model_manager import FractalModelManager
- from eva_ai.mlearning.hot_deployment.llama_cpp_hot import LlamaCppHotDeployment
- from eva_ai.memory.unified_fractal_memory import UnifiedFractalMemory
- from eva_ai.core.hybrid_pipeline_adapter import HybridPipelineAdapter
- from eva_ai.core.recursive_model_pipeline import RecursiveModelPipeline
- from eva_ai.core.event_bus import Event, EventTypes
- from eva_ai.memory.unified_fractal_memory import UnifiedFractalMemory
- from eva_ai.core.hybrid_pipeline_adapter import HybridPipelineAdapter
- ... и ещё 5

---

### core\brain_config.py

**Stdlib:**
- import os
- import json
- import logging

**External:**
- import sys
- from typing import Dict, Any
- import psutil

---

### core\brain_coordination.py

**Stdlib:**
- import time
- import logging
- import logging

**External:**
- from typing import Dict, Any, Optional
- from .event_bus import Event, EventTypes, EventPriority
- from .deferred_command_system import CommandPriority

---

### core\brain_init.py

**Stdlib:**
- import os
- import time
- import logging

**External:**
- from typing import Any, Dict, Optional
- from .base_component import ComponentState

**Internal (eva_ai):**
- from eva_ai.generation.generation_coordinator import initialize_generation_coordinator
- from eva_ai.knowledge.wikipedia_kb import get_wikipedia_kb
- from eva_ai.knowledge.wikipedia_loader import get_wikipedia_loader
- from eva_ai.reasoning.integration import ReasoningIntegration
- from eva_ai.core.metrics import PerformanceMonitor
- from eva_ai.knowledge.graph_curator import GraphCurator
- from eva_ai.training.gguf_training_system import GGUFTrainingSystem

---

### core\brain_memory.py

**Stdlib:**
- import time
- import logging
- import torch

**External:**
- from typing import Dict, Any
- import psutil

---

### core\brain_memory_manager.py

**Stdlib:**
- import time
- import logging
- import threading
- import torch
- import torch

**External:**
- import gc
- from typing import Dict, Any, Optional
- import psutil

---

### core\brain_monitoring.py

**Stdlib:**
- import time
- import logging
- import torch

**External:**
- from typing import Dict, Any, Optional
- import psutil

---

### core\brain_query.py

**Stdlib:**
- import time
- import logging
- import threading

**External:**
- import re
- import random
- from typing import Dict, Any, Optional, List
- import re
- import random

**Internal (eva_ai):**
- from eva_ai.mlearning.qwen_model_manager import get_qwen_model_manager
- from eva_ai.mlearning.qwen_model_manager import get_qwen_model_manager
- from eva_ai.mlearning.qwen_model_manager import get_qwen_model_manager

---

### core\brain_state.py

**Stdlib:**
- import logging

**External:**
- from typing import Dict, Any, Optional
- from .system_state import SystemState, SystemStateManager
- from enum import Enum

---

### core\component_initializer.py

**Internal (eva_ai):**
- from eva_ai.core.init_core import ComponentInitializer, create_component_initializer
- from eva_ai.core.init_factories import (
- from eva_ai.core.init_connections import (
- from eva_ai.core.init_validation import (

---

### core\component_managers.py

**Stdlib:**
- import logging

**External:**
- from typing import Dict, Any, Optional, List, Tuple
- from datetime import datetime

---

### core\component_readiness.py

**Stdlib:**
- import logging
- import time

**External:**
- from typing import Dict, Any, List, Optional

---

### core\component_types.py

**External:**
- from dataclasses import dataclass, field
- from typing import List, Dict, Any, Optional
- from enum import Enum
- from datetime import datetime

---

### core\config_manager.py

**Stdlib:**
- import os
- import json
- import logging

**External:**
- from typing import Dict, Any, Optional

---

### core\context_chunking.py

**Stdlib:**
- import logging
- import time
- import time

**External:**
- import re
- from typing import List, Dict, Optional, Tuple
- from dataclasses import dataclass
- from collections import Counter
- from typing import Generator
- from typing import List

---

### core\context_first_policy.py

**Stdlib:**
- import logging

**External:**
- from __future__ import annotations
- from typing import Any, Dict, Optional
- import psutil

---

### core\contradiction_resolver.py

**Stdlib:**
- import os
- import time
- import json
- import logging
- import numpy

**External:**
- import sys
- import hashlib
- from typing import Dict, Any, Optional, List, Tuple
- import hashlib

---

### core\coordinator.py

**Stdlib:**
- import logging

**External:**
- from typing import Any, Dict, List, Optional

---

### core\core_brain.py

**Stdlib:**
- import os
- import logging
- import time
- import threading
- import logging

**External:**
- import queue
- import collections
- import weakref
- from typing import Dict, Any, Optional
- from .brain_config import load_brain_config, mask_secrets, ConfigMixin
- from .brain_components import ComponentMixin, _init_managers, _init_fractal_model, _init_llama_cpp, _init_two_model_pipeline, _init_unified_generator, _init_preprocessing, _init_qwen_config, _init_background, _init_mode_controller
- from .brain_init import _init_fractal_final, _init_gen_coord, _init_wikipedia, _init_reasoning, _init_performance_monitor, _start_post_init_services, _connect_components, _start_components, _stop_components
- from .brain_query import QueryMixin, FALLBACK_RESPONSES, FALLBACK_RESPONSE_DEFAULT
- from .brain_monitoring import MonitoringMixin
- from .brain_memory import MemoryMixin
- ... и ещё 10

**Internal (eva_ai):**
- from eva_ai.core.fractal_attention_system import FractalAttentionSystem

---

### core\core_brain_types.py

**External:**
- from enum import Enum
- from typing import Dict, Any, Optional

---

### core\cot_logger.py

**Stdlib:**
- import json
- import logging
- import time

**External:**
- from typing import Dict, Any, List, Optional
- from dataclasses import dataclass, asdict, field
- from datetime import datetime

---

### core\deferred_command_system.py

**Stdlib:**
- import time
- import logging
- import threading

**External:**
- import queue
- from typing import Dict, Any, Callable, List, Optional, Tuple
- from enum import Enum
- from dataclasses import dataclass
- from concurrent.futures import ThreadPoolExecutor, as_completed
- from .event_bus import Event, EventPriority

---

### core\deferred_commands.py

**Stdlib:**
- import logging
- import time
- import threading

**External:**
- from typing import Dict, Any, Callable, List, Optional
- from queue import Queue, Empty

---

### core\device_resolver.py

**Stdlib:**
- import torch

**External:**
- from __future__ import annotations
- from contextlib import contextmanager
- from dataclasses import dataclass
- from typing import Iterator, Literal, Optional

---

### core\engine_analysis.py

**Stdlib:**
- import time
- import logging

**External:**
- from typing import Dict, Any, Optional, List
- from .engine_steps import (

---

### core\engine_core.py

**Stdlib:**
- import time
- import logging

**External:**
- from typing import Dict, Any, Optional
- from .engine_steps import (
- from .engine_analysis import ReasoningAnalysisMixin
- from .engine_synthesis import ReasoningSynthesisMixin

---

### core\engine_steps.py

**Stdlib:**
- import time
- import logging

**External:**
- import re
- from typing import Dict, Any, Optional, List
- from enum import Enum as _Enum
- from dataclasses import dataclass, field

---

### core\engine_synthesis.py

**Stdlib:**
- import time
- import logging

**External:**
- from typing import Dict, Any, Optional, List
- from .engine_steps import (

---

### core\enhanced_self_learning.py

**Stdlib:**
- import os
- import time
- import json
- import logging
- import threading
- import numpy

**External:**
- import sys
- from typing import Dict, Any, Optional, List, Tuple
- from datetime import datetime, timedelta
- from dataclasses import dataclass, field
- from enum import Enum
- import re

---

### core\event_bus.py

**Stdlib:**
- import logging
- import time
- import threading

**External:**
- import weakref
- from typing import Dict, List, Callable, Any, Optional, Set
- from enum import Enum
- from dataclasses import dataclass
- from collections import defaultdict, deque
- import queue
- import traceback
- import traceback
- import traceback
- import traceback
- ... и ещё 1

---

### core\event_bus_bridge.py

**Stdlib:**
- import logging
- import time

**External:**
- from typing import Any, Dict, Optional
- from .event_bus import Event, EventPriority

---

### core\event_management.py

**Stdlib:**
- import os
- import time
- import json
- import logging

**External:**
- import sys
- from typing import Dict, Any, Optional, List, Tuple, Callable

---

### core\event_system.py

**Stdlib:**
- import logging
- import time
- import threading

**External:**
- import weakref
- from typing import Dict, List, Callable, Any, Deque, Optional
- from collections import defaultdict, deque
- from .event_bus_bridge import EventBusBridge
- from .event_bus import Event, EventPriority

---

### core\feedback_processor.py

**Stdlib:**
- import time
- import logging

**External:**
- from dataclasses import dataclass
- from typing import Dict, Any, List, Optional

**Internal (eva_ai):**
- from eva_ai.core.event_bus import Event, EventTypes

---

### core\fractal_attention_system.py

**Stdlib:**
- import os
- import time
- import json
- import logging
- import threading

**External:**
- import sys
- from typing import Dict, Any, Optional, List, Tuple
- from datetime import datetime
- from .self_dialog_manager import SelfDialogManager
- from .contradiction_resolver import ContradictionResolver
- from .learning_scheduler import LearningScheduler
- from .system_optimizer import SystemOptimizer

---

### core\fractal_pipeline.py

**Stdlib:**
- import time
- import logging
- import os

**External:**
- from typing import Dict, List, Any, Optional
- from dataclasses import dataclass

**Internal (eva_ai):**
- from eva_ai.memory.fractal_graph_v2 import FractalGraphV2
- from eva_ai.memory.fractal_graph_v2.eva_generator import (
- from eva_ai.memory.fractal_graph_v2.gguf_shadow import GGUFShadowProfiler

---

### core\generation_tracker.py

**Stdlib:**
- import time
- import threading
- import logging

**External:**
- import uuid
- from typing import Dict, Any, Optional, Callable
- from enum import Enum

**Internal (eva_ai):**
- from eva_ai.core.deferred_command_system import CommandPriority

---

### core\global_resource_queue.py

**Stdlib:**
- import os
- import time
- import json
- import logging
- import threading

**External:**
- import sys
- from typing import Dict, Any, Optional, List, Tuple

---

### core\graph_ml_core.py

**Stdlib:**
- import os
- import time
- import json
- import logging
- import threading
- import numpy
- import torch
- import numpy

**External:**
- import sys
- from typing import Dict, Any, Optional, List, Tuple, Set, TYPE_CHECKING
- from dataclasses import dataclass, field
- from collections import defaultdict
- from datetime import datetime
- from sentence_transformers import SentenceTransformer
- import re
- from collections import defaultdict

**Internal (eva_ai):**
- from eva_ai.fractal.entity_fractal_store import EntityFractalStore
- from eva_ai.mlearning.sentence_transformers_cache import get_sentence_transformer

---

### core\graph_ml_inference.py

**Stdlib:**
- import logging
- import numpy

**External:**
- from typing import Dict, Any, Optional, List

---

### core\graph_ml_patterns.py

**Stdlib:**
- import logging
- import numpy

**External:**
- from typing import Dict, Any, Optional, List, Tuple
- from collections import defaultdict
- from sklearn.cluster import KMeans

---

### core\graph_ml_training.py

**Stdlib:**
- import time
- import logging
- import numpy

**External:**
- from typing import Dict, Any, Optional, List
- from collections import defaultdict
- from .graph_ml_core import GraphPattern

---

### core\hardware_optimizations.py

**Stdlib:**
- import os
- import logging
- import torch
- import torch
- import torch
- import torch
- import torch

**External:**
- from typing import Optional
- from ..utils.memory_info import memory_info

---

### core\hybrid_pipeline_adapter.py

**Stdlib:**
- import os
- import time
- import logging

**External:**
- from typing import Dict, List, Any, Optional
- from llama_cpp import Llama
- from llama_cpp import LogitsProcessorList

**Internal (eva_ai):**
- from eva_ai.core.fractal_pipeline import FractalPipeline
- from eva_ai.memory.fractal_graph_v2.dual_generator import DualGenerator
- from eva_ai.core.recursive_model_pipeline import RecursiveModelPipeline
- from eva_ai.memory.fractal_graph_v2 import (

---

### core\hybrid_token_cache.py

**Stdlib:**
- import os
- import time
- import json
- import logging
- import threading
- import numpy

**External:**
- import sys
- from typing import Dict, Any, Optional, List, Tuple

---

### core\init_connections.py

**Stdlib:**
- import logging

**External:**
- from typing import Any, Dict, List, Optional, Tuple

---

### core\init_core.py

**Stdlib:**
- import os
- import logging
- import time

**External:**
- import sys
- from typing import Dict, Any, List, Set, Optional, Callable, Tuple

**Internal (eva_ai):**
- from eva_ai.core.init_connections import define_dependencies
- from eva_ai.core.init_factories import register_all_factories
- from eva_ai.core.init_connections import validate_dependencies
- from eva_ai.core.event_bus import EventTypes, Event
- from eva_ai.core.init_connections import post_initialize_connections as _setup
- from eva_ai.core.init_validation import get_initialization_status
- from eva_ai.core.init_validation import retry_failed_components
- from eva_ai.core.init_validation import get_component_health
- from eva_ai.core.init_validation import get_all_component_health

---

### core\init_factories.py

**Stdlib:**
- import os
- import logging

**External:**
- import sys

**Internal (eva_ai):**
- from eva_ai.core.event_bus import EventBus
- from eva_ai.core.resource_manager import ResourceManager
- from eva_ai.core.config_manager import ConfigManager
- from eva_ai.memory.memory_manager import MemoryManager
- from eva_ai.memory.hybrid_token_cache import get_shared_cache
- from eva_ai.memory import get_shared_cache
- from eva_ai.knowledge.qwen_api_enhancer import QwenAPIEnhancer
- from eva_ai.mlearning.unified_text_processor import UnifiedTextProcessor
- from eva_ai.mlearning.ml_unit import MLUnit
- from eva_ai.mlearning.hybrid_model_manager import HybridModelManager
- from eva_ai.core.query_processor import QueryProcessor
- from eva_ai.core.response_generator import ResponseGenerator
- from eva_ai.core.reasoning_engine import ReasoningEngine
- from eva_ai.analytics.analytics_manager import AnalyticsManager
- from eva_ai.monitoring.system_monitor import SystemMonitor
- ... и ещё 16

---

### core\init_validation.py

**Stdlib:**
- import logging

**External:**
- from typing import Any, Dict, List, Optional

**Internal (eva_ai):**
- from eva_ai.core.init_connections import check_dependencies

---

### core\integration_adapters.py

**Stdlib:**
- import logging
- import time

**External:**
- from typing import Dict, Any, Optional, List

---

### core\integration_core.py

**Stdlib:**
- import logging
- import os
- import time
- import threading

**External:**
- from typing import Dict, Any, Optional, List
- from concurrent.futures import ThreadPoolExecutor, as_completed
- from .event_system import EventBus
- from .core_brain import CoreBrain
- from .fractal_attention_system import FractalAttentionSystem
- from .self_dialog_manager import SelfDialogManager
- from .contradiction_resolver import ContradictionResolver
- from .learning_scheduler import LearningScheduler
- from .system_optimizer import SystemOptimizer
- from .response_generator import ResponseGenerator
- ... и ещё 4

**Internal (eva_ai):**
- from eva_ai.generation.generation_coordinator import GenerationCoordinator
- from eva_ai.generation.generation_coordinator import UnifiedGenerationCoordinator as GenerationCoordinator
- from eva_ai.memory.memory_manager import MemoryManager
- from eva_ai.memory.fractal_graph_v2 import FractalGraphV2
- from eva_ai.memory.memory_manager import MemoryManager

---

### core\integration_events.py

**Stdlib:**
- import logging

**External:**
- from typing import Dict, Any

---

### core\integration_layer.py

**External:**
- from .integration_core import ЕВАIntegrator, IntegrationLayer
- from .integration_adapters import (
- from .integration_events import (
- from .integration_sync import (

---

### core\integration_sync.py

**Stdlib:**
- import logging
- import time

**External:**
- from typing import Dict, Any

---

### core\integration_types.py

**External:**
- from dataclasses import dataclass, field
- from typing import List, Dict, Any, Optional
- from enum import Enum

---

### core\knowledge_rollback.py

**Stdlib:**
- import time
- import logging

**External:**
- from typing import Dict, Any, List, Optional

**Internal (eva_ai):**
- from eva_ai.core.event_bus import Event, EventPriority

---

### core\learning_scheduler.py

**Stdlib:**
- import os
- import time
- import json
- import logging

**External:**
- import sys
- from typing import Dict, Any, Optional, List, Tuple

---

### core\memory_graph_ml.py

**External:**
- from .graph_ml_core import (
- from .graph_ml_training import (
- from .graph_ml_inference import (
- from .graph_ml_patterns import (

---

### core\memory_initializer.py

**Stdlib:**
- import os
- import logging

**External:**
- from typing import Optional, Dict, Any

**Internal (eva_ai):**
- from eva_ai.memory.memory_manager import MemoryManager

---

### core\metrics.py

**Stdlib:**
- import time
- import threading
- import logging

**External:**
- import statistics
- from typing import Dict, List, Any, Optional, Callable
- from dataclasses import dataclass, field
- from collections import deque, defaultdict
- from contextlib import contextmanager
- import psutil

---

### core\metrics_collector.py

**Stdlib:**
- import logging
- import time

**External:**
- from typing import Dict, Any, List, Optional
- from datetime import datetime

---

### core\opportunities\base_detector.py

**Stdlib:**
- import time

**External:**
- from __future__ import annotations
- from typing import List, Dict, Any

---

### core\opportunities\learning_detector.py

**External:**
- from __future__ import annotations
- from typing import List, Dict, Any
- from .base_detector import BaseDetector

---

### core\opportunities\recovery_detector.py

**External:**
- from __future__ import annotations
- from typing import List, Dict, Any
- from .base_detector import BaseDetector

---

### core\opportunities\web_discovery_detector.py

**Stdlib:**
- import logging

**External:**
- from __future__ import annotations
- from typing import List, Dict, Any
- from .base_detector import BaseDetector

---

### core\pie_adapters\config.py

**Stdlib:**
- import json

**External:**
- import yaml
- from pathlib import Path
- from typing import Dict, Any, Optional, List
- from dataclasses import dataclass, field, asdict

---

### core\pie_adapters\container.py

**Stdlib:**
- import os
- import json
- import logging

**External:**
- import shutil
- import hashlib
- import tempfile
- from pathlib import Path
- from typing import Dict, Optional, Any, Union
- from dataclasses import dataclass
- from datetime import datetime
- import tarfile
- import zipfile
- import tarfile
- ... и ещё 1

---

### core\pie_adapters\model.py

**Stdlib:**
- import logging
- import time
- import numpy
- import json

**External:**
- from pathlib import Path
- from typing import Dict, Any, List, Optional, Union, Iterator
- from ..core.container import EvaContainer
- from ..core.config import EvaConfig
- from ..backends import BaseBackend, create_backend, GenerationConfig, GenerationResult
- from ..embeddings.layer_wise import LayerWiseEmbedder, LayerConfig
- from memory.fractal_graph_l1_l2 import FractalGraphL1L2, create_l1l2_graph
- from profiles.activation_profiler import ActivationProfiler, create_default_profiler
- from routing.routing_engine import RoutingEngine, create_default_engine, RoutingParams
- import sys
- ... и ещё 5

---

### core\pie_fallback.py

**Stdlib:**
- import time
- import logging

**External:**
- from typing import Dict, List, Optional, Any, Union
- from dataclasses import dataclass
- import tempfile

**Internal (eva_ai):**
- from eva_ai.core.pie_fallback import PieFallbackPipeline
- from eva_ai.memory.pie_integration import (

---

### core\pie_model_paths.py

**External:**
- from pathlib import Path
- from typing import Dict, Optional

---

### core\pipeline_adapter.py

**Stdlib:**
- import logging

**External:**
- from typing import Dict, Any, Optional
- from .unified_generator import ModelType
- import traceback
- from pathlib import Path
- from .unified_generator import UnifiedGenerator
- import traceback

---

### core\pipeline_adaptive.py

**Stdlib:**
- import logging

**External:**
- import math
- from typing import Dict, Any, List, Optional

**Internal (eva_ai):**
- from eva_ai.mlearning.sentence_transformers_cache import get_sentence_transformer

---

### core\pipeline_core.py

**Stdlib:**
- import os
- import logging
- import json
- import time
- import threading
- import torch

**External:**
- import re
- import atexit
- from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
- from typing import Dict, Any, List, Optional
- from llama_cpp import Llama
- from .pipeline_adaptive import AdaptiveParameterController
- from .pipeline_quality import (
- from .pipeline_models import (
- from .resource_manager import ResourceManager
- from .contradiction_resolver import ContradictionResolver
- ... и ещё 3

**Internal (eva_ai):**
- from eva_ai.ethics.ethics_core import EthicsFramework

---

### core\pipeline_models.py

**Stdlib:**
- import os
- import logging
- import time
- import torch

**External:**
- import re
- from typing import Dict, Any, Optional
- from llama_cpp import Llama
- from .text_chunker import TextChunker, MAX_INPUT_TOKENS_MODEL_A, MAX_INPUT_TOKENS_MODEL_B
- from .pipeline_quality import check_quality, _sanitize_response, _clean_filler_start, _remove_looping_blocks, check_russian_quality
- import gc

---

### core\pipeline_quality.py

**Stdlib:**
- import logging
- import time

**External:**
- import re
- from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
- from typing import Dict, Any, Optional
- import atexit

---

### core\proactive_fallback.py

**Stdlib:**
- import logging
- import numpy

**External:**
- import re
- from typing import Dict, Any, Optional, List
- from dataclasses import dataclass, field

---

### core\processor_core.py

**Stdlib:**
- import time
- import logging
- import torch

**External:**
- import hashlib
- import re
- from typing import Dict, Any, Optional, List
- from concurrent.futures import ThreadPoolExecutor, as_completed
- from collections import OrderedDict

**Internal (eva_ai):**
- from eva_ai.knowledge.context_entity import EntityExtractor
- from eva_ai.knowledge.ambiguity_resolver import AmbiguityResolver

---

### core\query_processor.py

**External:**
- from .processor_core import QueryProcessor

---

### core\query_router.py

**Stdlib:**
- import logging

**External:**
- import re
- from typing import Dict, Optional, List
- from dataclasses import dataclass

---

### core\real_self_learning.py

**Stdlib:**
- import os
- import torch
- import logging
- import threading
- import time

**External:**
- import sys
- from pathlib import Path
- from typing import Dict, Any, Optional, List, Tuple
- from datetime import datetime, timedelta

---

### core\reasoning_engine.py

**External:**
- from .engine_core import ReasoningEngine, create_reasoning_engine
- from .engine_steps import ReasoningPhase, ReasoningStep, InternalDialogue

---

### core\reasoning_types.py

**External:**
- from dataclasses import dataclass, field
- from typing import List, Dict, Any, Optional
- from enum import Enum

---

### core\recursive_model_pipeline.py

**External:**
- from .pipeline_core import RecursiveModelPipeline, create_recursive_pipeline
- from .pipeline_adaptive import AdaptiveParameterController
- from .pipeline_quality import (
- from .pipeline_models import (

---

### core\resource_manager.py

**Stdlib:**
- import os
- import threading
- import time
- import logging
- import torch
- import torch
- import torch

**External:**
- import psutil
- from typing import Dict, Any, Optional, List
- import gc

---

### core\response_generator.py

**Stdlib:**
- import os
- import time
- import threading
- import logging
- import torch
- import json

**External:**
- import sys
- from typing import Dict, Any, Optional, List, Tuple, Union
- from transformers import AutoTokenizer, PreTrainedTokenizer
- from transformers import AutoTokenizer
- import gc

**Internal (eva_ai):**
- from eva_ai.memory.hybrid_token_cache import HybridTokenCache
- from eva_ai.core.event_system import EventBus, ComponentInitializationManager
- from eva_ai.knowledge.context_entity import EntityExtractor
- from eva_ai.learning.knowledge_awareness import KnowledgeAwareness
- from eva_ai.core.unified_cache_bridge import create_unified_bridge

---

### core\response_types.py

**External:**
- from dataclasses import dataclass, field
- from typing import List, Dict, Any, Optional
- from enum import Enum

---

### core\self_dialog_manager.py

**Stdlib:**
- import os
- import time
- import json
- import logging
- import threading

**External:**
- import sys
- from typing import Dict, Any, Optional, List, Tuple

---

### core\self_evaluation.py

**Stdlib:**
- import logging

**External:**
- from typing import Dict, Any, Optional
- from dataclasses import dataclass

---

### core\self_learning_system.py

**Stdlib:**
- import os
- import torch
- import logging
- import threading
- import time

**External:**
- import sys
- from pathlib import Path
- from typing import Dict, Any, Optional, List, Tuple
- from datetime import datetime, timedelta

---

### core\system_metrics.py

**Stdlib:**
- import time
- import json

**External:**
- from typing import Dict, Any, List, Optional, Tuple
- import psutil

---

### core\system_optimizer.py

**Stdlib:**
- import os
- import time
- import json
- import logging
- import threading

**External:**
- import sys
- import psutil
- from typing import Dict, Any, Optional, List, Tuple

---

### core\system_state.py

**Stdlib:**
- import time
- import threading
- import logging

**External:**
- from enum import Enum
- from typing import Dict, Any, Optional, List, Set
- from dataclasses import dataclass
- from .event_bus import EventBus, Event, EventTypes, get_event_bus
- from .base_component import ComponentState

---

### core\task_types.py

**External:**
- from dataclasses import dataclass, field
- from typing import List, Dict, Any, Optional
- from enum import Enum
- from datetime import datetime

---

### core\text_chunker.py

**Stdlib:**
- import logging

**External:**
- from typing import List, Dict, Any, Optional
- import re

---

### core\token_processor.py

**Stdlib:**
- import logging

**External:**
- import hashlib
- import re
- from typing import List, Dict, Any, Optional

---

### core\unified_cache_bridge.py

**Stdlib:**
- import os
- import time
- import json
- import logging
- import threading

**External:**
- import hashlib
- from typing import Dict, List, Optional, Any, Tuple
- from collections import OrderedDict

---

### core\unified_generator.py

**Stdlib:**
- import time
- import logging
- import os
- import threading
- import time

**External:**
- from typing import Dict, List, Optional, Any, Tuple, Generator
- from dataclasses import dataclass
- from pathlib import Path
- from enum import Enum
- from .context_chunking import ChunkedContextProcessor, StreamingGenerator, ContextChunk
- from llama_cpp import Llama
- import hashlib

**Internal (eva_ai):**
- from eva_ai.core.pie_model_paths import get_pie_model_path
- from eva_ai.memory.fractal_graph_v2 import get_fractal_graph
- from eva_ai.memory.fractal_graph_v2 import get_fractal_graph
- from eva_ai.core.brain_query import needs_web_search
- from eva_ai.websearch.web_search_integrated import get_web_search_engine

---

### core\utils.py

**Stdlib:**
- import logging
- import os

**External:**
- import sys
- import io
- from datetime import datetime
- from logging.handlers import RotatingFileHandler

---

### distributed\__init__.py

**External:**
- from .distributed_system import DistributedSystem
- from .cluster_manager import ClusterManager
- from .distributed_task_scheduler import TaskScheduler, SimpleTaskScheduler
- from .knowledge_sync import KnowledgeSync
- from .distributed_recovery_manager import RecoveryManager

---

### distributed\cluster_manager.py

**Stdlib:**
- import os
- import logging
- import time
- import threading
- import json
- import requests
- import sqlite3

**External:**
- import socket
- from typing import Dict, List, Optional, Any, Tuple
- from datetime import datetime, timedelta
- from collections import defaultdict
- import random
- import hashlib

---

### distributed\database_utils.py

**Stdlib:**
- import sqlite3
- import logging

---

### distributed\distributed_recovery_manager.py

**Stdlib:**
- import os
- import logging
- import json
- import time
- import threading

**External:**
- from typing import Dict, List, Optional, Any, Tuple
- from datetime import datetime, timedelta
- from .cluster_manager import ClusterNode

---

### distributed\distributed_system.py

**Stdlib:**
- import os
- import logging
- import time
- import threading
- import json
- import sqlite3
- import requests
- import threading

**External:**
- from typing import Dict, List, Optional, Any, Tuple, Callable
- from datetime import datetime, timedelta
- from collections import defaultdict
- from .distributed_task_scheduler import TaskScheduler
- from .knowledge_sync import KnowledgeSync
- from .distributed_recovery_manager import RecoveryManager

---

### distributed\distributed_task_scheduler.py

**Stdlib:**
- import logging
- import threading
- import time
- import os

**External:**
- from typing import Dict, List, Optional, Any, Callable

---

### distributed\distributed_tasks.py

**Stdlib:**
- import os
- import logging
- import time
- import threading
- import json
- import requests
- import numpy

**External:**
- import queue
- from .cluster_manager import ClusterNode
- from typing import Dict, List, Optional, Any, Tuple, Callable
- from datetime import datetime, timedelta
- import random
- import hashlib
- import matplotlib.pyplot
- from matplotlib.figure import Figure
- from matplotlib.backends.backend_agg import FigureCanvasAgg
- import base64
- ... и ещё 11

---

### distributed\distributed_types.py

**External:**
- from dataclasses import dataclass, field
- from typing import List, Dict, Any, Optional
- from enum import Enum

---

### distributed\knowledge_sync.py

**Stdlib:**
- import os
- import logging
- import time
- import threading
- import json
- import requests
- import sqlite3

**External:**
- import random
- from typing import Dict, List, Optional, Any, Tuple
- from datetime import datetime, timedelta
- from collections import defaultdict

---

### ethics\__init__.py

**External:**
- from .ethics_framework import EthicsFramework

---

### ethics\ethical_situations.py

**Stdlib:**
- import os
- import logging
- import json
- import time
- import numpy

**External:**
- import hashlib
- import base64
- from typing import Dict, List, Optional, Any, Tuple
- from collections import defaultdict
- from io import BytesIO
- from matplotlib.figure import Figure
- from matplotlib.backends.backend_agg import FigureCanvasAgg
- import matplotlib.pyplot
- from .situations_db import SituationsDBMixin, EthicalIssue
- from .situations_scenarios import SituationsScenariosMixin, EthicalAssessment, EthicalPrinciple, EthicalDecision
- ... и ещё 1

**Internal (eva_ai):**
- from eva_ai.ethics.ethics_framework import EthicalDecision as FrameworkEthicalDecision, EthicalIssue as FrameworkEthicalIssue
- from eva_ai.ethics.principles_manager import PrinciplesManager
- from eva_ai.ethics.risk_assessment import RiskAssessor

---

### ethics\ethics_core.py

**Stdlib:**
- import os
- import logging
- import time
- import threading
- import json

**External:**
- from typing import Dict, List, Optional, Any, Tuple, Callable

**Internal (eva_ai):**
- from eva_ai.ethics.ethics_framework import EthicalDecision, EthicalIssue
- from eva_ai.ethics.principles_manager import PrinciplesManager
- from eva_ai.ethics.risk_assessment import RiskAssessor
- from eva_ai.ethics.ethical_situations import EthicalSituationHandler

---

### ethics\ethics_framework.py

**External:**
- from .framework_core import EthicsFramework
- from .framework_principles import EthicalPrinciple, EthicsPrinciplesMixin
- from .framework_checks import EthicalDecision, EthicalAssessment, EthicsAnalysisResult, EthicsChecksMixin
- from .framework_violations import EthicsViolationsMixin
- from .situations_db import EthicalIssue

---

### ethics\ethics_integrated.py

**Stdlib:**
- import logging
- import time
- import os
- import json
- import json

**External:**
- from typing import Dict, List, Optional, Any, Tuple
- from datetime import datetime

**Internal (eva_ai):**
- from eva_ai.core.base_component import BaseComponent, ComponentState
- from eva_ai.core.event_bus import get_event_bus, Event, EventTypes
- from eva_ai.ethics.ethics_core import EthicsFramework

---

### ethics\framework_checks.py

**Stdlib:**
- import logging
- import time

**External:**
- import re
- from typing import Dict, List, Optional, Any
- from dataclasses import dataclass, field
- from .framework_principles import EthicalPrinciple
- from .framework_violations import EthicalDecision

---

### ethics\framework_core.py

**Stdlib:**
- import os
- import logging
- import time
- import threading
- import threading

**External:**
- from typing import Dict, List, Optional, Any
- from .framework_principles import EthicalPrinciple, EthicsPrinciplesMixin
- from .framework_checks import EthicalDecision, EthicalAssessment, EthicsAnalysisResult, EthicsChecksMixin
- from .framework_violations import EthicsViolationsMixin

---

### ethics\framework_principles.py

**Stdlib:**
- import os
- import logging
- import time
- import json

**External:**
- from typing import Dict, Any, Optional
- from dataclasses import dataclass, field

---

### ethics\framework_violations.py

**Stdlib:**
- import os
- import logging
- import time
- import json

**External:**
- from typing import Dict, List, Optional, Any
- from dataclasses import dataclass, field
- from collections import defaultdict
- from .violation_id_manager import (
- from .framework_principles import EthicalPrinciple

---

### ethics\principles_manager.py

**Stdlib:**
- import os
- import logging
- import json
- import sqlite3
- import time
- import threading

**External:**
- import base64
- from typing import Dict, List, Optional, Any, Callable, Tuple
- from datetime import datetime, timedelta
- from collections import defaultdict
- from io import BytesIO
- import hashlib
- import matplotlib
- import matplotlib.pyplot
- from matplotlib.figure import Figure
- from matplotlib.backends.backend_agg import FigureCanvasAgg

**Internal (eva_ai):**
- from eva_ai.ethics.ethics_framework import EthicalPrinciple

---

### ethics\risk_assessment.py

**Stdlib:**
- import os
- import logging
- import json
- import time
- import numpy

**External:**
- import re
- import base64
- from typing import Dict, List, Optional, Any, Tuple
- from collections import defaultdict
- from io import BytesIO
- from .principles_manager import PrinciplesManager
- from sklearn.feature_extraction.text import TfidfVectorizer
- from sklearn.metrics.pairwise import cosine_similarity
- import matplotlib
- import matplotlib.pyplot
- ... и ещё 2

**Internal (eva_ai):**
- from eva_ai.ethics.ethics_framework import EthicalAssessment, EthicalPrinciple

---

### ethics\situations_db.py

**Stdlib:**
- import os
- import logging
- import json
- import time

**External:**
- from typing import Dict, List, Optional, Any

---

### ethics\situations_evaluation.py

**Stdlib:**
- import os
- import logging
- import json
- import time
- import numpy

**External:**
- from typing import Dict, List, Optional, Any
- from io import BytesIO
- from matplotlib.figure import Figure
- from matplotlib.backends.backend_agg import FigureCanvasAgg
- import matplotlib.pyplot
- import base64
- from .situations_db import EthicalIssue

---

### ethics\situations_scenarios.py

**Stdlib:**
- import os
- import logging
- import json
- import time

**External:**
- import hashlib
- from typing import Dict, List, Optional, Any, Tuple
- from collections import defaultdict

---

### ethics\violation_id_manager.py

**Stdlib:**
- import time
- import logging
- import os

**External:**
- import hashlib
- import re
- from typing import List
- from typing import Dict, Optional, Tuple, NamedTuple

---

### fractal\__init__.py

**Internal (eva_ai):**
- from eva_ai.fractal.fractal_store import FractalStore

---

### fractal\entity_fractal_store.py

**Stdlib:**
- import time
- import logging
- import os
- import json
- import numpy

**External:**
- from __future__ import annotations
- from pathlib import Path
- from dataclasses import dataclass, field
- from typing import Dict, Any, Optional, List, Tuple
- from collections import defaultdict
- import re

---

### fractal\fractal_store.py

**Stdlib:**
- import time
- import logging
- import os
- import json
- import numpy
- import torch
- import torch.nn

**External:**
- from __future__ import annotations
- from pathlib import Path
- import hashlib
- import gc
- import math
- import sys
- import shutil
- from dataclasses import dataclass, field
- from typing import Any, Dict, Deque, Tuple, List, Optional, Iterable, Set
- from collections import deque, OrderedDict, defaultdict
- ... и ещё 2

---

### generation\__init__.py

**External:**
- from .generation_coordinator import UnifiedGenerationCoordinator

**Internal (eva_ai):**
- from eva_ai.memory.hybrid_token_cache import HybridTokenCache
- from eva_ai.memory import HybridTokenCache
- from eva_ai.mlearning.parallel_tokenization import ParallelTokenizer

---

### generation\generation_coordinator.py

**Stdlib:**
- import time
- import logging

**External:**
- from typing import Dict, Any, Optional, Union
- from abc import ABC, abstractmethod

---

### gui\__init__.py

**Stdlib:**
- import logging

**External:**
- from .core_gui import create_gui as _create_gui
- from .core_gui import ЕВАGUI
- from .chat_module import ChatModule
- from .memory_module import MemoryModule
- from .knowledge_graph_module import KnowledgeGraphModule
- from .contradiction_module import ContradictionModule
- from .analytics_module import AnalyticsModule
- from .learning_module import LearningModule
- from .neuromorphic_module import NeuromorphicModule
- from .settings_module import SettingsModule
- ... и ещё 1

---

### gui\analytics_module.py

**Stdlib:**
- import logging
- import numpy
- import time
- import json
- import os
- import threading

**External:**
- import tkinter
- from tkinter import ttk, messagebox, filedialog, scrolledtext
- import matplotlib.pyplot
- from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
- import datetime
- from datetime import timedelta
- from typing import Dict, List, Tuple, Optional, Any
- import queue
- import base64
- from io import BytesIO

---

### gui\analytics_types.py

**External:**
- from dataclasses import dataclass, field
- from typing import List, Dict, Any, Optional
- from enum import Enum

---

### gui\base_gui.py

**Stdlib:**
- import logging

**External:**
- from typing import Optional, Dict, Any, Callable
- import tkinter
- from tkinter import ttk

---

### gui\chat_actions.py

**Stdlib:**
- import threading
- import logging
- import os

**External:**
- import tkinter
- from tkinter import ttk, Menu, messagebox
- from typing import Optional
- import webbrowser

---

### gui\chat_history.py

**Stdlib:**
- import os
- import json
- import logging

**External:**
- from typing import Any, Dict

---

### gui\chat_input.py

**Stdlib:**
- import logging
- import time

**External:**
- import tkinter
- from tkinter import ttk, scrolledtext, Menu
- import random

---

### gui\chat_messages.py

**Stdlib:**
- import logging
- import time
- import json

**External:**
- import tkinter
- import re
- from datetime import datetime
- from typing import Dict, List, Optional, Any

**Internal (eva_ai):**
- from eva_ai.gui.chat_text_utils import _to_display_str, _fix_mojibake

---

### gui\chat_module.py

**Stdlib:**
- import logging
- import time
- import json
- import os
- import threading

**External:**
- import tkinter
- from tkinter import ttk, scrolledtext, Menu, messagebox, font, filedialog
- import webbrowser
- import re
- import random
- import queue
- from datetime import datetime
- from typing import Dict, List, Optional, Any, Tuple, Callable

**Internal (eva_ai):**
- from eva_ai.tools.import_pipeline import ImportPipeline
- from eva_ai.gui.chat_messages import ChatMessagesMixin
- from eva_ai.gui.chat_input import ChatInputMixin
- from eva_ai.gui.chat_history import ChatHistoryMixin
- from eva_ai.gui.chat_actions import ChatActionsMixin
- from eva_ai.gui.chat_reasoning import ChatReasoningMixin
- from eva_ai.gui.chat_text_utils import _to_display_str, _fix_mojibake
- from eva_ai.tools.document_reader import DocumentTextReader
- from eva_ai.knowledge.knowledge_integrator import KnowledgeIntegrator

---

### gui\chat_reasoning.py

**Stdlib:**
- import logging

**External:**
- import tkinter
- from tkinter import ttk, scrolledtext
- from typing import Optional

**Internal (eva_ai):**
- from eva_ai.gui.chat_text_utils import _fix_mojibake

---

### gui\chat_text_utils.py

**Stdlib:**
- import logging

**External:**
- from typing import Any

---

### gui\contradiction_module.py

**Stdlib:**
- import logging

**External:**
- import tkinter
- from tkinter import ttk, simpledialog
- from typing import Dict, Any, Optional, List

---

### gui\core\__init__.py

**Stdlib:**
- import logging
- import os

**External:**
- from pathlib import Path
- from typing import Optional, Dict, Any
- import sys
- from ..core_gui import ЕВАGUI, create_gui

---

### gui\core_gui.py

**Stdlib:**
- import os
- import logging
- import threading
- import time
- import json

**External:**
- import sys
- import queue
- from datetime import datetime
- from typing import Dict, Any, Optional, List
- import tkinter
- from tkinter import ttk, messagebox, filedialog
- import matplotlib
- from .settings import load_settings, save_settings
- from .gui_main import MainWindowMixin
- from .gui_tabs import TabManagerMixin, MemoryTab, SystemTab
- ... и ещё 2

---

### gui\gui_events.py

**Stdlib:**
- import os
- import logging
- import threading
- import time
- import json

**External:**
- import queue
- from datetime import datetime
- from typing import Dict, Any, Optional
- import tkinter
- from tkinter import messagebox

---

### gui\gui_graph_types.py

**External:**
- from dataclasses import dataclass, field
- from typing import List, Dict, Any, Optional
- from enum import Enum

---

### gui\gui_main.py

**Stdlib:**
- import os
- import logging
- import threading
- import time
- import json

**External:**
- import queue
- from datetime import datetime
- from typing import Dict, Any, Optional, List
- import tkinter
- from tkinter import ttk, messagebox
- import matplotlib
- from .settings import load_settings, save_settings

---

### gui\gui_modules.py

**Stdlib:**
- import logging

**External:**
- import tkinter
- from tkinter import ttk
- from .widgets import create_rounded_button
- from .chat_module import ChatModule
- from .analytics_module import AnalyticsModule
- from .knowledge_graph_module import KnowledgeGraphModule
- from .contradiction_module import ContradictionModule
- from .memory_module import MemoryModule
- from .learning_module import LearningModule
- from .settings_module import SettingsModule
- ... и ещё 1

---

### gui\gui_status.py

**Stdlib:**
- import logging
- import time

**External:**
- from typing import Dict, Any
- import tkinter
- from tkinter import ttk
- from datetime import datetime
- from tkinter import messagebox
- from tkinter import messagebox
- from tkinter import messagebox

---

### gui\gui_tabs.py

**Stdlib:**
- import logging

**External:**
- import tkinter
- from tkinter import ttk
- from .chat_module import ChatModule
- from tkinter import ttk

---

### gui\gui_themes.py

**External:**
- import tkinter.ttk

---

### gui\gui_types.py

**External:**
- from dataclasses import dataclass, field
- from typing import List, Dict, Any, Optional
- from datetime import datetime
- from enum import Enum

---

### gui\gui_util.py

**External:**
- import tkinter
- from tkinter import ttk

---

### gui\gui_utils.py

**Stdlib:**
- import json
- import os
- import logging

**External:**
- import queue
- import tkinter
- from tkinter import ttk
- from datetime import datetime
- from .gui_modules import switch_view
- from .gui_widgets import create_rounded_button

---

### gui\gui_widgets.py

**External:**
- import tkinter
- import datetime
- from tkinter import ttk
- from .widgets import create_rounded_button
- from .gui_modules import switch_view

---

### gui\kg_actions.py

**Stdlib:**
- import logging
- import time
- import json
- import os

**External:**
- import tkinter
- from tkinter import ttk, messagebox, filedialog

---

### gui\kg_nodes.py

**Stdlib:**
- import logging

**External:**
- import tkinter
- from tkinter import ttk, messagebox
- from tkinter import scrolledtext
- from datetime import datetime
- import networkx
- import matplotlib.pyplot
- from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
- from tkinter import scrolledtext

---

### gui\kg_search.py

**Stdlib:**
- import logging
- import time
- import json
- import os

**External:**
- import tkinter
- from tkinter import ttk, messagebox, filedialog
- from tkinter import Menu

---

### gui\kg_stats.py

**Stdlib:**
- import logging
- import time

**External:**
- import tkinter
- from tkinter import ttk, messagebox
- from datetime import datetime
- import matplotlib.pyplot

---

### gui\kg_visualization.py

**Stdlib:**
- import logging
- import numpy

**External:**
- import tkinter
- from tkinter import ttk, messagebox
- from tkinter import Menu
- import matplotlib.pyplot
- import networkx
- from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

---

### gui\knowledge_graph_module.py

**Stdlib:**
- import logging
- import numpy
- import time
- import json
- import os
- import threading

**External:**
- import tkinter
- from tkinter import ttk, messagebox, filedialog, scrolledtext, simpledialog, font
- import matplotlib.pyplot
- import networkx
- from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
- import queue
- import webbrowser
- import re
- from tkinter import Menu
- from datetime import datetime
- ... и ещё 1

**Internal (eva_ai):**
- from eva_ai.knowledge.knowledge_graph import KnowledgeGraph
- from eva_ai.gui.kg_visualization import (
- from eva_ai.gui.kg_search import (
- from eva_ai.gui.kg_stats import (
- from eva_ai.gui.kg_nodes import (
- from eva_ai.gui.kg_actions import (

---

### gui\learning_module.py

**Stdlib:**
- import logging
- import numpy
- import time
- import json
- import os
- import threading

**External:**
- import tkinter
- from tkinter import ttk, messagebox, filedialog, scrolledtext
- import matplotlib.pyplot
- from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
- from typing import Dict, List, Optional, Any, Union
- import re
- from datetime import datetime
- import uuid
- import tkinter.simpledialog
- import tkinter
- ... и ещё 1

**Internal (eva_ai):**
- from eva_ai.tools.import_pipeline import ImportPipeline
- from eva_ai.mlearning.text_quality_learning_integration import TextQualityLearningIntegration
- from eva_ai.mlearning.optimized_fractal_model_manager import OptimizedFractalModelManager

---

### gui\memory_module.py

**Stdlib:**
- import logging
- import json
- import numpy
- import os
- import time
- import threading

**External:**
- import tkinter
- from tkinter import ttk, messagebox, filedialog, simpledialog
- import matplotlib
- import matplotlib.pyplot
- from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
- from datetime import datetime, timedelta
- from typing import Dict, Any, List, Optional, Tuple, Union
- import queue
- import re

---

### gui\neuromorphic_module.py

**Stdlib:**
- import logging
- import time

**External:**
- import tkinter
- from tkinter import ttk
- from typing import Any, Dict, Optional, List

---

### gui\settings.py

**Stdlib:**
- import logging
- import json
- import os

**External:**
- from typing import Dict, Any

---

### gui\settings_module.py

**Stdlib:**
- import logging
- import json
- import os

**External:**
- import tkinter
- from tkinter import ttk, messagebox, filedialog
- from datetime import datetime
- import platform
- from tkinter import filedialog
- from tkinter import filedialog

---

### gui\web_gui\bridge.py

**Stdlib:**
- import logging
- import threading
- import time
- import json

**External:**
- from typing import Optional, Callable, Any, Dict
- import socket
- from .server import create_app
- from .server import create_app

**Internal (eva_ai):**
- from eva_ai.core.event_bus import EventTypes

---

### gui\web_gui\server.py

**Internal (eva_ai):**
- from eva_ai.gui.web_gui.server_main import (
- from eva_ai.gui.web_gui.server_auth import (
- from eva_ai.gui.web_gui.server_routes import (

---

### gui\web_gui\server_api_export.py

**Stdlib:**
- import logging
- import json

**External:**
- import csv
- import io
- from datetime import datetime
- from flask import jsonify, request

---

### gui\web_gui\server_api_knowledge.py

**Stdlib:**
- import logging

**External:**
- import uuid
- from datetime import datetime
- from flask import jsonify, request

---

### gui\web_gui\server_api_wikipedia.py

**Stdlib:**
- import logging
- import os
- import json
- import time
- import requests

**External:**
- from datetime import datetime, date
- from flask import jsonify, request

---

### gui\web_gui\server_auth.py

**Stdlib:**
- import os
- import logging
- import threading
- import json

**External:**
- import uuid
- import hashlib
- import secrets
- from datetime import datetime
- from typing import Dict, Any, Optional

---

### gui\web_gui\server_main.py

**Stdlib:**
- import os
- import time
- import logging
- import threading
- import json

**External:**
- import uuid
- import hashlib
- import secrets
- import socket
- from datetime import datetime
- from typing import Dict, Any, Optional
- from flask import Flask
- import click
- from werkzeug.serving import make_server

**Internal (eva_ai):**
- from eva_ai.gui.web_gui.server_auth import SessionManager, AuthManager, EntityExtractor, EthicsChecker
- from eva_ai.gui.web_gui.server_api_wikipedia import register_routes as register_wikipedia_routes
- from eva_ai.gui.web_gui.server_api_export import register_routes as register_export_routes
- from eva_ai.gui.web_gui.server_models import register_routes as register_model_routes
- from eva_ai.gui.web_gui.server_routes_graph import register_graph_routes
- from eva_ai.gui.web_gui.server_routes_core import register_core_routes
- from eva_ai.gui.web_gui.server_routes_chat import register_chat_routes
- from eva_ai.gui.web_gui.server_routes_auth import register_auth_routes
- from eva_ai.gui.web_gui.server_routes_analytics import register_analytics_routes
- from eva_ai.gui.web_gui.server_routes_knowledge import register_knowledge_routes
- from eva_ai.gui.web_gui.server_routes_upload import register_upload_routes
- from eva_ai.core.metrics import get_eva_metrics

---

### gui\web_gui\server_models.py

**Stdlib:**
- import os
- import logging

**External:**
- from flask import jsonify, request

---

### gui\web_gui\server_routes.py

**Stdlib:**
- import os
- import logging
- import json
- import time
- import threading
- import time
- import json

**External:**
- import uuid
- from datetime import datetime
- from flask import render_template, jsonify, request, abort, Response, stream_with_context
- import pytesseract
- import hashlib
- import traceback
- import re
- import psutil
- import psutil
- import traceback
- ... и ещё 14

**Internal (eva_ai):**
- from eva_ai.core.api_compat import API_VERSION, API_PREFIX, api_version
- from eva_ai.core.event_bus import EventTypes
- from eva_ai.core.metrics import get_metrics_registry
- from eva_ai.core.metrics import get_metrics_registry, get_eva_metrics
- from eva_ai.core.metrics import get_eva_metrics, get_metrics_registry

---

### gui\web_gui\server_routes_analytics.py

**Stdlib:**
- import os
- import logging
- import time
- import json
- import torch

**External:**
- from datetime import datetime
- from flask import jsonify, request, Response, stream_with_context
- import psutil
- import queue

**Internal (eva_ai):**
- from eva_ai.core.metrics import get_metrics_registry, get_eva_metrics

---

### gui\web_gui\server_routes_auth.py

**Stdlib:**
- import logging

**External:**
- from flask import jsonify, request

**Internal (eva_ai):**
- from eva_ai.core.api_compat import API_VERSION, API_PREFIX

---

### gui\web_gui\server_routes_backup.py

**Stdlib:**
- import os
- import logging
- import json
- import time
- import threading
- import time
- import json

**External:**
- import uuid
- from datetime import datetime
- from flask import render_template, jsonify, request, abort, Response, stream_with_context
- import pytesseract
- import hashlib
- import traceback
- import re
- import psutil
- import psutil
- import traceback
- ... и ещё 14

**Internal (eva_ai):**
- from eva_ai.core.api_compat import API_VERSION, API_PREFIX, api_version
- from eva_ai.core.event_bus import EventTypes
- from eva_ai.core.metrics import get_metrics_registry
- from eva_ai.core.metrics import get_metrics_registry, get_eva_metrics
- from eva_ai.core.metrics import get_eva_metrics, get_metrics_registry

---

### gui\web_gui\server_routes_chat.py

**Stdlib:**
- import json
- import logging
- import threading
- import time

**External:**
- from flask import jsonify, request, Response

**Internal (eva_ai):**
- from eva_ai.core.api_compat import API_VERSION, API_PREFIX, api_version

---

### gui\web_gui\server_routes_core.py

**Stdlib:**
- import logging
- import time
- import json
- import os
- import torch

**External:**
- from datetime import datetime
- from flask import render_template, jsonify, request, Response
- from .server_routes_utils import check_brain_initialized, get_brain_components
- import psutil

**Internal (eva_ai):**
- from eva_ai.core.api_compat import API_VERSION, API_PREFIX, api_version

---

### gui\web_gui\server_routes_graph.py

**Stdlib:**
- import logging

**External:**
- from flask import jsonify, request

---

### gui\web_gui\server_routes_knowledge.py

**Stdlib:**
- import os
- import logging
- import time
- import json

**External:**
- from datetime import datetime
- from flask import jsonify, request
- import glob
- import psutil

**Internal (eva_ai):**
- from eva_ai.core.api_compat import API_VERSION, API_PREFIX, api_version

---

### gui\web_gui\server_routes_upload.py

**Stdlib:**
- import os
- import logging

**External:**
- import uuid
- from flask import jsonify, request
- import pytesseract
- import fitz
- import pdfplumber
- import PyPDF2
- import pytesseract
- from PIL import Image
- import fitz
- from docx import Document
- ... и ещё 2

---

### gui\web_gui\server_routes_utils.py

**Stdlib:**
- import os
- import logging
- import json
- import time

**External:**
- import uuid
- from datetime import datetime
- from flask import request, jsonify
- import pytesseract
- import fitz
- import pdfplumber
- import PyPDF2
- import pytesseract
- from PIL import Image
- import fitz
- ... и ещё 3

**Internal (eva_ai):**
- from eva_ai.core.api_compat import API_VERSION, API_PREFIX, api_version

---

### gui\web_gui\server_types.py

**External:**
- from dataclasses import dataclass, field
- from typing import List, Dict, Any, Optional
- from enum import Enum

---

### gui\widgets.py

**Stdlib:**
- import logging

**External:**
- import tkinter
- from tkinter import ttk
- from typing import Dict, Any, Callable, Optional, List

---

### knowledge\__init__.py

**External:**
- from .kg_adapter import KnowledgeGraphAdapter
- from .knowledge_graph import KnowledgeGraph
- from .graph_curator import GraphCurator, create_graph_curator
- from .ambiguity_resolver import AmbiguityResolver
- from .context_entity import EntityExtractor
- from .wikipedia_kb import WikipediaKnowledgeBase, get_wikipedia_kb, clear_wikipedia_kb, get_wikipedia_loader
- from .knowledge_analytics import KnowledgeAnalytics
- from .qwen_api_enhancer import QwenAPIEnhancer
- from .concept_extractor import ConceptExtractor, Concept, create_concept_extractor
- from .concept_miner import ConceptMiner, ConceptStatus, PhantomCandidate, create_concept_miner

---

### knowledge\ambiguity_resolver.py

**Stdlib:**
- import logging

**External:**
- from typing import List, Dict, Any, Optional

---

### knowledge\concept_extractor.py

**Stdlib:**
- import time
- import logging

**External:**
- import re
- from typing import Dict, List, Any, Optional, Set
- from dataclasses import dataclass

---

### knowledge\concept_miner.py

**Stdlib:**
- import os
- import time
- import json
- import logging
- import threading
- import numpy

**External:**
- from typing import Dict, List, Any, Optional, Callable
- from dataclasses import dataclass, field, asdict
- from enum import Enum
- from datetime import datetime
- from collections import defaultdict
- from concurrent.futures import ThreadPoolExecutor
- import psutil

**Internal (eva_ai):**
- from eva_ai.core.deferred_command_system import CommandPriority

---

### knowledge\context_entity.py

**Stdlib:**
- import logging

**External:**
- from enum import Enum
- from typing import List, Dict, Any, Optional
- from dataclasses import dataclass

---

### knowledge\graph_curator.py

**Stdlib:**
- import logging
- import time
- import threading
- import numpy

**External:**
- from typing import Dict, List, Optional, Any, Set
- from enum import Enum
- from collections import defaultdict

---

### knowledge\kg_adapter.py

**Stdlib:**
- import logging

**External:**
- from typing import Dict, List, Any, Optional

---

### knowledge\knowledge_analytics.py

**Stdlib:**
- import logging

**External:**
- from typing import Dict, List, Any

---

### knowledge\knowledge_graph.py

**Stdlib:**
- import logging

**External:**
- from typing import Dict, List, Any, Optional
- from .kg_adapter import KnowledgeGraphAdapter

**Internal (eva_ai):**
- from eva_ai.memory.fractal_graph_v2 import FractalMemoryGraph

---

### knowledge\qwen_api_enhancer.py

**Stdlib:**
- import logging

**External:**
- from typing import Dict, Any, Optional

---

### knowledge\wikipedia_kb.py

**Stdlib:**
- import os
- import json
- import sqlite3
- import logging
- import threading
- import requests
- import requests
- import requests

**External:**
- import hashlib
- from typing import Optional, List, Dict, Any
- from datetime import datetime
- import re
- import math

**Internal (eva_ai):**
- from eva_ai.mlearning.sentence_transformers_cache import get_sentence_transformer

---

### knowledge\wikipedia_loader.py

**External:**
- from .wikipedia_kb import get_wikipedia_loader

---

### learning\__init__.py

**External:**
- from .learning_scheduler import LearningScheduler
- from .analyzer_core import AnalyzerCore
- from .learning_opportunity_manager import LearningOpportunityManager
- from .learning_opportunity import LearningOpportunity
- from .learning_manager import LearningManager
- from .self_analyzer import SelfAnalyzer
- from .self_dialog_learning import SelfDialogLearningSystem
- from .concept_dialog_integration import ConceptDialogIntegrator, create_concept_dialog_integrator
- from .dialog_core import SelfDialogLearning
- from .dialog_concepts import DialogConceptsMixin

---

### learning\analyzer_core.py

**Stdlib:**
- import sqlite3
- import os
- import logging
- import time
- import threading
- import json
- import sqlite3
- import sqlite3
- import sqlite3
- import json
- ... и ещё 1

**External:**
- import queue
- from typing import Dict, List, Any, Optional, Callable
- from dataclasses import dataclass, field

---

### learning\analyzer_types.py

**External:**
- from dataclasses import dataclass, field
- from typing import List, Dict, Any, Optional
- from enum import Enum

---

### learning\concept_dialog_integration.py

**Stdlib:**
- import logging
- import time
- import threading

**External:**
- from typing import Dict, List, Any, Optional
- from concurrent.futures import ThreadPoolExecutor

**Internal (eva_ai):**
- from eva_ai.core.event_bus import EventTypes

---

### learning\curiosity_engine.py

**Stdlib:**
- import logging
- import time

**External:**
- import re
- from typing import Dict, Any, List, Optional
- from dataclasses import dataclass
- from enum import Enum

**Internal (eva_ai):**
- from eva_ai.learning.learning_opportunity_manager import LearningOpportunityManager

---

### learning\data_processor.py

**Stdlib:**
- import time
- import threading
- import json
- import logging

**External:**
- from typing import Dict, List, Any, Optional
- from datetime import datetime, timedelta

---

### learning\dialog_concepts.py

**Stdlib:**
- import time
- import logging

**External:**
- from typing import Dict, List, Any, Optional
- import re

**Internal (eva_ai):**
- from eva_ai.learning.dialog_types import DialogRole, DialogTurn, SelfDialog
- from eva_ai.core.pipeline_adapter import PipelineAdapter

---

### learning\dialog_core.py

**Stdlib:**
- import logging
- import time
- import threading
- import json
- import os
- import sqlite3

**External:**
- import queue
- from typing import Dict, List, Any, Optional, Callable
- import re
- import re
- import re

**Internal (eva_ai):**
- from eva_ai.learning.dialog_types import DialogRole, DialogTurn, LearningType, SelfDialog
- from eva_ai.learning.dialog_topics import DialogTopicsMixin
- from eva_ai.learning.dialog_generation import DialogGenerationMixin
- from eva_ai.learning.dialog_learning import DialogLearningMixin
- from eva_ai.learning.dialog_concepts import DialogConceptsMixin
- from eva_ai.learning.interest_scorer import InterestScorer

---

### learning\dialog_generation.py

**Stdlib:**
- import logging
- import time

**External:**
- from __future__ import annotations
- from typing import Dict, List, Any, Optional

**Internal (eva_ai):**
- from eva_ai.learning.dialog_types import DialogRole, DialogTurn, LearningType, SelfDialog

---

### learning\dialog_learning.py

**Stdlib:**
- import logging
- import time
- import json
- import os
- import sqlite3

**External:**
- from __future__ import annotations
- from typing import Dict, List, Any, Optional

**Internal (eva_ai):**
- from eva_ai.learning.dialog_types import DialogRole, DialogTurn, LearningType, SelfDialog

---

### learning\dialog_topics.py

**Stdlib:**
- import logging
- import time

**External:**
- from __future__ import annotations
- import re
- from typing import Dict, List, Any, Optional

**Internal (eva_ai):**
- from eva_ai.learning.dialog_types import DialogRole, DialogTurn, LearningType, SelfDialog

---

### learning\dialog_types.py

**External:**
- from dataclasses import dataclass, field
- from enum import Enum
- from typing import Any, Dict, List, Optional

---

### learning\fractal_store.py

**Stdlib:**
- import time
- import logging
- import os
- import json
- import numpy
- import torch
- import torch.nn

**External:**
- from __future__ import annotations
- from pathlib import Path
- import hashlib
- import gc
- from typing import Any, Dict, List, Optional, Tuple, Union

---

### learning\integrated_learning_manager.py

**Stdlib:**
- import os
- import logging
- import time
- import threading
- import json
- import torch
- import numpy

**External:**
- from typing import Dict, Any, Optional, List, Tuple
- from datetime import datetime
- from dataclasses import dataclass

---

### learning\integration_manager.py

**Stdlib:**
- import time
- import threading
- import json
- import logging

**External:**
- from typing import Dict, List, Any, Optional, Callable
- from datetime import datetime
- from enum import Enum
- import copy

---

### learning\interest_scorer.py

**External:**
- from dataclasses import dataclass
- from typing import List, Dict, Any, Optional
- import math

---

### learning\knowledge_awareness.py

**Stdlib:**
- import time

**External:**
- from typing import Dict, Any, Optional, List
- from dataclasses import dataclass, field
- from enum import Enum

---

### learning\learning_integrated.py

**Stdlib:**
- import logging
- import time
- import os
- import json
- import json

**External:**
- from typing import Dict, List, Optional, Any, Tuple
- from datetime import datetime

**Internal (eva_ai):**
- from eva_ai.core.base_component import BaseComponent, ComponentState
- from eva_ai.core.event_bus import get_event_bus, Event, EventTypes
- from eva_ai.learning.learning_manager import LearningManager
- from eva_ai.learning.integrated_learning_manager import IntegratedLearningManager as OriginalIntegratedLearningManager

---

### learning\learning_manager.py

**Stdlib:**
- import logging
- import time

**External:**
- from typing import Dict, Any, Optional, List, Callable

---

### learning\learning_opportunity.py

**External:**
- from dataclasses import dataclass
- from typing import Any, Dict, List, Optional

---

### learning\learning_opportunity_manager.py

**Stdlib:**
- import json
- import os
- import sqlite3
- import logging
- import time

**External:**
- from typing import Dict, List, Any, Tuple

**Internal (eva_ai):**
- from eva_ai.learning.analyzer_core import AnalyzerCore
- from eva_ai.learning.learning_opportunity import LearningOpportunity

---

### learning\learning_processor.py

**Stdlib:**
- import time
- import threading
- import json
- import logging

**External:**
- from typing import Dict, List, Any, Optional
- from datetime import datetime, timedelta
- from enum import Enum
- from .data_processor import DataProcessor
- from .task_generator import LearningTaskGenerator, LearningTask, TaskType, TaskPriority
- from .integration_manager import LearningIntegrationManager, IntegrationStrategy

---

### learning\learning_scheduler.py

**External:**
- from .scheduler_core import LearningTask, ResourceAllocation, LearningSchedulerCore
- from .scheduler_tasks import TaskManagerMixin
- from .scheduler_triggers import TriggerMixin
- from .scheduler_monitor import MonitorMixin

---

### learning\learning_types.py

**External:**
- from dataclasses import dataclass, field
- from typing import List, Dict, Any, Optional
- from enum import Enum
- from datetime import datetime

---

### learning\performance_analyzer.py

**Stdlib:**
- import os
- import logging
- import time
- import sqlite3
- import json

**External:**
- import re
- from typing import Dict, List, Any, Optional
- from collections import defaultdict

**Internal (eva_ai):**
- from eva_ai.learning import AnalyzerCore

---

### learning\scheduler_core.py

**Stdlib:**
- import os
- import time
- import logging
- import threading
- import json

**External:**
- import sys
- import heapq
- from typing import Dict, List, Optional, Any, Set, Union, Callable, Deque, Tuple
- from dataclasses import dataclass, field

---

### learning\scheduler_monitor.py

**Stdlib:**
- import time
- import json
- import logging
- import numpy

**External:**
- from typing import Dict, List, Optional, Any
- from .scheduler_core import LearningTask

---

### learning\scheduler_tasks.py

**Stdlib:**
- import time
- import logging
- import numpy

**External:**
- import heapq
- from typing import Dict, List, Optional, Any
- from .scheduler_core import LearningTask, ResourceAllocation

---

### learning\scheduler_triggers.py

**Stdlib:**
- import time
- import logging
- import numpy

**External:**
- from typing import Dict, List, Optional, Any
- from .scheduler_core import LearningTask

---

### learning\self_analyzer.py

**Stdlib:**
- import os
- import logging
- import time

**External:**
- from typing import Dict, Any, Optional, List, Callable
- from .analyzer_core import AnalyzerCore

**Internal (eva_ai):**
- from eva_ai.system.health_monitor import HealthMonitor
- from eva_ai.learning.learning_opportunity_manager import LearningOpportunityManager
- from eva_ai.learning.performance_analyzer import PerformanceAnalyzer

---

### learning\self_dialog_learning.py

**Internal (eva_ai):**
- from eva_ai.learning.dialog_types import DialogRole, DialogTurn, LearningType, SelfDialog
- from eva_ai.learning.dialog_core import SelfDialogLearning, create_self_dialog_learning

---

### learning\self_dialog_new_methods.py

**External:**
- import re

---

### learning\task_generator.py

**Stdlib:**
- import time
- import threading
- import json
- import logging

**External:**
- from typing import Dict, List, Any, Optional, Tuple
- from datetime import datetime
- from enum import Enum
- import random

---

### memory\__init__.py

**External:**
- from .manager_core import MemoryManager
- from .hybrid_token_cache import HybridTokenCache
- from .working_memory import WorkingMemory
- from .long_term_memory import LongTermMemory

---

### memory\cache_core.py

**Stdlib:**
- import os
- import json
- import time
- import threading
- import logging
- import torch

**External:**
- from typing import Dict, List, Optional, Any
- import psutil
- from .cache_ram import LRUCache
- from .cache_disk import TokenDiskCache
- from .cache_eviction import _get_token_impl
- from .cache_eviction import _add_token_impl
- from .cache_eviction import _move_token_to_memory_impl
- from .cache_eviction import _evict_one_lru_impl
- from .cache_eviction import _start_memory_pressure_worker_impl
- from .cache_eviction import _offload_under_pressure_impl
- ... и ещё 10

---

### memory\cache_disk.py

**Stdlib:**
- import os
- import json
- import time
- import threading
- import logging

**External:**
- from typing import Dict, List, Optional
- import pickle
- import re
- import pickle

---

### memory\cache_eviction.py

**Stdlib:**
- import time
- import threading
- import logging

**External:**
- from typing import Dict, List, Optional, Any
- import psutil
- import re

---

### memory\cache_index.py

**Stdlib:**
- import os
- import sqlite3
- import threading
- import time
- import logging

**External:**
- from __future__ import annotations
- from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

---

### memory\cache_ram.py

**Stdlib:**
- import threading

**External:**
- from typing import Any, Optional
- from collections import OrderedDict

---

### memory\cache_router.py

**Stdlib:**
- import os
- import logging

**External:**
- from __future__ import annotations
- import hashlib
- from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
- from .cache_index import CacheIndex

---

### memory\cache_types.py

**External:**
- from dataclasses import dataclass, field
- from typing import List, Dict, Any, Optional
- from enum import Enum
- from datetime import datetime

---

### memory\context_book.py

**Stdlib:**
- import logging
- import time

**External:**
- from typing import Dict, List, Any, Optional, Tuple
- from dataclasses import dataclass, field
- from collections import defaultdict
- import re

---

### memory\disk_cache.py

**Stdlib:**
- import os
- import sqlite3
- import time
- import logging
- import threading

**External:**
- import pickle
- import zlib
- from typing import Any, Optional, Tuple
- import psutil
- import shutil

---

### memory\document_manager.py

**Stdlib:**
- import logging
- import time

**External:**
- import hashlib
- from typing import Dict, List, Any, Optional, Tuple
- from dataclasses import dataclass, field
- from collections import OrderedDict

---

### memory\embedding_cache.py

**Stdlib:**
- import os
- import json
- import sqlite3
- import logging
- import threading

**External:**
- import hashlib
- from typing import Optional, List, Dict
- from datetime import datetime

---

### memory\fg_generation_integration.py

**Stdlib:**
- import logging
- import os

**External:**
- from typing import Dict, Any, Optional, List
- from dataclasses import dataclass

**Internal (eva_ai):**
- from eva_ai.memory.fractal_graph_v2.tokenizer import GraphTokenizer
- from eva_ai.mlearning.storage.fractal_store import export_hf_model_to_fractal

---

### memory\fg_gguf_architecture_mapper.py

**Stdlib:**
- import logging

**External:**
- from typing import Dict, List, Optional, Any
- from dataclasses import dataclass

**Internal (eva_ai):**
- from eva_ai.memory.fractal_graph_v2.gguf_parser import GGUFModelParser

---

### memory\fg_gguf_quality_extraction.py

**Stdlib:**
- import logging
- import json

**External:**
- import re
- from typing import Dict, List, Optional, Any, Tuple
- from dataclasses import dataclass, field

**Internal (eva_ai):**
- from eva_ai.memory.fractal_graph_v2.gguf_parser import GGUFModelParser

---

### memory\fractal_cache\__init__.py

**External:**
- from .cache_manager import FractalCache
- from .semantic_embedder import SemanticEmbedder
- from .response_store import ResponseStore
- from .similarity_engine import SimilarityEngine
- from .eviction_policy import EvictionPolicy

---

### memory\fractal_cache\cache_manager.py

**Stdlib:**
- import os
- import json
- import time
- import logging

**External:**
- import hashlib
- from typing import Dict, Any, Optional, List, Tuple
- from .semantic_embedder import SemanticEmbedder
- from .response_store import ResponseStore
- from .similarity_engine import SimilarityEngine
- from .eviction_policy import EvictionPolicy

---

### memory\fractal_cache\eviction_policy.py

**Stdlib:**
- import time
- import logging

**External:**
- from typing import Dict, Optional, List
- from collections import OrderedDict

---

### memory\fractal_cache\response_store.py

**Stdlib:**
- import os
- import json
- import time
- import logging
- import threading

**External:**
- from typing import Dict, Any, Optional, List
- from collections import OrderedDict

---

### memory\fractal_cache\semantic_embedder.py

**Stdlib:**
- import logging

**External:**
- import hashlib
- from typing import List, Optional
- import math
- import re

---

### memory\fractal_cache\similarity_engine.py

**Stdlib:**
- import logging

**External:**
- import math
- from typing import List, Optional

---

### memory\fractal_graph_v2\__init__.py

**Stdlib:**
- import os
- import logging
- import time
- import threading
- import numpy
- import numpy
- import numpy

**External:**
- from typing import Dict, List, Optional, Any, Callable, Tuple
- from collections import OrderedDict
- from functools import wraps
- from .types import FractalNode, FractalEdge, SemanticGroup, NodeType, RelationType
- from .storage import FractalGraphV2, create_fractal_graph
- from .embeddings import EmbeddingsManager, create_embeddings_manager
- from .gguf_parser import parse_gguf_model, extract_to_graph
- from .gguf_extractor import GGUFKnowledgeExtractor, create_extractor
- from .gguf_shadow import GGUFShadowProfiler, create_gguf_shadow_profiler
- from .hybrid_tokenizer import HybridTokenizer, create_hybrid_tokenizer
- ... и ещё 6

---

### memory\fractal_graph_v2\dual_generator.py

**Stdlib:**
- import time
- import logging
- import time

**External:**
- from typing import Dict, List, Optional, Any
- from dataclasses import dataclass
- from .hybrid_tokenizer import HybridTokenizer
- from .gguf_shadow import GGUFShadowProfiler
- from .semantic_context_cache import SemanticContextCache

**Internal (eva_ai):**
- from eva_ai.memory.document_manager import DocumentVirtualMemory

---

### memory\fractal_graph_v2\dual_generator_pie.py

**Stdlib:**
- import time
- import logging

**External:**
- from typing import Dict, List, Optional, Any, Union
- from dataclasses import dataclass
- from .dual_generator import DualGenerator, CondensedGenerator, ExtendedGenerator
- from ..pie_integration import (
- from ...core.pie_fallback import PieFallbackPipeline

**Internal (eva_ai):**
- from eva_ai.memory.fractal_graph_v2.dual_generator_pie import DualGeneratorPie

---

### memory\fractal_graph_v2\embeddings.py

**Stdlib:**
- import os
- import time
- import logging
- import threading
- import numpy

**External:**
- from typing import Dict, List, Optional, Any, Tuple, Callable
- from dataclasses import dataclass
- from sentence_transformers import SentenceTransformer

---

### memory\fractal_graph_v2\eva_container.py

**Stdlib:**
- import os
- import json
- import logging
- import time
- import numpy
- import json
- import json

**External:**
- import struct
- import hashlib
- from typing import Dict, List, Optional, Any, Tuple
- from dataclasses import dataclass, field
- from pathlib import Path
- from .storage import FractalGraphV2
- import gzip
- import zstandard

---

### memory\fractal_graph_v2\eva_generator.py

**Stdlib:**
- import os
- import time
- import logging

**External:**
- import re
- from typing import Dict, List, Optional, Any, Tuple
- from dataclasses import dataclass
- from .hybrid_tokenizer import HybridTokenizer, Token
- from .gguf_shadow import GGUFShadowProfiler
- from .semantic_context_cache import SemanticContextCache
- from .prompt_templates import SYSTEM_PROMPTS, FEW_SHOT_EXAMPLES, REASONING_CHAIN_PROMPT

---

### memory\fractal_graph_v2\gguf_extractor.py

**Stdlib:**
- import os
- import json
- import time
- import logging
- import threading

**External:**
- import re
- from typing import Dict, List, Optional, Any, Tuple, Callable
- from dataclasses import dataclass, field
- from llama_cpp import Llama

---

### memory\fractal_graph_v2\gguf_parser.py

**Stdlib:**
- import os
- import json
- import logging

**External:**
- import struct
- from typing import Dict, List, Optional, Any, Tuple
- from dataclasses import dataclass, field
- from llama_cpp import Llama
- from gguf import GGUFReader

---

### memory\fractal_graph_v2\gguf_shadow.py

**Stdlib:**
- import os
- import json
- import logging
- import time
- import numpy

**External:**
- from typing import Dict, List, Optional, Any, Tuple
- from dataclasses import asdict
- from .types import FractalNode, FractalEdge, NodeType, RelationType
- from .gguf_parser import parse_gguf_model

---

### memory\fractal_graph_v2\hybrid_tokenizer.py

**Stdlib:**
- import logging

**External:**
- import re
- import hashlib
- from typing import Dict, List, Optional, Any, Tuple, Set
- from dataclasses import dataclass
- from collections import defaultdict

---

### memory\fractal_graph_v2\optimizations.py

**Stdlib:**
- import os
- import logging
- import numpy
- import time

**External:**
- from typing import Dict, List, Optional, Any, Tuple
- from collections import defaultdict
- import nmslib
- import faiss
- from transformers import pipeline

---

### memory\fractal_graph_v2\semantic_context_cache.py

**Stdlib:**
- import os
- import time
- import logging
- import threading
- import numpy
- import json
- import json

**External:**
- import hashlib
- from typing import List, Optional, Dict, Any, Tuple
- from dataclasses import dataclass, field
- from collections import OrderedDict
- import faiss

**Internal (eva_ai):**
- from eva_ai.mlearning.sentence_transformers_cache import get_sentence_transformer

---

### memory\fractal_graph_v2\snapshot_manager.py

**Stdlib:**
- import time
- import logging
- import threading

**External:**
- import hashlib
- from dataclasses import dataclass, field
- from typing import Dict, Optional, List, Any
- from copy import deepcopy

---

### memory\fractal_graph_v2\storage.py

**Stdlib:**
- import os
- import sqlite3
- import json
- import time
- import logging
- import threading
- import numpy

**External:**
- import uuid
- import hashlib
- import math
- from typing import Dict, List, Optional, Any, Tuple, Set
- from collections import defaultdict
- from dataclasses import asdict
- from .types import (
- import gzip
- import zstandard
- import gzip
- ... и ещё 1

---

### memory\fractal_graph_v2\tokenizer.py

**Stdlib:**
- import os
- import logging
- import numpy

**External:**
- import re
- from typing import Dict, List, Optional, Any, Tuple, Set
- from dataclasses import dataclass

---

### memory\fractal_graph_v2\types.py

**Stdlib:**
- import os
- import json
- import time
- import logging
- import threading
- import numpy

**External:**
- import uuid
- import hashlib
- from typing import Dict, List, Optional, Any, Tuple, Set
- from dataclasses import dataclass, field, asdict
- from enum import Enum
- from collections import defaultdict
- import math

---

### memory\fractal_graph_v2\virtual_token_handler.py

**Stdlib:**
- import logging

**External:**
- from typing import List, Dict, Optional, Any, Generator, Set, Tuple
- from dataclasses import dataclass

---

### memory\fractal_torch_storage\__init__.py

**External:**
- from .base_storage import FractalWeightStorage
- from .weight_index import WeightIndex
- from .layer_manager import LayerManager
- from .compression import WeightCompressor
- from .model_exporter import ModelExporter

---

### memory\fractal_torch_storage\base_storage.py

**Stdlib:**
- import os
- import logging
- import threading

**External:**
- from typing import Dict, Any, Optional, List
- from .weight_index import WeightIndex
- from .layer_manager import LayerManager
- from .compression import WeightCompressor
- import pickle
- import pickle

---

### memory\fractal_torch_storage\compression.py

**Stdlib:**
- import logging

**External:**
- from typing import Optional, Tuple
- import struct
- import struct
- import struct
- import struct
- import struct
- import struct

---

### memory\fractal_torch_storage\layer_manager.py

**Stdlib:**
- import logging
- import time

**External:**
- from typing import Dict, Set, Optional

---

### memory\fractal_torch_storage\model_exporter.py

**Stdlib:**
- import os
- import json
- import time
- import logging
- import torch
- import torch

**External:**
- from typing import Dict, Any, Optional, List
- from pathlib import Path
- from .base_storage import FractalWeightStorage
- from .weight_index import WeightIndex
- from transformers import AutoModelForCausalLM, BitsAndBytesConfig

---

### memory\fractal_torch_storage\weight_index.py

**Stdlib:**
- import logging

**External:**
- from typing import Dict, List, Optional

---

### memory\gguf_fractal_exporter.py

**Stdlib:**
- import os
- import json
- import logging

**External:**
- from typing import Dict, Any, List
- from llama_cpp import Llama

---

### memory\gguf_parser.py

**Stdlib:**
- import logging

**External:**
- import struct
- from typing import Dict, Any, List, Optional, Tuple
- from dataclasses import dataclass, field

---

### memory\graph_learning.py

**Stdlib:**
- import os
- import json
- import time
- import logging
- import threading
- import numpy
- import numpy

**External:**
- import hashlib
- from typing import Dict, Any, List, Optional
- from dataclasses import dataclass, field
- from collections import Counter

**Internal (eva_ai):**
- from eva_ai.neuromorphic.sim_core import NeuromorphicSimulator

---

### memory\hotset.py

**Stdlib:**
- import time
- import logging
- import torch

**External:**
- from __future__ import annotations
- from collections import OrderedDict
- from dataclasses import dataclass
- from typing import Any, Callable, Dict, Optional, Tuple

---

### memory\hybrid_token_cache.py

**External:**
- from .cache_core import HybridTokenCache, get_shared_cache
- from .cache_ram import LRUCache
- from .cache_disk import TokenDiskCache

---

### memory\long_term_memory.py

**Stdlib:**
- import logging
- import time

**External:**
- from typing import Dict, Any, Optional
- from collections import defaultdict

**Internal (eva_ai):**
- from eva_ai.memory.memory_core import MemoryNeuron, MemoryField, MemoryDatabase

---

### memory\longterm_types.py

**External:**
- from dataclasses import dataclass, field
- from typing import List, Dict, Any, Optional
- from enum import Enum

---

### memory\ltm_consolidation.py

**Stdlib:**
- import logging
- import time

**External:**
- from typing import Dict, List, Any
- from collections import defaultdict
- from .memory_core import MemoryNeuron, MemoryField

---

### memory\ltm_core.py

**Stdlib:**
- import logging
- import time
- import threading
- import numpy

**External:**
- from typing import Dict, List, Optional, Any
- from collections import defaultdict
- import re
- from .memory_core import MemoryNeuron, MemoryField, MemoryDatabase
- from .memory_working import WorkingMemory
- from .ltm_storage import _load_semantic_from_db
- from .ltm_storage import _update_knowledge_graph_impl
- from .ltm_consolidation import _consolidate_from_working_impl
- from .ltm_consolidation import _is_duplicate_impl
- from .ltm_retrieval import _retrieve_by_concept_impl
- ... и ещё 15

---

### memory\ltm_retrieval.py

**Stdlib:**
- import logging
- import time

**External:**
- from typing import Dict, List, Optional, Any
- from .memory_core import MemoryNeuron

---

### memory\ltm_storage.py

**Stdlib:**
- import logging
- import time

**External:**
- from typing import Dict, Optional, Any
- from .memory_core import MemoryNeuron, MemoryField

---

### memory\macro_archive.py

**Stdlib:**
- import json
- import threading

**External:**
- from __future__ import annotations
- from dataclasses import dataclass
- from pathlib import Path
- from typing import Iterable, List, Tuple, Optional
- from .paged_store import SuperblockMeta, ExtentMeta, SubBlockMeta, HierarchicalIndex
- import mmap

---

### memory\macro_integration.py

**Stdlib:**
- import os
- import threading
- import time
- import logging
- import torch

**External:**
- from __future__ import annotations
- from dataclasses import dataclass
- from pathlib import Path
- from typing import Dict, List, Optional, Sequence, Tuple, Callable, Any
- from .macro_archive import MacroArchive
- from .hotset import HotSetManager
- from .paged_store import (

---

### memory\manager_cache.py

**Stdlib:**
- import os
- import logging
- import time

**External:**
- import shutil
- from typing import Dict, Any
- import psutil
- from .manager_operations import _save_memory
- from .manager_operations import _save_memory
- from .manager_operations import _save_memory
- from .manager_operations import _save_memory

---

### memory\manager_core.py

**Stdlib:**
- import os
- import logging
- import json
- import time
- import threading

**External:**
- from typing import Dict, List, Optional, Any, Tuple, Iterable
- from pathlib import Path
- import psutil
- from .hybrid_token_cache import get_shared_cache
- from .hybrid_token_cache import HybridTokenCache, get_shared_cache
- from .manager_operations import (

**Internal (eva_ai):**
- from eva_ai.knowledge.context_entity import EntityExtractor
- from eva_ai.core.base_component import ComponentState
- from eva_ai.core.deferred_command_system import CommandPriority
- from eva_ai.memory.fractal_graph_v2 import FractalMemoryGraph

---

### memory\manager_gc.py

**Stdlib:**
- import os
- import logging
- import json
- import time

**External:**
- from typing import Dict, List, Optional, Any, Tuple, Iterable
- from pathlib import Path
- from .manager_operations import _save_memory
- from .manager_operations import _save_working_memory
- from .manager_operations import _save_semantic_memory
- from .manager_operations import _save_episodic_memory
- from .manager_operations import _save_user_profiles
- from .manager_operations import add_memory

---

### memory\manager_operations.py

**Stdlib:**
- import os
- import logging
- import json
- import time

**External:**
- from typing import Dict, List, Optional, Any

---

### memory\memory_cache.py

**Stdlib:**
- import logging
- import time
- import threading

**External:**
- from collections import OrderedDict
- from typing import Dict, List, Any, Optional

---

### memory\memory_consolidator.py

**Stdlib:**
- import logging
- import threading
- import time

**External:**
- from typing import Dict, Any, List, Optional
- from dataclasses import dataclass
- import psutil

---

### memory\memory_core.py

**Stdlib:**
- import os
- import logging
- import sqlite3
- import json
- import time

**External:**
- import hashlib
- from typing import Dict, Any, Optional, List, Tuple
- from collections import defaultdict

---

### memory\memory_long_term.py

**External:**
- from .ltm_core import SemanticMemory, EpisodicMemory, LongTermMemory

---

### memory\memory_manager.py

**External:**
- from .manager_core import MemoryManager
- from .manager_operations import (
- from .manager_cache import (
- from .manager_gc import (

---

### memory\memory_types.py

**External:**
- from typing import Dict, List, Any, Optional
- from datetime import datetime
- from dataclasses import dataclass, field
- from enum import Enum

---

### memory\memory_working.py

**Stdlib:**
- import os
- import logging
- import time
- import threading
- import json
- import numpy

**External:**
- import queue
- import re
- from typing import Dict, List, Optional, Tuple, Any, Set, Union, Callable, Deque
- from dataclasses import dataclass, field
- from collections import deque, defaultdict
- from .memory_core import MemoryNeuron, MemoryField, MemoryDatabase

---

### memory\metadata_manager.py

**Stdlib:**
- import logging
- import time
- import threading
- import os
- import json

**External:**
- from typing import Dict, List, Any

---

### memory\pie_integration\__init__.py

**External:**
- from .fractal_graph_l1_l2 import (
- from .activation_profiler import (
- from .routing_engine import (
- from .pie_adapter import (

---

### memory\pie_integration\activation_profiler.py

**Stdlib:**
- import numpy
- import logging

**External:**
- from typing import Dict, List, Optional, Any, Tuple
- from dataclasses import dataclass
- from sentence_transformers import SentenceTransformer
- from .fractal_graph_l1_l2 import FractalGraphL1L2, ActivationProfileData

---

### memory\pie_integration\fractal_graph_l1_l2.py

**Stdlib:**
- import numpy
- import time
- import json
- import sqlite3

**External:**
- from typing import Dict, List, Optional, Any, Tuple
- from dataclasses import dataclass
- import uuid
- import uuid

---

### memory\pie_integration\pie_adapter.py

**Stdlib:**
- import logging

**External:**
- from typing import Dict, List, Optional, Any, Tuple
- from dataclasses import dataclass
- from .fractal_graph_l1_l2 import FractalGraphL1L2, create_l1l2_graph
- from .activation_profiler import ActivationProfiler, create_default_profiler
- from .routing_engine import RoutingEngine, create_default_engine, RoutingParams

**Internal (eva_ai):**
- from eva_ai.memory.pie_integration import PieIntegration
- from eva_ai.memory.pie_integration import create_pie_integration

---

### memory\pie_integration\routing_engine.py

**Stdlib:**
- import logging
- import numpy

**External:**
- from typing import Dict, List, Optional, Any
- from dataclasses import dataclass
- from .fractal_graph_l1_l2 import FractalGraphL1L2, RoutingRuleData

---

### memory\semantic_cache.py

**Stdlib:**
- import logging
- import time
- import threading

**External:**
- from typing import Dict, Optional, Any, List
- from dataclasses import dataclass

---

### memory\token_disk_cache.py

**Stdlib:**
- import logging
- import os
- import json

**External:**
- from typing import Dict, List, Any

---

### memory\unified_fractal_memory.py

**Stdlib:**
- import os
- import json
- import time
- import logging
- import threading

**External:**
- import hashlib
- from typing import Dict, Any, List, Optional
- from dataclasses import dataclass, field, asdict
- from enum import Enum

**Internal (eva_ai):**
- from eva_ai.core.event_bus import Event, EventPriority
- from eva_ai.memory.graph_learning import DynamicContextBuilder, GraphLearningLoop, SnapshotManager
- from eva_ai.core.event_bus import get_event_bus
- from eva_ai.memory.gguf_fractal_exporter import GGUFFractalExporter

---

### memory\working_memory.py

**Stdlib:**
- import logging
- import time

**External:**
- from typing import Dict, Any, Optional

**Internal (eva_ai):**
- from eva_ai.memory.memory_core import MemoryNeuron, MemoryField, MemoryDatabase

---

### mlearning\__init__.py

**External:**
- from . import storage
- from .model_manager import ModelManager
- from .ml_unit import MLUnit
- from .eva_tokenizer import ЕВАTokenizer
- from .async_text_generator import AsyncTextGenerator
- from .unified_text_processor import UnifiedTextProcessor
- from .fractal_transformer import FractalTransformer, FractalConfig
- from .tokenization_fractal import ExtendedFractalTokenizer
- from .neuromorphic_memory import NeuromorphicMemoryLayer
- from .fractal_trainer import FractalKnowledgeTrainer
- ... и ещё 2

---

### mlearning\async_text_generator.py

**Stdlib:**
- import json
- import logging
- import time
- import os
- import torch

**External:**
- from __future__ import annotations
- import asyncio
- import hashlib
- import sys
- from dataclasses import dataclass, field
- from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
- from transformers.cache_utils import DynamicCache as _TF_DynamicCache  # type: ignore

**Internal (eva_ai):**
- from eva_ai.memory.disk_cache import DiskCache

---

### mlearning\bitnet_model_manager.py

**Stdlib:**
- import logging
- import os
- import torch

**External:**
- import subprocess
- import sys
- from typing import Optional, Dict, List, Any
- from transformers import AutoModelForCausalLM, AutoTokenizer
- from llama_cpp import Llama

---

### mlearning\comprehensive_learning_system.py

**Stdlib:**
- import os
- import json
- import time
- import logging
- import threading

**External:**
- from typing import Dict, Any, List, Optional
- from concurrent.futures import ThreadPoolExecutor
- from dataclasses import dataclass, field
- import random

---

### mlearning\current_manager.py

**Stdlib:**
- import os
- import json
- import time
- import logging
- import torch
- import threading

**External:**
- from __future__ import annotations
- from concurrent.futures import ThreadPoolExecutor
- from pathlib import Path
- from typing import Optional, Dict, Any, List, Tuple
- from safetensors.torch import load_file
- from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AutoTokenizer, AutoConfig
- from .text_quality_improver import TextQualityImprover
- from .text_quality_trainer import TextQualityTrainer, TrainingConfig
- from .text_quality_trainer import TextQualityTrainer
- import re

**Internal (eva_ai):**
- from eva_ai.memory.hybrid_token_cache import HybridTokenCache

---

### mlearning\enhanced_learning_integration.py

**Stdlib:**
- import os
- import json
- import time
- import logging
- import threading

**External:**
- from typing import Dict, Any, List, Optional
- from concurrent.futures import ThreadPoolExecutor
- from dataclasses import dataclass, field
- import random

---

### mlearning\eva_tokenizer.py

**Stdlib:**
- import os
- import logging
- import threading
- import torch

**External:**
- import hashlib
- from datetime import datetime
- from pathlib import Path
- from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
- from dataclasses import dataclass
- from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
- from tokenizers import Tokenizer as HFTokenizer
- from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

**Internal (eva_ai):**
- from eva_ai.core import CoreBrain
- from eva_ai.mlearning.model_manager import ModelMetadata
- from eva_ai.memory.hybrid_token_cache import HybridTokenCache

---

### mlearning\fractal_model_manager.py

**Stdlib:**
- import logging
- import json
- import os
- import torch
- import torch

**External:**
- from typing import Optional, Any, Dict
- import re
- import re
- import gc

**Internal (eva_ai):**
- from eva_ai.mlearning.hot_deployment.llama_cpp_hot import LlamaCppHotDeployment

---

### mlearning\fractal_qwen_manager.py

**Stdlib:**
- import os
- import logging
- import torch

**External:**
- from typing import Optional, Dict, Any, List
- from transformers import AutoTokenizer
- from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

**Internal (eva_ai):**
- from eva_ai.memory.fractal_torch_storage import ModelExporter

---

### mlearning\fractal_trainer.py

**Stdlib:**
- import os
- import torch
- import logging
- import json

**External:**
- import math
- from typing import Dict, List, Optional, Tuple, Union, Any
- from pathlib import Path
- from tqdm import tqdm
- from torch.utils.data import DataLoader, Dataset
- from torch.optim import AdamW
- from torch.optim.lr_scheduler import LambdaLR
- from transformers import get_linear_schedule_with_warmup
- from .fractal_transformer import FractalTransformer, FractalConfig
- from .tokenization_fractal import ExtendedFractalTokenizer
- ... и ещё 1

---

### mlearning\fractal_transformer.py

**Stdlib:**
- import torch
- import torch.nn
- import logging

**External:**
- from typing import Optional, Dict, Any, Tuple, List
- from transformers import PreTrainedModel, PretrainedConfig
- from pathlib import Path
- import math

---

### mlearning\hot_deployment\__init__.py

**Stdlib:**
- import os
- import json
- import time
- import threading
- import logging
- import torch

**External:**
- import hashlib
- from typing import Dict, Optional, Any, List, Tuple
- from dataclasses import dataclass, field
- from enum import Enum
- from collections import OrderedDict
- from transformers import AutoModelForCausalLM, AutoTokenizer

---

### mlearning\hot_deployment\convert_to_gguf.py

**Stdlib:**
- import os
- import logging
- import torch
- import numpy
- import json

**External:**
- import sys
- import subprocess
- from typing import Optional
- import llama_cpp
- from transformers import AutoModelForCausalLM, AutoTokenizer

---

### mlearning\hot_deployment\download_gguf.py

**Stdlib:**
- import os
- import logging
- import time

**External:**
- import sys
- import urllib.request
- import ssl
- from llama_cpp import Llama

---

### mlearning\hot_deployment\export_onnx.py

**Stdlib:**
- import os
- import logging
- import torch

**External:**
- import sys
- import subprocess
- from transformers import AutoModelForCausalLM, AutoTokenizer
- import traceback

---

### mlearning\hot_deployment\llama_cpp_hot.py

**Stdlib:**
- import os
- import time
- import logging
- import threading
- import torch

**External:**
- import sys
- from typing import Optional, Dict, Any, List
- from dataclasses import dataclass
- from llama_cpp import Llama
- import traceback
- import gc

**Internal (eva_ai):**
- from eva_ai.mlearning.hot_deployment import HotDeploymentManager, GraphNode, NodeState

---

### mlearning\hot_deployment\llama_cpp_wrapper.py

**Stdlib:**
- import os
- import time
- import logging

**External:**
- import sys
- from typing import Optional, List, Dict, Any
- from llama_cpp import Llama
- from llama_cpp.llama_chat_format import Llama2ChatHandler
- import traceback
- import urllib.request
- import ssl
- import subprocess

**Internal (eva_ai):**
- from eva_ai.mlearning.hot_deployment import HotDeploymentManager

---

### mlearning\hot_deployment\onnx_optimizer.py

**Stdlib:**
- import os
- import time
- import logging
- import torch

**External:**
- import sys
- from typing import Optional, Tuple
- from transformers import AutoModelForCausalLM, AutoTokenizer
- import onnx
- from onnxruntime import InferenceSession, SessionOptions
- from transformers import AutoTokenizer, AutoModelForCausalLM
- import traceback
- from transformers import AutoModelForCausalLM, AutoTokenizer
- from transformers import AutoModelForCausalLM, AutoTokenizer

**Internal (eva_ai):**
- from eva_ai.mlearning.hot_deployment import HotDeploymentManager
- from eva_ai.mlearning.hot_deployment import get_hot_deployment_manager
- import eva_ai.mlearning.hot_deployment

---

### mlearning\hot_deployment\onnx_runtime.py

**Stdlib:**
- import os
- import time
- import logging
- import numpy
- import torch

**External:**
- import sys
- from typing import Optional, List, Dict, Any
- import onnx
- import onnxruntime
- import traceback
- from transformers import AutoTokenizer
- from transformers import AutoModelForCausalLM, AutoTokenizer
- import traceback
- from onnxruntime.quantization import quantize_dynamic

---

### mlearning\hot_deployment\openvino_convert.py

**Stdlib:**
- import os
- import logging
- import torch

**External:**
- import sys
- import subprocess
- import openvino
- from transformers import AutoModelForCausalLM, AutoTokenizer
- import traceback

---

### mlearning\hot_deployment\openvino_inference.py

**Stdlib:**
- import os
- import time
- import logging
- import numpy
- import torch

**External:**
- import sys
- from typing import Optional, List, Dict, Any
- import openvino
- from openvino_tokenizers import convert_tokenizer
- from transformers import AutoModelForCausalLM, AutoTokenizer
- from optimum.exporters.openvino import export
- from optimum.utils import OvModel
- import traceback
- import traceback
- from transformers import AutoTokenizer
- ... и ещё 1

---

### mlearning\hot_deployment\openvino_via_optimum.py

**Stdlib:**
- import os
- import logging
- import time
- import numpy

**External:**
- import sys
- from optimum.intel.openvino import OVModelForCausalLM
- import traceback
- import openvino
- from transformers import AutoTokenizer
- import traceback

---

### mlearning\hot_deployment\optimized_inference.py

**Stdlib:**
- import os
- import time
- import logging
- import torch

**External:**
- import sys
- from typing import Optional, List, Dict, Any
- from transformers import AutoModelForCausalLM, AutoTokenizer
- import traceback

---

### mlearning\hybrid_model_manager.py

**Stdlib:**
- import os
- import json
- import logging
- import threading
- import time
- import torch
- import time

**External:**
- import sys
- import shutil
- from typing import Dict, Any, Optional, List, Tuple
- from dataclasses import dataclass
- from enum import Enum
- import psutil
- import sys
- import shutil
- from transformers import AutoTokenizer

---

### mlearning\language_filter.py

**Stdlib:**
- import logging

**External:**
- import re
- from typing import List, Optional, Set, Union, Dict, Any

---

### mlearning\ml_core.py

**Stdlib:**
- import os
- import logging
- import time
- import threading

**External:**
- from typing import Dict, List, Optional, Tuple, Any
- from dataclasses import dataclass, field

---

### mlearning\ml_types.py

**External:**
- from dataclasses import dataclass, field
- from typing import List, Dict, Any, Optional
- from enum import Enum

---

### mlearning\ml_unit.py

**External:**
- from .unit_core import MLUnit, _load_brain_config, _get_hybrid_cache_config, _get_project_root

---

### mlearning\model_manager.py

**Stdlib:**
- import os
- import time
- import logging
- import torch
- import torch

**External:**
- from typing import Dict, Any, Optional, Tuple, TypeVar
- from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
- from .storage.fractal_store import FractalWeightStore
- from .storage.fractal_store import FractalWeightStore
- from .storage.fractal_store import export_hf_model_to_fractal

**Internal (eva_ai):**
- from eva_ai.core.base_component import BaseComponent

---

### mlearning\model_selector.py

**Stdlib:**
- import os
- import logging
- import json

**External:**
- from typing import Optional, Dict, Any
- import psutil

**Internal (eva_ai):**
- from eva_ai.mlearning.qwen_model_manager import QwenModelManager
- from eva_ai.mlearning.bitnet_model_manager import BitNetModelManager

---

### mlearning\neuromorphic_memory.py

**Stdlib:**
- import torch
- import torch.nn
- import torch.nn.functional
- import logging

**External:**
- from typing import Optional, Tuple, Dict, Any, List
- import math

---

### mlearning\parallel_tokenization.py

**Stdlib:**
- import os
- import time
- import threading
- import logging

**External:**
- from __future__ import annotations
- import math
- import queue
- from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable

---

### mlearning\qwen_api_client.py

**Stdlib:**
- import os
- import logging
- import requests

**External:**
- from typing import Optional, Dict, Any, List, Iterator

---

### mlearning\qwen_model_manager.py

**Stdlib:**
- import logging
- import os
- import torch

**External:**
- from typing import Optional, Dict, List, Any
- from transformers import AutoTokenizer, GenerationConfig
- from transformers import AutoModelForCausalLM as QwenModelClass
- from transformers import BitsAndBytesConfig
- from transformers import BitsAndBytesConfig

---

### mlearning\sentence_transformers_cache.py

**Stdlib:**
- import logging
- import torch

**External:**
- from typing import Optional, List
- from sentence_transformers import SentenceTransformer

**Internal (eva_ai):**
- from eva_ai.memory.embedding_cache import get_embedding_cache

---

### mlearning\storage\__init__.py

**External:**
- from .fractal_store import FractalWeightStore
- from .memory_graph_store import MemoryGraphStore
- from .model_storage_adapter import ModelStorageAdapter
- from .model_storage_config import ModelStorageConfig
- from .fractal_model_loader import FractalModelLoader
- from .fractal_store_utils import FractalStoreUtils

---

### mlearning\storage\fractal_model_loader.py

**Stdlib:**
- import os
- import logging
- import torch

**External:**
- from pathlib import Path
- from typing import Dict, Optional, Any, Tuple, List
- from torch import nn
- from .fractal_store import FractalWeightStore
- from .memory_graph_store import MemoryGraphStore
- from .model_storage_config import ModelStorageConfig

---

### mlearning\storage\fractal_store.py

**External:**
- from .store_core import (

---

### mlearning\storage\fractal_store_core.py

**Stdlib:**
- import time
- import logging
- import os
- import json
- import numpy
- import torch
- import torch.nn

**External:**
- from __future__ import annotations
- from pathlib import Path
- import hashlib
- import gc
- from typing import Any, Dict, List, Optional, Tuple, Union

---

### mlearning\storage\fractal_store_utils.py

**Stdlib:**
- import time
- import logging
- import numpy
- import torch
- import json
- import json
- import time

**External:**
- from typing import Dict, List, Optional, Any, Tuple
- from .fractal_store_core import FractalWeightStore, FractalContainer

---

### mlearning\storage\fractal_weight_store.py

**Stdlib:**
- import time
- import torch
- import numpy
- import json

**External:**
- from typing import Any, Dict, Optional, Union, List
- from collections import OrderedDict
- from .fractal_store import FractalContainer
- import pickle
- from pathlib import Path

---

### mlearning\storage\memory_graph_store.py

**Stdlib:**
- import os
- import logging
- import numpy
- import torch

**External:**
- from pathlib import Path
- from typing import Dict, Optional, Any, List, Tuple, Union
- from torch import Tensor
- from .fractal_store import FractalWeightStore

---

### mlearning\storage\model_storage_adapter.py

**Stdlib:**
- import logging
- import torch

**External:**
- from pathlib import Path
- from typing import Optional, Dict, Any
- from torch import nn
- from .memory_graph_store import MemoryGraphStore

---

### mlearning\storage\model_storage_config.py

**Stdlib:**
- import os
- import logging
- import torch

**External:**
- from dataclasses import dataclass
- from typing import Dict, Any

---

### mlearning\storage\opt_cache.py

**Stdlib:**
- import logging
- import torch

**External:**
- from __future__ import annotations
- import gc

---

### mlearning\storage\opt_core.py

**Stdlib:**
- import os
- import json
- import time
- import logging
- import torch
- import threading

**External:**
- from __future__ import annotations
- from concurrent.futures import ThreadPoolExecutor
- from pathlib import Path
- from typing import Optional, Dict, Any, List, Tuple
- from safetensors.torch import load_file
- import sys
- from utils.text_quality import check_and_fix_response, TextQualityChecker
- from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AutoTokenizer, AutoConfig
- from .opt_models import (
- from .opt_cache import optimize_memory, clear_gpu_cache

---

### mlearning\storage\opt_models.py

**Stdlib:**
- import os
- import json
- import time
- import logging
- import torch

**External:**
- from __future__ import annotations
- from typing import Optional, Dict, Any, List
- from pathlib import Path
- from .text_quality_improver import TextQualityImprover
- from .text_quality_trainer import TextQualityTrainer, TrainingConfig
- from safetensors.torch import load_file
- from transformers import GPT2LMHeadModel, GPT2Config
- from .text_quality_trainer import TextQualityTrainer
- from .web_search_learning_integration import WebSearchLearningIntegration
- from transformers import GPT2Config, GPT2LMHeadModel
- ... и ещё 3

**Internal (eva_ai):**
- from eva_ai.memory.hybrid_token_cache import HybridTokenCache

---

### mlearning\storage\optimized_fractal_model_manager.py

**External:**
- from .opt_core import OptimizedFractalModelManager

---

### mlearning\storage\store_cache.py

**Stdlib:**
- import time
- import logging
- import numpy
- import torch

**External:**
- from __future__ import annotations
- from typing import Any, Dict, List, Optional, Tuple
- from collections import OrderedDict
- from collections import defaultdict

---

### mlearning\storage\store_core.py

**Stdlib:**
- import time
- import logging
- import os
- import json
- import numpy
- import torch
- import torch.nn

**External:**
- from __future__ import annotations
- from pathlib import Path
- import hashlib
- import gc
- import math
- import sys
- import shutil
- from dataclasses import dataclass, field
- from typing import Any, Dict, Deque, Tuple, List, Optional, Iterable, Set
- from collections import deque, OrderedDict, defaultdict
- ... и ещё 5

---

### mlearning\storage\store_operations.py

**Stdlib:**
- import time
- import logging
- import json
- import numpy
- import torch

**External:**
- from __future__ import annotations
- from pathlib import Path
- import hashlib
- from typing import Any, Dict, List, Optional, Set, Tuple
- from collections import OrderedDict, defaultdict
- from .store_core import FractalContainer
- from .store_core import FractalContainer
- from .store_core import FractalContainer
- from .store_core import FractalContainer
- from .store_core import FractalContainer
- ... и ещё 1

---

### mlearning\storage\store_queries.py

**Stdlib:**
- import time
- import logging
- import numpy
- import torch

**External:**
- from __future__ import annotations
- import hashlib
- from typing import Any, Dict, List, Optional
- from collections import defaultdict

---

### mlearning\storage\unified_fractal_store.py

**Stdlib:**
- import os
- import logging
- import json
- import torch
- import numpy

**External:**
- from pathlib import Path
- from typing import Dict, Optional, Any, Tuple, List, Union
- from dataclasses import dataclass
- from torch import nn
- from .fractal_store import FractalWeightStore, KnowledgeGraphProxy
- from .memory_graph_store import MemoryGraphStore
- from .model_storage_config import ModelStorageConfig

---

### mlearning\storage\unified_graph_store.py

**Stdlib:**
- import os
- import logging
- import torch
- import numpy

**External:**
- from pathlib import Path
- from typing import Dict, Optional, Any, List, Union, Tuple
- from torch import nn
- from .fractal_weight_store import FractalWeightStore

---

### mlearning\storage\unified_storage.py

**Stdlib:**
- import os
- import json
- import logging
- import torch
- import numpy

**External:**
- from pathlib import Path
- from typing import Dict, Optional, Any, Union, List, Tuple
- from dataclasses import dataclass
- from torch import nn

---

### mlearning\text_quality_improver.py

**Stdlib:**
- import os
- import json
- import time
- import logging
- import torch

**External:**
- from __future__ import annotations
- from typing import Dict, List, Optional, Tuple, Any
- from dataclasses import dataclass
- import re
- from transformers import GPT2LMHeadModel, GPT2Tokenizer

---

### mlearning\text_quality_learning_integration.py

**Stdlib:**
- import logging
- import os
- import time
- import threading
- import threading
- import threading

**External:**
- from __future__ import annotations
- import tkinter
- from tkinter import ttk
- from typing import Dict, Any, Optional, List
- from datetime import datetime
- from .text_quality_trainer import TextQualityTrainer, TrainingConfig
- from .text_quality_improver import TextQualityImprover
- from .text_quality_trainer import TextQualityTrainer, TrainingConfig
- from .text_quality_improver import TextQualityImprover
- from .text_quality_trainer import TextQualityTrainer, TrainingConfig
- ... и ещё 1

---

### mlearning\text_quality_trainer.py

**Stdlib:**
- import os
- import json
- import time
- import logging
- import torch
- import torch.nn
- import numpy
- import threading

**External:**
- from __future__ import annotations
- from typing import Dict, List, Optional, Any, Tuple
- from dataclasses import dataclass
- from torch.optim import AdamW
- from torch.utils.data import DataLoader, Dataset
- import re
- import re

---

### mlearning\tokenization_fractal.py

**Stdlib:**
- import os
- import json
- import logging
- import torch

**External:**
- from typing import Any, Dict, List, Optional, Tuple, Union
- from pathlib import Path
- from transformers import PreTrainedTokenizerFast, AutoTokenizer

---

### mlearning\tokenizer_registry.py

**Stdlib:**
- import os
- import json
- import threading
- import logging
- import torch

**External:**
- from typing import Any, Optional
- from transformers import AutoTokenizer

---

### mlearning\training_types.py

**External:**
- from dataclasses import dataclass, field
- from typing import List, Dict, Any, Optional
- from enum import Enum

---

### mlearning\unified_fractal_manager.py

**Stdlib:**
- import os
- import logging
- import json

**External:**
- from typing import Optional, Dict, Any, List
- from .fractal_model_manager import FractalModelManager
- from .optimized_fractal_model_manager import OptimizedFractalModelManager
- from .enhanced_learning_integration import EnhancedLearningIntegration
- from .comprehensive_learning_system import ComprehensiveLearningSystem
- from .comprehensive_learning_system import ComprehensiveLearningSystem

---

### mlearning\unified_text_processor.py

**Stdlib:**
- import os
- import time
- import logging
- import threading
- import numpy
- import json
- import torch

**External:**
- import multiprocessing
- import re
- import hashlib
- import nltk
- from collections import defaultdict, Counter, OrderedDict
- from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Type, TypeVar
- from dataclasses import dataclass, field
- from concurrent.futures import ThreadPoolExecutor
- from functools import partial
- import psutil
- ... и ещё 4

**Internal (eva_ai):**
- from eva_ai.config import is_embedding_loading_disabled, DISABLE_ALL_MODELS
- from eva_ai.core.base_component import BaseComponent
- from eva_ai.mlearning.sentence_transformers_cache import get_sentence_transformer

---

### mlearning\unit_components.py

**Stdlib:**
- import os
- import json
- import logging
- import time
- import threading
- import torch

**External:**
- from __future__ import annotations
- import sys
- from typing import Dict, Any, Optional, List, Callable
- from datetime import datetime

**Internal (eva_ai):**
- from eva_ai.mlearning.ml_core import MLCore
- from eva_ai.nlp.text_processor import TextProcessor
- from eva_ai.mlearning.model_manager import ModelManager as _ModelManager
- from eva_ai.core.response_generator import ResponseGenerator
- from eva_ai.memory.hybrid_token_cache import HybridTokenCache

---

### mlearning\unit_core.py

**Stdlib:**
- import os
- import json
- import logging
- import time
- import threading
- import torch

**External:**
- import sys
- import queue
- from typing import Dict, Any, Optional, List, Callable
- from datetime import datetime
- import sys
- from .unit_components import (

---

### mlearning\unit_training.py

**Stdlib:**
- import logging

**External:**
- from __future__ import annotations

---

### mlearning\universal_model_manager.py

**Stdlib:**
- import logging
- import torch
- import torch

**External:**
- from typing import Optional, Dict, Any
- import psutil
- from .bitnet_model_manager import BitNetModelManager
- from .qwen_model_manager import QwenModelManager
- from .fractal_model_manager import FractalModelManager
- import psutil

---

### mlearning\web_search_learning_integration.py

**Stdlib:**
- import os
- import logging
- import time

**External:**
- from typing import Dict, Any, List, Optional
- from concurrent.futures import ThreadPoolExecutor
- from ..websearch.web_search_engine import WebSearchEngine

---

### monitoring\system_monitor.py

**Stdlib:**
- import os
- import time
- import threading
- import logging
- import json

**External:**
- import platform
- import psutil
- from typing import Dict, List, Any, Optional, Callable
- from dataclasses import dataclass
- from datetime import datetime, timedelta
- from collections import defaultdict
- import statistics

---

### neuromorphic\__init__.py

**External:**
- from .neuromorphic_simulator import (

---

### neuromorphic\neuromorphic_memory.py

**Stdlib:**
- import logging

**External:**
- from typing import Dict, List, Optional, Any

**Internal (eva_ai):**
- from eva_ai.neuromorphic.neuromorphic_simulator import (

---

### neuromorphic\neuromorphic_simulator.py

**External:**
- from .sim_core import NeuromorphicSimulator
- from .sim_neurons import FallbackNeuralNetwork
- from .sim_spikes import NeuralActivity, SpikeEvent, SpikeGenerator, SpikePropagator
- from .sim_synapses import SynapseManager
- from .sim_plasticity import STDPPlasticity, AdaptiveThreshold, HomeostaticPlasticity

---

### neuromorphic\sim_core.py

**Stdlib:**
- import os
- import logging
- import time
- import threading
- import json
- import numpy

**External:**
- from typing import Dict, List, Optional, Any
- import base64
- import matplotlib.pyplot
- import nest
- from io import BytesIO
- from dataclasses import dataclass, field
- from .sim_neurons import FallbackNeuralNetwork
- from .sim_spikes import NeuralActivity

---

### neuromorphic\sim_neurons.py

**Stdlib:**
- import logging
- import numpy

**External:**
- from typing import Dict, Any, Optional

---

### neuromorphic\sim_plasticity.py

**Stdlib:**
- import logging
- import numpy

**External:**
- from typing import Dict, Any, List, Optional

---

### neuromorphic\sim_spikes.py

**Stdlib:**
- import logging
- import time
- import numpy

**External:**
- from typing import Dict, Any, List
- from dataclasses import dataclass, field

---

### neuromorphic\sim_synapses.py

**Stdlib:**
- import logging
- import numpy

**External:**
- from typing import Dict, Any, List

---

### nlp\__init__.py

**External:**
- from .text_processor import TextProcessor

---

### nlp\text_processor.py

**Stdlib:**
- import os
- import logging
- import torch

**External:**
- import re
- from typing import List, Dict, Any, Optional, Union
- from functools import lru_cache
- from transformers import AutoTokenizer, PreTrainedTokenizerBase
- from transformers import AutoTokenizer

**Internal (eva_ai):**
- from eva_ai.mlearning.tokenizer_registry import TokenizerRegistry

---

### nlp_fallbacks.py

**Stdlib:**
- import logging
- import time

**External:**
- from __future__ import annotations
- import re
- import concurrent.futures
- from typing import Any, Dict, List, Optional, Tuple, Union
- from collections import Counter
- import hashlib
- from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
- from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
- from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
- import spacy
- ... и ещё 1

---

### preprocess\__init__.py

**External:**
- from .preprocessing_pipeline import (

---

### preprocess\preprocessing_pipeline.py

**Stdlib:**
- import logging
- import json

**External:**
- import re
- from typing import Dict, Any, List, Optional, Set
- from dataclasses import dataclass, field

---

### reasoning\__init__.py

**External:**
- from .self_reasoning_engine import SelfReasoningEngine, create_reasoning_engine
- from .confidence_scorer import calculate_overall_confidence, should_terminate, CONFIDENCE_THRESHOLD, get_confidence_level
- from .clarification_generator import ClarificationGenerator
- from .reasoning_types import ReasoningStep, ReasoningResult, ReasoningPhase, AnalysisResult
- from .integration import ReasoningIntegration, integrate_reasoning
- from .enhanced_reasoning_engine import EnhancedReasoningEngine, ReasoningIteration
- from .fractal_ml import FractalStorage, FractalNode, FractalNodeType, FractalRetriever, FractalEmbedder

---

### reasoning\analytics_module.py

**Stdlib:**
- import logging

**External:**
- import re
- from typing import Dict, List, Any, Optional, Tuple
- from dataclasses import dataclass, field

---

### reasoning\clarification_generator.py

**Stdlib:**
- import logging

**External:**
- import re
- from typing import Dict, Any, List, Optional

---

### reasoning\combined_metric.py

**Stdlib:**
- import logging

**External:**
- from typing import Dict, List, Optional, Any
- from dataclasses import dataclass

---

### reasoning\confidence_scorer.py

**Stdlib:**
- import logging

**External:**
- from typing import Dict, Any, Optional

---

### reasoning\correlation_calculator.py

**Stdlib:**
- import logging

**External:**
- from typing import Dict, Any, Optional, List, Tuple

---

### reasoning\enhanced_reasoning_engine.py

**Stdlib:**
- import logging
- import time

**External:**
- import asyncio
- from typing import Dict, Any, Optional, List
- from dataclasses import dataclass, field
- from .analytics_module import AnalyticsModule, AnalyticsResult
- from .prompt_composer import PromptComposer, ComposedPrompt
- from .semantic_stability import SemanticStabilityChecker, StabilityResult
- from .combined_metric import CombinedMetricCalculator, ImprovementResult
- from .entity_extractor import EntityExtractor, ExtractedEntity
- from .correlation_calculator import CorrelationCalculator, CorrelationResult
- import traceback
- ... и ещё 2

**Internal (eva_ai):**
- from eva_ai.mlearning.fractal_qwen_manager import get_fractal_qwen

---

### reasoning\entity_extractor.py

**Stdlib:**
- import logging

**External:**
- import re
- from typing import Dict, List, Any, Optional, Set
- from dataclasses import dataclass, field

---

### reasoning\fractal_address.py

**Stdlib:**
- import numpy

**External:**
- import hashlib
- from typing import List, Optional, Tuple, Dict, Any

---

### reasoning\fractal_ml\__init__.py

**External:**
- from .fractal_base import (
- from .fractal_tokenizer import FractalTokenizer, FractalTokenizerWrapper
- from .fractal_storage import FractalStorage
- from .fractal_retriever import FractalRetriever
- from .fractal_embedder import FractalEmbedder

---

### reasoning\fractal_ml\fractal_base.py

**Stdlib:**
- import os
- import time
- import json
- import logging

**External:**
- import hashlib
- from typing import Dict, Any, Optional, List, Set, Tuple
- from dataclasses import dataclass, field, asdict
- from enum import Enum
- from collections import defaultdict

---

### reasoning\fractal_ml\fractal_embedder.py

**Stdlib:**
- import json
- import logging

**External:**
- import hashlib
- from typing import Dict, Any, List, Optional

**Internal (eva_ai):**
- from eva_ai.mlearning.sentence_transformers_cache import get_sentence_transformer

---

### reasoning\fractal_ml\fractal_retriever.py

**Stdlib:**
- import logging

**External:**
- from typing import Dict, Any, List, Optional

---

### reasoning\fractal_ml\fractal_storage.py

**Stdlib:**
- import os
- import json
- import logging
- import time

**External:**
- from typing import Dict, Any, List, Optional
- from .fractal_base import FractalNode, FractalNodeType, FractalEdge, FractalRelationType, create_fractal_id

---

### reasoning\fractal_ml\fractal_tokenizer.py

**Stdlib:**
- import logging
- import json
- import json

**External:**
- import re
- from typing import List, Dict, Set, Optional
- from collections import Counter

---

### reasoning\integration.py

**Stdlib:**
- import logging

**External:**
- from typing import Optional, Dict, Any

**Internal (eva_ai):**
- from eva_ai.reasoning.self_reasoning_engine import SelfReasoningEngine

---

### reasoning\prompt_composer.py

**Stdlib:**
- import logging

**External:**
- from typing import Dict, List, Any, Optional
- from dataclasses import dataclass, field

---

### reasoning\reasoning_nodes.py

**Stdlib:**
- import time

**External:**
- from enum import Enum
- from typing import Dict, Any, Optional, List
- from dataclasses import dataclass, field

---

### reasoning\reasoning_types.py

**Stdlib:**
- import time

**External:**
- from dataclasses import dataclass, field
- from typing import Dict, Any, List, Optional
- from enum import Enum

---

### reasoning\self_reasoning_engine.py

**Stdlib:**
- import time
- import logging
- import threading
- import os

**External:**
- from typing import Dict, Any, Optional, List
- from .reasoning_types import (
- from .confidence_scorer import (
- from .clarification_generator import ClarificationGenerator
- from .sre_context import (
- from .sre_quality import (
- from .sre_feedback import *
- from .sre_recursive import (

**Internal (eva_ai):**
- from eva_ai.reasoning.fractal_ml import FractalStorage
- from eva_ai.reasoning.fractal_ml.fractal_embedder import FractalEmbedder
- from eva_ai.reasoning.fractal_ml.fractal_retriever import FractalRetriever

---

### reasoning\semantic_stability.py

**Stdlib:**
- import logging

**External:**
- import re
- from typing import Dict, List, Optional, Tuple
- from dataclasses import dataclass

---

### reasoning\sre_context.py

**Stdlib:**
- import time
- import logging
- import threading

**External:**
- from typing import Dict, Any, Optional, List

---

### reasoning\sre_feedback.py

**Stdlib:**
- import time
- import logging

**External:**
- from typing import Dict, Any, List, Optional

---

### reasoning\sre_quality.py

**Stdlib:**
- import logging

**External:**
- from typing import Dict, Any

---

### reasoning\sre_recursive.py

**Stdlib:**
- import logging

**External:**
- from typing import Dict, Any, Optional, List
- from .reasoning_types import ReasoningStep, ReasoningResult, ReasoningPhase
- from .confidence_scorer import calculate_overall_confidence, should_terminate

**Internal (eva_ai):**
- from eva_ai.reasoning.fractal_ml.fractal_retriever import FractalRetriever

---

### recovery\recovery_system.py

**Stdlib:**
- import os
- import time
- import logging
- import threading
- import json
- import torch

**External:**
- import shutil
- from typing import Dict, List, Any, Optional, Callable
- from dataclasses import dataclass
- from datetime import datetime, timedelta
- from pathlib import Path
- import traceback
- import hashlib
- import traceback
- import traceback

---

### run.py

**Stdlib:**
- import os
- import logging
- import threading
- import time
- import torch
- import os

**External:**
- import sys
- import warnings
- import atexit
- import signal
- import concurrent.futures
- import concurrent.futures
- import subprocess
- import re
- import server

**Internal (eva_ai):**
- from eva_ai.core.utils import setup_logging
- from eva_ai.core.core_brain import CoreBrain

---

### runtime\simple_model.py

**Stdlib:**
- import torch

**External:**
- from __future__ import annotations
- from typing import Any, Dict

**Internal (eva_ai):**
- from eva_ai.adapters.torch_adapter import Batch

---

### runtime\worker_pool.py

**Stdlib:**
- import os
- import torch
- import logging

**External:**
- from __future__ import annotations
- import sys
- import queue
- import multiprocessing
- from contextlib import contextmanager
- from importlib import import_module
- from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

**Internal (eva_ai):**
- from eva_ai.adapters.torch_adapter import Batch
- from eva_ai.core.device_resolver import (
- from eva_ai.core.batch_wrapper import assert_clean_batch, emit_wrapper_event, WrapperMetadata

---

### scripts\activate_max_cache.py

**Stdlib:**
- import os
- import time
- import logging

**External:**
- import sys
- import traceback

**Internal (eva_ai):**
- from eva_ai.mlearning.unified_fractal_manager import UnifiedFractalManager

---

### scripts\complete_fractal_solution.py

**Stdlib:**
- import os
- import torch
- import json
- import logging
- import os
- import logging

**External:**
- import sys
- import shutil
- import hashlib
- from pathlib import Path
- from datetime import datetime
- from transformers import AutoTokenizer, GPT2Tokenizer, GPT2TokenizerFast
- from transformers import AutoModelForCausalLM
- from pathlib import Path

**Internal (eva_ai):**
- from eva_ai.mlearning.storage.fractal_store import export_hf_model_to_fractal
- from eva_ai.mlearning.storage.fractal_model_loader import FractalModelLoader
- from eva_ai.mlearning.storage.model_storage_config import ModelStorageConfig

---

### scripts\export_qwen.py

**Stdlib:**
- import os

**External:**
- import sys

**Internal (eva_ai):**
- from eva_ai.memory.fractal_torch_storage.model_exporter import ModelExporter

---

### scripts\load_gguf_to_fg.py

**Stdlib:**
- import os
- import logging

**External:**
- import sys
- import traceback

**Internal (eva_ai):**
- from eva_ai.memory.fractal_graph_v2 import FractalMemoryGraph
- from eva_ai.memory.fg_gguf_architecture_mapper import create_architecture_mapper

---

### scripts\migrate_events.py

**Stdlib:**
- import logging

**External:**
- from datetime import datetime

---

### scripts\migrate_kg_to_fg.py

**Stdlib:**
- import os
- import logging

**External:**
- import sys
- import traceback
- import traceback

**Internal (eva_ai):**
- from eva_ai.core.core_brain import CoreBrain
- from eva_ai.memory.fractal_graph_v2 import FractalMemoryGraph
- from eva_ai.knowledge.kg_to_fg_migration import migrate_knowledge_graph

---

### scripts\migrate_to_optimized.py

**Stdlib:**
- import os
- import logging
- import json

**External:**
- import sys
- import shutil
- import traceback

**Internal (eva_ai):**
- from eva_ai.mlearning.optimized_fractal_model_manager import OptimizedFractalModelManager

---

### scripts\simple_test.py

**Stdlib:**
- import os
- import time
- import torch

**External:**
- import sys
- from transformers import AutoModelForCausalLM, AutoTokenizer

---

### security\__init__.py

**External:**
- from .security_framework import (

---

### security\security_framework.py

**Stdlib:**
- import os
- import time
- import logging
- import threading
- import json

**External:**
- import hashlib
- import secrets
- from typing import Dict, List, Optional, Tuple, Any
- from dataclasses import dataclass
- from datetime import datetime, timedelta
- from collections import defaultdict

---

### server.py

**External:**
- from .server_main import (

---

### server_handlers.py

**Stdlib:**
- import logging

**External:**
- from datetime import datetime
- from flask import jsonify, request
- from .server_main import web_gui_instance, app
- import psutil

---

### server_routes.py

**Stdlib:**
- import os
- import logging

**External:**
- import uuid
- from datetime import datetime
- from flask import jsonify, request
- from .server_main import web_gui_instance, app, extract_text_from_file

---

### setup.py

**External:**
- from setuptools import setup, find_packages

---

### storage\fractal_storage.py

**Stdlib:**
- import os
- import logging
- import json
- import os
- import json
- import os
- import os

**External:**
- from typing import Optional, Any, Dict
- import pickle

**Internal (eva_ai):**
- from eva_ai.mlearning.eva_tokenizer import ЕВАTokenizer
- from eva_ai.mlearning.eva_tokenizer import ЕВАTokenizer
- from eva_ai.core.coordinator import Coordinator

---

### storage\storage_types.py

**External:**
- from dataclasses import dataclass, field
- from typing import List, Dict, Any, Optional
- from enum import Enum

---

### system\fault_tolerance.py

**Stdlib:**
- import logging
- import time

**External:**
- from typing import Dict, Any, Optional, List, Callable

---

### system\health_monitor.py

**Stdlib:**
- import os
- import logging
- import time
- import sqlite3
- import json

**External:**
- import re
- from typing import Dict, List, Any, Optional

**Internal (eva_ai):**
- from eva_ai.learning import AnalyzerCore

---

### system\system_types.py

**External:**
- from dataclasses import dataclass, field
- from typing import List, Dict, Any, Optional
- from enum import Enum

---

### system_selftest.py

**Stdlib:**
- import os
- import logging
- import time

**External:**
- import sys
- from datetime import datetime

**Internal (eva_ai):**
- from eva_ai.core.core_brain import ЕВАBrain

---

### tests\test_pie_integration.py

**Stdlib:**
- import numpy
- import os
- import os
- import os
- import os

**External:**
- import sys
- import tempfile
- from pathlib import Path
- import traceback
- import tempfile
- import traceback
- import tempfile
- import traceback
- import tempfile
- import traceback
- ... и ещё 2

**Internal (eva_ai):**
- from eva_ai.memory.pie_integration import (
- from eva_ai.core.pie_fallback import (
- from eva_ai.memory.fractal_graph_v2.dual_generator_pie import (
- from eva_ai.memory.pie_integration import create_l1l2_graph
- from eva_ai.memory.pie_integration import create_pie_integration
- from eva_ai.core.pie_fallback import PieFallbackPipeline
- from eva_ai.memory.fractal_graph_v2.dual_generator_pie import DualGeneratorPie

---

### tools\__init__.py

**External:**
- from .document_reader import DocumentTextReader, DocumentContent, read_text_file_simple
- from .import_pipeline import ImportPipeline, ImportedDocument

---

### tools\dependency_scan.py

**Stdlib:**
- import os

**External:**
- from typing import Dict, Set, List, Optional

---

### tools\document_reader.py

**Stdlib:**
- import os
- import logging

**External:**
- from typing import List, Optional, Dict, Any
- from dataclasses import dataclass

---

### tools\import_pipeline.py

**Stdlib:**
- import os
- import logging

**External:**
- from __future__ import annotations
- import io
- import re
- import hashlib
- from dataclasses import dataclass, field
- from typing import Any, Dict, Generator, Iterable, List, Optional
- from pdfminer.high_level import extract_text
- from pdf2image import convert_from_path  # may not be installed
- from ebooklib import epub
- from bs4 import BeautifulSoup  # optional but common

---

### tools\layer_expertise_analysis.py

**Stdlib:**
- import os
- import json
- import logging
- import numpy
- import torch

**External:**
- import sys
- from typing import Dict, List
- from transformers import AutoModelForCausalLM, AutoTokenizer
- from sklearn.preprocessing import StandardScaler
- import argparse

---

### tools\system_generation_analysis.py

**Stdlib:**
- import os
- import json
- import logging

**External:**
- import sys
- import inspect
- import importlib
- import importlib.util
- from datetime import datetime
- from pathlib import Path
- from typing import Dict, List, Optional, Any
- import re
- import traceback

**Internal (eva_ai):**
- from eva_ai.mlearning.optimized_fractal_model_manager import OptimizedFractalModelManager

---

### training\__init__.py

**External:**
- from .gguf_training_system import GGUFTrainingSystem, TrainingStatus, TrainingMetrics, VerifiedKnowledge

---

### training\gguf_training_system.py

**Stdlib:**
- import os
- import logging
- import time
- import json
- import threading

**External:**
- import hashlib
- from typing import Dict, List, Optional, Any, Tuple
- from dataclasses import dataclass, field
- from enum import Enum
- from pathlib import Path
- import shutil
- from llama_cpp import Llama

---

### utils\text_quality.py

**Stdlib:**
- import logging

**External:**
- import re
- from typing import Dict, Any, List, Tuple
- from collections import Counter

---

### websearch\__init__.py

**External:**
- from .search_models import SearchResult, SearchQuery
- from .web_search_engine import WebSearchEngine
- from .database_manager import DatabaseManager
- from .search_engines import SearchEngines
- from .cache_manager import CacheManager

---

### websearch\cache_manager.py

**Stdlib:**
- import os
- import json
- import time
- import logging

**External:**
- import hashlib
- from typing import List, Dict, Any, Optional
- from .search_models import SearchResult

---

### websearch\database_manager.py

**Stdlib:**
- import os
- import sqlite3
- import threading
- import logging

**External:**
- from typing import List, Dict, Any
- from datetime import datetime
- from .search_models import SearchResult

---

### websearch\search_engines.py

**Stdlib:**
- import requests
- import logging
- import json
- import time

**External:**
- import re
- import random
- from typing import List
- from urllib.parse import quote, urljoin
- from bs4 import BeautifulSoup
- from .search_models import SearchResult

---

### websearch\search_models.py

**Stdlib:**
- import time

**External:**
- from typing import List, Optional
- from dataclasses import dataclass, field

---

### websearch\search_types.py

**External:**
- from dataclasses import dataclass, field
- from typing import List, Dict, Any, Optional
- from enum import Enum

---

### websearch\web_search_engine.py

**Stdlib:**
- import os
- import json
- import sqlite3
- import logging
- import time
- import threading

**External:**
- import queue
- from typing import Dict, List, Optional, Any
- from collections import defaultdict
- from concurrent.futures import ThreadPoolExecutor, as_completed
- from .search_models import SearchResult, SearchQuery
- from .database_manager import DatabaseManager
- from .search_engines import SearchEngines
- from .cache_manager import CacheManager
- import heapq

---

### websearch\web_search_integrated.py

**Stdlib:**
- import logging
- import time
- import os
- import json
- import requests
- import threading
- import json
- import json

**External:**
- import re
- import asyncio
- import aiohttp
- from typing import Dict, List, Optional, Any, Tuple
- from datetime import datetime
- from functools import wraps
- import hashlib
- import re

**Internal (eva_ai):**
- from eva_ai.core.base_component import BaseComponent, ComponentState
- from eva_ai.core.event_bus import get_event_bus, Event, EventTypes
- from eva_ai.websearch.web_search_engine import WebSearchEngine

---

