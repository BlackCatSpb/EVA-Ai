"""
Модуль запросов графа знаний для ЕВА
Содержит методы поиска, фильтрации и получения данных из графа
"""
import os
import logging
import time
import sqlite3
import json
import hashlib
import math
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from .query_core import KnowledgeGraphQueryMixin as _QueryCore
from .query_search import KnowledgeGraphQuerySearch as _QuerySearch
from .query_traversal import KnowledgeGraphQueryTraversal as _QueryTraversal
from .query_analytics import KnowledgeGraphQueryAnalytics as _QueryAnalytics

logger = logging.getLogger("eva.knowledge_graph")


class KnowledgeGraphQueryMixin(_QueryCore, _QuerySearch, _QueryTraversal, _QueryAnalytics):
    """Mixin класс с методами запросов для KnowledgeGraph."""
    pass
