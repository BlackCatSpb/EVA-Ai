"""
Типы для Integration Layer
Часть модуля integration_layer.py (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class IntegrationType(Enum):
    """Типы интеграции."""
    SYNC = "sync"
    ASYNC = "async"
    EVENT_DRIVEN = "event_driven"
    STREAM = "stream"


class ConnectionStatus(Enum):
    """Статусы подключения."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    PENDING = "pending"
    ERROR = "error"


@dataclass
class IntegrationEndpoint:
    """Точка интеграции."""
    endpoint_id: str
    service_name: str
    url: str
    integration_type: IntegrationType
    status: ConnectionStatus
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "service_name": self.service_name,
            "url": self.url,
            "integration_type": self.integration_type.value if isinstance(self.integration_type, IntegrationType) else self.integration_type,
            "status": self.status.value if isinstance(self.status, ConnectionStatus) else self.status
        }


@dataclass
class IntegrationConfig:
    """Конфигурация интеграции."""
    timeout: int = 30
    retry_count: int = 3
    batch_size: int = 10
    enable_caching: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "batch_size": self.batch_size,
            "enable_caching": self.enable_caching
        }
