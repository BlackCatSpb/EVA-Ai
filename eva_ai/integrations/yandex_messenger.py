"""Интеграция с Яндекс Мессенджером для самообучения EVA"""
import logging
import requests
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger("eva_ai.yandex_messenger")


class YandexMessengerConnector:
    """Коннектор к Яндекс Мессенджеру для самообучения"""
    
    def __init__(self, oauth_token: str, enabled: bool = False):
        """
        Инициализация коннектора.
        
        Args:
            oauth_token: OAuth токен бота из Яндекс.Мессенджера
            enabled: Включить интеграцию
        """
        self.oauth_token = oauth_token
        self.enabled = enabled
        self.base_url = "https://dialogs.yandex.net/api/v1"
        self.headers = {
            "Authorization": f"OAuth {oauth_token}",
            "Content-Type": "application/json"
        }
        
        # ID чата для самообучения (будет определён при инициализации)
        self.learning_chat_id: Optional[str] = None
        
        logger.info(f"YandexMessengerConnector: enabled={enabled}")
    
    def is_available(self) -> bool:
        """Проверяет доступность API"""
        if not self.enabled or not self.oauth_token:
            return False
        
        try:
            r = requests.get(f"{self.base_url}/about", headers=self.headers, timeout=10)
            return r.status_code == 200
        except Exception as e:
            logger.warning(f"Yandex Messenger API unavailable: {e}")
            return False
    
    def get_chats(self) -> List[Dict[str, Any]]:
        """Получает список чатов"""
        if not self.is_available():
            return []
        
        try:
            r = requests.get(f"{self.base_url}/chats", headers=self.headers, timeout=15)
            if r.status_code == 200:
                return r.json().get("chats", [])
        except Exception as e:
            logger.error(f"Error getting chats: {e}")
        return []
    
    def send_message(self, text: str, chat_id: Optional[str] = None) -> bool:
        """Отправляет сообщение в чат"""
        if not self.is_available() or not chat_id:
            return False
        
        try:
            payload = {"text": text}
            r = requests.post(
                f"{self.base_url}/chats/{chat_id}/messages",
                json=payload,
                headers=self.headers,
                timeout=15
            )
            return r.status_code == 200
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    def get_messages(self, chat_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Получает сообщения из чата"""
        if not self.is_available():
            return []
        
        try:
            r = requests.get(
                f"{self.base_url}/chats/{chat_id}/messages",
                params={"limit": limit},
                headers=self.headers,
                timeout=15
            )
            if r.status_code == 200:
                return r.json().get("messages", [])
        except Exception as e:
            logger.error(f"Error getting messages: {e}")
        return []
    
    def poll_events(self, timeout: int = 30) -> List[Dict[str, Any]]:
        """Long polling для получения новых событий"""
        if not self.is_available():
            return []
        
        try:
            r = requests.get(
                f"{self.base_url}/events",
                params={"timeout": timeout},
                headers=self.headers,
                timeout=timeout + 5
            )
            if r.status_code == 200:
                return r.json().get("events", [])
        except Exception as e:
            logger.debug(f"Poll timeout or error: {e}")
        return []
    
    def setup_self_learning_chat(self, chat_name: str = "EVA Самообучение") -> Optional[str]:
        """Находит или создаёт чат для самообучения"""
        chats = self.get_chats()
        
        for chat in chats:
            if chat.get("name") == chat_name:
                self.learning_chat_id = chat.get("chat_id")
                logger.info(f"Found learning chat: {chat_name} ({self.learning_chat_id})")
                return self.learning_chat_id
        
        logger.warning(f"Chat '{chat_name}' not found. Available: {[c.get('name') for c in chats]}")
        return None
    
    def ask_question(self, question: str) -> bool:
        """Ева задаёт вопрос в чате самообучения"""
        if not self.learning_chat_id:
            return False
        
        formatted = f"🤔 Вопрос для изучения: {question}"
        return self.send_message(formatted, self.learning_chat_id)
    
    def get_answers(self) -> List[str]:
        """Получает ответы из чата самообучения"""
        if not self.learning_chat_id:
            return []
        
        messages = self.get_messages(self.learning_chat_id, limit=20)
        return [m.get("text", "") for m in messages if m.get("text")]
    
    def start_self_learning_mode(self) -> bool:
        """Запускает режим самообучения"""
        if not self.is_available():
            logger.warning("Cannot start self-learning: API unavailable")
            return False
        
        self.setup_self_learning_chat()
        if self.learning_chat_id:
            logger.info(f"Self-learning mode active in chat {self.learning_chat_id}")
            return True
        
        logger.warning("Self-learning chat not found")
        return False
    
    def check_for_new_knowledge(self) -> List[str]:
        """Проверяет новые знания в чате"""
        if not self.enabled or not self.learning_chat_id:
            return []
        
        messages = self.get_messages(self.learning_chat_id, limit=5)
        return [m.get("text", "") for m in messages if m.get("text")]


def create_yandex_messenger(config: Dict[str, Any]) -> YandexMessengerConnector:
    """Создаёт коннектор из конфига"""
    messenger_config = config.get("yandex_messenger", {})
    
    return YandexMessengerConnector(
        oauth_token=messenger_config.get("oauth_token", ""),
        enabled=messenger_config.get("enabled", False)
    )