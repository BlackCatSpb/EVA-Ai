#!/usr/bin/env python3
"""
ЕВА Security Framework Tests
Тесты для системы безопасности.
"""

import pytest
import time
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from eva.security.security_framework import (
    SecurityManager,
    AuthenticationManager,
    AuthorizationManager,
    RateLimiter,
    User,
    get_security_manager
)


class TestRateLimiter:
    """Тесты для RateLimiter."""

    def test_rate_limiter_initialization(self):
        """Тест инициализации RateLimiter."""
        limiter = RateLimiter(requests_per_minute=10, burst_limit=3)

        assert limiter.requests_per_minute == 10
        assert limiter.burst_limit == 3
        assert limiter.requests == {}

    @pytest.mark.unit
    def test_rate_limiter_allowed_requests(self):
        """Тест разрешенных запросов."""
        limiter = RateLimiter(requests_per_minute=5, burst_limit=3)

        # Первые запросы должны быть разрешены (до burst limit)
        assert limiter.is_allowed("test_user") is True
        assert limiter.is_allowed("test_user") is True
        assert limiter.is_allowed("test_user") is True  # burst limit = 3

        # Следующий запрос должен быть заблокирован burst limit
        assert limiter.is_allowed("test_user") is False

    def test_rate_limiter_burst_limit(self):
        """Тест burst limit."""
        limiter = RateLimiter(requests_per_minute=10, burst_limit=2)

        # Burst limit - только 2 запроса в секунду
        assert limiter.is_allowed("test_user") is True
        assert limiter.is_allowed("test_user") is True
        assert limiter.is_allowed("test_user") is False  # превышен burst limit

        # Ждем секунду и пробуем снова
        time.sleep(1.1)
        assert limiter.is_allowed("test_user") is True

    def test_rate_limiter_minute_limit(self):
        """Тест минутного лимита."""
        limiter = RateLimiter(requests_per_minute=3, burst_limit=5)

        # Делаем 3 запроса
        assert limiter.is_allowed("test_user") is True
        assert limiter.is_allowed("test_user") is True
        assert limiter.is_allowed("test_user") is True
        assert limiter.is_allowed("test_user") is False  # превышен лимит

    def test_rate_limiter_different_users(self):
        """Тест разных пользователей."""
        limiter = RateLimiter(requests_per_minute=2, burst_limit=2)

        # Каждый пользователь имеет свой лимит
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is False  # user1 превысил

        assert limiter.is_allowed("user2") is True  # user2 еще может
        assert limiter.is_allowed("user2") is True

    def test_rate_limiter_remaining_requests(self):
        """Тест подсчета оставшихся запросов."""
        limiter = RateLimiter(requests_per_minute=5, burst_limit=5)

        # В начале все запросы доступны
        assert limiter.get_remaining_requests("test_user") == 5

        # После использования
        limiter.is_allowed("test_user")
        limiter.is_allowed("test_user")
        assert limiter.get_remaining_requests("test_user") == 3


class TestAuthenticationManager:
    """Тесты для AuthenticationManager."""

    @pytest.fixture
    def auth_manager(self):
        """Создает AuthenticationManager для тестирования."""
        return AuthenticationManager()

    def test_auth_manager_initialization(self, auth_manager):
        """Тест инициализации AuthenticationManager."""
        assert auth_manager.users is not None
        assert "admin" in auth_manager.users
        assert auth_manager.sessions == {}

    def test_create_user(self, auth_manager):
        """Тест создания пользователя."""
        user = auth_manager.create_user("testuser", "testpass", "user")

        assert user is not None
        assert user.username == "testuser"
        assert user.role == "user"
        assert user.is_active is True

        # Проверка что пользователь добавлен
        assert "testuser" in auth_manager.users

    def test_create_duplicate_user(self, auth_manager):
        """Тест создания пользователя с существующим именем."""
        # Создаем первого пользователя
        auth_manager.create_user("testuser", "testpass", "user")

        # Пытаемся создать второго с тем же именем
        user2 = auth_manager.create_user("testuser", "testpass2", "user")

        assert user2 is None

    def test_authenticate_valid_user(self, auth_manager):
        """Тест аутентификации валидного пользователя."""
        # Создаем пользователя
        auth_manager.create_user("testuser", "testpass", "user")

        # Аутентифицируем
        session_token = auth_manager.authenticate("testuser", "testpass")

        assert session_token is not None
        assert session_token in auth_manager.sessions

        # Проверяем сессию
        session = auth_manager.sessions[session_token]
        assert session['username'] == "testuser"
        assert session['role'] == "user"

    def test_authenticate_invalid_credentials(self, auth_manager):
        """Тест аутентификации с неверными данными."""
        # Создаем пользователя
        auth_manager.create_user("testuser", "testpass", "user")

        # Пытаемся аутентифицировать с неверным паролем
        session_token = auth_manager.authenticate("testuser", "wrongpass")

        assert session_token is None

    def test_authenticate_nonexistent_user(self, auth_manager):
        """Тест аутентификации несуществующего пользователя."""
        session_token = auth_manager.authenticate("nonexistent", "testpass")

        assert session_token is None

    def test_validate_session(self, auth_manager):
        """Тест валидации сессии."""
        # Создаем и аутентифицируем пользователя
        auth_manager.create_user("testuser", "testpass", "user")
        session_token = auth_manager.authenticate("testuser", "testpass")

        # Валидируем сессию
        session = auth_manager.validate_session(session_token)

        assert session is not None
        assert session['username'] == "testuser"

    def test_validate_expired_session(self, auth_manager):
        """Тест валидации истекшей сессии."""
        # Создаем сессию с истекшим сроком
        expired_session = {
            'user_id': 'testuser',
            'username': 'testuser',
            'role': 'user',
            'created_at': datetime.now(),
            'expires_at': datetime.now() - timedelta(hours=1)  # Уже истекла
        }
        auth_manager.sessions['expired_token'] = expired_session

        # Проверяем что сессия считается истекшей
        session = auth_manager.validate_session('expired_token')

        assert session is None
        assert 'expired_token' not in auth_manager.sessions

    def test_logout(self, auth_manager):
        """Тест выхода из системы."""
        # Создаем и аутентифицируем пользователя
        auth_manager.create_user("testuser", "testpass", "user")
        session_token = auth_manager.authenticate("testuser", "testpass")

        # Выходим из системы
        result = auth_manager.logout(session_token)

        assert result is True
        assert session_token not in auth_manager.sessions

    def test_logout_invalid_session(self, auth_manager):
        """Тест выхода с невалидным токеном."""
        result = auth_manager.logout("invalid_token")

        assert result is False


class TestAuthorizationManager:
    """Тесты для AuthorizationManager."""

    @pytest.fixture
    def authz_manager(self):
        """Создает AuthorizationManager для тестирования."""
        return AuthorizationManager()

    def test_check_permission_admin(self, authz_manager):
        """Тест разрешений администратора."""
        assert authz_manager.check_permission("admin", "read") is True
        assert authz_manager.check_permission("admin", "write") is True
        assert authz_manager.check_permission("admin", "delete") is True
        assert authz_manager.check_permission("admin", "admin") is True
        assert authz_manager.check_permission("admin", "system") is True

    def test_check_permission_user(self, authz_manager):
        """Тест разрешений обычного пользователя."""
        assert authz_manager.check_permission("user", "read") is True
        assert authz_manager.check_permission("user", "write") is True
        assert authz_manager.check_permission("user", "chat") is True
        assert authz_manager.check_permission("user", "delete") is False
        assert authz_manager.check_permission("user", "admin") is False

    def test_check_permission_guest(self, authz_manager):
        """Тест разрешений гостя."""
        assert authz_manager.check_permission("guest", "read") is True
        assert authz_manager.check_permission("guest", "chat") is True
        assert authz_manager.check_permission("guest", "write") is False
        assert authz_manager.check_permission("guest", "delete") is False
        assert authz_manager.check_permission("guest", "admin") is False

    def test_check_permission_unknown_role(self, authz_manager):
        """Тест неизвестной роли."""
        assert authz_manager.check_permission("unknown", "read") is False
        assert authz_manager.check_permission("unknown", "admin") is False

    def test_get_user_permissions(self, authz_manager):
        """Тест получения разрешений пользователя."""
        admin_permissions = authz_manager.get_user_permissions("admin")
        user_permissions = authz_manager.get_user_permissions("user")
        guest_permissions = authz_manager.get_user_permissions("guest")
        unknown_permissions = authz_manager.get_user_permissions("unknown")

        assert "read" in admin_permissions
        assert "write" in admin_permissions
        assert "admin" in admin_permissions

        assert "read" in user_permissions
        assert "write" in user_permissions
        assert "admin" not in user_permissions

        assert "read" in guest_permissions
        assert "write" not in guest_permissions

        assert unknown_permissions == []


class TestSecurityManager:
    """Тесты для SecurityManager."""

    @pytest.fixture
    def security_mgr(self):
        """Создает SecurityManager для тестирования."""
        return SecurityManager()

    def test_security_manager_initialization(self, security_mgr):
        """Тест инициализации SecurityManager."""
        assert security_mgr.auth_manager is not None
        assert security_mgr.authz_manager is not None
        assert security_mgr.rate_limiter is not None
        assert security_mgr.event_log == []

    def test_authenticate_request_success(self, security_mgr):
        """Тест успешной аутентификации запроса."""
        # Создаем пользователя
        security_mgr.auth_manager.create_user("testuser", "testpass", "user")

        # Аутентифицируем запрос
        session_token, result = security_mgr.authenticate_request(
            "testuser", "testpass", "127.0.0.1", "test-agent"
        )

        assert session_token is not None
        assert result == "success"

        # Проверяем логирование события
        assert len(security_mgr.event_log) > 0
        assert security_mgr.event_log[-1].event_type == "authentication_success"

    def test_authenticate_request_invalid_credentials(self, security_mgr):
        """Тест аутентификации с неверными данными."""
        # Создаем пользователя
        security_mgr.auth_manager.create_user("testuser", "testpass", "user")

        # Пытаемся аутентифицировать с неверным паролем
        session_token, result = security_mgr.authenticate_request(
            "testuser", "wrongpass", "127.0.0.1", "test-agent"
        )

        assert session_token is None
        assert result == "authentication_failed"

        # Проверяем логирование события
        assert len(security_mgr.event_log) > 0
        assert security_mgr.event_log[-1].event_type == "authentication_failed"

    def test_authenticate_request_rate_limit(self, security_mgr):
        """Тест rate limiting при аутентификации."""
        # Устанавливаем низкий лимит для теста
        security_mgr.rate_limiter = RateLimiter(requests_per_minute=1, burst_limit=1)

        # Создаем пользователя
        security_mgr.auth_manager.create_user("testuser", "testpass", "user")

        # Первый запрос - успешен
        session_token1, result1 = security_mgr.authenticate_request(
            "testuser", "testpass", "127.0.0.1", "test-agent"
        )
        assert result1 == "success"

        # Второй запрос - rate limit
        session_token2, result2 = security_mgr.authenticate_request(
            "testuser", "testpass", "127.0.0.1", "test-agent"
        )
        assert result2 == "rate_limit_exceeded"

    def test_authorize_request_success(self, security_mgr):
        """Тест успешной авторизации запроса."""
        # Создаем и аутентифицируем пользователя
        security_mgr.auth_manager.create_user("testuser", "testpass", "user")
        session_token, _ = security_mgr.authenticate_request(
            "testuser", "testpass", "127.0.0.1", "test-agent"
        )

        # Авторизуем запрос
        authorized, result = security_mgr.authorize_request(
            session_token, "read", "127.0.0.1", "test-agent"
        )

        assert authorized is True
        assert result == "success"

    def test_authorize_request_insufficient_permissions(self, security_mgr):
        """Тест авторизации с недостаточными правами."""
        # Создаем пользователя без прав admin
        security_mgr.auth_manager.create_user("testuser", "testpass", "user")
        session_token, _ = security_mgr.authenticate_request(
            "testuser", "testpass", "127.0.0.1", "test-agent"
        )

        # Пытаемся выполнить admin операцию
        authorized, result = security_mgr.authorize_request(
            session_token, "admin", "127.0.0.1", "test-agent"
        )

        assert authorized is False
        assert result == "insufficient_permissions"

    def test_authorize_request_invalid_session(self, security_mgr):
        """Тест авторизации с невалидной сессией."""
        authorized, result = security_mgr.authorize_request(
            "invalid_token", "read", "127.0.0.1", "test-agent"
        )

        assert authorized is False
        assert result == "invalid_session"

    def test_validate_request_guest_access(self, security_mgr):
        """Тест валидации гостевого доступа."""
        authorized, result = security_mgr.validate_request(
            None, "guest", "127.0.0.1", "test-agent"
        )

        assert authorized is True
        assert result == "success"

    def test_validate_request_authenticated_access(self, security_mgr):
        """Тест валидации аутентифицированного доступа."""
        # Создаем и аутентифицируем пользователя
        security_mgr.auth_manager.create_user("testuser", "testpass", "user")
        session_token, _ = security_mgr.authenticate_request(
            "testuser", "testpass", "127.0.0.1", "test-agent"
        )

        # Проверяем доступ
        authorized, result = security_mgr.validate_request(
            session_token, "read", "127.0.0.1", "test-agent"
        )

        assert authorized is True
        assert result == "success"

    def test_security_events_logging(self, security_mgr):
        """Тест логирования событий безопасности."""
        initial_events = len(security_mgr.event_log)

        # Создаем пользователя и аутентифицируем
        security_mgr.auth_manager.create_user("testuser", "testpass", "user")
        security_mgr.authenticate_request(
            "testuser", "testpass", "127.0.0.1", "test-agent"
        )

        # Проверяем что события логируются
        assert len(security_mgr.event_log) > initial_events

        # Проверяем структуру события
        event = security_mgr.event_log[-1]
        assert hasattr(event, 'event_type')
        assert hasattr(event, 'user_id')
        assert hasattr(event, 'ip_address')
        assert hasattr(event, 'timestamp')

    def test_get_security_events(self, security_mgr):
        """Тест получения событий безопасности."""
        # Генерируем несколько событий
        security_mgr.auth_manager.create_user("testuser", "testpass", "user")
        security_mgr.authenticate_request("testuser", "testpass", "127.0.0.1", "test-agent")
        security_mgr.authenticate_request("testuser", "wrongpass", "127.0.0.1", "test-agent")

        events = security_mgr.get_security_events(limit=10)

        assert len(events) >= 2
        assert all(hasattr(event, 'event_type') for event in events)

    def test_rate_limit_status(self, security_mgr):
        """Тест получения статуса rate limit."""
        status = security_mgr.get_rate_limit_status("test_user")

        assert "remaining_requests" in status
        assert "total_limit" in status
        assert "is_allowed" in status
        assert status["remaining_requests"] > 0
        assert status["is_allowed"] is True


class TestSecurityIntegration:
    """Интеграционные тесты безопасности."""

    @pytest.fixture
    def integrated_security(self):
        """Создает полностью настроенную систему безопасности."""
        return get_security_manager()

    def test_full_authentication_flow(self, integrated_security):
        """Тест полного цикла аутентификации."""
        # 1. Создаем пользователя
        user = integrated_security.auth_manager.create_user("testuser", "testpass", "user")
        assert user is not None

        # 2. Аутентифицируем
        session_token, auth_result = integrated_security.authenticate_request(
            "testuser", "testpass", "127.0.0.1", "test-agent"
        )
        assert session_token is not None
        assert auth_result == "success"

        # 3. Авторизуем для чтения
        authorized, authz_result = integrated_security.authorize_request(
            session_token, "read", "127.0.0.1", "test-agent"
        )
        assert authorized is True
        assert authz_result == "success"

        # 4. Комплексная валидация
        valid, valid_result = integrated_security.validate_request(
            session_token, "read", "127.0.0.1", "test-agent"
        )
        assert valid is True
        assert valid_result == "success"

        # 5. Выход из системы
        logout_result = integrated_security.auth_manager.logout(session_token)
        assert logout_result is True

    def test_security_event_sequence(self, integrated_security):
        """Тест последовательности событий безопасности."""
        initial_events = len(integrated_security.event_log)

        # Создаем пользователя
        integrated_security.auth_manager.create_user("testuser", "testpass", "user")

        # Успешная аутентификация
        integrated_security.authenticate_request(
            "testuser", "testpass", "127.0.0.1", "test-agent"
        )

        # Неудачная аутентификация
        integrated_security.authenticate_request(
            "testuser", "wrongpass", "127.0.0.1", "test-agent"
        )

        # Проверяем события
        events = integrated_security.get_security_events()
        new_events = len(events) - initial_events

        assert new_events >= 2  # Минимум 2 события

        event_types = [event.event_type for event in events[-new_events:]]
        assert "authentication_success" in event_types
        assert "authentication_failed" in event_types


# Тесты для декораторов безопасности
class TestSecurityDecorators:
    """Тесты для декораторов безопасности."""

    def test_require_authentication_decorator(self):
        """Тест декоратора require_authentication."""
        from eva.security.security_framework import require_authentication

        @require_authentication("read")
        def protected_function(session_token=None, **kwargs):
            return "success"

        # Тест без токена - должно выбросить исключение
        with pytest.raises(PermissionError):
            protected_function()

        # Тест с токеном (mock)
        security_mgr = get_security_manager()
        security_mgr.auth_manager.create_user("testuser", "testpass", "user")
        session_token, _ = security_mgr.authenticate_request(
            "testuser", "testpass", "127.0.0.1", "test-agent"
        )

        result = protected_function(session_token=session_token)
        assert result == "success"

    def test_log_security_event_function(self):
        """Тест функции логирования событий безопасности."""
        from eva.security.security_framework import log_security_event

        security_mgr = get_security_manager()
        initial_events = len(security_mgr.event_log)

        # Логируем событие
        log_security_event(
            "test_event",
            user_id="testuser",
            ip_address="127.0.0.1",
            user_agent="test-agent",
            details={"test": "data"}
        )

        # Проверяем что событие добавлено
        assert len(security_mgr.event_log) > initial_events
        event = security_mgr.event_log[-1]
        assert event.event_type == "test_event"
        assert event.user_id == "testuser"
        assert event.details["test"] == "data"


if __name__ == "__main__":
    print("🚀 Запуск тестов системы безопасности ЕВА...")

    # Можно запускать тесты напрямую
    import subprocess
    result = subprocess.run([
        'python', '-m', 'pytest',
        __file__,
        '-v',
        '--tb=short'
    ], capture_output=True, text=True)

    print("Вывод тестов:")
    print(result.stdout)
    if result.stderr:
        print("Ошибки:")
        print(result.stderr)

    print(f"Код завершения: {result.returncode}")

    print("✅ Тесты безопасности завершены!")
