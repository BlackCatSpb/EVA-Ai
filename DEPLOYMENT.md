# EVA AI - Конфигурация домена

## Текущие домены
- eva-agi.ru
- eva-agi.online

## Конфигурация Nginx для production

```nginx
# /etc/nginx/sites-available/eva-agi

server {
    listen 80;
    server_name eva-agi.ru www.eva-agi.ru;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name eva-agi.ru www.eva-agi.ru;

    ssl_certificate /path/to/ssl/cert.pem;
    ssl_certificate_key /path/to/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:5555;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_connect_timeout 60s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
    }

    location /static {
        alias /path/to/CogniFlex/eva/gui/web_gui/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    location /uploads {
        alias /path/to/CogniFlex/eva/gui/web_gui/uploads;
        internal;
    }
}
```

## Команды запуска

### Development
```bash
python -m eva
```

### Production (Gunicorn)
```bash
gunicorn -c gunicorn_config.py wsgi:app
```

### Production (Systemd service)
```bash
# /etc/systemd/system/eva-agi.service
[Unit]
Description=EVA AI Cognitive System
After=network.target

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/path/to/CogniFlex
Environment="PATH=/path/to/venv/bin"
Environment="NO_COLOR=1"
ExecStart=/path/to/venv/bin/gunicorn -c gunicorn_config.py wsgi:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Переменные окружения для production

```bash
export COGNIFLEX_SECRET_KEY="your-secret-key-here"
export COGNIFLEX_ADMIN_USER="admin"
export COGNIFLEX_ADMIN_PASS="secure-password"
export NO_COLOR=1
export PYTHONIOENCODING=utf-8
```
