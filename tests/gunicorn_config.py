"""
Gunicorn configuration for EVA AI production deployment
"""
import os
import multiprocessing

# Server socket
bind = "0.0.0.0:5555"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 120
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = "eva-agi"

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
keyfile = None
certfile = None

# Preload app for faster worker spawn
preload_app = True

# Environment
raw_env = [
    'NO_COLOR=1',
    'PYTHONIOENCODING=utf-8',
]
