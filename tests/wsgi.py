"""
WSGI конфигурация для EVA AI production deployment
"""
import os
import logging

# Set environment variables before any imports
os.environ.setdefault('NO_COLOR', '1')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from eva_ai.gui.web_gui.server import app

if __name__ == '__main__':
    app.run()
