import os
from eva_ai.core.brain_config import load_brain_config

config = load_brain_config()

tavily_key = config.get('web_search', {}).get('tavily_api_key') or os.environ.get('TAVILY_API_KEY')
status = 'SET' if tavily_key else 'NOT SET'
print(f'Tavily API Key: {status}')

web_config = config.get('web_search', {})
print(f'Web Search Enabled: {web_config.get("enabled", "not set")}')
print(f'Web Search Config: {web_config}')
