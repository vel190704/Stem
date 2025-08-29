#!/usr/bin/env python3
"""
Debug script to check .env file loading
"""
import os
from dotenv import load_dotenv

print("=== Debugging .env file loading ===")

# Load .env file
load_dotenv()

# Check if API key is loaded
api_key = os.getenv('OPENAI_API_KEY')
print(f"API key loaded: {api_key is not None}")
print(f"API key length: {len(api_key) if api_key else 0}")
print(f"API key starts with: {api_key[:10] if api_key else 'None'}...")
print(f"API key ends with: {api_key[-10:] if api_key else 'None'}")

# Check for hidden characters
if api_key:
    print(f"API key repr: {repr(api_key)}")
    
# Also check the config loading
try:
    from config import settings
    print(f"Config API key loaded: {settings.OPENAI_API_KEY is not None}")
    print(f"Config API key length: {len(settings.OPENAI_API_KEY)}")
    print(f"Config API key starts with: {settings.OPENAI_API_KEY[:10]}...")
    print(f"Config API key ends with: {settings.OPENAI_API_KEY[-10:]}")
except Exception as e:
    print(f"Error loading config: {e}")
