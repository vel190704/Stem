#!/usr/bin/env python3
"""
Check API key format and characters
"""
import os
from dotenv import load_dotenv

def check_api_key():
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    print("=== API Key Analysis ===")
    print(f"Length: {len(api_key)}")
    print(f"First 20 chars: {repr(api_key[:20])}")
    print(f"Last 20 chars: {repr(api_key[-20:])}")
    
    # Check for unusual characters
    for i, char in enumerate(api_key):
        if ord(char) > 127 or ord(char) < 32:
            print(f"Unusual character at position {i}: {repr(char)} (ASCII: {ord(char)})")
    
    # Check if it matches expected pattern
    if api_key.startswith('sk-proj-'):
        print("✓ Starts with correct prefix")
    else:
        print("✗ Incorrect prefix")
    
    # Check length (typical OpenAI project keys are around 164 characters)
    if 160 <= len(api_key) <= 170:
        print("✓ Length seems reasonable")
    else:
        print("✗ Unusual length")

if __name__ == "__main__":
    check_api_key()
