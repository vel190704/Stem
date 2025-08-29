"""
Test OpenAI API connection
"""
from openai import OpenAI
from config import settings

def test_openai_connection():
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": "Say 'Hello, World!'"}],
            max_tokens=10
        )
        
        print("✅ OpenAI API connection successful!")
        print(f"Model: {settings.OPENAI_MODEL}")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    #!/usr/bin/env python3
"""
Test OpenAI API connection
"""
import openai
import os
from dotenv import load_dotenv

def test_openai_connection():
    """Test if OpenAI API key is working"""
    print("Testing OpenAI API connection...")
    
    # Load environment variables
    load_dotenv()
    
    # Get API key directly from environment
    api_key = os.getenv('OPENAI_API_KEY')
    model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    
    print(f"Using model: {model}")
    print(f"API key format: {api_key[:15]}...{api_key[-10:] if api_key else 'None'}")
    
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Test with a very simple request
        print("Making API request...")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10,
            temperature=0
        )
        
        print("✅ OpenAI API connection successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except openai.AuthenticationError as e:
        print(f"❌ Authentication Error: {e}")
        print("This usually means the API key is invalid or expired.")
        return False
    except openai.RateLimitError as e:
        print(f"❌ Rate Limit Error: {e}")
        return False
    except openai.APIError as e:
        print(f"❌ API Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_openai_connection()
    test_openai_connection()
