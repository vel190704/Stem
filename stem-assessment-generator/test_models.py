#!/usr/bin/env python3
"""
Test different OpenAI models and endpoints
"""
import openai
import os
from dotenv import load_dotenv

def test_different_endpoints():
    """Test different models and endpoints"""
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    
    # Test different models
    models_to_test = [
        'gpt-3.5-turbo',
        'gpt-4',
        'gpt-4o-mini'
    ]
    
    client = openai.OpenAI(api_key=api_key)
    
    print("Testing different models...")
    
    for model in models_to_test:
        print(f"\n--- Testing {model} ---")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            print(f"✅ {model} works!")
            print(f"Response: {response.choices[0].message.content}")
            break  # If one works, we're good
            
        except openai.AuthenticationError:
            print(f"❌ {model}: Authentication failed")
        except openai.NotFoundError:
            print(f"❌ {model}: Model not found or no access")
        except openai.PermissionDeniedError:
            print(f"❌ {model}: Permission denied")
        except Exception as e:
            print(f"❌ {model}: {e}")
    
    # Test embeddings
    print(f"\n--- Testing Embeddings ---")
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="test text"
        )
        print("✅ Embeddings work!")
    except Exception as e:
        print(f"❌ Embeddings failed: {e}")

if __name__ == "__main__":
    test_different_endpoints()
