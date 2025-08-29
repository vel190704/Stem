#!/usr/bin/env python3
"""
Test the updated configuration system
"""
from config import settings

def test_config():
    """Test the configuration system"""
    print("=== Testing Configuration System ===")
    
    # Print configuration overview
    print(f"\n{settings}")
    
    # Test OpenAI configuration
    print(f"\n--- OpenAI Configuration ---")
    print(f"OpenAI Configured: {settings.is_openai_configured()}")
    print(f"Model: {settings.OPENAI_MODEL}")
    print(f"Embedding Model: {settings.OPENAI_EMBEDDING_MODEL}")
    
    # Test paths
    print(f"\n--- Path Configuration ---")
    print(f"Project Root: {settings.PROJECT_ROOT}")
    print(f"Upload Path: {settings.upload_path}")
    print(f"Vector DB Path: {settings.vectordb_path}")
    print(f"Upload path exists: {settings.upload_path.exists()}")
    print(f"Vector DB path exists: {settings.vectordb_path.exists()}")
    
    # Test file configuration
    print(f"\n--- File Configuration ---")
    print(f"Max File Size: {settings.MAX_FILE_SIZE_MB} MB ({settings.get_file_size_bytes()} bytes)")
    print(f"Chunk Size: {settings.CHUNK_SIZE}")
    print(f"Chunk Overlap: {settings.CHUNK_OVERLAP}")
    print(f"Max Questions: {settings.MAX_QUESTIONS}")
    
    # Test app configuration
    print(f"\n--- App Configuration ---")
    print(f"Host: {settings.APP_HOST}")
    print(f"Port: {settings.APP_PORT}")
    print(f"Debug: {settings.DEBUG}")
    
    # Test OpenAI validation
    print(f"\n--- OpenAI Validation ---")
    try:
        if settings.is_openai_configured():
            key = settings.validate_openai_key()
            print(f"✓ OpenAI key validated: {key[:15]}...{key[-10:]}")
        else:
            print("⚠ OpenAI key not configured")
    except Exception as e:
        print(f"❌ OpenAI validation error: {e}")

if __name__ == "__main__":
    test_config()
