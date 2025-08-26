"""
Configuration settings for STEM Assessment Generator
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # App Configuration
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = True
    
    # File Configuration
    MAX_FILE_SIZE_MB: int = 10
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # Paths
    UPLOAD_DIR: str = "data/uploads"
    VECTOR_DB_DIR: str = "data/vectordb"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
