"""
Configuration settings for STEM Assessment Generator
"""
from pydantic_settings import BaseSettings
from pydantic import validator, Field
from typing import Optional
from pathlib import Path
import os

class Settings(BaseSettings):
    """Application settings with robust configuration management"""
    
    # Project root directory (auto-detected)
    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).parent.absolute())
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key (optional for development)")
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # App Configuration
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = True
    
    # File Configuration
    MAX_FILE_SIZE_MB: int = 10
    UPLOAD_DIR: str = "data/uploads"
    VECTORDB_DIR: str = "data/vectordb"
    
    # Processing Configuration
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MAX_QUESTIONS: int = 20
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories_exist()
        self._validate_directories_writable()
    
    @property
    def upload_path(self) -> Path:
        """Get absolute path to uploads directory"""
        return self.PROJECT_ROOT / self.UPLOAD_DIR
    
    @property
    def vectordb_path(self) -> Path:
        """Get absolute path to vector database directory"""
        return self.PROJECT_ROOT / self.VECTORDB_DIR
    
    @validator('MAX_FILE_SIZE_MB')
    def validate_file_size(cls, v):
        """Validate file size is reasonable"""
        if v <= 0 or v > 100:
            raise ValueError("MAX_FILE_SIZE_MB must be between 1 and 100")
        return v
    
    @validator('CHUNK_SIZE')
    def validate_chunk_size(cls, v):
        """Validate chunk size is reasonable"""
        if v <= 0 or v > 2000:
            raise ValueError("CHUNK_SIZE must be between 1 and 2000")
        return v
    
    @validator('CHUNK_OVERLAP')
    def validate_chunk_overlap(cls, v, values):
        """Validate chunk overlap is less than chunk size"""
        chunk_size = values.get('CHUNK_SIZE', 500)
        if v < 0 or v >= chunk_size:
            raise ValueError(f"CHUNK_OVERLAP must be between 0 and {chunk_size-1}")
        return v
    
    @validator('MAX_QUESTIONS')
    def validate_max_questions(cls, v):
        """Validate maximum questions is reasonable"""
        if v <= 0 or v > 50:
            raise ValueError("MAX_QUESTIONS must be between 1 and 50")
        return v
    
    def _ensure_directories_exist(self):
        """Ensure required directories exist"""
        directories = [self.upload_path, self.vectordb_path]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                print(f"✓ Directory ensured: {directory}")
            except Exception as e:
                raise RuntimeError(f"Failed to create directory {directory}: {e}")
    
    def _validate_directories_writable(self):
        """Validate that directories are writable"""
        directories = [self.upload_path, self.vectordb_path]
        
        for directory in directories:
            test_file = directory / ".write_test"
            try:
                # Test write access
                test_file.write_text("test")
                test_file.unlink()  # Remove test file
                print(f"✓ Directory writable: {directory}")
            except Exception as e:
                raise RuntimeError(f"Directory not writable {directory}: {e}")
    
    def is_openai_configured(self) -> bool:
        """Check if OpenAI API key is configured"""
        return bool(self.OPENAI_API_KEY and self.OPENAI_API_KEY.strip())
    
    def validate_openai_key(self) -> str:
        """Validate that OpenAI API key is provided and return it"""
        if not self.is_openai_configured():
            raise ValueError(
                "OPENAI_API_KEY is required. Please set it in your .env file.\n"
                "Get your API key from: https://platform.openai.com/account/api-keys"
            )
        return self.OPENAI_API_KEY
    
    def get_file_size_bytes(self) -> int:
        """Get max file size in bytes"""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024
    
    def __repr__(self):
        """String representation of settings"""
        return (
            f"Settings(\n"
            f"  PROJECT_ROOT='{self.PROJECT_ROOT}'\n"
            f"  OPENAI_CONFIGURED={self.is_openai_configured()}\n"
            f"  UPLOAD_DIR='{self.upload_path}'\n"
            f"  VECTORDB_DIR='{self.vectordb_path}'\n"
            f"  DEBUG={self.DEBUG}\n"
            f")"
        )

# Export single settings instance with error handling
def create_settings():
    """Create settings instance with proper error handling"""
    try:
        # Try to create settings normally
        return Settings()
    except Exception as e:
        print(f"⚠ Configuration warning: {e}")
        # Create minimal settings for development by clearing problematic env vars
        import os
        old_key = os.environ.get('OPENAI_API_KEY')
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
        
        try:
            settings = Settings()
            print("✓ Minimal configuration loaded for development")
            return settings
        finally:
            # Restore original env var if it existed
            if old_key is not None:
                os.environ['OPENAI_API_KEY'] = old_key

settings = create_settings()
