"""
Custom exception classes for STEM Assessment Generator
Provides user-friendly error messages and recovery suggestions
"""
from typing import Optional, Dict, Any
import traceback

class BaseAssessmentError(Exception):
    """Base exception class for all assessment generator errors"""
    
    def __init__(
        self, 
        message: str = None,
        user_message: str = None,
        technical_details: str = None,
        recovery_action: str = None,
        error_code: str = None
    ):
        self.message = message or self.default_message
        self.user_message = user_message or self.default_user_message
        self.technical_details = technical_details
        self.recovery_action = recovery_action or self.default_recovery_action
        self.error_code = error_code or self.default_error_code
        super().__init__(self.message)
    
    # Default values to be overridden by subclasses
    default_message = "An error occurred in the assessment generator"
    default_user_message = "Something went wrong. Please try again."
    default_recovery_action = "Contact support if the problem persists"
    default_error_code = "GENERAL_ERROR"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses"""
        return {
            "error_code": self.error_code,
            "message": self.user_message,
            "technical_details": self.technical_details,
            "recovery_action": self.recovery_action,
            "traceback": traceback.format_exc() if self.technical_details else None
        }

# =============================================================================
# PDF Processing Errors
# =============================================================================

class PDFProcessingError(BaseAssessmentError):
    """PDF processing related errors"""
    default_message = "Failed to process PDF file"
    default_user_message = "Unable to read the PDF file. Please check if the file is valid and not password-protected."
    default_recovery_action = "Try a different PDF file or ensure the file is not corrupted"
    default_error_code = "PDF_PROCESSING_ERROR"

class PDFCorruptedError(PDFProcessingError):
    """PDF file is corrupted or unreadable"""
    default_user_message = "The PDF file appears to be corrupted or damaged."
    default_recovery_action = "Try uploading a different PDF file"
    default_error_code = "PDF_CORRUPTED"

class PDFPasswordProtectedError(PDFProcessingError):
    """PDF is password protected"""
    default_user_message = "This PDF is password-protected. Please use an unprotected version."
    default_recovery_action = "Remove password protection or use a different PDF"
    default_error_code = "PDF_PASSWORD_PROTECTED"

class PDFTooLargeError(PDFProcessingError):
    """PDF file is too large to process"""
    default_user_message = "The PDF file is too large to process."
    default_recovery_action = "Try a smaller PDF file (under 10MB) or split large documents"
    default_error_code = "PDF_TOO_LARGE"

class PDFNoTextError(PDFProcessingError):
    """PDF contains no extractable text"""
    default_user_message = "This PDF doesn't contain readable text. It may be image-based."
    default_recovery_action = "Use a PDF with text content or try OCR conversion first"
    default_error_code = "PDF_NO_TEXT"

# =============================================================================
# OpenAI API Errors
# =============================================================================

class OpenAIError(BaseAssessmentError):
    """OpenAI API related errors"""
    default_message = "OpenAI API error occurred"
    default_user_message = "There was an issue with the AI service. Please try again."
    default_recovery_action = "Wait a moment and try again, or check your API configuration"
    default_error_code = "OPENAI_ERROR"

class OpenAIKeyError(OpenAIError):
    """OpenAI API key is invalid or missing"""
    default_user_message = "OpenAI API key is not configured or invalid."
    default_recovery_action = "Set a valid OPENAI_API_KEY in your environment variables"
    default_error_code = "OPENAI_KEY_ERROR"

class OpenAIQuotaError(OpenAIError):
    """OpenAI API quota exceeded"""
    default_user_message = "API usage limit reached. Please try again later."
    default_recovery_action = "Wait for quota reset or upgrade your OpenAI plan"
    default_error_code = "OPENAI_QUOTA_EXCEEDED"

class OpenAIRateLimitError(OpenAIError):
    """OpenAI API rate limit exceeded"""
    default_user_message = "Too many requests. Please wait a moment before trying again."
    default_recovery_action = "Wait 60 seconds and retry your request"
    default_error_code = "OPENAI_RATE_LIMIT"

# =============================================================================
# Question Generation Errors
# =============================================================================

class QuestionGenerationError(BaseAssessmentError):
    """Question generation related errors"""
    default_message = "Failed to generate questions"
    default_user_message = "Unable to generate questions from this content."
    default_recovery_action = "Try with different content or reduce the number of questions"
    default_error_code = "QUESTION_GENERATION_ERROR"

class InsufficientContentError(QuestionGenerationError):
    """Not enough content to generate requested questions"""
    default_user_message = "Not enough content to generate the requested number of questions."
    default_recovery_action = "Try a longer document or request fewer questions"
    default_error_code = "INSUFFICIENT_CONTENT"

class QualityValidationError(QuestionGenerationError):
    """Generated questions don't meet quality standards"""
    default_user_message = "Generated questions don't meet quality standards."
    default_recovery_action = "Try regenerating with different settings"
    default_error_code = "QUALITY_VALIDATION_FAILED"

class NoValidQuestionsError(QuestionGenerationError):
    """No valid questions could be generated"""
    default_user_message = "Unable to generate valid questions from this content."
    default_recovery_action = "Try different content or check if the PDF contains educational material"
    default_error_code = "NO_VALID_QUESTIONS"

# =============================================================================
# Vector Database Errors
# =============================================================================

class VectorDatabaseError(BaseAssessmentError):
    """Vector database related errors"""
    default_message = "Vector database error"
    default_user_message = "There was an issue with the document storage system."
    default_recovery_action = "Try restarting the service or contact support"
    default_error_code = "VECTOR_DB_ERROR"

class EmbeddingError(VectorDatabaseError):
    """Error generating embeddings"""
    default_user_message = "Failed to process document for similarity search."
    default_recovery_action = "Try a different document or check your internet connection"
    default_error_code = "EMBEDDING_ERROR"

# =============================================================================
# File System Errors
# =============================================================================

class FileSystemError(BaseAssessmentError):
    """File system related errors"""
    default_message = "File system error"
    default_user_message = "Unable to save or access files on the server."
    default_recovery_action = "Contact administrator to check disk space and permissions"
    default_error_code = "FILESYSTEM_ERROR"

class InsufficientStorageError(FileSystemError):
    """Insufficient disk space"""
    default_user_message = "Server is running low on storage space."
    default_recovery_action = "Contact administrator to free up disk space"
    default_error_code = "INSUFFICIENT_STORAGE"

class PermissionError(FileSystemError):
    """Permission denied for file operations"""
    default_user_message = "Server doesn't have permission to save files."
    default_recovery_action = "Contact administrator to fix file permissions"
    default_error_code = "PERMISSION_DENIED"

# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(BaseAssessmentError):
    """Configuration related errors"""
    default_message = "Configuration error"
    default_user_message = "The service is not properly configured."
    default_recovery_action = "Contact administrator to check service configuration"
    default_error_code = "CONFIGURATION_ERROR"

class DependencyError(ConfigurationError):
    """Missing or incompatible dependencies"""
    default_user_message = "Some required components are missing or incompatible."
    default_recovery_action = "Contact administrator to update service dependencies"
    default_error_code = "DEPENDENCY_ERROR"

# =============================================================================
# Export Errors
# =============================================================================

class ExportError(BaseAssessmentError):
    """Export related errors"""
    default_message = "Export failed"
    default_user_message = "Failed to export assessment in the requested format."
    default_recovery_action = "Try a different export format or contact support"
    default_error_code = "EXPORT_ERROR"

class UnsupportedFormatError(ExportError):
    """Unsupported export format"""
    default_user_message = "The requested export format is not supported."
    default_recovery_action = "Choose a different export format (PDF, JSON, DOCX, or TXT)"
    default_error_code = "UNSUPPORTED_FORMAT"

# =============================================================================
# Utility Functions
# =============================================================================

def create_error_response(error: BaseAssessmentError, include_technical: bool = False) -> Dict[str, Any]:
    """Create standardized error response for API"""
    response = {
        "success": False,
        "error": {
            "code": error.error_code,
            "message": error.user_message,
            "recovery_action": error.recovery_action
        }
    }
    
    if include_technical and error.technical_details:
        response["error"]["technical_details"] = error.technical_details
    
    return response

def wrap_api_errors(func):
    """Decorator to wrap API endpoints with error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BaseAssessmentError as e:
            return create_error_response(e)
        except Exception as e:
            # Wrap unexpected errors
            wrapped_error = BaseAssessmentError(
                message=str(e),
                user_message="An unexpected error occurred. Please try again.",
                technical_details=str(e),
                error_code="UNEXPECTED_ERROR"
            )
            return create_error_response(wrapped_error)
    return wrapper

# =============================================================================
# Error Context Manager
# =============================================================================

class ErrorContext:
    """Context manager for adding context to errors"""
    
    def __init__(self, operation: str, details: Dict[str, Any] = None):
        self.operation = operation
        self.details = details or {}
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and issubclass(exc_type, BaseAssessmentError):
            # Add context to existing custom errors
            if not exc_val.technical_details:
                exc_val.technical_details = f"During {self.operation}: {str(exc_val)}"
        elif exc_type:
            # Wrap other exceptions with context
            context_error = BaseAssessmentError(
                message=f"Error during {self.operation}: {str(exc_val)}",
                technical_details=f"Operation: {self.operation}, Details: {self.details}",
                error_code="OPERATION_FAILED"
            )
            raise context_error from exc_val
        return False
