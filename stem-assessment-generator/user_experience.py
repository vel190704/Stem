"""
User Experience Enhancement Module
Provides file validation, progress estimation, and user guidance
"""
import mimetypes
import magic
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import hashlib
import logging

from config import settings
from exceptions import PDFProcessingError, PDFTooLargeError, PDFPasswordProtectedError

logger = logging.getLogger(__name__)

class FileValidator:
    """Comprehensive file validation for uploads"""
    
    ALLOWED_MIME_TYPES = {
        'application/pdf',
        'application/x-pdf',
        'application/acrobat',
        'applications/vnd.pdf',
        'text/pdf',
        'text/x-pdf'
    }
    
    MAX_FILE_SIZE = settings.get_file_size_bytes()  # From config
    
    def __init__(self):
        self.validation_cache = {}
    
    def validate_file(self, file_path: Path, original_filename: str = None) -> Dict[str, Any]:
        """Comprehensive file validation"""
        
        # Calculate file hash for caching
        file_hash = self._calculate_file_hash(file_path)
        
        if file_hash in self.validation_cache:
            logger.info(f"Using cached validation for {original_filename}")
            return self.validation_cache[file_hash]
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "file_info": {},
            "estimated_processing_time": 0
        }
        
        try:
            # Basic file checks
            file_info = self._get_file_info(file_path, original_filename)
            validation_result["file_info"] = file_info
            
            # Size validation
            size_check = self._validate_file_size(file_path)
            if not size_check["valid"]:
                validation_result["valid"] = False
                validation_result["errors"].extend(size_check["errors"])
            
            # MIME type validation
            mime_check = self._validate_mime_type(file_path)
            if not mime_check["valid"]:
                validation_result["valid"] = False
                validation_result["errors"].extend(mime_check["errors"])
            
            # PDF-specific validation
            if validation_result["valid"]:
                pdf_check = self._validate_pdf_content(file_path)
                if not pdf_check["valid"]:
                    validation_result["valid"] = False
                    validation_result["errors"].extend(pdf_check["errors"])
                else:
                    validation_result["warnings"].extend(pdf_check.get("warnings", []))
                    validation_result["estimated_processing_time"] = self._estimate_processing_time(
                        file_info["size_bytes"], pdf_check.get("page_count", 1)
                    )
            
        except Exception as e:
            logger.error(f"File validation error: {e}")
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation failed: {str(e)}")
        
        # Cache the result
        self.validation_cache[file_hash] = validation_result
        
        return validation_result
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file for caching"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _get_file_info(self, file_path: Path, original_filename: str = None) -> Dict[str, Any]:
        """Get comprehensive file information"""
        stat = file_path.stat()
        
        return {
            "filename": original_filename or file_path.name,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": file_path.suffix.lower(),
            "mime_type": mimetypes.guess_type(str(file_path))[0]
        }
    
    def _validate_file_size(self, file_path: Path) -> Dict[str, Any]:
        """Validate file size"""
        file_size = file_path.stat().st_size
        
        if file_size == 0:
            return {
                "valid": False,
                "errors": ["File is empty"]
            }
        
        if file_size > self.MAX_FILE_SIZE:
            max_mb = self.MAX_FILE_SIZE / (1024 * 1024)
            current_mb = file_size / (1024 * 1024)
            return {
                "valid": False,
                "errors": [f"File too large: {current_mb:.1f}MB (max: {max_mb:.1f}MB)"]
            }
        
        return {"valid": True, "errors": []}
    
    def _validate_mime_type(self, file_path: Path) -> Dict[str, Any]:
        """Validate MIME type using multiple methods"""
        
        # Method 1: File extension
        guessed_type = mimetypes.guess_type(str(file_path))[0]
        
        # Method 2: Magic numbers (if python-magic is available)
        try:
            detected_type = magic.from_file(str(file_path), mime=True)
        except:
            detected_type = None
        
        # Check if any method indicates PDF
        is_pdf = (
            guessed_type in self.ALLOWED_MIME_TYPES or
            detected_type in self.ALLOWED_MIME_TYPES or
            str(file_path).lower().endswith('.pdf')
        )
        
        if not is_pdf:
            return {
                "valid": False,
                "errors": [
                    f"Invalid file type. Expected PDF, got: {guessed_type or 'unknown'}"
                ]
            }
        
        return {"valid": True, "errors": []}
    
    def _validate_pdf_content(self, file_path: Path) -> Dict[str, Any]:
        """Validate PDF content and structure"""
        try:
            import PyPDF2
            
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Check if encrypted
                if reader.is_encrypted:
                    return {
                        "valid": False,
                        "errors": ["PDF is password-protected. Please use an unprotected version."]
                    }
                
                # Get page count
                page_count = len(reader.pages)
                if page_count == 0:
                    return {
                        "valid": False,
                        "errors": ["PDF has no pages"]
                    }
                
                # Check for text content in first few pages
                text_found = False
                pages_to_check = min(3, page_count)
                total_text_length = 0
                
                for i in range(pages_to_check):
                    try:
                        page_text = reader.pages[i].extract_text()
                        if page_text and page_text.strip():
                            text_found = True
                            total_text_length += len(page_text.strip())
                    except Exception as e:
                        logger.warning(f"Could not extract text from page {i}: {e}")
                
                warnings = []
                if not text_found:
                    return {
                        "valid": False,
                        "errors": ["PDF appears to contain no readable text. It may be image-based."]
                    }
                
                if total_text_length < 100:
                    warnings.append("PDF has very little text content")
                
                if page_count > 50:
                    warnings.append(f"Large PDF ({page_count} pages) may take longer to process")
                
                return {
                    "valid": True,
                    "errors": [],
                    "warnings": warnings,
                    "page_count": page_count,
                    "text_length": total_text_length
                }
                
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Could not read PDF file: {str(e)}"]
            }
    
    def _estimate_processing_time(self, file_size_bytes: int, page_count: int) -> int:
        """Estimate processing time in seconds"""
        
        # Base time estimates (in seconds)
        base_time_per_page = 2  # 2 seconds per page
        base_time_per_mb = 5    # 5 seconds per MB
        
        # Calculate estimates
        page_time = page_count * base_time_per_page
        size_time = (file_size_bytes / (1024 * 1024)) * base_time_per_mb
        
        # Use the higher estimate with some padding
        estimated_time = max(page_time, size_time) * 1.2  # 20% padding
        
        # Minimum 10 seconds, maximum 300 seconds (5 minutes)
        return max(10, min(300, int(estimated_time)))

class ProgressEstimator:
    """Estimates and tracks processing progress"""
    
    def __init__(self):
        self.stage_weights = {
            "upload": 0.05,        # 5%
            "text_extraction": 0.20,  # 20%
            "chunking": 0.15,      # 15%
            "embedding_generation": 0.35,  # 35%
            "vector_storage": 0.10,   # 10%
            "question_generation": 0.15   # 15%
        }
        
        self.historical_times = {}  # Store historical processing times
    
    def estimate_total_time(self, file_size_mb: float, page_count: int, num_questions: int) -> Dict[str, Any]:
        """Estimate total processing time and breakdown"""
        
        # Base estimates (in seconds)
        estimates = {
            "upload": 2,
            "text_extraction": page_count * 1.5,
            "chunking": page_count * 0.5,
            "embedding_generation": max(10, page_count * 2.5),
            "vector_storage": 3,
            "question_generation": num_questions * 8
        }
        
        total_estimated = sum(estimates.values())
        
        # Add some variance based on file size
        size_factor = max(1.0, file_size_mb / 5.0)  # Factor increases with size
        total_estimated *= size_factor
        
        return {
            "total_seconds": int(total_estimated),
            "total_minutes": round(total_estimated / 60, 1),
            "stage_breakdown": estimates,
            "size_factor": size_factor
        }
    
    def get_progress_message(self, stage: str, percentage: float) -> str:
        """Get user-friendly progress message"""
        
        messages = {
            "upload": [
                "📤 Uploading your PDF...",
                "📤 Processing uploaded file...",
                "📤 Upload complete!"
            ],
            "text_extraction": [
                "📖 Reading PDF content...",
                "📖 Extracting text from pages...",
                "📖 Text extraction complete!"
            ],
            "chunking": [
                "✂️ Breaking down content into sections...",
                "✂️ Creating manageable chunks...",
                "✂️ Content chunking complete!"
            ],
            "embedding_generation": [
                "🧠 Analyzing content with AI...",
                "🧠 Creating semantic embeddings...",
                "🧠 AI analysis complete!"
            ],
            "vector_storage": [
                "💾 Storing content in database...",
                "💾 Indexing for fast retrieval...",
                "💾 Storage complete!"
            ],
            "question_generation": [
                "❓ Generating assessment questions...",
                "❓ Creating answer choices...",
                "❓ Questions generated successfully!"
            ]
        }
        
        stage_messages = messages.get(stage, ["Processing...", "Working...", "Complete!"])
        
        if percentage < 30:
            return stage_messages[0]
        elif percentage < 90:
            return stage_messages[1]
        else:
            return stage_messages[2]

class UserGuidance:
    """Provides user guidance and help"""
    
    @staticmethod
    def get_file_requirements() -> Dict[str, Any]:
        """Get file upload requirements and tips"""
        return {
            "supported_formats": ["PDF"],
            "max_file_size": f"{settings.MAX_FILE_SIZE_MB}MB",
            "requirements": [
                "PDF must contain readable text (not just images)",
                "PDF should not be password-protected",
                "Content should be educational/technical in nature",
                "Longer documents (3+ pages) work better for question generation"
            ],
            "tips": [
                "Academic papers and textbooks work best",
                "Include diagrams and examples for better questions",
                "Ensure good text quality (clear scanning if applicable)",
                "Consider breaking very large documents into chapters"
            ],
            "troubleshooting": {
                "file_too_large": "Split the PDF into smaller sections",
                "no_text_found": "Use OCR software to convert image-based PDFs",
                "password_protected": "Remove password protection before uploading",
                "poor_quality": "Ensure the PDF has clear, readable text"
            }
        }
    
    @staticmethod
    def get_sample_files_info() -> List[Dict[str, Any]]:
        """Get information about sample files for testing"""
        return [
            {
                "name": "Blockchain Basics",
                "description": "Introduction to blockchain technology and consensus mechanisms",
                "pages": 5,
                "topics": ["blockchain", "consensus", "mining", "wallets"],
                "estimated_questions": "8-12",
                "download_url": "/samples/blockchain_basics.pdf"
            },
            {
                "name": "Smart Contracts Guide",
                "description": "Ethereum smart contracts and gas mechanisms",
                "pages": 7,
                "topics": ["ethereum", "smart contracts", "gas", "deployment"],
                "estimated_questions": "10-15",
                "download_url": "/samples/smart_contracts.pdf"
            },
            {
                "name": "Advanced Blockchain Topics",
                "description": "Sharding, Layer 2 solutions, and bridges",
                "pages": 10,
                "topics": ["sharding", "layer2", "bridges", "scalability"],
                "estimated_questions": "15-20",
                "download_url": "/samples/advanced_blockchain.pdf"
            }
        ]
    
    @staticmethod
    def get_feature_tooltips() -> Dict[str, str]:
        """Get tooltips for UI features"""
        return {
            "num_questions": "Choose between 1-20 questions. More questions require longer, more detailed documents.",
            "difficulty_level": "Easy: Basic definitions. Medium: Conceptual understanding. Hard: Technical analysis.",
            "export_format": "PDF for printing, JSON for LMS integration, DOCX for editing.",
            "teacher_version": "Includes answers and explanations. Student version has questions only.",
            "quality_score": "Measures how well-crafted the questions are (clarity, distractor quality, etc.)",
            "regenerate": "Create new questions for low-quality items or try different approaches.",
            "analytics": "View detailed statistics about question quality and content coverage."
        }
    
    @staticmethod
    def get_keyboard_shortcuts() -> Dict[str, str]:
        """Get keyboard shortcuts for power users"""
        return {
            "Ctrl+U": "Focus upload area",
            "Ctrl+G": "Generate questions (when ready)",
            "Ctrl+E": "Export current assessment",
            "Ctrl+R": "Start over",
            "Escape": "Close modals/alerts",
            "F1": "Show help",
            "Ctrl+1,2,3": "Switch between Easy, Medium, Hard difficulty"
        }

# Global instances
file_validator = FileValidator()
progress_estimator = ProgressEstimator()
user_guidance = UserGuidance()
