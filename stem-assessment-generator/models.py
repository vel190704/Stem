"""
Pydantic models for STEM Assessment Generator
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union, Literal
from enum import Enum
from datetime import datetime
import uuid

# =============================================================================
# Enums
# =============================================================================

class DifficultyLevel(str, Enum):
    """Difficulty levels for assessment questions"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class QuestionType(str, Enum):
    """Types of assessment questions"""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"

class ProcessingStatus(str, Enum):
    """Status of document processing"""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"

# =============================================================================
# Request/Response Models
# =============================================================================

class UploadResponse(BaseModel):
    """Response model for file upload"""
    filename: str
    status: ProcessingStatus
    message: str
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_size: Optional[int] = None
    uploaded_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class GenerateRequest(BaseModel):
    """Request model for assessment generation"""
    task_id: str
    num_questions: int = Field(default=5, ge=1, le=20, description="Number of questions to generate")
    difficulty_level: DifficultyLevel = DifficultyLevel.MEDIUM
    question_types: List[QuestionType] = Field(default=[QuestionType.MULTIPLE_CHOICE])
    topics: Optional[List[str]] = Field(default=None, description="Specific topics to focus on")
    
    @validator('num_questions')
    def validate_num_questions(cls, v):
        if v < 1 or v > 20:
            raise ValueError("Number of questions must be between 1 and 20")
        return v

class Distractor(BaseModel):
    """Model for answer distractors"""
    text: str = Field(..., min_length=1, description="The distractor text")
    misconception_type: str = Field(..., description="Type of misconception this represents")
    explanation: str = Field(..., description="Why a student might choose this answer")
    
    @validator('text', 'misconception_type', 'explanation')
    def validate_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

class Question(BaseModel):
    """Model for assessment questions"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question_text: str = Field(..., min_length=10, description="The question text")
    question_type: QuestionType = QuestionType.MULTIPLE_CHOICE
    difficulty: DifficultyLevel
    correct_answer: str = Field(..., min_length=1, description="The correct answer")
    distractors: List[Distractor] = Field(..., min_items=1, max_items=4, description="List of distractors")
    explanation: str = Field(..., description="Explanation of the correct answer")
    source_context: Optional[str] = Field(default=None, description="Source text context")
    bloom_taxonomy_level: Optional[str] = Field(default="remember", description="Bloom's taxonomy level")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def options(self) -> Dict[str, Dict[str, Any]]:
        """Get formatted options with detailed explanations for frontend"""
        # Check if we have detailed options in metadata (from enhanced generation)
        detailed_options = self.metadata.get("detailed_options")
        if detailed_options and isinstance(detailed_options, dict):
            return detailed_options
        
        # Fallback to basic format if no detailed options
        letters = ['A', 'B', 'C', 'D']
        options = {}
        
        # Add correct answer first
        options['A'] = {
            "text": self.correct_answer,
            "is_correct": True,
            "explanation": {
                "why_correct": self.explanation or "This is the correct answer.",
                "key_points": ["Correct understanding of the concept"],
                "real_world_example": "This principle applies in practical scenarios.",
                "connects_to": "Related concepts and applications"
            }
        }
        
        # Add distractors
        for i, distractor in enumerate(self.distractors[:3]):  # Max 3 distractors
            letter = letters[i + 1]
            options[letter] = {
                "text": distractor.text,
                "is_correct": False,
                "explanation": {
                    "misconception_name": distractor.misconception_type or "Common Misconception",
                    "why_students_think_this": f"Students often select this due to {distractor.misconception_type or 'surface-level understanding'}.",
                    "why_its_wrong": distractor.explanation or "This represents a misunderstanding of the core concept.",
                    "correct_understanding": "The accurate understanding requires deeper knowledge.",
                    "remember_this": "Remember the key principles of this concept."
                }
            }
            
        return options
    
    @property
    def correct_position(self) -> str:
        """Get the position of the correct answer"""
        return 'A'  # Correct answer is always at position A in our structure
    
    @validator('question_text', 'correct_answer', 'explanation')
    def validate_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

class QuestionGeneration(BaseModel):
    """Model for generated questions from LLM"""
    question_text: str = Field(..., description="The question text")
    options: Dict[str, str] = Field(..., description="Answer options (A, B, C, D)")
    correct_position: str = Field(..., description="Position of correct answer (A, B, C, or D)")
    difficulty: str = Field(..., description="Question difficulty level")
    misconceptions: Dict[str, str] = Field(default_factory=dict, description="Misconception type for each option")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('correct_position')
    def validate_correct_position(cls, v):
        if v not in ['A', 'B', 'C', 'D']:
            raise ValueError("correct_position must be A, B, C, or D")
        return v
    
    @validator('options')
    def validate_options(cls, v):
        required_keys = {'A', 'B', 'C', 'D'}
        if set(v.keys()) != required_keys:
            raise ValueError("options must contain exactly keys A, B, C, D")
        return v

class GenerationProgress(BaseModel):
    """Model for tracking question generation progress"""
    task_id: str
    total_requested: int
    questions_generated: int
    current_batch: int = Field(default=1)
    status: str = Field(default="generating")
    failed_attempts: int = Field(default=0)
    last_error: Optional[str] = Field(default=None)
    generation_stats: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class GenerationRequest(BaseModel):
    """Enhanced request model for question generation"""
    task_id: str
    num_questions: int = Field(default=10, ge=1, le=50)
    difficulty_level: DifficultyLevel = DifficultyLevel.MEDIUM
    difficulty_mix: Optional[Dict[str, int]] = Field(default=None, description="Mix of difficulties")
    question_types: List[QuestionType] = Field(default_factory=lambda: [QuestionType.MULTIPLE_CHOICE])
    allow_streaming: bool = Field(default=False, description="Enable streaming progress updates")
    
    @validator('difficulty_mix')
    def validate_difficulty_mix(cls, v, values):
        if v is not None:
            total_requested = values.get('num_questions', 0)
            if sum(v.values()) != total_requested:
                raise ValueError("Sum of difficulty_mix values must equal num_questions")
        return v

class AssessmentResponse(BaseModel):
    """Response model for generated assessment"""
    questions: List[Question] = Field(..., description="List of generated questions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Assessment metadata")
    total_questions: int = Field(..., description="Total number of questions generated")
    source_file: Optional[str] = Field(default=None, description="Source filename")
    generated_at: datetime = Field(default_factory=datetime.now)
    processing_time: Optional[float] = Field(default=None, description="Time taken to generate (seconds)")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Generation statistics")
    
    @validator('total_questions')
    def validate_total_matches_questions(cls, v, values):
        questions = values.get('questions', [])
        if v != len(questions):
            raise ValueError("total_questions must match the length of questions list")
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# =============================================================================
# Internal Domain Models
# =============================================================================

class ChunkMetadata(BaseModel):
    """Metadata for text chunks"""
    text: str = Field(..., min_length=1, description="The chunk text content")
    page_num: Optional[int] = Field(default=None, ge=1, description="Page number (1-indexed)")
    chunk_index: int = Field(..., ge=0, description="Index of chunk in document")
    start_char: Optional[int] = Field(default=None, ge=0, description="Start character position")
    end_char: Optional[int] = Field(default=None, ge=0, description="End character position")
    word_count: Optional[int] = Field(default=None, ge=0, description="Number of words in chunk")
    
    @validator('text')
    def validate_text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Chunk text cannot be empty")
        return v.strip()
    
    @validator('end_char')
    def validate_end_after_start(cls, v, values):
        start_char = values.get('start_char')
        if start_char is not None and v is not None and v <= start_char:
            raise ValueError("end_char must be greater than start_char")
        return v

class MisconceptionPattern(BaseModel):
    """Model for misconception patterns"""
    pattern_type: str = Field(..., description="Type/category of misconception")
    description: str = Field(..., description="Description of the misconception")
    examples: List[str] = Field(default_factory=list, description="Example misconceptions")
    subject_area: Optional[str] = Field(default=None, description="Subject area (physics, chemistry, etc.)")
    difficulty_levels: List[DifficultyLevel] = Field(default_factory=list, description="Applicable difficulty levels")
    
    @validator('pattern_type', 'description')
    def validate_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

# =============================================================================
# Legacy Models (for backward compatibility)
# =============================================================================

class AssessmentRequest(BaseModel):
    """Legacy request model for assessment generation"""
    num_questions: int = Field(default=5, ge=1, le=20)
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    question_types: List[QuestionType] = [QuestionType.MULTIPLE_CHOICE]
    topics: Optional[List[str]] = None

class PDFContent(BaseModel):
    """Model for processed PDF content"""
    filename: str
    content: str
    chunks: List[str]
    metadata: Dict[str, Any]
    chunks_metadata: Optional[List[ChunkMetadata]] = Field(default=None)
    processing_status: ProcessingStatus = ProcessingStatus.READY
    
    @validator('filename', 'content')
    def validate_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip() if isinstance(v, str) else v

class EmbeddingResult(BaseModel):
    """Model for embedding results"""
    text: str
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_name: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.now)
    
    @validator('text')
    def validate_text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()
    
    @validator('embedding')
    def validate_embedding_not_empty(cls, v):
        if not v:
            raise ValueError("Embedding cannot be empty")
        return v

# =============================================================================
# Helper Models
# =============================================================================

class TaskStatus(BaseModel):
    """Model for tracking task status"""
    task_id: str
    status: ProcessingStatus
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    message: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    error_details: Optional[str] = Field(default=None)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ValidationResult(BaseModel):
    """Model for validation results"""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    field_errors: Dict[str, List[str]] = Field(default_factory=dict)

class TaskStatusResponse(BaseModel):
    """Response model for task status queries"""
    task_id: str
    status: ProcessingStatus
    message: str
    progress: Dict[str, Any] = Field(default_factory=dict)
    timestamps: Dict[str, datetime] = Field(default_factory=dict)
    error_details: Optional[Dict[str, Any]] = Field(default=None)
    created_at: datetime
    updated_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class StreamingGenerationResponse(BaseModel):
    """Response model for streaming generation updates"""
    event_type: str  # progress, question, complete, error
    task_id: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str  # healthy, degraded, unhealthy
    openai_status: str
    vectordb_status: str
    system_stats: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime

class RegenerateRequest(BaseModel):
    """Request model for question regeneration"""
    question_indices: Optional[List[int]] = None
    quality_threshold: float = 0.6
    regenerate_all_below_threshold: bool = False

class ExportRequest(BaseModel):
    """Request model for assessment export"""
    format_type: str = "json"
    teacher_version: bool = True
    include_explanations: bool = True
    include_difficulty: bool = True
    randomize_order: bool = False
    separate_answer_key: bool = False
    error: Optional[str] = Field(default=None)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
