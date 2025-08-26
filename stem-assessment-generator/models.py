"""
Pydantic models for STEM Assessment Generator
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

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

class MisconceptionPattern(BaseModel):
    """Model for misconception patterns"""
    name: str
    description: str
    examples: List[str]
    distractor_templates: List[str]

class Distractor(BaseModel):
    """Model for answer distractors"""
    text: str
    misconception_type: str
    explanation: str

class Question(BaseModel):
    """Model for assessment questions"""
    id: str
    question_text: str
    question_type: QuestionType
    difficulty: DifficultyLevel
    correct_answer: str
    distractors: List[Distractor]
    explanation: str
    source_context: str
    bloom_taxonomy_level: str

class AssessmentRequest(BaseModel):
    """Request model for assessment generation"""
    num_questions: int = Field(default=5, ge=1, le=20)
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    question_types: List[QuestionType] = [QuestionType.MULTIPLE_CHOICE]
    topics: Optional[List[str]] = None

class AssessmentResponse(BaseModel):
    """Response model for generated assessment"""
    questions: List[Question]
    metadata: Dict[str, Any]
    total_questions: int
    
class PDFContent(BaseModel):
    """Model for processed PDF content"""
    filename: str
    content: str
    chunks: List[str]
    metadata: Dict[str, Any]

class EmbeddingResult(BaseModel):
    """Model for embedding results"""
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]
