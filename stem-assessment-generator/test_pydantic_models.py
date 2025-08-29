#!/usr/bin/env python3
"""
Test the updated Pydantic models
"""
from models import *
from datetime import datetime
import json

def test_pydantic_models():
    """Test all the new Pydantic models"""
    print("=== Testing Updated Pydantic Models ===")
    
    # Test UploadResponse
    print("\n--- Testing UploadResponse ---")
    upload_response = UploadResponse(
        filename="test.pdf",
        status=ProcessingStatus.READY,
        message="File uploaded successfully",
        file_size=1024
    )
    print(f"✓ UploadResponse: {upload_response.filename} - {upload_response.status}")
    
    # Test GenerateRequest
    print("\n--- Testing GenerateRequest ---")
    generate_request = GenerateRequest(
        task_id="test-task-123",
        num_questions=5,
        difficulty_level=DifficultyLevel.MEDIUM
    )
    print(f"✓ GenerateRequest: {generate_request.num_questions} questions, {generate_request.difficulty_level}")
    
    # Test Distractor
    print("\n--- Testing Distractor ---")
    distractor = Distractor(
        text="Force is always needed to maintain motion",
        misconception_type="force_motion_confusion",
        explanation="Students often think continuous force is needed for constant velocity"
    )
    print(f"✓ Distractor: {distractor.misconception_type}")
    
    # Test Question
    print("\n--- Testing Question ---")
    question = Question(
        question_text="What happens to an object moving at constant velocity in space?",
        difficulty=DifficultyLevel.MEDIUM,
        correct_answer="It continues moving at constant velocity due to inertia",
        distractors=[distractor],
        explanation="Newton's first law states that objects in motion stay in motion unless acted upon by a force"
    )
    print(f"✓ Question: {question.id[:8]}... - {question.difficulty}")
    
    # Test AssessmentResponse
    print("\n--- Testing AssessmentResponse ---")
    assessment = AssessmentResponse(
        questions=[question],
        total_questions=1,
        source_file="test.pdf",
        statistics={"generation_time": 45.2, "model_used": "gpt-3.5-turbo"}
    )
    print(f"✓ AssessmentResponse: {assessment.total_questions} questions from {assessment.source_file}")
    
    # Test ChunkMetadata
    print("\n--- Testing ChunkMetadata ---")
    chunk = ChunkMetadata(
        text="This is a test chunk of text from a PDF document.",
        page_num=1,
        chunk_index=0,
        start_char=0,
        end_char=56,
        word_count=11
    )
    print(f"✓ ChunkMetadata: Page {chunk.page_num}, Chunk {chunk.chunk_index}, {chunk.word_count} words")
    
    # Test MisconceptionPattern
    print("\n--- Testing MisconceptionPattern ---")
    pattern = MisconceptionPattern(
        pattern_type="force_motion_confusion",
        description="Students confuse force with motion",
        examples=["Force needed for constant speed", "Heavier objects fall faster"],
        subject_area="physics",
        difficulty_levels=[DifficultyLevel.EASY, DifficultyLevel.MEDIUM]
    )
    print(f"✓ MisconceptionPattern: {pattern.pattern_type} ({pattern.subject_area})")
    
    # Test ProcessingStatus enum
    print("\n--- Testing ProcessingStatus ---")
    for status in ProcessingStatus:
        print(f"  Status: {status.value}")
    
    # Test TaskStatus
    print("\n--- Testing TaskStatus ---")
    task_status = TaskStatus(
        task_id="test-task-456",
        status=ProcessingStatus.PROCESSING,
        progress=0.75,
        message="Generating questions..."
    )
    print(f"✓ TaskStatus: {task_status.task_id} - {task_status.progress*100:.0f}% - {task_status.status}")
    
    # Test JSON serialization
    print("\n--- Testing JSON Serialization ---")
    try:
        json_data = assessment.model_dump_json(indent=2)
        print("✓ JSON serialization successful")
        print(f"  First 100 chars: {json_data[:100]}...")
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")
    
    # Test validation
    print("\n--- Testing Validation ---")
    try:
        # This should fail validation
        invalid_request = GenerateRequest(
            task_id="test",
            num_questions=25,  # Too many
            difficulty_level=DifficultyLevel.EASY
        )
        print("❌ Validation should have failed")
    except Exception as e:
        print(f"✓ Validation correctly caught error: {str(e)[:60]}...")
    
    print("\n=== All Pydantic Models Tests Completed ===")

if __name__ == "__main__":
    test_pydantic_models()
