"""
Main FastAPI application for STEM Assessment Generator
Enhanced with intelligent PDF processing and RAG pipeline integration
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
import uvicorn
import os
import asyncio
import uuid
from typing import Dict, Optional, List
from datetime import datetime
import shutil
import logging
import traceback
from pathlib import Path
from io import BytesIO

from config import settings
from models import (
    UploadResponse, GenerateRequest, AssessmentResponse, 
    ProcessingStatus, TaskStatus, DifficultyLevel,
    HealthResponse, TaskStatusResponse,
    GenerationRequest, GenerationProgress, StreamingGenerationResponse,
    QuestionGeneration, RegenerateRequest, ExportRequest
)
from pdf_processor import PDFProcessor
from rag_pipeline import RAGPipeline
from generator import AssessmentGenerator
from analytics import analytics_manager
from exporter import ExportManager
from exporter import ExportManager, ExportConfiguration, AssessmentExporter
from optimizations import optimization_manager
from exceptions import *
from system_validation import system_validator, validate_startup, get_system_health
from user_experience import file_validator, progress_estimator, user_guidance
from documentation import DocumentationGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="STEM Assessment Generator",
    version="2.0.0",
    description="AI-powered assessment question generator with intelligent PDF processing",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize components
pdf_processor = PDFProcessor()
rag_pipeline = RAGPipeline()
generator = AssessmentGenerator()
export_manager = ExportManager()
documentation_generator = DocumentationGenerator(app)

# Enhanced task tracking with detailed progress
task_storage: Dict[str, Dict] = {}
processed_documents: Dict[str, Dict] = {}

# =============================================================================
# Event Handlers
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize application with comprehensive validation"""
    logger.info("🚀 Starting STEM Assessment Generator...")
    
    # Run startup validation
    if not validate_startup():
        logger.error("❌ Startup validation failed. Check configuration.")
        raise RuntimeError("Critical configuration errors detected")
    
    # Log system health
    health = get_system_health()
    logger.info(f"System Status: {health['overall_status']}")
    
    # Start optimization manager
    await optimization_manager.startup()
    
    logger.info("✅ STEM Assessment Generator started successfully!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down STEM Assessment Generator...")
    
    # Cleanup optimization manager
    await optimization_manager.shutdown()
    
    logger.info("✅ Shutdown completed successfully!")

# =============================================================================
# Global Error Handlers
# =============================================================================

@app.exception_handler(BaseAssessmentError)
async def assessment_error_handler(request, exc: BaseAssessmentError):
    """Handle custom assessment errors"""
    logger.error(f"Assessment error: {exc.error_code} - {exc.message}")
    return JSONResponse(
        status_code=400,
        content=create_error_response(exc, include_technical=settings.DEBUG)
    )

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc: RequestValidationError):
    """Handle request validation errors"""
    error = BaseAssessmentError(
        message="Request validation failed",
        user_message="Invalid request data. Please check your input.",
        technical_details=str(exc),
        error_code="VALIDATION_ERROR"
    )
    return JSONResponse(
        status_code=422,
        content=create_error_response(error, include_technical=settings.DEBUG)
    )

@app.exception_handler(Exception)
async def general_error_handler(request, exc: Exception):
    """Handle unexpected errors"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    error = BaseAssessmentError(
        message=str(exc),
        user_message="An unexpected error occurred. Please try again or contact support.",
        technical_details=traceback.format_exc() if settings.DEBUG else None,
        error_code="UNEXPECTED_ERROR"
    )
    return JSONResponse(
        status_code=500,
        content=create_error_response(error, include_technical=settings.DEBUG)
    )

# =============================================================================
# Help and Documentation Endpoints
# =============================================================================

@app.get("/api/help")
async def get_api_help():
    """Get comprehensive API documentation"""
    return documentation_generator.get_api_documentation()

@app.get("/help", response_class=HTMLResponse)
async def get_user_guide():
    """Get HTML user guide"""
    return documentation_generator.get_user_guide_html()

@app.get("/api/file-requirements")
async def get_file_requirements():
    """Get file upload requirements and tips"""
    return user_guidance.get_file_requirements()

@app.get("/api/sample-files")
async def get_sample_files():
    """Get information about sample files for testing"""
    return {
        "sample_files": user_guidance.get_sample_files_info(),
        "download_instructions": "Download sample files to test the system"
    }

@app.get("/api/faq")
async def get_faq():
    """Get frequently asked questions"""
    return {
        "faq": documentation_generator.get_faq(),
        "contact": "For additional help, contact support"
    }

@app.get("/api/tooltips")
async def get_feature_tooltips():
    """Get tooltips for UI features"""
    return {
        "tooltips": user_guidance.get_feature_tooltips(),
        "keyboard_shortcuts": user_guidance.get_keyboard_shortcuts()
    }

@app.get("/api/system-validation")
async def get_system_validation():
    """Get detailed system validation report"""
    try:
        return get_system_health()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System validation failed: {str(e)}")

# =============================================================================
# Enhanced Upload with Validation
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for better error messages"""
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "message": str(exc),
            "type": type(exc).__name__
        }
    )

# =============================================================================
# Main Routes
# =============================================================================

@app.get("/", response_class=RedirectResponse)
async def root():
    """Redirect to static index.html"""
    return RedirectResponse(url="/static/index.html", status_code=302)

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "openai_configured": settings.is_openai_configured(),
        "directories": {
            "uploads": settings.upload_path.exists(),
            "vectordb": settings.vectordb_path.exists()
        }
    }

# =============================================================================
# API Routes
# =============================================================================

@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload PDF file with comprehensive validation and start intelligent processing"""
    try:
        # Basic file validation
        if not file.filename:
            raise PDFProcessingError(
                user_message="No file provided",
                recovery_action="Please select a PDF file to upload"
            )
        
        # Read file content
        file_content = await file.read()
        if len(file_content) == 0:
            raise PDFProcessingError(
                user_message="The uploaded file is empty",
                recovery_action="Please select a valid PDF file with content"
            )
        
        # Save file temporarily for validation
        task_id = str(uuid.uuid4())
        temp_file_path = settings.upload_path / f"{task_id}_{file.filename}"
        
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file_content)
        
        # Comprehensive file validation
        validation_result = file_validator.validate_file(temp_file_path, file.filename)
        
        if not validation_result["valid"]:
            # Clean up temp file
            temp_file_path.unlink(missing_ok=True)
            
            error_messages = "; ".join(validation_result["errors"])
            raise PDFProcessingError(
                user_message=f"File validation failed: {error_messages}",
                recovery_action="Please check the file requirements and try again"
            )
        
        # Show warnings if any
        if validation_result["warnings"]:
            logger.warning(f"File validation warnings: {validation_result['warnings']}")
        
        # Estimate processing time
        estimated_time = validation_result.get("estimated_processing_time", 60)
        file_info = validation_result["file_info"]
        file_size_bytes = file_info["size_bytes"]
        file_size_mb = file_info["size_mb"]
        
        # Create upload response with task ID
        upload_response = UploadResponse(
            filename=file.filename,
            status=ProcessingStatus.UPLOADING,
            message="File validated successfully, starting intelligent processing...",
            file_size=file_size_bytes
        )
        
        task_id = upload_response.task_id
        
        # Move temp file to final location
        file_path = settings.upload_path / f"{task_id}_{file.filename}"
        temp_file_path.rename(file_path)
        
        logger.info(f"File uploaded: {file.filename} ({file_size_mb:.2f}MB) -> {file_path}")
        
        # Initialize comprehensive task status
        task_storage[task_id] = {
            "status": ProcessingStatus.UPLOADING,
            "filename": file.filename,
            "file_path": str(file_path),
            "file_size_mb": file_size_mb,
            "progress": {
                "current_step": "upload",
                "steps_completed": ["upload"],
                "total_steps": ["upload", "text_extraction", "chunking", "embedding_generation", "vector_storage"],
                "chunks_processed": 0,
                "total_chunks": 0,
                "percentage": 10
            },
            "metadata": {
                "filename": file.filename,
                "upload_time": datetime.now().isoformat(),
                "file_size_mb": file_size_mb,
                "processing_method": "intelligent_chunking"
            },
            "timestamps": {
                "upload_start": datetime.now(),
                "upload_complete": datetime.now()
            },
            "message": "File uploaded successfully, starting intelligent processing...",
            "error_details": None,
            "processing_stats": {},
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        # Start enhanced background processing
        background_tasks.add_task(process_pdf_enhanced, task_id, file_path)
        
        return upload_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/process_status/{task_id}")
async def get_detailed_processing_status(task_id: str):
    """Get detailed processing status with comprehensive progress information"""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_data = task_storage[task_id]
    
    # Calculate processing time
    processing_time = None
    if "timestamps" in task_data:
        start_time = task_data["timestamps"].get("upload_start")
        if start_time:
            processing_time = (datetime.now() - start_time).total_seconds()
    
    return {
        "task_id": task_id,
        "status": task_data["status"],
        "progress": task_data.get("progress", {}),
        "metadata": task_data.get("metadata", {}),
        "message": task_data.get("message", ""),
        "error_details": task_data.get("error_details"),
        "processing_stats": task_data.get("processing_stats", {}),
        "processing_time_seconds": processing_time,
        "timestamps": {
            k: v.isoformat() if isinstance(v, datetime) else v 
            for k, v in task_data.get("timestamps", {}).items()
        },
        "created_at": task_data["created_at"].isoformat(),
        "updated_at": task_data["updated_at"].isoformat()
    }

@app.post("/api/retry/{task_id}")
async def retry_processing(task_id: str, background_tasks: BackgroundTasks):
    """Retry processing for a failed task"""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_data = task_storage[task_id]
    
    if task_data["status"] != ProcessingStatus.ERROR:
        raise HTTPException(
            status_code=400, 
            detail=f"Task is not in error state. Current status: {task_data['status']}"
        )
    
    # Reset task status for retry
    task_data["status"] = ProcessingStatus.PROCESSING
    task_data["progress"]["current_step"] = "text_extraction"
    task_data["progress"]["steps_completed"] = ["upload"]
    task_data["progress"]["percentage"] = 15
    task_data["message"] = "Retrying processing..."
    task_data["error_details"] = None
    task_data["updated_at"] = datetime.now()
    task_data["timestamps"]["retry_start"] = datetime.now()
    
    # Clean up any partial data
    if task_id in processed_documents:
        del processed_documents[task_id]
    
    # Restart processing
    file_path = task_data["file_path"]
    background_tasks.add_task(process_pdf_enhanced, task_id, file_path)
    
    logger.info(f"Retrying processing for task: {task_id}")
    
    return {
        "task_id": task_id,
        "status": "PROCESSING",
        "message": "Processing retry started"
    }

@app.get("/api/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get processing status for a task"""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_data = task_storage[task_id]
    
    # Extract progress percentage from progress dict
    progress_value = task_data.get("progress", {})
    if isinstance(progress_value, dict):
        progress_percentage = progress_value.get("percentage", 0) / 100.0  # Convert to 0-1 range
    else:
        progress_percentage = float(progress_value) if progress_value else 0.0
    
    # Extract error details as string
    error_details = task_data.get("error_details")
    if isinstance(error_details, dict):
        error_details = str(error_details)
    
    return TaskStatus(
        task_id=task_id,
        status=task_data["status"],
        progress=progress_percentage,
        message=task_data.get("message", ""),
        created_at=task_data["created_at"],
        updated_at=task_data["updated_at"],
        error_details=error_details
    )

@app.post("/api/generate", response_model=AssessmentResponse)
async def generate_assessment(request: GenerateRequest):
    """Generate assessment questions from processed PDF with enhanced tracking"""
    try:
        # Check if task exists and is ready
        if request.task_id not in task_storage:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task_data = task_storage[request.task_id]
        if task_data["status"] != ProcessingStatus.READY:
            raise HTTPException(
                status_code=400, 
                detail=f"Task not ready for question generation. Current status: {task_data['status']}"
            )
        
        # Check if document is processed
        if request.task_id not in processed_documents:
            raise HTTPException(status_code=400, detail="Document processing data not found")
        
        # Validate OpenAI configuration
        if not settings.is_openai_configured():
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured. Please set OPENAI_API_KEY in environment."
            )
        
        # Update task status for generation
        task_data["status"] = ProcessingStatus.PROCESSING
        task_data["message"] = f"Generating {request.num_questions} questions..."
        task_data["progress"]["current_step"] = "question_generation"
        task_data["progress"]["percentage"] = 90
        task_data["updated_at"] = datetime.now()
        task_data["timestamps"]["generation_start"] = datetime.now()
        
        # Get processed document data
        doc_data = processed_documents[request.task_id]
        retriever = doc_data["retriever"]
        
        logger.info(f"Starting question generation for task {request.task_id}: "
                   f"{request.num_questions} questions, difficulty: {request.difficulty_level.value}")
        
        # Check cache for existing questions
        content_hash = None
        cached_questions = None
        if hasattr(doc_data, 'content_hash'):
            content_hash = doc_data['content_hash']
            cached_questions = optimization_manager.question_cache.get_cached_questions(
                content_hash, request.num_questions
            )
        
        if cached_questions:
            logger.info(f"Using {len(cached_questions)} cached questions for task {request.task_id}")
            
            # Create assessment from cached questions
            from models import AssessmentResponse
            assessment = AssessmentResponse(
                questions=cached_questions,
                source_file=task_data["filename"],
                processing_time=0.1,  # Minimal time for cache retrieval
                statistics={
                    "task_id": request.task_id,
                    "from_cache": True,
                    "requested_questions": request.num_questions,
                    "generated_questions": len(cached_questions)
                }
            )
            generation_time = 0.1
        else:
            # Apply rate limiting for API calls
            await optimization_manager.rate_limit_manager.wait_if_needed('openai_completions')
            
            # Generate assessment using enhanced generator
            start_time = datetime.now()
            assessment = await generator.generate_assessment(
                retriever, 
                request.num_questions, 
                request.difficulty_level.value
            )
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            
            # Cache the generated questions for future use
            if content_hash and assessment.questions:
                optimization_manager.question_cache.cache_questions(
                    content_hash, 
                    [q.dict() for q in assessment.questions],
                    {"task_id": request.task_id, "difficulty": request.difficulty_level.value}
                )
        
        # Analyze quality and track performance
        generation_metadata = {
            "num_questions_requested": request.num_questions,
            "per_question_time": generation_time / len(assessment.questions) if assessment.questions else 0,
            "difficulty_requested": request.difficulty_level.value,
            "model_used": settings.OPENAI_MODEL
        }
        
        try:
            analytics_result = analytics_manager.analyze_generation_result(
                request.task_id, assessment, generation_metadata
            )
            
            # Add analytics summary to statistics
            assessment.statistics.update({
                "quality_score": analytics_result.get("assessment_analysis", {}).get("quality_score", 0),
                "quality_grade": analytics_result.get("assessment_analysis", {}).get("overview", {}).get("overall_grade", "C"),
                "quality_alerts": analytics_result.get("quality_alerts", [])
            })
        except Exception as e:
            logger.warning(f"Analytics processing failed: {e}")
        
        # Record performance statistics in optimization database
        try:
            success_rate = len(assessment.questions) / request.num_questions if request.num_questions > 0 else 0
            if content_hash:
                optimization_manager.db_optimizer.record_generation_stats(
                    content_hash=content_hash,
                    num_questions=request.num_questions,
                    difficulty=request.difficulty_level.value,
                    generation_time=generation_time,
                    success_rate=success_rate
                )
        except Exception as e:
            logger.warning(f"Performance tracking failed: {e}")
        
        # Enhance assessment with comprehensive metadata
        assessment.source_file = task_data["filename"]
        assessment.processing_time = generation_time
        
        # Add detailed statistics
        processing_stats = doc_data.get("processing_stats", {})
        assessment.statistics.update({
            "task_id": request.task_id,
            "generation_time": generation_time,
            "requested_questions": request.num_questions,
            "generated_questions": len(assessment.questions),
            "exact_count_achieved": len(assessment.questions) == request.num_questions,
            "difficulty": request.difficulty_level.value,
            "model_used": settings.OPENAI_MODEL,
            "source_chunks": processing_stats.get("total_chunks", 0),
            "source_pages": processing_stats.get("total_pages", 0),
            "chunking_method": processing_stats.get("chunking_method", "unknown"),
            "collection_name": doc_data.get("collection_name")
        })
        
        # Update metadata with processing information
        assessment.metadata.update({
            "processing_method": "intelligent_rag_pipeline",
            "pdf_processing_stats": processing_stats,
            "generation_stats": assessment.statistics.get("generation_stats", {}),
            "timestamps": {
                k: v.isoformat() if isinstance(v, datetime) else v 
                for k, v in task_data.get("timestamps", {}).items()
            }
        })
        
        # Store the assessment result in task_storage for exports
        task_data["assessment_result"] = assessment
        
        # Update task status to completed
        task_data["status"] = ProcessingStatus.READY
        task_data["message"] = f"Successfully generated {len(assessment.questions)} questions"
        task_data["progress"]["current_step"] = "completed"
        task_data["progress"]["percentage"] = 100
        task_data["timestamps"]["generation_complete"] = datetime.now()
        task_data["updated_at"] = datetime.now()
        
        logger.info(f"Question generation completed for task {request.task_id}: "
                   f"{len(assessment.questions)} questions in {generation_time:.2f}s")
        
        # Cache the assessment results for PDF export
        if request.task_id in processed_documents:
            processed_documents[request.task_id]["assessment_results"] = assessment
            logger.info(f"Cached assessment results for task {request.task_id}")
        
        return assessment
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Question generation failed for task {request.task_id}: {e}")
        
        # Update task status to error
        if request.task_id in task_storage:
            task_storage[request.task_id]["status"] = ProcessingStatus.ERROR
            task_storage[request.task_id]["message"] = f"Question generation failed: {str(e)}"
            task_storage[request.task_id]["error_details"] = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "failed_at_step": "question_generation"
            }
            task_storage[request.task_id]["timestamps"]["generation_error"] = datetime.now()
            task_storage[request.task_id]["updated_at"] = datetime.now()
        
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")

# =============================================================================
# Enhanced Generation Endpoints
# =============================================================================

@app.post("/api/generate-enhanced", response_model=AssessmentResponse)
async def generate_assessment_enhanced(request: GenerationRequest):
    """Enhanced assessment generation with better validation and progress tracking"""
    try:
        # Check if task exists and is ready
        if request.task_id not in task_storage:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task_data = task_storage[request.task_id]
        if task_data["status"] != ProcessingStatus.READY:
            raise HTTPException(
                status_code=400, 
                detail=f"Task not ready for question generation. Current status: {task_data['status']}"
            )
        
        # Check if document is processed
        if request.task_id not in processed_documents:
            raise HTTPException(status_code=400, detail="Document processing data not found")
        
        # Validate OpenAI configuration
        if not settings.is_openai_configured():
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured. Please set OPENAI_API_KEY in environment."
            )
        
        # Initialize enhanced task tracking
        task_data["status"] = ProcessingStatus.PROCESSING
        task_data["message"] = f"Generating {request.num_questions} questions with enhanced validation..."
        task_data["progress"] = {
            "current_step": "question_generation",
            "percentage": 90,
            "questions_generated": 0,
            "total_requested": request.num_questions,
            "current_batch": 1,
            "failed_attempts": 0
        }
        task_data["partial_results"] = []
        task_data["generation_log"] = []
        task_data["updated_at"] = datetime.now()
        task_data["timestamps"]["generation_start"] = datetime.now()
        
        # Get processed document data
        doc_data = processed_documents[request.task_id]
        retriever = doc_data["retriever"]
        
        logger.info(f"Starting enhanced question generation for task {request.task_id}: "
                   f"{request.num_questions} questions, difficulty: {request.difficulty_level.value}")
        
        # Generate assessment using enhanced generator with progress tracking
        start_time = datetime.now()
        
        # Use the enhanced generator with progress callback
        def progress_callback(progress_data):
            """Update task progress in real-time"""
            task_data["progress"].update(progress_data)
            task_data["updated_at"] = datetime.now()
            if "log_entry" in progress_data:
                task_data["generation_log"].append({
                    "timestamp": datetime.now().isoformat(),
                    "message": progress_data["log_entry"]
                })
        
        assessment = await generator.generate_assessment_enhanced(
            retriever, 
            request.num_questions, 
            request.difficulty_level.value,
            difficulty_mix=request.difficulty_mix,
            progress_callback=progress_callback
        )
        
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        # Enhance assessment with comprehensive metadata
        assessment.source_file = task_data["filename"]
        assessment.processing_time = generation_time
        
        # Add detailed statistics
        processing_stats = doc_data.get("processing_stats", {})
        assessment.statistics.update({
            "task_id": request.task_id,
            "generation_time": generation_time,
            "requested_questions": request.num_questions,
            "generated_questions": len(assessment.questions),
            "exact_count_achieved": len(assessment.questions) == request.num_questions,
            "difficulty": request.difficulty_level.value,
            "model_used": settings.OPENAI_MODEL,
            "source_chunks": processing_stats.get("total_chunks", 0),
            "source_pages": processing_stats.get("total_pages", 0),
            "chunking_method": processing_stats.get("chunking_method", "unknown"),
            "collection_name": doc_data.get("collection_name"),
            "generation_stats": task_data["progress"],
            "total_batches": task_data["progress"].get("current_batch", 1),
            "failed_attempts": task_data["progress"].get("failed_attempts", 0)
        })
        
        # Update metadata with processing information
        assessment.metadata.update({
            "processing_method": "enhanced_rag_pipeline",
            "pdf_processing_stats": processing_stats,
            "generation_log": task_data["generation_log"],
            "timestamps": {
                k: v.isoformat() if isinstance(v, datetime) else v 
                for k, v in task_data.get("timestamps", {}).items()
            }
        })
        
        # Store the assessment result in task_storage for exports
        task_data["assessment_result"] = assessment
        
        # Update task status to completed
        task_data["status"] = ProcessingStatus.READY
        task_data["message"] = f"Successfully generated {len(assessment.questions)} questions with enhanced validation"
        task_data["progress"]["current_step"] = "completed"
        task_data["progress"]["percentage"] = 100
        task_data["timestamps"]["generation_complete"] = datetime.now()
        task_data["updated_at"] = datetime.now()
        
        logger.info(f"Enhanced question generation completed for task {request.task_id}: "
                   f"{len(assessment.questions)} questions in {generation_time:.2f}s")
        
        return assessment
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced question generation failed for task {request.task_id}: {e}")
        
        # Update task status to error
        if request.task_id in task_storage:
            task_storage[request.task_id]["status"] = ProcessingStatus.ERROR
            task_storage[request.task_id]["message"] = f"Enhanced generation failed: {str(e)}"
            task_storage[request.task_id]["error_details"] = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "failed_at_step": "enhanced_question_generation"
            }
            task_storage[request.task_id]["timestamps"]["generation_error"] = datetime.now()
            task_storage[request.task_id]["updated_at"] = datetime.now()
        
        raise HTTPException(status_code=500, detail=f"Enhanced question generation failed: {str(e)}")

@app.get("/api/generate-with-progress/{task_id}")
async def generate_with_streaming_progress(task_id: str, num_questions: int = 10, difficulty: str = "medium"):
    """Stream question generation progress using Server-Sent Events"""
    
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task_id not in processed_documents:
        raise HTTPException(status_code=400, detail="Document not processed")
    
    if not settings.is_openai_configured():
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    async def generate_stream():
        try:
            # Initialize streaming generation
            task_data = task_storage[task_id]
            task_data["status"] = ProcessingStatus.PROCESSING
            task_data["message"] = "Starting streaming generation..."
            
            # Send initial progress
            initial_progress = StreamingGenerationResponse(
                event_type="progress",
                task_id=task_id,
                data={
                    "questions_generated": 0,
                    "total_requested": num_questions,
                    "current_batch": 1,
                    "message": "Initializing generation..."
                }
            )
            yield f"data: {initial_progress.json()}\n\n"
            
            doc_data = processed_documents[task_id]
            retriever = doc_data["retriever"]
            
            # Generate questions with streaming updates
            async def stream_callback(event_type: str, data: dict):
                response = StreamingGenerationResponse(
                    event_type=event_type,
                    task_id=task_id,
                    data=data
                )
                yield f"data: {response.json()}\n\n"
            
            # Start generation with streaming
            assessment = await generator.generate_assessment_streaming(
                retriever,
                num_questions,
                difficulty,
                stream_callback
            )
            
            # Send completion event
            completion_response = StreamingGenerationResponse(
                event_type="complete",
                task_id=task_id,
                data={
                    "assessment": assessment.dict(),
                    "total_generated": len(assessment.questions),
                    "processing_time": assessment.processing_time
                }
            )
            yield f"data: {completion_response.json()}\n\n"
            
        except Exception as e:
            # Send error event
            error_response = StreamingGenerationResponse(
                event_type="error",
                task_id=task_id,
                data={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            yield f"data: {error_response.json()}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )

@app.get("/api/generation-progress/{task_id}", response_model=GenerationProgress)
async def get_generation_progress(task_id: str):
    """Get detailed generation progress for a task"""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_data = task_storage[task_id]
    progress_data = task_data.get("progress", {})
    
    return GenerationProgress(
        task_id=task_id,
        total_requested=progress_data.get("total_requested", 0),
        questions_generated=progress_data.get("questions_generated", 0),
        current_batch=progress_data.get("current_batch", 1),
        status=task_data["status"],
        failed_attempts=progress_data.get("failed_attempts", 0),
        last_error=task_data.get("error_details", {}).get("error_message"),
        generation_stats=progress_data,
        timestamp=task_data["updated_at"]
    )

# =============================================================================
# Collection Management Endpoints
# =============================================================================

@app.get("/api/collections")
async def list_collections():
    """List all vector store collections"""
    try:
        collections = rag_pipeline.list_collections()
        return {
            "collections": collections,
            "total_count": len(collections)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")

@app.delete("/api/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a specific collection"""
    try:
        success = rag_pipeline.delete_collection(collection_name)
        if success:
            return {"message": f"Collection '{collection_name}' deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Collection not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {str(e)}")

@app.post("/api/clear_cache")
async def clear_cache():
    """Clear all caches"""
    try:
        rag_pipeline.clear_cache()
        return {"message": "All caches cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@app.get("/api/tasks")
async def list_all_tasks():
    """List all tasks with their current status"""
    tasks = []
    for task_id, task_data in task_storage.items():
        task_summary = {
            "task_id": task_id,
            "status": task_data["status"],
            "filename": task_data["filename"],
            "progress_percentage": task_data.get("progress", {}).get("percentage", 0),
            "message": task_data["message"],
            "created_at": task_data["created_at"].isoformat(),
            "updated_at": task_data["updated_at"].isoformat()
        }
        tasks.append(task_summary)
    
    return {
        "tasks": tasks,
        "total_count": len(tasks)
    }

# =============================================================================
# Export Endpoints
# =============================================================================

@app.post("/api/export/{task_id}")
async def export_assessment(
    task_id: str,
    request: ExportRequest
):
    """Export assessment to various formats"""
    try:
        # Check if task exists and has results
        if task_id not in task_storage:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task_data = task_storage[task_id]
        if task_data["status"] != ProcessingStatus.READY:
            raise HTTPException(
                status_code=400, 
                detail=f"Task not ready for export. Current status: {task_data['status']}"
            )
        
        # Use cached assessment from task_storage
        assessment = task_data.get("assessment_result")
        if not assessment:
            raise HTTPException(status_code=400, detail="No assessment result found for export")
        
        logger.info(f"Using cached assessment for export - Task ID: {task_id} with {len(assessment.questions)} questions")
        
        # Map frontend format names to backend format names
        format_mapping = {
            'docx': 'word',
            'pdf-teacher': 'pdf',
            'pdf-student': 'pdf'
        }
        
        # Use mapped format or original format
        backend_format = format_mapping.get(request.format_type, request.format_type)
        
        # Configure export settings
        config = ExportConfiguration()
        config.teacher_version = request.teacher_version
        config.include_explanations = request.include_explanations
        config.include_difficulty = request.include_difficulty
        
        # Determine output filename
        base_filename = f"assessment_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_path = export_manager.validate_output_path(
            f"/tmp/{base_filename}", 
            backend_format
        )
        
        # Export assessment
        exported_file = export_manager.export_assessment(
            assessment, 
            backend_format, 
            output_path, 
            config
        )
        
        logger.info(f"Assessment exported successfully: {exported_file}")
        
        # Return file for download
        if Path(exported_file).exists():
            from fastapi.responses import FileResponse
            
            # Determine media type
            media_types = {
                'json': 'application/json',
                'pdf': 'application/pdf',
                'pdf-teacher': 'application/pdf',
                'pdf-student': 'application/pdf',
                'word': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'text': 'text/plain',
                'markdown': 'text/markdown',
                'qti': 'application/xml',
                'moodle': 'application/xml',
                'canvas': 'text/csv',
                'google_forms': 'text/csv'
            }
            
            media_type = media_types.get(request.format_type, 'application/octet-stream')
            filename = Path(exported_file).name
            
            return FileResponse(
                path=exported_file,
                media_type=media_type,
                filename=filename
            )
        else:
            raise HTTPException(status_code=500, detail="Export file not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export failed for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.post("/api/export/{task_id}/pdf")
async def export_assessment_pdf(
    task_id: str,
    request: ExportRequest
):
    """Export assessment to PDF format with proper download headers"""
    try:
        logger.info(f"PDF export requested for task {task_id}, teacher_version: {request.teacher_version}")
        
        # Check if task exists and has results
        if task_id not in task_storage:
            logger.error(f"Task {task_id} not found in task_storage")
            raise HTTPException(status_code=404, detail="Task not found")
        
        task_data = task_storage[task_id]
        logger.info(f"Task data status: {task_data['status']}")
        
        if task_data["status"] != ProcessingStatus.READY:
            raise HTTPException(
                status_code=400, 
                detail=f"Task not ready for export. Current status: {task_data['status']}"
            )
        
        # Use cached assessment from task_storage 
        assessment = task_data.get("assessment_result")
        if not assessment:
            raise HTTPException(status_code=400, detail="No assessment result found for export")
        
        logger.info(f"Using cached assessment with {len(assessment.questions)} questions")
        
        # Create exporter and generate PDF via Word document conversion
        try:
            exporter = AssessmentExporter()
            logger.info("Creating PDF by converting Word document...")
            
            # First create Word document in memory
            word_buffer = exporter.export_to_docx_buffer_assessment(
                assessment, 
                include_answers=request.teacher_version,
                include_explanations=request.include_explanations
            )
            
            # Convert Word to PDF using LibreOffice
            pdf_buffer = exporter.convert_word_to_pdf(word_buffer)
            
            logger.info("PDF created successfully from Word document")
        except Exception as pdf_error:
            logger.error(f"PDF creation failed: {str(pdf_error)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(pdf_error)}")
        
        # Generate filename
        version_suffix = "_teacher" if request.teacher_version else "_student"
        filename = f"assessment_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{version_suffix}.pdf"
        logger.info(f"Generated filename: {filename}")
        
        # Return PDF with proper headers
        try:
            pdf_content = pdf_buffer.read()
            logger.info(f"PDF content size: {len(pdf_content)} bytes")
            
            return StreamingResponse(
                BytesIO(pdf_content),
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}",
                    "Content-Type": "application/pdf"
                }
            )
        except Exception as response_error:
            logger.error(f"PDF response creation failed: {str(response_error)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"PDF response failed: {str(response_error)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF export failed for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF export failed: {str(e)}")

@app.post("/api/export/{task_id}/docx")
async def export_assessment_docx(
    task_id: str,
    teacher_version: bool = True,
    include_explanations: bool = True
):
    """Export assessment to Word document format with proper download headers"""
    try:
        logger.info(f"Word export requested for task {task_id}, teacher_version: {teacher_version}")
        
        # Check if task exists and has results
        if task_id not in task_storage:
            logger.error(f"Task {task_id} not found in task_storage")
            raise HTTPException(status_code=404, detail="Task not found")
        
        task_data = task_storage[task_id]
        logger.info(f"Task data status: {task_data['status']}")
        
        if task_data["status"] != ProcessingStatus.READY:
            raise HTTPException(
                status_code=400, 
                detail=f"Task not ready for export. Current status: {task_data['status']}"
            )
        
        # Check if we have assessment results stored
        if task_id not in processed_documents:
            logger.error(f"Task {task_id} not found in processed_documents")
            raise HTTPException(status_code=400, detail="No assessment results found for export")
        
        # Get stored assessment data
        doc_data = processed_documents[task_id]
        retriever = doc_data["retriever"]
        
        # Generate assessment for export
        logger.info(f"Generating Word assessment for export - Task ID: {task_id}")
        assessment = await generator.generate_assessment(retriever, 10, "medium")
        logger.info(f"Generated assessment with {len(assessment.questions)} questions")
        
        # Convert assessment to simple format for Word generation
        questions = []
        for i, q in enumerate(assessment.questions):
            logger.debug(f"Processing question {i+1}: {q.question_text[:50]}...")
            
            question_dict = {
                'question_text': q.question_text,
                'correct_answer': q.correct_answer,
                'explanation': q.explanation,
                'difficulty': q.difficulty.value if hasattr(q.difficulty, 'value') else str(q.difficulty),
                'options': {}
            }
            
            # Create options dict
            question_dict['options']['A'] = {
                'text': q.correct_answer,
                'is_correct': True
            }
            
            for j, distractor in enumerate(q.distractors[:3]):
                letter = chr(66 + j)  # B, C, D
                distractor_text = distractor.text if hasattr(distractor, 'text') else str(distractor)
                question_dict['options'][letter] = {
                    'text': distractor_text,
                    'is_correct': False
                }
            
            questions.append(question_dict)
            logger.debug(f"Question {i+1} formatted successfully")
        
        logger.info(f"All {len(questions)} questions formatted for Word generation")
        
        # Create exporter and generate Word document
        try:
            exporter = AssessmentExporter()
            logger.info("Creating Word document buffer...")
            docx_buffer = exporter.export_to_docx_buffer(questions, include_answers=teacher_version)
            logger.info("Word document buffer created successfully")
        except Exception as docx_error:
            logger.error(f"Word document buffer creation failed: {str(docx_error)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Word document generation failed: {str(docx_error)}")
        
        # Generate filename
        version_suffix = "_teacher" if teacher_version else "_student"
        filename = f"assessment_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{version_suffix}.docx"
        logger.info(f"Generated filename: {filename}")
        
        # Return Word document with proper headers
        try:
            docx_content = docx_buffer.read()
            logger.info(f"Word document content size: {len(docx_content)} bytes")
            
            return StreamingResponse(
                BytesIO(docx_content),
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}",
                    "Content-Type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                }
            )
        except Exception as response_error:
            logger.error(f"Word document response creation failed: {str(response_error)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Word document response failed: {str(response_error)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Word export failed for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Word export failed: {str(e)}")

@app.get("/api/export/formats")
async def get_export_formats():
    """Get list of supported export formats"""
    formats = export_manager.get_supported_formats()
    
    format_info = {
        'json': {
            'name': 'JSON',
            'description': 'Machine-readable format with full metadata',
            'extension': '.json',
            'use_cases': ['API integration', 'data analysis', 'backup']
        },
        'pdf': {
            'name': 'PDF Document',
            'description': 'Professional formatted document',
            'extension': '.pdf',
            'use_cases': ['printing', 'sharing', 'archival']
        },
        'word': {
            'name': 'Word Document',
            'description': 'Microsoft Word compatible format',
            'extension': '.docx',
            'use_cases': ['editing', 'collaboration', 'institutional use']
        },
        'text': {
            'name': 'Plain Text',
            'description': 'Simple text format for basic use',
            'extension': '.txt',
            'use_cases': ['basic sharing', 'email', 'simple viewing']
        },
        'markdown': {
            'name': 'Markdown',
            'description': 'Formatted text for documentation',
            'extension': '.md',
            'use_cases': ['documentation', 'GitHub', 'technical writing']
        },
        'qti': {
            'name': 'QTI XML',
            'description': 'Standard e-learning format',
            'extension': '.xml',
            'use_cases': ['LMS integration', 'e-learning platforms']
        },
        'moodle': {
            'name': 'Moodle XML',
            'description': 'Moodle-specific import format',
            'extension': '.xml',
            'use_cases': ['Moodle courses', 'educational institutions']
        },
        'canvas': {
            'name': 'Canvas CSV',
            'description': 'Canvas LMS import format',
            'extension': '.csv',
            'use_cases': ['Canvas courses', 'university systems']
        },
        'google_forms': {
            'name': 'Google Forms CSV',
            'description': 'Google Forms compatible format',
            'extension': '.csv',
            'use_cases': ['Google Classroom', 'quick surveys']
        }
    }
    
    return {
        "supported_formats": formats,
        "format_details": {fmt: format_info.get(fmt, {}) for fmt in formats}
    }

# =============================================================================
# Question Regeneration Endpoints
# =============================================================================

@app.post("/api/regenerate/{task_id}")
async def regenerate_questions(
    task_id: str,
    request: RegenerateRequest
):
    """Regenerate specific questions or all questions below quality threshold"""
    try:
        # Check if task exists and has results
        if task_id not in task_storage:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task_data = task_storage[task_id]
        if task_data["status"] != ProcessingStatus.READY:
            raise HTTPException(
                status_code=400, 
                detail=f"Task not ready for regeneration. Current status: {task_data['status']}"
            )
        
        # Check if document is processed
        if task_id not in processed_documents:
            raise HTTPException(status_code=400, detail="Document processing data not found")
        
        # Validate OpenAI configuration
        if not settings.is_openai_configured():
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured. Please set OPENAI_API_KEY in environment."
            )
        
        # Get processed document data
        doc_data = processed_documents[task_id]
        retriever = doc_data["retriever"]
        
        # Update task status
        task_data["status"] = ProcessingStatus.PROCESSING
        task_data["message"] = "Regenerating questions..."
        task_data["updated_at"] = datetime.now()
        
        logger.info(f"Starting question regeneration for task {task_id}")
        
        # Get existing assessment from task storage
        existing_assessment = task_data.get("result")
        if not existing_assessment:
            # If no cached result, try to get from processed documents and generate new
            if task_id not in processed_documents:
                raise HTTPException(status_code=400, detail="No assessment or document data found for regeneration")
            
            doc_data = processed_documents[task_id]
            retriever = doc_data["retriever"]
            
            # Generate fresh assessment
            logger.info("Generating new assessment for regeneration")
            existing_assessment = await generator.generate_assessment(retriever, 10, "medium")
            
            # Cache it
            task_data["result"] = existing_assessment
        
        # If we need to regenerate all questions below threshold
        if request.regenerate_all_below_threshold:
            # For now, just generate a completely new assessment
            assessment = await generator.generate_assessment(
                retriever, 
                len(existing_assessment.questions),  # Same number as existing
                "medium"
            )
            regenerated_count = len(assessment.questions)
        else:
            # For specific question regeneration, replace only specified questions
            if not request.question_indices:
                raise HTTPException(status_code=400, detail="No question indices specified for regeneration")
            
            # Generate replacement questions
            num_to_regenerate = len(request.question_indices)
            new_questions_assessment = await generator.generate_assessment(
                retriever, 
                num_to_regenerate,
                "medium"
            )
            
            # Replace specific questions in existing assessment
            assessment = existing_assessment
            for i, question_index in enumerate(request.question_indices):
                if 0 <= question_index < len(assessment.questions):
                    if i < len(new_questions_assessment.questions):
                        assessment.questions[question_index] = new_questions_assessment.questions[i]
            
            regenerated_count = min(len(request.question_indices), len(new_questions_assessment.questions))
        
        # Enhance assessment with metadata
        assessment.source_file = task_data["filename"]
        assessment.statistics.update({
            "task_id": task_id,
            "regeneration_type": "selective" if request.question_indices else "quality_threshold",
            "regenerated_count": regenerated_count,
            "quality_threshold": request.quality_threshold,
            "model_used": settings.OPENAI_MODEL
        })
        
        # IMPORTANT: Store the updated assessment back to task storage
        task_data["result"] = assessment
        
        # Update task status
        task_data["status"] = ProcessingStatus.READY
        task_data["message"] = f"Regenerated {regenerated_count} questions"
        task_data["updated_at"] = datetime.now()
        
        return {
            "task_id": task_id,
            "regenerated_count": regenerated_count,
            "statistics": assessment.statistics,
            "message": f"Successfully regenerated {regenerated_count} questions"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Question regeneration failed for task {task_id}: {e}")
        
        # Update task status to error
        if task_id in task_storage:
            task_storage[task_id]["status"] = ProcessingStatus.ERROR
            task_storage[task_id]["message"] = f"Regeneration failed: {str(e)}"
            task_storage[task_id]["updated_at"] = datetime.now()
        
        raise HTTPException(status_code=500, detail=f"Question regeneration failed: {str(e)}")

# =============================================================================
# Analytics Endpoints
# =============================================================================

@app.get("/api/analytics/{task_id}")
async def get_task_analytics(task_id: str):
    """Get detailed analytics for a specific task"""
    try:
        if task_id not in task_storage:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Generate analytics report
        report = analytics_manager.generate_analytics_report(task_id)
        return report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics generation failed: {str(e)}")

@app.get("/api/analytics/performance/summary")
async def get_performance_summary(days: int = 30):
    """Get performance summary for the last N days"""
    try:
        summary = analytics_manager.performance_tracker.get_performance_summary(days)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance summary failed: {str(e)}")

@app.post("/api/analytics/analyze")
async def analyze_assessment_quality(assessment: AssessmentResponse):
    """Analyze quality of an assessment"""
    try:
        analysis = analytics_manager.assessment_analyzer.analyze_assessment(assessment)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality analysis failed: {str(e)}")

# =============================================================================
# Optimization & Performance Endpoints
# =============================================================================

@app.get("/api/optimization/status")
async def get_optimization_status():
    """Get comprehensive optimization and performance status"""
    try:
        status = optimization_manager.get_optimization_status()
        return {
            "status": "success",
            "optimization_data": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get optimization status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get optimization status: {str(e)}")

@app.get("/api/optimization/cache/stats")
async def get_cache_statistics():
    """Get detailed cache performance statistics"""
    try:
        cache_stats = optimization_manager.question_cache.get_cache_stats()
        return {
            "cache_performance": cache_stats,
            "recommendations": optimization_manager.get_optimization_recommendations(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache statistics: {str(e)}")

@app.post("/api/optimization/cache/clear")
async def clear_optimization_cache():
    """Clear all optimization caches"""
    try:
        # Clear question cache
        optimization_manager.question_cache.cleanup_expired_cache()
        
        # Force cleanup of temporary files
        optimization_manager.memory_manager.cleanup_temp_files()
        
        # Force garbage collection
        optimization_manager.memory_manager.force_garbage_collection()
        
        return {
            "status": "success",
            "message": "All caches cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@app.get("/api/optimization/memory")
async def get_memory_status():
    """Get current memory usage and management status"""
    try:
        memory_usage = optimization_manager.memory_manager.get_memory_usage_mb()
        memory_within_limits = optimization_manager.memory_manager.check_memory_limit()
        
        return {
            "memory_usage_mb": memory_usage,
            "within_limits": memory_within_limits,
            "max_limit_mb": optimization_manager.memory_manager.max_memory_mb,
            "temp_files_count": len(optimization_manager.memory_manager.temp_files),
            "recommendations": optimization_manager.get_optimization_recommendations()
        }
    except Exception as e:
        logger.error(f"Failed to get memory status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory status: {str(e)}")

@app.get("/api/optimization/performance/analytics")
async def get_performance_analytics():
    """Get detailed performance analytics from database"""
    try:
        analytics = optimization_manager.db_optimizer.get_performance_analytics()
        return {
            "performance_data": analytics,
            "optimization_suggestions": optimization_manager.get_optimization_recommendations(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get performance analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance analytics: {str(e)}")

@app.post("/api/optimization/database/vacuum")
async def vacuum_optimization_database():
    """Optimize database by running VACUUM operation"""
    try:
        optimization_manager.db_optimizer.vacuum_database()
        return {
            "status": "success",
            "message": "Database optimization completed",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to vacuum database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to vacuum database: {str(e)}")

# =============================================================================
# Health & System Status
# =============================================================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check for the system"""
    try:
        # Test OpenAI connectivity
        openai_status = "healthy" if settings.is_openai_configured() else "unhealthy"
        
        # Test vector database
        try:
            collections = rag_pipeline.list_collections()
            vectordb_status = "healthy"
        except Exception as e:
            logger.warning(f"Vector database health check failed: {e}")
            vectordb_status = "unhealthy"
        
        # System statistics
        system_stats = {
            "active_tasks": len([t for t in task_storage.values() if t["status"] == ProcessingStatus.PROCESSING]),
            "completed_tasks": len([t for t in task_storage.values() if t["status"] == ProcessingStatus.READY]),
            "error_tasks": len([t for t in task_storage.values() if t["status"] == ProcessingStatus.ERROR]),
            "total_collections": len(collections) if vectordb_status == "healthy" else 0,
            "cache_size": rag_pipeline.get_cache_info() if hasattr(rag_pipeline, 'get_cache_info') else {},
            "uptime": datetime.now().isoformat()
        }
        
        overall_status = "healthy" if openai_status == "healthy" and vectordb_status == "healthy" else "degraded"
        
        return HealthResponse(
            status=overall_status,
            openai_status=openai_status,
            vectordb_status=vectordb_status,
            system_stats=system_stats,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            openai_status="unknown",
            vectordb_status="unknown",
            system_stats={},
            timestamp=datetime.now(),
            error=str(e)
        )

# =============================================================================
# Legacy endpoint for backward compatibility
# =============================================================================

@app.post("/upload", response_model=AssessmentResponse)
async def upload_and_generate_legacy(
    file: UploadFile = File(...),
    num_questions: int = Form(5),
    difficulty: str = Form("medium")
):
    """Legacy endpoint: Upload PDF and generate assessment questions directly"""
    try:
        # Validate OpenAI configuration
        if not settings.is_openai_configured():
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured. Please set OPENAI_API_KEY in environment."
            )
        
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Process PDF
        content = await pdf_processor.process_pdf(file)
        
        # Build RAG pipeline
        retriever = await rag_pipeline.build_pipeline(content)
        
        # Generate assessment
        assessment = await generator.generate_assessment(
            retriever, num_questions, difficulty
        )
        
        return assessment
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Enhanced Background Tasks
# =============================================================================

async def process_pdf_enhanced(task_id: str, file_path: str):
    """Enhanced background task for intelligent PDF processing with detailed progress tracking"""
    try:
        logger.info(f"Starting enhanced PDF processing for task: {task_id}")
        
        # Step 1: Text Extraction
        update_task_progress(task_id, "text_extraction", "Extracting text from PDF...", 15)
        
        # Process PDF with intelligent chunking
        task_storage[task_id]["timestamps"]["text_extraction_start"] = datetime.now()
        pdf_result = pdf_processor.process_pdf(file_path)  # Use sync version with file path
        task_storage[task_id]["timestamps"]["text_extraction_complete"] = datetime.now()
        
        # Convert dictionary result to object-like structure for compatibility
        class PDFContentWrapper:
            def __init__(self, result_dict, filename):
                self.chunks = result_dict.get("chunks", [])
                self.metadata = result_dict.get("metadata", {})
                self.pages_data = result_dict.get("pages_data", [])
                self.filename = filename
                
                # Add intelligent chunks to metadata for backward compatibility
                # The chunks from process_pdf are already intelligent chunks
                if self.chunks:
                    self.metadata["intelligent_chunks"] = self.chunks
        
        filename = task_storage[task_id]["filename"]
        pdf_content = PDFContentWrapper(pdf_result, filename)
        
        # Extract processing stats
        processing_stats = {
            "total_pages": pdf_content.metadata.get("num_pages", 0),
            "extraction_method": pdf_content.metadata.get("extraction_method", "unknown"),
            "processing_warnings": pdf_content.metadata.get("processing_warnings", [])
        }
        
        # Step 2: Chunking Analysis
        update_task_progress(task_id, "chunking", "Analyzing document structure and creating chunks...", 35)
        
        # Get intelligent chunks if available
        intelligent_chunks = pdf_content.metadata.get("intelligent_chunks", [])
        total_chunks = len(intelligent_chunks) if intelligent_chunks else len(pdf_content.chunks)
        
        processing_stats.update({
            "total_chunks": total_chunks,
            "chunking_method": "intelligent" if intelligent_chunks else "simple",
            "avg_chunk_size": sum(len(chunk.get("text", "")) for chunk in intelligent_chunks) // max(1, total_chunks) if intelligent_chunks else 0
        })
        
        task_storage[task_id]["progress"]["total_chunks"] = total_chunks
        task_storage[task_id]["processing_stats"] = processing_stats
        
        # Step 3: Embedding Generation
        update_task_progress(task_id, "embedding_generation", "Generating embeddings for content chunks...", 55)
        task_storage[task_id]["timestamps"]["embedding_start"] = datetime.now()
        
        # Process PDF content through RAG pipeline
        collection_name = f"task_{task_id}"
        success = await rag_pipeline.process_pdf_content(pdf_content, collection_name)
        
        if not success:
            raise Exception("Failed to process PDF content through RAG pipeline")
        
        task_storage[task_id]["timestamps"]["embedding_complete"] = datetime.now()
        
        # Step 4: Vector Storage
        update_task_progress(task_id, "vector_storage", "Building vector index for intelligent retrieval...", 85)
        
        # Get retriever for question generation
        retriever = rag_pipeline.get_retriever(collection_name)
        
        # Get collection info for stats
        collection_info = rag_pipeline.get_collection_info(collection_name)
        processing_stats.update({
            "vector_store_count": collection_info.get("count", 0),
            "collection_name": collection_name
        })
        
        # Step 5: Completion
        update_task_progress(task_id, "completed", "Processing completed successfully!", 100)
        
        # Store processed document data
        processed_documents[task_id] = {
            "content": pdf_content,
            "retriever": retriever,
            "collection_name": collection_name,
            "processing_stats": processing_stats,
            "processed_at": datetime.now()
        }
        
        # Final status update
        task_storage[task_id]["status"] = ProcessingStatus.READY
        task_storage[task_id]["progress"]["current_step"] = "ready"
        task_storage[task_id]["progress"]["steps_completed"].append("completed")
        task_storage[task_id]["message"] = f"Ready for question generation! Processed {total_chunks} chunks."
        task_storage[task_id]["timestamps"]["processing_complete"] = datetime.now()
        task_storage[task_id]["updated_at"] = datetime.now()
        
        # Calculate total processing time
        start_time = task_storage[task_id]["timestamps"]["upload_start"]
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        task_storage[task_id]["processing_stats"]["total_processing_time"] = processing_time
        
        logger.info(f"PDF processing completed for task {task_id} in {processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"PDF processing failed for task {task_id}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Update task status to error
        task_storage[task_id]["status"] = ProcessingStatus.ERROR
        task_storage[task_id]["message"] = f"Processing failed: {str(e)}"
        task_storage[task_id]["error_details"] = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "failed_at_step": task_storage[task_id]["progress"].get("current_step", "unknown")
        }
        task_storage[task_id]["timestamps"]["error_time"] = datetime.now()
        task_storage[task_id]["updated_at"] = datetime.now()

def update_task_progress(task_id: str, step: str, message: str, percentage: int):
    """Helper function to update task progress consistently"""
    if task_id not in task_storage:
        return
    
    task_data = task_storage[task_id]
    
    # Update progress details
    task_data["progress"]["current_step"] = step
    task_data["progress"]["percentage"] = percentage
    
    # Add to completed steps if not already there
    if step not in task_data["progress"]["steps_completed"]:
        task_data["progress"]["steps_completed"].append(step)
    
    # Update message and timestamp
    task_data["message"] = message
    task_data["updated_at"] = datetime.now()
    
    logger.info(f"Task {task_id}: {step} - {message} ({percentage}%)")

async def process_pdf_background(task_id: str, file_path):
    """Legacy background task - redirects to enhanced version"""
    await process_pdf_enhanced(task_id, file_path)

# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG
    )
