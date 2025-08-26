"""
Main FastAPI application for STEM Assessment Generator
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import os

from config import settings
from models import AssessmentRequest, AssessmentResponse
from pdf_processor import PDFProcessor
from rag_pipeline import RAGPipeline
from generator import AssessmentGenerator

app = FastAPI(title="STEM Assessment Generator", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize components
pdf_processor = PDFProcessor()
rag_pipeline = RAGPipeline()
generator = AssessmentGenerator()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/upload", response_model=AssessmentResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    num_questions: int = Form(5),
    difficulty: str = Form("medium")
):
    """Upload PDF and generate assessment questions"""
    try:
        # Validate file
        if not file.filename.endswith('.pdf'):
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG
    )
