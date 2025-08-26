"""
PDF processing and text chunking for STEM Assessment Generator
"""
import PyPDF2
import io
import os
from typing import List, Dict, Any
from fastapi import UploadFile

from config import settings
from models import PDFContent

class PDFProcessor:
    """Handles PDF parsing and text chunking"""
    
    def __init__(self):
        self.max_file_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
    
    async def process_pdf(self, file: UploadFile) -> PDFContent:
        """
        Process uploaded PDF file and extract text content
        
        Args:
            file: Uploaded PDF file
            
        Returns:
            PDFContent: Processed content with chunks
        """
        # Validate file size
        content = await file.read()
        if len(content) > self.max_file_size:
            raise ValueError(f"File size exceeds {settings.MAX_FILE_SIZE_MB}MB limit")
        
        # Extract text from PDF
        text = self._extract_text_from_pdf(content)
        
        # Create chunks
        chunks = self._create_chunks(text)
        
        # Create metadata
        metadata = {
            "filename": file.filename,
            "file_size": len(content),
            "num_chunks": len(chunks),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
        
        return PDFContent(
            filename=file.filename,
            content=text,
            chunks=chunks,
            metadata=metadata
        )
    
    def _extract_text_from_pdf(self, content: bytes) -> str:
        """
        Extract text content from PDF bytes
        
        Args:
            content: PDF file content as bytes
            
        Returns:
            str: Extracted text content
        """
        try:
            pdf_stream = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
            
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    def _create_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            
        Returns:
            List[str]: List of text chunks
        """
        if not text:
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
            
            # Stop if we've reached the end
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    def save_upload(self, file_content: bytes, filename: str) -> str:
        """
        Save uploaded file to disk (optional)
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            str: Path to saved file
        """
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        return file_path
