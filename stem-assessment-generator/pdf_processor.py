"""
Enhanced PDF processing and intelligent chunking for STEM Assessment Generator
Features: semantic chunking, section detection, content type classification, 
mathematical equation detection, table extraction, code block preservation
"""
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
import io
import os
import re
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from fastapi import UploadFile
from pathlib import Path

from config import settings
from models import PDFContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Advanced PDF parsing and intelligent text chunking"""
    
    def __init__(self):
        self.max_file_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
        self.default_chunk_size = 500  # tokens
        self.default_overlap = 50  # tokens
        
        # Enhanced patterns for content detection
        self.header_patterns = [
            r'^(Chapter|Section|Part)\s+\d+',
            r'^\d+\.?\s+[A-Z][a-z]',
            r'^[A-Z][A-Z\s]{2,20}$',
            r'^\d+\.\d+\s+[A-Z]'
        ]
        
        # Mathematical equation patterns (enhanced)
        self.equation_patterns = [
            r'\$[^$]+\$',  # LaTeX inline math
            r'\$\$[^$]+\$\$',  # LaTeX display math
            r'\\begin\{equation\}.*?\\end\{equation\}',
            r'\\begin\{align\}.*?\\end\{align\}',
            r'\\begin\{matrix\}.*?\\end\{matrix\}',
            r'\\begin\{cases\}.*?\\end\{cases\}',
            r'\\frac\{[^}]+\}\{[^}]+\}',  # Fractions
            r'\\sum_\{[^}]*\}\^\{[^}]*\}',  # Summations
            r'\\int_\{[^}]*\}\^\{[^}]*\}',  # Integrals
            r'\\lim_\{[^}]*\}',  # Limits
            r'\b(?:\w+\s*[=≠<>≤≥±∞]+\s*\w+|\w+\s*[+\-*/∇∂]+\s*\w+)',  # Basic equations with symbols
            r'[α-ωΑ-Ω]+',  # Greek letters
            r'[∑∫∂∇∞±≤≥≠≈∈∉⊂⊃∪∩∧∨¬→↔∀∃]+',  # Mathematical symbols
        ]
        
        # Code patterns (enhanced)
        self.code_patterns = [
            r'```[^`]*```',  # Markdown code blocks
            r'`[^`]+`',  # Inline code
            r'def\s+\w+\s*\([^)]*\):',  # Python functions
            r'function\s+\w+\s*\([^)]*\)',  # JavaScript functions
            r'class\s+\w+\s*[\{:]',  # Class definitions
            r'for\s*\([^)]+\)\s*\{',  # For loops
            r'if\s*\([^)]+\)\s*\{',  # If statements
            r'while\s*\([^)]+\)\s*\{',  # While loops
            r'#include\s*<[^>]+>',  # C/C++ includes
            r'import\s+\w+',  # Import statements
            r'from\s+\w+\s+import',  # Python imports
            r'^\s{4,}\w+.*$',  # Indented code blocks (4+ spaces)
        ]
        
        # Table detection patterns
        self.table_patterns = [
            r'\|[^|]*\|[^|]*\|',  # Markdown tables
            r'┌[─┬]*┐',  # ASCII table borders
            r'├[─┼]*┤',  # ASCII table separators
            r'└[─┴]*┘',  # ASCII table bottom
            r'\+[-+]*\+',  # Simple ASCII tables
        ]
        
        # Figure/diagram patterns
        self.figure_patterns = [
            r'Figure\s+\d+[:\-\s]*[^\n]*',
            r'Fig\.\s+\d+[:\-\s]*[^\n]*',
            r'Diagram\s+\d+[:\-\s]*[^\n]*',
            r'Chart\s+\d+[:\-\s]*[^\n]*',
            r'Graph\s+\d+[:\-\s]*[^\n]*',
            r'\[FIGURE\s+\d+[^\]]*\]',
            r'\[IMAGE[^\]]*\]',
            r'\[DIAGRAM[^\]]*\]',
        ]
        
        # Symbol mapping for preservation
        self.symbol_map = {
            # Greek letters
            'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon',
            'ζ': 'zeta', 'η': 'eta', 'θ': 'theta', 'ι': 'iota', 'κ': 'kappa',
            'λ': 'lambda', 'μ': 'mu', 'ν': 'nu', 'ξ': 'xi', 'π': 'pi',
            'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon', 'φ': 'phi',
            'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',
            # Mathematical operators
            '∑': 'sum', '∫': 'integral', '∂': 'partial', '∇': 'nabla',
            '∞': 'infinity', '±': 'plus_minus', '≤': 'less_equal', '≥': 'greater_equal',
            '≠': 'not_equal', '≈': 'approximately', '∈': 'element_of', '∉': 'not_element_of',
            '⊂': 'subset', '⊃': 'superset', '∪': 'union', '∩': 'intersection',
            '∧': 'and', '∨': 'or', '¬': 'not', '→': 'implies', '↔': 'if_and_only_if',
            '∀': 'for_all', '∃': 'exists',
        }
        
        # Keywords for importance scoring
        self.importance_keywords = {
            'high': ['definition', 'theorem', 'principle', 'law', 'algorithm', 'formula', 'proof', 'concept'],
            'medium': ['example', 'application', 'method', 'process', 'procedure', 'technique', 'approach'],
            'low': ['note', 'remark', 'aside', 'reference', 'see also', 'further reading']
        }
        
        # Section title patterns
        self.section_patterns = [
            r'^(\d+\.?\d*\.?\d*)\s+([A-Z][^.!?]*)',  # Numbered sections
            r'^([A-Z][A-Z\s]{2,50})$',  # All caps headings
            r'^(Introduction|Conclusion|Summary|Abstract|Overview|Background)',  # Common sections
            r'^(Chapter|Section|Part)\s+(\d+)[:\-\s]*([A-Z][^.!?]*)',
        ]
    
    async def process_pdf(self, file: UploadFile) -> PDFContent:
        """
        Main entry point for PDF processing with intelligent chunking and STEM content analysis
        
        Args:
            file: Uploaded PDF file
            
        Returns:
            PDFContent: Processed content with intelligent chunks and STEM analysis
        """
        logger.info(f"Processing PDF: {file.filename}")
        
        # Validate file size
        content = await file.read()
        if len(content) > self.max_file_size:
            raise ValueError(f"File size exceeds {settings.MAX_FILE_SIZE_MB}MB limit")
        
        # Save file temporarily for processing
        file_path = self.save_upload(content, file.filename)
        
        try:
            # Extract structured text from PDF with enhanced analysis
            pages_data = self.extract_text_from_pdf(file_path)
            
            # Perform STEM-specific content analysis
            equations_data = self.extract_equations_from_pages(pages_data)
            tables_data = self.extract_tables_from_pdf(file_path)
            figures_data = self.extract_figures_from_pages(pages_data)
            code_blocks = self.extract_code_blocks_from_pages(pages_data)
            
            # Create intelligent chunks with STEM context
            chunks_data = self.create_chunks_with_stem_context(
                pages_data, equations_data, tables_data, figures_data, code_blocks
            )
            
            # Combine all text for legacy compatibility
            full_text = "\n".join([page["text"] for page in pages_data])
            
            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(
                equations_data, tables_data, code_blocks, pages_data
            )
            
            # Create enhanced metadata
            metadata = {
                "filename": file.filename,
                "file_size": len(content),
                "num_pages": len(pages_data),
                "num_chunks": len(chunks_data),
                "extraction_method": "intelligent_chunking_with_stem",
                "processing_warnings": self._get_processing_warnings(pages_data),
                "has_math": len(equations_data) > 0,
                "has_code": len(code_blocks) > 0,
                "has_tables": len(tables_data) > 0,
                "has_figures": len(figures_data) > 0,
                "complexity_score": complexity_score,
                "equations_count": len(equations_data),
                "tables_count": len(tables_data),
                "figures_count": len(figures_data),
                "code_blocks_count": len(code_blocks)
            }
            
            return PDFContent(
                filename=file.filename,
                content=full_text,
                chunks=[chunk["text"] for chunk in chunks_data],  # For legacy compatibility
                metadata={
                    **metadata,
                    "intelligent_chunks": chunks_data,  # Store full chunk data
                    "equations": equations_data,
                    "tables": tables_data,
                    "figures": figures_data,
                    "code_blocks": code_blocks
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing PDF {file.filename}: {e}")
            # Fallback to basic processing
            return await self._fallback_processing(file, content)
        
        finally:
            # Clean up temporary file
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def extract_text_from_pdf(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF with page-level metadata and error handling
        
        Args:
            filepath: Path to PDF file
            
        Returns:
            List[Dict]: Page data with text and metadata
        """
        pages_data = []
        
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                logger.info(f"PDF has {len(pdf_reader.pages)} pages")
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        # Extract text with multiple strategies
                        raw_text = self._extract_page_text(page, page_num)
                        
                        if not raw_text.strip():
                            logger.warning(f"Page {page_num}: No extractable text found")
                            continue
                        
                        # Clean and preprocess text
                        cleaned_text = self.clean_text(raw_text)
                        
                        # Detect content features
                        page_metadata = self._analyze_page_content(cleaned_text, page_num)
                        
                        page_data = {
                            "page_num": page_num,
                            "text": cleaned_text,
                            "metadata": page_metadata
                        }
                        
                        pages_data.append(page_data)
                        
                    except Exception as e:
                        logger.warning(f"Page {page_num}: Failed to extract text - {e}")
                        continue
                
        except Exception as e:
            logger.error(f"Failed to read PDF file: {e}")
            raise ValueError(f"Failed to process PDF: {str(e)}")
        
        if not pages_data:
            raise ValueError("No text could be extracted from the PDF")
        
        return pages_data
    
    def _extract_page_text(self, page, page_num: int) -> str:
        """Extract text from a single page with multiple strategies"""
        try:
            # Primary extraction method
            text = page.extract_text()
            
            # If primary method fails, try alternative extraction
            if not text.strip():
                # Try different extraction parameters if available
                if hasattr(page, 'extract_text_simple'):
                    text = page.extract_text_simple()
            
            return text
            
        except Exception as e:
            logger.warning(f"Page {page_num}: Text extraction failed - {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text while preserving important elements
        
        Args:
            text: Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Store equations and code blocks temporarily
        equations = []
        code_blocks = []
        
        # Extract and preserve equations
        for pattern in self.equation_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                placeholder = f"__EQUATION_{len(equations)}__"
                equations.append(match)
                text = text.replace(match, placeholder, 1)
        
        # Extract and preserve code blocks
        for pattern in self.code_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                placeholder = f"__CODE_{len(code_blocks)}__"
                code_blocks.append(match)
                text = text.replace(match, placeholder, 1)
        
        # Basic text cleaning
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Remove excessive line breaks
        text = re.sub(r'^[\s\n]+|[\s\n]+$', '', text)  # Trim leading/trailing whitespace
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between lowercase and uppercase
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)  # Add space between digits and letters
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Add space after sentence endings
        
        # Preserve bullet points and numbered lists
        text = re.sub(r'•\s*', '• ', text)  # Normalize bullet points
        text = re.sub(r'(\d+)\.\s*', r'\1. ', text)  # Normalize numbered lists
        
        # Restore equations and code blocks
        for i, equation in enumerate(equations):
            text = text.replace(f"__EQUATION_{i}__", equation)
        
        for i, code_block in enumerate(code_blocks):
            text = text.replace(f"__CODE_{i}__", code_block)
        
        return text.strip()
    
    def _analyze_page_content(self, text: str, page_num: int) -> Dict[str, Any]:
        """Analyze page content to extract metadata"""
        metadata = {
            "page_num": page_num,
            "word_count": len(text.split()),
            "char_count": len(text),
            "has_equations": bool(self._find_equations(text)),
            "has_code": bool(self._find_code_blocks(text)),
            "section_titles": self._extract_section_titles(text),
            "content_density": len(text.split()) / max(1, text.count('\n') + 1)
        }
        
        return metadata
    
    def _find_equations(self, text: str) -> List[str]:
        """Find mathematical equations in text"""
        equations = []
        for pattern in self.equation_patterns:
            equations.extend(re.findall(pattern, text, re.DOTALL | re.IGNORECASE))
        return equations
    
    def _find_code_blocks(self, text: str) -> List[str]:
        """Find code blocks in text"""
        code_blocks = []
        for pattern in self.code_patterns:
            code_blocks.extend(re.findall(pattern, text, re.DOTALL))
        return code_blocks
    
    def _extract_section_titles(self, text: str) -> List[str]:
        """Extract section titles from text"""
        titles = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in self.section_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    titles.append(line)
                    break
        
        return titles
    
    def create_chunks(self, pages_data: List[Dict[str, Any]], 
                     chunk_size: int = None, overlap: int = None) -> List[Dict[str, Any]]:
        """
        Create intelligent chunks with semantic boundaries and metadata
        
        Args:
            pages_data: List of page data dictionaries
            chunk_size: Target chunk size in tokens (default: 500)
            overlap: Overlap size in tokens (default: 50)
            
        Returns:
            List[Dict]: Intelligent chunks with metadata
        """
        chunk_size = chunk_size or self.default_chunk_size
        overlap = overlap or self.default_overlap
        
        all_chunks = []
        global_chunk_index = 0
        
        for page_data in pages_data:
            page_num = page_data["page_num"]
            text = page_data["text"]
            page_metadata = page_data["metadata"]
            
            # Create chunks for this page
            page_chunks = self._create_page_chunks(
                text, page_num, page_metadata, chunk_size, overlap, global_chunk_index
            )
            
            all_chunks.extend(page_chunks)
            global_chunk_index += len(page_chunks)
        
        # Add inter-chunk relationships
        self._link_chunks(all_chunks)
        
        return all_chunks
    
    def _create_page_chunks(self, text: str, page_num: int, page_metadata: Dict[str, Any],
                           chunk_size: int, overlap: int, start_index: int) -> List[Dict[str, Any]]:
        """Create semantic chunks for a single page"""
        if not text.strip():
            return []
        
        # Split text into semantic units (paragraphs, sections)
        semantic_units = self._split_into_semantic_units(text)
        
        chunks = []
        current_chunk = ""
        current_chunk_units = []
        chunk_index = start_index
        
        for unit in semantic_units:
            # Check if adding this unit would exceed chunk size
            potential_chunk = current_chunk + ("\n" if current_chunk else "") + unit
            
            if len(potential_chunk.split()) <= chunk_size or not current_chunk:
                # Add unit to current chunk
                current_chunk = potential_chunk
                current_chunk_units.append(unit)
            else:
                # Finalize current chunk and start new one
                if current_chunk:
                    chunk_data = self._create_chunk_data(
                        current_chunk, current_chunk_units, page_num, chunk_index, page_metadata
                    )
                    chunks.append(chunk_data)
                    chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + ("\n" if overlap_text else "") + unit
                current_chunk_units = [unit]
        
        # Add final chunk
        if current_chunk:
            chunk_data = self._create_chunk_data(
                current_chunk, current_chunk_units, page_num, chunk_index, page_metadata
            )
            chunks.append(chunk_data)
        
        return chunks
    
    def _split_into_semantic_units(self, text: str) -> List[str]:
        """Split text into semantic units (paragraphs, sections)"""
        # First split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        
        units = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if paragraph is too long and needs further splitting
            if len(paragraph.split()) > 200:  # Split very long paragraphs
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                current_unit = ""
                
                for sentence in sentences:
                    if len((current_unit + " " + sentence).split()) <= 200:
                        current_unit += (" " if current_unit else "") + sentence
                    else:
                        if current_unit:
                            units.append(current_unit)
                        current_unit = sentence
                
                if current_unit:
                    units.append(current_unit)
            else:
                units.append(paragraph)
        
        return units
    
    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get the last N tokens for overlap"""
        words = text.split()
        if len(words) <= overlap_tokens:
            return text
        return " ".join(words[-overlap_tokens:])
    
    def _create_chunk_data(self, chunk_text: str, units: List[str], page_num: int,
                          chunk_index: int, page_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive chunk data structure"""
        
        # Determine chunk type
        chunk_type = self._classify_chunk_type(chunk_text)
        
        # Calculate importance score
        importance_score = self._calculate_importance_score(chunk_text)
        
        # Detect section title
        section_title = self._detect_section_title(units)
        
        # Generate unique chunk ID
        chunk_id = f"page_{page_num}_chunk_{chunk_index}"
        
        chunk_data = {
            "chunk_id": chunk_id,
            "text": chunk_text,
            "metadata": {
                "page_num": page_num,
                "chunk_index": chunk_index,
                "section": section_title,
                "chunk_type": chunk_type,
                "importance_score": importance_score,
                "word_count": len(chunk_text.split()),
                "char_count": len(chunk_text),
                "has_equations": bool(self._find_equations(chunk_text)),
                "has_code": bool(self._find_code_blocks(chunk_text)),
                "semantic_units": len(units),
                "prev_chunk": None,  # Will be set by _link_chunks
                "next_chunk": None   # Will be set by _link_chunks
            }
        }
        
        return chunk_data
    
    def _classify_chunk_type(self, text: str) -> str:
        """Classify the type of content in the chunk"""
        text_lower = text.lower()
        
        # Check for equations
        if self._find_equations(text):
            return "equation"
        
        # Check for code
        if self._find_code_blocks(text):
            return "code"
        
        # Check for definitions
        if any(keyword in text_lower for keyword in ['definition', 'defined as', 'is defined', 'refers to']):
            return "definition"
        
        # Check for examples
        if any(keyword in text_lower for keyword in ['example', 'for instance', 'for example', 'consider']):
            return "example"
        
        # Check for procedures/methods
        if any(keyword in text_lower for keyword in ['step', 'procedure', 'method', 'algorithm', 'process']):
            return "procedure"
        
        return "normal"
    
    def _calculate_importance_score(self, text: str) -> float:
        """Calculate importance score based on keyword density and content features"""
        text_lower = text.lower()
        score = 0.5  # Base score
        
        # Count high-importance keywords
        high_count = sum(1 for keyword in self.importance_keywords['high'] if keyword in text_lower)
        medium_count = sum(1 for keyword in self.importance_keywords['medium'] if keyword in text_lower)
        low_count = sum(1 for keyword in self.importance_keywords['low'] if keyword in text_lower)
        
        # Adjust score based on keyword presence
        score += high_count * 0.2
        score += medium_count * 0.1
        score -= low_count * 0.05
        
        # Boost for equations and technical content
        if self._find_equations(text):
            score += 0.2
        
        if self._find_code_blocks(text):
            score += 0.15
        
        # Boost for section headers
        if any(re.match(pattern, text, re.IGNORECASE) for pattern in self.section_patterns):
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _detect_section_title(self, units: List[str]) -> Optional[str]:
        """Detect section title from semantic units"""
        for unit in units:
            for pattern in self.section_patterns:
                match = re.match(pattern, unit.strip(), re.IGNORECASE)
                if match:
                    return unit.strip()
        
        return None
    
    def _link_chunks(self, chunks: List[Dict[str, Any]]):
        """Add previous/next chunk relationships"""
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk["metadata"]["prev_chunk"] = chunks[i-1]["chunk_id"]
            if i < len(chunks) - 1:
                chunk["metadata"]["next_chunk"] = chunks[i+1]["chunk_id"]
    
    def _get_processing_warnings(self, pages_data: List[Dict[str, Any]]) -> List[str]:
        """Generate processing warnings based on extracted data"""
        warnings = []
        
        # Check for pages with no text
        empty_pages = [p["page_num"] for p in pages_data if not p["text"].strip()]
        if empty_pages:
            warnings.append(f"Pages with no extractable text: {empty_pages}")
        
        # Check for very short pages
        short_pages = [p["page_num"] for p in pages_data if len(p["text"].split()) < 10]
        if short_pages:
            warnings.append(f"Pages with very little text: {short_pages}")
        
        return warnings
    
    async def _fallback_processing(self, file: UploadFile, content: bytes) -> PDFContent:
        """Fallback to basic processing when advanced processing fails"""
        logger.warning("Falling back to basic PDF processing")
        
        try:
            # Basic text extraction
            text = self._basic_text_extraction(content)
            
            # Simple chunking
            simple_chunks = self._simple_chunking(text)
            
            metadata = {
                "filename": file.filename,
                "file_size": len(content),
                "num_chunks": len(simple_chunks),
                "extraction_method": "fallback_basic",
                "processing_warnings": ["Used fallback processing due to errors"]
            }
            
            return PDFContent(
                filename=file.filename,
                content=text,
                chunks=simple_chunks,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Fallback processing also failed: {e}")
            raise ValueError("Failed to process PDF with all available methods")
    
    def _basic_text_extraction(self, content: bytes) -> str:
        """Basic text extraction as fallback"""
        try:
            pdf_stream = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            
            text = ""
            for page in pdf_reader.pages:
                try:
                    text += page.extract_text() + "\n"
                except:
                    continue
            
            return text.strip()
            
        except Exception as e:
            raise ValueError(f"Basic text extraction failed: {str(e)}")
    
    def _simple_chunking(self, text: str) -> List[str]:
        """Simple word-based chunking as fallback"""
        if not text:
            return []
        
        words = text.split()
        chunk_size = 400  # Conservative chunk size
        overlap = 40
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def save_upload(self, file_content: bytes, filename: str) -> str:
        """
        Save uploaded file to disk with unique naming
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            str: Path to saved file
        """
        os.makedirs(settings.upload_path, exist_ok=True)
        
        # Create unique filename to avoid conflicts
        file_hash = hashlib.md5(file_content).hexdigest()[:8]
        name, ext = os.path.splitext(filename)
        unique_filename = f"{file_hash}_{filename}"
        
        file_path = settings.upload_path / unique_filename
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        logger.info(f"Saved PDF to: {file_path}")
        return str(file_path)

    # =============================================================================
    # STEM-Specific Content Analysis Methods
    # =============================================================================
    
    def detect_equations(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect mathematical equations in text with position mapping
        
        Args:
            text: Input text to analyze
            
        Returns:
            List[Dict]: Equations with positions and types
        """
        equations = []
        
        for pattern_name, pattern in [
            ('latex_inline', r'\$([^$]+)\$'),
            ('latex_display', r'\$\$([^$]+)\$\$'),
            ('latex_equation', r'\\begin\{equation\}(.*?)\\end\{equation\}'),
            ('latex_align', r'\\begin\{align\}(.*?)\\end\{align\}'),
            ('fraction', r'\\frac\{([^}]+)\}\{([^}]+)\}'),
            ('summation', r'\\sum_\{([^}]*)\}\^\{([^}]*)\}'),
            ('integral', r'\\int_\{([^}]*)\}\^\{([^}]*)\}'),
            ('basic_equation', r'\b(\w+\s*[=≠<>≤≥±]+\s*\w+)'),
            ('symbols', r'([α-ωΑ-Ω∑∫∂∇∞±≤≥≠≈∈∉⊂⊃∪∩∧∨¬→↔∀∃]+)')
        ]:
            for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
                equations.append({
                    'type': pattern_name,
                    'content': match.group(1) if match.groups() else match.group(0),
                    'full_match': match.group(0),
                    'start_pos': match.start(),
                    'end_pos': match.end(),
                    'preserved_formatting': True
                })
        
        # Sort by position
        equations.sort(key=lambda x: x['start_pos'])
        
        return equations
    
    def extract_equations_from_pages(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract equations from all pages with page context"""
        all_equations = []
        
        for page_num, page_data in enumerate(pages_data, 1):
            page_equations = self.detect_equations(page_data["text"])
            for eq in page_equations:
                eq.update({
                    'page_number': page_num,
                    'id': f"eq_{page_num}_{len(all_equations)}"
                })
                all_equations.append(eq)
        
        logger.info(f"Found {len(all_equations)} equations across {len(pages_data)} pages")
        return all_equations
    
    def extract_tables_from_pdf(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Extract tables using pdfplumber for better table detection
        
        Args:
            filepath: Path to PDF file
            
        Returns:
            List[Dict]: Tables with structured data
        """
        tables = []
        
        try:
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_tables = page.extract_tables()
                    
                    for table_idx, table_data in enumerate(page_tables):
                        if table_data and len(table_data) > 1:  # Valid table
                            # Convert to markdown format
                            markdown_table = self._table_to_markdown(table_data)
                            
                            tables.append({
                                'id': f"table_{page_num}_{table_idx}",
                                'page_number': page_num,
                                'raw_data': table_data,
                                'markdown': markdown_table,
                                'rows': len(table_data),
                                'columns': len(table_data[0]) if table_data else 0,
                                'chunk_type': 'table'
                            })
                            
        except Exception as e:
            logger.warning(f"Could not extract tables with pdfplumber: {e}")
            # Fallback to text-based table detection
            tables = self._extract_tables_from_text(filepath)
        
        logger.info(f"Found {len(tables)} tables")
        return tables
    
    def extract_figures_from_pages(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract figure references and descriptions from pages"""
        figures = []
        
        for page_num, page_data in enumerate(pages_data, 1):
            text = page_data["text"]
            
            for pattern in self.figure_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    figures.append({
                        'id': f"figure_{page_num}_{len(figures)}",
                        'page_number': page_num,
                        'description': match.group(0).strip(),
                        'start_pos': match.start(),
                        'end_pos': match.end(),
                        'type': 'figure_reference'
                    })
        
        logger.info(f"Found {len(figures)} figure references")
        return figures
    
    def extract_code_blocks_from_pages(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract code blocks while preserving formatting"""
        code_blocks = []
        
        for page_num, page_data in enumerate(pages_data, 1):
            text = page_data["text"]
            
            # Detect various code patterns
            for pattern_name, pattern in [
                ('markdown_code', r'```(\w+)?\n(.*?)\n```'),
                ('inline_code', r'`([^`]+)`'),
                ('function_def', r'(def\s+\w+\s*\([^)]*\):)'),
                ('indented_code', r'^(\s{4,}.+)$')
            ]:
                for match in re.finditer(pattern, text, re.DOTALL | re.MULTILINE):
                    # Detect programming language if possible
                    language = self._detect_programming_language(match.group(0))
                    
                    code_blocks.append({
                        'id': f"code_{page_num}_{len(code_blocks)}",
                        'page_number': page_num,
                        'type': pattern_name,
                        'content': match.group(0).strip(),
                        'language': language,
                        'start_pos': match.start(),
                        'end_pos': match.end(),
                        'preserve_formatting': True
                    })
        
        logger.info(f"Found {len(code_blocks)} code blocks")
        return code_blocks
    
    def create_chunks_with_stem_context(self, pages_data: List[Dict], equations_data: List[Dict], 
                                      tables_data: List[Dict], figures_data: List[Dict], 
                                      code_blocks: List[Dict]) -> List[Dict[str, Any]]:
        """
        Create intelligent chunks with STEM context preservation
        
        Args:
            pages_data: Page text data
            equations_data: Detected equations
            tables_data: Extracted tables
            figures_data: Figure references
            code_blocks: Code blocks
            
        Returns:
            List[Dict]: Enhanced chunks with STEM context
        """
        chunks = []
        
        for page_num, page_data in enumerate(pages_data, 1):
            text = page_data["text"]
            
            # Find STEM elements on this page
            page_equations = [eq for eq in equations_data if eq['page_number'] == page_num]
            page_tables = [tb for tb in tables_data if tb['page_number'] == page_num]
            page_figures = [fg for fg in figures_data if fg['page_number'] == page_num]
            page_code = [cb for cb in code_blocks if cb['page_number'] == page_num]
            
            # Create chunks considering STEM boundaries
            page_chunks = self._create_stem_aware_chunks(
                text, page_num, page_equations, page_tables, page_figures, page_code
            )
            
            chunks.extend(page_chunks)
        
        logger.info(f"Created {len(chunks)} STEM-aware chunks")
        return chunks
    
    def _create_stem_aware_chunks(self, text: str, page_num: int, equations: List[Dict], 
                                tables: List[Dict], figures: List[Dict], 
                                code_blocks: List[Dict]) -> List[Dict[str, Any]]:
        """Create chunks that respect STEM content boundaries"""
        
        # Collect all STEM boundaries
        boundaries = []
        for eq in equations:
            boundaries.append((eq['start_pos'], eq['end_pos'], 'equation', eq))
        for tb in tables:
            boundaries.append((0, len(text), 'table', tb))  # Tables span full context
        for fg in figures:
            boundaries.append((fg['start_pos'], fg['end_pos'], 'figure', fg))
        for cb in code_blocks:
            boundaries.append((cb['start_pos'], cb['end_pos'], 'code', cb))
        
        # Sort boundaries by position
        boundaries.sort(key=lambda x: x[0])
        
        # Create chunks respecting boundaries
        chunks = []
        current_pos = 0
        chunk_id = 0
        
        while current_pos < len(text):
            # Find next boundary or end of text
            next_boundary = None
            for start, end, elem_type, elem_data in boundaries:
                if start >= current_pos:
                    next_boundary = (start, end, elem_type, elem_data)
                    break
            
            if next_boundary:
                start, end, elem_type, elem_data = next_boundary
                
                # Create chunk before STEM element
                if start > current_pos:
                    chunk_text = text[current_pos:start].strip()
                    if chunk_text:
                        chunks.append(self._create_chunk_with_context(
                            chunk_text, page_num, chunk_id, 'text'
                        ))
                        chunk_id += 1
                
                # Create chunk for STEM element
                stem_text = text[start:end].strip()
                if stem_text:
                    chunk = self._create_chunk_with_context(
                        stem_text, page_num, chunk_id, elem_type
                    )
                    chunk['stem_element'] = elem_data
                    chunks.append(chunk)
                    chunk_id += 1
                
                current_pos = end
            else:
                # No more boundaries, create final chunk
                remaining_text = text[current_pos:].strip()
                if remaining_text:
                    chunks.append(self._create_chunk_with_context(
                        remaining_text, page_num, chunk_id, 'text'
                    ))
                break
        
        return chunks
    
    def _create_chunk_with_context(self, text: str, page_num: int, chunk_id: int, 
                                 content_type: str) -> Dict[str, Any]:
        """Create a chunk with enhanced context information"""
        return {
            'id': f"chunk_{page_num}_{chunk_id}",
            'text': text,
            'page_number': page_num,
            'chunk_index': chunk_id,
            'content_type': content_type,
            'char_count': len(text),
            'word_count': len(text.split()),
            'has_equations': bool(re.search(r'[α-ωΑ-Ω∑∫∂∇∞±≤≥≠≈∈∉⊂⊃∪∩∧∨¬→↔∀∃]|\$[^$]+\$', text)),
            'has_code': bool(re.search(r'```|def\s+\w+|function\s+\w+|class\s+\w+', text)),
            'complexity_indicators': {
                'equations': len(re.findall(r'\$[^$]+\$', text)),
                'symbols': len(re.findall(r'[α-ωΑ-Ω∑∫∂∇∞±≤≥≠≈∈∉⊂⊃∪∩∧∨¬→↔∀∃]', text)),
                'code_patterns': len(re.findall(r'def\s+\w+|function\s+\w+|class\s+\w+', text))
            }
        }
    
    def _calculate_complexity_score(self, equations: List[Dict], tables: List[Dict], 
                                  code_blocks: List[Dict], pages_data: List[Dict]) -> float:
        """Calculate document complexity score based on STEM content"""
        total_text = sum(len(page['text']) for page in pages_data)
        
        if total_text == 0:
            return 0.0
        
        # Weight different types of content
        equation_score = len(equations) * 0.3
        table_score = len(tables) * 0.2
        code_score = len(code_blocks) * 0.25
        
        # Calculate symbol density
        symbol_count = 0
        for page in pages_data:
            symbol_count += len(re.findall(r'[α-ωΑ-Ω∑∫∂∇∞±≤≥≠≈∈∉⊂⊃∪∩∧∨¬→↔∀∃]', page['text']))
        
        symbol_density = symbol_count / total_text * 1000  # per 1000 chars
        
        # Combine scores
        complexity = (equation_score + table_score + code_score + symbol_density) / 10
        
        # Normalize to 0-1 range
        return min(1.0, complexity)
    
    def _table_to_markdown(self, table_data: List[List[str]]) -> str:
        """Convert table data to markdown format"""
        if not table_data or not table_data[0]:
            return ""
        
        markdown = []
        
        # Header row
        header = " | ".join(str(cell or "") for cell in table_data[0])
        markdown.append(f"| {header} |")
        
        # Separator row
        separator = " | ".join("---" for _ in table_data[0])
        markdown.append(f"| {separator} |")
        
        # Data rows
        for row in table_data[1:]:
            row_text = " | ".join(str(cell or "") for cell in row)
            markdown.append(f"| {row_text} |")
        
        return "\n".join(markdown)
    
    def _extract_tables_from_text(self, filepath: str) -> List[Dict[str, Any]]:
        """Fallback method for table extraction using text patterns"""
        tables = []
        
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    
                    # Look for table patterns in text
                    for pattern in self.table_patterns:
                        matches = re.finditer(pattern, text, re.MULTILINE)
                        for match_idx, match in enumerate(matches):
                            tables.append({
                                'id': f"text_table_{page_num}_{match_idx}",
                                'page_number': page_num,
                                'content': match.group(0),
                                'type': 'text_pattern_table',
                                'chunk_type': 'table'
                            })
                            
        except Exception as e:
            logger.warning(f"Text-based table extraction failed: {e}")
        
        return tables
    
    def _detect_programming_language(self, code_text: str) -> Optional[str]:
        """Detect programming language from code patterns"""
        code_lower = code_text.lower()
        
        # Language detection patterns
        language_patterns = {
            'python': [r'def\s+\w+', r'import\s+\w+', r'from\s+\w+\s+import', r'print\s*\('],
            'javascript': [r'function\s+\w+', r'var\s+\w+', r'let\s+\w+', r'console\.log'],
            'java': [r'public\s+class', r'public\s+static\s+void\s+main', r'System\.out\.print'],
            'c++': [r'#include\s*<[^>]+>', r'int\s+main\s*\(', r'std::'],
            'c': [r'#include\s*<[^>]+>', r'int\s+main\s*\(', r'printf'],
            'sql': [r'SELECT\s+', r'FROM\s+', r'WHERE\s+', r'INSERT\s+INTO'],
            'html': [r'<[^>]+>', r'<!DOCTYPE', r'<html>', r'<body>'],
            'css': [r'[^{]+\s*\{[^}]*\}', r'@media', r'font-family:'],
        }
        
        for language, patterns in language_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code_lower):
                    return language
        
        return 'unknown'

    # =============================================================================
    # End of STEM-Specific Methods
    # =============================================================================
    
    def save_upload(self, file_content: bytes, filename: str) -> str:
        """
        Save uploaded file content to temporary location
        
        Args:
            file_content: Binary file content
            filename: Original filename
            
        Returns:
            str: Path to saved file
        """
        # Ensure upload directory exists
        upload_dir = settings.upload_path
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unique filename to avoid conflicts
        file_hash = hashlib.md5(file_content[:1000]).hexdigest()[:8]
        safe_filename = f"{file_hash}_{filename}"
        file_path = upload_dir / safe_filename
        
        # Write file
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        logger.info(f"Saved PDF to: {file_path}")
        return str(file_path)
    
    def process_pdf(self, filepath: str) -> Dict[str, Any]:
        """
        Process PDF from file path (for direct file processing)
        
        Args:
            filepath: Path to PDF file
            
        Returns:
            Dict: Processing results with chunks and metadata
        """
        try:
            # Extract text from PDF
            pages_data = self.extract_text_from_pdf(filepath)
            
            # Create intelligent chunks
            chunks_data = self.create_chunks(pages_data)
            
            # Create metadata
            metadata = {
                "filepath": filepath,
                "num_pages": len(pages_data),
                "num_chunks": len(chunks_data),
                "extraction_method": "intelligent_chunking",
                "processing_warnings": self._get_processing_warnings(pages_data)
            }
            
            return {
                "chunks": chunks_data,
                "metadata": metadata,
                "pages_data": pages_data
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF {filepath}: {e}")
            raise
