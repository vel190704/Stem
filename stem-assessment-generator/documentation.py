"""
Help and Documentation Module
Provides comprehensive documentation and sample API calls
"""
from typing import Dict, List, Any
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import json

class DocumentationGenerator:
    """Generates comprehensive documentation"""
    
    def __init__(self, app: FastAPI):
        self.app = app
    
    def get_api_documentation(self) -> Dict[str, Any]:
        """Get comprehensive API documentation"""
        return {
            "title": "STEM Assessment Generator API",
            "version": "2.0.0",
            "description": "AI-powered assessment question generator with intelligent PDF processing",
            "base_url": "http://localhost:8000",
            "endpoints": self._get_endpoint_documentation(),
            "authentication": {
                "type": "Environment Variable",
                "description": "Set OPENAI_API_KEY in your environment",
                "example": "export OPENAI_API_KEY=sk-your-api-key-here"
            },
            "examples": self._get_curl_examples(),
            "error_codes": self._get_error_codes(),
            "best_practices": self._get_best_practices()
        }
    
    def _get_endpoint_documentation(self) -> Dict[str, Any]:
        """Document all API endpoints"""
        return {
            "upload": {
                "method": "POST",
                "path": "/api/upload",
                "description": "Upload a PDF file for processing",
                "content_type": "multipart/form-data",
                "parameters": {
                    "file": {
                        "type": "file",
                        "required": True,
                        "description": "PDF file to process",
                        "constraints": ["PDF format", "Max 10MB", "Text-based content"]
                    }
                },
                "response": {
                    "task_id": "Unique identifier for tracking",
                    "filename": "Original filename",
                    "status": "Processing status",
                    "message": "Human-readable status message"
                },
                "example_response": {
                    "task_id": "abc123-def456",
                    "filename": "blockchain_basics.pdf",
                    "status": "processing",
                    "message": "File uploaded successfully, processing started"
                }
            },
            "process_status": {
                "method": "GET",
                "path": "/api/process_status/{task_id}",
                "description": "Check processing status of uploaded file",
                "parameters": {
                    "task_id": {
                        "type": "string",
                        "required": True,
                        "description": "Task ID from upload response"
                    }
                },
                "response": {
                    "status": "Current status (processing/ready/error)",
                    "progress": "Detailed progress information",
                    "message": "Current operation description"
                }
            },
            "generate": {
                "method": "POST",
                "path": "/api/generate",
                "description": "Generate assessment questions from processed PDF",
                "content_type": "application/json",
                "parameters": {
                    "task_id": {
                        "type": "string",
                        "required": True,
                        "description": "Task ID from successful upload"
                    },
                    "num_questions": {
                        "type": "integer",
                        "required": True,
                        "description": "Number of questions to generate (1-20)",
                        "default": 10
                    },
                    "difficulty_level": {
                        "type": "string",
                        "required": True,
                        "description": "Question difficulty",
                        "options": ["easy", "medium", "hard"],
                        "default": "medium"
                    }
                },
                "response": {
                    "questions": "Array of generated questions",
                    "statistics": "Generation statistics",
                    "metadata": "Additional information"
                }
            },
            "export": {
                "method": "POST",
                "path": "/api/export/{task_id}",
                "description": "Export generated questions in various formats",
                "parameters": {
                    "format_type": {
                        "type": "string",
                        "options": ["json", "pdf-teacher", "pdf-student", "docx", "txt"],
                        "default": "json"
                    },
                    "teacher_version": {
                        "type": "boolean",
                        "description": "Include answers and explanations",
                        "default": True
                    }
                }
            },
            "analytics": {
                "method": "GET",
                "path": "/api/analytics/{task_id}",
                "description": "Get quality analytics for generated questions",
                "response": {
                    "summary": "Overall quality metrics",
                    "per_question": "Individual question analysis",
                    "recommendations": "Improvement suggestions"
                }
            },
            "health": {
                "method": "GET",
                "path": "/api/health",
                "description": "Check system health and configuration",
                "response": {
                    "status": "System status (healthy/degraded/unhealthy)",
                    "openai_status": "OpenAI API connectivity",
                    "vectordb_status": "Vector database status",
                    "system_stats": "System statistics"
                }
            }
        }
    
    def _get_curl_examples(self) -> Dict[str, str]:
        """Get comprehensive curl examples"""
        return {
            "upload_file": '''curl -X POST "http://localhost:8000/api/upload" \\
  -H "accept: application/json" \\
  -H "Content-Type: multipart/form-data" \\
  -F "file=@blockchain_basics.pdf"''',
            
            "check_status": '''curl -X GET "http://localhost:8000/api/process_status/abc123-def456" \\
  -H "accept: application/json"''',
            
            "generate_questions": '''curl -X POST "http://localhost:8000/api/generate" \\
  -H "accept: application/json" \\
  -H "Content-Type: application/json" \\
  -d '{
    "task_id": "abc123-def456",
    "num_questions": 10,
    "difficulty_level": "medium"
  }'\\''',
            
            "export_pdf": '''curl -X POST "http://localhost:8000/api/export/abc123-def456" \\
  -H "accept: application/json" \\
  -H "Content-Type: application/json" \\
  -d '{
    "format_type": "pdf-teacher",
    "teacher_version": true,
    "include_explanations": true
  }' \\
  --output assessment.pdf''',
            
            "get_analytics": '''curl -X GET "http://localhost:8000/api/analytics/abc123-def456" \\
  -H "accept: application/json"''',
            
            "health_check": '''curl -X GET "http://localhost:8000/api/health" \\
  -H "accept: application/json"'''
        }
    
    def _get_error_codes(self) -> Dict[str, Dict[str, str]]:
        """Document error codes and their meanings"""
        return {
            "PDF_PROCESSING_ERROR": {
                "description": "General PDF processing failure",
                "common_causes": ["Corrupted file", "Unsupported PDF version", "No text content"],
                "solutions": ["Try a different PDF", "Ensure file has readable text", "Check file integrity"]
            },
            "PDF_TOO_LARGE": {
                "description": "PDF file exceeds size limit",
                "common_causes": ["File larger than 10MB"],
                "solutions": ["Compress PDF", "Split into smaller files", "Use a shorter document"]
            },
            "PDF_PASSWORD_PROTECTED": {
                "description": "PDF requires password to open",
                "common_causes": ["Document security settings"],
                "solutions": ["Remove password protection", "Use unprotected version"]
            },
            "OPENAI_KEY_ERROR": {
                "description": "OpenAI API key not configured or invalid",
                "common_causes": ["Missing environment variable", "Invalid API key"],
                "solutions": ["Set OPENAI_API_KEY environment variable", "Verify API key is correct"]
            },
            "OPENAI_QUOTA_EXCEEDED": {
                "description": "OpenAI API usage limit reached",
                "common_causes": ["Monthly quota exceeded", "Rate limits"],
                "solutions": ["Wait for quota reset", "Upgrade OpenAI plan", "Try again later"]
            },
            "INSUFFICIENT_CONTENT": {
                "description": "Not enough content for requested questions",
                "common_causes": ["Short document", "Requesting too many questions"],
                "solutions": ["Use longer document", "Request fewer questions", "Add more detailed content"]
            },
            "QUALITY_VALIDATION_FAILED": {
                "description": "Generated questions don't meet quality standards",
                "common_causes": ["Poor source content", "Technical complexity"],
                "solutions": ["Try different content", "Regenerate questions", "Adjust difficulty level"]
            }
        }
    
    def _get_best_practices(self) -> List[str]:
        """Get best practices for using the API"""
        return [
            "Upload high-quality PDF files with clear, readable text",
            "Use educational or technical documents for best results",
            "Start with fewer questions (5-10) for testing",
            "Monitor processing status before generating questions",
            "Check analytics to improve question quality",
            "Use appropriate difficulty levels for your audience",
            "Export in multiple formats for different use cases",
            "Keep API keys secure and never commit them to code",
            "Handle errors gracefully with proper retry logic",
            "Cache results when possible to reduce API calls",
            "Use teacher/student versions appropriately",
            "Monitor system health before heavy usage"
        ]
    
    def get_user_guide_html(self) -> str:
        """Generate comprehensive HTML user guide"""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>STEM Assessment Generator - User Guide</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .code-block {{
            background: #1f2937;
            color: #f9fafb;
            padding: 1rem;
            border-radius: 0.5rem;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
        }}
        .step {{
            border-left: 4px solid #3b82f6;
            padding-left: 1rem;
            margin: 1rem 0;
        }}
    </style>
</head>
<body class="bg-gray-50">
    <div class="container mx-auto px-4 py-8 max-w-4xl">
        <h1 class="text-4xl font-bold text-gray-800 mb-8">📚 STEM Assessment Generator</h1>
        <p class="text-xl text-gray-600 mb-8">Complete guide to generating high-quality assessment questions from PDF documents</p>
        
        <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
            <h2 class="text-2xl font-semibold mb-4">🚀 Quick Start</h2>
            <div class="space-y-4">
                <div class="step">
                    <h3 class="font-semibold text-lg">1. Upload Your PDF</h3>
                    <p class="text-gray-600">Select a PDF file containing educational content. Best results with 3+ pages of text.</p>
                </div>
                <div class="step">
                    <h3 class="font-semibold text-lg">2. Wait for Processing</h3>
                    <p class="text-gray-600">The system will extract text, analyze content, and prepare for question generation.</p>
                </div>
                <div class="step">
                    <h3 class="font-semibold text-lg">3. Configure Questions</h3>
                    <p class="text-gray-600">Choose number of questions (1-20) and difficulty level (Easy/Medium/Hard).</p>
                </div>
                <div class="step">
                    <h3 class="font-semibold text-lg">4. Generate & Export</h3>
                    <p class="text-gray-600">Generate questions and export in your preferred format (PDF, Word, JSON, etc.).</p>
                </div>
            </div>
        </div>
        
        <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
            <h2 class="text-2xl font-semibold mb-4">📋 File Requirements</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                    <h3 class="font-semibold text-lg mb-2">✅ Supported</h3>
                    <ul class="list-disc list-inside text-gray-600 space-y-1">
                        <li>PDF files with readable text</li>
                        <li>Academic papers and textbooks</li>
                        <li>Technical documentation</li>
                        <li>Educational materials</li>
                        <li>Files up to 10MB</li>
                    </ul>
                </div>
                <div>
                    <h3 class="font-semibold text-lg mb-2">❌ Not Supported</h3>
                    <ul class="list-disc list-inside text-gray-600 space-y-1">
                        <li>Password-protected PDFs</li>
                        <li>Image-only PDFs (scanned without OCR)</li>
                        <li>Files larger than 10MB</li>
                        <li>Non-educational content</li>
                        <li>Corrupted or damaged files</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
            <h2 class="text-2xl font-semibold mb-4">⚙️ Question Settings</h2>
            <div class="space-y-4">
                <div>
                    <h3 class="font-semibold">Number of Questions</h3>
                    <p class="text-gray-600">Choose 1-20 questions. More questions require longer documents with diverse content.</p>
                </div>
                <div>
                    <h3 class="font-semibold">Difficulty Levels</h3>
                    <ul class="list-disc list-inside text-gray-600 space-y-1 ml-4">
                        <li><strong>Easy:</strong> Basic definitions and simple concepts</li>
                        <li><strong>Medium:</strong> Conceptual understanding and relationships</li>
                        <li><strong>Hard:</strong> Advanced analysis and application</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
            <h2 class="text-2xl font-semibold mb-4">📤 Export Options</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                    <h3 class="font-semibold">Formats Available</h3>
                    <ul class="list-disc list-inside text-gray-600 space-y-1">
                        <li><strong>PDF:</strong> Print-ready teacher/student versions</li>
                        <li><strong>Word:</strong> Editable document format</li>
                        <li><strong>JSON:</strong> LMS integration (Moodle, Canvas)</li>
                        <li><strong>Text:</strong> Simple copy-paste format</li>
                    </ul>
                </div>
                <div>
                    <h3 class="font-semibold">Version Options</h3>
                    <ul class="list-disc list-inside text-gray-600 space-y-1">
                        <li><strong>Teacher:</strong> Includes answers & explanations</li>
                        <li><strong>Student:</strong> Questions only</li>
                        <li><strong>Custom:</strong> Choose what to include</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
            <h2 class="text-2xl font-semibold mb-4">📊 Quality Analytics</h2>
            <p class="text-gray-600 mb-4">Our system analyzes question quality across multiple dimensions:</p>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="bg-blue-50 p-4 rounded">
                    <h4 class="font-semibold text-blue-800">Clarity Score</h4>
                    <p class="text-sm text-blue-600">How clear and unambiguous the question is</p>
                </div>
                <div class="bg-green-50 p-4 rounded">
                    <h4 class="font-semibold text-green-800">Distractor Quality</h4>
                    <p class="text-sm text-green-600">How plausible the wrong answers are</p>
                </div>
                <div class="bg-purple-50 p-4 rounded">
                    <h4 class="font-semibold text-purple-800">Difficulty Match</h4>
                    <p class="text-sm text-purple-600">How well difficulty matches the setting</p>
                </div>
            </div>
        </div>
        
        <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
            <h2 class="text-2xl font-semibold mb-4">🔧 Troubleshooting</h2>
            <div class="space-y-4">
                <div>
                    <h3 class="font-semibold text-red-600">File Upload Issues</h3>
                    <ul class="list-disc list-inside text-gray-600 space-y-1 ml-4">
                        <li>Ensure PDF is not password-protected</li>
                        <li>Check file size is under 10MB</li>
                        <li>Verify PDF contains readable text</li>
                    </ul>
                </div>
                <div>
                    <h3 class="font-semibold text-red-600">Processing Failures</h3>
                    <ul class="list-disc list-inside text-gray-600 space-y-1 ml-4">
                        <li>Try a different PDF with more text content</li>
                        <li>Reduce number of requested questions</li>
                        <li>Check internet connection</li>
                    </ul>
                </div>
                <div>
                    <h3 class="font-semibold text-red-600">Poor Question Quality</h3>
                    <ul class="list-disc list-inside text-gray-600 space-y-1 ml-4">
                        <li>Use more detailed source material</li>
                        <li>Try different difficulty settings</li>
                        <li>Regenerate specific questions</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="bg-blue-50 rounded-lg p-6">
            <h2 class="text-2xl font-semibold mb-4">💡 Tips for Best Results</h2>
            <ul class="list-disc list-inside text-gray-700 space-y-2">
                <li>Use academic papers, textbooks, or technical documentation</li>
                <li>Ensure content includes examples, diagrams, and explanations</li>
                <li>Start with 5-10 questions to test content quality</li>
                <li>Review analytics to improve future uploads</li>
                <li>Export in multiple formats for different uses</li>
                <li>Keep source PDFs organized for reuse</li>
            </ul>
        </div>
    </div>
</body>
</html>"""
    
    def get_faq(self) -> List[Dict[str, str]]:
        """Get frequently asked questions"""
        return [
            {
                "question": "What types of PDFs work best?",
                "answer": "Academic papers, textbooks, technical documentation, and educational materials with clear text content work best. Documents should be 3+ pages with detailed explanations and examples."
            },
            {
                "question": "How many questions can I generate?",
                "answer": "You can generate 1-20 questions per document. The number depends on content length and complexity. Longer documents can support more questions."
            },
            {
                "question": "What's the difference between difficulty levels?",
                "answer": "Easy questions test basic definitions, Medium questions test conceptual understanding, and Hard questions require analysis and application of concepts."
            },
            {
                "question": "Can I edit the generated questions?",
                "answer": "Yes! Export to Word format for easy editing, or use the regenerate feature to create alternative versions of specific questions."
            },
            {
                "question": "How do I integrate with my LMS?",
                "answer": "Export in JSON format for most LMS platforms, or use specific formats like Moodle XML for direct import into course management systems."
            },
            {
                "question": "Why did question generation fail?",
                "answer": "Common causes include insufficient content, password-protected PDFs, image-only documents, or very technical content that's difficult to process."
            },
            {
                "question": "How can I improve question quality?",
                "answer": "Use detailed source material with examples, check analytics for quality scores, regenerate low-quality questions, and ensure content matches your chosen difficulty level."
            },
            {
                "question": "Is my uploaded content stored permanently?",
                "answer": "No, uploaded files and generated questions are stored temporarily for your session. Export your questions to save them permanently."
            }
        ]

# Global documentation generator instance (will be initialized in app.py)
documentation_generator = None
