#!/usr/bin/env python3
"""
Manual Testing Guide for STEM Assessment Generator
Provides structured manual testing procedures and utilities
"""
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import settings
from rag_pipeline import RAGPipeline

def print_manual_testing_guide():
    """Print comprehensive manual testing guide"""
    print("="*80)
    print("MANUAL TESTING GUIDE - STEM ASSESSMENT GENERATOR")
    print("="*80)
    
    print("\n🎯 TESTING OBJECTIVES:")
    print("   • Verify end-to-end PDF processing workflow")
    print("   • Test web interface functionality")
    print("   • Validate question generation quality")
    print("   • Check system persistence and recovery")
    
    print("\n📋 PRE-TESTING CHECKLIST:")
    print("   □ OpenAI API key configured")
    print("   □ All dependencies installed")
    print("   □ Application starts without errors")
    print("   □ Test PDF files available")
    
    print(f"\n🔧 CURRENT CONFIGURATION:")
    print(f"   • Project Root: {settings.PROJECT_ROOT}")
    print(f"   • Upload Directory: {settings.upload_path}")
    print(f"   • Vector DB Directory: {settings.vectordb_path}")
    print(f"   • OpenAI Configured: {settings.is_openai_configured()}")
    print(f"   • Debug Mode: {settings.DEBUG}")
    
    print("\n" + "="*80)
    print("MANUAL TESTING PROCEDURES")
    print("="*80)
    
    print("\n🔄 TEST 1: APPLICATION STARTUP")
    print("   1. Start the application:")
    print("      cd /home/manivel/project/cleaned-Stem/stem-assessment-generator")
    print("      python app.py")
    print("   ")
    print("   2. Expected output:")
    print("      ✓ Configuration loaded successfully")
    print("      ✓ Directory ensured: data/uploads")
    print("      ✓ Directory ensured: data/vectordb")
    print("      ✓ Application startup on http://0.0.0.0:8000")
    print("   ")
    print("   3. Verify browser access:")
    print("      Open: http://localhost:8000")
    print("      Should see: Modern upload interface with blockchain theme")
    
    print("\n📤 TEST 2: PDF UPLOAD & PROCESSING")
    print("   1. Prepare test PDF:")
    print("      • Use blockchain/cryptocurrency content")
    print("      • 2-5 pages recommended")
    print("      • Mix of text, concepts, and technical details")
    print("   ")
    print("   2. Upload via web interface:")
    print("      • Drag & drop or click to upload")
    print("      • File size should be under 10MB")
    print("      • Watch for upload progress")
    print("   ")
    print("   3. Monitor console logs for:")
    print("      ✓ File upload successful")
    print("      ✓ Text extraction started")
    print("      ✓ Chunking process initiated")
    print("      ✓ Embedding generation started")
    print("      ✓ Vector storage completed")
    print("   ")
    print("   4. Check progress tracking:")
    print("      • Progress bar should advance through 5 steps")
    print("      • Status should update in real-time")
    print("      • Final status should be 'ready'")
    
    print("\n🗃️ TEST 3: DATA PERSISTENCE")
    print("   1. Check file system:")
    print("      ls -la data/uploads/")
    print("      → Should contain uploaded PDF with UUID prefix")
    print("   ")
    print("      ls -la data/vectordb/")
    print("      → Should contain ChromaDB files:")
    print("        • chroma.sqlite3")
    print("        • Collection directories with UUID names")
    print("   ")
    print("   2. Verify vector storage:")
    print("      python -c \"from rag_pipeline import RAGPipeline; rag = RAGPipeline(); print('Collections:', rag.list_collections())\"")
    
    print("\n🎯 TEST 4: QUESTION GENERATION")
    print("   1. Generate questions via web interface:")
    print("      • Select difficulty level (easy/medium/hard)")
    print("      • Choose number of questions (5-20)")
    print("      • Click 'Generate Assessment'")
    print("   ")
    print("   2. Monitor generation process:")
    print("      ✓ Question generation started")
    print("      ✓ Context retrieval from vector database")
    print("      ✓ Batch generation with perspective switching")
    print("      ✓ Quality validation and deduplication")
    print("   ")
    print("   3. Evaluate generated questions:")
    print("      • Questions should be relevant to uploaded content")
    print("      • Multiple choice options should be plausible")
    print("      • Difficulty should match selected level")
    print("      • Should achieve exact requested quantity")
    
    print("\n🔍 TEST 5: VECTOR STORE QUERIES")
    print("   Manual vector store testing commands:")
    print("   (Run these in Python console after uploading a PDF)")
    
    print("\n🚀 TEST 6: ERROR HANDLING & RECOVERY")
    print("   1. Test invalid file uploads:")
    print("      • Try uploading non-PDF files")
    print("      • Upload oversized files")
    print("      • Upload corrupted PDFs")
    print("   ")
    print("   2. Test API endpoints:")
    print("      GET  /api/health")
    print("      GET  /api/collections")
    print("      GET  /api/tasks")
    print("   ")
    print("   3. Test retry functionality:")
    print("      • Simulate processing failures")
    print("      • Use retry endpoint for failed tasks")
    
    print("\n📊 TEST 7: PERFORMANCE & SCALABILITY")
    print("   1. Upload multiple PDFs concurrently")
    print("   2. Generate multiple assessments simultaneously")
    print("   3. Monitor memory usage and response times")
    print("   4. Test cache performance with repeated queries")

def create_test_queries():
    """Create sample queries for manual vector store testing"""
    return [
        "What is blockchain technology?",
        "How does proof of work consensus work?",
        "Explain cryptocurrency mining",
        "What are smart contracts?",
        "Describe blockchain security features",
        "How do hash functions work in blockchain?",
        "What is decentralization?",
        "Compare different consensus mechanisms",
        "Explain Bitcoin and its features",
        "What are the applications of blockchain?"
    ]

async def manual_vector_test(collection_name: str = None):
    """Interactive vector store testing"""
    print("\n" + "="*60)
    print("INTERACTIVE VECTOR STORE TEST")
    print("="*60)
    
    pipeline = RAGPipeline()
    
    # List available collections
    collections = pipeline.list_collections()
    print(f"\nAvailable collections: {collections}")
    
    if not collections:
        print("❌ No collections found. Upload a PDF first.")
        return
    
    # Use specified collection or first available
    if collection_name and collection_name in collections:
        test_collection = collection_name
    else:
        test_collection = collections[0]
    
    print(f"Using collection: {test_collection}")
    
    # Test queries
    test_queries = create_test_queries()
    print(f"\nTesting {len(test_queries)} sample queries...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        
        try:
            # Basic retrieval
            results = await pipeline.retrieve_relevant_chunks(
                query, test_collection, top_k=3
            )
            
            print(f"   Retrieved {len(results)} chunks:")
            for j, (chunk, score) in enumerate(results):
                print(f"     {j+1}. Score: {score:.3f} | Type: {chunk.content_type}")
                print(f"        {chunk.content[:100]}...")
            
            # MMR diversity test
            diverse_results = await pipeline.retrieve_with_mmr(
                query, test_collection, top_k=3, diversity_lambda=0.7
            )
            
            content_types = set(chunk.content_type for chunk, _ in diverse_results)
            print(f"   Diversity: {len(content_types)} different content types")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

def print_api_testing_commands():
    """Print curl commands for API testing"""
    print("\n" + "="*60)
    print("API TESTING COMMANDS")
    print("="*60)
    
    base_url = f"http://localhost:{settings.APP_PORT}"
    
    commands = [
        ("Health Check", f"curl -X GET {base_url}/api/health"),
        ("List Collections", f"curl -X GET {base_url}/api/collections"),
        ("List Tasks", f"curl -X GET {base_url}/api/tasks"),
        ("Clear Cache", f"curl -X POST {base_url}/api/clear_cache"),
        ("Upload File", f"curl -X POST -F 'file=@your_file.pdf' {base_url}/api/upload"),
    ]
    
    for name, command in commands:
        print(f"\n{name}:")
        print(f"  {command}")
    
    print(f"\nGenerate Assessment (replace TASK_ID):")
    print(f"  curl -X POST {base_url}/api/generate \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f"    -d '{{")
    print(f"      \"task_id\": \"YOUR_TASK_ID\",")
    print(f"      \"num_questions\": 5,")
    print(f"      \"difficulty_level\": \"medium\"")
    print(f"    }}'")

def check_environment():
    """Check environment setup for testing"""
    print("\n" + "="*60)
    print("ENVIRONMENT CHECK")
    print("="*60)
    
    checks = [
        ("Python version", sys.version),
        ("Current directory", Path.cwd()),
        ("Config loaded", "✓" if settings else "❌"),
        ("OpenAI configured", "✓" if settings.is_openai_configured() else "❌"),
        ("Upload directory exists", "✓" if settings.upload_path.exists() else "❌"),
        ("Vector DB directory exists", "✓" if settings.vectordb_path.exists() else "❌"),
    ]
    
    for check_name, result in checks:
        print(f"  {check_name}: {result}")
    
    # Check dependencies
    missing_deps = []
    required_modules = [
        'fastapi', 'uvicorn', 'pydantic', 'openai', 
        'chromadb', 'PyPDF2', 'numpy', 'tiktoken'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_deps.append(module)
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {missing_deps}")
        print("   Install with: pip install " + " ".join(missing_deps))
    else:
        print("\n✓ All dependencies available")

if __name__ == "__main__":
    print(f"Manual Testing Guide - Generated at {datetime.now()}")
    
    # Run environment check
    check_environment()
    
    # Print main testing guide
    print_manual_testing_guide()
    
    # Print API testing commands
    print_api_testing_commands()
    
    print("\n" + "="*80)
    print("QUICK START TESTING SEQUENCE")
    print("="*80)
    print("1. python app.py                     # Start the application")
    print("2. Open http://localhost:8000        # Test web interface")
    print("3. Upload a blockchain PDF           # Test upload & processing")
    print("4. Generate 5 medium questions       # Test question generation")
    print("5. python manual_testing_guide.py   # Run vector store tests")
    print("="*80)
    
    # Offer interactive testing
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        print("\nStarting interactive vector store testing...")
        collection = sys.argv[2] if len(sys.argv) > 2 else None
        asyncio.run(manual_vector_test(collection))
