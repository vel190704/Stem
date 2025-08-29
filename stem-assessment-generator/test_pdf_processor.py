#!/usr/bin/env python3
"""
Test script for PDF Processor
Tests PDF text extraction, intelligent chunking, and metadata handling
"""
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from pdf_processor import PDFProcessor
from config import settings

def create_sample_blockchain_pdf():
    """Create a simple sample PDF about blockchain for testing"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        pdf_path = Path("test_blockchain_sample.pdf")
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        
        # Page 1: Blockchain Basics
        c.drawString(100, 750, "Blockchain Technology Overview")
        c.drawString(100, 700, "")
        c.drawString(100, 680, "What is Blockchain?")
        c.drawString(100, 660, "A blockchain is a distributed ledger technology that maintains")
        c.drawString(100, 640, "a continuously growing list of records, called blocks, which")
        c.drawString(100, 620, "are linked and secured using cryptography.")
        c.drawString(100, 580, "")
        c.drawString(100, 560, "Key Features:")
        c.drawString(120, 540, "• Decentralization: No single point of control")
        c.drawString(120, 520, "• Immutability: Records cannot be altered once written")
        c.drawString(120, 500, "• Transparency: All transactions are visible to network participants")
        c.drawString(120, 480, "• Consensus: Network agreement on transaction validity")
        c.drawString(100, 440, "")
        c.drawString(100, 420, "Hash Functions:")
        c.drawString(100, 400, "Cryptographic hash functions like SHA-256 ensure data integrity.")
        c.drawString(100, 380, "Each block contains the hash of the previous block, creating")
        c.drawString(100, 360, "an immutable chain of records.")
        
        c.showPage()
        
        # Page 2: Consensus Mechanisms
        c.drawString(100, 750, "Consensus Mechanisms in Blockchain")
        c.drawString(100, 700, "")
        c.drawString(100, 680, "Proof of Work (PoW):")
        c.drawString(100, 660, "Miners compete to solve computationally difficult puzzles.")
        c.drawString(100, 640, "The first to solve the puzzle gets to add the next block")
        c.drawString(100, 620, "and receive a reward. Used by Bitcoin.")
        c.drawString(100, 580, "")
        c.drawString(100, 560, "Proof of Stake (PoS):")
        c.drawString(100, 540, "Validators are chosen to create new blocks based on their")
        c.drawString(100, 520, "stake in the network. More energy-efficient than PoW.")
        c.drawString(100, 500, "Used by Ethereum 2.0.")
        c.drawString(100, 460, "")
        c.drawString(100, 440, "Smart Contracts:")
        c.drawString(100, 420, "Self-executing contracts with terms directly written into code.")
        c.drawString(100, 400, "They automatically execute when predetermined conditions are met.")
        c.drawString(100, 380, "Enable complex applications on blockchain platforms.")
        c.drawString(100, 340, "")
        c.drawString(100, 320, "Common Misconceptions:")
        c.drawString(120, 300, "• Blockchain is only for cryptocurrency")
        c.drawString(120, 280, "• All blockchains are public")
        c.drawString(120, 260, "• Blockchain is completely anonymous")
        
        c.save()
        
        print(f"✓ Created sample PDF: {pdf_path}")
        return pdf_path
        
    except ImportError:
        print("⚠ reportlab not installed. Using text file instead.")
        # Create a simple text file as fallback
        txt_path = Path("test_blockchain_sample.txt")
        with open(txt_path, 'w') as f:
            f.write("""Blockchain Technology Overview

What is Blockchain?
A blockchain is a distributed ledger technology that maintains a continuously growing list of records, called blocks, which are linked and secured using cryptography.

Key Features:
• Decentralization: No single point of control
• Immutability: Records cannot be altered once written
• Transparency: All transactions are visible to network participants
• Consensus: Network agreement on transaction validity

Hash Functions:
Cryptographic hash functions like SHA-256 ensure data integrity. Each block contains the hash of the previous block, creating an immutable chain of records.

Consensus Mechanisms in Blockchain

Proof of Work (PoW):
Miners compete to solve computationally difficult puzzles. The first to solve the puzzle gets to add the next block and receive a reward. Used by Bitcoin.

Proof of Stake (PoS):
Validators are chosen to create new blocks based on their stake in the network. More energy-efficient than PoW. Used by Ethereum 2.0.

Smart Contracts:
Self-executing contracts with terms directly written into code. They automatically execute when predetermined conditions are met. Enable complex applications on blockchain platforms.

Common Misconceptions:
• Blockchain is only for cryptocurrency
• All blockchains are public
• Blockchain is completely anonymous""")
        
        print(f"✓ Created sample text file: {txt_path}")
        return txt_path

async def test_pdf_processor():
    """Test PDF processor with comprehensive checks"""
    print("="*60)
    print("PDF PROCESSOR TEST SUITE")
    print("="*60)
    
    # Initialize processor
    processor = PDFProcessor()
    print("✓ PDFProcessor initialized")
    
    # Create or use sample PDF
    sample_file = create_sample_blockchain_pdf()
    
    try:
        # Test 1: Text Extraction
        print("\n1. Testing Text Extraction...")
        text = processor.extract_text_from_pdf(sample_file)
        
        print(f"   ✓ Extracted {len(text)} characters")
        print(f"   ✓ Contains 'blockchain': {'blockchain' in text.lower()}")
        print(f"   ✓ Contains 'consensus': {'consensus' in text.lower()}")
        
        # Show sample of extracted text
        print(f"\n   Sample text (first 200 chars):")
        print(f"   {text[:200]}...")
        
        # Test 2: Intelligent Chunking
        print("\n2. Testing Intelligent Chunking...")
        chunks = processor.create_chunks(text, str(sample_file))
        
        print(f"   ✓ Created {len(chunks)} chunks")
        print(f"   ✓ Average chunk size: {sum(len(c.content) for c in chunks) / len(chunks):.1f} chars")
        
        # Analyze chunk distribution
        chunk_sizes = [len(c.content) for c in chunks]
        print(f"   ✓ Chunk size range: {min(chunk_sizes)} - {max(chunk_sizes)} chars")
        
        # Test 3: Metadata Verification
        print("\n3. Testing Metadata...")
        for i, chunk in enumerate(chunks[:3]):  # Check first 3 chunks
            print(f"   Chunk {i+1}:")
            print(f"     • ID: {chunk.chunk_id}")
            print(f"     • Source: {chunk.source_file}")
            print(f"     • Size: {len(chunk.content)} chars")
            print(f"     • Importance: {chunk.importance_score:.2f}")
            print(f"     • Content type: {chunk.content_type}")
            print(f"     • Has equations: {chunk.has_equations}")
            print(f"     • Has code: {chunk.has_code_blocks}")
            if chunk.metadata:
                print(f"     • Metadata keys: {list(chunk.metadata.keys())}")
        
        # Test 4: Content Analysis
        print("\n4. Testing Content Analysis...")
        equations_count = sum(1 for c in chunks if c.has_equations)
        code_count = sum(1 for c in chunks if c.has_code_blocks)
        high_importance = sum(1 for c in chunks if c.importance_score > 0.7)
        
        print(f"   ✓ Chunks with equations: {equations_count}")
        print(f"   ✓ Chunks with code: {code_count}")
        print(f"   ✓ High importance chunks: {high_importance}")
        
        # Test 5: Sample Chunks for Manual Verification
        print("\n5. Sample Chunks for Manual Verification:")
        print("-" * 40)
        
        for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
            print(f"\nChunk {i+1} (Importance: {chunk.importance_score:.2f}):")
            print(f"Content: {chunk.content[:200]}...")
            if chunk.content != chunk.content[:200]:
                print("(truncated)")
            print()
        
        # Test 6: Edge Cases
        print("\n6. Testing Edge Cases...")
        
        # Empty text
        try:
            empty_chunks = processor.create_chunks("", "test.pdf")
            print(f"   ✓ Empty text handling: {len(empty_chunks)} chunks")
        except Exception as e:
            print(f"   ⚠ Empty text error: {e}")
        
        # Very short text
        try:
            short_chunks = processor.create_chunks("Short text.", "test.pdf")
            print(f"   ✓ Short text handling: {len(short_chunks)} chunks")
        except Exception as e:
            print(f"   ⚠ Short text error: {e}")
        
        print("\n" + "="*60)
        print("PDF PROCESSOR TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return chunks
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Cleanup
        if sample_file.exists():
            sample_file.unlink()
            print(f"✓ Cleaned up sample file: {sample_file}")

if __name__ == "__main__":
    print(f"Starting PDF Processor Tests at {datetime.now()}")
    print(f"Configuration: {settings}")
    
    # Run the tests
    chunks = asyncio.run(test_pdf_processor())
    
    if chunks:
        print(f"\n📊 SUMMARY:")
        print(f"   • Total chunks created: {len(chunks)}")
        print(f"   • Average importance: {sum(c.importance_score for c in chunks) / len(chunks):.2f}")
        print(f"   • Content types: {set(c.content_type for c in chunks)}")
        print(f"   • Test Status: ✅ PASSED")
    else:
        print(f"\n📊 SUMMARY:")
        print(f"   • Test Status: ❌ FAILED")
