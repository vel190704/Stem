#!/usr/bin/env python3
"""
Full Pipeline Integration Test for STEM Assessment Generator
Tests the complete workflow from PDF upload to question generation
"""

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any
import statistics

# Test imports
from pdf_processor import PDFProcessor
from rag_pipeline import RAGPipeline
from generator import AssessmentGenerator
from config import settings

class PipelineTestRunner:
    """Comprehensive test runner for the full pipeline"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.rag_pipeline = RAGPipeline()
        self.generator = AssessmentGenerator()
        self.test_results = {}
        
    async def run_full_pipeline_test(self, pdf_path: str, num_questions: int = 10):
        """Run complete pipeline test"""
        print(f"\n🧪 Starting Full Pipeline Test")
        print(f"📄 PDF: {pdf_path}")
        print(f"🎯 Target Questions: {num_questions}")
        print("=" * 60)
        
        # Track overall timing
        start_time = time.time()
        
        try:
            # Step 1: PDF Processing
            print("\n📖 Step 1: PDF Processing...")
            processing_start = time.time()
            chunks = await self.test_pdf_processing(pdf_path)
            processing_time = time.time() - processing_start
            print(f"✅ PDF processed in {processing_time:.2f}s → {len(chunks)} chunks")
            
            # Step 2: RAG Pipeline
            print("\n🔍 Step 2: RAG Pipeline Setup...")
            rag_start = time.time()
            retriever = await self.test_rag_pipeline(chunks, pdf_path)
            rag_time = time.time() - rag_start
            print(f"✅ RAG pipeline ready in {rag_time:.2f}s")
            
            # Step 3: Question Generation
            print("\n❓ Step 3: Question Generation...")
            generation_start = time.time()
            assessment = await self.test_question_generation(retriever, num_questions)
            generation_time = time.time() - generation_start
            print(f"✅ Questions generated in {generation_time:.2f}s")
            
            # Step 4: Quality Analysis
            print("\n🔍 Step 4: Quality Analysis...")
            quality_metrics = self.analyze_question_quality(assessment)
            
            # Summary
            total_time = time.time() - start_time
            self.print_test_summary(assessment, quality_metrics, {
                'processing_time': processing_time,
                'rag_time': rag_time,
                'generation_time': generation_time,
                'total_time': total_time,
                'chunks_created': len(chunks)
            })
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_pdf_processing(self, pdf_path: str) -> List[Dict]:
        """Test PDF processing functionality"""
        if not Path(pdf_path).exists():
            # Create a sample PDF if it doesn't exist
            await self.create_sample_pdf(pdf_path)
        
        # Extract text and create chunks
        text_data = self.pdf_processor.extract_text_from_pdf(pdf_path)
        chunks = self.pdf_processor.create_chunks(text_data)
        
        # Validate chunks
        assert len(chunks) > 0, "No chunks created from PDF"
        assert all('text' in chunk for chunk in chunks), "Chunks missing text field"
        assert all('metadata' in chunk for chunk in chunks), "Chunks missing metadata"
        
        # Check chunk quality
        avg_chunk_length = statistics.mean(len(chunk['text']) for chunk in chunks)
        print(f"  📊 Average chunk length: {avg_chunk_length:.0f} characters")
        print(f"  📄 Pages processed: {len(set(chunk['metadata'].get('page', 0) for chunk in chunks))}")
        
        return chunks
    
    async def test_rag_pipeline(self, chunks: List[Dict], source_file: str):
        """Test RAG pipeline functionality"""
        collection_name = f"test_{uuid.uuid4().hex[:8]}"
        
        # Process chunks and create retriever
        retriever = await self.rag_pipeline.process_chunks(chunks, collection_name, source_file)
        
        # Test retrieval
        test_queries = [
            "What is blockchain consensus?",
            "How does mining work?",
            "What are smart contracts?"
        ]
        
        for query in test_queries:
            results = retriever.retrieve(query, k=3)
            assert len(results) > 0, f"No results for query: {query}"
            print(f"  🔍 Query '{query[:30]}...' → {len(results)} results")
        
        return retriever
    
    async def test_question_generation(self, retriever, num_questions: int):
        """Test question generation with quality validation"""
        difficulties = ["easy", "medium", "hard"]
        
        # Generate assessment
        assessment = await self.generator.generate_assessment(
            retriever, 
            num_questions,
            difficulties[1]  # medium difficulty
        )
        
        # Basic validation
        assert assessment is not None, "Assessment generation failed"
        assert len(assessment.questions) > 0, "No questions generated"
        
        print(f"  📊 Generated {len(assessment.questions)} questions")
        print(f"  🎯 Target was {num_questions} questions")
        print(f"  ⚡ Success rate: {len(assessment.questions)/num_questions*100:.1f}%")
        
        return assessment
    
    def analyze_question_quality(self, assessment) -> Dict[str, Any]:
        """Analyze quality metrics of generated questions"""
        questions = assessment.questions
        metrics = {
            'total_questions': len(questions),
            'placeholder_count': 0,
            'short_options_count': 0,
            'duplicate_questions': 0,
            'difficulty_distribution': {'easy': 0, 'medium': 0, 'hard': 0},
            'avg_option_length': 0,
            'correct_position_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0},
            'quality_issues': []
        }
        
        question_texts = set()
        option_lengths = []
        
        for i, question in enumerate(questions):
            q_text = question.question_text
            
            # Check for duplicates
            if q_text in question_texts:
                metrics['duplicate_questions'] += 1
                metrics['quality_issues'].append(f"Question {i+1}: Duplicate question text")
            question_texts.add(q_text)
            
            # Check for placeholders
            if any(placeholder in str(question.dict()).lower() for placeholder in 
                   ['placeholder', 'option a', 'option b', 'option c', 'option d', 'xxx', '...']):
                metrics['placeholder_count'] += 1
                metrics['quality_issues'].append(f"Question {i+1}: Contains placeholders")
            
            # Analyze options
            if hasattr(question, 'options') and question.options:
                for opt_key, opt_text in question.options.items():
                    if len(opt_text.split()) < 3:  # Less than 3 words
                        metrics['short_options_count'] += 1
                        metrics['quality_issues'].append(f"Question {i+1} Option {opt_key}: Too short")
                    option_lengths.append(len(opt_text.split()))
            
            # Track difficulty distribution
            difficulty = getattr(question, 'difficulty', 'medium').lower()
            if difficulty in metrics['difficulty_distribution']:
                metrics['difficulty_distribution'][difficulty] += 1
            
            # Track correct answer position
            correct_pos = getattr(question, 'correct_position', 'A')
            if correct_pos in metrics['correct_position_distribution']:
                metrics['correct_position_distribution'][correct_pos] += 1
        
        # Calculate averages
        if option_lengths:
            metrics['avg_option_length'] = statistics.mean(option_lengths)
        
        return metrics
    
    def print_test_summary(self, assessment, quality_metrics: Dict, timing_data: Dict):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        
        # Basic Results
        print(f"✅ Questions Generated: {quality_metrics['total_questions']}")
        print(f"⏱️  Total Time: {timing_data['total_time']:.2f}s")
        print(f"⚡ Time per Question: {timing_data['generation_time']/quality_metrics['total_questions']:.2f}s")
        
        # Quality Metrics
        print(f"\n🎯 QUALITY METRICS:")
        print(f"   Placeholder Issues: {quality_metrics['placeholder_count']} (target: 0)")
        print(f"   Short Options: {quality_metrics['short_options_count']} (target: 0)")
        print(f"   Duplicate Questions: {quality_metrics['duplicate_questions']} (target: 0)")
        print(f"   Avg Option Length: {quality_metrics['avg_option_length']:.1f} words")
        
        # Difficulty Distribution
        print(f"\n📊 DIFFICULTY DISTRIBUTION:")
        for difficulty, count in quality_metrics['difficulty_distribution'].items():
            percentage = (count / quality_metrics['total_questions']) * 100 if quality_metrics['total_questions'] > 0 else 0
            print(f"   {difficulty.title()}: {count} ({percentage:.1f}%)")
        
        # Correct Answer Distribution
        print(f"\n🎲 CORRECT ANSWER DISTRIBUTION:")
        for position, count in quality_metrics['correct_position_distribution'].items():
            percentage = (count / quality_metrics['total_questions']) * 100 if quality_metrics['total_questions'] > 0 else 0
            print(f"   Position {position}: {count} ({percentage:.1f}%)")
        
        # Performance Breakdown
        print(f"\n⏱️  PERFORMANCE BREAKDOWN:")
        print(f"   PDF Processing: {timing_data['processing_time']:.2f}s")
        print(f"   RAG Setup: {timing_data['rag_time']:.2f}s")  
        print(f"   Question Generation: {timing_data['generation_time']:.2f}s")
        
        # Quality Issues
        if quality_metrics['quality_issues']:
            print(f"\n⚠️  QUALITY ISSUES ({len(quality_metrics['quality_issues'])}):")
            for issue in quality_metrics['quality_issues'][:10]:  # Show first 10
                print(f"   • {issue}")
            if len(quality_metrics['quality_issues']) > 10:
                print(f"   ... and {len(quality_metrics['quality_issues']) - 10} more")
        
        # Overall Assessment
        print(f"\n🏆 OVERALL ASSESSMENT:")
        issues = quality_metrics['placeholder_count'] + quality_metrics['short_options_count'] + quality_metrics['duplicate_questions']
        if issues == 0:
            print("   ✅ EXCELLENT - No quality issues detected!")
        elif issues <= 2:
            print("   ✅ GOOD - Minor issues detected")
        elif issues <= 5:
            print("   ⚠️  FAIR - Some quality issues need attention")
        else:
            print("   ❌ POOR - Significant quality issues detected")
    
    async def create_sample_pdf(self, pdf_path: str):
        """Create a sample PDF for testing if none exists"""
        print(f"📝 Creating sample PDF: {pdf_path}")
        
        # For testing purposes, create a simple text file that can be used
        # In a real scenario, you'd want to use a PDF library like reportlab
        sample_content = """
        Blockchain Technology Overview
        
        Blockchain is a distributed ledger technology that maintains a continuously growing list of records, called blocks, which are linked and secured using cryptography. Each block contains a cryptographic hash of the previous block, a timestamp, and transaction data.
        
        Consensus Mechanisms
        
        Consensus mechanisms are protocols that ensure all nodes in a blockchain network agree on the validity of transactions. The most common consensus mechanisms include:
        
        1. Proof of Work (PoW): Miners compete to solve complex mathematical problems to validate transactions and create new blocks. This process requires significant computational power and energy consumption.
        
        2. Proof of Stake (PoS): Validators are chosen to create new blocks based on their stake in the network. This mechanism is more energy-efficient than PoW.
        
        Mining Process
        
        Mining is the process of validating transactions and adding them to the blockchain. Miners use computational power to solve cryptographic puzzles, and the first to solve the puzzle gets to add the next block to the chain and receive a reward.
        
        Smart Contracts
        
        Smart contracts are self-executing contracts with the terms directly written into code. They automatically execute when predetermined conditions are met, eliminating the need for intermediaries. Smart contracts run on blockchain platforms like Ethereum.
        
        Cryptocurrency Wallets
        
        A cryptocurrency wallet is a digital tool that allows users to store, send, and receive cryptocurrencies. Wallets contain private keys that provide access to the user's cryptocurrency holdings. There are different types of wallets including hot wallets (online) and cold wallets (offline).
        
        Gas Fees
        
        Gas fees are transaction costs paid to miners or validators for processing transactions on a blockchain network. The amount of gas required depends on the complexity of the transaction or smart contract execution.
        """
        
        # Save as text file for now (in real implementation, convert to PDF)
        Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
        with open(pdf_path.replace('.pdf', '.txt'), 'w') as f:
            f.write(sample_content)
        
        print(f"✅ Sample content created: {pdf_path.replace('.pdf', '.txt')}")

async def main():
    """Run comprehensive pipeline tests"""
    runner = PipelineTestRunner()
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Basic Blockchain Test',
            'pdf_path': 'test_data/basic_blockchain.pdf',
            'num_questions': 5
        },
        {
            'name': 'Medium Size Test', 
            'pdf_path': 'test_data/smart_contracts.pdf',
            'num_questions': 10
        },
        {
            'name': 'Large Test',
            'pdf_path': 'test_data/advanced_topics.pdf', 
            'num_questions': 15
        }
    ]
    
    print("🚀 Starting Comprehensive Pipeline Tests")
    print("=" * 60)
    
    results = {}
    for scenario in test_scenarios:
        print(f"\n🧪 Running: {scenario['name']}")
        success = await runner.run_full_pipeline_test(
            scenario['pdf_path'], 
            scenario['num_questions']
        )
        results[scenario['name']] = success
        
        if not success:
            print(f"❌ {scenario['name']} failed!")
            break
        
        print(f"✅ {scenario['name']} passed!")
        
        # Wait between tests
        await asyncio.sleep(1)
    
    # Final summary
    print("\n" + "=" * 60)
    print("🏁 FINAL TEST RESULTS")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Pipeline is working correctly.")
    else:
        print(f"\n⚠️ {total-passed} test(s) failed. Please review the issues above.")

if __name__ == "__main__":
    asyncio.run(main())
