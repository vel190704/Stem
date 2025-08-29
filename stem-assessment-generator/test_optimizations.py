#!/usr/bin/env python3
"""
Test script for optimization features
"""
import asyncio
import time
import json
from pathlib import Path

# Add the project root to Python path
import sys
sys.path.append(str(Path(__file__).parent))

from optimizations import (
    QuestionCache, 
    RateLimitManager, 
    MemoryManager, 
    DatabaseOptimizer,
    optimization_manager
)

async def test_question_cache():
    """Test question caching functionality"""
    print("🧪 Testing Question Cache...")
    
    cache = QuestionCache()
    
    # Test content hash generation
    content = "Blockchain is a distributed ledger technology"
    content_hash = cache.get_content_hash(content, "medium", 5)
    print(f"✓ Generated content hash: {content_hash}")
    
    # Test caching questions
    sample_questions = [
        {
            "question_text": "What is blockchain?",
            "options": {"A": "Database", "B": "Distributed ledger", "C": "Network", "D": "Protocol"},
            "correct_position": "B",
            "difficulty": "EASY"
        },
        {
            "question_text": "What is consensus?",
            "options": {"A": "Agreement", "B": "Conflict", "C": "Process", "D": "Algorithm"},
            "correct_position": "A",
            "difficulty": "MEDIUM"
        }
    ]
    
    cache.cache_questions(content_hash, sample_questions, {"test": True})
    print(f"✓ Cached {len(sample_questions)} questions")
    
    # Test retrieving cached questions
    cached = cache.get_cached_questions(content_hash, 2)
    if cached and len(cached) == 2:
        print(f"✓ Retrieved {len(cached)} cached questions")
    else:
        print("❌ Failed to retrieve cached questions")
    
    # Test cache statistics
    stats = cache.get_cache_stats()
    print(f"✓ Cache stats: {stats}")

async def test_rate_limiting():
    """Test rate limiting functionality"""
    print("\n🧪 Testing Rate Limiting...")
    
    rate_manager = RateLimitManager()
    
    # Simulate API calls
    start_time = time.time()
    for i in range(3):
        await rate_manager.wait_if_needed('openai_completions')
        print(f"✓ API call {i+1} processed")
    
    elapsed = time.time() - start_time
    print(f"✓ Rate limiting test completed in {elapsed:.2f}s")
    
    # Test estimation
    estimated = rate_manager.get_estimated_completion_time('openai_completions', 10)
    print(f"✓ Estimated completion time for 10 calls: {estimated:.1f}s")

def test_memory_management():
    """Test memory management functionality"""
    print("\n🧪 Testing Memory Management...")
    
    memory_manager = MemoryManager()
    
    # Create some temporary files
    test_file = Path("test_temp_file.txt")
    test_file.write_text("test data")
    memory_manager.register_temp_file(test_file)
    print(f"✓ Registered temporary file: {test_file}")
    
    # Test memory usage
    memory_usage = memory_manager.get_memory_usage_mb()
    print(f"✓ Current memory usage: {memory_usage:.1f} MB")
    
    # Test cleanup
    memory_manager.cleanup_temp_files()
    if not test_file.exists():
        print("✓ Temporary file cleaned up successfully")
    else:
        print("❌ Temporary file cleanup failed")
    
    # Test garbage collection
    memory_manager.force_garbage_collection()
    print("✓ Forced garbage collection")

def test_database_optimizer():
    """Test database optimization functionality"""
    print("\n🧪 Testing Database Optimizer...")
    
    db_optimizer = DatabaseOptimizer()
    
    # Record some test statistics
    db_optimizer.record_generation_stats(
        content_hash="test_hash_123",
        num_questions=5,
        difficulty="medium",
        generation_time=2.5,
        success_rate=1.0
    )
    print("✓ Recorded test generation statistics")
    
    # Get performance analytics
    analytics = db_optimizer.get_performance_analytics()
    print(f"✓ Retrieved performance analytics: {json.dumps(analytics, indent=2)}")
    
    # Test database vacuum
    db_optimizer.vacuum_database()
    print("✓ Database vacuum completed")

async def test_optimization_manager():
    """Test the main optimization manager"""
    print("\n🧪 Testing Optimization Manager...")
    
    # Get optimization status
    status = optimization_manager.get_optimization_status()
    print("✓ Retrieved optimization status:")
    print(json.dumps(status, indent=2, default=str))
    
    # Get recommendations
    recommendations = optimization_manager.get_optimization_recommendations()
    print(f"✓ Optimization recommendations: {recommendations}")

async def run_all_tests():
    """Run all optimization tests"""
    print("🚀 Starting Optimization Tests\n")
    
    try:
        await test_question_cache()
        await test_rate_limiting()
        test_memory_management()
        test_database_optimizer()
        await test_optimization_manager()
        
        print("\n🎉 All optimization tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_all_tests())
