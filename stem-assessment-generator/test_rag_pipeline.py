#!/usr/bin/env python3
"""
Test script for RAG Pipeline
Tests embedding generation, ChromaDB storage, and intelligent retrieval
"""
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from rag_pipeline import RAGPipeline
from models import Chunk
from config import settings
import uuid

def create_dummy_blockchain_chunks() -> List[Chunk]:
    """Create dummy chunks for testing when no PDF is available"""
    dummy_data = [
        {
            "content": "Blockchain is a distributed ledger technology that maintains a continuously growing list of records, called blocks, which are linked and secured using cryptography. Each block contains a cryptographic hash of the previous block, a timestamp, and transaction data.",
            "content_type": "conceptual",
            "importance_score": 0.9,
            "has_equations": False,
            "has_code_blocks": False
        },
        {
            "content": "Hash functions are mathematical algorithms that take an input and return a fixed-size string of bytes. In blockchain, SHA-256 is commonly used. The hash function ensures data integrity by producing a unique fingerprint for each block. Example: SHA-256('Hello') = 185f8db32271fe25f561a6fc938b2e264306ec304eda518007d1764826381969",
            "content_type": "technical",
            "importance_score": 0.8,
            "has_equations": True,
            "has_code_blocks": False
        },
        {
            "content": "Proof of Work (PoW) is a consensus mechanism where miners compete to solve computationally difficult puzzles. The first miner to solve the puzzle gets to add the next block and receive a reward. This process requires significant computational power and energy consumption.",
            "content_type": "procedural",
            "importance_score": 0.85,
            "has_equations": False,
            "has_code_blocks": False
        },
        {
            "content": "Smart contracts are self-executing contracts with the terms of the agreement directly written into code. Here's a simple example in Solidity: contract SimpleStorage { uint256 public storedData; function set(uint256 x) public { storedData = x; } function get() public view returns (uint256) { return storedData; } }",
            "content_type": "technical",
            "importance_score": 0.75,
            "has_equations": False,
            "has_code_blocks": True
        },
        {
            "content": "Decentralization means that no single entity has control over the entire blockchain network. This is achieved through a distributed network of nodes that validate and record transactions. Each node maintains a copy of the entire blockchain ledger.",
            "content_type": "conceptual",
            "importance_score": 0.8,
            "has_equations": False,
            "has_code_blocks": False
        },
        {
            "content": "Consensus mechanisms ensure that all nodes in the blockchain network agree on the current state of the ledger. Popular mechanisms include Proof of Work (PoW), Proof of Stake (PoS), and Delegated Proof of Stake (DPoS). Each has different trade-offs between security, speed, and energy efficiency.",
            "content_type": "comparative",
            "importance_score": 0.9,
            "has_equations": False,
            "has_code_blocks": False
        },
        {
            "content": "Bitcoin was the first successful implementation of blockchain technology, created by the pseudonymous Satoshi Nakamoto in 2008. It introduced the concept of a peer-to-peer electronic cash system without the need for a trusted third party.",
            "content_type": "historical",
            "importance_score": 0.7,
            "has_equations": False,
            "has_code_blocks": False
        },
        {
            "content": "Merkle trees are binary trees used in blockchain to efficiently summarize all transactions in a block. They allow for quick verification of large data structures and enable light clients to verify transactions without downloading the entire blockchain.",
            "content_type": "technical",
            "importance_score": 0.8,
            "has_equations": False,
            "has_code_blocks": False
        }
    ]
    
    chunks = []
    for i, data in enumerate(dummy_data):
        chunk = Chunk(
            chunk_id=f"test_chunk_{i+1}_{uuid.uuid4().hex[:8]}",
            content=data["content"],
            source_file="test_blockchain_data.txt",
            chunk_index=i,
            content_type=data["content_type"],
            importance_score=data["importance_score"],
            has_equations=data["has_equations"],
            has_code_blocks=data["has_code_blocks"],
            metadata={
                "page_number": (i // 3) + 1,  # Simulate multiple pages
                "section": f"Section {i+1}",
                "created_for_testing": True
            }
        )
        chunks.append(chunk)
    
    print(f"✓ Created {len(chunks)} dummy blockchain chunks")
    return chunks

async def test_rag_pipeline():
    """Test RAG pipeline with comprehensive checks"""
    print("="*60)
    print("RAG PIPELINE TEST SUITE")
    print("="*60)
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    print("✓ RAGPipeline initialized")
    
    # Create test collection name
    test_collection = f"test_blockchain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"✓ Using test collection: {test_collection}")
    
    try:
        # Create dummy chunks
        print("\n1. Creating Test Data...")
        chunks = create_dummy_blockchain_chunks()
        
        # Test 1: Embedding Generation
        print("\n2. Testing Embedding Generation...")
        if not settings.is_openai_configured():
            print("   ⚠ OpenAI not configured - skipping embedding tests")
            print("   💡 Set OPENAI_API_KEY environment variable to test embeddings")
            return
        
        # Test single embedding
        test_text = "What is blockchain technology?"
        embedding = await pipeline.generate_embedding(test_text)
        
        print(f"   ✓ Generated embedding for test text")
        print(f"   ✓ Embedding dimension: {len(embedding)}")
        print(f"   ✓ Embedding type: {type(embedding[0])}")
        print(f"   ✓ Sample values: {embedding[:5]}...")
        
        # Test 2: Batch Embedding Generation
        print("\n3. Testing Batch Embedding Generation...")
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = await pipeline.generate_embeddings_batch(chunk_texts)
        
        print(f"   ✓ Generated {len(embeddings)} embeddings")
        print(f"   ✓ All embeddings same dimension: {len(set(len(e) for e in embeddings)) == 1}")
        
        # Test 3: ChromaDB Storage
        print("\n4. Testing ChromaDB Storage...")
        await pipeline.store_chunks(chunks, test_collection)
        
        print(f"   ✓ Stored {len(chunks)} chunks in ChromaDB")
        
        # Verify storage
        collections = pipeline.list_collections()
        print(f"   ✓ Available collections: {collections}")
        print(f"   ✓ Test collection exists: {test_collection in collections}")
        
        # Test 4: Basic Retrieval
        print("\n5. Testing Basic Retrieval...")
        
        test_queries = [
            "What is blockchain?",
            "How does proof of work consensus work?",
            "Explain smart contracts",
            "What are hash functions?",
            "Bitcoin and Satoshi Nakamoto"
        ]
        
        for query in test_queries:
            print(f"\n   Query: '{query}'")
            results = await pipeline.retrieve_relevant_chunks(
                query, test_collection, top_k=3
            )
            
            print(f"   ✓ Retrieved {len(results)} chunks")
            for i, (chunk, score) in enumerate(results):
                print(f"     {i+1}. Score: {score:.3f} | Type: {chunk.content_type} | "
                      f"Content: {chunk.content[:100]}...")
        
        # Test 5: MMR Diversity Retrieval
        print("\n6. Testing MMR Diversity Retrieval...")
        
        diverse_results = await pipeline.retrieve_with_mmr(
            "blockchain consensus mechanisms", 
            test_collection, 
            top_k=4, 
            diversity_lambda=0.7
        )
        
        print(f"   ✓ Retrieved {len(diverse_results)} diverse chunks")
        print("   Diversity analysis:")
        
        content_types = [chunk.content_type for chunk, _ in diverse_results]
        importance_scores = [chunk.importance_score for chunk, _ in diverse_results]
        
        print(f"     • Content types: {set(content_types)}")
        print(f"     • Importance range: {min(importance_scores):.2f} - {max(importance_scores):.2f}")
        
        for i, (chunk, score) in enumerate(diverse_results):
            print(f"     {i+1}. Score: {score:.3f} | Type: {chunk.content_type} | "
                  f"Importance: {chunk.importance_score:.2f}")
            print(f"        {chunk.content[:80]}...")
        
        # Test 6: Context Building
        print("\n7. Testing Context Building...")
        
        context = await pipeline.build_context(
            "Explain blockchain consensus and smart contracts",
            test_collection,
            max_chunks=5
        )
        
        print(f"   ✓ Built context with {len(context.split('---'))-1} chunks")
        print(f"   ✓ Context length: {len(context)} characters")
        print(f"\n   Sample context (first 300 chars):")
        print(f"   {context[:300]}...")
        
        # Test 7: Cache Testing
        print("\n8. Testing Cache Performance...")
        
        # Test embedding cache
        start_time = datetime.now()
        cached_embedding = await pipeline.generate_embedding(test_text)  # Should be cached
        cached_time = (datetime.now() - start_time).total_seconds()
        
        print(f"   ✓ Cached embedding retrieval time: {cached_time:.4f}s")
        print(f"   ✓ Cache hit (same result): {embedding == cached_embedding}")
        
        # Test query cache
        start_time = datetime.now()
        cached_results = await pipeline.retrieve_relevant_chunks(
            "What is blockchain?", test_collection, top_k=3
        )
        cached_query_time = (datetime.now() - start_time).total_seconds()
        
        print(f"   ✓ Cached query retrieval time: {cached_query_time:.4f}s")
        
        # Test 8: Edge Cases
        print("\n9. Testing Edge Cases...")
        
        # Empty query
        try:
            empty_results = await pipeline.retrieve_relevant_chunks("", test_collection)
            print(f"   ✓ Empty query handling: {len(empty_results)} results")
        except Exception as e:
            print(f"   ⚠ Empty query error: {e}")
        
        # Very long query
        try:
            long_query = "blockchain " * 100  # Very long query
            long_results = await pipeline.retrieve_relevant_chunks(long_query, test_collection, top_k=2)
            print(f"   ✓ Long query handling: {len(long_results)} results")
        except Exception as e:
            print(f"   ⚠ Long query error: {e}")
        
        # Non-existent collection
        try:
            fake_results = await pipeline.retrieve_relevant_chunks("test", "fake_collection")
            print(f"   ⚠ Non-existent collection returned: {len(fake_results)} results")
        except Exception as e:
            print(f"   ✓ Non-existent collection error handled: {type(e).__name__}")
        
        print("\n" + "="*60)
        print("RAG PIPELINE TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup test collection
        try:
            if test_collection in pipeline.list_collections():
                pipeline.delete_collection(test_collection)
                print(f"✓ Cleaned up test collection: {test_collection}")
        except Exception as e:
            print(f"⚠ Cleanup warning: {e}")

async def test_cache_performance():
    """Test cache performance specifically"""
    print("\n" + "="*60)
    print("CACHE PERFORMANCE TEST")
    print("="*60)
    
    if not settings.is_openai_configured():
        print("⚠ OpenAI not configured - skipping cache performance tests")
        return
    
    pipeline = RAGPipeline()
    
    # Test repeated embedding generation
    test_text = "Blockchain is a revolutionary technology"
    times = []
    
    print("Testing embedding cache performance...")
    for i in range(5):
        start = datetime.now()
        await pipeline.generate_embedding(test_text)
        duration = (datetime.now() - start).total_seconds()
        times.append(duration)
        print(f"  Attempt {i+1}: {duration:.4f}s")
    
    print(f"\nCache analysis:")
    print(f"  First call (no cache): {times[0]:.4f}s")
    print(f"  Subsequent calls (cached): {min(times[1:]):.4f}s - {max(times[1:]):.4f}s")
    print(f"  Speedup ratio: {times[0] / min(times[1:]):.1f}x")

if __name__ == "__main__":
    print(f"Starting RAG Pipeline Tests at {datetime.now()}")
    print(f"Configuration: {settings}")
    print(f"OpenAI Configured: {settings.is_openai_configured()}")
    
    # Run the tests
    success = asyncio.run(test_rag_pipeline())
    
    if success and settings.is_openai_configured():
        asyncio.run(test_cache_performance())
    
    print(f"\n📊 FINAL SUMMARY:")
    if success:
        print(f"   • RAG Pipeline Status: ✅ PASSED")
        if settings.is_openai_configured():
            print(f"   • Full tests completed with OpenAI integration")
        else:
            print(f"   • Limited tests completed (OpenAI not configured)")
    else:
        print(f"   • RAG Pipeline Status: ❌ FAILED")
    
    print(f"   • Test completed at: {datetime.now()}")
