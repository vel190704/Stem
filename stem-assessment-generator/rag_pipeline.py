"""
Enhanced RAG pipeline with intelligent retrieval and vector storage
Features: ChromaDB integration, MMR diversity, context building, caching
"""
import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import uuid

import chromadb
from chromadb.config import Settings
import numpy as np
import openai
from openai import OpenAI

from config import settings
from models import PDFContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingCache:
    """Cache for embeddings to avoid re-computation"""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or (settings.vectordb_path / "embedding_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.memory_cache = {}  # In-memory cache for recent embeddings
        self.max_memory_cache = 1000
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text to use as cache key"""
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)
        return hashlib.sha256(text.encode()).hexdigest()
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding if available"""
        text_hash = self._get_text_hash(text)
        
        # Check memory cache first
        if text_hash in self.memory_cache:
            return self.memory_cache[text_hash]
        
        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"{text_hash}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    embedding = data['embedding']
                    
                    # Add to memory cache
                    self._add_to_memory_cache(text_hash, embedding)
                    return embedding
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
        
        return None
    
    def store_embedding(self, text: str, embedding: List[float]):
        """Store embedding in cache"""
        text_hash = self._get_text_hash(text)
        
        # Store in memory cache
        self._add_to_memory_cache(text_hash, embedding)
        
        # Store on disk
        cache_file = os.path.join(self.cache_dir, f"{text_hash}.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'text_hash': text_hash,
                    'embedding': embedding,
                    'timestamp': datetime.now().isoformat(),
                    'text_preview': text[:100]  # For debugging
                }, f)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    def _add_to_memory_cache(self, text_hash: str, embedding: List[float]):
        """Add embedding to memory cache with size limit"""
        if len(self.memory_cache) >= self.max_memory_cache:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        self.memory_cache[text_hash] = embedding

class QueryCache:
    """Cache for query results to improve response time"""
    
    def __init__(self, ttl_minutes: int = 30):
        self.cache = {}
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def _get_query_key(self, query: str, top_k: int, diversity: float, 
                      collection_name: str) -> str:
        """Generate cache key for query"""
        key_data = f"{query}:{top_k}:{diversity}:{collection_name}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_result(self, query: str, top_k: int, diversity: float, 
                   collection_name: str) -> Optional[Dict[str, Any]]:
        """Get cached query result if available and fresh"""
        key = self._get_query_key(query, top_k, diversity, collection_name)
        
        if key in self.cache:
            result, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                logger.debug(f"Using cached result for query: {query[:50]}...")
                return result
            else:
                # Remove expired entry
                del self.cache[key]
        
        return None
    
    def store_result(self, query: str, top_k: int, diversity: float,
                    collection_name: str, result: Dict[str, Any]):
        """Store query result in cache"""
        key = self._get_query_key(query, top_k, diversity, collection_name)
        self.cache[key] = (result, datetime.now())
        
        # Clean up old entries periodically
        if len(self.cache) > 100:
            self._cleanup_expired()
    
    def _cleanup_expired(self):
        """Remove expired cache entries"""
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp >= self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]

class RAGPipeline:
    """Enhanced RAG pipeline with intelligent retrieval and vector storage"""
    
    def __init__(self):
        """Initialize RAG pipeline with ChromaDB and OpenAI clients"""
        # Initialize OpenAI client
        settings.validate_openai_key()
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.embedding_model = "text-embedding-3-small"
        
        # Initialize caches
        self.embedding_cache = EmbeddingCache()
        self.query_cache = QueryCache()
        
        # ChromaDB setup
        self.vectorstore_path = settings.vectordb_path
        self.chroma_client = None
        self.collection = None
        self.default_collection_name = "stem_assessment"
        
        # Retrieval parameters
        self.max_batch_size = 100  # For embedding generation
        self.max_context_length = 4000  # Characters
        
        # Synonyms for query expansion
        self.synonyms = {
            "blockchain": ["distributed ledger", "cryptocurrency", "decentralized"],
            "consensus": ["agreement", "validation", "verification"],
            "smart contract": ["automated contract", "self-executing contract"],
            "mining": ["proof of work", "computation", "validation"],
            "token": ["cryptocurrency", "digital asset", "coin"],
            "node": ["participant", "peer", "network member"],
            "transaction": ["transfer", "exchange", "payment"],
            "security": ["cryptography", "protection", "safety"]
        }
        
        # Initialize vector store
        self.initialize_vectorstore()
    
    def initialize_vectorstore(self) -> chromadb.Client:
        """
        Initialize persistent ChromaDB client and collection
        
        Returns:
            ChromaDB client instance
        """
        try:
            # Ensure directory exists
            os.makedirs(self.vectorstore_path, exist_ok=True)
            
            # Initialize ChromaDB with persistent storage
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.vectorstore_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            logger.info(f"ChromaDB initialized at: {self.vectorstore_path}")
            
            # Get or create default collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name=self.default_collection_name
                )
                logger.info(f"Loaded existing collection: {self.default_collection_name}")
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    name=self.default_collection_name,
                    metadata={"description": "STEM Assessment content chunks"}
                )
                logger.info(f"Created new collection: {self.default_collection_name}")
            
            return self.chroma_client
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    async def generate_embeddings(self, texts: List[str], 
                                show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for texts with batching and caching
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress for large batches
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Ensure all texts are strings
        processed_texts = []
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                if isinstance(text, dict):
                    # Try to extract meaningful text from dict
                    text = str(text.get('text', text.get('content', str(text))))
                else:
                    text = str(text)
            
            if not text.strip():
                text = f"Empty text content {i}"
                
            processed_texts.append(text)
        
        texts = processed_texts
        embeddings = []
        cache_hits = 0
        
        # Check cache first
        texts_to_embed = []
        cached_embeddings = {}
        
        for i, text in enumerate(texts):
            cached_embedding = self.embedding_cache.get_embedding(text)
            if cached_embedding:
                cached_embeddings[i] = cached_embedding
                cache_hits += 1
            else:
                texts_to_embed.append((i, text))
        
        logger.info(f"Cache hits: {cache_hits}/{len(texts)}")
        
        # Generate embeddings for uncached texts in batches
        new_embeddings = {}
        
        if texts_to_embed:
            for batch_start in range(0, len(texts_to_embed), self.max_batch_size):
                batch_end = min(batch_start + self.max_batch_size, len(texts_to_embed))
                batch_texts = [texts_to_embed[i][1] for i in range(batch_start, batch_end)]
                batch_indices = [texts_to_embed[i][0] for i in range(batch_start, batch_end)]
                
                if show_progress and len(texts_to_embed) > 10:
                    logger.info(f"Generating embeddings: batch {batch_start//self.max_batch_size + 1}"
                              f"/{(len(texts_to_embed) - 1)//self.max_batch_size + 1}")
                
                try:
                    # Generate embeddings with retry logic
                    batch_embeddings = await self._generate_batch_embeddings(batch_texts)
                    
                    # Store in cache and result
                    for idx, embedding, original_idx in zip(range(len(batch_embeddings)), 
                                                          batch_embeddings, batch_indices):
                        new_embeddings[original_idx] = embedding
                        self.embedding_cache.store_embedding(batch_texts[idx], embedding)
                    
                    # Rate limiting
                    if batch_end < len(texts_to_embed):
                        await asyncio.sleep(0.1)  # Small delay between batches
                        
                except Exception as e:
                    logger.error(f"Failed to generate embeddings for batch: {e}")
                    # Continue with next batch, but mark missing embeddings
                    for original_idx in batch_indices:
                        new_embeddings[original_idx] = [0.0] * 1536  # Default embedding size
        
        # Combine cached and new embeddings in original order
        all_embeddings = []
        for i in range(len(texts)):
            if i in cached_embeddings:
                all_embeddings.append(cached_embeddings[i])
            elif i in new_embeddings:
                all_embeddings.append(new_embeddings[i])
            else:
                # Fallback for failed embeddings
                all_embeddings.append([0.0] * 1536)
        
        return all_embeddings
    
    async def _generate_batch_embeddings(self, texts: List[str], 
                                       max_retries: int = 3) -> List[List[float]]:
        """Generate embeddings for a batch with retry logic"""
        
        for attempt in range(max_retries):
            try:
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=texts
                )
                
                embeddings = [data.embedding for data in response.data]
                return embeddings
                
            except Exception as e:
                logger.warning(f"Embedding generation attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    raise
    
    async def store_chunks(self, chunks: List[Dict[str, Any]], 
                          collection_name: str = None) -> bool:
        """
        Store chunks with embeddings and metadata in ChromaDB
        
        Args:
            chunks: List of chunk dictionaries with text and metadata
            collection_name: Collection name (uses default if None)
            
        Returns:
            bool: Success status
        """
        if not chunks:
            logger.warning("No chunks to store")
            return False
        
        collection_name = collection_name or self.default_collection_name
        
        try:
            # Get or create collection
            try:
                collection = self.chroma_client.get_collection(name=collection_name)
            except Exception:
                collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": f"Content chunks for {collection_name}"}
                )
            
            # Prepare data for storage
            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []
            
            for i, chunk in enumerate(chunks):
                # Generate unique ID
                chunk_id = chunk.get('chunk_id') or f"chunk_{uuid.uuid4().hex[:8]}_{i}"
                chunk_ids.append(chunk_id)
                
                # Extract text - ensure it's always a string
                chunk_text = chunk.get('text') or chunk.get('content', '')
                
                # Convert to string if it's not already
                if not isinstance(chunk_text, str):
                    if isinstance(chunk_text, dict):
                        # If it's a dict, try to get meaningful text
                        chunk_text = str(chunk_text.get('text', chunk_text.get('content', str(chunk_text))))
                    else:
                        chunk_text = str(chunk_text)
                
                # Ensure we have some text content
                if not chunk_text.strip():
                    chunk_text = f"Empty chunk content {i}"
                    
                chunk_texts.append(chunk_text)
                
                # Prepare metadata (ChromaDB requires simple types)
                metadata = chunk.get('metadata', {}).copy()
                
                # Convert complex metadata to strings
                for key, value in metadata.items():
                    if isinstance(value, (dict, list)):
                        metadata[key] = json.dumps(value)
                    elif not isinstance(value, (str, int, float, bool)):
                        metadata[key] = str(value)
                
                # Add storage timestamp
                metadata['stored_at'] = datetime.now().isoformat()
                chunk_metadatas.append(metadata)
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
            embeddings = await self.generate_embeddings(chunk_texts)
            
            # Store in ChromaDB
            logger.info(f"Storing {len(chunks)} chunks in collection '{collection_name}'...")
            collection.add(
                ids=chunk_ids,
                documents=chunk_texts,
                embeddings=embeddings,
                metadatas=chunk_metadatas
            )
            
            logger.info(f"Successfully stored {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store chunks: {e}")
            return False
    
    async def retrieve_chunks(self, query: str, top_k: int = 10, diversity: float = 0.3,
                             collection_name: str = None, filter_metadata: Dict = None,
                             boost_importance: bool = True) -> Dict[str, Any]:
        """
        Intelligent retrieval with diversity and importance boosting
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            diversity: MMR diversity parameter (0.0 = no diversity, 1.0 = max diversity)
            collection_name: Collection to search (uses default if None)
            filter_metadata: Metadata filters for retrieval
            boost_importance: Whether to boost chunks with higher importance scores
            
        Returns:
            Dict with retrieved chunks and metadata
        """
        collection_name = collection_name or self.default_collection_name
        
        # Check cache first
        cached_result = self.query_cache.get_result(query, top_k, diversity, collection_name)
        if cached_result:
            return cached_result
        
        try:
            # Get collection
            collection = self.chroma_client.get_collection(name=collection_name)
            
            # Expand query with synonyms
            expanded_query = self._expand_query(query)
            
            # Generate query embedding
            query_embeddings = await self.generate_embeddings([expanded_query])
            query_embedding = query_embeddings[0]
            
            # Initial retrieval (get more than needed for diversity selection)
            initial_k = min(top_k * 3, 50)  # Get 3x more for diversity
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=initial_k,
                where=filter_metadata,
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'][0]:
                logger.warning(f"No results found for query: {query}")
                return {
                    "query": query,
                    "retrieved_chunks": [],
                    "total_chunks": 0,
                    "context_length": 0
                }
            
            # Process and enhance results
            enhanced_chunks = self._process_retrieval_results(
                results, query, boost_importance
            )
            
            # Apply MMR for diversity if requested
            if diversity > 0 and len(enhanced_chunks) > top_k:
                diverse_chunks = self._apply_mmr_diversity(
                    enhanced_chunks, query_embedding, top_k, diversity
                )
            else:
                diverse_chunks = enhanced_chunks[:top_k]
            
            # Build context for each chunk
            final_chunks = []
            for chunk in diverse_chunks:
                context_chunk = self._build_chunk_context(chunk, collection)
                final_chunks.append(context_chunk)
            
            # Calculate total context length
            total_context = sum(len(chunk.get('text', '')) for chunk in final_chunks)
            
            result = {
                "query": query,
                "expanded_query": expanded_query,
                "retrieved_chunks": final_chunks,
                "total_chunks": len(final_chunks),
                "context_length": total_context,
                "retrieval_stats": {
                    "initial_results": len(enhanced_chunks),
                    "after_diversity": len(diverse_chunks),
                    "diversity_applied": diversity > 0,
                    "importance_boosted": boost_importance
                }
            }
            
            # Cache the result
            self.query_cache.store_result(query, top_k, diversity, collection_name, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")
            return {
                "query": query,
                "retrieved_chunks": [],
                "total_chunks": 0,
                "context_length": 0,
                "error": str(e)
            }
    
    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms for better retrieval"""
        expanded_terms = [query]
        query_lower = query.lower()
        
        for term, synonyms in self.synonyms.items():
            if term in query_lower:
                expanded_terms.extend(synonyms)
        
        return " ".join(expanded_terms)
    
    def _process_retrieval_results(self, results: Dict, query: str, 
                                 boost_importance: bool) -> List[Dict[str, Any]]:
        """Process and enhance raw retrieval results"""
        enhanced_chunks = []
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        for doc, metadata, distance in zip(documents, metadatas, distances):
            # Convert distance to similarity score
            similarity_score = 1.0 - distance
            
            # Parse JSON metadata fields
            processed_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, str) and value.startswith(('[', '{')):
                    try:
                        processed_metadata[key] = json.loads(value)
                    except:
                        processed_metadata[key] = value
                else:
                    processed_metadata[key] = value
            
            # Apply importance boosting
            final_score = similarity_score
            if boost_importance:
                importance_score = processed_metadata.get('importance_score', 0.5)
                final_score = (similarity_score * 0.7) + (importance_score * 0.3)
            
            chunk = {
                "text": doc,
                "similarity_score": similarity_score,
                "final_score": final_score,
                "metadata": processed_metadata,
                "distance": distance
            }
            
            enhanced_chunks.append(chunk)
        
        # Sort by final score
        enhanced_chunks.sort(key=lambda x: x['final_score'], reverse=True)
        return enhanced_chunks
    
    def _apply_mmr_diversity(self, chunks: List[Dict], query_embedding: List[float],
                           top_k: int, diversity: float) -> List[Dict]:
        """Apply Maximum Marginal Relevance for diverse retrieval"""
        if len(chunks) <= top_k:
            return chunks
        
        selected = []
        remaining = chunks.copy()
        
        # Select first chunk (highest relevance)
        selected.append(remaining.pop(0))
        
        while len(selected) < top_k and remaining:
            best_chunk = None
            best_score = -1
            best_idx = -1
            
            for i, chunk in enumerate(remaining):
                # Get chunk embedding (would need to store or regenerate)
                # For now, use similarity score as proxy
                relevance_score = chunk['similarity_score']
                
                # Calculate diversity (simple approximation)
                # In full implementation, would use embedding similarity
                diversity_penalty = 0
                for selected_chunk in selected:
                    # Simple text overlap as diversity measure
                    overlap = len(set(chunk['text'].split()) & 
                                set(selected_chunk['text'].split()))
                    max_words = max(len(chunk['text'].split()), 
                                  len(selected_chunk['text'].split()))
                    if max_words > 0:
                        diversity_penalty += overlap / max_words
                
                diversity_penalty /= len(selected)  # Average penalty
                
                # MMR score
                mmr_score = (diversity * relevance_score - 
                           (1 - diversity) * diversity_penalty)
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_chunk = chunk
                    best_idx = i
            
            if best_chunk:
                selected.append(remaining.pop(best_idx))
            else:
                break
        
        return selected
    
    def _build_chunk_context(self, chunk: Dict, collection) -> Dict[str, Any]:
        """Build context for a chunk using previous/next chunks"""
        metadata = chunk.get('metadata', {})
        
        # Get related chunks using prev/next IDs
        context_chunks = []
        
        # Try to get previous chunk
        prev_chunk_id = metadata.get('prev_chunk')
        if prev_chunk_id:
            try:
                prev_result = collection.get(ids=[prev_chunk_id])
                if prev_result['documents']:
                    context_chunks.append({
                        'position': 'previous',
                        'text': prev_result['documents'][0][:200] + "..."
                    })
            except:
                pass
        
        # Try to get next chunk
        next_chunk_id = metadata.get('next_chunk')
        if next_chunk_id:
            try:
                next_result = collection.get(ids=[next_chunk_id])
                if next_result['documents']:
                    context_chunks.append({
                        'position': 'next',
                        'text': next_result['documents'][0][:200] + "..."
                    })
            except:
                pass
        
        # Enhanced chunk with context
        enhanced_chunk = chunk.copy()
        enhanced_chunk['context_chunks'] = context_chunks
        
        return enhanced_chunk
    
    def build_context(self, chunks: List[Dict[str, Any]], 
                     max_length: int = None) -> str:
        """
        Build coherent context from retrieved chunks
        
        Args:
            chunks: List of retrieved chunk dictionaries
            max_length: Maximum context length in characters
            
        Returns:
            str: Built context string
        """
        max_length = max_length or self.max_context_length
        
        if not chunks:
            return ""
        
        # Sort chunks by page and position for coherent reading
        sorted_chunks = self._sort_chunks_by_position(chunks)
        
        context_parts = []
        total_length = 0
        
        for i, chunk in enumerate(sorted_chunks):
            chunk_text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})
            
            # Add section header if available
            section = metadata.get('section')
            if section and (i == 0 or 
                          sorted_chunks[i-1].get('metadata', {}).get('section') != section):
                header = f"\n=== {section} ===\n"
                if total_length + len(header) + len(chunk_text) > max_length:
                    break
                context_parts.append(header)
                total_length += len(header)
            
            # Add chunk text
            if total_length + len(chunk_text) > max_length:
                # Add partial chunk if it fits
                remaining_space = max_length - total_length - 50  # Leave space for "..."
                if remaining_space > 100:
                    partial_text = chunk_text[:remaining_space] + "..."
                    context_parts.append(partial_text)
                break
            
            context_parts.append(chunk_text)
            total_length += len(chunk_text)
            
            # Add separator between chunks
            if i < len(sorted_chunks) - 1:
                separator = "\n\n"
                context_parts.append(separator)
                total_length += len(separator)
        
        return "".join(context_parts)
    
    def _sort_chunks_by_position(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort chunks by page number and chunk index for coherent reading"""
        def sort_key(chunk):
            metadata = chunk.get('metadata', {})
            page_num = metadata.get('page_num', 0)
            chunk_index = metadata.get('chunk_index', 0)
            return (page_num, chunk_index)
        
        return sorted(chunks, key=sort_key)
    
    def get_collection_info(self, collection_name: str = None) -> Dict[str, Any]:
        """Get information about a collection"""
        collection_name = collection_name or self.default_collection_name
        
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
            count = collection.count()
            
            return {
                "name": collection_name,
                "count": count,
                "metadata": collection.metadata
            }
        except Exception as e:
            return {"error": str(e)}
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections in the vector store"""
        try:
            collections = self.chroma_client.list_collections()
            return [
                {
                    "name": col.name,
                    "count": col.count(),
                    "metadata": col.metadata
                }
                for col in collections
            ]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            self.chroma_client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False
    
    def clear_cache(self):
        """Clear all caches"""
        self.embedding_cache.memory_cache.clear()
        self.query_cache.cache.clear()
        logger.info("Cleared all caches")
    
    async def process_pdf_content(self, pdf_content: PDFContent, 
                                collection_name: str = None) -> bool:
        """
        Process PDF content and store in vector database
        
        Args:
            pdf_content: PDFContent object with chunks and metadata
            collection_name: Collection name for storage
            
        Returns:
            bool: Success status
        """
        collection_name = collection_name or self.default_collection_name
        
        try:
            # Extract intelligent chunks if available
            chunks_data = []
            
            if 'intelligent_chunks' in pdf_content.metadata:
                # Use intelligent chunks with full metadata
                chunks_data = pdf_content.metadata['intelligent_chunks']
                logger.info(f"Using {len(chunks_data)} intelligent chunks")
            else:
                # Fallback to simple chunks
                for i, chunk_text in enumerate(pdf_content.chunks):
                    chunk_data = {
                        'chunk_id': f"simple_{i}",
                        'text': chunk_text,
                        'metadata': {
                            'chunk_index': i,
                            'source_file': pdf_content.filename,
                            'chunk_type': 'simple',
                            'importance_score': 0.5
                        }
                    }
                    chunks_data.append(chunk_data)
                logger.info(f"Using {len(chunks_data)} simple chunks")
            
            # Store chunks in vector database
            success = await self.store_chunks(chunks_data, collection_name)
            
            if success:
                logger.info(f"Successfully processed PDF: {pdf_content.filename}")
            else:
                logger.error(f"Failed to process PDF: {pdf_content.filename}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing PDF content: {e}")
            return False
    
    def get_retriever(self, collection_name: str = None):
        """
        Get a retriever interface compatible with existing code
        
        Args:
            collection_name: Collection to use for retrieval
            
        Returns:
            Retriever object with get_relevant_documents method
        """
        return ChromaRetriever(self, collection_name)

    async def build_pipeline(self, pdf_content: PDFContent) -> Any:
        """
        Build the RAG pipeline from PDF content (compatibility method)
        
        Args:
            pdf_content: Processed PDF content
            
        Returns:
            Retriever object for querying
        """
        # Process and store the PDF content
        success = await self.process_pdf_content(pdf_content)
        
        if success:
            # Return a compatible retriever
            return self.get_retriever()
        else:
            raise ValueError("Failed to build RAG pipeline from PDF content")

class ChromaRetriever:
    """Retriever interface for compatibility with existing code"""
    
    def __init__(self, rag_pipeline: RAGPipeline, collection_name: str = None):
        self.rag_pipeline = rag_pipeline
        self.collection_name = collection_name
    
    async def get_relevant_documents(self, query: str, top_k: int = 5) -> List[Any]:
        """Get relevant documents for a query"""
        
        # Retrieve chunks using RAG pipeline
        result = await self.rag_pipeline.retrieve_chunks(
            query=query,
            top_k=top_k,
            collection_name=self.collection_name
        )
        
        # Convert to document-like objects
        documents = []
        for chunk in result.get('retrieved_chunks', []):
            # Create simple document object
            doc = SimpleDocument(
                page_content=chunk.get('text', ''),
                metadata=chunk.get('metadata', {}),
                score=chunk.get('similarity_score', 0.0)
            )
            documents.append(doc)
        
        return documents

class SimpleDocument:
    """Simple document class for compatibility"""
    
    def __init__(self, page_content: str, metadata: Dict, score: float = 0.0):
        self.page_content = page_content
        self.metadata = metadata
        self.score = score
