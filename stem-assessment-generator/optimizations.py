"""
Performance optimizations and caching for STEM Assessment Generator
"""
import os
import json
import time
import hashlib
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pickle
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import gc

from config import settings

logger = logging.getLogger(__name__)

# =============================================================================
# Intelligent Caching System
# =============================================================================

class QuestionCache:
    """Intelligent caching system for questions and embeddings"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(settings.PROJECT_ROOT) / cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache subdirectories
        self.question_cache_dir = self.cache_dir / "questions"
        self.embedding_cache_dir = self.cache_dir / "embeddings"
        self.pdf_cache_dir = self.cache_dir / "pdfs"
        
        for cache_subdir in [self.question_cache_dir, self.embedding_cache_dir, self.pdf_cache_dir]:
            cache_subdir.mkdir(exist_ok=True)
        
        # Cache statistics
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "saves": 0,
            "evictions": 0
        }
        
        # TTL settings (in seconds)
        self.question_ttl = 86400 * 7  # 7 days
        self.embedding_ttl = 86400 * 30  # 30 days
        self.pdf_ttl = 86400 * 14  # 14 days
        
        logger.info(f"✓ Question cache initialized at {self.cache_dir}")

    def get_content_hash(self, content: str, difficulty: str = "", num_questions: int = 0) -> str:
        """Generate hash for content-based caching"""
        combined = f"{content}_{difficulty}_{num_questions}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def get_cached_questions(self, content_hash: str, num_needed: int) -> Optional[List[Dict]]:
        """Retrieve cached questions if available"""
        try:
            cache_file = self.question_cache_dir / f"{content_hash}.json"
            
            if not cache_file.exists():
                self.cache_stats["misses"] += 1
                return None
            
            # Check TTL
            if time.time() - cache_file.stat().st_mtime > self.question_ttl:
                cache_file.unlink()  # Remove expired cache
                self.cache_stats["evictions"] += 1
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            questions = cached_data.get('questions', [])
            
            # Return subset if we have enough questions
            if len(questions) >= num_needed:
                self.cache_stats["hits"] += 1
                logger.info(f"Cache hit: Retrieved {num_needed} questions from cache")
                return questions[:num_needed]
            
            self.cache_stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            self.cache_stats["misses"] += 1
            return None

    def cache_questions(self, content_hash: str, questions: List[Dict], metadata: Dict = None):
        """Cache generated questions"""
        try:
            cache_file = self.question_cache_dir / f"{content_hash}.json"
            
            cache_data = {
                'questions': questions,
                'metadata': metadata or {},
                'cached_at': datetime.now().isoformat(),
                'cache_version': '1.0'
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            self.cache_stats["saves"] += 1
            logger.info(f"Cached {len(questions)} questions with hash {content_hash}")
            
        except Exception as e:
            logger.error(f"Failed to cache questions: {e}")

    def get_cached_embeddings(self, text_hash: str) -> Optional[List[float]]:
        """Retrieve cached embeddings"""
        try:
            cache_file = self.embedding_cache_dir / f"{text_hash}.pkl"
            
            if not cache_file.exists():
                return None
            
            # Check TTL
            if time.time() - cache_file.stat().st_mtime > self.embedding_ttl:
                cache_file.unlink()
                return None
            
            with open(cache_file, 'rb') as f:
                embeddings = pickle.load(f)
            
            logger.debug(f"Retrieved cached embeddings for hash {text_hash}")
            return embeddings
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached embeddings: {e}")
            return None

    def cache_embeddings(self, text_hash: str, embeddings: List[float]):
        """Cache generated embeddings"""
        try:
            cache_file = self.embedding_cache_dir / f"{text_hash}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
            
            logger.debug(f"Cached embeddings for hash {text_hash}")
            
        except Exception as e:
            logger.error(f"Failed to cache embeddings: {e}")

    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests,
            "cache_size_mb": self.get_cache_size_mb()
        }

    def get_cache_size_mb(self) -> float:
        """Calculate total cache size in MB"""
        total_size = 0
        for cache_dir in [self.question_cache_dir, self.embedding_cache_dir, self.pdf_cache_dir]:
            for file_path in cache_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        return round(total_size / (1024 * 1024), 2)

    def cleanup_expired_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_count = 0
        
        cache_configs = [
            (self.question_cache_dir, self.question_ttl),
            (self.embedding_cache_dir, self.embedding_ttl),
            (self.pdf_cache_dir, self.pdf_ttl)
        ]
        
        for cache_dir, ttl in cache_configs:
            for file_path in cache_dir.iterdir():
                if file_path.is_file():
                    if current_time - file_path.stat().st_mtime > ttl:
                        file_path.unlink()
                        expired_count += 1
        
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired cache entries")
            self.cache_stats["evictions"] += expired_count

# =============================================================================
# Batch Processing Optimization
# =============================================================================

class BatchProcessor:
    """Optimize batch processing of PDFs and questions"""
    
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_queue = asyncio.Queue()
        self.active_tasks = {}
        
    async def process_pdf_batch(self, pdf_files: List[Path]) -> Dict[str, Any]:
        """Process multiple PDFs in parallel"""
        tasks = []
        
        for pdf_file in pdf_files:
            task = asyncio.create_task(self.process_single_pdf(pdf_file))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate successful results from exceptions
        successful = []
        failed = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed.append({"file": pdf_files[i], "error": str(result)})
            else:
                successful.append(result)
        
        return {
            "successful": successful,
            "failed": failed,
            "total_processed": len(successful)
        }

    async def process_single_pdf(self, pdf_file: Path) -> Dict[str, Any]:
        """Process a single PDF with optimizations"""
        # This would integrate with the existing PDF processor
        # Implementation would depend on the specific PDF processor used
        pass

    def optimize_chunk_processing(self, chunks: List[Dict]) -> List[Dict]:
        """Optimize chunk processing order and batching"""
        # Sort chunks by importance score (if available)
        scored_chunks = []
        unscored_chunks = []
        
        for chunk in chunks:
            if 'importance_score' in chunk:
                scored_chunks.append(chunk)
            else:
                unscored_chunks.append(chunk)
        
        # Sort by importance (highest first)
        scored_chunks.sort(key=lambda x: x.get('importance_score', 0), reverse=True)
        
        # Combine with scored chunks first
        return scored_chunks + unscored_chunks

# =============================================================================
# Memory Management
# =============================================================================

class MemoryManager:
    """Manage memory usage and cleanup"""
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_mb = max_memory_mb
        self.temp_files = set()
        
    def register_temp_file(self, file_path: Path):
        """Register a temporary file for cleanup"""
        self.temp_files.add(file_path)

    def cleanup_temp_files(self):
        """Remove all registered temporary files"""
        cleaned_count = 0
        for file_path in list(self.temp_files):
            try:
                if file_path.exists():
                    file_path.unlink()
                    cleaned_count += 1
                self.temp_files.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} temporary files")

    def force_garbage_collection(self):
        """Force garbage collection to free memory"""
        collected = gc.collect()
        logger.debug(f"Garbage collection freed {collected} objects")

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage (simplified)"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            # Fallback if psutil not available
            return 0.0

    def check_memory_limit(self) -> bool:
        """Check if memory usage is within limits"""
        current_usage = self.get_memory_usage_mb()
        return current_usage < self.max_memory_mb

# =============================================================================
# API Rate Limit Management
# =============================================================================

class RateLimitManager:
    """Manage API rate limits and backoff strategies"""
    
    def __init__(self):
        self.api_calls = {}  # Track calls per endpoint
        self.backoff_delays = {}  # Current backoff delays
        self.rate_limits = {
            'openai_embeddings': {'per_minute': 60, 'per_hour': 3000},
            'openai_completions': {'per_minute': 20, 'per_hour': 500}
        }
        
    async def wait_if_needed(self, endpoint: str):
        """Wait if rate limit is approaching"""
        if endpoint not in self.api_calls:
            self.api_calls[endpoint] = []
        
        now = time.time()
        
        # Remove old entries (older than 1 hour)
        self.api_calls[endpoint] = [
            call_time for call_time in self.api_calls[endpoint] 
            if now - call_time < 3600
        ]
        
        # Check if we need to wait
        recent_calls = len([
            call_time for call_time in self.api_calls[endpoint] 
            if now - call_time < 60
        ])
        
        if endpoint in self.rate_limits:
            limit = self.rate_limits[endpoint]['per_minute']
            if recent_calls >= limit:
                wait_time = 60 - (now - max(self.api_calls[endpoint]))
                logger.info(f"Rate limit approached for {endpoint}, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        # Record this API call
        self.api_calls[endpoint].append(now)

    def get_estimated_completion_time(self, endpoint: str, remaining_calls: int) -> float:
        """Estimate completion time based on rate limits"""
        if endpoint not in self.rate_limits:
            return 0.0
        
        calls_per_minute = self.rate_limits[endpoint]['per_minute']
        estimated_minutes = remaining_calls / calls_per_minute
        return estimated_minutes * 60  # Return in seconds

# =============================================================================
# Database Optimization (SQLite)
# =============================================================================

class DatabaseOptimizer:
    """Optimize database operations"""
    
    def __init__(self, db_path: str = "cache/optimization.db"):
        self.db_path = Path(settings.PROJECT_ROOT) / db_path
        self.db_path.parent.mkdir(exist_ok=True)
        self.init_database()

    def init_database(self):
        """Initialize optimization database with proper indexes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS generation_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT NOT NULL,
                num_questions INTEGER,
                difficulty TEXT,
                generation_time REAL,
                success_rate REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_type TEXT NOT NULL,
                operation TEXT NOT NULL,
                hit_rate REAL,
                size_mb REAL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_content_hash ON generation_stats(content_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON generation_stats(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_type ON cache_usage(cache_type)')
        
        conn.commit()
        conn.close()
        
        logger.info(f"✓ Database optimizer initialized at {self.db_path}")

    def record_generation_stats(self, content_hash: str, num_questions: int, 
                              difficulty: str, generation_time: float, success_rate: float):
        """Record generation statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO generation_stats 
            (content_hash, num_questions, difficulty, generation_time, success_rate)
            VALUES (?, ?, ?, ?, ?)
        ''', (content_hash, num_questions, difficulty, generation_time, success_rate))
        
        conn.commit()
        conn.close()

    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get performance analytics from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Average generation time by difficulty
        cursor.execute('''
            SELECT difficulty, AVG(generation_time), COUNT(*) 
            FROM generation_stats 
            GROUP BY difficulty
        ''')
        difficulty_stats = {row[0]: {"avg_time": row[1], "count": row[2]} 
                          for row in cursor.fetchall()}
        
        # Recent performance trends
        cursor.execute('''
            SELECT DATE(created_at), AVG(generation_time), AVG(success_rate)
            FROM generation_stats 
            WHERE created_at >= datetime('now', '-7 days')
            GROUP BY DATE(created_at)
            ORDER BY DATE(created_at)
        ''')
        recent_trends = [{"date": row[0], "avg_time": row[1], "avg_success": row[2]} 
                        for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "difficulty_performance": difficulty_stats,
            "recent_trends": recent_trends
        }

    def vacuum_database(self):
        """Optimize database by running VACUUM"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('VACUUM')
        conn.close()
        logger.info("Database vacuum completed")

# =============================================================================
# Main Optimization Manager
# =============================================================================

class OptimizationManager:
    """Central optimization manager"""
    
    def __init__(self):
        self.question_cache = QuestionCache()
        self.batch_processor = BatchProcessor()
        self.memory_manager = MemoryManager()
        self.rate_limit_manager = RateLimitManager()
        self.db_optimizer = DatabaseOptimizer()
        
        # Background cleanup task will be started when needed
        self._cleanup_task = None
        
        logger.info("✓ Optimization manager initialized")

    async def startup(self):
        """Start background tasks when the app starts"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self.background_cleanup())
            logger.info("✓ Background cleanup task started")

    async def shutdown(self):
        """Stop background tasks when the app shuts down"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("✓ Background cleanup task stopped")

    async def background_cleanup(self):
        """Background task for periodic cleanup"""
        while True:
            try:
                # Run cleanup every hour
                await asyncio.sleep(3600)
                
                self.question_cache.cleanup_expired_cache()
                self.memory_manager.cleanup_temp_files()
                self.memory_manager.force_garbage_collection()
                
                # Run database vacuum weekly
                if datetime.now().weekday() == 0:  # Monday
                    self.db_optimizer.vacuum_database()
                
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status"""
        return {
            "cache_stats": self.question_cache.get_cache_stats(),
            "memory_usage_mb": self.memory_manager.get_memory_usage_mb(),
            "performance_analytics": self.db_optimizer.get_performance_analytics(),
            "active_batch_tasks": len(self.batch_processor.active_tasks),
            "optimization_recommendations": self.get_optimization_recommendations()
        }

    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current state"""
        recommendations = []
        
        cache_stats = self.question_cache.get_cache_stats()
        if cache_stats["hit_rate_percent"] < 20:
            recommendations.append("Consider increasing cache TTL or improving content similarity detection")
        
        if self.memory_manager.get_memory_usage_mb() > 400:
            recommendations.append("High memory usage detected - consider reducing batch sizes")
        
        if cache_stats["cache_size_mb"] > 100:
            recommendations.append("Cache size is large - consider running cleanup")
        
        return recommendations

# Global optimization manager instance
optimization_manager = OptimizationManager()
