"""
RAG (Retrieval Augmented Generation) pipeline for STEM Assessment Generator
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import os

from config import settings
from models import PDFContent, EmbeddingResult

class RAGPipeline:
    """Handles embeddings, vector storage, and retrieval"""
    
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize OpenAI embeddings"""
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_EMBEDDING_MODEL
        )
    
    async def build_pipeline(self, pdf_content: PDFContent) -> Any:
        """
        Build the RAG pipeline from PDF content
        
        Args:
            pdf_content: Processed PDF content
            
        Returns:
            Retriever object for querying
        """
        # Create vector store
        self.vector_store = await self._create_vector_store(pdf_content.chunks)
        
        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        return self.retriever
    
    async def _create_vector_store(self, chunks: List[str]) -> Chroma:
        """
        Create ChromaDB vector store from text chunks
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Chroma vector store
        """
        # Ensure vector DB directory exists
        os.makedirs(settings.VECTOR_DB_DIR, exist_ok=True)
        
        # Initialize Chroma client
        client = chromadb.PersistentClient(
            path=settings.VECTOR_DB_DIR,
            settings=ChromaSettings(allow_reset=True)
        )
        
        # Create vector store
        vector_store = Chroma(
            client=client,
            collection_name="stem_content",
            embedding_function=self.embeddings,
            persist_directory=settings.VECTOR_DB_DIR
        )
        
        # Add documents to vector store
        if chunks:
            vector_store.add_texts(
                texts=chunks,
                metadatas=[{"chunk_id": i} for i in range(len(chunks))]
            )
        
        return vector_store
    
    async def retrieve_relevant_content(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant content for a query
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents with metadata
        """
        if not self.retriever:
            raise ValueError("RAG pipeline not initialized. Call build_pipeline first.")
        
        docs = self.retriever.get_relevant_documents(query)
        
        results = []
        for doc in docs[:k]:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": getattr(doc, "score", None)
            })
        
        return results
    
    async def similarity_search(self, query: str, k: int = 5) -> List[str]:
        """
        Perform similarity search on vector store
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar text chunks
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized.")
        
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    def reset_vector_store(self):
        """Reset the vector store (useful for testing)"""
        if self.vector_store:
            self.vector_store.delete_collection()
            self.vector_store = None
            self.retriever = None
