# services/vector_service.py - Production Vector Service
"""
Production Vector Service with Pinecone and OpenAI Embeddings
Handles semantic search, conversation memory, and research indexing
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

import openai
import pinecone
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import settings


class VectorNamespace(Enum):
    """Vector database namespaces for different data types"""
    CONVERSATIONS = "conversations"
    RESEARCH_REPORTS = "research"
    USER_MEMORIES = "memories"
    TRADE_INSIGHTS = "trades"
    MARKET_ANALYSIS = "analysis"


@dataclass
class VectorDocument:
    """Structured vector document for storage"""
    id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]
    namespace: VectorNamespace
    created_at: datetime


class VectorService:
    """
    Production Vector Service for semantic search and memory
    
    Features:
    - OpenAI text-embedding-3-small for cost-effective embeddings
    - Pinecone for scalable vector storage
    - Automatic retry logic with exponential backoff
    - Batch processing for efficiency
    - Comprehensive error handling and logging
    - Security controls and rate limiting
    """
    
    def __init__(self, base_service):
        self.base_service = base_service
        self.openai_client = None
        self.pinecone_index = None
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimensions = 1536
        self.batch_size = 100
        self.max_text_length = 8000  # Prevent token limit issues
        
        # Rate limiting
        self.last_embedding_time = 0
        self.embedding_rate_limit = 0.1  # 10 requests per second max
        
        logger.info("🔗 VectorService initialized")

    async def initialize(self):
        """Initialize OpenAI and Pinecone connections"""
        try:
            # Initialize OpenAI client
            self.openai_client = openai.AsyncOpenAI(
                api_key=settings.openai_api_key,
                timeout=30.0,
                max_retries=3
            )
            
            # Initialize Pinecone
            pinecone.init(
                api_key=settings.pinecone_api_key,
                environment=settings.pinecone_environment
            )
            
            # Connect to index (create if doesn't exist)
            await self._ensure_index_exists()
            self.pinecone_index = pinecone.Index(settings.pinecone_index_name)
            
            # Test connections
            await self._test_connections()
            
            logger.info("✅ VectorService connections established")
            
        except Exception as e:
            logger.exception(f"❌ VectorService initialization failed: {e}")
            raise

    async def _ensure_index_exists(self):
        """Ensure Pinecone index exists with correct configuration"""
        try:
            existing_indexes = pinecone.list_indexes()
            
            if settings.pinecone_index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {settings.pinecone_index_name}")
                
                pinecone.create_index(
                    name=settings.pinecone_index_name,
                    dimension=self.embedding_dimensions,
                    metric="cosine",
                    pods=1,
                    replicas=1,
                    pod_type="p1.x1"
                )
                
                # Wait for index to be ready
                await asyncio.sleep(10)
                
                logger.info("✅ Pinecone index created successfully")
            else:
                logger.info(f"✅ Pinecone index {settings.pinecone_index_name} already exists")
                
        except Exception as e:
            logger.exception(f"❌ Error ensuring Pinecone index exists: {e}")
            raise

    async def _test_connections(self):
        """Test OpenAI and Pinecone connections"""
        try:
            # Test OpenAI
            test_embedding = await self._generate_embedding("test connection")
            if not test_embedding or len(test_embedding) != self.embedding_dimensions:
                raise Exception("OpenAI embedding test failed")
            
            # Test Pinecone
            stats = self.pinecone_index.describe_index_stats()
            logger.info(f"Pinecone index stats: {stats}")
            
            logger.info("✅ Vector service connections tested successfully")
            
        except Exception as e:
            logger.exception(f"❌ Vector service connection test failed: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError))
    )
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding with retry logic and rate limiting"""
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_embedding_time
            if time_since_last < self.embedding_rate_limit:
                await asyncio.sleep(self.embedding_rate_limit - time_since_last)
            
            self.last_embedding_time = time.time()
            
            # Truncate text if too long
            if len(text) > self.max_text_length:
                text = text[:self.max_text_length]
                logger.warning(f"Text truncated to {self.max_text_length} characters")
            
            # Generate embedding
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            
            if len(embedding) != self.embedding_dimensions:
                raise ValueError(f"Unexpected embedding dimensions: {len(embedding)}")
            
            return embedding
            
        except openai.RateLimitError as e:
            logger.warning(f"OpenAI rate limit hit, retrying: {e}")
            raise
        except openai.APITimeoutError as e:
            logger.warning(f"OpenAI timeout, retrying: {e}")
            raise
        except Exception as e:
            logger.exception(f"❌ Error generating embedding: {e}")
            raise

    async def _generate_document_id(self, text: str, user_id: str = None) -> str:
        """Generate deterministic document ID"""
        content = f"{user_id or 'global'}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def health_check(self) -> Dict[str, Any]:
        """Health check for vector service"""
        health = {"status": "healthy", "components": {}}
        
        try:
            # Test OpenAI
            test_embedding = await self._generate_embedding("health check")
            health["components"]["openai"] = {
                "status": "healthy",
                "embedding_dimensions": len(test_embedding)
            }
        except Exception as e:
            health["components"]["openai"] = {"status": "unhealthy", "error": str(e)}
            health["status"] = "degraded"
        
        try:
            # Test Pinecone
            stats = self.pinecone_index.describe_index_stats()
            health["components"]["pinecone"] = {
                "status": "healthy",
                "total_vectors": stats.get("total_vector_count", 0)
            }
        except Exception as e:
            health["components"]["pinecone"] = {"status": "unhealthy", "error": str(e)}
            health["status"] = "degraded"
        
        return health

    async def upsert_vector(self, namespace: str, doc_id: str, 
                           embedding: List[float], metadata: Dict) -> bool:
        """Store vector in Pinecone with metadata"""
        try:
            # Validate inputs
            if not embedding or len(embedding) != self.embedding_dimensions:
                raise ValueError(f"Invalid embedding dimensions: {len(embedding) if embedding else 0}")
            
            if not metadata:
                metadata = {}
            
            # Add system metadata
            metadata.update({
                "created_at": datetime.utcnow().isoformat(),
                "service_version": "1.0"
            })
            
            # Upsert to Pinecone
            vectors = [(doc_id, embedding, metadata)]
            
            upsert_response = self.pinecone_index.upsert(
                vectors=vectors,
                namespace=namespace
            )
            
            if upsert_response.get("upserted_count", 0) > 0:
                logger.debug(f"✅ Vector upserted: {namespace}:{doc_id}")
                return True
            else:
                logger.error(f"❌ Vector upsert failed: {upsert_response}")
                return False
                
        except Exception as e:
            logger.exception(f"❌ Error upserting vector {namespace}:{doc_id}: {e}")
            return False

    async def upsert_text(self, namespace: str, text: str, metadata: Dict,
                         user_id: str = None, doc_id: str = None) -> Optional[str]:
        """Generate embedding and upsert text document"""
        try:
            # Generate embedding
            embedding = await self._generate_embedding(text)
            if not embedding:
                return None
            
            # Generate document ID if not provided
            if not doc_id:
                doc_id = await self._generate_document_id(text, user_id)
            
            # Add text to metadata
            metadata = metadata.copy() if metadata else {}
            metadata.update({
                "text": text[:1000],  # Store truncated text for reference
                "text_length": len(text),
                "user_id": user_id
            })
            
            # Upsert vector
            success = await self.upsert_vector(namespace, doc_id, embedding, metadata)
            
            return doc_id if success else None
            
        except Exception as e:
            logger.exception(f"❌ Error upserting text to {namespace}: {e}")
            return None

    async def query_similar(self, namespace: str, query_embedding: List[float],
                           top_k: int = 5, filter_dict: Dict = None,
                           include_metadata: bool = True) -> List[Dict]:
        """Query similar vectors from Pinecone"""
        try:
            if not query_embedding or len(query_embedding) != self.embedding_dimensions:
                raise ValueError(f"Invalid query embedding dimensions: {len(query_embedding) if query_embedding else 0}")
            
            if top_k > 100:
                top_k = 100  # Pinecone limit
                logger.warning("top_k limited to 100")
            
            # Query Pinecone
            query_response = self.pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                filter=filter_dict,
                include_metadata=include_metadata
            )
            
            results = []
            for match in query_response.get("matches", []):
                result = {
                    "id": match["id"],
                    "score": match["score"],
                }
                
                if include_metadata and "metadata" in match:
                    result["metadata"] = match["metadata"]
                
                results.append(result)
            
            logger.debug(f"✅ Vector query returned {len(results)} results from {namespace}")
            return results
            
        except Exception as e:
            logger.exception(f"❌ Error querying vectors from {namespace}: {e}")
            return []

    async def query_text(self, namespace: str, query_text: str, top_k: int = 5,
                        filter_dict: Dict = None, user_id: str = None) -> List[Dict]:
        """Query similar documents using text query"""
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query_text)
            if not query_embedding:
                return []
            
            # Add user filter if provided
            if user_id and filter_dict is None:
                filter_dict = {"user_id": user_id}
            elif user_id and filter_dict:
                filter_dict = {**filter_dict, "user_id": user_id}
            
            # Query similar vectors
            results = await self.query_similar(
                namespace, query_embedding, top_k, filter_dict
            )
            
            return results
            
        except Exception as e:
            logger.exception(f"❌ Error querying text from {namespace}: {e}")
            return []

    async def delete_vector(self, namespace: str, doc_id: str) -> bool:
        """Delete vector from Pinecone"""
        try:
            delete_response = self.pinecone_index.delete(
                ids=[doc_id],
                namespace=namespace
            )
            
            logger.debug(f"✅ Vector deleted: {namespace}:{doc_id}")
            return True
            
        except Exception as e:
            logger.exception(f"❌ Error deleting vector {namespace}:{doc_id}: {e}")
            return False

    async def delete_by_filter(self, namespace: str, filter_dict: Dict) -> bool:
        """Delete vectors by metadata filter"""
        try:
            delete_response = self.pinecone_index.delete(
                filter=filter_dict,
                namespace=namespace
            )
            
            logger.info(f"✅ Vectors deleted by filter from {namespace}: {filter_dict}")
            return True
            
        except Exception as e:
            logger.exception(f"❌ Error deleting vectors by filter from {namespace}: {e}")
            return False

    async def delete_user_data(self, user_id: str) -> Dict[str, Any]:
        """Delete all user data from vector database (GDPR compliance)"""
        results = {"deleted": False, "details": {}}
        
        try:
            # Delete from all namespaces
            for namespace in VectorNamespace:
                try:
                    success = await self.delete_by_filter(
                        namespace.value, 
                        {"user_id": user_id}
                    )
                    results["details"][namespace.value] = success
                except Exception as e:
                    logger.error(f"Error deleting user data from {namespace.value}: {e}")
                    results["details"][namespace.value] = False
            
            # Check if all deletions succeeded
            results["deleted"] = all(results["details"].values())
            
            logger.info(f"User data deletion for {user_id}: {results}")
            return results
            
        except Exception as e:
            logger.exception(f"❌ Error deleting user data for {user_id}: {e}")
            results["error"] = str(e)
            return results

    async def get_namespace_stats(self, namespace: str) -> Dict[str, Any]:
        """Get statistics for a specific namespace"""
        try:
            # Get overall index stats
            stats = self.pinecone_index.describe_index_stats()
            
            # Extract namespace-specific stats if available
            namespace_stats = stats.get("namespaces", {}).get(namespace, {})
            
            return {
                "namespace": namespace,
                "vector_count": namespace_stats.get("vector_count", 0),
                "total_index_vectors": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension", self.embedding_dimensions)
            }
            
        except Exception as e:
            logger.exception(f"❌ Error getting namespace stats for {namespace}: {e}")
            return {"namespace": namespace, "error": str(e)}

    async def batch_upsert_texts(self, namespace: str, documents: List[Dict],
                                user_id: str = None) -> Dict[str, Any]:
        """Batch upsert multiple text documents for efficiency"""
        results = {"successful": 0, "failed": 0, "errors": []}
        
        try:
            # Process in batches to avoid memory issues
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i + self.batch_size]
                
                # Generate embeddings for batch
                texts = [doc.get("text", "") for doc in batch]
                embeddings = []
                
                for text in texts:
                    try:
                        embedding = await self._generate_embedding(text)
                        embeddings.append(embedding)
                    except Exception as e:
                        logger.error(f"Error generating embedding for batch item: {e}")
                        embeddings.append(None)
                
                # Prepare vectors for upsert
                vectors = []
                for j, doc in enumerate(batch):
                    if embeddings[j] is None:
                        results["failed"] += 1
                        results["errors"].append(f"Failed to generate embedding for doc {i+j}")
                        continue
                    
                    doc_id = doc.get("id") or await self._generate_document_id(
                        doc.get("text", ""), user_id
                    )
                    
                    metadata = doc.get("metadata", {}).copy()
                    metadata.update({
                        "text": doc.get("text", "")[:1000],
                        "text_length": len(doc.get("text", "")),
                        "user_id": user_id,
                        "created_at": datetime.utcnow().isoformat()
                    })
                    
                    vectors.append((doc_id, embeddings[j], metadata))
                
                # Upsert batch
                if vectors:
                    try:
                        upsert_response = self.pinecone_index.upsert(
                            vectors=vectors,
                            namespace=namespace
                        )
                        
                        successful_count = upsert_response.get("upserted_count", 0)
                        results["successful"] += successful_count
                        
                        if successful_count != len(vectors):
                            failed_count = len(vectors) - successful_count
                            results["failed"] += failed_count
                            results["errors"].append(f"Batch {i//self.batch_size}: {failed_count} upserts failed")
                        
                    except Exception as e:
                        results["failed"] += len(vectors)
                        results["errors"].append(f"Batch {i//self.batch_size} upsert failed: {str(e)}")
                        logger.exception(f"Batch upsert failed: {e}")
            
            logger.info(f"Batch upsert completed: {results}")
            return results
            
        except Exception as e:
            logger.exception(f"❌ Error in batch upsert: {e}")
            results["errors"].append(f"Batch operation failed: {str(e)}")
            return results

    # Memory-specific convenience methods
    async def save_conversation_memory(self, user_id: str, conversation_text: str,
                                     summary: str, topics: List[str]) -> Optional[str]:
        """Save conversation memory with structured metadata"""
        metadata = {
            "type": "conversation",
            "summary": summary,
            "topics": topics,
            "conversation_length": len(conversation_text)
        }
        
        return await self.upsert_text(
            VectorNamespace.CONVERSATIONS.value,
            conversation_text,
            metadata,
            user_id
        )

    async def save_research_report(self, user_id: str, symbol: str, report_text: str,
                                 report_type: str, analysis_date: datetime) -> Optional[str]:
        """Save research report with structured metadata"""
        metadata = {
            "type": "research_report",
            "symbol": symbol.upper(),
            "report_type": report_type,
            "analysis_date": analysis_date.isoformat(),
            "report_length": len(report_text)
        }
        
        return await self.upsert_text(
            VectorNamespace.RESEARCH_REPORTS.value,
            report_text,
            metadata,
            user_id
        )

    async def search_user_memories(self, user_id: str, query: str, 
                                 memory_type: str = None, top_k: int = 5) -> List[Dict]:
        """Search user's memories with optional type filtering"""
        filter_dict = {"user_id": user_id}
        if memory_type:
            filter_dict["type"] = memory_type
        
        return await self.query_text(
            VectorNamespace.USER_MEMORIES.value,
            query,
            top_k,
            filter_dict
        )

    async def search_research_by_symbol(self, user_id: str, symbol: str,
                                      top_k: int = 5) -> List[Dict]:
        """Search research reports for specific symbol"""
        filter_dict = {
            "user_id": user_id,
            "symbol": symbol.upper(),
            "type": "research_report"
        }
        
        return await self.query_similar(
            VectorNamespace.RESEARCH_REPORTS.value,
            await self._generate_embedding(f"research analysis {symbol}"),
            top_k,
            filter_dict
        )
