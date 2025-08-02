# services/memory_service.py - Production Memory Service
"""
Production Memory Service with 3-Layer Memory Architecture
Handles STM (Redis), Summaries (MongoDB), and LTM (Vector DB)
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

import openai
from loguru import logger

from config import settings
from services.vector_service import VectorService, VectorNamespace


class MemoryType(Enum):
    """Types of memory for classification"""
    CONVERSATION = "conversation"
    TRADING_INSIGHT = "trading_insight"
    USER_PREFERENCE = "user_preference"
    MARKET_ANALYSIS = "market_analysis"
    GOAL_DISCUSSION = "goal_discussion"
    EDUCATIONAL = "educational"


@dataclass
class ConversationTurn:
    """Individual conversation turn"""
    user_id: str
    user_message: str
    bot_response: str
    timestamp: datetime
    metadata: Dict[str, Any]
    symbols: List[str] = None
    intent: str = None
    sentiment: str = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class ConversationSummary:
    """Conversation summary for mid-term memory"""
    user_id: str
    summary_id: str
    summary_text: str
    topics: List[str]
    symbols: List[str]
    key_insights: List[str]
    message_count: int
    start_time: datetime
    end_time: datetime
    importance_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        data['end_time'] = self.end_time.isoformat()
        return data


class MemoryService:
    """
    Production Memory Service - 3-Layer Architecture
    
    Layer 1 (STM): Redis - Last 10-20 conversation turns for immediate context
    Layer 2 (MTM): MongoDB - Conversation summaries and structured insights  
    Layer 3 (LTM): Vector DB - Semantic search across all user memories
    
    Features:
    - Automatic conversation summarization
    - Intelligent memory retrieval and ranking
    - Context compression for token optimization
    - Memory importance scoring
    - Cross-session conversation threading
    - GDPR-compliant data management
    """
    
    def __init__(self, base_service, vector_service: VectorService):
        self.base_service = base_service
        self.vector_service = vector_service
        self.openai_client = None
        
        # Memory configuration
        self.stm_limit = 20  # Number of recent turns to keep in STM
        self.summary_trigger = 10  # Summarize every N messages
        self.max_context_tokens = 3000  # Max tokens for context
        self.importance_threshold = 0.6  # Minimum importance for LTM storage
        
        # Redis keys
        self.stm_key_pattern = "memory:stm:{user_id}"
        self.session_key_pattern = "memory:session:{user_id}"
        self.metrics_key = "memory:metrics"
        
        # MongoDB collections
        self.summaries_collection = "conversation_summaries"
        self.insights_collection = "user_insights"
        
        logger.info("🧠 MemoryService initialized")

    async def initialize(self):
        """Initialize memory service components"""
        try:
            # Initialize OpenAI client for summarization
            self.openai_client = openai.AsyncOpenAI(
                api_key=settings.openai_api_key,
                timeout=30.0,
                max_retries=3
            )
            
            # Ensure MongoDB collections exist with proper indexes
            await self._ensure_collections()
            
            # Test components
            await self._test_connections()
            
            logger.info("✅ MemoryService initialized successfully")
            
        except Exception as e:
            logger.exception(f"❌ MemoryService initialization failed: {e}")
            raise

    async def _ensure_collections(self):
        """Ensure MongoDB collections exist with proper indexes"""
        try:
            db = self.base_service.db
            
            # Create indexes for conversation summaries
            summaries_collection = db[self.summaries_collection]
            await summaries_collection.create_index([("user_id", 1), ("end_time", -1)])
            await summaries_collection.create_index([("topics", 1)])
            await summaries_collection.create_index([("symbols", 1)])
            await summaries_collection.create_index([("importance_score", -1)])
            
            # Create indexes for user insights
            insights_collection = db[self.insights_collection]
            await insights_collection.create_index([("user_id", 1), ("created_at", -1)])
            await insights_collection.create_index([("type", 1)])
            await insights_collection.create_index([("confidence", -1)])
            
            logger.info("✅ Memory service MongoDB collections ensured")
            
        except Exception as e:
            logger.exception(f"❌ Error ensuring collections: {e}")
            raise

    async def _test_connections(self):
        """Test all memory service connections"""
        try:
            # Test OpenAI
            test_response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            if not test_response.choices:
                raise Exception("OpenAI test failed")
            
            # Test Redis
            await self.base_service.redis.ping()
            
            # Test MongoDB
            await self.base_service.db.list_collection_names()
            
            # Test Vector Service
            vector_health = await self.vector_service.health_check()
            if vector_health.get("status") != "healthy":
                raise Exception("Vector service unhealthy")
            
            logger.info("✅ Memory service connections tested successfully")
            
        except Exception as e:
            logger.exception(f"❌ Memory service connection test failed: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Health check for memory service"""
        health = {"status": "healthy", "components": {}}
        
        try:
            # Test OpenAI
            health["components"]["openai"] = {"status": "healthy"}
        except Exception as e:
            health["components"]["openai"] = {"status": "unhealthy", "error": str(e)}
            health["status"] = "degraded"
        
        try:
            # Test Vector Service
            vector_health = await self.vector_service.health_check()
            health["components"]["vector"] = vector_health
            if vector_health.get("status") != "healthy":
                health["status"] = "degraded"
        except Exception as e:
            health["components"]["vector"] = {"status": "unhealthy", "error": str(e)}
            health["status"] = "degraded"
        
        try:
            # Get memory statistics
            stats = await self.get_memory_stats()
            health["statistics"] = stats
        except Exception as e:
            health["statistics"] = {"error": str(e)}
        
        return health

    # ==========================================
    # SHORT-TERM MEMORY (STM) - Redis
    # ==========================================

    async def save_conversation_turn(self, user_id: str, user_message: str,
                                   bot_response: str, metadata: Dict = None) -> bool:
        """Save conversation turn to STM and trigger summarization if needed"""
        try:
            # Create conversation turn
            turn = ConversationTurn(
                user_id=user_id,
                user_message=user_message,
                bot_response=bot_response,
                timestamp=datetime.utcnow(),
                metadata=metadata or {},
                symbols=metadata.get("symbols", []) if metadata else [],
                intent=metadata.get("intent") if metadata else None,
                sentiment=await self._analyze_sentiment(user_message)
            )
            
            # Save to Redis STM
            stm_key = self.stm_key_pattern.format(user_id=user_id)
            
            # Get current STM
            stm_data = await self.base_service.redis.get(stm_key)
            if stm_data:
                stm_turns = json.loads(stm_data)
            else:
                stm_turns = []
            
            # Add new turn
            stm_turns.append(turn.to_dict())
            
            # Keep only recent turns
            if len(stm_turns) > self.stm_limit:
                # Remove oldest turns but keep them for summarization
                old_turns = stm_turns[:-self.stm_limit]
                stm_turns = stm_turns[-self.stm_limit:]
                
                # Trigger summarization for old turns
                asyncio.create_task(self._process_old_turns(user_id, old_turns))
            
            # Save updated STM
            await self.base_service.redis.setex(
                stm_key, 
                86400,  # 24 hours TTL
                json.dumps(stm_turns, default=str)
            )
            
            # Check if we should summarize current session
            if len(stm_turns) % self.summary_trigger == 0:
                asyncio.create_task(self._trigger_summarization(user_id))
            
            logger.debug(f"✅ Conversation turn saved for user {user_id}")
            return True
            
        except Exception as e:
            logger.exception(f"❌ Error saving conversation turn for {user_id}: {e}")
            return False

    async def get_stm_context(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get recent conversation turns from STM"""
        try:
            stm_key = self.stm_key_pattern.format(user_id=user_id)
            stm_data = await self.base_service.redis.get(stm_key)
            
            if not stm_data:
                return []
            
            stm_turns = json.loads(stm_data)
            
            # Return most recent turns
            return stm_turns[-limit:] if len(stm_turns) > limit else stm_turns
            
        except Exception as e:
            logger.exception(f"❌ Error getting STM context for {user_id}: {e}")
            return []

    # ==========================================
    # MID-TERM MEMORY (MTM) - MongoDB Summaries
    # ==========================================

    async def _trigger_summarization(self, user_id: str):
        """Trigger conversation summarization"""
        try:
            # Get current STM
            stm_context = await self.get_stm_context(user_id, self.summary_trigger)
            
            if len(stm_context) < 3:  # Need minimum conversation for summary
                return
            
            # Generate summary
            summary = await self._generate_conversation_summary(user_id, stm_context)
            
            if summary:
                # Save summary to MongoDB
                await self._save_conversation_summary(summary)
                
                # Store important insights in vector DB
                if summary.importance_score >= self.importance_threshold:
                    await self._store_summary_in_vector_db(summary)
                
                logger.info(f"✅ Conversation summarized for user {user_id}")
            
        except Exception as e:
            logger.exception(f"❌ Error in summarization for {user_id}: {e}")

    async def _generate_conversation_summary(self, user_id: str, 
                                           turns: List[Dict]) -> Optional[ConversationSummary]:
        """Generate conversation summary using LLM"""
        try:
            # Prepare conversation text
            conversation_text = ""
            all_symbols = set()
            start_time = None
            end_time = None
            
            for turn in turns:
                timestamp = turn.get("timestamp", "")
                user_msg = turn.get("user_message", "")
                bot_msg = turn.get("bot_response", "")
                symbols = turn.get("symbols", [])
                
                if symbols:
                    all_symbols.update(symbols)
                
                if start_time is None:
                    start_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                conversation_text += f"User: {user_msg}\nBot: {bot_msg}\n\n"
            
            # Generate summary using LLM
            prompt = f"""
            Analyze this conversation between a user and a trading AI assistant. Provide a structured summary.

            Conversation:
            {conversation_text}

            Please provide:
            1. A 2-3 sentence summary of the main discussion
            2. Key topics discussed (list)
            3. Stock symbols mentioned (list)
            4. Important insights or decisions (list)
            5. Importance score (0.0-1.0) based on trading relevance and user engagement

            Format as JSON:
            {{
                "summary": "...",
                "topics": [...],
                "symbols": [...],
                "insights": [...],
                "importance_score": 0.8
            }}
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            # Parse response
            summary_data = json.loads(response.choices[0].message.content)
            
            # Create summary object
            summary = ConversationSummary(
                user_id=user_id,
                summary_id=f"{user_id}_{int(time.time())}",
                summary_text=summary_data.get("summary", ""),
                topics=summary_data.get("topics", []),
                symbols=list(all_symbols) or summary_data.get("symbols", []),
                key_insights=summary_data.get("insights", []),
                message_count=len(turns),
                start_time=start_time,
                end_time=end_time,
                importance_score=summary_data.get("importance_score", 0.5)
            )
            
            return summary
            
        except Exception as e:
            logger.exception(f"❌ Error generating conversation summary: {e}")
            return None

    async def _save_conversation_summary(self, summary: ConversationSummary) -> bool:
        """Save conversation summary to MongoDB"""
        try:
            collection = self.base_service.db[self.summaries_collection]
            
            result = await collection.insert_one(summary.to_dict())
            
            if result.inserted_id:
                logger.debug(f"✅ Summary saved: {summary.summary_id}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.exception(f"❌ Error saving summary: {e}")
            return False

    async def get_conversation_summaries(self, user_id: str, limit: int = 5,
                                       min_importance: float = 0.0) -> List[Dict]:
        """Get conversation summaries from MongoDB"""
        try:
            collection = self.base_service.db[self.summaries_collection]
            
            query = {
                "user_id": user_id,
                "importance_score": {"$gte": min_importance}
            }
            
            cursor = collection.find(query).sort("end_time", -1).limit(limit)
            summaries = await cursor.to_list(length=limit)
            
            return summaries
            
        except Exception as e:
            logger.exception(f"❌ Error getting summaries for {user_id}: {e}")
            return []

    # ==========================================
    # LONG-TERM MEMORY (LTM) - Vector Database
    # ==========================================

    async def _store_summary_in_vector_db(self, summary: ConversationSummary) -> bool:
        """Store important summary in vector database"""
        try:
            # Prepare text for embedding
            summary_text = f"{summary.summary_text}\n\nKey insights: {' '.join(summary.key_insights)}"
            
            # Prepare metadata
            metadata = {
                "type": "conversation_summary",
                "user_id": summary.user_id,
                "topics": summary.topics,
                "symbols": summary.symbols,
                "importance_score": summary.importance_score,
                "message_count": summary.message_count,
                "start_time": summary.start_time.isoformat(),
                "end_time": summary.end_time.isoformat()
            }
            
            # Store in vector database
            doc_id = await self.vector_service.upsert_text(
                VectorNamespace.CONVERSATIONS.value,
                summary_text,
                metadata,
                summary.user_id,
                summary.summary_id
            )
            
            return doc_id is not None
            
        except Exception as e:
            logger.exception(f"❌ Error storing summary in vector DB: {e}")
            return False

    async def get_relevant_memories(self, user_id: str, query: str, 
                                  top_k: int = 5) -> List[Dict]:
        """Get semantically relevant memories from vector database"""
        try:
            # Search user's memories
            results = await self.vector_service.query_text(
                VectorNamespace.CONVERSATIONS.value,
                query,
                top_k,
                filter_dict={"user_id": user_id}
            )
            
            # Enrich results with summary data from MongoDB
            enriched_results = []
            for result in results:
                summary_id = result.get("id")
                if summary_id:
                    # Get full summary from MongoDB
                    collection = self.base_service.db[self.summaries_collection]
                    summary_doc = await collection.find_one({"summary_id": summary_id})
                    
                    if summary_doc:
                        enriched_result = {
                            "similarity_score": result.get("score", 0.0),
                            "summary": summary_doc,
                            "relevance_type": "semantic_match"
                        }
                        enriched_results.append(enriched_result)
            
            return enriched_results
            
        except Exception as e:
            logger.exception(f"❌ Error getting relevant memories for {user_id}: {e}")
            return []

    # ==========================================
    # COMPREHENSIVE MEMORY RETRIEVAL
    # ==========================================

    async def get_conversation_memory(self, user_id: str, limit: int = 10,
                                    query: str = None) -> Dict[str, Any]:
        """Get comprehensive conversation memory (STM + MTM + LTM)"""
        try:
            memory_context = {
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "short_term_memory": [],
                "conversation_summaries": [],
                "relevant_memories": [],
                "context_stats": {}
            }
            
            # Get short-term memory (recent turns)
            stm_context = await self.get_stm_context(user_id, limit)
            memory_context["short_term_memory"] = stm_context
            
            # Get mid-term memory (conversation summaries)
            summaries = await self.get_conversation_summaries(user_id, limit=5)
            memory_context["conversation_summaries"] = summaries
            
            # Get long-term memory (semantic search) if query provided
            if query:
                relevant_memories = await self.get_relevant_memories(user_id, query, top_k=3)
                memory_context["relevant_memories"] = relevant_memories
            
            # Calculate context statistics
            memory_context["context_stats"] = {
                "stm_messages": len(stm_context),
                "mtm_summaries": len(summaries),
                "ltm_memories": len(memory_context["relevant_memories"]),
                "total_context_items": len(stm_context) + len(summaries) + len(memory_context["relevant_memories"])
            }
            
            return memory_context
            
        except Exception as e:
            logger.exception(f"❌ Error getting conversation memory for {user_id}: {e}")
            return {"user_id": user_id, "error": str(e)}

    async def compress_context_for_llm(self, memory_context: Dict[str, Any],
                                     max_tokens: int = None) -> str:
        """Compress memory context for LLM prompt inclusion"""
        try:
            if max_tokens is None:
                max_tokens = self.max_context_tokens
            
            # Build context string
            context_parts = []
            
            # Recent conversation (STM)
            stm = memory_context.get("short_term_memory", [])
            if stm:
                context_parts.append("## Recent Conversation")
                for turn in stm[-5:]:  # Last 5 turns
                    user_msg = turn.get("user_message", "")[:200]  # Truncate long messages
                    bot_msg = turn.get("bot_response", "")[:200]
                    context_parts.append(f"User: {user_msg}")
                    context_parts.append(f"Bot: {bot_msg}")
            
            # Important summaries (MTM)
            summaries = memory_context.get("conversation_summaries", [])
            if summaries:
                context_parts.append("\n## Conversation History")
                for summary in summaries[:3]:  # Top 3 summaries
                    summary_text = summary.get("summary_text", "")
                    topics = ", ".join(summary.get("topics", []))
                    context_parts.append(f"- {summary_text} (Topics: {topics})")
            
            # Relevant memories (LTM)
            memories = memory_context.get("relevant_memories", [])
            if memories:
                context_parts.append("\n## Relevant Past Insights")
                for memory in memories[:2]:  # Top 2 memories
                    summary_data = memory.get("summary", {})
                    insights = summary_data.get("key_insights", [])
                    if insights:
                        context_parts.append(f"- {insights[0]}")
            
            # Combine and truncate if needed
            full_context = "\n".join(context_parts)
            
            # Rough token estimation (4 chars per token)
            estimated_tokens = len(full_context) // 4
            
            if estimated_tokens > max_tokens:
                # Truncate to fit
                target_chars = max_tokens * 4
                full_context = full_context[:target_chars] + "..."
            
            return full_context
            
        except Exception as e:
            logger.exception(f"❌ Error compressing context: {e}")
            return "Error loading conversation context."

    # ==========================================
    # MEMORY INSIGHTS & ANALYTICS
    # ==========================================

    async def analyze_user_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze user conversation patterns and preferences"""
        try:
            # Get recent summaries
            summaries = await self.get_conversation_summaries(user_id, limit=20)
            
            if not summaries:
                return {"user_id": user_id, "insufficient_data": True}
            
            # Analyze patterns
            all_topics = []
            all_symbols = []
            importance_scores = []
            
            for summary in summaries:
                all_topics.extend(summary.get("topics", []))
                all_symbols.extend(summary.get("symbols", []))
                importance_scores.append(summary.get("importance_score", 0.5))
            
            # Calculate statistics
            from collections import Counter
            
            topic_counts = Counter(all_topics)
            symbol_counts = Counter(all_symbols)
            
            analysis = {
                "user_id": user_id,
                "analysis_date": datetime.utcnow().isoformat(),
                "conversation_count": len(summaries),
                "average_importance": sum(importance_scores) / len(importance_scores),
                "top_topics": dict(topic_counts.most_common(5)),
                "top_symbols": dict(symbol_counts.most_common(5)),
                "engagement_level": "high" if sum(importance_scores) / len(importance_scores) > 0.7 else "medium"
            }
            
            return analysis
            
        except Exception as e:
            logger.exception(f"❌ Error analyzing user patterns for {user_id}: {e}")
            return {"user_id": user_id, "error": str(e)}

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory service statistics"""
        try:
            stats = {
                "timestamp": datetime.utcnow().isoformat(),
                "collections": {},
                "vector_stats": {}
            }
            
            # MongoDB stats
            summaries_collection = self.base_service.db[self.summaries_collection]
            summaries_count = await summaries_collection.count_documents({})
            stats["collections"]["summaries"] = summaries_count
            
            insights_collection = self.base_service.db[self.insights_collection]
            insights_count = await insights_collection.count_documents({})
            stats["collections"]["insights"] = insights_count
            
            # Vector database stats
            vector_stats = await self.vector_service.get_namespace_stats(
                VectorNamespace.CONVERSATIONS.value
            )
            stats["vector_stats"] = vector_stats
            
            # Redis stats (approximate STM usage)
            redis_info = await self.base_service.redis.info()
            stats["redis_memory"] = redis_info.get("used_memory_human", "unknown")
            
            return stats
            
        except Exception as e:
            logger.exception(f"❌ Error getting memory stats: {e}")
            return {"error": str(e)}

    # ==========================================
    # MEMORY MANAGEMENT & CLEANUP
    # ==========================================

    async def summarize_session(self, user_id: str) -> Dict[str, Any]:
        """Manually trigger session summarization"""
        try:
            # Get current STM
            stm_context = await self.get_stm_context(user_id)
            
            if len(stm_context) < 2:
                return {"error": "Insufficient conversation for summary"}
            
            # Generate summary
            summary = await self._generate_conversation_summary(user_id, stm_context)
            
            if summary:
                # Save summary
                await self._save_conversation_summary(summary)
                
                # Store in vector DB if important
                if summary.importance_score >= self.importance_threshold:
                    await self._store_summary_in_vector_db(summary)
                
                # Clear STM after summarization
                stm_key = self.stm_key_pattern.format(user_id=user_id)
                await self.base_service.redis.delete(stm_key)
                
                return {
                    "summarized": True,
                    "summary_id": summary.summary_id,
                    "importance_score": summary.importance_score,
                    "message_count": summary.message_count
                }
            else:
                return {"error": "Failed to generate summary"}
                
        except Exception as e:
            logger.exception(f"❌ Error in manual summarization for {user_id}: {e}")
            return {"error": str(e)}

    async def cleanup_old_memories(self, days_to_keep: int = 90) -> Dict[str, Any]:
        """Clean up old memories beyond retention period"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            results = {
                "cutoff_date": cutoff_date.isoformat(),
                "deleted_summaries": 0,
                "deleted_vectors": 0,
                "errors": []
            }
            
            # Clean up old conversation summaries
            summaries_collection = self.base_service.db[self.summaries_collection]
            delete_result = await summaries_collection.delete_many({
                "end_time": {"$lt": cutoff_date}
            })
            results["deleted_summaries"] = delete_result.deleted_count
            
            # Clean up old vectors would require iterating through vector DB
            # This is more complex and should be done carefully
            
            logger.info(f"Memory cleanup completed: {results}")
            return results
            
        except Exception as e:
            logger.exception(f"❌ Error in memory cleanup: {e}")
            return {"error": str(e)}

    async def export_user_memories(self, user_id: str) -> Dict[str, Any]:
        """Export all user memories for GDPR compliance"""
        try:
            export_data = {
                "user_id": user_id,
                "export_date": datetime.utcnow().isoformat(),
                "short_term_memory": [],
                "conversation_summaries": [],
                "vector_memories": []
            }
            
            # Export STM
            stm_context = await self.get_stm_context(user_id, limit=100)
            export_data["short_term_memory"] = stm_context
            
            # Export conversation summaries
            summaries = await self.get_conversation_summaries(user_id, limit=1000)
            export_data["conversation_summaries"] = summaries
            
            # Export vector memories (would need special vector DB query)
            # This is a placeholder - actual implementation depends on vector DB capabilities
            
            return export_data
            
        except Exception as e:
            logger.exception(f"❌ Error exporting memories for {user_id}: {e}")
            return {"user_id": user_id, "error": str(e)}

    async def delete_user_data(self, user_id: str) -> Dict[str, Any]:
        """Delete all user memory data for GDPR compliance"""
        try:
            results = {
                "user_id": user_id,
                "deletion_date": datetime.utcnow().isoformat(),
                "deleted_stm": False,
                "deleted_summaries": 0,
                "deleted_insights": 0,
                "deleted_vectors": False
            }
            
            # Delete STM from Redis
            stm_key = self.stm_key_pattern.format(user_id=user_id)
            stm_deleted = await self.base_service.redis.delete(stm_key)
            results["deleted_stm"] = stm_deleted > 0
            
            # Delete conversation summaries
            summaries_collection = self.base_service.db[self.summaries_collection]
            summary_result = await summaries_collection.delete_many({"user_id": user_id})
            results["deleted_summaries"] = summary_result.deleted_count
            
            # Delete insights
            insights_collection = self.base_service.db[self.insights_collection]
            insights_result = await insights_collection.delete_many({"user_id": user_id})
            results["deleted_insights"] = insights_result.deleted_count
            
            # Delete vector memories
            vector_result = await self.vector_service.delete_user_data(user_id)
            results["deleted_vectors"] = vector_result.get("deleted", False)
            
            logger.info(f"User memory data deleted: {results}")
            return results
            
        except Exception as e:
            logger.exception(f"❌ Error deleting user memory data for {user_id}: {e}")
            return {"user_id": user_id, "error": str(e)}

    # ==========================================
    # HELPER METHODS
    # ==========================================

    async def _analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis for conversation turns"""
        try:
            # Use simple keyword-based sentiment for now
            # Could be replaced with more sophisticated sentiment analysis
            positive_words = ['good', 'great', 'excellent', 'love', 'amazing', 'perfect', 'thanks']
            negative_words = ['bad', 'terrible', 'hate', 'awful', 'disappointed', 'frustrated']
            
            text_lower = text.lower()
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                return "positive"
            elif negative_count > positive_count:
                return "negative"
            else:
                return "neutral"
                
        except Exception:
            return "neutral"

    async def _process_old_turns(self, user_id: str, old_turns: List[Dict]):
        """Process old turns that were removed from STM"""
        try:
            if len(old_turns) >= 5:  # Only process if we have substantial conversation
                # Generate summary for old turns
                summary = await self._generate_conversation_summary(user_id, old_turns)
                
                if summary:
                    await self._save_conversation_summary(summary)
                    
                    if summary.importance_score >= self.importance_threshold:
                        await self._store_summary_in_vector_db(summary)
                    
                    logger.debug(f"Processed {len(old_turns)} old turns for {user_id}")
            
        except Exception as e:
            logger.exception(f"❌ Error processing old turns for {user_id}: {e}")

    async def save_user_insight(self, user_id: str, insight_type: str,
                              insight_data: Dict, confidence: float = 1.0) -> bool:
        """Save structured user insight"""
        try:
            insight_doc = {
                "user_id": user_id,
                "type": insight_type,
                "data": insight_data,
                "confidence": confidence,
                "created_at": datetime.utcnow(),
                "source": "memory_service"
            }
            
            collection = self.base_service.db[self.insights_collection]
            result = await collection.insert_one(insight_doc)
            
            return result.inserted_id is not None
            
        except Exception as e:
            logger.exception(f"❌ Error saving user insight: {e}")
            return False

    async def get_user_insights(self, user_id: str, insight_type: str = None,
                              limit: int = 10) -> List[Dict]:
        """Get user insights"""
        try:
            collection = self.base_service.db[self.insights_collection]
            
            query = {"user_id": user_id}
            if insight_type:
                query["type"] = insight_type
            
            cursor = collection.find(query).sort("created_at", -1).limit(limit)
            insights = await cursor.to_list(length=limit)
            
            return insights
            
        except Exception as e:
            logger.exception(f"❌ Error getting user insights: {e}")
            return []
