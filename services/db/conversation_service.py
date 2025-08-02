
"""
Conversation Service - Conversation management, context, daily sessions
Focused on message storage, context building, and conversation intelligence
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from loguru import logger

from models.conversation import ChatMessage
from services.db.base_db_service import BaseDBService


class ConversationService:
    """Specialized service for conversation management and context"""
    
    def __init__(self, base_service: BaseDBService):
        self.base = base_service
        self._context_cache = {}

    async def initialize(self):
        """Initialize conversation-specific indexes"""
        try:
            # Conversation indexes
            await self.base.db.conversations.create_index([("user_id", 1), ("session_start", -1)])
            await self.base.db.conversations.create_index("session_id", unique=True)
            
            # Enhanced conversation indexes
            await self.base.db.enhanced_conversations.create_index([("phone_number", 1), ("timestamp", -1)])
            await self.base.db.enhanced_conversations.create_index("symbols_mentioned")
            await self.base.db.enhanced_conversations.create_index("date")
            
            # Context indexes
            await self.base.db.conversation_context.create_index("phone_number", unique=True)
            await self.base.db.daily_sessions.create_index([("phone_number", 1), ("date", -1)])
            
            logger.info("✅ Conversation service initialized")
        except Exception as e:
            logger.exception(f"❌ Conversation service initialization failed: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Health check for conversation service"""
        try:
            conv_count = await self.base.db.enhanced_conversations.count_documents({})
            context_count = await self.base.db.conversation_context.count_documents({})
            
            return {
                "status": "healthy",
                "conversation_count": conv_count,
                "context_count": context_count,
                "cache_size": len(self._context_cache)
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def get_context(self, phone_number: str) -> Dict[str, Any]:
        """Get comprehensive conversation context"""
        try:
            cache_key = f"context:{phone_number}"
            if cache_key in self._context_cache:
                cached_time, cached_context = self._context_cache[cache_key]
                if (datetime.now(timezone.utc) - cached_time).total_seconds() < 600:  # 10 minutes
                    return cached_context
            
            # Get conversation context
            context = await self.base.db.conversation_context.find_one({"phone_number": phone_number})
            
            # Get today's session
            today = datetime.now(timezone.utc).date().isoformat()
            today_session = await self.base.db.daily_sessions.find_one({
                "phone_number": phone_number,
                "date": today
            })
            
            # Get recent messages
            recent_messages = await self.get_recent_messages(phone_number, limit=10)
            
            # Build context if missing
            if not context:
                context = await self._create_default_context(phone_number)
            
            comprehensive_context = {
                "conversation_context": {
                    "recent_symbols": context.get("last_discussed_symbols", [])[-5:],
                    "recent_topics": context.get("recent_topics", [])[-3:],
                    "pending_decisions": context.get("pending_decisions", []),
                    "relationship_stage": context.get("relationship_stage", "new"),
                    "total_conversations": context.get("total_conversations", 0),
                    "conversation_frequency": context.get("conversation_frequency", "occasional"),
                    "preferred_analysis_depth": context.get("preferred_analysis_depth", "standard")
                },
                "today_session": {
                    "message_count": today_session.get("message_count", 0) if today_session else 0,
                    "topics_discussed": today_session.get("topics_discussed", []) if today_session else [],
                    "symbols_mentioned": today_session.get("symbols_mentioned", []) if today_session else [],
                    "session_mood": today_session.get("session_mood", "neutral") if today_session else "neutral",
                    "is_first_message_today": not today_session or today_session.get("message_count", 0) == 0
                },
                "recent_messages": recent_messages,
                "context_summary": self._build_context_summary(context, today_session),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Cache the context
            self._context_cache[cache_key] = (datetime.now(timezone.utc), comprehensive_context)
            
            return comprehensive_context
            
        except Exception as e:
            logger.exception(f"❌ Error getting conversation context: {e}")
            return self._get_fallback_context()

    async def save_message(self, phone_number: str, user_message: str, bot_response: str, 
                          intent_data: Dict, symbols: List[str] = None, 
                          context_used: Dict = None) -> bool:
        """Save enhanced message with context updates"""
        try:
            now = datetime.now(timezone.utc)
            today = now.date().isoformat()
            
            message_doc = {
                "phone_number": phone_number,
                "user_message": user_message,
                "bot_response": bot_response,
                "intent_data": intent_data,
                "symbols_mentioned": symbols or [],
                "context_used": context_used or {},
                "timestamp": now,
                "date": today
            }
            
            # Save to enhanced conversations
            await self.base.db.enhanced_conversations.insert_one(message_doc)
            
            # Update conversation context
            await self._update_conversation_context(phone_number, user_message, symbols or [], intent_data.get("intent", "general"))
            
            # Update daily session
            await self._update_daily_session(phone_number, today, symbols or [], intent_data.get("intent", "general"))
            
            # Invalidate context cache
            cache_key = f"context:{phone_number}"
            if cache_key in self._context_cache:
                del self._context_cache[cache_key]
            
            return True
            
        except Exception as e:
            logger.exception(f"❌ Error saving enhanced message: {e}")
            return False

    async def get_recent_messages(self, phone_number: str, limit: int = 10) -> List[Dict]:
        """Get recent message history"""
        try:
            cache_key = f"recent_messages:{phone_number}:{limit}"
            cached_messages = await self.base.key_builder.get(cache_key)
            if cached_messages:
                return cached_messages
            
            messages = await self.base.db.enhanced_conversations.find(
                {"phone_number": phone_number}
            ).sort("timestamp", -1).limit(limit).to_list(length=limit)
            
            # Process messages
            for msg in messages:
                if "_id" in msg:
                    msg["_id"] = str(msg["_id"])
                if isinstance(msg.get("timestamp"), datetime):
                    msg["timestamp"] = msg["timestamp"].isoformat()
            
            # Reverse for chronological order
            messages = list(reversed(messages))
            
            # Cache for 2 minutes
            await self.base.key_builder.set(cache_key, messages, ttl=120)
            
            return messages
            
        except Exception as e:
            logger.exception(f"❌ Error getting recent messages: {e}")
            return []

    # Legacy compatibility methods
    async def save_legacy_message(self, message: ChatMessage) -> str:
        """Legacy compatibility: Save chat message"""
        try:
            message_dict = message.__dict__.copy()
            result = await self.base.db.conversations.update_one(
                {"session_id": message.session_id},
                {
                    "$push": {"messages": message_dict},
                    "$inc": {"total_messages": 1},
                    "$setOnInsert": {
                        "user_id": message.user_id,
                        "session_start": datetime.utcnow()
                    }
                },
                upsert=True
            )
            return str(result.upserted_id) if result.upserted_id else "updated"
        except Exception as e:
            logger.exception(f"❌ Error saving legacy message: {e}")
            raise

    async def get_legacy_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Legacy compatibility: Get conversation history"""
        try:
            conversations = await self.base.db.conversations.find(
                {"user_id": user_id}
            ).sort("session_start", -1).limit(limit).to_list(length=limit)
            
            for conv in conversations:
                conv['_id'] = str(conv['_id'])
            
            return conversations
        except Exception as e:
            logger.exception(f"❌ Error getting legacy history: {e}")
            return []

    async def _create_default_context(self, phone_number: str) -> Dict[str, Any]:
        """Create default conversation context"""
        try:
            default_context = {
                "phone_number": phone_number,
                "recent_topics": [],
                "last_discussed_symbols": [],
                "pending_decisions": [],
                "total_conversations": 0,
                "relationship_stage": "new",
                "conversation_frequency": "occasional",
                "preferred_analysis_depth": "standard",
                "successful_response_patterns": [],
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            
            await self.base.db.conversation_context.insert_one(default_context)
            return default_context
        except Exception as e:
            logger.exception(f"❌ Error creating default context: {e}")
            return {}

    def _build_context_summary(self, context: Dict, today_session: Dict) -> str:
        """Build human-readable context summary"""
        try:
            summary_parts = []
            
            if context:
                stage = context.get("relationship_stage", "new")
                total_convos = context.get("total_conversations", 0)
                summary_parts.append(f"Relationship: {stage} ({total_convos} conversations)")
            
            if today_session:
                msg_count = today_session.get("message_count", 0)
                mood = today_session.get("session_mood", "neutral")
                summary_parts.append(f"Today: {msg_count} messages, {mood} mood")
            
            if context and context.get("last_discussed_symbols"):
                recent_symbols = context["last_discussed_symbols"][-3:]
                summary_parts.append(f"Recent symbols: {', '.join(recent_symbols)}")
            
            return " | ".join(summary_parts) if summary_parts else "New conversation"
        except Exception as e:
            logger.exception(f"❌ Error building context summary: {e}")
            return "Context unavailable"

    def _get_fallback_context(self) -> Dict[str, Any]:
        """Fallback context when operations fail"""
        return {
            "conversation_context": {
                "recent_symbols": [],
                "recent_topics": [],
                "pending_decisions": [],
                "relationship_stage": "new",
                "total_conversations": 0,
                "conversation_frequency": "occasional",
                "preferred_analysis_depth": "standard"
            },
            "today_session": {
                "message_count": 0,
                "topics_discussed": [],
                "symbols_mentioned": [],
                "session_mood": "neutral",
                "is_first_message_today": True
            },
            "recent_messages": [],
            "context_summary": "Context unavailable - fallback mode"
        }

    async def _update_conversation_context(self, phone_number: str, message: str, 
                                         symbols: List[str], intent: str):
        """Update conversation context in database"""
        try:
            now = datetime.now(timezone.utc)
            
            updates = {
                "$inc": {"total_conversations": 1},
                "$set": {"updated_at": now}
            }
            
            if symbols:
                updates["$addToSet"] = {"last_discussed_symbols": {"$each": symbols}}
            
            if intent != "general":
                topic_entry = {
                    "topic": intent,
                    "symbols": symbols,
                    "date": now.isoformat(),
                    "message_snippet": message[:50] + "..." if len(message) > 50 else message
                }
                updates["$push"] = {"recent_topics": topic_entry}
            
            await self.base.db.conversation_context.update_one(
                {"phone_number": phone_number},
                updates,
                upsert=True
            )
            
            # Clean up old data
            cleanup_date = (now - timedelta(days=30)).isoformat()
            await self.base.db.conversation_context.update_one(
                {"phone_number": phone_number},
                {
                    "$pull": {"recent_topics": {"date": {"$lt": cleanup_date}}},
                    "$push": {"last_discussed_symbols": {"$each": [], "$slice": -10}}
                }
            )
        except Exception as e:
            logger.exception(f"❌ Error updating conversation context: {e}")

    async def _update_daily_session(self, phone_number: str, date: str, 
                                  symbols: List[str], intent: str):
        """Update daily session in database"""
        try:
            updates = {
                "$inc": {"message_count": 1},
                "$set": {"updated_at": datetime.now(timezone.utc)}
            }
            
            if symbols:
                updates["$addToSet"] = {"symbols_mentioned": {"$each": symbols}}
            
            if intent != "general":
                updates.setdefault("$addToSet", {})["topics_discussed"] = intent
            
            await self.base.db.daily_sessions.update_one(
                {"phone_number": phone_number, "date": date},
                updates,
                upsert=True
            )
        except Exception as e:
            logger.exception(f"❌ Error updating daily session: {e}")
