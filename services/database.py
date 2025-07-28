# ===== services/database.py - ENHANCED WITH CONTEXT-RICH OPERATIONS =====
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as aioredis  # ✅ FIXED: Use correct alias
from typing import Optional, Dict, List, Any
from bson import ObjectId
import json
from datetime import datetime, timedelta, timezone
from loguru import logger

from models.user import UserProfile
from models.conversation import ChatMessage, Conversation
from models.trading import TradingData
from config import settings

class DatabaseService:
    def __init__(self):
        self.mongo_client = None
        self.db = None
        self.redis = None

    async def initialize(self):
        """Initialize database connections"""
        try:
            # MongoDB connection
            self.mongo_client = AsyncIOMotorClient(settings.mongodb_url)
            self.db = self.mongo_client.ai
            
            # Redis connection - NOW MATCHES THE IMPORT!
            self.redis = await aioredis.from_url(settings.redis_url)
            
            # Setup indexes
            await self._setup_indexes()
            
            logger.info("✅ Database connections initialized")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    async def _setup_indexes(self):
        """Setup database indexes for performance"""
        try:
            # User indexes
            await self.db.users.create_index("phone_number", unique=True)
            await self.db.users.create_index([("plan_type", 1), ("subscription_status", 1)])
            
            # Conversation indexes
            await self.db.conversations.create_index([("user_id", 1), ("session_start", -1)])
            await self.db.conversations.create_index("session_id", unique=True)
            
            # Trading data indexes
            await self.db.trading_data.create_index("user_id", unique=True)
            
            # NEW: Context-rich indexes
            await self.db.daily_sessions.create_index([("phone_number", 1), ("date", -1)])
            await self.db.conversation_context.create_index("phone_number", unique=True)
            await self.db.enhanced_conversations.create_index([("phone_number", 1), ("timestamp", -1)])
            
            logger.info("✅ Database indexes created")
        except Exception as e:
            logger.error(f"❌ Index creation failed: {e}")
    
    # EXISTING METHODS - UNCHANGED
    async def get_user_by_phone(self, phone_number: str) -> Optional[UserProfile]:
        """Get user by phone number"""
        try:
            user_doc = await self.db.users.find_one({"phone_number": phone_number})
            if not user_doc:
                return None
            
            user_doc['_id'] = str(user_doc['_id'])
            return UserProfile(**user_doc)
        except Exception as e:
            logger.error(f"❌ Error getting user by phone {phone_number}: {e}")
            return None
    
    async def save_user(self, user: UserProfile) -> str:
        """Save or update user profile"""
        try:
            user_dict = user.__dict__.copy()
            user_dict['updated_at'] = datetime.utcnow()
            
            if user._id:
                # Update existing - remove _id from update data
                user_dict.pop('_id', None)
                await self.db.users.update_one(
                    {"_id": ObjectId(user._id)},
                    {"$set": user_dict}
                )
                return user._id
            else:
                # Create new - remove _id if it's None
                user_dict.pop('_id', None)
                result = await self.db.users.insert_one(user_dict)
                return str(result.inserted_id)
        except Exception as e:
            logger.error(f"❌ Error saving user: {e}")
            raise
    
    async def save_message(self, message: ChatMessage) -> str:
        """Save chat message"""
        try:
            message_dict = message.__dict__.copy()
            result = await self.db.conversations.update_one(
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
            logger.error(f"❌ Error saving message: {e}")
            raise
    
    async def get_usage_count(self, user_id: str, period: str) -> int:
        """Get user's message usage for period"""
        try:
            key = f"usage:{user_id}:{period}"
            count = await self.redis.get(key)
            return int(count) if count else 0
        except Exception as e:
            logger.error(f"❌ Error getting usage count: {e}")
            return 0
    
    async def increment_usage(self, user_id: str, period: str, ttl: int):
        """Increment usage counter with TTL"""
        try:
            key = f"usage:{user_id}:{period}"
            await self.redis.incr(key)
            await self.redis.expire(key, ttl)
        except Exception as e:
            logger.error(f"❌ Error incrementing usage: {e}")
    
    async def cleanup_invalid_users(self):
        """Remove users with null _id"""
        try:
            result = await self.db.users.delete_many({"_id": None})
            logger.info(f"Cleaned up {result.deleted_count} invalid users")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    # EXISTING METHOD - KEEP UNCHANGED
    async def update_user_activity(self, phone_number: str):
        """Update user activity timestamp"""
        try:
            if self.db:
                await self.db.users.update_one(
                    {"phone": phone_number},
                    {
                        "$set": {"last_activity": datetime.utcnow()},
                        "$setOnInsert": {"phone": phone_number, "created_at": datetime.utcnow()}
                    },
                    upsert=True
                )
                logger.info(f"✅ Updated user activity for {phone_number}")
        except Exception as e:
            logger.error(f"❌ Failed to update user activity: {e}")
    
    async def close(self):
        """Close database connections"""
        if self.mongo_client:
            self.mongo_client.close()
        if self.redis:
            await self.redis.close()
    
    # ===== NEW: CONTEXT-RICH CONVERSATION OPERATIONS =====
    
    async def get_conversation_context(self, phone_number: str) -> Dict[str, Any]:
        """Get rich conversation context for response generation"""
        try:
            # Get conversation context
            context = await self.db.conversation_context.find_one({"phone_number": phone_number})
            
            # Get today's session
            today = datetime.now(timezone.utc).date().isoformat()
            today_session = await self.db.daily_sessions.find_one({
                "phone_number": phone_number,
                "date": today
            })
            
            # Get recent message history (last 10 messages)
            recent_messages = await self.get_recent_messages(phone_number, limit=10)
            
            # Get user profile for personality context
            user_profile = await self.get_user_by_phone(phone_number)
            
            # Build comprehensive context
            if not context:
                context = await self._create_default_context(phone_number)
            
            return {
                "user_profile": user_profile.__dict__ if user_profile else {},
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
                "context_summary": self._build_context_summary(context, today_session, user_profile)
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting conversation context for {phone_number}: {e}")
            return self._get_fallback_context()
    
    async def _create_default_context(self, phone_number: str) -> Dict[str, Any]:
        """Create default conversation context for new user"""
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
            
            await self.db.conversation_context.insert_one(default_context)
            logger.info(f"✅ Created default conversation context for {phone_number}")
            
            return default_context
            
        except Exception as e:
            logger.error(f"❌ Error creating default context for {phone_number}: {e}")
            return {}
    
    def _build_context_summary(self, context: Dict, today_session: Dict, user_profile) -> str:
        """Build human-readable context summary"""
        try:
            summary_parts = []
            
            # Relationship context
            if context:
                stage = context.get("relationship_stage", "new")
                total_convos = context.get("total_conversations", 0)
                summary_parts.append(f"Relationship: {stage} ({total_convos} conversations)")
            
            # Communication style
            if user_profile:
                comm_style = user_profile.__dict__.get("communication_style", {})
                formality = comm_style.get("formality", "casual")
                energy = comm_style.get("energy", "moderate")
                summary_parts.append(f"Style: {formality}/{energy}")
            
            # Today's activity
            if today_session:
                msg_count = today_session.get("message_count", 0)
                mood = today_session.get("session_mood", "neutral")
                summary_parts.append(f"Today: {msg_count} messages, {mood} mood")
            
            # Recent focus
            if context and context.get("last_discussed_symbols"):
                recent_symbols = context["last_discussed_symbols"][-3:]
                summary_parts.append(f"Recent symbols: {', '.join(recent_symbols)}")
            
            return " | ".join(summary_parts) if summary_parts else "New conversation"
            
        except Exception as e:
            logger.error(f"❌ Error building context summary: {e}")
            return "Context unavailable"
    
    def _get_fallback_context(self) -> Dict[str, Any]:
        """Fallback context when database operations fail"""
        return {
            "user_profile": {},
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
    
    async def get_recent_messages(self, phone_number: str, limit: int = 10) -> List[Dict]:
        """Get recent message history for context"""
        try:
            messages = await self.db.enhanced_conversations.find(
                {"phone_number": phone_number}
            ).sort("timestamp", -1).limit(limit).to_list(length=limit)
            
            # Reverse to get chronological order (oldest first)
            return list(reversed(messages))
            
        except Exception as e:
            logger.error(f"❌ Error getting recent messages for {phone_number}: {e}")
            return []
    
    async def save_enhanced_message(self, phone_number: str, user_message: str, bot_response: str, 
                                   intent_data: Dict, symbols: List[str] = None, 
                                   context_used: Dict = None) -> bool:
        """Save enhanced message with full context"""
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
            await self.db.enhanced_conversations.insert_one(message_doc)
            
            # Update conversation context
            await self._update_conversation_context_db(phone_number, user_message, symbols or [], intent_data.get("intent", "general"))
            
            # Update daily session
            await self._update_daily_session_db(phone_number, today, symbols or [], intent_data.get("intent", "general"))
            
            logger.info(f"✅ Saved enhanced message for {phone_number}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving enhanced message for {phone_number}: {e}")
            return False
    
    async def _update_conversation_context_db(self, phone_number: str, message: str, 
                                            symbols: List[str], intent: str):
        """Update conversation context in database"""
        try:
            now = datetime.now(timezone.utc)
            
            updates = {
                "$inc": {"total_conversations": 1},
                "$set": {"updated_at": now}
            }
            
            # Add symbols to recent list
            if symbols:
                updates["$addToSet"] = {"last_discussed_symbols": {"$each": symbols}}
            
            # Add topic if not general
            if intent != "general":
                topic_entry = {
                    "topic": intent,
                    "symbols": symbols,
                    "date": now.isoformat(),
                    "message_snippet": message[:50] + "..." if len(message) > 50 else message
                }
                updates["$push"] = {"recent_topics": topic_entry}
            
            await self.db.conversation_context.update_one(
                {"phone_number": phone_number},
                updates,
                upsert=True
            )
            
            # Clean up old data (keep last 30 entries)
            cleanup_date = (now - timedelta(days=30)).isoformat()
            await self.db.conversation_context.update_one(
                {"phone_number": phone_number},
                {
                    "$pull": {
                        "recent_topics": {"date": {"$lt": cleanup_date}}
                    },
                    "$push": {
                        "last_discussed_symbols": {
                            "$each": [],
                            "$slice": -10  # Keep only last 10 symbols
                        }
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Error updating conversation context in DB for {phone_number}: {e}")
    
    async def _update_daily_session_db(self, phone_number: str, date: str, 
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
            
            await self.db.daily_sessions.update_one(
                {"phone_number": phone_number, "date": date},
                updates,
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"❌ Error updating daily session in DB for {phone_number}: {e}")
    
    async def add_pending_decision(self, phone_number: str, decision: str, 
                                 context: str, urgency: str = "medium") -> bool:
        """Add pending decision to track"""
        try:
            decision_entry = {
                "decision": decision,
                "context": context,
                "urgency": urgency,
                "date": datetime.now(timezone.utc).isoformat()
            }
            
            await self.db.conversation_context.update_one(
                {"phone_number": phone_number},
                {"$push": {"pending_decisions": decision_entry}},
                upsert=True
            )
            
            logger.info(f"📝 Added pending decision for {phone_number}: {decision}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding pending decision for {phone_number}: {e}")
            return False
    
    async def get_user_conversation_summary(self, phone_number: str, days: int = 7) -> Dict[str, Any]:
        """Get conversation summary for the last N days"""
        try:
            start_date = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
            
            # Get sessions in date range
            sessions = await self.db.daily_sessions.find({
                "phone_number": phone_number,
                "date": {"$gte": start_date}
            }).sort("date", -1).to_list(length=days)
            
            # Get conversation context
            context = await self.db.conversation_context.find_one({"phone_number": phone_number})
            
            # Build summary
            total_messages = sum(session.get("message_count", 0) for session in sessions)
            all_symbols = []
            all_topics = []
            
            for session in sessions:
                all_symbols.extend(session.get("symbols_mentioned", []))
                all_topics.extend(session.get("topics_discussed", []))
            
            unique_symbols = list(set(all_symbols))
            unique_topics = list(set(all_topics))
            
            return {
                "phone_number": phone_number,
                "period": f"Last {days} days",
                "total_messages": total_messages,
                "active_days": len(sessions),
                "symbols_discussed": unique_symbols,
                "topics_covered": unique_topics,
                "relationship_stage": context.get("relationship_stage", "new") if context else "new",
                "pending_decisions": context.get("pending_decisions", []) if context else [],
                "daily_breakdown": sessions
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting conversation summary for {phone_number}: {e}")
            return {}
    
    async def update_relationship_stage(self, phone_number: str) -> str:
        """Update and return relationship stage based on activity"""
        try:
            context = await self.db.conversation_context.find_one({"phone_number": phone_number})
            if not context:
                return "new"
            
            total_conversations = context.get("total_conversations", 0)
            current_stage = context.get("relationship_stage", "new")
            
            # Determine new stage
            if total_conversations < 3:
                new_stage = "new"
            elif total_conversations < 10:
                new_stage = "getting_acquainted"
            elif total_conversations < 25:
                new_stage = "building_trust"
            else:
                new_stage = "established"
            
            # Update if changed
            if new_stage != current_stage:
                await self.db.conversation_context.update_one(
                    {"phone_number": phone_number},
                    {"$set": {"relationship_stage": new_stage}}
                )
                logger.info(f"🤝 Updated relationship stage for {phone_number}: {current_stage} → {new_stage}")
            
            return new_stage
            
        except Exception as e:
            logger.error(f"❌ Error updating relationship stage for {phone_number}: {e}")
            return "new"
