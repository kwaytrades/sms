# services/database.py - INTEGRATED WITH OPTIMIZED PERSONALITY ENGINE V2.0
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as aioredis
from typing import Optional, Dict, List, Any
from bson import ObjectId
import json
from datetime import datetime, timedelta, timezone
from loguru import logger

from models.user import UserProfile
from models.conversation import ChatMessage, Conversation
from models.trading import TradingData
from config import settings


class KeyBuilder:
    """KeyBuilder class compatible with OptimizedPersonalityEngine v2.0"""
    
    def __init__(self, redis_client, mongo_db):
        self.redis = redis_client
        self.db = mongo_db
    
    async def get(self, key: str) -> Optional[Dict]:
        """Get data from Redis with JSON deserialization"""
        try:
            data = await self.redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"❌ KeyBuilder get error for {key}: {e}")
            return None
    
    async def set(self, key: str, value: Dict, ttl: int = 86400) -> bool:
        """Set data in Redis with JSON serialization"""
        try:
            serialized_value = json.dumps(value, default=str)
            await self.redis.setex(key, ttl, serialized_value)
            return True
        except Exception as e:
            logger.error(f"❌ KeyBuilder set error for {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"❌ KeyBuilder delete error for {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        try:
            result = await self.redis.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"❌ KeyBuilder exists error for {key}: {e}")
            return False


class DatabaseService:
    """
    Enhanced Database Service fully compatible with OptimizedPersonalityEngine v2.0
    Provides both traditional MongoDB operations and KeyBuilder integration
    """
    
    def __init__(self):
        self.mongo_client = None
        self.db = None
        self.redis = None
        self.key_builder = None

    async def initialize(self):
        """Initialize database connections with KeyBuilder integration"""
        try:
            # MongoDB connection
            self.mongo_client = AsyncIOMotorClient(settings.mongodb_url)
            self.db = self.mongo_client.ai
            
            # Redis connection
            self.redis = await aioredis.from_url(settings.redis_url)
            
            # Initialize KeyBuilder for OptimizedPersonalityEngine v2.0 compatibility
            self.key_builder = KeyBuilder(self.redis, self.db)
            
            # Setup indexes
            await self._setup_indexes()
            
            # Clean up invalid users on startup
            await self.cleanup_invalid_users()
            
            logger.info("✅ Database connections initialized with KeyBuilder support")
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
            
            # Context-rich indexes for enhanced functionality
            await self.db.daily_sessions.create_index([("phone_number", 1), ("date", -1)])
            await self.db.conversation_context.create_index("phone_number", unique=True)
            await self.db.enhanced_conversations.create_index([("phone_number", 1), ("timestamp", -1)])
            
            # Personality engine indexes
            await self.db.personality_profiles.create_index("user_id", unique=True)
            await self.db.personality_analytics.create_index([("user_id", 1), ("timestamp", -1)])
            
            logger.info("✅ Database indexes created")
        except Exception as e:
            logger.error(f"❌ Index creation failed: {e}")
    
    # ==========================================
    # PERSONALITY ENGINE INTEGRATION METHODS
    # ==========================================
    
    async def get_personality_profile(self, user_id: str) -> Optional[Dict]:
        """Get personality profile with KeyBuilder integration"""
        try:
            # Try KeyBuilder first (Redis)
            personality_key = f"user:{user_id}:personality"
            profile = await self.key_builder.get(personality_key)
            
            if profile:
                return profile
            
            # Fallback to MongoDB
            profile_doc = await self.db.personality_profiles.find_one({"user_id": user_id})
            if profile_doc:
                profile_doc.pop('_id', None)  # Remove MongoDB ID
                # Cache in Redis for future requests
                await self.key_builder.set(personality_key, profile_doc, ttl=86400 * 7)  # 7 days
                return profile_doc
            
            return None
        except Exception as e:
            logger.error(f"❌ Error getting personality profile for {user_id}: {e}")
            return None
    
    async def save_personality_profile(self, user_id: str, profile: Dict) -> bool:
        """Save personality profile with dual storage (Redis + MongoDB)"""
        try:
            # Save to Redis via KeyBuilder
            personality_key = f"user:{user_id}:personality"
            redis_success = await self.key_builder.set(personality_key, profile, ttl=86400 * 7)
            
            # Save to MongoDB as backup
            mongo_profile = profile.copy()
            mongo_profile["user_id"] = user_id
            mongo_profile["updated_at"] = datetime.now(timezone.utc)
            
            await self.db.personality_profiles.update_one(
                {"user_id": user_id},
                {"$set": mongo_profile},
                upsert=True
            )
            
            logger.info(f"✅ Saved personality profile for {user_id} (Redis: {redis_success})")
            return True
        except Exception as e:
            logger.error(f"❌ Error saving personality profile for {user_id}: {e}")
            return False
    
    async def cache_analysis_result(self, user_id: str, analysis_data: Dict, ttl: int = 3600) -> bool:
        """Cache analysis result for OptimizedPersonalityEngine"""
        try:
            cache_key = f"user:{user_id}:last_analysis"
            return await self.key_builder.set(cache_key, analysis_data, ttl=ttl)
        except Exception as e:
            logger.error(f"❌ Error caching analysis for {user_id}: {e}")
            return False
    
    async def get_cached_analysis(self, user_id: str) -> Optional[Dict]:
        """Get cached analysis result"""
        try:
            cache_key = f"user:{user_id}:last_analysis"
            return await self.key_builder.get(cache_key)
        except Exception as e:
            logger.error(f"❌ Error getting cached analysis for {user_id}: {e}")
            return None
    
    # ==========================================
    # TRADITIONAL DATABASE OPERATIONS
    # ==========================================
    
    async def get_user_by_phone(self, phone_number: str) -> Optional[UserProfile]:
        """Get user by phone number with validation"""
        try:
            if not phone_number or phone_number.strip() == "":
                logger.warning("❌ get_user_by_phone called with empty phone_number")
                return None
            
            user_doc = await self.db.users.find_one({"phone_number": phone_number})
            if not user_doc:
                return None
            
            user_doc['_id'] = str(user_doc['_id'])
            return UserProfile(**user_doc)
        except Exception as e:
            logger.error(f"❌ Error getting user by phone {phone_number}: {e}")
            return None
    
    async def save_user(self, user: UserProfile) -> str:
        """Save or update user profile with validation"""
        try:
            if not user.phone_number or user.phone_number.strip() == "":
                logger.error("❌ Cannot save user with empty phone_number")
                raise ValueError("User must have a valid phone_number")
            
            user_dict = user.__dict__.copy()
            user_dict['updated_at'] = datetime.utcnow()
            
            if user._id:
                user_dict.pop('_id', None)
                await self.db.users.update_one(
                    {"_id": ObjectId(user._id)},
                    {"$set": user_dict}
                )
                return user._id
            else:
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
        """Remove users with null or empty phone_number"""
        try:
            result_null = await self.db.users.delete_many({"phone_number": None})
            result_empty = await self.db.users.delete_many({"phone_number": ""})
            result_missing = await self.db.users.delete_many({"phone_number": {"$exists": False}})
            
            total_cleaned = result_null.deleted_count + result_empty.deleted_count + result_missing.deleted_count
            
            if total_cleaned > 0:
                logger.info(f"🧹 Cleaned up {total_cleaned} invalid users")
            else:
                logger.info("✅ No invalid users found to clean up")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
    
    async def update_user_activity(self, phone_number: str) -> bool:
        """Update user activity with proper validation"""
        try:
            if not phone_number or phone_number.strip() == "":
                logger.error("❌ update_user_activity called with empty phone_number")
                return False
            
            user = await self.get_user_by_phone(phone_number)
            if not user:
                logger.warning(f"⚠️ User not found for activity update: {phone_number}")
                return False
            
            now = datetime.utcnow()
            result = await self.db.users.update_one(
                {"phone_number": phone_number},
                {
                    "$set": {
                        "last_active_at": now,
                        "updated_at": now
                    },
                    "$inc": {
                        "total_messages_received": 1
                    }
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"✅ Updated activity for {phone_number}")
                return True
            else:
                logger.warning(f"⚠️ No document modified for {phone_number}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to update user activity: {e}")
            return False
    
    async def store_conversation(self, phone_number: str, user_message: str, bot_response: str, 
                               intent: Dict, tool_results: Dict) -> bool:
        """Store conversation with enhanced context"""
        try:
            # Extract symbols from intent or tool results
            symbols = intent.get("symbols", []) if intent else []
            
            if not symbols and tool_results:
                for tool_name, tool_data in tool_results.items():
                    if isinstance(tool_data, dict):
                        if any(isinstance(key, str) and key.isupper() and len(key) <= 5 
                              for key in tool_data.keys()):
                            symbols.extend([key for key in tool_data.keys() 
                                          if isinstance(key, str) and key.isupper() and len(key) <= 5])
                            break
            
            # Use enhanced message saving
            return await self.save_enhanced_message(
                phone_number=phone_number,
                user_message=user_message,
                bot_response=bot_response,
                intent_data=intent,
                symbols=symbols,
                context_used=tool_results
            )
        except Exception as e:
            logger.error(f"❌ Error storing conversation for {phone_number}: {e}")
            return False
    
    # ==========================================
    # CONTEXT-RICH CONVERSATION OPERATIONS
    # ==========================================
    
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
            
            # Get recent message history
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
            
            if context:
                stage = context.get("relationship_stage", "new")
                total_convos = context.get("total_conversations", 0)
                summary_parts.append(f"Relationship: {stage} ({total_convos} conversations)")
            
            if user_profile:
                comm_style = user_profile.__dict__.get("communication_style", {})
                formality = comm_style.get("formality", "casual")
                energy = comm_style.get("energy", "moderate")
                summary_parts.append(f"Style: {formality}/{energy}")
            
            if today_session:
                msg_count = today_session.get("message_count", 0)
                mood = today_session.get("session_mood", "neutral")
                summary_parts.append(f"Today: {msg_count} messages, {mood} mood")
            
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
            
            await self.db.conversation_context.update_one(
                {"phone_number": phone_number},
                updates,
                upsert=True
            )
            
            # Clean up old data
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
                            "$slice": -10
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
    
    # ==========================================
    # UTILITY AND MANAGEMENT METHODS
    # ==========================================
    
    async def health_check(self) -> Dict[str, str]:
        """Check database health"""
        health = {
            "mongodb": "unknown",
            "redis": "unknown",
            "key_builder": "unknown"
        }
        
        try:
            await self.db.command("ping")
            health["mongodb"] = "connected"
        except Exception as e:
            health["mongodb"] = f"error: {str(e)}"
        
        try:
            await self.redis.ping()
            health["redis"] = "connected"
        except Exception as e:
            health["redis"] = f"error: {str(e)}"
        
        try:
            test_key = "health_check_test"
            test_value = {"test": True, "timestamp": datetime.now(timezone.utc).isoformat()}
            await self.key_builder.set(test_key, test_value, ttl=60)
            retrieved = await self.key_builder.get(test_key)
            await self.key_builder.delete(test_key)
            
            if retrieved and retrieved.get("test"):
                health["key_builder"] = "working"
            else:
                health["key_builder"] = "not_working"
        except Exception as e:
            health["key_builder"] = f"error: {str(e)}"
        
        return health
    
    async def close(self):
        """Close database connections"""
        if self.mongo_client:
            self.mongo_client.close()
        if self.redis:
            await self.redis.close()
    
    # ==========================================
    # ADDITIONAL UTILITY METHODS
    # ==========================================
    
    async def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get recent conversation history for user"""
        try:
            conversations = await self.db.conversations.find(
                {"user_id": user_id}
            ).sort("session_start", -1).limit(limit).to_list(length=limit)
            
            for conv in conversations:
                conv['_id'] = str(conv['_id'])
            
            return conversations
        except Exception as e:
            logger.error(f"❌ Error getting conversation history: {e}")
            return []
    
    async def get_user_stats(self, user_id: str) -> Dict:
        """Get user statistics"""
        try:
            stats = {
                "total_conversations": 0,
                "total_messages": 0,
                "first_interaction": None,
                "last_interaction": None
            }
            
            pipeline = [
                {"$match": {"user_id": user_id}},
                {"$group": {
                    "_id": None,
                    "total_conversations": {"$sum": 1},
                    "total_messages": {"$sum": "$total_messages"},
                    "first_interaction": {"$min": "$session_start"},
                    "last_interaction": {"$max": "$session_start"}
                }}
            ]
            
            result = await self.db.conversations.aggregate(pipeline).to_list(length=1)
            if result:
                stats.update(result[0])
                stats.pop('_id', None)
            
            return stats
        except Exception as e:
            logger.error(f"❌ Error getting user stats: {e}")
            return {}
    
    async def save_trading_data(self, user_id: str, symbol: str, data: Dict) -> str:
        """Save trading analysis data"""
        try:
            trading_data = {
                "user_id": user_id,
                "symbol": symbol,
                "data": data,
                "timestamp": datetime.utcnow()
            }
            
            result = await self.db.trading_data.insert_one(trading_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"❌ Error saving trading data: {e}")
            raise
    
    async def get_trading_data(self, user_id: str, symbol: str = None, limit: int = 10) -> List[Dict]:
        """Get trading data for user"""
        try:
            query = {"user_id": user_id}
            if symbol:
                query["symbol"] = symbol
            
            data = await self.db.trading_data.find(query).sort("timestamp", -1).limit(limit).to_list(length=limit)
            
            for item in data:
                item['_id'] = str(item['_id'])
            
            return data
        except Exception as e:
            logger.error(f"❌ Error getting trading data: {e}")
            return []
