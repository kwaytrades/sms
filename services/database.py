# ===== services/database.py - FIXED FOR DEPLOYMENT =====
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as aioredis  # Use redis.asyncio instead of aioredis package
from typing import Optional, Dict, List, Any
from bson import ObjectId
import json
from datetime import datetime
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
        """Initialize database connections with proper error handling"""
        try:
            # MongoDB connection
            logger.info("🔗 Connecting to MongoDB...")
            self.mongo_client = AsyncIOMotorClient(settings.mongodb_url)
            self.db = self.mongo_client.ai
            
            # Test MongoDB connection
            await self.db.command("ping")
            logger.info("✅ MongoDB connected successfully")
            
            # Redis connection with fallback
            logger.info("🔗 Connecting to Redis...")
            try:
                if settings.redis_url:
                    # Parse Redis URL and create connection
                    self.redis = await aioredis.from_url(
                        settings.redis_url,
                        encoding="utf-8",
                        decode_responses=True,
                        socket_timeout=5,
                        socket_connect_timeout=5,
                        retry_on_timeout=True,
                        health_check_interval=30
                    )
                    
                    # Test Redis connection
                    await self.redis.ping()
                    logger.info("✅ Redis connected successfully")
                else:
                    logger.warning("⚠️ No Redis URL provided, continuing without cache")
                    self.redis = None
                    
            except Exception as redis_error:
                logger.warning(f"⚠️ Redis connection failed: {redis_error}, continuing without cache")
                self.redis = None
            
            # Setup indexes
            await self._setup_indexes()
            
            logger.info("✅ Database service initialized successfully")
            
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
            
            logger.info("✅ Database indexes created")
        except Exception as e:
            logger.error(f"❌ Index creation failed: {e}")
    
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
        """Get user's message usage for period with Redis fallback"""
        try:
            if self.redis:
                key = f"usage:{user_id}:{period}"
                count = await self.redis.get(key)
                return int(count) if count else 0
            else:
                # Fallback to MongoDB for usage tracking
                result = await self.db.usage_tracking.find_one({
                    "user_id": user_id,
                    "period": period
                })
                return result.get("count", 0) if result else 0
        except Exception as e:
            logger.error(f"❌ Error getting usage count: {e}")
            return 0
    
    async def increment_usage(self, user_id: str, period: str, ttl: int):
        """Increment usage counter with TTL and MongoDB fallback"""
        try:
            if self.redis:
                key = f"usage:{user_id}:{period}"
                await self.redis.incr(key)
                await self.redis.expire(key, ttl)
            
            # Also update MongoDB for persistence
            await self.db.usage_tracking.update_one(
                {"user_id": user_id, "period": period},
                {
                    "$inc": {"count": 1},
                    "$setOnInsert": {"created_at": datetime.utcnow()},
                    "$set": {"updated_at": datetime.utcnow()}
                },
                upsert=True
            )
        except Exception as e:
            logger.error(f"❌ Error incrementing usage: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all database connections"""
        health = {
            "mongodb": {"status": "unknown", "error": None},
            "redis": {"status": "unknown", "error": None}
        }
        
        # Test MongoDB
        try:
            await self.db.command("ping")
            health["mongodb"]["status"] = "connected"
        except Exception as e:
            health["mongodb"]["status"] = "error"
            health["mongodb"]["error"] = str(e)
        
        # Test Redis
        try:
            if self.redis:
                await self.redis.ping()
                health["redis"]["status"] = "connected"
            else:
                health["redis"]["status"] = "not_configured"
        except Exception as e:
            health["redis"]["status"] = "error"
            health["redis"]["error"] = str(e)
        
        return health
    
    async def cleanup_invalid_users(self):
        """Remove users with null _id"""
        try:
            result = await self.db.users.delete_many({"_id": None})
            logger.info(f"Cleaned up {result.deleted_count} invalid users")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    async def close(self):
        """Close database connections"""
        try:
            if self.mongo_client:
                self.mongo_client.close()
                logger.info("MongoDB connection closed")
            
            if self.redis:
                await self.redis.close()
                logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
