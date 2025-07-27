# ===== services/database.py - FIXED REDIS IMPORT =====
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as aioredis  # ✅ FIXED: Use correct alias
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

# Add this method to your DatabaseService class
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
    
    async def close(self):
        """Close database connections"""
        if self.mongo_client:
            self.mongo_client.close()
        if self.redis:
            await self.redis.close()
