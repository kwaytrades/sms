# services/database.py - Unified Database Service
from motor.motor_asyncio import AsyncIOMotorClient
import aioredis
from typing import Optional, Dict, List, Any
from bson import ObjectId
import json
from datetime import datetime, timedelta
from loguru import logger
import asyncio

from config import settings

class DatabaseService:
    def __init__(self):
        self.mongo_client = None
        self.db = None
        self.redis = None
        
    async def initialize(self):
        """Initialize all database connections"""
        try:
            # MongoDB connection
            self.mongo_client = AsyncIOMotorClient(settings.mongodb_url)
            self.db = self.mongo_client.sms_trading_bot
            
            # Test MongoDB connection
            await self.mongo_client.admin.command('ping')
            logger.info("✅ MongoDB connected successfully")
            
            # Redis connection
            self.redis = await aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
                retry_on_timeout=True
            )
            
            # Test Redis connection
            await self.redis.ping()
            logger.info("✅ Redis connected successfully")
            
            # Setup indexes
            await self._setup_indexes()
            
            logger.info("✅ All database connections initialized")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    async def _setup_indexes(self):
        """Setup database indexes for performance"""
        try:
            # User indexes
            await self.db.users.create_index("phone_number", unique=True)
            await self.db.users.create_index([("plan_type", 1), ("subscription_status", 1)])
            await self.db.users.create_index("stripe_customer_id")
            await self.db.users.create_index("last_active_at")
            
            # Conversation indexes
            await self.db.conversations.create_index([("phone_number", 1), ("timestamp", -1)])
            await self.db.conversations.create_index("session_id")
            await self.db.conversations.create_index("date")
            
            # Usage tracking indexes
            await self.db.usage_tracking.create_index([("phone_number", 1), ("date", 1)])
            await self.db.usage_tracking.create_index("action")
            
            logger.info("✅ Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"❌ Index creation failed: {e}")
    
    async def close(self):
        """Close all database connections"""
        try:
            if self.redis:
                await self.redis.close()
            if self.mongo_client:
                self.mongo_client.close()
            logger.info("✅ Database connections closed")
        except Exception as e:
            logger.error(f"❌ Error closing connections: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {
                "mongodb": {
                    "status": "connected" if self.db else "disconnected",
                    "database": "sms_trading_bot"
                },
                "redis": {
                    "status": "connected" if self.redis else "disconnected"
                }
            }
            
            if self.db:
                # Get collection stats
                stats["mongodb"]["collections"] = {
                    "users": await self.db.users.count_documents({}),
                    "conversations": await self.db.conversations.count_documents({}),
                    "usage_tracking": await self.db.usage_tracking.count_documents({})
                }
            
            if self.redis:
                # Get Redis info
                info = await self.redis.info()
                stats["redis"]["memory_used"] = info.get('used_memory_human', 'N/A')
                stats["redis"]["total_keys"] = info.get('db0', {}).get('keys', 0) if 'db0' in info else 0
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting database stats: {e}")
            return {"error": str(e)}
    
    # Redis Cache Operations
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get data from Redis cache"""
        try:
            if not self.redis:
                return None
            
            data = await self.redis.get(key)
            if data:
                return json.loads(data)
            return None
            
        except Exception as e:
            logger.error(f"❌ Cache get failed for key {key}: {e}")
            return None
    
    async def cache_set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set data in Redis cache with TTL"""
        try:
            if not self.redis:
                return False
            
            serialized = json.dumps(value, default=self._json_serializer)
            await self.redis.setex(key, ttl, serialized)
            return True
            
        except Exception as e:
            logger.error(f"❌ Cache set failed for key {key}: {e}")
            return False
    
    async def cache_delete(self, key: str) -> bool:
        """Delete key from Redis cache"""
        try:
            if not self.redis:
                return False
            
            result = await self.redis.delete(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"❌ Cache delete failed for key {key}: {e}")
            return False
    
    async def cache_exists(self, key: str) -> bool:
        """Check if key exists in Redis cache"""
        try:
            if not self.redis:
                return False
            
            result = await self.redis.exists(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"❌ Cache exists check failed for key {key}: {e}")
            return False
    
    # Usage Tracking Operations
    
    async def increment_usage(self, phone_number: str, action: str = "message") -> bool:
        """Increment usage counter for user"""
        try:
            today = datetime.utcnow().date().isoformat()
            
            # Increment in MongoDB
            await self.db.usage_tracking.update_one(
                {
                    "phone_number": phone_number,
                    "date": today,
                    "action": action
                },
                {
                    "$inc": {"count": 1},
                    "$setOnInsert": {
                        "phone_number": phone_number,
                        "date": today,
                        "action": action,
                        "created_at": datetime.utcnow()
                    },
                    "$set": {"updated_at": datetime.utcnow()}
                },
                upsert=True
            )
            
            # Also increment weekly counter in Redis
            week_key = f"usage:weekly:{phone_number}"
            await self.redis.incr(week_key)
            
            # Set expiry to end of current week (Monday 9:30 AM EST)
            now = datetime.utcnow()
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0 and now.hour >= 14:  # After 9:30 AM EST (14:30 UTC)
                days_until_monday = 7
            
            expiry_seconds = days_until_monday * 24 * 3600 + (14.5 * 3600)  # Monday 9:30 AM EST
            await self.redis.expire(week_key, int(expiry_seconds))
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Increment usage failed for {phone_number}: {e}")
            return False
    
    async def get_weekly_usage(self, phone_number: str) -> int:
        """Get weekly usage count for user"""
        try:
            week_key = f"usage:weekly:{phone_number}"
            count = await self.redis.get(week_key)
            return int(count) if count else 0
            
        except Exception as e:
            logger.error(f"❌ Get weekly usage failed for {phone_number}: {e}")
            return 0
    
    async def get_daily_usage(self, phone_number: str, date: str = None) -> int:
        """Get daily usage count for user"""
        try:
            if not date:
                date = datetime.utcnow().date().isoformat()
            
            result = await self.db.usage_tracking.find_one({
                "phone_number": phone_number,
                "date": date,
                "action": "message"
            })
            
            return result.get("count", 0) if result else 0
            
        except Exception as e:
            logger.error(f"❌ Get daily usage failed for {phone_number}: {e}")
            return 0
    
    # Conversation Operations
    
    async def save_conversation(self, phone_number: str, user_message: str, 
                              bot_response: str, intent: str, symbols: List[str] = None) -> bool:
        """Save conversation to database"""
        try:
            conversation = {
                "phone_number": phone_number,
                "user_message": user_message,
                "bot_response": bot_response,
                "intent": intent,
                "symbols": symbols or [],
                "timestamp": datetime.utcnow(),
                "date": datetime.utcnow().date().isoformat()
            }
            
            await self.db.conversations.insert_one(conversation)
            return True
            
        except Exception as e:
            logger.error(f"❌ Save conversation failed for {phone_number}: {e}")
            return False
    
    async def get_conversation_history(self, phone_number: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get conversation history for user"""
        try:
            since_date = datetime.utcnow() - timedelta(days=days)
            
            cursor = self.db.conversations.find({
                "phone_number": phone_number,
                "timestamp": {"$gte": since_date}
            }).sort("timestamp", -1).limit(50)
            
            conversations = []
            async for conv in cursor:
                conv["_id"] = str(conv["_id"])
                conversations.append(conv)
            
            return conversations
            
        except Exception as e:
            logger.error(f"❌ Get conversation history failed for {phone_number}: {e}")
            return []
    
    # Utility methods
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for datetime and ObjectId"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, ObjectId):
            return str(obj)
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    async def cleanup_old_data(self, days: int = 30) -> Dict[str, int]:
        """Clean up old data older than specified days"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Clean old conversations
            conv_result = await self.db.conversations.delete_many({
                "timestamp": {"$lt": cutoff_date}
            })
            
            # Clean old usage tracking
            usage_result = await self.db.usage_tracking.delete_many({
                "created_at": {"$lt": cutoff_date}
            })
            
            logger.info(f"🧹 Cleaned up {conv_result.deleted_count} conversations and {usage_result.deleted_count} usage records")
            
            return {
                "conversations_deleted": conv_result.deleted_count,
                "usage_records_deleted": usage_result.deleted_count
            }
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return {"error": str(e)}
