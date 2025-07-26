# ===== services/database.py - FIXED ASYNC DATABASE SERVICE =====
from motor.motor_asyncio import AsyncIOMotorClient
import aioredis
from typing import Optional, Dict, List, Any
from bson import ObjectId
import json
from datetime import datetime, timezone, timedelta
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
            # MongoDB connection with Motor (async)
            self.mongo_client = AsyncIOMotorClient(settings.mongodb_url)
            
            # Use configurable database name instead of hardcoded 'ai'
            db_name = getattr(settings, 'database_name', 'ai')
            self.db = self.mongo_client[db_name]
            
            # Test MongoDB connection
            await self.db.admin.command('ping')
            
            # Redis connection
            if settings.redis_url:
                self.redis = await aioredis.from_url(settings.redis_url)
                # Test Redis connection
                await self.redis.ping()
            
            # Setup indexes
            await self._setup_indexes()
            
            logger.info("✅ Database connections initialized successfully")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    async def _setup_indexes(self):
        """Setup database indexes for performance"""
        try:
            # User indexes
            await self.db.users.create_index("phone_number", unique=True)
            await self.db.users.create_index([("plan_type", 1), ("subscription_status", 1)])
            await self.db.users.create_index("last_active_at")
            
            # Conversation indexes
            await self.db.conversations.create_index([("user_id", 1), ("session_start", -1)])
            await self.db.conversations.create_index("session_id", unique=True)
            
            # Trading data indexes
            await self.db.trading_data.create_index("user_id", unique=True)
            
            # Usage tracking indexes
            await self.db.usage_tracking.create_index([("phone_number", 1), ("date", 1)])
            await self.db.usage_tracking.create_index("timestamp")
            
            logger.info("✅ Database indexes created successfully")
        except Exception as e:
            logger.warning(f"⚠️ Index creation warning (may already exist): {e}")
    
    # ===== USER MANAGEMENT =====
    
    async def get_user_by_phone(self, phone_number: str) -> Optional[UserProfile]:
        """Get user by phone number - ASYNC VERSION"""
        try:
            user_doc = await self.db.users.find_one({"phone_number": phone_number})
            if not user_doc:
                return None
            
            # Convert ObjectId to string for serialization
            if user_doc.get('_id'):
                user_doc['_id'] = str(user_doc['_id'])
            
            return UserProfile(**user_doc)
        except Exception as e:
            logger.error(f"❌ Error getting user by phone {phone_number}: {e}")
            return None
    
    async def get_or_create_user(self, phone_number: str) -> UserProfile:
        """Get existing user or create new one - ASYNC VERSION"""
        user = await self.get_user_by_phone(phone_number)
        
        if user:
            # Update existing user with any missing fields
            await self._update_user_schema(phone_number)
            # Get the updated user
            user = await self.get_user_by_phone(phone_number)
            return user
        else:
            # Create completely new user
            return await self._create_new_user(phone_number)
    
    async def _create_new_user(self, phone_number: str) -> UserProfile:
        """Create completely new user with full schema - ASYNC VERSION"""
        now = datetime.now(timezone.utc)
        
        new_user_data = {
            "phone_number": phone_number,
            "email": None,
            "first_name": None,
            "timezone": "US/Eastern",
            "plan_type": "free",
            "subscription_status": "trialing",
            "stripe_customer_id": None,
            "stripe_subscription_id": None,
            "trial_ends_at": None,
            
            # Trading profile
            "risk_tolerance": "medium",
            "trading_experience": "intermediate", 
            "preferred_sectors": [],
            "watchlist": [],
            "trading_style": "swing",
            
            # Communication preferences
            "communication_style": {
                "formality": "casual",
                "message_length": "medium",
                "emoji_usage": True,
                "technical_depth": "medium"
            },
            
            # Behavioral patterns
            "response_patterns": {
                "preferred_response_time": "immediate",
                "engagement_triggers": [],
                "successful_trade_patterns": [],
                "loss_triggers": []
            },
            
            # Trading behavior tracking
            "trading_behavior": {
                "successful_sectors": [],
                "preferred_position_sizes": [],
                "typical_hold_time": None,
                "win_rate": None,
                "average_gain": None,
                "average_loss": None
            },
            
            # Speech and interaction patterns
            "speech_patterns": {
                "vocabulary_level": "intermediate",
                "question_types": [],
                "common_tickers": [],
                "interaction_frequency": "moderate"
            },
            
            # Plaid integration
            "plaid_access_tokens": [],
            "connected_accounts": [],
            
            # Usage tracking
            "total_messages_sent": 0,
            "total_messages_received": 0,
            "messages_this_period": 0,
            "period_start": now,
            "last_message_at": None,
            "last_active_at": now,
            
            # Daily insights
            "daily_insights_enabled": True,
            "premarket_time": "09:00",
            "market_close_time": "16:00",
            
            # Timestamps
            "created_at": now,
            "updated_at": now
        }
        
        try:
            result = await self.db.users.insert_one(new_user_data)
            new_user_data["_id"] = str(result.inserted_id)
            logger.info(f"✅ Created new user: {phone_number}")
            return UserProfile(**new_user_data)
        except Exception as e:
            logger.error(f"❌ Error creating user {phone_number}: {e}")
            raise
    
    async def _update_user_schema(self, phone_number: str) -> bool:
        """Update existing user to new schema without overwriting data - ASYNC VERSION"""
        try:
            user = await self.db.users.find_one({"phone_number": phone_number})
            if not user:
                return False
            
            # Only add missing fields, don't overwrite existing ones
            updates = {}
            now = datetime.now(timezone.utc)
            
            # Add missing message tracking fields
            if "total_messages_sent" not in user:
                updates["total_messages_sent"] = 0
            if "total_messages_received" not in user:
                updates["total_messages_received"] = 0
            if "messages_this_period" not in user:
                updates["messages_this_period"] = 0
            if "period_start" not in user:
                updates["period_start"] = now
            if "last_message_at" not in user:
                updates["last_message_at"] = None
                
            # Add missing communication style structure
            if "communication_style" not in user or not isinstance(user.get("communication_style"), dict):
                updates["communication_style"] = {
                    "formality": "casual",
                    "message_length": "medium", 
                    "emoji_usage": True,
                    "technical_depth": "medium"
                }
            
            # Add missing speech patterns structure
            if "speech_patterns" not in user or not isinstance(user.get("speech_patterns"), dict):
                updates["speech_patterns"] = {
                    "vocabulary_level": "intermediate",
                    "question_types": [],
                    "common_tickers": [],
                    "interaction_frequency": "moderate"
                }
            
            # Add missing behavioral patterns
            if "response_patterns" not in user or not isinstance(user.get("response_patterns"), dict):
                updates["response_patterns"] = {
                    "preferred_response_time": "immediate",
                    "engagement_triggers": [],
                    "successful_trade_patterns": [],
                    "loss_triggers": []
                }
            
            # Add missing trading behavior
            if "trading_behavior" not in user or not isinstance(user.get("trading_behavior"), dict):
                updates["trading_behavior"] = {
                    "successful_sectors": [],
                    "preferred_position_sizes": [],
                    "typical_hold_time": None,
                    "win_rate": None,
                    "average_gain": None,
                    "average_loss": None
                }
            
            # Add daily insights if missing
            if "daily_insights_enabled" not in user:
                updates["daily_insights_enabled"] = True
            if "premarket_time" not in user:
                updates["premarket_time"] = "09:00"
            if "market_close_time" not in user:
                updates["market_close_time"] = "16:00"
            
            # Always update the updated_at field
            updates["updated_at"] = now
            
            if updates:
                result = await self.db.users.update_one(
                    {"phone_number": phone_number},
                    {"$set": updates}
                )
                logger.info(f"✅ Updated schema for existing user {phone_number}")
                return result.modified_count > 0
            
            return True  # User already has all fields
            
        except Exception as e:
            logger.error(f"❌ Error updating user schema for {phone_number}: {e}")
            return False
    
    async def save_user(self, user: UserProfile) -> str:
        """Save or update user profile - ASYNC VERSION"""
        try:
            user_dict = user.__dict__.copy()
            user_dict['updated_at'] = datetime.now(timezone.utc)
            
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
    
    async def update_user_activity(self, phone_number: str, message_type: str = "received") -> bool:
        """Update user activity and message counts - ASYNC VERSION"""
        now = datetime.now(timezone.utc)
        
        try:
            user = await self.get_user_by_phone(phone_number)
            if not user:
                logger.error(f"❌ User not found for activity update: {phone_number}")
                return False
            
            # Build update operations
            set_updates = {
                "last_active_at": now,
                "updated_at": now
            }
            
            inc_updates = {}
            
            if message_type == "received":
                set_updates["last_message_at"] = now
                inc_updates["total_messages_received"] = 1
                inc_updates["messages_this_period"] = 1
                
                # Log to usage_tracking collection
                await self._log_usage(phone_number, "message_received")
                
            elif message_type == "sent":
                inc_updates["total_messages_sent"] = 1
                
                # Log to usage_tracking collection  
                await self._log_usage(phone_number, "message_sent")
            
            # Perform the update
            update_doc = {"$set": set_updates}
            if inc_updates:
                update_doc["$inc"] = inc_updates
            
            result = await self.db.users.update_one(
                {"phone_number": phone_number},
                update_doc
            )
            
            if result.modified_count > 0:
                logger.info(f"✅ Updated activity for {phone_number}: {message_type}")
                return True
            else:
                logger.warning(f"⚠️ No document modified for {phone_number}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error updating activity for {phone_number}: {e}")
            return False
    
    async def check_message_limits(self, phone_number: str) -> Dict[str, Any]:
        """Check if user can send more messages - ASYNC VERSION"""
        user_doc = await self.db.users.find_one({"phone_number": phone_number})
        if not user_doc:
            return {"can_send": False, "reason": "User not found"}
        
        plan_type = user_doc.get('plan_type', 'free')
        messages_this_period = user_doc.get('messages_this_period', 0)
        
        limits = {
            'free': 10,
            'paid': 100,
            'pro': 999999  # Effectively unlimited
        }
        
        limit = limits.get(plan_type, 10)
        
        # Special handling for pro users (cooloff after 50/day)
        if plan_type == 'pro':
            daily_count = await self._get_daily_message_count(phone_number)
            if daily_count >= 50:
                return {
                    "can_send": False,
                    "reason": "Daily cooloff active",
                    "limit": 50,
                    "used": daily_count,
                    "plan": plan_type
                }
        
        can_send = messages_this_period < limit
        
        return {
            "can_send": can_send,
            "reason": "Limit exceeded" if not can_send else "OK",
            "limit": limit,
            "used": messages_this_period,
            "remaining": max(0, limit - messages_this_period),
            "plan": plan_type
        }
    
    async def _get_daily_message_count(self, phone_number: str) -> int:
        """Get message count in last 24 hours - ASYNC VERSION"""
        try:
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            
            count = await self.db.usage_tracking.count_documents({
                "phone_number": phone_number,
                "action": "message_received",
                "timestamp": {"$gte": yesterday}
            })
            
            return count
        except Exception as e:
            logger.error(f"Error getting daily count for {phone_number}: {e}")
            return 0
    
    async def _log_usage(self, phone_number: str, action: str, metadata: Dict[str, Any] = None):
        """Log usage to usage_tracking collection - ASYNC VERSION"""
        try:
            usage_log = {
                "phone_number": phone_number,
                "action": action,
                "timestamp": datetime.now(timezone.utc),
                "date": datetime.now(timezone.utc).date().isoformat(),
                "metadata": metadata or {}
            }
            
            await self.db.usage_tracking.insert_one(usage_log)
            logger.info(f"📊 Logged usage: {phone_number} - {action}")
            
        except Exception as e:
            logger.error(f"❌ Error logging usage for {phone_number}: {e}")
    
    async def log_conversation(self, phone_number: str, message: str, response: str, intent: str, symbols: List[str] = None):
        """Log conversation to conversations collection - ASYNC VERSION"""
        try:
            conversation_log = {
                "phone_number": phone_number,
                "user_message": message,
                "bot_response": response,
                "intent": intent,
                "symbols": symbols or [],
                "timestamp": datetime.now(timezone.utc),
                "date": datetime.now(timezone.utc).date().isoformat()
            }
            
            await self.db.conversations.insert_one(conversation_log)
            logger.info(f"💬 Logged conversation: {phone_number}")
            
        except Exception as e:
            logger.error(f"❌ Error logging conversation for {phone_number}: {e}")
    
    async def learn_from_interaction(self, phone_number: str, message: str, symbols: List[str], intent: str):
        """Learn from user interaction patterns - ASYNC VERSION"""
        try:
            # Get current user data
            user_doc = await self.db.users.find_one({"phone_number": phone_number})
            if not user_doc:
                return False
            
            # Prepare learning updates
            updates = {}
            
            # Update common tickers
            current_tickers = user_doc.get('speech_patterns', {}).get('common_tickers', [])
            for symbol in symbols:
                if symbol not in current_tickers:
                    current_tickers.append(symbol)
            
            # Keep only last 20 tickers
            if len(current_tickers) > 20:
                current_tickers = current_tickers[-20:]
            
            updates["speech_patterns.common_tickers"] = current_tickers
            
            # Update question types
            current_question_types = user_doc.get('speech_patterns', {}).get('question_types', [])
            if intent not in current_question_types:
                current_question_types.append(intent)
            
            updates["speech_patterns.question_types"] = current_question_types
            
            # Analyze message characteristics
            message_length = "short" if len(message) < 20 else "long" if len(message) > 60 else "medium"
            has_emojis = any(ord(char) > 127 for char in message)
            
            # Update communication style
            updates["communication_style.message_length"] = message_length
            updates["communication_style.emoji_usage"] = has_emojis
            
            # Determine technical depth
            if intent in ['technical', 'analyze'] or any(word in message.lower() for word in ['rsi', 'macd', 'support', 'resistance']):
                updates["communication_style.technical_depth"] = "advanced"
            elif intent in ['price', 'general']:
                updates["communication_style.technical_depth"] = "basic"
            else:
                updates["communication_style.technical_depth"] = "medium"
            
            # Always update timestamp
            updates["updated_at"] = datetime.now(timezone.utc)
            
            # Perform the update
            result = await self.db.users.update_one(
                {"phone_number": phone_number},
                {"$set": updates}
            )
            
            if result.modified_count > 0:
                logger.info(f"📚 Learned from interaction: {phone_number}")
                
                # Log the learning event
                await self._log_usage(phone_number, "behavioral_learning", {
                    "intent": intent,
                    "symbols": symbols,
                    "message_length": len(message),
                    "learned_fields": list(updates.keys())
                })
                
                return True
            else:
                logger.warning(f"⚠️ Learning update didn't modify document for {phone_number}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error learning from interaction for {phone_number}: {e}")
            return False
    
    async def update_subscription(self, phone_number: str, plan_type: str, stripe_customer_id: str = None, stripe_subscription_id: str = None) -> bool:
        """Update user subscription status - ASYNC VERSION"""
        try:
            updates = {
                "plan_type": plan_type,
                "subscription_status": "active" if plan_type in ['paid', 'pro'] else "trialing",
                "updated_at": datetime.now(timezone.utc)
            }
            
            if stripe_customer_id:
                updates["stripe_customer_id"] = stripe_customer_id
            if stripe_subscription_id:
                updates["stripe_subscription_id"] = stripe_subscription_id
            
            # Reset usage counters when upgrading
            if plan_type in ['paid', 'pro']:
                updates["messages_this_period"] = 0
                updates["period_start"] = datetime.now(timezone.utc)
            
            result = await self.db.users.update_one(
                {"phone_number": phone_number},
                {"$set": updates}
            )
            
            logger.info(f"💳 Updated subscription for {phone_number}: {plan_type}")
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"❌ Error updating subscription for {phone_number}: {e}")
            return False
    
    async def get_user_stats(self) -> Dict[str, Any]:
        """Get overall user statistics - ASYNC VERSION"""
        try:
            total_users = await self.db.users.count_documents({})
            
            # Get plan breakdown
            pipeline = [
                {"$group": {"_id": "$plan_type", "count": {"$sum": 1}}}
            ]
            plan_breakdown_cursor = self.db.users.aggregate(pipeline)
            plan_breakdown = {item["_id"]: item["count"] async for item in plan_breakdown_cursor}
            
            # Get active users today
            today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            active_users = await self.db.users.count_documents({
                "last_active_at": {"$gte": today_start}
            })
            
            return {
                "total_users": total_users,
                "active_today": active_users,
                "plan_breakdown": plan_breakdown
            }
        except Exception as e:
            logger.error(f"❌ Error getting user stats: {e}")
            return {}
    
    # ===== CONVERSATION MANAGEMENT =====
    
    async def save_message(self, message: ChatMessage) -> str:
        """Save chat message - ASYNC VERSION"""
        try:
            message_dict = message.__dict__.copy()
            result = await self.db.conversations.update_one(
                {"session_id": message.session_id},
                {
                    "$push": {"messages": message_dict},
                    "$inc": {"total_messages": 1},
                    "$setOnInsert": {
                        "user_id": message.user_id,
                        "session_start": datetime.now(timezone.utc)
                    }
                },
                upsert=True
            )
            return str(result.upserted_id) if result.upserted_id else "updated"
        except Exception as e:
            logger.error(f"❌ Error saving message: {e}")
            raise
    
    # ===== REDIS OPERATIONS =====
    
    async def get_usage_count(self, user_id: str, period: str) -> int:
        """Get user's message usage for period - ASYNC VERSION"""
        try:
            if not self.redis:
                return 0
            
            key = f"usage:{user_id}:{period}"
            count = await self.redis.get(key)
            return int(count) if count else 0
        except Exception as e:
            logger.error(f"❌ Error getting usage count: {e}")
            return 0
    
    async def increment_usage(self, user_id: str, period: str, ttl: int):
        """Increment usage counter with TTL - ASYNC VERSION"""
        try:
            if not self.redis:
                return
            
            key = f"usage:{user_id}:{period}"
            await self.redis.incr(key)
            await self.redis.expire(key, ttl)
        except Exception as e:
            logger.error(f"❌ Error incrementing usage: {e}")
    
    async def close(self):
        """Close database connections"""
        if self.mongo_client:
            self.mongo_client.close()
        if self.redis:
            await self.redis.close()
