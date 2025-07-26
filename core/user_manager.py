# core/user_manager.py - Unified User Management
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
from bson import ObjectId
from loguru import logger

from config import settings, PLAN_LIMITS
from services.database import DatabaseService

class UserManager:
    def __init__(self, db_service: DatabaseService):
        self.db = db_service
    
    async def get_user_by_phone(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """Get user by phone number"""
        try:
            user = await self.db.db.users.find_one({"phone_number": phone_number})
            if user:
                user["_id"] = str(user["_id"])
                logger.info(f"✅ Found user: {phone_number}")
                return user
            return None
            
        except Exception as e:
            logger.error(f"❌ Error finding user {phone_number}: {e}")
            return None
    
    async def get_or_create_user(self, phone_number: str) -> Dict[str, Any]:
        """Get existing user or create new one"""
        user = await self.get_user_by_phone(phone_number)
        
        if user:
            # Update existing user schema if needed
            await self._update_user_schema(user)
            return await self.get_user_by_phone(phone_number)
        else:
            # Create new user
            return await self._create_new_user(phone_number)
    
    async def _create_new_user(self, phone_number: str) -> Dict[str, Any]:
        """Create a new user with complete schema"""
        now = datetime.now(timezone.utc)
        
        new_user = {
            "phone_number": phone_number,
            "email": None,
            "first_name": None,
            "timezone": "US/Eastern",
            
            # Subscription
            "plan_type": "free",
            "subscription_status": "active",
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
            
            # Behavioral patterns (learned over time)
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
            
            # Usage tracking
            "total_messages_sent": 0,
            "total_messages_received": 0,
            "last_message_at": None,
            "last_active_at": now,
            
            # Daily insights
            "daily_insights_enabled": True,
            "premarket_time": "09:00",
            "market_close_time": "16:00",
            
            # Plaid integration (future)
            "plaid_access_tokens": [],
            "connected_accounts": [],
            
            # Timestamps
            "created_at": now,
            "updated_at": now
        }
        
        try:
            result = await self.db.db.users.insert_one(new_user)
            new_user["_id"] = str(result.inserted_id)
            logger.info(f"✅ Created new user: {phone_number}")
            return new_user
            
        except Exception as e:
            logger.error(f"❌ Error creating user {phone_number}: {e}")
            raise
    
    async def _update_user_schema(self, user: Dict[str, Any]) -> bool:
        """Update existing user to latest schema without overwriting data"""
        try:
            phone_number = user["phone_number"]
            updates = {}
            now = datetime.now(timezone.utc)
            
            # Add missing message tracking fields
            if "total_messages_sent" not in user:
                updates["total_messages_sent"] = 0
            if "total_messages_received" not in user:
                updates["total_messages_received"] = 0
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
            
            # Add missing response patterns
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
            
            # Always update timestamp
            updates["updated_at"] = now
            
            if updates:
                await self.db.db.users.update_one(
                    {"phone_number": phone_number},
                    {"$set": updates}
                )
                logger.info(f"✅ Updated schema for user {phone_number}")
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating user schema for {phone_number}: {e}")
            return False
    
    async def update_user_activity(self, phone_number: str, message_type: str = "received") -> bool:
        """Update user activity and message counts"""
        now = datetime.now(timezone.utc)
        
        try:
            updates = {
                "last_active_at": now,
                "updated_at": now
            }
            
            increments = {}
            
            if message_type == "received":
                updates["last_message_at"] = now
                increments["total_messages_received"] = 1
                
                # Also increment usage tracking
                await self.db.increment_usage(phone_number, "message")
                
            elif message_type == "sent":
                increments["total_messages_sent"] = 1
            
            # Build update operation
            update_doc = {"$set": updates}
            if increments:
                update_doc["$inc"] = increments
            
            result = await self.db.db.users.update_one(
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
        """Check if user can send more messages based on weekly limits"""
        try:
            user = await self.get_user_by_phone(phone_number)
            if not user:
                return {"can_send": False, "reason": "User not found"}
            
            plan_type = user.get('plan_type', 'free')
            plan_config = PLAN_LIMITS.get(plan_type, PLAN_LIMITS['free'])
            
            # Get current weekly usage
            weekly_usage = await self.db.get_weekly_usage(phone_number)
            
            # Check subscription status
            if user.get('subscription_status') != 'active':
                return {
                    "can_send": False,
                    "reason": "Subscription inactive",
                    "plan": plan_type,
                    "used": weekly_usage,
                    "limit": plan_config['weekly_limit']
                }
            
            # Check weekly limits
            weekly_limit = plan_config['weekly_limit']
            
            # Special handling for pro users (daily cooloff after 50 messages)
            if plan_type == 'pro':
                daily_usage = await self.db.get_daily_usage(phone_number)
                if daily_usage >= 50:
                    return {
                        "can_send": False,
                        "reason": "Daily cooloff active (Pro plan)",
                        "plan": plan_type,
                        "used": daily_usage,
                        "limit": 50,
                        "reset_type": "daily"
                    }
            
            can_send = weekly_usage < weekly_limit
            
            return {
                "can_send": can_send,
                "reason": "Weekly limit exceeded" if not can_send else "OK",
                "plan": plan_type,
                "used": weekly_usage,
                "limit": weekly_limit,
                "remaining": max(0, weekly_limit - weekly_usage),
                "upgrade_message": plan_config.get('upgrade_message', '')
            }
            
        except Exception as e:
            logger.error(f"❌ Error checking limits for {phone_number}: {e}")
            return {"can_send": False, "reason": f"Error: {str(e)}"}
    
    async def learn_from_interaction(self, phone_number: str, message: str, 
                                   symbols: List[str], intent: str) -> bool:
        """Learn from user interaction patterns"""
        try:
            user = await self.get_user_by_phone(phone_number)
            if not user:
                return False
            
            updates = {}
            
            # Update common tickers
            current_tickers = user.get('speech_patterns', {}).get('common_tickers', [])
            for symbol in symbols:
                if symbol not in current_tickers:
                    current_tickers.append(symbol)
            
            # Keep only last 20 tickers
            if len(current_tickers) > 20:
                current_tickers = current_tickers[-20:]
            
            updates["speech_patterns.common_tickers"] = current_tickers
            
            # Update question types
            current_question_types = user.get('speech_patterns', {}).get('question_types', [])
            if intent not in current_question_types:
                current_question_types.append(intent)
            
            if len(current_question_types) > 10:
                current_question_types = current_question_types[-10:]
            
            updates["speech_patterns.question_types"] = current_question_types
            
            # Analyze message characteristics
            message_length = "short" if len(message) < 20 else "long" if len(message) > 60 else "medium"
            has_emojis = any(ord(char) > 127 for char in message)
            
            # Update communication style
            updates["communication_style.message_length"] = message_length
            updates["communication_style.emoji_usage"] = has_emojis
            
            # Determine technical depth
            technical_terms = ['rsi', 'macd', 'support', 'resistance', 'bollinger', 'ema', 'fibonacci']
            if any(term in message.lower() for term in technical_terms):
                updates["communication_style.technical_depth"] = "advanced"
            elif intent in ['price', 'general']:
                updates["communication_style.technical_depth"] = "basic"
            else:
                updates["communication_style.technical_depth"] = "medium"
            
            # Update timestamp
            updates["updated_at"] = datetime.now(timezone.utc)
            
            # Perform update
            result = await self.db.db.users.update_one(
                {"phone_number": phone_number},
                {"$set": updates}
            )
            
            if result.modified_count > 0:
                logger.info(f"📚 Learned from interaction: {phone_number}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error learning from interaction for {phone_number}: {e}")
            return False
    
    async def update_subscription(self, phone_number: str, plan_type: str, 
                                stripe_customer_id: str = None, 
                                stripe_subscription_id: str = None) -> bool:
        """Update user subscription status"""
        try:
            updates = {
                "plan_type": plan_type,
                "subscription_status": "active" if plan_type in ['paid', 'pro'] else "active",
                "updated_at": datetime.now(timezone.utc)
            }
            
            if stripe_customer_id:
                updates["stripe_customer_id"] = stripe_customer_id
            if stripe_subscription_id:
                updates["stripe_subscription_id"] = stripe_subscription_id
            
            result = await self.db.db.users.update_one(
                {"phone_number": phone_number},
                {"$set": updates}
            )
            
            logger.info(f"💳 Updated subscription for {phone_number}: {plan_type}")
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"❌ Error updating subscription for {phone_number}: {e}")
            return False
    
    async def update_subscription_from_stripe(self, customer_id: str, subscription: Dict[str, Any]) -> bool:
        """Update subscription from Stripe webhook"""
        try:
            # Find user by Stripe customer ID
            user = await self.db.db.users.find_one({"stripe_customer_id": customer_id})
            if not user:
                logger.warning(f"⚠️ No user found for Stripe customer {customer_id}")
                return False
            
            # Determine plan type from subscription
            price_id = subscription['items']['data'][0]['price']['id']
            plan_type = "free"
            
            if price_id == settings.stripe_paid_price_id:
                plan_type = "paid"
            elif price_id == settings.stripe_pro_price_id:
                plan_type = "pro"
            
            # Update user
            success = await self.update_subscription(
                user["phone_number"],
                plan_type,
                customer_id,
                subscription['id']
            )
            
            logger.info(f"💳 Updated subscription from Stripe webhook: {user['phone_number']} -> {plan_type}")
            return success
            
        except Exception as e:
            logger.error(f"❌ Error updating subscription from Stripe: {e}")
            return False
    
    async def get_user_stats(self) -> Dict[str, Any]:
        """Get overall user statistics"""
        try:
            # Total users
            total_users = await self.db.db.users.count_documents({})
            
            # Users by plan
            pipeline = [
                {"$group": {"_id": "$plan_type", "count": {"$sum": 1}}}
            ]
            plan_breakdown = {}
            async for result in self.db.db.users.aggregate(pipeline):
                plan_breakdown[result["_id"]] = result["count"]
            
            # Active users (last 24 hours)
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            active_users = await self.db.db.users.count_documents({
                "last_active_at": {"$gte": yesterday}
            })
            
            # New users this week
            week_ago = datetime.now(timezone.utc) - timedelta(days=7)
            new_users_week = await self.db.db.users.count_documents({
                "created_at": {"$gte": week_ago}
            })
            
            return {
                "total_users": total_users,
                "plan_breakdown": plan_breakdown,
                "active_24h": active_users,
                "new_this_week": new_users_week,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting user stats: {e}")
            return {"error": str(e)}
    
    async def get_user_for_analysis(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """Get user data optimized for AI analysis"""
        try:
            user = await self.get_user_by_phone(phone_number)
            if not user:
                return None
            
            # Return only the fields needed for AI personalization
            return {
                "plan_type": user.get("plan_type", "free"),
                "trading_experience": user.get("trading_experience", "intermediate"),
                "risk_tolerance": user.get("risk_tolerance", "medium"),
                "communication_style": user.get("communication_style", {}),
                "speech_patterns": user.get("speech_patterns", {}),
                "preferred_sectors": user.get("preferred_sectors", []),
                "watchlist": user.get("watchlist", [])
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting user for analysis {phone_number}: {e}")
            return None
