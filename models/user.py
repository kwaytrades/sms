# models/user.py - ENHANCED WITH CONTEXT-RICH CONVERSATION TRACKING

from pymongo import MongoClient
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import os
import logging

logger = logging.getLogger(__name__)

# EXISTING CLASSES - UNCHANGED
@dataclass
class PlanLimits:
    """Plan limits configuration"""
    plan_type: str
    messages_per_period: int
    period: str  # "weekly", "monthly", etc.
    price: float
    features: List[str] = field(default_factory=list)
    
    @staticmethod
    def get_plan_config() -> Dict[str, 'PlanLimits']:
        """Get plan configuration"""
        return {
            "free": PlanLimits(
                plan_type="free",
                messages_per_period=10,
                period="weekly",
                price=0.0,
                features=["Basic stock analysis", "Weekly insights"]
            ),
            "paid": PlanLimits(
                plan_type="paid", 
                messages_per_period=100,
                period="monthly",
                price=29.0,
                features=["Advanced analysis", "Portfolio tracking", "Monthly insights"]
            ),
            "pro": PlanLimits(
                plan_type="pro",
                messages_per_period=999999,  # Unlimited
                period="monthly", 
                price=99.0,
                features=["Unlimited messages", "Real-time alerts", "Priority support"]
            )
        }

@dataclass 
class UserProfile:
    """User profile data structure"""
    phone_number: str
    email: Optional[str] = None
    first_name: Optional[str] = None
    timezone: str = "US/Eastern"
    plan_type: str = "free"
    subscription_status: str = "trialing"
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    trial_ends_at: Optional[datetime] = None
    
    # Trading profile
    risk_tolerance: str = "medium"
    trading_experience: str = "intermediate"
    preferred_sectors: List[str] = field(default_factory=list)
    watchlist: List[str] = field(default_factory=list)
    trading_style: str = "swing"
    
    # Communication preferences
    communication_style: Dict[str, Any] = field(default_factory=lambda: {
        "formality": "casual",
        "message_length": "medium",
        "emoji_usage": True,
        "technical_depth": "medium"
    })
    
    # Behavioral patterns
    response_patterns: Dict[str, Any] = field(default_factory=lambda: {
        "preferred_response_time": "immediate",
        "engagement_triggers": [],
        "successful_trade_patterns": [],
        "loss_triggers": []
    })
    
    # Trading behavior tracking
    trading_behavior: Dict[str, Any] = field(default_factory=lambda: {
        "successful_sectors": [],
        "preferred_position_sizes": [],
        "typical_hold_time": None,
        "win_rate": None,
        "average_gain": None,
        "average_loss": None
    })
    
    # Speech and interaction patterns
    speech_patterns: Dict[str, Any] = field(default_factory=lambda: {
        "vocabulary_level": "intermediate",
        "question_types": [],
        "common_tickers": [],
        "interaction_frequency": "moderate"
    })
    
    # Plaid integration
    plaid_access_tokens: List[str] = field(default_factory=list)
    connected_accounts: List[Dict] = field(default_factory=list)
    
    # Usage tracking
    total_messages_sent: int = 0
    total_messages_received: int = 0
    messages_this_period: int = 0
    period_start: Optional[datetime] = None
    last_message_at: Optional[datetime] = None
    last_active_at: Optional[datetime] = None
    
    # Daily insights
    daily_insights_enabled: bool = True
    premarket_enabled: bool = True
    market_close_enabled: bool = True
    premarket_time: str = "09:00"
    market_close_time: str = "16:00"
    promotional_messages: bool = True
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    _id: Optional[str] = None

# NEW: CONTEXT-RICH CONVERSATION TRACKING STRUCTURES
@dataclass
class ConversationContext:
    """Rich conversation context for better responses"""
    # Recent conversation memory (last 7 days)
    recent_topics: List[Dict] = field(default_factory=list)  # {topic, symbols, date, sentiment}
    last_discussed_symbols: List[str] = field(default_factory=list)  # Last 10 symbols discussed
    pending_decisions: List[Dict] = field(default_factory=list)  # {decision, context, urgency, date}
    
    # Relationship tracking
    total_conversations: int = 0
    relationship_stage: str = "new"  # new, getting_acquainted, building_trust, established
    conversation_frequency: str = "occasional"  # rare, occasional, regular, frequent
    
    # Session tracking
    last_session_date: Optional[str] = None
    session_message_count: int = 0
    session_mood: str = "neutral"  # excited, neutral, worried, frustrated
    
    # Response effectiveness
    successful_response_patterns: List[str] = field(default_factory=list)
    preferred_analysis_depth: str = "standard"  # quick, standard, detailed, comprehensive

@dataclass
class DailySession:
    """Daily conversation session summary"""
    date: str
    message_count: int = 0
    topics_discussed: List[str] = field(default_factory=list)
    symbols_mentioned: List[str] = field(default_factory=list)
    session_mood: str = "neutral"
    key_decisions: List[str] = field(default_factory=list)
    session_summary: str = ""

# ENHANCED UserManager CLASS WITH CONTEXT TRACKING
class UserManager:
    def __init__(self):
        mongodb_url = os.getenv('MONGODB_URL')
        self.client = MongoClient(mongodb_url)
        
        # Use the 'ai' database where your existing users are!
        self.db = self.client.ai
        self.users = self.db.users
        
        # Also set up usage tracking collection
        self.usage_tracking = self.db.usage_tracking
        self.conversations = self.db.conversations
        
        # NEW: Add context-rich collections
        self.daily_sessions = self.db.daily_sessions
        self.conversation_context = self.db.conversation_context
        
        # Create indexes for performance
        try:
            self.users.create_index("phone_number", unique=True)
            self.users.create_index("last_active_at")
            self.usage_tracking.create_index([("phone_number", 1), ("date", 1)])
            self.conversations.create_index([("phone_number", 1), ("timestamp", -1)])
            
            # NEW: Context indexes
            self.daily_sessions.create_index([("phone_number", 1), ("date", -1)])
            self.conversation_context.create_index("phone_number", unique=True)
            
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")
    
    # EXISTING METHODS - UNCHANGED
    def get_user_by_phone(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """Get user profile by phone number"""
        try:
            user = self.users.find_one({"phone_number": phone_number})
            if user:
                logger.info(f"✅ Found user: {phone_number}")
                return user
            return None
        except Exception as e:
            logger.error(f"❌ Error finding user {phone_number}: {e}")
            return None
    
    def update_existing_user_schema(self, phone_number: str) -> bool:
        """Update existing user to new schema without overwriting data"""
        try:
            user = self.users.find_one({"phone_number": phone_number})
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
                result = self.users.update_one(
                    {"phone_number": phone_number},
                    {"$set": updates}
                )
                logger.info(f"✅ Updated schema for existing user {phone_number}: {list(updates.keys())}")
                return result.modified_count > 0
            
            return True  # User already has all fields
            
        except Exception as e:
            logger.error(f"❌ Error updating user schema for {phone_number}: {e}")
            return False
    
    def get_or_create_user(self, phone_number: str) -> Dict[str, Any]:
        """Get existing user or create new one"""
        user = self.get_user_by_phone(phone_number)
        
        if user:
            # Update existing user with any missing fields
            self.update_existing_user_schema(phone_number)
            # Get the updated user
            user = self.get_user_by_phone(phone_number)
            
            # NEW: Initialize conversation context if missing
            self._ensure_conversation_context(phone_number)
            
            return user
        else:
            # Create completely new user
            new_user = self.create_new_user(phone_number)
            # NEW: Create conversation context for new user
            self._create_conversation_context(phone_number)
            return new_user
    
    def create_new_user(self, phone_number: str) -> Dict[str, Any]:
        """Create completely new user with full schema"""
        now = datetime.now(timezone.utc)
        
        new_user = {
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
            result = self.users.insert_one(new_user)
            new_user["_id"] = result.inserted_id
            logger.info(f"✅ Created new user: {phone_number}")
            return new_user
        except Exception as e:
            logger.error(f"❌ Error creating user {phone_number}: {e}")
            raise
    
    # EXISTING METHODS CONTINUE UNCHANGED...
    def update_user_activity(self, phone_number: str, message_type: str = "received") -> bool:
        """Update user activity and message counts with detailed logging"""
        now = datetime.now(timezone.utc)
        
        try:
            user = self.get_user_by_phone(phone_number)
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
                self.log_usage(phone_number, "message_received")
                
                # NEW: Update conversation context
                self._update_conversation_activity(phone_number)
                
            elif message_type == "sent":
                inc_updates["total_messages_sent"] = 1
                
                # Log to usage_tracking collection  
                self.log_usage(phone_number, "message_sent")
            
            # Perform the update
            update_doc = {"$set": set_updates}
            if inc_updates:
                update_doc["$inc"] = inc_updates
            
            result = self.users.update_one(
                {"phone_number": phone_number},
                update_doc
            )
            
            if result.modified_count > 0:
                logger.info(f"✅ Updated activity for {phone_number}: {message_type} - set: {list(set_updates.keys())}, inc: {list(inc_updates.keys())}")
                return True
            else:
                logger.warning(f"⚠️ No document modified for {phone_number}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error updating activity for {phone_number}: {e}")
            return False
    
    def check_message_limits(self, phone_number: str) -> Dict[str, Any]:
        """Check if user can send more messages"""
        user = self.get_user_by_phone(phone_number)
        if not user:
            return {"can_send": False, "reason": "User not found"}
        
        plan_type = user.get('plan_type', 'free')
        messages_this_period = user.get('messages_this_period', 0)
        
        limits = {
            'free': 10,
            'paid': 100,
            'pro': 999999  # Effectively unlimited
        }
        
        limit = limits.get(plan_type, 10)
        
        # Special handling for pro users (cooloff after 50/day)
        if plan_type == 'pro':
            # Check daily usage
            last_message = user.get('last_message_at')
            if last_message:
                hours_since_last = (datetime.now(timezone.utc) - last_message).total_seconds() / 3600
                if hours_since_last < 24:
                    # Count messages in last 24 hours
                    daily_count = self.get_daily_message_count(phone_number)
                    if daily_count >= 50:
                        return {
                            "can_send": False,
                            "reason": "Daily limit reached. Pro users get cooloff after 50 messages/day.",
                            "limit": 50,
                            "used": daily_count,
                            "resets_in_hours": 24 - hours_since_last
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
    
    def get_daily_message_count(self, phone_number: str) -> int:
        """Get message count in last 24 hours"""
        try:
            user = self.get_user_by_phone(phone_number)
            if not user:
                return 0
                
            last_message = user.get('last_message_at')
            if not last_message:
                return 0
                
            hours_ago = (datetime.now(timezone.utc) - last_message).total_seconds() / 3600
            if hours_ago > 24:
                return 0
            
            # Estimate based on recent activity (you'd improve this with proper message logging)
            return min(user.get('messages_this_period', 0), 50)
        except Exception as e:
            logger.error(f"Error getting daily count for {phone_number}: {e}")
            return 0
    
    def log_usage(self, phone_number: str, action: str, metadata: Dict[str, Any] = None):
        """Log usage to usage_tracking collection"""
        try:
            usage_log = {
                "phone_number": phone_number,
                "action": action,  # message_received, message_sent, analysis_request, etc.
                "timestamp": datetime.now(timezone.utc),
                "date": datetime.now(timezone.utc).date().isoformat(),
                "metadata": metadata or {}
            }
            
            self.usage_tracking.insert_one(usage_log)
            logger.info(f"📊 Logged usage: {phone_number} - {action}")
            
        except Exception as e:
            logger.error(f"❌ Error logging usage for {phone_number}: {e}")
    
    def log_conversation(self, phone_number: str, message: str, response: str, intent: str, symbols: List[str] = None):
        """Log conversation to conversations collection"""
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
            
            self.conversations.insert_one(conversation_log)
            logger.info(f"💬 Logged conversation: {phone_number}")
            
        except Exception as e:
            logger.error(f"❌ Error logging conversation for {phone_number}: {e}")
    
    def learn_from_interaction(self, phone_number: str, message: str, symbols: List[str], intent: str):
        """Learn from user interaction patterns with detailed updates"""
        try:
            # Get current user data
            user = self.get_user_by_phone(phone_number)
            if not user:
                return False
            
            # Prepare learning updates
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
            
            updates["speech_patterns.question_types"] = current_question_types
            
            # Analyze message characteristics
            message_length = "short" if len(message) < 20 else "long" if len(message) > 60 else "medium"
            has_emojis = any(ord(char) > 127 for char in message)
            
            # Update communication style
            updates["communication_style.message_length"] = message_length
            updates["communication_style.emoji_usage"] = has_emojis
            
            # Determine technical depth based on intent and symbols
            if intent in ['technical', 'analyze'] or any(word in message.lower() for word in ['rsi', 'macd', 'support', 'resistance']):
                updates["communication_style.technical_depth"] = "advanced"
            elif intent in ['price', 'general']:
                updates["communication_style.technical_depth"] = "basic"
            else:
                updates["communication_style.technical_depth"] = "medium"
            
            # Always update timestamp
            updates["updated_at"] = datetime.now(timezone.utc)
            
            # Perform the update
            result = self.users.update_one(
                {"phone_number": phone_number},
                {"$set": updates}
            )
            
            if result.modified_count > 0:
                logger.info(f"📚 Learned from interaction: {phone_number} - updated: {list(updates.keys())}")
                
                # Log the learning event
                self.log_usage(phone_number, "behavioral_learning", {
                    "intent": intent,
                    "symbols": symbols,
                    "message_length": len(message),
                    "learned_fields": list(updates.keys())
                })
                
                # NEW: Update conversation context with this interaction
                self._update_conversation_context(phone_number, message, symbols, intent)
                
                return True
            else:
                logger.warning(f"⚠️ Learning update didn't modify document for {phone_number}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error learning from interaction for {phone_number}: {e}")
            return False
    
    def update_subscription(self, phone_number: str, plan_type: str, stripe_customer_id: str = None, stripe_subscription_id: str = None) -> bool:
        """Update user subscription status"""
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
            
            result = self.users.update_one(
                {"phone_number": phone_number},
                {"$set": updates}
            )
            
            logger.info(f"💳 Updated subscription for {phone_number}: {plan_type}")
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"❌ Error updating subscription for {phone_number}: {e}")
            return False
    
    def get_user_stats(self) -> Dict[str, Any]:
        """Get overall user statistics"""
        try:
            total_users = self.users.count_documents({})
            
            plan_breakdown = list(self.users.aggregate([
                {"$group": {"_id": "$plan_type", "count": {"$sum": 1}}}
            ]))
            
            active_users = self.users.count_documents({
                "last_active_at": {"$gte": datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)}
            })
            
            return {
                "total_users": total_users,
                "active_today": active_users,
                "plan_breakdown": {item["_id"]: item["count"] for item in plan_breakdown}
            }
        except Exception as e:
            logger.error(f"❌ Error getting user stats: {e}")
            return {}
    
    # ===== NEW: CONTEXT-RICH CONVERSATION METHODS =====
    
    def _ensure_conversation_context(self, phone_number: str):
        """Ensure conversation context exists for user"""
        try:
            existing_context = self.conversation_context.find_one({"phone_number": phone_number})
            if not existing_context:
                self._create_conversation_context(phone_number)
        except Exception as e:
            logger.error(f"❌ Error ensuring conversation context for {phone_number}: {e}")
    
    def _create_conversation_context(self, phone_number: str):
        """Create initial conversation context for new user"""
        try:
            context = {
                "phone_number": phone_number,
                "recent_topics": [],
                "last_discussed_symbols": [],
                "pending_decisions": [],
                "total_conversations": 0,
                "relationship_stage": "new",
                "conversation_frequency": "occasional",
                "last_session_date": None,
                "session_message_count": 0,
                "session_mood": "neutral",
                "successful_response_patterns": [],
                "preferred_analysis_depth": "standard",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            
            self.conversation_context.insert_one(context)
            logger.info(f"✅ Created conversation context for {phone_number}")
            
        except Exception as e:
            logger.error(f"❌ Error creating conversation context for {phone_number}: {e}")
    
    def _update_conversation_activity(self, phone_number: str):
        """Update conversation activity and session tracking"""
        try:
            today = datetime.now(timezone.utc).date().isoformat()
            
            # Update conversation context
            self.conversation_context.update_one(
                {"phone_number": phone_number},
                {
                    "$inc": {"total_conversations": 1},
                    "$set": {
                        "last_session_date": today,
                        "updated_at": datetime.now(timezone.utc)
                    }
                },
                upsert=True
            )
            
            # Update or create daily session
            self.daily_sessions.update_one(
                {"phone_number": phone_number, "date": today},
                {
                    "$inc": {"message_count": 1},
                    "$setOnInsert": {
                        "topics_discussed": [],
                        "symbols_mentioned": [],
                        "session_mood": "neutral",
                        "key_decisions": [],
                        "session_summary": ""
                    },
                    "$set": {"updated_at": datetime.now(timezone.utc)}
                },
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"❌ Error updating conversation activity for {phone_number}: {e}")
    
    def _update_conversation_context(self, phone_number: str, message: str, symbols: List[str], intent: str):
        """Update conversation context with latest interaction"""
        try:
            now = datetime.now(timezone.utc)
            today = now.date().isoformat()
            
            # Prepare context updates
            updates = {
                "$set": {"updated_at": now},
                "$addToSet": {}
            }
            
            # Add symbols to recent symbols
            if symbols:
                updates["$addToSet"]["last_discussed_symbols"] = {"$each": symbols}
            
            # Add topic to recent topics
            if intent != "general":
                topic_entry = {
                    "topic": intent,
                    "symbols": symbols,
                    "date": now.isoformat(),
                    "sentiment": "neutral"  # Could be enhanced with sentiment analysis
                }
                updates["$push"] = {"recent_topics": topic_entry}
            
            # Update conversation context
            self.conversation_context.update_one(
                {"phone_number": phone_number},
                updates,
                upsert=True
            )
            
            # Update daily session with symbols and topics
            if symbols or intent != "general":
                session_updates = {"$set": {"updated_at": now}}
                
                if symbols:
                    session_updates["$addToSet"] = {"symbols_mentioned": {"$each": symbols}}
                
                if intent != "general":
                    session_updates.setdefault("$addToSet", {})["topics_discussed"] = intent
                
                self.daily_sessions.update_one(
                    {"phone_number": phone_number, "date": today},
                    session_updates,
                    upsert=True
                )
            
            # Clean up old context (keep last 30 days)
            cleanup_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            self.conversation_context.update_one(
                {"phone_number": phone_number},
                {
                    "$pull": {
                        "recent_topics": {"date": {"$lt": cleanup_date}},
                        "pending_decisions": {"date": {"$lt": cleanup_date}}
                    },
                    "$set": {
                        "last_discussed_symbols": {"$slice": ["$last_discussed_symbols", -10]}
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Error updating conversation context for {phone_number}: {e}")
    
    def get_conversation_context(self, phone_number: str) -> Dict[str, Any]:
        """Get conversation context for response generation"""
        try:
            # Ensure context exists
            self._ensure_conversation_context(phone_number)
            
            # Get context
            context = self.conversation_context.find_one({"phone_number": phone_number})
            if not context:
                return self._get_default_context()
            
            # Get today's session
            today = datetime.now(timezone.utc).date().isoformat()
            today_session = self.daily_sessions.find_one({
                "phone_number": phone_number,
                "date": today
            })
            
            # Get recent conversations (last 3 days)
            recent_date = (datetime.now(timezone.utc) - timedelta(days=3)).date().isoformat()
            recent_sessions = list(self.daily_sessions.find({
                "phone_number": phone_number,
                "date": {"$gte": recent_date}
            }).sort("date", -1).limit(3))
            
            # Build comprehensive context
            return {
                "recent_symbols": context.get("last_discussed_symbols", [])[-5:],  # Last 5 symbols
                "recent_topics": context.get("recent_topics", [])[-3:],  # Last 3 topics
                "pending_decisions": context.get("pending_decisions", []),
                "relationship_stage": context.get("relationship_stage", "new"),
                "total_conversations": context.get("total_conversations", 0),
                "preferred_analysis_depth": context.get("preferred_analysis_depth", "standard"),
                "today_session": {
                    "message_count": today_session.get("message_count", 0) if today_session else 0,
                    "topics_discussed": today_session.get("topics_discussed", []) if today_session else [],
                    "symbols_mentioned": today_session.get("symbols_mentioned", []) if today_session else [],
                    "session_mood": today_session.get("session_mood", "neutral") if today_session else "neutral"
                },
                "recent_sessions": recent_sessions,
                "conversation_frequency": context.get("conversation_frequency", "occasional"),
                "is_first_message_today": not today_session or today_session.get("message_count", 0) == 0
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting conversation context for {phone_number}: {e}")
            return self._get_default_context()
    
    def _get_default_context(self) -> Dict[str, Any]:
        """Get default context for new users or error cases"""
        return {
            "recent_symbols": [],
            "recent_topics": [],
            "pending_decisions": [],
            "relationship_stage": "new",
            "total_conversations": 0,
            "preferred_analysis_depth": "standard",
            "today_session": {
                "message_count": 0,
                "topics_discussed": [],
                "symbols_mentioned": [],
                "session_mood": "neutral"
            },
            "recent_sessions": [],
            "conversation_frequency": "occasional",
            "is_first_message_today": True
        }
    
    def add_pending_decision(self, phone_number: str, decision: str, context: str, urgency: str = "medium"):
        """Add a pending decision to track"""
        try:
            decision_entry = {
                "decision": decision,
                "context": context,
                "urgency": urgency,
                "date": datetime.now(timezone.utc).isoformat()
            }
            
            self.conversation_context.update_one(
                {"phone_number": phone_number},
                {"$push": {"pending_decisions": decision_entry}},
                upsert=True
            )
            
            logger.info(f"📝 Added pending decision for {phone_number}: {decision}")
            
        except Exception as e:
            logger.error(f"❌ Error adding pending decision for {phone_number}: {e}")
    
    def update_relationship_stage(self, phone_number: str):
        """Update relationship stage based on conversation count"""
        try:
            context = self.conversation_context.find_one({"phone_number": phone_number})
            if not context:
                return
            
            total_conversations = context.get("total_conversations", 0)
            current_stage = context.get("relationship_stage", "new")
            
            # Determine new stage
            new_stage = current_stage
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
                self.conversation_context.update_one(
                    {"phone_number": phone_number},
                    {"$set": {"relationship_stage": new_stage}}
                )
                logger.info(f"🤝 Updated relationship stage for {phone_number}: {current_stage} → {new_stage}")
        
        except Exception as e:
            logger.error(f"❌ Error updating relationship stage for {phone_number}: {e}")
    
    def generate_session_summary(self, phone_number: str, date: str = None) -> str:
        """Generate a summary of the day's conversation"""
        try:
            if not date:
                date = datetime.now(timezone.utc).date().isoformat()
            
            session = self.daily_sessions.find_one({
                "phone_number": phone_number,
                "date": date
            })
            
            if not session:
                return "No conversation today"
            
            # Build simple summary
            topics = session.get("topics_discussed", [])
            symbols = session.get("symbols_mentioned", [])
            message_count = session.get("message_count", 0)
            mood = session.get("session_mood", "neutral")
            
            summary_parts = [f"{message_count} messages exchanged"]
            
            if topics:
                summary_parts.append(f"discussed {', '.join(topics[:3])}")
            
            if symbols:
                summary_parts.append(f"analyzed {', '.join(symbols[:3])}")
            
            if mood != "neutral":
                summary_parts.append(f"user mood: {mood}")
            
            summary = " | ".join(summary_parts)
            
            # Save summary
            self.daily_sessions.update_one(
                {"phone_number": phone_number, "date": date},
                {"$set": {"session_summary": summary}}
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error generating session summary for {phone_number}: {e}")
            return "Unable to generate summary"
