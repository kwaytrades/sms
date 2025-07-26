# models/user.py
from pymongo import MongoClient
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import os
import logging

logger = logging.getLogger(__name__)

class UserManager:
    def __init__(self):
        mongodb_url = os.getenv('MONGODB_URL')
        self.client = MongoClient(mongodb_url)
        self.db = self.client.sms_trading_bot
        self.users = self.db.users
        
        # Create indexes for performance
        self.users.create_index("phone_number", unique=True)
        self.users.create_index("last_active_at")
        
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
    
    def create_user(self, phone_number: str) -> Dict[str, Any]:
        """Create new user with default profile"""
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
                "formality": "casual",  # casual, professional, technical
                "message_length": "medium",  # short, medium, long
                "emoji_usage": True,
                "technical_depth": "medium"  # basic, medium, advanced
            },
            
            # Behavioral patterns (will be learned)
            "response_patterns": {
                "preferred_response_time": "immediate",  # immediate, morning, evening
                "engagement_triggers": [],  # price_alerts, technical_analysis, news
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
                "vocabulary_level": "intermediate",  # basic, intermediate, advanced
                "question_types": [],  # analysis, price, news, screener
                "common_tickers": [],
                "interaction_frequency": "moderate"  # low, moderate, high
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
    
    def get_or_create_user(self, phone_number: str) -> Dict[str, Any]:
        """Get existing user or create new one"""
        user = self.get_user_by_phone(phone_number)
        if not user:
            user = self.create_user(phone_number)
        return user
    
    def update_user_activity(self, phone_number: str, message_type: str = "received") -> bool:
        """Update user activity and message counts"""
        now = datetime.now(timezone.utc)
        
        # Determine if we need to reset period (weekly for free users, monthly for paid)
        user = self.get_user_by_phone(phone_number)
        if not user:
            return False
        
        period_start = user.get('period_start', now)
        plan_type = user.get('plan_type', 'free')
        
        # Reset period logic
        reset_period = False
        if plan_type == 'free':
            # Weekly reset for free users
            days_since_start = (now - period_start).days
            if days_since_start >= 7:
                reset_period = True
        else:
            # Monthly reset for paid users
            days_since_start = (now - period_start).days
            if days_since_start >= 30:
                reset_period = True
        
        update_fields = {
            "last_active_at": now,
            "updated_at": now
        }
        
        if message_type == "received":
            update_fields["last_message_at"] = now
            if reset_period:
                update_fields["messages_this_period"] = 1
                update_fields["period_start"] = now
                update_fields["$inc"] = {"total_messages_received": 1}
            else:
                update_fields["$inc"] = {
                    "total_messages_received": 1,
                    "messages_this_period": 1
                }
        elif message_type == "sent":
            if reset_period:
                update_fields["period_start"] = now
                update_fields["$inc"] = {"total_messages_sent": 1}
            else:
                update_fields["$inc"] = {"total_messages_sent": 1}
        
        try:
            result = self.users.update_one(
                {"phone_number": phone_number},
                {"$set": {k: v for k, v in update_fields.items() if k != "$inc"},
                 "$inc": update_fields.get("$inc", {})}
            )
            
            logger.info(f"✅ Updated activity for {phone_number}: {message_type}")
            return result.modified_count > 0
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
            pipeline = [
                {"$match": {"phone_number": phone_number}},
                {"$project": {
                    "total_messages_received": 1,
                    "last_message_at": 1
                }}
            ]
            # This is simplified - in production you'd query a messages collection
            # For now, return approximate based on recent activity
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
    
    def update_user_profile(self, phone_number: str, updates: Dict[str, Any]) -> bool:
        """Update user profile fields"""
        try:
            updates["updated_at"] = datetime.now(timezone.utc)
            
            result = self.users.update_one(
                {"phone_number": phone_number},
                {"$set": updates}
            )
            
            logger.info(f"✅ Updated profile for {phone_number}: {list(updates.keys())}")
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"❌ Error updating profile for {phone_number}: {e}")
            return False
    
    def learn_from_interaction(self, phone_number: str, message: str, symbols: List[str], intent: str, satisfaction_score: Optional[float] = None):
        """Learn from user interaction patterns"""
        try:
            user = self.get_user_by_phone(phone_number)
            if not user:
                return False
            
            # Update common tickers
            current_tickers = user.get('speech_patterns', {}).get('common_tickers', [])
            for symbol in symbols:
                if symbol not in current_tickers:
                    current_tickers.append(symbol)
            
            # Limit to top 20 most mentioned
            if len(current_tickers) > 20:
                current_tickers = current_tickers[-20:]
            
            # Update question types
            current_question_types = user.get('speech_patterns', {}).get('question_types', [])
            if intent not in current_question_types:
                current_question_types.append(intent)
            
            # Update message length preference based on user message
            message_length = "short" if len(message) < 20 else "long" if len(message) > 60 else "medium"
            
            # Detect emoji usage
            has_emojis = any(ord(char) > 127 for char in message)
            
            learning_updates = {
                "speech_patterns.common_tickers": current_tickers,
                "speech_patterns.question_types": current_question_types,
                "communication_style.message_length": message_length,
                "communication_style.emoji_usage": has_emojis,
                "updated_at": datetime.now(timezone.utc)
            }
            
            result = self.users.update_one(
                {"phone_number": phone_number},
                {"$set": learning_updates}
            )
            
            logger.info(f"📚 Learned from interaction: {phone_number}")
            return result.modified_count > 0
            
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

# Integration with your existing SMS handler
# Add this to your main app.py

user_manager = UserManager()

@app.post("/sms")
async def handle_sms(request: Request):
    """Enhanced SMS handler with proper user management"""
    form = await request.form()
    from_number = form.get('From')
    message_body = form.get('Body', '').strip()
    
    logger.info(f"📱 SMS from {from_number}: {message_body}")
    
    try:
        # Get or create user
        user = user_manager.get_or_create_user(from_number)
        
        # Check message limits
        limit_check = user_manager.check_message_limits(from_number)
        
        if not limit_check["can_send"]:
            # Send limit exceeded message
            limit_message = f"You've reached your {limit_check['plan']} plan limit ({limit_check['used']}/{limit_check['limit']} messages). "
            
            if user.get('plan_type') == 'free':
                limit_message += "Upgrade to paid ($29/month) for 100 messages! Reply UPGRADE"
            elif user.get('plan_type') == 'paid':
                limit_message += "Upgrade to Pro ($99/month) for unlimited! Reply UPGRADE"
            else:
                limit_message += f"Daily cooloff active. Resets in {limit_check.get('resets_in_hours', 24):.1f} hours."
            
            twiml_response = MessagingResponse()
            twiml_response.message(limit_message)
            return Response(content=str(twiml_response), media_type="application/xml")
        
        # Update user activity
        user_manager.update_user_activity(from_number, "received")
        
        # Analyze message intent
        intent = message_analyzer.detect_intent(message_body)
        
        # Learn from interaction
        user_manager.learn_from_interaction(
            from_number, 
            message_body, 
            intent.symbols, 
            intent.action
        )
        
        # Fetch TA data if needed
        ta_data = None
        if intent.symbols and intent.action in ['analyze', 'price', 'technical']:
            symbol = intent.symbols[0]
            ta_data = await ta_client.get_stock_analysis(symbol)
        
        # Generate personalized response
        response_text = await response_generator.generate_personalized_response(
            message_body, intent, ta_data, user
        )
        
        # Send response
        twiml_response = MessagingResponse()
        twiml_response.message(response_text)
        
        # Update sent message count
        user_manager.update_user_activity(from_number, "sent")
        
        logger.info(f"✅ Sent response to {from_number}: {response_text[:50]}...")
        
        return Response(content=str(twiml_response), media_type="application/xml")
        
    except Exception as e:
        logger.error(f"💥 SMS handler error: {e}")
        
        twiml_response = MessagingResponse()
        twiml_response.message("Sorry, I'm having technical difficulties. Please try again in a moment!")
        return Response(content=str(twiml_response), media_type="application/xml")

# Add user management endpoints
@app.get("/admin/users/{phone_number}")
async def get_user_profile(phone_number: str):
    """Get user profile for admin"""
    user = user_manager.get_user_by_phone(phone_number)
    if user:
        # Convert ObjectId to string for JSON serialization
        user["_id"] = str(user["_id"])
        return user
    return {"error": "User not found"}

@app.get("/admin/users/stats")
async def get_user_stats():
    """Get user statistics"""
    return user_manager.get_user_stats()

@app.post("/admin/users/{phone_number}/subscription")
async def update_user_subscription(phone_number: str, plan_data: dict):
    """Update user subscription"""
    success = user_manager.update_subscription(
        phone_number,
        plan_data.get('plan_type'),
        plan_data.get('stripe_customer_id'),
        plan_data.get('stripe_subscription_id')
    )
    
    return {"success": success}
