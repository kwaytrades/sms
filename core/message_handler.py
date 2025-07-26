
# ===== core/message_handler.py - MINIMAL VERSION =====
from typing import Dict, Optional
from datetime import datetime, timedelta, timezone
from loguru import logger
from models.user import UserProfile

class MessageHandler:
    def __init__(self, db_service, openai_service, twilio_service):
        self.db = db_service
        self.openai = openai_service
        self.twilio = twilio_service
        logger.info("✅ Message handler initialized")
    
    async def process_incoming_message(self, phone_number: str, message_body: str) -> bool:
        """Process incoming SMS message"""
        try:
            logger.info(f"📱 Processing message from {phone_number}: {message_body}")
            
            # Get or create user
            user = await self._get_or_create_user(phone_number)
            
            # Check limits
            usage_check = await self._check_weekly_limits(user)
            if not usage_check["can_send"]:
                await self._send_limit_message(user, usage_check)
                return True
            
            # Generate response
            response = await self._generate_response(user, message_body)
            
            # Send response
            await self.twilio.send_sms(phone_number, response)
            
            # Update usage
            await self._update_weekly_usage(user)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing message from {phone_number}: {e}")
            return False
    
    async def _get_or_create_user(self, phone_number: str) -> UserProfile:
        """Get existing user or create new one"""
        if self.db:
            user = await self.db.get_user_by_phone(phone_number)
            if user:
                return user
            
            # Create new user
            new_user = UserProfile(
                phone_number=phone_number,
                plan_type="free",
                subscription_status="trialing"
            )
            user_id = await self.db.save_user(new_user)
            new_user._id = user_id
            return new_user
        else:
            # Return mock user if no database
            return UserProfile(
                phone_number=phone_number,
                plan_type="free",
                subscription_status="trialing"
            )
    
    async def _check_weekly_limits(self, user: UserProfile) -> Dict:
        """Check weekly usage limits"""
        limits = {"free": 10, "paid": 100, "pro": 999999}
        limit = limits.get(user.plan_type, 10)
        
        # Mock usage check
        current_usage = user.messages_this_period if hasattr(user, 'messages_this_period') else 0
        
        return {
            "can_send": current_usage < limit,
            "limit": limit,
            "used": current_usage,
            "remaining": max(0, limit - current_usage)
        }
    
    async def _generate_response(self, user: UserProfile, message: str) -> str:
        """Generate response to user message"""
        try:
            return await self.openai.generate_personalized_response(
                user_query=message,
                user_profile=user.__dict__ if hasattr(user, '__dict__') else {},
                conversation_history=[]
            )
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return "I'm having trouble processing your request. Please try again."
    
    async def _send_limit_message(self, user: UserProfile, usage_info: Dict):
        """Send limit reached message"""
        plan = user.plan_type.upper()
        used = usage_info["used"]
        limit = usage_info["limit"]
        
        message = f"You've reached your {plan} plan limit ({used}/{limit} messages this week). "
        
        if user.plan_type == "free":
            message += "Upgrade to PAID for 100 messages/month! Reply UPGRADE"
        elif user.plan_type == "paid":
            message += "Upgrade to PRO for unlimited! Reply UPGRADE"
        else:
            message += "Limits reset Monday 9:30 AM EST."
        
        await self.twilio.send_sms(user.phone_number, message)
    
    async def _update_weekly_usage(self, user: UserProfile):
        """Update weekly usage counter"""
        try:
            if self.db:
                await self.db.update_user_activity(user.phone_number, "received")
        except Exception as e:
            logger.error(f"❌ Error updating usage: {e}")
