# ===== core/message_handler.py =====
from typing import Dict, Optional
import uuid
from datetime import datetime, timedelta, timezone
from loguru import logger

from models.user import UserProfile, PlanLimits
from models.conversation import ChatMessage
from services.database import DatabaseService
from services.openai_service import OpenAIService
from services.twilio_service import TwilioService
from utils.validators import validate_phone_number
from utils.helpers import extract_command, generate_session_id

class MessageHandler:
    def __init__(self, db: DatabaseService, openai: OpenAIService, twilio: TwilioService):
        self.db = db
        self.openai = openai
        self.twilio = twilio
    
    async def process_incoming_message(self, phone_number: str, message_body: str) -> bool:
        """Process incoming SMS message with weekly rate limiting"""
        try:
            # Validate phone number
            phone_number = validate_phone_number(phone_number)
            
            # Get or create user
            user = await self._get_or_create_user(phone_number)
            
            # Check if it's a command
            command = extract_command(message_body)
            if command:
                await self._handle_command(user, command)
                return True
            
            # Check weekly usage limits
            usage_check = await self._check_weekly_limits(user)
            if not usage_check["can_send"]:
                await self._send_limit_message(user, usage_check)
                return True
            
            # Save incoming message
            session_id = generate_session_id(user._id)
            incoming_msg = ChatMessage(
                user_id=user._id,
                content=message_body,
                direction="inbound",
                message_type="user_query",
                session_id=session_id
            )
            await self.db.save_message(incoming_msg)
            
            # Generate AI response
            response = await self._generate_response(user, message_body, session_id)
            
            # Send response
            await self.twilio.send_sms(phone_number, response)
            
            # Save outgoing message
            outgoing_msg = ChatMessage(
                user_id=user._id,
                content=response,
                direction="outbound",
                message_type="bot_response",
                session_id=session_id
            )
            await self.db.save_message(outgoing_msg)
            
            # Update weekly usage with smart warnings
            await self._update_weekly_usage(user)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing message from {phone_number}: {e}")
            return False
    
    async def _get_or_create_user(self, phone_number: str) -> UserProfile:
        """Get existing user or create new one"""
        user = await self.db.get_user_by_phone(phone_number)
        
        if not user:
            # Create new user with trial
            user = UserProfile(
                phone_number=phone_number,
                plan_type="free",
                subscription_status="trialing"
            )
            user._id = await self.db.save_user(user)
            logger.info(f"✅ Created new user: {phone_number}")
        
        return user
    
    async def _check_weekly_limits(self, user: UserProfile) -> Dict:
        """Check weekly usage limits with detailed response"""
        plan_limits = self._get_plan_limits(user.plan_type)
        
        # Check subscription status
        if user.subscription_status not in ["active", "trialing"]:
            return {
                "can_send": False, 
                "reason": "subscription_inactive",
                "reset_date": None,
                "current_usage": 0,
                "limit": 0
            }
        
        # Get current weekly usage
        current_usage = await self.db.get_usage_count(user._id, "weekly")
        
        # Calculate next Monday 9:30 AM EST reset
        reset_date = self._get_next_weekly_reset()
        
        if current_usage >= plan_limits["weekly_limit"]:
            return {
                "can_send": False, 
                "reason": "weekly_limit_reached",
                "reset_date": reset_date,
                "current_usage": current_usage,
                "limit": plan_limits["weekly_limit"]
            }
        
        return {
            "can_send": True,
            "reset_date": reset_date,
            "current_usage": current_usage,
            "limit": plan_limits["weekly_limit"]
        }
    
    def _get_plan_limits(self, plan_type: str) -> Dict:
        """Get weekly limits for each plan"""
        limits = {
            "free": {"weekly_limit": 4, "multiplier": "10x"},
            "standard": {"weekly_limit": 40, "multiplier": "3x"}, 
            "vip": {"weekly_limit": 120, "multiplier": "unlimited"}
        }
        return limits.get(plan_type, limits["free"])
    
    def _get_next_weekly_reset(self) -> datetime:
        """Calculate next Monday 9:30 AM EST reset"""
        # EST timezone (UTC-5, or UTC-4 during DST)
        est = timezone(timedelta(hours=-5))  # Simplified - you may want proper timezone handling
        
        now = datetime.now(est)
        
        # Find next Monday
        days_until_monday = (7 - now.weekday()) % 7
        if days_until_monday == 0 and now.hour >= 9 and now.minute >= 30:
            days_until_monday = 7  # If it's Monday after 9:30 AM, go to next Monday
        
        next_monday = now + timedelta(days=days_until_monday)
        reset_time = next_monday.replace(hour=9, minute=30, second=0, microsecond=0)
        
        return reset_time
    
    async def _generate_response(self, user: UserProfile, message: str, session_id: str) -> str:
        """Generate personalized AI response"""
        try:
            # Get conversation history
            conversation_history = await self._get_conversation_history(session_id)
            
            # Generate response using OpenAI
            response = await self.openai.generate_personalized_response(
                user_query=message,
                user_profile=user.__dict__,
                conversation_history=conversation_history
            )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return "I'm having trouble processing your request. Please try again."
    
    async def _handle_command(self, user: UserProfile, command: str):
        """Handle SMS commands"""
        try:
            # Try to use enhanced command handler if available
            try:
                from core.enhanced_command_handler import EnhancedCommandHandler
                from services.stripe_integration import StripeIntegrationService
                
                stripe_service = StripeIntegrationService()
                enhanced_handler = EnhancedCommandHandler(self.db, self.twilio, stripe_service)
                
                return await enhanced_handler.handle_command(user, command, user.phone_number)
                
            except ImportError:
                # Fallback to basic command handling
                response = await self._handle_basic_command(user, command)
                await self.twilio.send_sms(user.phone_number, response)
                return True
                
        except Exception as e:
            logger.error(f"❌ Command handling failed: {e}")
            await self.twilio.send_sms(user.phone_number, "Sorry, something went wrong. Please try again.")
            return False
    
    async def _handle_basic_command(self, user: UserProfile, command: str) -> str:
        """Handle basic commands as fallback"""
        command = command.lower().strip()
        
        if command == '/start':
            return f"Welcome {user.name or 'there'}! I'm your AI trading assistant. Send me any trading question or use /help for commands."
        
        elif command == '/help':
            return """Trading Bot Commands:
/start - Welcome message
/help - Show this help  
/usage - Check weekly usage
/upgrade - View plans

Send any trading question for AI analysis!"""
        
        elif command == '/usage':
            usage_info = await self._check_weekly_limits(user)
            reset_str = usage_info["reset_date"].strftime("%A %I:%M %p EST")
            
            return f"""📊 Weekly Usage: {usage_info['current_usage']}/{usage_info['limit']} messages

Resets: {reset_str}
Plan: {user.plan_type.title()}

Need more? /upgrade for higher limits!"""
        
        elif command == '/upgrade':
            return """💎 Upgrade Plans:

📈 Standard ($29/mo): 40 msgs/week
💎 VIP ($99/mo): 120 msgs/week + priority

[Upgrade Now] [Learn More]"""
        
        elif command == '/portfolio':
            return "📊 Portfolio feature coming soon! For now, ask me any trading questions."
        
        elif command == '/market':
            return "📈 Market updates feature coming soon! Ask me about specific stocks or crypto."
        
        elif command == '/subscribe':
            return "💎 Premium plans coming soon! Currently testing with unlimited access."
        
        else:
            return f"Unknown command: {command}. Use /help to see available commands."
    
    async def _get_conversation_history(self, session_id: str) -> list:
        """Get conversation history for context"""
        try:
            # This would fetch recent messages from the session
            # For now, return empty list
            return []
        except Exception as e:
            logger.error(f"❌ Error getting conversation history: {e}")
            return []
    
    async def _send_limit_message(self, user: UserProfile, usage_info: Dict):
        """Send appropriate limit reached message"""
        reason = usage_info["reason"]
        
        if reason == "subscription_inactive":
            message = """⚠️ Your subscription is inactive. 

Reactivate now to continue getting AI trading insights:
[Reactivate Account]"""
            
        elif reason == "weekly_limit_reached":
            reset_date = usage_info["reset_date"]
            plan_limits = self._get_plan_limits(user.plan_type)
            
            # Format reset date
            reset_str = reset_date.strftime("%A %I:%M %p EST")
            
            if user.plan_type == "free":
                message = f"""⚡ Weekly limit reached!

Your AI trading insights reset {reset_str}.

Can't wait? Upgrade now for {plan_limits['multiplier']} more daily analysis:

📈 Standard: 40 msgs/week  
💎 VIP: 120 msgs/week + priority alerts

[Upgrade Now] [View Plans]"""
            
            else:  # Standard or VIP users
                message = f"""⚡ Weekly limit reached!

Your {user.plan_type.title()} plan resets {reset_str}.

Need more insights? Upgrade to VIP for {plan_limits['multiplier']} analysis:

💎 VIP: 120 msgs/week + priority features

[Upgrade to VIP] [View Plans]"""
        
        await self.twilio.send_sms(user.phone_number, message)
    
    async def _update_weekly_usage(self, user: UserProfile):
        """Update weekly usage counter with smart warnings"""
        try:
            # Weekly TTL: 7 days in seconds
            weekly_ttl = 7 * 24 * 3600
            await self.db.increment_usage(user._id, "weekly", weekly_ttl)
            
            # Get updated usage count
            current_usage = await self.db.get_usage_count(user._id, "weekly")
            plan_limits = self._get_plan_limits(user.plan_type)
            limit = plan_limits['weekly_limit']
            
            # Send low usage warnings
            await self._check_and_send_usage_warning(user, current_usage, limit)
            
            logger.info(f"📊 User {user.phone_number} usage: {current_usage}/{limit} weekly")
            
        except Exception as e:
            logger.error(f"❌ Error updating weekly usage: {e}")
    
    async def _check_and_send_usage_warning(self, user: UserProfile, current_usage: int, limit: int):
        """Send usage warnings at specific thresholds"""
        try:
            # Calculate warning thresholds
            warning_thresholds = {
                "75_percent": int(limit * 0.75),
                "90_percent": int(limit * 0.9),
                "last_message": limit - 1
            }
            
            # Check if we should send a warning
            warning_sent_key = f"warning_sent:{user._id}:{current_usage}"
            
            if current_usage == warning_thresholds["75_percent"]:
                message = f"""⚠️ Usage Alert: {current_usage}/{limit} messages used this week.

{limit - current_usage} insights remaining until Monday reset.

Running low? Upgrade for 3x more weekly analysis:
[View Plans]"""
                
                await self.twilio.send_sms(user.phone_number, message)
                # Note: You'll need to implement set_warning_flag in database service
                # await self.db.set_warning_flag(warning_sent_key, "75_percent")
                
            elif current_usage == warning_thresholds["90_percent"]:
                message = f"""🚨 Almost out: {current_usage}/{limit} messages used.

Only {limit - current_usage} insights left this week!

Don't get caught without analysis - upgrade now:
📈 Standard: 40/week | 💎 VIP: 120/week
[Upgrade Now]"""
                
                await self.twilio.send_sms(user.phone_number, message)
                # await self.db.set_warning_flag(warning_sent_key, "90_percent")
                
            elif current_usage == warning_thresholds["last_message"]:
                reset_date = self._get_next_weekly_reset()
                reset_str = reset_date.strftime("%A %I:%M %p EST")
                
                message = f"""🔥 FINAL message this week!

Next insight resets {reset_str}.

Can't wait? Upgrade for immediate access:
[Upgrade Now] [View Plans]"""
                
                await self.twilio.send_sms(user.phone_number, message)
                # await self.db.set_warning_flag(warning_sent_key, "final")
                
        except Exception as e:
            logger.error(f"❌ Error sending usage warning: {e}")
