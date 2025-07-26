# ===== core/message_handler.py - FIXED ASYNC MESSAGE HANDLER =====
from typing import Dict, Optional, List
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
        """Process incoming SMS message with unified async database operations"""
        try:
            # Validate phone number
            phone_number = validate_phone_number(phone_number)
            
            # Get or create user (NOW ASYNC)
            user = await self.db.get_or_create_user(phone_number)
            
            # Check if it's a command
            command = extract_command(message_body)
            if command:
                await self._handle_command(user, command)
                return True
            
            # Check message limits (NOW ASYNC)
            usage_check = await self.db.check_message_limits(phone_number)
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
            
            # Update user activity and learning (NOW ASYNC)
            await self.db.update_user_activity(phone_number, "received")
            await self._update_usage_tracking(user)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing message from {phone_number}: {e}")
            return False
    
    async def _update_usage_tracking(self, user: UserProfile):
        """Update usage tracking with smart warnings (ASYNC)"""
        try:
            # Weekly TTL: 7 days in seconds
            weekly_ttl = 7 * 24 * 3600
            await self.db.increment_usage(user._id, "weekly", weekly_ttl)
            
            # Get updated usage count
            current_usage = await self.db.get_usage_count(user._id, "weekly")
            plan_limits = user.plan_limits
            limit = plan_limits.messages_per_period
            
            # Send low usage warnings
            await self._check_and_send_usage_warning(user, current_usage, limit)
            
            logger.info(f"📊 User {user.phone_number} usage: {current_usage}/{limit} {plan_limits.period}")
            
        except Exception as e:
            logger.error(f"❌ Error updating usage tracking: {e}")
    
    async def _check_and_send_usage_warning(self, user: UserProfile, current_usage: int, limit: int):
        """Send usage warnings at specific thresholds (ASYNC)"""
        try:
            if user.has_unlimited_messages:
                return  # No warnings for unlimited users
            
            # Calculate warning thresholds
            warning_thresholds = {
                "75_percent": int(limit * 0.75),
                "90_percent": int(limit * 0.9),
                "last_message": limit - 1
            }
            
            # Check if we should send a warning
            if current_usage == warning_thresholds["75_percent"]:
                message = f"""⚠️ Usage Alert: {current_usage}/{limit} messages used this {user.plan_limits.period}.

{limit - current_usage} insights remaining until reset.

Running low? Upgrade for more weekly analysis:
[View Plans]"""
                
                await self.twilio.send_sms(user.phone_number, message)
                
            elif current_usage == warning_thresholds["90_percent"]:
                message = f"""🚨 Almost out: {current_usage}/{limit} messages used.

Only {limit - current_usage} insights left this {user.plan_limits.period}!

Don't get caught without analysis - upgrade now:
📈 {user.plan_limits.period.title()}: {limit} messages | 💎 Upgrade available
[Upgrade Now]"""
                
                await self.twilio.send_sms(user.phone_number, message)
                
            elif current_usage == warning_thresholds["last_message"]:
                reset_info = self._get_reset_info(user)
                
                message = f"""🔥 FINAL message this {user.plan_limits.period}!

Next insight resets {reset_info}.

Can't wait? Upgrade for immediate access:
[Upgrade Now] [View Plans]"""
                
                await self.twilio.send_sms(user.phone_number, message)
                
        except Exception as e:
            logger.error(f"❌ Error sending usage warning: {e}")
    
    def _get_reset_info(self, user: UserProfile) -> str:
        """Get reset time information for user's plan"""
        if user.plan_limits.period == "weekly":
            # Find next Monday 9:30 AM EST
            now = datetime.now(timezone.utc)
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0 and now.hour >= 14 and now.minute >= 30:  # 9:30 AM EST = 14:30 UTC
                days_until_monday = 7
            
            next_monday = now + timedelta(days=days_until_monday)
            reset_time = next_monday.replace(hour=14, minute=30, second=0, microsecond=0)  # 9:30 AM EST
            return reset_time.strftime("%A %I:%M %p EST")
            
        elif user.plan_limits.period == "monthly":
            # Find next month
            now = datetime.now(timezone.utc)
            if now.month == 12:
                next_month = now.replace(year=now.year + 1, month=1, day=1)
            else:
                next_month = now.replace(month=now.month + 1, day=1)
            return next_month.strftime("%B %d")
            
        return "next period"
    
    async def _generate_response(self, user: UserProfile, message: str, session_id: str) -> str:
        """Generate personalized AI response (ASYNC)"""
        try:
            # Get conversation history
            conversation_history = await self._get_conversation_history(session_id)
            
            # Extract symbols and intent for learning
            symbols = self._extract_symbols(message)
            intent = self._classify_intent(message)
            
            # Learn from interaction (NOW ASYNC)
            await self.db.learn_from_interaction(user.phone_number, message, symbols, intent)
            
            # Generate response using OpenAI
            response = await self.openai.generate_personalized_response(
                user_query=message,
                user_profile=user.to_dict(),
                conversation_history=conversation_history
            )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return "I'm having trouble processing your request. Please try again."
    
    def _extract_symbols(self, message: str) -> List[str]:
        """Extract stock symbols from message"""
        import re
        # Simple regex to find potential stock symbols (1-5 uppercase letters)
        symbols = re.findall(r'\b[A-Z]{1,5}\b', message.upper())
        
        # Filter out common false positives
        false_positives = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BY', 'DO', 'GET', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'WAY', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
        
        return [s for s in symbols if s not in false_positives and len(s) >= 2]
    
    def _classify_intent(self, message: str) -> str:
        """Classify message intent"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['price', 'cost', 'trading at', 'current']):
            return 'price'
        elif any(word in message_lower for word in ['analyze', 'analysis', 'look at', 'check']):
            return 'analyze'
        elif any(word in message_lower for word in ['find', 'search', 'recommend', 'suggest']):
            return 'screener'
        elif any(word in message_lower for word in ['news', 'updates', 'happened']):
            return 'news'
        else:
            return 'general'
    
    async def _handle_command(self, user: UserProfile, command: str):
        """Handle SMS commands (ASYNC)"""
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
        """Handle basic commands as fallback (ASYNC)"""
        command = command.lower().strip()
        
        if command == '/start':
            return f"Welcome {user.name}! I'm your AI trading assistant. Send me any trading question or use /help for commands."
        
        elif command == '/help':
            return """Trading Bot Commands:
/start - Welcome message
/help - Show this help  
/usage - Check usage
/upgrade - View plans
/status - Account status

Send any trading question for AI analysis!"""
        
        elif command == '/usage':
            usage_pct = user.get_usage_percentage()
            remaining = user.get_messages_remaining()
            
            return f"""📊 Usage: {user.messages_this_period}/{user.plan_limits.messages_per_period} messages ({usage_pct:.1f}%)

Remaining: {remaining} messages
Plan: {user.plan_type.title()} ({user.plan_limits.period})
Resets: {self._get_reset_info(user)}

Need more? /upgrade for higher limits!"""
        
        elif command == '/status':
            return f"""📊 Account Status:

Plan: {user.plan_type.title()} (${user.plan_limits.price}/month)
Status: {user.subscription_status.title()}
Usage: {user.messages_this_period}/{user.plan_limits.messages_per_period} messages
Connected accounts: {len(user.connected_accounts)}

Manage: /upgrade | /help"""
        
        elif command == '/upgrade':
            config = PlanLimits.get_plan_config()
            return f"""💎 Upgrade Plans:

📈 Paid (${config['paid'].price}/month): {config['paid'].messages_per_period} messages/{config['paid'].period}
💎 Pro (${config['pro'].price}/month): Unlimited messages + alerts

[Upgrade Now] [Learn More]"""
        
        else:
            return f"Unknown command: {command}. Use /help to see available commands."
    
    async def _get_conversation_history(self, session_id: str) -> list:
        """Get conversation history for context (ASYNC)"""
        try:
            # This would fetch recent messages from the session
            # For now, return empty list as the full implementation would require
            # querying the conversations collection
            return []
        except Exception as e:
            logger.error(f"❌ Error getting conversation history: {e}")
            return []
    
    async def _send_limit_message(self, user: UserProfile, usage_info: Dict):
        """Send appropriate limit reached message (ASYNC)"""
        reason = usage_info["reason"]
        
        if reason == "subscription_inactive":
            message = """⚠️ Your subscription is inactive. 

Reactivate now to continue getting AI trading insights:
[Reactivate Account]"""
            
        elif reason in ["weekly_limit_reached", "Limit exceeded"]:
            reset_info = self._get_reset_info(user)
            
            if user.plan_type == "free":
                message = f"⚡ Weekly limit hit! Resets {reset_info} or upgrade for 10x more: [Upgrade Now]"
            elif user.plan_type == "paid":
                message = f"⚡ Monthly limit hit! Resets {reset_info} or upgrade to Pro for unlimited: [Upgrade]"
            else:  # Pro with daily cooloff
                message = f"⚡ Daily cooloff active! Resets {reset_info}. You're on our highest tier."
        
        elif reason == "Daily cooloff active":
            message = f"⚡ Pro daily cooloff active (50+ messages today). Resets in a few hours. You're on unlimited!"
        
        else:
            message = "Message limit reached. Please upgrade or wait for reset."
        
        await self.twilio.send_sms(user.phone_number, message)
