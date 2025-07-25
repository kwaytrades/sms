# ===== core/message_handler.py =====
from typing import Dict, Optional
import uuid
from datetime import datetime
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
        """Process incoming SMS message"""
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
            
            # Check usage limits
            usage_check = await self._check_usage_limits(user)
            if not usage_check["can_send"]:
                await self._send_limit_message(user, usage_check["reason"])
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
            
            # Update usage
            await self._update_usage(user)
            
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
    
    async def _check_usage_limits(self, user: UserProfile) -> Dict:
        """Check if user can receive messages"""
        plan_config = PlanLimits.get_plan_config()[user.plan_type]
        
        # Check subscription status
        if user.subscription_status not in ["active", "trialing"]:
            return {"can_send": False, "reason": "subscription_inactive"}
        
        # Check period limits
        current_usage = await self.db.get_usage_count(user._id, plan_config.period)
        
        if current_usage >= plan_config.messages_per_period:
            return {"can_send": False, "reason": "limit_reached"}
        
        return {"can_send": True}
    
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
            return f"👋 Welcome {user.name or 'there'}! I'm your AI trading assistant. Send me any trading question or use /help for commands."
        
        elif command == '/help':
            return """🤖 Trading Bot Commands:
/start - Welcome message
/help - Show this help
/portfolio - View your portfolio
/market - Get market updates
/subscribe - Upgrade your plan

Just send me any trading question and I'll help! 📈"""
        
        elif command == '/portfolio':
            return "📊 Portfolio feature coming soon! For now, ask me any trading questions."
        
        elif command == '/market':
            return "📈 Market updates feature coming soon! Ask me about specific stocks or crypto."
        
        elif command == '/subscribe':
            return "💎 Premium plans coming soon! Currently testing with unlimited access."
        
        else:
            return f"❓ Unknown command: {command}. Use /help to see available commands."
    
    async def _get_conversation_history(self, session_id: str) -> list:
        """Get conversation history for context"""
        try:
            # This would fetch recent messages from the session
            # For now, return empty list
            return []
        except Exception as e:
            logger.error(f"❌ Error getting conversation history: {e}")
            return []
    
    async def _send_limit_message(self, user: UserProfile, reason: str):
        """Send usage limit message"""
        if reason == "subscription_inactive":
            message = "⚠️ Your subscription is inactive. Please reactivate to continue using the service."
        elif reason == "limit_reached":
            message = "📈 You've reached your message limit for this period. Upgrade your plan for unlimited access!"
        else:
            message = "⚠️ Usage limit reached. Please check your plan or upgrade for more access."
        
        await self.twilio.send_sms(user.phone_number, message)
    
    async def _update_usage(self, user: UserProfile):
        """Update user's usage count"""
        try:
            plan_config = PlanLimits.get_plan_config()[user.plan_type]
            period = plan_config.period
            ttl = 24 * 3600 if period == "daily" else 30 * 24 * 3600  # 1 day or 30 days
            
            await self.db.increment_usage(user._id, period, ttl)
        except Exception as e:
            logger.error(f"❌ Error updating usage: {e}")
