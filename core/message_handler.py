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
    
    # In your existing message_handler.py, replace _handle_command with:
async def _handle_command(self, user: UserProfile, command: str):
    from core.enhanced_command_handler import EnhancedCommandHandler
    from services.stripe_integration import StripeIntegrationService
    
    stripe_service = StripeIntegrationService()
    enhanced_handler = EnhancedCommandHandler(self.db, self.twilio, stripe_service)
    
    return await enhanced_handler.handle_command(user, command, user.phone_number)
