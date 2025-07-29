# ===== core/message_handler.py - MINIMAL UPDATES FOR ORCHESTRATOR =====
from typing import Dict, Optional
from datetime import datetime, timedelta, timezone
from loguru import logger
from models.user import UserProfile

class MessageHandler:
    def __init__(self, db_service, openai_service, twilio_service, user_manager=None, message_processor=None):
        self.db = db_service
        self.openai = openai_service
        self.twilio = twilio_service
        self.user_manager = user_manager
        self.message_processor = message_processor  # NEW: Add orchestrator processor
        logger.info("✅ Enhanced message handler initialized with context support")
    
    async def process_incoming_message(self, phone_number: str, message_body: str) -> bool:
        """Process incoming SMS message with rich conversation context"""
        try:
            logger.info(f"📱 Processing message from {phone_number}: {message_body}")
            
            # Get or create user (existing functionality)
            user = await self._get_or_create_user(phone_number)
            
            # Check limits (existing functionality)
            usage_check = await self._check_weekly_limits(user)
            if not usage_check["can_send"]:
                await self._send_limit_message(user, usage_check)
                return True
            
            # NEW: Use orchestrator if available, otherwise use existing logic
            if self.message_processor:
                # Use new orchestrator processor
                response = await self.message_processor.process_message(
                    message_body, phone_number
                )
            else:
                # Fallback to existing enhanced response generation
                conversation_context = await self._get_conversation_context(phone_number)
                response = await self._generate_context_aware_response(user, message_body, conversation_context)
                # Update conversation context for existing flow
                await self._update_conversation_context(phone_number, message_body, response)
            
            # Send response (existing functionality)
            await self.twilio.send_sms(phone_number, response)
            
            # Update usage (existing functionality)
            await self._update_weekly_usage(user)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing message from {phone_number}: {e}")
            return False
    
    # ALL EXISTING METHODS REMAIN UNCHANGED
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
    
    # EXISTING CONTEXT METHODS REMAIN UNCHANGED
    async def _get_conversation_context(self, phone_number: str) -> Dict:
        """Get conversation context for enhanced response generation"""
        try:
            # Try database service first (new enhanced context)
            if self.db and hasattr(self.db, 'get_conversation_context'):
                return await self.db.get_conversation_context(phone_number)
            
            # Fallback to user manager context (if available)
            elif self.user_manager and hasattr(self.user_manager, 'get_conversation_context'):
                return self.user_manager.get_conversation_context(phone_number)
            
            # No context available
            else:
                return self._get_default_context()
                
        except Exception as e:
            logger.warning(f"Failed to get conversation context for {phone_number}: {e}")
            return self._get_default_context()
    
    def _get_default_context(self) -> Dict:
        """Default context when context services unavailable"""
        return {
            "conversation_context": {
                "recent_symbols": [],
                "recent_topics": [],
                "pending_decisions": [],
                "relationship_stage": "new",
                "total_conversations": 0,
                "conversation_frequency": "occasional"
            },
            "today_session": {
                "message_count": 0,
                "topics_discussed": [],
                "symbols_mentioned": [],
                "session_mood": "neutral",
                "is_first_message_today": True
            },
            "recent_messages": [],
            "context_summary": "No conversation history available"
        }
    
    async def _generate_context_aware_response(self, user: UserProfile, message: str, conversation_context: Dict) -> str:
        """Generate response with conversation context awareness"""
        try:
            # Check if we have enhanced LLM agent with context support
            if hasattr(self.openai, 'generate_context_aware_response'):
                return await self.openai.generate_context_aware_response(
                    user_query=message,
                    user_profile=user.__dict__ if hasattr(user, '__dict__') else {},
                    conversation_context=conversation_context
                )
            
            # Check if we have personality-aware response generation
            elif hasattr(self.openai, 'generate_personalized_response'):
                # Enhanced call with context information
                user_profile_dict = user.__dict__ if hasattr(user, '__dict__') else {}
                
                # Enhance user profile with conversation context
                if conversation_context:
                    user_profile_dict['conversation_context'] = conversation_context
                
                return await self.openai.generate_personalized_response(
                    user_query=message,
                    user_profile=user_profile_dict,
                    conversation_history=conversation_context.get('recent_messages', [])
                )
            
            # Fallback to basic response generation
            else:
                return await self._generate_basic_response(user, message, conversation_context)
                
        except Exception as e:
            logger.error(f"❌ Error generating context-aware response: {e}")
            return self._generate_fallback_response(conversation_context)
    
    async def _generate_basic_response(self, user: UserProfile, message: str, conversation_context: Dict) -> str:
        """Generate basic response with available context"""
        try:
            # Build context-aware prompt
            context_info = ""
            if conversation_context:
                conv_ctx = conversation_context.get("conversation_context", {})
                today_session = conversation_context.get("today_session", {})
                
                # Add relationship context
                relationship = conv_ctx.get("relationship_stage", "new")
                total_convos = conv_ctx.get("total_conversations", 0)
                context_info += f"Relationship: {relationship} ({total_convos} conversations). "
                
                # Add recent context
                recent_symbols = conv_ctx.get("recent_symbols", [])
                if recent_symbols:
                    context_info += f"Recently discussed: {', '.join(recent_symbols[-3:])}. "
                
                # Add today's context
                if not today_session.get("is_first_message_today", True):
                    context_info += f"Continuing today's conversation. "
                else:
                    context_info += f"First message today. "
            
            # Enhanced prompt with context
            enhanced_prompt = f"""Respond as a knowledgeable trading buddy.

Context: {context_info}

User message: "{message}"

Instructions:
- Be conversational and helpful
- Reference conversation history naturally if relevant
- Match their communication style
- Provide valuable trading insights
- Keep response under 300 characters for SMS
"""
            
            # Use basic OpenAI generation if available
            if hasattr(self.openai, 'client'):
                response = await self.openai.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": enhanced_prompt}],
                    temperature=0.7,
                    max_tokens=150
                )
                return response.choices[0].message.content.strip()
            
            # Fallback response
            else:
                return "I'm here to help with your trading questions! Can you tell me more about what you're looking for?"
                
        except Exception as e:
            logger.error(f"❌ Error in basic response generation: {e}")
            return "I'm having trouble processing your request. Please try again."
    
    def _generate_fallback_response(self, conversation_context: Dict) -> str:
        """Generate fallback response when all else fails"""
        # Use context to determine appropriate fallback
        if conversation_context:
            relationship = conversation_context.get("conversation_context", {}).get("relationship_stage", "new")
            
            if relationship == "new":
                return "Hey there! I'm having some technical issues but I'm here to help with your trading questions."
            elif relationship in ["getting_acquainted", "building_trust"]:
                return "Hey! Running into some tech issues on my end but still here to help. What's on your mind?"
            else:
                return "yo tech gremlins acting up but I got you! what you need help with?"
        else:
            return "I'm having trouble processing your request. Please try again."
    
    async def _update_conversation_context(self, phone_number: str, user_message: str, bot_response: str):
        """Update conversation context after successful interaction"""
        try:
            # Update through database service if available
            if self.db and hasattr(self.db, 'save_enhanced_message'):
                await self.db.save_enhanced_message(
                    phone_number=phone_number,
                    user_message=user_message,
                    bot_response=bot_response,
                    intent_data={"intent": "general"},  # Basic intent for now
                    symbols=[],
                    context_used="message_handler_basic"
                )
            
            # Update through user manager if available
            elif self.user_manager:
                # Extract any symbols from the message for learning
                symbols = self._extract_basic_symbols(user_message)
                
                # Update user learning
                if hasattr(self.user_manager, 'learn_from_interaction'):
                    self.user_manager.learn_from_interaction(
                        phone_number=phone_number,
                        message=user_message,
                        symbols=symbols,
                        intent="general"
                    )
                
                # Update activity
                if hasattr(self.user_manager, 'update_user_activity'):
                    self.user_manager.update_user_activity(phone_number, "received")
                
                # Log conversation
                if hasattr(self.user_manager, 'log_conversation'):
                    self.user_manager.log_conversation(
                        phone_number=phone_number,
                        message=user_message,
                        response=bot_response,
                        intent="general",
                        symbols=symbols
                    )
            
        except Exception as e:
            logger.warning(f"Failed to update conversation context for {phone_number}: {e}")
    
    def _extract_basic_symbols(self, message: str) -> list:
        """Extract basic stock symbols from message"""
        try:
            # Simple symbol extraction
            words = message.upper().split()
            symbols = []
            
            # Look for obvious ticker patterns (2-5 uppercase letters)
            for word in words:
                cleaned_word = ''.join(c for c in word if c.isalpha())
                if 2 <= len(cleaned_word) <= 5 and cleaned_word.isupper():
                    symbols.append(cleaned_word)
            
            # Basic company name mapping
            company_mappings = {
                'google': 'GOOGL', 'tesla': 'TSLA', 'apple': 'AAPL',
                'microsoft': 'MSFT', 'amazon': 'AMZN', 'meta': 'META',
                'nvidia': 'NVDA', 'netflix': 'NFLX'
            }
            
            message_lower = message.lower()
            for company, symbol in company_mappings.items():
                if company in message_lower and symbol not in symbols:
                    symbols.append(symbol)
            
            return symbols[:3]  # Limit to 3 symbols max
            
        except Exception as e:
            logger.warning(f"Failed to extract symbols from message: {e}")
            return []
