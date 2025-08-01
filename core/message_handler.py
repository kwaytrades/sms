# ===== core/message_handler.py - MINIMAL UPDATES FOR ORCHESTRATOR =====
from typing import Dict, Optional
from datetime import datetime, timedelta, timezone
from loguru import logger
from models.user import UserProfile

class MessageHandler:
    def __init__(self, db_service, claude_service, twilio_service, user_manager=None, message_processor=None):
        self.db = db_service
        self.claude = claude_service
        self.twilio = twilio_service
        self.user_manager = user_manager
        self.message_processor = message_processor  # NEW: Add orchestrator processor
        logger.info("✅ Enhanced message handler initialized with Claude and context support")
    
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
            
            # Send response (FIXED METHOD NAME)
            await self.twilio.send_message(phone_number, response)  # ✅ FIXED: send_sms → send_message
            
            # Update usage (existing functionality)
            await self._update_weekly_usage(user)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing message from {phone_number}: {e}")
            return False
    
    # ALL EXISTING METHODS REMAIN UNCHANGED
    async def _get_or_create_user(self, phone_number: str) -> UserProfile:
        """Get existing user or create new one using KeyBuilder"""
        if self.db:
            # Try KeyBuilder first for user lookup
            if hasattr(self.db, 'key_builder'):
                try:
                    user_profile = await self.db.key_builder.get_user_profile(phone_number)
                    if user_profile:
                        # Convert dict to UserProfile object if needed
                        if isinstance(user_profile, dict):
                            return UserProfile(**user_profile)
                        return user_profile
                except Exception as e:
                    logger.warning(f"KeyBuilder user lookup failed: {e}")
            
            # Fallback to existing database method
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
        """Check weekly usage limits using KeyBuilder"""
        limits = {"free": 10, "paid": 100, "pro": 999999}
        limit = limits.get(user.plan_type, 10)
        
        try:
            current_usage = 0
            
            # Try KeyBuilder for usage data
            if self.db and hasattr(self.db, 'key_builder'):
                try:
                    usage_data = await self.db.key_builder.get_user_usage(user.phone_number, "weekly")
                    current_usage = usage_data.get("message_count", 0) if usage_data else 0
                except Exception as e:
                    logger.warning(f"KeyBuilder usage lookup failed: {e}")
            
            # Fallback to user object attribute
            if current_usage == 0:
                current_usage = user.messages_this_period if hasattr(user, 'messages_this_period') else 0
            
            return {
                "can_send": current_usage < limit,
                "limit": limit,
                "used": current_usage,
                "remaining": max(0, limit - current_usage)
            }
            
        except Exception as e:
            logger.warning(f"Usage check failed: {e}")
            # Safe fallback
            return {"can_send": True, "limit": limit, "used": 0, "remaining": limit}
    
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
        
        await self.twilio.send_message(user.phone_number, message)  # ✅ FIXED: send_sms → send_message
    
    async def _update_weekly_usage(self, user: UserProfile):
        """Update weekly usage counter with FIXED method call"""
        try:
            if self.db:
                # FIXED: Use correct method signature - only phone_number argument
                await self.db.update_user_activity(user.phone_number)
        except Exception as e:
            logger.error(f"❌ Error updating usage: {e}")
    
    # EXISTING CONTEXT METHODS WITH KEYBUILDER INTEGRATION
    async def _get_conversation_context(self, phone_number: str) -> Dict:
        """Get conversation context for enhanced response generation using KeyBuilder"""
        try:
            # Try KeyBuilder first for conversation context
            if self.db and hasattr(self.db, 'key_builder'):
                try:
                    context_data = await self.db.key_builder.get_user_context(phone_number)
                    if context_data:
                        return context_data
                except Exception as e:
                    logger.warning(f"KeyBuilder context lookup failed: {e}")
            
            # Try database service (new enhanced context)
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
                "conversation_frequency": "occasional",
                "conversation_type": "unknown"  # trading, customer_service, sales
            },
            "today_session": {
                "message_count": 0,
                "topics_discussed": [],
                "symbols_mentioned": [],
                "session_mood": "neutral",
                "is_first_message_today": True,
                "session_type": "mixed"  # trading, support, sales
            },
            "recent_messages": [],
            "context_summary": "No conversation history available"
        }
    
    async def _generate_context_aware_response(self, user: UserProfile, message: str, conversation_context: Dict) -> str:
        """Generate response with conversation context awareness for trading AND customer service"""
        try:
            # Check if we have enhanced Claude agent with context support
            if hasattr(self.claude, 'generate_context_aware_response'):
                return await self.claude.generate_context_aware_response(
                    user_query=message,
                    user_profile=user.__dict__ if hasattr(user, '__dict__') else {},
                    conversation_context=conversation_context
                )
            
            # Check if we have personality-aware response generation
            elif hasattr(self.claude, 'generate_personalized_response'):
                # Enhanced call with context information
                user_profile_dict = user.__dict__ if hasattr(user, '__dict__') else {}
                
                # Enhance user profile with conversation context
                if conversation_context:
                    user_profile_dict['conversation_context'] = conversation_context
                
                return await self.claude.generate_personalized_response(
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
        """Generate basic response with available context for trading AND customer service conversations"""
        try:
            # Determine conversation type from context or message content
            conversation_type = self._determine_conversation_type(message, conversation_context)
            
            # Build context-aware prompt
            context_info = ""
            if conversation_context:
                conv_ctx = conversation_context.get("conversation_context", {})
                today_session = conversation_context.get("today_session", {})
                
                # Add relationship context
                relationship = conv_ctx.get("relationship_stage", "new")
                total_convos = conv_ctx.get("total_conversations", 0)
                context_info += f"Relationship: {relationship} ({total_convos} conversations). "
                
                # Add conversation type context
                conv_type = conv_ctx.get("conversation_type", conversation_type)
                context_info += f"Conversation type: {conv_type}. "
                
                # Add recent context based on conversation type
                if conv_type == "trading":
                    recent_symbols = conv_ctx.get("recent_symbols", [])
                    if recent_symbols:
                        context_info += f"Recently discussed: {', '.join(recent_symbols[-3:])}. "
                elif conv_type in ["customer_service", "support"]:
                    recent_topics = conv_ctx.get("recent_topics", [])
                    if recent_topics:
                        context_info += f"Recent support topics: {', '.join(recent_topics[-2:])}. "
                elif conv_type == "sales":
                    context_info += f"Sales conversation in progress. "
                
                # Add today's context
                if not today_session.get("is_first_message_today", True):
                    context_info += f"Continuing today's conversation. "
                else:
                    context_info += f"First message today. "
            
            # Enhanced prompt with conversation type awareness
            if conversation_type == "trading":
                enhanced_prompt = f"""Respond as a knowledgeable trading buddy.

Context: {context_info}

User message: "{message}"

Instructions:
- Be conversational and helpful with trading insights
- Reference conversation history naturally if relevant
- Match their communication style
- Provide valuable trading insights
- Keep response under 300 characters for SMS
"""
            elif conversation_type in ["customer_service", "support"]:
                enhanced_prompt = f"""Respond as a helpful customer service representative.

Context: {context_info}

User message: "{message}"

Instructions:
- Be professional and helpful with support issues
- Reference previous support interactions if relevant
- Provide clear solutions or next steps
- Match their communication style
- Keep response under 300 characters for SMS
"""
            elif conversation_type == "sales":
                enhanced_prompt = f"""Respond as a friendly sales representative.

Context: {context_info}

User message: "{message}"

Instructions:
- Be helpful and not pushy about service offerings
- Reference previous conversations if relevant
- Focus on user benefits and value
- Match their communication style
- Keep response under 300 characters for SMS
"""
            else:
                enhanced_prompt = f"""Respond as a helpful AI assistant.

Context: {context_info}

User message: "{message}"

Instructions:
- Be conversational and helpful
- Reference conversation history naturally if relevant
- Match their communication style
- Provide valuable assistance
- Keep response under 300 characters for SMS
"""
            
            # Use basic Claude generation if available
            if hasattr(self.claude, 'client'):
                response = await self.claude.client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=150,
                    temperature=0.7,
                    messages=[{"role": "user", "content": enhanced_prompt}]
                )
                return response.content[0].text.strip()
            
            # Fallback response based on conversation type
            else:
                if conversation_type == "trading":
                    return "I'm here to help with your trading questions! Can you tell me more about what you're looking for?"
                elif conversation_type in ["customer_service", "support"]:
                    return "I'm here to help with any questions or issues you have. How can I assist you today?"
                elif conversation_type == "sales":
                    return "I'd be happy to help you learn more about our services. What interests you most?"
                else:
                    return "I'm here to help! How can I assist you today?"
                
        except Exception as e:
            logger.error(f"❌ Error in basic response generation: {e}")
            return "I'm having trouble processing your request. Please try again."
    
    def _determine_conversation_type(self, message: str, conversation_context: Dict) -> str:
        """Determine if this is trading, customer service, or sales conversation"""
        message_lower = message.lower()
        
        # Check context first
        if conversation_context:
            conv_ctx = conversation_context.get("conversation_context", {})
            existing_type = conv_ctx.get("conversation_type")
            if existing_type and existing_type != "unknown":
                return existing_type
        
        # Trading keywords
        trading_keywords = [
            'stock', 'stocks', 'trade', 'trading', 'buy', 'sell', 'portfolio',
            'analysis', 'chart', 'price', 'market', 'ticker', 'options',
            'calls', 'puts', 'rsi', 'macd', 'support', 'resistance'
        ]
        
        # Customer service keywords
        support_keywords = [
            'help', 'problem', 'issue', 'error', 'bug', 'not working',
            'cancel', 'refund', 'billing', 'account', 'login', 'password',
            'support', 'complaint', 'feedback'
        ]
        
        # Sales keywords
        sales_keywords = [
            'upgrade', 'plan', 'pricing', 'features', 'demo', 'trial',
            'subscription', 'pro', 'premium', 'paid'
        ]
        
        # Count keyword matches
        trading_score = sum(1 for keyword in trading_keywords if keyword in message_lower)
        support_score = sum(1 for keyword in support_keywords if keyword in message_lower)
        sales_score = sum(1 for keyword in sales_keywords if keyword in message_lower)
        
        # Determine type based on highest score
        if trading_score > support_score and trading_score > sales_score:
            return "trading"
        elif support_score > trading_score and support_score > sales_score:
            return "customer_service"
        elif sales_score > 0:
            return "sales"
        else:
            return "general"
    
    def _generate_fallback_response(self, conversation_context: Dict) -> str:
        """Generate fallback response when all else fails"""
        # Use context to determine appropriate fallback
        if conversation_context:
            conv_ctx = conversation_context.get("conversation_context", {})
            relationship = conv_ctx.get("relationship_stage", "new")
            conv_type = conv_ctx.get("conversation_type", "general")
            
            if conv_type == "trading":
                if relationship == "new":
                    return "Hey there! I'm having some technical issues but I'm here to help with your trading questions."
                else:
                    return "yo tech gremlins acting up but I got you! what trading help you need?"
            elif conv_type in ["customer_service", "support"]:
                return "I'm experiencing some technical difficulties but I'm still here to help with your support needs."
            elif conv_type == "sales":
                return "Technical issues on my end, but I'm still available to discuss our services with you."
            else:
                if relationship == "new":
                    return "Hey there! I'm having some technical issues but I'm here to help."
                else:
                    return "tech issues on my end but still here to help! what's up?"
        else:
            return "I'm having trouble processing your request. Please try again."
    
    async def _update_conversation_context(self, phone_number: str, user_message: str, bot_response: str):
        """Update conversation context after successful interaction"""
        try:
            # Determine conversation type for context
            conversation_type = self._determine_conversation_type(user_message, {})
            
            # Update through database service if available
            if self.db and hasattr(self.db, 'save_enhanced_message'):
                await self.db.save_enhanced_message(
                    phone_number=phone_number,
                    user_message=user_message,
                    bot_response=bot_response,
                    intent_data={"intent": conversation_type},
                    symbols=self._extract_basic_symbols(user_message),
                    context_used="message_handler_basic"
                )
            
            # Update through user manager if available
            elif self.user_manager:
                # Extract symbols and topics based on conversation type
                symbols = self._extract_basic_symbols(user_message) if conversation_type == "trading" else []
                
                # Update user learning
                if hasattr(self.user_manager, 'learn_from_interaction'):
                    self.user_manager.learn_from_interaction(
                        phone_number=phone_number,
                        message=user_message,
                        symbols=symbols,
                        intent=conversation_type
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
                        intent=conversation_type,
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
