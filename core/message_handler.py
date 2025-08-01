# core/message_handler.py
import asyncio
import time
from typing import Dict, List, Optional, Any
from loguru import logger
from datetime import datetime, timezone

from core.personality_engine_v3_gemini import EnhancedPersonalityEngine, MessageAnalysis
from services.llm_agent import TradingAgent
from services.database import DatabaseService


class EnhancedMessageHandler:
    """
    Enhanced Message Handler with real-time Gemini personality analysis integration
    Maintains sub-4-second SMS response times while adding semantic intelligence
    """
    
    def __init__(
        self, 
        db_service: DatabaseService,
        trading_agent: TradingAgent,
        gemini_api_key: str = None,
        enable_background_analysis: bool = True
    ):
        """Initialize enhanced message handler with Gemini integration"""
        self.db_service = db_service
        self.trading_agent = trading_agent
        self.enable_background_analysis = enable_background_analysis
        
        # Initialize enhanced personality engine
        self.personality_engine = EnhancedPersonalityEngine(
            db_service=db_service,
            gemini_api_key=gemini_api_key
        )
        
        # Performance tracking
        self.response_times = []
        self.analysis_times = []
        self.total_messages_processed = 0
        
        logger.info("🚀 Enhanced Message Handler initialized with Gemini personality analysis")
    
    async def process_message(
        self, 
        user_phone: str, 
        message_content: str,
        message_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process incoming SMS message with real-time personality analysis
        Optimized for sub-4-second response times
        """
        start_time = time.time()
        
        try:
            # Step 1: Get or create user profile (fast lookup)
            user_profile = await self.personality_engine.get_user_profile(user_phone)
            
            # Step 2: Real-time personality context (optimized for speed)
            personality_context = await self.personality_engine.get_real_time_personality_context(
                user_phone, 
                message_content
            )
            
            context_time = time.time()
            logger.debug(f"⚡ Personality context generated in {(context_time - start_time) * 1000:.0f}ms")
            
            # Step 3: Generate enhanced prompt with personality context
            enhanced_context = self._build_enhanced_context(
                user_profile, 
                personality_context, 
                message_context
            )
            
            # Step 4: Get AI response with personality-aware prompt
            ai_response = await self.trading_agent.process_message_with_context(
                user_phone=user_phone,
                message=message_content,
                personality_context=enhanced_context
            )
            
            # Step 5: Learn from interaction (async, don't wait)
            asyncio.create_task(
                self._async_learning_update(user_phone, message_content, ai_response, personality_context)
            )
            
            # Step 6: Schedule background analysis if enabled
            if self.enable_background_analysis:
                asyncio.create_task(
                    self._schedule_background_analysis(user_phone)
                )
            
            # Step 7: Track performance
            total_time = time.time() - start_time
            self._track_performance(total_time, personality_context)
            
            return {
                "response": ai_response,
                "personality_applied": True,
                "analysis_method": personality_context["meta"]["analysis_method"],
                "confidence_score": personality_context["meta"]["confidence"],
                "processing_time_ms": int(total_time * 1000),
                "personality_context": enhanced_context
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced message processing failed: {e}")
            # Fallback to basic processing
            return await self._fallback_processing(user_phone, message_content, start_time)
    
    def _build_enhanced_context(
        self, 
        user_profile: Dict, 
        personality_context: Dict,
        message_context: Dict = None
    ) -> Dict[str, Any]:
        """Build comprehensive context for AI response generation"""
        
        # Extract communication preferences
        comm_style = personality_context.get("communication_style", {})
        trading_focus = personality_context.get("trading_focus", {})
        emotional_state = personality_context.get("emotional_state", {})
        response_guidance = personality_context.get("response_guidance", {})
        
        # Build enhanced context
        enhanced_context = {
            # Communication style adaptation
            "communication_preferences": {
                "formality_level": self._interpret_formality(comm_style.get("formality", 0.5)),
                "energy_matching": comm_style.get("energy", "moderate"),
                "technical_depth": comm_style.get("technical_preference", "basic"),
                "use_emojis": comm_style.get("emoji_appropriate", False),
                "response_length": self._determine_response_length(user_profile, comm_style)
            },
            
            # Trading context
            "trading_context": {
                "symbols_mentioned": trading_focus.get("symbols", []),
                "trading_intent": trading_focus.get("action_intent", "unclear"),
                "risk_profile": trading_focus.get("risk_appetite", "moderate"),
                "requires_analysis_tools": response_guidance.get("requires_tools", [])
            },
            
            # Emotional awareness
            "emotional_context": {
                "current_emotion": emotional_state.get("primary_emotion", "neutral"),
                "emotional_intensity": emotional_state.get("intensity", 0.0),
                "support_approach": emotional_state.get("support_needed", "standard_guidance"),
                "reassurance_needed": emotional_state.get("intensity", 0.0) > 0.6
            },
            
            # Response optimization
            "response_optimization": {
                "urgency_level": response_guidance.get("urgency_level", 0.0),
                "personalization_strength": response_guidance.get("personalization_strength", 0.5),
                "follow_up_likely": response_guidance.get("follow_up_likely", 0.5),
                "confidence_in_analysis": personality_context["meta"]["confidence"]
            },
            
            # Historical context
            "user_history": {
                "total_conversations": user_profile.get("conversation_history", {}).get("total_conversations", 0),
                "preferred_sectors": list(user_profile.get("trading_personality", {}).get("sector_preferences", {}).keys())[:3],
                "recent_symbols": user_profile.get("trading_personality", {}).get("common_symbols", [])[-5:],
                "experience_level": user_profile.get("trading_personality", {}).get("experience_level", "intermediate")
            },
            
            # Meta information
            "meta": {
                "analysis_method": personality_context["meta"]["analysis_method"],
                "processing_time_budget_ms": 3000,  # Reserve time for response generation
                "gemini_powered": personality_context["meta"]["analysis_method"] == "gemini"
            }
        }
        
        # Add message-specific context if provided
        if message_context:
            enhanced_context["message_context"] = message_context
        
        return enhanced_context
    
    def _interpret_formality(self, formality_score: float) -> str:
        """Convert formality score to descriptive level"""
        if formality_score > 0.7:
            return "professional"
# core/message_handler.py
import asyncio
import time
from typing import Dict, List, Optional, Any
from loguru import logger
from datetime import datetime, timezone

from core.personality_engine_v3_gemini import EnhancedPersonalityEngine, MessageAnalysis
from services.llm_agent import TradingAgent
from services.database import DatabaseService
from models.user import UserProfile


class MessageHandler:
    """
    Message Handler with real-time Gemini personality analysis integration
    INCLUDES ALL ORIGINAL FUNCTIONALITY + Gemini enhancements
    """
    
    def __init__(
        self, 
        db_service: DatabaseService,
        claude_service=None,  # Keep original parameter name for compatibility
        twilio_service=None,
        user_manager=None,
        message_processor=None,
        gemini_api_key: str = None,
        enable_background_analysis: bool = True
    ):
        """Initialize message handler with full backward compatibility"""
        self.db = db_service
        self.claude = claude_service  # Keep original attribute name
        self.trading_agent = claude_service  # Also support as trading_agent
        self.twilio = twilio_service
        self.user_manager = user_manager
        self.message_processor = message_processor  # Orchestrator processor
        self.enable_background_analysis = enable_background_analysis
        
        # Initialize enhanced personality engine
        self.personality_engine = EnhancedPersonalityEngine(
            db_service=db_service,
            gemini_api_key=gemini_api_key
        )
        
        # Performance tracking
        self.response_times = []
        self.analysis_times = []
        self.total_messages_processed = 0
        
        logger.info("✅ Enhanced message handler initialized with Claude and context support + Gemini")
    
    # ==========================================
    # MAIN PROCESSING METHOD (ENHANCED)
    # ==========================================
    
    async def process_incoming_message(self, phone_number: str, message_body: str) -> bool:
        """
        Process incoming SMS message with enhanced personality analysis
        MAINTAINS ORIGINAL SIGNATURE AND BEHAVIOR
        """
        start_time = time.time()
        
        try:
            logger.info(f"📱 Processing message from {phone_number}: {message_body}")
            
            # Step 1: Get or create user (ORIGINAL FUNCTIONALITY)
            user = await self._get_or_create_user(phone_number)
            
            # Step 2: Check limits (ORIGINAL FUNCTIONALITY)
            usage_check = await self._check_weekly_limits(user)
            if not usage_check["can_send"]:
                await self._send_limit_message(user, usage_check)
                return True
            
            # Step 3: Enhanced response generation with personality analysis
            response = await self._generate_enhanced_response(user, message_body)
            
            # Step 4: Send response (ORIGINAL FUNCTIONALITY - FIXED METHOD NAME)
            if self.twilio:
                await self.twilio.send_message(phone_number, response)
            
            # Step 5: Update usage (ORIGINAL FUNCTIONALITY)
            await self._update_weekly_usage(user)
            
            # Step 6: Background learning (NEW ENHANCEMENT)
            if self.enable_background_analysis:
                asyncio.create_task(
                    self._async_learning_update(phone_number, message_body, response)
                )
            
            # Step 7: Track performance
            total_time = time.time() - start_time
            self._track_performance(total_time)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing message from {phone_number}: {e}")
            return False
    
    async def _generate_enhanced_response(self, user: UserProfile, message_body: str) -> str:
        """Enhanced response generation with multiple fallback strategies"""
        try:
            # Strategy 1: Use new orchestrator if available
            if self.message_processor:
                logger.debug("Using message processor (orchestrator)")
                return await self.message_processor.process_message(
                    message_body, user.phone_number
                )
            
            # Strategy 2: Use enhanced personality-aware generation
            try:
                personality_context = await self.personality_engine.get_real_time_personality_context(
                    user.phone_number, message_body
                )
                
                enhanced_context = self._build_enhanced_context_for_response(
                    user, personality_context
                )
                
                # Try enhanced trading agent methods
                if hasattr(self.claude, 'process_message_with_context'):
                    return await self.claude.process_message_with_context(
                        user_phone=user.phone_number,
                        message=message_body,
                        personality_context=enhanced_context
                    )
                elif hasattr(self.trading_agent, 'process_message_with_context'):
                    return await self.trading_agent.process_message_with_context(
                        user_phone=user.phone_number,
                        message=message_body,
                        personality_context=enhanced_context
                    )
                
            except Exception as e:
                logger.warning(f"Enhanced personality generation failed: {e}")
            
            # Strategy 3: Use original context-aware response (FALLBACK)
            conversation_context = await self._get_conversation_context(user.phone_number)
            return await self._generate_context_aware_response(user, message_body, conversation_context)
            
        except Exception as e:
            logger.error(f"All response generation strategies failed: {e}")
            return self._generate_fallback_response({})
    
    def _build_enhanced_context_for_response(self, user: UserProfile, personality_context: Dict) -> Dict:
        """Build context for enhanced response generation"""
        comm_style = personality_context.get("communication_style", {})
        trading_focus = personality_context.get("trading_focus", {})
        emotional_state = personality_context.get("emotional_state", {})
        
        return {
            "user_profile": {
                "phone_number": user.phone_number,
                "plan_type": user.plan_type,
                "subscription_status": user.subscription_status
            },
            "communication_preferences": {
                "formality_level": self._interpret_formality(comm_style.get("formality", 0.5)),
                "energy_matching": comm_style.get("energy", "moderate"),
                "technical_depth": comm_style.get("technical_preference", "basic"),
                "use_emojis": comm_style.get("emoji_appropriate", False)
            },
            "trading_context": {
                "symbols_mentioned": trading_focus.get("symbols", []),
                "trading_intent": trading_focus.get("action_intent", "unclear"),
                "risk_profile": trading_focus.get("risk_appetite", "moderate")
            },
            "emotional_context": {
                "current_emotion": emotional_state.get("primary_emotion", "neutral"),
                "emotional_intensity": emotional_state.get("intensity", 0.0),
                "support_approach": emotional_state.get("support_needed", "standard_guidance")
            }
        }
    
    def _interpret_formality(self, formality_score: float) -> str:
        """Convert formality score to descriptive level"""
        if formality_score > 0.7:
            return "professional"
        elif formality_score < 0.3:
            return "casual"
        else:
            return "friendly"
    
    # ==========================================
    # ORIGINAL USER MANAGEMENT METHODS (PRESERVED)
    # ==========================================
    
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
        
        if self.twilio:
            await self.twilio.send_message(user.phone_number, message)
    
    async def _update_weekly_usage(self, user: UserProfile):
        """Update weekly usage counter"""
        try:
            if self.db:
                await self.db.update_user_activity(user.phone_number)
        except Exception as e:
            logger.error(f"❌ Error updating usage: {e}")
    
    # ==========================================
    # ORIGINAL CONTEXT METHODS (PRESERVED)
    # ==========================================
    
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
                "conversation_type": "unknown"
            },
            "today_session": {
                "message_count": 0,
                "topics_discussed": [],
                "symbols_mentioned": [],
                "session_mood": "neutral",
                "is_first_message_today": True,
                "session_type": "mixed"
            },
            "recent_messages": [],
            "context_summary": "No conversation history available"
        }
    
    async def _generate_context_aware_response(self, user: UserProfile, message: str, conversation_context: Dict) -> str:
        """Generate response with conversation context awareness (ORIGINAL METHOD)"""
        try:
            # Check if we have enhanced Claude agent with context support
            if hasattr(self.trading_agent, 'generate_context_aware_response'):
                return await self.trading_agent.generate_context_aware_response(
                    user_query=message,
                    user_profile=user.__dict__ if hasattr(user, '__dict__') else {},
                    conversation_context=conversation_context
                )
            
            # Check if we have personality-aware response generation
            elif hasattr(self.trading_agent, 'generate_personalized_response'):
                user_profile_dict = user.__dict__ if hasattr(user, '__dict__') else {}
                
                if conversation_context:
                    user_profile_dict['conversation_context'] = conversation_context
                
                return await self.trading_agent.generate_personalized_response(
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
        """Generate basic response with available context (ORIGINAL METHOD)"""
        try:
            conversation_type = self._determine_conversation_type(message, conversation_context)
            
            # Build context-aware prompt
            context_info = ""
            if conversation_context:
                conv_ctx = conversation_context.get("conversation_context", {})
                today_session = conversation_context.get("today_session", {})
                
                relationship = conv_ctx.get("relationship_stage", "new")
                total_convos = conv_ctx.get("total_conversations", 0)
                context_info += f"Relationship: {relationship} ({total_convos} conversations). "
                
                conv_type = conv_ctx.get("conversation_type", conversation_type)
                context_info += f"Conversation type: {conv_type}. "
                
                if conv_type == "trading":
                    recent_symbols = conv_ctx.get("recent_symbols", [])
                    if recent_symbols:
                        context_info += f"Recently discussed: {', '.join(recent_symbols[-3:])}. "
                elif conv_type in ["customer_service", "support"]:
                    recent_topics = conv_ctx.get("recent_topics", [])
                    if recent_topics:
                        context_info += f"Recent support topics: {', '.join(recent_topics[-2:])}. "
                
                if not today_session.get("is_first_message_today", True):
                    context_info += f"Continuing today's conversation. "
                else:
                    context_info += f"First message today. "
            
            # Generate enhanced prompt based on conversation type
            enhanced_prompt = self._build_prompt_for_conversation_type(
                conversation_type, message, context_info
            )
            
            # Use trading agent if available
            if hasattr(self.trading_agent, 'client') or hasattr(self.trading_agent, 'generate_response'):
                if hasattr(self.trading_agent, 'generate_response'):
                    return await self.trading_agent.generate_response(enhanced_prompt)
                else:
                    # Direct LLM call
                    response = await self.trading_agent.client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=150,
                        temperature=0.7,
                        messages=[{"role": "user", "content": enhanced_prompt}]
                    )
                    return response.content[0].text.strip()
            
            # Fallback response based on conversation type
            else:
                return self._get_fallback_response_by_type(conversation_type)
                
        except Exception as e:
            logger.error(f"❌ Error in basic response generation: {e}")
            return "I'm having trouble processing your request. Please try again."
    
    def _build_prompt_for_conversation_type(self, conversation_type: str, message: str, context_info: str) -> str:
        """Build prompt based on conversation type"""
        if conversation_type == "trading":
            return f"""Respond as a knowledgeable trading buddy.

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
            return f"""Respond as a helpful customer service representative.

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
            return f"""Respond as a friendly sales representative.

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
            return f"""Respond as a helpful AI assistant.

Context: {context_info}

User message: "{message}"

Instructions:
- Be conversational and helpful
- Reference conversation history naturally if relevant
- Match their communication style
- Provide valuable assistance
- Keep response under 300 characters for SMS
"""
    
    def _get_fallback_response_by_type(self, conversation_type: str) -> str:
        """Get fallback response by conversation type"""
        fallbacks = {
            "trading": "I'm here to help with your trading questions! Can you tell me more about what you're looking for?",
            "customer_service": "I'm here to help with any questions or issues you have. How can I assist you today?",
            "support": "I'm here to help with any questions or issues you have. How can I assist you today?",
            "sales": "I'd be happy to help you learn more about our services. What interests you most?",
            "general": "I'm here to help! How can I assist you today?"
        }
        return fallbacks.get(conversation_type, fallbacks["general"])
    
    def _determine_conversation_type(self, message: str, conversation_context: Dict) -> str:
        """Determine if this is trading, customer service, or sales conversation (ORIGINAL METHOD)"""
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
        """Generate fallback response when all else fails (ORIGINAL METHOD)"""
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
        """Update conversation context after successful interaction (ORIGINAL METHOD)"""
        try:
            conversation_type = self._determine_conversation_type(user_message, {})
            
            # Update through database service if available
            if self.db and hasattr(self.db, 'save_enhanced_message'):
                await self.db.save_enhanced_message(
                    phone_number=phone_number,
                    user_message=user_message,
                    bot_response=bot_response,
                    intent_data={"intent": conversation_type},
                    symbols=self._extract_basic_symbols(user_message),
                    context_used="message_handler_enhanced"
                )
            
            # Update through user manager if available
            elif self.user_manager:
                symbols = self._extract_basic_symbols(user_message) if conversation_type == "trading" else []
                
                if hasattr(self.user_manager, 'learn_from_interaction'):
                    self.user_manager.learn_from_interaction(
                        phone_number=phone_number,
                        message=user_message,
                        symbols=symbols,
                        intent=conversation_type
                    )
                
                if hasattr(self.user_manager, 'update_user_activity'):
                    self.user_manager.update_user_activity(phone_number, "received")
                
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
        """Extract basic stock symbols from message (ORIGINAL METHOD)"""
        try:
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
    
    # ==========================================
    # NEW ENHANCED METHODS
    # ==========================================
    
    async def _async_learning_update(self, user_phone: str, message_content: str, ai_response: str) -> None:
        """Async learning update - doesn't block response generation"""
        try:
            learning_result = await self.personality_engine.analyze_and_learn(
                user_id=user_phone,
                user_message=message_content,
                bot_response=ai_response,
                context={"enhanced_handler": True}
            )
            
            logger.debug(f"📚 Learning update completed for {user_phone}: {learning_result.get('learning_applied', False)}")
            
        except Exception as e:
            logger.error(f"❌ Async learning update failed: {e}")
    
    def _track_performance(self, total_time: float) -> None:
        """Track performance metrics for monitoring"""
        self.response_times.append(total_time)
        self.total_messages_processed += 1
        
        # Keep only recent measurements
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
        
        # Log performance warning if slow
        if total_time > 4.0:
            logger.warning(f"⚠️ Slow response: {total_time:.2f}s (target: <4s)")
        elif total_time > 3.0:
            logger.info(f"⏱️ Response time: {total_time:.2f}s")
    
    # ==========================================
    # ADDITIONAL ENHANCED METHODS (FROM ORIGINAL)
    # ==========================================
    
    async def process_message(self, user_phone: str, message_content: str, message_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhanced process_message method for direct integration
        Returns detailed response information
        """
        start_time = time.time()
        
        try:
            # Get user
            user = await self._get_or_create_user(user_phone)
            
            # Check limits
            usage_check = await self._check_weekly_limits(user)
            if not usage_check["can_send"]:
                limit_message = self._build_limit_message(user, usage_check)
                return {
                    "response": limit_message,
                    "personality_applied": False,
                    "analysis_method": "limit_check",
                    "processing_time_ms": int((time.time() - start_time) * 1000),
                    "limit_reached": True
                }
            
            # Enhanced response generation
            try:
                personality_context = await self.personality_engine.get_real_time_personality_context(
                    user_phone, message_content
                )
                
                enhanced_context = self._build_enhanced_context_for_response(user, personality_context)
                
                if hasattr(self.trading_agent, 'process_message_with_context'):
                    response = await self.trading_agent.process_message_with_context(
                        user_phone=user_phone,
                        message=message_content,
                        personality_context=enhanced_context
                    )
                    analysis_method = "gemini_enhanced"
                    personality_applied = True
                else:
                    # Fallback to original method
                    conversation_context = await self._get_conversation_context(user_phone)
                    response = await self._generate_context_aware_response(user, message_content, conversation_context)
                    analysis_method = "context_aware_fallback"
                    personality_applied = False
                
            except Exception as e:
                logger.warning(f"Enhanced response failed, using basic fallback: {e}")
                response = await self._generate_basic_response(user, message_content, {})
                analysis_method = "basic_fallback"
                personality_applied = False
            
            # Update usage
            await self._update_weekly_usage(user)
            
            # Background learning
            if self.enable_background_analysis and personality_applied:
                asyncio.create_task(
                    self._async_learning_update(user_phone, message_content, response)
                )
            
            # Track performance
            total_time = time.time() - start_time
            self._track_performance(total_time)
            
            return {
                "response": response,
                "personality_applied": personality_applied,
                "analysis_method": analysis_method,
                "processing_time_ms": int(total_time * 1000),
                "confidence_score": personality_context.get("meta", {}).get("confidence", 0.5) if 'personality_context' in locals() else 0.5,
                "limit_reached": False
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced message processing failed: {e}")
            return {
                "response": "I'm experiencing technical difficulties. Please try again in a moment.",
                "personality_applied": False,
                "analysis_method": "error_fallback",
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "error": str(e)
            }
    
    def _build_limit_message(self, user: UserProfile, usage_info: Dict) -> str:
        """Build limit reached message"""
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
    
    # ==========================================
    # ORIGINAL USER MANAGEMENT METHODS (PRESERVED)
    # ==========================================
    
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
        
        if self.twilio:
            await self.twilio.send_message(user.phone_number, message)  # FIXED: send_sms → send_message
    
    async def _update_weekly_usage(self, user: UserProfile):
        """Update weekly usage counter with FIXED method call"""
        try:
            if self.db:
                # FIXED: Use correct method signature - only phone_number argument
                await self.db.update_user_activity(user.phone_number)
        except Exception as e:
            logger.error(f"❌ Error updating usage: {e}")
    
    # ==========================================
    # ORIGINAL CONTEXT METHODS (PRESERVED)
    # ==========================================
    
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
            enhanced_prompt = self._build_prompt_for_conversation_type(
                conversation_type, message, context_info
            )
            
            # Use basic Claude generation if available
            try:
                if hasattr(self.claude, 'client'):
                    response = await self.claude.client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=150,
                        temperature=0.7,
                        messages=[{"role": "user", "content": enhanced_prompt}]
                    )
                    return response.content[0].text.strip()
                elif hasattr(self.claude, 'generate_response'):
                    return await self.claude.generate_response(enhanced_prompt)
            except Exception as e:
                logger.warning(f"Claude response generation failed: {e}")
            
            # Fallback response based on conversation type
            return self._get_fallback_response_by_type(conversation_type)
                
        except Exception as e:
            logger.error(f"❌ Error in basic response generation: {e}")
            return "I'm having trouble processing your request. Please try again."
    
    def _build_prompt_for_conversation_type(self, conversation_type: str, message: str, context_info: str) -> str:
        """Build prompt based on conversation type"""
        if conversation_type == "trading":
            return f"""Respond as a knowledgeable trading buddy.

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
            return f"""Respond as a helpful customer service representative.

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
            return f"""Respond as a friendly sales representative.

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
            return f"""Respond as a helpful AI assistant.

Context: {context_info}

User message: "{message}"

Instructions:
- Be conversational and helpful
- Reference conversation history naturally if relevant
- Match their communication style
- Provide valuable assistance
- Keep response under 300 characters for SMS
"""
    
    def _get_fallback_response_by_type(self, conversation_type: str) -> str:
        """Get fallback response by conversation type"""
        fallbacks = {
            "trading": "I'm here to help with your trading questions! Can you tell me more about what you're looking for?",
            "customer_service": "I'm here to help with any questions or issues you have. How can I assist you today?",
            "support": "I'm here to help with any questions or issues you have. How can I assist you today?",
            "sales": "I'd be happy to help you learn more about our services. What interests you most?",
            "general": "I'm here to help! How can I assist you today?"
        }
        return fallbacks.get(conversation_type, fallbacks["general"])
    
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
                    context_used="message_handler_enhanced"
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
    
    # ==========================================
    # NEW ENHANCED METHODS
    # ==========================================
    
    async def _async_learning_update(self, user_phone: str, message_content: str, ai_response: str) -> None:
        """Async learning update - doesn't block response generation"""
        try:
            learning_result = await self.personality_engine.analyze_and_learn(
                user_id=user_phone,
                user_message=message_content,
                bot_response=ai_response,
                context={"enhanced_handler": True}
            )
            
            logger.debug(f"📚 Learning update completed for {user_phone}: {learning_result.get('learning_applied', False)}")
            
        except Exception as e:
            logger.error(f"❌ Async learning update failed: {e}")
    
    def _track_performance(self, total_time: float) -> None:
        """Track performance metrics for monitoring"""
        self.response_times.append(total_time)
        self.total_messages_processed += 1
        
        # Keep only recent measurements
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
        
        # Log performance warning if slow
        if total_time > 4.0:
            logger.warning(f"⚠️ Slow response: {total_time:.2f}s (target: <4s)")
        elif total_time > 3.0:
            logger.info(f"⏱️ Response time: {total_time:.2f}s")
    
    # ==========================================
    # ADDITIONAL API METHODS (FOR INTEGRATION)
    # ==========================================
    
    async def process_message(self, user_phone: str, message_content: str, message_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhanced process_message method for direct integration
        Returns detailed response information
        """
        start_time = time.time()
        
        try:
            # Get user
            user = await self._get_or_create_user(user_phone)
            
            # Check limits
            usage_check = await self._check_weekly_limits(user)
            if not usage_check["can_send"]:
                limit_message = self._build_limit_message(user, usage_check)
                return {
                    "response": limit_message,
                    "personality_applied": False,
                    "analysis_method": "limit_check",
                    "processing_time_ms": int((time.time() - start_time) * 1000),
                    "limit_reached": True
                }
            
            # Enhanced response generation
            try:
                personality_context = await self.personality_engine.get_real_time_personality_context(
                    user_phone, message_content
                )
                
                enhanced_context = self._build_enhanced_context_for_response(user, personality_context)
                
                if hasattr(self.claude, 'process_message_with_context'):
                    response = await self.claude.process_message_with_context(
                        user_phone=user_phone,
                        message=message_content,
                        personality_context=enhanced_context
                    )
                    analysis_method = "gemini_enhanced"
                    personality_applied = True
                else:
                    # Fallback to original method
                    conversation_context = await self._get_conversation_context(user_phone)
                    response = await self._generate_context_aware_response(user, message_content, conversation_context)
                    analysis_method = "context_aware_fallback"
                    personality_applied = False
                
            except Exception as e:
                logger.warning(f"Enhanced response failed, using basic fallback: {e}")
                response = await self._generate_basic_response(user, message_content, {})
                analysis_method = "basic_fallback"
                personality_applied = False
            
            # Update usage
            await self._update_weekly_usage(user)
            
            # Background learning
            if self.enable_background_analysis and personality_applied:
                asyncio.create_task(
                    self._async_learning_update(user_phone, message_content, response)
                )
            
            # Track performance
            total_time = time.time() - start_time
            self._track_performance(total_time)
            
            return {
                "response": response,
                "personality_applied": personality_applied,
                "analysis_method": analysis_method,
                "processing_time_ms": int(total_time * 1000),
                "confidence_score": personality_context.get("meta", {}).get("confidence", 0.5) if 'personality_context' in locals() else 0.5,
                "limit_reached": False
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced message processing failed: {e}")
            return {
                "response": "I'm experiencing technical difficulties. Please try again in a moment.",
                "personality_applied": False,
                "analysis_method": "error_fallback",
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "error": str(e)
            }
    
    def _build_limit_message(self, user: UserProfile, usage_info: Dict) -> str:
        """Build limit reached message"""
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
        
        return message
    
    # ==========================================
    # PERFORMANCE AND MONITORING METHODS
    # ==========================================
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if not self.response_times:
            return {"error": "No performance data available"}
        
        import statistics
        
        return {
            "response_times": {
                "average_ms": statistics.mean(self.response_times) * 1000,
                "median_ms": statistics.median(self.response_times) * 1000,
                "p95_ms": statistics.quantiles(self.response_times, n=20)[18] * 1000 if len(self.response_times) >= 20 else max(self.response_times) * 1000,
                "max_ms": max(self.response_times) * 1000,
                "under_4s_rate": sum(1 for t in self.response_times if t < 4.0) / len(self.response_times)
            },
            "system_metrics": {
                "total_messages_processed": self.total_messages_processed,
                "gemini_enabled": self.personality_engine.gemini_service is not None,
                "background_analysis_enabled": self.enable_background_analysis
            },
            "gemini_usage": self.personality_engine.get_gemini_usage_stats() if hasattr(self.personality_engine, 'get_gemini_usage_stats') else {}
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for monitoring"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {}
        }
        
        try:
            # Check personality engine
            try:
                test_analysis = await self.personality_engine.get_real_time_personality_context(
                    "health_check", 
                    "test message"
                )
                health_status["components"]["personality_engine"] = {
                    "status": "healthy",
                    "analysis_method": test_analysis["meta"]["analysis_method"],
                    "response_time_ms": test_analysis["meta"]["processing_time_ms"]
                }
            except Exception as e:
                health_status["components"]["personality_engine"] = {
                    "status": "degraded",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
            
            # Check database connection
            try:
                test_profile = await self.personality_engine.get_user_profile("health_check")
                health_status["components"]["database"] = {
                    "status": "healthy",
                    "profile_accessible": test_profile is not None
                }
            except Exception as e:
                health_status["components"]["database"] = {
                    "status": "error",
                    "error": str(e)
                }
                health_status["status"] = "error"
            
            # Performance metrics
            if self.response_times:
                health_status["performance"] = {
                    "avg_response_time_ms": sum(self.response_times[-10:]) / len(self.response_times[-10:]) * 1000,
                    "messages_processed": self.total_messages_processed,
                    "performance_target_met": sum(self.response_times[-10:]) / len(self.response_times[-10:]) < 4.0
                }
            
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {
                "status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }
    
    def _determine_response_length(self, user_profile: Dict, comm_style: Dict) -> str:
        """Determine optimal response length based on user preferences"""
        # Check historical preferences
        historical_length = user_profile.get("communication_style", {}).get("message_length", "medium")
        
        # Check current context
        current_preference = comm_style.get("message_length_preference", "medium")
        
        # Prioritize recent analysis over historical
        if current_preference and current_preference != "medium":
            return current_preference
        
        return historical_length
    
    async def _async_learning_update(
        self, 
        user_phone: str, 
        message_content: str, 
        ai_response: str,
        personality_context: Dict
    ) -> None:
        """Async learning update - doesn't block response generation"""
        try:
            # Learn from the interaction with enhanced context
            learning_result = await self.personality_engine.analyze_and_learn(
                user_id=user_phone,
                user_message=message_content,
                bot_response=ai_response,
                context={"personality_context": personality_context}
            )
            
            logger.debug(f"📚 Learning update completed for {user_phone}: {learning_result.get('learning_applied', False)}")
            
        except Exception as e:
            logger.error(f"❌ Async learning update failed: {e}")
    
    async def _schedule_background_analysis(self, user_phone: str) -> None:
        """Schedule background deep analysis for comprehensive profiling"""
        try:
            # Check if user has enough conversation history
            user_profile = await self.personality_engine.get_user_profile(user_phone)
            conversation_count = user_profile.get("conversation_history", {}).get("total_conversations", 0)
            
            # Only run background analysis for users with sufficient history
            if conversation_count >= 10 and conversation_count % 25 == 0:  # Every 25 conversations after first 10
                logger.info(f"🔍 Scheduling background analysis for {user_phone} (conversations: {conversation_count})")
                
                # Get conversation history
                conversation_history = await self.db_service.get_user_conversation_history(user_phone, limit=50)
                
                # Run background deep analysis
                deep_analysis_result = await self.personality_engine.run_background_deep_analysis(
                    user_phone, 
                    conversation_history
                )
                
                logger.info(f"✅ Background analysis completed: {deep_analysis_result.get('confidence_score', 0.5)}")
                
        except Exception as e:
            logger.error(f"❌ Background analysis scheduling failed: {e}")
    
    async def _fallback_processing(
        self, 
        user_phone: str, 
        message_content: str, 
        start_time: float
    ) -> Dict[str, Any]:
        """Fallback processing when enhanced analysis fails"""
        try:
            # Basic AI response without personality enhancement
            ai_response = await self.trading_agent.process_message(user_phone, message_content)
            
            total_time = time.time() - start_time
            
            return {
                "response": ai_response,
                "personality_applied": False,
                "analysis_method": "fallback",
                "confidence_score": 0.1,
                "processing_time_ms": int(total_time * 1000),
                "warning": "Enhanced personality analysis unavailable"
            }
            
        except Exception as e:
            logger.error(f"❌ Fallback processing also failed: {e}")
            return {
                "response": "I'm experiencing technical difficulties. Please try again in a moment.",
                "personality_applied": False,
                "analysis_method": "error_fallback",
                "confidence_score": 0.0,
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "error": str(e)
            }
    
    def _track_performance(self, total_time: float, personality_context: Dict) -> None:
        """Track performance metrics for monitoring"""
        self.response_times.append(total_time)
        self.analysis_times.append(personality_context["meta"]["processing_time_ms"])
        self.total_messages_processed += 1
        
        # Keep only recent measurements
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
            self.analysis_times = self.analysis_times[-100:]
        
        # Log performance warning if slow
        if total_time > 4.0:
            logger.warning(f"⚠️ Slow response: {total_time:.2f}s (target: <4s)")
        elif total_time > 3.0:
            logger.info(f"⏱️ Response time: {total_time:.2f}s")
    
    # ==========================================
    # REAL-TIME PERSONALITY INTEGRATION
    # ==========================================
    
    async def get_personality_enhanced_prompt(
        self, 
        user_phone: str, 
        message: str
    ) -> str:
        """Generate personality-enhanced prompt for direct LLM usage"""
        try:
            # Get real-time personality context
            personality_context = await self.personality_engine.get_real_time_personality_context(
                user_phone, 
                message
            )
            
            # Get user profile
            user_profile = await self.personality_engine.get_user_profile(user_phone)
            
            # Build enhanced context
            enhanced_context = self._build_enhanced_context(user_profile, personality_context)
            
            # Generate personality-aware prompt
            prompt = self._build_personality_prompt(message, enhanced_context)
            
            return prompt
            
        except Exception as e:
            logger.error(f"❌ Personality prompt generation failed: {e}")
            return f"USER MESSAGE: {message}\n\nRespond as a helpful trading assistant:"
    
    def _build_personality_prompt(self, message: str, context: Dict) -> str:
        """Build comprehensive personality-aware prompt"""
        comm_prefs = context.get("communication_preferences", {})
        trading_ctx = context.get("trading_context", {})
        emotional_ctx = context.get("emotional_context", {})
        user_history = context.get("user_history", {})
        meta = context.get("meta", {})
        
        prompt_parts = [
            "You are a highly personalized SMS trading assistant with deep user understanding.",
            "",
            "🎯 USER PERSONALITY PROFILE:",
            f"• Communication Style: {comm_prefs.get('formality_level', 'friendly')} tone, {comm_prefs.get('energy_matching', 'moderate')} energy",
            f"• Technical Level: {comm_prefs.get('technical_depth', 'basic')} explanations preferred",
            f"• Response Length: {comm_prefs.get('response_length', 'medium')} responses",
            f"• Emoji Usage: {'Use sparingly' if comm_prefs.get('use_emojis') else 'Avoid emojis'}",
            "",
            "📊 TRADING CONTEXT:",
            f"• Experience Level: {user_history.get('experience_level', 'intermediate')}",
            f"• Risk Profile: {trading_ctx.get('risk_profile', 'moderate')}",
            f"• Current Intent: {trading_ctx.get('trading_intent', 'general inquiry')}",
            f"• Symbols Mentioned: {', '.join(trading_ctx.get('symbols_mentioned', [])) or 'None'}",
            f"• Recent Focus: {', '.join(user_history.get('recent_symbols', [])) or 'Diversified'}",
            "",
            "🧠 EMOTIONAL AWARENESS:",
            f"• Current State: {emotional_ctx.get('current_emotion', 'neutral')} (intensity: {emotional_ctx.get('emotional_intensity', 0.0):.1f})",
            f"• Support Approach: {emotional_ctx.get('support_approach', 'standard_guidance')}",
            f"• Reassurance: {'High priority' if emotional_ctx.get('reassurance_needed') else 'Standard approach'}",
            "",
            "⚡ RESPONSE OPTIMIZATION:",
            f"• Urgency Level: {context.get('response_optimization', {}).get('urgency_level', 0.0):.1f}/1.0",
            f"• Personalization Strength: {context.get('response_optimization', {}).get('personalization_strength', 0.5):.1f}/1.0",
            f"• Analysis Confidence: {context.get('response_optimization', {}).get('confidence_in_analysis', 0.5):.1f}/1.0",
            f"• Analysis Method: {meta.get('analysis_method', 'standard')}",
            "",
            f"📱 USER MESSAGE: {message}",
            "",
            "RESPONSE GUIDELINES:",
            f"• Match {comm_prefs.get('formality_level', 'friendly')} communication style exactly",
            f"• Provide {comm_prefs.get('technical_depth', 'basic')}-level technical analysis",
            f"• Keep response {comm_prefs.get('response_length', 'medium')}-length",
            f"• Address {emotional_ctx.get('current_emotion', 'neutral')} emotional state appropriately",
            "• Focus on requested symbols and trading intent",
            "• Use personality insights to maximize relevance and engagement",
            "",
            "Respond as their perfectly personalized trading assistant:"
        ]
        
        return "\n".join(prompt_parts)
    
    # ==========================================
    # BATCH PROCESSING FOR ALERTS
    # ==========================================
    
    async def process_batch_alerts(
        self, 
        alert_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process multiple alerts with personality awareness"""
        results = []
        
        try:
            # Group alerts by user for batch personality analysis
            user_alerts = {}
            for alert in alert_data:
                user_phone = alert.get("user_phone")
                if user_phone not in user_alerts:
                    user_alerts[user_phone] = []
                user_alerts[user_phone].append(alert)
            
            # Process each user's alerts with personality context
            for user_phone, alerts in user_alerts.items():
                try:
                    # Get user profile once per user
                    user_profile = await self.personality_engine.get_user_profile(user_phone)
                    
                    # Process all alerts for this user
                    for alert in alerts:
                        alert_message = alert.get("message", "")
                        
                        # Get personality context for alert
                        personality_context = await self.personality_engine.get_real_time_personality_context(
                            user_phone, 
                            alert_message
                        )
                        
                        # Build enhanced context
                        enhanced_context = self._build_enhanced_context(
                            user_profile, 
                            personality_context, 
                            {"alert_type": alert.get("type", "general")}
                        )
                        
                        # Generate personalized alert response
                        personalized_alert = await self._generate_personalized_alert(
                            alert, 
                            enhanced_context
                        )
                        
                        results.append({
                            "user_phone": user_phone,
                            "original_alert": alert,
                            "personalized_message": personalized_alert,
                            "personality_applied": True,
                            "confidence": personality_context["meta"]["confidence"]
                        })
                        
                except Exception as e:
                    logger.error(f"❌ Batch alert processing failed for {user_phone}: {e}")
                    # Add fallback alert
                    for alert in alerts:
                        results.append({
                            "user_phone": user_phone,
                            "original_alert": alert,
                            "personalized_message": alert.get("message", "Alert notification"),
                            "personality_applied": False,
                            "error": str(e)
                        })
            
            logger.info(f"✅ Processed {len(results)} personalized alerts")
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch alert processing failed: {e}")
            return []
    
    async def _generate_personalized_alert(
        self, 
        alert: Dict, 
        enhanced_context: Dict
    ) -> str:
        """Generate personalized alert message"""
        try:
            # Build alert-specific prompt
            alert_prompt = self._build_alert_prompt(alert, enhanced_context)
            
            # Generate personalized alert through trading agent
            personalized_message = await self.trading_agent.generate_response_with_context(
                prompt=alert_prompt,
                max_tokens=160  # SMS limit
            )
            
            return personalized_message
            
        except Exception as e:
            logger.error(f"❌ Personalized alert generation failed: {e}")
            return alert.get("message", "Trading alert notification")
    
    def _build_alert_prompt(self, alert: Dict, context: Dict) -> str:
        """Build prompt for personalized alert generation"""
        comm_prefs = context.get("communication_preferences", {})
        trading_ctx = context.get("trading_context", {})
        emotional_ctx = context.get("emotional_context", {})
        
        return f"""Generate a personalized SMS alert based on user's communication style.

ALERT DATA:
Type: {alert.get('type', 'general')}
Symbol: {alert.get('symbol', 'N/A')}
Price: {alert.get('price', 'N/A')}
Message: {alert.get('message', '')}

USER PREFERENCES:
• Style: {comm_prefs.get('formality_level', 'friendly')}
• Technical Level: {comm_prefs.get('technical_depth', 'basic')}
• Emotional State: {emotional_ctx.get('current_emotion', 'neutral')}

Requirements:
• Keep under 160 characters for SMS
• Match user's communication style
• Include relevant trading context
• Be actionable and clear

Generate personalized alert:"""
    
    # ==========================================
    # PERFORMANCE MONITORING
    # ==========================================
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if not self.response_times:
            return {"error": "No performance data available"}
        
        import statistics
        
        return {
            "response_times": {
                "average_ms": statistics.mean(self.response_times) * 1000,
                "median_ms": statistics.median(self.response_times) * 1000,
                "p95_ms": statistics.quantiles(self.response_times, n=20)[18] * 1000,  # 95th percentile
                "max_ms": max(self.response_times) * 1000,
                "under_4s_rate": sum(1 for t in self.response_times if t < 4.0) / len(self.response_times)
            },
            "personality_analysis": {
                "average_analysis_ms": statistics.mean(self.analysis_times),
                "median_analysis_ms": statistics.median(self.analysis_times),
                "max_analysis_ms": max(self.analysis_times)
            },
            "system_metrics": {
                "total_messages_processed": self.total_messages_processed,
                "gemini_enabled": self.personality_engine.gemini_service is not None,
                "background_analysis_enabled": self.enable_background_analysis
            },
            "gemini_usage": self.personality_engine.get_gemini_usage_stats()
        }
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Optimize system performance"""
        optimization_results = {}
        
        # Clear caches if performance is degrading
        avg_response_time = sum(self.response_times[-10:]) / len(self.response_times[-10:]) if self.response_times else 0
        
        if avg_response_time > 3.5:  # If average response time > 3.5s
            # Clear personality engine caches
            cache_optimization = await self.personality_engine.optimize_analysis_costs()
            optimization_results.update(cache_optimization)
            
            # Enable aggressive caching
            self.personality_engine.enable_aggressive_caching(cache_ttl=7200)  # 2 hours
            optimization_results["aggressive_caching_enabled"] = True
        
        # Reset performance tracking
        self.response_times = []
        self.analysis_times = []
        optimization_results["performance_tracking_reset"] = True
        
        logger.info(f"🔧 Performance optimization completed: {optimization_results}")
        return optimization_results
    
    # ==========================================
    # HEALTH CHECK AND MONITORING
    # ==========================================
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for monitoring"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {}
        }
        
        try:
            # Check personality engine
            try:
                test_analysis = await self.personality_engine.get_real_time_personality_context(
                    "health_check", 
                    "test message"
                )
                health_status["components"]["personality_engine"] = {
                    "status": "healthy",
                    "analysis_method": test_analysis["meta"]["analysis_method"],
                    "response_time_ms": test_analysis["meta"]["processing_time_ms"]
                }
            except Exception as e:
                health_status["components"]["personality_engine"] = {
                    "status": "degraded",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
            
            # Check Gemini service
            if self.personality_engine.gemini_service:
                try:
                    gemini_stats = self.personality_engine.get_gemini_usage_stats()
                    health_status["components"]["gemini_service"] = {
                        "status": "healthy",
                        "total_analyses": gemini_stats.get("total_analyses", 0),
                        "total_cost": gemini_stats.get("total_cost", 0.0)
                    }
                except Exception as e:
                    health_status["components"]["gemini_service"] = {
                        "status": "error",
                        "error": str(e)
                    }
                    health_status["status"] = "degraded"
            else:
                health_status["components"]["gemini_service"] = {
                    "status": "disabled",
                    "note": "Using regex fallback"
                }
            
            # Check database connection
            try:
                test_profile = await self.personality_engine.get_user_profile("health_check")
                health_status["components"]["database"] = {
                    "status": "healthy",
                    "profile_accessible": test_profile is not None
                }
            except Exception as e:
                health_status["components"]["database"] = {
                    "status": "error",
                    "error": str(e)
                }
                health_status["status"] = "error"
            
            # Check trading agent
            try:
                # Simple agent health check
                health_status["components"]["trading_agent"] = {
                    "status": "healthy",
                    "agent_type": type(self.trading_agent).__name__
                }
            except Exception as e:
                health_status["components"]["trading_agent"] = {
                    "status": "error",
                    "error": str(e)
                }
                health_status["status"] = "error"
            
            # Performance metrics
            if self.response_times:
                health_status["performance"] = {
                    "avg_response_time_ms": sum(self.response_times[-10:]) / len(self.response_times[-10:]) * 1000,
                    "messages_processed": self.total_messages_processed,
                    "performance_target_met": sum(self.response_times[-10:]) / len(self.response_times[-10:]) < 4.0
                }
            
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {
                "status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }


# Factory function for easy integration
async def create_enhanced_message_handler(
    db_service: DatabaseService,
    trading_agent: TradingAgent,
    gemini_api_key: str = None,
    enable_background_analysis: bool = True
) -> EnhancedMessageHandler:
    """Factory function to create enhanced message handler"""
    return EnhancedMessageHandler(
        db_service=db_service,
        trading_agent=trading_agent,
        gemini_api_key=gemini_api_key,
        enable_background_analysis=enable_background_analysis
    )
