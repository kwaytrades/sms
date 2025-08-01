# core/message_handler_enhanced.py
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
        elif formality_score < 0.3:
            return "casual"
        else:
            return "friendly"
    
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
