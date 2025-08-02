# core/personality_engine_v4_unified.py
import json
import asyncio
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from collections import defaultdict, Counter
from loguru import logger
from dataclasses import dataclass
import numpy as np
from functools import lru_cache

# Import the new Gemini service
from services.gemini_service import (
    GeminiPersonalityService, 
    PersonalityAnalysisResult,
    convert_to_message_analysis
)


@dataclass
class MessageAnalysis:
    """Enhanced message analysis with Gemini insights"""
    communication_insights: Dict[str, Any]
    trading_insights: Dict[str, Any]
    emotional_state: Dict[str, Any]
    service_needs: Dict[str, Any]
    sales_opportunity: Dict[str, Any]
    intent_analysis: Dict[str, Any]
    confidence_score: float
    analysis_method: str  # "gemini" or "fallback"
    processing_time_ms: int
    gemini_analysis: Optional[PersonalityAnalysisResult] = None


class UnifiedPersonalityEngine:
    """
    UNIFIED PersonalityEngine v4.0 - SMS + Social Media Intelligence
    Combines Gemini semantic analysis with frequency-weighted learning
    Handles both SMS and social media interactions with one smart system
    """
    
    PROFILE_VERSION = 4.0
    DEFAULT_LEARNING_RATE = 0.2
    MIN_LEARNING_RATE = 0.05
    MAX_LEARNING_RATE = 0.4
    TIME_DECAY_FACTOR = 0.95
    
    def __init__(self, db_service=None, gemini_api_key: str = None):
        """Initialize unified engine for SMS and social media"""
        self.db_service = db_service
        self.key_builder = db_service.key_builder if db_service and hasattr(db_service, 'key_builder') else None
        
        # Initialize Gemini service for semantic analysis
        self.gemini_service = None
        if gemini_api_key:
            self.gemini_service = GeminiPersonalityService(gemini_api_key)
            logger.info("🤖 Gemini semantic analysis enabled")
        else:
            logger.warning("⚠️ Gemini API key not provided - falling back to regex detection")
        
        # Load authoritative ticker lists for symbol validation
        self.authoritative_tickers = self._load_authoritative_tickers()
        
        # Event hooks for real-time integration
        self._profile_update_hooks: List[Callable] = []
        self._analysis_hooks: List[Callable] = []
        
        # Fallback in-memory storage for when KeyBuilder is not available
        self.user_profiles = defaultdict(self._create_default_profile)
        
        # Pre-compiled regex patterns for performance (fallback only)
        self._compiled_patterns = self._compile_regex_patterns()
        
        # Analysis patterns (loaded once) - kept for fallback
        self.communication_patterns = self._load_communication_patterns()
        self.trading_patterns = self._load_trading_patterns()
        self.sales_indicators = self._load_sales_indicators()
        self.service_patterns = self._load_service_patterns()
        
        # Cached analysis results (prevents duplicate processing)
        self._analysis_cache = {}
        self._cache_max_size = 1000
        
        # Global intelligence layer for pattern aggregation
        self._global_patterns = defaultdict(lambda: defaultdict(int))
        
        logger.info(f"🧠 Unified PersonalityEngine v{self.PROFILE_VERSION} initialized")

    def _create_default_profile(self) -> Dict[str, Any]:
        """Create enhanced default user profile with frequency-weighted features"""
        return {
            "profile_version": self.PROFILE_VERSION,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "confidence_score": 0.1,  # Start with low confidence
            
            # ENHANCED COMMUNICATION STYLE ANALYSIS
            "communication_style": {
                "formality": 0.5,  # 0.0 = very casual, 1.0 = very formal
                "energy": "moderate",  # low, moderate, high, very_high
                "emoji_usage": 0.0,
                "message_length": "medium",  # short, medium, long
                "technical_depth": "basic",  # basic, intermediate, advanced
                "question_frequency": 0.0,
                "urgency_patterns": [],
                "consistency_score": 1.0,
                # NEW: Platform adaptation patterns
                "platform_adaptation": {
                    "sms": {"formality": 0.5, "emoji_usage": 0.2},
                    "twitter": {"formality": 0.4, "emoji_usage": 0.6},
                    "tiktok": {"formality": 0.2, "emoji_usage": 0.8}
                }
            },
            
            # ENHANCED TRADING PERSONALITY PROFILE
            "trading_personality": {
                "risk_tolerance": "moderate",  # conservative, moderate, aggressive
                "trading_style": "swing",  # day, swing, long_term, mixed
                "experience_level": "intermediate",  # novice, intermediate, advanced, expert
                "common_symbols": [],
                "sector_preferences": [],
                "strategy_preferences": [],
                "decision_making_style": "analytical",  # impulsive, analytical, consensus_seeking
                "loss_reaction": "neutral",  # emotional, neutral, analytical
                # NEW: Frequency-weighted learning
                "symbol_frequency": {},      # {"AAPL": 12.5, "TSLA": 8.3}
                "topic_frequency": {},       # {"technical_analysis": 15.2, "earnings": 8.7}
                "symbol_relationships": {},  # {"AAPL": {"TSLA": 5.2, "NVDA": 3.1}}
                "trading_focus": []          # Enhanced with frequency data
            },
            
            # EMOTIONAL INTELLIGENCE PROFILE
            "emotional_intelligence": {
                "dominant_emotion": "neutral",
                "emotional_volatility": 0.0,
                "emotional_consistency": 1.0,
                "frustration_triggers": [],
                "excitement_triggers": [],
                "support_needs": "standard_guidance"
            },
            
            # ENHANCED CONTEXT MEMORY SYSTEM
            "context_memory": {
                "last_discussed_stocks": [],
                "recent_topics": [],
                "goals_mentioned": [],
                "concerns_expressed": [],
                "relationship_stage": "new",  # new, developing, established
                "conversation_frequency": "occasional",  # rare, occasional, regular, frequent
                # NEW: Cross-channel context
                "channel_activity": {
                    "sms": {"last_active": None, "message_count": 0},
                    "twitter": {"last_active": None, "message_count": 0},
                    "tiktok": {"last_active": None, "message_count": 0}
                }
            },
            
            # ENHANCED LEARNING DATA
            "learning_data": {
                "total_messages": 0,
                "successful_predictions": 0,
                "learning_rate": self.DEFAULT_LEARNING_RATE,
                "last_learning_update": datetime.utcnow().isoformat(),
                "successful_trades_mentioned": 0,
                "loss_trades_mentioned": 0,
                "pattern_recognition_score": 0.0,
                # NEW: LLM learning optimization
                "llm_update_count": 0,
                "last_llm_update": datetime.utcnow().isoformat(),
                "llm_confidence_threshold": 0.6,
                "learning_velocity": "unknown"  # low, moderate, high, very_high
            },
            
            # RESPONSE ADAPTATION SETTINGS
            "response_patterns": {
                "preferred_response_length": "medium",
                "technical_detail_level": "standard",
                "humor_acceptance": 0.5,
                "educational_content_preference": 0.5,
                "news_update_frequency": "daily"
            },
            
            # ENHANCED GEMINI INSIGHTS
            "gemini_insights": {
                "semantic_profile_confidence": 0.0,
                "last_gemini_analysis": None,
                "communication_nuances": {},
                "trading_sentiment_patterns": {},
                "emotional_depth_analysis": {},
                "cross_conversation_insights": {},
                # NEW: Cross-platform semantic insights
                "platform_behavior_patterns": {},
                "unified_personality_traits": {}
            },
            
            # NEW: SEMANTIC CLASSIFICATION STORAGE (from social system)
            "semantic_classifications": {
                "intent_classifications": [],
                "journey_progression": [],
                "engagement_style_evolution": [],
                "conversion_opportunities": []
            },
            
            # NEW: SOCIAL MEDIA FEATURES
            "social_media_profile": {
                "platform_accounts": {},  # {"twitter": {...}, "tiktok": {...}}
                "influence_level": "unknown",  # micro, macro, mega, unknown
                "engagement_style": "lurker",  # question_asker, advice_giver, skeptic, supporter, lurker, technical_analyst
                "conversion_stage": "discovery",  # discovery, engagement, recognition, collaboration, advocacy, converted
                "relationship_score": 0,  # 0-100 scale
                "cross_platform_consistency": 0.0
            },
            
            # ADVANCED PERSONALIZATION METADATA
            "personalization_metadata": {
                "timezone_detected": None,
                "session_patterns": {},
                "response_feedback_loop": [],
                "a_b_testing_group": "control",
                "personalization_effectiveness": 0.5,
                # NEW: Unified intelligence features
                "unified_intelligence_enabled": True,
                "primary_channel": "sms",  # sms, twitter, tiktok, etc.
                "channel_preferences": {}
            }
        }

    # ==========================================
    # FREQUENCY-WEIGHTED LEARNING METHODS (from social system)
    # ==========================================
    
    def update_trading_interests_with_frequency(self, user_id: str, symbols: List[str], 
                                              topics: List[str], confidence: float = 1.0,
                                              channel: str = "sms"):
        """
        FREQUENCY-WEIGHTED topic and symbol tracking
        Higher mentioned topics/symbols get priority in response generation
        """
        try:
            profile = self.user_profiles[user_id]
            
            # Get confidence threshold for this user
            threshold = profile["learning_data"]["llm_confidence_threshold"]
            
            # Only update if we have reasonable confidence
            if confidence < threshold:
                logger.info(f"Skipping low confidence update for {user_id}: {confidence} < {threshold}")
                return
            
            # Update topic frequency with weighting
            topic_freq = profile["trading_personality"]["topic_frequency"]
            for topic in topics:
                if topic not in topic_freq:
                    topic_freq[topic] = 0
                topic_freq[topic] += confidence  # Weighted by confidence
                
                # Add to trading focus if high frequency
                if (topic not in profile["trading_personality"]["trading_focus"] and 
                    topic_freq[topic] >= 2.0):  # Threshold for inclusion
                    profile["trading_personality"]["trading_focus"].append(topic)
            
            # Update symbol frequency with weighting
            symbol_freq = profile["trading_personality"]["symbol_frequency"]
            for symbol in symbols:
                if symbol not in symbol_freq:
                    symbol_freq[symbol] = 0
                symbol_freq[symbol] += confidence
                
                # Add to common symbols if high frequency
                if (symbol not in profile["trading_personality"]["common_symbols"] and 
                    symbol_freq[symbol] >= 2.0):
                    profile["trading_personality"]["common_symbols"].append(symbol)
            
            # Build symbol relationship graph
            self._update_symbol_relationships(user_id, symbols, confidence)
            
            # Update LLM tracking
            profile["learning_data"]["last_llm_update"] = datetime.utcnow().isoformat()
            profile["learning_data"]["llm_update_count"] += 1
            
            # Update channel activity
            profile["context_memory"]["channel_activity"][channel]["last_active"] = datetime.utcnow().isoformat()
            profile["context_memory"]["channel_activity"][channel]["message_count"] += 1
            
            # Update learning velocity
            self._calculate_learning_velocity(user_id)
            
            logger.info(f"Updated interests for {user_id}: topics={topics}, symbols={symbols}, confidence={confidence}, channel={channel}")
            
        except Exception as e:
            logger.error(f"Failed to update trading interests with frequency: {e}")
    
    def _update_symbol_relationships(self, user_id: str, symbols: List[str], confidence: float):
        """
        BUILD SYMBOL AFFINITY GRAPH
        Track which symbols user mentions together (co-occurrence)
        """
        try:
            if len(symbols) < 2:
                return  # Need at least 2 symbols for relationships
            
            profile = self.user_profiles[user_id]
            symbol_relationships = profile["trading_personality"]["symbol_relationships"]
            
            # Create relationships between all symbol pairs
            for i, symbol1 in enumerate(symbols):
                if symbol1 not in symbol_relationships:
                    symbol_relationships[symbol1] = {}
                
                for j, symbol2 in enumerate(symbols):
                    if i != j:  # Don't relate symbol to itself
                        if symbol2 not in symbol_relationships[symbol1]:
                            symbol_relationships[symbol1][symbol2] = 0
                        
                        # Weighted by confidence
                        symbol_relationships[symbol1][symbol2] += confidence
            
            logger.debug(f"Updated symbol relationships for {user_id}: {len(symbols)} symbols co-occurred")
            
        except Exception as e:
            logger.error(f"Failed to update symbol relationships: {e}")
    
    def get_top_topics_by_frequency(self, user_id: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Get most frequently mentioned topics for response personalization"""
        try:
            profile = self.user_profiles[user_id]
            topic_freq = profile["trading_personality"]["topic_frequency"]
            
            sorted_topics = sorted(
                topic_freq.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            return sorted_topics[:limit]
        except Exception as e:
            logger.error(f"Failed to get top topics for {user_id}: {e}")
            return []
    
    def get_top_symbols_by_frequency(self, user_id: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Get most frequently mentioned symbols for response personalization"""
        try:
            profile = self.user_profiles[user_id]
            symbol_freq = profile["trading_personality"]["symbol_frequency"]
            
            sorted_symbols = sorted(
                symbol_freq.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_symbols[:limit]
        except Exception as e:
            logger.error(f"Failed to get top symbols for {user_id}: {e}")
            return []
    
    def get_related_symbols(self, user_id: str, symbol: str, limit: int = 3) -> List[Tuple[str, float]]:
        """
        GET SYMBOL AFFINITY for response personalization
        If user mentions AAPL, suggest related symbols they often discuss
        """
        try:
            profile = self.user_profiles[user_id]
            symbol_relationships = profile["trading_personality"]["symbol_relationships"]
            
            if symbol not in symbol_relationships:
                return []
            
            related = sorted(
                symbol_relationships[symbol].items(),
                key=lambda x: x[1],
                reverse=True
            )
            return related[:limit]
        except Exception as e:
            logger.error(f"Failed to get related symbols for {user_id}: {e}")
            return []
    
    def _calculate_learning_velocity(self, user_id: str) -> str:
        """Calculate how fast we're learning about the user"""
        try:
            profile = self.user_profiles[user_id]
            
            created_at = datetime.fromisoformat(profile["created_at"])
            days_since_creation = (datetime.utcnow() - created_at).days + 1
            
            llm_updates = profile["learning_data"]["llm_update_count"]
            updates_per_day = llm_updates / days_since_creation
            
            if updates_per_day > 2.0:
                velocity = "very_high"
            elif updates_per_day > 1.0:
                velocity = "high"
            elif updates_per_day > 0.3:
                velocity = "moderate"
            else:
                velocity = "low"
            
            profile["learning_data"]["learning_velocity"] = velocity
            return velocity
            
        except Exception as e:
            logger.error(f"Failed to calculate learning velocity for {user_id}: {e}")
            return "unknown"
    
    def get_personalization_context(self, user_id: str) -> Dict[str, Any]:
        """
        COMPREHENSIVE CONTEXT for response generation
        Combines frequency data, relationships, and learning velocity
        """
        try:
            profile = self.user_profiles[user_id]
            
            top_topics = self.get_top_topics_by_frequency(user_id, 3)
            top_symbols = self.get_top_symbols_by_frequency(user_id, 3)
            
            topic_freq = profile["trading_personality"]["topic_frequency"]
            symbol_freq = profile["trading_personality"]["symbol_frequency"]
            
            return {
                "top_topics": top_topics,
                "top_symbols": top_symbols,
                "learning_velocity": {
                    "rate": profile["learning_data"]["learning_velocity"],
                    "updates_per_day": profile["learning_data"]["llm_update_count"] / max(1, 
                        (datetime.utcnow() - datetime.fromisoformat(profile["created_at"])).days + 1),
                    "total_updates": profile["learning_data"]["llm_update_count"],
                    "last_update": profile["learning_data"]["last_llm_update"]
                },
                "total_topic_mentions": sum(topic_freq.values()),
                "total_symbol_mentions": sum(symbol_freq.values()),
                "relationship_graph_size": len(profile["trading_personality"]["symbol_relationships"]),
                "confidence_threshold": profile["learning_data"]["llm_confidence_threshold"],
                "cross_channel_data": profile["context_memory"]["channel_activity"]
            }
        except Exception as e:
            logger.error(f"Failed to get personalization context for {user_id}: {e}")
            return {}

    # ==========================================
    # MAIN ANALYSIS METHOD - ENHANCED GEMINI INTEGRATION
    # ==========================================
    
    async def analyze_message_unified(
        self, 
        user_id: str,
        user_message: str, 
        channel: str = "sms",
        context: Dict[str, Any] = None,
        force_gemini: bool = False
    ) -> MessageAnalysis:
        """
        UNIFIED message analysis for SMS and social media
        Uses Gemini semantic intelligence with frequency-weighted learning
        """
        try:
            # Get existing user profile for context
            profile = self.user_profiles[user_id]
            
            # Build enhanced context with user history
            enhanced_context = {
                "channel": channel,
                "user_profile": profile,
                "personalization_context": self.get_personalization_context(user_id),
                "cross_channel_history": self._get_cross_channel_context(user_id),
                **(context or {})
            }
            
            # Check analysis cache first
            cache_key = f"{hash(user_message)}_{hash(str(enhanced_context))}_{channel}"
            if cache_key in self._analysis_cache and not force_gemini:
                logger.debug("📋 Using cached analysis result")
                return self._analysis_cache[cache_key]
            
            # Try Gemini semantic analysis first
            if self.gemini_service:
                try:
                    gemini_result = await self.gemini_service.analyze_personality_semantic(
                        user_message, enhanced_context
                    )
                    
                    # Convert to MessageAnalysis format
                    analysis = MessageAnalysis(
                        communication_insights=gemini_result.communication_insights,
                        trading_insights=gemini_result.trading_insights,
                        emotional_state=gemini_result.emotional_state,
                        service_needs=gemini_result.service_needs,
                        sales_opportunity=gemini_result.sales_opportunity,
                        intent_analysis=gemini_result.intent_analysis,
                        confidence_score=gemini_result.confidence_score,
                        analysis_method="gemini",
                        processing_time_ms=gemini_result.processing_time_ms,
                        gemini_analysis=gemini_result
                    )
                    
                    # Enhanced symbol validation using authoritative list
                    analysis = self._enhance_symbol_validation(analysis, user_message)
                    
                    # FREQUENCY-WEIGHTED LEARNING INTEGRATION
                    await self._integrate_analysis_into_profile(user_id, analysis, channel)
                    
                    logger.info(f"✅ Gemini semantic analysis completed for {user_id} (confidence: {gemini_result.confidence_score:.2f})")
                    
                except Exception as e:
                    logger.error(f"❌ Gemini analysis failed, using fallback: {e}")
                    analysis = await self._run_fallback_analysis(user_message, enhanced_context)
            else:
                # Use fallback regex-based analysis
                analysis = await self._run_fallback_analysis(user_message, enhanced_context)
            
            # Cache result
            if len(self._analysis_cache) >= self._cache_max_size:
                oldest_keys = list(self._analysis_cache.keys())[:100]
                for key in oldest_keys:
                    del self._analysis_cache[key]
            
            self._analysis_cache[cache_key] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error in unified analysis: {e}")
            return await self._create_minimal_analysis(user_message)
    
    async def _integrate_analysis_into_profile(self, user_id: str, analysis: MessageAnalysis, channel: str):
        """
        INTEGRATE ANALYSIS RESULTS with frequency-weighted learning
        """
        try:
            # Extract symbols and topics from analysis
            symbols = analysis.trading_insights.get('symbols_mentioned', [])
            topics = self._extract_topics_from_analysis(analysis)
            confidence = analysis.confidence_score
            
            # Update trading interests with frequency weighting
            self.update_trading_interests_with_frequency(user_id, symbols, topics, confidence, channel)
            
            # Store semantic classification
            self._store_semantic_classification(user_id, analysis, channel)
            
            # Update communication style from analysis
            self._update_communication_style_from_analysis(user_id, analysis, channel)
            
            # Update profile metadata
            profile = self.user_profiles[user_id]
            profile["updated_at"] = datetime.utcnow().isoformat()
            profile["learning_data"]["total_messages"] += 1
            
            logger.debug(f"Integrated analysis into profile for {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to integrate analysis into profile: {e}")
    
    def _extract_topics_from_analysis(self, analysis: MessageAnalysis) -> List[str]:
        """Extract trading topics from analysis results"""
        topics = []
        
        # From trading insights
        trading_insights = analysis.trading_insights
        if trading_insights.get('trading_action') and trading_insights['trading_action'] != 'unclear':
            topics.append(trading_insights['trading_action'])
        
        # From intent analysis
        intent_analysis = analysis.intent_analysis
        if 'technical_analysis' in intent_analysis.get('requires_tools', []):
            topics.append('technical_analysis')
        
        # Add more topic extraction logic based on your analysis structure
        
        return topics
    
    def _store_semantic_classification(self, user_id: str, analysis: MessageAnalysis, channel: str):
        """Store semantic classification for longitudinal analysis"""
        try:
            profile = self.user_profiles[user_id]
            
            classification_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "channel": channel,
                "content_hash": hash(str(analysis.trading_insights.get('symbols_mentioned', []))),
                "intent": analysis.intent_analysis.get("primary_intent", "general"),
                "sentiment": analysis.emotional_state.get("primary_emotion", "neutral"),
                "confidence": analysis.confidence_score,
                "symbols_mentioned": analysis.trading_insights.get('symbols_mentioned', []),
                "topics_discussed": self._extract_topics_from_analysis(analysis),
                "analysis_method": analysis.analysis_method
            }
            
            # Add to semantic classifications
            semantic_classifications = profile["semantic_classifications"]["intent_classifications"]
            semantic_classifications.append(classification_record)
            
            # Keep only last 50 classifications
            if len(semantic_classifications) > 50:
                profile["semantic_classifications"]["intent_classifications"] = semantic_classifications[-50:]
            
        except Exception as e:
            logger.error(f"Failed to store semantic classification: {e}")
    
    def _update_communication_style_from_analysis(self, user_id: str, analysis: MessageAnalysis, channel: str):
        """Update communication style from analysis with channel-specific adaptation"""
        try:
            profile = self.user_profiles[user_id]
            comm_style = profile["communication_style"]
            
            # Update platform-specific communication patterns
            platform_adaptation = comm_style["platform_adaptation"]
            if channel not in platform_adaptation:
                platform_adaptation[channel] = {"formality": 0.5, "emoji_usage": 0.2}
            
            # Get insights from analysis
            comm_insights = analysis.communication_insights
            
            # Update formality (weighted average with existing)
            if 'formality_score' in comm_insights:
                current_formality = platform_adaptation[channel]["formality"]
                new_formality = (current_formality * 0.7) + (comm_insights['formality_score'] * 0.3)
                platform_adaptation[channel]["formality"] = new_formality
                
                # Update global formality
                comm_style["formality"] = (comm_style["formality"] * 0.8) + (new_formality * 0.2)
            
            # Update emoji usage
            if 'emoji_usage' in comm_insights:
                emoji_score = comm_insights['emoji_usage']
                current_emoji = platform_adaptation[channel]["emoji_usage"]
                new_emoji = (current_emoji * 0.7) + (emoji_score * 0.3)
                platform_adaptation[channel]["emoji_usage"] = new_emoji
                
                # Update global emoji usage
                comm_style["emoji_usage"] = (comm_style["emoji_usage"] * 0.8) + (emoji_score * 0.2)
            
        except Exception as e:
            logger.error(f"Failed to update communication style from analysis: {e}")
    
    def _get_cross_channel_context(self, user_id: str) -> Dict[str, Any]:
        """Get cross-channel context for better analysis"""
        try:
            profile = self.user_profiles[user_id]
            channel_activity = profile["context_memory"]["channel_activity"]
            
            # Calculate cross-channel patterns
            total_messages = sum(ch.get("message_count", 0) for ch in channel_activity.values())
            active_channels = [ch for ch, data in channel_activity.items() 
                             if data.get("message_count", 0) > 0]
            
            return {
                "total_cross_channel_messages": total_messages,
                "active_channels": active_channels,
                "primary_channel": max(channel_activity.items(), 
                                     key=lambda x: x[1].get("message_count", 0))[0] if channel_activity else "sms",
                "channel_diversity": len(active_channels)
            }
            
        except Exception as e:
            logger.error(f"Failed to get cross-channel context: {e}")
            return {}

    # ==========================================
    # RESPONSE GENERATION CONTEXT
    # ==========================================
    
    async def get_response_context(self, user_id: str, message: str, channel: str = "sms") -> Dict[str, Any]:
        """
        Get comprehensive context for response generation
        Combines personality insights with frequency-weighted data
        """
        try:
            # Run analysis
            analysis = await self.analyze_message_unified(user_id, message, channel)
            
            # Get user profile
            profile = self.user_profiles[user_id]
            
            # Get personalization context
            personalization = self.get_personalization_context(user_id)
            
            # Build comprehensive context
            context = {
                "user_id": user_id,
                "channel": channel,
                "analysis": {
                    "communication_style": analysis.communication_insights,
                    "trading_focus": analysis.trading_insights,
                    "emotional_state": analysis.emotional_state,
                    "confidence": analysis.confidence_score,
                    "analysis_method": analysis.analysis_method
                },
                "personality": {
                    "formality": profile["communication_style"]["formality"],
                    "energy_level": profile["communication_style"]["energy"],
                    "technical_depth": profile["communication_style"]["technical_depth"],
                    "emoji_preference": profile["communication_style"]["emoji_usage"],
                    "risk_tolerance": profile["trading_personality"]["risk_tolerance"],
                    "experience_level": profile["trading_personality"]["experience_level"]
                },
                "frequency_insights": {
                    "top_topics": personalization["top_topics"],
                    "top_symbols": personalization["top_symbols"],
                    "learning_velocity": personalization["learning_velocity"],
                    "symbol_relationships": self._get_symbol_relationships_for_context(user_id, 
                        analysis.trading_insights.get('symbols_mentioned', []))
                },
                "cross_channel": {
                    "active_channels": personalization["cross_channel_data"],
                    "primary_channel": profile["personalization_metadata"]["primary_channel"],
                    "channel_consistency": profile["social_media_profile"]["cross_platform_consistency"]
                },
                "relationship": {
                    "stage": profile["context_memory"]["relationship_stage"],
                    "total_messages": profile["learning_data"]["total_messages"],
                    "confidence_score": profile["confidence_score"]
                }
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get response context: {e}")
            return self._get_minimal_response_context(user_id, channel)
    
    def _get_symbol_relationships_for_context(self, user_id: str, mentioned_symbols: List[str]) -> List[str]:
        """Get related symbols for mentioned symbols"""
        related_symbols = []
        
        for symbol in mentioned_symbols[:2]:  # Check first 2 mentioned symbols
            related = self.get_related_symbols(user_id, symbol, 2)
            for rel_symbol, strength in related:
                if rel_symbol not in mentioned_symbols and rel_symbol not in related_symbols:
                    related_symbols.append(rel_symbol)
        
        return related_symbols[:3]  # Return max 3 related symbols
    
    def _get_minimal_response_context(self, user_id: str, channel: str) -> Dict[str, Any]:
        """Get minimal context when full analysis fails"""
        return {
            "user_id": user_id,
            "channel": channel,
            "analysis": {"confidence": 0.1, "analysis_method": "minimal"},
            "personality": {"formality": 0.5, "energy_level": "moderate", "technical_depth": "basic"},
            "frequency_insights": {"top_topics": [], "top_symbols": [], "learning_velocity": {"rate": "unknown"}},
            "cross_channel": {"active_channels": {}, "primary_channel": channel},
            "relationship": {"stage": "new", "total_messages": 0, "confidence_score": 0.1}
        }

    # ==========================================
    # SOCIAL MEDIA INTEGRATION METHODS
    # ==========================================
    
    async def add_social_platform(self, user_id: str, platform: str, username: str, 
                                 follower_count: int = 0, verified: bool = False):
        """Add social media platform to user profile"""
        try:
            profile = self.user_profiles[user_id]
            
            platform_data = {
                "username": username,
                "follower_count": follower_count,
                "verified": verified,
                "last_updated": datetime.utcnow().isoformat(),
                "engagement_rate": 0.0,
                "bio": "",
                "profile_url": ""
            }
            
            profile["social_media_profile"]["platform_accounts"][platform] = platform_data
            
            # Update influence level
            self._calculate_influence_level(user_id)
            
            # Update primary channel if this is their first social platform
            if profile["personalization_metadata"]["primary_channel"] == "sms" and follower_count > 100:
                profile["personalization_metadata"]["primary_channel"] = platform
            
            logger.info(f"Added {platform} account for {user_id}: @{username}")
            
        except Exception as e:
            logger.error(f"Failed to add social platform: {e}")
    
    def _calculate_influence_level(self, user_id: str):
        """Calculate user's influence level across platforms"""
        try:
            profile = self.user_profiles[user_id]
            platform_accounts = profile["social_media_profile"]["platform_accounts"]
            
            total_followers = sum(acc.get("follower_count", 0) for acc in platform_accounts.values())
            
            if total_followers >= 100000:
                influence_level = "mega"
            elif total_followers >= 10000:
                influence_level = "macro"
            elif total_followers >= 1000:
                influence_level = "micro"
            else:
                influence_level = "unknown"
            
            profile["social_media_profile"]["influence_level"] = influence_level
            
        except Exception as e:
            logger.error(f"Failed to calculate influence level: {e}")
    
    def update_engagement_style(self, user_id: str, engagement_style: str, confidence: float = 0.8):
        """Update user's engagement style (for social media)"""
        try:
            if confidence < 0.7:  # High threshold for style changes
                return
            
            profile = self.user_profiles[user_id]
            old_style = profile["social_media_profile"]["engagement_style"]
            
            if engagement_style != old_style:
                profile["social_media_profile"]["engagement_style"] = engagement_style
                
                # Track evolution
                evolution_record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "from_style": old_style,
                    "to_style": engagement_style,
                    "confidence": confidence
                }
                
                profile["semantic_classifications"]["engagement_style_evolution"].append(evolution_record)
                
                logger.info(f"User {user_id} engagement style evolved: {old_style} → {engagement_style}")
            
        except Exception as e:
            logger.error(f"Failed to update engagement style: {e}")

    # ==========================================
    # KEEP EXISTING METHODS FOR COMPATIBILITY
    # ==========================================
    
    def _load_authoritative_tickers(self) -> set:
        """Load authoritative ticker list - same as v3"""
        popular_tickers = [
            # 🔥 MAGNIFICENT 7 + AI GIANTS
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            
            # 🤖 AI & MACHINE LEARNING
            'NVDA', 'AMD', 'SMCI', 'ARM', 'AVGO', 'MRVL', 'QCOM', 'MU',
            'PLTR', 'C3AI', 'AI', 'BBAI', 'SOUN', 'STEM', 'PATH', 'UPST',
            
            # 🔬 QUANTUM COMPUTING
            'IBM', 'GOOGL', 'IONQ', 'RGTI', 'QBTS', 'ARQQ',
            
            # 💻 SEMICONDUCTOR POWERHOUSES
            'NVDA', 'AMD', 'INTC', 'TSM', 'ASML', 'LRCX', 'KLAC', 'AMAT',
            
            # Popular ETFs and crypto
            'SPY', 'QQQ', 'BTC', 'ETH', 'DOGE'
        ]
        return set(popular_tickers)
    
    def _compile_regex_patterns(self) -> Dict[str, re.Pattern]:
        """Pre-compile regex patterns for performance - same as v3"""
        return {
            'potential_symbols': re.compile(r'\b[A-Z]{2,5}\b'),
            'money_amounts': re.compile(r'\$([0-9,]+(?:\.[0-9]{2})?)'),
            'percentages': re.compile(r'([0-9]+(?:\.[0-9]+)?)%'),
            'emojis': re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U0001f900-\U0001f9ff\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f1e0-\U0001f1ff]'),
        }
    
    def _load_communication_patterns(self) -> Dict[str, Any]:
        """Load communication analysis patterns - same as v3"""
        return {
            'formality_indicators': {
                'formal': ['please', 'thank you', 'kindly', 'respectfully', 'sincerely'],
                'casual': ['yo', 'hey', 'sup', 'dude', 'bro', 'lol', 'omg', 'wtf'],
                'professional': ['analyze', 'assessment', 'evaluation', 'consideration', 'recommendation']
            },
            'energy_indicators': {
                'high': ['!', '!!', '!!!', 'excited', 'pumped', 'amazing', 'awesome', 'love it'],
                'low': ['tired', 'meh', 'okay', 'fine', 'whatever', 'sure'],
                'moderate': ['good', 'nice', 'cool', 'interesting', 'thanks']
            }
        }
    
    def _load_trading_patterns(self) -> Dict[str, Any]:
        """Load trading behavior analysis patterns - same as v3"""
        return {
            'risk_indicators': {
                'conservative': ['safe', 'secure', 'stable', 'dividend', 'blue chip'],
                'aggressive': ['yolo', 'moon', 'rocket', 'all in', 'bet', 'gamble']
            },
            'trading_actions': {
                'buying': ['buy', 'purchase', 'get', 'acquire', 'long', 'calls'],
                'selling': ['sell', 'dump', 'exit', 'short', 'puts', 'close']
            }
        }
    
    def _load_sales_indicators(self) -> Dict[str, Any]:
        """Load sales opportunity detection patterns - same as v3"""
        return {
            'buying_signals': {
                'strong': ['need help', 'premium', 'upgrade', 'better service'],
                'moderate': ['interested', 'tell me more', 'pricing', 'cost']
            }
        }
    
    def _load_service_patterns(self) -> Dict[str, Any]:
        """Load service need detection patterns - same as v3"""
        return {
            'service_types': {
                'technical_analysis': ['chart', 'pattern', 'indicator', 'signal'],
                'fundamental_analysis': ['earnings', 'revenue', 'pe ratio', 'financials']
            }
        }
    
    # Keep all other existing methods from v3 (fallback analysis, etc.)
    async def _run_fallback_analysis(self, message: str, context: Dict = None) -> MessageAnalysis:
        """Run fallback regex-based analysis when Gemini is unavailable - same as v3"""
        start_time = asyncio.get_event_loop().time()
        
        # Use original regex-based preprocessing
        preprocessed = self._preprocess_message(message)
        
        # Basic analysis
        communication_analysis = {"formality_score": 0.5, "energy_level": "moderate"}
        trading_analysis = {"symbols_mentioned": preprocessed.get('symbols', []), "trading_action": "unclear"}
        emotional_analysis = {"primary_emotion": "neutral", "emotional_intensity": 0.0}
        intent_analysis = {"primary_intent": "general_chat"}
        service_analysis = {"service_type": "none", "urgency_level": 0.0}
        sales_analysis = {"sales_readiness_score": 0.5}
        
        processing_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        return MessageAnalysis(
            communication_insights=communication_analysis,
            trading_insights=trading_analysis,
            emotional_state=emotional_analysis,
            service_needs=service_analysis,
            sales_opportunity=sales_analysis,
            intent_analysis=intent_analysis,
            confidence_score=0.6,
            analysis_method="regex_fallback",
            processing_time_ms=processing_time_ms,
            gemini_analysis=None
        )
    
    def _preprocess_message(self, message: str) -> Dict[str, Any]:
        """Basic preprocessing for fallback analysis - same as v3"""
        symbols = []
        # Basic symbol extraction
        words = message.upper().split()
        for word in words:
            if word in self.authoritative_tickers:
                symbols.append(word)
        
        return {
            'original': message,
            'lower': message.lower(),
            'symbols': symbols,
            'words': message.split()
        }
    
    def _enhance_symbol_validation(self, analysis: MessageAnalysis, message: str) -> MessageAnalysis:
        """Enhance symbol detection with authoritative ticker validation - same as v3"""
        symbols_mentioned = analysis.trading_insights.get('symbols_mentioned', [])
        
        # Validate symbols
        validated_symbols = []
        for symbol in symbols_mentioned:
            if symbol.upper() in self.authoritative_tickers:
                validated_symbols.append(symbol.upper())
        
        analysis.trading_insights['symbols_mentioned'] = validated_symbols
        
        return analysis
    
    async def _create_minimal_analysis(self, message: str) -> MessageAnalysis:
        """Create minimal analysis when everything fails - same as v3"""
        return MessageAnalysis(
            communication_insights={"formality_score": 0.5, "energy_level": "moderate"},
            trading_insights={"symbols_mentioned": [], "trading_action": "unclear"},
            emotional_state={"primary_emotion": "neutral", "emotional_intensity": 0.0},
            service_needs={"service_type": "none", "urgency_level": 0.0},
            sales_opportunity={"sales_readiness_score": 0.5},
            intent_analysis={"primary_intent": "general_chat"},
            confidence_score=0.1,
            analysis_method="minimal_fallback",
            processing_time_ms=1,
            gemini_analysis=None
        )

    # ==========================================
    # LEGACY COMPATIBILITY METHODS
    # ==========================================
    
    async def analyze_and_learn(self, user_id: str, user_message: str, bot_response: str = None, 
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Legacy compatibility method - redirects to new unified analysis
        """
        try:
            channel = context.get("channel", "sms") if context else "sms"
            analysis = await self.analyze_message_unified(user_id, user_message, channel, context)
            
            return {
                "communication_insights": analysis.communication_insights,
                "trading_insights": analysis.trading_insights,
                "emotional_state": analysis.emotional_state,
                "service_needs": analysis.service_needs,
                "sales_opportunity": analysis.sales_opportunity,
                "analysis_method": analysis.analysis_method,
                "confidence": analysis.confidence_score,
                "processing_time_ms": analysis.processing_time_ms,
                "unified_intelligence": True,
                "frequency_data": self.get_personalization_context(user_id)
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_and_learn for {user_id}: {e}")
            return {"error": str(e)}
    
    def learn_from_message(self, user_id: str, message: str, intent_data: Dict, channel: str = "sms"):
        """
        Legacy compatibility method with frequency-weighted learning
        """
        try:
            # Extract symbols and topics from intent data
            symbols = intent_data.get('symbols', [])
            topics = []
            
            # Map intent to topics
            intent = intent_data.get('intent', 'general')
            if intent == 'analyze':
                topics.append('technical_analysis')
            elif intent == 'portfolio':
                topics.append('portfolio_management')
            
            confidence = intent_data.get('confidence', 0.8)
            
            # Update with frequency weighting
            self.update_trading_interests_with_frequency(user_id, symbols, topics, confidence, channel)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to learn from message: {e}")
            return False
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile - maintains compatibility"""
        return self.user_profiles[user_id]
    
    def get_personality_summary(self, user_id: str) -> Dict[str, Any]:
        """Get personality summary in legacy format"""
        try:
            profile = self.user_profiles[user_id]
            personalization = self.get_personalization_context(user_id)
            
            return {
                "communication_style": profile["communication_style"],
                "trading_personality": profile["trading_personality"],
                "learning_progress": {
                    "total_messages": profile["learning_data"]["total_messages"],
                    "confidence_score": profile["confidence_score"],
                    "learning_velocity": profile["learning_data"]["learning_velocity"]
                },
                "frequency_insights": {
                    "top_topics": personalization["top_topics"],
                    "top_symbols": personalization["top_symbols"],
                    "relationship_graph_size": personalization["relationship_graph_size"]
                },
                "unified_intelligence": True,
                "profile_version": self.PROFILE_VERSION
            }
            
        except Exception as e:
            logger.error(f"Failed to get personality summary: {e}")
            return {"error": str(e)}


# ==========================================
# FACTORY FUNCTION
# ==========================================

async def create_unified_personality_engine(
    db_service=None, 
    gemini_api_key: str = None
) -> UnifiedPersonalityEngine:
    """Factory function to create unified personality engine"""
    return UnifiedPersonalityEngine(db_service, gemini_api_key)


# ==========================================
# BACKWARD COMPATIBILITY ALIASES
# ==========================================

# For existing SMS bot code
UserPersonalityEngine = UnifiedPersonalityEngine
EnhancedPersonalityEngine = UnifiedPersonalityEngine

# Export for use in main application
__all__ = [
    'UnifiedPersonalityEngine',
    'UserPersonalityEngine',  # Backward compatibility
    'EnhancedPersonalityEngine',  # Backward compatibility
    'MessageAnalysis',
    'create_unified_personality_engine'
]
