v4_unified_complete.py
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
    UNIFIED PersonalityEngine v4.0 - Complete Feature Set
    Combines ALL v3 advanced features with v4 frequency-weighted learning
    
    NEW FEATURES MERGED FROM V3:
    - Deep background analysis pipeline
    - Rich fallback regex analysis
    - Global insights & response strategy layer
    - Advanced reporting, batch processing & insights
    - Cost optimization & caching controls
    - Extended profile metadata with confirmed traits
    
    V4 UNIFIED FEATURES:
    - SMS + Social Media Intelligence
    - Frequency-weighted learning
    - Cross-platform personality adaptation
    - Real-time semantic analysis
    """
    
    PROFILE_VERSION = 4.0
    DEFAULT_LEARNING_RATE = 0.2
    MIN_LEARNING_RATE = 0.05
    MAX_LEARNING_RATE = 0.4
    TIME_DECAY_FACTOR = 0.95
    
    def __init__(self, db_service=None, gemini_api_key: str = None):
        """Initialize unified engine with all v3 + v4 features"""
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
        
        # Pre-compiled regex patterns for performance (enhanced from v3)
        self._compiled_patterns = self._compile_regex_patterns()
        
        # Enhanced analysis patterns (from v3)
        self.communication_patterns = self._load_communication_patterns()
        self.trading_patterns = self._load_trading_patterns()
        self.sales_indicators = self._load_sales_indicators()
        self.service_patterns = self._load_service_patterns()
        
        # Cached analysis results (prevents duplicate processing)
        self._analysis_cache = {}
        self._cache_max_size = 1000
        
        # Global intelligence layer for pattern aggregation (restored from v3)
        self._global_patterns = defaultdict(lambda: defaultdict(int))
        
        logger.info(f"🧠 Unified PersonalityEngine v{self.PROFILE_VERSION} initialized with complete feature set")

    def _create_default_profile(self) -> Dict[str, Any]:
        """Create enhanced default user profile with complete v3 + v4 feature set"""
        return {
            "profile_version": self.PROFILE_VERSION,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "confidence_score": 0.1,  # Start with low confidence
            
            # ENHANCED COMMUNICATION STYLE ANALYSIS (v4 + v3 confirmed traits)
            "communication_style": {
                "formality": 0.5,  # 0.0 = very casual, 1.0 = very formal
                "energy": "moderate",  # low, moderate, high, very_high
                "emoji_usage": 0.0,
                "message_length": "medium",  # short, medium, long
                "technical_depth": "basic",  # basic, intermediate, advanced
                "question_frequency": 0.0,
                "urgency_patterns": [],
                "consistency_score": 1.0,
                # v3 CONFIRMED TRAITS
                "formality_from_deep_analysis": 0.5,
                "energy_consistent": "moderate",
                "technical_depth_confirmed": "basic",
                # v4 PLATFORM ADAPTATION
                "platform_adaptation": {
                    "sms": {"formality": 0.5, "emoji_usage": 0.2},
                    "twitter": {"formality": 0.4, "emoji_usage": 0.6},
                    "tiktok": {"formality": 0.2, "emoji_usage": 0.8}
                }
            },
            
            # ENHANCED TRADING PERSONALITY PROFILE (v4 frequency + v3 confirmed)
            "trading_personality": {
                "risk_tolerance": "moderate",  # conservative, moderate, aggressive
                "trading_style": "swing",  # day, swing, long_term, mixed
                "experience_level": "intermediate",  # novice, intermediate, advanced, expert
                "common_symbols": [],
                "sector_preferences": [],
                "strategy_preferences": [],
                "decision_making_style": "analytical",  # impulsive, analytical, consensus_seeking
                "loss_reaction": "neutral",  # emotional, neutral, analytical
                # v3 CONFIRMED TRAITS
                "verified_symbols": {},  # from deep analysis
                "risk_tolerance_confirmed": "moderate",
                "preferred_action": "unclear",
                "trading_focus_score": 0.0,
                # v4 FREQUENCY-WEIGHTED LEARNING
                "symbol_frequency": {},      # {"AAPL": 12.5, "TSLA": 8.3}
                "topic_frequency": {},       # {"technical_analysis": 15.2, "earnings": 8.7}
                "symbol_relationships": {},  # {"AAPL": {"TSLA": 5.2, "NVDA": 3.1}}
                "trading_focus": []          # Enhanced with frequency data
            },
            
            # EMOTIONAL INTELLIGENCE PROFILE (v3 confirmed + v4 cross-platform)
            "emotional_intelligence": {
                "dominant_emotion": "neutral",
                "emotional_volatility": 0.0,
                "emotional_consistency": 1.0,
                "frustration_triggers": [],
                "excitement_triggers": [],
                "support_needs": "standard_guidance",
                # v3 CONFIRMED TRAITS
                "dominant_emotion_confirmed": "neutral"
            },
            
            # ENHANCED CONTEXT MEMORY SYSTEM (v4 cross-channel)
            "context_memory": {
                "last_discussed_stocks": [],
                "recent_topics": [],
                "goals_mentioned": [],
                "concerns_expressed": [],
                "relationship_stage": "new",  # new, developing, established
                "conversation_frequency": "occasional",  # rare, occasional, regular, frequent
                # v4 CROSS-CHANNEL CONTEXT
                "channel_activity": {
                    "sms": {"last_active": None, "message_count": 0},
                    "twitter": {"last_active": None, "message_count": 0},
                    "tiktok": {"last_active": None, "message_count": 0}
                }
            },
            
            # ENHANCED LEARNING DATA (v4 LLM optimization)
            "learning_data": {
                "total_messages": 0,
                "successful_predictions": 0,
                "learning_rate": self.DEFAULT_LEARNING_RATE,
                "last_learning_update": datetime.utcnow().isoformat(),
                "successful_trades_mentioned": 0,
                "loss_trades_mentioned": 0,
                "pattern_recognition_score": 0.0,
                # v4 LLM LEARNING OPTIMIZATION
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
            
            # ENHANCED GEMINI INSIGHTS (v3 + v4)
            "gemini_insights": {
                "semantic_profile_confidence": 0.0,
                "last_gemini_analysis": None,
                "communication_nuances": {},
                "trading_sentiment_patterns": {},
                "emotional_depth_analysis": {},
                "cross_conversation_insights": {},
                # v4 CROSS-PLATFORM SEMANTIC INSIGHTS
                "platform_behavior_patterns": {},
                "unified_personality_traits": {}
            },
            
            # v4 SEMANTIC CLASSIFICATION STORAGE
            "semantic_classifications": {
                "intent_classifications": [],
                "journey_progression": [],
                "engagement_style_evolution": [],
                "conversion_opportunities": []
            },
            
            # v4 SOCIAL MEDIA FEATURES
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
                # v4 UNIFIED INTELLIGENCE FEATURES
                "unified_intelligence_enabled": True,
                "primary_channel": "sms",  # sms, twitter, tiktok, etc.
                "channel_preferences": {}
            },
            
            # v3 DEEP ANALYSIS METADATA
            "deep_analysis_completed": None,
            "deep_analysis_confidence": 0.0
        }

    # ==========================================
    # v3 DEEP BACKGROUND ANALYSIS PIPELINE (RESTORED)
    # ==========================================
    
    async def run_background_deep_analysis(
        self, 
        user_id: str, 
        conversation_history: List[Dict],
        max_messages: int = 50
    ) -> Dict[str, Any]:
        """
        Run comprehensive background analysis for deep personality profiling
        Uses batch Gemini analysis for cost efficiency (RESTORED FROM v3)
        """
        try:
            if not self.gemini_service:
                logger.warning("⚠️ Background analysis requires Gemini service")
                return {"error": "Gemini service not available"}
            
            # Prepare messages for batch analysis
            recent_messages = conversation_history[-max_messages:]
            batch_data = []
            
            for msg in recent_messages:
                if msg.get('sender') == 'user':
                    batch_data.append({
                        'message': msg['content'],
                        'context': {
                            'timestamp': msg.get('timestamp'),
                            'conversation_context': True
                        }
                    })
            
            if not batch_data:
                return {"error": "No user messages found for analysis"}
            
            # Run batch analysis
            logger.info(f"🔍 Running background deep analysis for {len(batch_data)} messages")
            analysis_results = await self.gemini_service.batch_analyze(batch_data)
            
            # Aggregate insights across all messages
            deep_insights = self._aggregate_deep_insights(analysis_results)
            
            # Update user profile with deep insights
            await self._apply_deep_insights_to_profile(user_id, deep_insights)
            
            return {
                "analysis_method": "background_deep_analysis",
                "messages_analyzed": len(batch_data),
                "deep_insights": deep_insights,
                "total_cost": self.gemini_service.get_usage_stats()["total_cost"],
                "confidence_score": deep_insights.get("overall_confidence", 0.5)
            }
            
        except Exception as e:
            logger.error(f"❌ Background deep analysis failed: {e}")
            return {"error": str(e)}
    
    def _aggregate_deep_insights(self, results: List[PersonalityAnalysisResult]) -> Dict[str, Any]:
        """Aggregate insights from multiple analysis results (RESTORED FROM v3)"""
        if not results:
            return {}
        
        # Aggregate communication patterns
        formality_scores = [r.communication_insights.get('formality_score', 0.5) for r in results]
        energy_levels = [r.communication_insights.get('energy_level', 'moderate') for r in results]
        tech_depths = [r.communication_insights.get('technical_depth', 'basic') for r in results]
        
        # Aggregate trading patterns
        all_symbols = []
        risk_tolerances = []
        trading_actions = []
        
        for result in results:
            symbols = result.trading_insights.get('symbols_mentioned', [])
            all_symbols.extend(symbols)
            
            risk_tolerance = result.trading_insights.get('risk_tolerance', 'moderate')
            risk_tolerances.append(risk_tolerance)
            
            trading_action = result.trading_insights.get('trading_action', 'unclear')
            if trading_action != 'unclear':
                trading_actions.append(trading_action)
        
        # Aggregate emotional patterns
        primary_emotions = [r.emotional_state.get('primary_emotion', 'neutral') for r in results]
        emotional_intensities = [r.emotional_state.get('emotional_intensity', 0.0) for r in results]
        
        # Calculate confidence based on consistency
        confidence_scores = [r.confidence_score for r in results]
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
        
        return {
            "communication_patterns": {
                "average_formality": np.mean(formality_scores),
                "dominant_energy_level": Counter(energy_levels).most_common(1)[0][0] if energy_levels else "moderate",
                "dominant_tech_depth": Counter(tech_depths).most_common(1)[0][0] if tech_depths else "basic",
                "consistency_score": 1.0 - np.std(formality_scores) if len(formality_scores) > 1 else 1.0
            },
            "trading_patterns": {
                "symbol_frequency": dict(Counter(all_symbols).most_common(10)),
                "total_unique_symbols": len(set(all_symbols)),
                "dominant_risk_tolerance": Counter(risk_tolerances).most_common(1)[0][0] if risk_tolerances else "moderate",
                "dominant_trading_action": Counter(trading_actions).most_common(1)[0][0] if trading_actions else "unclear",
                "trading_focus_score": len(all_symbols) / max(1, len(results))
            },
            "emotional_patterns": {
                "dominant_emotion": Counter(primary_emotions).most_common(1)[0][0] if primary_emotions else "neutral",
                "average_intensity": np.mean(emotional_intensities),
                "emotional_volatility": np.std(emotional_intensities) if len(emotional_intensities) > 1 else 0.0,
                "emotional_diversity": len(set(primary_emotions))
            },
            "overall_confidence": overall_confidence,
            "analysis_depth": "comprehensive",
            "messages_analyzed": len(results)
        }
    
    async def _apply_deep_insights_to_profile(self, user_id: str, deep_insights: Dict) -> None:
        """Apply deep insights to user profile with high confidence (RESTORED FROM v3)"""
        try:
            profile = await self.get_user_profile(user_id)
            
            # Apply communication patterns with high learning rate
            comm_patterns = deep_insights.get("communication_patterns", {})
            if comm_patterns:
                profile["communication_style"]["formality_from_deep_analysis"] = comm_patterns.get("average_formality", 0.5)
                profile["communication_style"]["energy_consistent"] = comm_patterns.get("dominant_energy_level", "moderate")
                profile["communication_style"]["technical_depth_confirmed"] = comm_patterns.get("dominant_tech_depth", "basic")
                profile["communication_style"]["consistency_score"] = comm_patterns.get("consistency_score", 1.0)
            
            # Apply trading patterns
            trading_patterns = deep_insights.get("trading_patterns", {})
            if trading_patterns:
                profile["trading_personality"]["verified_symbols"] = trading_patterns.get("symbol_frequency", {})
                profile["trading_personality"]["risk_tolerance_confirmed"] = trading_patterns.get("dominant_risk_tolerance", "moderate")
                profile["trading_personality"]["preferred_action"] = trading_patterns.get("dominant_trading_action", "unclear")
                profile["trading_personality"]["trading_focus_score"] = trading_patterns.get("trading_focus_score", 0.0)
            
            # Apply emotional patterns
            emotional_patterns = deep_insights.get("emotional_patterns", {})
            if emotional_patterns:
                profile["emotional_intelligence"]["dominant_emotion_confirmed"] = emotional_patterns.get("dominant_emotion", "neutral")
                profile["emotional_intelligence"]["emotional_volatility"] = emotional_patterns.get("emotional_volatility", 0.0)
                profile["emotional_intelligence"]["emotional_consistency"] = 1.0 - emotional_patterns.get("emotional_volatility", 0.0)
            
            # Update profile metadata
            profile["deep_analysis_completed"] = datetime.now(timezone.utc).isoformat()
            profile["deep_analysis_confidence"] = deep_insights.get("overall_confidence", 0.5)
            profile["profile_version"] = self.PROFILE_VERSION
            
            # Save updated profile
            await self.update_user_profile(user_id, profile)
            
            logger.info(f"✅ Applied deep insights to profile for user {user_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply deep insights: {e}")

    # ==========================================
    # v4 FREQUENCY-WEIGHTED LEARNING METHODS
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
                    
                    # LLM-ONLY TICKER VALIDATION: Validate Gemini-detected tickers against universe
                    raw_symbols = gemini_result.trading_insights.get('symbols_mentioned', [])
                    verified_symbols = [s.upper() for s in raw_symbols if s.upper() in self.authoritative_tickers]
                    
                    # Force only verified symbols into trading insights
                    analysis.trading_insights['symbols_mentioned'] = verified_symbols
                    
                    # Additional company name mapping (LLM context enhancement)
                    company_symbols = self._extract_company_name_symbols(user_message)
                    for symbol in company_symbols:
                        if symbol not in verified_symbols and symbol in self.authoritative_tickers:
                            verified_symbols.append(symbol)
                    
                    # Update final verified symbols
                    analysis.trading_insights['symbols_mentioned'] = list(set(verified_symbols))
                    
                    # Boost confidence if verified symbols found
                    if verified_symbols and analysis.confidence_score < 0.8:
                        analysis.confidence_score = min(1.0, analysis.confidence_score + 0.1)
                    
                    # FREQUENCY-WEIGHTED LEARNING INTEGRATION
                    await self._integrate_analysis_into_profile(user_id, analysis, channel)
                    
                    logger.info(f"✅ Gemini semantic analysis completed for {user_id} (confidence: {gemini_result.confidence_score:.2f})")
                    
                except Exception as e:
                    logger.error(f"❌ Gemini analysis failed, using fallback: {e}")
                    analysis = await self._run_fallback_analysis(user_message, enhanced_context)
            else:
                # Use enhanced fallback regex-based analysis
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

    # ==========================================
    # v3 ENHANCED FALLBACK ANALYSIS (RESTORED)
    # ==========================================
    
    async def _run_fallback_analysis(self, message: str, context: Dict = None) -> MessageAnalysis:
        """Run enhanced fallback regex-based analysis when Gemini is unavailable (SAFE REGEX MODE)"""
        start_time = asyncio.get_event_loop().time()
        
        # Use enhanced regex-based preprocessing (but with universe validation)
        preprocessed = self._preprocess_message_safe(message)
        
        # Run enhanced analysis methods (restored from v3)
        communication_analysis = self._analyze_communication_style_regex(message, preprocessed)
        trading_analysis = self._analyze_trading_content_regex(message, preprocessed)
        emotional_analysis = self._analyze_emotional_state_regex(message, preprocessed)
        intent_analysis = self._analyze_user_intent_regex(message, preprocessed)
        service_analysis = self._analyze_service_needs_regex(message, preprocessed)
        sales_analysis = self._analyze_sales_opportunity_regex(message, preprocessed)
        
        processing_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        return MessageAnalysis(
            communication_insights=communication_analysis,
            trading_insights=trading_analysis,
            emotional_state=emotional_analysis,
            service_needs=service_analysis,
            sales_opportunity=sales_analysis,
            intent_analysis=intent_analysis,
            confidence_score=0.6,  # Lower confidence for regex analysis
            analysis_method="safe_regex_fallback",
            processing_time_ms=processing_time_ms,
            gemini_analysis=None
        )
    
    def _analyze_communication_style_regex(self, message: str, preprocessed: Dict) -> Dict[str, Any]:
        """Analyze communication style using enhanced regex patterns (RESTORED FROM v3)"""
        patterns = self.communication_patterns
        
        # Formality analysis
        formal_count = sum(1 for word in patterns['formality_indicators']['formal'] 
                          if word in preprocessed['lower'])
        casual_count = sum(1 for word in patterns['formality_indicators']['casual'] 
                          if word in preprocessed['lower'])
        
        formality_score = 0.5
        if formal_count > casual_count:
            formality_score = min(1.0, 0.5 + (formal_count * 0.1))
        elif casual_count > formal_count:
            formality_score = max(0.0, 0.5 - (casual_count * 0.1))
        
        # Energy analysis
        high_energy = sum(1 for word in patterns['energy_indicators']['high'] 
                         if word in preprocessed['lower'])
        energy_level = "moderate"
        if high_energy > 2:
            energy_level = "high"
        elif preprocessed['char_analysis']['exclamation_marks'] > 2:
            energy_level = "high"
        
        # Technical depth
        basic_count = sum(1 for word in patterns['technical_depth']['basic'] 
                         if word in preprocessed['lower'])
        advanced_count = sum(1 for word in patterns['technical_depth']['advanced'] 
                            if word in preprocessed['lower'])
        
        technical_depth = "basic"
        if advanced_count > 0:
            technical_depth = "advanced"
        elif advanced_count == 0 and basic_count == 0:
            technical_depth = "intermediate"
        
        return {
            'formality_score': formality_score,
            'energy_level': energy_level,
            'technical_depth': technical_depth,
            'emoji_usage': preprocessed['char_analysis']['emoji_count'],
            'message_length': len(preprocessed['words'])
        }

    def _analyze_trading_content_regex(self, message: str, preprocessed: Dict) -> Dict[str, Any]:
        """Analyze trading content using enhanced regex patterns (RESTORED FROM v3)"""
        patterns = self.trading_patterns
        
        # Risk tolerance analysis
        conservative_count = sum(1 for word in patterns['risk_indicators']['conservative'] 
                               if word in preprocessed['lower'])
        aggressive_count = sum(1 for word in patterns['risk_indicators']['aggressive'] 
                              if word in preprocessed['lower'])
        
        risk_tolerance = "moderate"
        if aggressive_count > conservative_count and aggressive_count > 0:
            risk_tolerance = "aggressive"
        elif conservative_count > aggressive_count and conservative_count > 0:
            risk_tolerance = "conservative"
        
        # Trading action analysis
        trading_action = "unclear"
        for action, keywords in patterns['trading_actions'].items():
            if any(keyword in preprocessed['lower'] for keyword in keywords):
                trading_action = action
                break
        
        return {
            'symbols_mentioned': preprocessed['symbols'],
            'trading_action': trading_action,
            'risk_tolerance': risk_tolerance,
            'money_amounts': preprocessed['patterns'].get('money_amounts', []),
            'percentages': preprocessed['patterns'].get('percentages', [])
        }

    def _analyze_emotional_state_regex(self, message: str, preprocessed: Dict) -> Dict[str, Any]:
        """Analyze emotional state using enhanced regex patterns (RESTORED FROM v3)"""
        emotional_words = {
            'excited': ['excited', 'pumped', 'thrilled', 'amazing', 'awesome'],
            'worried': ['worried', 'concerned', 'scared', 'nervous', 'anxious'],
            'frustrated': ['frustrated', 'annoyed', 'upset', 'angry', 'mad'],
            'confident': ['confident', 'sure', 'certain', 'bullish', 'optimistic'],
            'uncertain': ['uncertain', 'confused', 'unsure', 'maybe', 'not sure']
        }
        
        primary_emotion = "neutral"
        emotional_intensity = 0.0
        
        for emotion, keywords in emotional_words.items():
            count = sum(1 for keyword in keywords if keyword in preprocessed['lower'])
            if count > 0:
                primary_emotion = emotion
                emotional_intensity = min(1.0, count * 0.3)
                break
        
        # Check for intensity indicators
        if preprocessed['char_analysis']['exclamation_marks'] > 2:
            emotional_intensity = min(1.0, emotional_intensity + 0.2)
        
        return {
            'primary_emotion': primary_emotion,
            'emotional_intensity': emotional_intensity,
            'support_needed': 'high_support' if emotional_intensity > 0.7 else 'standard_guidance'
        }

    def _analyze_user_intent_regex(self, message: str, preprocessed: Dict) -> Dict[str, Any]:
        """Analyze user intent using enhanced regex patterns (RESTORED FROM v3)"""
        intent_patterns = {
            'question': message.count('?') > 0 or any(q in preprocessed['lower'] for q in ['what', 'how', 'when', 'where', 'why']),
            'request_analysis': any(word in preprocessed['lower'] for word in ['analyze', 'analysis', 'chart', 'technical']),
            'general_chat': len(preprocessed['symbols']) == 0 and not any(t in preprocessed['lower'] for t in ['buy', 'sell', 'trade'])
        }
        
        primary_intent = "general_chat"
        if intent_patterns['request_analysis']:
            primary_intent = "request_analysis"
        elif intent_patterns['question']:
            primary_intent = "question"
        
        return {
            'primary_intent': primary_intent,
            'requires_tools': ['technical_analysis'] if primary_intent == 'request_analysis' else [],
            'follow_up_likelihood': 0.7 if primary_intent == 'question' else 0.3
        }

    def _analyze_service_needs_regex(self, message: str, preprocessed: Dict) -> Dict[str, Any]:
        """Analyze service needs using enhanced regex patterns (RESTORED FROM v3)"""
        patterns = self.service_patterns
        
        service_type = "none"
        urgency_level = 0.0
        
        for stype, keywords in patterns['service_types'].items():
            if any(keyword in preprocessed['lower'] for keyword in keywords):
                service_type = stype
                break
        
        urgency_patterns = {
            'immediate': ['now', 'urgent', 'asap', 'quick', 'emergency'],
            'today': ['today', 'this morning', 'this afternoon', 'tonight'],
            'this_week': ['this week', 'soon', 'in a few days'],
            'general': ['when', 'sometime', 'eventually', 'later']
        }
        
        for urgency, keywords in urgency_patterns.items():
            if any(keyword in preprocessed['lower'] for keyword in keywords):
                urgency_level = {'immediate': 1.0, 'today': 0.8, 'this_week': 0.5, 'general': 0.2}[urgency]
                break
        
        return {
            'service_type': service_type,
            'urgency_level': urgency_level
        }

    def _analyze_sales_opportunity_regex(self, message: str, preprocessed: Dict) -> Dict[str, Any]:
        """Analyze sales opportunity using enhanced regex patterns (RESTORED FROM v3)"""
        patterns = self.sales_indicators
        
        buying_signal_strength = 0.0
        for strength, keywords in patterns['buying_signals'].items():
            if any(keyword in preprocessed['lower'] for keyword in keywords):
                buying_signal_strength = {'strong': 0.9, 'moderate': 0.6, 'weak': 0.3}[strength]
                break
        
        return {
            'sales_readiness_score': buying_signal_strength,
            'opportunity_type': 'premium_upgrade' if buying_signal_strength > 0.6 else 'none'
        }

    # ==========================================
    # v3 GLOBAL INSIGHTS & RESPONSE STRATEGY (RESTORED)
    # ==========================================
    
    def _update_global_patterns(self, user_id: str, analysis: MessageAnalysis) -> None:
        """Update global patterns for intelligence aggregation (RESTORED FROM v3)"""
        # Update global symbol patterns
        symbols = analysis.trading_insights.get('symbols_mentioned', [])
        for symbol in symbols:
            self._global_patterns['symbols'][symbol] += 1
        
        # Update global communication patterns
        energy = analysis.communication_insights.get('energy_level', 'moderate')
        self._global_patterns['energy'][energy] += 1
        
        # Update global trading actions
        trading_action = analysis.trading_insights.get('trading_action', 'unclear')
        if trading_action != 'unclear':
            self._global_patterns['trading_actions'][trading_action] += 1

    def _get_global_insights(self, symbols: List[str], profile: Dict) -> Dict[str, Any]:
        """Get global insights for the user (RESTORED FROM v3)"""
        return {
            'popular_symbols': dict(self._global_patterns['symbols'].most_common(5)),
            'trending_actions': dict(self._global_patterns['trading_actions'].most_common(3)),
            'energy_distribution': dict(self._global_patterns['energy'].most_common(3)),
            'user_uniqueness': len(set(symbols)) / max(1, len(symbols)) if symbols else 0.0,
            'symbol_overlap_with_trends': len([s for s in symbols if s in self._global_patterns['symbols']]) / max(1, len(symbols)) if symbols else 0.0
        }

    def _generate_response_strategy_enhanced(self, profile: Dict, analysis: MessageAnalysis, global_insights: Dict) -> Dict[str, Any]:
        """Generate enhanced response strategy with global context (RESTORED FROM v3)"""
        return {
            'communication_style': analysis.communication_insights.get('energy_level', 'moderate'),
            'technical_level': analysis.communication_insights.get('technical_depth', 'basic'),
            'personalization_strength': profile.get('confidence_score', 0.5),
            'global_context': global_insights,
            'recommended_symbols': global_insights.get('popular_symbols', {}),
            'trending_focus': global_insights.get('trending_actions', {}),
            'adaptation_needed': global_insights.get('user_uniqueness', 0.0) < 0.3  # User follows trends
        }

    # ==========================================
    # v3 COST OPTIMIZATION & CACHING (RESTORED)
    # ==========================================
    
    async def optimize_analysis_costs(self) -> Dict[str, Any]:
        """Optimize analysis costs by clearing caches and adjusting settings (RESTORED FROM v3)"""
        optimization_results = {}
        
        # Clear analysis cache
        cache_cleared = self.clear_analysis_cache()
        optimization_results["analysis_cache_cleared"] = cache_cleared
        
        # Clear Gemini cache
        if self.gemini_service:
            gemini_cache_cleared = self.gemini_service.clear_cache()
            optimization_results["gemini_cache_cleared"] = gemini_cache_cleared
        
        # Optimize global patterns
        pattern_optimization = self.optimize_global_patterns()
        optimization_results["global_patterns_optimized"] = pattern_optimization
        
        logger.info(f"🔧 Cost optimization completed: {optimization_results}")
        return optimization_results
    
    def enable_aggressive_caching(self, cache_ttl: int = 7200) -> None:
        """Enable aggressive caching for cost optimization (RESTORED FROM v3)"""
        if self.gemini_service:
            self.gemini_service.cache_ttl = cache_ttl
            logger.info(f"💰 Enabled aggressive caching (TTL: {cache_ttl}s)")

    def clear_analysis_cache(self) -> int:
        """Clear analysis cache and return number of entries cleared (RESTORED FROM v3)"""
        cache_size = len(self._analysis_cache)
        self._analysis_cache.clear()
        return cache_size

    def optimize_global_patterns(self) -> Dict[str, int]:
        """Optimize global patterns storage (RESTORED FROM v3)"""
        # Keep only top N patterns to save memory
        for pattern_type in self._global_patterns:
            if len(self._global_patterns[pattern_type]) > 1000:
                # Keep only top 500 most common
                top_patterns = dict(self._global_patterns[pattern_type].most_common(500))
                self._global_patterns[pattern_type] = Counter(top_patterns)
        
        return {k: len(v) for k, v in self._global_patterns.items()}

    def get_gemini_usage_stats(self) -> Dict[str, Any]:
        """Get Gemini usage statistics for cost monitoring (RESTORED FROM v3)"""
        if self.gemini_service:
            return self.gemini_service.get_usage_stats()
        return {"error": "Gemini service not available"}

    # ==========================================
    # ENHANCED PATTERN LOADING (v3 + v4)
    # ==========================================
    
    def _load_authoritative_tickers(self) -> set:
        """Load authoritative ticker list from centralized JSON file"""
        ticker_file = os.path.join(os.path.dirname(__file__), "data", "ticker_universe.json")
        try:
            with open(ticker_file, "r") as f:
                data = json.load(f)
            all_tickers = set(data.get("stocks", []) + data.get("etfs", []) + data.get("crypto", []))
            logger.info(f"✅ Loaded {len(all_tickers)} tickers from universe file")
            return all_tickers
        except Exception as e:
            logger.error(f"❌ Failed to load ticker universe: {e}")
            # Fallback to basic set if file not found
            return {
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
                'SPY', 'QQQ', 'BTC', 'ETH', 'DOGE'
            }

    def _compile_regex_patterns(self) -> Dict[str, re.Pattern]:
        """Enhanced pre-compiled regex patterns (v3 + improvements)"""
        return {
            'potential_symbols': re.compile(r'\b[A-Z]{2,5}\b'),
            'money_amounts': re.compile(r'\$([0-9,]+(?:\.[0-9]{2})?)'),
            'percentages': re.compile(r'([0-9]+(?:\.[0-9]+)?)%'),
            'emojis': re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U0001f900-\U0001f9ff\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f1e0-\U0001f1ff]'),
            'caps_words': re.compile(r'\b[A-Z]{2,}\b'),
            'excessive_punct': re.compile(r'[!?]{3,}'),
            'repeated_chars': re.compile(r'(.)\1{2,}'),
            'time_expressions': re.compile(r'\b(today|tomorrow|next week|this week|asap|now|urgent)\b', re.I),
            'price_targets': re.compile(r'\$([0-9]+(?:\.[0-9]{2})?)\s*(?:target|price|level)', re.I)
        }
    
    def _load_communication_patterns(self) -> Dict[str, Any]:
        """Enhanced communication analysis patterns (v3 expanded)"""
        return {
            'formality_indicators': {
                'formal': ['please', 'thank you', 'kindly', 'respectfully', 'sincerely', 'appreciate', 'gratitude'],
                'casual': ['yo', 'hey', 'sup', 'dude', 'bro', 'lol', 'omg', 'wtf', 'thx', 'ur'],
                'professional': ['analyze', 'assessment', 'evaluation', 'consideration', 'recommendation', 'strategic', 'metrics']
            },
            'energy_indicators': {
                'high': ['!', '!!', '!!!', 'excited', 'pumped', 'amazing', 'awesome', 'love it', 'fantastic', 'incredible'],
                'low': ['tired', 'meh', 'okay', 'fine', 'whatever', 'sure', 'bored', 'disappointed'],
                'moderate': ['good', 'nice', 'cool', 'interesting', 'thanks', 'appreciate', 'helpful']
            },
            'technical_depth': {
                'basic': ['buy', 'sell', 'up', 'down', 'good', 'bad', 'price', 'stock'],
                'intermediate': ['rsi', 'macd', 'support', 'resistance', 'volume', 'trend', 'breakout', 'pattern'],
                'advanced': ['fibonacci', 'bollinger', 'stochastic', 'divergence', 'consolidation', 'volatility', 'correlation', 'beta']
            }
        }

    def _load_trading_patterns(self) -> Dict[str, Any]:
        """Enhanced trading behavior analysis patterns (v3 expanded)"""
        return {
            'risk_indicators': {
                'conservative': ['safe', 'secure', 'stable', 'dividend', 'blue chip', 'worried', 'scared', 'careful', 'steady'],
                'moderate': ['growth', 'balanced', 'reasonable', 'consider', 'think about', 'moderate', 'cautious'],
                'aggressive': ['yolo', 'moon', 'rocket', 'all in', 'bet', 'gamble', 'risky', 'swing big', 'high risk']
            },
            'trading_actions': {
                'buying': ['buy', 'purchase', 'get', 'acquire', 'long', 'calls', 'bullish', 'accumulate'],
                'selling': ['sell', 'dump', 'exit', 'short', 'puts', 'bearish', 'close', 'liquidate'],
                'holding': ['hold', 'keep', 'hodl', 'diamond hands', 'stay', 'maintain', 'patient'],
                'researching': ['analyze', 'research', 'study', 'look into', 'investigate', 'examine', 'evaluate']
            },
            'experience_indicators': {
                'novice': ['new', 'beginner', 'start', 'learn', 'help', 'confused', 'what is', 'first time'],
                'intermediate': ['understand', 'know', 'familiar', 'experience', 'usually', 'sometimes', 'decent'],
                'advanced': ['strategy', 'algorithm', 'model', 'backtest', 'optimize', 'correlate', 'systematic', 'quantitative']
            }
        }

    def _load_sales_indicators(self) -> Dict[str, Any]:
        """Enhanced sales opportunity detection patterns (v3 expanded)"""
        return {
            'buying_signals': {
                'strong': ['need help', 'premium', 'upgrade', 'better service', 'more features', 'professional', 'advanced'],
                'moderate': ['interested', 'tell me more', 'pricing', 'cost', 'worth it', 'benefits', 'value'],
                'weak': ['maybe', 'someday', 'later', 'thinking about', 'not sure', 'hesitant']
            },
            'pain_points': {
                'performance': ['slow', 'delayed', 'late', 'timing', 'missing opportunities', 'lagging', 'behind'],
                'accuracy': ['wrong', 'incorrect', 'bad advice', 'lost money', 'mistake', 'error', 'unreliable'],
                'features': ['limited', 'basic', 'need more', 'lacking', 'insufficient', 'missing', 'incomplete']
            },
            'urgency_indicators': {
                'high': ['urgent', 'asap', 'now', 'immediately', 'quick', 'fast', 'emergency'],
                'medium': ['soon', 'today', 'this week', 'need', 'want', 'prefer'],
                'low': ['eventually', 'someday', 'when', 'if', 'maybe', 'future']
            }
        }

    def _load_service_patterns(self) -> Dict[str, Any]:
        """Enhanced service need detection patterns (v3 expanded)"""
        return {
            'service_types': {
                'technical_analysis': ['chart', 'pattern', 'indicator', 'signal', 'trend', 'analysis', 'ta'],
                'fundamental_analysis': ['earnings', 'revenue', 'pe ratio', 'financials', 'valuation', 'fa', 'balance sheet'],
                'news_analysis': ['news', 'announcement', 'earnings call', 'merger', 'acquisition', 'catalyst', 'events'],
                'portfolio_management': ['portfolio', 'diversify', 'allocation', 'balance', 'risk', 'holdings', 'positions'],
                'education': ['learn', 'explain', 'teach', 'understand', 'how to', 'what is', 'tutorial', 'guide'],
                'screening': ['screener', 'filter', 'find stocks', 'search', 'criteria', 'scan', 'discover'],
                'alerts': ['alert', 'notify', 'notification', 'reminder', 'watch', 'monitor', 'track']
            },
            'urgency_patterns': {
                'immediate': ['now', 'urgent', 'asap', 'quick', 'emergency', 'immediately'],
                'today': ['today', 'this morning', 'this afternoon', 'tonight', 'before close'],
                'this_week': ['this week', 'soon', 'in a few days', 'by friday'],
                'general': ['when', 'sometime', 'eventually', 'later', 'no rush']
            }
        }

    def _preprocess_message_safe(self, message: str) -> Dict[str, Any]:
        """
        SAFE preprocessing for fallback analysis - LLM-FIRST approach
        Only uses regex for basic analysis, NOT primary ticker detection
        """
        message_lower = message.lower()
        words = message.split()
        words_lower = message_lower.split()
        
        patterns = {}
        for pattern_name, compiled_regex in self._compiled_patterns.items():
            patterns[pattern_name] = compiled_regex.findall(message)
        
        char_analysis = {
            'total_length': len(message),
            'emoji_count': len(patterns.get('emojis', [])),
            'caps_words': patterns.get('caps_words', []),
            'question_marks': message.count('?'),
            'exclamation_marks': message.count('!'),
            'excessive_punctuation': patterns.get('excessive_punct', []),
            'repeated_chars': patterns.get('repeated_chars', []),
            'time_expressions': patterns.get('time_expressions', []),
            'price_targets': patterns.get('price_targets', [])
        }
        
        # SAFE SYMBOL EXTRACTION: Only company name mapping + universe validation
        symbols = []
        
        # Primary: Company name to symbol mapping (contextual)
        company_symbols = self._extract_company_name_symbols(message)
        symbols.extend(company_symbols)
        
        # Secondary: Regex potential symbols BUT validate against universe before adding
        potential_symbols = patterns.get('potential_symbols', [])
        for symbol in potential_symbols:
            if symbol.upper() in self.authoritative_tickers and symbol.upper() not in symbols:
                symbols.append(symbol.upper())
        
        # Final deduplication
        symbols = list(set(symbols))
        
        return {
            'original': message,
            'lower': message_lower,
            'words': words,
            'words_lower': words_lower,
            'patterns': patterns,
            'char_analysis': char_analysis,
            'symbols': symbols,  # These are now universe-validated
            'preprocessing_timestamp': datetime.now(timezone.utc).isoformat()
        }

    def _extract_company_name_symbols(self, message: str) -> List[str]:
        """Enhanced company name to symbol mapping with universe validation"""
        company_mappings = {
            # Core tech giants
            'apple': 'AAPL', 'tesla': 'TSLA', 'microsoft': 'MSFT', 'amazon': 'AMZN',
            'google': 'GOOGL', 'alphabet': 'GOOGL', 'meta': 'META', 'facebook': 'META', 
            'netflix': 'NFLX', 'nvidia': 'NVDA', 'amd': 'AMD', 'intel': 'INTC', 
            
            # Financial & payments
            'ibm': 'IBM', 'oracle': 'ORCL', 'salesforce': 'CRM', 'adobe': 'ADBE', 
            'paypal': 'PYPL', 'visa': 'V', 'mastercard': 'MA', 'jpmorgan': 'JPM',
            'goldman': 'GS', 'goldman sachs': 'GS', 'bank of america': 'BAC',
            
            # Transportation & logistics
            'uber': 'UBER', 'lyft': 'LYFT', 'fedex': 'FDX', 'ups': 'UPS',
            
            # Retail & consumer
            'walmart': 'WMT', 'target': 'TGT', 'costco': 'COST', 'home depot': 'HD',
            
            # Industrial & aerospace
            'boeing': 'BA', 'caterpillar': 'CAT', 'deere': 'DE', 'john deere': 'DE',
            
            # Crypto companies & assets
            'bitcoin': 'BTC', 'ethereum': 'ETH', 'dogecoin': 'DOGE', 'solana': 'SOL',
            'coinbase': 'COIN', 'microstrategy': 'MSTR', 'riot': 'RIOT',
            
            # Popular growth stocks
            'palantir': 'PLTR', 'snowflake': 'SNOW', 'crowdstrike': 'CRWD', 
            'zoom': 'ZM', 'slack': 'WORK', 'datadog': 'DDOG',
            
            # ETFs
            'spy': 'SPY', 'qqq': 'QQQ', 'arkk': 'ARKK', 'vti': 'VTI'
        }
        
        message_lower = message.lower()
        found_symbols = []
        
        for company, symbol in company_mappings.items():
            if company in message_lower and symbol in self.authoritative_tickers:
                found_symbols.append(symbol)
        
        return found_symbols

    def _validate_symbol_with_authority(self, symbol: str) -> bool:
        """Enhanced symbol validation against authoritative ticker list"""
        return symbol.upper() in self.authoritative_tickers

    def _enhance_symbol_validation(self, analysis: MessageAnalysis, message: str) -> MessageAnalysis:
        """
        DEPRECATED: Enhanced symbol detection - replaced by LLM-first approach
        Kept for legacy compatibility but not used in main flow
        """
        logger.warning("⚠️ _enhance_symbol_validation is deprecated - using LLM-first ticker detection")
        return analysis

    # ==========================================
    # INTEGRATION AND PROFILE MANAGEMENT
    # ==========================================
    
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
            
            # Update global patterns
            self._update_global_patterns(user_id, analysis)
            
            # Update profile metadata
            profile = self.user_profiles[user_id]
            profile["updated_at"] = datetime.utcnow().isoformat()
            profile["learning_data"]["total_messages"] += 1
            
            # Trigger analysis hooks
            await self._trigger_analysis_hooks(user_id, analysis)
            
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
        
        primary_intent = intent_analysis.get('primary_intent', 'general')
        if primary_intent != 'general_chat':
            topics.append(primary_intent)
        
        # From service needs
        service_type = analysis.service_needs.get('service_type', 'none')
        if service_type != 'none':
            topics.append(service_type)
        
        return list(set(topics))  # Remove duplicates
    
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
            
            # Update energy level
            if 'energy_level' in comm_insights:
                comm_style["energy"] = comm_insights['energy_level']
            
            # Update technical depth
            if 'technical_depth' in comm_insights:
                comm_style["technical_depth"] = comm_insights['technical_depth']
            
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

    async def _trigger_analysis_hooks(self, user_id: str, analysis: MessageAnalysis) -> None:
        """Trigger registered analysis hooks"""
        for hook in self._analysis_hooks:
            try:
                await hook(user_id, analysis)
            except Exception as e:
                logger.error(f"Analysis hook failed: {e}")

    async def _create_minimal_analysis(self, message: str) -> MessageAnalysis:
        """Create minimal analysis when everything fails"""
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
            
            # Get global insights
            global_insights = self._get_global_insights(
                analysis.trading_insights.get('symbols_mentioned', []), 
                profile
            )
            
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
                },
                "global_insights": global_insights,
                "response_strategy": self._generate_response_strategy_enhanced(profile, analysis, global_insights)
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
    # PROFILE MANAGEMENT METHODS
    # ==========================================

    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile with KeyBuilder integration"""
        if self.key_builder:
            try:
                profile_data = await self.key_builder.get_user_personality(user_id)
                if profile_data:
                    return profile_data
            except Exception as e:
                logger.warning(f"KeyBuilder profile retrieval failed: {e}")
        
        # Fallback to in-memory storage
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = self._create_default_profile()
        
        return self.user_profiles[user_id]

    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user profile with KeyBuilder integration"""
        try:
            # Update timestamp
            updates['updated_at'] = datetime.now(timezone.utc).isoformat()
            
            if self.key_builder:
                try:
                    success = await self.key_builder.update_user_personality(user_id, updates)
                    if success:
                        return True
                except Exception as e:
                    logger.warning(f"KeyBuilder profile update failed: {e}")
            
            # Fallback to in-memory storage
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = self._create_default_profile()
            
            # Deep merge updates
            self._deep_merge_dict(self.user_profiles[user_id], updates)
            
            # Trigger profile update hooks
            await self._trigger_profile_update_hooks(user_id, updates)
            
            return True
            
        except Exception as e:
            logger.error(f"Profile update failed for {user_id}: {e}")
            return False

    def _deep_merge_dict(self, target: Dict, source: Dict) -> None:
        """Deep merge source dict into target dict"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge_dict(target[key], value)
            else:
                target[key] = value

    async def learn_from_analysis(self, user_id: str, profile: Dict, analysis: MessageAnalysis) -> Dict[str, Any]:
        """Generate learning updates from analysis"""
        updates = {}
        
        # Update communication style
        comm_insights = analysis.communication_insights
        if 'formality_score' in comm_insights:
            current_formality = profile.get('communication_style', {}).get('formality', 0.5)
            new_formality = (current_formality * 0.8) + (comm_insights['formality_score'] * 0.2)
            updates['communication_style'] = {'formality': new_formality}
        
        # Update trading personality
        trading_insights = analysis.trading_insights
        if 'symbols_mentioned' in trading_insights and trading_insights['symbols_mentioned']:
            current_symbols = profile.get('trading_personality', {}).get('common_symbols', [])
            new_symbols = list(set(current_symbols + trading_insights['symbols_mentioned']))
            if 'trading_personality' not in updates:
                updates['trading_personality'] = {}
            updates['trading_personality']['common_symbols'] = new_symbols[-20:]  # Keep last 20
        
        # Update confidence score
        current_confidence = profile.get('confidence_score', 0.1)
        new_confidence = min(1.0, current_confidence + 0.05)
        updates['confidence_score'] = new_confidence
        
        # Update message count
        current_count = profile.get('learning_data', {}).get('total_messages', 0)
        if 'learning_data' not in updates:
            updates['learning_data'] = {}
        updates['learning_data']['total_messages'] = current_count + 1
        
        return updates

    async def _trigger_profile_update_hooks(self, user_id: str, updates: Dict[str, Any]) -> None:
        """Trigger registered profile update hooks"""
        for hook in self._profile_update_hooks:
            try:
                await hook(user_id, updates)
            except Exception as e:
                logger.error(f"Profile update hook failed: {e}")

    # ==========================================
    # LEGACY COMPATIBILITY METHODS
    # ==========================================
    
    async def analyze_and_learn(self, user_id: str, user_message: str, bot_response: str = None, 
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Combined analysis and learning workflow - now powered by Gemini
        Maintains API compatibility with existing code
        """
        try:
            # Step 1: Run enhanced analysis (Gemini or fallback)
            channel = context.get("channel", "sms") if context else "sms"
            analysis = await self.analyze_message_unified(user_id, user_message, channel, context)
            
            # Step 2: Get current profile
            profile = await self.get_user_profile(user_id)
            
            # Step 3: Update global intelligence patterns
            self._update_global_patterns(user_id, analysis)
            
            # Step 4: Generate learning updates (enhanced with Gemini insights)
            updates = await self.learn_from_analysis(user_id, profile, analysis)
            
            # Step 5: Apply updates
            if updates:
                await self.update_user_profile(user_id, updates)
            
            # Step 6: Trigger analysis hooks
            await self._trigger_analysis_hooks(user_id, analysis)
            
            # Step 7: Generate enhanced response strategy with global insights
            global_insights = self._get_global_insights(
                analysis.trading_insights.get('symbols_mentioned', []), 
                profile
            )
            response_strategy = self._generate_response_strategy_enhanced(profile, analysis, global_insights)
            
            return {
                "communication_insights": analysis.communication_insights,
                "trading_insights": analysis.trading_insights,
                "emotional_state": analysis.emotional_state,
                "service_needs": analysis.service_needs,
                "sales_opportunity": analysis.sales_opportunity,
                "recommended_approach": response_strategy,
                "global_insights": global_insights,
                "profile_confidence": profile.get("confidence_score", 0.5),
                "learning_applied": bool(updates),
                "analysis_method": analysis.analysis_method,
                "gemini_confidence": analysis.confidence_score,
                "processing_time_ms": analysis.processing_time_ms,
                "unified_intelligence": True,
                "frequency_data": self.get_personalization_context(user_id)
            }
            
        except Exception as e:
            logger.error(f"❌ Error in analyze_and_learn for {user_id}: {e}")
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
    
    def get_personality_summary(self, user_id: str) -> Dict[str, Any]:
        """Get personality summary in enhanced format"""
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
                "profile_version": self.PROFILE_VERSION,
                "deep_analysis_available": profile.get("deep_analysis_completed") is not None
            }
            
        except Exception as e:
            logger.error(f"Failed to get personality summary: {e}")
            return {"error": str(e)}

    # ==========================================
    # REAL-TIME CONTEXT FOR SMS/SOCIAL RESPONSES
    # ==========================================
    
    async def get_real_time_personality_context(self, user_id: str, message: str, channel: str = "sms") -> Dict[str, Any]:
        """
        Get real-time personality context for SMS/social response generation
        Optimized for sub-1-second response time
        """
        try:
            # Get current user profile
            profile = await self.get_user_profile(user_id)
            
            # Run fast analysis (use cache aggressively)
            analysis = await self.analyze_message_unified(user_id, message, channel, {"real_time": True})
            
            # Extract key context for response generation
            context = {
                "communication_style": {
                    "formality": analysis.communication_insights.get('formality_score', 0.5),
                    "energy": analysis.communication_insights.get('energy_level', 'moderate'),
                    "technical_preference": analysis.communication_insights.get('technical_depth', 'basic'),
                    "emoji_appropriate": analysis.communication_insights.get('emoji_usage', 0) > 0
                },
                "trading_focus": {
                    "symbols": analysis.trading_insights.get('symbols_mentioned', []),
                    "action_intent": analysis.trading_insights.get('trading_action', 'unclear'),
                    "risk_appetite": analysis.trading_insights.get('risk_tolerance', 'moderate')
                },
                "emotional_state": {
                    "primary_emotion": analysis.emotional_state.get('primary_emotion', 'neutral'),
                    "intensity": analysis.emotional_state.get('emotional_intensity', 0.0),
                    "support_needed": analysis.emotional_state.get('support_needed', 'standard_guidance')
                },
                "response_guidance": {
                    "urgency_level": analysis.service_needs.get('urgency_level', 0.0),
                    "personalization_strength": profile.get("confidence_score", 0.5),
                    "requires_tools": analysis.intent_analysis.get('requires_tools', []),
                    "follow_up_likely": analysis.intent_analysis.get('follow_up_likelihood', 0.5)
                },
                "frequency_context": self.get_personalization_context(user_id),
                "meta": {
                    "analysis_method": analysis.analysis_method,
                    "confidence": analysis.confidence_score,
                    "processing_time_ms": analysis.processing_time_ms,
                    "channel": channel
                }
            }
            
            return context
            
        except Exception as e:
            logger.error(f"❌ Real-time personality context failed: {e}")
            return self._create_minimal_context(message, channel)
    
    def _create_minimal_context(self, message: str, channel: str) -> Dict[str, Any]:
        """Create minimal context when analysis fails"""
        return {
            "communication_style": {"formality": 0.5, "energy": "moderate", "technical_preference": "basic"},
            "trading_focus": {"symbols": [], "action_intent": "unclear", "risk_appetite": "moderate"},
            "emotional_state": {"primary_emotion": "neutral", "intensity": 0.0, "support_needed": "standard_guidance"},
            "response_guidance": {"urgency_level": 0.0, "personalization_strength": 0.2, "requires_tools": []},
            "frequency_context": {"top_topics": [], "top_symbols": [], "learning_velocity": {"rate": "unknown"}},
            "meta": {"analysis_method": "minimal_fallback", "confidence": 0.1, "processing_time_ms": 1, "channel": channel}
        }

    # ==========================================
    # HOOK MANAGEMENT
    # ==========================================

    def add_profile_update_hook(self, hook_func: Callable) -> None:
        """Add a hook function to be called when profiles are updated"""
        self._profile_update_hooks.append(hook_func)

    def add_analysis_hook(self, hook_func: Callable) -> None:
        """Add a hook function to be called after analysis"""
        self._analysis_hooks.append(hook_func)

    def remove_profile_update_hook(self, hook_func: Callable) -> bool:
        """Remove a profile update hook"""
        try:
            self._profile_update_hooks.remove(hook_func)
            return True
        except ValueError:
            return False

    def remove_analysis_hook(self, hook_func: Callable) -> bool:
        """Remove an analysis hook"""
        try:
            self._analysis_hooks.remove(hook_func)
            return True
        except ValueError:
            return False


# ==========================================
# v3 ADVANCED FEATURES AND EXTENSIONS (RESTORED)
# ==========================================

class PersonalityInsightsGenerator:
    """Advanced personality insights and reporting (RESTORED FROM v3)"""
    
    def __init__(self, personality_engine: UnifiedPersonalityEngine):
        self.engine = personality_engine
    
    async def generate_personality_report(self, user_id: str) -> Dict[str, Any]:
        """Generate comprehensive personality report"""
        try:
            profile = await self.engine.get_user_profile(user_id)
            
            # Calculate personality metrics
            communication_score = self._calculate_communication_score(profile)
            trading_sophistication = self._calculate_trading_sophistication(profile)
            engagement_level = self._calculate_engagement_level(profile)
            
            return {
                "user_id": user_id,
                "report_timestamp": datetime.now(timezone.utc).isoformat(),
                "personality_scores": {
                    "communication_effectiveness": communication_score,
                    "trading_sophistication": trading_sophistication,
                    "engagement_level": engagement_level
                },
                "communication_profile": {
                    "style": profile["communication_style"]["energy"],
                    "formality": profile["communication_style"]["formality"],
                    "technical_depth": profile["communication_style"]["technical_depth"],
                    "consistency": profile["communication_style"]["consistency_score"]
                },
                "trading_profile": {
                    "risk_tolerance": profile["trading_personality"]["risk_tolerance"],
                    "experience_level": profile["trading_personality"]["experience_level"],
                    "preferred_symbols": profile["trading_personality"]["common_symbols"][:10],
                    "trading_focus": len(profile["trading_personality"]["common_symbols"])
                },
                "learning_metrics": {
                    "total_interactions": profile["learning_data"]["total_messages"],
                    "profile_confidence": profile["confidence_score"],
                    "learning_progression": self._calculate_learning_progression(profile)
                },
                "frequency_insights": {
                    "top_topics": self.engine.get_top_topics_by_frequency(user_id, 5),
                    "top_symbols": self.engine.get_top_symbols_by_frequency(user_id, 5),
                    "learning_velocity": profile["learning_data"]["learning_velocity"]
                },
                "recommendations": self._generate_personalization_recommendations(profile)
            }
            
        except Exception as e:
            logger.error(f"Personality report generation failed: {e}")
            return {"error": str(e)}
    
    def _calculate_communication_score(self, profile: Dict) -> float:
        """Calculate communication effectiveness score"""
        style = profile["communication_style"]
        consistency = style.get("consistency_score", 1.0)
        message_count = profile["learning_data"]["total_messages"]
        
        # Score based on consistency and interaction frequency
        base_score = consistency * 0.6
        activity_bonus = min(0.4, message_count / 50)  # Up to 0.4 bonus for activity
        
        return round(base_score + activity_bonus, 2)
    
    def _calculate_trading_sophistication(self, profile: Dict) -> float:
        """Calculate trading sophistication score"""
        trading = profile["trading_personality"]
        
        # Experience level mapping
        exp_scores = {"novice": 0.2, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}
        base_score = exp_scores.get(trading["experience_level"], 0.5)
        
        # Symbol diversity bonus
        symbol_count = len(trading["common_symbols"])
        diversity_bonus = min(0.3, symbol_count / 20)  # Up to 0.3 bonus
        
        return round(base_score + diversity_bonus, 2)
    
    def _calculate_engagement_level(self, profile: Dict) -> float:
        """Calculate user engagement level"""
        total_messages = profile["learning_data"]["total_messages"]
        confidence = profile["confidence_score"]
        
        # Engagement based on activity and profile development
        activity_score = min(0.7, total_messages / 30)  # Up to 0.7 for activity
        development_score = confidence * 0.3  # Up to 0.3 for profile development
        
        return round(activity_score + development_score, 2)
    
    def _calculate_learning_progression(self, profile: Dict) -> str:
        """Calculate learning progression stage"""
        total_messages = profile["learning_data"]["total_messages"]
        confidence = profile["confidence_score"]
        
        if total_messages < 5:
            return "getting_started"
        elif total_messages < 15:
            return "building_profile"
        elif confidence < 0.7:
            return "developing_preferences"
        else:
            return "personalized_experience"
    
    def _generate_personalization_recommendations(self, profile: Dict) -> List[str]:
        """Generate personalization recommendations"""
        recommendations = []
        
        # Communication recommendations
        if profile["communication_style"]["formality"] < 0.3:
            recommendations.append("Consider more casual, friendly communication style")
        elif profile["communication_style"]["formality"] > 0.7:
            recommendations.append("Maintain professional, detailed responses")
        
        # Trading recommendations
        risk_tolerance = profile["trading_personality"]["risk_tolerance"]
        if risk_tolerance == "conservative":
            recommendations.append("Focus on stable, blue-chip stock recommendations")
        elif risk_tolerance == "aggressive":
            recommendations.append("Include high-growth and volatile stock opportunities")
        
        # Learning recommendations
        if profile["learning_data"]["total_messages"] < 10:
            recommendations.append("Encourage more interaction to improve personalization")
        
        return recommendations


class PersonalityBatchProcessor:
    """Batch processing for personality analytics (RESTORED FROM v3)"""
    
    def __init__(self, personality_engine: UnifiedPersonalityEngine):
        self.engine = personality_engine
    
    async def process_user_batch(self, user_ids: List[str]) -> Dict[str, Any]:
        """Process personality analysis for multiple users"""
        results = {}
        
        for user_id in user_ids:
            try:
                profile = await self.engine.get_user_profile(user_id)
                results[user_id] = {
                    "profile_confidence": profile["confidence_score"],
                    "total_messages": profile["learning_data"]["total_messages"],
                    "communication_style": profile["communication_style"]["energy"],
                    "risk_tolerance": profile["trading_personality"]["risk_tolerance"],
                    "common_symbols": profile["trading_personality"]["common_symbols"][:5],
                    "learning_velocity": profile["learning_data"]["learning_velocity"]
                }
            except Exception as e:
                results[user_id] = {"error": str(e)}
        
        return {
            "batch_results": results,
            "total_processed": len(user_ids),
            "successful": len([r for r in results.values() if "error" not in r]),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def generate_aggregate_insights(self, user_profiles: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate aggregate insights across all users"""
        if not user_profiles:
            return {"error": "No profiles provided"}
        
        # Aggregate communication styles
        energy_levels = [p["communication_style"]["energy"] for p in user_profiles.values() 
                        if "communication_style" in p]
        
        # Aggregate risk tolerances
        risk_tolerances = [p["trading_personality"]["risk_tolerance"] for p in user_profiles.values()
                          if "trading_personality" in p]
        
        # Aggregate symbol preferences
        all_symbols = []
        for profile in user_profiles.values():
            if "trading_personality" in profile:
                all_symbols.extend(profile["trading_personality"].get("common_symbols", []))
        
        return {
            "total_users": len(user_profiles),
            "communication_distribution": dict(Counter(energy_levels)),
            "risk_distribution": dict(Counter(risk_tolerances)),
            "popular_symbols": dict(Counter(all_symbols).most_common(20)),
            "average_confidence": np.mean([p.get("confidence_score", 0) for p in user_profiles.values()]),
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }


class PersonalityDataManager:
    """Data management utilities for personality profiles (RESTORED FROM v3)"""
    
    def __init__(self, personality_engine: UnifiedPersonalityEngine):
        self.engine = personality_engine
    
    async def export_user_profile(self, user_id: str, format: str = "json") -> Union[str, Dict]:
        """Export user profile in specified format"""
        try:
            profile = await self.engine.get_user_profile(user_id)
            
            if format.lower() == "json":
                return json.dumps(profile, indent=2, default=str)
            elif format.lower() == "dict":
                return profile
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Profile export failed: {e}")
            return {"error": str(e)}
    
    async def import_user_profile(self, user_id: str, profile_data: Union[str, Dict]) -> bool:
        """Import user profile from data"""
        try:
            if isinstance(profile_data, str):
                profile = json.loads(profile_data)
            else:
                profile = profile_data
            
            # Validate profile structure
            if not self._validate_profile_structure(profile):
                raise ValueError("Invalid profile structure")
            
            # Update profile
            success = await self.engine.update_user_profile(user_id, profile)
            
            if success:
                logger.info(f"✅ Profile imported successfully for user {user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Profile import failed: {e}")
            return False
    
    def _validate_profile_structure(self, profile: Dict) -> bool:
        """Validate profile structure"""
        required_keys = [
            "profile_version", "communication_style", 
            "trading_personality", "learning_data"
        ]
        
        for key in required_keys:
            if key not in profile:
                return False
        
        return True
    
    async def backup_all_profiles(self) -> Dict[str, Any]:
        """Backup all user profiles"""
        try:
            backup_data = {
                "backup_timestamp": datetime.now(timezone.utc).isoformat(),
                "engine_version": self.engine.PROFILE_VERSION,
                "profiles": {}
            }
            
            for user_id in self.engine.user_profiles.keys():
                profile = await self.engine.get_user_profile(user_id)
                backup_data["profiles"][user_id] = profile
            
            backup_data["total_profiles"] = len(backup_data["profiles"])
            
            return backup_data
            
        except Exception as e:
            logger.error(f"Profile backup failed: {e}")
            return {"error": str(e)}
    
    async def restore_from_backup(self, backup_data: Dict) -> Dict[str, Any]:
        """Restore profiles from backup"""
        try:
            if "profiles" not in backup_data:
                raise ValueError("Invalid backup format")
            
            results = {"successful": 0, "failed": 0, "errors": []}
            
            for user_id, profile_data in backup_data["profiles"].items():
                success = await self.import_user_profile(user_id, profile_data)
                if success:
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(f"Failed to restore profile for {user_id}")
            
            logger.info(f"✅ Backup restore completed: {results['successful']} successful, {results['failed']} failed")
            
            return results
            
        except Exception as e:
            logger.error(f"Profile restore failed: {e}")
            return {"error": str(e)}


class PersonalityEngineMonitor:
    """Performance monitoring for personality engine (RESTORED FROM v3)"""
    
    def __init__(self, personality_engine: UnifiedPersonalityEngine):
        self.engine = personality_engine
        self.metrics = {
            "analysis_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "gemini_calls": 0,
            "fallback_calls": 0,
            "errors": 0,
            "total_processing_time": 0.0
        }
    
    def record_analysis(self, analysis: MessageAnalysis) -> None:
        """Record analysis metrics"""
        self.metrics["analysis_count"] += 1
        self.metrics["total_processing_time"] += analysis.processing_time_ms
        
        if analysis.analysis_method == "gemini":
            self.metrics["gemini_calls"] += 1
        else:
            self.metrics["fallback_calls"] += 1
    
    def record_cache_hit(self) -> None:
        """Record cache hit"""
        self.metrics["cache_hits"] += 1
    
    def record_cache_miss(self) -> None:
        """Record cache miss"""
        self.metrics["cache_misses"] += 1
    
    def record_error(self) -> None:
        """Record error"""
        self.metrics["errors"] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_analyses = self.metrics["analysis_count"]
        
        return {
            "total_analyses": total_analyses,
            "average_processing_time": (
                self.metrics["total_processing_time"] / max(1, total_analyses)
            ),
            "cache_hit_rate": (
                self.metrics["cache_hits"] / 
                max(1, self.metrics["cache_hits"] + self.metrics["cache_misses"])
            ),
            "gemini_usage_rate": (
                self.metrics["gemini_calls"] / max(1, total_analyses)
            ),
            "error_rate": (
                self.metrics["errors"] / max(1, total_analyses)
            ),
            "engine_efficiency": {
                "cache_efficiency": f"{self.metrics['cache_hits']}/{self.metrics['cache_hits'] + self.metrics['cache_misses']}",
                "gemini_vs_fallback": f"{self.metrics['gemini_calls']}/{self.metrics['fallback_calls']}",
                "total_profiles": len(self.engine.user_profiles)
            }
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        for key in self.metrics:
            if isinstance(self.metrics[key], (int, float)):
                self.metrics[key] = 0 if isinstance(self.metrics[key], int) else 0.0


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
    'PersonalityInsightsGenerator',
    'PersonalityBatchProcessor', 
    'PersonalityDataManager',
    'PersonalityEngineMonitor',
    'create_unified_personality_engine'
]
