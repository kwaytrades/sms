# core/personality_engine.py
import json
import asyncio
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from collections import defaultdict, Counter
from loguru import logger
from dataclasses import dataclass
import numpy as np


@dataclass
class ConversationMetrics:
    """Metrics for conversation analysis"""
    response_time_preference: float  # Preferred response speed
    complexity_preference: str      # simple/moderate/complex
    explanation_depth: str          # brief/standard/detailed  
    example_preference: bool        # Likes examples or not
    follow_up_tendency: float       # Likelihood to ask follow-ups


@dataclass
class TradingBehavior:
    """Trading behavior patterns"""
    decision_speed: str             # fast/moderate/slow
    research_depth: str             # minimal/moderate/thorough
    risk_communication: str         # direct/gentle/detailed
    profit_loss_sharing: bool       # Shares P&L openly
    advice_seeking_style: str       # specific/general/validation


class ComprehensivePersonalityEngine:
    """
    Advanced personality learning system with customer service, sales, and trading intelligence.
    Integrates with KeyBuilder for unified data management across Redis and MongoDB.
    """
    
    def __init__(self, db_service=None):
        """Initialize with optional database service for KeyBuilder integration"""
        self.db_service = db_service
        self.key_builder = db_service.key_builder if db_service and hasattr(db_service, 'key_builder') else None
        
        # Fallback in-memory storage for when KeyBuilder is not available
        self.user_profiles = defaultdict(self._create_default_profile)
        
        # Analysis patterns
        self.communication_patterns = self._load_communication_patterns()
        self.trading_patterns = self._load_trading_patterns()
        self.sales_indicators = self._load_sales_indicators()
        self.service_patterns = self._load_service_patterns()
        
        logger.info(f"🧠 PersonalityEngine initialized with KeyBuilder: {self.key_builder is not None}")
    
    # ==========================================
    # CORE USER PROFILE MANAGEMENT
    # ==========================================
    
    async def get_user_profile(self, user_id: str, create_if_missing: bool = True) -> Dict[str, Any]:
        """Get comprehensive user profile with KeyBuilder integration"""
        try:
            # Try KeyBuilder first if available
            if self.key_builder:
                profile = await self.key_builder.get_user_personality(user_id)
                if profile:
                    logger.debug(f"✅ Retrieved profile from KeyBuilder for {user_id}")
                    return profile
                
                if not create_if_missing:
                    return None
                
                # Create new profile with KeyBuilder
                logger.info(f"🆕 Creating new profile for {user_id}")
                profile = self._create_comprehensive_profile()
                await self.key_builder.set_user_personality(user_id, profile)
                return profile
            
            # Fallback to in-memory storage
            if user_id not in self.user_profiles and not create_if_missing:
                return None
                
            profile = self.user_profiles[user_id]
            if not profile or len(profile) < 10:  # Basic profile check
                self.user_profiles[user_id] = self._create_comprehensive_profile()
            
            return self.user_profiles[user_id]
            
        except Exception as e:
            logger.error(f"❌ Error getting user profile for {user_id}: {e}")
            return self._create_comprehensive_profile() if create_if_missing else None
    
    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user profile with KeyBuilder integration"""
        try:
            profile = await self.get_user_profile(user_id, create_if_missing=True)
            
            # Deep merge updates
            self._deep_merge_dict(profile, updates)
            profile["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            # Save with KeyBuilder if available
            if self.key_builder:
                success = await self.key_builder.set_user_personality(user_id, profile)
                if success:
                    logger.debug(f"✅ Updated profile via KeyBuilder for {user_id}")
                    return True
                else:
                    logger.warning(f"⚠️ KeyBuilder update failed for {user_id}, falling back to memory")
            
            # Fallback to in-memory
            self.user_profiles[user_id] = profile
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating profile for {user_id}: {e}")
            return False
    
    # ==========================================
    # MESSAGE ANALYSIS & LEARNING
    # ==========================================
    
    async def analyze_and_learn(self, user_id: str, user_message: str, bot_response: str = None, 
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive message analysis and learning"""
        try:
            profile = await self.get_user_profile(user_id)
            
            # Multi-dimensional analysis
            communication_analysis = self._analyze_communication_style(user_message, profile)
            trading_analysis = self._analyze_trading_content(user_message, profile)
            emotional_analysis = self._analyze_emotional_state(user_message, profile)
            intent_analysis = self._analyze_user_intent(user_message, profile)
            service_analysis = self._analyze_service_needs(user_message, profile)
            sales_analysis = self._analyze_sales_opportunity(user_message, profile)
            
            # Update profile with learnings
            updates = {
                "communication_style": self._update_communication_style(profile["communication_style"], communication_analysis),
                "trading_personality": self._update_trading_personality(profile["trading_personality"], trading_analysis),
                "emotional_intelligence": self._update_emotional_profile(profile["emotional_intelligence"], emotional_analysis),
                "service_profile": self._update_service_profile(profile["service_profile"], service_analysis),
                "sales_profile": self._update_sales_profile(profile["sales_profile"], sales_analysis),
                "conversation_history": self._update_conversation_history(profile["conversation_history"], user_message, bot_response, context),
                "analytics": self._update_analytics(profile["analytics"], intent_analysis, context)
            }
            
            await self.update_user_profile(user_id, updates)
            
            return {
                "communication_insights": communication_analysis,
                "trading_insights": trading_analysis,
                "emotional_state": emotional_analysis,
                "service_needs": service_analysis,
                "sales_opportunity": sales_analysis,
                "recommended_approach": self._generate_response_strategy(profile, communication_analysis, emotional_analysis)
            }
            
        except Exception as e:
            logger.error(f"❌ Error in analyze_and_learn for {user_id}: {e}")
            return {"error": str(e)}
    
    # ==========================================
    # COMMUNICATION STYLE ANALYSIS
    # ==========================================
    
    def _analyze_communication_style(self, message: str, profile: Dict) -> Dict[str, Any]:
        """Analyze communication patterns"""
        analysis = {
            "formality_score": self._detect_formality(message),
            "energy_level": self._detect_energy_level(message),
            "technical_depth": self._detect_technical_depth(message),
            "emoji_usage": self._count_emojis(message),
            "message_length": len(message),
            "urgency_level": self._detect_urgency(message),
            "question_type": self._classify_question_type(message),
            "communication_tone": self._detect_tone(message)
        }
        
        # Pattern learning
        analysis["patterns"] = {
            "greeting_style": self._extract_greeting_pattern(message),
            "closing_style": self._extract_closing_pattern(message),
            "question_structure": self._analyze_question_structure(message),
            "vocabulary_level": self._assess_vocabulary_level(message)
        }
        
        return analysis
    
    def _detect_formality(self, message: str) -> float:
        """Detect formality level (0.0 = very casual, 1.0 = very formal)"""
        casual_indicators = ['yo', 'hey', 'sup', 'gonna', 'wanna', 'lol', 'lmao', 'nah', 'yeah', 'kinda', 'sorta']
        formal_indicators = ['please', 'thank you', 'appreciate', 'kindly', 'regards', 'analysis', 'evaluation', 'assessment']
        
        message_lower = message.lower()
        casual_count = sum(1 for indicator in casual_indicators if indicator in message_lower)
        formal_count = sum(1 for indicator in formal_indicators if indicator in message_lower)
        
        total_words = len(message.split())
        if total_words == 0:
            return 0.5
        
        formality = (formal_count - casual_count) / total_words + 0.5
        return max(0.0, min(1.0, formality))
    
    def _detect_energy_level(self, message: str) -> str:
        """Detect energy level from message"""
        high_energy = ['!!!', '!!!', 'wow', 'amazing', 'awesome', 'excited', 'love', 'PUMP', 'MOON', '🚀', '💪', '🔥']
        low_energy = ['...', 'maybe', 'not sure', 'probably', 'might', 'whatever', 'ok', 'fine']
        
        message_lower = message.lower()
        high_count = sum(1 for indicator in high_energy if indicator in message_lower)
        low_count = sum(1 for indicator in low_energy if indicator in message_lower)
        
        if high_count > low_count:
            return "high"
        elif low_count > high_count:
            return "low"
        else:
            return "moderate"
    
    def _detect_technical_depth(self, message: str) -> str:
        """Detect preferred technical analysis depth"""
        basic_terms = ['price', 'up', 'down', 'buy', 'sell', 'good', 'bad']
        intermediate_terms = ['support', 'resistance', 'trend', 'volume', 'moving average', 'rsi', 'macd']
        advanced_terms = ['fibonacci', 'bollinger', 'stochastic', 'ichimoku', 'elliott wave', 'harmonic', 'divergence']
        
        message_lower = message.lower()
        basic_count = sum(1 for term in basic_terms if term in message_lower)
        intermediate_count = sum(1 for term in intermediate_terms if term in message_lower)
        advanced_count = sum(1 for term in advanced_terms if term in message_lower)
        
        if advanced_count > 0:
            return "advanced"
        elif intermediate_count > basic_count:
            return "intermediate"
        else:
            return "basic"
    
    # ==========================================
    # TRADING PERSONALITY ANALYSIS
    # ==========================================
    
    def _analyze_trading_content(self, message: str, profile: Dict) -> Dict[str, Any]:
        """Analyze trading-related content and behavior"""
        analysis = {
            "symbols_mentioned": self._extract_symbols(message),
            "trading_action": self._detect_trading_action(message),
            "risk_sentiment": self._analyze_risk_sentiment(message),
            "time_horizon": self._detect_time_horizon(message),
            "position_size_hints": self._detect_position_size_hints(message),
            "profit_loss_mention": self._detect_profit_loss_mention(message),
            "sector_preferences": self._detect_sector_mentions(message),
            "analysis_request": self._classify_analysis_request(message)
        }
        
        # Trading behavior patterns
        analysis["behavior_indicators"] = {
            "fomo_tendency": self._detect_fomo(message),
            "research_oriented": self._detect_research_orientation(message),
            "contrarian_tendency": self._detect_contrarian_signals(message),
            "momentum_focused": self._detect_momentum_focus(message),
            "news_driven": self._detect_news_sensitivity(message)
        }
        
        return analysis
    
    def _extract_symbols(self, message: str) -> List[str]:
        """Extract stock symbols from message"""
        # Pattern for stock symbols (3-5 capital letters)
        symbol_pattern = r'\b[A-Z]{2,5}\b'
        potential_symbols = re.findall(symbol_pattern, message)
        
        # Filter out common false positives
        false_positives = {'USD', 'USA', 'CEO', 'IPO', 'ETF', 'API', 'AI', 'VR', 'AR', 'IT', 'TV', 'PC', 'ATM', 'GPS', 'DNA', 'FBI', 'CIA', 'NBA', 'NFL', 'MLB', 'NHL'}
        
        return [symbol for symbol in potential_symbols if symbol not in false_positives]
    
    def _detect_trading_action(self, message: str) -> str:
        """Detect intended trading action"""
        buy_indicators = ['buy', 'buying', 'long', 'bullish', 'calls', 'purchase', 'invest in', 'go long']
        sell_indicators = ['sell', 'selling', 'short', 'bearish', 'puts', 'exit', 'close', 'dump']
        hold_indicators = ['hold', 'holding', 'keep', 'maintain', 'stay', 'wait']
        research_indicators = ['analyze', 'research', 'study', 'look into', 'thoughts on', 'opinion']
        
        message_lower = message.lower()
        
        buy_score = sum(1 for indicator in buy_indicators if indicator in message_lower)
        sell_score = sum(1 for indicator in sell_indicators if indicator in message_lower)
        hold_score = sum(1 for indicator in hold_indicators if indicator in message_lower)
        research_score = sum(1 for indicator in research_indicators if indicator in message_lower)
        
        scores = {'buy': buy_score, 'sell': sell_score, 'hold': hold_score, 'research': research_score}
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'unclear'
    
    def _analyze_risk_sentiment(self, message: str) -> Dict[str, Any]:
        """Analyze risk tolerance and sentiment"""
        conservative_terms = ['safe', 'stable', 'conservative', 'careful', 'protect', 'preserve', 'low risk']
        aggressive_terms = ['risky', 'aggressive', 'volatile', 'high risk', 'gamble', 'yolo', 'moon', 'diamond hands']
        
        message_lower = message.lower()
        conservative_count = sum(1 for term in conservative_terms if term in message_lower)
        aggressive_count = sum(1 for term in aggressive_terms if term in message_lower)
        
        return {
            "risk_tolerance": "conservative" if conservative_count > aggressive_count else "aggressive" if aggressive_count > 0 else "moderate",
            "confidence_level": self._detect_confidence_level(message),
            "uncertainty_indicators": self._detect_uncertainty(message)
        }
    
    # ==========================================
    # EMOTIONAL INTELLIGENCE
    # ==========================================
    
    def _analyze_emotional_state(self, message: str, profile: Dict) -> Dict[str, Any]:
        """Advanced emotional state analysis"""
        emotions = {
            "excitement": self._detect_excitement(message),
            "anxiety": self._detect_anxiety(message),
            "frustration": self._detect_frustration(message),
            "confidence": self._detect_confidence_level(message),
            "impatience": self._detect_impatience(message),
            "satisfaction": self._detect_satisfaction(message),
            "confusion": self._detect_confusion(message)
        }
        
        # Emotional triggers
        triggers = {
            "loss_aversion": self._detect_loss_aversion(message),
            "fomo": self._detect_fomo(message),
            "greed_indicators": self._detect_greed(message),
            "fear_indicators": self._detect_fear(message)
        }
        
        return {
            "primary_emotion": max(emotions, key=emotions.get),
            "emotional_intensity": max(emotions.values()),
            "emotional_triggers": triggers,
            "support_needed": self._assess_support_needs(emotions, triggers),
            "communication_approach": self._recommend_emotional_approach(emotions)
        }
    
    # ==========================================
    # CUSTOMER SERVICE ANALYSIS
    # ==========================================
    
    def _analyze_service_needs(self, message: str, profile: Dict) -> Dict[str, Any]:
        """Analyze customer service requirements"""
        service_indicators = {
            "complaint": self._detect_complaint(message),
            "technical_issue": self._detect_technical_issue(message),
            "billing_question": self._detect_billing_question(message),
            "feature_request": self._detect_feature_request(message),
            "urgent_issue": self._detect_urgency(message),
            "praise": self._detect_praise(message),
            "confusion": self._detect_confusion(message)
        }
        
        escalation_risk = self._assess_escalation_risk(message, service_indicators)
        
        return {
            "service_type": max(service_indicators, key=service_indicators.get),
            "urgency_level": self._calculate_urgency_score(service_indicators),
            "escalation_risk": escalation_risk,
            "recommended_response_time": self._recommend_response_time(service_indicators),
            "tone_recommendation": self._recommend_service_tone(service_indicators),
            "follow_up_needed": self._assess_follow_up_needs(service_indicators)
        }
    
    # ==========================================
    # SALES OPPORTUNITY ANALYSIS
    # ==========================================
    
    def _analyze_sales_opportunity(self, message: str, profile: Dict) -> Dict[str, Any]:
        """Analyze sales opportunities and readiness"""
        buying_signals = {
            "interest_level": self._detect_interest_level(message),
            "budget_indicators": self._detect_budget_indicators(message),
            "timeline_urgency": self._detect_timeline_urgency(message),
            "decision_authority": self._detect_decision_authority(message),
            "comparison_shopping": self._detect_comparison_shopping(message),
            "objection_indicators": self._detect_objections(message)
        }
        
        sales_stage = self._determine_sales_stage(buying_signals, profile)
        
        return {
            "sales_readiness_score": self._calculate_sales_readiness(buying_signals),
            "current_sales_stage": sales_stage,
            "recommended_approach": self._recommend_sales_approach(sales_stage, buying_signals),
            "objections_to_address": self._identify_objections(message),
            "upsell_opportunities": self._identify_upsell_opportunities(message, profile),
            "next_action": self._recommend_next_sales_action(sales_stage, buying_signals)
        }
    
    # ==========================================
    # RESPONSE STRATEGY GENERATION
    # ==========================================
    
    def _generate_response_strategy(self, profile: Dict, communication_analysis: Dict, 
                                   emotional_analysis: Dict) -> Dict[str, Any]:
        """Generate comprehensive response strategy"""
        return {
            "tone": self._determine_optimal_tone(profile, communication_analysis, emotional_analysis),
            "technical_level": self._determine_technical_level(profile, communication_analysis),
            "message_length": self._determine_optimal_length(profile, communication_analysis),
            "emoji_usage": self._determine_emoji_strategy(profile, communication_analysis),
            "urgency_handling": self._determine_urgency_approach(emotional_analysis),
            "personalization_elements": self._extract_personalization_elements(profile),
            "risk_communication": self._determine_risk_communication_style(profile, emotional_analysis),
            "follow_up_strategy": self._determine_follow_up_strategy(profile, emotional_analysis)
        }
    
    # ==========================================
    # PROFILE UPDATE METHODS
    # ==========================================
    
    def _update_communication_style(self, current_style: Dict, analysis: Dict) -> Dict:
        """Update communication style with weighted learning"""
        alpha = 0.2  # Learning rate
        
        # Update formality
        if "formality_score" in analysis:
            current_formality = {"casual": 0.0, "professional": 1.0, "friendly": 0.5}.get(current_style.get("formality", "friendly"), 0.5)
            new_formality = current_formality * (1 - alpha) + analysis["formality_score"] * alpha
            
            if new_formality < 0.33:
                current_style["formality"] = "casual"
            elif new_formality > 0.67:
                current_style["formality"] = "professional"
            else:
                current_style["formality"] = "friendly"
        
        # Update other communication attributes
        if "energy_level" in analysis:
            current_style["energy"] = analysis["energy_level"]
        
        if "technical_depth" in analysis:
            current_style["technical_depth"] = analysis["technical_depth"]
        
        # Update message length preference
        if "message_length" in analysis:
            if analysis["message_length"] < 50:
                current_style["message_length"] = "short"
            elif analysis["message_length"] > 200:
                current_style["message_length"] = "long"
            else:
                current_style["message_length"] = "medium"
        
        current_style["last_updated"] = datetime.now(timezone.utc).isoformat()
        return current_style
    
    def _update_trading_personality(self, current_trading: Dict, analysis: Dict) -> Dict:
        """Update trading personality with analysis"""
        # Update symbols tracking
        if "symbols_mentioned" in analysis and analysis["symbols_mentioned"]:
            current_symbols = current_trading.get("common_symbols", [])
            for symbol in analysis["symbols_mentioned"]:
                if symbol not in current_symbols:
                    current_symbols.append(symbol)
            current_trading["common_symbols"] = current_symbols[-20:]  # Keep last 20
        
        # Update risk tolerance
        if "risk_sentiment" in analysis:
            risk_tolerance = analysis["risk_sentiment"].get("risk_tolerance")
            if risk_tolerance:
                current_trading["risk_tolerance"] = risk_tolerance
        
        # Update trading action patterns
        if "trading_action" in analysis:
            actions = current_trading.get("recent_actions", [])
            actions.append({
                "action": analysis["trading_action"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            current_trading["recent_actions"] = actions[-10:]  # Keep last 10
        
        current_trading["last_updated"] = datetime.now(timezone.utc).isoformat()
        return current_trading
    
    # ==========================================
    # UTILITY METHODS
    # ==========================================
    
    def _create_comprehensive_profile(self) -> Dict[str, Any]:
        """Create a comprehensive default user profile"""
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            
            # Communication Style Analysis
            "communication_style": {
                "formality": "friendly",                    # casual/friendly/professional
                "energy": "moderate",                       # low/moderate/high
                "technical_depth": "intermediate",          # basic/intermediate/advanced
                "message_length": "medium",                 # short/medium/long
                "emoji_usage": "some",                      # none/minimal/some/lots
                "response_time_preference": "normal",       # immediate/normal/patient
                "explanation_preference": "balanced",       # brief/balanced/detailed
                "example_preference": True,                 # Likes examples
                "question_style": "direct",                 # direct/indirect/probing
                "feedback_style": "constructive"            # gentle/constructive/direct
            },
            
            # Trading Personality & Behavior
            "trading_personality": {
                "experience_level": "intermediate",         # beginner/intermediate/advanced
                "risk_tolerance": "moderate",               # conservative/moderate/aggressive
                "trading_style": "swing",                   # scalping/day/swing/position/long_term
                "decision_speed": "moderate",               # fast/moderate/deliberate
                "research_depth": "moderate",               # minimal/moderate/thorough
                "common_symbols": [],                       # Recently mentioned symbols
                "preferred_sectors": [],                    # Preferred market sectors
                "position_sizing": "moderate",              # small/moderate/large
                "profit_sharing_openness": False,           # Shares P&L information
                "loss_sharing_openness": False,             # Discusses losses
                "advice_seeking_style": "general",          # specific/general/validation
                "momentum_vs_value": "balanced",            # momentum/value/balanced
                "fundamental_vs_technical": "balanced"      # fundamental/technical/balanced
            },
            
            # Emotional Intelligence Profile
            "emotional_intelligence": {
                "primary_emotions": [],                     # Recent emotional states
                "emotional_triggers": {                     # What triggers emotions
                    "loss_aversion": 0.5,                   # 0.0-1.0 sensitivity
                    "fomo_susceptibility": 0.5,
                    "greed_indicators": 0.5,
                    "fear_indicators": 0.5,
                    "confidence_volatility": 0.5
                },
                "support_preferences": {                    # How they like support
                    "reassurance_needed": "moderate",       # low/moderate/high
                    "directness_preference": "balanced",    # gentle/balanced/direct
                    "education_vs_action": "balanced"       # education/balanced/action
                },
                "stress_indicators": [],                    # Patterns when stressed
                "celebration_style": "moderate"             # How they celebrate wins
            },
            
            # Customer Service Profile
            "service_profile": {
                "complaint_style": "constructive",          # passive/constructive/aggressive
                "issue_escalation_tendency": "moderate",    # low/moderate/high
                "technical_comfort_level": "intermediate",  # basic/intermediate/advanced
                "patience_level": "moderate",               # low/moderate/high
                "feedback_provision": "moderate",           # minimal/moderate/detailed
                "praise_expression": "moderate",            # minimal/moderate/enthusiastic
                "help_seeking_style": "specific",           # vague/specific/detailed
                "follow_up_preference": "moderate",         # none/moderate/frequent
                "communication_channel_preference": "sms",  # sms/email/phone/chat
                "response_time_expectation": "normal"       # immediate/normal/patient
            },
            
            # Sales & Conversion Profile
            "sales_profile": {
                "buying_readiness": 0.5,                    # 0.0-1.0 readiness score
                "price_sensitivity": "moderate",            # low/moderate/high
                "feature_prioritization": "value",          # features/value/price/support
                "decision_making_style": "analytical",      # impulsive/balanced/analytical
                "research_behavior": "moderate",            # minimal/moderate/extensive
                "social_proof_influence": "moderate",       # low/moderate/high
                "urgency_response": "moderate",             # low/moderate/high
                "objection_patterns": [],                   # Common objections
                "upsell_receptiveness": "moderate",         # low/moderate/high
                "loyalty_indicators": "developing"          # new/developing/strong/at_risk
            },
            
            # Conversation History & Context
            "conversation_history": {
                "total_conversations": 0,
                "recent_topics": [],                        # Last 10 topics
                "successful_interactions": 0,
                "problematic_interactions": 0,
                "preferred_interaction_types": [],          # question/analysis/validation/educational
                "context_memory": {                         # Things they've told us
                    "goals_mentioned": [],
                    "concerns_expressed": [],
                    "preferences_stated": [],
                    "plans_discussed": []
                },
                "engagement_patterns": {
                    "typical_session_length": "medium",     # short/medium/long
                    "follow_up_frequency": "moderate",      # rare/moderate/frequent
                    "topic_depth_preference": "moderate",   # surface/moderate/deep
                    "multi_topic_comfort": True             # Handles multiple topics
                }
            },
            
            # Analytics & Performance Tracking
            "analytics": {
                "personalization_accuracy": 0.5,           # How well we understand them
                "response_satisfaction_estimate": 0.5,     # Estimated satisfaction
                "engagement_score": 0.5,                   # Overall engagement
                "conversion_indicators": {                  # Sales conversion tracking
                    "feature_interest_score": 0.5,
                    "pricing_acceptance_score": 0.5,
                    "timing_readiness_score": 0.5,
                    "trust_building_score": 0.5
                },
                "learning_velocity": 0.5,                  # How fast we learn about them
                "prediction_accuracy": {},                 # How well we predict their needs
                "optimization_opportunities": []           # Areas for improvement
            }
        }
    
    def _create_default_profile(self) -> Dict[str, Any]:
        """Fallback method for defaultdict compatibility"""
        return self._create_comprehensive_profile()
    
    @staticmethod
    def _deep_merge_dict(base_dict: Dict, update_dict: Dict) -> None:
        """Deep merge update_dict into base_dict"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                ComprehensivePersonalityEngine._deep_merge_dict(base_dict[key], value)
            else:
                base_dict[key] = value
    
    # ==========================================
    # PATTERN DEFINITIONS
    # ==========================================
    
    def _load_communication_patterns(self) -> Dict:
        """Load communication analysis patterns"""
        return {
            "formality_indicators": {
                "casual": ['yo', 'hey', 'sup', 'gonna', 'wanna', 'lol', 'lmao', 'nah', 'yeah', 'kinda'],
                "professional": ['please', 'thank you', 'appreciate', 'kindly', 'regards', 'sincerely'],
                "friendly": ['thanks', 'great', 'awesome', 'cool', 'nice', 'good', 'perfect']
            },
            "energy_indicators": {
                "high": ['!!!', 'wow', 'amazing', 'awesome', 'excited', 'love', '🚀', '💪', '🔥', 'PUMP'],
                "low": ['...', 'maybe', 'not sure', 'probably', 'might', 'whatever', 'ok', 'fine'],
                "moderate": ['good', 'nice', 'thanks', 'sure', 'alright', 'sounds good']
            }
        }
    
    def _load_trading_patterns(self) -> Dict:
        """Load trading analysis patterns"""
        return {
            "action_indicators": {
                "buy": ['buy', 'buying', 'long', 'bullish', 'calls', 'purchase', 'invest in'],
                "sell": ['sell', 'selling', 'short', 'bearish', 'puts', 'exit', 'close'],
                "hold": ['hold', 'holding', 'keep', 'maintain', 'stay', 'wait'],
                "research": ['analyze', 'research', 'study', 'thoughts on', 'opinion', 'DD']
            },
            "risk_indicators": {
                "conservative": ['safe', 'stable', 'conservative', 'careful', 'protect', 'preserve'],
                "aggressive": ['risky', 'aggressive', 'volatile', 'gamble', 'yolo', 'moon', 'diamond hands']
            }
        }
    
    def _load_sales_indicators(self) -> Dict:
        """Load sales opportunity patterns"""
        return {
            "buying_signals": ['interested in', 'want to', 'need', 'looking for', 'considering', 'thinking about'],
            "price_sensitivity": ['cost', 'price', 'expensive', 'cheap', 'affordable', 'budget', 'worth it'],
            "urgency_signals": ['asap', 'urgent', 'quickly', 'soon', 'immediately', 'rush', 'deadline'],
            "comparison_signals": ['vs', 'versus', 'compared to', 'better than', 'alternatives', 'options']
        }
    
    def _load_service_patterns(self) -> Dict:
        """Load customer service patterns"""
        return {
            "complaint_indicators": ['problem', 'issue', 'bug', 'error', 'broken', 'not working', 'wrong'],
            "praise_indicators": ['great', 'excellent', 'amazing', 'love', 'perfect', 'awesome', 'fantastic'],
            "confusion_indicators": ['confused', 'don\'t understand', 'unclear', 'help', 'how do', 'what is']
        }
    
    # ==========================================
    # DETECTION HELPER METHODS
    # ==========================================
    
    def _count_emojis(self, message: str) -> int:
        """Count emoji usage in message"""
        emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U0001f900-\U0001f9ff\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f1e0-\U0001f1ff]'
        return len(re.findall(emoji_pattern, message))
    
    def _detect_urgency(self, message: str) -> float:
        """Detect urgency level (0.0-1.0)"""
        urgency_indicators = ['urgent', 'asap', 'quickly', 'immediately', 'rush', 'emergency', '!!!', 'help!']
        message_lower = message.lower()
        urgency_count = sum(1 for indicator in urgency_indicators if indicator in message_lower)
        return min(1.0, urgency_count * 0.3)
    
    def _detect_confidence_level(self, message: str) -> float:
        """Detect confidence level (0.0-1.0)"""
        confident_indicators = ['definitely', 'sure', 'certain', 'confident', 'absolutely', 'clearly']
        uncertain_indicators = ['maybe', 'perhaps', 'might', 'not sure', 'uncertain', 'probably']
        
        message_lower = message.lower()
        confident_count = sum(1 for indicator in confident_indicators if indicator in message_lower)
        uncertain_count = sum(1 for indicator in uncertain_indicators if indicator in message_lower)
        
        return max(0.0, min(1.0, (confident_count - uncertain_count) * 0.2 + 0.5))
    
    def _detect_fomo(self, message: str) -> float:
        """Detect FOMO indicators (0.0-1.0)"""
        fomo_indicators = ['missing out', 'everyone is', 'too late', 'should have', 'wish I', 'regret', 'fomo']
        message_lower = message.lower()
        fomo_count = sum(1 for indicator in fomo_indicators if indicator in message_lower)
        return min(1.0, fomo_count * 0.4)
    
    def _detect_anxiety(self, message: str) -> float:
        """Detect anxiety indicators (0.0-1.0)"""
        anxiety_indicators = ['worried', 'nervous', 'anxious', 'scared', 'afraid', 'concerned', 'stress']
        message_lower = message.lower()
        anxiety_count = sum(1 for indicator in anxiety_indicators if indicator in message_lower)
        return min(1.0, anxiety_count * 0.3)
    
    def _detect_excitement(self, message: str) -> float:
        """Detect excitement level (0.0-1.0)"""
        excitement_indicators = ['excited', 'pumped', 'thrilled', 'amazing', 'awesome', '🚀', '💪', '!!!']
        message_lower = message.lower()
        excitement_count = sum(1 for indicator in excitement_indicators if indicator in message_lower)
        emoji_boost = self._count_emojis(message) * 0.1
        return min(1.0, excitement_count * 0.3 + emoji_boost)
    
    # Complete detection methods for PersonalityEngine
# Add these methods to the ComprehensivePersonalityEngine class

    # ==========================================
    # COMPLETE DETECTION HELPER METHODS
    # ==========================================
    
    # EXISTING METHODS (already implemented):
    # _count_emojis, _detect_urgency, _detect_confidence_level, 
    # _detect_fomo, _detect_anxiety, _detect_excitement
    
    def _detect_frustration(self, message: str) -> float:
        """Detect frustration indicators (0.0-1.0)"""
        frustration_indicators = [
            'frustrated', 'annoyed', 'irritated', 'pissed', 'mad', 'angry',
            'ridiculous', 'stupid', 'waste of time', 'fed up', 'sick of',
            'this sucks', 'terrible', 'awful', 'horrible', 'wtf', 'seriously?',
            'come on', 'give me a break', 'unbelievable', 'ugh'
        ]
        message_lower = message.lower()
        frustration_count = sum(1 for indicator in frustration_indicators if indicator in message_lower)
        
        # Check for excessive punctuation (!!!, ???)
        excessive_punct = len(re.findall(r'[!?]{3,}', message))
        caps_words = len(re.findall(r'\b[A-Z]{2,}\b', message))
        
        return min(1.0, (frustration_count * 0.3) + (excessive_punct * 0.2) + (caps_words * 0.1))
    
    def _detect_satisfaction(self, message: str) -> float:
        """Detect satisfaction indicators (0.0-1.0)"""
        satisfaction_indicators = [
            'satisfied', 'happy', 'pleased', 'content', 'glad', 'delighted',
            'perfect', 'excellent', 'great job', 'well done', 'fantastic',
            'love it', 'exactly what', 'just what I needed', 'spot on',
            'nailed it', 'awesome', 'brilliant', 'outstanding', 'superb'
        ]
        message_lower = message.lower()
        satisfaction_count = sum(1 for indicator in satisfaction_indicators if indicator in message_lower)
        
        # Positive emoji boost
        positive_emojis = ['😊', '😀', '😃', '😄', '👍', '👏', '🎉', '✅', '💪', '🙌']
        emoji_boost = sum(1 for emoji in positive_emojis if emoji in message) * 0.2
        
        return min(1.0, (satisfaction_count * 0.3) + emoji_boost)
    
    def _detect_confusion(self, message: str) -> float:
        """Detect confusion indicators (0.0-1.0)"""
        confusion_indicators = [
            'confused', 'don\'t understand', 'unclear', 'lost', 'puzzled',
            'what do you mean', 'can you explain', 'not sure what',
            'how does', 'what is', 'help me understand', 'clarify',
            'not following', 'doesn\'t make sense', 'huh?', 'what?',
            'i\'m lost', 'can\'t figure out', 'struggling with'
        ]
        message_lower = message.lower()
        confusion_count = sum(1 for indicator in confusion_indicators if indicator in message_lower)
        
        # Question marks as confusion indicators
        question_marks = message.count('?')
        
        return min(1.0, (confusion_count * 0.4) + (question_marks * 0.1))
    
    def _detect_impatience(self, message: str) -> float:
        """Detect impatience indicators (0.0-1.0)"""
        impatience_indicators = [
            'hurry up', 'taking too long', 'when will', 'how long',
            'still waiting', 'any update', 'come on', 'quickly',
            'asap', 'urgent', 'need now', 'can\'t wait', 'time sensitive',
            'deadline', 'running out of time', 'behind schedule'
        ]
        message_lower = message.lower()
        impatience_count = sum(1 for indicator in impatience_indicators if indicator in message_lower)
        
        # Repeated characters as impatience (sooo, pleeeease)
        repeated_chars = len(re.findall(r'(.)\1{2,}', message_lower))
        
        return min(1.0, (impatience_count * 0.4) + (repeated_chars * 0.1))
    
    def _detect_uncertainty(self, message: str) -> float:
        """Detect uncertainty indicators (0.0-1.0)"""
        uncertainty_indicators = [
            'not sure', 'maybe', 'perhaps', 'might', 'could be',
            'i think', 'probably', 'possibly', 'uncertain', 'unsure',
            'don\'t know', 'hard to say', 'tough call', 'on the fence',
            'hesitant', 'second guessing', 'having doubts'
        ]
        message_lower = message.lower()
        uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator in message_lower)
        
        # Hedging language
        hedging_words = ['kind of', 'sort of', 'somewhat', 'rather', 'fairly']
        hedging_count = sum(1 for hedge in hedging_words if hedge in message_lower)
        
        return min(1.0, (uncertainty_count * 0.3) + (hedging_count * 0.2))
    
    def _detect_greed(self, message: str) -> float:
        """Detect greed indicators in trading context (0.0-1.0)"""
        greed_indicators = [
            'get rich', 'make bank', 'huge gains', 'massive profits',
            'to the moon', 'lambo', 'retirement money', 'life changing',
            'yolo', 'all in', 'diamond hands', 'hodl', 'moon mission',
            'millionaire', 'jackpot', 'easy money', 'quick buck'
        ]
        message_lower = message.lower()
        greed_count = sum(1 for indicator in greed_indicators if indicator in message_lower)
        
        # Money emojis and symbols
        money_symbols = ['💰', '💵', '💸', '🤑', '💎', '🚀', '$$$']
        money_boost = sum(1 for symbol in money_symbols if symbol in message) * 0.2
        
        return min(1.0, (greed_count * 0.4) + money_boost)
    
    def _detect_fear(self, message: str) -> float:
        """Detect fear indicators in trading context (0.0-1.0)"""
        fear_indicators = [
            'scared', 'afraid', 'terrified', 'panic', 'worried sick',
            'losing everything', 'going broke', 'can\'t afford',
            'bankruptcy', 'disaster', 'nightmare', 'catastrophe',
            'freaking out', 'heart attack', 'stressed out', 'sleepless'
        ]
        message_lower = message.lower()
        fear_count = sum(1 for indicator in fear_indicators if indicator in message_lower)
        
        # Fear emojis
        fear_emojis = ['😰', '😱', '😨', '😖', '😵', '💔']
        fear_boost = sum(1 for emoji in fear_emojis if emoji in message) * 0.2
        
        return min(1.0, (fear_count * 0.4) + fear_boost)
    
    def _detect_loss_aversion(self, message: str) -> float:
        """Detect loss aversion indicators (0.0-1.0)"""
        loss_aversion_indicators = [
            'can\'t lose', 'afraid to lose', 'protect my money',
            'cut losses', 'stop loss', 'minimize risk', 'play it safe',
            'don\'t want to lose', 'scared of losing', 'risk management',
            'preservation', 'conservative approach', 'safe investment'
        ]
        message_lower = message.lower()
        loss_aversion_count = sum(1 for indicator in loss_aversion_indicators if indicator in message_lower)
        
        return min(1.0, loss_aversion_count * 0.3)
    
    # ==========================================
    # TRADING-SPECIFIC DETECTION METHODS
    # ==========================================
    
    def _detect_time_horizon(self, message: str) -> str:
        """Detect trading time horizon"""
        scalping_indicators = ['scalp', 'quick trade', 'minutes', 'seconds', 'in and out']
        day_trading_indicators = ['day trade', 'intraday', 'same day', 'daily', 'today only']
        swing_indicators = ['swing', 'few days', 'week or two', 'short term', 'couple weeks']
        position_indicators = ['position', 'months', 'long term', 'hold for', 'buy and hold']
        
        message_lower = message.lower()
        
        scores = {
            'scalping': sum(1 for ind in scalping_indicators if ind in message_lower),
            'day_trading': sum(1 for ind in day_trading_indicators if ind in message_lower),
            'swing': sum(1 for ind in swing_indicators if ind in message_lower),
            'position': sum(1 for ind in position_indicators if ind in message_lower)
        }
        
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'unknown'
    
    def _detect_position_size_hints(self, message: str) -> str:
        """Detect position sizing hints"""
        small_indicators = ['small position', 'test the waters', 'dip my toe', 'conservative', 'little bit']
        medium_indicators = ['normal position', 'usual amount', 'regular size', 'moderate']
        large_indicators = ['big position', 'heavy', 'significant', 'major investment', 'all in', 'yolo']
        
        message_lower = message.lower()
        
        scores = {
            'small': sum(1 for ind in small_indicators if ind in message_lower),
            'medium': sum(1 for ind in medium_indicators if ind in message_lower),
            'large': sum(1 for ind in large_indicators if ind in message_lower)
        }
        
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'unknown'
    
    def _detect_profit_loss_mention(self, message: str) -> Dict[str, Any]:
        """Detect profit/loss mentions and amounts"""
        profit_indicators = ['profit', 'gain', 'made money', 'up', 'green', 'winner', 'successful']
        loss_indicators = ['loss', 'lost money', 'down', 'red', 'loser', 'underwater', 'baghold']
        
        message_lower = message.lower()
        
        # Extract monetary amounts
        money_pattern = r'\$([0-9,]+(?:\.[0-9]{2})?)'
        percentage_pattern = r'([0-9]+(?:\.[0-9]+)?)%'
        
        money_amounts = re.findall(money_pattern, message)
        percentages = re.findall(percentage_pattern, message)
        
        profit_count = sum(1 for ind in profit_indicators if ind in message_lower)
        loss_count = sum(1 for ind in loss_indicators if ind in message_lower)
        
        return {
            'type': 'profit' if profit_count > loss_count else 'loss' if loss_count > 0 else 'neutral',
            'money_amounts': money_amounts,
            'percentages': percentages,
            'sharing_openness': profit_count > 0 or loss_count > 0
        }
    
    def _detect_sector_mentions(self, message: str) -> List[str]:
        """Detect sector/industry mentions"""
        sectors = {
            'technology': ['tech', 'technology', 'software', 'ai', 'artificial intelligence', 'cloud', 'saas'],
            'healthcare': ['healthcare', 'biotech', 'pharma', 'medical', 'drug', 'vaccine'],
            'finance': ['bank', 'financial', 'fintech', 'insurance', 'credit'],
            'energy': ['oil', 'energy', 'renewable', 'solar', 'wind', 'gas', 'electric'],
            'retail': ['retail', 'consumer', 'ecommerce', 'shopping', 'fashion'],
            'automotive': ['auto', 'car', 'electric vehicle', 'ev', 'tesla'],
            'real_estate': ['real estate', 'reit', 'property', 'housing'],
            'aerospace': ['aerospace', 'defense', 'aviation', 'space']
        }
        
        message_lower = message.lower()
        mentioned_sectors = []
        
        for sector, keywords in sectors.items():
            if any(keyword in message_lower for keyword in keywords):
                mentioned_sectors.append(sector)
        
        return mentioned_sectors
    
    def _classify_analysis_request(self, message: str) -> str:
        """Classify type of analysis being requested"""
        technical_indicators = ['chart', 'technical', 'ta', 'support', 'resistance', 'rsi', 'macd', 'moving average']
        fundamental_indicators = ['fundamental', 'earnings', 'revenue', 'pe ratio', 'valuation', 'balance sheet']
        news_indicators = ['news', 'recent news', 'headlines', 'events', 'announcements']
        options_indicators = ['options', 'calls', 'puts', 'strike', 'expiration', 'iv', 'implied volatility']
        
        message_lower = message.lower()
        
        scores = {
            'technical': sum(1 for ind in technical_indicators if ind in message_lower),
            'fundamental': sum(1 for ind in fundamental_indicators if ind in message_lower),
            'news': sum(1 for ind in news_indicators if ind in message_lower),
            'options': sum(1 for ind in options_indicators if ind in message_lower)
        }
        
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'general'
    
    def _detect_research_orientation(self, message: str) -> float:
        """Detect how research-oriented the user is (0.0-1.0)"""
        research_indicators = [
            'research', 'dd', 'due diligence', 'analyze', 'study', 'investigate',
            'deep dive', 'look into', 'examine', 'evaluation', 'assessment',
            'data', 'metrics', 'fundamentals', 'technicals', 'reports'
        ]
        message_lower = message.lower()
        research_count = sum(1 for indicator in research_indicators if indicator in message_lower)
        
        return min(1.0, research_count * 0.3)
    
    def _detect_contrarian_signals(self, message: str) -> float:
        """Detect contrarian trading tendency (0.0-1.0)"""
        contrarian_indicators = [
            'against the crowd', 'contrarian', 'opposite', 'everyone else',
            'bucking the trend', 'fade the move', 'sell when others buy',
            'buy the dip', 'when others are fearful', 'unpopular opinion'
        ]
        message_lower = message.lower()
        contrarian_count = sum(1 for indicator in contrarian_indicators if indicator in message_lower)
        
        return min(1.0, contrarian_count * 0.4)
    
    def _detect_momentum_focus(self, message: str) -> float:
        """Detect momentum trading focus (0.0-1.0)"""
        momentum_indicators = [
            'momentum', 'trending', 'breakout', 'follow the trend',
            'riding the wave', 'hot stocks', 'moving fast', 'volume surge',
            'price action', 'strong move', 'uptrend', 'downtrend'
        ]
        message_lower = message.lower()
        momentum_count = sum(1 for indicator in momentum_indicators if indicator in message_lower)
        
        return min(1.0, momentum_count * 0.3)
    
    def _detect_news_sensitivity(self, message: str) -> float:
        """Detect sensitivity to news events (0.0-1.0)"""
        news_indicators = [
            'news', 'announcement', 'earnings', 'report', 'event',
            'catalyst', 'breaking news', 'headlines', 'press release',
            'rumors', 'speculation', 'insider', 'leak'
        ]
        message_lower = message.lower()
        news_count = sum(1 for indicator in news_indicators if indicator in message_lower)
        
        return min(1.0, news_count * 0.3)
    
    # ==========================================
    # COMMUNICATION PATTERN DETECTION
    # ==========================================
    
    def _classify_question_type(self, message: str) -> str:
        """Classify the type of question being asked"""
        if '?' not in message:
            return 'statement'
        
        yes_no_indicators = ['is', 'are', 'was', 'were', 'will', 'would', 'should', 'can', 'could', 'do', 'does', 'did']
        what_indicators = ['what', 'which', 'who', 'where', 'when', 'why', 'how']
        
        message_lower = message.lower()
        
        if any(message_lower.startswith(ind) for ind in yes_no_indicators):
            return 'yes_no'
        elif any(ind in message_lower for ind in what_indicators):
            return 'open_ended'
        else:
            return 'clarification'
    
    def _detect_tone(self, message: str) -> str:
        """Detect overall communication tone"""
        positive_indicators = ['great', 'awesome', 'love', 'excellent', 'fantastic', 'amazing', 'perfect']
        negative_indicators = ['hate', 'terrible', 'awful', 'horrible', 'worst', 'sucks', 'disappointed']
        neutral_indicators = ['okay', 'fine', 'alright', 'normal', 'regular', 'standard']
        
        message_lower = message.lower()
        
        positive_count = sum(1 for ind in positive_indicators if ind in message_lower)
        negative_count = sum(1 for ind in negative_indicators if ind in message_lower)
        neutral_count = sum(1 for ind in neutral_indicators if ind in message_lower)
        
        if positive_count > negative_count and positive_count > neutral_count:
            return 'positive'
        elif negative_count > positive_count and negative_count > neutral_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _extract_greeting_pattern(self, message: str) -> str:
        """Extract greeting style from message"""
        casual_greetings = ['yo', 'hey', 'sup', 'what\'s up', 'howdy']
        formal_greetings = ['hello', 'good morning', 'good afternoon', 'good evening', 'greetings']
        friendly_greetings = ['hi', 'hey there', 'hello there', 'good day']
        
        message_lower = message.lower()
        
        if any(greeting in message_lower for greeting in casual_greetings):
            return 'casual'
        elif any(greeting in message_lower for greeting in formal_greetings):
            return 'formal'
        elif any(greeting in message_lower for greeting in friendly_greetings):
            return 'friendly'
        else:
            return 'none'
    
    def _extract_closing_pattern(self, message: str) -> str:
        """Extract closing style from message"""
        casual_closings = ['later', 'bye', 'see ya', 'peace', 'ttyl', 'talk later']
        formal_closings = ['thank you', 'regards', 'sincerely', 'best regards', 'kind regards']
        friendly_closings = ['thanks', 'cheers', 'take care', 'have a good one']
        
        message_lower = message.lower()
        
        if any(closing in message_lower for closing in casual_closings):
            return 'casual'
        elif any(closing in message_lower for closing in formal_closings):
            return 'formal'
        elif any(closing in message_lower for closing in friendly_closings):
            return 'friendly'
        else:
            return 'none'
    
    def _analyze_question_structure(self, message: str) -> Dict[str, Any]:
        """Analyze how user structures questions"""
        question_count = message.count('?')
        
        direct_questions = ['what is', 'how much', 'when will', 'where can', 'why did']
        indirect_questions = ['i wonder', 'curious about', 'thinking about', 'not sure if']
        
        message_lower = message.lower()
        
        direct_count = sum(1 for q in direct_questions if q in message_lower)
        indirect_count = sum(1 for q in indirect_questions if q in message_lower)
        
        return {
            'question_count': question_count,
            'style': 'direct' if direct_count > indirect_count else 'indirect' if indirect_count > 0 else 'mixed',
            'complexity': 'simple' if question_count <= 1 else 'complex'
        }
    
    def _assess_vocabulary_level(self, message: str) -> str:
        """Assess vocabulary sophistication level"""
        basic_words = ['good', 'bad', 'big', 'small', 'nice', 'okay', 'cool', 'great']
        advanced_words = ['sophisticated', 'comprehensive', 'substantial', 'exceptional', 'intricate', 'nuanced']
        technical_words = ['volatility', 'correlation', 'diversification', 'allocation', 'optimization']
        
        words = message.lower().split()
        
        basic_count = sum(1 for word in words if word in basic_words)
        advanced_count = sum(1 for word in words if word in advanced_words)
        technical_count = sum(1 for word in words if word in technical_words)
        
        if technical_count > 0:
            return 'technical'
        elif advanced_count > basic_count:
            return 'advanced'
        elif basic_count > 0:
            return 'basic'
        else:
            return 'intermediate'
    
    # ==========================================
    # CUSTOMER SERVICE DETECTION METHODS
    # ==========================================
    
    def _detect_complaint(self, message: str) -> float:
        """Detect complaint indicators (0.0-1.0)"""
        complaint_indicators = [
            'complaint', 'problem', 'issue', 'trouble', 'error', 'bug',
            'not working', 'broken', 'failed', 'wrong', 'incorrect',
            'disappointed', 'frustrated', 'upset', 'angry', 'dissatisfied'
        ]
        message_lower = message.lower()
        complaint_count = sum(1 for indicator in complaint_indicators if indicator in message_lower)
        
        return min(1.0, complaint_count * 0.3)
    
    def _detect_technical_issue(self, message: str) -> float:
        """Detect technical issue indicators (0.0-1.0)"""
        technical_indicators = [
            'won\'t load', 'can\'t access', 'error message', 'crashed',
            'frozen', 'slow', 'timeout', 'connection', 'login issue',
            'password', 'reset', 'sync problem', 'glitch'
        ]
        message_lower = message.lower()
        technical_count = sum(1 for indicator in technical_indicators if indicator in message_lower)
        
        return min(1.0, technical_count * 0.4)
    
    def _detect_billing_question(self, message: str) -> float:
        """Detect billing/payment related questions (0.0-1.0)"""
        billing_indicators = [
            'billing', 'payment', 'charge', 'invoice', 'subscription',
            'refund', 'cancel', 'upgrade', 'downgrade', 'plan',
            'credit card', 'account', 'fee', 'cost', 'price'
        ]
        message_lower = message.lower()
        billing_count = sum(1 for indicator in billing_indicators if indicator in message_lower)
        
        return min(1.0, billing_count * 0.4)
    
    def _detect_feature_request(self, message: str) -> float:
        """Detect feature request indicators (0.0-1.0)"""
        feature_indicators = [
            'feature request', 'suggestion', 'would be nice', 'could you add',
            'enhancement', 'improvement', 'new feature', 'wish you had',
            'it would be great if', 'please add', 'consider adding'
        ]
        message_lower = message.lower()
        feature_count = sum(1 for indicator in feature_indicators if indicator in message_lower)
        
        return min(1.0, feature_count * 0.4)
    
    def _detect_praise(self, message: str) -> float:
        """Detect praise/positive feedback (0.0-1.0)"""
        praise_indicators = [
            'great job', 'excellent', 'amazing', 'fantastic', 'awesome',
            'love it', 'perfect', 'brilliant', 'outstanding', 'wonderful',
            'impressed', 'exceeded expectations', 'exactly what I needed'
        ]
        message_lower = message.lower()
        praise_count = sum(1 for indicator in praise_indicators if indicator in message_lower)
        
        # Positive emojis boost
        positive_emojis = ['😊', '😀', '👍', '👏', '🎉', '❤️', '💯']
        emoji_boost = sum(1 for emoji in positive_emojis if emoji in message) * 0.2
        
        return min(1.0, (praise_count * 0.3) + emoji_boost)
    
    # ==========================================
    # SALES OPPORTUNITY DETECTION METHODS
    # ==========================================
    
    def _detect_interest_level(self, message: str) -> float:
        """Detect sales interest level (0.0-1.0)"""
        high_interest = ['very interested', 'definitely want', 'need this', 'when can I', 'ready to buy']
        medium_interest = ['interested', 'might be good', 'considering', 'tell me more', 'sounds good']
        low_interest = ['maybe later', 'not sure', 'just looking', 'not ready', 'still thinking']
        
        message_lower = message.lower()
        
        high_count = sum(1 for ind in high_interest if ind in message_lower)
        medium_count = sum(1 for ind in medium_interest if ind in message_lower)
        low_count = sum(1 for ind in low_interest if ind in message_lower)
        
        if high_count > 0:
            return min(1.0, high_count * 0.8)
        elif medium_count > 0:
            return min(1.0, medium_count * 0.5)
        elif low_count > 0:
            return max(0.0, 0.5 - (low_count * 0.2))
        else:
            return 0.5
    
    def _detect_budget_indicators(self, message: str) -> Dict[str, Any]:
        """Detect budget-related information"""
        budget_indicators = {
            'has_budget': ['budget', 'can afford', 'price range', 'spending limit'],
            'no_budget': ['no budget', 'can\'t afford', 'too expensive', 'out of budget'],
            'flexible': ['flexible', 'depends on value', 'worth it', 'investment']
        }
        
        message_lower = message.lower()
        
        # Extract monetary amounts
        money_pattern = r'\$([0-9,]+(?:\.[0-9]{2})?)'
        amounts = re.findall(money_pattern, message)
        
        budget_status = 'unknown'
        for status, indicators in budget_indicators.items():
            if any(ind in message_lower for ind in indicators):
                budget_status = status
                break
        
        return {
            'status': budget_status,
            'amounts_mentioned': amounts,
            'price_sensitivity': self._detect_price_sensitivity(message)
        }
    
    def _detect_price_sensitivity(self, message: str) -> str:
        """Detect price sensitivity level"""
        high_sensitivity = ['too expensive', 'can\'t afford', 'cheaper option', 'budget conscious']
        low_sensitivity = ['price doesn\'t matter', 'money no object', 'best quality', 'premium']
        
        message_lower = message.lower()
        
        if any(ind in message_lower for ind in high_sensitivity):
            return 'high'
        elif any(ind in message_lower for ind in low_sensitivity):
            return 'low'
        else:
            return 'medium'
    
    def _detect_timeline_urgency(self, message: str) -> str:
        """Detect purchase timeline urgency"""
        immediate = ['now', 'today', 'asap', 'immediately', 'urgent', 'right away']
        soon = ['this week', 'soon', 'quickly', 'in a few days', 'by friday']
        later = ['next month', 'eventually', 'sometime', 'no rush', 'when I\'m ready']
        
        message_lower = message.lower()
        
        if any(ind in message_lower for ind in immediate):
            return 'immediate'
        elif any(ind in message_lower for ind in soon):
            return 'soon'
        elif any(ind in message_lower for ind in later):
            return 'later'
        else:
            return 'unspecified'
    
    def _detect_decision_authority(self, message: str) -> str:
        """Detect decision-making authority"""
        decision_maker = ['I decide', 'my choice', 'up to me', 'I\'m in charge']
        influencer = ['need to discuss', 'talk to my', 'get approval', 'team decision']
        no_authority = ['not my decision', 'someone else decides', 'I just research']
        
        message_lower = message.lower()
        
        if any(ind in message_lower for ind in decision_maker):
            return 'decision_maker'
        elif any(ind in message_lower for ind in influencer):
            return 'influencer'
        elif any(ind in message_lower for ind in no_authority):
            return 'no_authority'
        else:
            return 'unknown'
    
    def _detect_comparison_shopping(self, message: str) -> float:
        """Detect comparison shopping behavior (0.0-1.0)"""
        comparison_indicators = [
            'compared to', 'vs', 'versus', 'better than', 'alternatives',
            'other options', 'competitors', 'shopping around', 'best deal'
        ]
        message_lower = message.lower()
        comparison_count = sum(1 for indicator in comparison_indicators if indicator in message_lower)
        
        return min(1.0, comparison_count * 0.4)
    
    def _detect_objections(self, message: str) -> List[str]:
        """Detect sales objections"""
        objection_patterns = {
            'price': ['too expensive', 'costs too much', 'over budget', 'can\'t afford'],
            'features': ['doesn\'t have', 'missing', 'lacks', 'wish it had'],
            'timing': ['not ready', 'wrong time', 'maybe later', 'in the future'],
            'trust': ['not sure', 'hesitant', 'concerned', 'worried about'],
            'competition': ['using something else', 'happy with current', 'already have']
        }
        
        message_lower = message.lower()
        detected_objections = []
        
        for objection_type, indicators in objection_patterns.items():
            if any(ind in message_lower for ind in indicators):
                detected_objections.append(objection_type)
        
        return detected_objections
    
    # ==========================================
    # ASSESSMENT AND RECOMMENDATION METHODS
    # ==========================================
    
    def _assess_support_needs(self, emotions: Dict[str, float], triggers: Dict[str, float]) -> str:
        """Assess what kind of support the user needs"""
        high_anxiety = emotions.get('anxiety', 0) > 0.6
        high_frustration = emotions.get('frustration', 0) > 0.6
        high_confusion = emotions.get('confusion', 0) > 0.6
        high_fear = triggers.get('fear_indicators', 0) > 0.6
        
        if high_fear or high_anxiety:
            return 'emotional_reassurance'
        elif high_frustration:
            return 'problem_solving'
        elif high_confusion:
            return 'education_and_clarity'
        else:
            return 'standard_guidance'
    
    def _recommend_emotional_approach(self, emotions: Dict[str, float]) -> str:
        """Recommend emotional communication approach"""
        dominant_emotion = max(emotions, key=emotions.get)
        intensity = emotions[dominant_emotion]
        
        if dominant_emotion == 'anxiety' and intensity > 0.6:
            return 'calm_and_reassuring'
        elif dominant_emotion == 'excitement' and intensity > 0.6:
            return 'enthusiastic_but_cautious'
        elif dominant_emotion == 'frustration' and intensity > 0.6:
            return 'empathetic_and_solution_focused'
        elif dominant_emotion == 'confusion' and intensity > 0.6:
            return 'clear_and_educational'
        else:
            return 'balanced_and_professional'
    
    def _assess_escalation_risk(self, message: str, service_indicators: Dict[str, float]) -> str:
        """Assess risk of customer service escalation"""
        complaint_level = service_indicators.get('complaint', 0)
        urgency_level = service_indicators.get('urgent_issue', 0)
        frustration_detected = self._detect_frustration(message)
        
        escalation_score = (complaint_level * 0.4) + (urgency_level * 0.3) + (frustration_detected * 0.3)
        
        if escalation_score > 0.7:
            return 'high'
        elif escalation_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_urgency_score(self, service_indicators: Dict[str, float]) -> float:
        """Calculate overall urgency score"""
        urgency_factors = [
            service_indicators.get('urgent_issue', 0),
            service_indicators.get('technical_issue', 0) * 0.7,
            service_indicators.get('billing_question', 0) * 0.5
        ]
        
        return min(1.0, sum(urgency_factors))
    
    def _recommend_response_time(self, service_indicators: Dict[str, float]) -> str:
        """Recommend response time based on indicators"""
        urgency_score = self._calculate_urgency_score(service_indicators)
        
        if urgency_score > 0.8:
            return 'immediate'
        elif urgency_score > 0.5:
            return 'within_hour'
        elif urgency_score > 0.2:
            return 'within_day'
        else:
            return 'standard'
    
    def _recommend_service_tone(self, service_indicators: Dict[str, float]) -> str:
        """Recommend customer service tone"""
        if service_indicators.get('praise', 0) > 0.5:
            return 'appreciative'
        elif service_indicators.get('complaint', 0) > 0.5:
            return 'empathetic_and_solution_focused'
        elif service_indicators.get('confusion', 0) > 0.5:
            return 'patient_and_educational'
        else:
            return 'professional_and_helpful'
    
    def _assess_follow_up_needs(self, service_indicators: Dict[str, float]) -> bool:
        """Assess if follow-up is needed"""
        needs_followup = (
            service_indicators.get('technical_issue', 0) > 0.3 or
            service_indicators.get('complaint', 0) > 0.3 or
            service_indicators.get('billing_question', 0) > 0.3
        )
        
        return needs_followup
    
    def _calculate_sales_readiness(self, buying_signals: Dict[str, float]) -> float:
        """Calculate overall sales readiness score"""
        factors = [
            buying_signals.get('interest_level', 0) * 0.3,
            buying_signals.get('budget_indicators', 0) * 0.2,
            buying_signals.get('timeline_urgency', 0) * 0.2,
            buying_signals.get('decision_authority', 0) * 0.15,
            (1.0 - buying_signals.get('objection_indicators', 0)) * 0.15
        ]
        
        return sum(factors)
    
    def _determine_sales_stage(self, buying_signals: Dict[str, Any], profile: Dict) -> str:
        """Determine current sales stage"""
        readiness = self._calculate_sales_readiness(buying_signals)
        
        if readiness > 0.8:
            return 'ready_to_buy'
        elif readiness > 0.6:
            return 'evaluation'
        elif readiness > 0.4:
            return 'consideration'
        elif readiness > 0.2:
            return 'awareness'
        else:
            return 'unqualified'
    
    def _recommend_sales_approach(self, sales_stage: str, buying_signals: Dict) -> str:
        """Recommend sales approach based on stage"""
        approaches = {
            'ready_to_buy': 'close_the_sale',
            'evaluation': 'provide_detailed_information',
            'consideration': 'build_value_and_trust',
            'awareness': 'educate_and_nurture',
            'unqualified': 'qualify_and_discover_needs'
        }
        
        return approaches.get(sales_stage, 'standard_engagement')
    
    def _identify_objections(self, message: str) -> List[str]:
        """Identify specific objections to address"""
        return self._detect_objections(message)
    
    def _identify_upsell_opportunities(self, message: str, profile: Dict) -> List[str]:
        """Identify upsell opportunities"""
        opportunities = []
        
        # Check for advanced feature interest
        if 'advanced' in message.lower() or 'more features' in message.lower():
            opportunities.append('premium_features')
        
        # Check for volume/frequency indicators
        if any(word in message.lower() for word in ['daily', 'frequently', 'often', 'multiple']):
            opportunities.append('higher_usage_plan')
        
        # Check for team/collaboration mentions
        if any(word in message.lower() for word in ['team', 'colleagues', 'share', 'collaborate']):
            opportunities.append('team_plan')
        
        return opportunities
    
    def _recommend_next_sales_action(self, sales_stage: str, buying_signals: Dict) -> str:
        """Recommend next sales action"""
        actions = {
            'ready_to_buy': 'send_purchase_link',
            'evaluation': 'schedule_demo',
            'consideration': 'send_case_studies',
            'awareness': 'provide_educational_content',
            'unqualified': 'ask_qualifying_questions'
        }
        
        return actions.get(sales_stage, 'continue_conversation')
    
    async def get_personalized_prompt(self, user_id: str, user_message: str, context: Dict = None) -> str:
        """Generate personalized prompt for LLM based on user profile"""
        try:
            profile = await self.get_user_profile(user_id)
            analysis = await self.analyze_and_learn(user_id, user_message, context=context)
            
            # Build comprehensive prompt
            prompt_parts = [
                "You are a highly personalized SMS trading assistant. Here's what you know about this user:",
                "",
                f"COMMUNICATION STYLE:",
                f"- Formality: {profile['communication_style']['formality']}",
                f"- Energy: {profile['communication_style']['energy']}",
                f"- Technical depth: {profile['communication_style']['technical_depth']}",
                f"- Message length preference: {profile['communication_style']['message_length']}",
                f"- Emoji usage: {profile['communication_style']['emoji_usage']}",
                "",
                f"TRADING PERSONALITY:",
                f"- Experience: {profile['trading_personality']['experience_level']}",
                f"- Risk tolerance: {profile['trading_personality']['risk_tolerance']}",
                f"- Trading style: {profile['trading_personality']['trading_style']}",
                f"- Recent symbols: {', '.join(profile['trading_personality']['common_symbols'][-5:]) if profile['trading_personality']['common_symbols'] else 'None yet'}",
                "",
                f"EMOTIONAL STATE: {analysis.get('emotional_state', {}).get('primary_emotion', 'neutral')}",
                "",
                f"RESPONSE STRATEGY: {analysis.get('recommended_approach', {})}",
                "",
                f"USER MESSAGE: {user_message}",
                "",
                "Respond as their personalized trading assistant, matching their style perfectly:"
            ]
            
            return "\n".join(prompt_parts)
            
        except Exception as e:
            logger.error(f"❌ Error generating personalized prompt: {e}")
            return f"USER MESSAGE: {user_message}\n\nRespond as a helpful trading assistant:"
    
    async def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user insights for admin/debugging"""
        try:
            profile = await self.get_user_profile(user_id, create_if_missing=False)
            if not profile:
                return {"error": "User profile not found"}
            
            return {
                "user_id": user_id,
                "profile_completeness": self._calculate_profile_completeness(profile),
                "communication_summary": self._summarize_communication_style(profile),
                "trading_summary": self._summarize_trading_personality(profile),
                "emotional_summary": self._summarize_emotional_profile(profile),
                "service_summary": self._summarize_service_profile(profile),
                "sales_summary": self._summarize_sales_profile(profile),
                "personalization_score": profile.get("analytics", {}).get("personalization_accuracy", 0.5),
                "recommendations": self._generate_improvement_recommendations(profile)
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting user insights: {e}")
            return {"error": str(e)}
    
    def _calculate_profile_completeness(self, profile: Dict) -> float:
        """Calculate how complete the user profile is (0.0-1.0)"""
        total_conversations = profile.get("conversation_history", {}).get("total_conversations", 0)
        
        # Base completeness on conversation count and data richness
        conversation_score = min(1.0, total_conversations / 20)  # Full score at 20+ conversations
        
        # Data richness factors
        has_symbols = len(profile.get("trading_personality", {}).get("common_symbols", [])) > 0
        has_preferences = len(profile.get("conversation_history", {}).get("context_memory", {}).get("preferences_stated", [])) > 0
        has_emotional_data = len(profile.get("emotional_intelligence", {}).get("primary_emotions", [])) > 0
        
        richness_score = sum([has_symbols, has_preferences, has_emotional_data]) / 3
        
        return (conversation_score * 0.7) + (richness_score * 0.3)
    
    # Summary methods for insights
    def _summarize_communication_style(self, profile: Dict) -> str:
        style = profile.get("communication_style", {})
        return f"{style.get('formality', 'friendly').title()} tone, {style.get('energy', 'moderate')} energy, prefers {style.get('message_length', 'medium')} responses"
    
    def _summarize_trading_personality(self, profile: Dict) -> str:
        trading = profile.get("trading_personality", {})
        return f"{trading.get('experience_level', 'intermediate').title()} {trading.get('trading_style', 'swing')} trader with {trading.get('risk_tolerance', 'moderate')} risk tolerance"
    
    def _summarize_emotional_profile(self, profile: Dict) -> str:
        emotions = profile.get("emotional_intelligence", {}).get("primary_emotions", [])
        if emotions:
            return f"Recent emotions: {', '.join(emotions[-3:])}"
        return "Emotional patterns still developing"
    
    def _summarize_service_profile(self, profile: Dict) -> str:
        service = profile.get("service_profile", {})
        return f"{service.get('patience_level', 'moderate').title()} patience, {service.get('technical_comfort_level', 'intermediate')} technical comfort"
    
    def _summarize_sales_profile(self, profile: Dict) -> str:
        sales = profile.get("sales_profile", {})
        readiness = sales.get("buying_readiness", 0.5)
        return f"Buying readiness: {readiness:.1%}, {sales.get('decision_making_style', 'analytical')} decision maker"
