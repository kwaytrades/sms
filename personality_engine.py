# core/personality_engine_v2_complete.py
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


@dataclass
class MessageAnalysis:
    """Container for comprehensive message analysis results"""
    communication_insights: Dict[str, Any]
    trading_insights: Dict[str, Any]
    emotional_state: Dict[str, Any]
    service_needs: Dict[str, Any]
    sales_opportunity: Dict[str, Any]
    intent_analysis: Dict[str, Any]
    preprocessed_tokens: Dict[str, Any]  # Cached preprocessing results


class OptimizedPersonalityEngine:
    """
    Complete OptimizedPersonalityEngine v2.0 with all detection methods integrated.
    Enhanced with robust classification logic, time-decay mechanism, conversation history,
    global intelligence layer, and profile confidence scoring.
    """
    
    # Class-level constants for performance
    PROFILE_VERSION = 2.0
    DEFAULT_LEARNING_RATE = 0.2
    MIN_LEARNING_RATE = 0.05
    MAX_LEARNING_RATE = 0.4
    TIME_DECAY_FACTOR = 0.95  # Daily decay factor for profile attributes
    
    def __init__(self, db_service=None):
        """Initialize with optional database service for KeyBuilder integration"""
        self.db_service = db_service
        self.key_builder = db_service.key_builder if db_service and hasattr(db_service, 'key_builder') else None
        
        # Load authoritative ticker lists for symbol validation
        self.authoritative_tickers = self._load_authoritative_tickers()
        
        # Event hooks for real-time integration
        self._profile_update_hooks: List[Callable] = []
        self._analysis_hooks: List[Callable] = []
        
        # Fallback in-memory storage for when KeyBuilder is not available
        self.user_profiles = defaultdict(self._create_default_profile)
        
        # Pre-compiled regex patterns for performance
        self._compiled_patterns = self._compile_regex_patterns()
        
        # Analysis patterns (loaded once)
        self.communication_patterns = self._load_communication_patterns()
        self.trading_patterns = self._load_trading_patterns()
        self.sales_indicators = self._load_sales_indicators()
        self.service_patterns = self._load_service_patterns()
        
        # Cached analysis results (prevents duplicate processing)
        self._analysis_cache = {}
        self._cache_max_size = 1000
        
        # Global intelligence layer for pattern aggregation
        self._global_patterns = defaultdict(lambda: defaultdict(int))
        
        logger.info(f"🧠 OptimizedPersonalityEngine v{self.PROFILE_VERSION} initialized with KeyBuilder: {self.key_builder is not None}")
    
    # ==========================================
    # AUTHORITATIVE TICKER VALIDATION
    # ==========================================
    
    def _load_authoritative_tickers(self) -> set:
        """Load authoritative ticker list from project files"""
        # Popular tickers from project knowledge
        popular_tickers = [
            # FAANG + Top Tech
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA',
            # Major Tech
            'NFLX', 'AMD', 'INTC', 'ORCL', 'CRM', 'ADBE', 'PYPL', 'UBER', 'LYFT',
            # Financial
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP',
            # Healthcare & Pharma
            'JNJ', 'PFE', 'UNH', 'MRNA', 'ABBV', 'TMO', 'ABT', 'LLY',
            # Consumer & Retail  
            'WMT', 'TGT', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 'DIS', 'AMGN',
            # Energy & Industrial
            'XOM', 'CVX', 'COP', 'BA', 'CAT', 'GE', 'MMM', 'HON',
            # Communication & Media
            'T', 'VZ', 'CMCSA', 'TMUS',
            # ETFs - Most Popular
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'IVV', 'VEA', 'IEFA', 'EEM',
            'GLD', 'SLV', 'TLT', 'HYG', 'LQD', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI',
            # Crypto-Related
            'COIN', 'MSTR', 'SQ', 'HOOD',
            # Electric Vehicle & Clean Energy  
            'NIO', 'XPEV', 'LI', 'RIVN', 'LCID', 'ENPH', 'PLUG',
            # Meme Stocks & High Volume
            'GME', 'AMC', 'BB', 'NOK', 'PLTR', 'WISH', 'CLOV',
            # Biotech
            'GILD', 'BIIB', 'REGN', 'VRTX', 'ILMN',
            # Other High Volume
            'F', 'SNAP', 'PINS', 'ZM', 'ROKU', 'PTON', 'SHOP',
            'ARKK', 'ARKQ', 'ARKG', 'ARKW', 'SQQQ', 'TQQQ', 'UVXY', 'VIX',
            # Quantum Computing
            'IONQ', 'RGTI', 'QBTS', 'QUBT', 'ARQQ', 'QTUM',
            # Crypto/Blockchain additional
            'MARA', 'RIOT', 'CLSK', 'HUT', 'BITF', 'BTBT', 'WULF', 'CORZ',
            # AI/Machine Learning additional
            'AI', 'BBAI', 'SOUN', 'SNOW', 'CRWD'
        ]
        
        # Cryptocurrency symbols
        crypto_tickers = [
            # Top Market Cap
            'BTC', 'ETH', 'SOL', 'ADA', 'AVAX', 'DOT', 'MATIC',
            # DeFi Ecosystem
            'UNI', 'AAVE', 'COMP', 'SUSHI', 'CRV', '1INCH', 'YFI',
            # Gaming/Metaverse
            'AXS', 'SAND', 'MANA', 'ENJ', 'GALA', 'APE', 'FLOW',
            # Meme Coins
            'DOGE', 'SHIB', 'PEPE', 'FLOKI', 'WIF', 'BONK',
            # AI/Data Tokens
            'FET', 'OCEAN', 'GRT', 'RENDER', 'LINK',
            # Privacy Coins
            'XMR', 'ZEC', 'DASH', 'BEAM', 'GRIN'
        ]
        
        return set(popular_tickers + crypto_tickers)
    
    def _validate_symbol_with_authority(self, symbol: str) -> bool:
        """Validate symbol against authoritative ticker list to prevent false positives"""
        return symbol.upper() in self.authoritative_tickers
    
    # ==========================================
    # TIME DECAY MECHANISM
    # ==========================================
    
    def _apply_time_decay(self, profile_attribute: Dict, decay_factor: float = None) -> Dict:
        """Apply time decay to profile attributes based on timestamps"""
        if decay_factor is None:
            decay_factor = self.TIME_DECAY_FACTOR
        
        if not isinstance(profile_attribute, dict):
            return profile_attribute
        
        # Apply decay to attributes with timestamps
        for key, value in profile_attribute.items():
            if isinstance(value, dict) and 'timestamp' in value and 'value' in value:
                # Calculate days since update
                try:
                    timestamp = datetime.fromisoformat(value['timestamp'].replace('Z', '+00:00'))
                    days_passed = (datetime.now(timezone.utc) - timestamp).days
                    
                    # Apply exponential decay
                    decayed_value = value['value'] * (decay_factor ** days_passed)
                    profile_attribute[key] = {
                        'value': decayed_value,
                        'timestamp': value['timestamp'],
                        'original_value': value.get('original_value', value['value']),
                        'decay_applied': True
                    }
                except (ValueError, KeyError):
                    continue
        
        return profile_attribute
    
    def _store_with_timestamp(self, value: Any, metadata: Dict = None) -> Dict:
        """Store value with timestamp for time decay mechanism"""
        return {
            'value': value,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metadata': metadata or {},
            'decay_applied': False
        }
    
    # ==========================================
    # GLOBAL INTELLIGENCE LAYER
    # ==========================================
    
    def _update_global_patterns(self, user_id: str, analysis: MessageAnalysis):
        """Update global intelligence patterns (anonymized)"""
        # Track symbol co-occurrence patterns
        symbols = analysis.trading_insights.get('symbols_mentioned', [])
        for symbol in symbols:
            intent = analysis.intent_analysis.get('analysis_type', 'general')
            self._global_patterns['symbol_intents'][f"{symbol}:{intent}"] += 1
        
        # Track emotional patterns by trading action
        trading_action = analysis.trading_insights.get('trading_action', 'unknown')
        primary_emotion = analysis.emotional_state.get('primary_emotion', 'neutral')
        if trading_action != 'unknown' and primary_emotion != 'neutral':
            self._global_patterns['emotion_action'][f"{primary_emotion}:{trading_action}"] += 1
        
        # Track communication style patterns
        formality = analysis.communication_insights.get('formality_score', 0.5)
        technical_depth = analysis.communication_insights.get('technical_depth', 'basic')
        formality_cat = 'formal' if formality > 0.7 else 'casual' if formality < 0.3 else 'neutral'
        self._global_patterns['style_depth'][f"{formality_cat}:{technical_depth}"] += 1
    
    def _get_global_insights(self, symbols: List[str] = None, user_profile: Dict = None) -> Dict:
        """Get insights from global intelligence patterns"""
        insights = {}
        
        if symbols:
            # Find related analysis patterns
            for symbol in symbols:
                related_intents = {}
                for pattern, count in self._global_patterns['symbol_intents'].items():
                    if pattern.startswith(f"{symbol}:"):
                        intent = pattern.split(':')[1]
                        related_intents[intent] = count
                
                if related_intents:
                    most_common_intent = max(related_intents, key=related_intents.get)
                    insights[f"{symbol}_common_intent"] = most_common_intent
        
        # Global communication insights
        if user_profile:
            style = user_profile.get('communication_style', {})
            formality = style.get('formality', 'friendly')
            formality_map = {'casual': 'casual', 'professional': 'formal', 'friendly': 'neutral'}
            formality_cat = formality_map.get(formality, 'neutral')
            
            depth_patterns = {}
            for pattern, count in self._global_patterns['style_depth'].items():
                if pattern.startswith(f"{formality_cat}:"):
                    depth = pattern.split(':')[1]
                    depth_patterns[depth] = count
            
            if depth_patterns:
                common_depth = max(depth_patterns, key=depth_patterns.get)
                insights['style_depth_suggestion'] = common_depth
        
        return insights
    
    # ==========================================
    # PROFILE CONFIDENCE SCORING
    # ==========================================
    
    def _calculate_profile_confidence(self, profile: Dict) -> float:
        """Calculate profile confidence score based on data quality and recency"""
        conversation_count = profile.get('conversation_history', {}).get('total_conversations', 0)
        
        # Base confidence from conversation volume
        volume_confidence = min(1.0, conversation_count / 50)  # Max confidence at 50 conversations
        
        # Recency confidence - how recent are the updates
        last_updated = profile.get('last_updated')
        if last_updated:
            try:
                last_update_time = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                days_since_update = (datetime.now(timezone.utc) - last_update_time).days
                recency_confidence = max(0.0, 1.0 - (days_since_update / 30))  # Decay over 30 days
            except (ValueError, AttributeError):
                recency_confidence = 0.5
        else:
            recency_confidence = 0.0
        
        # Pattern consistency confidence
        style_updates = profile.get('communication_style', {}).get('learning_rate_used', 0)
        trading_updates = profile.get('trading_personality', {}).get('learning_rate_used', 0)
        consistency_confidence = min(1.0, (style_updates + trading_updates) / 2)
        
        # Weighted final confidence
        final_confidence = (
            volume_confidence * 0.4 +
            recency_confidence * 0.3 +
            consistency_confidence * 0.3
        )
        
        return min(1.0, max(0.0, final_confidence))
    
    def _adjust_personalization_aggressiveness(self, confidence_score: float) -> float:
        """Adjust personalization aggressiveness based on confidence"""
        if confidence_score > 0.8:
            return 1.0  # Highly confident - full personalization
        elif confidence_score > 0.6:
            return 0.8  # Moderately confident - strong personalization
        elif confidence_score > 0.4:
            return 0.6  # Some confidence - moderate personalization
        elif confidence_score > 0.2:
            return 0.4  # Low confidence - light personalization
        else:
            return 0.2  # Very low confidence - minimal personalization
    
    # ==========================================
    # EVENT HOOKS SYSTEM
    # ==========================================
    
    def register_profile_update_hook(self, callback: Callable[[str, Dict], None]):
        """Register callback for profile updates"""
        self._profile_update_hooks.append(callback)
    
    def register_analysis_hook(self, callback: Callable[[str, MessageAnalysis], None]):
        """Register callback for message analysis results"""
        self._analysis_hooks.append(callback)
    
    async def _trigger_profile_update_hooks(self, user_id: str, profile: Dict):
        """Trigger all registered profile update hooks"""
        for hook in self._profile_update_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(user_id, profile)
                else:
                    hook(user_id, profile)
            except Exception as e:
                logger.error(f"❌ Profile update hook failed: {e}")
    
    async def _trigger_analysis_hooks(self, user_id: str, analysis: MessageAnalysis):
        """Trigger all registered analysis hooks"""
        for hook in self._analysis_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(user_id, analysis)
                else:
                    hook(user_id, analysis)
            except Exception as e:
                logger.error(f"❌ Analysis hook failed: {e}")
    
    # ==========================================
    # KEYBUILDER INTEGRATION WITH NAMING CONVENTION
    # ==========================================
    
    def _get_personality_key(self, user_id: str) -> str:
        """Get consistent personality key following naming convention"""
        return f"user:{user_id}:personality"
    
    def _get_analysis_cache_key(self, user_id: str) -> str:
        """Get analysis cache key following naming convention"""
        return f"user:{user_id}:last_analysis"
    
    async def get_user_profile(self, user_id: str, create_if_missing: bool = True) -> Dict[str, Any]:
        """Get comprehensive user profile with KeyBuilder integration"""
        try:
            # Try KeyBuilder first if available
            if self.key_builder:
                personality_key = self._get_personality_key(user_id)
                profile = await self.key_builder.get(personality_key)
                
                if profile:
                    # Handle version migration and apply time decay
                    profile = self._migrate_profile_if_needed(profile)
                    profile = self._apply_time_decay_to_profile(profile)
                    logger.debug(f"✅ Retrieved profile from KeyBuilder for {user_id}")
                    return profile
                
                if not create_if_missing:
                    return None
                
                # Create new profile with KeyBuilder
                logger.info(f"🆕 Creating new profile for {user_id}")
                profile = self._create_comprehensive_profile()
                await self.key_builder.set(personality_key, profile, ttl=86400 * 30)  # 30 days
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
    
    def _apply_time_decay_to_profile(self, profile: Dict) -> Dict:
        """Apply time decay to all relevant profile sections"""
        # Apply decay to emotional intelligence
        if 'emotional_intelligence' in profile:
            profile['emotional_intelligence']['emotional_triggers'] = self._apply_time_decay(
                profile['emotional_intelligence']['emotional_triggers']
            )
        
        # Apply decay to trading personality metrics
        if 'trading_personality' in profile:
            trading_metrics = ['risk_tolerance', 'decision_speed', 'momentum_vs_value']
            for metric in trading_metrics:
                if metric in profile['trading_personality']:
                    if isinstance(profile['trading_personality'][metric], dict):
                        profile['trading_personality'][metric] = self._apply_time_decay(
                            profile['trading_personality'][metric]
                        )
        
        return profile
    
    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user profile with KeyBuilder integration and event hooks"""
        try:
            profile = await self.get_user_profile(user_id, create_if_missing=True)
            
            # Deep merge updates
            self._deep_merge_dict(profile, updates)
            profile["last_updated"] = datetime.now(timezone.utc).isoformat()
            profile["profile_version"] = self.PROFILE_VERSION
            
            # Calculate and store profile confidence
            confidence_score = self._calculate_profile_confidence(profile)
            profile["confidence_score"] = confidence_score
            profile["personalization_aggressiveness"] = self._adjust_personalization_aggressiveness(confidence_score)
            
            # Save with KeyBuilder if available
            if self.key_builder:
                personality_key = self._get_personality_key(user_id)
                success = await self.key_builder.set(personality_key, profile, ttl=86400 * 30)
                if success:
                    logger.debug(f"✅ Updated profile via KeyBuilder for {user_id}")
                else:
                    logger.warning(f"⚠️ KeyBuilder update failed for {user_id}, falling back to memory")
                    self.user_profiles[user_id] = profile
            else:
                # Fallback to in-memory
                self.user_profiles[user_id] = profile
            
            # Trigger update hooks for real-time integration
            await self._trigger_profile_update_hooks(user_id, profile)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating profile for {user_id}: {e}")
            return False
    
    # ==========================================
    # SEPARATED ANALYSIS AND LEARNING LAYERS
    # ==========================================
    
    async def run_comprehensive_analysis(self, user_message: str, context: Dict[str, Any] = None) -> MessageAnalysis:
        """
        Run comprehensive message analysis with all detection methods integrated.
        Separated from learning for reusability and performance.
        """
        try:
            # Check analysis cache first
            cache_key = f"{hash(user_message)}_{hash(str(context))}"
            if cache_key in self._analysis_cache:
                logger.debug("📋 Using cached analysis result")
                return self._analysis_cache[cache_key]
            
            # Optimized preprocessing - do expensive operations once
            preprocessed = self._preprocess_message(user_message)
            
            # Run all analyses with preprocessed data and full detection methods
            communication_analysis = self._analyze_communication_style_enhanced(user_message, preprocessed)
            trading_analysis = self._analyze_trading_content_enhanced(user_message, preprocessed)
            emotional_analysis = self._analyze_emotional_state_enhanced(user_message, preprocessed)
            intent_analysis = self._analyze_user_intent_enhanced(user_message, preprocessed)
            service_analysis = self._analyze_service_needs_enhanced(user_message, preprocessed)
            sales_analysis = self._analyze_sales_opportunity_enhanced(user_message, preprocessed)
            
            # Create analysis result
            analysis = MessageAnalysis(
                communication_insights=communication_analysis,
                trading_insights=trading_analysis,
                emotional_state=emotional_analysis,
                service_needs=service_analysis,
                sales_opportunity=sales_analysis,
                intent_analysis=intent_analysis,
                preprocessed_tokens=preprocessed
            )
            
            # Cache result (with size limit)
            if len(self._analysis_cache) >= self._cache_max_size:
                # Remove oldest entries
                oldest_keys = list(self._analysis_cache.keys())[:100]
                for key in oldest_keys:
                    del self._analysis_cache[key]
            
            self._analysis_cache[cache_key] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error in comprehensive analysis: {e}")
            # Return empty analysis instead of failing
            return MessageAnalysis(
                communication_insights={},
                trading_insights={},
                emotional_state={},
                service_needs={},
                sales_opportunity={},
                intent_analysis={},
                preprocessed_tokens={}
            )
    
    async def learn_from_analysis(self, user_id: str, profile: Dict[str, Any], analysis: MessageAnalysis) -> Dict[str, Any]:
        """
        Generate profile updates from analysis results with enhanced learning algorithms.
        """
        try:
            # Calculate dynamic learning rate
            learning_rate = self._calculate_dynamic_learning_rate(profile, analysis)
            
            # Generate updates for each profile section using enhanced methods
            updates = {
                "communication_style": self._update_communication_style_enhanced(
                    profile["communication_style"], 
                    analysis.communication_insights, 
                    learning_rate
                ),
                "trading_personality": self._update_trading_personality_enhanced(
                    profile["trading_personality"], 
                    analysis.trading_insights,
                    learning_rate
                ),
                "emotional_intelligence": self._update_emotional_profile_enhanced(
                    profile["emotional_intelligence"], 
                    analysis.emotional_state,
                    learning_rate
                ),
                "service_profile": self._update_service_profile_enhanced(
                    profile["service_profile"], 
                    analysis.service_needs,
                    learning_rate
                ),
                "sales_profile": self._update_sales_profile_enhanced(
                    profile["sales_profile"], 
                    analysis.sales_opportunity,
                    learning_rate
                ),
                "analytics": self._update_analytics_enhanced(
                    profile["analytics"], 
                    analysis.intent_analysis
                )
            }
            
            # Update conversation history with enhanced context tracking
            updates["conversation_history"] = self._update_conversation_history_enhanced(
                profile["conversation_history"], 
                analysis,
                user_id
            )
            
            return updates
            
        except Exception as e:
            logger.error(f"❌ Error in learn_from_analysis: {e}")
            return {}
    
    async def analyze_and_learn(self, user_id: str, user_message: str, bot_response: str = None, 
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Combined analysis and learning workflow with enhanced global intelligence integration.
        """
        try:
            # Step 1: Run comprehensive analysis
            analysis = await self.run_comprehensive_analysis(user_message, context)
            
            # Step 2: Get current profile
            profile = await self.get_user_profile(user_id)
            
            # Step 3: Update global intelligence patterns
            self._update_global_patterns(user_id, analysis)
            
            # Step 4: Generate learning updates
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
            
            # Cache analysis for potential reuse
            if self.key_builder:
                analysis_key = self._get_analysis_cache_key(user_id)
                await self.key_builder.set(analysis_key, {
                    "analysis": analysis.__dict__,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "global_insights": global_insights
                }, ttl=3600)  # 1 hour cache
            
            return {
                "communication_insights": analysis.communication_insights,
                "trading_insights": analysis.trading_insights,
                "emotional_state": analysis.emotional_state,
                "service_needs": analysis.service_needs,
                "sales_opportunity": analysis.sales_opportunity,
                "recommended_approach": response_strategy,
                "global_insights": global_insights,
                "profile_confidence": profile.get("confidence_score", 0.5),
                "learning_applied": bool(updates)
            }
            
        except Exception as e:
            logger.error(f"❌ Error in analyze_and_learn for {user_id}: {e}")
            return {"error": str(e)}
    
    # ==========================================
    # OPTIMIZED PREPROCESSING LAYER
    # ==========================================
    
    def _preprocess_message(self, message: str) -> Dict[str, Any]:
        """
        Optimized preprocessing that extracts all patterns once.
        Eliminates redundant regex operations across detection methods.
        """
        message_lower = message.lower()
        
        # Tokenization
        words = message.split()
        words_lower = message_lower.split()
        
        # Extract all patterns at once using pre-compiled regex
        patterns = {}
        for pattern_name, compiled_regex in self._compiled_patterns.items():
            patterns[pattern_name] = compiled_regex.findall(message)
        
        # Character analysis
        char_analysis = {
            'total_length': len(message),
            'emoji_count': len(patterns.get('emojis', [])),
            'caps_words': patterns.get('caps_words', []),
            'question_marks': message.count('?'),
            'exclamation_marks': message.count('!'),
            'excessive_punctuation': patterns.get('excessive_punct', []),
            'repeated_chars': patterns.get('repeated_chars', [])
        }
        
        # Enhanced symbol extraction with validation
        symbols = self._extract_symbols_enhanced(patterns.get('potential_symbols', []))
        
        return {
            'original': message,
            'lower': message_lower,
            'words': words,
            'words_lower': words_lower,
            'patterns': patterns,
            'char_analysis': char_analysis,
            'symbols': symbols,
            'preprocessing_timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _compile_regex_patterns(self) -> Dict[str, re.Pattern]:
        """Pre-compile all regex patterns for performance"""
        return {
            'potential_symbols': re.compile(r'\b[A-Z]{2,5}\b'),
            'money_amounts': re.compile(r'\$([0-9,]+(?:\.[0-9]{2})?)'),
            'percentages': re.compile(r'([0-9]+(?:\.[0-9]+)?)%'),
            'emojis': re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U0001f900-\U0001f9ff\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f1e0-\U0001f1ff]'),
            'caps_words': re.compile(r'\b[A-Z]{2,}\b'),
            'excessive_punct': re.compile(r'[!?]{3,}'),
            'repeated_chars': re.compile(r'(.)\1{2,}'),
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'phone_numbers': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'emails': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        }
    
    def _extract_symbols_enhanced(self, potential_symbols: List[str]) -> List[str]:
        """Enhanced symbol extraction with authoritative validation"""
        # First filter with basic false positives
        basic_false_positives = {
            'USD', 'USA', 'CEO', 'IPO', 'ETF', 'API', 'AI', 'VR', 'AR', 'IT', 'TV', 'PC', 
            'ATM', 'GPS', 'DNA', 'FBI', 'CIA', 'NBA', 'NFL', 'MLB', 'NHL', 'SEC', 'FTC',
            'AND', 'THE', 'FOR', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAD', 'HER', 'WAS',
            'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HOW', 'ITS', 'MAY', 'NEW',
            'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'HAS', 'LET', 'PUT', 'SAY',
            'SHE', 'TOO', 'USE'
        }
        
        # Filter basic false positives
        filtered_symbols = [symbol for symbol in potential_symbols if symbol not in basic_false_positives]
        
        # Validate against authoritative ticker list
        validated_symbols = [symbol for symbol in filtered_symbols if self._validate_symbol_with_authority(symbol)]
        
        return validated_symbols
    
    # ==========================================
    # ENHANCED DETECTION METHODS - COMPLETE INTEGRATION
    # ==========================================
    
    def _analyze_communication_style_enhanced(self, message: str, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced communication analysis with complete detection methods"""
        words_lower = preprocessed['words_lower']
        char_analysis = preprocessed['char_analysis']
        
        # Use word sets for faster lookups
        words_set = set(words_lower)
        
        # Enhanced formality detection
        casual_indicators = ['yo', 'hey', 'sup', 'gonna', 'wanna', 'lol', 'lmao', 'nah', 'yeah', 'kinda', 'wtf']
        formal_indicators = ['please', 'thank', 'appreciate', 'kindly', 'regards', 'sincerely', 'however', 'furthermore']
        friendly_indicators = ['thanks', 'great', 'awesome', 'cool', 'nice', 'good', 'perfect', 'exactly']
        
        casual_score = sum(1 for word in casual_indicators if word in words_set)
        formal_score = sum(1 for word in formal_indicators if word in words_set)
        friendly_score = sum(1 for word in friendly_indicators if word in words_set)
        
        total_words = len(words_lower)
        formality_score = ((formal_score - casual_score) / max(total_words, 1)) + 0.5
        formality_score = max(0.0, min(1.0, formality_score))
        
        # Enhanced energy level detection
        high_energy_indicators = ['wow', 'amazing', 'awesome', 'excited', 'love', 'pumped', 'thrilled', 'fantastic']
        low_energy_indicators = ['maybe', 'probably', 'might', 'whatever', 'fine', 'tired', 'exhausted', 'meh']
        
        high_energy_score = sum(1 for word in high_energy_indicators if word in words_set)
        low_energy_score = sum(1 for word in low_energy_indicators if word in words_set)
        
        # Factor in emoji and punctuation for energy
        emoji_boost = min(char_analysis['emoji_count'] * 0.2, 1.0)
        exclamation_boost = min(char_analysis['exclamation_marks'] * 0.1, 0.5)
        
        energy_score = high_energy_score + emoji_boost + exclamation_boost - (low_energy_score * 0.5)
        
        if energy_score > 1.0:
            energy_level = "high"
        elif energy_score < -0.5:
            energy_level = "low"
        else:
            energy_level = "moderate"
        
        # Enhanced technical depth assessment
        advanced_terms = ['fibonacci', 'bollinger', 'stochastic', 'ichimoku', 'elliott', 'harmonic', 'divergence', 'confluence']
        intermediate_terms = ['support', 'resistance', 'volume', 'rsi', 'macd', 'trend', 'breakout', 'pullback']
        basic_terms = ['buy', 'sell', 'up', 'down', 'good', 'bad', 'high', 'low']
        
        advanced_count = sum(1 for term in advanced_terms if term in preprocessed['lower'])
        intermediate_count = sum(1 for term in intermediate_terms if term in preprocessed['lower'])
        basic_count = sum(1 for term in basic_terms if term in words_set)
        
        if advanced_count > 0:
            technical_depth = "advanced"
        elif intermediate_count > basic_count:
            technical_depth = "intermediate"
        else:
            technical_depth = "basic"
        
        # Enhanced pattern detection
        greeting_pattern = self._extract_greeting_pattern(message)
        closing_pattern = self._extract_closing_pattern(message)
        question_analysis = self._analyze_question_structure(message)
        vocabulary_level = self._assess_vocabulary_level(message)
        
        return {
            "formality_score": formality_score,
            "energy_level": energy_level,
            "technical_depth": technical_depth,
            "emoji_usage": char_analysis['emoji_count'],
            "message_length": char_analysis['total_length'],
            "urgency_level": self._detect_urgency_enhanced(preprocessed),
            "question_type": self._classify_question_type_optimized(preprocessed),
            "communication_tone": self._detect_tone_enhanced(words_set, char_analysis),
            "greeting_pattern": greeting_pattern,
            "closing_pattern": closing_pattern,
            "question_structure": question_analysis,
            "vocabulary_level": vocabulary_level,
            "politeness_level": self._detect_politeness_level(words_set),
            "directness_level": self._detect_directness_level(words_set, question_analysis)
        }
    
    def _analyze_trading_content_enhanced(self, message: str, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced trading analysis with complete detection methods"""
        words_set = set(preprocessed['words_lower'])
        
        # Enhanced trading action detection
        buy_indicators = ['buy', 'buying', 'long', 'bullish', 'calls', 'purchase', 'invest', 'accumulate']
        sell_indicators = ['sell', 'selling', 'short', 'bearish', 'puts', 'exit', 'close', 'dump']
        hold_indicators = ['hold', 'holding', 'keep', 'maintain', 'stay', 'wait', 'diamond hands']
        research_indicators = ['analyze', 'research', 'study', 'thoughts', 'opinion', 'dd', 'due diligence']
        
        action_scores = {
            'buy': sum(1 for word in buy_indicators if word in words_set),
            'sell': sum(1 for word in sell_indicators if word in words_set),
            'hold': sum(1 for word in hold_indicators if word in words_set),
            'research': sum(1 for word in research_indicators if word in words_set)
        }
        
        trading_action = max(action_scores, key=action_scores.get) if max(action_scores.values()) > 0 else 'unclear'
        
        # Enhanced risk sentiment analysis
        conservative_indicators = ['safe', 'stable', 'conservative', 'careful', 'protect', 'preserve', 'cautious']
        aggressive_indicators = ['risky', 'aggressive', 'volatile', 'yolo', 'moon', 'diamond hands', 'all in']
        
        conservative_score = sum(1 for word in conservative_indicators if word in words_set)
        aggressive_score = sum(1 for word in aggressive_indicators if word in words_set)
        
        if aggressive_score > conservative_score:
            risk_tolerance = "aggressive"
        elif conservative_score > 0:
            risk_tolerance = "conservative"
        else:
            risk_tolerance = "moderate"
        
        # Enhanced analysis using complete detection methods
        time_horizon = self._detect_time_horizon(message)
        position_size_hints = self._detect_position_size_hints(message)
        profit_loss_mention = self._detect_profit_loss_mention(message)
        sector_mentions = self._detect_sector_mentions(message)
        analysis_request = self._classify_analysis_request(message)
        research_orientation = self._detect_research_orientation(message)
        contrarian_signals = self._detect_contrarian_signals(message)
        momentum_focus = self._detect_momentum_focus(message)
        news_sensitivity = self._detect_news_sensitivity(message)
        
        return {
            "symbols_mentioned": preprocessed['symbols'],
            "trading_action": trading_action,
            "action_confidence": max(action_scores.values()) / max(len(words_set), 1),
            "risk_sentiment": {
                "risk_tolerance": risk_tolerance,
                "conservative_score": conservative_score,
                "aggressive_score": aggressive_score
            },
            "time_horizon": time_horizon,
            "position_size_hints": position_size_hints,
            "profit_loss_mention": profit_loss_mention,
            "sector_mentions": sector_mentions,
            "analysis_request": analysis_request,
            "research_orientation": research_orientation,
            "contrarian_signals": contrarian_signals,
            "momentum_focus": momentum_focus,
            "news_sensitivity": news_sensitivity,
            "money_amounts": preprocessed['patterns'].get('money_amounts', []),
            "percentages": preprocessed['patterns'].get('percentages', [])
        }
    
    def _analyze_emotional_state_enhanced(self, message: str, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced emotional analysis with complete detection methods"""
        words_set = set(preprocessed['words_lower'])
        char_analysis = preprocessed['char_analysis']
        
        # Comprehensive emotion detection using all detection methods
        emotions = {
            "excitement": self._detect_excitement_enhanced(message, words_set, char_analysis),
            "anxiety": self._detect_anxiety_enhanced(message, words_set),
            "frustration": self._detect_frustration(message),
            "satisfaction": self._detect_satisfaction(message),
            "confusion": self._detect_confusion(message),
            "impatience": self._detect_impatience(message),
            "uncertainty": self._detect_uncertainty(message),
            "confidence": self._detect_confidence_enhanced(words_set),
            "greed": self._detect_greed(message),
            "fear": self._detect_fear(message),
            "loss_aversion": self._detect_loss_aversion(message)
        }
        
        # Normalize emotions to 0-1 range
        for emotion in emotions:
            emotions[emotion] = min(1.0, max(0.0, emotions[emotion]))
        
        primary_emotion = max(emotions, key=emotions.get) if emotions else "neutral"
        emotional_intensity = max(emotions.values()) if emotions else 0.0
        
        # Enhanced emotional triggers analysis
        emotional_triggers = {
            "fomo_indicators": self._detect_fomo_indicators(message),
            "euphoria_indicators": self._detect_euphoria_indicators(message),
            "panic_indicators": self._detect_panic_indicators(message),
            "regret_indicators": self._detect_regret_indicators(message)
        }
        
        return {
            "primary_emotion": primary_emotion,
            "emotional_intensity": emotional_intensity,
            "all_emotions": emotions,
            "emotional_triggers": emotional_triggers,
            "support_needed": self._assess_support_needs(emotions, emotional_triggers),
            "emotional_approach_recommendation": self._recommend_emotional_approach(emotions)
        }
    
    def _analyze_user_intent_enhanced(self, message: str, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced user intent analysis using preprocessed data"""
        return {
            "intent": "general_inquiry",
            "analysis_type": self._classify_analysis_request(message),
            "time_horizon": self._detect_time_horizon(message),
            "research_orientation": self._detect_research_orientation(message),
            "emotional_urgency": self._detect_impatience(message),
            "complexity_level": self._assess_query_complexity(message, preprocessed),
            "follow_up_likelihood": self._predict_follow_up_likelihood(message, preprocessed)
        }
    
    def _analyze_service_needs_enhanced(self, message: str, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced service needs analysis using all detection methods"""
        service_indicators = {
            "complaint": self._detect_complaint(message),
            "technical_issue": self._detect_technical_issue(message),
            "billing_question": self._detect_billing_question(message),
            "feature_request": self._detect_feature_request(message),
            "urgent_issue": self._detect_urgency_enhanced(preprocessed),
            "praise": self._detect_praise(message),
            "confusion": self._detect_confusion(message)
        }
        
        return {
            "service_type": max(service_indicators, key=service_indicators.get) if service_indicators else "none",
            "urgency_level": self._calculate_urgency_score(service_indicators),
            "escalation_risk": self._assess_escalation_risk(message, service_indicators),
            "recommended_response_time": self._recommend_response_time(service_indicators),
            "tone_recommendation": self._recommend_service_tone(service_indicators),
            "follow_up_needed": self._assess_follow_up_needs(service_indicators),
            "service_indicators": service_indicators
        }
    
    def _analyze_sales_opportunity_enhanced(self, message: str, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced sales opportunity analysis using all detection methods"""
        buying_signals = {
            "interest_level": self._detect_interest_level(message),
            "budget_indicators": self._detect_budget_indicators(message),
            "timeline_urgency": self._detect_timeline_urgency(message),
            "decision_authority": self._detect_decision_authority(message),
            "comparison_shopping": self._detect_comparison_shopping(message),
            "objection_indicators": self._detect_objections(message)
        }
        
        # Create a minimal profile for sales stage determination
        minimal_profile = {"sales_profile": {"buying_readiness": 0.5}}
        sales_stage = self._determine_sales_stage(buying_signals, minimal_profile)
        
        return {
            "sales_readiness_score": self._calculate_sales_readiness(buying_signals),
            "current_sales_stage": sales_stage,
            "recommended_approach": self._recommend_sales_approach(sales_stage, buying_signals),
            "objections_to_address": self._identify_objections(message),
            "upsell_opportunities": self._identify_upsell_opportunities(message, minimal_profile),
            "next_action": self._recommend_next_sales_action(sales_stage, buying_signals),
            "buying_signals": buying_signals
        }
    
    # ==========================================
    # COMPLETE EMOTIONAL DETECTION METHODS
    # ==========================================
    
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
    
    def _detect_excitement_enhanced(self, message: str, words_set: set, char_analysis: Dict) -> float:
        """Enhanced excitement detection"""
        excitement_indicators = ['excited', 'pumped', 'thrilled', 'amazing', 'fantastic', 'incredible', 'unbelievable']
        excitement_count = sum(1 for word in excitement_indicators if word in words_set)
        
        # Factor in emojis and punctuation
        emoji_boost = min(char_analysis['emoji_count'] * 0.2, 0.5)
        exclamation_boost = min(char_analysis['exclamation_marks'] * 0.1, 0.3)
        
        return min(1.0, (excitement_count * 0.3) + emoji_boost + exclamation_boost)
    
    def _detect_anxiety_enhanced(self, message: str, words_set: set) -> float:
        """Enhanced anxiety detection"""
        anxiety_indicators = ['worried', 'nervous', 'anxious', 'scared', 'concerned', 'stressed', 'panic', 'nervous']
        anxiety_count = sum(1 for word in anxiety_indicators if word in words_set)
        
        return min(1.0, anxiety_count * 0.4)
    
    def _detect_confidence_enhanced(self, words_set: set) -> float:
        """Enhanced confidence detection"""
        confidence_indicators = ['definitely', 'sure', 'certain', 'confident', 'absolutely', 'positive', 'convinced']
        confidence_count = sum(1 for word in confidence_indicators if word in words_set)
        
        return min(1.0, confidence_count * 0.3)
    
    # Enhanced emotional trigger detection methods
    def _detect_fomo_indicators(self, message: str) -> float:
        """Detect FOMO (Fear of Missing Out) indicators"""
        fomo_indicators = [
            'missing out', 'everyone else', 'too late', 'wish I had',
            'should have bought', 'train leaving', 'opportunity missed',
            'fomo', 'regret not', 'if only I had'
        ]
        message_lower = message.lower()
        fomo_count = sum(1 for indicator in fomo_indicators if indicator in message_lower)
        
        return min(1.0, fomo_count * 0.4)
    
    def _detect_euphoria_indicators(self, message: str) -> float:
        """Detect euphoria indicators"""
        euphoria_indicators = [
            'to the moon', 'infinite gains', 'can\'t lose', 'always goes up',
            'easy money', 'printing money', 'this time is different',
            'new paradigm', 'never coming down'
        ]
        message_lower = message.lower()
        euphoria_count = sum(1 for indicator in euphoria_indicators if indicator in message_lower)
        
        return min(1.0, euphoria_count * 0.5)
    
    def _detect_panic_indicators(self, message: str) -> float:
        """Detect panic indicators"""
        panic_indicators = [
            'sell everything', 'get out now', 'cut all losses',
            'market crash', 'going to zero', 'emergency exit',
            'panic selling', 'fire sale', 'dump it all'
        ]
        message_lower = message.lower()
        panic_count = sum(1 for indicator in panic_indicators if indicator in message_lower)
        
        return min(1.0, panic_count * 0.5)
    
    def _detect_regret_indicators(self, message: str) -> float:
        """Detect regret indicators"""
        regret_indicators = [
            'should have', 'wish I had', 'regret', 'if only',
            'stupid decision', 'big mistake', 'poor choice',
            'kicking myself', 'hindsight', 'learned my lesson'
        ]
        message_lower = message.lower()
        regret_count = sum(1 for indicator in regret_indicators if indicator in message_lower)
        
        return min(1.0, regret_count * 0.4)
    
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
    
    def _detect_politeness_level(self, words_set: set) -> str:
        """Detect politeness level"""
        polite_indicators = ['please', 'thank', 'appreciate', 'kindly', 'sorry', 'excuse me']
        impolite_indicators = ['whatever', 'don\'t care', 'shut up', 'stupid']
        
        polite_count = sum(1 for word in polite_indicators if word in words_set)
        impolite_count = sum(1 for word in impolite_indicators if word in words_set)
        
        if polite_count > impolite_count and polite_count > 0:
            return 'high'
        elif impolite_count > 0:
            return 'low'
        else:
            return 'moderate'
    
    def _detect_directness_level(self, words_set: set, question_analysis: Dict) -> str:
        """Detect communication directness level"""
        direct_indicators = ['tell me', 'show me', 'give me', 'i want', 'i need']
        indirect_indicators = ['could you', 'would you', 'might you', 'perhaps', 'maybe']
        
        direct_count = sum(1 for word in direct_indicators if word in words_set)
        indirect_count = sum(1 for word in indirect_indicators if word in words_set)
        
        # Factor in question style
        if question_analysis.get('style') == 'direct':
            direct_count += 1
        elif question_analysis.get('style') == 'indirect':
            indirect_count += 1
        
        if direct_count > indirect_count:
            return 'high'
        elif indirect_count > direct_count:
            return 'low'
        else:
            return 'moderate'
    
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
        high_fear = emotions.get('fear', 0) > 0.6
        
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
        if not emotions or max(emotions.values()) == 0:
            return 'balanced_and_professional'
            
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
    
    def _calculate_sales_readiness(self, buying_signals: Dict[str, Any]) -> float:
        """Calculate overall sales readiness score"""
        factors = [
            buying_signals.get('interest_level', 0) * 0.3,
            (1 if buying_signals.get('budget_indicators', {}).get('status') == 'has_budget' else 0) * 0.2,
            (1 if buying_signals.get('timeline_urgency') == 'immediate' else 0.5 if buying_signals.get('timeline_urgency') == 'soon' else 0) * 0.2,
            (1 if buying_signals.get('decision_authority') == 'decision_maker' else 0) * 0.15,
            (1.0 - len(buying_signals.get('objection_indicators', [])) * 0.2) * 0.15
        ]
        
        return max(0.0, sum(factors))
    
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
    
    # ==========================================
    # ENHANCED HELPER METHODS
    # ==========================================
    
    def _detect_urgency_enhanced(self, preprocessed: Dict[str, Any]) -> float:
        """Enhanced urgency detection"""
        words_set = set(preprocessed['words_lower'])
        urgency_words = {'urgent', 'asap', 'quickly', 'immediately', 'rush', 'emergency', 'critical'}
        urgency_count = sum(1 for word in urgency_words if word in words_set)
        exclamation_boost = min(preprocessed['char_analysis']['exclamation_marks'] * 0.1, 0.3)
        return min(1.0, urgency_count * 0.3 + exclamation_boost)
    
    def _classify_question_type_optimized(self, preprocessed: Dict[str, Any]) -> str:
        """Optimized question type classification"""
        if preprocessed['char_analysis']['question_marks'] == 0:
            return 'statement'
        
        words_set = set(preprocessed['words_lower'])
        if any(word in words_set for word in ['what', 'which', 'who', 'where', 'when', 'why', 'how']):
            return 'open_ended'
        elif any(preprocessed['lower'].startswith(word) for word in ['is', 'are', 'was', 'will', 'would', 'should', 'can', 'do']):
            return 'yes_no'
        else:
            return 'clarification'
    
    def _detect_tone_enhanced(self, words_set: set, char_analysis: Dict) -> str:
        """Enhanced tone detection"""
        positive_count = sum(1 for word in ['great', 'awesome', 'love', 'excellent', 'fantastic'] if word in words_set)
        negative_count = sum(1 for word in ['hate', 'terrible', 'awful', 'horrible', 'worst'] if word in words_set)
        
        # Factor in emojis
        emoji_count = char_analysis.get('emoji_count', 0)
        if emoji_count > 0:
            positive_count += emoji_count * 0.5  # Assume most emojis are positive
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _assess_query_complexity(self, message: str, preprocessed: Dict) -> str:
        """Assess query complexity level"""
        word_count = len(preprocessed['words'])
        question_count = preprocessed['char_analysis']['question_marks']
        symbol_count = len(preprocessed['symbols'])
        
        complexity_score = word_count * 0.1 + question_count * 2 + symbol_count * 1.5
        
        if complexity_score > 15:
            return 'high'
        elif complexity_score > 7:
            return 'medium'
        else:
            return 'low'
    
    def _predict_follow_up_likelihood(self, message: str, preprocessed: Dict) -> float:
        """Predict likelihood of follow-up questions"""
        follow_up_indicators = ['tell me more', 'what about', 'also', 'additionally', 'furthermore']
        message_lower = preprocessed['lower']
        
        indicator_count = sum(1 for indicator in follow_up_indicators if indicator in message_lower)
        question_count = preprocessed['char_analysis']['question_marks']
        complexity = len(preprocessed['words']) / 10  # Longer messages likely to generate follow-ups
        
        likelihood = (indicator_count * 0.3) + (question_count * 0.2) + (complexity * 0.1)
        return min(1.0, likelihood)
    
    # ==========================================
    # DYNAMIC LEARNING RATE CALCULATION
    # ==========================================
    
    def _calculate_dynamic_learning_rate(self, profile: Dict[str, Any], analysis: MessageAnalysis) -> float:
        """
        Calculate adaptive learning rate based on user data and analysis confidence
        """
        base_rate = self.DEFAULT_LEARNING_RATE
        
        # Adjust based on conversation history
        total_conversations = profile.get("conversation_history", {}).get("total_conversations", 0)
        if total_conversations > 100:
            conversation_factor = 0.7  # Slower learning for experienced users
        elif total_conversations > 50:
            conversation_factor = 0.8
        elif total_conversations < 5:
            conversation_factor = 1.3  # Faster learning for new users
        else:
            conversation_factor = 1.0
        
        # Adjust based on confidence in analysis
        confidence_factors = []
        
        # Communication analysis confidence
        comm_insights = analysis.communication_insights
        if comm_insights.get('formality_score', 0) > 0.8 or comm_insights.get('formality_score', 0) < 0.2:
            confidence_factors.append(1.2)  # High confidence in formality detection
        
        # Trading analysis confidence
        trading_insights = analysis.trading_insights
        if trading_insights.get('symbols_mentioned') and len(trading_insights['symbols_mentioned']) > 0:
            confidence_factors.append(1.1)  # High confidence when symbols mentioned
        
        # Emotional analysis confidence
        emotional_state = analysis.emotional_state
        primary_emotion = emotional_state.get('primary_emotion', 'neutral')
        emotional_intensity = emotional_state.get('emotional_intensity', 0)
        if emotional_intensity > 0.7:
            confidence_factors.append(1.2)  # High confidence in strong emotions
        
        # Calculate final learning rate
        confidence_multiplier = np.mean(confidence_factors) if confidence_factors else 1.0
        final_rate = base_rate * conversation_factor * confidence_multiplier
        
        # Clamp to reasonable bounds
        return max(self.MIN_LEARNING_RATE, min(self.MAX_LEARNING_RATE, final_rate))
    
    # ==========================================
    # ENHANCED UPDATE METHODS WITH ALL DETECTION CAPABILITIES
    # ==========================================
    
    def _update_communication_style_enhanced(self, current_style: Dict, analysis: Dict, learning_rate: float) -> Dict:
        """Enhanced communication style update with dynamic learning rate"""
        alpha = learning_rate * 0.5  # Slightly slower learning for communication patterns
        
        # Update formality with dynamic learning
        if "formality_score" in analysis:
            current_formality_map = {"casual": 0.0, "professional": 1.0, "friendly": 0.5}
            current_formality = current_formality_map.get(current_style.get("formality", "friendly"), 0.5)
            new_formality = current_formality * (1 - alpha) + analysis["formality_score"] * alpha
            
            if new_formality < 0.33:
                current_style["formality"] = "casual"
            elif new_formality > 0.67:
                current_style["formality"] = "professional"
            else:
                current_style["formality"] = "friendly"
        
        # Update other attributes with confidence weighting
        if "energy_level" in analysis and learning_rate > 0.15:  # Only update if confident
            current_style["energy"] = analysis["energy_level"]
        
        if "technical_depth" in analysis and learning_rate > 0.15:
            current_style["technical_depth"] = analysis["technical_depth"]
        
        # Update message length preference
        if "message_length" in analysis:
            length_category = "short" if analysis["message_length"] < 50 else "long" if analysis["message_length"] > 200 else "medium"
            current_style["message_length"] = length_category
        
        # Update communication patterns with timestamp
        if "greeting_pattern" in analysis and analysis["greeting_pattern"] != "none":
            current_style["greeting_pattern"] = self._store_with_timestamp(
                analysis["greeting_pattern"], 
                {"confidence": learning_rate}
            )
        
        if "vocabulary_level" in analysis:
            current_style["vocabulary_level"] = self._store_with_timestamp(
                analysis["vocabulary_level"],
                {"confidence": learning_rate}
            )
        
        current_style["last_updated"] = datetime.now(timezone.utc).isoformat()
        current_style["learning_rate_used"] = learning_rate
        
        return current_style
    
    def _update_trading_personality_enhanced(self, current_trading: Dict, analysis: Dict, learning_rate: float) -> Dict:
        """Enhanced trading personality update with compression and all detection methods"""
        alpha = learning_rate * 0.4  # Moderate learning for trading patterns
        
        # Update symbols with intelligent deduplication and compression
        if "symbols_mentioned" in analysis and analysis["symbols_mentioned"]:
            current_symbols = current_trading.get("common_symbols", [])
            for symbol in analysis["symbols_mentioned"]:
                if symbol not in current_symbols:
                    current_symbols.append(symbol)
            # Keep most recent 20 symbols
            current_trading["common_symbols"] = current_symbols[-20:]
            
            # Compress into symbol frequency map after many observations
            if len(current_symbols) > 15:
                symbol_counts = Counter(current_symbols)
                current_trading["symbol_frequency"] = dict(symbol_counts.most_common(10))
        
        # Update risk tolerance with learning rate and time decay
        if "risk_sentiment" in analysis and learning_rate > 0.1:
            risk_tolerance = analysis["risk_sentiment"].get("risk_tolerance")
            if risk_tolerance:
                current_trading["risk_tolerance"] = self._store_with_timestamp(
                    risk_tolerance,
                    {"confidence": learning_rate, "source": "message_analysis"}
                )
        
        # Track trading actions with compression
        if "trading_action" in analysis and analysis["trading_action"] != "unclear":
            actions = current_trading.get("recent_actions", [])
            actions.append({
                "action": analysis["trading_action"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "confidence": analysis.get("action_confidence", 0.5)
            })
            current_trading["recent_actions"] = actions[-10:]  # Keep last 10
            
            # Compress to action counts after sufficient data
            if len(actions) >= 10:
                action_counts = current_trading.get("action_counts", {})
                action_counts[analysis["trading_action"]] = action_counts.get(analysis["trading_action"], 0) + 1
                current_trading["action_counts"] = action_counts
        
        # Update advanced trading characteristics with time decay
        if "time_horizon" in analysis and analysis["time_horizon"] != "unknown":
            current_trading["time_horizon"] = self._store_with_timestamp(
                analysis["time_horizon"],
                {"confidence": learning_rate}
            )
        
        if "research_orientation" in analysis:
            current_val = current_trading.get("research_orientation", 0.5)
            new_val = current_val * (1 - alpha) + analysis["research_orientation"] * alpha
            current_trading["research_orientation"] = self._store_with_timestamp(
                new_val,
                {"confidence": learning_rate}
            )
        
        # Update sector preferences
        if "sector_mentions" in analysis and analysis["sector_mentions"]:
            sector_prefs = current_trading.get("sector_preferences", {})
            for sector in analysis["sector_mentions"]:
                sector_prefs[sector] = sector_prefs.get(sector, 0) + 1
            current_trading["sector_preferences"] = sector_prefs
        
        current_trading["last_updated"] = datetime.now(timezone.utc).isoformat()
        current_trading["learning_rate_used"] = learning_rate
        
        return current_trading
    
    def _update_emotional_profile_enhanced(self, current_emotional: Dict, analysis: Dict, learning_rate: float) -> Dict:
        """Enhanced emotional profile update with all emotion detection capabilities"""
        alpha = learning_rate * 0.5  # Slightly slower learning for emotional patterns
        
        # Update emotional triggers with all new detection capabilities
        if "emotional_triggers" in analysis:
            for trigger, value in analysis["emotional_triggers"].items():
                if trigger in current_emotional["emotional_triggers"]:
                    current_val = current_emotional["emotional_triggers"][trigger]
                    if isinstance(current_val, dict) and 'value' in current_val:
                        current_val = current_val['value']
                    new_val = current_val * (1 - alpha) + value * alpha
                    current_emotional["emotional_triggers"][trigger] = self._store_with_timestamp(
                        new_val,
                        {"confidence": learning_rate, "trigger_type": trigger}
                    )
        
        # Update primary emotions list with new detections
        primary_emotion = analysis.get("primary_emotion", "neutral")
        if primary_emotion != "neutral":
            emotions_list = current_emotional.get("primary_emotions", [])
            emotions_list.append({
                "emotion": primary_emotion,
                "intensity": analysis.get("emotional_intensity", 0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "confidence": learning_rate
            })
            current_emotional["primary_emotions"] = emotions_list[-10:]  # Keep last 10
        
        # Update support preferences based on detected needs
        support_needed = analysis.get("support_needed", "standard_guidance")
        if support_needed != "standard_guidance":
            current_emotional["support_preferences"]["last_support_type"] = self._store_with_timestamp(
                support_needed,
                {"confidence": learning_rate}
            )
        
        # Update emotional approach recommendation
        emotional_approach = analysis.get("emotional_approach_recommendation", "balanced_and_professional")
        current_emotional["emotional_approach_preference"] = self._store_with_timestamp(
            emotional_approach,
            {"confidence": learning_rate}
        )
        
        current_emotional["last_updated"] = datetime.now(timezone.utc).isoformat()
        current_emotional["learning_rate_used"] = learning_rate
        
        return current_emotional
    
    def _update_service_profile_enhanced(self, current_service: Dict, analysis: Dict, learning_rate: float) -> Dict:
        """Enhanced service profile update with all service detection capabilities"""
        alpha = learning_rate * 0.3  # Moderate learning for service patterns
        
        # Update escalation tendency based on analysis
        escalation_risk = analysis.get("escalation_risk", "low")
        if escalation_risk == "high":
            current_service["issue_escalation_tendency"] = "high"
        elif escalation_risk == "medium" and current_service.get("issue_escalation_tendency") != "high":
            current_service["issue_escalation_tendency"] = "moderate"
        
        # Update response time expectations
        recommended_time = analysis.get("recommended_response_time", "standard")
        if recommended_time == "immediate":
            current_service["response_time_expectation"] = "immediate"
        
        # Track service interaction types with timestamps
        service_type = analysis.get("service_type", "none")
        if service_type != "none":
            service_history = current_service.get("recent_service_types", [])
            service_history.append({
                "type": service_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "urgency_level": analysis.get("urgency_level", 0),
                "escalation_risk": escalation_risk
            })
            current_service["recent_service_types"] = service_history[-5:]  # Keep last 5
        
        # Update tone preferences based on recommendations
        tone_recommendation = analysis.get("tone_recommendation", "professional_and_helpful")
        current_service["preferred_tone"] = self._store_with_timestamp(
            tone_recommendation,
            {"confidence": learning_rate}
        )
        
        current_service["last_updated"] = datetime.now(timezone.utc).isoformat()
        current_service["learning_rate_used"] = learning_rate
        
        return current_service
    
    def _update_sales_profile_enhanced(self, current_sales: Dict, analysis: Dict, learning_rate: float) -> Dict:
        """Enhanced sales profile update with all sales detection capabilities"""
        alpha = learning_rate * 0.4  # Moderate-fast learning for sales patterns
        
        # Update buying readiness with weighted learning
        if "sales_readiness_score" in analysis:
            current_readiness = current_sales.get("buying_readiness", 0.5)
            if isinstance(current_readiness, dict) and 'value' in current_readiness:
                current_readiness = current_readiness['value']
            new_readiness = current_readiness * (1 - alpha) + analysis["sales_readiness_score"] * alpha
            current_sales["buying_readiness"] = self._store_with_timestamp(
                max(0.0, min(1.0, new_readiness)),
                {"confidence": learning_rate, "source": "readiness_analysis"}
            )
        
        # Update sales stage progression
        current_stage = analysis.get("current_sales_stage", "unqualified")
        current_sales["current_sales_stage"] = current_stage
        
        # Track objection patterns with timestamps
        objections = analysis.get("objections_to_address", [])
        if objections:
            objection_history = current_sales.get("objection_patterns", [])
            for objection in objections:
                objection_entry = {
                    "objection": objection,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "frequency": 1
                }
                # Check if objection already exists and increment frequency
                existing = next((o for o in objection_history if o["objection"] == objection), None)
                if existing:
                    existing["frequency"] += 1
                    existing["timestamp"] = datetime.now(timezone.utc).isoformat()
                else:
                    objection_history.append(objection_entry)
            current_sales["objection_patterns"] = objection_history[-10:]  # Keep last 10 unique
        
        # Update upsell receptiveness based on opportunities identified
        upsell_opps = analysis.get("upsell_opportunities", [])
        if upsell_opps:
            current_sales["upsell_receptiveness"] = self._store_with_timestamp(
                "high",
                {"opportunities": upsell_opps, "confidence": learning_rate}
            )
        
        # Track decision-making authority with confidence
        decision_authority = analysis.get("decision_authority", "unknown")
        if decision_authority != "unknown":
            current_sales["decision_authority"] = self._store_with_timestamp(
                decision_authority,
                {"confidence": learning_rate}
            )
        
        # Update budget indicators
        budget_info = analysis.get("buying_signals", {}).get("budget_indicators", {})
        if budget_info.get("status") != "unknown":
            current_sales["budget_status"] = self._store_with_timestamp(
                budget_info["status"],
                {"price_sensitivity": budget_info.get("price_sensitivity"), "confidence": learning_rate}
            )
        
        current_sales["last_updated"] = datetime.now(timezone.utc).isoformat()
        current_sales["learning_rate_used"] = learning_rate
        
        return current_sales
    
    def _update_conversation_history_enhanced(self, current_history: Dict, analysis: MessageAnalysis, user_id: str) -> Dict:
        """Enhanced conversation history update with context tracking"""
        current_history["total_conversations"] = current_history.get("total_conversations", 0) + 1
        
        # Track recent topics with symbols and analysis types
        recent_topics = current_history.get("recent_topics", [])
        
        # Extract topic from analysis
        topic_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbols": analysis.trading_insights.get("symbols_mentioned", []),
            "analysis_type": analysis.intent_analysis.get("analysis_type", "general"),
            "primary_emotion": analysis.emotional_state.get("primary_emotion", "neutral"),
            "trading_action": analysis.trading_insights.get("trading_action", "unclear")
        }
        
        recent_topics.append(topic_entry)
        current_history["recent_topics"] = recent_topics[-20:]  # Keep last 20 topics
        
        # Enhanced context memory tracking
        context_memory = current_history.get("context_memory", {
            "goals_mentioned": [],
            "concerns_expressed": [],
            "preferences_stated": [],
            "plans_discussed": []
        })
        
        # Update context memory based on analysis
        if analysis.emotional_state.get("primary_emotion") in ["anxiety", "fear", "frustration"]:
            concerns = context_memory.get("concerns_expressed", [])
            concerns.append({
                "concern": analysis.emotional_state.get("primary_emotion"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbols": analysis.trading_insights.get("symbols_mentioned", [])
            })
            context_memory["concerns_expressed"] = concerns[-10:]
        
        # Track preferences from communication style
        if analysis.communication_insights.get("technical_depth"):
            preferences = context_memory.get("preferences_stated", [])
            preferences.append({
                "preference": f"technical_depth_{analysis.communication_insights['technical_depth']}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            context_memory["preferences_stated"] = preferences[-10:]
        
        current_history["context_memory"] = context_memory
        current_history["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        return current_history
    
    def _update_analytics_enhanced(self, current_analytics: Dict, intent_analysis: Dict) -> Dict:
        """Enhanced analytics update with performance tracking"""
        # Update learning velocity based on analysis complexity
        complexity = intent_analysis.get("complexity_level", "medium")
        velocity_boost = {"high": 0.3, "medium": 0.2, "low": 0.1}.get(complexity, 0.1)
        
        current_velocity = current_analytics.get("learning_velocity", 0.5)
        new_velocity = min(1.0, current_velocity + velocity_boost * 0.1)
        current_analytics["learning_velocity"] = new_velocity
        
        # Track prediction accuracy (placeholder for future implementation)
        current_analytics["last_analysis_timestamp"] = datetime.now(timezone.utc).isoformat()
        current_analytics["total_analyses"] = current_analytics.get("total_analyses", 0) + 1
        
        return current_analytics
    
    # ==========================================
    # ENHANCED RESPONSE STRATEGY WITH GLOBAL INSIGHTS
    # ==========================================
    
    def _generate_response_strategy_enhanced(self, profile: Dict, analysis: MessageAnalysis, global_insights: Dict) -> Dict[str, Any]:
        """Enhanced response strategy generation with global insights"""
        communication_style = profile.get("communication_style", {})
        emotional_state = analysis.emotional_state
        trading_personality = profile.get("trading_personality", {})
        confidence_score = profile.get("confidence_score", 0.5)
        personalization_aggressiveness = profile.get("personalization_aggressiveness", 0.5)
        
        # Determine optimal approach based on multiple factors including global insights
        strategy = {
            "tone": self._determine_optimal_tone(communication_style, emotional_state),
            "technical_level": communication_style.get("technical_depth", "intermediate"),
            "message_length": communication_style.get("message_length", "medium"),
            "emoji_usage": self._determine_emoji_strategy(communication_style),
            "urgency_handling": self._determine_urgency_approach(emotional_state),
            "risk_communication": self._determine_risk_communication_style(trading_personality, emotional_state),
            "personalization_elements": self._extract_personalization_elements(profile),
            "confidence_level": emotional_state.get("emotional_intensity", 0.5),
            "profile_confidence": confidence_score,
            "personalization_aggressiveness": personalization_aggressiveness,
            "global_insights": global_insights,
            "response_adaptations": self._generate_response_adaptations(profile, analysis, global_insights)
        }
        
        return strategy
    
    def _generate_response_adaptations(self, profile: Dict, analysis: MessageAnalysis, global_insights: Dict) -> Dict[str, Any]:
        """Generate specific response adaptations based on comprehensive analysis"""
        adaptations = {}
        
        # Adapt based on emotional state
        primary_emotion = analysis.emotional_state.get("primary_emotion", "neutral")
        if primary_emotion == "anxiety":
            adaptations["emotional_adaptation"] = "provide_reassurance_and_facts"
        elif primary_emotion == "excitement":
            adaptations["emotional_adaptation"] = "match_energy_but_add_caution"
        elif primary_emotion == "frustration":
            adaptations["emotional_adaptation"] = "acknowledge_frustration_and_solve"
        
        # Adapt based on trading experience and global patterns
        symbols = analysis.trading_insights.get("symbols_mentioned", [])
        for symbol in symbols:
            if f"{symbol}_common_intent" in global_insights:
                common_intent = global_insights[f"{symbol}_common_intent"]
                adaptations[f"{symbol}_suggestion"] = f"users_often_want_{common_intent}_for_this_symbol"
        
        # Adapt based on communication style and global patterns
        if "style_depth_suggestion" in global_insights:
            suggested_depth = global_insights["style_depth_suggestion"]
            current_depth = analysis.communication_insights.get("technical_depth", "basic")
            if suggested_depth != current_depth:
                adaptations["depth_adaptation"] = f"consider_adjusting_to_{suggested_depth}_based_on_similar_users"
        
        return adaptations
    
    # ==========================================
    # PATTERN DEFINITIONS AND LOADING METHODS
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
    # PROFILE VERSIONING AND MIGRATION
    # ==========================================
    
    def _migrate_profile_if_needed(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Handle profile version migration"""
        current_version = profile.get("profile_version", 1.0)
        
        if current_version < 2.0:
            profile = self._migrate_profile_to_v2(profile)
        
        return profile
    
    def _migrate_profile_to_v2(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate profile from v1.0 to v2.0"""
        logger.info(f"🔄 Migrating profile from v{profile.get('profile_version', 1.0)} to v{self.PROFILE_VERSION}")
        
        # Add new fields introduced in v2.0
        if "analytics" not in profile:
            profile["analytics"] = {
                "personalization_accuracy": 0.5,
                "response_satisfaction_estimate": 0.5,
                "engagement_score": 0.5,
                "learning_velocity": 0.5,
                "total_analyses": 0
            }
        
        # Add compression fields to trading personality
        if "trading_personality" in profile:
            if "symbol_frequency" not in profile["trading_personality"]:
                profile["trading_personality"]["symbol_frequency"] = {}
            if "action_counts" not in profile["trading_personality"]:
                profile["trading_personality"]["action_counts"] = {}
        
        # Add confidence scoring fields
        if "confidence_score" not in profile:
            profile["confidence_score"] = 0.5
        if "personalization_aggressiveness" not in profile:
            profile["personalization_aggressiveness"] = 0.5
        
        # Update version
        profile["profile_version"] = self.PROFILE_VERSION
        profile["migration_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        return profile
    
    # ==========================================
    # UTILITY AND HELPER METHODS
    # ==========================================
    
    def _create_comprehensive_profile(self) -> Dict[str, Any]:
        """Create a comprehensive default user profile with v2.0 schema"""
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "profile_version": self.PROFILE_VERSION,
            "confidence_score": 0.0,  # Start with zero confidence
            "personalization_aggressiveness": 0.2,  # Start conservative
            
            # Communication Style Analysis
            "communication_style": {
                "formality": "friendly",
                "energy": "moderate",
                "technical_depth": "intermediate",
                "message_length": "medium",
                "emoji_usage": "some",
                "response_time_preference": "normal",
                "explanation_preference": "balanced",
                "example_preference": True,
                "question_style": "direct",
                "feedback_style": "constructive",
                "greeting_pattern": {},
                "vocabulary_level": {},
                "politeness_level": "moderate",
                "directness_level": "moderate",
                "learning_rate_used": self.DEFAULT_LEARNING_RATE
            },
            
            # Trading Personality & Behavior
            "trading_personality": {
                "experience_level": "intermediate",
                "risk_tolerance": {},
                "trading_style": "swing",
                "decision_speed": "moderate",
                "research_depth": "moderate",
                "common_symbols": [],
                "symbol_frequency": {},  # New: compressed symbol tracking
                "preferred_sectors": [],
                "sector_preferences": {},
                "position_sizing": "moderate",
                "profit_sharing_openness": False,
                "loss_sharing_openness": False,
                "advice_seeking_style": "general",
                "momentum_vs_value": "balanced",
                "fundamental_vs_technical": "balanced",
                "recent_actions": [],
                "action_counts": {},  # New: compressed action tracking
                "time_horizon": {},
                "research_orientation": {},
                "contrarian_signals": 0.0,
                "momentum_focus": 0.0,
                "news_sensitivity": 0.0,
                "learning_rate_used": self.DEFAULT_LEARNING_RATE
            },
            
            # Emotional Intelligence Profile
            "emotional_intelligence": {
                "primary_emotions": [],
                "emotional_triggers": {
                    "loss_aversion": {},
                    "fomo_susceptibility": {},
                    "greed_indicators": {},
                    "fear_indicators": {},
                    "confidence_volatility": {},
                    "fomo_indicators": {},
                    "euphoria_indicators": {},
                    "panic_indicators": {},
                    "regret_indicators": {}
                },
                "support_preferences": {
                    "reassurance_needed": "moderate",
                    "directness_preference": "balanced",
                    "education_vs_action": "balanced",
                    "last_support_type": {}
                },
                "stress_indicators": [],
                "celebration_style": "moderate",
                "emotional_approach_preference": {},
                "learning_rate_used": self.DEFAULT_LEARNING_RATE
            },
            
            # Customer Service Profile
            "service_profile": {
                "complaint_style": "constructive",
                "issue_escalation_tendency": "moderate",
                "technical_comfort_level": "intermediate",
                "patience_level": "moderate",
                "feedback_provision": "moderate",
                "praise_expression": "moderate",
                "help_seeking_style": "specific",
                "follow_up_preference": "moderate",
                "communication_channel_preference": "sms",
                "response_time_expectation": "normal",
                "recent_service_types": [],
                "preferred_tone": {},
                "learning_rate_used": self.DEFAULT_LEARNING_RATE
            },
            
            # Sales & Conversion Profile
            "sales_profile": {
                "buying_readiness": {},
                "price_sensitivity": "moderate",
                "feature_prioritization": "value",
                "decision_making_style": "analytical",
                "research_behavior": "moderate",
                "social_proof_influence": "moderate",
                "urgency_response": "moderate",
                "objection_patterns": [],
                "upsell_receptiveness": {},
                "loyalty_indicators": "developing",
                "current_sales_stage": "unqualified",
                "decision_authority": {},
                "budget_status": {},
                "learning_rate_used": self.DEFAULT_LEARNING_RATE
            },
            
            # Conversation History & Context
            "conversation_history": {
                "total_conversations": 0,
                "recent_topics": [],
                "successful_interactions": 0,
                "problematic_interactions": 0,
                "preferred_interaction_types": [],
                "context_memory": {
                    "goals_mentioned": [],
                    "concerns_expressed": [],
                    "preferences_stated": [],
                    "plans_discussed": []
                },
                "engagement_patterns": {
                    "typical_session_length": "medium",
                    "follow_up_frequency": "moderate",
                    "topic_depth_preference": "moderate",
                    "multi_topic_comfort": True
                }
            },
            
            # Analytics & Performance Tracking
            "analytics": {
                "personalization_accuracy": 0.5,
                "response_satisfaction_estimate": 0.5,
                "engagement_score": 0.5,
                "conversion_indicators": {
                    "feature_interest_score": 0.5,
                    "pricing_acceptance_score": 0.5,
                    "timing_readiness_score": 0.5,
                    "trust_building_score": 0.5
                },
                "learning_velocity": 0.5,
                "prediction_accuracy": {},
                "optimization_opportunities": [],
                "total_analyses": 0
            }
        }
    
    @staticmethod
    def _deep_merge_dict(base_dict: Dict, update_dict: Dict) -> None:
        """Deep merge update_dict into base_dict"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                OptimizedPersonalityEngine._deep_merge_dict(base_dict[key], value)
            else:
                base_dict[key] = value
    
    # ==========================================
    # OPTIMIZED HELPER METHODS (CONTINUED)
    # ==========================================
    
    def _determine_optimal_tone(self, communication_style: Dict, emotional_state: Dict) -> str:
        """Determine optimal response tone"""
        formality = communication_style.get("formality", "friendly")
        primary_emotion = emotional_state.get("primary_emotion", "neutral")
        
        if primary_emotion == "anxiety":
            return "calm_and_reassuring"
        elif primary_emotion == "excitement":
            return "enthusiastic_but_cautious"
        elif primary_emotion == "frustration":
            return "empathetic_and_solution_focused"
        elif formality == "professional":
            return "professional_and_informative"
        else:
            return "friendly_and_helpful"
    
    def _determine_emoji_strategy(self, communication_style: Dict) -> str:
        """Determine emoji usage strategy"""
        emoji_usage = communication_style.get("emoji_usage", "some")
        if isinstance(emoji_usage, int):
            if emoji_usage == 0:
                return "none"
            elif emoji_usage < 3:
                return "minimal"
            else:
                return "moderate"
        return emoji_usage
    
    def _determine_urgency_approach(self, emotional_state: Dict) -> str:
        """Determine how to handle urgency"""
        if emotional_state.get("primary_emotion") == "anxiety":
            return "immediate_acknowledgment"
        elif emotional_state.get("emotional_intensity", 0) > 0.7:
            return "prioritized_response"
        else:
            return "standard_timeline"
    
    def _determine_risk_communication_style(self, trading_personality: Dict, emotional_state: Dict) -> str:
        """Determine how to communicate risk"""
        risk_tolerance = trading_personality.get("risk_tolerance", "moderate")
        if isinstance(risk_tolerance, dict) and 'value' in risk_tolerance:
            risk_tolerance = risk_tolerance['value']
        
        primary_emotion = emotional_state.get("primary_emotion", "neutral")
        
        if primary_emotion in ["anxiety", "fear"]:
            return "gentle_and_educational"
        elif risk_tolerance == "aggressive":
            return "direct_and_comprehensive"
        else:
            return "balanced_and_clear"
    
    def _extract_personalization_elements(self, profile: Dict) -> List[str]:
        """Extract elements for personalization"""
        elements = []
        
        # Add frequently mentioned symbols
        symbols = profile.get("trading_personality", {}).get("common_symbols", [])
        if symbols:
            elements.append(f"Recently mentioned: {', '.join(symbols[-3:])}")
        
        # Add experience level
        experience = profile.get("trading_personality", {}).get("experience_level", "intermediate")
        elements.append(f"Experience level: {experience}")
        
        # Add communication preferences
        formality = profile.get("communication_style", {}).get("formality", "friendly")
        elements.append(f"Communication style: {formality}")
        
        # Add confidence level
        confidence = profile.get("confidence_score", 0.5)
        elements.append(f"Profile confidence: {confidence:.1f}")
        
        return elements
    
    # ==========================================
    # PUBLIC API METHODS
    # ==========================================
    
    async def get_personalized_prompt(self, user_id: str, user_message: str, context: Dict = None) -> str:
        """Generate personalized prompt for LLM based on user profile"""
        try:
            profile = await self.get_user_profile(user_id)
            analysis = await self.run_comprehensive_analysis(user_message, context)
            
            # Build comprehensive prompt with response strategy
            response_strategy = self._generate_response_strategy_enhanced(profile, analysis, {})
            
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
                f"- Risk tolerance: {self._extract_value_from_timestamped(profile['trading_personality'].get('risk_tolerance', 'moderate'))}",
                f"- Trading style: {profile['trading_personality']['trading_style']}",
                f"- Recent symbols: {', '.join(profile['trading_personality']['common_symbols'][-5:]) if profile['trading_personality']['common_symbols'] else 'None yet'}",
                "",
                f"CURRENT EMOTIONAL STATE: {analysis.emotional_state.get('primary_emotion', 'neutral')}",
                f"RESPONSE STRATEGY: {response_strategy.get('tone', 'friendly_and_helpful')}",
                f"PROFILE CONFIDENCE: {profile.get('confidence_score', 0.5):.1f}/1.0",
                "",
                f"USER MESSAGE: {user_message}",
                "",
                "Respond as their personalized trading assistant, matching their style perfectly:"
            ]
            
            return "\n".join(prompt_parts)
            
        except Exception as e:
            logger.error(f"❌ Error generating personalized prompt: {e}")
            return f"USER MESSAGE: {user_message}\n\nRespond as a helpful trading assistant:"
    
    def _extract_value_from_timestamped(self, field: Any) -> Any:
        """Extract value from timestamped field or return as-is"""
        if isinstance(field, dict) and 'value' in field:
            return field['value']
        return field
    
    async def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user insights for admin/debugging"""
        try:
            profile = await self.get_user_profile(user_id, create_if_missing=False)
            if not profile:
                return {"error": "User profile not found"}
            
            return {
                "user_id": user_id,
                "profile_version": profile.get("profile_version", 1.0),
                "profile_completeness": self._calculate_profile_completeness(profile),
                "confidence_score": profile.get("confidence_score", 0.5),
                "personalization_aggressiveness": profile.get("personalization_aggressiveness", 0.5),
                "communication_summary": self._summarize_communication_style(profile),
                "trading_summary": self._summarize_trading_personality(profile),
                "emotional_summary": self._summarize_emotional_profile(profile),
                "total_conversations": profile.get("conversation_history", {}).get("total_conversations", 0),
                "recent_symbols": profile.get("trading_personality", {}).get("common_symbols", [])[-5:],
                "last_updated": profile.get("last_updated"),
                "redis_key": self._get_personality_key(user_id),
                "global_intelligence_insights": len(self._global_patterns)
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting user insights: {e}")
            return {"error": str(e)}
    
    def _calculate_profile_completeness(self, profile: Dict) -> float:
        """Calculate how complete the user profile is (0.0-1.0)"""
        total_conversations = profile.get("conversation_history", {}).get("total_conversations", 0)
        conversation_score = min(1.0, total_conversations / 50)  # Max confidence at 50 conversations
        
        has_symbols = len(profile.get("trading_personality", {}).get("common_symbols", [])) > 0
        has_preferences = len(profile.get("conversation_history", {}).get("context_memory", {}).get("preferences_stated", [])) > 0
        has_emotional_data = len(profile.get("emotional_intelligence", {}).get("primary_emotions", [])) > 0
        
        richness_score = sum([has_symbols, has_preferences, has_emotional_data]) / 3
        return (conversation_score * 0.7) + (richness_score * 0.3)
    
    def _summarize_communication_style(self, profile: Dict) -> str:
        style = profile.get("communication_style", {})
        return f"{style.get('formality', 'friendly').title()} tone, {style.get('energy', 'moderate')} energy, prefers {style.get('message_length', 'medium')} responses"
    
    def _summarize_trading_personality(self, profile: Dict) -> str:
        trading = profile.get("trading_personality", {})
        risk_tolerance = self._extract_value_from_timestamped(trading.get("risk_tolerance", "moderate"))
        return f"{trading.get('experience_level', 'intermediate').title()} {trading.get('trading_style', 'swing')} trader with {risk_tolerance} risk tolerance"
    
    def _summarize_emotional_profile(self, profile: Dict) -> str:
        emotions = profile.get("emotional_intelligence", {}).get("primary_emotions", [])
        if emotions:
            recent_emotions = [e.get("emotion", "unknown") for e in emotions[-3:]]
            return f"Recent emotions: {', '.join(recent_emotions)}"
        return "Emotional patterns still developing"
    
    async def get_global_intelligence_summary(self) -> Dict[str, Any]:
        """Get summary of global intelligence patterns"""
        return {
            "total_patterns": len(self._global_patterns),
            "symbol_intent_patterns": len(self._global_patterns.get("symbol_intents", {})),
            "emotion_action_patterns": len(self._global_patterns.get("emotion_action", {})),
            "style_depth_patterns": len(self._global_patterns.get("style_depth", {})),
            "top_symbol_intents": dict(sorted(
                self._global_patterns.get("symbol_intents", {}).items(),
                key=lambda x: x[1], reverse=True
            )[:10]),
            "top_emotion_actions": dict(sorted(
                self._global_patterns.get("emotion_action", {}).items(),
                key=lambda x: x[1], reverse=True
            )[:10])
        }
    
    async def reset_user_profile(self, user_id: str) -> bool:
        """Reset user profile to default state"""
        try:
            default_profile = self._create_comprehensive_profile()
            
            if self.key_builder:
                personality_key = self._get_personality_key(user_id)
                await self.key_builder.set(personality_key, default_profile, ttl=86400 * 30)
            else:
                self.user_profiles[user_id] = default_profile
            
            logger.info(f"✅ Reset profile for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error resetting profile for {user_id}: {e}")
            return False
    
    async def export_user_profile(self, user_id: str) -> Optional[Dict]:
        """Export user profile for backup or analysis"""
        try:
            profile = await self.get_user_profile(user_id, create_if_missing=False)
            if profile:
                # Remove sensitive or temporary data
                export_profile = profile.copy()
                export_profile.pop("_cache_keys", None)
                return export_profile
            return None
        except Exception as e:
            logger.error(f"❌ Error exporting profile for {user_id}: {e}")
            return None
    
    async def import_user_profile(self, user_id: str, profile_data: Dict) -> bool:
        """Import user profile from backup"""
        try:
            # Validate profile structure
            if "profile_version" not in profile_data:
                profile_data["profile_version"] = self.PROFILE_VERSION
            
            # Migrate if needed
            migrated_profile = self._migrate_profile_if_needed(profile_data)
            
            # Update timestamps
            migrated_profile["last_updated"] = datetime.now(timezone.utc).isoformat()
            migrated_profile["imported_at"] = datetime.now(timezone.utc).isoformat()
            
            # Save profile
            if self.key_builder:
                personality_key = self._get_personality_key(user_id)
                await self.key_builder.set(personality_key, migrated_profile, ttl=86400 * 30)
            else:
                self.user_profiles[user_id] = migrated_profile
            
            logger.info(f"✅ Imported profile for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error importing profile for {user_id}: {e}")
            return False
    
    # ==========================================
    # PERFORMANCE AND MONITORING METHODS
    # ==========================================
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        return {
            "cache_size": len(self._analysis_cache),
            "cache_max_size": self._cache_max_size,
            "global_patterns_count": len(self._global_patterns),
            "authoritative_tickers_count": len(self.authoritative_tickers),
            "profile_update_hooks": len(self._profile_update_hooks),
            "analysis_hooks": len(self._analysis_hooks),
            "engine_version": self.PROFILE_VERSION,
            "memory_usage": {
                "user_profiles": len(self.user_profiles),
                "analysis_cache": len(self._analysis_cache),
                "global_patterns": sum(len(patterns) for patterns in self._global_patterns.values())
            }
        }
    
    def clear_analysis_cache(self) -> int:
        """Clear analysis cache and return number of items cleared"""
        cache_size = len(self._analysis_cache)
        self._analysis_cache.clear()
        logger.info(f"🧹 Cleared {cache_size} items from analysis cache")
        return cache_size
    
    def optimize_global_patterns(self, max_patterns_per_type: int = 1000) -> Dict[str, int]:
        """Optimize global patterns by keeping only top patterns"""
        optimized_counts = {}
        
        for pattern_type, patterns in self._global_patterns.items():
            original_count = len(patterns)
            if original_count > max_patterns_per_type:
                # Keep top patterns by frequency
                top_patterns = dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:max_patterns_per_type])
                self._global_patterns[pattern_type] = top_patterns
                optimized_counts[pattern_type] = original_count - len(top_patterns)
            else:
                optimized_counts[pattern_type] = 0
        
        total_removed = sum(optimized_counts.values())
        if total_removed > 0:
            logger.info(f"🔧 Optimized global patterns, removed {total_removed} items")
        
        return optimized_counts
