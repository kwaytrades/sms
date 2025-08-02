# social/models/social_user.py
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

class PlatformType(Enum):
    TIKTOK = "tiktok"
    TWITTER = "twitter"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"
    FACEBOOK = "facebook"
    REDDIT = "reddit"

class InfluenceLevel(Enum):
    MICRO = "micro"  # <10k followers
    MACRO = "macro"  # 10k-100k followers
    MEGA = "mega"    # 100k+ followers
    UNKNOWN = "unknown"

class EngagementStyle(Enum):
    QUESTION_ASKER = "question_asker"
    ADVICE_GIVER = "advice_giver"
    SKEPTIC = "skeptic"
    SUPPORTER = "supporter"
    LURKER = "lurker"
    TECHNICAL_ANALYST = "technical_analyst"

class ConversionStage(Enum):
    DISCOVERY = "discovery"           # First interaction
    ENGAGEMENT = "engagement"         # Regular interaction
    RECOGNITION = "recognition"       # Acknowledged valuable member
    COLLABORATION = "collaboration"   # Partnership discussions
    ADVOCACY = "advocacy"            # Actively promotes us
    CONVERTED = "converted"          # Signed up for SMS

@dataclass
class PlatformAccount:
    """Single platform account information"""
    platform: PlatformType
    username: str
    follower_count: int = 0
    following_count: int = 0
    engagement_rate: float = 0.0
    verified: bool = False
    bio: str = ""
    profile_url: str = ""
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class TradingPersonality:
    """User's trading preferences and behavior"""
    trading_focus: List[str] = field(default_factory=list)  # ["tech_stocks", "crypto", "day_trading"]
    risk_tolerance: str = "unknown"  # conservative, moderate, aggressive
    experience_level: str = "unknown"  # beginner, intermediate, advanced, expert
    common_symbols: List[str] = field(default_factory=list)  # ["AAPL", "TSLA", "NVDA"]
    preferred_timeframes: List[str] = field(default_factory=list)  # ["intraday", "swing", "long_term"]
    sentiment_bias: str = "neutral"  # bullish, bearish, neutral

@dataclass
class CommunicationStyle:
    """How user communicates and prefers to be communicated with"""
    formality: str = "neutral"  # formal, casual, neutral
    emoji_usage: str = "moderate"  # none, light, moderate, heavy
    response_preference: str = "balanced"  # technical, simple, balanced
    energy_level: str = "moderate"  # low, moderate, high
    language_patterns: List[str] = field(default_factory=list)  # ["uses_slang", "technical_terms"]

@dataclass
class InfluenceMetrics:
    """User's influence and reach metrics"""
    influence_level: InfluenceLevel = InfluenceLevel.UNKNOWN
    total_followers: int = 0
    avg_engagement_rate: float = 0.0
    community_respect: str = "unknown"  # low, medium, high
    content_quality: str = "unknown"  # low, medium, high
    partnership_potential: str = "unknown"  # low, medium, high

@dataclass
class InteractionHistory:
    """Single interaction record"""
    platform: PlatformType
    interaction_type: str  # comment, dm, like, share, mention
    content: str
    our_response: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    engagement_result: str = ""  # liked_our_response, replied, ignored, blocked
    content_id: str = ""  # ID of our content they interacted with
    sentiment: str = "neutral"  # positive, negative, neutral

class SocialUser:
    """Complete social media user profile across all platforms"""
    
    def __init__(self, primary_username: str, primary_platform: PlatformType):
        self.user_id = f"{primary_platform.value}_{primary_username}"
        self.primary_username = primary_username
        self.primary_platform = primary_platform
        
        # Platform accounts
        self.platform_accounts: Dict[PlatformType, PlatformAccount] = {}
        
        # User personality and preferences
        self.trading_personality = TradingPersonality()
        self.communication_style = CommunicationStyle()
        self.influence_metrics = InfluenceMetrics()
        
        # Engagement tracking
        self.engagement_style = EngagementStyle.LURKER
        self.conversion_stage = ConversionStage.DISCOVERY
        self.last_interaction = datetime.utcnow()
        self.total_interactions = 0
        
        # Interaction history
        self.interaction_history: List[InteractionHistory] = []
        self.max_history_length = 100  # Keep last 100 interactions
        
        # SEMANTIC CLASSIFICATION STORAGE - NEW
        self.intent_classifications: List[Dict[str, Any]] = []
        self.journey_progression: List[Dict[str, Any]] = []
        self.engagement_style_evolution: List[Dict[str, Any]] = []
        
        # Relationship building
        self.relationship_score = 0  # 0-100 scale
        self.conversion_potential = 0  # 0-100 scale
        self.partnership_potential = 0  # 0-100 scale
        
        # SMS conversion tracking
        self.sms_phone_number: Optional[str] = None
        self.conversion_date: Optional[datetime] = None
        self.conversion_source: str = ""  # which platform/content led to conversion
        
        # Metadata
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.notes = ""  # Manual notes about this user
        
        # Change tracking for efficient updates
        self._changed_fields = set()
    
    def add_platform_account(self, account: PlatformAccount):
        """Add or update a platform account"""
        self.platform_accounts[account.platform] = account
        self.updated_at = datetime.utcnow()
        self._changed_fields.add('platform_accounts')
        
        # Update influence metrics
        self._recalculate_influence_metrics()
    
    def add_interaction(self, interaction: InteractionHistory):
        """Add a new interaction to history"""
        self.interaction_history.append(interaction)
        
        # Trim history if too long
        if len(self.interaction_history) > self.max_history_length:
            self.interaction_history = self.interaction_history[-self.max_history_length:]
        
        self.total_interactions += 1
        self.last_interaction = interaction.timestamp
        self.updated_at = datetime.utcnow()
        
        # Update engagement patterns
        self._analyze_engagement_patterns()
        self._update_relationship_score()
        
        self._changed_fields.update(['interaction_history', 'total_interactions', 'last_interaction'])
    
    def _analyze_engagement_patterns(self):
        """Analyze interaction patterns to determine engagement style"""
        if len(self.interaction_history) < 3:
            return
        
        recent_interactions = self.interaction_history[-10:]  # Last 10 interactions
        
        # Count interaction types
        questions = sum(1 for i in recent_interactions if '?' in i.content)
        advice_giving = sum(1 for i in recent_interactions if any(word in i.content.lower() 
                           for word in ['should', 'recommend', 'suggest', 'think', 'strategy']))
        skeptical = sum(1 for i in recent_interactions if any(word in i.content.lower() 
                       for word in ['scam', 'fake', 'doubt', 'suspicious', 'risky']))
        supportive = sum(1 for i in recent_interactions if any(word in i.content.lower() 
                        for word in ['great', 'thanks', 'helpful', 'awesome', 'love']))
        technical = sum(1 for i in recent_interactions if any(word in i.content.lower() 
                       for word in ['rsi', 'macd', 'support', 'resistance', 'fibonacci']))
        
        # Determine primary engagement style
        if questions >= 3:
            self.engagement_style = EngagementStyle.QUESTION_ASKER
        elif advice_giving >= 3:
            self.engagement_style = EngagementStyle.ADVICE_GIVER
        elif skeptical >= 2:
            self.engagement_style = EngagementStyle.SKEPTIC
        elif supportive >= 3:
            self.engagement_style = EngagementStyle.SUPPORTER
        elif technical >= 2:
            self.engagement_style = EngagementStyle.TECHNICAL_ANALYST
        else:
            self.engagement_style = EngagementStyle.LURKER
    
    def _update_relationship_score(self):
        """Calculate relationship strength score (0-100)"""
        score = 0
        
        # Interaction frequency (30 points max)
        days_active = (datetime.utcnow() - self.created_at).days + 1
        interaction_frequency = self.total_interactions / days_active
        score += min(30, int(interaction_frequency * 10))
        
        # Engagement quality (25 points max)
        if len(self.interaction_history) > 0:
            positive_responses = sum(1 for i in self.interaction_history 
                                   if i.engagement_result in ['liked_our_response', 'replied'])
            quality_score = positive_responses / len(self.interaction_history)
            score += int(quality_score * 25)
        
        # Conversation depth (20 points max)
        recent_interactions = self.interaction_history[-5:]
        avg_length = sum(len(i.content) for i in recent_interactions) / max(len(recent_interactions), 1)
        score += min(20, int(avg_length / 10))
        
        # Influence bonus (15 points max)
        if self.influence_metrics.influence_level != InfluenceLevel.UNKNOWN:
            influence_bonus = {
                InfluenceLevel.MICRO: 5,
                InfluenceLevel.MACRO: 10,
                InfluenceLevel.MEGA: 15
            }
            score += influence_bonus.get(self.influence_metrics.influence_level, 0)
        
        # Consistency bonus (10 points max)
        if self.total_interactions >= 5:
            score += 10
        
        self.relationship_score = min(100, score)
        self._changed_fields.add('relationship_score')
    
    def _recalculate_influence_metrics(self):
        """Recalculate influence metrics based on platform accounts"""
        total_followers = sum(account.follower_count for account in self.platform_accounts.values())
        self.influence_metrics.total_followers = total_followers
        
        if total_followers >= 100000:
            self.influence_metrics.influence_level = InfluenceLevel.MEGA
        elif total_followers >= 10000:
            self.influence_metrics.influence_level = InfluenceLevel.MACRO
        elif total_followers >= 1000:
            self.influence_metrics.influence_level = InfluenceLevel.MICRO
        else:
            self.influence_metrics.influence_level = InfluenceLevel.UNKNOWN
        
        # Calculate average engagement rate
        if self.platform_accounts:
            self.influence_metrics.avg_engagement_rate = sum(
                account.engagement_rate for account in self.platform_accounts.values()
            ) / len(self.platform_accounts)
        
        self._changed_fields.add('influence_metrics')
    
    def advance_conversion_stage(self, new_stage: ConversionStage):
        """Move user to next conversion stage"""
        if new_stage.value != self.conversion_stage.value:
            self.conversion_stage = new_stage
            self.updated_at = datetime.utcnow()
            self._changed_fields.add('conversion_stage')
            logger.info(f"User {self.primary_username} advanced to {new_stage.value}")
    
    def mark_sms_conversion(self, phone_number: str, source_platform: str):
        """Mark user as converted to SMS"""
        self.sms_phone_number = phone_number
        self.conversion_date = datetime.utcnow()
        self.conversion_source = source_platform
        self.conversion_stage = ConversionStage.CONVERTED
        self.updated_at = datetime.utcnow()
        self._changed_fields.update(['sms_phone_number', 'conversion_date', 'conversion_source'])
        logger.info(f"User {self.primary_username} converted to SMS from {source_platform}")
    
    def update_trading_interests(self, symbols: List[str], topics: List[str]):
        """Update user's trading interests based on interactions"""
        # Update symbols
        for symbol in symbols:
            if symbol not in self.trading_personality.common_symbols:
                self.trading_personality.common_symbols.append(symbol)
        
        # Keep only recent symbols (last 20)
        if len(self.trading_personality.common_symbols) > 20:
            self.trading_personality.common_symbols = self.trading_personality.common_symbols[-20:]
        
        # Update focus areas
        for topic in topics:
            if topic not in self.trading_personality.trading_focus:
                self.trading_personality.trading_focus.append(topic)
        
        self.updated_at = datetime.utcnow()
        self._changed_fields.add('trading_personality')
    
    def get_recent_interactions(self, days: int = 30) -> List[InteractionHistory]:
        """Get interactions from the last N days"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        return [i for i in self.interaction_history if i.timestamp >= cutoff]
    
    def is_active(self, days: int = 7) -> bool:
        """Check if user has been active recently"""
        return (datetime.utcnow() - self.last_interaction).days <= days
    
    def get_engagement_style_trend(self) -> Dict[str, Any]:
        """Get trend analysis of engagement style changes over time"""
        try:
            if not hasattr(self, 'intent_classifications') or len(self.intent_classifications) < 3:
                return {"trend": "insufficient_data", "confidence": 0.0}
            
            # Analyze last 10 classifications
            recent_styles = [
                c.get("engagement_style", "unknown") 
                for c in self.intent_classifications[-10:] 
                if c.get("engagement_style") != "unknown"
            ]
            
            if len(recent_styles) < 3:
                return {"trend": "insufficient_data", "confidence": 0.0}
            
            # Check for progression patterns
            first_half = recent_styles[:len(recent_styles)//2]
            second_half = recent_styles[len(recent_styles)//2:]
            
            first_dominant = max(set(first_half), key=first_half.count) if first_half else "unknown"
            second_dominant = max(set(second_half), key=second_half.count) if second_half else "unknown"
            
            if first_dominant != second_dominant:
                trend = f"evolving_{first_dominant}_to_{second_dominant}"
                confidence = 0.8
            else:
                trend = f"stable_{first_dominant}"
                confidence = 0.9
            
            return {
                "trend": trend,
                "first_half_style": first_dominant,
                "second_half_style": second_dominant,
                "confidence": confidence,
                "total_classifications": len(recent_styles)
            }
            
        except Exception as e:
            logger.error(f"Failed to get engagement style trend: {e}")
            return {"trend": "error", "confidence": 0.0}
    
    def get_conversion_readiness_score(self) -> float:
        """Calculate conversion readiness based on semantic analysis"""
        try:
            score = 0.0
            
            # Base relationship score (40% of total)
            score += (self.relationship_score / 100) * 0.4
            
            # Journey stage progression (30% of total)
            stage_scores = {
                ConversionStage.DISCOVERY: 0.1,
                ConversionStage.ENGAGEMENT: 0.3,
                ConversionStage.RECOGNITION: 0.6,
                ConversionStage.COLLABORATION: 0.8,
                ConversionStage.ADVOCACY: 0.9,
                ConversionStage.CONVERTED: 1.0
            }
            score += stage_scores.get(self.conversion_stage, 0.1) * 0.3
            
            # Recent sentiment trend (20% of total)
            if hasattr(self, 'intent_classifications') and self.intent_classifications:
                recent_sentiments = [
                    c.get("sentiment", "neutral") 
                    for c in self.intent_classifications[-5:]
                ]
                positive_ratio = sum(1 for s in recent_sentiments if s == "positive") / len(recent_sentiments)
                score += positive_ratio * 0.2
            
            # Engagement frequency (10% of total)
            if self.total_interactions >= 10:
                score += 0.1
            elif self.total_interactions >= 5:
                score += 0.05
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate conversion readiness: {e}")
            return 0.0
    
    def should_prioritize_response(self) -> bool:
        """Determine if this user should get priority responses - ENHANCED WITH SEMANTIC ANALYSIS"""
        # Original priority criteria
        basic_priority = (
            self.influence_metrics.influence_level in [InfluenceLevel.MACRO, InfluenceLevel.MEGA] or
            self.relationship_score > 70 or
            self.conversion_stage in [ConversionStage.COLLABORATION, ConversionStage.ADVOCACY]
        )
        
        # Semantic priority criteria
        semantic_priority = False
        try:
            if hasattr(self, 'intent_classifications') and self.intent_classifications:
                # High conversion potential users
                recent_conversion_potential = [
                    c.get("conversion_potential", "low") 
                    for c in self.intent_classifications[-3:]
                ]
                if "high" in recent_conversion_potential:
                    semantic_priority = True
                
                # Users showing journey acceleration
                journey_velocity = self._calculate_journey_velocity()
                if journey_velocity.get("velocity") == "accelerating":
                    semantic_priority = True
        except Exception as e:
            logger.error(f"Failed to check semantic priority: {e}")
        
        return basic_priority or semantic_priority
    
    def _calculate_journey_velocity(self) -> Dict[str, Any]:
        """Calculate how quickly user is progressing through journey stages"""
        try:
            if not hasattr(self, 'journey_progression') or len(self.journey_progression) < 2:
                return {"velocity": "unknown", "direction": "stable"}
            
            # Calculate time between progressions
            progressions = self.journey_progression[-3:]  # Last 3 progressions
            forward_moves = sum(1 for p in progressions if p.get("progression_direction") == "forward")
            backward_moves = sum(1 for p in progressions if p.get("progression_direction") == "backward")
            
            if forward_moves > backward_moves:
                velocity = "accelerating" if forward_moves >= 2 else "progressing"
                direction = "forward"
            elif backward_moves > forward_moves:
                velocity = "declining"
                direction = "backward"
            else:
                velocity = "stable"
                direction = "stable"
            
            return {
                "velocity": velocity,
                "direction": direction,
                "forward_moves": forward_moves,
                "backward_moves": backward_moves
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate journey velocity: {e}")
            return {"velocity": "error", "direction": "unknown"}
    
    def _analyze_content_preferences(self) -> Dict[str, Any]:
        """Analyze what type of content user engages with most"""
        try:
            if not hasattr(self, 'intent_classifications'):
                return {"preferences": "unknown"}
            
            # Count interaction types
            interaction_types = {}
            topics_mentioned = {}
            
            for classification in self.intent_classifications[-20:]:  # Last 20 interactions
                intent = classification.get("intent", "unknown")
                interaction_types[intent] = interaction_types.get(intent, 0) + 1
                
                # Extract topics from content if available
                content = classification.get("content", "")
                for topic in ["technical", "earnings", "options", "crypto", "news"]:
                    if topic in content.lower():
                        topics_mentioned[topic] = topics_mentioned.get(topic, 0) + 1
            
            preferences = {
                "most_common_interaction": max(interaction_types, key=interaction_types.get) if interaction_types else "unknown",
                "preferred_topics": sorted(topics_mentioned.items(), key=lambda x: x[1], reverse=True)[:3],
                "interaction_diversity": len(interaction_types),
                "topic_diversity": len(topics_mentioned)
            }
            
            return preferences
            
        except Exception as e:
            logger.error(f"Failed to analyze content preferences: {e}")
            return {"preferences": "error"}
    
    def _suggest_engagement_strategy(self) -> Dict[str, Any]:
        """Suggest optimal engagement strategy based on semantic analysis"""
        try:
            readiness = self.get_conversion_readiness_score()
            style_trend = self.get_engagement_style_trend()
            journey_velocity = self._calculate_journey_velocity()
            
            # Determine strategy based on analysis
            if readiness > 0.8:
                strategy = "conversion_focus"
                approach = "Direct conversion attempt with trial offer"
            elif readiness > 0.6:
                strategy = "nurture_to_conversion"
                approach = "Value demonstration and soft conversion"
            elif journey_velocity["direction"] == "forward":
                strategy = "momentum_building"
                approach = "Continue current engagement pattern"
            elif style_trend["trend"].startswith("evolving"):
                strategy = "adaptive_engagement"
                approach = "Match evolving communication style"
            else:
                strategy = "relationship_building"
                approach = "Focus on increasing relationship score"
            
            return {
                "recommended_strategy": strategy,
                "approach": approach,
                "confidence": min(readiness + 0.2, 1.0),
                "next_actions": self._get_next_action_recommendations(strategy)
            }
            
        except Exception as e:
            logger.error(f"Failed to suggest engagement strategy: {e}")
            return {"recommended_strategy": "default_engagement"}
    
    def _get_next_action_recommendations(self, strategy: str) -> List[str]:
        """Get specific next action recommendations"""
        action_map = {
            "conversion_focus": [
                "Send direct trial offer DM",
                "Highlight success stories in response",
                "Offer personalized demo"
            ],
            "nurture_to_conversion": [
                "Share relevant technical analysis",
                "Mention user testimonials",
                "Provide educational value"
            ],
            "momentum_building": [
                "Continue current engagement style",
                "Increase response frequency",
                "Reference their expertise"
            ],
            "adaptive_engagement": [
                "Match their evolving communication style",
                "Experiment with different response types",
                "Monitor style changes"
            ],
            "relationship_building": [
                "Increase personalization",
                "Remember and reference past interactions",
                "Provide consistent value"
            ]
        }
        
        return action_map.get(strategy, ["Continue standard engagement"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage"""
        return {
            "user_id": self.user_id,
            "primary_username": self.primary_username,
            "primary_platform": self.primary_platform.value,
            
            # Platform accounts
            "platform_accounts": {
                platform.value: {
                    "platform": account.platform.value,
                    "username": account.username,
                    "follower_count": account.follower_count,
                    "following_count": account.following_count,
                    "engagement_rate": account.engagement_rate,
                    "verified": account.verified,
                    "bio": account.bio,
                    "profile_url": account.profile_url,
                    "last_updated": account.last_updated.isoformat()
                }
                for platform, account in self.platform_accounts.items()
            },
            
            # Personality
            "trading_personality": {
                "trading_focus": self.trading_personality.trading_focus,
                "risk_tolerance": self.trading_personality.risk_tolerance,
                "experience_level": self.trading_personality.experience_level,
                "common_symbols": self.trading_personality.common_symbols,
                "preferred_timeframes": self.trading_personality.preferred_timeframes,
                "sentiment_bias": self.trading_personality.sentiment_bias
            },
            
            "communication_style": {
                "formality": self.communication_style.formality,
                "emoji_usage": self.communication_style.emoji_usage,
                "response_preference": self.communication_style.response_preference,
                "energy_level": self.communication_style.energy_level,
                "language_patterns": self.communication_style.language_patterns
            },
            
            "influence_metrics": {
                "influence_level": self.influence_metrics.influence_level.value,
                "total_followers": self.influence_metrics.total_followers,
                "avg_engagement_rate": self.influence_metrics.avg_engagement_rate,
                "community_respect": self.influence_metrics.community_respect,
                "content_quality": self.influence_metrics.content_quality,
                "partnership_potential": self.influence_metrics.partnership_potential
            },
            
            # Engagement
            "engagement_style": self.engagement_style.value,
            "conversion_stage": self.conversion_stage.value,
            "last_interaction": self.last_interaction.isoformat(),
            "total_interactions": self.total_interactions,
            
            # Interaction history (recent only)
            "interaction_history": [
                {
                    "platform": i.platform.value,
                    "interaction_type": i.interaction_type,
                    "content": i.content,
                    "our_response": i.our_response,
                    "timestamp": i.timestamp.isoformat(),
                    "engagement_result": i.engagement_result,
                    "content_id": i.content_id,
                    "sentiment": i.sentiment
                }
                for i in self.interaction_history[-50:]  # Store only last 50
            ],
            
            # SEMANTIC CLASSIFICATION STORAGE - NEW
            "intent_classifications": getattr(self, 'intent_classifications', []),
            "journey_progression": getattr(self, 'journey_progression', []),
            "engagement_style_evolution": getattr(self, 'engagement_style_evolution', []),
            
            # Scores
            "relationship_score": self.relationship_score,
            "conversion_potential": self.conversion_potential,
            "partnership_potential": self.partnership_potential,
            
            # SMS conversion
            "sms_phone_number": self.sms_phone_number,
            "conversion_date": self.conversion_date.isoformat() if self.conversion_date else None,
            "conversion_source": self.conversion_source,
            
            # Metadata
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "notes": self.notes
        }

    # OPTIMIZATION PATCH: Production-Ready LLM Integration
# Adds confidence filtering, delta tracking, topic weighting, and symbol relationship graphs

# ===========================================
# UPDATED: social/models/social_user.py
# ADD these new fields and methods to SocialUser class
# ===========================================

class SocialUser:
    """Enhanced SocialUser with LLM optimization features"""
    
    def __init__(self, primary_username: str, primary_platform: PlatformType):
        # ... existing initialization ...
        
        # LLM OPTIMIZATION FEATURES - NEW
        self.last_llm_update = datetime.utcnow()
        self.llm_confidence_threshold = 0.6
        self.topic_frequency = {}  # {"technical_analysis": 15, "earnings": 8}
        self.symbol_relationships = {}  # {"AAPL": {"TSLA": 5, "NVDA": 3}}
        self.symbol_frequency = {}  # {"AAPL": 12, "TSLA": 8}
        self.llm_update_count = 0
    
    def update_trading_interests_with_frequency(self, symbols: List[str], topics: List[str], 
                                              confidence: float = 1.0):
        """
        FREQUENCY-WEIGHTED topic and symbol tracking
        Higher mentioned topics/symbols get priority in response generation
        """
        try:
            # Only update if we have reasonable confidence
            if confidence < self.llm_confidence_threshold:
                logger.info(f"Skipping low confidence update: {confidence}")
                return
            
            # Update topic frequency with weighting
            for topic in topics:
                if topic not in self.topic_frequency:
                    self.topic_frequency[topic] = 0
                self.topic_frequency[topic] += confidence  # Weighted by confidence
                
                # Add to trading focus if high frequency
                if (topic not in self.trading_personality.trading_focus and 
                    self.topic_frequency[topic] >= 2.0):  # Threshold for inclusion
                    self.trading_personality.trading_focus.append(topic)
            
            # Update symbol frequency with weighting
            for symbol in symbols:
                if symbol not in self.symbol_frequency:
                    self.symbol_frequency[symbol] = 0
                self.symbol_frequency[symbol] += confidence
                
                # Add to common symbols if high frequency
                if (symbol not in self.trading_personality.common_symbols and 
                    self.symbol_frequency[symbol] >= 2.0):
                    self.trading_personality.common_symbols.append(symbol)
            
            # Build symbol relationship graph
            self._update_symbol_relationships(symbols, confidence)
            
            # Update LLM tracking
            self.last_llm_update = datetime.utcnow()
            self.llm_update_count += 1
            
            logger.info(f"Updated interests for @{self.primary_username}: topics={topics}, symbols={symbols}, confidence={confidence}")
            
        except Exception as e:
            logger.error(f"Failed to update trading interests with frequency: {e}")
    
    def _update_symbol_relationships(self, symbols: List[str], confidence: float):
        """
        BUILD SYMBOL AFFINITY GRAPH
        Track which symbols user mentions together (co-occurrence)
        """
        try:
            if len(symbols) < 2:
                return  # Need at least 2 symbols for relationships
            
            # Create relationships between all symbol pairs
            for i, symbol1 in enumerate(symbols):
                if symbol1 not in self.symbol_relationships:
                    self.symbol_relationships[symbol1] = {}
                
                for j, symbol2 in enumerate(symbols):
                    if i != j:  # Don't relate symbol to itself
                        if symbol2 not in self.symbol_relationships[symbol1]:
                            self.symbol_relationships[symbol1][symbol2] = 0
                        
                        # Weighted by confidence
                        self.symbol_relationships[symbol1][symbol2] += confidence
            
            logger.debug(f"Updated symbol relationships: {len(symbols)} symbols co-occurred")
            
        except Exception as e:
            logger.error(f"Failed to update symbol relationships: {e}")
    
    def get_top_topics_by_frequency(self, limit: int = 5) -> List[Tuple[str, float]]:
        """Get most frequently mentioned topics for response personalization"""
        try:
            sorted_topics = sorted(
                self.topic_frequency.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            return sorted_topics[:limit]
        except Exception as e:
            logger.error(f"Failed to get top topics: {e}")
            return []
    
    def get_top_symbols_by_frequency(self, limit: int = 5) -> List[Tuple[str, float]]:
        """Get most frequently mentioned symbols for response personalization"""
        try:
            sorted_symbols = sorted(
                self.symbol_frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_symbols[:limit]
        except Exception as e:
            logger.error(f"Failed to get top symbols: {e}")
            return []
    
    def get_related_symbols(self, symbol: str, limit: int = 3) -> List[Tuple[str, float]]:
        """
        GET SYMBOL AFFINITY for response personalization
        If user mentions AAPL, suggest related symbols they often discuss
        """
        try:
            if symbol not in self.symbol_relationships:
                return []
            
            related = sorted(
                self.symbol_relationships[symbol].items(),
                key=lambda x: x[1],
                reverse=True
            )
            return related[:limit]
        except Exception as e:
            logger.error(f"Failed to get related symbols: {e}")
            return []
    
    def get_llm_learning_velocity(self) -> Dict[str, Any]:
        """
        DELTA TRACKING - measure how fast LLM is learning about user
        """
        try:
            days_since_creation = (datetime.utcnow() - self.created_at).days + 1
            days_since_last_update = (datetime.utcnow() - self.last_llm_update).days
            
            learning_velocity = {
                "updates_per_day": self.llm_update_count / days_since_creation,
                "days_since_last_update": days_since_last_update,
                "total_llm_updates": self.llm_update_count,
                "learning_rate": "high" if self.llm_update_count / days_since_creation > 1.0 else "moderate" if self.llm_update_count / days_since_creation > 0.3 else "low",
                "last_update": self.last_llm_update.isoformat()
            }
            
            return learning_velocity
        except Exception as e:
            logger.error(f"Failed to get learning velocity: {e}")
            return {"learning_rate": "unknown"}
    
    def get_personalization_context(self) -> Dict[str, Any]:
        """
        COMPREHENSIVE CONTEXT for response generation
        Combines frequency data, relationships, and learning velocity
        """
        try:
            return {
                "top_topics": self.get_top_topics_by_frequency(3),
                "top_symbols": self.get_top_symbols_by_frequency(3),
                "learning_velocity": self.get_llm_learning_velocity(),
                "total_topic_mentions": sum(self.topic_frequency.values()),
                "total_symbol_mentions": sum(self.symbol_frequency.values()),
                "relationship_graph_size": len(self.symbol_relationships),
                "confidence_threshold": self.llm_confidence_threshold
            }
        except Exception as e:
            logger.error(f"Failed to get personalization context: {e}")
            return {}

# ===========================================
# UPDATED: social/services/social_trading_agent.py
# ENHANCE the LLM integration with confidence filtering
# ===========================================

    async def _integrate_llm_results_into_user_profile(self, user, intent_data: Dict, 
                                                     symbols: List[str], topics: List[str], 
                                                     engagement_style: str):
        """
        CONFIDENCE-FILTERED LLM integration with frequency tracking
        """
        try:
            confidence = intent_data.get("confidence", 0.0)
            
            # CONFIDENCE-BASED FILTERING
            if confidence < user.llm_confidence_threshold:
                logger.info(f"Skipping LLM update for @{user.primary_username}: confidence {confidence} below threshold {user.llm_confidence_threshold}")
                return
            
            # 1. STORE INTENT CLASSIFICATION (only high confidence)
            classification_record = {
                "timestamp": intent_data.get("timestamp"),
                "content": intent_data.get("content", ""),
                "intent": intent_data.get("intent"),
                "sentiment": intent_data.get("sentiment"),
                "engagement_style": engagement_style,
                "journey_stage": intent_data.get("user_journey_stage"),
                "conversion_potential": intent_data.get("conversion_potential"),
                "confidence": confidence,
                "symbols_mentioned": symbols,
                "topics_discussed": topics,
                "filtered": False  # Passed confidence filter
            }
            
            if not hasattr(user, 'intent_classifications'):
                user.intent_classifications = []
            user.intent_classifications.append(classification_record)
            
            # Keep only last 20 classifications
            if len(user.intent_classifications) > 20:
                user.intent_classifications = user.intent_classifications[-20:]
            
            # 2. UPDATE ENGAGEMENT STYLE (confidence-based)
            await self._update_engagement_style_from_llm_with_confidence(user, engagement_style, confidence)
            
            # 3. UPDATE JOURNEY PROGRESSION (confidence-based)
            await self._update_journey_progression_from_llm_with_confidence(user, intent_data.get("user_journey_stage"), confidence)
            
            # 4. FREQUENCY-WEIGHTED TRADING INTERESTS UPDATE
            user.update_trading_interests_with_frequency(symbols, topics, confidence)
            
            # 5. UPDATE COMMUNICATION STYLE
            await self._update_communication_style_from_llm(user, intent_data.get("content", ""), intent_data)
            
            # Save all updates
            await self.memory_service._save_user_updates(user)
            
            logger.info(f"LLM integration complete for @{user.primary_username} (confidence: {confidence})")
            
        except Exception as e:
            logger.error(f"Failed to integrate LLM results: {e}")

    async def _update_engagement_style_from_llm_with_confidence(self, user, llm_engagement_style: str, confidence: float):
        """Update engagement style only if confidence is high enough"""
        try:
            # Require higher confidence for style changes
            if confidence < 0.7:  # Higher threshold for style changes
                logger.debug(f"Skipping engagement style update: confidence {confidence} too low")
                return
            
            from ..models.social_user import EngagementStyle
            
            style_mapping = {
                "question_asker": EngagementStyle.QUESTION_ASKER,
                "advice_giver": EngagementStyle.ADVICE_GIVER,
                "skeptic": EngagementStyle.SKEPTIC,
                "supporter": EngagementStyle.SUPPORTER,
                "lurker": EngagementStyle.LURKER,
                "technical_analyst": EngagementStyle.TECHNICAL_ANALYST
            }
            
            new_style = style_mapping.get(llm_engagement_style, EngagementStyle.LURKER)
            old_style = user.engagement_style
            
            # Track evolution if style changed
            if new_style != old_style:
                if not hasattr(user, 'engagement_style_evolution'):
                    user.engagement_style_evolution = []
                
                evolution_record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "from_style": old_style.value,
                    "to_style": new_style.value,
                    "confidence": confidence,
                    "llm_triggered": True
                }
                
                user.engagement_style_evolution.append(evolution_record)
                user.engagement_style = new_style
                
                logger.info(f"User @{user.primary_username} evolved: {old_style.value} → {new_style.value} (confidence: {confidence})")
            
        except Exception as e:
            logger.error(f"Failed to update engagement style from LLM: {e}")

    async def _update_journey_progression_from_llm_with_confidence(self, user, llm_journey_stage: str, confidence: float):
        """Update journey progression only if confidence is high enough"""
        try:
            # Require high confidence for journey stage changes
            if confidence < 0.75:  # Even higher threshold for journey changes
                logger.debug(f"Skipping journey progression update: confidence {confidence} too low")
                return
            
            from ..models.social_user import ConversionStage
            
            stage_mapping = {
                "discovery": ConversionStage.DISCOVERY,
                "engagement": ConversionStage.ENGAGEMENT,
                "recognition": ConversionStage.RECOGNITION,
                "collaboration": ConversionStage.COLLABORATION,
                "advocacy": ConversionStage.ADVOCACY,
                "converted": ConversionStage.CONVERTED
            }
            
            new_stage = stage_mapping.get(llm_journey_stage, ConversionStage.DISCOVERY)
            old_stage = user.conversion_stage
            
            # Track progression if stage changed
            if new_stage != old_stage:
                if not hasattr(user, 'journey_progression'):
                    user.journey_progression = []
                
                progression_record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "from_stage": old_stage.value,
                    "to_stage": new_stage.value,
                    "progression_direction": self._get_progression_direction(old_stage.value, new_stage.value),
                    "confidence": confidence,
                    "llm_triggered": True
                }
                
                user.journey_progression.append(progression_record)
                user.advance_conversion_stage(new_stage)
                
                logger.info(f"User @{user.primary_username} journey: {old_stage.value} → {new_stage.value} (confidence: {confidence})")
            
        except Exception as e:
            logger.error(f"Failed to update journey progression from LLM: {e}")

# ===========================================
# ENHANCED: Response generation with frequency data
# ===========================================

    def _build_response_prompt(self, context: Dict[str, Any]) -> str:
        """
        ENHANCED response prompt with frequency-weighted personalization
        """
        intent = context["intent"]
        user_context = context["user_context"]
        tool_results = context["tool_results"]
        platform = context["platform"]
        
        # User personalization with FREQUENCY DATA
        user_profile = user_context.get("user_profile", {})
        username = user_profile.get("primary_username", "")
        
        # GET FREQUENCY-WEIGHTED PERSONALIZATION
        personalization_context = {}
        if hasattr(user_profile, 'get_personalization_context'):
            personalization_context = user_profile.get_personalization_context()
        
        top_topics = personalization_context.get("top_topics", [])
        top_symbols = personalization_context.get("top_symbols", [])
        learning_velocity = personalization_context.get("learning_velocity", {})
        
        # SYMBOL RELATIONSHIP suggestions
        mentioned_symbols = intent.get("symbols", [])
        related_symbols = []
        if mentioned_symbols and hasattr(user_profile, 'get_related_symbols'):
            for symbol in mentioned_symbols[:1]:  # Check first mentioned symbol
                related = user_profile.get_related_symbols(symbol, 2)
                related_symbols.extend([r[0] for r in related])
        
        # Platform-specific voice rules
        platform_voice_rules = {
            "tiktok": "Casual, energetic, use emojis, trending language, under 150 chars",
            "twitter": "Concise, witty, hashtag-friendly, professional but accessible",
            "instagram": "Visual storytelling tone, emoji-rich, engaging, community-focused",
            "youtube": "Conversational, educational, longer-form OK, professional tone",
            "linkedin": "Professional, business-focused, industry expertise, formal tone"
        }
        
        voice_rule = platform_voice_rules.get(platform, "Professional but friendly tone")
        
        prompt = f"""
You are an expert social media manager for a trading bot company. Generate a personalized response using FREQUENCY-WEIGHTED user insights.

=== INTERACTION CONTEXT ===
Platform: {platform}
User: @{username}
Intent: {intent.get('primary_intent')} - {intent.get('sub_intent')}
Response Strategy: {intent.get('response_strategy')}
Current Symbols Mentioned: {', '.join(mentioned_symbols) if mentioned_symbols else 'None'}

=== FREQUENCY-WEIGHTED USER INSIGHTS ===
Top Discussed Topics: {', '.join([f"{topic}({score:.1f})" for topic, score in top_topics[:3]]) if top_topics else 'None'}
Top Discussed Symbols: {', '.join([f"{symbol}({score:.1f})" for symbol, score in top_symbols[:3]]) if top_symbols else 'None'}
Related Symbols: {', '.join(related_symbols) if related_symbols else 'None'}
Learning Velocity: {learning_velocity.get('learning_rate', 'unknown')} ({learning_velocity.get('updates_per_day', 0):.1f} updates/day)

=== USER PROFILE ===
Relationship Score: {user_context.get('relationship_score', 0)}/100
Journey Stage: {intent.get('user_journey_stage', 'discovery')}
Engagement Style: {intent.get('engagement_style_detected', 'unknown')}
Communication Style: {user_context.get('communication_style', {}).get('formality', 'neutral')}

=== TOOL RESULTS ===
{json.dumps(tool_results, indent=2) if tool_results else 'No tool results available'}

=== PERSONALIZATION STRATEGY ===
**Use Frequency Data for Personalization:**
- Reference their most discussed topics when relevant
- Mention their frequently traded symbols for context
- Suggest related symbols from their relationship graph
- Adapt depth based on their learning velocity

**Platform Optimization:**
- {voice_rule}
- Max Length: {context.get('max_length', 280)} characters
- Supports Emoji: {context.get('supports_emoji', True)}

**Response Guidelines:**
- If high learning velocity: Use more technical depth
- If top topics include "technical_analysis": Reference charts/indicators
- If related symbols available: Mention symbol relationships naturally
- If low frequency data: Keep response general but engaging

Generate a response that leverages the frequency-weighted insights for maximum personalization. Return only the response text.
"""
        return prompt

# ===========================================
# UPDATED: Storage methods to include new fields
# ===========================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - UPDATED with new fields"""
        base_dict = super().to_dict()  # Get existing fields
        
        # Add LLM optimization fields
        base_dict.update({
            # LLM tracking
            "last_llm_update": safe_datetime_to_iso(getattr(self, 'last_llm_update', datetime.utcnow())),
            "llm_confidence_threshold": getattr(self, 'llm_confidence_threshold', 0.6),
            "llm_update_count": getattr(self, 'llm_update_count', 0),
            
            # Frequency tracking
            "topic_frequency": getattr(self, 'topic_frequency', {}),
            "symbol_frequency": getattr(self, 'symbol_frequency', {}),
            
            # Relationship graph
            "symbol_relationships": getattr(self, 'symbol_relationships', {}),
        })
        
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SocialUser':
        """Create from dictionary - UPDATED to load new fields"""
        user = super().from_dict(data)  # Load existing fields
        
        # Load LLM optimization fields
        user.last_llm_update = safe_iso_to_datetime(data.get("last_llm_update")) or datetime.utcnow()
        user.llm_confidence_threshold = data.get("llm_confidence_threshold", 0.6)
        user.llm_update_count = data.get("llm_update_count", 0)
        
        # Load frequency tracking
        user.topic_frequency = data.get("topic_frequency", {})
        user.symbol_frequency = data.get("symbol_frequency", {})
        
        # Load relationship graph
        user.symbol_relationships = data.get("symbol_relationships", {})
        
        return user

# ===========================================
# ANALYTICS: New methods for monitoring LLM performance
# ===========================================

class SocialMemoryService:
    """ENHANCED with LLM optimization analytics"""
    
    async def get_llm_performance_analytics(self) -> Dict[str, Any]:
        """Get analytics on LLM integration performance"""
        try:
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "total_users": {"$sum": 1},
                        "avg_confidence_threshold": {"$avg": "$llm_confidence_threshold"},
                        "avg_llm_updates": {"$avg": "$llm_update_count"},
                        "total_topic_mentions": {"$sum": {"$sum": {"$objectToArray": "$topic_frequency"}.value}},
                        "total_symbol_mentions": {"$sum": {"$sum": {"$objectToArray": "$symbol_frequency"}.value}},
                    }
                }
            ]
            
            result = await self.users_collection.aggregate(pipeline).to_list(1)
            
            if result:
                analytics = result[0]
                analytics["avg_learning_velocity"] = analytics["avg_llm_updates"] / max(analytics["total_users"], 1)
                return analytics
            else:
                return {"total_users": 0}
                
        except Exception as e:
            logger.error(f"Failed to get LLM analytics: {e}")
            return {"error": str(e)}
    
    async def get_top_topics_across_users(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most frequently mentioned topics across all users"""
        try:
            # Aggregate topic frequencies across all users
            pipeline = [
                {"$project": {"topic_frequency": {"$objectToArray": "$topic_frequency"}}},
                {"$unwind": "$topic_frequency"},
                {
                    "$group": {
                        "_id": "$topic_frequency.k",
                        "total_frequency": {"$sum": "$topic_frequency.v"}
                    }
                },
                {"$sort": {"total_frequency": -1}},
                {"$limit": limit}
            ]
            
            results = await self.users_collection.aggregate(pipeline).to_list(limit)
            return [(result["_id"], result["total_frequency"]) for result in results]
            
        except Exception as e:
            logger.error(f"Failed to get top topics: {e}")
            return []
    
    async def get_symbol_relationship_network(self, min_strength: float = 2.0) -> Dict[str, Dict[str, float]]:
        """Get global symbol relationship network for content strategy"""
        try:
            # Find all users with symbol relationships
            users_with_relationships = await self.users_collection.find(
                {"symbol_relationships": {"$exists": True, "$ne": {}}}
            ).to_list(None)
            
            # Aggregate relationships
            global_network = {}
            for user_doc in users_with_relationships:
                user_relationships = user_doc.get("symbol_relationships", {})
                
                for symbol1, relationships in user_relationships.items():
                    if symbol1 not in global_network:
                        global_network[symbol1] = {}
                    
                    for symbol2, strength in relationships.items():
                        if symbol2 not in global_network[symbol1]:
                            global_network[symbol1][symbol2] = 0
                        global_network[symbol1][symbol2] += strength
            
            # Filter by minimum strength
            filtered_network = {}
            for symbol1, relationships in global_network.items():
                strong_relationships = {
                    symbol2: strength 
                    for symbol2, strength in relationships.items() 
                    if strength >= min_strength
                }
                if strong_relationships:
                    filtered_network[symbol1] = strong_relationships
            
            return filtered_network
            
        except Exception as e:
            logger.error(f"Failed to get symbol relationship network: {e}")
            return {}

# ===========================================

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SocialUser':
        """Create SocialUser from dictionary"""
        user = cls(
            primary_username=data["primary_username"],
            primary_platform=PlatformType(data["primary_platform"])
        )
        
        # Basic fields
        user.user_id = data["user_id"]
        
        # Platform accounts
        for platform_str, account_data in data.get("platform_accounts", {}).items():
            platform = PlatformType(platform_str)
            account = PlatformAccount(
                platform=PlatformType(account_data["platform"]),
                username=account_data["username"],
                follower_count=account_data.get("follower_count", 0),
                following_count=account_data.get("following_count", 0),
                engagement_rate=account_data.get("engagement_rate", 0.0),
                verified=account_data.get("verified", False),
                bio=account_data.get("bio", ""),
                profile_url=account_data.get("profile_url", ""),
                last_updated=datetime.fromisoformat(account_data.get("last_updated", datetime.utcnow().isoformat()))
            )
            user.platform_accounts[platform] = account
        
        # Personality data
        trading_data = data.get("trading_personality", {})
        user.trading_personality = TradingPersonality(
            trading_focus=trading_data.get("trading_focus", []),
            risk_tolerance=trading_data.get("risk_tolerance", "unknown"),
            experience_level=trading_data.get("experience_level", "unknown"),
            common_symbols=trading_data.get("common_symbols", []),
            preferred_timeframes=trading_data.get("preferred_timeframes", []),
            sentiment_bias=trading_data.get("sentiment_bias", "neutral")
        )
        
        comm_data = data.get("communication_style", {})
        user.communication_style = CommunicationStyle(
            formality=comm_data.get("formality", "neutral"),
            emoji_usage=comm_data.get("emoji_usage", "moderate"),
            response_preference=comm_data.get("response_preference", "balanced"),
            energy_level=comm_data.get("energy_level", "moderate"),
            language_patterns=comm_data.get("language_patterns", [])
        )
        
        influence_data = data.get("influence_metrics", {})
        user.influence_metrics = InfluenceMetrics(
            influence_level=InfluenceLevel(influence_data.get("influence_level", "unknown")),
            total_followers=influence_data.get("total_followers", 0),
            avg_engagement_rate=influence_data.get("avg_engagement_rate", 0.0),
            community_respect=influence_data.get("community_respect", "unknown"),
            content_quality=influence_data.get("content_quality", "unknown"),
            partnership_potential=influence_data.get("partnership_potential", "unknown")
        )
        
        # Engagement data
        user.engagement_style = EngagementStyle(data.get("engagement_style", "lurker"))
        user.conversion_stage = ConversionStage(data.get("conversion_stage", "discovery"))
        user.last_interaction = datetime.fromisoformat(data.get("last_interaction", datetime.utcnow().isoformat()))
        user.total_interactions = data.get("total_interactions", 0)
        
        # Interaction history
        for interaction_data in data.get("interaction_history", []):
            interaction = InteractionHistory(
                platform=PlatformType(interaction_data["platform"]),
                interaction_type=interaction_data["interaction_type"],
                content=interaction_data["content"],
                our_response=interaction_data.get("our_response", ""),
                timestamp=datetime.fromisoformat(interaction_data["timestamp"]),
                engagement_result=interaction_data.get("engagement_result", ""),
                content_id=interaction_data.get("content_id", ""),
                sentiment=interaction_data.get("sentiment", "neutral")
            )
            user.interaction_history.append(interaction)
        
        # Scores
        user.relationship_score = data.get("relationship_score", 0)
        user.conversion_potential = data.get("conversion_potential", 0)
        user.partnership_potential = data.get("partnership_potential", 0)
        
        # SMS conversion
        user.sms_phone_number = data.get("sms_phone_number")
        if data.get("conversion_date"):
            user.conversion_date = datetime.fromisoformat(data["conversion_date"])
        user.conversion_source = data.get("conversion_source", "")
        
        # Metadata
        user.created_at = datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat()))
        user.updated_at = datetime.fromisoformat(data.get("updated_at", datetime.utcnow().isoformat()))
        user.notes = data.get("notes", "")
        
        # Clear changed fields after loading
        user._changed_fields = set()
        
        return user
    
    def __str__(self) -> str:
        return f"SocialUser({self.primary_username}, {self.primary_platform.value}, score={self.relationship_score})"
    
    def __repr__(self) -> str:
        return self.__str__()
