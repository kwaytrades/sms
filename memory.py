"""
Enhanced SMS Trading AI Agent - MemoryManager System with Emotional Intelligence
Complete implementation with Redis, MongoDB, Pinecone integration plus emotional awareness,
context intelligence, and response adaptation for multi-agent support.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib
import re
import statistics
from collections import deque, defaultdict

# External dependencies
import redis.asyncio as redis
from motor.motor_asyncio import AsyncIOMotorClient
import pinecone
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageDirection(Enum):
    """Message direction enumeration"""
    USER = "user"
    BOT = "bot"

class MemoryType(Enum):
    """Memory type classification"""
    CASUAL = "casual"
    TRADING = "trading"
    CUSTOMER_SERVICE = "customer_service"
    SALES = "sales"
    IMPORTANT = "important"
    SYSTEM = "system"
    EMOTIONAL = "emotional"

class AgentType(Enum):
    """Agent type classification"""
    TRADING = "trading"
    CUSTOMER_SERVICE = "customer_service" 
    SALES = "sales"

class EmotionType(Enum):
    """Emotion classification"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    FRUSTRATED = "frustrated"
    EXCITED = "excited"
    CONFUSED = "confused"
    CONFIDENT = "confident"
    ANXIOUS = "anxious"

@dataclass
class EmotionalState:
    """Emotional state tracking"""
    emotion_type: EmotionType
    emotion_score: float  # 0.0 to 1.0
    sentiment_label: str  # positive/neutral/negative
    confidence: float
    timestamp: datetime
    triggers: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['emotion_type'] = self.emotion_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class Message:
    """Enhanced message structure with emotional context"""
    user_id: str
    content: str
    direction: MessageDirection
    timestamp: datetime
    agent_type: AgentType = AgentType.TRADING
    message_type: MemoryType = MemoryType.CASUAL
    topics: List[str] = None
    importance_score: float = 0.5
    metadata: Dict[str, Any] = None
    emotional_state: Optional[EmotionalState] = None
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['direction'] = self.direction.value
        data['agent_type'] = self.agent_type.value
        data['message_type'] = self.message_type.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.emotional_state:
            data['emotional_state'] = self.emotional_state.to_dict()
        return data

@dataclass
class ConversationSummary:
    """Enhanced conversation summary with emotional insights"""
    user_id: str
    summary: str
    topics: List[str]
    timestamp: datetime
    importance_score: float
    message_count: int
    session_id: str
    agent_type: AgentType
    agent_insights: Dict[str, Any] = None
    emotional_summary: Dict[str, Any] = None
    engagement_level: float = 0.5
    avg_sentiment_score: float = 0.5
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['agent_type'] = self.agent_type.value
        return data

@dataclass
class UserProfile:
    """Enhanced multi-agent user profile with emotional intelligence"""
    user_id: str
    
    # Trading agent profile
    risk_tolerance: str = "medium"
    trading_style: str = "swing"
    watchlist: List[str] = None
    portfolio_insights: Dict[str, Any] = None
    confidence_score: float = 0.5  # Trading confidence
    
    # Customer service profile  
    support_history: List[Dict[str, Any]] = None
    satisfaction_scores: List[float] = None
    preferred_contact_method: str = "sms"
    escalation_triggers: List[str] = None
    
    # Sales agent profile
    lead_stage: str = "prospect"
    interest_level: float = 0.5
    budget_range: str = "unknown"
    decision_timeline: str = "unknown"
    pain_points: List[str] = None
    objections_history: List[str] = None
    
    # Emotional intelligence profile
    emotional_trend: List[float] = None  # Rolling sentiment average
    last_emotion: Optional[EmotionalState] = None
    tone_preference: str = "adaptive"  # casual, formal, adaptive
    frustration_count: int = 0
    engagement_score: float = 0.5
    
    # Shared profile data
    communication_preferences: Dict[str, Any] = None
    timezone: str = "UTC"
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.watchlist is None:
            self.watchlist = []
        if self.portfolio_insights is None:
            self.portfolio_insights = {}
        if self.support_history is None:
            self.support_history = []
        if self.satisfaction_scores is None:
            self.satisfaction_scores = []
        if self.escalation_triggers is None:
            self.escalation_triggers = []
        if self.pain_points is None:
            self.pain_points = []
        if self.objections_history is None:
            self.objections_history = []
        if self.emotional_trend is None:
            self.emotional_trend = []
        if self.communication_preferences is None:
            self.communication_preferences = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.last_emotion:
            data['last_emotion'] = self.last_emotion.to_dict()
        return data

class EmotionalStateTracker:
    """Advanced emotional intelligence and sentiment analysis"""
    
    def __init__(self):
        # Emotion detection patterns
        self.emotion_patterns = {
            EmotionType.FRUSTRATED: {
                'keywords': ['frustrated', 'annoying', 'terrible', 'awful', 'hate', 'worst', 'useless', 'stupid'],
                'phrases': ['not working', 'broken', 'waste of time', 'fed up', 'sick of'],
                'punctuation_weight': 2.0  # !! increases frustration
            },
            EmotionType.EXCITED: {
                'keywords': ['amazing', 'awesome', 'fantastic', 'excellent', 'love', 'perfect', 'great'],
                'phrases': ['cant wait', 'so excited', 'looking forward', 'this is great'],
                'punctuation_weight': 1.5  # ! increases excitement
            },
            EmotionType.CONFUSED: {
                'keywords': ['confused', 'unclear', 'understand', 'explain', 'help', 'lost'],
                'phrases': ['dont get it', 'not sure', 'how do i', 'what does', 'can you explain'],
                'punctuation_weight': 1.2  # ? increases confusion
            },
            EmotionType.CONFIDENT: {
                'keywords': ['confident', 'sure', 'certain', 'ready', 'definitely', 'absolutely'],
                'phrases': ['im ready', 'lets do it', 'go for it', 'sounds good'],
                'punctuation_weight': 1.0
            },
            EmotionType.ANXIOUS: {
                'keywords': ['worried', 'nervous', 'scared', 'concerned', 'risky', 'dangerous'],
                'phrases': ['not sure about', 'worried about', 'what if', 'too risky'],
                'punctuation_weight': 1.3
            }
        }
        
        # Sentiment words
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'awesome', 'perfect', 'love',
            'happy', 'satisfied', 'pleased', 'wonderful', 'fantastic', 'brilliant'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'frustrated',
            'angry', 'disappointed', 'annoying', 'useless', 'broken', 'stupid'
        }
        
        # Trading-specific emotion patterns
        self.trading_emotions = {
            'fear': ['scared', 'risky', 'loss', 'crash', 'drop', 'bear'],
            'greed': ['moon', 'rocket', 'huge gains', 'rich', 'millionaire'],
            'fomo': ['missing out', 'everyone buying', 'too late', 'wish i bought'],
            'confidence': ['bullish', 'strong buy', 'confident', 'easy money']
        }
    
    def analyze_emotion(self, content: str, direction: MessageDirection) -> EmotionalState:
        """
        Analyze emotional state from message content
        
        Args:
            content: Message content to analyze
            direction: USER or BOT message
            
        Returns:
            EmotionalState object with detected emotion
        """
        content_lower = content.lower()
        detected_emotions = {}
        triggers = []
        
        # Analyze each emotion type
        for emotion_type, patterns in self.emotion_patterns.items():
            score = 0.0
            emotion_triggers = []
            
            # Check keywords
            for keyword in patterns['keywords']:
                if keyword in content_lower:
                    score += 0.3
                    emotion_triggers.append(f"keyword:{keyword}")
            
            # Check phrases
            for phrase in patterns['phrases']:
                if phrase in content_lower:
                    score += 0.4
                    emotion_triggers.append(f"phrase:{phrase}")
            
            # Check punctuation intensity
            if emotion_type == EmotionType.FRUSTRATED:
                score += content.count('!') * 0.2
                score += content.count('!!') * 0.3
            elif emotion_type == EmotionType.EXCITED:
                score += content.count('!') * 0.15
            elif emotion_type == EmotionType.CONFUSED:
                score += content.count('?') * 0.2
            
            if score > 0:
                detected_emotions[emotion_type] = {
                    'score': min(score, 1.0),
                    'triggers': emotion_triggers
                }
        
        # Calculate sentiment
        sentiment_score = self._calculate_sentiment_score(content_lower)
        sentiment_label = self._get_sentiment_label(sentiment_score)
        
        # Determine primary emotion
        if detected_emotions:
            primary_emotion = max(detected_emotions.keys(), 
                                key=lambda e: detected_emotions[e]['score'])
            emotion_score = detected_emotions[primary_emotion]['score']
            triggers = detected_emotions[primary_emotion]['triggers']
        else:
            # Default based on sentiment
            if sentiment_score > 0.6:
                primary_emotion = EmotionType.POSITIVE
                emotion_score = sentiment_score
            elif sentiment_score < 0.4:
                primary_emotion = EmotionType.NEGATIVE
                emotion_score = 1.0 - sentiment_score
            else:
                primary_emotion = EmotionType.NEUTRAL
                emotion_score = 0.5
        
        return EmotionalState(
            emotion_type=primary_emotion,
            emotion_score=emotion_score,
            sentiment_label=sentiment_label,
            confidence=self._calculate_confidence(content, detected_emotions),
            timestamp=datetime.utcnow(),
            triggers=triggers
        )
    
    def _calculate_sentiment_score(self, content_lower: str) -> float:
        """Calculate sentiment score (0.0 = negative, 1.0 = positive)"""
        positive_count = sum(1 for word in self.positive_words if word in content_lower)
        negative_count = sum(1 for word in self.negative_words if word in content_lower)
        
        total_words = len(content_lower.split())
        if total_words == 0:
            return 0.5
        
        # Normalize scores
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        
        # Calculate final sentiment (0.0 to 1.0)
        if positive_ratio + negative_ratio == 0:
            return 0.5  # Neutral
        
        sentiment = positive_ratio / (positive_ratio + negative_ratio)
        return max(0.0, min(1.0, sentiment))
    
    def _get_sentiment_label(self, sentiment_score: float) -> str:
        """Convert sentiment score to label"""
        if sentiment_score > 0.6:
            return "positive"
        elif sentiment_score < 0.4:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_confidence(self, content: str, detected_emotions: Dict) -> float:
        """Calculate confidence in emotion detection"""
        base_confidence = 0.5
        
        # More text = higher confidence
        text_length_factor = min(len(content.split()) / 20, 0.3)
        
        # Strong emotion indicators = higher confidence
        emotion_strength = max([e['score'] for e in detected_emotions.values()]) if detected_emotions else 0
        
        # Multiple emotion indicators = lower confidence
        emotion_count_penalty = len(detected_emotions) * 0.1 if len(detected_emotions) > 1 else 0
        
        confidence = base_confidence + text_length_factor + (emotion_strength * 0.3) - emotion_count_penalty
        return max(0.1, min(0.95, confidence))
    
    def detect_escalation(self, emotional_history: List[EmotionalState]) -> bool:
        """
        Detect if user needs escalation based on emotional history
        
        Args:
            emotional_history: List of recent emotional states
            
        Returns:
            bool: True if escalation is needed
        """
        if len(emotional_history) < 3:
            return False
        
        # Check last 3 emotions for persistent frustration
        recent_emotions = emotional_history[-3:]
        frustration_count = sum(1 for emotion in recent_emotions 
                              if emotion.emotion_type == EmotionType.FRUSTRATED 
                              and emotion.emotion_score > 0.6)
        
        # Check for escalating negative sentiment
        sentiment_scores = [emotion.emotion_score if emotion.sentiment_label == "positive" 
                          else 1.0 - emotion.emotion_score for emotion in recent_emotions]
        
        # Escalate if 3+ frustrated messages or declining sentiment trend
        declining_sentiment = all(sentiment_scores[i] >= sentiment_scores[i+1] 
                                for i in range(len(sentiment_scores)-1))
        
        return frustration_count >= 3 or (declining_sentiment and sentiment_scores[-1] < 0.3)

class MemoryManager:
    """
    Enhanced memory management system with emotional intelligence,
    context awareness, and multi-agent support for SMS trading AI
    """
    
    def __init__(
        self,
        redis_url: str,
        mongodb_url: str,
        pinecone_api_key: str,
        pinecone_environment: str,
        openai_api_key: str,
        stm_limit: int = 15,
        summary_trigger: int = 10
    ):
        """Initialize Enhanced MemoryManager with emotional intelligence"""
        
        # Configuration
        self.stm_limit = stm_limit
        self.summary_trigger = summary_trigger
        
        # Initialize emotional intelligence
        self.emotion_tracker = EmotionalStateTracker()
        
        # Database connections (initialized in setup)
        self.redis_client = None
        self.mongo_client = None
        self.db = None
        self.pinecone_index = None
        self.openai_client = None
        
        # Connection strings
        self.redis_url = redis_url
        self.mongodb_url = mongodb_url
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_environment = pinecone_environment
        self.openai_api_key = openai_api_key
        
        # Enhanced topic extraction patterns
        self.stock_pattern = re.compile(r'\b[A-Z]{1,5}\b')
        
        # Trading terms
        self.trading_terms = {
            'buy', 'sell', 'hold', 'calls', 'puts', 'options', 'earnings', 
            'support', 'resistance', 'breakout', 'technical', 'fundamental',
            'portfolio', 'risk', 'volatility', 'dividend', 'swing', 'day trading'
        }
        
        # Customer service terms
        self.service_terms = {
            'help', 'support', 'issue', 'problem', 'bug', 'error', 'complaint',
            'refund', 'cancel', 'billing', 'account', 'password', 'login',
            'frustrated', 'angry', 'disappointed', 'confused', 'urgent'
        }
        
        # Sales terms
        self.sales_terms = {
            'price', 'cost', 'budget', 'discount', 'trial', 'demo', 'features',
            'comparison', 'competitor', 'upgrade', 'premium', 'subscription',
            'interested', 'considering', 'evaluate', 'decision', 'timeline'
        }
        
        # Context caching for performance
        self.context_cache = {}
        self.semantic_cache = {}
        
    async def setup(self):
        """Initialize all database connections and emotional intelligence"""
        try:
            # Redis connection
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            logger.info("Redis connection established")
            
            # MongoDB connection
            self.mongo_client = AsyncIOMotorClient(self.mongodb_url)
            self.db = self.mongo_client.trading_ai_memory
            
            # Test MongoDB connection
            await self.db.command("ping")
            logger.info("MongoDB connection established")
            
            # Pinecone connection
            pinecone.init(
                api_key=self.pinecone_api_key,
                environment=self.pinecone_environment
            )
            
            # Create or connect to index
            index_name = "trading-memory-enhanced"
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=1536,  # OpenAI embedding dimension
                    metric="cosine"
                )
            
            self.pinecone_index = pinecone.Index(index_name)
            logger.info("Pinecone connection established")
            
            # OpenAI client
            self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
            logger.info("OpenAI client initialized")
            
            # Create MongoDB indexes for optimization
            await self._create_mongodb_indexes()
            
            logger.info("Enhanced MemoryManager with Emotional Intelligence ready!")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise
    
    async def save_message(
        self, 
        user_id: str, 
        content: str, 
        direction: MessageDirection,
        agent_type: AgentType = AgentType.TRADING,
        topics: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Save message with emotional analysis and intelligent context tracking
        
        Args:
            user_id: Unique user identifier
            content: Message content
            direction: USER or BOT
            agent_type: Which agent is handling this message
            topics: Optional list of topics/symbols
            metadata: Agent-specific metadata
            
        Returns:
            bool: Success status
        """
        try:
            # Analyze emotional state
            emotional_state = self.emotion_tracker.analyze_emotion(content, direction)
            
            # Create enhanced message object
            message = Message(
                user_id=user_id,
                content=content,
                direction=direction,
                agent_type=agent_type,
                timestamp=datetime.utcnow(),
                topics=topics or self._extract_topics(content, agent_type),
                importance_score=self._calculate_importance(content, agent_type, emotional_state),
                metadata=metadata or {},
                emotional_state=emotional_state
            )
            
            # Classify message type based on agent and content
            message.message_type = self._classify_message_type(content, agent_type)
            
            # Agent-specific Redis key for session memory
            stm_key = f"session:{agent_type.value}:{user_id}"
            
            # Add message to Redis list (FIFO)
            await self.redis_client.lpush(stm_key, json.dumps(message.to_dict()))
            
            # Maintain STM limit
            await self.redis_client.ltrim(stm_key, 0, self.stm_limit - 1)
            
            # Set expiration (24 hours)
            await self.redis_client.expire(stm_key, 86400)
            
            # Store emotional state in Redis for trend tracking
            await self._store_emotional_state(user_id, emotional_state)
            
            # Update user profile with emotional insights
            await self._update_emotional_profile(user_id, emotional_state)
            
            # Check for escalation needs
            await self._check_escalation_needs(user_id, emotional_state)
            
            # Get current message count for this agent
            message_count = await self.redis_client.llen(stm_key)
            
            # Trigger summarization if threshold reached
            if message_count >= self.summary_trigger:
                await self._trigger_summarization(user_id, agent_type)
            
            # Store important messages in vector database
            if message.importance_score > 0.7 or message.message_type in [
                MemoryType.IMPORTANT, MemoryType.CUSTOMER_SERVICE, 
                MemoryType.SALES, MemoryType.EMOTIONAL
            ]:
                await self._store_in_vector_db(
                    user_id, content, message.topics, 
                    f"{agent_type.value}_message", agent_type, emotional_state
                )
            
            logger.info(f"Enhanced message saved for user {user_id} via {agent_type.value} agent, "
                       f"emotion: {emotional_state.emotion_type.value}, count: {message_count}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save enhanced message: {e}")
            return False
    
    async def get_context(
        self, 
        user_id: str, 
        agent_type: AgentType = AgentType.TRADING,
        query: str = None,
        include_cross_agent: bool = True,
        include_emotional_context: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve comprehensive context with emotional intelligence and 3-layer memory
        
        Args:
            user_id: Unique user identifier
            agent_type: Which agent is requesting context
            query: Optional query for semantic search
            include_cross_agent: Whether to include memories from other agents
            include_emotional_context: Whether to include emotional analysis
            
        Returns:
            Dict containing enhanced context with emotional intelligence
        """
        try:
            # Check cache first
            cache_key = f"context:{user_id}:{agent_type.value}:{query or 'general'}"
            cached_context = await self._get_cached_context(cache_key)
            if cached_context:
                return cached_context
            
            # Layer 1: Short-term memory (Redis) - recent messages
            stm_task = self._get_short_term_memory(user_id, agent_type)
            
            # Layer 2: Session summaries (MongoDB) - summarized conversations  
            summaries_task = self._get_conversation_summaries(user_id, agent_type, limit=5)
            
            # Layer 3: Enhanced user profile
            profile_task = self._get_user_profile(user_id)
            
            # Emotional context
            if include_emotional_context:
                emotional_task = self._get_emotional_context(user_id)
            else:
                emotional_task = asyncio.create_task(self._get_empty_result())
            
            # Cross-agent context if requested
            if include_cross_agent:
                cross_agent_task = self._get_cross_agent_context(user_id, agent_type)
            else:
                cross_agent_task = asyncio.create_task(self._get_empty_result())
            
            # Layer 3: Enhanced semantic search with emotional weighting
            if query:
                vector_task = self._enhanced_vector_search(user_id, query, agent_type, top_k=3)
            else:
                vector_task = asyncio.create_task(self._get_empty_vector_result())
            
            # Wait for all tasks
            stm, summaries, profile, emotional_context, cross_agent_data, vector_results = await asyncio.gather(
                stm_task, summaries_task, profile_task, emotional_task, cross_agent_task, vector_task,
                return_exceptions=True
            )
            
            # Handle exceptions gracefully
            if isinstance(stm, Exception):
                logger.warning(f"STM retrieval failed: {stm}")
                stm = []
            
            if isinstance(summaries, Exception):
                logger.warning(f"Summaries retrieval failed: {summaries}")
                summaries = []
            
            if isinstance(profile, Exception):
                logger.warning(f"Profile retrieval failed: {profile}")
                profile = {}
            
            if isinstance(emotional_context, Exception):
                logger.warning(f"Emotional context retrieval failed: {emotional_context}")
                emotional_context = {}
            
            if isinstance(cross_agent_data, Exception):
                logger.warning(f"Cross-agent data retrieval failed: {cross_agent_data}")
                cross_agent_data = {}
            
            if isinstance(vector_results, Exception):
                logger.warning(f"Vector search failed: {vector_results}")
                vector_results = []
            
            # Apply relevance weighting based on emotional state and importance
            weighted_memories = self._apply_relevance_weighting(
                vector_results, emotional_context, profile
            )
            
            # Compile enhanced context with 3-layer architecture
            context = {
                "agent_type": agent_type.value,
                
                # Layer 1: Immediate memory
                "short_term_memory": stm,
                
                # Layer 2: Session intelligence
                "conversation_summaries": summaries,
                
                # Layer 3: Long-term and semantic memory
                "user_profile": self._filter_profile_for_agent(profile, agent_type),
                "relevant_memories": weighted_memories,
                
                # Enhanced context layers
                "emotional_context": emotional_context,
                "cross_agent_insights": cross_agent_data,
                
                # Context metadata
                "context_metadata": {
                    "primary_agent": agent_type.value,
                    "stm_count": len(stm),
                    "summary_count": len(summaries),
                    "has_profile": bool(profile),
                    "has_emotional_context": bool(emotional_context),
                    "cross_agent_data": bool(cross_agent_data),
                    "vector_matches": len(weighted_memories),
                    "timestamp": datetime.utcnow().isoformat(),
                    "context_layers": 3
                }
            }
            
            # Intelligent context compression if needed
            context = await self._intelligent_context_compression(context)
            
            # Cache context for performance
            await self._cache_context(cache_key, context)
            
            logger.info(f"Enhanced context retrieved for user {user_id} via {agent_type.value} agent")
            return context
            
        except Exception as e:
            logger.error(f"Failed to get enhanced context: {e}")
            return {"error": str(e)}
    
    async def generate_contextual_prompt(
        self,
        context: Dict[str, Any],
        emotional_state: EmotionalState,
        user_profile: Dict[str, Any],
        agent_type: AgentType
    ) -> str:
        """
        Generate tone-optimized prompt based on emotional state and context
        
        Args:
            context: Full context from get_context()
            emotional_state: Current emotional state
            user_profile: User profile data
            agent_type: Current agent type
            
        Returns:
            str: Optimized prompt for LLM
        """
        try:
            # Base prompt components
            base_context = f"""
You are a {agent_type.value.replace('_', ' ')} AI agent for a personalized SMS trading platform.

USER CONTEXT:
- Communication Preference: {user_profile.get('tone_preference', 'adaptive')}
- Trading Style: {user_profile.get('trading_style', 'unknown')}
- Risk Tolerance: {user_profile.get('risk_tolerance', 'unknown')}
"""
            
            # Emotional adaptation
            emotional_guidance = ""
            
            if emotional_state.emotion_type == EmotionType.FRUSTRATED:
                emotional_guidance = """
EMOTIONAL STATE: User is frustrated (score: {:.2f})
RESPONSE TONE: Be empathetic, acknowledge their frustration, provide clear step-by-step solutions.
- Use phrases like "I understand your frustration" or "Let me help you resolve this quickly"
- Avoid technical jargon, be direct and solution-focused
- Offer escalation if needed
""".format(emotional_state.emotion_score)
                
            elif emotional_state.emotion_type == EmotionType.EXCITED:
                emotional_guidance = """
EMOTIONAL STATE: User is excited (score: {:.2f})
RESPONSE TONE: Match their enthusiasm while providing grounded advice.
- Use positive language and confirm their excitement
- Provide balanced perspective to prevent overconfidence
- Encourage but add risk awareness
""".format(emotional_state.emotion_score)
                
            elif emotional_state.emotion_type == EmotionType.CONFUSED:
                emotional_guidance = """
EMOTIONAL STATE: User is confused (score: {:.2f})
RESPONSE TONE: Be patient, educational, and structured.
- Break down complex concepts into simple steps
- Use analogies and examples
- Ask clarifying questions to understand their confusion
- Provide educational resources
""".format(emotional_state.emotion_score)
                
            elif emotional_state.emotion_type == EmotionType.ANXIOUS:
                emotional_guidance = """
EMOTIONAL STATE: User is anxious (score: {:.2f})
RESPONSE TONE: Be reassuring and focus on risk management.
- Acknowledge their concerns as valid
- Provide data-driven reassurance
- Focus on risk mitigation strategies
- Suggest conservative approaches
""".format(emotional_state.emotion_score)
            
            # Context integration
            context_summary = ""
            if context.get('relevant_memories'):
                relevant_topics = [mem.get('topics', []) for mem in context['relevant_memories']]
                flat_topics = [topic for sublist in relevant_topics for topic in sublist]
                top_topics = list(set(flat_topics))[:5]
                context_summary = f"\nRELEVANT CONTEXT: Previous discussions about {', '.join(top_topics)}"
            
            # Conversation history
            recent_messages = context.get('short_term_memory', [])[:3]
            if recent_messages:
                context_summary += f"\nRECENT CONVERSATION: {len(recent_messages)} messages in current session"
            
            # Combine all components
            full_prompt = f"""
{base_context}
{emotional_guidance}
{context_summary}

INSTRUCTIONS:
1. Respond in a tone that matches the user's emotional state
2. Reference relevant context from previous conversations when helpful
3. Keep responses concise for SMS (under 160 characters when possible)
4. Always prioritize user safety and financial responsibility
5. If user seems frustrated or confused, offer additional help

Remember: You're having a conversation via SMS, so be conversational and helpful.
"""
            
            return full_prompt.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate contextual prompt: {e}")
            return "You are a helpful AI assistant. Please provide a helpful response."
    
    async def get_user_engagement_metrics(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive user engagement metrics with emotional intelligence
        
        Args:
            user_id: User to analyze
            
        Returns:
            Dict with engagement metrics
        """
        try:
            # Get emotional history
            emotional_states = await self._get_emotional_history(user_id, days=30)
            
            # Calculate metrics
            sentiment_scores = [state.emotion_score for state in emotional_states]
            avg_sentiment = statistics.mean(sentiment_scores) if sentiment_scores else 0.5
            
            # Frustration analysis
            frustration_events = [state for state in emotional_states 
                                if state.emotion_type == EmotionType.FRUSTRATED]
            
            # Engagement calculation
            total_messages = len(emotional_states)
            positive_interactions = sum(1 for state in emotional_states 
                                      if state.sentiment_label == "positive")
            engagement_rate = positive_interactions / max(total_messages, 1)
            
            # Trading-specific metrics
            trading_insights = await self._get_trading_insights_consumed(user_id, days=30)
            
            return {
                "user_id": user_id,
                "avg_sentiment_30d": avg_sentiment,
                "frustration_triggers": len(frustration_events),
                "engagement_rate": engagement_rate,
                "total_interactions": total_messages,
                "trading_insights_consumed": len(trading_insights),
                "emotional_trend": sentiment_scores[-7:] if len(sentiment_scores) >= 7 else sentiment_scores,
                "last_emotion": emotional_states[-1].emotion_type.value if emotional_states else "unknown",
                "analysis_period": "30 days"
            }
            
        except Exception as e:
            logger.error(f"Failed to get user engagement metrics: {e}")
            return {"error": str(e)}
    
    async def get_global_metrics(self) -> Dict[str, Any]:
        """
        Get platform-wide analytics and insights
        
        Returns:
            Dict with global metrics
        """
        try:
            # User engagement metrics
            total_users = await self.db.users.count_documents({})
            
            # Sentiment analysis across platform
            recent_conversations = await self.db.conversations.find({
                "timestamp": {"$gte": datetime.utcnow() - timedelta(days=7)}
            }).to_list(1000)
            
            avg_sentiment = statistics.mean([
                conv.get('avg_sentiment_score', 0.5) for conv in recent_conversations
            ]) if recent_conversations else 0.5
            
            # Top trading symbols
            trading_insights = await self.db.trade_insights.find({
                "timestamp": {"$gte": datetime.utcnow() - timedelta(days=7)}
            }).to_list(1000)
            
            all_symbols = []
            for insight in trading_insights:
                all_symbols.extend(insight.get('symbols', []))
            
            symbol_counts = defaultdict(int)
            for symbol in all_symbols:
                symbol_counts[symbol] += 1
            
            top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Escalation metrics
            escalations = await self.db.customer_escalations.find({
                "timestamp": {"$gte": datetime.utcnow() - timedelta(days=7)}
            }).to_list(1000)
            
            return {
                "platform_metrics": {
                    "total_users": total_users,
                    "avg_sentiment_7d": avg_sentiment,
                    "total_conversations_7d": len(recent_conversations),
                    "total_escalations_7d": len(escalations)
                },
                "trading_metrics": {
                    "top_symbols_7d": top_symbols,
                    "total_trading_insights": len(trading_insights),
                    "avg_insights_per_user": len(trading_insights) / max(total_users, 1)
                },
                "emotional_metrics": {
                    "engagement_score": avg_sentiment,
                    "escalation_rate": len(escalations) / max(len(recent_conversations), 1)
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get global metrics: {e}")
            return {"error": str(e)}
    
    # Enhanced helper methods with emotional intelligence
    
    async def _store_emotional_state(self, user_id: str, emotional_state: EmotionalState):
        """Store emotional state in Redis for trend tracking"""
        try:
            emotion_key = f"emotions:{user_id}"
            await self.redis_client.lpush(emotion_key, json.dumps(emotional_state.to_dict()))
            await self.redis_client.ltrim(emotion_key, 0, 19)  # Keep last 20 emotions
            await self.redis_client.expire(emotion_key, 86400 * 7)  # 7 days
        except Exception as e:
            logger.error(f"Failed to store emotional state: {e}")
    
    async def _update_emotional_profile(self, user_id: str, emotional_state: EmotionalState):
        """Update user profile with emotional insights"""
        try:
            # Get current emotional trend
            emotion_key = f"emotions:{user_id}"
            emotions_json = await self.redis_client.lrange(emotion_key, 0, 9)  # Last 10
            
            sentiment_scores = []
            for emotion_json in emotions_json:
                try:
                    emotion_data = json.loads(emotion_json)
                    score = emotion_data.get('emotion_score', 0.5)
                    if emotion_data.get('sentiment_label') == 'positive':
                        sentiment_scores.append(score)
                    elif emotion_data.get('sentiment_label') == 'negative':
                        sentiment_scores.append(1.0 - score)
                    else:
                        sentiment_scores.append(0.5)
                except:
                    continue
            
            # Calculate rolling average
            rolling_avg = statistics.mean(sentiment_scores) if sentiment_scores else 0.5
            
            # Update profile
            updates = {
                "last_emotion": emotional_state.to_dict(),
                "emotional_trend": sentiment_scores,
                "engagement_score": rolling_avg,
                "updated_at": datetime.utcnow()
            }
            
            # Increment frustration count if frustrated
            if emotional_state.emotion_type == EmotionType.FRUSTRATED:
                current_profile = await self.db.users.find_one({"user_id": user_id})
                frustration_count = current_profile.get('frustration_count', 0) + 1 if current_profile else 1
                updates["frustration_count"] = frustration_count
            
            await self.db.users.update_one(
                {"user_id": user_id},
                {"$set": updates},
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"Failed to update emotional profile: {e}")
    
    async def _check_escalation_needs(self, user_id: str, emotional_state: EmotionalState):
        """Check if user needs escalation based on emotional state"""
        try:
            if emotional_state.emotion_type == EmotionType.FRUSTRATED and emotional_state.emotion_score > 0.7:
                # Get recent emotional history
                emotion_key = f"emotions:{user_id}"
                emotions_json = await self.redis_client.lrange(emotion_key, 0, 2)  # Last 3
                
                emotional_history = []
                for emotion_json in emotions_json:
                    try:
                        emotion_data = json.loads(emotion_json)
                        emotion_obj = EmotionalState(
                            emotion_type=EmotionType(emotion_data['emotion_type']),
                            emotion_score=emotion_data['emotion_score'],
                            sentiment_label=emotion_data['sentiment_label'],
                            confidence=emotion_data['confidence'],
                            timestamp=datetime.fromisoformat(emotion_data['timestamp']),
                            triggers=emotion_data.get('triggers', [])
                        )
                        emotional_history.append(emotion_obj)
                    except:
                        continue
                
                # Check for escalation
                if self.emotion_tracker.detect_escalation(emotional_history):
                    await self.store_customer_escalation(
                        user_id,
                        "User showing persistent frustration pattern",
                        f"Emotional escalation detected: {emotional_state.triggers}",
                        "high"
                    )
                    logger.warning(f"Escalation triggered for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to check escalation needs: {e}")
    
    async def _get_emotional_context(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive emotional context for user"""
        try:
            emotion_key = f"emotions:{user_id}"
            emotions_json = await self.redis_client.lrange(emotion_key, 0, 9)  # Last 10
            
            emotional_history = []
            for emotion_json in emotions_json:
                try:
                    emotion_data = json.loads(emotion_json)
                    emotional_history.append(emotion_data)
                except:
                    continue
            
            if not emotional_history:
                return {}
            
            # Analyze trends
            recent_emotions = [EmotionType(e['emotion_type']) for e in emotional_history[:5]]
            sentiment_trend = [e['emotion_score'] for e in emotional_history[:5]]
            
            # Calculate emotional stability
            stability_score = 1.0 - (statistics.stdev(sentiment_trend) if len(sentiment_trend) > 1 else 0)
            
            return {
                "current_emotion": emotional_history[0] if emotional_history else None,
                "emotional_trend": emotional_history[:5],
                "dominant_emotions": [e.value for e in set(recent_emotions)],
                "emotional_stability": stability_score,
                "needs_escalation": self.emotion_tracker.detect_escalation([
                    EmotionalState(
                        emotion_type=EmotionType(e['emotion_type']),
                        emotion_score=e['emotion_score'],
                        sentiment_label=e['sentiment_label'],
                        confidence=e['confidence'],
                        timestamp=datetime.fromisoformat(e['timestamp']),
                        triggers=e.get('triggers', [])
                    ) for e in emotional_history[:3]
                ])
            }
            
        except Exception as e:
            logger.error(f"Failed to get emotional context: {e}")
            return {}
    
    async def _enhanced_vector_search(
        self,
        user_id: str,
        query: str,
        agent_type: AgentType,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Enhanced vector search with emotional weighting"""
        try:
            # Check semantic cache first
            cache_key = f"semantic:{user_id}:{query}:{agent_type.value}"
            cached_results = self.semantic_cache.get(cache_key)
            if cached_results:
                return cached_results
            
            # Generate embedding for query
            embedding = await self._get_embedding(query)
            
            # Search Pinecone with enhanced filters
            search_filter = {"user_id": user_id}
            
            # Include agent-specific memories and cross-agent important ones
            agent_filter = {
                "$or": [
                    {"agent_type": agent_type.value},
                    {"memory_type": {"$in": ["important", "trade_insight", "customer_escalation", "sales_opportunity", "emotional"]}}
                ]
            }
            search_filter.update(agent_filter)
            
            results = self.pinecone_index.query(
                vector=embedding,
                top_k=top_k * 3,  # Get more results for intelligent filtering
                filter=search_filter,
                include_metadata=True
            )
            
            # Enhanced result processing with emotional weighting
            formatted_results = []
            for match in results.matches:
                result = {
                    "content": match.metadata.get("content", ""),
                    "topics": match.metadata.get("topics", []),
                    "timestamp": match.metadata.get("timestamp", ""),
                    "relevance_score": float(match.score),
                    "memory_type": match.metadata.get("memory_type", "unknown"),
                    "agent_type": match.metadata.get("agent_type", "unknown"),
                    "emotional_weight": match.metadata.get("emotional_weight", 1.0)
                }
                
                # Apply relevance boosting
                if result["agent_type"] == agent_type.value:
                    result["relevance_score"] *= 1.2
                
                # Boost emotional memories
                if result["memory_type"] == "emotional":
                    result["relevance_score"] *= 1.3
                
                # Boost high-importance memories
                emotional_weight = result.get("emotional_weight", 1.0)
                result["relevance_score"] *= emotional_weight
                
                formatted_results.append(result)
            
            # Sort by enhanced relevance and return top_k
            formatted_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            final_results = formatted_results[:top_k]
            
            # Cache results
            self.semantic_cache[cache_key] = final_results
            
            return final_results
            
        except Exception as e:
            logger.error(f"Enhanced vector search failed: {e}")
            return []
    
    async def _apply_relevance_weighting(
        self,
        memories: List[Dict[str, Any]],
        emotional_context: Dict[str, Any],
        profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply intelligent relevance weighting based on emotional state and user profile"""
        try:
            if not memories:
                return memories
            
            # Get current emotional state
            current_emotion = emotional_context.get('current_emotion')
            if not current_emotion:
                return memories
            
            current_emotion_type = current_emotion.get('emotion_type', 'neutral')
            
            # Apply weighting based on emotional relevance
            for memory in memories:
                base_score = memory['relevance_score']
                
                # Boost memories related to current emotional state
                if memory['memory_type'] == 'emotional':
                    memory['relevance_score'] = base_score * 1.4
                
                # Boost frustration-related memories if user is frustrated
                if current_emotion_type == 'frustrated' and memory['memory_type'] == 'customer_service':
                    memory['relevance_score'] = base_score * 1.3
                
                # Boost trading memories if user is confident/excited about trading
                if (current_emotion_type in ['confident', 'excited'] and 
                    memory['memory_type'] == 'trade_insight'):
                    memory['relevance_score'] = base_score * 1.25
            
            # Re-sort by updated relevance
            memories.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to apply relevance weighting: {e}")
            return memories
    
    async def _intelligent_context_compression(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligent context compression prioritizing emotional and high-importance content"""
        try:
            # Estimate token count
            context_str = json.dumps(context)
            estimated_tokens = len(context_str) / 4
            
            if estimated_tokens <= 3000:
                return context
            
            compressed = context.copy()
            
            # Compression priority: keep emotional and important content
            # 1. Reduce STM, but keep emotional messages
            stm = context.get("short_term_memory", [])
            if len(stm) > 8:
                # Keep emotional messages and recent important ones
                emotional_messages = [msg for msg in stm if msg.get('emotional_state')]
                important_messages = [msg for msg in stm if msg.get('importance_score', 0) > 0.7]
                recent_messages = stm[:3]  # Always keep 3 most recent
                
                # Combine and deduplicate
                keep_messages = []
                seen_ids = set()
                for msg_list in [recent_messages, emotional_messages, important_messages]:
                    for msg in msg_list:
                        msg_id = f"{msg.get('timestamp', '')}_{msg.get('content', '')[:50]}"
                        if msg_id not in seen_ids:
                            keep_messages.append(msg)
                            seen_ids.add(msg_id)
                            if len(keep_messages) >= 8:
                                break
                    if len(keep_messages) >= 8:
                        break
                
                compressed["short_term_memory"] = keep_messages[:8]
            
            # 2. Reduce summaries, prioritize recent and high-importance
            summaries = context.get("conversation_summaries", [])
            if len(summaries) > 3:
                # Sort by importance and recency
                sorted_summaries = sorted(summaries, 
                                        key=lambda s: (s.get('importance_score', 0), s.get('timestamp', '')), 
                                        reverse=True)
                compressed["conversation_summaries"] = sorted_summaries[:3]
            
            # 3. Keep top 2 relevant memories with highest emotional relevance
            memories = context.get("relevant_memories", [])
            if len(memories) > 2:
                sorted_memories = sorted(memories, 
                                       key=lambda m: (m.get('emotional_weight', 1.0) * m.get('relevance_score', 0)),
                                       reverse=True)
                compressed["relevant_memories"] = sorted_memories[:2]
            
            # 4. Always preserve emotional context (it's small but crucial)
            # Emotional context is kept as-is
            
            # Mark as compressed
            compressed["context_metadata"]["compressed"] = True
            compressed["context_metadata"]["compression_strategy"] = "emotional_priority"
            
            return compressed
            
        except Exception as e:
            logger.error(f"Context compression failed: {e}")
            return context
    
    async def _cache_context(self, cache_key: str, context: Dict[str, Any]):
        """Cache context for performance with TTL"""
        try:
            # Cache for 5 minutes
            await self.redis_client.setex(
                f"context_cache:{cache_key}",
                300,  # 5 minutes
                json.dumps(context, default=str)
            )
        except Exception as e:
            logger.error(f"Failed to cache context: {e}")
    
    async def _get_cached_context(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached context if available"""
        try:
            cached = await self.redis_client.get(f"context_cache:{cache_key}")
            if cached:
                return json.loads(cached)
            return None
        except Exception as e:
            logger.error(f"Failed to get cached context: {e}")
            return None
    
    # Include all existing helper methods with enhancements...
    # (Previous helper methods with emotional intelligence integration)
    
    async def _calculate_importance(
        self, 
        content: str, 
        agent_type: AgentType = AgentType.TRADING,
        emotional_state: Optional[EmotionalState] = None
    ) -> float:
        """Enhanced importance calculation including emotional factors"""
        importance = 0.3  # Base score
        content_lower = content.lower()
        
        # Apply existing importance logic
        if agent_type == AgentType.TRADING:
            high_importance_terms = [
                'buy', 'sell', 'portfolio', 'risk', 'loss', 'profit', 'earnings',
                'recommendation', 'strategy', 'analysis', 'alert', 'breakout'
            ]
            
            for term in high_importance_terms:
                if term in content_lower:
                    importance += 0.1
            
            symbols = self.stock_pattern.findall(content.upper())
            importance += len(symbols) * 0.05
            
        elif agent_type == AgentType.CUSTOMER_SERVICE:
            high_importance_terms = [
                'frustrated', 'angry', 'disappointed', 'urgent', 'critical', 
                'escalate', 'manager', 'refund', 'cancel', 'lawsuit', 'complaint'
            ]
            
            for term in high_importance_terms:
                if term in content_lower:
                    importance += 0.15
                    
            if any(word in content_lower for word in ['hate', 'terrible', 'worst', 'horrible']):
                importance += 0.2
                
        elif agent_type == AgentType.SALES:
            high_importance_terms = [
                'buy', 'purchase', 'budget', 'decision', 'timeline', 'competitor',
                'interested', 'considering', 'evaluate', 'demo', 'trial'
            ]
            
            for term in high_importance_terms:
                if term in content_lower:
                    importance += 0.12
                    
            if any(word in content_lower for word in ['ready to buy', 'place order', 'sign up']):
                importance += 0.3
        
        # Emotional importance boosting
        if emotional_state:
            if emotional_state.emotion_type in [EmotionType.FRUSTRATED, EmotionType.ANXIOUS]:
                importance += 0.2  # High emotional states are important
            elif emotional_state.emotion_type == EmotionType.EXCITED:
                importance += 0.15  # Excitement indicates engagement
            
            # High emotion scores increase importance
            importance += emotional_state.emotion_score * 0.1
        
        # Question marks increase importance
        importance += content.count('?') * 0.05
        
        return min(importance, 1.0)
    
    async def _store_in_vector_db(
        self, 
        user_id: str, 
        content: str, 
        topics: List[str], 
        memory_type: str = "message",
        agent_type: AgentType = AgentType.TRADING,
        emotional_state: Optional[EmotionalState] = None
    ):
        """Enhanced vector storage with emotional weighting"""
        try:
            # Generate embedding
            embedding = await self._get_embedding(content)
            
            # Create unique ID
            content_id = hashlib.md5(
                f"{user_id}_{agent_type.value}_{content}_{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()
            
            # Calculate emotional weight
            emotional_weight = 1.0
            if emotional_state:
                if emotional_state.emotion_type in [EmotionType.FRUSTRATED, EmotionType.EXCITED]:
                    emotional_weight = 1.0 + emotional_state.emotion_score * 0.5
                elif emotional_state.emotion_type in [EmotionType.ANXIOUS, EmotionType.CONFUSED]:
                    emotional_weight = 1.0 + emotional_state.emotion_score * 0.3
            
            # Prepare enhanced metadata
            metadata = {
                "user_id": user_id,
                "agent_type": agent_type.value,
                "content": content[:1000],
                "topics": topics[:10],
                "memory_type": memory_type,
                "timestamp": datetime.utcnow().isoformat(),
                "emotional_weight": emotional_weight
            }
            
            # Add emotional metadata if available
            if emotional_state:
                metadata.update({
                    "emotion_type": emotional_state.emotion_type.value,
                    "emotion_score": emotional_state.emotion_score,
                    "sentiment_label": emotional_state.sentiment_label
                })
            
            # Upsert to Pinecone
            self.pinecone_index.upsert(
                vectors=[(content_id, embedding, metadata)]
            )
            
            logger.debug(f"Enhanced content stored in vector DB for user {user_id} via {agent_type.value}")
            
        except Exception as e:
            logger.error(f"Enhanced vector storage failed: {e}")
    
    # Add remaining helper methods from previous implementation...
    # (Include all the methods from the previous version with emotional enhancements)
    
    async def _get_empty_result(self) -> Dict:
        """Return empty result"""
        return {}
    
    async def _get_empty_vector_result(self) -> List:
        """Return empty vector result"""
        return []

    async def _generate_summary(self, messages: List[Dict[str, Any]], agent_type: AgentType) -> str:
        """Generate LLM-based summary of conversation for specific agent"""
        try:
            # Format messages for LLM
            conversation_text = "\n".join([
                f"{msg['direction']}: {msg['content']}" 
                for msg in messages[-10:]  # Last 10 messages
            ])
            
            # Agent-specific prompts
            if agent_type == AgentType.TRADING:
                system_prompt = """You are a trading conversation summarizer. Create a concise summary focusing on:
                1. Main trading topics discussed (stocks, options, market analysis)
                2. User's trading preferences, risk tolerance, or concerns
                3. Any specific stocks, strategies, or recommendations mentioned
                4. Key insights or market analysis provided
                
                Keep the summary under 150 words and focus on actionable trading information."""
                
            elif agent_type == AgentType.CUSTOMER_SERVICE:
                system_prompt = """You are a customer service conversation summarizer. Create a concise summary focusing on:
                1. Main issue or problem the customer is experiencing
                2. Customer's emotional state and satisfaction level
                3. Resolution steps taken or needed
                4. Any escalation triggers or urgent concerns
                
                Keep the summary under 150 words and focus on customer satisfaction and issue resolution."""
                
            elif agent_type == AgentType.SALES:
                system_prompt = """You are a sales conversation summarizer. Create a concise summary focusing on:
                1. Customer's interest level and buying intent
                2. Budget, timeline, and decision-making process
                3. Pain points, objections, or concerns raised
                4. Opportunities for upselling or feature interest
                
                Keep the summary under 150 words and focus on sales pipeline advancement."""
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Summarize this conversation:\n\n{conversation_text}"}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Summary generation failed for {agent_type.value}: {e}")
            return ""
    
    async def _store_in_vector_db(
        self, 
        user_id: str, 
        content: str, 
        topics: List[str], 
        memory_type: str = "message",
        agent_type: AgentType = AgentType.TRADING,
        emotional_state: Optional[EmotionalState] = None
    ):
        """Enhanced vector storage with emotional weighting"""
        try:
            # Generate embedding
            embedding = await self._get_embedding(content)
            
            # Create unique ID
            content_id = hashlib.md5(
                f"{user_id}_{agent_type.value}_{content}_{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()
            
            # Calculate emotional weight
            emotional_weight = 1.0
            if emotional_state:
                if emotional_state.emotion_type in [EmotionType.FRUSTRATED, EmotionType.EXCITED]:
                    emotional_weight = 1.0 + emotional_state.emotion_score * 0.5
                elif emotional_state.emotion_type in [EmotionType.ANXIOUS, EmotionType.CONFUSED]:
                    emotional_weight = 1.0 + emotional_state.emotion_score * 0.3
            
            # Prepare enhanced metadata
            metadata = {
                "user_id": user_id,
                "agent_type": agent_type.value,
                "content": content[:1000],
                "topics": topics[:10],
                "memory_type": memory_type,
                "timestamp": datetime.utcnow().isoformat(),
                "emotional_weight": emotional_weight
            }
            
            # Add emotional metadata if available
            if emotional_state:
                metadata.update({
                    "emotion_type": emotional_state.emotion_type.value,
                    "emotion_score": emotional_state.emotion_score,
                    "sentiment_label": emotional_state.sentiment_label
                })
            
            # Upsert to Pinecone
            self.pinecone_index.upsert(
                vectors=[(content_id, embedding, metadata)]
            )
            
            logger.debug(f"Enhanced content stored in vector DB for user {user_id} via {agent_type.value}")
            
        except Exception as e:
            logger.error(f"Enhanced vector storage failed: {e}")
    
    async def _get_emotional_history(self, user_id: str, days: int = 30) -> List[EmotionalState]:
        """Get emotional history for user"""
        try:
            emotion_key = f"emotions:{user_id}"
            emotions_json = await self.redis_client.lrange(emotion_key, 0, -1)
            
            emotional_history = []
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            for emotion_json in emotions_json:
                try:
                    emotion_data = json.loads(emotion_json)
                    timestamp = datetime.fromisoformat(emotion_data['timestamp'])
                    
                    if timestamp >= cutoff_date:
                        emotion_obj = EmotionalState(
                            emotion_type=EmotionType(emotion_data['emotion_type']),
                            emotion_score=emotion_data['emotion_score'],
                            sentiment_label=emotion_data['sentiment_label'],
                            confidence=emotion_data['confidence'],
                            timestamp=timestamp,
                            triggers=emotion_data.get('triggers', [])
                        )
                        emotional_history.append(emotion_obj)
                except:
                    continue
            
            return emotional_history
            
        except Exception as e:
            logger.error(f"Failed to get emotional history: {e}")
            return []

    async def _get_trading_insights_consumed(self, user_id: str, days: int = 30) -> List[Dict]:
        """Get trading insights consumed by user"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            cursor = self.db.trade_insights.find({
                "user_id": user_id,
                "timestamp": {"$gte": cutoff_date}
            })
            
            insights = []
            async for doc in cursor:
                insights.append(doc)
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get trading insights: {e}")
            return []
    
    async def get_agent_analytics(self, agent_type: AgentType, days: int = 30) -> Dict[str, Any]:
        """
        Get analytics for specific agent type
        
        Args:
            agent_type: Agent to analyze
            days: Number of days to analyze
            
        Returns:
            Dict with analytics data
        """
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            if agent_type == AgentType.TRADING:
                # Trading analytics
                pipeline = [
                    {"$match": {"agent_type": "trading", "timestamp": {"$gte": start_date}}},
                    {"$group": {
                        "_id": None,
                        "total_insights": {"$sum": 1},
                        "avg_importance": {"$avg": "$importance_score"},
                        "top_symbols": {"$push": "$symbols"}
                    }}
                ]
                result = await self.db.trade_insights.aggregate(pipeline).to_list(1)
                
            elif agent_type == AgentType.CUSTOMER_SERVICE:
                # Customer service analytics
                pipeline = [
                    {"$match": {"agent_type": "customer_service", "timestamp": {"$gte": start_date}}},
                    {"$group": {
                        "_id": "$priority",
                        "count": {"$sum": 1}
                    }}
                ]
                result = await self.db.customer_escalations.aggregate(pipeline).to_list(None)
                
            elif agent_type == AgentType.SALES:
                # Sales analytics
                pipeline = [
                    {"$match": {"agent_type": "sales", "timestamp": {"$gte": start_date}}},
                    {"$group": {
                        "_id": "$stage",
                        "count": {"$sum": 1},
                        "total_value": {"$sum": "$value_estimate"}
                    }}
                ]
                result = await self.db.sales_opportunities.aggregate(pipeline).to_list(None)
            
            return {"agent_type": agent_type.value, "analytics": result, "period_days": days}
            
        except Exception as e:
            logger.error(f"Failed to get agent analytics: {e}")
            return {"error": str(e)}
    
    async def store_trade_insight(
        self, 
        user_id: str, 
        insight: str, 
        symbols: List[str],
        insight_type: str = "analysis"
    ) -> bool:
        """
        Store trading-specific insights
        
        Args:
            user_id: Unique user identifier
            insight: Trading insight content
            symbols: Related stock symbols
            insight_type: Type of insight (analysis, recommendation, etc.)
            
        Returns:
            bool: Success status
        """
        try:
            insight_doc = {
                "user_id": user_id,
                "insight": insight,
                "symbols": symbols,
                "insight_type": insight_type,
                "timestamp": datetime.utcnow(),
                "topics": self._extract_topics(insight, AgentType.TRADING),
                "agent_type": "trading",
                "importance_score": self._calculate_importance(insight, AgentType.TRADING)
            }
            
            result = await self.db.trade_insights.insert_one(insight_doc)
            
            # Also store in vector database for semantic search
            await self._store_in_vector_db(user_id, insight, symbols, "trade_insight", AgentType.TRADING)
            
            logger.info(f"Trade insight stored for user {user_id}")
            return bool(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to store trade insight: {e}")
            return False
    
    async def update_user_profile(
        self, 
        user_id: str, 
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update user profile with new information
        
        Args:
            user_id: Unique user identifier
            updates: Dictionary of updates to apply
            
        Returns:
            bool: Success status
        """
        try:
            updates["updated_at"] = datetime.utcnow()
            
            result = await self.db.users.update_one(
                {"user_id": user_id},
                {"$set": updates},
                upsert=True
            )
            
            logger.info(f"User profile updated for {user_id}")
            return result.acknowledged
            
        except Exception as e:
            logger.error(f"Failed to update user profile: {e}")
            return False
    
    async def summarize_session(self, user_id: str, agent_type: AgentType = AgentType.TRADING) -> bool:
        """
        Manually trigger session summarization for specific agent
        
        Args:
            user_id: Unique user identifier
            agent_type: Which agent's session to summarize
            
        Returns:
            bool: Success status
        """
        return await self._trigger_summarization(user_id, agent_type)
    
    async def search_vector_memory(
        self, 
        user_id: str, 
        query: str, 
        agent_type: AgentType = AgentType.TRADING,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search vector memory for semantically similar content
        
        Args:
            user_id: Unique user identifier
            query: Search query
            agent_type: Requesting agent type for filtering
            top_k: Number of results to return
            
        Returns:
            List of relevant memories with scores
        """
        try:
            # Generate embedding for query
            embedding = await self._get_embedding(query)
            
            # Search Pinecone with user and agent filters
            search_filter = {"user_id": user_id}
            
            # Include agent-specific memories and cross-agent important ones
            agent_filter = {
                "$or": [
                    {"agent_type": agent_type.value},
                    {"memory_type": {"$in": ["important", "trade_insight", "customer_escalation", "sales_opportunity"]}}
                ]
            }
            search_filter.update(agent_filter)
            
            results = self.pinecone_index.query(
                vector=embedding,
                top_k=top_k * 2,  # Get more results to filter
                filter=search_filter,
                include_metadata=True
            )
            
            # Format and rank results by agent relevance
            formatted_results = []
            for match in results.matches:
                result = {
                    "content": match.metadata.get("content", ""),
                    "topics": match.metadata.get("topics", []),
                    "timestamp": match.metadata.get("timestamp", ""),
                    "relevance_score": float(match.score),
                    "memory_type": match.metadata.get("memory_type", "unknown"),
                    "agent_type": match.metadata.get("agent_type", "unknown")
                }
                
                # Boost relevance for same-agent memories
                if result["agent_type"] == agent_type.value:
                    result["relevance_score"] *= 1.2
                
                formatted_results.append(result)
            
            # Sort by relevance and return top_k
            formatted_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            formatted_results = formatted_results[:top_k]
            
            logger.info(f"Vector search returned {len(formatted_results)} results for user {user_id} via {agent_type.value}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    # Include all previous methods with emotional intelligence enhancements
    # ... (all the existing helper methods from the previous implementation)

    async def cleanup(self):
        """Cleanup database connections and caches"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            if self.mongo_client:
                self.mongo_client.close()
            
            # Clear caches
            self.context_cache.clear()
            self.semantic_cache.clear()
            
            logger.info("Enhanced MemoryManager connections and caches cleaned up")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


# Enhanced demo showcasing emotional intelligence
async def enhanced_demo():
    """
    Comprehensive demo showing emotional intelligence and context awareness
    """
    print("=== ENHANCED MEMORY MANAGER WITH EMOTIONAL INTELLIGENCE DEMO ===\n")
    
    # Initialize Enhanced MemoryManager
    memory_manager = MemoryManager(
        redis_url="redis://localhost:6379",
        mongodb_url="mongodb://localhost:27017/trading_ai_enhanced",
        pinecone_api_key="your-pinecone-key",
        pinecone_environment="your-environment",
        openai_api_key="your-openai-key"
    )
    
    await memory_manager.setup()
    user_id = "demo_user_123"
    
    print("1. FRUSTRATED USER SCENARIO")
    print("=" * 40)
    
    # Simulate frustrated user interaction
    await memory_manager.save_message(
        user_id,
        "This platform is terrible! My orders keep failing and I'm losing money!!",
        MessageDirection.USER,
        AgentType.CUSTOMER_SERVICE
    )
    
    await memory_manager.save_message(
        user_id,
        "I understand your frustration. Let me help you resolve these order issues immediately.",
        MessageDirection.BOT,
        AgentType.CUSTOMER_SERVICE
    )
    
    await memory_manager.save_message(
        user_id,
        "Still not working! This is the worst trading app ever!!!",
        MessageDirection.USER,
        AgentType.CUSTOMER_SERVICE
    )
    
    # Get context with emotional intelligence
    context = await memory_manager.get_context(
        user_id,
        AgentType.CUSTOMER_SERVICE,
        "order execution problems",
        include_emotional_context=True
    )
    
    print(f"Emotional Context Detected:")
    emotional_ctx = context.get('emotional_context', {})
    if emotional_ctx.get('current_emotion'):
        print(f"- Current Emotion: {emotional_ctx['current_emotion']['emotion_type']}")
        print(f"- Emotion Score: {emotional_ctx['current_emotion']['emotion_score']:.2f}")
        print(f"- Needs Escalation: {emotional_ctx.get('needs_escalation', False)}")
    
    # Generate contextual prompt
    if emotional_ctx.get('current_emotion'):
        current_emotion = EmotionalState(
            emotion_type=EmotionType(emotional_ctx['current_emotion']['emotion_type']),
            emotion_score=emotional_ctx['current_emotion']['emotion_score'],
            sentiment_label=emotional_ctx['current_emotion']['sentiment_label'],
            confidence=emotional_ctx['current_emotion']['confidence'],
            timestamp=datetime.fromisoformat(emotional_ctx['current_emotion']['timestamp']),
            triggers=emotional_ctx['current_emotion'].get('triggers', [])
        )
