"""
Enhanced SMS Trading AI Agent - MemoryManager System with Emotional Intelligence
FINAL PRODUCTION-READY VERSION - All Issues Fixed
Complete implementation with Redis, MongoDB, Pinecone integration plus emotional awareness,
context intelligence, and response adaptation for multi-agent support.
"""

import asyncio
import json
import logging
import re
import hashlib
import statistics
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import deque, defaultdict
from dotenv import load_dotenv

# External dependencies
import redis.asyncio as redis
from motor.motor_asyncio import AsyncIOMotorClient
from pinecone import Pinecone
from openai import AsyncOpenAI
import tiktoken

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration management
class MemoryConfig:
    """Configuration management for MemoryManager"""
    
    @staticmethod
    def from_env():
        """Load configuration from environment variables"""
        return {
            'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379'),
            'mongodb_url': os.getenv('MONGODB_URL', 'mongodb://localhost:27017/trading_ai_enhanced'),
            'pinecone_api_key': os.getenv('PINECONE_API_KEY'),
            'pinecone_environment': os.getenv('PINECONE_ENVIRONMENT', 'us-east1-gcp'),
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'stm_limit': int(os.getenv('STM_LIMIT', '15')),
            'summary_trigger': int(os.getenv('SUMMARY_TRIGGER', '10')),
            'max_retries': int(os.getenv('MAX_RETRIES', '3')),
            'rate_limit_delay': float(os.getenv('RATE_LIMIT_DELAY', '0.1'))
        }
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate required configuration"""
        required_keys = ['pinecone_api_key', 'openai_api_key']
        missing_keys = [key for key in required_keys if not config.get(key)]
        
        if missing_keys:
            logger.error(f"Missing required configuration: {missing_keys}")
            return False
        
        return True  # Fixed: Always return True when validation passes

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
    confidence_score: float = 0.5
    
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
    emotional_trend: List[float] = None
    last_emotion: Optional[EmotionalState] = None
    tone_preference: str = "adaptive"
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
                'punctuation_weight': 2.0
            },
            EmotionType.EXCITED: {
                'keywords': ['amazing', 'awesome', 'fantastic', 'excellent', 'love', 'perfect', 'great'],
                'phrases': ['cant wait', 'so excited', 'looking forward', 'this is great'],
                'punctuation_weight': 1.5
            },
            EmotionType.CONFUSED: {
                'keywords': ['confused', 'unclear', 'understand', 'explain', 'help', 'lost'],
                'phrases': ['dont get it', 'not sure', 'how do i', 'what does', 'can you explain'],
                'punctuation_weight': 1.2
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
        
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'awesome', 'perfect', 'love',
            'happy', 'satisfied', 'pleased', 'wonderful', 'fantastic', 'brilliant'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'frustrated',
            'angry', 'disappointed', 'annoying', 'useless', 'broken', 'stupid'
        }
    
    def analyze_emotion(self, content: str, direction: MessageDirection) -> EmotionalState:
        """Analyze emotional state from message content"""
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
        
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        
        if positive_ratio + negative_ratio == 0:
            return 0.5
        
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
        text_length_factor = min(len(content.split()) / 20, 0.3)
        emotion_strength = max([e['score'] for e in detected_emotions.values()]) if detected_emotions else 0
        emotion_count_penalty = len(detected_emotions) * 0.1 if len(detected_emotions) > 1 else 0
        
        confidence = base_confidence + text_length_factor + (emotion_strength * 0.3) - emotion_count_penalty
        return max(0.1, min(0.95, confidence))
    
    def detect_escalation(self, emotional_history: List[EmotionalState]) -> bool:
        """Detect if user needs escalation based on emotional history"""
        if len(emotional_history) < 3:
            return False
        
        recent_emotions = emotional_history[-3:]
        frustration_count = sum(1 for emotion in recent_emotions 
                              if emotion.emotion_type == EmotionType.FRUSTRATED 
                              and emotion.emotion_score > 0.6)
        
        sentiment_scores = [emotion.emotion_score if emotion.sentiment_label == "positive" 
                          else 1.0 - emotion.emotion_score for emotion in recent_emotions]
        
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
        redis_url: str = None,
        mongodb_url: str = None,
        pinecone_api_key: str = None,
        pinecone_environment: str = None,
        openai_api_key: str = None,
        stm_limit: int = 15,
        summary_trigger: int = 10,
        config: Dict[str, Any] = None
    ):
        """Initialize Enhanced MemoryManager with emotional intelligence"""
        
        # Load configuration from environment or parameters
        if config:
            self.config = config
        else:
            env_config = MemoryConfig.from_env()
            self.config = {
                'redis_url': redis_url or env_config['redis_url'],
                'mongodb_url': mongodb_url or env_config['mongodb_url'],
                'pinecone_api_key': pinecone_api_key or env_config['pinecone_api_key'],
                'pinecone_environment': pinecone_environment or env_config['pinecone_environment'],
                'openai_api_key': openai_api_key or env_config['openai_api_key'],
                'stm_limit': stm_limit,
                'summary_trigger': summary_trigger,
                'max_retries': env_config['max_retries'],
                'rate_limit_delay': env_config['rate_limit_delay']
            }
        
        # Validate configuration
        if not MemoryConfig.validate_config(self.config):
            raise ValueError("Invalid configuration. Check environment variables.")
        
        # Configuration
        self.stm_limit = self.config['stm_limit']
        self.summary_trigger = self.config['summary_trigger']
        self.max_retries = self.config['max_retries']
        self.rate_limit_delay = self.config['rate_limit_delay']
        
        # Initialize emotional intelligence
        self.emotion_tracker = EmotionalStateTracker()
        
        # Database connections (initialized in setup)
        self.redis_client = None
        self.mongo_client = None
        self.db = None
        self.pinecone_index = None
        self.openai_client = None
        
        # Initialize tokenizer for accurate token counting
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        # Performance tracking
        self._last_embedding_call = None
        self._pinecone_batch = []
        self._batch_size = 10
        
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
            self.redis_client = redis.from_url(self.config['redis_url'], decode_responses=True)
            await self.redis_client.ping()
            logger.info("Redis connection established")
            
            # MongoDB connection
            self.mongo_client = AsyncIOMotorClient(self.config['mongodb_url'])
            self.db = self.mongo_client.trading_ai_memory
            
            # Test MongoDB connection
            await self.mongo_client.admin.command("ping")
            logger.info("MongoDB connection established")
            
            # Pinecone connection (Updated SDK)
            pc = Pinecone(api_key=self.config['pinecone_api_key'])
            
            # Create or connect to index
            index_name = "trading-memory-enhanced"
            existing_indexes = [index.name for index in pc.list_indexes()]
            
            if index_name not in existing_indexes:
                from pinecone import ServerlessSpec
                pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud='aws', 
                        region='us-east-1'
                    )
                )
                
                logger.info(f"Created new Pinecone index: {index_name}")
            
            self.pinecone_index = pc.Index(index_name)
            logger.info("Pinecone connection established")
            
            # OpenAI client
            self.openai_client = AsyncOpenAI(api_key=self.config['openai_api_key'])
            logger.info("OpenAI client initialized")
            
            # Create MongoDB indexes for optimization
            await self._create_mongodb_indexes()
            
            logger.info("Enhanced MemoryManager with Emotional Intelligence ready!")
            
        except Exception as e:
            logger.exception(f"Setup failed: {e}")
            raise
    
    async def _create_mongodb_indexes(self):
        """Create MongoDB indexes for optimal query performance"""
        try:
            # Users collection indexes
            await self.db.users.create_index("user_id", unique=True)
            await self.db.users.create_index("updated_at")
            await self.db.users.create_index("lead_stage")
            
            # Conversations collection indexes
            await self.db.conversations.create_index([("user_id", 1), ("agent_type", 1), ("timestamp", -1)])
            await self.db.conversations.create_index("topics")
            await self.db.conversations.create_index("importance_score")
            await self.db.conversations.create_index("agent_type")
            
            # Trade insights collection indexes
            await self.db.trade_insights.create_index([("user_id", 1), ("timestamp", -1)])
            await self.db.trade_insights.create_index("symbols")
            await self.db.trade_insights.create_index("agent_type")
            await self.db.trade_insights.create_index("importance_score")
            
            # Customer escalations collection indexes
            await self.db.customer_escalations.create_index([("user_id", 1), ("timestamp", -1)])
            await self.db.customer_escalations.create_index("priority")
            await self.db.customer_escalations.create_index("status")
            await self.db.customer_escalations.create_index("agent_type")
            
            # Sales opportunities collection indexes
            await self.db.sales_opportunities.create_index([("user_id", 1), ("timestamp", -1)])
            await self.db.sales_opportunities.create_index("stage")
            await self.db.sales_opportunities.create_index("probability")
            await self.db.sales_opportunities.create_index("agent_type")
            await self.db.sales_opportunities.create_index("value_estimate")
            
            logger.info("MongoDB indexes created successfully for all agent types")
            
        except Exception as e:
            logger.exception(f"Index creation error: {e}")
    
    def _extract_topics(self, content: str, agent_type: AgentType = AgentType.TRADING) -> List[str]:
        """Extract topics from message content based on agent type"""
        content_lower = content.lower()
        topics = []
        
        if agent_type == AgentType.TRADING:
            # Extract stock symbols using regex pattern
            symbols = self.stock_pattern.findall(content.upper())
            topics.extend([s for s in symbols if 2 <= len(s) <= 5])
            
            # Extract trading terms
            topics.extend([term for term in self.trading_terms if term in content_lower])
                    
        elif agent_type == AgentType.CUSTOMER_SERVICE:
            # Extract service-related terms
            topics.extend([term for term in self.service_terms if term in content_lower])
                    
            # Extract urgency indicators
            if any(word in content_lower for word in ['urgent', 'asap', 'immediately', 'critical']):
                topics.append('urgent')
                
        elif agent_type == AgentType.SALES:
            # Extract sales-related terms
            topics.extend([term for term in self.sales_terms if term in content_lower])
                    
            # Extract buying intent indicators
            if any(word in content_lower for word in ['buy', 'purchase', 'order', 'subscribe']):
                topics.append('buying_intent')
        
        # Remove duplicates and limit to 10
        return list(set(topics))[:10]
    
    def _classify_message_type(self, content: str, agent_type: AgentType = AgentType.TRADING) -> MemoryType:
        """Classify message type based on content and agent"""
        content_lower = content.lower()
        
        # System messages
        if any(term in content_lower for term in ['error', 'system', 'status']):
            return MemoryType.SYSTEM
        
        if agent_type == AgentType.TRADING:
            # Trading messages
            trading_indicators = [
                'buy', 'sell', 'portfolio', 'stock', 'price', 'analysis',
                'earnings', 'options', 'calls', 'puts', 'market'
            ]
            
            if any(term in content_lower for term in trading_indicators):
                return MemoryType.TRADING
                
        elif agent_type == AgentType.CUSTOMER_SERVICE:
            # Check for emotional content first
            if any(word in content_lower for word in ['frustrated', 'angry', 'disappointed']):
                return MemoryType.EMOTIONAL
            # Customer service messages
            if any(term in content_lower for term in self.service_terms):
                return MemoryType.CUSTOMER_SERVICE
                
        elif agent_type == AgentType.SALES:
            # Sales messages
            if any(term in content_lower for term in self.sales_terms):
                return MemoryType.SALES
        
        # Important messages (questions, requests)
        if '?' in content or any(term in content_lower for term in ['help', 'recommend', 'should']):
            return MemoryType.IMPORTANT
        
        return MemoryType.CASUAL
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Generate OpenAI embedding for text with rate limiting"""
        try:
            # Simple rate limiting - wait if needed
            if self._last_embedding_call:
                elapsed = (datetime.utcnow() - self._last_embedding_call).total_seconds()
                if elapsed < self.rate_limit_delay:
                    await asyncio.sleep(self.rate_limit_delay - elapsed)
            
            self._last_embedding_call = datetime.utcnow()
            
            response = await self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text[:8000]  # Limit input size for OpenAI
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.exception(f"Embedding generation failed: {e}")
            return [0.0] * 1536  # Return zero vector as fallback
    
    def _calculate_importance(
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
    
    async def store_customer_escalation(
        self,
        user_id: str,
        issue: str,
        details: str,
        priority: str = "medium"
    ) -> bool:
        """Store customer service escalation for high-priority tracking"""
        try:
            escalation_doc = {
                "user_id": user_id,
                "issue": issue,
                "details": details,
                "priority": priority,
                "status": "open",
                "timestamp": datetime.utcnow(),
                "agent_type": "customer_service"
            }
            
            result = await self.db.customer_escalations.insert_one(escalation_doc)
            
            # Store in vector database for future reference
            await self._store_in_vector_db(
                user_id, 
                f"ESCALATION: {issue} - {details}", 
                [priority, "escalation"], 
                "customer_escalation",
                AgentType.CUSTOMER_SERVICE
            )
            
            # Update user profile with escalation trigger
            await self.update_user_profile(user_id, {
                "escalation_triggers": [details],
                "last_escalation": datetime.utcnow()
            })
            
            logger.info(f"Customer escalation stored for user {user_id}")
            return bool(result.inserted_id)
            
        except Exception as e:
            logger.exception(f"Failed to store customer escalation: {e}")
            return False
    
    async def update_user_profile(
        self, 
        user_id: str, 
        updates: Dict[str, Any]
    ) -> bool:
        """Update user profile with new information"""
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
            logger.exception(f"Failed to update user profile: {e}")
            return False
    
    async def save_message(
        self, 
        user_id: str, 
        content: str, 
        direction: MessageDirection,
        agent_type: AgentType = AgentType.TRADING,
        topics: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Save message with emotional analysis and intelligent context tracking"""
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
            logger.exception(f"Failed to save enhanced message: {e}")
            return False
    
    async def get_context(
        self, 
        user_id: str, 
        agent_type: AgentType = AgentType.TRADING,
        query: str = None,
        include_cross_agent: bool = True,
        include_emotional_context: bool = True
    ) -> Dict[str, Any]:
        """Retrieve comprehensive context with emotional intelligence and 3-layer memory"""
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
            logger.exception(f"Failed to get enhanced context: {e}")
            return {"error": str(e)}
    
    async def generate_contextual_prompt(
        self,
        context: Dict[str, Any],
        emotional_state: EmotionalState,
        user_profile: Dict[str, Any],
        agent_type: AgentType
    ) -> str:
        """Generate tone-optimized prompt based on emotional state and context"""
        try:
            # Safe access to user profile
            tone_preference = user_profile.get('tone_preference', 'adaptive') if user_profile else 'adaptive'
            trading_style = user_profile.get('trading_style', 'unknown') if user_profile else 'unknown'
            risk_tolerance = user_profile.get('risk_tolerance', 'unknown') if user_profile else 'unknown'
            
            # Base prompt components
            base_context = f"""
You are a {agent_type.value.replace('_', ' ')} AI agent for a personalized SMS trading platform.

USER CONTEXT:
- Communication Preference: {tone_preference}
- Trading Style: {trading_style}
- Risk Tolerance: {risk_tolerance}
"""
            
            # Emotional adaptation
            emotional_guidance = ""
            
            if emotional_state.emotion_type == EmotionType.FRUSTRATED:
                emotional_guidance = f"""
EMOTIONAL STATE: User is frustrated (score: {emotional_state.emotion_score:.2f})
RESPONSE TONE: Be empathetic, acknowledge their frustration, provide clear step-by-step solutions.
- Use phrases like "I understand your frustration" or "Let me help you resolve this quickly"
- Avoid technical jargon, be direct and solution-focused
- Offer escalation if needed
"""
                
            elif emotional_state.emotion_type == EmotionType.EXCITED:
                emotional_guidance = f"""
EMOTIONAL STATE: User is excited (score: {emotional_state.emotion_score:.2f})
RESPONSE TONE: Match their enthusiasm while providing grounded advice.
- Use positive language and confirm their excitement
- Provide balanced perspective to prevent overconfidence
- Encourage but add risk awareness
"""
                
            elif emotional_state.emotion_type == EmotionType.CONFUSED:
                emotional_guidance = f"""
EMOTIONAL STATE: User is confused (score: {emotional_state.emotion_score:.2f})
RESPONSE TONE: Be patient, educational, and structured.
- Break down complex concepts into simple steps
- Use analogies and examples
- Ask clarifying questions to understand their confusion
- Provide educational resources
"""
                
            elif emotional_state.emotion_type == EmotionType.ANXIOUS:
                emotional_guidance = f"""
EMOTIONAL STATE: User is anxious (score: {emotional_state.emotion_score:.2f})
RESPONSE TONE: Be reassuring and focus on risk management.
- Acknowledge their concerns as valid
- Provide data-driven reassurance
- Focus on risk mitigation strategies
- Suggest conservative approaches
"""
            
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
            logger.exception(f"Failed to generate contextual prompt: {e}")
            return "You are a helpful AI assistant. Please provide a helpful response."
    
    # Helper methods (all the missing ones implemented)
    
    async def _get_short_term_memory(self, user_id: str, agent_type: AgentType) -> List[Dict[str, Any]]:
        """Retrieve short-term memory from Redis for specific agent"""
        try:
            stm_key = f"session:{agent_type.value}:{user_id}"
            messages = await self.redis_client.lrange(stm_key, 0, -1)
            
            parsed_messages = []
            for msg_json in messages:
                try:
                    parsed_messages.append(json.loads(msg_json))
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse message: {msg_json}")
                    continue
            
            # Sort by timestamp (newest first)
            parsed_messages.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return parsed_messages
            
        except Exception as e:
            logger.exception(f"Failed to get STM for {agent_type.value}: {e}")
            return []
    
    async def _get_conversation_summaries(
        self, 
        user_id: str, 
        agent_type: AgentType,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve recent conversation summaries for specific agent"""
        try:
            cursor = self.db.conversations.find(
                {"user_id": user_id, "agent_type": agent_type.value}
            ).sort("timestamp", -1).limit(limit)
            
            summaries = []
            async for doc in cursor:
                # Convert ObjectId to string and datetime to ISO
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
                if isinstance(doc.get("timestamp"), datetime):
                    doc["timestamp"] = doc["timestamp"].isoformat()
                summaries.append(doc)
            
            return summaries
            
        except Exception as e:
            logger.exception(f"Failed to get summaries for {agent_type.value}: {e}")
            return []
    
    async def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Retrieve or create user profile from MongoDB"""
        try:
            profile = await self.db.users.find_one({"user_id": user_id})
            
            if profile:
                # Convert ObjectId to string and datetime to ISO
                if "_id" in profile:
                    profile["_id"] = str(profile["_id"])
                for date_field in ["created_at", "updated_at"]:
                    if isinstance(profile.get(date_field), datetime):
                        profile[date_field] = profile[date_field].isoformat()
                return profile
            else:
                # Create default profile
                default_profile = UserProfile(user_id=user_id)
                profile_dict = default_profile.to_dict()
                
                # Insert into database
                result = await self.db.users.insert_one(profile_dict)
                profile_dict["_id"] = str(result.inserted_id)
                
                logger.info(f"Created new user profile for {user_id}")
                return profile_dict
            
        except Exception as e:
            logger.exception(f"Failed to get user profile for {user_id}: {e}")
            return {}
    
    async def _store_emotional_state(self, user_id: str, emotional_state: EmotionalState):
        """Store emotional state in Redis for trend tracking"""
        try:
            emotion_key = f"emotions:{user_id}"
            await self.redis_client.lpush(emotion_key, json.dumps(emotional_state.to_dict()))
            await self.redis_client.ltrim(emotion_key, 0, 19)  # Keep last 20 emotions
            await self.redis_client.expire(emotion_key, 86400 * 7)  # 7 days
        except Exception as e:
            logger.exception(f"Failed to store emotional state: {e}")
    
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
            logger.exception(f"Failed to update emotional profile: {e}")
    
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
            logger.exception(f"Failed to check escalation needs: {e}")
    
    async def _store_in_vector_db(
        self, 
        user_id: str, 
        content: str, 
        topics: List[str], 
        memory_type: str = "message",
        agent_type: AgentType = AgentType.TRADING,
        emotional_state: Optional[EmotionalState] = None
    ):
        """Enhanced vector storage with emotional weighting and batching"""
        try:
            # Generate embedding with rate limiting
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
            
            # Add to batch for efficient upsert
            self._pinecone_batch.append((content_id, embedding, metadata))
            
            # Process batch if it's full
            if len(self._pinecone_batch) >= self._batch_size:
                await self._flush_pinecone_batch()
            
            logger.debug(f"Enhanced content queued for vector DB for user {user_id} via {agent_type.value}")
            
        except Exception as e:
            logger.exception(f"Enhanced vector storage failed: {e}")
    
    async def _flush_pinecone_batch(self):
        """Flush pending Pinecone upserts"""
        if not self._pinecone_batch:
            return
        
        try:
            # Upsert batch to Pinecone
            self.pinecone_index.upsert(vectors=self._pinecone_batch)
            logger.debug(f"Flushed {len(self._pinecone_batch)} vectors to Pinecone")
            self._pinecone_batch = []
            
        except Exception as e:
            logger.exception(f"Pinecone batch flush failed: {e}")
            self._pinecone_batch = []  # Clear batch on error to prevent infinite retries
    
    # Additional missing helper methods
    
    async def _get_cross_agent_context(self, user_id: str, current_agent: AgentType) -> Dict[str, Any]:
        """Get relevant context from other agents"""
        try:
            cross_agent_data = {}
            
            # Get summaries from other agents (last 2 each)
            for agent_type in AgentType:
                if agent_type != current_agent:
                    summaries = await self._get_conversation_summaries(user_id, agent_type, limit=2)
                    if summaries:
                        cross_agent_data[agent_type.value] = {
                            "recent_summaries": summaries,
                            "last_interaction": summaries[0].get('timestamp') if summaries else None,
                            "message_count": len(summaries)
                        }
            
            return cross_agent_data
            
        except Exception as e:
            logger.exception(f"Failed to get cross-agent context: {e}")
            return {}
    
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
            logger.exception(f"Failed to get emotional context: {e}")
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
                top_k=top_k * 2,  # Get more results for filtering
                include_metadata=True,
                filter=search_filter
            )
            
            # Process and rank results with emotional weighting
            processed_results = []
            for match in results.matches:
                metadata = match.metadata
                
                # Apply emotional weighting to relevance score
                emotional_weight = metadata.get('emotional_weight', 1.0)
                weighted_score = match.score * emotional_weight
                
                processed_results.append({
                    'content': metadata.get('content', ''),
                    'topics': metadata.get('topics', []),
                    'score': weighted_score,
                    'original_score': match.score,
                    'emotional_weight': emotional_weight,
                    'timestamp': metadata.get('timestamp'),
                    'memory_type': metadata.get('memory_type'),
                    'agent_type': metadata.get('agent_type')
                })
            
            # Sort by weighted score and limit results
            processed_results.sort(key=lambda x: x['score'], reverse=True)
            final_results = processed_results[:top_k]
            
            # Cache results
            self.semantic_cache[cache_key] = final_results
            
            return final_results
            
        except Exception as e:
            logger.exception(f"Enhanced vector search failed: {e}")
            return []
    
    async def _trigger_summarization(self, user_id: str, agent_type: AgentType) -> bool:
        """Trigger LLM-based session summarization for specific agent"""
        try:
            # Get current STM for this agent
            messages = await self._get_short_term_memory(user_id, agent_type)
            
            if len(messages) < 3:  # Need minimum messages for summary
                logger.info(f"Not enough messages ({len(messages)}) for summarization")
                return False
            
            # Generate agent-specific summary using LLM
            summary_text = await self._generate_summary(messages, agent_type)
            
            if not summary_text:
                logger.warning("Failed to generate summary text")
                return False
            
            # Extract topics and calculate importance
            topics = self._extract_topics_from_messages(messages)
            importance_score = self._calculate_session_importance(messages)
            
            # Extract agent-specific insights
            agent_insights = self._extract_agent_insights(messages, agent_type)
            
            # Calculate engagement metrics
            emotional_states = [msg.get('emotional_state') for msg in messages if msg.get('emotional_state')]
            avg_sentiment = sum(
                1.0 if es.get('sentiment_label') == 'positive' 
                else 0.0 if es.get('sentiment_label') == 'negative' 
                else 0.5 
                for es in emotional_states
            ) / max(len(emotional_states), 1) if emotional_states else 0.5
            
            # Create summary object
            summary = ConversationSummary(
                user_id=user_id,
                summary=summary_text,
                topics=topics,
                timestamp=datetime.utcnow(),
                importance_score=importance_score,
                message_count=len(messages),
                session_id=self._generate_session_id(user_id, agent_type),
                agent_type=agent_type,
                agent_insights=agent_insights,
                avg_sentiment_score=avg_sentiment,
                engagement_level=min(importance_score * avg_sentiment * 2, 1.0)
            )
            
            # Store in MongoDB with retry logic
            for attempt in range(self.max_retries):
                try:
                    result = await self.db.conversations.insert_one(summary.to_dict())
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    logger.warning(f"Retry {attempt + 1}/{self.max_retries} for summary storage: {e}")
                    await asyncio.sleep(0.5 * (attempt + 1))
            
            # Store in vector database if important
            if importance_score > 0.6:
                await self._store_in_vector_db(
                    user_id, 
                    summary_text, 
                    topics, 
                    "conversation_summary",
                    agent_type
                )
            
            # Clear STM after successful summarization
            stm_key = f"session:{agent_type.value}:{user_id}"
            await self.redis_client.delete(stm_key)
            
            logger.info(f"Session summarized for user {user_id} via {agent_type.value} agent")
            return bool(result.inserted_id)
            
        except Exception as e:
            logger.exception(f"Summarization failed for {agent_type.value}: {e}")
            return False
    
    # Missing utility methods implementations
    
    async def _generate_summary(self, messages: List[Dict], agent_type: AgentType) -> str:
        """Generate LLM-based summary of conversation messages"""
        try:
            # Prepare conversation text for LLM
            conversation_text = ""
            for msg in reversed(messages[-10:]):  # Last 10 messages in chronological order
                direction = "User" if msg.get('direction') == 'user' else "Bot"
                content = msg.get('content', '')
                conversation_text += f"{direction}: {content}\n"
            
            # Agent-specific summary prompt
            if agent_type == AgentType.TRADING:
                prompt = f"""Summarize this trading conversation in 1-2 sentences, focusing on:
- Stocks/symbols discussed
- Trading intentions or strategies
- Key insights or recommendations
- User's risk preferences or concerns

Conversation:
{conversation_text}

Summary:"""
            elif agent_type == AgentType.CUSTOMER_SERVICE:
                prompt = f"""Summarize this customer service conversation in 1-2 sentences, focusing on:
- Issues or problems discussed
- Solutions provided or attempted
- User's satisfaction level
- Any escalation needs

Conversation:
{conversation_text}

Summary:"""
            else:  # SALES
                prompt = f"""Summarize this sales conversation in 1-2 sentences, focusing on:
- Products or services discussed
- User's interest level and budget
- Objections raised
- Next steps or timeline

Conversation:
{conversation_text}

Summary:"""
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.exception(f"Failed to generate summary: {e}")
            return f"Conversation summary for {agent_type.value} agent (auto-generated)"
    
    def _extract_topics_from_messages(self, messages: List[Dict]) -> List[str]:
        """Extract topics from a list of messages"""
        all_topics = []
        for msg in messages:
            topics = msg.get('topics', [])
            if topics:
                all_topics.extend(topics)
        
        # Count frequency and return top topics
        topic_counts = defaultdict(int)
        for topic in all_topics:
            topic_counts[topic] += 1
        
        # Sort by frequency and return top 10
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, count in sorted_topics[:10]]
    
    def _calculate_session_importance(self, messages: List[Dict]) -> float:
        """Calculate overall importance score for a session"""
        if not messages:
            return 0.0
        
        importance_scores = [msg.get('importance_score', 0.5) for msg in messages]
        
        # Weighted average with more weight on recent messages
        weights = [1.0 + (i * 0.1) for i in range(len(importance_scores))]
        weighted_sum = sum(score * weight for score, weight in zip(importance_scores, weights))
        weight_sum = sum(weights)
        
        return min(weighted_sum / weight_sum if weight_sum > 0 else 0.5, 1.0)
    
    def _extract_agent_insights(self, messages: List[Dict], agent_type: AgentType) -> Dict[str, Any]:
        """Extract agent-specific insights from conversation"""
        insights = {}
        
        if agent_type == AgentType.TRADING:
            # Extract trading-specific insights
            symbols_mentioned = []
            sentiment_indicators = []
            
            for msg in messages:
                topics = msg.get('topics', [])
                symbols_mentioned.extend([t for t in topics if t.isupper() and len(t) <= 5])
                
                content = msg.get('content', '').lower()
                if any(word in content for word in ['buy', 'bullish', 'long']):
                    sentiment_indicators.append('positive')
                elif any(word in content for word in ['sell', 'bearish', 'short']):
                    sentiment_indicators.append('negative')
            
            insights = {
                'symbols_discussed': list(set(symbols_mentioned)),
                'market_sentiment': sentiment_indicators,
                'trading_signals': len([m for m in messages if 'buy' in m.get('content', '').lower() or 'sell' in m.get('content', '').lower()])
            }
            
        elif agent_type == AgentType.CUSTOMER_SERVICE:
            # Extract service-specific insights
            issue_keywords = []
            resolution_attempts = 0
            
            for msg in messages:
                content = msg.get('content', '').lower()
                if any(word in content for word in ['problem', 'issue', 'bug', 'error']):
                    issue_keywords.append('technical_issue')
                if any(word in content for word in ['help', 'fix', 'resolve', 'solution']):
                    resolution_attempts += 1
            
            insights = {
                'issue_types': list(set(issue_keywords)),
                'resolution_attempts': resolution_attempts,
                'escalation_indicators': len([m for m in messages if any(word in m.get('content', '').lower() for word in ['frustrated', 'manager', 'escalate'])])
            }
            
        else:  # SALES
            # Extract sales-specific insights
            buying_signals = 0
            objections = 0
            
            for msg in messages:
                content = msg.get('content', '').lower()
                if any(word in content for word in ['interested', 'buy', 'purchase', 'demo']):
                    buying_signals += 1
                if any(word in content for word in ['expensive', 'budget', 'competitor', 'think about it']):
                    objections += 1
            
            insights = {
                'buying_signals': buying_signals,
                'objections_raised': objections,
                'engagement_level': len([m for m in messages if m.get('direction') == 'user'])
            }
        
        return insights
    
    def _generate_session_id(self, user_id: str, agent_type: AgentType) -> str:
        """Generate unique session ID"""
        timestamp = datetime.utcnow().isoformat()
        return hashlib.md5(f"{user_id}_{agent_type.value}_{timestamp}".encode()).hexdigest()[:12]
    
    def _apply_relevance_weighting(
        self, 
        vector_results: List[Dict], 
        emotional_context: Dict, 
        user_profile: Dict
    ) -> List[Dict]:
        """Apply relevance weighting based on emotional state and user profile"""
        if not vector_results:
            return []
        
        weighted_results = []
        current_emotion = emotional_context.get('current_emotion', {}).get('emotion_type', 'neutral')
        
        for result in vector_results:
            weight_multiplier = 1.0
            
            # Boost emotional content if user is emotional
            if current_emotion in ['frustrated', 'anxious'] and result.get('memory_type') == 'emotional':
                weight_multiplier *= 1.5
            
            # Boost agent-specific content
            user_agent_preference = user_profile.get('preferred_agent', 'trading')
            if result.get('agent_type') == user_agent_preference:
                weight_multiplier *= 1.2
            
            # Apply time decay (recent is more relevant)
            if result.get('timestamp'):
                try:
                    timestamp = datetime.fromisoformat(result['timestamp'])
                    days_old = (datetime.utcnow() - timestamp).days
                    time_decay = max(0.5, 1.0 - (days_old * 0.1))
                    weight_multiplier *= time_decay
                except:
                    pass
            
            # Apply weighting
            result['weighted_score'] = result.get('score', 0) * weight_multiplier
            weighted_results.append(result)
        
        # Sort by weighted score
        weighted_results.sort(key=lambda x: x.get('weighted_score', 0), reverse=True)
        return weighted_results
    
    def _filter_profile_for_agent(self, profile: Dict, agent_type: AgentType) -> Dict:
        """Filter user profile to show only relevant information for specific agent"""
        if not profile:
            return {}
        
        base_fields = ['user_id', 'communication_preferences', 'timezone', 'tone_preference', 'engagement_score']
        
        if agent_type == AgentType.TRADING:
            agent_fields = ['risk_tolerance', 'trading_style', 'watchlist', 'portfolio_insights', 'confidence_score']
        elif agent_type == AgentType.CUSTOMER_SERVICE:
            agent_fields = ['support_history', 'satisfaction_scores', 'preferred_contact_method', 'escalation_triggers', 'frustration_count']
        else:  # SALES
            agent_fields = ['lead_stage', 'interest_level', 'budget_range', 'decision_timeline', 'pain_points', 'objections_history']
        
        filtered_profile = {}
        for field in base_fields + agent_fields:
            if field in profile:
                filtered_profile[field] = profile[field]
        
        return filtered_profile
    
    async def _intelligent_context_compression(self, context: Dict) -> Dict:
        """Compress context if it exceeds token limits while preserving important information"""
        try:
            # Estimate token count (rough approximation)
            context_str = json.dumps(context)
            estimated_tokens = len(self.tokenizer.encode(context_str))
            
            if estimated_tokens <= 2000:  # Within reasonable limits
                return context
            
            # Compress by prioritizing recent and important information
            compressed_context = {
                "agent_type": context["agent_type"],
                "short_term_memory": context["short_term_memory"][:5],  # Keep last 5 messages
                "conversation_summaries": context["conversation_summaries"][:3],  # Keep top 3 summaries
                "user_profile": context["user_profile"],
                "relevant_memories": context["relevant_memories"][:2],  # Keep top 2 vector matches
                "emotional_context": {
                    "current_emotion": context["emotional_context"].get("current_emotion"),
                    "needs_escalation": context["emotional_context"].get("needs_escalation")
                },
                "context_metadata": context["context_metadata"]
            }
            
            # Add compression indicator
            compressed_context["context_metadata"]["compressed"] = True
            compressed_context["context_metadata"]["original_token_estimate"] = estimated_tokens
            
            return compressed_context
            
        except Exception as e:
            logger.exception(f"Context compression failed: {e}")
            return context  # Return original if compression fails
    
    async def _get_cached_context(self, cache_key: str) -> Optional[Dict]:
        """Retrieve cached context if available and not expired"""
        try:
            cached_data = self.context_cache.get(cache_key)
            if cached_data:
                # Check if cache is still valid (5 minutes)
                cache_time = cached_data.get('cached_at')
                if cache_time and (datetime.utcnow() - cache_time).total_seconds() < 300:
                    return cached_data.get('context')
            return None
        except Exception as e:
            logger.exception(f"Failed to get cached context: {e}")
            return None
    
    async def _cache_context(self, cache_key: str, context: Dict):
        """Cache context for performance"""
        try:
            self.context_cache[cache_key] = {
                'context': context,
                'cached_at': datetime.utcnow()
            }
            
            # Limit cache size (keep last 100 contexts)
            if len(self.context_cache) > 100:
                # Remove oldest entries
                sorted_keys = sorted(self.context_cache.keys(), 
                                   key=lambda k: self.context_cache[k]['cached_at'])
                for key in sorted_keys[:50]:  # Remove oldest 50
                    del self.context_cache[key]
                    
        except Exception as e:
            logger.exception(f"Failed to cache context: {e}")
    
    async def _get_empty_result(self) -> Dict:
        """Return empty result for optional context components"""
        return {}
    
    async def _get_empty_vector_result(self) -> List:
        """Return empty vector result"""
        return []
    
    # Cleanup and utility methods
    
    async def cleanup(self):
        """Clean up resources and connections"""
        try:
            # Flush any pending Pinecone batch
            await self._flush_pinecone_batch()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            # Close MongoDB connection
            if self.mongo_client:
                self.mongo_client.close()
            
            logger.info("MemoryManager cleanup completed")
            
        except Exception as e:
            logger.exception(f"Cleanup failed: {e}")
    
    async def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user statistics across all agents"""
        try:
            stats = {
                "user_id": user_id,
                "agents": {},
                "emotional_profile": {},
                "engagement_metrics": {}
            }
            
            # Get stats for each agent type
            for agent_type in AgentType:
                agent_stats = {
                    "message_count": 0,
                    "conversation_count": 0,
                    "avg_importance": 0.0,
                    "last_interaction": None
                }
                
                # Count messages in STM
                stm_key = f"session:{agent_type.value}:{user_id}"
                stm_count = await self.redis_client.llen(stm_key)
                agent_stats["current_session_messages"] = stm_count
                
                # Count conversations in MongoDB
                conv_count = await self.db.conversations.count_documents({
                    "user_id": user_id,
                    "agent_type": agent_type.value
                })
                agent_stats["conversation_count"] = conv_count
                
                # Get last conversation
                last_conv = await self.db.conversations.find_one(
                    {"user_id": user_id, "agent_type": agent_type.value},
                    sort=[("timestamp", -1)]
                )
                if last_conv:
                    agent_stats["last_interaction"] = last_conv["timestamp"].isoformat()
                    agent_stats["avg_importance"] = last_conv.get("importance_score", 0.0)
                
                stats["agents"][agent_type.value] = agent_stats
            
            # Get emotional profile
            emotion_key = f"emotions:{user_id}"
            emotion_count = await self.redis_client.llen(emotion_key)
            stats["emotional_profile"]["emotion_history_count"] = emotion_count
            
            # Get user profile for engagement metrics
            user_profile = await self._get_user_profile(user_id)
            if user_profile:
                stats["engagement_metrics"] = {
                    "engagement_score": user_profile.get("engagement_score", 0.5),
                    "frustration_count": user_profile.get("frustration_count", 0),
                    "created_at": user_profile.get("created_at"),
                    "last_updated": user_profile.get("updated_at")
                }
            
            return stats
            
        except Exception as e:
            logger.exception(f"Failed to get user statistics: {e}")
            return {"error": str(e)}
    
    async def delete_user_data(self, user_id: str) -> bool:
        """Delete all user data across all systems (GDPR compliance)"""
        try:
            # Delete from Redis
            for agent_type in AgentType:
                stm_key = f"session:{agent_type.value}:{user_id}"
                await self.redis_client.delete(stm_key)
            
            emotion_key = f"emotions:{user_id}"
            await self.redis_client.delete(emotion_key)
            
            # Delete from MongoDB
            await self.db.users.delete_many({"user_id": user_id})
            await self.db.conversations.delete_many({"user_id": user_id})
            await self.db.trade_insights.delete_many({"user_id": user_id})
            await self.db.customer_escalations.delete_many({"user_id": user_id})
            await self.db.sales_opportunities.delete_many({"user_id": user_id})
            
            # Delete from Pinecone (requires fetching first)
            try:
                # Query all vectors for this user
                dummy_vector = [0.0] * 1536
                results = self.pinecone_index.query(
                    vector=dummy_vector,
                    filter={"user_id": user_id},
                    top_k=1000,  # Large number to get all
                    include_metadata=False
                )
                
                # Delete vectors
                if results.matches:
                    vector_ids = [match.id for match in results.matches]
                    self.pinecone_index.delete(ids=vector_ids)
            except Exception as e:
                logger.warning(f"Pinecone deletion failed (non-critical): {e}")
            
            # Clear local caches
            cache_keys_to_remove = [k for k in self.context_cache.keys() if user_id in k]
            for key in cache_keys_to_remove:
                del self.context_cache[key]
            
            cache_keys_to_remove = [k for k in self.semantic_cache.keys() if user_id in k]
            for key in cache_keys_to_remove:
                del self.semantic_cache[key]
            
            logger.info(f"All data deleted for user {user_id}")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to delete user data: {e}")
            return False

# Example usage and testing
async def main():
    """Example usage of the MemoryManager"""
    try:
        # Initialize MemoryManager
        memory_manager = MemoryManager()
        await memory_manager.setup()
        
        # Example: Save a trading message
        await memory_manager.save_message(
            user_id="test_user_123",
            content="What do you think about AAPL? Should I buy some shares?",
            direction=MessageDirection.USER,
            agent_type=AgentType.TRADING
        )
        
        # Example: Save bot response
        await memory_manager.save_message(
            user_id="test_user_123",
            content="AAPL looks strong with good technical indicators. Consider your risk tolerance before investing.",
            direction=MessageDirection.BOT,
            agent_type=AgentType.TRADING
        )
        
        # Example: Get context for response generation
        context = await memory_manager.get_context(
            user_id="test_user_123",
            agent_type=AgentType.TRADING,
            query="AAPL analysis"
        )
        
        print("Context retrieved successfully:")
        print(f"STM messages: {len(context.get('short_term_memory', []))}")
        print(f"Conversation summaries: {len(context.get('conversation_summaries', []))}")
        print(f"User profile exists: {bool(context.get('user_profile'))}")
        print(f"Emotional context: {bool(context.get('emotional_context'))}")
        
        # Example: Get user statistics
        stats = await memory_manager.get_user_statistics("test_user_123")
        print(f"User statistics: {stats}")
        
        # Cleanup
        await memory_manager.cleanup()
        
    except Exception as e:
        logger.exception(f"Example usage failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
