# social/models/social_conversation.py
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Utility functions for safe enum handling
def safe_enum(enum_cls, value, default):
    """Safely convert value to enum with fallback"""
    try:
        if isinstance(value, enum_cls):
            return value
        return enum_cls(value)
    except (ValueError, TypeError):
        logger.warning(f"Invalid enum value {value} for {enum_cls.__name__}, using default {default}")
        return default

def safe_datetime_to_iso(dt: Optional[datetime]) -> Optional[str]:
    """Safely convert datetime to ISO string"""
    return dt.isoformat() if dt else None

def safe_iso_to_datetime(iso_string: Optional[str]) -> Optional[datetime]:
    """Safely convert ISO string to datetime"""
    if not iso_string:
        return None
    try:
        return datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        logger.warning(f"Invalid datetime string: {iso_string}")
        return datetime.utcnow()

class InteractionType(Enum):
    COMMENT = "comment"  # Public comment on our content
    DM = "dm"  # Private direct message
    REPLY = "reply"  # Reply to their comment
    DM_REPLY = "dm_reply"  # Reply to their DM
    COMPETITOR_COMMENT = "competitor_comment"  # Comment on competitor content
    COMPETITOR_REPLY = "competitor_reply"  # Our reply on competitor content

class ConversationType(Enum):
    PUBLIC_THREAD = "public_thread"  # Comment thread on our post/video
    PRIVATE_CONVERSATION = "private_conversation"  # DM conversation
    CROSS_PLATFORM = "cross_platform"  # Same user across multiple platforms
    COMPETITIVE_ENGAGEMENT = "competitive_engagement"  # Engagement on competitor content

class ContentOwnership(Enum):
    OUR_CONTENT = "our_content"  # Comments on our posts/videos
    COMPETITOR_CONTENT = "competitor_content"  # Comments on competitor posts
    NEUTRAL_CONTENT = "neutral_content"  # Comments on general finance content

class MessageDirection(Enum):
    INBOUND = "inbound"  # They sent to us
    OUTBOUND = "outbound"  # We sent to them

class ConversationStatus(Enum):
    ACTIVE = "active"  # Ongoing conversation
    DORMANT = "dormant"  # No recent activity
    CONVERTED = "converted"  # User signed up for SMS
    BLOCKED = "blocked"  # User blocked or we stopped responding

@dataclass
class MessageEngagement:
    """Engagement metrics for a specific message"""
    likes: int = 0
    replies: int = 0
    shares: int = 0
    views: int = 0  # For DMs that support read receipts
    user_reacted: bool = False  # Did they like/react to our response?

@dataclass
class ConversationContext:
    """Context about the conversation"""
    our_content_id: Optional[str] = None  # ID of our post/video being discussed
    our_content_type: Optional[str] = None  # "video", "post", "story"
    our_content_topic: List[str] = field(default_factory=list)  # ["AAPL", "technical_analysis"]
    conversation_starter: str = ""  # What initiated this conversation
    platform_thread_id: Optional[str] = None  # Platform's conversation/thread ID
    
    # Competitor content tracking
    content_ownership: ContentOwnership = ContentOwnership.OUR_CONTENT
    competitor_account: str = ""  # Username of competitor if applicable
    competitor_content_id: Optional[str] = None  # ID of competitor's content
    competitor_content_topic: List[str] = field(default_factory=list)  # Topics in competitor content
    
@dataclass
class CompetitiveEngagement:
    """Tracking competitive engagement metrics"""
    strategy_type: str = "audience_education"  # Type of competitive strategy
    quality_gap_identified: str = ""  # What gap we're filling vs competitor
    response_performance: Dict[str, int] = field(default_factory=dict)  # Likes, replies, profile visits
    audience_interest_level: str = "unknown"  # How interested their audience is in our response
    conversion_potential: str = "unknown"  # Likelihood of converting their audience
    competitor_response: str = ""  # Did the competitor respond to our engagement?

class SocialMessage:
    """Individual message in a conversation"""
    
    def __init__(self, 
                 direction: MessageDirection,
                 content: str,
                 interaction_type: InteractionType,
                 message_id: str = ""):
        self.message_id = message_id
        self.direction = direction
        self.content = content
        self.interaction_type = interaction_type
        self.timestamp = datetime.utcnow()
        
        # Engagement tracking
        self.engagement = MessageEngagement()
        
        # Analysis
        self.topics_mentioned = []  # ["AAPL", "NVDA"]
        self.sentiment = "neutral"  # positive, negative, neutral
        self.contains_question = False
        self.contains_stock_symbol = False
        self.urgency_level = "normal"  # low, normal, high
        
        # Response metadata (for our outbound messages)
        self.response_strategy = ""  # "educational", "supportive", "conversion"
        self.generated_by = "manual"  # "manual", "automated", "ai_assisted"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "direction": self.direction.value,
            "content": self.content,
            "interaction_type": self.interaction_type.value,
            "timestamp": safe_datetime_to_iso(self.timestamp),
            "engagement": {
                "likes": self.engagement.likes,
                "replies": self.engagement.replies,
                "shares": self.engagement.shares,
                "views": self.engagement.views,
                "user_reacted": self.engagement.user_reacted
            },
            "topics_mentioned": self.topics_mentioned,
            "sentiment": self.sentiment,
            "contains_question": self.contains_question,
            "contains_stock_symbol": self.contains_stock_symbol,
            "urgency_level": self.urgency_level,
            "response_strategy": self.response_strategy,
            "generated_by": self.generated_by
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SocialMessage':
        message = cls(
            direction=safe_enum(MessageDirection, data.get("direction"), MessageDirection.INBOUND),
            content=data.get("content", ""),
            interaction_type=safe_enum(InteractionType, data.get("interaction_type"), InteractionType.COMMENT),
            message_id=data.get("message_id", "")
        )
        
        message.timestamp = safe_iso_to_datetime(data.get("timestamp"))
        
        # Engagement
        engagement_data = data.get("engagement", {})
        message.engagement = MessageEngagement(
            likes=engagement_data.get("likes", 0),
            replies=engagement_data.get("replies", 0),
            shares=engagement_data.get("shares", 0),
            views=engagement_data.get("views", 0),
            user_reacted=engagement_data.get("user_reacted", False)
        )
        
        # Analysis fields
        message.topics_mentioned = data.get("topics_mentioned", [])
        message.sentiment = data.get("sentiment", "neutral")
        message.contains_question = data.get("contains_question", False)
        message.contains_stock_symbol = data.get("contains_stock_symbol", False)
        message.urgency_level = data.get("urgency_level", "normal")
        message.response_strategy = data.get("response_strategy", "")
        message.generated_by = data.get("generated_by", "manual")
        
        return message

class SocialConversation:
    """Complete conversation thread with a social user"""
    
    def __init__(self, 
                 platform_user_id: str,
                 conversation_type: ConversationType,
                 conversation_id: str = ""):
        self.conversation_id = conversation_id or f"{platform_user_id}_{int(datetime.utcnow().timestamp())}"
        self.platform_user_id = platform_user_id
        self.conversation_type = conversation_type
        
        # Conversation metadata
        self.status = ConversationStatus.ACTIVE
        self.context = ConversationContext()
        
        # Competitive engagement tracking
        self.competitive_engagement = None  # CompetitiveEngagement object if applicable
        self.audience_poaching = False  # Is this targeting competitor audience?
        
        # Messages in this conversation
        self.messages: List[SocialMessage] = []
        
        # Timestamps
        self.started_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        # Analysis
        self.conversation_summary = ""
        self.topics_discussed = []  # ["AAPL", "technical_analysis", "earnings"]
        self.dominant_sentiment = "neutral"
        self.user_satisfaction = "unknown"  # based on their responses
        
        # Conversion tracking
        self.conversion_opportunity_identified = False
        self.conversion_attempt_made = False
        self.conversion_successful = False
        
        # Quality metrics (cached for performance)
        self.avg_response_time = 0  # Our average response time in minutes
        self.user_engagement_score = 0  # Cached engagement score
        self._engagement_score_last_calculated = datetime.utcnow()
        self._engagement_score_stale_threshold = timedelta(hours=1)  # Recalc every hour
        
        # Message storage strategy - keep lightweight summary
        self.message_count = 0
        self.user_message_count = 0
        self.our_response_count = 0
        self.last_message_snippet = ""  # Last 100 chars for quick context
    
    def add_message(self, message: SocialMessage):
        """Add a new message to the conversation"""
        self.messages.append(message)
        self.last_activity = message.timestamp
        self.updated_at = datetime.utcnow()
        
        # Update lightweight counters
        self.message_count += 1
        if message.direction == MessageDirection.INBOUND:
            self.user_message_count += 1
        else:
            self.our_response_count += 1
        
        # Update last message snippet for quick context
        self.last_message_snippet = message.content[:100]
        
        # Update conversation topics
        for topic in message.topics_mentioned:
            if topic not in self.topics_discussed:
                self.topics_discussed.append(topic)
        
        # Mark engagement score as stale
        self._engagement_score_last_calculated = datetime.utcnow() - self._engagement_score_stale_threshold
    
    def get_recent_messages(self, count: int = 5) -> List[SocialMessage]:
        """Get the most recent messages for context"""
        return self.messages[-count:] if self.messages else []
    
    def get_message_pairs(self, count: int = 3) -> List[tuple]:
        """Get recent message pairs (their message + our response) for context"""
        pairs = []
        recent = self.messages[-count*2:] if len(self.messages) >= count*2 else self.messages
        
        for i in range(0, len(recent) - 1, 2):
            if (recent[i].direction == MessageDirection.INBOUND and 
                recent[i+1].direction == MessageDirection.OUTBOUND):
                pairs.append((recent[i], recent[i+1]))
        
        return pairs
    
    def calculate_user_engagement_score(self, force_recalc: bool = False) -> int:
        """Calculate how engaged the user is in this conversation with caching"""
        # Check if cached score is still valid
        if (not force_recalc and 
            datetime.utcnow() - self._engagement_score_last_calculated < self._engagement_score_stale_threshold):
            return self.user_engagement_score
        
        if not self.messages:
            self.user_engagement_score = 0
            return 0
        
        # Single-pass calculation for performance
        score = 0
        user_messages = []
        our_messages = []
        total_likes = 0
        total_user_reacted = 0
        questions_count = 0
        
        # Single iteration over messages
        for message in self.messages:
            if message.direction == MessageDirection.INBOUND:
                user_messages.append(message)
                if message.contains_question:
                    questions_count += 1
            else:
                our_messages.append(message)
                total_likes += message.engagement.likes
                if message.engagement.user_reacted:
                    total_user_reacted += 1
        
        # Message count contribution (0-30 points)
        message_count = len(user_messages)
        if message_count >= 10:
            score += 30
        elif message_count >= 5:
            score += 20
        elif message_count >= 3:
            score += 15
        elif message_count >= 1:
            score += 10
        
        # Response ratio (0-25 points)
        if our_messages:
            response_ratio = len(user_messages) / len(our_messages)
            score += min(25, int(response_ratio * 25))
        
        # Question asking (0-20 points) - engaged users ask questions
        score += min(20, questions_count * 5)
        
        # Engagement with our responses (0-25 points)
        if our_messages:
            avg_likes = total_likes / len(our_messages)
            reaction_rate = total_user_reacted / len(our_messages)
            
            score += min(15, int(avg_likes * 3))  # Likes contribution
            score += min(10, int(reaction_rate * 10))  # Reaction rate contribution
        
        self.user_engagement_score = min(score, 100)
        self._engagement_score_last_calculated = datetime.utcnow()
        return self.user_engagement_score
    
    def is_dormant(self, days_threshold: int = 7) -> bool:
        """Check if conversation has been dormant"""
        days_since_activity = (datetime.utcnow() - self.last_activity).days
        return days_since_activity >= days_threshold
    
    def needs_response(self) -> bool:
        """Check if we need to respond (last message was from user)"""
        if not self.messages:
            return False
        
        last_message = self.messages[-1]
        return (last_message.direction == MessageDirection.INBOUND and 
                self.status == ConversationStatus.ACTIVE)
    
    def update_conversation_summary(self, summary: str):
        """Update the conversation summary"""
        self.conversation_summary = summary
        self.updated_at = datetime.utcnow()
    
    def mark_conversion_opportunity(self, opportunity_type: str):
        """Mark that a conversion opportunity was identified"""
        self.conversion_opportunity_identified = True
        self.context.conversation_starter = f"conversion_opportunity_{opportunity_type}"
        self.updated_at = datetime.utcnow()
    
    def record_conversion_attempt(self, successful: bool = False):
        """Record a conversion attempt"""
        self.conversion_attempt_made = True
        self.conversion_successful = successful
        if successful:
            self.status = ConversationStatus.CONVERTED
        self.updated_at = datetime.utcnow()
    
    def mark_as_competitive_engagement(self, competitor_account: str, strategy_type: str, quality_gap: str):
        """Mark this conversation as competitive audience engagement"""
        self.conversation_type = ConversationType.COMPETITIVE_ENGAGEMENT
        self.audience_poaching = True
        self.context.content_ownership = ContentOwnership.COMPETITOR_CONTENT
        self.context.competitor_account = competitor_account
        
        self.competitive_engagement = CompetitiveEngagement(
            strategy_type=strategy_type,
            quality_gap_identified=quality_gap
        )
        self.updated_at = datetime.utcnow()
    
    def update_competitive_performance(self, performance_data: Dict[str, int]):
        """Update performance metrics for competitive engagement"""
        if self.competitive_engagement:
            self.competitive_engagement.response_performance.update(performance_data)
            self.updated_at = datetime.utcnow()
    
    def assess_competitor_audience_interest(self, interest_level: str):
        """Assess how interested the competitor's audience is in our response"""
        if self.competitive_engagement:
            self.competitive_engagement.audience_interest_level = interest_level
            self.updated_at = datetime.utcnow()
    
    def record_competitor_response(self, response_text: str):
        """Record if/how the competitor responded to our engagement"""
        if self.competitive_engagement:
            self.competitive_engagement.competitor_response = response_text
            self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage"""
        return {
            "conversation_id": self.conversation_id,
            "platform_user_id": self.platform_user_id,
            "conversation_type": self.conversation_type.value,
            "status": self.status.value,
            
            # Context
            "context": {
                "our_content_id": self.context.our_content_id,
                "our_content_type": self.context.our_content_type,
                "our_content_topic": self.context.our_content_topic,
                "conversation_starter": self.context.conversation_starter,
                "platform_thread_id": self.context.platform_thread_id,
                "content_ownership": self.context.content_ownership.value,
                "competitor_account": self.context.competitor_account,
                "competitor_content_id": self.context.competitor_content_id,
                "competitor_content_topic": self.context.competitor_content_topic
            },
            
            # Competitive engagement
            "competitive_engagement": {
                "strategy_type": self.competitive_engagement.strategy_type if self.competitive_engagement else "",
                "quality_gap_identified": self.competitive_engagement.quality_gap_identified if self.competitive_engagement else "",
                "response_performance": self.competitive_engagement.response_performance if self.competitive_engagement else {},
                "audience_interest_level": self.competitive_engagement.audience_interest_level if self.competitive_engagement else "unknown",
                "conversion_potential": self.competitive_engagement.conversion_potential if self.competitive_engagement else "unknown",
                "competitor_response": self.competitive_engagement.competitor_response if self.competitive_engagement else ""
            } if self.competitive_engagement else None,
            
            "audience_poaching": self.audience_poaching,
            
            # Messages
            "messages": [message.to_dict() for message in self.messages],
            
            # Timestamps (ISO format for JSON compatibility)
            "started_at": safe_datetime_to_iso(self.started_at),
            "last_activity": safe_datetime_to_iso(self.last_activity),
            "created_at": safe_datetime_to_iso(self.created_at),
            "updated_at": safe_datetime_to_iso(self.updated_at),
            
            # Analysis
            "conversation_summary": self.conversation_summary,
            "topics_discussed": self.topics_discussed,
            "dominant_sentiment": self.dominant_sentiment,
            "user_satisfaction": self.user_satisfaction,
            
            # Conversion tracking
            "conversion_opportunity_identified": self.conversion_opportunity_identified,
            "conversion_attempt_made": self.conversion_attempt_made,
            "conversion_successful": self.conversion_successful,
            
            # Quality metrics (cached values)
            "avg_response_time": self.avg_response_time,
            "user_engagement_score": self.user_engagement_score,
            "_engagement_score_last_calculated": safe_datetime_to_iso(self._engagement_score_last_calculated),
            
            # Lightweight message tracking
            "message_count": self.message_count,
            "user_message_count": self.user_message_count,
            "our_response_count": self.our_response_count,
            "last_message_snippet": self.last_message_snippet
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SocialConversation':
        """Create SocialConversation from dictionary"""
        conversation = cls(
            platform_user_id=data["platform_user_id"],
            conversation_type=ConversationType(data["conversation_type"]),
            conversation_id=data["conversation_id"]
        )
        
        # Basic fields
        conversation.status = safe_enum(ConversationStatus, data.get("status"), ConversationStatus.ACTIVE)
        
        # Context with safe enum handling
        context_data = data.get("context", {})
        conversation.context = ConversationContext(
            our_content_id=context_data.get("our_content_id"),
            our_content_type=context_data.get("our_content_type"),
            our_content_topic=context_data.get("our_content_topic", []),
            conversation_starter=context_data.get("conversation_starter", ""),
            platform_thread_id=context_data.get("platform_thread_id"),
            content_ownership=safe_enum(ContentOwnership, context_data.get("content_ownership"), ContentOwnership.OUR_CONTENT),
            competitor_account=context_data.get("competitor_account", ""),
            competitor_content_id=context_data.get("competitor_content_id"),
            competitor_content_topic=context_data.get("competitor_content_topic", [])
        )
        
        # Competitive engagement
        if data.get("competitive_engagement"):
            comp_data = data["competitive_engagement"]
            conversation.competitive_engagement = CompetitiveEngagement(
                strategy_type=comp_data.get("strategy_type", "audience_education"),
                quality_gap_identified=comp_data.get("quality_gap_identified", ""),
                response_performance=comp_data.get("response_performance", {}),
                audience_interest_level=comp_data.get("audience_interest_level", "unknown"),
                conversion_potential=comp_data.get("conversion_potential", "unknown"),
                competitor_response=comp_data.get("competitor_response", "")
            )
        
        conversation.audience_poaching = data.get("audience_poaching", False)
        
        # Messages
        conversation.messages = [
            SocialMessage.from_dict(msg_data) 
            for msg_data in data.get("messages", [])
        ]
        
        # Timestamps with safe parsing
        conversation.started_at = safe_iso_to_datetime(data.get("started_at"))
        conversation.last_activity = safe_iso_to_datetime(data.get("last_activity"))
        conversation.created_at = safe_iso_to_datetime(data.get("created_at"))
        conversation.updated_at = safe_iso_to_datetime(data.get("updated_at"))
        
        # Analysis
        conversation.conversation_summary = data.get("conversation_summary", "")
        conversation.topics_discussed = data.get("topics_discussed", [])
        conversation.dominant_sentiment = data.get("dominant_sentiment", "neutral")
        conversation.user_satisfaction = data.get("user_satisfaction", "unknown")
        
        # Conversion tracking
        conversation.conversion_opportunity_identified = data.get("conversion_opportunity_identified", False)
        conversation.conversion_attempt_made = data.get("conversion_attempt_made", False)
        conversation.conversion_successful = data.get("conversion_successful", False)
        
        # Quality metrics (cached values)
        conversation.avg_response_time = data.get("avg_response_time", 0)
        conversation.user_engagement_score = data.get("user_engagement_score", 0)
        conversation._engagement_score_last_calculated = safe_iso_to_datetime(
            data.get("_engagement_score_last_calculated")
        ) or datetime.utcnow()
        
        # Lightweight message tracking
        conversation.message_count = data.get("message_count", len(conversation.messages))
        conversation.user_message_count = data.get("user_message_count", 0)
        conversation.our_response_count = data.get("our_response_count", 0)
        conversation.last_message_snippet = data.get("last_message_snippet", "")
        
        return conversation
    
    def __str__(self) -> str:
        return f"SocialConversation({self.conversation_id}, type={self.conversation_type.value}, messages={len(self.messages)})"
    
    def __repr__(self) -> str:
        return self.__str__()

# Utility functions for conversation management
def create_comment_conversation(platform_user_id: str, our_content_id: str, content_topic: List[str]) -> SocialConversation:
    """Create a new public comment conversation"""
    conversation = SocialConversation(platform_user_id, ConversationType.PUBLIC_THREAD)
    conversation.context.our_content_id = our_content_id
    conversation.context.our_content_topic = content_topic
    conversation.context.conversation_starter = "comment_on_content"
    return conversation

def create_dm_conversation(platform_user_id: str, initiated_by: str = "user") -> SocialConversation:
    """Create a new private DM conversation"""
    conversation = SocialConversation(platform_user_id, ConversationType.PRIVATE_CONVERSATION)
    conversation.context.conversation_starter = f"dm_initiated_by_{initiated_by}"
    return conversation

def create_competitor_engagement(platform_user_id: str, competitor_account: str, 
                               competitor_content_id: str, strategy_type: str) -> SocialConversation:
    """Create a new competitive engagement conversation"""
    conversation = SocialConversation(platform_user_id, ConversationType.COMPETITIVE_ENGAGEMENT)
    conversation.context.content_ownership = ContentOwnership.COMPETITOR_CONTENT
    conversation.context.competitor_account = competitor_account
    conversation.context.competitor_content_id = competitor_content_id
    conversation.context.conversation_starter = f"competitive_engagement_{strategy_type}"
    conversation.audience_poaching = True
    
    conversation.competitive_engagement = CompetitiveEngagement(
        strategy_type=strategy_type
    )
    
    return conversation
