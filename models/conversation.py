# ===== models/conversation.py =====
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any

@dataclass
class ChatMessage:
    user_id: str
    content: str
    direction: str  # "inbound" or "outbound"
    message_type: str  # "user_query", "bot_response", "premarket_insight", "market_close", "alert", "command"
    
    # Analysis
    sentiment: float = 0.0
    urgency_level: str = "normal"
    topics_mentioned: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None
    response_time_ms: Optional[int] = None
    
    _id: Optional[str] = None

@dataclass
class Conversation:
    user_id: str
    session_id: str
    messages: List[Dict] = field(default_factory=list)
    
    # Session tracking
    session_start: datetime = field(default_factory=datetime.utcnow)
    session_end: Optional[datetime] = None
    total_messages: int = 0
    
    # Analysis
    topics_discussed: List[str] = field(default_factory=list)
    sentiment_trend: List[float] = field(default_factory=list)
    user_satisfaction: Optional[int] = None
    
    # Outcomes
    actions_taken: List[str] = field(default_factory=list)
    follow_up_needed: bool = False
    
    _id: Optional[str] = None
