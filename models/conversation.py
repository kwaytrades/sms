# ===== models/conversation.py =====
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any

@dataclass
class ChatMessage:
    """Simple chat message"""
    user_id: str
    content: str
    direction: str  # "inbound" or "outbound"
    message_type: str
    timestamp: datetime = None
    session_id: Optional[str] = None
    _id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class Conversation:
    """Simple conversation container"""
    user_id: str
    session_id: str
    messages: List[Dict] = field(default_factory=list)
    session_start: datetime = None
    total_messages: int = 0
    _id: Optional[str] = None
    
    def __post_init__(self):
        if self.session_start is None:
            self.session_start = datetime.utcnow()
