# ===== models/conversation.py - SIMPLIFIED VERSION =====
# Add these imports to the TOP of models/conversation.py:
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

# ===== models/trading.py - SIMPLIFIED VERSION =====
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class TradingData:
    """Simple trading data"""
    user_id: str
    daily_pnl: float = 0.0
    ytd_return: float = 0.0
    last_updated: datetime = None
    _id: Optional[str] = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()
