# ===== models/user.py - SIMPLIFIED VERSION =====
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

@dataclass
class UserProfile:
    """Simplified user profile"""
    phone_number: str
    plan_type: str = "free"
    subscription_status: str = "trialing"
    email: Optional[str] = None
    first_name: Optional[str] = None
    
    # Usage tracking
    total_messages_sent: int = 0
    total_messages_received: int = 0
    messages_this_period: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Internal
    _id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            key: value for key, value in self.__dict__.items()
        }
