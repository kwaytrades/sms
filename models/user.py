# ===== models/user.py =====
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum

class PlanType(Enum):
    FREE = "free"
    PAID = "paid"
    PRO = "pro"

class SubscriptionStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRIALING = "trialing"
    CANCELED = "canceled"
    PAST_DUE = "past_due"

@dataclass
class PlanLimits:
    plan_type: str
    price: float
    messages_per_period: int
    period: str  # "week" or "month"
    features: List[str]
    cooloff_threshold: Optional[int] = None
    
    @classmethod
    def get_plan_config(cls):
        return {
            "free": cls(
                plan_type="free",
                price=0.0,
                messages_per_period=10,
                period="week",
                features=["basic_messages", "market_updates"]
            ),
            "paid": cls(
                plan_type="paid",
                price=29.0,
                messages_per_period=100,
                period="month",
                features=["advanced_messaging", "personalized_insights", "analytics", "portfolio_tracking"]
            ),
            "pro": cls(
                plan_type="pro",
                price=99.0,
                messages_per_period=999999,
                period="month",
                features=["unlimited_messaging", "trade_alerts", "real_time_notifications", "priority_support"],
                cooloff_threshold=50
            )
        }

@dataclass
class UserProfile:
    # Identity
    phone_number: str
    email: Optional[str] = None
    first_name: Optional[str] = None
    timezone: str = "US/Eastern"
    
    # Subscription
    plan_type: str = "free"
    subscription_status: str = "inactive"
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    trial_ends_at: Optional[datetime] = None
    
    # Trading Profile
    risk_tolerance: str = "medium"
    trading_experience: str = "intermediate"
    preferred_sectors: List[str] = field(default_factory=list)
    watchlist: List[str] = field(default_factory=list)
    trading_style: str = "swing"
    
    # Behavioral Patterns (Learned)
    communication_style: Dict[str, Any] = field(default_factory=dict)
    response_patterns: Dict[str, Any] = field(default_factory=dict)
    trading_behavior: Dict[str, Any] = field(default_factory=dict)
    speech_patterns: Dict[str, Any] = field(default_factory=dict)
    
    # Connected Accounts
    plaid_access_tokens: List[Dict] = field(default_factory=list)
    
    # Usage Tracking
    total_messages_sent: int = 0
    total_messages_received: int = 0
    last_active_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    
    # Preferences
    daily_insights_enabled: bool = True
    premarket_enabled: bool = True
    market_close_enabled: bool = True
    promotional_messages: bool = True
    
    _id: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.last_active_at is None:
            self.last_active_at = datetime.utcnow()
