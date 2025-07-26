# ===== models/user.py - FIXED ASYNC-COMPATIBLE USER MODELS =====
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from enum import Enum

class PlanType(Enum):
    """Subscription plan types."""
    FREE = "free"
    PAID = "paid"
    PRO = "pro"

class SubscriptionStatus(Enum):
    """Subscription status options."""
    TRIALING = "trialing"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    UNPAID = "unpaid"

@dataclass
class PlanLimits:
    """Plan-specific limits and features."""
    messages_per_period: int
    period: str  # "weekly", "monthly", "daily"
    price: float
    features: List[str] = field(default_factory=list)
    
    @classmethod
    def get_plan_config(cls) -> Dict[str, 'PlanLimits']:
        """Get configuration for all plans."""
        return {
            "free": cls(
                messages_per_period=10,
                period="weekly",
                price=0.0,
                features=["Basic market updates", "Stock analysis on demand"]
            ),
            "paid": cls(
                messages_per_period=100,
                period="monthly", 
                price=29.0,
                features=[
                    "Personalized insights", "Portfolio tracking", 
                    "Market analytics", "Priority support"
                ]
            ),
            "pro": cls(
                messages_per_period=999999,  # Unlimited
                period="monthly",
                price=99.0,
                features=[
                    "Unlimited messages", "Real-time trade alerts",
                    "Advanced screeners", "Priority support",
                    "Custom watchlists", "Portfolio analytics"
                ]
            )
        }

@dataclass
class CommunicationStyle:
    """User's communication preferences and patterns."""
    formality: str = "casual"  # "formal", "casual", "mixed"
    message_length: str = "medium"  # "short", "medium", "long"
    emoji_usage: bool = True
    technical_depth: str = "medium"  # "basic", "medium", "advanced"
    preferred_response_time: str = "immediate"  # "immediate", "delayed"

@dataclass
class SpeechPatterns:
    """Learned speech and interaction patterns."""
    vocabulary_level: str = "intermediate"  # "basic", "intermediate", "advanced"
    question_types: List[str] = field(default_factory=list)
    common_tickers: List[str] = field(default_factory=list)
    interaction_frequency: str = "moderate"  # "low", "moderate", "high"
    preferred_topics: List[str] = field(default_factory=list)

@dataclass
class ResponsePatterns:
    """User's response and engagement patterns."""
    preferred_response_time: str = "immediate"
    engagement_triggers: List[str] = field(default_factory=list)
    successful_trade_patterns: List[str] = field(default_factory=list)
    loss_triggers: List[str] = field(default_factory=list)
    satisfaction_indicators: List[str] = field(default_factory=list)

@dataclass
class TradingBehavior:
    """Learned trading behavior and preferences."""
    successful_sectors: List[str] = field(default_factory=list)
    preferred_position_sizes: List[str] = field(default_factory=list)
    typical_hold_time: Optional[str] = None
    win_rate: Optional[float] = None
    average_gain: Optional[float] = None
    average_loss: Optional[float] = None
    risk_patterns: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UserProfile:
    """Complete user profile with all data."""
    
    # ===== IDENTITY =====
    phone_number: str
    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    timezone: str = "US/Eastern"
    
    # ===== SUBSCRIPTION =====
    plan_type: str = "free"
    subscription_status: str = "trialing"
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    trial_ends_at: Optional[datetime] = None
    
    # ===== TRADING PROFILE =====
    risk_tolerance: str = "medium"  # "low", "medium", "high"
    trading_experience: str = "intermediate"  # "beginner", "intermediate", "expert"
    preferred_sectors: List[str] = field(default_factory=list)
    watchlist: List[str] = field(default_factory=list)
    trading_style: str = "swing"  # "day", "swing", "position", "long_term"
    
    # ===== BEHAVIORAL LEARNING =====
    communication_style: Dict[str, Any] = field(default_factory=lambda: {
        "formality": "casual",
        "message_length": "medium",
        "emoji_usage": True,
        "technical_depth": "medium"
    })
    
    speech_patterns: Dict[str, Any] = field(default_factory=lambda: {
        "vocabulary_level": "intermediate",
        "question_types": [],
        "common_tickers": [],
        "interaction_frequency": "moderate"
    })
    
    response_patterns: Dict[str, Any] = field(default_factory=lambda: {
        "preferred_response_time": "immediate",
        "engagement_triggers": [],
        "successful_trade_patterns": [],
        "loss_triggers": []
    })
    
    trading_behavior: Dict[str, Any] = field(default_factory=lambda: {
        "successful_sectors": [],
        "preferred_position_sizes": [],
        "typical_hold_time": None,
        "win_rate": None,
        "average_gain": None,
        "average_loss": None
    })
    
    # ===== EXTERNAL INTEGRATIONS =====
    plaid_access_tokens: List[str] = field(default_factory=list)
    connected_accounts: List[Dict[str, Any]] = field(default_factory=list)
    
    # ===== USAGE TRACKING =====
    total_messages_sent: int = 0
    total_messages_received: int = 0
    messages_this_period: int = 0
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_message_at: Optional[datetime] = None
    last_active_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # ===== NOTIFICATION PREFERENCES =====
    daily_insights_enabled: bool = True
    premarket_enabled: bool = True
    market_close_enabled: bool = True
    premarket_time: str = "09:00"
    market_close_time: str = "16:00"
    promotional_messages: bool = True
    
    # ===== TIMESTAMPS =====
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # ===== INTERNAL =====
    _id: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Ensure timestamps are timezone-aware
        if self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)
        if self.updated_at.tzinfo is None:
            self.updated_at = self.updated_at.replace(tzinfo=timezone.utc)
        if self.period_start.tzinfo is None:
            self.period_start = self.period_start.replace(tzinfo=timezone.utc)
        if self.last_active_at.tzinfo is None:
            self.last_active_at = self.last_active_at.replace(tzinfo=timezone.utc)
    
    @property
    def name(self) -> str:
        """Get user's display name."""
        if self.first_name:
            return self.first_name
        return self.phone_number
    
    @property
    def full_name(self) -> str:
        """Get user's full name."""
        parts = []
        if self.first_name:
            parts.append(self.first_name)
        if self.last_name:
            parts.append(self.last_name)
        return " ".join(parts) if parts else self.phone_number
    
    @property
    def plan_limits(self) -> PlanLimits:
        """Get current plan limits."""
        config = PlanLimits.get_plan_config()
        return config.get(self.plan_type, config["free"])
    
    @property
    def is_premium(self) -> bool:
        """Check if user has premium subscription."""
        return self.plan_type in ["paid", "pro"]
    
    @property
    def is_trial(self) -> bool:
        """Check if user is on trial."""
        return self.subscription_status == "trialing"
    
    @property
    def can_connect_portfolio(self) -> bool:
        """Check if user can connect portfolio (paid feature)."""
        return self.plan_type in ["paid", "pro"]
    
    @property
    def has_unlimited_messages(self) -> bool:
        """Check if user has unlimited messages."""
        return self.plan_type == "pro"
    
    def get_usage_percentage(self) -> float:
        """Get current usage as percentage of limit."""
        if self.has_unlimited_messages:
            return 0.0  # No limit
        
        limit = self.plan_limits.messages_per_period
        if limit == 0:
            return 100.0
        
        return min(100.0, (self.messages_this_period / limit) * 100)
    
    def get_messages_remaining(self) -> int:
        """Get number of messages remaining in current period."""
        if self.has_unlimited_messages:
            return 999999  # Effectively unlimited
        
        limit = self.plan_limits.messages_per_period
        return max(0, limit - self.messages_this_period)
    
    def should_send_usage_warning(self, threshold: float = 80.0) -> bool:
        """Check if usage warning should be sent."""
        return not self.has_unlimited_messages and self.get_usage_percentage() >= threshold
    
    def get_communication_preference(self, key: str, default: Any = None) -> Any:
        """Get communication style preference."""
        return self.communication_style.get(key, default)
    
    def update_communication_style(self, **updates):
        """Update communication style preferences."""
        self.communication_style.update(updates)
        self.updated_at = datetime.now(timezone.utc)
    
    def add_to_watchlist(self, ticker: str) -> bool:
        """Add ticker to watchlist."""
        ticker = ticker.upper()
        if ticker not in self.watchlist:
            self.watchlist.append(ticker)
            self.updated_at = datetime.now(timezone.utc)
            return True
        return False
    
    def remove_from_watchlist(self, ticker: str) -> bool:
        """Remove ticker from watchlist."""
        ticker = ticker.upper()
        if ticker in self.watchlist:
            self.watchlist.remove(ticker)
            self.updated_at = datetime.now(timezone.utc)
            return True
        return False
    
    def learn_from_message(self, message: str, symbols: List[str], intent: str):
        """Learn from user message and update patterns."""
        # Update speech patterns
        for symbol in symbols:
            if symbol not in self.speech_patterns["common_tickers"]:
                self.speech_patterns["common_tickers"].append(symbol)
        
        # Keep only last 20 tickers
        if len(self.speech_patterns["common_tickers"]) > 20:
            self.speech_patterns["common_tickers"] = self.speech_patterns["common_tickers"][-20:]
        
        # Update question types
        if intent not in self.speech_patterns["question_types"]:
            self.speech_patterns["question_types"].append(intent)
        
        # Analyze message characteristics
        message_length = "short" if len(message) < 20 else "long" if len(message) > 60 else "medium"
        has_emojis = any(ord(char) > 127 for char in message)
        
        # Update communication style
        self.communication_style["message_length"] = message_length
        self.communication_style["emoji_usage"] = has_emojis
        
        # Update timestamp
        self.updated_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value
            elif isinstance(value, list):
                result[key] = value.copy()
            elif isinstance(value, dict):
                result[key] = value.copy()
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Create UserProfile from dictionary."""
        # Handle datetime fields
        for field_name in ['created_at', 'updated_at', 'last_active_at', 'period_start', 'trial_ends_at']:
            if field_name in data and data[field_name] is not None:
                if isinstance(data[field_name], str):
                    data[field_name] = datetime.fromisoformat(data[field_name].replace('Z', '+00:00'))
                elif isinstance(data[field_name], datetime) and data[field_name].tzinfo is None:
                    data[field_name] = data[field_name].replace(tzinfo=timezone.utc)
        
        # Handle last_message_at separately as it can be None
        if 'last_message_at' in data and data['last_message_at'] is not None:
            if isinstance(data['last_message_at'], str):
                data['last_message_at'] = datetime.fromisoformat(data['last_message_at'].replace('Z', '+00:00'))
            elif isinstance(data['last_message_at'], datetime) and data['last_message_at'].tzinfo is None:
                data['last_message_at'] = data['last_message_at'].replace(tzinfo=timezone.utc)
        
        return cls(**data)
