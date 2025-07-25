# models/trading.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any  # ← Add Any here

@dataclass
class Portfolio:
    accounts: List[Dict] = field(default_factory=list)
    positions: List[Dict] = field(default_factory=list)
    transactions: List[Dict] = field(default_factory=list)
    total_value: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class TradingData:
    user_id: str
    
    # Portfolio Data
    portfolio: Portfolio = field(default_factory=Portfolio)
    
    # Performance Metrics
    daily_pnl: float = 0.0
    ytd_return: float = 0.0
    win_rate: float = 0.0
    
    # Behavioral Insights (Learned)
    successful_patterns: Dict[str, float] = field(default_factory=dict)
    loss_patterns: Dict[str, float] = field(default_factory=dict)
    timing_patterns: Dict[str, Any] = field(default_factory=dict)  # This line was causing the error
    
    # Risk Profile
    actual_risk_tolerance: float = 0.5
    position_sizing_pattern: str = "conservative"
    hold_time_preference: int = 5
    
    # Data Quality
    last_updated: datetime = field(default_factory=datetime.utcnow)
    data_completeness: float = 0.0
    
    _id: Optional[str] = None

