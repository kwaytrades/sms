# config.py - Unified Configuration
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Environment
    environment: str = "development"
    log_level: str = "INFO"
    
    # Database (Required)
    mongodb_url: str
    redis_url: str = "redis://localhost:6379"
    
    # External APIs
    openai_api_key: Optional[str] = None
    
    # EODHD for market data (REMOVED HARDCODED KEY!)
    eodhd_api_key: Optional[str] = None
    
    # Stripe for payments
    stripe_secret_key: Optional[str] = None
    stripe_webhook_secret: Optional[str] = None
    stripe_paid_price_id: Optional[str] = None
    stripe_pro_price_id: Optional[str] = None
    
    # Twilio for SMS
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_phone_number: Optional[str] = None
    
    # Plaid for brokerage connections (future feature)
    plaid_client_id: Optional[str] = None
    plaid_secret: Optional[str] = None
    plaid_env: str = "sandbox"
    
    # Technical Analysis Settings
    cache_popular_ttl: int = 1800  # 30 minutes for popular stocks
    cache_ondemand_ttl: int = 300  # 5 minutes for other stocks
    cache_afterhours_ttl: int = 3600  # 1 hour after market close
    
    # Rate Limiting
    free_plan_weekly_limit: int = 4
    paid_plan_weekly_limit: int = 40
    pro_plan_weekly_limit: int = 120
    
    # Message Costs (SMS segments)
    max_message_segments: int = 3  # Limit responses to 3 SMS segments
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Plan configurations
PLAN_LIMITS = {
    "free": {
        "weekly_limit": settings.free_plan_weekly_limit,
        "price": 0,
        "features": ["Basic stock analysis", "Limited messages"],
        "upgrade_message": "Upgrade to Paid for 10x more analysis!"
    },
    "paid": {
        "weekly_limit": settings.paid_plan_weekly_limit,
        "price": 29,
        "features": ["Advanced analysis", "Portfolio tracking", "Personalized insights"],
        "upgrade_message": "Upgrade to Pro for unlimited messages!"
    },
    "pro": {
        "weekly_limit": settings.pro_plan_weekly_limit,
        "price": 99,
        "features": ["Unlimited messages", "Real-time alerts", "Priority support"],
        "upgrade_message": "You're on our highest tier!"
    }
}

# Popular tickers for enhanced caching
POPULAR_TICKERS = [
    # FAANG + Mega Cap
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA',
    
    # Major Tech
    'NFLX', 'AMD', 'INTC', 'ORCL', 'CRM', 'ADBE', 'UBER', 'LYFT',
    
    # Financial
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP',
    
    # Healthcare & Pharma
    'JNJ', 'PFE', 'UNH', 'MRNA', 'ABBV', 'TMO', 'ABT', 'LLY',
    
    # Consumer & Retail
    'WMT', 'TGT', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 'DIS',
    
    # Energy & Industrial
    'XOM', 'CVX', 'COP', 'BA', 'CAT', 'GE', 'MMM', 'HON',
    
    # ETFs - Most Popular
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'IVV',
    'GLD', 'SLV', 'TLT', 'XLF', 'XLK', 'XLE', 'XLV',
    
    # Crypto-Related
    'COIN', 'MSTR', 'SQ', 'HOOD',
    
    # Meme Stocks
    'GME', 'AMC', 'BB', 'PLTR'
]

def is_popular_ticker(symbol: str) -> bool:
    """Check if ticker is in popular list for enhanced caching"""
    return symbol.upper() in POPULAR_TICKERS
