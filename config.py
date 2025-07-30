# ===== config.py - COMPLETE MERGED VERSION =====
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any
from enum import Enum

class PlanType(Enum):
    FREE = "free"
    PAID = "paid"  # Restored original naming
    PRO = "pro"

class SubscriptionStatus(Enum):
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    PAUSED = "paused"

class Settings(BaseSettings):
    # Database (Required)
    mongodb_url: str
    redis_url: str = "redis://localhost:6379"
    
    # External APIs (Optional for testing)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None  # RESTORED: Claude support
    stripe_secret_key: Optional[str] = None
    stripe_webhook_secret: Optional[str] = None
    stripe_paid_price_id: Optional[str] = "price_mock_paid"  # RESTORED: Original naming
    stripe_pro_price_id: Optional[str] = "price_mock_pro"   # RESTORED: Original naming
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_phone_number: Optional[str] = None
    plaid_client_id: Optional[str] = None
    plaid_secret: Optional[str] = None
    plaid_env: str = "sandbox"
    eodhd_api_key: Optional[str] = None
    ta_service_url: Optional[str] = "https://mock-ta-service.com"
    
    # Memory Manager Settings (RESTORED)
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "us-east1-gcp"
    memory_stm_limit: int = 15
    memory_summary_trigger: int = 10
    
    # App settings
    environment: str = "development"
    log_level: str = "INFO"
    testing_mode: bool = True
    database_name: str = "ai"
    prefer_claude: bool = True  # RESTORED: Claude preference setting
    
    # SIMPLIFIED Plan Limits (Restored Original Approach)
    free_weekly_limit: int = 10
    paid_monthly_limit: int = 100  # RESTORED: Original naming (not "basic")
    pro_daily_cooloff: int = 50
    
    # Enhanced Plan Pricing
    paid_price: int = 29   # NEW: Pricing for paid plan
    pro_price: int = 99    # NEW: Pricing for pro plan
    
    # Enhanced Subscription Management (NEW)
    retention_discount_percent: int = 50
    retention_duration_months: int = 3
    max_retention_offers: int = 2
    payment_retry_attempts: int = 3
    
    # Feature flags (NEW)
    enable_portfolio_tracking: bool = True
    enable_real_time_alerts: bool = True
    enable_advanced_analysis: bool = True
    enable_options_analysis: bool = True
    enable_personal_trade_alerts: bool = True   # NEW: For PRO plan
    enable_portfolio_optimizer: bool = True     # NEW: For PRO plan
    enable_trading_coach: bool = True           # NEW: For PRO plan
    
    # Support settings (NEW)
    support_email: str = "support@tradingbot.com"
    support_phone: str = "1-800-TRADING"
    support_hours: str = "9am-5pm ET"
    
    class Config:
        env_file = ".env"
    
    def get_plan_config(self, plan_type: str) -> Dict[str, Any]:
        """Get comprehensive plan configuration - ENHANCED"""
        
        if plan_type == PlanType.FREE.value:
            return {
                "name": "Free",
                "price": 0,
                "weekly_limit": self.free_weekly_limit,
                "monthly_limit": self.free_weekly_limit * 4,  # Approximate monthly
                "features": [
                    "Basic stock analysis",
                    "Daily market insights", 
                    "Price quotes",
                    "Basic charts"
                ],
                "restrictions": [
                    "Limited messages",
                    "No portfolio tracking",
                    "No real-time alerts",
                    "Basic support only"
                ],
                "stripe_price_id": None
            }
        
        elif plan_type == PlanType.PAID.value:  # RESTORED: Original "paid" naming
            return {
                "name": "Paid",
                "price": self.paid_price,
                "weekly_limit": self.paid_monthly_limit // 4,  # Weekly approximation
                "monthly_limit": self.paid_monthly_limit,
                "features": [
                    "Advanced stock analysis",
                    "Portfolio tracking", 
                    "Personalized insights",
                    "Daily market updates",
                    "Technical indicators",
                    "Email support",
                    "Watchlist management",
                    "Performance tracking"
                ],
                "restrictions": [
                    "Limited messages per month",
                    "No real-time alerts",
                    "Standard support"
                ],
                "stripe_price_id": self.stripe_paid_price_id
            }
        
        elif plan_type == PlanType.PRO.value:
            return {
                "name": "Pro",
                "price": self.pro_price,
                "weekly_limit": None,  # Unlimited
                "monthly_limit": None,  # Unlimited
                "daily_cooloff": self.pro_daily_cooloff,
                "features": [
                    "Unlimited messages",
                    "Real-time price alerts",
                    "Advanced technical analysis",
                    "Personal trade alerts",      # NEW: Requested feature
                    "Portfolio optimizer",        # NEW: Requested feature  
                    "Trading coach",             # NEW: Requested feature
                    "Options analysis",
                    "Custom alerts",
                    "Priority support",
                    "Advanced charting",
                    "Market scanner",
                    "Risk analysis",
                    "Earnings calendar",
                    "News sentiment analysis"
                ],
                "restrictions": [
                    "Daily cooloff after 50 messages"
                ],
                "stripe_price_id": self.stripe_pro_price_id
            }
        
        else:
            # Default to free plan
            return self.get_plan_config(PlanType.FREE.value)
    
    def get_all_plans(self) -> Dict[str, Dict[str, Any]]:
        """Get all plan configurations"""
        return {
            plan.value: self.get_plan_config(plan.value) 
            for plan in PlanType
        }
    
    def get_capability_summary(self) -> Dict[str, Any]:
        """Get summary of available capabilities - RESTORED + ENHANCED"""
        return {
            "openai_available": bool(self.openai_api_key),
            "claude_available": bool(self.anthropic_api_key),  # RESTORED
            "stripe_available": bool(self.stripe_secret_key),
            "twilio_available": bool(self.twilio_account_sid and self.twilio_auth_token),
            "plaid_available": bool(self.plaid_client_id and self.plaid_secret),
            "eodhd_available": bool(self.eodhd_api_key),
            "redis_configured": bool(self.redis_url),
            "mongodb_configured": bool(self.mongodb_url),
            "memory_available": bool(self.pinecone_api_key and self.openai_api_key),  # RESTORED
            "testing_mode": self.testing_mode,
            "environment": self.environment,
            "prefer_claude": self.prefer_claude,  # RESTORED
            "stripe_prices_configured": bool(self.stripe_paid_price_id and self.stripe_pro_price_id),  # NEW
            "feature_flags": {  # NEW
                "portfolio_tracking": self.enable_portfolio_tracking,
                "real_time_alerts": self.enable_real_time_alerts,
                "advanced_analysis": self.enable_advanced_analysis,
                "options_analysis": self.enable_options_analysis,
                "personal_trade_alerts": self.enable_personal_trade_alerts,
                "portfolio_optimizer": self.enable_portfolio_optimizer,
                "trading_coach": self.enable_trading_coach
            }
        }
    
    def get_ai_config(self) -> Dict[str, Any]:
        """Get AI service configuration - RESTORED"""
        return {
            "openai_available": bool(self.openai_api_key),
            "claude_available": bool(self.anthropic_api_key),
            "prefer_claude": self.prefer_claude,
            "active_provider": "claude" if self.prefer_claude and self.anthropic_api_key else "openai" if self.openai_api_key else "none",
            "fallback_available": bool(self.openai_api_key and self.anthropic_api_key),
            "memory_enhanced": bool(self.pinecone_api_key and self.openai_api_key)
        }
    
    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory manager configuration - RESTORED"""
        return {
            "pinecone_available": bool(self.pinecone_api_key),
            "pinecone_environment": self.pinecone_environment,
            "stm_limit": self.memory_stm_limit,
            "summary_trigger": self.memory_summary_trigger,
            "memory_enabled": bool(self.pinecone_api_key and self.openai_api_key),
            "emotional_intelligence": bool(self.pinecone_api_key),
            "conversation_continuity": bool(self.pinecone_api_key)
        }
    
    def get_scheduler_config(self) -> Dict[str, Any]:
        """Get scheduler configuration - RESTORED"""
        return {
            "enabled": True,
            "timezone": "US/Eastern",
            "premarket_time": "09:00",
            "market_close_time": "16:00",
            "reminder_time": "Sunday 09:30",
            "reset_time": "Monday 09:30"
        }
    
    def get_feature_matrix(self) -> Dict[str, Dict[str, bool]]:
        """Get feature availability matrix by plan - NEW"""
        return {
            "basic_analysis": {
                PlanType.FREE.value: True,
                PlanType.PAID.value: True,
                PlanType.PRO.value: True
            },
            "portfolio_tracking": {
                PlanType.FREE.value: False,
                PlanType.PAID.value: True,
                PlanType.PRO.value: True
            },
            "real_time_alerts": {
                PlanType.FREE.value: False,
                PlanType.PAID.value: False,
                PlanType.PRO.value: True
            },
            "advanced_analysis": {
                PlanType.FREE.value: False,
                PlanType.PAID.value: True,
                PlanType.PRO.value: True
            },
            "personal_trade_alerts": {
                PlanType.FREE.value: False,
                PlanType.PAID.value: False,
                PlanType.PRO.value: True
            },
            "portfolio_optimizer": {
                PlanType.FREE.value: False,
                PlanType.PAID.value: False,
                PlanType.PRO.value: True
            },
            "trading_coach": {
                PlanType.FREE.value: False,
                PlanType.PAID.value: False,
                PlanType.PRO.value: True
            },
            "options_analysis": {
                PlanType.FREE.value: False,
                PlanType.PAID.value: False,
                PlanType.PRO.value: True
            },
            "priority_support": {
                PlanType.FREE.value: False,
                PlanType.PAID.value: False,
                PlanType.PRO.value: True
            },
            "unlimited_messages": {
                PlanType.FREE.value: False,
                PlanType.PAID.value: False,
                PlanType.PRO.value: True
            }
        }
    
    def get_business_metrics_config(self) -> Dict[str, Any]:
        """Get business metrics and KPI configuration - NEW"""
        return {
            "pricing": {
                "paid_monthly": self.paid_price,
                "pro_monthly": self.pro_price,
                "annual_discount": 20
            },
            "retention": {
                "discount_percent": self.retention_discount_percent,
                "duration_months": self.retention_duration_months,
                "max_offers": self.max_retention_offers
            },
            "limits": {
                "free": {
                    "weekly": self.free_weekly_limit
                },
                "paid": {
                    "monthly": self.paid_monthly_limit
                },
                "pro": {
                    "daily_cooloff": self.pro_daily_cooloff,
                    "unlimited": True
                }
            },
            "payment": {
                "retry_attempts": self.payment_retry_attempts,
                "currencies": ["USD"],
                "billing_cycles": ["monthly"]
            }
        }
    
    def validate_runtime_requirements(self) -> Dict[str, Any]:
        """Validate if system is ready for production - RESTORED + ENHANCED"""
        issues = []
        capabilities = self.get_capability_summary()
        
        # Critical services
        if not capabilities["mongodb_configured"]:
            issues.append("MongoDB URL not configured")
        if not capabilities["twilio_available"] and not self.testing_mode:
            issues.append("Twilio not configured for SMS")
        
        # AI services - RESTORED logic
        if not capabilities["openai_available"] and not capabilities["claude_available"]:
            issues.append("No AI service configured - need OpenAI or Claude")
        
        # Enhanced validation - NEW
        if not capabilities["stripe_available"] and not self.testing_mode:
            issues.append("Stripe not configured for payments")
        if not capabilities["stripe_prices_configured"] and not self.testing_mode:
            issues.append("Stripe price IDs not configured")
        
        # Recommended services - RESTORED + ENHANCED
        warnings = []
        if not capabilities["openai_available"] and not capabilities["claude_available"]:
            warnings.append("No AI services configured")
        elif not capabilities["openai_available"]:
            warnings.append("OpenAI not configured - no fallback if Claude fails")
        elif not capabilities["claude_available"]:
            warnings.append("Claude not configured - missing superior reasoning")
        
        if not capabilities["stripe_available"]:
            warnings.append("Stripe not configured - payments disabled")
        if not capabilities["redis_configured"]:
            warnings.append("Redis not configured - caching disabled")
        if not capabilities["eodhd_available"]:
            warnings.append("EODHD not configured - market data limited")
        if not capabilities["memory_available"]:
            warnings.append("Memory manager not configured - no conversation continuity")
        if not capabilities["plaid_available"]:
            warnings.append("Plaid not configured - portfolio tracking disabled")
        
        ready_for_production = len(issues) == 0 and not self.testing_mode
        
        return {
            "ready_for_production": ready_for_production,
            "critical_issues": issues,
            "warnings": warnings,
            "capabilities": capabilities,
            "ai_config": self.get_ai_config(),  # RESTORED
            "memory_config": self.get_memory_config(),  # RESTORED
            "plan_configs": self.get_all_plans(),  # NEW
            "testing_mode": self.testing_mode
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration status - RESTORED + ENHANCED"""
        return {
            "environment": self.environment,
            "testing_mode": self.testing_mode,
            "webhook_secrets_configured": bool(self.stripe_webhook_secret),
            "api_keys_configured": {
                "openai": bool(self.openai_api_key),
                "claude": bool(self.anthropic_api_key),  # RESTORED
                "stripe": bool(self.stripe_secret_key),
                "twilio": bool(self.twilio_auth_token),
                "plaid": bool(self.plaid_secret),
                "eodhd": bool(self.eodhd_api_key),
                "pinecone": bool(self.pinecone_api_key)  # RESTORED
            },
            "encryption": {  # NEW
                "database": "mongodb_tls_enabled" if "ssl" in self.mongodb_url else "standard",
                "redis": "tls_enabled" if "rediss://" in self.redis_url else "standard",
                "api_calls": "https_enforced"
            }
        }

# Initialize settings
settings = Settings()

# SIMPLIFIED Plan Limits (Restored Original Structure + Enhanced)
PLAN_LIMITS = {
    "free": {
        "weekly_limit": settings.free_weekly_limit,
        "price": 0,
        "features": settings.get_plan_config("free")["features"]
    },
    "paid": {  # RESTORED: Original naming (not "basic")
        "monthly_limit": settings.paid_monthly_limit,
        "price": settings.paid_price,
        "features": settings.get_plan_config("paid")["features"]
    },
    "pro": {
        "daily_cooloff": settings.pro_daily_cooloff,
        "price": settings.pro_price,
        "features": settings.get_plan_config("pro")["features"]
    }
}

# Popular tickers for caching (RESTORED)
POPULAR_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK.B', 
    'V', 'JNJ', 'WMT', 'JPM', 'PG', 'UNH', 'MA', 'HD', 'CVX', 'LLY',
    'ABBV', 'KO', 'AVGO', 'MRK', 'PEP', 'TMO', 'BAC', 'COST', 'XOM',
    'NKE', 'ABT', 'DHR', 'VZ', 'ADBE', 'ACN', 'MCD', 'PFE', 'TXN',
    'LIN', 'CRM', 'WFC', 'PM', 'NEE', 'NFLX', 'RTX', 'DIS', 'T', 'CMCSA',
    'AMD', 'ORCL', 'HON', 'QCOM', 'UPS', 'IBM', 'AMGN', 'INTU', 'CAT'
]

# Export everything for backward compatibility
__all__ = [
    'settings',
    'Settings', 
    'PlanType',
    'SubscriptionStatus',
    'PLAN_LIMITS',
    'POPULAR_TICKERS'
]
