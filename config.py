# ===== config.py - PRODUCTION READY CONFIGURATION =====
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Unified application settings using Pydantic."""
    
    # ===== DATABASE CONFIGURATION =====
    mongodb_url: str
    database_name: str = "ai"  # Now configurable instead of hardcoded
    redis_url: Optional[str] = "redis://localhost:6379"
    
    # ===== UPSTASH REDIS (Alternative to standard Redis) =====
    upstash_redis_rest_url: Optional[str] = None
    upstash_redis_rest_token: Optional[str] = None
    
    # ===== EXTERNAL APIS =====
    # OpenAI
    openai_api_key: Optional[str] = None
    
    # Stripe
    stripe_secret_key: Optional[str] = None
    stripe_webhook_secret: Optional[str] = None
    stripe_paid_price_id: Optional[str] = "price_default_paid"
    stripe_pro_price_id: Optional[str] = "price_default_pro"
    
    # Twilio
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_phone_number: Optional[str] = None
    
    # Plaid
    plaid_client_id: Optional[str] = None
    plaid_secret: Optional[str] = None
    plaid_env: str = "sandbox"
    
    # EODHD (NO MORE HARDCODED VALUES)
    eodhd_api_token: Optional[str] = None
    
    # Technical Analysis Service
    ta_service_url: Optional[str] = None
    
    # ===== APPLICATION SETTINGS =====
    environment: str = "development"
    log_level: str = "INFO"
    testing_mode: bool = False  # FIXED: Changed from True to False for production default
    
    # ===== CACHE CONFIGURATION =====
    cache_popular_ttl: int = 1800      # 30 minutes
    cache_ondemand_ttl: int = 300      # 5 minutes  
    cache_afterhours_ttl: int = 3600   # 1 hour
    market_timezone: str = "US/Eastern"
    
    # ===== SUBSCRIPTION LIMITS =====
    free_weekly_limit: int = 10
    paid_monthly_limit: int = 100
    pro_daily_cooloff: int = 50
    
    # ===== SECURITY SETTINGS =====
    enable_webhook_verification: bool = True
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    max_message_length: int = 1600
    
    # ===== NOTIFICATION SETTINGS =====
    enable_weekly_scheduler: bool = True
    scheduler_timezone: str = "US/Eastern"
    weekly_reset_day: int = 0  # Monday = 0
    weekly_reset_hour: int = 9
    weekly_reset_minute: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_critical_settings()
        self._log_configuration_status()
    
    def _validate_critical_settings(self):
        """Validate critical settings on startup."""
        
        # Database is always required
        if not self.mongodb_url:
            raise ValueError("MONGODB_URL is required")
        
        # Redis - either standard or Upstash required for caching
        has_standard_redis = bool(self.redis_url and self.redis_url != "redis://localhost:6379")
        has_upstash_redis = bool(self.upstash_redis_rest_url and self.upstash_redis_rest_token)
        
        if not (has_standard_redis or has_upstash_redis):
            if not self.testing_mode:
                raise ValueError("Either REDIS_URL or UPSTASH credentials required for production")
        
        # In production, require key external APIs
        if self.environment == "production" and not self.testing_mode:
            required_apis = [
                ("OPENAI_API_KEY", self.openai_api_key),
                ("TWILIO_ACCOUNT_SID", self.twilio_account_sid),
                ("TWILIO_AUTH_TOKEN", self.twilio_auth_token),
                ("EODHD_API_TOKEN", self.eodhd_api_token)
            ]
            
            missing_apis = [name for name, value in required_apis if not value]
            if missing_apis:
                raise ValueError(f"Production requires: {', '.join(missing_apis)}")
        
        # Validate webhook secrets if verification is enabled
        if self.enable_webhook_verification and self.environment == "production":
            if self.twilio_auth_token and not self.stripe_webhook_secret:
                import warnings
                warnings.warn("Webhook verification enabled but Stripe webhook secret missing")
    
    def _log_configuration_status(self):
        """Log configuration status for debugging"""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Configuration loaded - Environment: {self.environment}")
        logger.info(f"Testing mode: {self.testing_mode}")
        logger.info(f"Database: {'✅' if self.mongodb_url else '❌'} MongoDB")
        logger.info(f"Cache: {'✅' if self.has_redis_connection else '❌'} Redis")
        logger.info(f"SMS: {'✅' if self.has_sms_capabilities else '❌'} Twilio")
        logger.info(f"AI: {'✅' if self.has_ai_capabilities else '❌'} OpenAI")
        logger.info(f"Payments: {'✅' if self.has_payment_capabilities else '❌'} Stripe")
        logger.info(f"Market Data: {'✅' if self.eodhd_api_token else '❌'} EODHD")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development.""" 
        return self.environment.lower() == "development"
    
    @property
    def is_testing(self) -> bool:
        """Check if in testing mode."""
        return self.testing_mode
    
    @property
    def has_redis_connection(self) -> bool:
        """Check if any Redis connection is available."""
        return bool(
            (self.redis_url and self.redis_url != "redis://localhost:6379") or
            (self.upstash_redis_rest_url and self.upstash_redis_rest_token)
        )
    
    @property
    def has_sms_capabilities(self) -> bool:
        """Check if SMS capabilities are configured."""
        return bool(self.twilio_account_sid and self.twilio_auth_token and self.twilio_phone_number)
    
    @property
    def has_ai_capabilities(self) -> bool:
        """Check if AI capabilities are configured."""
        return bool(self.openai_api_key)
    
    @property
    def has_payment_capabilities(self) -> bool:
        """Check if payment processing is configured."""
        return bool(self.stripe_secret_key)
    
    @property
    def has_market_data(self) -> bool:
        """Check if market data is configured."""
        return bool(self.eodhd_api_token)
    
    @property
    def has_portfolio_tracking(self) -> bool:
        """Check if portfolio tracking is configured."""
        return bool(self.plaid_client_id and self.plaid_secret)
    
    def get_capability_summary(self) -> dict:
        """Get summary of configured capabilities."""
        return {
            "database": bool(self.mongodb_url),
            "cache": self.has_redis_connection,
            "sms": self.has_sms_capabilities,
            "ai": self.has_ai_capabilities,
            "payments": self.has_payment_capabilities,
            "market_data": self.has_market_data,
            "portfolio_tracking": self.has_portfolio_tracking,
            "weekly_scheduler": self.enable_weekly_scheduler,
            "webhook_verification": self.enable_webhook_verification,
            "rate_limiting": self.enable_rate_limiting
        }
    
    def get_security_config(self) -> dict:
        """Get security-related configuration."""
        return {
            "webhook_verification": self.enable_webhook_verification,
            "rate_limiting": self.enable_rate_limiting,
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_message_length": self.max_message_length,
            "environment": self.environment,
            "testing_mode": self.testing_mode
        }
    
    def get_scheduler_config(self) -> dict:
        """Get scheduler configuration."""
        return {
            "enabled": self.enable_weekly_scheduler,
            "timezone": self.scheduler_timezone,
            "reset_day": self.weekly_reset_day,
            "reset_hour": self.weekly_reset_hour,
            "reset_minute": self.weekly_reset_minute,
            "reset_time_display": f"{'Monday' if self.weekly_reset_day == 0 else 'Day ' + str(self.weekly_reset_day)} {self.weekly_reset_hour:02d}:{self.weekly_reset_minute:02d}"
        }
    
    def validate_runtime_requirements(self) -> dict:
        """Validate all runtime requirements and return status."""
        requirements = {
            "database": {
                "required": True,
                "configured": bool(self.mongodb_url),
                "status": "✅" if self.mongodb_url else "❌",
                "message": "MongoDB connection string" if self.mongodb_url else "Missing MongoDB URL"
            },
            "cache": {
                "required": not self.testing_mode,
                "configured": self.has_redis_connection,
                "status": "✅" if self.has_redis_connection or self.testing_mode else "❌",
                "message": "Redis caching available" if self.has_redis_connection else "No Redis connection"
            },
            "sms": {
                "required": not self.testing_mode,
                "configured": self.has_sms_capabilities,
                "status": "✅" if self.has_sms_capabilities or self.testing_mode else "❌",
                "message": "Twilio SMS ready" if self.has_sms_capabilities else "Missing Twilio credentials"
            },
            "ai": {
                "required": not self.testing_mode,
                "configured": self.has_ai_capabilities,
                "status": "✅" if self.has_ai_capabilities or self.testing_mode else "❌",
                "message": "OpenAI integration ready" if self.has_ai_capabilities else "Missing OpenAI API key"
            }
        }
        
        all_required_met = all(
            req["configured"] or not req["required"] 
            for req in requirements.values()
        )
        
        return {
            "overall_status": "✅ Ready" if all_required_met else "❌ Missing Requirements",
            "ready_for_production": all_required_met and not self.testing_mode,
            "requirements": requirements
        }

# Global settings instance
settings = Settings()

# Convenience functions for common checks
def require_database():
    """Decorator to require database connection."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not settings.mongodb_url:
                raise RuntimeError("Database connection required but not configured")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_redis():
    """Decorator to require Redis connection."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not settings.has_redis_connection:
                raise RuntimeError("Redis connection required but not configured")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_openai():
    """Decorator to require OpenAI API."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not settings.openai_api_key:
                raise RuntimeError("OpenAI API key required but not configured")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_production():
    """Decorator to require production environment."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if settings.testing_mode:
                raise RuntimeError("This function requires production mode")
            return func(*args, **kwargs)
        return wrapper
    return decorator
