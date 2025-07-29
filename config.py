# ===== config.py - CLAUDE-ENABLED VERSION =====
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any

class Settings(BaseSettings):
    # Database (Required)
    mongodb_url: str
    redis_url: str = "redis://localhost:6379"
    
    # External APIs (Optional for testing)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None  # NEW: Claude support
    stripe_secret_key: Optional[str] = None
    stripe_webhook_secret: Optional[str] = None
    stripe_paid_price_id: Optional[str] = "price_mock_paid"
    stripe_pro_price_id: Optional[str] = "price_mock_pro"
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_phone_number: Optional[str] = None
    plaid_client_id: Optional[str] = None
    plaid_secret: Optional[str] = None
    plaid_env: str = "sandbox"
    eodhd_api_key: Optional[str] = None
    ta_service_url: Optional[str] = "https://mock-ta-service.com"
    
    # App settings
    environment: str = "development"
    log_level: str = "INFO"
    testing_mode: bool = True
    database_name: str = "ai"
    prefer_claude: bool = True  # NEW: Claude preference setting
    
    # Plan limits (used by main.py)
    free_weekly_limit: int = 10
    paid_monthly_limit: int = 100
    pro_daily_cooloff: int = 50
    
    class Config:
        env_file = ".env"
    
    def get_capability_summary(self) -> Dict[str, Any]:
        """Get summary of available capabilities."""
        return {
            "openai_available": bool(self.openai_api_key),
            "claude_available": bool(self.anthropic_api_key),  # NEW
            "stripe_available": bool(self.stripe_secret_key),
            "twilio_available": bool(self.twilio_account_sid and self.twilio_auth_token),
            "plaid_available": bool(self.plaid_client_id and self.plaid_secret),
            "eodhd_available": bool(self.eodhd_api_key),
            "redis_configured": bool(self.redis_url),
            "mongodb_configured": bool(self.mongodb_url),
            "testing_mode": self.testing_mode,
            "environment": self.environment,
            "prefer_claude": self.prefer_claude  # NEW
        }
    
    def get_ai_config(self) -> Dict[str, Any]:
        """Get AI service configuration."""
        return {
            "openai_available": bool(self.openai_api_key),
            "claude_available": bool(self.anthropic_api_key),
            "prefer_claude": self.prefer_claude,
            "active_provider": "claude" if self.prefer_claude and self.anthropic_api_key else "openai" if self.openai_api_key else "none",
            "fallback_available": bool(self.openai_api_key and self.anthropic_api_key)
        }
    
    def validate_runtime_requirements(self) -> Dict[str, Any]:
        """Validate if system is ready for production."""
        issues = []
        capabilities = self.get_capability_summary()
        
        # Critical services
        if not capabilities["mongodb_configured"]:
            issues.append("MongoDB URL not configured")
        if not capabilities["twilio_available"] and not self.testing_mode:
            issues.append("Twilio not configured for SMS")
        
        # AI services
        if not capabilities["openai_available"] and not capabilities["claude_available"]:
            issues.append("No AI service configured - need OpenAI or Claude")
        
        # Recommended services
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
        
        ready_for_production = len(issues) == 0 and not self.testing_mode
        
        return {
            "ready_for_production": ready_for_production,
            "critical_issues": issues,
            "warnings": warnings,
            "capabilities": capabilities,
            "ai_config": self.get_ai_config(),
            "testing_mode": self.testing_mode
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration status."""
        return {
            "environment": self.environment,
            "testing_mode": self.testing_mode,
            "webhook_secrets_configured": bool(self.stripe_webhook_secret),
            "api_keys_configured": {
                "openai": bool(self.openai_api_key),
                "claude": bool(self.anthropic_api_key),  # NEW
                "stripe": bool(self.stripe_secret_key),
                "twilio": bool(self.twilio_auth_token),
                "plaid": bool(self.plaid_secret),
                "eodhd": bool(self.eodhd_api_key)
            }
        }
    
    def get_scheduler_config(self) -> Dict[str, Any]:
        """Get scheduler configuration."""
        return {
            "enabled": True,
            "timezone": "US/Eastern",
            "premarket_time": "09:00",
            "market_close_time": "16:00",
            "reminder_time": "Sunday 09:30",
            "reset_time": "Monday 09:30"
        }

settings = Settings()
