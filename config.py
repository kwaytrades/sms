# config.py - Enhanced with Gemini Integration + Legacy Features
import os
from typing import Optional, Dict, Any
from loguru import logger
from pydantic_settings import BaseSettings
from enum import Enum

class PlanType(Enum):
    FREE = "free"
    PAID = "paid"
    PRO = "pro"

class SubscriptionStatus(Enum):
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    PAUSED = "paused"

class Config(BaseSettings):
    """Enhanced configuration with Gemini personality analysis integration + legacy features"""
    
    # ==========================================
    # EXISTING CONFIGURATION (PRESERVED)
    # ==========================================
    
    # Database Configuration
    mongodb_url: str
    redis_url: str = "redis://localhost:6379"
    
    # API Keys (Existing + Restored)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None  # RESTORED: Claude support
    eodhd_api_key: Optional[str] = None
    marketaux_api_key: Optional[str] = None
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_phone_number: Optional[str] = None
    stripe_secret_key: Optional[str] = None
    stripe_webhook_secret: Optional[str] = None
    stripe_paid_price_id: Optional[str] = "price_mock_paid"
    stripe_pro_price_id: Optional[str] = "price_mock_pro"
    plaid_client_id: Optional[str] = None
    plaid_secret: Optional[str] = None
    plaid_env: str = "sandbox"
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
    
    # ==========================================
    # NEW GEMINI CONFIGURATION
    # ==========================================
    
    # Gemini API Configuration
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-1.5-flash"
    gemini_temperature: float = 0.1
    gemini_max_output_tokens: int = 2048
    
    # Personality Analysis Configuration
    personality_analysis_enabled: bool = True
    personality_cache_ttl: int = 3600  # 1 hour
    background_analysis_enabled: bool = True
    background_analysis_interval: int = 25  # Every 25 conversations
    
    # Cost Management
    gemini_cost_alert_threshold: float = 10.0  # $10 daily threshold
    gemini_cost_limit: float = 50.0  # $50 daily limit
    enable_aggressive_caching: bool = False
    
    # Performance Optimization
    max_concurrent_gemini_requests: int = 5
    personality_analysis_timeout: int = 3000  # 3 seconds
    fallback_to_regex: bool = True
    
    # Response Time Targets
    target_response_time_ms: int = 4000  # 4 seconds
    target_analysis_time_ms: int = 800   # 800ms for personality analysis
    
    # ==========================================
    # FEATURE FLAGS (ENHANCED)
    # ==========================================
    
    # Core Features
    enable_real_time_personality: bool = True
    enable_background_deep_analysis: bool = True
    enable_batch_alert_processing: bool = True
    
    # Advanced Features
    enable_global_intelligence: bool = True
    enable_personality_hooks: bool = True
    enable_confidence_scoring: bool = True
    
    # Development & Testing
    debug_personality_analysis: bool = False
    save_analysis_logs: bool = False
    mock_gemini_responses: bool = False
    
    class Config:
        env_file = ".env"
    
    # ==========================================
    # VALIDATION AND WARNINGS
    # ==========================================
    
    def get_plan_config(self, plan_type: str) -> Dict[str, Any]:
        """Get comprehensive plan configuration - ENHANCED"""
        
        if plan_type == PlanType.FREE.value:
            return {
                "name": "Free",
                "price": 0,
                "weekly_limit": self.free_weekly_limit,
                "monthly_limit": self.free_weekly_limit * 4,
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
        
        elif plan_type == PlanType.PAID.value:
            return {
                "name": "Paid",
                "price": self.paid_price,
                "weekly_limit": self.paid_monthly_limit // 4,
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
                    "Personal trade alerts",
                    "Portfolio optimizer",
                    "Trading coach",
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
            "gemini_available": bool(self.gemini_api_key),  # NEW
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
            "stripe_prices_configured": bool(self.stripe_paid_price_id and self.stripe_pro_price_id),
            "personality_analysis_enabled": self.personality_analysis_enabled,
            "feature_flags": {
                "portfolio_tracking": self.enable_portfolio_tracking,
                "real_time_alerts": self.enable_real_time_alerts,
                "advanced_analysis": self.enable_advanced_analysis,
                "options_analysis": self.enable_options_analysis,
                "personal_trade_alerts": self.enable_personal_trade_alerts,
                "portfolio_optimizer": self.enable_portfolio_optimizer,
                "trading_coach": self.enable_trading_coach,
                "gemini_personality": self.personality_analysis_enabled
            }
        }
    
    def get_ai_config(self) -> Dict[str, Any]:
        """Get AI service configuration - RESTORED + ENHANCED"""
        return {
            "openai_available": bool(self.openai_api_key),
            "claude_available": bool(self.anthropic_api_key),
            "gemini_available": bool(self.gemini_api_key),
            "prefer_claude": self.prefer_claude,
            "active_provider": "claude" if self.prefer_claude and self.anthropic_api_key else "openai" if self.openai_api_key else "none",
            "fallback_available": bool(self.openai_api_key and self.anthropic_api_key),
            "memory_enhanced": bool(self.pinecone_api_key and self.openai_api_key),
            "personality_analysis": {
                "gemini_enabled": bool(self.gemini_api_key),
                "fallback_to_regex": self.fallback_to_regex,
                "background_analysis": self.background_analysis_enabled,
                "real_time_analysis": self.enable_real_time_personality
            }
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
    
    def get_feature_matrix(self) -> Dict[str, Dict[str, bool]]:
        """Get feature availability matrix by plan - RESTORED"""
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
            },
            "gemini_personality": {
                PlanType.FREE.value: True,  # Available to all if API key provided
                PlanType.PAID.value: True,
                PlanType.PRO.value: True
            }
        }
    
    def get_business_metrics_config(self) -> Dict[str, Any]:
        """Get business metrics and KPI configuration - RESTORED"""
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
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration and return status report"""
        validation_results = {
            "status": "valid",
            "warnings": [],
            "errors": [],
            "features_enabled": []
        }
        
        # Check critical API keys
        if not self.gemini_api_key and self.personality_analysis_enabled:
            validation_results["warnings"].append(
                "Gemini API key not provided - personality analysis will use regex fallback"
            )
            validation_results["features_enabled"].append("regex_personality_fallback")
        elif self.gemini_api_key:
            validation_results["features_enabled"].append("gemini_personality_analysis")
        
        if not self.openai_api_key and not self.anthropic_api_key:
            validation_results["errors"].append("At least one AI API key (OpenAI or Claude) is required")
        elif self.anthropic_api_key:
            validation_results["features_enabled"].append("claude_ai")
        elif self.openai_api_key:
            validation_results["features_enabled"].append("openai_ai")
        
        if not self.mongodb_url or self.mongodb_url == "mongodb://localhost:27017/sms_trading_bot":
            validation_results["warnings"].append("Using local MongoDB - ensure it's running for production")
        
        # Check memory manager
        if self.pinecone_api_key and self.openai_api_key:
            validation_results["features_enabled"].append("memory_manager")
        elif self.pinecone_api_key:
            validation_results["warnings"].append("Pinecone configured but OpenAI missing - memory manager disabled")
        
        # Check performance configuration
        if self.target_response_time_ms > 5000:
            validation_results["warnings"].append(
                f"Response time target ({self.target_response_time_ms}ms) exceeds recommended 4000ms"
            )
        
        if self.personality_analysis_timeout > self.target_response_time_ms * 0.8:
            validation_results["warnings"].append(
                "Personality analysis timeout too high relative to response time target"
            )
        
        # Check cost management
        if self.gemini_cost_limit > 100:
            validation_results["warnings"].append(
                f"Daily Gemini cost limit (${self.gemini_cost_limit}) is quite high - monitor usage"
            )
        
        # Feature validation
        if self.background_analysis_enabled and not self.gemini_api_key:
            validation_results["warnings"].append(
                "Background analysis enabled but Gemini API not available - limited functionality"
            )
        
        # Set overall status
        if validation_results["errors"]:
            validation_results["status"] = "invalid"
        elif validation_results["warnings"]:
            validation_results["status"] = "valid_with_warnings"
        
        return validation_results
    
    def get_personality_config(self) -> Dict[str, Any]:
        """Get personality analysis specific configuration"""
        return {
            "enabled": self.personality_analysis_enabled,
            "gemini_api_key_provided": bool(self.gemini_api_key),
            "model": self.gemini_model,
            "temperature": self.gemini_temperature,
            "max_output_tokens": self.gemini_max_output_tokens,
            "cache_ttl": self.personality_cache_ttl,
            "background_analysis": self.background_analysis_enabled,
            "cost_management": {
                "alert_threshold": self.gemini_cost_alert_threshold,
                "daily_limit": self.gemini_cost_limit,
                "aggressive_caching": self.enable_aggressive_caching
            },
            "performance": {
                "max_concurrent_requests": self.max_concurrent_gemini_requests,
                "analysis_timeout_ms": self.personality_analysis_timeout,
                "fallback_enabled": self.fallback_to_regex
            }
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance-related configuration"""
        return {
            "targets": {
                "response_time_ms": self.target_response_time_ms,
                "analysis_time_ms": self.target_analysis_time_ms
            },
            "optimization": {
                "aggressive_caching": self.enable_aggressive_caching,
                "max_concurrent_gemini": self.max_concurrent_gemini_requests,
                "personality_timeout_ms": self.personality_analysis_timeout
            },
            "monitoring": {
                "debug_analysis": self.debug_personality_analysis,
                "save_logs": self.save_analysis_logs
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
        
        # Enhanced validation
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
        if not capabilities["gemini_available"]:
            warnings.append("Gemini not configured - personality analysis limited to regex")
        
        ready_for_production = len(issues) == 0 and not self.testing_mode
        
        return {
            "ready_for_production": ready_for_production,
            "critical_issues": issues,
            "warnings": warnings,
            "capabilities": capabilities,
            "ai_config": self.get_ai_config(),
            "memory_config": self.get_memory_config(),
            "plan_configs": self.get_all_plans(),
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
                "gemini": bool(self.gemini_api_key),  # NEW
                "stripe": bool(self.stripe_secret_key),
                "twilio": bool(self.twilio_auth_token),
                "plaid": bool(self.plaid_secret),
                "eodhd": bool(self.eodhd_api_key),
                "pinecone": bool(self.pinecone_api_key)  # RESTORED
            },
            "encryption": {
                "database": "mongodb_tls_enabled" if "ssl" in self.mongodb_url else "standard",
                "redis": "tls_enabled" if "rediss://" in self.redis_url else "standard",
                "api_calls": "https_enforced"
            }
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
    
    def log_configuration_status(self) -> None:
        """Log configuration status for debugging"""
        validation = self.validate_configuration()
        
        logger.info("🔧 Configuration Status:")
        logger.info(f"  Status: {validation['status']}")
        
        if validation["features_enabled"]:
            logger.info(f"  Features: {', '.join(validation['features_enabled'])}")
        
        if validation["warnings"]:
            for warning in validation["warnings"]:
                logger.warning(f"  ⚠️ {warning}")
        
        if validation["errors"]:
            for error in validation["errors"]:
                logger.error(f"  ❌ {error}")
        
        # Log personality analysis status
        personality_config = self.get_personality_config()
        if personality_config["enabled"]:
            if personality_config["gemini_api_key_provided"]:
                logger.info(f"  🤖 Gemini Analysis: Enabled ({personality_config['model']})")
            else:
                logger.info("  🤖 Personality Analysis: Regex Fallback Mode")
        else:
            logger.info("  🤖 Personality Analysis: Disabled")
        
        # Log AI services
        ai_config = self.get_ai_config()
        logger.info(f"  🧠 AI Provider: {ai_config['active_provider']}")
        if ai_config["fallback_available"]:
            logger.info("  🧠 AI Fallback: Available")
        
        # Log memory manager
        memory_config = self.get_memory_config()
        if memory_config["memory_enabled"]:
            logger.info("  🧠 Memory Manager: Enabled")


# Initialize settings instance using Pydantic
settings = Config()

# SIMPLIFIED Plan Limits (Restored Original Structure + Enhanced)
PLAN_LIMITS = {
    "free": {
        "weekly_limit": settings.free_weekly_limit,
        "price": 0,
        "features": settings.get_plan_config("free")["features"]
    },
    "paid": {
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
    'Config', 
    'PlanType',
    'SubscriptionStatus',
    'PLAN_LIMITS',
    'POPULAR_TICKERS'
]
