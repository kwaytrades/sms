# services/database/config.py
"""
Database Service Configuration
Centralized, environment-aware configuration management
"""

import os
from typing import Dict, List, Any, Optional
from pydantic import BaseSettings, Field, validator
from datetime import timedelta


class DatabaseServiceConfig(BaseSettings):
    """
    Centralized database service configuration with environment overrides
    
    All settings can be overridden via environment variables with DB_SERVICE_ prefix
    Example: DB_SERVICE_INFERENCE_ENABLED=false
    """
    
    # ==========================================
    # AUTO-INFERENCE SETTINGS
    # ==========================================
    
    INFERENCE_ENABLED: bool = Field(
        default=True,
        description="Enable automatic profile inference"
    )
    
    INFERENCE_INTERVAL_HOURS: int = Field(
        default=24,
        ge=1,
        le=168,  # Max 1 week
        description="Hours between inference runs"
    )
    
    PROFILE_COMPLETENESS_THRESHOLD: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum profile completeness to skip inference"
    )
    
    # ==========================================
    # SERVICE CONFIGURATION
    # ==========================================
    
    REQUIRED_SERVICES: List[str] = Field(
        default=["users", "trading", "migrations"],
        description="Services required for operation"
    )
    
    OPTIONAL_SERVICES: List[str] = Field(
        default=["conversations", "analytics", "feature_flags", "compliance"],
        description="Services that can fail without breaking core functionality"
    )
    
    SERVICE_INITIALIZATION_TIMEOUT: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Service initialization timeout in seconds"
    )
    
    # ==========================================
    # HEALTH CHECK SETTINGS
    # ==========================================
    
    HEALTH_CHECK_TIMEOUT: float = Field(
        default=5.0,
        ge=1.0,
        le=30.0,
        description="Health check timeout in seconds"
    )
    
    DEGRADED_THRESHOLD: float = Field(
        default=0.7,
        ge=0.1,
        le=1.0,
        description="Minimum ratio of healthy services to avoid degraded status"
    )
    
    HEALTH_HISTORY_SIZE: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of health check results to keep in history"
    )
    
    # ==========================================
    # RATE LIMITING
    # ==========================================
    
    DEFAULT_RATE_LIMIT: int = Field(
        default=1000,
        ge=1,
        description="Default rate limit per window"
    )
    
    DEFAULT_RATE_WINDOW: int = Field(
        default=3600,
        ge=60,
        description="Default rate limit window in seconds"
    )
    
    RATE_LIMIT_SLIDING_WINDOW: bool = Field(
        default=True,
        description="Use sliding window for rate limiting"
    )
    
    # ==========================================
    # CACHING CONFIGURATION
    # ==========================================
    
    # Cache TTL values
    USER_CACHE_TTL: int = Field(
        default=3600,
        ge=300,
        description="User data cache TTL in seconds"
    )
    
    PERSONALITY_CACHE_TTL: int = Field(
        default=86400 * 7,  # 1 week
        ge=3600,
        description="Personality profile cache TTL in seconds"
    )
    
    ANALYSIS_CACHE_TTL: int = Field(
        default=3600,
        ge=300,
        description="Analysis result cache TTL in seconds"
    )
    
    GOAL_CACHE_TTL: int = Field(
        default=86400,
        ge=3600,
        description="Financial goal cache TTL in seconds"
    )
    
    ALERT_CACHE_TTL: int = Field(
        default=1800,
        ge=300,
        description="Alert cache TTL in seconds"
    )
    
    TRADE_CACHE_TTL: int = Field(
        default=3600,
        ge=300,
        description="Trade data cache TTL in seconds"
    )
    
    # Cache behavior
    CACHE_ENABLED: bool = Field(
        default=True,
        description="Enable caching across all services"
    )
    
    CACHE_FALLBACK_ENABLED: bool = Field(
        default=True,
        description="Enable fallback when cache operations fail"
    )
    
    # ==========================================
    # ERROR HANDLING & RETRY
    # ==========================================
    
    MAX_RETRIES: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for operations"
    )
    
    RETRY_BASE_DELAY: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Base delay for exponential backoff in seconds"
    )
    
    MAX_RETRY_DELAY: int = Field(
        default=60,
        ge=1,
        le=300,
        description="Maximum retry delay in seconds"
    )
    
    CIRCUIT_BREAKER_ENABLED: bool = Field(
        default=True,
        description="Enable circuit breaker pattern for service failures"
    )
    
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Consecutive failures before opening circuit"
    )
    
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Seconds to wait before attempting recovery"
    )
    
    # ==========================================
    # MONITORING & METRICS
    # ==========================================
    
    METRICS_ENABLED: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    
    PERFORMANCE_TRACKING: bool = Field(
        default=True,
        description="Enable detailed performance tracking"
    )
    
    METRIC_HISTORY_SIZE: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Number of metric entries to keep in memory"
    )
    
    SLOW_OPERATION_THRESHOLD: float = Field(
        default=1.0,
        ge=0.1,
        le=30.0,
        description="Threshold for logging slow operations in seconds"
    )
    
    # ==========================================
    # DATABASE SPECIFIC SETTINGS
    # ==========================================
    
    # MongoDB settings
    MONGO_MAX_POOL_SIZE: int = Field(
        default=50,
        ge=5,
        le=200,
        description="MongoDB maximum connection pool size"
    )
    
    MONGO_MIN_POOL_SIZE: int = Field(
        default=5,
        ge=1,
        le=50,
        description="MongoDB minimum connection pool size"
    )
    
    MONGO_CONNECTION_TIMEOUT: int = Field(
        default=10000,
        ge=1000,
        le=30000,
        description="MongoDB connection timeout in milliseconds"
    )
    
    # Redis settings
    REDIS_MAX_CONNECTIONS: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Redis maximum connection pool size"
    )
    
    REDIS_CONNECTION_TIMEOUT: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Redis connection timeout in seconds"
    )
    
    REDIS_SOCKET_TIMEOUT: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Redis socket timeout in seconds"
    )
    
    # ==========================================
    # FEATURE FLAGS
    # ==========================================
    
    FEATURE_AUTO_INFERENCE: bool = Field(
        default=True,
        description="Enable automatic profile inference"
    )
    
    FEATURE_BACKGROUND_JOBS: bool = Field(
        default=True,
        description="Enable background job processing"
    )
    
    FEATURE_ADVANCED_CACHING: bool = Field(
        default=True,
        description="Enable advanced caching strategies"
    )
    
    FEATURE_REAL_TIME_ANALYTICS: bool = Field(
        default=True,
        description="Enable real-time analytics"
    )
    
    FEATURE_COMPLIANCE_TRACKING: bool = Field(
        default=True,
        description="Enable compliance and audit tracking"
    )
    
    # ==========================================
    # ENVIRONMENT SPECIFIC
    # ==========================================
    
    ENVIRONMENT: str = Field(
        default="development",
        regex="^(development|staging|production)$",
        description="Current environment"
    )
    
    DEBUG_MODE: bool = Field(
        default=False,
        description="Enable debug mode with verbose logging"
    )
    
    ENABLE_REQUEST_LOGGING: bool = Field(
        default=True,
        description="Enable detailed request logging"
    )
    
    LOG_LEVEL: str = Field(
        default="INFO",
        regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Logging level"
    )
    
    # ==========================================
    # SECURITY SETTINGS
    # ==========================================
    
    ENABLE_AUDIT_LOGGING: bool = Field(
        default=True,
        description="Enable audit logging for sensitive operations"
    )
    
    DATA_ENCRYPTION_AT_REST: bool = Field(
        default=False,
        description="Enable data encryption at rest"
    )
    
    REQUIRE_TLS: bool = Field(
        default=True,
        description="Require TLS for all connections"
    )
    
    # ==========================================
    # BUSINESS LOGIC LIMITS
    # ==========================================
    
    MAX_GOALS_PER_USER: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum financial goals per user"
    )
    
    MAX_ALERTS_PER_USER: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum alerts per user"
    )
    
    MAX_TRADE_MARKERS_PER_QUERY: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Maximum trade markers returned per query"
    )
    
    MAX_USAGE_HISTORY_DAYS: int = Field(
        default=90,
        ge=7,
        le=365,
        description="Maximum days of usage history to retain"
    )
    
    # ==========================================
    # MIGRATION SETTINGS
    # ==========================================
    
    MIGRATION_BATCH_SIZE: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Default migration batch size"
    )
    
    MIGRATION_MAX_CONCURRENT: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent migrations"
    )
    
    MIGRATION_CHECKPOINT_INTERVAL: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Save checkpoint every N items during migration"
    )
    
    # ==========================================
    # VALIDATORS
    # ==========================================
    
    @validator('DEGRADED_THRESHOLD')
    def validate_degraded_threshold(cls, v):
        if not 0.1 <= v <= 1.0:
            raise ValueError('DEGRADED_THRESHOLD must be between 0.1 and 1.0')
        return v
    
    @validator('PROFILE_COMPLETENESS_THRESHOLD')
    def validate_completeness_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('PROFILE_COMPLETENESS_THRESHOLD must be between 0.0 and 1.0')
        return v
    
    @validator('MONGO_MIN_POOL_SIZE', 'MONGO_MAX_POOL_SIZE')
    def validate_mongo_pool_sizes(cls, v, values):
        if 'MONGO_MIN_POOL_SIZE' in values and 'MONGO_MAX_POOL_SIZE' in values:
            if values['MONGO_MIN_POOL_SIZE'] > values['MONGO_MAX_POOL_SIZE']:
                raise ValueError('MONGO_MIN_POOL_SIZE cannot be greater than MONGO_MAX_POOL_SIZE')
        return v
    
    # ==========================================
    # UTILITY METHODS
    # ==========================================
    
    def get_cache_ttl(self, cache_type: str) -> int:
        """Get TTL for specific cache type"""
        ttl_mapping = {
            "user": self.USER_CACHE_TTL,
            "personality": self.PERSONALITY_CACHE_TTL,
            "analysis": self.ANALYSIS_CACHE_TTL,
            "goal": self.GOAL_CACHE_TTL,
            "alert": self.ALERT_CACHE_TTL,
            "trade": self.TRADE_CACHE_TTL
        }
        return ttl_mapping.get(cache_type, 3600)  # Default 1 hour
    
    def get_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff"""
        delay = min(self.RETRY_BASE_DELAY * (2 ** attempt), self.MAX_RETRY_DELAY)
        return delay
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.ENVIRONMENT == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.ENVIRONMENT == "development"
    
    def get_mongo_config(self) -> Dict[str, Any]:
        """Get MongoDB configuration dictionary"""
        return {
            "maxPoolSize": self.MONGO_MAX_POOL_SIZE,
            "minPoolSize": self.MONGO_MIN_POOL_SIZE,
            "connectTimeoutMS": self.MONGO_CONNECTION_TIMEOUT,
            "serverSelectionTimeoutMS": 5000,
            "socketTimeoutMS": 5000,
            "retryWrites": True,
            "retryReads": True
        }
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration dictionary"""
        return {
            "decode_responses": True,
            "max_connections": self.REDIS_MAX_CONNECTIONS,
            "retry_on_timeout": True,
            "socket_connect_timeout": self.REDIS_CONNECTION_TIMEOUT,
            "socket_timeout": self.REDIS_SOCKET_TIMEOUT
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.dict()
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get all feature flags"""
        return {
            "auto_inference": self.FEATURE_AUTO_INFERENCE,
            "background_jobs": self.FEATURE_BACKGROUND_JOBS,
            "advanced_caching": self.FEATURE_ADVANCED_CACHING,
            "real_time_analytics": self.FEATURE_REAL_TIME_ANALYTICS,
            "compliance_tracking": self.FEATURE_COMPLIANCE_TRACKING
        }
    
    # ==========================================
    # CONFIGURATION VALIDATION
    # ==========================================
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate entire configuration and return validation report"""
        validation_report = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Production-specific validations
        if self.is_production():
            if self.DEBUG_MODE:
                validation_report["warnings"].append("DEBUG_MODE is enabled in production")
            
            if not self.REQUIRE_TLS:
                validation_report["errors"].append("TLS should be required in production")
                validation_report["valid"] = False
            
            if not self.ENABLE_AUDIT_LOGGING:
                validation_report["warnings"].append("Audit logging should be enabled in production")
        
        # Performance validations
        if self.MONGO_MAX_POOL_SIZE < 10:
            validation_report["warnings"].append("MongoDB pool size might be too small for production")
        
        if self.REDIS_MAX_CONNECTIONS < 10:
            validation_report["warnings"].append("Redis connection pool might be too small")
        
        # Security validations
        if not self.DATA_ENCRYPTION_AT_REST and self.is_production():
            validation_report["recommendations"].append("Consider enabling data encryption at rest")
        
        # Cache configuration validations
        if not self.CACHE_ENABLED:
            validation_report["warnings"].append("Caching is disabled - performance may be impacted")
        
        # Rate limiting validations
        if self.DEFAULT_RATE_LIMIT > 10000:
            validation_report["warnings"].append("Rate limit might be too high")
        
        return validation_report
    
    class Config:
        env_prefix = "DB_SERVICE_"
        case_sensitive = True
        validate_assignment = True
        extra = "forbid"  # Prevent unknown configuration keys


# Environment-specific configurations
class DevelopmentConfig(DatabaseServiceConfig):
    """Development environment configuration"""
    ENVIRONMENT: str = "development"
    DEBUG_MODE: bool = True
    LOG_LEVEL: str = "DEBUG"
    METRICS_ENABLED: bool = True
    CACHE_ENABLED: bool = True
    REQUIRE_TLS: bool = False


class StagingConfig(DatabaseServiceConfig):
    """Staging environment configuration"""
    ENVIRONMENT: str = "staging"
    DEBUG_MODE: bool = False
    LOG_LEVEL: str = "INFO"
    METRICS_ENABLED: bool = True
    CACHE_ENABLED: bool = True
    REQUIRE_TLS: bool = True
    ENABLE_AUDIT_LOGGING: bool = True


class ProductionConfig(DatabaseServiceConfig):
    """Production environment configuration"""
    ENVIRONMENT: str = "production"
    DEBUG_MODE: bool = False
    LOG_LEVEL: str = "WARNING"
    METRICS_ENABLED: bool = True
    CACHE_ENABLED: bool = True
    REQUIRE_TLS: bool = True
    ENABLE_AUDIT_LOGGING: bool = True
    DATA_ENCRYPTION_AT_REST: bool = True
    
    # More conservative settings for production
    MAX_RETRIES: int = 5
    HEALTH_CHECK_TIMEOUT: float = 3.0
    CIRCUIT_BREAKER_ENABLED: bool = True


# Configuration factory
def get_config(environment: Optional[str] = None) -> DatabaseServiceConfig:
    """Get configuration for specified environment"""
    env = environment or os.getenv("DB_SERVICE_ENVIRONMENT", "development")
    
    config_map = {
        "development": DevelopmentConfig,
        "staging": StagingConfig,
        "production": ProductionConfig
    }
    
    config_class = config_map.get(env, DatabaseServiceConfig)
    return config_class()


# Export configuration classes
__all__ = [
    'DatabaseServiceConfig',
    'DevelopmentConfig',
    'StagingConfig', 
    'ProductionConfig',
    'get_config'
]
