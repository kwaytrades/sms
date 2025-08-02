# services/database_service.py - REFACTORED FACADE v5.0
"""
Unified Database Service - Modular Architecture
Provides unified interface while maintaining clean separation of concerns
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from loguru import logger

# Import specialized service modules
from services.db.base_db_service import BaseDBService
from services.db.users_service import UsersService
from services.db.conversation_service import ConversationService
from services.db.trading_service import TradingService
from services.db.analytics_service import AnalyticsService
from services.db.migration_service import MigrationService
from services.db.feature_flags_service import FeatureFlagsService
from services.db.compliance_service import ComplianceService

# Import inference services
from services.inference.profile_inference import ProfileInferenceService
from services.inference.context_inference import ContextInferenceService

# Legacy compatibility imports
from models.user import UserProfile
from models.conversation import ChatMessage


class UnifiedDatabaseService:
    """
    Unified Database Service v5.0 - Modular Architecture
    
    Provides a single interface for all database operations while maintaining
    clean separation of concerns through specialized service modules.
    
    This facade orchestrates:
    - Core database services (users, conversations, trading, analytics)
    - Auto-inference services for progressive profiling
    - Migration utilities for seamless upgrades
    - Compliance tools for GDPR/privacy requirements
    - Feature flags for controlled rollouts
    """
    
    def __init__(self):
        # Core infrastructure
        self.base_service: Optional[BaseDBService] = None
        
        # Specialized services
        self.users: Optional[UsersService] = None
        self.conversations: Optional[ConversationService] = None
        self.trading: Optional[TradingService] = None
        self.analytics: Optional[AnalyticsService] = None
        self.migrations: Optional[MigrationService] = None
        self.feature_flags: Optional[FeatureFlagsService] = None
        self.compliance: Optional[ComplianceService] = None
        
        # Auto-inference services
        self.profile_inference: Optional[ProfileInferenceService] = None
        self.context_inference: Optional[ContextInferenceService] = None
        
        logger.info("🚀 UnifiedDatabaseService v5.0 (Modular) initialized")

    async def initialize(self):
        """Initialize all database services with dependency injection"""
        try:
            # Initialize base infrastructure
            self.base_service = BaseDBService()
            await self.base_service.initialize()
            
            # Initialize specialized services with shared base
            self.users = UsersService(self.base_service)
            self.conversations = ConversationService(self.base_service)
            self.trading = TradingService(self.base_service)
            self.analytics = AnalyticsService(self.base_service)
            self.migrations = MigrationService(self.base_service)
            self.feature_flags = FeatureFlagsService(self.base_service)
            self.compliance = ComplianceService(self.base_service)
            
            # Initialize inference services
            self.profile_inference = ProfileInferenceService(self.base_service)
            self.context_inference = ContextInferenceService(self.base_service)
            
            # Initialize all services
            await self._initialize_all_services()
            
            logger.info("🎉 UnifiedDatabaseService fully initialized with modular architecture")
            
        except Exception as e:
            logger.exception(f"❌ Database initialization failed: {e}")
            raise

    async def _initialize_all_services(self):
        """Initialize all specialized services"""
        services = [
            self.users, self.conversations, self.trading, self.analytics,
            self.migrations, self.feature_flags, self.compliance,
            self.profile_inference, self.context_inference
        ]
        
        for service in services:
            if hasattr(service, 'initialize'):
                await service.initialize()

    # ==========================================
    # HEALTH & MONITORING
    # ==========================================

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check across all services"""
        health = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "unknown",
            "services": {}
        }
        
        try:
            # Base infrastructure health
            base_health = await self.base_service.health_check()
            health["services"]["infrastructure"] = base_health
            
            # Service-specific health checks
            service_healths = await asyncio.gather(
                self.users.health_check(),
                self.conversations.health_check(),
                self.trading.health_check(),
                self.analytics.health_check(),
                return_exceptions=True
            )
            
            health["services"]["users"] = service_healths[0] if not isinstance(service_healths[0], Exception) else {"status": "error", "error": str(service_healths[0])}
            health["services"]["conversations"] = service_healths[1] if not isinstance(service_healths[1], Exception) else {"status": "error", "error": str(service_healths[1])}
            health["services"]["trading"] = service_healths[2] if not isinstance(service_healths[2], Exception) else {"status": "error", "error": str(service_healths[2])}
            health["services"]["analytics"] = service_healths[3] if not isinstance(service_healths[3], Exception) else {"status": "error", "error": str(service_healths[3])}
            
            # Determine overall health
            all_healthy = all(
                service.get("status") == "healthy" 
                for service in health["services"].values()
                if isinstance(service, dict)
            )
            health["overall_status"] = "healthy" if all_healthy else "degraded"
            
            return health
            
        except Exception as e:
            logger.exception(f"❌ Health check failed: {e}")
            health["overall_status"] = "error"
            health["error"] = str(e)
            return health

    # ==========================================
    # USER OPERATIONS (Delegate to UsersService)
    # ==========================================

    async def get_user_by_phone(self, phone_number: str) -> Optional[UserProfile]:
        """Get user by phone number with auto-inference"""
        user = await self.users.get_by_phone(phone_number)
        
        # Auto-inference hook
        if user and await self._should_run_inference(user):
            enhanced_user = await self.profile_inference.enhance_user_profile(user)
            if enhanced_user != user:
                await self.users.save(enhanced_user)
                return enhanced_user
        
        return user

    async def get_user_by_id(self, user_id: str) -> Optional[UserProfile]:
        """Get user by ID with auto-inference"""
        user = await self.users.get_by_id(user_id)
        
        # Auto-inference hook
        if user and await self._should_run_inference(user):
            enhanced_user = await self.profile_inference.enhance_user_profile(user)
            if enhanced_user != user:
                await self.users.save(enhanced_user)
                return enhanced_user
        
        return user

    async def save_user(self, user: UserProfile) -> str:
        """Save user with auto-inference"""
        # Pre-save inference
        enhanced_user = await self.profile_inference.enhance_user_profile(user)
        return await self.users.save(enhanced_user)

    async def update_user_activity(self, phone_number: str) -> bool:
        """Update user activity"""
        return await self.users.update_activity(phone_number)

    # ==========================================
    # CONVERSATION OPERATIONS
    # ==========================================

    async def get_conversation_context(self, phone_number: str) -> Dict[str, Any]:
        """Get conversation context with auto-enhancement"""
        context = await self.conversations.get_context(phone_number)
        
        # Context inference hook
        enhanced_context = await self.context_inference.enhance_context(context, phone_number)
        
        return enhanced_context

    async def save_enhanced_message(self, phone_number: str, user_message: str, 
                                   bot_response: str, intent_data: Dict, 
                                   symbols: List[str] = None, 
                                   context_used: Dict = None) -> bool:
        """Save enhanced message with context updates"""
        success = await self.conversations.save_message(
            phone_number, user_message, bot_response, intent_data, symbols, context_used
        )
        
        # Trigger context inference after message save
        if success:
            await self.context_inference.update_from_message(
                phone_number, user_message, bot_response, intent_data, symbols
            )
        
        return success

    async def get_recent_messages(self, phone_number: str, limit: int = 10) -> List[Dict]:
        """Get recent messages"""
        return await self.conversations.get_recent_messages(phone_number, limit)

    # ==========================================
    # USAGE TRACKING & LIMITS
    # ==========================================

    async def get_usage_count(self, user_id: str, period: str) -> int:
        """Get usage count"""
        return await self.users.get_usage_count(user_id, period)

    async def increment_usage(self, user_id: str, period: str, ttl: int):
        """Increment usage"""
        await self.users.increment_usage(user_id, period, ttl)

    async def get_user_usage_summary(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive usage summary"""
        return await self.users.get_usage_summary(user_id)

    # ==========================================
    # GOAL MANAGEMENT
    # ==========================================

    async def save_financial_goal(self, user_id: str, goal_data: Dict) -> str:
        """Save financial goal"""
        return await self.trading.save_goal(user_id, goal_data)

    async def get_user_goals(self, user_id: str) -> List[Dict]:
        """Get user goals"""
        return await self.trading.get_goals(user_id)

    async def update_goal_progress(self, user_id: str, goal_id: str, new_amount: float) -> bool:
        """Update goal progress"""
        return await self.trading.update_goal_progress(user_id, goal_id, new_amount)

    # ==========================================
    # ALERT SYSTEM
    # ==========================================

    async def save_user_alert(self, user_id: str, alert_data: Dict) -> str:
        """Save user alert"""
        return await self.trading.save_alert(user_id, alert_data)

    async def get_active_alerts(self, user_id: str = None) -> List[Dict]:
        """Get active alerts"""
        return await self.trading.get_active_alerts(user_id)

    async def record_alert_trigger(self, alert_id: str, trigger_data: Dict) -> bool:
        """Record alert trigger"""
        return await self.trading.record_alert_trigger(alert_id, trigger_data)

    # ==========================================
    # TRADE TRACKING
    # ==========================================

    async def save_trade_marker(self, user_id: str, marker_data: Dict) -> str:
        """Save trade marker"""
        return await self.trading.save_trade_marker(user_id, marker_data)

    async def get_trade_performance(self, user_id: str, symbol: str = None) -> List[Dict]:
        """Get trade performance"""
        return await self.trading.get_trade_performance(user_id, symbol)

    # ==========================================
    # ANALYTICS & METRICS
    # ==========================================

    async def get_user_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get user analytics"""
        return await self.analytics.get_user_analytics(user_id, days)

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        return await self.analytics.get_system_metrics()

    # ==========================================
    # ONBOARDING SUPPORT
    # ==========================================

    async def save_onboarding_progress(self, user_id: str, progress_data: Dict) -> bool:
        """Save onboarding progress"""
        return await self.users.save_onboarding_progress(user_id, progress_data)

    async def get_onboarding_progress(self, user_id: str) -> Optional[Dict]:
        """Get onboarding progress"""
        return await self.users.get_onboarding_progress(user_id)

    # ==========================================
    # FEATURE FLAGS
    # ==========================================

    async def get_feature_flags(self, user_id: str = None) -> Dict[str, bool]:
        """Get feature flags"""
        return await self.feature_flags.get_flags(user_id)

    async def set_feature_flag(self, flag_name: str, enabled: bool, user_id: str = None) -> bool:
        """Set feature flag"""
        return await self.feature_flags.set_flag(flag_name, enabled, user_id)

    # ==========================================
    # STOCK DATA OPERATIONS (KeyBuilder Compatible)
    # ==========================================

    async def get_stock_technical(self, symbol: str) -> Optional[Dict]:
        """Get technical analysis data with migration"""
        return await self.migrations.get_stock_technical(symbol)

    async def get_stock_fundamental(self, symbol: str) -> Optional[Dict]:
        """Get fundamental analysis data with migration"""
        return await self.migrations.get_stock_fundamental(symbol)

    async def set_stock_data(self, symbol: str, data_type: str, data: Dict, ttl: int = 3600) -> bool:
        """Set stock data"""
        return await self.migrations.set_stock_data(symbol, data_type, data, ttl)

    # ==========================================
    # PERSONALITY ENGINE COMPATIBILITY
    # ==========================================

    async def get_personality_profile(self, user_id: str) -> Optional[Dict]:
        """Get personality profile"""
        return await self.users.get_personality_profile(user_id)

    async def save_personality_profile(self, user_id: str, profile: Dict) -> bool:
        """Save personality profile"""
        return await self.users.save_personality_profile(user_id, profile)

    async def cache_analysis_result(self, user_id: str, analysis_data: Dict, ttl: int = 3600) -> bool:
        """Cache analysis result"""
        return await self.users.cache_analysis_result(user_id, analysis_data, ttl)

    async def get_cached_analysis(self, user_id: str) -> Optional[Dict]:
        """Get cached analysis"""
        return await self.users.get_cached_analysis(user_id)

    # ==========================================
    # MIGRATION & MAINTENANCE
    # ==========================================

    async def migrate_all_users(self, limit: int = None) -> Dict:
        """Migrate all users"""
        return await self.migrations.migrate_all_users(limit)

    async def get_migration_stats(self) -> Dict:
        """Get migration statistics"""
        return await self.migrations.get_stats()

    async def cleanup_old_keys(self, dry_run: bool = True) -> Dict:
        """Clean up old Redis keys"""
        return await self.migrations.cleanup_old_keys(dry_run)

    # ==========================================
    # GDPR COMPLIANCE
    # ==========================================

    async def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all user data for GDPR compliance"""
        return await self.compliance.export_user_data(user_id)

    async def delete_user_data(self, user_id: str) -> Dict[str, Any]:
        """Delete all user data for GDPR compliance"""
        return await self.compliance.delete_user_data(user_id)

    # ==========================================
    # RATE LIMITING
    # ==========================================

    async def check_rate_limit(self, identifier: str, limit: int, window_seconds: int) -> Tuple[bool, int]:
        """Check rate limit"""
        return await self.base_service.check_rate_limit(identifier, limit, window_seconds)

    async def reset_rate_limit(self, identifier: str, window_seconds: int) -> bool:
        """Reset rate limit"""
        return await self.base_service.reset_rate_limit(identifier, window_seconds)

    # ==========================================
    # LEGACY COMPATIBILITY
    # ==========================================

    async def save_message(self, message: ChatMessage) -> str:
        """Legacy compatibility: Save chat message"""
        return await self.conversations.save_legacy_message(message)

    async def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Legacy compatibility: Get conversation history"""
        return await self.conversations.get_legacy_history(user_id, limit)

    async def save_trading_data(self, user_id: str, symbol: str, data: Dict) -> str:
        """Legacy compatibility: Save trading data"""
        return await self.trading.save_legacy_trading_data(user_id, symbol, data)

    async def get_trading_data(self, user_id: str, symbol: str = None, limit: int = 10) -> List[Dict]:
        """Legacy compatibility: Get trading data"""
        return await self.trading.get_legacy_trading_data(user_id, symbol, limit)

    # ==========================================
    # HELPER METHODS
    # ==========================================

    async def _should_run_inference(self, user: UserProfile) -> bool:
        """Determine if auto-inference should run for user"""
        try:
            # Check if enough time has passed since last inference
            if hasattr(user, 'last_inference_at') and user.last_inference_at:
                from datetime import timedelta
                if datetime.utcnow() - user.last_inference_at < timedelta(hours=24):
                    return False
            
            # Check if profile is significantly incomplete
            profile_completeness = await self.profile_inference.calculate_completeness(user)
            
            # Run inference if profile is less than 80% complete
            return profile_completeness < 0.8
            
        except Exception as e:
            logger.exception(f"❌ Error checking inference requirements: {e}")
            return False

    # ==========================================
    # PROPERTIES FOR DIRECT ACCESS
    # ==========================================

    @property
    def mongo_client(self):
        """Direct access to MongoDB client"""
        return self.base_service.mongo_client if self.base_service else None

    @property
    def db(self):
        """Direct access to MongoDB database"""
        return self.base_service.db if self.base_service else None

    @property
    def redis(self):
        """Direct access to Redis client"""
        return self.base_service.redis if self.base_service else None

    @property
    def key_builder(self):
        """Direct access to KeyBuilder interface"""
        return self.base_service.key_builder if self.base_service else None

    # ==========================================
    # CLEANUP & SHUTDOWN
    # ==========================================

    async def close(self):
        """Close all database connections"""
        try:
            if self.base_service:
                await self.base_service.close()
            logger.info("🔒 UnifiedDatabaseService shutdown complete")
        except Exception as e:
            logger.exception(f"❌ Error during shutdown: {e}")


# ==========================================
# SPECIALIZED SERVICE MODULES (Interfaces)
# ==========================================

# services/db/base_db_service.py
"""
Base Database Service - Shared Infrastructure
Handles MongoDB, Redis connections, KeyBuilder interface, and common utilities
"""

import asyncio
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone
import redis.asyncio as aioredis
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from loguru import logger
from config import settings


class BaseDBService:
    """Base database service providing shared infrastructure"""
    
    def __init__(self):
        self.mongo_client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.redis: Optional[aioredis.Redis] = None
        self.key_builder: Optional['KeyBuilderInterface'] = None

    async def initialize(self):
        """Initialize MongoDB and Redis connections"""
        try:
            # MongoDB connection
            self.mongo_client = AsyncIOMotorClient(
                settings.mongodb_url,
                maxPoolSize=50,
                minPoolSize=5,
                serverSelectionTimeoutMS=5000
            )
            self.db = self.mongo_client.ai
            await self.mongo_client.admin.command("ping")
            
            # Redis connection
            self.redis = await aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
                max_connections=20
            )
            await self.redis.ping()
            
            # KeyBuilder interface
            self.key_builder = KeyBuilderInterface(self.redis, self.db)
            
            logger.info("✅ Base database infrastructure initialized")
            
        except Exception as e:
            logger.exception(f"❌ Base database initialization failed: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check health of base infrastructure"""
        health = {"status": "healthy", "components": {}}
        
        try:
            await self.mongo_client.admin.command("ping")
            health["components"]["mongodb"] = {"status": "healthy"}
        except Exception as e:
            health["components"]["mongodb"] = {"status": "unhealthy", "error": str(e)}
            health["status"] = "degraded"
        
        try:
            await self.redis.ping()
            health["components"]["redis"] = {"status": "healthy"}
        except Exception as e:
            health["components"]["redis"] = {"status": "unhealthy", "error": str(e)}
            health["status"] = "degraded"
        
        return health

    async def check_rate_limit(self, identifier: str, limit: int, window_seconds: int) -> Tuple[bool, int]:
        """Rate limiting implementation"""
        try:
            cache_key = f"rate_limit:{identifier}:{window_seconds}"
            current_data = await self.key_builder.get(cache_key)
            
            if not current_data:
                await self.key_builder.set(cache_key, {"count": 1}, ttl=window_seconds)
                return True, 1
            
            current_count = current_data.get("count", 0)
            if current_count >= limit:
                return False, current_count
            
            new_count = current_count + 1
            await self.key_builder.set(cache_key, {"count": new_count}, ttl=window_seconds)
            return True, new_count
            
        except Exception as e:
            logger.exception(f"❌ Rate limit check failed: {e}")
            return True, 0

    async def reset_rate_limit(self, identifier: str, window_seconds: int) -> bool:
        """Reset rate limit"""
        try:
            cache_key = f"rate_limit:{identifier}:{window_seconds}"
            return await self.key_builder.delete(cache_key)
        except Exception as e:
            logger.exception(f"❌ Rate limit reset failed: {e}")
            return False

    async def close(self):
        """Close all connections"""
        try:
            if self.mongo_client:
                self.mongo_client.close()
            if self.redis:
                await self.redis.close()
            logger.info("✅ Base database connections closed")
        except Exception as e:
            logger.exception(f"❌ Error closing connections: {e}")


class KeyBuilderInterface:
    """KeyBuilder compatibility interface"""
    
    def __init__(self, redis_client, mongo_db):
        self.redis = redis_client
        self.db = mongo_db
    
    async def get(self, key: str) -> Optional[Dict]:
        """Get data from Redis with JSON deserialization"""
        try:
            import json
            data = await self.redis.get(key)
            if data:
                if isinstance(data, str):
                    return json.loads(data)
                return data
            return None
        except Exception as e:
            logger.error(f"❌ KeyBuilder get error for {key}: {e}")
            return None
    
    async def set(self, key: str, value: Dict, ttl: int = 86400) -> bool:
        """Set data in Redis with JSON serialization"""
        try:
            import json
            if isinstance(value, dict):
                serialized_value = json.dumps(value, default=str)
            else:
                serialized_value = str(value)
            
            await self.redis.setex(key, ttl, serialized_value)
            return True
        except Exception as e:
            logger.error(f"❌ KeyBuilder set error for {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"❌ KeyBuilder delete error for {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        try:
            result = await self.redis.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"❌ KeyBuilder exists error for {key}: {e}")
            return False
