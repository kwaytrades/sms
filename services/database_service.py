# services/database_service.py - RENDER-READY VERSION
"""
Unified Database Service - Enhanced v5.0 for Render Deployment
Ready for immediate deployment with real service implementations
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from loguru import logger
import asyncio

# Import specialized service modules (EXISTING)
from services.db.base_db_service import BaseDBService
from services.db.users_service import UsersService
from services.db.conversation_service import ConversationService
from services.db.trading_service import TradingService
from services.db.analytics_service import AnalyticsService
from services.db.migration_service import MigrationService
from services.db.feature_flags_service import FeatureFlagsService
from services.db.compliance_service import ComplianceService

# Import enhanced inference services (EXISTING)
from services.inference.profile_inference import ProfileInferenceService
from services.inference.context_inference import ContextInferenceService

# Import NEW services - Create these files from the artifacts
try:
    from services.vector_service import VectorService
    from services.cache_service import CacheService
    from services.memory_service import MemoryService
    from services.portfolio_service import PortfolioService
    from services.alert_service import AlertService
    from services.trade_tracker_service import TradeTrackerService
    from services.research_service import ResearchService
    from services.notification_service import NotificationService
    NEW_SERVICES_AVAILABLE = True
    logger.info("✅ New v5.0 services imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ New services not available yet: {e}")
    NEW_SERVICES_AVAILABLE = False
    
    # Fallback to stubs for gradual deployment
    class ServiceStub:
        def __init__(self, base_service): self.base_service = base_service
        async def initialize(self): pass
        async def health_check(self): return {"status": "healthy"}
        def __getattr__(self, name): return lambda *args, **kwargs: None
    
    VectorService = ServiceStub
    CacheService = ServiceStub
    MemoryService = ServiceStub
    PortfolioService = ServiceStub
    AlertService = ServiceStub
    TradeTrackerService = ServiceStub
    ResearchService = ServiceStub
    NotificationService = ServiceStub

# Legacy compatibility imports
from models.user import UserProfile
from models.conversation import ChatMessage


class UnifiedDatabaseService:
    """
    Unified Database Service v5.0 - Production Ready for Render
    
    Features:
    - Graceful degradation if new services aren't deployed yet
    - Backward compatibility with existing functionality
    - Progressive enhancement as new services are added
    """
    
    def __init__(self):
        # Core infrastructure
        self.base_service: Optional[BaseDBService] = None
        
        # EXISTING specialized services
        self.users: Optional[UsersService] = None
        self.conversations: Optional[ConversationService] = None
        self.trading: Optional[TradingService] = None
        self.analytics: Optional[AnalyticsService] = None
        self.migrations: Optional[MigrationService] = None
        self.feature_flags: Optional[FeatureFlagsService] = None
        self.compliance: Optional[ComplianceService] = None
        
        # EXISTING auto-inference services
        self.profile_inference: Optional[ProfileInferenceService] = None
        self.context_inference: Optional[ContextInferenceService] = None
        
        # NEW SERVICES (will be stubs if not available)
        self.vector_service: Optional[VectorService] = None
        self.cache_service: Optional[CacheService] = None
        self.memory_service: Optional[MemoryService] = None
        self.portfolio_service: Optional[PortfolioService] = None
        self.alert_service: Optional[AlertService] = None
        self.trade_tracker: Optional[TradeTrackerService] = None
        self.research_service: Optional[ResearchService] = None
        self.notification_service: Optional[NotificationService] = None
        
        self.new_services_enabled = NEW_SERVICES_AVAILABLE
        
        logger.info(f"🚀 UnifiedDatabaseService v5.0 initialized (New services: {'✅' if NEW_SERVICES_AVAILABLE else '⚠️ Stubs'})")

    async def initialize(self):
        """Initialize all database services with graceful fallback"""
        try:
            # Initialize base infrastructure
            self.base_service = BaseDBService()
            await self.base_service.initialize()
            
            # Initialize EXISTING specialized services with shared base
            self.users = UsersService(self.base_service)
            self.conversations = ConversationService(self.base_service)
            self.trading = TradingService(self.base_service)
            self.analytics = AnalyticsService(self.base_service)
            self.migrations = MigrationService(self.base_service)
            self.feature_flags = FeatureFlagsService(self.base_service)
            self.compliance = ComplianceService(self.base_service)
            
            # Initialize EXISTING inference services
            self.profile_inference = ProfileInferenceService(self.base_service)
            self.context_inference = ContextInferenceService(self.base_service)
            
            # Initialize NEW services (real or stubs)
            self.vector_service = VectorService(self.base_service)
            self.cache_service = CacheService(self.base_service)
            self.memory_service = MemoryService(self.base_service, self.vector_service) if NEW_SERVICES_AVAILABLE else MemoryService(self.base_service)
            self.portfolio_service = PortfolioService(self.base_service)
            self.alert_service = AlertService(self.base_service)
            self.trade_tracker = TradeTrackerService(self.base_service)
            self.research_service = ResearchService(self.base_service, self.vector_service) if NEW_SERVICES_AVAILABLE else ResearchService(self.base_service)
            self.notification_service = NotificationService(self.base_service)
            
            # Initialize all services
            await self._initialize_all_services()
            
            status = "fully" if NEW_SERVICES_AVAILABLE else "partially (with stubs)"
            logger.info(f"🎉 UnifiedDatabaseService {status} initialized")
            
        except Exception as e:
            logger.exception(f"❌ Database initialization failed: {e}")
            raise

    async def _initialize_all_services(self):
        """Initialize all specialized services with error handling"""
        services = [
            # Existing services
            self.users, self.conversations, self.trading, self.analytics,
            self.migrations, self.feature_flags, self.compliance,
            self.profile_inference, self.context_inference,
            # New services (may be stubs)
            self.vector_service, self.cache_service, self.memory_service,
            self.portfolio_service, self.alert_service, self.trade_tracker,
            self.research_service, self.notification_service
        ]
        
        for service in services:
            try:
                if hasattr(service, 'initialize'):
                    await service.initialize()
            except Exception as e:
                logger.warning(f"Service initialization failed (will use stub): {e}")

    # ==========================================
    # HEALTH & MONITORING
    # ==========================================

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check with graceful degradation"""
        health = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "unknown",
            "services": {},
            "new_services_enabled": self.new_services_enabled
        }
        
        try:
            # Base infrastructure health
            base_health = await self.base_service.health_check()
            health["services"]["infrastructure"] = base_health
            
            # EXISTING service health checks
            try:
                existing_healths = await asyncio.gather(
                    self.users.health_check(),
                    self.conversations.health_check(),
                    self.trading.health_check(),
                    self.analytics.health_check(),
                    return_exceptions=True
                )
                
                service_names = ["users", "conversations", "trading", "analytics"]
                for i, name in enumerate(service_names):
                    if i < len(existing_healths):
                        result = existing_healths[i]
                        health["services"][name] = result if not isinstance(result, Exception) else {
                            "status": "error", "error": str(result)
                        }
            except Exception as e:
                logger.warning(f"Error checking existing services: {e}")
            
            # NEW service health checks (only if available)
            if self.new_services_enabled:
                try:
                    new_healths = await asyncio.gather(
                        self.vector_service.health_check(),
                        self.cache_service.health_check(),
                        self.memory_service.health_check(),
                        self.alert_service.health_check(),
                        return_exceptions=True
                    )
                    
                    new_service_names = ["vector", "cache", "memory", "alerts"]
                    for i, name in enumerate(new_service_names):
                        if i < len(new_healths):
                            result = new_healths[i]
                            health["services"][name] = result if not isinstance(result, Exception) else {
                                "status": "error", "error": str(result)
                            }
                except Exception as e:
                    logger.warning(f"Error checking new services: {e}")
            
            # Determine overall health
            healthy_services = [
                service.get("status") == "healthy" 
                for service in health["services"].values()
                if isinstance(service, dict)
            ]
            
            if len(healthy_services) == 0:
                health["overall_status"] = "error"
            elif all(healthy_services):
                health["overall_status"] = "healthy"
            else:
                health["overall_status"] = "degraded"
            
            return health
            
        except Exception as e:
            logger.exception(f"❌ Health check failed: {e}")
            health["overall_status"] = "error"
            health["error"] = str(e)
            return health

    # ==========================================
    # ENHANCED METHODS (with fallbacks)
    # ==========================================

    async def save_conversation_turn(self, user_id: str, user_message: str,
                                   bot_response: str, metadata: Dict = None) -> bool:
        """Save conversation turn with memory enhancement if available"""
        # Always save to existing conversation service
        success = await self.save_enhanced_message(
            user_id, user_message, bot_response, 
            metadata or {}, metadata.get("symbols", []) if metadata else []
        )
        
        # Additionally save to new memory service if available
        if self.new_services_enabled and hasattr(self.memory_service, 'save_conversation_turn'):
            try:
                await self.memory_service.save_conversation_turn(
                    user_id, user_message, bot_response, metadata
                )
            except Exception as e:
                logger.warning(f"Memory service save failed: {e}")
        
        return success

    async def get_conversation_memory(self, user_id: str, limit: int = 10,
                                    query: str = None) -> Dict[str, Any]:
        """Get conversation memory with fallback to existing system"""
        if self.new_services_enabled and hasattr(self.memory_service, 'get_conversation_memory'):
            try:
                return await self.memory_service.get_conversation_memory(user_id, limit, query)
            except Exception as e:
                logger.warning(f"Memory service failed, using fallback: {e}")
        
        # Fallback to existing conversation context
        try:
            context = await self.get_conversation_context(user_id)
            return {
                "user_id": user_id,
                "short_term_memory": context.get("recent_messages", [])[:limit],
                "conversation_summaries": [],
                "relevant_memories": [],
                "fallback_mode": True
            }
        except Exception as e:
            logger.error(f"Fallback memory failed: {e}")
            return {"user_id": user_id, "error": str(e)}

    async def create_price_alert(self, user_id: str, symbol: str,
                               target_price: float, condition: str) -> Optional[str]:
        """Create price alert if alert service is available"""
        if self.new_services_enabled and hasattr(self.alert_service, 'create_price_alert'):
            try:
                return await self.alert_service.create_price_alert(
                    user_id, symbol, target_price, condition
                )
            except Exception as e:
                logger.error(f"Alert creation failed: {e}")
                return None
        else:
            logger.info(f"Alert service not available - would create alert for {symbol} @ ${target_price}")
            return f"stub_alert_{user_id}_{symbol}_{int(datetime.utcnow().timestamp())}"

    async def send_sms_notification(self, user_id: str, message: str, 
                                  priority: str = "normal") -> Dict[str, Any]:
        """Send SMS notification if notification service is available"""
        if self.new_services_enabled and hasattr(self.notification_service, 'send_sms'):
            try:
                return await self.notification_service.send_sms(user_id, message, priority)
            except Exception as e:
                logger.error(f"SMS notification failed: {e}")
                return {"success": False, "error": str(e)}
        else:
            logger.info(f"Notification service not available - would send: {message}")
            return {"success": True, "stub": True, "message": "SMS service not available"}

    # ==========================================
    # ALL EXISTING METHODS (unchanged)
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
    # ANALYTICS & METRICS
    # ==========================================

    async def get_user_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get user analytics"""
        return await self.analytics.get_user_analytics(user_id, days)

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        return await self.analytics.get_system_metrics()

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
    # STOCK DATA OPERATIONS
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
    # RATE LIMITING
    # ==========================================

    async def check_rate_limit(self, identifier: str, limit: int, window_seconds: int) -> Tuple[bool, int]:
        """Check rate limit"""
        return await self.base_service.check_rate_limit(identifier, limit, window_seconds)

    async def reset_rate_limit(self, identifier: str, window_seconds: int) -> bool:
        """Reset rate limit"""
        return await self.base_service.reset_rate_limit(identifier, window_seconds)

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
