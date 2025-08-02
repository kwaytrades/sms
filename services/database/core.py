# services/database/core.py
"""
Core Database Service - Main Entry Point
Focused modular architecture with clean separation of concerns
"""

from typing import Dict, List, Optional, Any, Tuple, Union, Type
from datetime import datetime, timezone
import asyncio
from loguru import logger

from .config import DatabaseServiceConfig
from .exceptions import DatabaseServiceException, ServiceInitializationError
from .monitoring import DatabaseServiceMonitor
from .base import EnhancedBaseDBService
from .service_manager import ServiceManager
from .mixins import (
    ErrorHandlingMixin, 
    CachingMixin, 
    InferenceMixin, 
    HealthCheckMixin
)
from .interfaces import DatabaseServiceInterface


class CoreDatabaseService(
    ErrorHandlingMixin,
    CachingMixin, 
    InferenceMixin,
    HealthCheckMixin,
    DatabaseServiceInterface
):
    """
    Core Database Service - Main Entry Point
    
    Provides unified interface with clean modular architecture:
    - Dependency injection for all specialized services
    - Comprehensive error handling and monitoring
    - Auto-inference and caching capabilities
    - Health monitoring and graceful degradation
    """
    
    def __init__(self, config: Optional[DatabaseServiceConfig] = None):
        self.config = config or DatabaseServiceConfig()
        self.monitor = DatabaseServiceMonitor()
        self.base_service: Optional[EnhancedBaseDBService] = None
        self.service_manager = ServiceManager(self.config, self.monitor)
        
        # Initialize mixins
        super().__init__(self.config, self.monitor)
        
        logger.info(f"🚀 CoreDatabaseService v6.1 initialized")

    @property
    def service_id(self) -> str:
        """Get unique service identifier"""
        return self.service_manager.service_id

    async def initialize(self) -> None:
        """Initialize all database services with enhanced error handling"""
        async with self.monitor.track_operation("service_initialization"):
            try:
                # Initialize base infrastructure
                await self._initialize_base_infrastructure()
                
                # Initialize all services through service manager
                await self.service_manager.initialize_all_services(self.base_service)
                
                # Validate initialization
                await self._validate_initialization()
                
                logger.info("🎉 CoreDatabaseService fully initialized")
                
            except Exception as e:
                logger.exception(f"❌ Database initialization failed: {e}")
                raise ServiceInitializationError(f"Failed to initialize service: {e}")

    async def _initialize_base_infrastructure(self) -> None:
        """Initialize base infrastructure with retry logic"""
        @self.with_retry(max_retries=3, base_delay=2.0)
        async def init_base():
            self.base_service = EnhancedBaseDBService()
            await self.base_service.initialize()
            return self.base_service
        
        await init_base()
        logger.info("✅ Base infrastructure initialized")

    async def _validate_initialization(self) -> None:
        """Validate that required services are available"""
        validation_result = await self.service_manager.validate_services()
        
        if not validation_result["all_required_available"]:
            missing = validation_result["missing_services"]
            raise ServiceInitializationError(f"Required services missing: {missing}")
        
        health_ratio = validation_result["health_ratio"]
        if health_ratio < self.config.DEGRADED_THRESHOLD:
            logger.warning(f"⚠️ Service initialization degraded: {health_ratio:.1%} healthy")

    # ==========================================
    # ENHANCED SERVICE ACCESS
    # ==========================================

    @property
    def users(self):
        """Access users service"""
        return self.service_manager.get_service("users")

    @property
    def trading(self):
        """Access trading service"""
        return self.service_manager.get_service("trading")

    @property
    def migrations(self):
        """Access migrations service"""
        return self.service_manager.get_service("migrations")

    @property
    def conversations(self):
        """Access conversations service"""
        return self.service_manager.get_service("conversations")

    @property
    def analytics(self):
        """Access analytics service"""
        return self.service_manager.get_service("analytics")

    @property
    def feature_flags(self):
        """Access feature flags service"""
        return self.service_manager.get_service("feature_flags")

    @property
    def compliance(self):
        """Access compliance service"""
        return self.service_manager.get_service("compliance")

    # ==========================================
    # DIRECT DATABASE ACCESS
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
    # SERVICE UTILITIES
    # ==========================================

    async def get_service_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics"""
        return await self.service_manager.get_comprehensive_metrics()

    async def validate_service_availability(self, required_services: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate that required services are available"""
        return await self.service_manager.validate_services(required_services)

    async def get_service_uptime(self) -> Dict[str, Any]:
        """Get service uptime and operational statistics"""
        return self.monitor.get_uptime_stats()

    async def reset_service_metrics(self) -> bool:
        """Reset service metrics"""
        try:
            self.monitor.reset_metrics()
            await self.service_manager.reset_all_metrics()
            return True
        except Exception as e:
            logger.exception(f"❌ Error resetting metrics: {e}")
            return False

    # ==========================================
    # TESTING INTERFACE
    # ==========================================

    def inject_test_services(self, service_mocks: Dict[str, Any]) -> None:
        """Inject mock services for testing"""
        self.service_manager.inject_test_services(service_mocks)

    def get_test_hooks(self) -> Dict[str, Any]:
        """Get testing hooks and utilities"""
        return {
            "service_manager": self.service_manager,
            "monitor": self.monitor,
            "config": self.config,
            "injection_method": self.inject_test_services
        }

    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive service test"""
        return await self.service_manager.run_comprehensive_test()

    # ==========================================
    # CLEANUP & SHUTDOWN
    # ==========================================

    async def close(self) -> None:
        """Close all database connections with enhanced cleanup"""
        try:
            async with self.monitor.track_operation("service_shutdown"):
                logger.info(f"🔒 Initiating shutdown for service {self.service_id}")
                
                # Shutdown all services
                await self.service_manager.shutdown_all_services()
                
                # Close base infrastructure
                if self.base_service:
                    await self.base_service.close()
                
                # Log final metrics
                final_metrics = self.monitor.get_final_metrics()
                logger.info(f"📊 Final service metrics: {final_metrics}")
                
                logger.info(f"🔒 CoreDatabaseService {self.service_id} shutdown complete")
                
        except Exception as e:
            logger.exception(f"❌ Error during shutdown: {e}")


# Export main service
__all__ = ['CoreDatabaseService']
