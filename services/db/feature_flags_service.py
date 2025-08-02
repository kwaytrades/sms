services/db/feature_flags_service.py
"""
Feature Flags Service - Feature flag management and A/B testing
Focused on controlled feature rollouts and experimentation
"""

from typing import Dict, Optional, Any
from datetime import datetime, timezone
from loguru import logger

from services.db.base_db_service import BaseDBService


class FeatureFlagsService:
    """Specialized service for feature flags and A/B testing"""
    
    def __init__(self, base_service: BaseDBService):
        self.base = base_service

    async def initialize(self):
        """Initialize feature flags collections"""
        try:
            # Feature flags collection indexes
            await self.base.db.feature_flags.create_index("flag_name", unique=True)
            await self.base.db.feature_flags.create_index("enabled")
            await self.base.db.feature_flags.create_index("rollout_percentage")
            
            # User feature overrides
            await self.base.db.user_feature_overrides.create_index([("user_id", 1), ("flag_name", 1)], unique=True)
            
            logger.info("✅ Feature flags service initialized")
        except Exception as e:
            logger.exception(f"❌ Feature flags service initialization failed: {e}")

    async def get_flags(self, user_id: str = None) -> Dict[str, bool]:
        """Get feature flags for user or global"""
        try:
            cache_key = f"feature_flags:{user_id or 'global'}"
            cached_flags = await self.base.key_builder.get(cache_key)
            if cached_flags:
                return cached_flags
            
            # Default feature flags
            flags = {
                "personality_engine_v5": True,
                "enhanced_memory": True,
                "goal_tracking": True,
                "trade_tracking": True,
                "real_time_alerts": True,
                "portfolio_integration": False,
                "advanced_research": False,
                "social_features": False,
                "ai_strategy_generation": False
            }
            
            # Get global flags from database
            global_flags = await self.base.db.feature_flags.find({}).to_list(length=None)
            for flag in global_flags:
                flags[flag["flag_name"]] = flag.get("enabled", False)
            
            # User-specific overrides
            if user_id:
                user_overrides = await self.base.db.user_feature_overrides.find({"user_id": user_id}).to_list(length=None)
                for override in user_overrides:
                    flags[override["flag_name"]] = override.get("enabled", False)
            
            # Cache for 1 hour
            await self.base.key_builder.set(cache_key, flags, ttl=3600)
            
            return flags
            
        except Exception as e:
            logger.exception(f"❌ Error getting feature flags: {e}")
            return {}

    async def set_flag(self, flag_name: str, enabled: bool, user_id: str = None) -> bool:
        """Set feature flag for user or globally"""
        try:
            if user_id:
                # User-specific flag
                await self.base.db.user_feature_overrides.update_one(
                    {"user_id": user_id, "flag_name": flag_name},
                    {
                        "$set": {
                            "enabled": enabled,
                            "updated_at": datetime.now(timezone.utc)
                        }
                    },
                    upsert=True
                )
                
                # Invalidate user cache
                cache_key = f"feature_flags:{user_id}"
                await self.base.key_builder.delete(cache_key)
            else:
                # Global flag
                await self.base.db.feature_flags.update_one(
                    {"flag_name": flag_name},
                    {
                        "$set": {
                            "enabled": enabled,
                            "updated_at": datetime.now(timezone.utc)
                        }
                    },
                    upsert=True
                )
                
                # Invalidate global cache
                cache_key = "feature_flags:global"
                await self.base.key_builder.delete(cache_key)
            
            logger.info(f"✅ Set feature flag {flag_name}={enabled} for {user_id or 'global'}")
            return True
            
        except Exception as e:
            logger.exception(f"❌ Error setting feature flag: {e}")
            return False

