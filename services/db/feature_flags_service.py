"""
Enhanced Feature Flags Service - Complete Standalone Implementation
Advanced feature flag management, A/B testing, and intelligent rollouts
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import asyncio
import hashlib
import json
import math
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from pydantic import BaseModel, validator, Field
from scipy import stats
import numpy as np
from loguru import logger

from services.db.base_db_service import BaseDBService


class FeatureFlagScope(Enum):
    GLOBAL = "global"
    USER_SPECIFIC = "user_specific"
    PLAN_BASED = "plan_based"
    GEOGRAPHIC = "geographic"
    AB_TEST = "ab_test"


class RolloutStrategy(Enum):
    IMMEDIATE = "immediate"
    PERCENTAGE = "percentage"
    WHITELIST = "whitelist"
    GRADUAL = "gradual"
    AB_TEST = "ab_test"
    SCHEDULED = "scheduled"


class FeatureFlagValidation(BaseModel):
    """Pydantic model for feature flag validation"""
    flag_name: str = Field(..., min_length=1, max_length=100, regex=r'^[a-zA-Z][a-zA-Z0-9_]*
    flag_name: str
    enabled: bool
    rollout_strategy: RolloutStrategy
    rollout_percentage: float = 0.0
    target_users: List[str] = None
    target_plans: List[str] = None
    target_regions: List[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    ab_test_groups: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.target_users is None:
            self.target_users = []
        if self.target_plans is None:
            self.target_plans = []
        if self.target_regions is None:
            self.target_regions = []
        if self.ab_test_groups is None:
            self.ab_test_groups = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ABTestResult:
    flag_name: str
    user_id: str
    group: str
    event_type: str
    value: float = 0.0
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class IntelligentFeatureFlagsService:
    """Advanced feature flags with A/B testing and intelligent rollouts"""
    
    def __init__(self, base_service: BaseDBService):
        self.base = base_service
        self.flag_cache = {}
        self.ab_test_assignments = {}
        
        # Default feature configuration with realistic SMS trading bot flags
        self.default_flags = {
            "personality_engine_v5": FeatureFlag(
                flag_name="personality_engine_v5",
                enabled=True,
                rollout_strategy=RolloutStrategy.IMMEDIATE,
                metadata={"description": "Enhanced personality learning engine"}
            ),
            "enhanced_memory": FeatureFlag(
                flag_name="enhanced_memory",
                enabled=True,
                rollout_strategy=RolloutStrategy.PERCENTAGE,
                rollout_percentage=85.0,
                metadata={"description": "3-layer conversation memory system"}
            ),
            "goal_tracking": FeatureFlag(
                flag_name="goal_tracking",
                enabled=True,
                rollout_strategy=RolloutStrategy.WHITELIST,
                target_plans=["premium", "enterprise"],
                metadata={"description": "Financial goal tracking and management"}
            ),
            "trade_tracking": FeatureFlag(
                flag_name="trade_tracking",
                enabled=False,
                rollout_strategy=RolloutStrategy.AB_TEST,
                ab_test_groups={
                    "control": {"enabled": False, "weight": 50},
                    "test": {"enabled": True, "weight": 50}
                },
                metadata={"description": "Automatic trade detection and performance tracking"}
            ),
            "real_time_alerts": FeatureFlag(
                flag_name="real_time_alerts",
                enabled=True,
                rollout_strategy=RolloutStrategy.GRADUAL,
                rollout_percentage=20.0,
                metadata={
                    "description": "Real-time market alerts",
                    "gradual_rollout_days": 14
                }
            ),
            "portfolio_integration": FeatureFlag(
                flag_name="portfolio_integration",
                enabled=False,
                rollout_strategy=RolloutStrategy.WHITELIST,
                target_plans=["enterprise"],
                metadata={"description": "Plaid portfolio connectivity"}
            ),
            "advanced_research": FeatureFlag(
                flag_name="advanced_research",
                enabled=False,
                rollout_strategy=RolloutStrategy.SCHEDULED,
                start_date=datetime.now(timezone.utc) + timedelta(days=7),
                rollout_percentage=25.0,
                metadata={"description": "Multi-engine research coordinator"}
            ),
            "social_features": FeatureFlag(
                flag_name="social_features",
                enabled=False,
                rollout_strategy=RolloutStrategy.PERCENTAGE,
                rollout_percentage=0.0,
                metadata={"description": "Community features and social trading"}
            ),
            "ai_strategy_generation": FeatureFlag(
                flag_name="ai_strategy_generation",
                enabled=False,
                rollout_strategy=RolloutStrategy.WHITELIST,
                target_users=[],  # Manually managed whitelist
                metadata={"description": "AI-generated trading strategies"}
            )
        }

    async def initialize(self):
        """Initialize enhanced feature flags system"""
        try:
            # Feature flags collection
            await self.base.db.feature_flags.create_index("flag_name", unique=True)
            await self.base.db.feature_flags.create_index("enabled")
            await self.base.db.feature_flags.create_index("rollout_strategy")
            await self.base.db.feature_flags.create_index([("start_date", 1), ("end_date", 1)])
            await self.base.db.feature_flags.create_index("rollout_percentage")
            
            # User feature assignments
            await self.base.db.user_feature_assignments.create_index([("user_id", 1), ("flag_name", 1)], unique=True)
            await self.base.db.user_feature_assignments.create_index("assigned_at")
            await self.base.db.user_feature_assignments.create_index("ab_test_group")
            
            # A/B test results
            await self.base.db.ab_test_results.create_index([("flag_name", 1), ("user_id", 1), ("event_type", 1)])
            await self.base.db.ab_test_results.create_index("timestamp")
            await self.base.db.ab_test_results.create_index([("flag_name", 1), ("group", 1)])
            
            # User feature overrides (for admin overrides)
            await self.base.db.user_feature_overrides.create_index([("user_id", 1), ("flag_name", 1)], unique=True)
            
            # Initialize default flags
            await self._initialize_default_flags()
            
            logger.info("✅ Enhanced Feature Flags service initialized")
        except Exception as e:
            logger.exception(f"❌ Enhanced Feature Flags service initialization failed: {e}")

    async def _initialize_default_flags(self):
        """Initialize default feature flags in database"""
        for flag_name, flag_config in self.default_flags.items():
            try:
                flag_dict = asdict(flag_config)
                flag_dict["created_at"] = datetime.now(timezone.utc)
                flag_dict["updated_at"] = datetime.now(timezone.utc)
                
                await self.base.db.feature_flags.update_one(
                    {"flag_name": flag_name},
                    {"$setOnInsert": flag_dict},
                    upsert=True
                )
            except Exception as e:
                logger.warning(f"Failed to initialize flag {flag_name}: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Health check for feature flags service"""
        try:
            total_flags = await self.base.db.feature_flags.count_documents({})
            enabled_flags = await self.base.db.feature_flags.count_documents({"enabled": True})
            ab_tests = await self.base.db.feature_flags.count_documents({"rollout_strategy": "ab_test"})
            
            # Cache statistics
            cache_size = len(self.flag_cache)
            
            return {
                "status": "healthy",
                "total_flags": total_flags,
                "enabled_flags": enabled_flags,
                "ab_tests_running": ab_tests,
                "cache_size": cache_size
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def get_user_flags(self, user_id: str, user_context: Dict[str, Any] = None) -> Dict[str, bool]:
        """Get feature flags for user with intelligent rollout logic"""
        try:
            cache_key = f"feature_flags:user:{user_id}"
            
            # Check cache first (with shorter TTL for dynamic flags)
            if cache_key in self.flag_cache:
                cache_entry = self.flag_cache[cache_key]
                cache_age = datetime.now(timezone.utc) - cache_entry["timestamp"]
                
                # Use cached result if less than 30 minutes old
                if cache_age < timedelta(minutes=30):
                    return cache_entry["flags"]
            
            # Get user context
            if not user_context:
                user_context = await self._get_user_context(user_id)
            
            # Get all active flags
            flags_cursor = self.base.db.feature_flags.find({})
            flags = {}
            
            async for flag_doc in flags_cursor:
                try:
                    # Convert to FeatureFlag object
                    flag = FeatureFlag(**{k: v for k, v in flag_doc.items() if k != '_id'})
                    enabled = await self._evaluate_flag_for_user(flag, user_id, user_context)
                    flags[flag.flag_name] = enabled
                except Exception as e:
                    logger.warning(f"Error evaluating flag {flag_doc.get('flag_name')}: {e}")
                    flags[flag_doc.get('flag_name', 'unknown')] = False
            
            # Check for user-specific overrides
            overrides = await self.base.db.user_feature_overrides.find({"user_id": user_id}).to_list(length=None)
            for override in overrides:
                flags[override["flag_name"]] = override.get("enabled", False)
            
            # Cache result
            self.flag_cache[cache_key] = {
                "flags": flags,
                "timestamp": datetime.now(timezone.utc)
            }
            
            return flags
            
        except Exception as e:
            logger.exception(f"❌ Error getting user flags: {e}")
            # Return safe defaults
            return {flag_name: False for flag_name in self.default_flags.keys()}

    async def get_global_flags(self) -> Dict[str, bool]:
        """Get global feature flags (not user-specific)"""
        try:
            cache_key = "feature_flags:global"
            
            # Check cache
            cached_flags = await self.base.key_builder.get(cache_key)
            if cached_flags:
                return cached_flags
            
            # Get flags that are immediately enabled for everyone
            flags_cursor = self.base.db.feature_flags.find({
                "enabled": True,
                "rollout_strategy": "immediate"
            })
            
            flags = {}
            async for flag_doc in flags_cursor:
                flags[flag_doc["flag_name"]] = True
            
            # Add default disabled flags
            for flag_name in self.default_flags.keys():
                if flag_name not in flags:
                    flags[flag_name] = False
            
            # Cache for 1 hour
            await self.base.key_builder.set(cache_key, flags, ttl=3600)
            
            return flags
            
        except Exception as e:
            logger.exception(f"❌ Error getting global flags: {e}")
            return {}

    async def _evaluate_flag_for_user(self, flag: FeatureFlag, user_id: str, user_context: Dict[str, Any]) -> bool:
        """Evaluate whether flag should be enabled for specific user"""
        
        # Check if flag is globally disabled
        if not flag.enabled:
            return False
        
        # Check date constraints
        now = datetime.now(timezone.utc)
        if flag.start_date and now < flag.start_date:
            return False
        if flag.end_date and now > flag.end_date:
            return False
        
        # Strategy-based evaluation
        if flag.rollout_strategy == RolloutStrategy.IMMEDIATE:
            return True
        
        elif flag.rollout_strategy == RolloutStrategy.PERCENTAGE:
            return self._evaluate_percentage_rollout(user_id, flag.rollout_percentage)
        
        elif flag.rollout_strategy == RolloutStrategy.WHITELIST:
            return await self._evaluate_whitelist(user_id, flag, user_context)
        
        elif flag.rollout_strategy == RolloutStrategy.AB_TEST:
            return await self._evaluate_ab_test(user_id, flag)
        
        elif flag.rollout_strategy == RolloutStrategy.GRADUAL:
            return await self._evaluate_gradual_rollout(user_id, flag)
        
        elif flag.rollout_strategy == RolloutStrategy.SCHEDULED:
            return await self._evaluate_scheduled_rollout(user_id, flag)
        
        return False

    def _evaluate_percentage_rollout(self, user_id: str, percentage: float) -> bool:
        """Evaluate percentage-based rollout using consistent hashing"""
        if percentage <= 0:
            return False
        if percentage >= 100:
            return True
        
        # Use hash of user_id for consistent assignment
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)
        user_percentage = (hash_value % 10000) / 100.0  # 0-99.99
        
        return user_percentage < percentage

    async def _evaluate_whitelist(self, user_id: str, flag: FeatureFlag, user_context: Dict[str, Any]) -> bool:
        """Evaluate whitelist-based rollout"""
        
        # Check user whitelist
        if user_id in flag.target_users:
            return True
        
        # Check plan-based access
        user_plan = user_context.get("plan_type", "free")
        if user_plan in flag.target_plans:
            return True
        
        # Check geographic restrictions
        user_region = user_context.get("region")
        if flag.target_regions and user_region in flag.target_regions:
            return True
        
        return False

    async def _evaluate_ab_test(self, user_id: str, flag: FeatureFlag) -> bool:
        """Evaluate A/B test assignment"""
        
        # Check existing assignment
        existing_assignment = await self.base.db.user_feature_assignments.find_one({
            "user_id": user_id,
            "flag_name": flag.flag_name
        })
        
        if existing_assignment:
            return existing_assignment.get("enabled", False)
        
        # Assign to test group
        total_weight = sum(group["weight"] for group in flag.ab_test_groups.values())
        if total_weight == 0:
            return False
        
        hash_value = int(hashlib.md5(f"{user_id}:{flag.flag_name}".encode()).hexdigest()[:8], 16)
        assignment_value = hash_value % total_weight
        
        current_weight = 0
        assigned_group = None
        
        for group_name, group_config in flag.ab_test_groups.items():
            current_weight += group_config["weight"]
            if assignment_value < current_weight:
                assigned_group = group_name
                break
        
        if assigned_group:
            group_config = flag.ab_test_groups[assigned_group]
            enabled = group_config.get("enabled", False)
            
            # Save assignment
            await self.base.db.user_feature_assignments.update_one(
                {"user_id": user_id, "flag_name": flag.flag_name},
                {
                    "$set": {
                        "enabled": enabled,
                        "ab_test_group": assigned_group,
                        "assigned_at": datetime.now(timezone.utc)
                    }
                },
                upsert=True
            )
            
            return enabled
        
        return False

    async def _evaluate_gradual_rollout(self, user_id: str, flag: FeatureFlag) -> bool:
        """Evaluate gradual rollout based on time progression"""
        
        if not flag.created_at:
            return False
        
        # Calculate rollout percentage based on time since flag creation
        flag_age = datetime.now(timezone.utc) - flag.created_at
        target_days = flag.metadata.get("gradual_rollout_days", 30)
        
        progress = min(1.0, flag_age.days / target_days)
        current_percentage = progress * flag.rollout_percentage
        
        return self._evaluate_percentage_rollout(user_id, current_percentage)

    async def _evaluate_scheduled_rollout(self, user_id: str, flag: FeatureFlag) -> bool:
        """Evaluate scheduled rollout"""
        
        now = datetime.now(timezone.utc)
        
        # Check if we're in the rollout window
        if flag.start_date and now < flag.start_date:
            return False
        
        # If we're past start date, use percentage rollout
        if flag.start_date and now >= flag.start_date:
            return self._evaluate_percentage_rollout(user_id, flag.rollout_percentage)
        
        return False

    async def _get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get user context for flag evaluation"""
        try:
            user_doc = await self.base.db.users.find_one({"user_id": user_id})
            if user_doc:
                return {
                    "plan_type": user_doc.get("plan_type", "free"),
                    "created_at": user_doc.get("created_at"),
                    "region": user_doc.get("region"),
                    "beta_user": user_doc.get("beta_user", False),
                    "trading_experience": user_doc.get("trading_experience", "beginner")
                }
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
        
        return {}

    async def create_or_update_flag(self, flag: FeatureFlag) -> Dict[str, Any]:
        """Create or update a feature flag with validation"""
        try:
            # Validate flag configuration
            validation_errors = flag.validate()
            if validation_errors:
                return {
                    "success": False,
                    "errors": validation_errors
                }
            
            flag.updated_at = datetime.now(timezone.utc)
            if not flag.created_at:
                flag.created_at = datetime.now(timezone.utc)
            
            flag_dict = asdict(flag)
            
            result = await self.base.db.feature_flags.update_one(
                {"flag_name": flag.flag_name},
                {"$set": flag_dict},
                upsert=True
            )
            
            # Invalidate cache
            await self._invalidate_flag_cache(flag.flag_name)
            
            logger.info(f"✅ Updated feature flag: {flag.flag_name}")
            
            return {
                "success": True,
                "flag_name": flag.flag_name,
                "upserted": result.upserted_id is not None,
                "modified": result.modified_count > 0
            }
            
        except Exception as e:
            logger.exception(f"❌ Error updating feature flag: {e}")
            return {
                "success": False,
                "errors": [str(e)]
            }

    async def set_flag(self, flag_name: str, enabled: bool, user_id: str = None, 
                      rollout_percentage: float = None) -> bool:
        """Set feature flag for user or globally"""
        try:
            if user_id:
                # User-specific override
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
                cache_key = f"feature_flags:user:{user_id}"
                self.flag_cache.pop(cache_key, None)
                
            else:
                # Global flag update
                update_fields = {
                    "enabled": enabled,
                    "updated_at": datetime.now(timezone.utc)
                }
                
                if rollout_percentage is not None:
                    update_fields["rollout_percentage"] = rollout_percentage
                
                await self.base.db.feature_flags.update_one(
                    {"flag_name": flag_name},
                    {"$set": update_fields},
                    upsert=True
                )
                
                # Invalidate all caches
                await self._invalidate_flag_cache()
            
            logger.info(f"✅ Set feature flag {flag_name}={enabled} for {user_id or 'global'}")
            return True
            
        except Exception as e:
            logger.exception(f"❌ Error setting feature flag: {e}")
            return False

    async def get_flag(self, flag_name: str, user_id: str = None) -> bool:
        """Get a specific flag for user or globally"""
        try:
            if user_id:
                flags = await self.get_user_flags(user_id)
                return flags.get(flag_name, False)
            else:
                flags = await self.get_global_flags()
                return flags.get(flag_name, False)
        except Exception as e:
            logger.error(f"Error getting flag {flag_name}: {e}")
            return False

    async def record_ab_test_event(self, flag_name: str, user_id: str, 
                                  event_type: str, value: float = 1.0) -> bool:
        """Record an A/B test conversion event"""
        try:
            # Get user's test group
            assignment = await self.base.db.user_feature_assignments.find_one({
                "user_id": user_id,
                "flag_name": flag_name
            })
            
            if not assignment:
                return False
            
            # Record the event
            event = ABTestResult(
                flag_name=flag_name,
                user_id=user_id,
                group=assignment.get("ab_test_group", "unknown"),
                event_type=event_type,
                value=value
            )
            
            await self.base.db.ab_test_results.insert_one(asdict(event))
            
            logger.debug(f"📊 Recorded A/B test event: {flag_name}, {event_type}, {value}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording A/B test event: {e}")
            return False

    async def get_ab_test_results(self, flag_name: str, days: int = 30) -> Dict[str, Any]:
        """Get A/B test results for analysis"""
        try:
            # Get test configuration
            flag_doc = await self.base.db.feature_flags.find_one({"flag_name": flag_name})
            if not flag_doc or flag_doc.get("rollout_strategy") != "ab_test":
                return {"error": "Flag not found or not an A/B test"}
            
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Get assignments by group
            assignments_pipeline = [
                {"$match": {"flag_name": flag_name}},
                {"$group": {
                    "_id": "$ab_test_group",
                    "count": {"$sum": 1}
                }}
            ]
            
            assignments = await self.base.db.user_feature_assignments.aggregate(assignments_pipeline).to_list(length=None)
            
            # Get conversion events by group
            conversions_pipeline = [
                {"$match": {
                    "flag_name": flag_name,
                    "timestamp": {"$gte": start_date}
                }},
                {"$group": {
                    "_id": {
                        "group": "$group",
                        "event_type": "$event_type"
                    },
                    "count": {"$sum": 1},
                    "total_value": {"$sum": "$value"}
                }}
            ]
            
            conversions = await self.base.db.ab_test_results.aggregate(conversions_pipeline).to_list(length=None)
            
            # Process results
            results = {
                "flag_name": flag_name,
                "period_days": days,
                "assignments": {item["_id"]: item["count"] for item in assignments},
                "conversions": {},
                "conversion_rates": {},
                "statistical_analysis": {}
            }
            
            # Organize conversion data
            for conv in conversions:
                group = conv["_id"]["group"]
                event_type = conv["_id"]["event_type"]
                
                if group not in results["conversions"]:
                    results["conversions"][group] = {}
                
                results["conversions"][group][event_type] = {
                    "count": conv["count"],
                    "total_value": conv["total_value"]
                }
            
            # Calculate conversion rates
            for group, assignment_count in results["assignments"].items():
                if group in results["conversions"]:
                    results["conversion_rates"][group] = {}
                    for event_type, conv_data in results["conversions"][group].items():
                        rate = conv_data["count"] / assignment_count if assignment_count > 0 else 0
                        results["conversion_rates"][group][event_type] = round(rate, 4)
            
            # Add statistical significance (simplified)
            results["statistical_analysis"] = await self._calculate_statistical_significance(results)
            
            return results
            
        except Exception as e:
            logger.exception(f"❌ Error getting A/B test results: {e}")
            return {"error": str(e)}

    async def _calculate_statistical_significance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistical significance of A/B test results using proper statistics"""
        
        assignments = results.get("assignments", {})
        conversions = results.get("conversions", {})
        
        groups = list(assignments.keys())
        if len(groups) < 2:
            return {"error": "Need at least 2 groups for comparison"}
        
        # Find control and test groups
        control_group = groups[0]  # Assume first group is control
        test_groups = groups[1:]
        
        control_total = assignments.get(control_group, 0)
        control_conversions = 0
        
        # Sum all conversion events for control group
        if control_group in conversions:
            control_conversions = sum(
                event_data.get("count", 0) 
                for event_data in conversions[control_group].values()
            )
        
        analysis_results = {
            "total_sample_size": sum(assignments.values()),
            "groups_analyzed": len(groups),
            "control_group": control_group,
            "comparisons": {}
        }
        
        # Analyze each test group against control
        for test_group in test_groups:
            test_total = assignments.get(test_group, 0)
            test_conversions = 0
            
            if test_group in conversions:
                test_conversions = sum(
                    event_data.get("count", 0) 
                    for event_data in conversions[test_group].values()
                )
            
            # Calculate statistical significance
            comparison = StatisticalAnalysis.calculate_statistical_significance(
                control_conversions, control_total,
                test_conversions, test_total
            )
            
            # Add sample size recommendations
            if control_total > 0:
                control_rate = control_conversions / control_total
                recommended_sample_size = StatisticalAnalysis.calculate_sample_size_needed(
                    baseline_rate=max(control_rate, 0.01),  # Minimum 1% to avoid division by zero
                    minimum_detectable_effect=0.2  # 20% relative improvement
                )
                comparison["recommended_sample_size_per_group"] = recommended_sample_size
                comparison["current_power"] = min(control_total / recommended_sample_size, 1.0) if recommended_sample_size > 0 else 0
            
            analysis_results["comparisons"][f"{control_group}_vs_{test_group}"] = comparison
        
        # Overall test status
        any_significant = any(
            comp.get("significant", False) 
            for comp in analysis_results["comparisons"].values()
        )
        
        analysis_results["overall_significant"] = any_significant
        analysis_results["min_sample_size_reached"] = analysis_results["total_sample_size"] >= 100
        analysis_results["analysis_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        return analysis_results

    async def get_all_flags(self) -> List[Dict[str, Any]]:
        """Get all feature flags with their configurations"""
        try:
            flags = await self.base.db.feature_flags.find({}).to_list(length=None)
            
            for flag in flags:
                flag["_id"] = str(flag["_id"])
                
                # Add runtime statistics
                if flag.get("rollout_strategy") == "ab_test":
                    assignments = await self.base.db.user_feature_assignments.count_documents({
                        "flag_name": flag["flag_name"]
                    })
                    flag["active_users"] = assignments
            
            return flags
            
        except Exception as e:
            logger.error(f"Error getting all flags: {e}")
            return []

    async def delete_flag(self, flag_name: str) -> bool:
        """Delete a feature flag and all related data"""
        try:
            # Delete flag
            flag_result = await self.base.db.feature_flags.delete_one({"flag_name": flag_name})
            
            # Delete assignments
            await self.base.db.user_feature_assignments.delete_many({"flag_name": flag_name})
            
            # Delete overrides
            await self.base.db.user_feature_overrides.delete_many({"flag_name": flag_name})
            
            # Delete A/B test results
            await self.base.db.ab_test_results.delete_many({"flag_name": flag_name})
            
            # Invalidate cache
            await self._invalidate_flag_cache()
            
            return flag_result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting flag: {e}")
            return False

    async def _invalidate_flag_cache(self, flag_name: str = None):
        """Invalidate feature flag cache with comprehensive cleanup"""
        if flag_name:
            # Invalidate specific flag cache entries
            keys_to_remove = [key for key in self.flag_cache.keys() 
                            if flag_name in key or "global" in key]
        else:
            # Invalidate all cache
            keys_to_remove = list(self.flag_cache.keys())
        
        for key in keys_to_remove:
            self.flag_cache.pop(key, None)
        
        # Also invalidate Redis cache with pattern matching
        try:
            if flag_name:
                # Invalidate specific patterns
                patterns = [
                    f"feature_flags:user:*",  # All user caches might be affected
                    f"feature_flags:global",
                    f"feature_flags:*{flag_name}*"
                ]
                for pattern in patterns:
                    await self.base.key_builder.delete(pattern)
            else:
                await self.base.key_builder.delete("feature_flags:*")
        except Exception as e:
            logger.warning(f"Failed to invalidate Redis cache: {e}")


# Test utilities for feature flags
class MockFeatureFlagsDatabase:
    """Mock database for testing feature flags"""
    
    def __init__(self):
        self.feature_flags = {}
        self.user_feature_assignments = {}
        self.user_feature_overrides = {}
        self.ab_test_results = []
    
    def find_one(self, query: Dict) -> Optional[Dict]:
        if "flag_name" in query:
            return self.feature_flags.get(query["flag_name"])
        return None
    
    def find(self, query: Dict):
        class MockCursor:
            def __init__(self, data):
                self.data = list(data.values()) if isinstance(data, dict) else data
            
            async def to_list(self, length=None):
                return self.data[:length] if length else self.data
            
            def __aiter__(self):
                return iter(self.data)
        
        return MockCursor(self.feature_flags)
    
    async def update_one(self, filter_dict: Dict, update_dict: Dict, upsert: bool = False):
        flag_name = filter_dict.get("flag_name")
        if flag_name and "$set" in update_dict:
            self.feature_flags[flag_name] = update_dict["$set"]
        
        class MockResult:
            modified_count = 1
            upserted_id = None if not upsert else "mock_id"
        
        return MockResult()


class FeatureFlagsTestHooks:
    """Test hooks for feature flags service testing"""
    
    @staticmethod
    def create_test_feature_flags_service(mock_db: bool = True):
        """Create feature flags service with test doubles"""
        from unittest.mock import MagicMock
        
        # Mock base service
        base_service = MagicMock()
        
        if mock_db:
            base_service.db = MagicMock()
            mock_db_instance = MockFeatureFlagsDatabase()
            base_service.db.feature_flags = mock_db_instance
            base_service.db.user_feature_assignments = mock_db_instance
            base_service.db.user_feature_overrides = mock_db_instance
            base_service.db.ab_test_results = mock_db_instance
            base_service.key_builder = MagicMock()
        
        return IntelligentFeatureFlagsService(base_service)
    
    @staticmethod
    def create_test_feature_flag(
        flag_name: str = "test_flag",
        enabled: bool = True,
        rollout_strategy: RolloutStrategy = RolloutStrategy.IMMEDIATE,
        **kwargs
    ) -> FeatureFlag:
        """Create test feature flag"""
        return FeatureFlag(
            flag_name=flag_name,
            enabled=enabled,
            rollout_strategy=rollout_strategy,
            **kwargs
        )
    
    @staticmethod
    def create_ab_test_flag(
        flag_name: str = "ab_test_flag",
        control_weight: int = 50,
        test_weight: int = 50
    ) -> FeatureFlag:
        """Create A/B test feature flag"""
        return FeatureFlag(
            flag_name=flag_name,
            enabled=True,
            rollout_strategy=RolloutStrategy.AB_TEST,
            ab_test_groups={
                "control": {"enabled": False, "weight": control_weight},
                "test": {"enabled": True, "weight": test_weight}
            }
        )
    
    @staticmethod
    def simulate_ab_test_data(
        control_users: int = 1000,
        test_users: int = 1000,
        control_conversion_rate: float = 0.1,
        test_conversion_rate: float = 0.12
    ) -> Dict[str, Any]:
        """Simulate A/B test data for testing"""
        import random
        
        control_conversions = int(control_users * control_conversion_rate)
        test_conversions = int(test_users * test_conversion_rate)
        
        return {
            "assignments": {
                "control": control_users,
                "test": test_users
            },
            "conversions": {
                "control": {
                    "conversion": {"count": control_conversions, "total_value": control_conversions}
                },
                "test": {
                    "conversion": {"count": test_conversions, "total_value": test_conversions}
                }
            }
        }


# Export the enhanced feature flags service
__all__ = [
    'IntelligentFeatureFlagsService', 
    'FeatureFlag', 
    'ABTestResult',
    'FeatureFlagScope', 
    'RolloutStrategy'
])
    enabled: bool
    rollout_strategy: RolloutStrategy
    rollout_percentage: float = Field(0.0, ge=0.0, le=100.0)
    target_users: List[str] = Field(default_factory=list, max_items=10000)
    target_plans: List[str] = Field(default_factory=list, max_items=50)
    target_regions: List[str] = Field(default_factory=list, max_items=200)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    ab_test_groups: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('end_date')
    def end_date_after_start_date(cls, v, values):
        if v and values.get('start_date') and v <= values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v
    
    @validator('ab_test_groups')
    def validate_ab_test_groups(cls, v, values):
        if values.get('rollout_strategy') == RolloutStrategy.AB_TEST:
            if not v:
                raise ValueError('ab_test_groups required for AB_TEST strategy')
            
            total_weight = sum(group.get('weight', 0) for group in v.values())
            if total_weight <= 0:
                raise ValueError('ab_test_groups must have positive total weight')
            
            for group_name, group_config in v.items():
                if not isinstance(group_config, dict):
                    raise ValueError(f'ab_test_group {group_name} must be a dict')
                if 'weight' not in group_config or group_config['weight'] <= 0:
                    raise ValueError(f'ab_test_group {group_name} must have positive weight')
                if 'enabled' not in group_config:
                    raise ValueError(f'ab_test_group {group_name} must have enabled field')
        
        return v
    
    @validator('rollout_percentage')
    def validate_percentage_strategy(cls, v, values):
        strategy = values.get('rollout_strategy')
        if strategy in [RolloutStrategy.PERCENTAGE, RolloutStrategy.GRADUAL, RolloutStrategy.SCHEDULED]:
            if v < 0 or v > 100:
                raise ValueError('rollout_percentage must be between 0 and 100')
        return v


class StatisticalAnalysis:
    """Statistical analysis utilities for A/B testing"""
    
    @staticmethod
    def calculate_conversion_rate(conversions: int, total: int) -> float:
        """Calculate conversion rate with safety check"""
        return conversions / total if total > 0 else 0.0
    
    @staticmethod
    def calculate_confidence_interval(conversions: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for conversion rate"""
        if total == 0:
            return 0.0, 0.0
        
        rate = conversions / total
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin_error = z_score * math.sqrt(rate * (1 - rate) / total)
        
        return max(0, rate - margin_error), min(1, rate + margin_error)
    
    @staticmethod
    def calculate_statistical_significance(
        control_conversions: int, control_total: int,
        test_conversions: int, test_total: int
    ) -> Dict[str, Any]:
        """Calculate statistical significance between two groups"""
        
        if control_total == 0 or test_total == 0:
            return {
                "significant": False,
                "p_value": 1.0,
                "confidence": 0,
                "error": "Insufficient data"
            }
        
        # Calculate conversion rates
        control_rate = control_conversions / control_total
        test_rate = test_conversions / test_total
        
        # Pooled standard error
        pooled_rate = (control_conversions + test_conversions) / (control_total + test_total)
        se = math.sqrt(pooled_rate * (1 - pooled_rate) * (1/control_total + 1/test_total))
        
        if se == 0:
            return {
                "significant": False,
                "p_value": 1.0,
                "confidence": 0,
                "error": "Zero standard error"
            }
        
        # Z-test
        z_score = (test_rate - control_rate) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Effect size (Cohen's h)
        effect_size = 2 * (math.asin(math.sqrt(test_rate)) - math.asin(math.sqrt(control_rate)))
        
        return {
            "significant": p_value < 0.05,
            "p_value": round(p_value, 6),
            "z_score": round(z_score, 4),
            "effect_size": round(effect_size, 4),
            "control_rate": round(control_rate, 4),
            "test_rate": round(test_rate, 4),
            "lift": round((test_rate - control_rate) / control_rate * 100, 2) if control_rate > 0 else 0,
            "confidence_intervals": {
                "control": StatisticalAnalysis.calculate_confidence_interval(control_conversions, control_total),
                "test": StatisticalAnalysis.calculate_confidence_interval(test_conversions, test_total)
            }
        }
    
    @staticmethod
    def calculate_sample_size_needed(
        baseline_rate: float, 
        minimum_detectable_effect: float, 
        power: float = 0.8, 
        alpha: float = 0.05
    ) -> int:
        """Calculate required sample size for A/B test"""
        if baseline_rate <= 0 or baseline_rate >= 1:
            return 0
        
        test_rate = baseline_rate * (1 + minimum_detectable_effect)
        test_rate = min(test_rate, 0.99)  # Cap at 99%
        
        # Z-scores for alpha and power
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        # Sample size calculation
        pooled_rate = (baseline_rate + test_rate) / 2
        numerator = (z_alpha * math.sqrt(2 * pooled_rate * (1 - pooled_rate)) + 
                    z_beta * math.sqrt(baseline_rate * (1 - baseline_rate) + test_rate * (1 - test_rate)))
        denominator = abs(test_rate - baseline_rate)
        
        return math.ceil((numerator / denominator) ** 2)
    flag_name: str
    enabled: bool
    rollout_strategy: RolloutStrategy
    rollout_percentage: float = 0.0
    target_users: List[str] = None
    target_plans: List[str] = None
    target_regions: List[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    ab_test_groups: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.target_users is None:
            self.target_users = []
        if self.target_plans is None:
            self.target_plans = []
        if self.target_regions is None:
            self.target_regions = []
        if self.ab_test_groups is None:
            self.ab_test_groups = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ABTestResult:
    flag_name: str
    user_id: str
    group: str
    event_type: str
    value: float = 0.0
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class IntelligentFeatureFlagsService:
    """Advanced feature flags with A/B testing and intelligent rollouts"""
    
    def __init__(self, base_service: BaseDBService):
        self.base = base_service
        self.flag_cache = {}
        self.ab_test_assignments = {}
        
        # Default feature configuration with realistic SMS trading bot flags
        self.default_flags = {
            "personality_engine_v5": FeatureFlag(
                flag_name="personality_engine_v5",
                enabled=True,
                rollout_strategy=RolloutStrategy.IMMEDIATE,
                metadata={"description": "Enhanced personality learning engine"}
            ),
            "enhanced_memory": FeatureFlag(
                flag_name="enhanced_memory",
                enabled=True,
                rollout_strategy=RolloutStrategy.PERCENTAGE,
                rollout_percentage=85.0,
                metadata={"description": "3-layer conversation memory system"}
            ),
            "goal_tracking": FeatureFlag(
                flag_name="goal_tracking",
                enabled=True,
                rollout_strategy=RolloutStrategy.WHITELIST,
                target_plans=["premium", "enterprise"],
                metadata={"description": "Financial goal tracking and management"}
            ),
            "trade_tracking": FeatureFlag(
                flag_name="trade_tracking",
                enabled=False,
                rollout_strategy=RolloutStrategy.AB_TEST,
                ab_test_groups={
                    "control": {"enabled": False, "weight": 50},
                    "test": {"enabled": True, "weight": 50}
                },
                metadata={"description": "Automatic trade detection and performance tracking"}
            ),
            "real_time_alerts": FeatureFlag(
                flag_name="real_time_alerts",
                enabled=True,
                rollout_strategy=RolloutStrategy.GRADUAL,
                rollout_percentage=20.0,
                metadata={
                    "description": "Real-time market alerts",
                    "gradual_rollout_days": 14
                }
            ),
            "portfolio_integration": FeatureFlag(
                flag_name="portfolio_integration",
                enabled=False,
                rollout_strategy=RolloutStrategy.WHITELIST,
                target_plans=["enterprise"],
                metadata={"description": "Plaid portfolio connectivity"}
            ),
            "advanced_research": FeatureFlag(
                flag_name="advanced_research",
                enabled=False,
                rollout_strategy=RolloutStrategy.SCHEDULED,
                start_date=datetime.now(timezone.utc) + timedelta(days=7),
                rollout_percentage=25.0,
                metadata={"description": "Multi-engine research coordinator"}
            ),
            "social_features": FeatureFlag(
                flag_name="social_features",
                enabled=False,
                rollout_strategy=RolloutStrategy.PERCENTAGE,
                rollout_percentage=0.0,
                metadata={"description": "Community features and social trading"}
            ),
            "ai_strategy_generation": FeatureFlag(
                flag_name="ai_strategy_generation",
                enabled=False,
                rollout_strategy=RolloutStrategy.WHITELIST,
                target_users=[],  # Manually managed whitelist
                metadata={"description": "AI-generated trading strategies"}
            )
        }

    async def initialize(self):
        """Initialize enhanced feature flags system"""
        try:
            # Feature flags collection
            await self.base.db.feature_flags.create_index("flag_name", unique=True)
            await self.base.db.feature_flags.create_index("enabled")
            await self.base.db.feature_flags.create_index("rollout_strategy")
            await self.base.db.feature_flags.create_index([("start_date", 1), ("end_date", 1)])
            await self.base.db.feature_flags.create_index("rollout_percentage")
            
            # User feature assignments
            await self.base.db.user_feature_assignments.create_index([("user_id", 1), ("flag_name", 1)], unique=True)
            await self.base.db.user_feature_assignments.create_index("assigned_at")
            await self.base.db.user_feature_assignments.create_index("ab_test_group")
            
            # A/B test results
            await self.base.db.ab_test_results.create_index([("flag_name", 1), ("user_id", 1), ("event_type", 1)])
            await self.base.db.ab_test_results.create_index("timestamp")
            await self.base.db.ab_test_results.create_index([("flag_name", 1), ("group", 1)])
            
            # User feature overrides (for admin overrides)
            await self.base.db.user_feature_overrides.create_index([("user_id", 1), ("flag_name", 1)], unique=True)
            
            # Initialize default flags
            await self._initialize_default_flags()
            
            logger.info("✅ Enhanced Feature Flags service initialized")
        except Exception as e:
            logger.exception(f"❌ Enhanced Feature Flags service initialization failed: {e}")

    async def _initialize_default_flags(self):
        """Initialize default feature flags in database"""
        for flag_name, flag_config in self.default_flags.items():
            try:
                flag_dict = asdict(flag_config)
                flag_dict["created_at"] = datetime.now(timezone.utc)
                flag_dict["updated_at"] = datetime.now(timezone.utc)
                
                await self.base.db.feature_flags.update_one(
                    {"flag_name": flag_name},
                    {"$setOnInsert": flag_dict},
                    upsert=True
                )
            except Exception as e:
                logger.warning(f"Failed to initialize flag {flag_name}: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Health check for feature flags service"""
        try:
            total_flags = await self.base.db.feature_flags.count_documents({})
            enabled_flags = await self.base.db.feature_flags.count_documents({"enabled": True})
            ab_tests = await self.base.db.feature_flags.count_documents({"rollout_strategy": "ab_test"})
            
            # Cache statistics
            cache_size = len(self.flag_cache)
            
            return {
                "status": "healthy",
                "total_flags": total_flags,
                "enabled_flags": enabled_flags,
                "ab_tests_running": ab_tests,
                "cache_size": cache_size
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def get_user_flags(self, user_id: str, user_context: Dict[str, Any] = None) -> Dict[str, bool]:
        """Get feature flags for user with intelligent rollout logic"""
        try:
            cache_key = f"feature_flags:user:{user_id}"
            
            # Check cache first (with shorter TTL for dynamic flags)
            if cache_key in self.flag_cache:
                cache_entry = self.flag_cache[cache_key]
                cache_age = datetime.now(timezone.utc) - cache_entry["timestamp"]
                
                # Use cached result if less than 30 minutes old
                if cache_age < timedelta(minutes=30):
                    return cache_entry["flags"]
            
            # Get user context
            if not user_context:
                user_context = await self._get_user_context(user_id)
            
            # Get all active flags
            flags_cursor = self.base.db.feature_flags.find({})
            flags = {}
            
            async for flag_doc in flags_cursor:
                try:
                    # Convert to FeatureFlag object
                    flag = FeatureFlag(**{k: v for k, v in flag_doc.items() if k != '_id'})
                    enabled = await self._evaluate_flag_for_user(flag, user_id, user_context)
                    flags[flag.flag_name] = enabled
                except Exception as e:
                    logger.warning(f"Error evaluating flag {flag_doc.get('flag_name')}: {e}")
                    flags[flag_doc.get('flag_name', 'unknown')] = False
            
            # Check for user-specific overrides
            overrides = await self.base.db.user_feature_overrides.find({"user_id": user_id}).to_list(length=None)
            for override in overrides:
                flags[override["flag_name"]] = override.get("enabled", False)
            
            # Cache result
            self.flag_cache[cache_key] = {
                "flags": flags,
                "timestamp": datetime.now(timezone.utc)
            }
            
            return flags
            
        except Exception as e:
            logger.exception(f"❌ Error getting user flags: {e}")
            # Return safe defaults
            return {flag_name: False for flag_name in self.default_flags.keys()}

    async def get_global_flags(self) -> Dict[str, bool]:
        """Get global feature flags (not user-specific)"""
        try:
            cache_key = "feature_flags:global"
            
            # Check cache
            cached_flags = await self.base.key_builder.get(cache_key)
            if cached_flags:
                return cached_flags
            
            # Get flags that are immediately enabled for everyone
            flags_cursor = self.base.db.feature_flags.find({
                "enabled": True,
                "rollout_strategy": "immediate"
            })
            
            flags = {}
            async for flag_doc in flags_cursor:
                flags[flag_doc["flag_name"]] = True
            
            # Add default disabled flags
            for flag_name in self.default_flags.keys():
                if flag_name not in flags:
                    flags[flag_name] = False
            
            # Cache for 1 hour
            await self.base.key_builder.set(cache_key, flags, ttl=3600)
            
            return flags
            
        except Exception as e:
            logger.exception(f"❌ Error getting global flags: {e}")
            return {}

    async def _evaluate_flag_for_user(self, flag: FeatureFlag, user_id: str, user_context: Dict[str, Any]) -> bool:
        """Evaluate whether flag should be enabled for specific user"""
        
        # Check if flag is globally disabled
        if not flag.enabled:
            return False
        
        # Check date constraints
        now = datetime.now(timezone.utc)
        if flag.start_date and now < flag.start_date:
            return False
        if flag.end_date and now > flag.end_date:
            return False
        
        # Strategy-based evaluation
        if flag.rollout_strategy == RolloutStrategy.IMMEDIATE:
            return True
        
        elif flag.rollout_strategy == RolloutStrategy.PERCENTAGE:
            return self._evaluate_percentage_rollout(user_id, flag.rollout_percentage)
        
        elif flag.rollout_strategy == RolloutStrategy.WHITELIST:
            return await self._evaluate_whitelist(user_id, flag, user_context)
        
        elif flag.rollout_strategy == RolloutStrategy.AB_TEST:
            return await self._evaluate_ab_test(user_id, flag)
        
        elif flag.rollout_strategy == RolloutStrategy.GRADUAL:
            return await self._evaluate_gradual_rollout(user_id, flag)
        
        elif flag.rollout_strategy == RolloutStrategy.SCHEDULED:
            return await self._evaluate_scheduled_rollout(user_id, flag)
        
        return False

    def _evaluate_percentage_rollout(self, user_id: str, percentage: float) -> bool:
        """Evaluate percentage-based rollout using consistent hashing"""
        if percentage <= 0:
            return False
        if percentage >= 100:
            return True
        
        # Use hash of user_id for consistent assignment
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)
        user_percentage = (hash_value % 10000) / 100.0  # 0-99.99
        
        return user_percentage < percentage

    async def _evaluate_whitelist(self, user_id: str, flag: FeatureFlag, user_context: Dict[str, Any]) -> bool:
        """Evaluate whitelist-based rollout"""
        
        # Check user whitelist
        if user_id in flag.target_users:
            return True
        
        # Check plan-based access
        user_plan = user_context.get("plan_type", "free")
        if user_plan in flag.target_plans:
            return True
        
        # Check geographic restrictions
        user_region = user_context.get("region")
        if flag.target_regions and user_region in flag.target_regions:
            return True
        
        return False

    async def _evaluate_ab_test(self, user_id: str, flag: FeatureFlag) -> bool:
        """Evaluate A/B test assignment"""
        
        # Check existing assignment
        existing_assignment = await self.base.db.user_feature_assignments.find_one({
            "user_id": user_id,
            "flag_name": flag.flag_name
        })
        
        if existing_assignment:
            return existing_assignment.get("enabled", False)
        
        # Assign to test group
        total_weight = sum(group["weight"] for group in flag.ab_test_groups.values())
        if total_weight == 0:
            return False
        
        hash_value = int(hashlib.md5(f"{user_id}:{flag.flag_name}".encode()).hexdigest()[:8], 16)
        assignment_value = hash_value % total_weight
        
        current_weight = 0
        assigned_group = None
        
        for group_name, group_config in flag.ab_test_groups.items():
            current_weight += group_config["weight"]
            if assignment_value < current_weight:
                assigned_group = group_name
                break
        
        if assigned_group:
            group_config = flag.ab_test_groups[assigned_group]
            enabled = group_config.get("enabled", False)
            
            # Save assignment
            await self.base.db.user_feature_assignments.update_one(
                {"user_id": user_id, "flag_name": flag.flag_name},
                {
                    "$set": {
                        "enabled": enabled,
                        "ab_test_group": assigned_group,
                        "assigned_at": datetime.now(timezone.utc)
                    }
                },
                upsert=True
            )
            
            return enabled
        
        return False

    async def _evaluate_gradual_rollout(self, user_id: str, flag: FeatureFlag) -> bool:
        """Evaluate gradual rollout based on time progression"""
        
        if not flag.created_at:
            return False
        
        # Calculate rollout percentage based on time since flag creation
        flag_age = datetime.now(timezone.utc) - flag.created_at
        target_days = flag.metadata.get("gradual_rollout_days", 30)
        
        progress = min(1.0, flag_age.days / target_days)
        current_percentage = progress * flag.rollout_percentage
        
        return self._evaluate_percentage_rollout(user_id, current_percentage)

    async def _evaluate_scheduled_rollout(self, user_id: str, flag: FeatureFlag) -> bool:
        """Evaluate scheduled rollout"""
        
        now = datetime.now(timezone.utc)
        
        # Check if we're in the rollout window
        if flag.start_date and now < flag.start_date:
            return False
        
        # If we're past start date, use percentage rollout
        if flag.start_date and now >= flag.start_date:
            return self._evaluate_percentage_rollout(user_id, flag.rollout_percentage)
        
        return False

    async def _get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get user context for flag evaluation"""
        try:
            user_doc = await self.base.db.users.find_one({"user_id": user_id})
            if user_doc:
                return {
                    "plan_type": user_doc.get("plan_type", "free"),
                    "created_at": user_doc.get("created_at"),
                    "region": user_doc.get("region"),
                    "beta_user": user_doc.get("beta_user", False),
                    "trading_experience": user_doc.get("trading_experience", "beginner")
                }
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
        
        return {}

    async def create_or_update_flag(self, flag: FeatureFlag) -> bool:
        """Create or update a feature flag"""
        try:
            flag.updated_at = datetime.now(timezone.utc)
            if not flag.created_at:
                flag.created_at = datetime.now(timezone.utc)
            
            flag_dict = asdict(flag)
            
            await self.base.db.feature_flags.update_one(
                {"flag_name": flag.flag_name},
                {"$set": flag_dict},
                upsert=True
            )
            
            # Invalidate cache
            await self._invalidate_flag_cache(flag.flag_name)
            
            logger.info(f"✅ Updated feature flag: {flag.flag_name}")
            return True
            
        except Exception as e:
            logger.exception(f"❌ Error updating feature flag: {e}")
            return False

    async def set_flag(self, flag_name: str, enabled: bool, user_id: str = None, 
                      rollout_percentage: float = None) -> bool:
        """Set feature flag for user or globally"""
        try:
            if user_id:
                # User-specific override
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
                cache_key = f"feature_flags:user:{user_id}"
                self.flag_cache.pop(cache_key, None)
                
            else:
                # Global flag update
                update_fields = {
                    "enabled": enabled,
                    "updated_at": datetime.now(timezone.utc)
                }
                
                if rollout_percentage is not None:
                    update_fields["rollout_percentage"] = rollout_percentage
                
                await self.base.db.feature_flags.update_one(
                    {"flag_name": flag_name},
                    {"$set": update_fields},
                    upsert=True
                )
                
                # Invalidate all caches
                await self._invalidate_flag_cache()
            
            logger.info(f"✅ Set feature flag {flag_name}={enabled} for {user_id or 'global'}")
            return True
            
        except Exception as e:
            logger.exception(f"❌ Error setting feature flag: {e}")
            return False

    async def get_flag(self, flag_name: str, user_id: str = None) -> bool:
        """Get a specific flag for user or globally"""
        try:
            if user_id:
                flags = await self.get_user_flags(user_id)
                return flags.get(flag_name, False)
            else:
                flags = await self.get_global_flags()
                return flags.get(flag_name, False)
        except Exception as e:
            logger.error(f"Error getting flag {flag_name}: {e}")
            return False

    async def record_ab_test_event(self, flag_name: str, user_id: str, 
                                  event_type: str, value: float = 1.0) -> bool:
        """Record an A/B test conversion event"""
        try:
            # Get user's test group
            assignment = await self.base.db.user_feature_assignments.find_one({
                "user_id": user_id,
                "flag_name": flag_name
            })
            
            if not assignment:
                return False
            
            # Record the event
            event = ABTestResult(
                flag_name=flag_name,
                user_id=user_id,
                group=assignment.get("ab_test_group", "unknown"),
                event_type=event_type,
                value=value
            )
            
            await self.base.db.ab_test_results.insert_one(asdict(event))
            
            logger.debug(f"📊 Recorded A/B test event: {flag_name}, {event_type}, {value}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording A/B test event: {e}")
            return False

    async def get_ab_test_results(self, flag_name: str, days: int = 30) -> Dict[str, Any]:
        """Get A/B test results for analysis"""
        try:
            # Get test configuration
            flag_doc = await self.base.db.feature_flags.find_one({"flag_name": flag_name})
            if not flag_doc or flag_doc.get("rollout_strategy") != "ab_test":
                return {"error": "Flag not found or not an A/B test"}
            
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Get assignments by group
            assignments_pipeline = [
                {"$match": {"flag_name": flag_name}},
                {"$group": {
                    "_id": "$ab_test_group",
                    "count": {"$sum": 1}
                }}
            ]
            
            assignments = await self.base.db.user_feature_assignments.aggregate(assignments_pipeline).to_list(length=None)
            
            # Get conversion events by group
            conversions_pipeline = [
                {"$match": {
                    "flag_name": flag_name,
                    "timestamp": {"$gte": start_date}
                }},
                {"$group": {
                    "_id": {
                        "group": "$group",
                        "event_type": "$event_type"
                    },
                    "count": {"$sum": 1},
                    "total_value": {"$sum": "$value"}
                }}
            ]
            
            conversions = await self.base.db.ab_test_results.aggregate(conversions_pipeline).to_list(length=None)
            
            # Process results
            results = {
                "flag_name": flag_name,
                "period_days": days,
                "assignments": {item["_id"]: item["count"] for item in assignments},
                "conversions": {},
                "conversion_rates": {},
                "statistical_analysis": {}
            }
            
            # Organize conversion data
            for conv in conversions:
                group = conv["_id"]["group"]
                event_type = conv["_id"]["event_type"]
                
                if group not in results["conversions"]:
                    results["conversions"][group] = {}
                
                results["conversions"][group][event_type] = {
                    "count": conv["count"],
                    "total_value": conv["total_value"]
                }
            
            # Calculate conversion rates
            for group, assignment_count in results["assignments"].items():
                if group in results["conversions"]:
                    results["conversion_rates"][group] = {}
                    for event_type, conv_data in results["conversions"][group].items():
                        rate = conv_data["count"] / assignment_count if assignment_count > 0 else 0
                        results["conversion_rates"][group][event_type] = round(rate, 4)
            
            # Add statistical significance (simplified)
            results["statistical_analysis"] = await self._calculate_statistical_significance(results)
            
            return results
            
        except Exception as e:
            logger.exception(f"❌ Error getting A/B test results: {e}")
            return {"error": str(e)}

    async def _calculate_statistical_significance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistical significance of A/B test results"""
        # Simplified statistical analysis
        # In production, would implement proper statistical tests (chi-square, t-test, etc.)
        
        groups = list(results["assignments"].keys())
        if len(groups) < 2:
            return {"error": "Need at least 2 groups for comparison"}
        
        total_sample_size = sum(results["assignments"].values())
        
        return {
            "sample_size": total_sample_size,
            "groups_compared": len(groups),
            "min_sample_size_reached": total_sample_size >= 100,  # Arbitrary threshold
            "confidence_level": 95,
            "note": "Statistical significance calculation simplified for demo"
        }

    async def get_all_flags(self) -> List[Dict[str, Any]]:
        """Get all feature flags with their configurations"""
        try:
            flags = await self.base.db.feature_flags.find({}).to_list(length=None)
            
            for flag in flags:
                flag["_id"] = str(flag["_id"])
                
                # Add runtime statistics
                if flag.get("rollout_strategy") == "ab_test":
                    assignments = await self.base.db.user_feature_assignments.count_documents({
                        "flag_name": flag["flag_name"]
                    })
                    flag["active_users"] = assignments
            
            return flags
            
        except Exception as e:
            logger.error(f"Error getting all flags: {e}")
            return []

    async def delete_flag(self, flag_name: str) -> bool:
        """Delete a feature flag and all related data"""
        try:
            # Delete flag
            flag_result = await self.base.db.feature_flags.delete_one({"flag_name": flag_name})
            
            # Delete assignments
            await self.base.db.user_feature_assignments.delete_many({"flag_name": flag_name})
            
            # Delete overrides
            await self.base.db.user_feature_overrides.delete_many({"flag_name": flag_name})
            
            # Delete A/B test results
            await self.base.db.ab_test_results.delete_many({"flag_name": flag_name})
            
            # Invalidate cache
            await self._invalidate_flag_cache()
            
            return flag_result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting flag: {e}")
            return False

    async def _invalidate_flag_cache(self, flag_name: str = None):
        """Invalidate feature flag cache"""
        if flag_name:
            # Invalidate specific flag cache entries
            keys_to_remove = [key for key in self.flag_cache.keys() 
                            if flag_name in key or "global" in key]
        else:
            # Invalidate all cache
            keys_to_remove = list(self.flag_cache.keys())
        
        for key in keys_to_remove:
            self.flag_cache.pop(key, None)
        
        # Also invalidate Redis cache
        try:
            if flag_name:
                await self.base.key_builder.delete(f"feature_flags:*{flag_name}*")
            else:
                await self.base.key_builder.delete("feature_flags:*")
        except Exception as e:
            logger.warning(f"Failed to invalidate Redis cache: {e}")


# Export the enhanced feature flags service
__all__ = [
    'IntelligentFeatureFlagsService', 
    'FeatureFlag', 
    'ABTestResult',
    'FeatureFlagScope', 
    'RolloutStrategy'
]
