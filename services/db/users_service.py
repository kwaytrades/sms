# services/db/users_service.py
"""
Users Service - User management, onboarding, personality profiles
Focused on user lifecycle, authentication, profile management
"""

from typing import Optional, Dict, List, Any
from datetime import datetime, timezone, timedelta
from bson import ObjectId
from loguru import logger

from models.user import UserProfile
from services.db.base_db_service import BaseDBService


class UsersService:
    """Specialized service for user management and personality profiles"""
    
    def __init__(self, base_service: BaseDBService):
        self.base = base_service
        self._cache_stats = {"hits": 0, "misses": 0}

    async def initialize(self):
        """Initialize user-specific indexes and setup"""
        try:
            # User collection indexes
            await self.base.db.users.create_index("phone_number", unique=True)
            await self.base.db.users.create_index("user_id", unique=True, sparse=True)
            await self.base.db.users.create_index([("plan_type", 1), ("subscription_status", 1)])
            await self.base.db.users.create_index("last_active_at")
            
            # Personality and onboarding indexes
            await self.base.db.personality_profiles.create_index("user_id", unique=True)
            await self.base.db.onboarding_progress.create_index("user_id", unique=True)
            
            logger.info("✅ Users service initialized")
        except Exception as e:
            logger.exception(f"❌ Users service initialization failed: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Health check for users service"""
        try:
            # Test user operations
            user_count = await self.base.db.users.count_documents({})
            return {
                "status": "healthy",
                "user_count": user_count,
                "cache_stats": self._cache_stats
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def get_by_phone(self, phone_number: str) -> Optional[UserProfile]:
        """Get user by phone number with caching"""
        try:
            if not phone_number or phone_number.strip() == "":
                return None
            
            # Check cache first
            cache_key = f"user:phone:{phone_number}"
            cached_user = await self.base.key_builder.get(cache_key)
            if cached_user:
                self._cache_stats["hits"] += 1
                return UserProfile(**cached_user)
            
            self._cache_stats["misses"] += 1
            
            # Query database
            user_doc = await self.base.db.users.find_one({"phone_number": phone_number})
            if not user_doc:
                return None
            
            user_doc['_id'] = str(user_doc['_id'])
            user_profile = UserProfile(**user_doc)
            
            # Cache for future requests
            await self.base.key_builder.set(cache_key, user_doc, ttl=3600)
            
            return user_profile
            
        except Exception as e:
            logger.exception(f"❌ Error getting user by phone {phone_number}: {e}")
            return None

    async def get_by_id(self, user_id: str) -> Optional[UserProfile]:
        """Get user by user_id with caching"""
        try:
            if not user_id:
                return None
            
            cache_key = f"user:id:{user_id}"
            cached_user = await self.base.key_builder.get(cache_key)
            if cached_user:
                self._cache_stats["hits"] += 1
                return UserProfile(**cached_user)
            
            self._cache_stats["misses"] += 1
            
            # Try both ObjectId and string ID
            try:
                user_doc = await self.base.db.users.find_one({"_id": ObjectId(user_id)})
            except:
                user_doc = await self.base.db.users.find_one({"user_id": user_id})
            
            if not user_doc:
                return None
            
            user_doc['_id'] = str(user_doc['_id'])
            user_profile = UserProfile(**user_doc)
            
            await self.base.key_builder.set(cache_key, user_doc, ttl=3600)
            return user_profile
            
        except Exception as e:
            logger.exception(f"❌ Error getting user by ID {user_id}: {e}")
            return None

    async def save(self, user: UserProfile) -> str:
        """Save or update user profile with cache invalidation"""
        try:
            if not user.phone_number or user.phone_number.strip() == "":
                raise ValueError("User must have a valid phone_number")
            
            user_dict = user.__dict__.copy()
            user_dict['updated_at'] = datetime.utcnow()
            
            user_id = None
            if user._id:
                user_dict.pop('_id', None)
                await self.base.db.users.update_one(
                    {"_id": ObjectId(user._id)},
                    {"$set": user_dict}
                )
                user_id = user._id
            else:
                user_dict.pop('_id', None)
                result = await self.base.db.users.insert_one(user_dict)
                user_id = str(result.inserted_id)
            
            # Invalidate caches
            await self._invalidate_user_caches(user.phone_number, user_id)
            
            return user_id
            
        except Exception as e:
            logger.exception(f"❌ Error saving user: {e}")
            raise

    async def update_activity(self, phone_number: str) -> bool:
        """Update user activity timestamp"""
        try:
            if not phone_number:
                return False
            
            result = await self.base.db.users.update_one(
                {"phone_number": phone_number},
                {
                    "$set": {
                        "last_active_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    },
                    "$inc": {"total_messages_received": 1}
                }
            )
            
            if result.modified_count > 0:
                # Invalidate cache
                await self.base.key_builder.delete(f"user:phone:{phone_number}")
                return True
            
            return False
            
        except Exception as e:
            logger.exception(f"❌ Error updating user activity: {e}")
            return False

    async def get_usage_count(self, user_id: str, period: str) -> int:
        """Get usage count for period"""
        try:
            key = f"usage:{user_id}:{period}"
            count_data = await self.base.key_builder.get(key)
            if count_data and isinstance(count_data, dict):
                return count_data.get("count", 0)
            elif count_data:
                return int(count_data)
            return 0
        except Exception as e:
            logger.exception(f"❌ Error getting usage count: {e}")
            return 0

    async def increment_usage(self, user_id: str, period: str, ttl: int):
        """Increment usage counter"""
        try:
            key = f"usage:{user_id}:{period}"
            current_data = await self.base.key_builder.get(key) or {
                "count": 0, 
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            new_count = current_data.get("count", 0) + 1
            usage_data = {
                "count": new_count,
                "last_increment": datetime.now(timezone.utc).isoformat(),
                "period": period,
                "user_id": user_id
            }
            
            if "created_at" in current_data:
                usage_data["created_at"] = current_data["created_at"]
            else:
                usage_data["created_at"] = datetime.now(timezone.utc).isoformat()
            
            await self.base.key_builder.set(key, usage_data, ttl=ttl)
            
        except Exception as e:
            logger.exception(f"❌ Error incrementing usage: {e}")

    async def get_usage_summary(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive usage summary"""
        try:
            usage_summary = {
                "user_id": user_id,
                "current_usage": {},
                "limits": {},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Get current usage
            periods = ["week", "month", "day"]
            for period in periods:
                count = await self.get_usage_count(user_id, period)
                usage_summary["current_usage"][period] = count
            
            # Get limits based on user plan
            user = await self.get_by_id(user_id)
            if user:
                from config import settings
                if user.plan_type == "free":
                    usage_summary["limits"]["week"] = settings.free_weekly_limit
                elif user.plan_type == "paid":
                    usage_summary["limits"]["month"] = settings.paid_monthly_limit
                elif user.plan_type == "pro":
                    usage_summary["limits"]["day"] = settings.pro_daily_cooloff
            
            return usage_summary
            
        except Exception as e:
            logger.exception(f"❌ Error getting usage summary: {e}")
            return {"error": str(e)}

    # Personality Profile Methods
    async def get_personality_profile(self, user_id: str) -> Optional[Dict]:
        """Get personality profile with caching"""
        try:
            cache_key = f"user:{user_id}:personality"
            profile = await self.base.key_builder.get(cache_key)
            
            if profile:
                return profile
            
            profile_doc = await self.base.db.personality_profiles.find_one({"user_id": user_id})
            if profile_doc:
                profile_doc.pop('_id', None)
                await self.base.key_builder.set(cache_key, profile_doc, ttl=86400 * 7)
                return profile_doc
            
            return None
        except Exception as e:
            logger.exception(f"❌ Error getting personality profile: {e}")
            return None

    async def save_personality_profile(self, user_id: str, profile: Dict) -> bool:
        """Save personality profile with dual storage"""
        try:
            # Save to Redis cache
            cache_key = f"user:{user_id}:personality"
            await self.base.key_builder.set(cache_key, profile, ttl=86400 * 7)
            
            # Save to MongoDB
            mongo_profile = profile.copy()
            mongo_profile["user_id"] = user_id
            mongo_profile["updated_at"] = datetime.now(timezone.utc)
            
            await self.base.db.personality_profiles.update_one(
                {"user_id": user_id},
                {"$set": mongo_profile},
                upsert=True
            )
            
            return True
        except Exception as e:
            logger.exception(f"❌ Error saving personality profile: {e}")
            return False

    async def cache_analysis_result(self, user_id: str, analysis_data: Dict, ttl: int = 3600) -> bool:
        """Cache analysis result"""
        try:
            cache_key = f"user:{user_id}:last_analysis"
            return await self.base.key_builder.set(cache_key, analysis_data, ttl=ttl)
        except Exception as e:
            logger.exception(f"❌ Error caching analysis: {e}")
            return False

    async def get_cached_analysis(self, user_id: str) -> Optional[Dict]:
        """Get cached analysis result"""
        try:
            cache_key = f"user:{user_id}:last_analysis"
            return await self.base.key_builder.get(cache_key)
        except Exception as e:
            logger.exception(f"❌ Error getting cached analysis: {e}")
            return None

    # Onboarding Methods
    async def save_onboarding_progress(self, user_id: str, progress_data: Dict) -> bool:
        """Save onboarding progress"""
        try:
            progress_doc = {
                "user_id": user_id,
                "current_step": progress_data.get("current_step", 1),
                "completed_steps": progress_data.get("completed_steps", []),
                "collected_data": progress_data.get("collected_data", {}),
                "completion_percentage": progress_data.get("completion_percentage", 0),
                "started_at": progress_data.get("started_at", datetime.now(timezone.utc)),
                "updated_at": datetime.now(timezone.utc),
                "status": progress_data.get("status", "in_progress")
            }
            
            await self.base.db.onboarding_progress.update_one(
                {"user_id": user_id},
                {"$set": progress_doc},
                upsert=True
            )
            
            # Cache the progress
            cache_key = f"onboarding:{user_id}"
            await self.base.key_builder.set(cache_key, progress_doc, ttl=86400)
            
            return True
            
        except Exception as e:
            logger.exception(f"❌ Error saving onboarding progress: {e}")
            return False

    async def get_onboarding_progress(self, user_id: str) -> Optional[Dict]:
        """Get onboarding progress"""
        try:
            cache_key = f"onboarding:{user_id}"
            cached_progress = await self.base.key_builder.get(cache_key)
            if cached_progress:
                return cached_progress
            
            progress = await self.base.db.onboarding_progress.find_one({"user_id": user_id})
            if progress:
                progress.pop("_id", None)
                
                # Convert datetime fields
                for field in ["started_at", "updated_at"]:
                    if isinstance(progress.get(field), datetime):
                        progress[field] = progress[field].isoformat()
                
                await self.base.key_builder.set(cache_key, progress, ttl=86400)
                return progress
            
            return None
            
        except Exception as e:
            logger.exception(f"❌ Error getting onboarding progress: {e}")
            return None

    async def _invalidate_user_caches(self, phone_number: str, user_id: str):
        """Invalidate all user-related caches"""
        try:
            cache_keys = [
                f"user:phone:{phone_number}",
                f"user:id:{user_id}",
                f"user:{user_id}:personality",
                f"onboarding:{user_id}"
            ]
            
            for key in cache_keys:
                await self.base.key_builder.delete(key)
                
        except Exception as e:
            logger.exception(f"❌ Error invalidating user caches: {e}")
