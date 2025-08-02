# services/db/users_service.py
"""
Enhanced Users Service - User management, onboarding, personality profiles
Focused on user lifecycle, authentication, profile management
Enhanced with custom exceptions, base service patterns, monitoring, and validation
"""

from typing import Optional, Dict, List, Any, Union
from datetime import datetime, timezone, timedelta
from bson import ObjectId
from loguru import logger
import asyncio
import time
import re
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, validator, EmailStr
from enum import Enum
import uuid

from models.user import UserProfile
from services.db.base_db_service import BaseDBService


# Configuration for Users Service
class UsersConfig:
    """Users service configuration"""
    
    # Cache TTL values
    USER_CACHE_TTL = 3600  # 1 hour
    PERSONALITY_CACHE_TTL = 86400 * 7  # 1 week
    USAGE_CACHE_TTL = 86400  # 1 day
    ONBOARDING_CACHE_TTL = 86400  # 1 day
    ANALYSIS_CACHE_TTL = 3600  # 1 hour
    
    # Processing limits
    MAX_USAGE_HISTORY_DAYS = 90
    MAX_PERSONALITY_PROFILE_SIZE = 10240  # 10KB
    MAX_ONBOARDING_STEPS = 20
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0
    MAX_RETRY_DELAY = 30.0
    
    # Validation patterns
    PHONE_PATTERN = r'^\+?1?[2-9]\d{2}[2-9]\d{2}\d{4}$'
    
    # Plan types and limits
    PLAN_LIMITS = {
        "free": {
            "weekly_limit": 10,
            "monthly_limit": None,
            "daily_cooloff": None
        },
        "paid": {
            "weekly_limit": None,
            "monthly_limit": 1000,
            "daily_cooloff": None
        },
        "pro": {
            "weekly_limit": None,
            "monthly_limit": None,
            "daily_cooloff": 5  # 5 second cooloff
        }
    }


# Custom Exception Hierarchy
class UsersServiceException(Exception):
    """Base exception for users service errors"""
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc)


class UserValidationError(UsersServiceException):
    """User data validation errors"""
    pass


class UserNotFoundError(UsersServiceException):
    """User not found errors"""
    pass


class UserExistsError(UsersServiceException):
    """User already exists errors"""
    pass


class UsageExceededError(UsersServiceException):
    """Usage limit exceeded errors"""
    pass


class ProfileDataError(UsersServiceException):
    """Profile data operation errors"""
    pass


class OnboardingError(UsersServiceException):
    """Onboarding operation errors"""
    pass


# Enhanced Data Models
class PlanType(str, Enum):
    FREE = "free"
    PAID = "paid"
    PRO = "pro"


class UserStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class SubscriptionStatus(str, Enum):
    ACTIVE = "active"
    CANCELLED = "cancelled"
    PAST_DUE = "past_due"
    UNPAID = "unpaid"


class OnboardingStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class EnhancedUserProfile(BaseModel):
    """Enhanced user profile with comprehensive validation"""
    _id: Optional[str] = None
    user_id: Optional[str] = None
    phone_number: str = Field(..., min_length=10, max_length=15)
    email: Optional[EmailStr] = None
    first_name: Optional[str] = Field(None, min_length=1, max_length=50)
    last_name: Optional[str] = Field(None, min_length=1, max_length=50)
    plan_type: PlanType = Field(PlanType.FREE)
    status: UserStatus = Field(UserStatus.ACTIVE)
    subscription_status: Optional[SubscriptionStatus] = None
    
    # Activity tracking
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_active_at: Optional[datetime] = None
    total_messages_sent: int = Field(0, ge=0)
    total_messages_received: int = Field(0, ge=0)
    
    # Preferences and settings
    timezone: str = Field("UTC")
    language: str = Field("en")
    notifications_enabled: bool = Field(True)
    marketing_emails: bool = Field(False)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('phone_number')
    def validate_phone_number(cls, v):
        """Validate and normalize phone number"""
        if not v:
            raise ValueError('Phone number is required')
        
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', v)
        
        # Validate length
        if len(digits) < 10:
            raise ValueError('Phone number too short')
        if len(digits) > 15:
            raise ValueError('Phone number too long')
        
        # Handle different formats
        if len(digits) == 11 and digits.startswith('1'):
            return f"+1{digits[1:]}"
        elif len(digits) == 10:
            return f"+1{digits}"
        elif len(digits) > 10:
            return f"+{digits}"
        
        return v
    
    @validator('first_name', 'last_name')
    def validate_name_fields(cls, v):
        """Validate name fields"""
        if v is not None:
            v = v.strip()
            if not v:
                return None
            # Remove any numbers or special characters except hyphens and apostrophes
            if not re.match(r"^[a-zA-Z\s\-']+$", v):
                raise ValueError('Names can only contain letters, spaces, hyphens, and apostrophes')
        return v
    
    class Config:
        validate_assignment = True
        use_enum_values = True


class PersonalityProfile(BaseModel):
    """Enhanced personality profile model"""
    user_id: str = Field(..., min_length=1)
    risk_tolerance: str = Field("moderate", regex="^(conservative|moderate|aggressive)$")
    investment_experience: str = Field("beginner", regex="^(beginner|intermediate|advanced|expert)$")
    trading_style: str = Field("long_term", regex="^(day_trading|swing|long_term|buy_and_hold)$")
    communication_style: str = Field("balanced", regex="^(concise|detailed|balanced)$")
    
    # Behavioral traits
    interests: List[str] = Field(default_factory=list)
    goals: List[str] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    
    # AI insights
    ai_generated_traits: Dict[str, Any] = Field(default_factory=dict)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @validator('interests', 'goals')
    def validate_list_length(cls, v):
        if len(v) > 20:
            raise ValueError('Too many items in list (max 20)')
        return v


class OnboardingProgress(BaseModel):
    """Enhanced onboarding progress model"""
    user_id: str = Field(..., min_length=1)
    current_step: int = Field(1, ge=1, le=20)
    completed_steps: List[int] = Field(default_factory=list)
    collected_data: Dict[str, Any] = Field(default_factory=dict)
    completion_percentage: float = Field(0, ge=0, le=100)
    status: OnboardingStatus = Field(OnboardingStatus.NOT_STARTED)
    
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @validator('completed_steps')
    def validate_completed_steps(cls, v, values):
        current_step = values.get('current_step', 1)
        if v and max(v) > current_step:
            raise ValueError('Completed steps cannot exceed current step')
        return v


# Monitoring and Metrics
class UsersServiceMonitor:
    """Users service monitoring and metrics collection"""
    
    def __init__(self):
        self.metrics = {}
        self.active_operations = {}
        self.start_time = time.time()
        self.operation_counts = {
            "users_created": 0,
            "users_updated": 0,
            "profiles_saved": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "validation_errors": 0,
            "not_found_errors": 0
        }
    
    @asynccontextmanager
    async def track_operation(self, operation_name: str, metadata: Dict = None):
        """Context manager for tracking operation metrics"""
        start_time = time.time()
        operation_id = f"{operation_name}_{int(start_time * 1000)}"
        
        self.active_operations[operation_id] = {
            "name": operation_name,
            "start_time": start_time,
            "metadata": metadata or {}
        }
        
        try:
            yield operation_id
        except Exception as e:
            self._record_error(operation_name, str(e))
            raise
        finally:
            duration = time.time() - start_time
            self._record_success(operation_name, duration)
            self.active_operations.pop(operation_id, None)
    
    def _record_success(self, operation: str, duration: float):
        if operation not in self.metrics:
            self.metrics[operation] = {
                "count": 0,
                "total_duration": 0,
                "errors": 0,
                "avg_duration": 0,
                "min_duration": float('inf'),
                "max_duration": 0
            }
        
        metrics = self.metrics[operation]
        metrics["count"] += 1
        metrics["total_duration"] += duration
        metrics["avg_duration"] = metrics["total_duration"] / metrics["count"]
        metrics["min_duration"] = min(metrics["min_duration"], duration)
        metrics["max_duration"] = max(metrics["max_duration"], duration)
    
    def _record_error(self, operation: str, error: str):
        if operation not in self.metrics:
            self.metrics[operation] = {"count": 0, "total_duration": 0, "errors": 0}
        self.metrics[operation]["errors"] += 1
    
    def increment_counter(self, counter_name: str):
        """Increment operation counter"""
        if counter_name in self.operation_counts:
            self.operation_counts[counter_name] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        return {
            "metrics": self.metrics,
            "operation_counts": self.operation_counts,
            "active_operations": len(self.active_operations),
            "uptime_seconds": uptime,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Base Enhanced Service for Users
class BaseUsersService:
    """Base class with common patterns for user operations"""
    
    def __init__(self, base_service: BaseDBService, config: UsersConfig = None):
        self.base = base_service
        self.config = config or UsersConfig()
        self.monitor = UsersServiceMonitor()
        self._service_id = str(uuid.uuid4())
        self._cache_stats = {"hits": 0, "misses": 0}
    
    async def with_retry(self, operation, max_retries: int = None, base_delay: float = None):
        """Generic retry wrapper with exponential backoff"""
        max_retries = max_retries or self.config.MAX_RETRIES
        base_delay = base_delay or self.config.RETRY_BASE_DELAY
        
        for attempt in range(max_retries + 1):
            try:
                result = await operation()
                return result
            except Exception as e:
                if attempt == max_retries:
                    raise UsersServiceException(
                        f"Operation failed after {max_retries} retries: {str(e)}",
                        "MAX_RETRIES_EXCEEDED",
                        {"attempt": attempt, "max_retries": max_retries}
                    )
                
                delay = min(base_delay * (2 ** attempt), self.config.MAX_RETRY_DELAY)
                logger.warning(f"Users operation retry {attempt + 1}/{max_retries} in {delay}s: {str(e)}")
                await asyncio.sleep(delay)
    
    async def cache_with_fallback(self, cache_key: str, fallback_func, ttl: int):
        """Standard cache-with-fallback pattern"""
        try:
            cached = await self.base.key_builder.get(cache_key)
            if cached:
                self._cache_stats["hits"] += 1
                self.monitor.increment_counter("cache_hits")
                return cached
        except Exception as e:
            logger.warning(f"Cache read failed for {cache_key}: {e}")
        
        self._cache_stats["misses"] += 1
        self.monitor.increment_counter("cache_misses")
        result = await fallback_func()
        
        try:
            await self.base.key_builder.set(cache_key, result, ttl=ttl)
        except Exception as e:
            logger.warning(f"Cache write failed for {cache_key}: {e}")
        
        return result
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics"""
        return {
            "service_id": self._service_id,
            "cache_stats": self._cache_stats.copy(),
            "monitor_metrics": self.monitor.get_metrics(),
            "config": {
                "user_cache_ttl": self.config.USER_CACHE_TTL,
                "max_retries": self.config.MAX_RETRIES,
                "plan_limits": self.config.PLAN_LIMITS
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class UsersService(BaseUsersService):
    """Enhanced specialized service for user management and personality profiles"""
    
    def __init__(self, base_service: BaseDBService, config: UsersConfig = None):
        super().__init__(base_service, config)

    async def initialize(self):
        """Initialize user-specific indexes and setup"""
        try:
            # User collection indexes
            await self.base.db.users.create_index("phone_number", unique=True)
            await self.base.db.users.create_index("user_id", unique=True, sparse=True)
            await self.base.db.users.create_index("email", sparse=True)
            await self.base.db.users.create_index([("plan_type", 1), ("subscription_status", 1)])
            await self.base.db.users.create_index("last_active_at")
            await self.base.db.users.create_index("status")
            await self.base.db.users.create_index("created_at")
            
            # Personality and onboarding indexes
            await self.base.db.personality_profiles.create_index("user_id", unique=True)
            await self.base.db.personality_profiles.create_index("risk_tolerance")
            await self.base.db.personality_profiles.create_index("trading_style")
            await self.base.db.onboarding_progress.create_index("user_id", unique=True)
            await self.base.db.onboarding_progress.create_index("status")
            
            logger.info("✅ Enhanced Users service initialized")
        except Exception as e:
            logger.exception(f"❌ Enhanced Users service initialization failed: {e}")
            raise UsersServiceException(f"Failed to initialize users service: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check for users service"""
        try:
            async with self.monitor.track_operation("health_check"):
                # Test user operations
                user_count = await self.base.db.users.count_documents({})
                active_users = await self.base.db.users.count_documents({"status": "active"})
                profiles_count = await self.base.db.personality_profiles.count_documents({})
                onboarding_count = await self.base.db.onboarding_progress.count_documents({})
                
                # Check for recent activity
                recent_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
                recent_active = await self.base.db.users.count_documents({
                    "last_active_at": {"$gte": recent_cutoff}
                })
                
                return {
                    "status": "healthy",
                    "user_count": user_count,
                    "active_users": active_users,
                    "profiles_count": profiles_count,
                    "onboarding_count": onboarding_count,
                    "recent_active": recent_active,
                    "service_metrics": self.get_service_metrics()
                }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    # Enhanced User Management Methods
    async def get_by_phone(self, phone_number: str) -> Optional[UserProfile]:
        """Get user by phone number with enhanced caching and validation"""
        if not phone_number or phone_number.strip() == "":
            raise UserValidationError("Phone number cannot be empty")
        
        try:
            # Normalize phone number
            normalized_phone = self._normalize_phone_number(phone_number)
            
            cache_key = f"user:phone:{normalized_phone}"
            
            async def fetch_user():
                user_doc = await self.base.db.users.find_one({"phone_number": normalized_phone})
                if not user_doc:
                    return None
                
                user_doc['_id'] = str(user_doc['_id'])
                return user_doc
            
            async with self.monitor.track_operation("get_by_phone", {"phone": normalized_phone}):
                user_doc = await self.cache_with_fallback(cache_key, fetch_user, self.config.USER_CACHE_TTL)
                
                if user_doc:
                    return UserProfile(**user_doc)
                
                return None
                
        except UserValidationError:
            raise
        except Exception as e:
            logger.exception(f"❌ Error getting user by phone {phone_number}: {e}")
            raise UsersServiceException(f"Failed to get user by phone: {e}")

    async def get_by_id(self, user_id: str) -> Optional[UserProfile]:
        """Get user by user_id with enhanced caching and validation"""
        if not user_id:
            raise UserValidationError("User ID cannot be empty")
        
        try:
            cache_key = f"user:id:{user_id}"
            
            async def fetch_user():
                # Try both ObjectId and string ID
                try:
                    user_doc = await self.base.db.users.find_one({"_id": ObjectId(user_id)})
                except:
                    user_doc = await self.base.db.users.find_one({"user_id": user_id})
                
                if not user_doc:
                    return None
                
                user_doc['_id'] = str(user_doc['_id'])
                return user_doc
            
            async with self.monitor.track_operation("get_by_id", {"user_id": user_id}):
                user_doc = await self.cache_with_fallback(cache_key, fetch_user, self.config.USER_CACHE_TTL)
                
                if user_doc:
                    return UserProfile(**user_doc)
                
                return None
                
        except Exception as e:
            logger.exception(f"❌ Error getting user by ID {user_id}: {e}")
            raise UsersServiceException(f"Failed to get user by ID: {e}")

    async def save(self, user: UserProfile) -> str:
        """Save or update user profile with enhanced validation and cache management"""
        try:
            # Validate using enhanced model
            enhanced_user = EnhancedUserProfile(**user.__dict__)
            
            async with self.monitor.track_operation("save_user", {"plan_type": enhanced_user.plan_type.value}):
                user_dict = enhanced_user.dict(exclude={'_id'})
                user_dict['updated_at'] = datetime.now(timezone.utc)
                
                # Convert enums to strings for storage
                user_dict['plan_type'] = enhanced_user.plan_type.value
                user_dict['status'] = enhanced_user.status.value
                if enhanced_user.subscription_status:
                    user_dict['subscription_status'] = enhanced_user.subscription_status.value
                
                user_id = None
                if user._id:
                    # Update existing user
                    result = await self.base.db.users.update_one(
                        {"_id": ObjectId(user._id)},
                        {"$set": user_dict}
                    )
                    if result.modified_count == 0:
                        raise UserNotFoundError(f"User {user._id} not found for update")
                    user_id = user._id
                    self.monitor.increment_counter("users_updated")
                else:
                    # Create new user
                    user_dict['created_at'] = datetime.now(timezone.utc)
                    
                    # Check if user already exists
                    existing = await self.base.db.users.find_one({"phone_number": enhanced_user.phone_number})
                    if existing:
                        raise UserExistsError(f"User with phone {enhanced_user.phone_number} already exists")
                    
                    result = await self.base.db.users.insert_one(user_dict)
                    user_id = str(result.inserted_id)
                    self.monitor.increment_counter("users_created")
                
                # Invalidate caches
                await self._invalidate_user_caches(enhanced_user.phone_number, user_id)
                
                logger.info(f"✅ Saved user profile: {enhanced_user.phone_number}")
                return user_id
                
        except (UserValidationError, UserExistsError, UserNotFoundError):
            raise
        except Exception as e:
            logger.exception(f"❌ Error saving user: {e}")
            raise UsersServiceException(f"Failed to save user: {e}")

    async def update_activity(self, phone_number: str, increment_messages: bool = True) -> bool:
        """Update user activity timestamp with enhanced tracking"""
        if not phone_number:
            raise UserValidationError("Phone number cannot be empty")
        
        try:
            normalized_phone = self._normalize_phone_number(phone_number)
            
            async with self.monitor.track_operation("update_activity", {"phone": normalized_phone}):
                update_doc = {
                    "last_active_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc)
                }
                
                if increment_messages:
                    update_doc = {"$set": update_doc, "$inc": {"total_messages_received": 1}}
                else:
                    update_doc = {"$set": update_doc}
                
                result = await self.base.db.users.update_one(
                    {"phone_number": normalized_phone},
                    update_doc
                )
                
                if result.modified_count > 0:
                    # Invalidate cache
                    await self.base.key_builder.delete(f"user:phone:{normalized_phone}")
                    return True
                
                return False
                
        except UserValidationError:
            raise
        except Exception as e:
            logger.exception(f"❌ Error updating user activity: {e}")
            raise UsersServiceException(f"Failed to update user activity: {e}")

    # Enhanced Usage Tracking Methods
    async def get_usage_count(self, user_id: str, period: str) -> int:
        """Get usage count for period with validation"""
        if not user_id:
            raise UserValidationError("User ID cannot be empty")
        
        if period not in ["day", "week", "month"]:
            raise UserValidationError("Period must be 'day', 'week', or 'month'")
        
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
        """Increment usage counter with enhanced tracking"""
        if not user_id:
            raise UserValidationError("User ID cannot be empty")
        
        if period not in ["day", "week", "month"]:
            raise UserValidationError("Period must be 'day', 'week', or 'month'")
        
        try:
            async with self.monitor.track_operation("increment_usage", {"user_id": user_id, "period": period}):
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
                
        except UserValidationError:
            raise
        except Exception as e:
            logger.exception(f"❌ Error incrementing usage: {e}")
            raise UsersServiceException(f"Failed to increment usage: {e}")

    async def check_usage_limits(self, user_id: str) -> Dict[str, Any]:
        """Check usage limits for user with comprehensive analysis"""
        if not user_id:
            raise UserValidationError("User ID cannot be empty")
        
        try:
            async with self.monitor.track_operation("check_usage_limits", {"user_id": user_id}):
                # Get user to determine plan
                user = await self.get_by_id(user_id)
                if not user:
                    raise UserNotFoundError(f"User {user_id} not found")
                
                plan_limits = self.config.PLAN_LIMITS.get(user.plan_type, self.config.PLAN_LIMITS["free"])
                
                usage_status = {
                    "user_id": user_id,
                    "plan_type": user.plan_type,
                    "limits": plan_limits,
                    "current_usage": {},
                    "limits_exceeded": {},
                    "remaining": {},
                    "reset_times": {},
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # Check each applicable limit
                for period, limit in plan_limits.items():
                    if limit is not None:
                        current_count = await self.get_usage_count(user_id, period.replace("_limit", ""))
                        usage_status["current_usage"][period] = current_count
                        usage_status["limits_exceeded"][period] = current_count >= limit
                        usage_status["remaining"][period] = max(0, limit - current_count)
                        
                        # Calculate reset time
                        if period == "weekly_limit":
                            # Reset on Monday
                            now = datetime.now(timezone.utc)
                            days_until_monday = (7 - now.weekday()) % 7
                            reset_time = now + timedelta(days=days_until_monday)
                            usage_status["reset_times"][period] = reset_time.isoformat()
                        elif period == "monthly_limit":
                            # Reset on first of next month
                            now = datetime.now(timezone.utc)
                            if now.month == 12:
                                reset_time = now.replace(year=now.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                            else:
                                reset_time = now.replace(month=now.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
                            usage_status["reset_times"][period] = reset_time.isoformat()
                
                return usage_status
                
        except (UserValidationError, UserNotFoundError):
            raise
        except Exception as e:
            logger.exception(f"❌ Error checking usage limits: {e}")
            raise UsersServiceException(f"Failed to check usage limits: {e}")

    async def get_usage_summary(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive usage summary with enhanced analytics"""
        if not user_id:
            raise UserValidationError("User ID cannot be empty")
        
        try:
            async with self.monitor.track_operation("get_usage_summary", {"user_id": user_id}):
                usage_limits = await self.check_usage_limits(user_id)
                
                # Get historical usage (if available)
                historical_usage = {}
                for period in ["day", "week", "month"]:
                    historical_usage[period] = await self.get_usage_count(user_id, period)
                
                usage_summary = {
                    "user_id": user_id,
                    "current_usage": historical_usage,
                    "limits": usage_limits["limits"],
                    "limits_exceeded": usage_limits["limits_exceeded"],
                    "remaining": usage_limits["remaining"],
                    "reset_times": usage_limits["reset_times"],
                    "plan_type": usage_limits["plan_type"],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                return usage_summary
                
        except (UserValidationError, UserNotFoundError):
            raise
        except Exception as e:
            logger.exception(f"❌ Error getting usage summary: {e}")
            raise UsersServiceException(f"Failed to get usage summary: {e}")

    # Enhanced Personality Profile Methods
    async def get_personality_profile(self, user_id: str) -> Optional[Dict]:
        """Get personality profile with enhanced caching and validation"""
        if not user_id:
            raise UserValidationError("User ID cannot be empty")
        
        try:
            cache_key = f"user:{user_id}:personality"
            
            async def fetch_profile():
                profile_doc = await self.base.db.personality_profiles.find_one({"user_id": user_id})
                if profile_doc:
                    profile_doc.pop('_id', None)
                    # Convert datetime fields
                    for field in ["created_at", "updated_at"]:
                        if isinstance(profile_doc.get(field), datetime):
                            profile_doc[field] = profile_doc[field].isoformat()
                    return profile_doc
                return None
            
            async with self.monitor.track_operation("get_personality_profile", {"user_id": user_id}):
                profile = await self.cache_with_fallback(cache_key, fetch_profile, self.config.PERSONALITY_CACHE_TTL)
                return profile
                
        except Exception as e:
            logger.exception(f"❌ Error getting personality profile: {e}")
            raise ProfileDataError(f"Failed to get personality profile: {e}")

    async def save_personality_profile(self, user_id: str, profile: Dict) -> bool:
        """Save personality profile with enhanced validation and dual storage"""
        if not user_id:
            raise UserValidationError("User ID cannot be empty")
        
        if not profile:
            raise UserValidationError("Profile data cannot be empty")
        
        try:
            # Validate using enhanced model
            enhanced_profile = PersonalityProfile(user_id=user_id, **profile)
            
            async with self.monitor.track_operation("save_personality_profile", {"user_id": user_id}):
                # Save to Redis cache
                cache_key = f"user:{user_id}:personality"
                profile_dict = enhanced_profile.dict()
                
                # Convert datetime fields to ISO format for cache
                for field in ["created_at", "updated_at"]:
                    if isinstance(profile_dict.get(field), datetime):
                        profile_dict[field] = profile_dict[field].isoformat()
                
                await self.base.key_builder.set(cache_key, profile_dict, ttl=self.config.PERSONALITY_CACHE_TTL)
                
                # Save to MongoDB
                mongo_profile = enhanced_profile.dict()
                mongo_profile["updated_at"] = datetime.now(timezone.utc)
                
                await self.base.db.personality_profiles.update_one(
                    {"user_id": user_id},
                    {"$set": mongo_profile},
                    upsert=True
                )
                
                self.monitor.increment_counter("profiles_saved")
                logger.info(f"✅ Saved personality profile for user {user_id}")
                return True
                
        except UserValidationError:
            raise
        except Exception as e:
            logger.exception(f"❌ Error saving personality profile: {e}")
            raise ProfileDataError(f"Failed to save personality profile: {e}")

    async def update_personality_traits(self, user_id: str, traits: Dict[str, Any]) -> bool:
        """Update specific personality traits without overwriting entire profile"""
        if not user_id:
            raise UserValidationError("User ID cannot be empty")
        
        if not traits:
            raise UserValidationError("Traits data cannot be empty")
        
        try:
            async with self.monitor.track_operation("update_personality_traits", {"user_id": user_id}):
                # Get current profile
                current_profile = await self.get_personality_profile(user_id)
                if not current_profile:
                    # Create new profile with traits
                    new_profile = {"user_id": user_id, **traits}
                    return await self.save_personality_profile(user_id, new_profile)
                
                # Update existing profile
                current_profile.update(traits)
                current_profile["updated_at"] = datetime.now(timezone.utc).isoformat()
                
                return await self.save_personality_profile(user_id, current_profile)
                
        except (UserValidationError, ProfileDataError):
            raise
        except Exception as e:
            logger.exception(f"❌ Error updating personality traits: {e}")
            raise ProfileDataError(f"Failed to update personality traits: {e}")

    # Enhanced Analysis and Caching Methods
    async def cache_analysis_result(self, user_id: str, analysis_data: Dict, ttl: int = None) -> bool:
        """Cache analysis result with validation"""
        if not user_id:
            raise UserValidationError("User ID cannot be empty")
        
        if not analysis_data:
            raise UserValidationError("Analysis data cannot be empty")
        
        ttl = ttl or self.config.ANALYSIS_CACHE_TTL
        
        try:
            async with self.monitor.track_operation("cache_analysis_result", {"user_id": user_id}):
                cache_key = f"user:{user_id}:last_analysis"
                
                # Add metadata
                cache_data = {
                    "analysis": analysis_data,
                    "cached_at": datetime.now(timezone.utc).isoformat(),
                    "user_id": user_id
                }
                
                return await self.base.key_builder.set(cache_key, cache_data, ttl=ttl)
                
        except UserValidationError:
            raise
        except Exception as e:
            logger.exception(f"❌ Error caching analysis: {e}")
            return False

    async def get_cached_analysis(self, user_id: str) -> Optional[Dict]:
        """Get cached analysis result with validation"""
        if not user_id:
            raise UserValidationError("User ID cannot be empty")
        
        try:
            async with self.monitor.track_operation("get_cached_analysis", {"user_id": user_id}):
                cache_key = f"user:{user_id}:last_analysis"
                cached_data = await self.base.key_builder.get(cache_key)
                
                if cached_data and isinstance(cached_data, dict):
                    return cached_data.get("analysis")
                
                return cached_data
                
        except Exception as e:
            logger.exception(f"❌ Error getting cached analysis: {e}")
            return None

    # Enhanced Onboarding Methods
    async def save_onboarding_progress(self, user_id: str, progress_data: Dict) -> bool:
        """Save onboarding progress with enhanced validation"""
        if not user_id:
            raise UserValidationError("User ID cannot be empty")
        
        if not progress_data:
            raise UserValidationError("Progress data cannot be empty")
        
        try:
            # Validate using enhanced model
            enhanced_progress = OnboardingProgress(user_id=user_id, **progress_data)
            
            async with self.monitor.track_operation("save_onboarding_progress", {"user_id": user_id}):
                progress_dict = enhanced_progress.dict()
                progress_dict["updated_at"] = datetime.now(timezone.utc)
                
                # Convert enums to strings for storage
                progress_dict["status"] = enhanced_progress.status.value
                
                # Set started_at if not already set and status is in_progress
                if not progress_dict.get("started_at") and enhanced_progress.status == OnboardingStatus.IN_PROGRESS:
                    progress_dict["started_at"] = datetime.now(timezone.utc)
                
                # Set completed_at if status is completed
                if enhanced_progress.status == OnboardingStatus.COMPLETED and not progress_dict.get("completed_at"):
                    progress_dict["completed_at"] = datetime.now(timezone.utc)
                
                await self.base.db.onboarding_progress.update_one(
                    {"user_id": user_id},
                    {"$set": progress_dict},
                    upsert=True
                )
                
                # Cache the progress
                cache_key = f"onboarding:{user_id}"
                # Convert datetime fields to ISO format for cache
                cache_dict = progress_dict.copy()
                for field in ["started_at", "completed_at", "updated_at"]:
                    if isinstance(cache_dict.get(field), datetime):
                        cache_dict[field] = cache_dict[field].isoformat()
                
                await self.base.key_builder.set(cache_key, cache_dict, ttl=self.config.ONBOARDING_CACHE_TTL)
                
                logger.info(f"✅ Saved onboarding progress for user {user_id}: step {enhanced_progress.current_step}")
                return True
                
        except UserValidationError:
            raise
        except Exception as e:
            logger.exception(f"❌ Error saving onboarding progress: {e}")
            raise OnboardingError(f"Failed to save onboarding progress: {e}")

    async def get_onboarding_progress(self, user_id: str) -> Optional[Dict]:
        """Get onboarding progress with enhanced caching"""
        if not user_id:
            raise UserValidationError("User ID cannot be empty")
        
        try:
            cache_key = f"onboarding:{user_id}"
            
            async def fetch_progress():
                progress = await self.base.db.onboarding_progress.find_one({"user_id": user_id})
                if progress:
                    progress.pop("_id", None)
                    
                    # Convert datetime fields
                    for field in ["started_at", "completed_at", "updated_at"]:
                        if isinstance(progress.get(field), datetime):
                            progress[field] = progress[field].isoformat()
                    
                    return progress
                return None
            
            async with self.monitor.track_operation("get_onboarding_progress", {"user_id": user_id}):
                progress = await self.cache_with_fallback(cache_key, fetch_progress, self.config.ONBOARDING_CACHE_TTL)
                return progress
                
        except Exception as e:
            logger.exception(f"❌ Error getting onboarding progress: {e}")
            raise OnboardingError(f"Failed to get onboarding progress: {e}")

    async def complete_onboarding_step(self, user_id: str, step_number: int, step_data: Dict = None) -> bool:
        """Complete a specific onboarding step with validation"""
        if not user_id:
            raise UserValidationError("User ID cannot be empty")
        
        if step_number < 1 or step_number > self.config.MAX_ONBOARDING_STEPS:
            raise UserValidationError(f"Step number must be between 1 and {self.config.MAX_ONBOARDING_STEPS}")
        
        try:
            async with self.monitor.track_operation("complete_onboarding_step", 
                                                  {"user_id": user_id, "step": step_number}):
                # Get current progress
                progress = await self.get_onboarding_progress(user_id) or {
                    "current_step": 1,
                    "completed_steps": [],
                    "collected_data": {},
                    "completion_percentage": 0,
                    "status": "not_started"
                }
                
                # Update progress
                if step_number not in progress["completed_steps"]:
                    progress["completed_steps"].append(step_number)
                    progress["completed_steps"].sort()
                
                # Update collected data
                if step_data:
                    progress["collected_data"].update(step_data)
                
                # Update current step to next incomplete step
                all_steps = set(range(1, self.config.MAX_ONBOARDING_STEPS + 1))
                completed_steps = set(progress["completed_steps"])
                remaining_steps = all_steps - completed_steps
                
                if remaining_steps:
                    progress["current_step"] = min(remaining_steps)
                    progress["status"] = "in_progress"
                else:
                    progress["current_step"] = self.config.MAX_ONBOARDING_STEPS
                    progress["status"] = "completed"
                
                # Calculate completion percentage
                progress["completion_percentage"] = (len(progress["completed_steps"]) / self.config.MAX_ONBOARDING_STEPS) * 100
                
                return await self.save_onboarding_progress(user_id, progress)
                
        except (UserValidationError, OnboardingError):
            raise
        except Exception as e:
            logger.exception(f"❌ Error completing onboarding step: {e}")
            raise OnboardingError(f"Failed to complete onboarding step: {e}")

    # Enhanced Helper Methods
    async def _invalidate_user_caches(self, phone_number: str, user_id: str):
        """Invalidate all user-related caches"""
        try:
            cache_keys = [
                f"user:phone:{phone_number}",
                f"user:id:{user_id}",
                f"user:{user_id}:personality",
                f"onboarding:{user_id}",
                f"user:{user_id}:last_analysis"
            ]
            
            for key in cache_keys:
                await self.base.key_builder.delete(key)
                
        except Exception as e:
            logger.exception(f"❌ Error invalidating user caches: {e}")

    def _normalize_phone_number(self, phone: str) -> str:
        """Enhanced phone number normalization with comprehensive validation"""
        if not phone:
            raise UserValidationError("Phone number cannot be empty")
        
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', phone)
        
        # Validate length
        if len(digits) < 10:
            raise UserValidationError(f"Phone number too short: {phone}")
        if len(digits) > 15:
            raise UserValidationError(f"Phone number too long: {phone}")
        
        # Handle different formats
        if len(digits) == 11 and digits.startswith('1'):
            # US number with country code
            normalized = f"+1{digits[1:]}"
        elif len(digits) == 10:
            # US number without country code
            normalized = f"+1{digits}"
        elif len(digits) > 10:
            # International number
            normalized = f"+{digits}"
        else:
            raise UserValidationError(f"Invalid phone number format: {phone}")
        
        # Additional validation using regex
        if not re.match(self.config.PHONE_PATTERN, normalized):
            raise UserValidationError(f"Phone number format not supported: {phone}")
        
        return normalized

    # Enhanced Analytics Methods
    async def get_user_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive user analytics"""
        if not user_id:
            raise UserValidationError("User ID cannot be empty")
        
        if days <= 0 or days > self.config.MAX_USAGE_HISTORY_DAYS:
            raise UserValidationError(f"Days must be between 1 and {self.config.MAX_USAGE_HISTORY_DAYS}")
        
        try:
            async with self.monitor.track_operation("get_user_analytics", {"user_id": user_id, "days": days}):
                # Get user profile
                user = await self.get_by_id(user_id)
                if not user:
                    raise UserNotFoundError(f"User {user_id} not found")
                
                # Get usage summary
                usage_summary = await self.get_usage_summary(user_id)
                
                # Get personality profile
                personality = await self.get_personality_profile(user_id)
                
                # Get onboarding progress
                onboarding = await self.get_onboarding_progress(user_id)
                
                # Calculate account age
                account_age_days = 0
                if user.created_at:
                    account_age = datetime.now(timezone.utc) - user.created_at
                    account_age_days = account_age.days
                
                # Calculate activity metrics
                last_active_days = None
                if user.last_active_at:
                    last_active = datetime.now(timezone.utc) - user.last_active_at
                    last_active_days = last_active.days
                
                analytics = {
                    "user_id": user_id,
                    "account_age_days": account_age_days,
                    "last_active_days": last_active_days,
                    "plan_type": user.plan_type,
                    "status": user.status,
                    "total_messages": {
                        "sent": user.total_messages_sent,
                        "received": user.total_messages_received
                    },
                    "usage_summary": usage_summary,
                    "personality_complete": personality is not None,
                    "onboarding_complete": onboarding and onboarding.get("status") == "completed",
                    "onboarding_progress": onboarding.get("completion_percentage", 0) if onboarding else 0,
                    "preferences_set": bool(user.metadata),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                return analytics
                
        except (UserValidationError, UserNotFoundError):
            raise
        except Exception as e:
            logger.exception(f"❌ Error getting user analytics: {e}")
            raise UsersServiceException(f"Failed to get user analytics: {e}")

    async def get_user_segments(self, limit: int = 1000) -> Dict[str, Any]:
        """Get user segmentation data"""
        if limit <= 0 or limit > 10000:
            raise UserValidationError("Limit must be between 1 and 10000")
        
        try:
            async with self.monitor.track_operation("get_user_segments"):
                # Aggregate user data for segmentation
                pipeline = [
                    {"$limit": limit},
                    {"$group": {
                        "_id": {
                            "plan_type": "$plan_type",
                            "status": "$status"
                        },
                        "count": {"$sum": 1},
                        "avg_messages_sent": {"$avg": "$total_messages_sent"},
                        "avg_messages_received": {"$avg": "$total_messages_received"}
                    }},
                    {"$sort": {"count": -1}}
                ]
                
                segments = await self.base.db.users.aggregate(pipeline).to_list(length=None)
                
                # Get personality profile distribution
                personality_pipeline = [
                    {"$group": {
                        "_id": {
                            "risk_tolerance": "$risk_tolerance",
                            "trading_style": "$trading_style"
                        },
                        "count": {"$sum": 1}
                    }},
                    {"$sort": {"count": -1}}
                ]
                
                personality_segments = await self.base.db.personality_profiles.aggregate(personality_pipeline).to_list(length=None)
                
                # Get onboarding completion rates
                onboarding_pipeline = [
                    {"$group": {
                        "_id": "$status",
                        "count": {"$sum": 1},
                        "avg_completion": {"$avg": "$completion_percentage"}
                    }}
                ]
                
                onboarding_segments = await self.base.db.onboarding_progress.aggregate(onboarding_pipeline).to_list(length=None)
                
                return {
                    "user_segments": segments,
                    "personality_segments": personality_segments,
                    "onboarding_segments": onboarding_segments,
                    "total_analyzed": limit,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            logger.exception(f"❌ Error getting user segments: {e}")
            raise UsersServiceException(f"Failed to get user segments: {e}")


# Test utilities for enhanced testing
class UsersServiceTestHooks:
    """Enhanced test hooks for users service testing"""
    
    @staticmethod
    def create_test_users_service(mock_db: bool = True, config: UsersConfig = None):
        """Create users service with test doubles"""
        from unittest.mock import MagicMock
        
        # Mock base service
        base_service = MagicMock()
        
        if mock_db:
            base_service.db = MagicMock()
            base_service.db.users = MagicMock()
            base_service.db.personality_profiles = MagicMock()
            base_service.db.onboarding_progress = MagicMock()
            base_service.key_builder = MagicMock()
        
        # Use test config if not provided
        test_config = config or UsersConfig()
        
        service = UsersService(base_service, test_config)
        return service
    
    @staticmethod
    def create_test_user_data(**overrides) -> Dict[str, Any]:
        """Create test user data"""
        base_data = {
            "phone_number": "+1234567890",
            "email": "test@example.com",
            "first_name": "Test",
            "last_name": "User",
            "plan_type": "free",
            "status": "active"
        }
        base_data.update(overrides)
        return base_data
    
    @staticmethod
    def create_test_personality_data(**overrides) -> Dict[str, Any]:
        """Create test personality profile data"""
        base_data = {
            "risk_tolerance": "moderate",
            "investment_experience": "beginner",
            "trading_style": "long_term",
            "communication_style": "balanced",
            "interests": ["stocks", "crypto"],
            "goals": ["retirement", "emergency_fund"]
        }
        base_data.update(overrides)
        return base_data
    
    @staticmethod
    def create_test_onboarding_data(**overrides) -> Dict[str, Any]:
        """Create test onboarding progress data"""
        base_data = {
            "current_step": 1,
            "completed_steps": [],
            "collected_data": {},
            "completion_percentage": 0,
            "status": "not_started"
        }
        base_data.update(overrides)
        return base_data


# Export the enhanced users service
__all__ = [
    'UsersService',
    'UsersConfig',
    'UsersServiceException',
    'UserValidationError',
    'UserNotFoundError',
    'UserExistsError',
    'UsageExceededError',
    'ProfileDataError',
    'OnboardingError',
    'EnhancedUserProfile',
    'PersonalityProfile',
    'OnboardingProgress',
    'PlanType',
    'UserStatus',
    'SubscriptionStatus',
    'OnboardingStatus',
    'UsersServiceTestHooks'
]
