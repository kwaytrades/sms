# services/db/trading_service.py
"""
Enhanced Trading Service - Goals, alerts, trade tracking, performance analysis
Focused on trading-related data management and analysis
Enhanced with custom exceptions, base service patterns, and comprehensive monitoring
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone, timedelta
from bson import ObjectId
from loguru import logger
import asyncio
import time
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, validator
from enum import Enum
import uuid

from services.db.base_db_service import BaseDBService


# Configuration for Trading Service
class TradingConfig:
    """Trading service configuration"""
    
    # Cache TTL values
    GOAL_CACHE_TTL = 86400  # 24 hours
    ALERT_CACHE_TTL = 1800  # 30 minutes
    TRADE_CACHE_TTL = 3600  # 1 hour
    PERFORMANCE_CACHE_TTL = 7200  # 2 hours
    
    # Processing limits
    MAX_ALERTS_PER_USER = 50
    MAX_GOALS_PER_USER = 20
    MAX_TRADE_MARKERS_PER_QUERY = 1000
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0
    MAX_RETRY_DELAY = 30.0
    
    # Performance thresholds
    MILESTONE_CHECK_INTERVAL = 100  # Check milestones every N goal updates
    ALERT_BATCH_SIZE = 20
    
    # Alert types
    SUPPORTED_ALERT_TYPES = [
        "price_above", "price_below", "volume_spike", 
        "technical_signal", "news_sentiment", "earnings"
    ]
    
    # Goal types
    SUPPORTED_GOAL_TYPES = [
        "savings", "investment", "retirement", "emergency_fund", 
        "house_down_payment", "education", "custom"
    ]


# Custom Exception Hierarchy
class TradingServiceException(Exception):
    """Base exception for trading service errors"""
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc)


class TradingValidationError(TradingServiceException):
    """Trading data validation errors"""
    pass


class TradingExecutionError(TradingServiceException):
    """Trading operation execution errors"""
    pass


class TradingDataError(TradingServiceException):
    """Trading data access errors"""
    pass


class AlertLimitExceededError(TradingServiceException):
    """Alert limit exceeded errors"""
    pass


class GoalLimitExceededError(TradingServiceException):
    """Goal limit exceeded errors"""
    pass


# Enhanced Data Models
class AlertType(str, Enum):
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    VOLUME_SPIKE = "volume_spike"
    TECHNICAL_SIGNAL = "technical_signal"
    NEWS_SENTIMENT = "news_sentiment"
    EARNINGS = "earnings"


class AlertStatus(str, Enum):
    ACTIVE = "active"
    TRIGGERED = "triggered"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class GoalType(str, Enum):
    SAVINGS = "savings"
    INVESTMENT = "investment"
    RETIREMENT = "retirement"
    EMERGENCY_FUND = "emergency_fund"
    HOUSE_DOWN_PAYMENT = "house_down_payment"
    EDUCATION = "education"
    CUSTOM = "custom"


class GoalStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class MarkerType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    WATCH = "watch"
    ANALYSIS = "analysis"


class EnhancedAlert(BaseModel):
    """Enhanced alert model with validation"""
    user_id: str = Field(..., min_length=1)
    alert_type: AlertType
    symbol: str = Field(..., min_length=1, max_length=10)
    condition: str = Field(..., min_length=1)
    target_value: float = Field(..., gt=0)
    status: AlertStatus = Field(AlertStatus.ACTIVE)
    notification_method: str = Field("sms", regex="^(sms|email|both)$")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    
    @validator('symbol')
    def symbol_uppercase(cls, v):
        return v.upper()
    
    @validator('expires_at')
    def expires_in_future(cls, v):
        if v and v <= datetime.now(timezone.utc):
            raise ValueError('Expiration date must be in the future')
        return v


class EnhancedGoal(BaseModel):
    """Enhanced goal model with validation"""
    user_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    target_amount: float = Field(..., gt=0)
    current_amount: float = Field(0, ge=0)
    target_date: Optional[datetime] = None
    goal_type: GoalType = Field(GoalType.SAVINGS)
    status: GoalStatus = Field(GoalStatus.ACTIVE)
    strategy: Dict[str, Any] = Field(default_factory=dict)
    milestones: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @validator('current_amount')
    def current_not_negative(cls, v):
        if v < 0:
            raise ValueError('Current amount cannot be negative')
        return v
    
    @validator('target_date')
    def target_date_future(cls, v):
        if v and v <= datetime.now(timezone.utc):
            raise ValueError('Target date must be in the future')
        return v
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self.target_amount <= 0:
            return 0
        return min((self.current_amount / self.target_amount) * 100, 100)


class EnhancedTradeMarker(BaseModel):
    """Enhanced trade marker model with validation"""
    user_id: str = Field(..., min_length=1)
    symbol: str = Field(..., min_length=1, max_length=10)
    marker_type: MarkerType
    entry_price: float = Field(..., gt=0)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    context: str = Field("", max_length=1000)
    user_sentiment: str = Field("neutral", regex="^(bullish|bearish|neutral)$")
    market_context: str = Field("", max_length=500)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    conversation_id: Optional[str] = None
    status: str = Field("active", regex="^(active|closed|cancelled)$")
    
    @validator('symbol')
    def symbol_uppercase(cls, v):
        return v.upper()


# Monitoring and Metrics
class TradingServiceMonitor:
    """Trading service monitoring and metrics collection"""
    
    def __init__(self):
        self.metrics = {}
        self.active_operations = {}
        self.start_time = time.time()
        self.operation_counts = {
            "goals_created": 0,
            "goals_updated": 0,
            "alerts_created": 0,
            "alerts_triggered": 0,
            "trades_recorded": 0,
            "milestones_achieved": 0
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


# Base Enhanced Service for Trading
class BaseTradingService:
    """Base class with common patterns for trading operations"""
    
    def __init__(self, base_service: BaseDBService, config: TradingConfig = None):
        self.base = base_service
        self.config = config or TradingConfig()
        self.monitor = TradingServiceMonitor()
        self._service_id = str(uuid.uuid4())
        self._operational_metrics = {
            "operations": 0,
            "errors": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def with_retry(self, operation, max_retries: int = None, base_delay: float = None):
        """Generic retry wrapper with exponential backoff"""
        max_retries = max_retries or self.config.MAX_RETRIES
        base_delay = base_delay or self.config.RETRY_BASE_DELAY
        
        for attempt in range(max_retries + 1):
            try:
                result = await operation()
                self._operational_metrics["operations"] += 1
                return result
            except Exception as e:
                if attempt == max_retries:
                    self._operational_metrics["errors"] += 1
                    raise TradingExecutionError(
                        f"Operation failed after {max_retries} retries: {str(e)}",
                        "MAX_RETRIES_EXCEEDED",
                        {"attempt": attempt, "max_retries": max_retries}
                    )
                
                delay = min(base_delay * (2 ** attempt), self.config.MAX_RETRY_DELAY)
                logger.warning(f"Trading operation retry {attempt + 1}/{max_retries} in {delay}s: {str(e)}")
                await asyncio.sleep(delay)
    
    async def cache_with_fallback(self, cache_key: str, fallback_func, ttl: int):
        """Standard cache-with-fallback pattern"""
        try:
            cached = await self.base.key_builder.get(cache_key)
            if cached:
                self._operational_metrics["cache_hits"] += 1
                return cached
        except Exception as e:
            logger.warning(f"Cache read failed for {cache_key}: {e}")
        
        self._operational_metrics["cache_misses"] += 1
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
            "operational_metrics": self._operational_metrics.copy(),
            "monitor_metrics": self.monitor.get_metrics(),
            "config": {
                "max_alerts_per_user": self.config.MAX_ALERTS_PER_USER,
                "max_goals_per_user": self.config.MAX_GOALS_PER_USER,
                "max_retries": self.config.MAX_RETRIES
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class TradingService(BaseTradingService):
    """Enhanced specialized service for trading data, goals, alerts, and performance"""
    
    def __init__(self, base_service: BaseDBService, config: TradingConfig = None):
        super().__init__(base_service, config)

    async def initialize(self):
        """Initialize trading-specific indexes"""
        try:
            # Goal management indexes
            await self.base.db.financial_goals.create_index([("user_id", 1), ("status", 1)])
            await self.base.db.financial_goals.create_index("target_date")
            await self.base.db.financial_goals.create_index("goal_type")
            await self.base.db.goal_milestones.create_index([("user_id", 1), ("goal_id", 1)])
            
            # Alert system indexes
            await self.base.db.user_alerts.create_index([("user_id", 1), ("status", 1)])
            await self.base.db.user_alerts.create_index("alert_type")
            await self.base.db.user_alerts.create_index("symbol")
            await self.base.db.user_alerts.create_index("expires_at")
            await self.base.db.alert_history.create_index([("user_id", 1), ("triggered_at", -1)])
            
            # Trade tracking indexes
            await self.base.db.trade_markers.create_index([("user_id", 1), ("symbol", 1), ("timestamp", -1)])
            await self.base.db.trade_markers.create_index("marker_type")
            await self.base.db.trade_markers.create_index("status")
            await self.base.db.trade_performance.create_index([("user_id", 1), ("symbol", 1)])
            
            # Legacy trading data indexes
            await self.base.db.trading_data.create_index("user_id")
            await self.base.db.trading_data.create_index("symbol")
            await self.base.db.trading_data.create_index("timestamp")
            
            logger.info("✅ Enhanced Trading service initialized")
        except Exception as e:
            logger.exception(f"❌ Enhanced Trading service initialization failed: {e}")
            raise TradingExecutionError(f"Failed to initialize trading service: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Health check for trading service"""
        try:
            async with self.monitor.track_operation("health_check"):
                goals_count = await self.base.db.financial_goals.count_documents({})
                alerts_count = await self.base.db.user_alerts.count_documents({"status": "active"})
                trades_count = await self.base.db.trade_markers.count_documents({})
                
                # Check for expired alerts
                expired_alerts = await self.base.db.user_alerts.count_documents({
                    "expires_at": {"$lt": datetime.now(timezone.utc)},
                    "status": "active"
                })
                
                return {
                    "status": "healthy",
                    "goals_count": goals_count,
                    "active_alerts": alerts_count,
                    "trade_markers": trades_count,
                    "expired_alerts": expired_alerts,
                    "service_metrics": self.get_service_metrics()
                }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    # Enhanced Goal Management Methods
    async def save_goal(self, user_id: str, goal_data: Dict) -> str:
        """Save financial goal with enhanced validation and milestone tracking"""
        if not user_id:
            raise TradingValidationError("User ID is required")
        
        try:
            # Validate goal data using Pydantic model
            enhanced_goal = EnhancedGoal(
                user_id=user_id,
                **goal_data
            )
            
            # Check user goal limit
            async with self.monitor.track_operation("check_goal_limit", {"user_id": user_id}):
                user_goal_count = await self.base.db.financial_goals.count_documents({
                    "user_id": user_id,
                    "status": {"$ne": "cancelled"}
                })
                
                if user_goal_count >= self.config.MAX_GOALS_PER_USER:
                    raise GoalLimitExceededError(
                        f"User has reached maximum goal limit of {self.config.MAX_GOALS_PER_USER}",
                        "GOAL_LIMIT_EXCEEDED"
                    )
            
            async with self.monitor.track_operation("save_goal", {"user_id": user_id, "goal_type": enhanced_goal.goal_type.value}):
                goal_doc = enhanced_goal.dict()
                goal_doc["created_at"] = datetime.now(timezone.utc)
                goal_doc["updated_at"] = datetime.now(timezone.utc)
                
                # Convert enums to strings for storage
                goal_doc["goal_type"] = enhanced_goal.goal_type.value
                goal_doc["status"] = enhanced_goal.status.value
                
                result = await self.base.db.financial_goals.insert_one(goal_doc)
                goal_id = str(result.inserted_id)
                
                # Cache the goal
                cache_key = f"goal:{user_id}:{goal_id}"
                goal_doc["_id"] = goal_id
                await self.base.key_builder.set(cache_key, goal_doc, ttl=self.config.GOAL_CACHE_TTL)
                
                # Invalidate user goals cache
                await self.base.key_builder.delete(f"goals:{user_id}")
                
                self.monitor.increment_counter("goals_created")
                logger.info(f"✅ Saved financial goal for user {user_id}: {enhanced_goal.title}")
                return goal_id
                
        except TradingServiceException:
            raise
        except Exception as e:
            logger.exception(f"❌ Error saving financial goal: {e}")
            raise TradingExecutionError(f"Failed to save goal: {e}")

    async def get_goals(self, user_id: str, include_completed: bool = False) -> List[Dict]:
        """Get all goals for user with enhanced filtering and caching"""
        if not user_id:
            raise TradingValidationError("User ID is required")
        
        try:
            cache_key = f"goals:{user_id}:{include_completed}"
            
            async def fetch_goals():
                query = {"user_id": user_id}
                if not include_completed:
                    query["status"] = {"$ne": "completed"}
                
                goals = await self.base.db.financial_goals.find(query).sort("created_at", -1).to_list(length=None)
                
                # Process goals
                for goal in goals:
                    goal["_id"] = str(goal["_id"])
                    
                    # Convert datetime fields to ISO format
                    for field in ["created_at", "updated_at", "target_date"]:
                        if isinstance(goal.get(field), datetime):
                            goal[field] = goal[field].isoformat()
                    
                    # Calculate progress percentage
                    target = goal.get("target_amount", 1)
                    current = goal.get("current_amount", 0)
                    goal["progress_percentage"] = min((current / target) * 100, 100) if target > 0 else 0
                    
                    # Calculate days remaining
                    if goal.get("target_date"):
                        try:
                            target_date = datetime.fromisoformat(goal["target_date"].replace('Z', '+00:00'))
                            days_remaining = (target_date - datetime.now(timezone.utc)).days
                            goal["days_remaining"] = max(0, days_remaining)
                        except:
                            goal["days_remaining"] = None
                
                return goals
            
            async with self.monitor.track_operation("get_goals", {"user_id": user_id}):
                goals = await self.cache_with_fallback(cache_key, fetch_goals, self.config.GOAL_CACHE_TTL)
                return goals
                
        except Exception as e:
            logger.exception(f"❌ Error getting user goals: {e}")
            raise TradingDataError(f"Failed to get goals: {e}")

    async def update_goal_progress(self, user_id: str, goal_id: str, new_amount: float) -> bool:
        """Update goal progress with enhanced milestone checking"""
        if not user_id or not goal_id:
            raise TradingValidationError("User ID and Goal ID are required")
        
        if new_amount < 0:
            raise TradingValidationError("Amount cannot be negative")
        
        try:
            async with self.monitor.track_operation("update_goal_progress", 
                                                  {"user_id": user_id, "goal_id": goal_id}):
                result = await self.base.db.financial_goals.update_one(
                    {"_id": ObjectId(goal_id), "user_id": user_id},
                    {
                        "$set": {
                            "current_amount": new_amount,
                            "updated_at": datetime.now(timezone.utc)
                        }
                    }
                )
                
                if result.modified_count > 0:
                    # Invalidate caches
                    await self.base.key_builder.delete(f"goals:{user_id}")
                    await self.base.key_builder.delete(f"goals:{user_id}:True")
                    await self.base.key_builder.delete(f"goal:{user_id}:{goal_id}")
                    
                    # Check for milestone achievements
                    await self._check_milestone_achievements(user_id, goal_id, new_amount)
                    
                    self.monitor.increment_counter("goals_updated")
                    return True
                
                return False
                
        except Exception as e:
            logger.exception(f"❌ Error updating goal progress: {e}")
            raise TradingExecutionError(f"Failed to update goal progress: {e}")

    # Enhanced Alert System Methods
    async def save_alert(self, user_id: str, alert_data: Dict) -> str:
        """Save user alert configuration with enhanced validation"""
        if not user_id:
            raise TradingValidationError("User ID is required")
        
        try:
            # Validate alert data using Pydantic model
            enhanced_alert = EnhancedAlert(
                user_id=user_id,
                **alert_data
            )
            
            # Check user alert limit
            async with self.monitor.track_operation("check_alert_limit", {"user_id": user_id}):
                user_alert_count = await self.base.db.user_alerts.count_documents({
                    "user_id": user_id,
                    "status": "active"
                })
                
                if user_alert_count >= self.config.MAX_ALERTS_PER_USER:
                    raise AlertLimitExceededError(
                        f"User has reached maximum alert limit of {self.config.MAX_ALERTS_PER_USER}",
                        "ALERT_LIMIT_EXCEEDED"
                    )
            
            async with self.monitor.track_operation("save_alert", 
                                                  {"user_id": user_id, "alert_type": enhanced_alert.alert_type.value}):
                alert_doc = enhanced_alert.dict()
                alert_doc["created_at"] = datetime.now(timezone.utc)
                alert_doc["last_checked"] = None
                alert_doc["trigger_count"] = 0
                
                # Convert enums to strings for storage
                alert_doc["alert_type"] = enhanced_alert.alert_type.value
                alert_doc["status"] = enhanced_alert.status.value
                
                result = await self.base.db.user_alerts.insert_one(alert_doc)
                alert_id = str(result.inserted_id)
                
                # Invalidate active alerts cache
                await self.base.key_builder.delete(f"alerts:active:{user_id}")
                
                self.monitor.increment_counter("alerts_created")
                logger.info(f"✅ Saved alert for user {user_id}: {enhanced_alert.symbol} {enhanced_alert.alert_type.value}")
                return alert_id
                
        except TradingServiceException:
            raise
        except Exception as e:
            logger.exception(f"❌ Error saving user alert: {e}")
            raise TradingExecutionError(f"Failed to save alert: {e}")

    async def get_active_alerts(self, user_id: str = None, symbol: str = None, alert_type: str = None) -> List[Dict]:
        """Get active alerts with enhanced filtering"""
        try:
            query = {"status": "active"}
            cache_key_parts = ["alerts", "active"]
            
            if user_id:
                query["user_id"] = user_id
                cache_key_parts.append(user_id)
            
            if symbol:
                query["symbol"] = symbol.upper()
                cache_key_parts.append(symbol.upper())
            
            if alert_type:
                if alert_type not in self.config.SUPPORTED_ALERT_TYPES:
                    raise TradingValidationError(f"Unsupported alert type: {alert_type}")
                query["alert_type"] = alert_type
                cache_key_parts.append(alert_type)
            
            cache_key = ":".join(cache_key_parts)
            
            async def fetch_alerts():
                # Add expiration filter
                query["$or"] = [
                    {"expires_at": {"$exists": False}},
                    {"expires_at": None},
                    {"expires_at": {"$gt": datetime.now(timezone.utc)}}
                ]
                
                alerts = await self.base.db.user_alerts.find(query).to_list(length=None)
                
                # Process alerts
                for alert in alerts:
                    alert["_id"] = str(alert["_id"])
                    for field in ["created_at", "last_checked", "expires_at"]:
                        if isinstance(alert.get(field), datetime):
                            alert[field] = alert[field].isoformat()
                
                return alerts
            
            async with self.monitor.track_operation("get_active_alerts", {"filters": len(query)}):
                alerts = await self.cache_with_fallback(cache_key, fetch_alerts, self.config.ALERT_CACHE_TTL)
                return alerts
                
        except Exception as e:
            logger.exception(f"❌ Error getting active alerts: {e}")
            raise TradingDataError(f"Failed to get alerts: {e}")

    async def record_alert_trigger(self, alert_id: str, trigger_data: Dict) -> bool:
        """Record alert trigger event with enhanced tracking"""
        if not alert_id:
            raise TradingValidationError("Alert ID is required")
        
        try:
            async with self.monitor.track_operation("record_alert_trigger", {"alert_id": alert_id}):
                now = datetime.now(timezone.utc)
                
                # Update alert
                await self.base.db.user_alerts.update_one(
                    {"_id": ObjectId(alert_id)},
                    {
                        "$set": {"last_checked": now},
                        "$inc": {"trigger_count": 1}
                    }
                )
                
                # Record trigger history
                history_doc = {
                    "alert_id": alert_id,
                    "user_id": trigger_data.get("user_id"),
                    "symbol": trigger_data.get("symbol"),
                    "alert_type": trigger_data.get("alert_type"),
                    "condition_met": trigger_data.get("condition_met"),
                    "current_value": trigger_data.get("current_value"),
                    "target_value": trigger_data.get("target_value"),
                    "triggered_at": now,
                    "notification_sent": trigger_data.get("notification_sent", False),
                    "notification_method": trigger_data.get("notification_method"),
                    "metadata": trigger_data.get("metadata", {})
                }
                
                await self.base.db.alert_history.insert_one(history_doc)
                
                # Invalidate relevant caches
                if trigger_data.get("user_id"):
                    await self.base.key_builder.delete(f"alerts:active:{trigger_data['user_id']}")
                
                self.monitor.increment_counter("alerts_triggered")
                logger.info(f"✅ Recorded alert trigger for alert {alert_id}")
                return True
                
        except Exception as e:
            logger.exception(f"❌ Error recording alert trigger: {e}")
            raise TradingExecutionError(f"Failed to record alert trigger: {e}")

    # Enhanced Trade Tracking Methods
    async def save_trade_marker(self, user_id: str, marker_data: Dict) -> str:
        """Save trade marker with enhanced validation and tracking"""
        if not user_id:
            raise TradingValidationError("User ID is required")
        
        try:
            # Validate marker data using Pydantic model
            enhanced_marker = EnhancedTradeMarker(
                user_id=user_id,
                **marker_data
            )
            
            async with self.monitor.track_operation("save_trade_marker", 
                                                  {"user_id": user_id, "symbol": enhanced_marker.symbol}):
                marker_doc = enhanced_marker.dict()
                marker_doc["created_at"] = datetime.now(timezone.utc)
                
                # Convert enums to strings for storage
                marker_doc["marker_type"] = enhanced_marker.marker_type.value
                
                result = await self.base.db.trade_markers.insert_one(marker_doc)
                marker_id = str(result.inserted_id)
                
                # Invalidate trade performance cache
                await self.base.key_builder.delete(f"trade_performance:{user_id}")
                await self.base.key_builder.delete(f"trade_performance:{user_id}:{enhanced_marker.symbol}")
                
                self.monitor.increment_counter("trades_recorded")
                logger.info(f"✅ Saved trade marker for {user_id}:{enhanced_marker.symbol} ({enhanced_marker.marker_type.value})")
                return marker_id
                
        except Exception as e:
            logger.exception(f"❌ Error saving trade marker: {e}")
            raise TradingExecutionError(f"Failed to save trade marker: {e}")

    async def get_trade_performance(self, user_id: str, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get trade performance data with enhanced filtering"""
        if not user_id:
            raise TradingValidationError("User ID is required")
        
        if limit <= 0 or limit > self.config.MAX_TRADE_MARKERS_PER_QUERY:
            raise TradingValidationError(f"Limit must be between 1 and {self.config.MAX_TRADE_MARKERS_PER_QUERY}")
        
        try:
            cache_key = f"trade_performance:{user_id}"
            if symbol:
                cache_key += f":{symbol.upper()}"
            cache_key += f":{limit}"
            
            async def fetch_trade_data():
                query = {"user_id": user_id}
                if symbol:
                    query["symbol"] = symbol.upper()
                
                markers = await self.base.db.trade_markers.find(query).sort("timestamp", -1).limit(limit).to_list(length=None)
                
                # Process markers
                for marker in markers:
                    marker["_id"] = str(marker["_id"])
                    if isinstance(marker.get("timestamp"), datetime):
                        marker["timestamp"] = marker["timestamp"].isoformat()
                    if isinstance(marker.get("created_at"), datetime):
                        marker["created_at"] = marker["created_at"].isoformat()
                
                return markers
            
            async with self.monitor.track_operation("get_trade_performance", {"user_id": user_id, "symbol": symbol}):
                markers = await self.cache_with_fallback(cache_key, fetch_trade_data, self.config.TRADE_CACHE_TTL)
                return markers
                
        except Exception as e:
            logger.exception(f"❌ Error getting trade performance: {e}")
            raise TradingDataError(f"Failed to get trade performance: {e}")

    async def get_trade_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive trade analytics for user"""
        if not user_id:
            raise TradingValidationError("User ID is required")
        
        if days <= 0 or days > 365:
            raise TradingValidationError("Days must be between 1 and 365")
        
        try:
            async with self.monitor.track_operation("get_trade_analytics", {"user_id": user_id, "days": days}):
                start_date = datetime.now(timezone.utc) - timedelta(days=days)
                
                # Get recent markers
                markers = await self.base.db.trade_markers.find({
                    "user_id": user_id,
                    "timestamp": {"$gte": start_date}
                }).to_list(length=None)
                
                # Calculate analytics
                analytics = {
                    "total_markers": len(markers),
                    "by_type": {},
                    "by_symbol": {},
                    "by_sentiment": {},
                    "average_confidence": 0,
                    "most_active_symbols": [],
                    "recent_activity": []
                }
                
                if markers:
                    # Count by type
                    for marker in markers:
                        marker_type = marker.get("marker_type", "unknown")
                        analytics["by_type"][marker_type] = analytics["by_type"].get(marker_type, 0) + 1
                        
                        symbol = marker.get("symbol", "unknown")
                        analytics["by_symbol"][symbol] = analytics["by_symbol"].get(symbol, 0) + 1
                        
                        sentiment = marker.get("user_sentiment", "neutral")
                        analytics["by_sentiment"][sentiment] = analytics["by_sentiment"].get(sentiment, 0) + 1
                    
                    # Calculate average confidence
                    confidences = [marker.get("confidence", 0) for marker in markers]
                    analytics["average_confidence"] = sum(confidences) / len(confidences)
                    
                    # Most active symbols
                    sorted_symbols = sorted(analytics["by_symbol"].items(), key=lambda x: x[1], reverse=True)
                    analytics["most_active_symbols"] = sorted_symbols[:10]
                    
                    # Recent activity (last 7 days)
                    recent_date = datetime.now(timezone.utc) - timedelta(days=7)
                    recent_markers = [m for m in markers if m.get("timestamp", datetime.min.replace(tzinfo=timezone.utc)) >= recent_date]
                    analytics["recent_activity"] = len(recent_markers)
                
                return analytics
                
        except Exception as e:
            logger.exception(f"❌ Error getting trade analytics: {e}")
            raise TradingDataError(f"Failed to get trade analytics: {e}")

    # Legacy Trading Data Methods (Enhanced)
    async def save_legacy_trading_data(self, user_id: str, symbol: str, data: Dict) -> str:
        """Legacy compatibility: Save trading analysis data with validation"""
        if not user_id or not symbol or not data:
            raise TradingValidationError("User ID, symbol, and data are required")
        
        try:
            async with self.monitor.track_operation("save_legacy_trading_data", {"user_id": user_id, "symbol": symbol}):
                trading_data = {
                    "user_id": user_id,
                    "symbol": symbol.upper(),
                    "data": data,
                    "timestamp": datetime.now(timezone.utc),
                    "created_at": datetime.now(timezone.utc)
                }
                
                result = await self.base.db.trading_data.insert_one(trading_data)
                return str(result.inserted_id)
                
        except Exception as e:
            logger.exception(f"❌ Error saving legacy trading data: {e}")
            raise TradingExecutionError(f"Failed to save legacy trading data: {e}")

    async def get_legacy_trading_data(self, user_id: str, symbol: str = None, limit: int = 10) -> List[Dict]:
        """Legacy compatibility: Get trading data for user with validation"""
        if not user_id:
            raise TradingValidationError("User ID is required")
        
        if limit <= 0 or limit > 100:
            raise TradingValidationError("Limit must be between 1 and 100")
        
        try:
            async with self.monitor.track_operation("get_legacy_trading_data", {"user_id": user_id, "symbol": symbol}):
                query = {"user_id": user_id}
                if symbol:
                    query["symbol"] = symbol.upper()
                
                data = await self.base.db.trading_data.find(query).sort("timestamp", -1).limit(limit).to_list(length=limit)
                
                for item in data:
                    item['_id'] = str(item['_id'])
                    if isinstance(item.get("timestamp"), datetime):
                        item["timestamp"] = item["timestamp"].isoformat()
                    if isinstance(item.get("created_at"), datetime):
                        item["created_at"] = item["created_at"].isoformat()
                
                return data
                
        except Exception as e:
            logger.exception(f"❌ Error getting legacy trading data: {e}")
            raise TradingDataError(f"Failed to get legacy trading data: {e}")

    # Enhanced Helper Methods
    async def _check_milestone_achievements(self, user_id: str, goal_id: str, new_amount: float):
        """Enhanced milestone checking with detailed tracking"""
        try:
            async with self.monitor.track_operation("check_milestones", {"user_id": user_id, "goal_id": goal_id}):
                # Get goal with milestones
                goal = await self.base.db.financial_goals.find_one({"_id": ObjectId(goal_id)})
                if not goal:
                    return
                
                target_amount = goal.get("target_amount", 0)
                milestones = goal.get("milestones", [])
                
                if target_amount > 0:
                    progress_percentage = (new_amount / target_amount) * 100
                    
                    # Check for milestone achievements
                    for milestone in milestones:
                        milestone_percentage = milestone.get("percentage", 0)
                        if progress_percentage >= milestone_percentage and not milestone.get("achieved", False):
                            # Record milestone achievement
                            milestone_doc = {
                                "user_id": user_id,
                                "goal_id": goal_id,
                                "milestone_name": milestone.get("name", f"{milestone_percentage}% milestone"),
                                "milestone_percentage": milestone_percentage,
                                "achieved_at": datetime.now(timezone.utc),
                                "current_amount": new_amount,
                                "target_amount": target_amount,
                                "progress_percentage": progress_percentage
                            }
                            
                            await self.base.db.goal_milestones.insert_one(milestone_doc)
                            
                            # Mark milestone as achieved in goal
                            await self.base.db.financial_goals.update_one(
                                {"_id": ObjectId(goal_id), "milestones.percentage": milestone_percentage},
                                {
                                    "$set": {
                                        "milestones.$.achieved": True,
                                        "milestones.$.achieved_at": datetime.now(timezone.utc)
                                    }
                                }
                            )
                            
                            self.monitor.increment_counter("milestones_achieved")
                            logger.info(f"🎉 Milestone achieved: {user_id} reached {milestone_percentage}% of goal {goal_id}")
        
        except Exception as e:
            logger.exception(f"❌ Error checking milestone achievements: {e}")

    async def cleanup_expired_alerts(self) -> Dict[str, Any]:
        """Cleanup expired alerts and return statistics"""
        try:
            async with self.monitor.track_operation("cleanup_expired_alerts"):
                now = datetime.now(timezone.utc)
                
                # Find expired alerts
                expired_alerts = await self.base.db.user_alerts.find({
                    "expires_at": {"$lt": now},
                    "status": "active"
                }).to_list(length=None)
                
                # Update status to expired
                result = await self.base.db.user_alerts.update_many(
                    {
                        "expires_at": {"$lt": now},
                        "status": "active"
                    },
                    {
                        "$set": {
                            "status": "expired",
                            "updated_at": now
                        }
                    }
                )
                
                # Invalidate caches for affected users
                affected_users = set()
                for alert in expired_alerts:
                    user_id = alert.get("user_id")
                    if user_id:
                        affected_users.add(user_id)
                        await self.base.key_builder.delete(f"alerts:active:{user_id}")
                
                cleanup_stats = {
                    "expired_alerts_count": result.modified_count,
                    "affected_users": len(affected_users),
                    "cleanup_timestamp": now.isoformat()
                }
                
                if result.modified_count > 0:
                    logger.info(f"🧹 Cleaned up {result.modified_count} expired alerts for {len(affected_users)} users")
                
                return cleanup_stats
                
        except Exception as e:
            logger.exception(f"❌ Error cleaning up expired alerts: {e}")
            raise TradingExecutionError(f"Failed to cleanup expired alerts: {e}")


# Test utilities for enhanced testing
class TradingServiceTestHooks:
    """Test hooks for trading service testing"""
    
    @staticmethod
    def create_test_trading_service(mock_db: bool = True, config: TradingConfig = None):
        """Create trading service with test doubles"""
        from unittest.mock import MagicMock
        
        # Mock base service
        base_service = MagicMock()
        
        if mock_db:
            base_service.db = MagicMock()
            base_service.db.financial_goals = MagicMock()
            base_service.db.goal_milestones = MagicMock()
            base_service.db.user_alerts = MagicMock()
            base_service.db.alert_history = MagicMock()
            base_service.db.trade_markers = MagicMock()
            base_service.db.trade_performance = MagicMock()
            base_service.db.trading_data = MagicMock()
            
            base_service.key_builder = MagicMock()
        
        # Use test config if not provided
        test_config = config or TradingConfig()
        test_config.MAX_GOALS_PER_USER = 5
        test_config.MAX_ALERTS_PER_USER = 10
        
        service = TradingService(base_service, test_config)
        return service
    
    @staticmethod
    def create_test_goal_data(**overrides) -> Dict[str, Any]:
        """Create test goal data"""
        base_data = {
            "title": "Test Goal",
            "description": "Test goal description",
            "target_amount": 10000.0,
            "current_amount": 0.0,
            "goal_type": "savings",
            "status": "active"
        }
        base_data.update(overrides)
        return base_data
    
    @staticmethod
    def create_test_alert_data(**overrides) -> Dict[str, Any]:
        """Create test alert data"""
        base_data = {
            "alert_type": "price_above",
            "symbol": "AAPL",
            "condition": "price > 150",
            "target_value": 150.0,
            "notification_method": "sms"
        }
        base_data.update(overrides)
        return base_data
    
    @staticmethod
    def create_test_marker_data(**overrides) -> Dict[str, Any]:
        """Create test trade marker data"""
        base_data = {
            "symbol": "AAPL",
            "marker_type": "buy",
            "entry_price": 150.0,
            "confidence": 0.8,
            "context": "Test trade marker",
            "user_sentiment": "bullish"
        }
        base_data.update(overrides)
        return base_data


# Export the enhanced trading service
__all__ = [
    'TradingService',
    'TradingConfig',
    'TradingServiceException',
    'TradingValidationError',
    'TradingExecutionError',
    'TradingDataError',
    'AlertLimitExceededError',
    'GoalLimitExceededError',
    'EnhancedAlert',
    'EnhancedGoal',
    'EnhancedTradeMarker',
    'AlertType',
    'AlertStatus',
    'GoalType',
    'GoalStatus',
    'MarkerType',
    'TradingServiceTestHooks'
]
