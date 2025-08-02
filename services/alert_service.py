# services/alert_service.py - Production Alert Service
"""
Production Alert Service for Real-time Price and Technical Alerts
Handles price alerts, technical indicators, portfolio alerts, and news alerts
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone, timedelta
from enum import Enum
from dataclasses import dataclass, asdict

import aiohttp
from loguru import logger

from config import settings
from services.notification_service import NotificationService


class AlertType(Enum):
    """Types of alerts"""
    PRICE = "price"
    TECHNICAL = "technical"
    PORTFOLIO = "portfolio"
    NEWS = "news"
    VOLUME = "volume"
    EARNINGS = "earnings"


class AlertCondition(Enum):
    """Alert conditions"""
    ABOVE = "above"
    BELOW = "below"
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"
    PERCENTAGE_CHANGE = "percentage_change"
    CUSTOM = "custom"


class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    TRIGGERED = "triggered"
    EXPIRED = "expired"
    DISABLED = "disabled"
    DELETED = "deleted"


class AlertFrequency(Enum):
    """Alert frequency"""
    ONCE = "once"
    DAILY = "daily"
    RECURRING = "recurring"


@dataclass
class PriceAlert:
    """Price alert configuration"""
    alert_id: str
    user_id: str
    symbol: str
    condition: AlertCondition
    target_price: float
    current_price: float
    created_at: datetime
    expires_at: Optional[datetime]
    frequency: AlertFrequency
    status: AlertStatus
    triggered_count: int = 0
    last_triggered: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['expires_at'] = self.expires_at.isoformat() if self.expires_at else None
        data['last_triggered'] = self.last_triggered.isoformat() if self.last_triggered else None
        data['condition'] = self.condition.value
        data['frequency'] = self.frequency.value
        data['status'] = self.status.value
        return data


@dataclass
class TechnicalAlert:
    """Technical indicator alert"""
    alert_id: str
    user_id: str
    symbol: str
    indicator: str
    condition: str
    threshold: float
    timeframe: str
    created_at: datetime
    expires_at: Optional[datetime]
    status: AlertStatus
    triggered_count: int = 0
    last_triggered: Optional[datetime] = None
    metadata: Dict[str, Any] = None


class AlertService:
    """
    Production Alert Service
    
    Features:
    - Real-time price monitoring
    - Technical indicator alerts (RSI, MACD, etc.)
    - Portfolio-based alerts
    - News sentiment alerts
    - Volume spike detection
    - Earnings calendar alerts
    - Smart delivery timing
    - Rate limiting and batching
    - Historical alert analytics
    """
    
    def __init__(self, base_service):
        self.base_service = base_service
        self.notification_service = None
        
        # Configuration
        self.check_interval = 60  # Check alerts every minute
        self.batch_size = 100     # Process alerts in batches
        self.max_alerts_per_user = 50
        self.default_expiry_hours = 24 * 7  # 1 week
        
        # Collections
        self.alerts_collection = "alerts"
        self.alert_history_collection = "alert_history"
        self.price_cache_collection = "price_cache"
        
        # Monitoring
        self.last_check_time = None
        self.alerts_processed = 0
        self.alerts_triggered = 0
        
        # Technical indicators cache
        self.indicator_cache = {}
        self.indicator_cache_ttl = 300  # 5 minutes
        
        logger.info("🚨 AlertService initialized")

    async def initialize(self):
        """Initialize alert service"""
        try:
            # Initialize notification service
            self.notification_service = NotificationService(self.base_service)
            await self.notification_service.initialize()
            
            # Ensure collections and indexes
            await self._ensure_collections()
            
            # Start background alert processor
            asyncio.create_task(self._alert_processor_loop())
            
            logger.info("✅ AlertService initialized successfully")
            
        except Exception as e:
            logger.exception(f"❌ AlertService initialization failed: {e}")
            raise

    async def _ensure_collections(self):
        """Ensure MongoDB collections with proper indexes"""
        try:
            db = self.base_service.db
            
            # Alerts collection
            alerts = db[self.alerts_collection]
            await alerts.create_index([("user_id", 1), ("status", 1)])
            await alerts.create_index([("symbol", 1), ("status", 1)])
            await alerts.create_index([("alert_type", 1), ("status", 1)])
            await alerts.create_index([("expires_at", 1)])
            await alerts.create_index([("created_at", -1)])
            
            # Alert history collection
            history = db[self.alert_history_collection]
            await history.create_index([("user_id", 1), ("triggered_at", -1)])
            await history.create_index([("symbol", 1), ("triggered_at", -1)])
            await history.create_index([("alert_id", 1)])
            
            # Price cache collection
            cache = db[self.price_cache_collection]
            await cache.create_index([("symbol", 1)], unique=True)
            await cache.create_index([("updated_at", 1)])
            
            logger.info("✅ Alert service collections ensured")
            
        except Exception as e:
            logger.exception(f"❌ Error ensuring collections: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Health check for alert service"""
        health = {"status": "healthy", "components": {}}
        
        try:
            # Check notification service
            notification_health = await self.notification_service.health_check()
            health["components"]["notifications"] = notification_health
            
            if notification_health.get("status") != "healthy":
                health["status"] = "degraded"
            
            # Get alert statistics
            stats = await self.get_alert_stats()
            health["statistics"] = stats
            
            # Check if alert processor is running
            health["components"]["processor"] = {
                "status": "healthy" if self.last_check_time else "unknown",
                "last_check": self.last_check_time.isoformat() if self.last_check_time else None,
                "alerts_processed": self.alerts_processed,
                "alerts_triggered": self.alerts_triggered
            }
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        return health

    # ==========================================
    # PRICE ALERTS
    # ==========================================

    async def create_price_alert(self, user_id: str, symbol: str,
                               target_price: float, condition: str,
                               expires_in_hours: int = None) -> str:
        """Create price alert"""
        try:
            # Validate inputs
            symbol = symbol.upper()
            
            if condition not in [c.value for c in AlertCondition]:
                raise ValueError(f"Invalid condition: {condition}")
            
            # Check user alert limit
            user_alert_count = await self._get_user_alert_count(user_id)
            if user_alert_count >= self.max_alerts_per_user:
                raise ValueError(f"Maximum {self.max_alerts_per_user} alerts per user")
            
            # Get current price for validation
            current_price = await self._get_current_price(symbol)
            if not current_price:
                raise ValueError(f"Could not get current price for {symbol}")
            
            # Calculate expiry
            expires_at = None
            if expires_in_hours:
                expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
            else:
                expires_at = datetime.utcnow() + timedelta(hours=self.default_expiry_hours)
            
            # Create alert
            alert_id = f"{user_id}_{symbol}_{int(time.time())}"
            
            alert = PriceAlert(
                alert_id=alert_id,
                user_id=user_id,
                symbol=symbol,
                condition=AlertCondition(condition),
                target_price=target_price,
                current_price=current_price,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                frequency=AlertFrequency.ONCE,
                status=AlertStatus.ACTIVE,
                metadata={
                    "alert_type": AlertType.PRICE.value,
                    "created_price": current_price
                }
            )
            
            # Save to database
            collection = self.base_service.db[self.alerts_collection]
            alert_doc = alert.to_dict()
            result = await collection.insert_one(alert_doc)
            
            if result.inserted_id:
                logger.info(f"✅ Price alert created: {alert_id} for {symbol} @ ${target_price}")
                return alert_id
            else:
                raise Exception("Failed to save alert to database")
                
        except Exception as e:
            logger.exception(f"❌ Error creating price alert: {e}")
            raise

    async def create_technical_alert(self, user_id: str, symbol: str,
                                   indicator: str, conditions: Dict) -> str:
        """Create technical indicator alert"""
        try:
            symbol = symbol.upper()
            
            # Validate indicator
            supported_indicators = ["RSI", "MACD", "SMA", "EMA", "BOLLINGER", "STOCH"]
            if indicator.upper() not in supported_indicators:
                raise ValueError(f"Unsupported indicator: {indicator}")
            
            # Check user alert limit
            user_alert_count = await self._get_user_alert_count(user_id)
            if user_alert_count >= self.max_alerts_per_user:
                raise ValueError(f"Maximum {self.max_alerts_per_user} alerts per user")
            
            # Create alert
            alert_id = f"{user_id}_{symbol}_{indicator}_{int(time.time())}"
            
            expires_at = datetime.utcnow() + timedelta(hours=self.default_expiry_hours)
            
            alert_doc = {
                "alert_id": alert_id,
                "user_id": user_id,
                "symbol": symbol,
                "alert_type": AlertType.TECHNICAL.value,
                "indicator": indicator.upper(),
                "conditions": conditions,
                "status": AlertStatus.ACTIVE.value,
                "created_at": datetime.utcnow(),
                "expires_at": expires_at,
                "triggered_count": 0,
                "metadata": {
                    "timeframe": conditions.get("timeframe", "1D"),
                    "threshold": conditions.get("threshold", 0)
                }
            }
            
            # Save to database
            collection = self.base_service.db[self.alerts_collection]
            result = await collection.insert_one(alert_doc)
            
            if result.inserted_id:
                logger.info(f"✅ Technical alert created: {alert_id} for {symbol} {indicator}")
                return alert_id
            else:
                raise Exception("Failed to save technical alert")
                
        except Exception as e:
            logger.exception(f"❌ Error creating technical alert: {e}")
            raise

    # ==========================================
    # ALERT MANAGEMENT
    # ==========================================

    async def get_user_alerts(self, user_id: str, active_only: bool = True) -> List[Dict]:
        """Get user's alerts"""
        try:
            collection = self.base_service.db[self.alerts_collection]
            
            query = {"user_id": user_id}
            if active_only:
                query["status"] = AlertStatus.ACTIVE.value
            
            cursor = collection.find(query).sort("created_at", -1)
            alerts = await cursor.to_list(length=None)
            
            # Convert ObjectId to string
            for alert in alerts:
                alert["_id"] = str(alert["_id"])
            
            return alerts
            
        except Exception as e:
            logger.exception(f"❌ Error getting alerts for {user_id}: {e}")
            return []

    async def update_alert_status(self, alert_id: str, status: str, 
                                metadata: Dict = None) -> bool:
        """Update alert status"""
        try:
            if status not in [s.value for s in AlertStatus]:
                raise ValueError(f"Invalid status: {status}")
            
            update_doc = {
                "status": status,
                "updated_at": datetime.utcnow()
            }
            
            if metadata:
                update_doc["metadata"] = metadata
            
            collection = self.base_service.db[self.alerts_collection]
            result = await collection.update_one(
                {"alert_id": alert_id},
                {"$set": update_doc}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.exception(f"❌ Error updating alert status: {e}")
            return False

    async def delete_alert(self, alert_id: str, user_id: str) -> bool:
        """Delete/disable alert"""
        try:
            collection = self.base_service.db[self.alerts_collection]
            
            result = await collection.update_one(
                {"alert_id": alert_id, "user_id": user_id},
                {
                    "$set": {
                        "status": AlertStatus.DELETED.value,
                        "deleted_at": datetime.utcnow()
                    }
                }
            )
            
            if result.modified_count
