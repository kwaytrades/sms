# services/db/trading_service.py
"""
Trading Service - Goals, alerts, trade tracking, performance analysis
Focused on trading-related data management and analysis
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from bson import ObjectId
from loguru import logger

from services.db.base_db_service import BaseDBService


class TradingService:
    """Specialized service for trading data, goals, alerts, and performance"""
    
    def __init__(self, base_service: BaseDBService):
        self.base = base_service

    async def initialize(self):
        """Initialize trading-specific indexes"""
        try:
            # Goal management indexes
            await self.base.db.financial_goals.create_index([("user_id", 1), ("status", 1)])
            await self.base.db.financial_goals.create_index("target_date")
            await self.base.db.goal_milestones.create_index([("user_id", 1), ("goal_id", 1)])
            
            # Alert system indexes
            await self.base.db.user_alerts.create_index([("user_id", 1), ("status", 1)])
            await self.base.db.user_alerts.create_index("alert_type")
            await self.base.db.alert_history.create_index([("user_id", 1), ("triggered_at", -1)])
            
            # Trade tracking indexes
            await self.base.db.trade_markers.create_index([("user_id", 1), ("symbol", 1), ("timestamp", -1)])
            await self.base.db.trade_markers.create_index("marker_type")
            await self.base.db.trade_performance.create_index([("user_id", 1), ("symbol", 1)])
            
            # Legacy trading data indexes
            await self.base.db.trading_data.create_index("user_id")
            await self.base.db.trading_data.create_index("symbol")
            await self.base.db.trading_data.create_index("timestamp")
            
            logger.info("✅ Trading service initialized")
        except Exception as e:
            logger.exception(f"❌ Trading service initialization failed: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Health check for trading service"""
        try:
            goals_count = await self.base.db.financial_goals.count_documents({})
            alerts_count = await self.base.db.user_alerts.count_documents({"status": "active"})
            trades_count = await self.base.db.trade_markers.count_documents({})
            
            return {
                "status": "healthy",
                "goals_count": goals_count,
                "active_alerts": alerts_count,
                "trade_markers": trades_count
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    # Goal Management Methods
    async def save_goal(self, user_id: str, goal_data: Dict) -> str:
        """Save financial goal with milestone tracking"""
        try:
            goal_doc = {
                "user_id": user_id,
                "title": goal_data.get("title"),
                "description": goal_data.get("description"),
                "target_amount": goal_data.get("target_amount"),
                "current_amount": goal_data.get("current_amount", 0),
                "target_date": goal_data.get("target_date"),
                "goal_type": goal_data.get("goal_type", "savings"),
                "status": goal_data.get("status", "active"),
                "strategy": goal_data.get("strategy", {}),
                "milestones": goal_data.get("milestones", []),
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            
            result = await self.base.db.financial_goals.insert_one(goal_doc)
            goal_id = str(result.inserted_id)
            
            # Cache the goal
            cache_key = f"goal:{user_id}:{goal_id}"
            goal_doc["_id"] = goal_id
            await self.base.key_builder.set(cache_key, goal_doc, ttl=86400)
            
            logger.info(f"✅ Saved financial goal for user {user_id}")
            return goal_id
            
        except Exception as e:
            logger.exception(f"❌ Error saving financial goal: {e}")
            raise

    async def get_goals(self, user_id: str) -> List[Dict]:
        """Get all goals for user with progress calculation"""
        try:
            cache_key = f"goals:{user_id}"
            cached_goals = await self.base.key_builder.get(cache_key)
            if cached_goals:
                return cached_goals
            
            goals = await self.base.db.financial_goals.find(
                {"user_id": user_id}
            ).sort("created_at", -1).to_list(length=None)
            
            # Process goals
            for goal in goals:
                goal["_id"] = str(goal["_id"])
                if isinstance(goal.get("created_at"), datetime):
                    goal["created_at"] = goal["created_at"].isoformat()
                if isinstance(goal.get("updated_at"), datetime):
                    goal["updated_at"] = goal["updated_at"].isoformat()
                if isinstance(goal.get("target_date"), datetime):
                    goal["target_date"] = goal["target_date"].isoformat()
                
                # Calculate progress percentage
                target = goal.get("target_amount", 1)
                current = goal.get("current_amount", 0)
                goal["progress_percentage"] = min((current / target) * 100, 100) if target > 0 else 0
            
            # Cache for 1 hour
            await self.base.key_builder.set(cache_key, goals, ttl=3600)
            
            return goals
            
        except Exception as e:
            logger.exception(f"❌ Error getting user goals: {e}")
            return []

    async def update_goal_progress(self, user_id: str, goal_id: str, new_amount: float) -> bool:
        """Update goal progress and check milestones"""
        try:
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
                await self.base.key_builder.delete(f"goal:{user_id}:{goal_id}")
                
                # Check for milestone achievements
                await self._check_milestone_achievements(user_id, goal_id, new_amount)
                
                return True
            
            return False
            
        except Exception as e:
            logger.exception(f"❌ Error updating goal progress: {e}")
            return False

    # Alert System Methods
    async def save_alert(self, user_id: str, alert_data: Dict) -> str:
        """Save user alert configuration"""
        try:
            alert_doc = {
                "user_id": user_id,
                "alert_type": alert_data.get("alert_type"),
                "symbol": alert_data.get("symbol"),
                "condition": alert_data.get("condition"),
                "target_value": alert_data.get("target_value"),
                "status": alert_data.get("status", "active"),
                "notification_method": alert_data.get("notification_method", "sms"),
                "created_at": datetime.now(timezone.utc),
                "last_checked": None,
                "trigger_count": 0
            }
            
            result = await self.base.db.user_alerts.insert_one(alert_doc)
            alert_id = str(result.inserted_id)
            
            # Invalidate active alerts cache
            await self.base.key_builder.delete(f"alerts:active:{user_id}")
            
            logger.info(f"✅ Saved alert for user {user_id}")
            return alert_id
            
        except Exception as e:
            logger.exception(f"❌ Error saving user alert: {e}")
            raise

    async def get_active_alerts(self, user_id: str = None) -> List[Dict]:
        """Get active alerts for user or all users"""
        try:
            query = {"status": "active"}
            if user_id:
                query["user_id"] = user_id
                
                # Check cache first for user-specific alerts
                cache_key = f"alerts:active:{user_id}"
                cached_alerts = await self.base.key_builder.get(cache_key)
                if cached_alerts:
                    return cached_alerts
            
            alerts = await self.base.db.user_alerts.find(query).to_list(length=None)
            
            # Process alerts
            for alert in alerts:
                alert["_id"] = str(alert["_id"])
                if isinstance(alert.get("created_at"), datetime):
                    alert["created_at"] = alert["created_at"].isoformat()
                if isinstance(alert.get("last_checked"), datetime):
                    alert["last_checked"] = alert["last_checked"].isoformat()
            
            # Cache user-specific alerts
            if user_id:
                await self.base.key_builder.set(cache_key, alerts, ttl=1800)  # 30 minutes
            
            return alerts
            
        except Exception as e:
            logger.exception(f"❌ Error getting active alerts: {e}")
            return []

    async def record_alert_trigger(self, alert_id: str, trigger_data: Dict) -> bool:
        """Record alert trigger event"""
        try:
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
                "condition_met": trigger_data.get("condition_met"),
                "current_value": trigger_data.get("current_value"),
                "target_value": trigger_data.get("target_value"),
                "triggered_at": now,
                "notification_sent": trigger_data.get("notification_sent", False)
            }
            
            await self.base.db.alert_history.insert_one(history_doc)
            
            # Invalidate relevant caches
            if trigger_data.get("user_id"):
                await self.base.key_builder.delete(f"alerts:active:{trigger_data['user_id']}")
            
            logger.info(f"✅ Recorded alert trigger for alert {alert_id}")
            return True
            
        except Exception as e:
            logger.exception(f"❌ Error recording alert trigger: {e}")
            return False

    # Trade Tracking Methods
    async def save_trade_marker(self, user_id: str, marker_data: Dict) -> str:
        """Save trade marker for performance tracking"""
        try:
            marker_doc = {
                "user_id": user_id,
                "symbol": marker_data.get("symbol"),
                "marker_type": marker_data.get("marker_type"),
                "entry_price": marker_data.get("entry_price"),
                "confidence": marker_data.get("confidence", 0.5),
                "context": marker_data.get("context", ""),
                "user_sentiment": marker_data.get("user_sentiment", "neutral"),
                "market_context": marker_data.get("market_context", ""),
                "timestamp": datetime.now(timezone.utc),
                "conversation_id": marker_data.get("conversation_id"),
                "status": "active"
            }
            
            result = await self.base.db.trade_markers.insert_one(marker_doc)
            marker_id = str(result.inserted_id)
            
            logger.info(f"✅ Saved trade marker for {user_id}:{marker_data.get('symbol')}")
            return marker_id
            
        except Exception as e:
            logger.exception(f"❌ Error saving trade marker: {e}")
            raise

    async def get_trade_performance(self, user_id: str, symbol: str = None) -> List[Dict]:
        """Get trade performance data for user"""
        try:
            query = {"user_id": user_id}
            if symbol:
                query["symbol"] = symbol
            
            markers = await self.base.db.trade_markers.find(query).sort("timestamp", -1).to_list(length=None)
            
            # Process markers
            for marker in markers:
                marker["_id"] = str(marker["_id"])
                if isinstance(marker.get("timestamp"), datetime):
                    marker["timestamp"] = marker["timestamp"].isoformat()
            
            return markers
            
        except Exception as e:
            logger.exception(f"❌ Error getting trade performance: {e}")
            return []

    # Legacy Trading Data Methods
    async def save_legacy_trading_data(self, user_id: str, symbol: str, data: Dict) -> str:
        """Legacy compatibility: Save trading analysis data"""
        try:
            trading_data = {
                "user_id": user_id,
                "symbol": symbol,
                "data": data,
                "timestamp": datetime.utcnow()
            }
            
            result = await self.base.db.trading_data.insert_one(trading_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.exception(f"❌ Error saving legacy trading data: {e}")
            raise

    async def get_legacy_trading_data(self, user_id: str, symbol: str = None, limit: int = 10) -> List[Dict]:
        """Legacy compatibility: Get trading data for user"""
        try:
            query = {"user_id": user_id}
            if symbol:
                query["symbol"] = symbol
            
            data = await self.base.db.trading_data.find(query).sort("timestamp", -1).limit(limit).to_list(length=limit)
            
            for item in data:
                item['_id'] = str(item['_id'])
            
            return data
        except Exception as e:
            logger.exception(f"❌ Error getting legacy trading data: {e}")
            return []

    async def _check_milestone_achievements(self, user_id: str, goal_id: str, new_amount: float):
        """Check and record milestone achievements"""
        try:
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
                            "target_amount": target_amount
                        }
                        
                        await self.base.db.goal_milestones.insert_one(milestone_doc)
                        
                        # Mark milestone as achieved in goal
                        await self.base.db.financial_goals.update_one(
                            {"_id": ObjectId(goal_id), "milestones.percentage": milestone_percentage},
                            {"$set": {"milestones.$.achieved": True, "milestones.$.achieved_at": datetime.now(timezone.utc)}}
                        )
                        
                        logger.info(f"🎉 Milestone achieved: {user_id} reached {milestone_percentage}% of goal {goal_id}")
        
        except Exception as e:
            logger.exception(f"❌ Error checking milestone achievements: {e}")
