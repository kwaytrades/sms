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
                        "update
