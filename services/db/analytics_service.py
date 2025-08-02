services/db/analytics_service.py
"""
Analytics Service - User analytics, system metrics, performance tracking
Focused on data analysis, reporting, and business intelligence
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from loguru import logger

from services.db.base_db_service import BaseDBService


class AnalyticsService:
    """Specialized service for analytics and metrics"""
    
    def __init__(self, base_service: BaseDBService):
        self.base = base_service

    async def initialize(self):
        """Initialize analytics-specific indexes"""
        try:
            # User analytics indexes
            await self.base.db.user_analytics.create_index([("user_id", 1), ("date", -1)])
            await self.base.db.user_analytics.create_index("metric_type")
            
            # Performance tracking indexes
            await self.base.db.performance_metrics.create_index([("timestamp", -1)])
            await self.base.db.performance_metrics.create_index("metric_name")
            
            logger.info("✅ Analytics service initialized")
        except Exception as e:
            logger.exception(f"❌ Analytics service initialization failed: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Health check for analytics service"""
        try:
            analytics_count = await self.base.db.user_analytics.count_documents({})
            metrics_count = await self.base.db.performance_metrics.count_documents({})
            
            return {
                "status": "healthy",
                "analytics_records": analytics_count,
                "performance_metrics": metrics_count
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def get_user_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive user analytics"""
        try:
            cache_key = f"analytics:{user_id}:{days}"
            cached_analytics = await self.base.key_builder.get(cache_key)
            if cached_analytics:
                return cached_analytics
            
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Get user from users service (we'll need to import or inject this)
            user_doc = await self.base.db.users.find_one({"_id": ObjectId(user_id) if len(user_id) == 24 else None, "user_id": user_id})
            phone_number = user_doc.get("phone_number") if user_doc else ""
            
            # Message analytics
            message_count = await self.base.db.enhanced_conversations.count_documents({
                "phone_number": phone_number,
                "timestamp": {"$gte": start_date}
            }) if phone_number else 0
            
            # Symbol mentions
            symbol_pipeline = [
                {"$match": {
                    "phone_number": phone_number,
                    "timestamp": {"$gte": start_date}
                }},
                {"$unwind": "$symbols_mentioned"},
                {"$group": {
                    "_id": "$symbols_mentioned",
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}},
                {"$limit": 10}
            ] if phone_number else []
            
            top_symbols = await self.base.db.enhanced_conversations.aggregate(symbol_pipeline).to_list(length=None) if symbol_pipeline else []
            
            # Goals analytics
            goals_pipeline = [
                {"$match": {"user_id": user_id}},
                {"$group": {
                    "_id": "$status",
                    "count": {"$sum": 1}
                }}
            ]
            goals_by_status = await self.base.db.financial_goals.aggregate(goals_pipeline).to_list(length=None)
            
            # Usage analytics
            usage_stats = {}
            periods = ["week", "month", "day"]
            for period in periods:
                key = f"usage:{user_id}:{period}"
                usage_data = await self.base.key_builder.get(key)
                usage_stats[period] = usage_data.get("count", 0) if usage_data else 0
            
            analytics = {
                "user_id": user_id,
                "period_days": days,
                "message_count": message_count,
                "top_symbols": top_symbols,
                "goals_by_status": {item["_id"]: item["count"] for item in goals_by_status},
                "usage_stats": usage_stats,
                "user_profile": {
                    "plan_type": user_doc.get("plan_type", "unknown") if user_doc else "unknown",
                    "created_at": user_doc.get("created_at").isoformat() if user_doc and user_doc.get("created_at") else None,
                    "last_active": user_doc.get("last_active_at").isoformat() if user_doc and user_doc.get("last_active_at") else None
                },
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Cache for 1 hour
            await self.base.key_builder.set(cache_key, analytics, ttl=3600)
            
            return analytics
            
        except Exception as e:
            logger.exception(f"❌ Error getting user analytics: {e}")
            return {"error": str(e)}

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        try:
            cache_key = "system:metrics"
            cached_metrics = await self.base.key_builder.get(cache_key)
            if cached_metrics:
                return cached_metrics
            
            # User metrics
            total_users = await self.base.db.users.count_documents({})
            active_users_7d = await self.base.db.users.count_documents({
                "last_active_at": {"$gte": datetime.now(timezone.utc) - timedelta(days=7)}
            })
            
            # Plan distribution
            plan_pipeline = [
                {"$group": {
                    "_id": "$plan_type",
                    "count": {"$sum": 1}
                }}
            ]
            plan_distribution = await self.base.db.users.aggregate(plan_pipeline).to_list(length=None)
            
            # Message volume (last 24 hours)
            messages_24h = await self.base.db.enhanced_conversations.count_documents({
                "timestamp": {"$gte": datetime.now(timezone.utc) - timedelta(hours=24)}
            })
            
            # Goals metrics
            total_goals = await self.base.db.financial_goals.count_documents({})
            active_goals = await self.base.db.financial_goals.count_documents({"status": "active"})
            
            # Alert metrics
            total_alerts = await self.base.db.user_alerts.count_documents({})
            active_alerts = await self.base.db.user_alerts.count_documents({"status": "active"})
            
            metrics = {
                "users": {
                    "total": total_users,
                    "active_7d": active_users_7d,
                    "plan_distribution": {item["_id"]: item["count"] for item in plan_distribution}
                },
                "activity": {
                    "messages_24h": messages_24h
                },
                "goals": {
                    "total": total_goals,
                    "active": active_goals
                },
                "alerts": {
                    "total": total_alerts,
                    "active": active_alerts
                },
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Cache for 15 minutes
            await self.base.key_builder.set(cache_key, metrics, ttl=900)
            
            return metrics
            
        except Exception as e:
            logger.exception(f"❌ Error getting system metrics: {e}")
            return {"error": str(e)}

    async def record_performance_metric(self, metric_name: str, value: float, metadata: Dict = None) -> bool:
        """Record a performance metric"""
        try:
            metric_doc = {
                "metric_name": metric_name,
                "value": value,
                "metadata": metadata or {},
                "timestamp": datetime.now(timezone.utc)
            }
            
            await self.base.db.performance_metrics.insert_one(metric_doc)
            return True
            
        except Exception as e:
            logger.exception(f"❌ Error recording performance metric: {e}")
            return False

    async def get_performance_metrics(self, metric_name: str = None, hours: int = 24) -> List[Dict]:
        """Get performance metrics for analysis"""
        try:
            start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            query = {"timestamp": {"$gte": start_time}}
            if metric_name:
                query["metric_name"] = metric_name
            
            metrics = await self.base.db.performance_metrics.find(query).sort("timestamp", -1).to_list(length=None)
            
            # Process metrics
            for metric in metrics:
                metric["_id"] = str(metric["_id"])
                if isinstance(metric.get("timestamp"), datetime):
                    metric["timestamp"] = metric["timestamp"].isoformat()
            
            return metrics
            
        except Exception as e:
            logger.exception(f"❌ Error getting performance metrics: {e}")
            return []
