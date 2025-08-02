services/db/compliance_service.py
"""
Compliance Service - GDPR compliance, data export, user data deletion
Focused on privacy compliance and data protection
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from bson import ObjectId
from loguru import logger

from services.db.base_db_service import BaseDBService


class ComplianceService:
    """Specialized service for GDPR compliance and data protection"""
    
    def __init__(self, base_service: BaseDBService):
        self.base = base_service

    async def initialize(self):
        """Initialize compliance tracking"""
        try:
            # Data requests tracking
            await self.base.db.data_requests.create_index("user_id")
            await self.base.db.data_requests.create_index("request_type")
            await self.base.db.data_requests.create_index("created_at")
            
            logger.info("✅ Compliance service initialized")
        except Exception as e:
            logger.exception(f"❌ Compliance service initialization failed: {e}")

    async def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all user data for GDPR compliance"""
        try:
            # Record the export request
            request_doc = {
                "user_id": user_id,
                "request_type": "export",
                "status": "processing",
                "created_at": datetime.now(timezone.utc)
            }
            request_result = await self.base.db.data_requests.insert_one(request_doc)
            
            export_data = {
                "user_id": user_id,
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": str(request_result.inserted_id),
                "data": {}
            }
            
            # User profile
            user = await self.base.db.users.find_one({"_id": ObjectId(user_id) if len(user_id) == 24 else None, "user_id": user_id})
            if user:
                user["_id"] = str(user["_id"])
                export_data["data"]["profile"] = user
                phone_number = user.get("phone_number")
            else:
                phone_number = None
            
            # Conversations
            if phone_number:
                conversations = await self.base.db.enhanced_conversations.find({"phone_number": phone_number}).to_list(length=None)
                for conv in conversations:
                    conv["_id"] = str(conv["_id"])
                    if isinstance(conv.get("timestamp"), datetime):
                        conv["timestamp"] = conv["timestamp"].isoformat()
                export_data["data"]["conversations"] = conversations
                
                # Conversation context
                context = await self.base.db.conversation_context.find_one({"phone_number": phone_number})
                if context:
                    context["_id"] = str(context["_id"])
                    export_data["data"]["conversation_context"] = context
            
            # Goals
            goals = await self.base.db.financial_goals.find({"user_id": user_id}).to_list(length=None)
            for goal in goals:
                goal["_id"] = str(goal["_id"])
                if isinstance(goal.get("created_at"), datetime):
                    goal["created_at"] = goal["created_at"].isoformat()
                if isinstance(goal.get("updated_at"), datetime):
                    goal["updated_at"] = goal["updated_at"].isoformat()
            export_data["data"]["goals"] = goals
            
            # Alerts
            alerts = await self.base.db.user_alerts.find({"user_id": user_id}).to_list(length=None)
            for alert in alerts:
                alert["_id"] = str(alert["_id"])
                if isinstance(alert.get("created_at"), datetime):
                    alert["created_at"] = alert["created_at"].isoformat()
            export_data["data"]["alerts"] = alerts
            
            # Trade markers
            trade_markers = await self.base.db.trade_markers.find({"user_id": user_id}).to_list(length=None)
            for marker in trade_markers:
                marker["_id"] = str(marker["_id"])
                if isinstance(marker.get("timestamp"), datetime):
                    marker["timestamp"] = marker["timestamp"].isoformat()
            export_data["data"]["trade_markers"] = trade_markers
            
            # Personality profile
            personality = await self.base.db.personality_profiles.find_one({"user_id": user_id})
            if personality:
                personality["_id"] = str(personality["_id"])
                export_data["data"]["personality"] = personality
            
            # Update request status
            await self.base.db.data_requests.update_one(
                {"_id": request_result.inserted_id},
                {"$set": {"status": "completed", "completed_at": datetime.now(timezone.utc)}}
            )
            
            logger.info(f"✅ Exported user data for {user_id}")
            return export_data
            
        except Exception as e:
            logger.exception(f"❌ Error exporting user data: {e}")
            return {"error": str(e)}

    async def delete_user_data(self, user_id: str) -> Dict[str, Any]:
        """Delete all user data for GDPR compliance"""
        try:
            # Record the deletion request
            request_doc = {
                "user_id": user_id,
                "request_type": "deletion",
                "status": "processing",
                "created_at": datetime.now(timezone.utc)
            }
            request_result = await self.base.db.data_requests.insert_one(request_doc)
            
            deletion_report = {
                "user_id": user_id,
                "deletion_timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": str(request_result.inserted_id),
                "deleted_collections": [],
                "deleted_cache_keys": [],
                "errors": []
            }
            
            # Get user first to get phone number
            user = await self.base.db.users.find_one({"_id": ObjectId(user_id) if len(user_id) == 24 else None, "user_id": user_id})
            phone_number = user.get("phone_number") if user else None
            
            # Delete from MongoDB collections
            collections_to_clean = [
                "users",
                "personality_profiles", 
                "financial_goals",
                "goal_milestones",
                "user_alerts",
                "alert_history",
                "trade_markers",
                "trade_performance",
                "onboarding_progress",
                "notification_logs",
                "user_analytics",
                "user_feature_overrides"
            ]
            
            for collection_name in collections_to_clean:
                try:
                    collection = getattr(self.base.db, collection_name)
                    result = await collection.delete_many({"user_id": user_id})
                    if result.deleted_count > 0:
                        deletion_report["deleted_collections"].append(f"{collection_name}: {result.deleted_count}")
                except Exception as e:
                    deletion_report["errors"].append(f"Error deleting from {collection_name}: {e}")
            
            # Delete phone-number-based data
            if phone_number:
                phone_collections = [
                    "enhanced_conversations",
                    "conversation_context", 
                    "daily_sessions"
                ]
                
                for collection_name in phone_collections:
                    try:
                        collection = getattr(self.base.db, collection_name)
                        result = await collection.delete_many({"phone_number": phone_number})
                        if result.deleted_count > 0:
                            deletion_report["deleted_collections"].append(f"{collection_name}: {result.deleted_count}")
                    except Exception as e:
                        deletion_report["errors"].append(f"Error deleting from {collection_name}: {e}")
            
            # Delete from Redis cache
            cache_keys = [
                f"user:{user_id}:personality",
                f"user:id:{user_id}",
                f"user:phone:{phone_number}" if phone_number else "",
                f"context:{phone_number}" if phone_number else "",
                f"goals:{user_id}",
                f"alerts:active:{user_id}",
                f"analytics:{user_id}:30",
                f"onboarding:{user_id}",
                f"feature_flags:{user_id}"
            ]
            
            for key in cache_keys:
                if not key:
                    continue
                try:
                    deleted = await self.base.key_builder.delete(key)
                    if deleted:
                        deletion_report["deleted_cache_keys"].append(key)
                except Exception as e:
                    deletion_report["errors"].append(f"Error deleting cache {key}: {e}")
            
            # Update request status
            await self.base.db.data_requests.update_one(
                {"_id": request_result.inserted_id},
                {"$set": {"status": "completed", "completed_at": datetime.now(timezone.utc)}}
            )
            
            logger.info(f"✅ Deleted user data for {user_id}: {len(deletion_report['deleted_collections'])} collections, {len(deletion_report['deleted_cache_keys'])} cache keys")
            return deletion_report
            
        except Exception as e:
            logger.exception(f"❌ Error deleting user data: {e}")
            return {"error": str(e)}
