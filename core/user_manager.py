# core/user_manager.py - Enhanced User Management with Advanced Stripe Integration
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
from bson import ObjectId
from loguru import logger
from enum import Enum

from config import settings
from services.database import DatabaseService

class SubscriptionStatus(Enum):
    ACTIVE = "active"
    TRIALING = "trialing"
    PAST_DUE = "past_due"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    PAUSED = "paused"

class AccountStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    DELETED = "deleted"

class PlanType(Enum):
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"

class PlanLimits:
    PLANS = {
        PlanType.FREE.value: {
            "weekly_limit": 10,
            "monthly_limit": 40,
            "price": 0,
            "features": ["basic_analysis", "daily_insights"],
            "daily_cooloff": None
        },
        PlanType.BASIC.value: {
            "weekly_limit": 100,
            "monthly_limit": 400,
            "price": 29,
            "features": ["basic_analysis", "daily_insights", "portfolio_tracking", "personalized_insights"],
            "daily_cooloff": None
        },
        PlanType.PRO.value: {
            "weekly_limit": None,  # Unlimited
            "monthly_limit": None,  # Unlimited
            "price": 99,
            "features": ["all_features", "real_time_alerts", "priority_support", "advanced_analysis"],
            "daily_cooloff": 50  # Soft limit before cooloff
        }
    }

    @classmethod
    def get_plan_config(cls, plan_type: str) -> Dict[str, Any]:
        return cls.PLANS.get(plan_type, cls.PLANS[PlanType.FREE.value])

class UserManager:
    def __init__(self, db_service: DatabaseService):
        self.db = db_service
    
    async def get_user_by_phone(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """Get user by phone number"""
        try:
            user = await self.db.db.users.find_one({"phone_number": phone_number})
            if user:
                user["_id"] = str(user["_id"])
                logger.info(f"✅ Found user: {phone_number}")
                return user
            return None
            
        except Exception as e:
            logger.error(f"❌ Error finding user {phone_number}: {e}")
            return None
    
    async def get_user_by_stripe_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get user by Stripe customer ID"""
        try:
            user = await self.db.db.users.find_one({"stripe_customer_id": customer_id})
            if user:
                user["_id"] = str(user["_id"])
                return user
            return None
        except Exception as e:
            logger.error(f"❌ Error finding user by Stripe customer {customer_id}: {e}")
            return None
    
    async def get_or_create_user(self, phone_number: str) -> Dict[str, Any]:
        """Get existing user or create new one"""
        user = await self.get_user_by_phone(phone_number)
        
        if user:
            # Update existing user schema if needed
            await self._update_user_schema(user)
            return await self.get_user_by_phone(phone_number)
        else:
            # Create new user
            return await self._create_new_user(phone_number)
    
    async def _create_new_user(self, phone_number: str) -> Dict[str, Any]:
        """Create a new user with complete schema"""
        now = datetime.now(timezone.utc)
        trial_end = now + timedelta(days=7)  # 7-day trial for new users
        
        new_user = {
            "phone_number": phone_number,
            "email": None,
            "first_name": None,
            "timezone": "US/Eastern",
            
            # Enhanced Subscription Management
            "plan_type": PlanType.FREE.value,
            "subscription_status": SubscriptionStatus.TRIALING.value,
            "account_status": AccountStatus.ACTIVE.value,
            "billing_status": "current",
            "stripe_customer_id": None,
            "stripe_subscription_id": None,
            "trial_ends_at": trial_end,
            "subscription_started_at": now,
            "subscription_cancelled_at": None,
            "subscription_paused_at": None,
            "subscription_resume_at": None,
            
            # Cancellation and Retention Tracking
            "cancellation_reason": None,
            "retention_offers_sent": 0,
            "last_retention_offer": None,
            "retention_offer_accepted": False,
            "downgrade_history": [],
            "payment_failures": 0,
            "last_payment_failure": None,
            
            # Enhanced Usage Tracking
            "total_messages_sent": 0,
            "total_messages_received": 0,
            "messages_this_week": 0,
            "messages_this_month": 0,
            "week_start": now.replace(hour=0, minute=0, second=0, microsecond=0),
            "month_start": now.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
            "daily_message_count": 0,
            "daily_reset_time": now.replace(hour=0, minute=0, second=0, microsecond=0),
            "last_message_at": None,
            "last_active_at": now,
            
            # Trading profile
            "risk_tolerance": "medium",
            "trading_experience": "intermediate",
            "preferred_sectors": [],
            "watchlist": [],
            "trading_style": "swing",
            
            # Communication preferences
            "communication_style": {
                "formality": "casual",
                "message_length": "medium", 
                "emoji_usage": True,
                "technical_depth": "medium"
            },
            
            # Behavioral patterns
            "response_patterns": {
                "preferred_response_time": "immediate",
                "engagement_triggers": [],
                "successful_trade_patterns": [],
                "loss_triggers": []
            },
            
            # Feature usage tracking
            "feature_usage": {
                "basic_analysis": 0,
                "portfolio_tracking": 0,
                "daily_insights": 0,
                "real_time_alerts": 0,
                "advanced_analysis": 0
            },
            
            # Engagement metrics
            "engagement_score": 0.0,
            "satisfaction_rating": None,
            "nps_score": None,
            "feedback_history": [],
            
            # Timestamps
            "created_at": now,
            "updated_at": now,
            
            # Internal tracking
            "_id": None
        }
        
        try:
            result = await self.db.db.users.insert_one(new_user)
            new_user["_id"] = str(result.inserted_id)
            logger.info(f"✅ Created new user: {phone_number} with trial until {trial_end}")
            return new_user
            
        except Exception as e:
            logger.error(f"❌ Error creating user {phone_number}: {e}")
            raise
    
    async def _update_user_schema(self, user: Dict[str, Any]) -> bool:
        """Update existing user to latest schema"""
        try:
            updates = {}
            now = datetime.now(timezone.utc)
            
            # Add missing fields with defaults
            if "subscription_status" not in user:
                updates["subscription_status"] = SubscriptionStatus.ACTIVE.value
            if "account_status" not in user:
                updates["account_status"] = AccountStatus.ACTIVE.value
            if "billing_status" not in user:
                updates["billing_status"] = "current"
            if "retention_offers_sent" not in user:
                updates["retention_offers_sent"] = 0
            if "feature_usage" not in user:
                updates["feature_usage"] = {
                    "basic_analysis": 0,
                    "portfolio_tracking": 0,
                    "daily_insights": 0,
                    "real_time_alerts": 0,
                    "advanced_analysis": 0
                }
            if "engagement_score" not in user:
                updates["engagement_score"] = 0.0
            if "messages_this_week" not in user:
                updates["messages_this_week"] = 0
            if "messages_this_month" not in user:
                updates["messages_this_month"] = 0
            
            if updates:
                updates["updated_at"] = now
                await self.db.db.users.update_one(
                    {"phone_number": user["phone_number"]},
                    {"$set": updates}
                )
                logger.info(f"✅ Updated user schema for {user['phone_number']}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"❌ Error updating user schema: {e}")
            return False
    
    async def update_subscription(self, 
                                phone_number: str, 
                                plan_type: str,
                                stripe_customer_id: str = None,
                                stripe_subscription_id: str = None,
                                subscription_status: str = None) -> bool:
        """Enhanced subscription update with full lifecycle tracking"""
        try:
            now = datetime.now(timezone.utc)
            
            # Validate plan type
            if plan_type not in [p.value for p in PlanType]:
                logger.error(f"❌ Invalid plan type: {plan_type}")
                return False
            
            updates = {
                "plan_type": plan_type,
                "updated_at": now
            }
            
            # Set subscription status based on plan and context
            if subscription_status:
                updates["subscription_status"] = subscription_status
            elif plan_type == PlanType.FREE.value:
                updates["subscription_status"] = SubscriptionStatus.ACTIVE.value
            else:
                updates["subscription_status"] = SubscriptionStatus.ACTIVE.value
            
            # Update Stripe IDs if provided
            if stripe_customer_id:
                updates["stripe_customer_id"] = stripe_customer_id
            if stripe_subscription_id:
                updates["stripe_subscription_id"] = stripe_subscription_id
            
            # Reset usage counters for upgrades
            if plan_type in [PlanType.BASIC.value, PlanType.PRO.value]:
                updates["messages_this_week"] = 0
                updates["messages_this_month"] = 0
                updates["week_start"] = now.replace(hour=0, minute=0, second=0, microsecond=0)
                updates["month_start"] = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                updates["subscription_started_at"] = now
                updates["billing_status"] = "current"
                updates["payment_failures"] = 0
            
            # Track subscription history
            if plan_type != PlanType.FREE.value:
                updates["subscription_started_at"] = now
            
            result = await self.db.db.users.update_one(
                {"phone_number": phone_number},
                {"$set": updates}
            )
            
            logger.info(f"💳 Updated subscription for {phone_number}: {plan_type} -> {subscription_status or 'active'}")
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"❌ Error updating subscription for {phone_number}: {e}")
            return False
    
    async def handle_subscription_cancellation(self, 
                                             phone_number: str, 
                                             reason: str = None,
                                             immediate: bool = False) -> bool:
        """Handle subscription cancellation with retention tracking"""
        try:
            now = datetime.now(timezone.utc)
            updates = {
                "subscription_cancelled_at": now,
                "cancellation_reason": reason,
                "updated_at": now
            }
            
            if immediate:
                # Immediate cancellation - downgrade to free
                updates["plan_type"] = PlanType.FREE.value
                updates["subscription_status"] = SubscriptionStatus.EXPIRED.value
                updates["billing_status"] = "cancelled"
            else:
                # End of billing period cancellation
                updates["subscription_status"] = SubscriptionStatus.CANCELLED.value
            
            result = await self.db.db.users.update_one(
                {"phone_number": phone_number},
                {"$set": updates}
            )
            
            logger.info(f"🚫 Cancelled subscription for {phone_number}: {reason}")
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"❌ Error cancelling subscription: {e}")
            return False
    
    async def handle_payment_failure(self, phone_number: str) -> bool:
        """Handle payment failure with progressive responses"""
        try:
            user = await self.get_user_by_phone(phone_number)
            if not user:
                return False
                
            now = datetime.now(timezone.utc)
            payment_failures = user.get("payment_failures", 0) + 1
            
            updates = {
                "payment_failures": payment_failures,
                "last_payment_failure": now,
                "billing_status": "past_due" if payment_failures < 3 else "failed",
                "updated_at": now
            }
            
            # Suspend service after 3 failures
            if payment_failures >= 3:
                updates["subscription_status"] = SubscriptionStatus.SUSPENDED.value
                updates["account_status"] = AccountStatus.SUSPENDED.value
            
            result = await self.db.db.users.update_one(
                {"phone_number": phone_number},
                {"$set": updates}
            )
            
            logger.warning(f"💳 Payment failure #{payment_failures} for {phone_number}")
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"❌ Error handling payment failure: {e}")
            return False
    
    async def track_retention_offer(self, 
                                  phone_number: str, 
                                  offer_type: str,
                                  accepted: bool = False) -> bool:
        """Track retention offers sent to users"""
        try:
            now = datetime.now(timezone.utc)
            user = await self.get_user_by_phone(phone_number)
            
            updates = {
                "retention_offers_sent": user.get("retention_offers_sent", 0) + 1,
                "last_retention_offer": now,
                "updated_at": now
            }
            
            if accepted:
                updates["retention_offer_accepted"] = True
            
            # Track offer in history
            offer_record = {
                "type": offer_type,
                "sent_at": now,
                "accepted": accepted
            }
            
            await self.db.db.users.update_one(
                {"phone_number": phone_number},
                {
                    "$set": updates,
                    "$push": {"retention_offer_history": offer_record}
                }
            )
            
            logger.info(f"📈 Retention offer tracked for {phone_number}: {offer_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error tracking retention offer: {e}")
            return False
    
    async def check_usage_limits(self, phone_number: str) -> Dict[str, Any]:
        """Check if user has exceeded usage limits"""
        try:
            user = await self.get_user_by_phone(phone_number)
            if not user:
                return {"allowed": False, "reason": "User not found"}
            
            plan_config = PlanLimits.get_plan_config(user["plan_type"])
            now = datetime.now(timezone.utc)
            
            # Check trial status
            if user.get("subscription_status") == SubscriptionStatus.TRIALING.value:
                trial_end = user.get("trial_ends_at")
                if trial_end and now > trial_end:
                    # Trial expired, downgrade to free
                    await self.update_subscription(phone_number, PlanType.FREE.value)
                    user["plan_type"] = PlanType.FREE.value
                    plan_config = PlanLimits.get_plan_config(PlanType.FREE.value)
            
            # Check account status
            if user.get("account_status") == AccountStatus.SUSPENDED.value:
                return {
                    "allowed": False, 
                    "reason": "Account suspended",
                    "action_required": "contact_support"
                }
            
            # Pro plan unlimited (with soft daily limit)
            if user["plan_type"] == PlanType.PRO.value:
                daily_count = user.get("daily_message_count", 0)
                if daily_count >= plan_config["daily_cooloff"]:
                    return {
                        "allowed": True,
                        "reason": "Daily cooloff reached",
                        "remaining": 0,
                        "cooloff": True
                    }
                return {"allowed": True, "unlimited": True}
            
            # Check weekly limits for Free and Basic
            weekly_limit = plan_config["weekly_limit"]
            weekly_used = user.get("messages_this_week", 0)
            
            if weekly_used >= weekly_limit:
                return {
                    "allowed": False,
                    "reason": "Weekly limit exceeded",
                    "limit": weekly_limit,
                    "used": weekly_used,
                    "reset_date": user.get("week_start", now) + timedelta(days=7)
                }
            
            return {
                "allowed": True,
                "remaining": weekly_limit - weekly_used,
                "limit": weekly_limit,
                "used": weekly_used
            }
            
        except Exception as e:
            logger.error(f"❌ Error checking usage limits: {e}")
            return {"allowed": False, "reason": "System error"}
    
    async def increment_usage(self, phone_number: str, feature: str = None) -> bool:
        """Increment usage counters"""
        try:
            now = datetime.now(timezone.utc)
            user = await self.get_user_by_phone(phone_number)
            
            # Reset counters if needed
            week_start = user.get("week_start", now)
            month_start = user.get("month_start", now)
            daily_reset = user.get("daily_reset_time", now)
            
            updates = {
                "total_messages_sent": user.get("total_messages_sent", 0) + 1,
                "last_message_at": now,
                "last_active_at": now,
                "updated_at": now
            }
            
            # Reset weekly counter if new week
            if (now - week_start).days >= 7:
                updates["messages_this_week"] = 1
                updates["week_start"] = now.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                updates["messages_this_week"] = user.get("messages_this_week", 0) + 1
            
            # Reset monthly counter if new month
            if now.month != month_start.month or now.year != month_start.year:
                updates["messages_this_month"] = 1
                updates["month_start"] = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                updates["messages_this_month"] = user.get("messages_this_month", 0) + 1
            
            # Reset daily counter if new day
            if (now - daily_reset).days >= 1:
                updates["daily_message_count"] = 1
                updates["daily_reset_time"] = now.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                updates["daily_message_count"] = user.get("daily_message_count", 0) + 1
            
            # Track feature usage
            if feature and feature in user.get("feature_usage", {}):
                feature_path = f"feature_usage.{feature}"
                updates[feature_path] = user["feature_usage"][feature] + 1
            
            result = await self.db.db.users.update_one(
                {"phone_number": phone_number},
                {"$set": updates}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"❌ Error incrementing usage: {e}")
            return False
    
    async def get_user_stats(self) -> Dict[str, Any]:
        """Get comprehensive user statistics"""
        try:
            total_users = await self.db.db.users.count_documents({})
            
            # Plan breakdown
            plan_pipeline = [
                {"$group": {"_id": "$plan_type", "count": {"$sum": 1}}}
            ]
            plan_breakdown = await self.db.db.users.aggregate(plan_pipeline).to_list(None)
            plan_stats = {item["_id"]: item["count"] for item in plan_breakdown}
            
            # Subscription status breakdown
            status_pipeline = [
                {"$group": {"_id": "$subscription_status", "count": {"$sum": 1}}}
            ]
            status_breakdown = await self.db.db.users.aggregate(status_pipeline).to_list(None)
            status_stats = {item["_id"]: item["count"] for item in status_breakdown}
            
            # Active users (last 24 hours)
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            active_users = await self.db.db.users.count_documents({
                "last_active_at": {"$gte": yesterday}
            })
            
            # Revenue metrics
            revenue_pipeline = [
                {"$match": {"plan_type": {"$in": ["basic", "pro"]}}},
                {"$group": {
                    "_id": "$plan_type",
                    "count": {"$sum": 1}
                }}
            ]
            revenue_breakdown = await self.db.db.users.aggregate(revenue_pipeline).to_list(None)
            
            mrr = 0
            for item in revenue_breakdown:
                plan_config = PlanLimits.get_plan_config(item["_id"])
                mrr += item["count"] * plan_config["price"]
            
            # Retention metrics
            total_cancellations = await self.db.db.users.count_documents({
                "subscription_cancelled_at": {"$exists": True}
            })
            
            return {
                "total_users": total_users,
                "active_today": active_users,
                "plan_breakdown": plan_stats,
                "subscription_status": status_stats,
                "revenue": {
                    "mrr": mrr,
                    "breakdown": {item["_id"]: item["count"] for item in revenue_breakdown}
                },
                "retention": {
                    "total_cancellations": total_cancellations,
                    "cancellation_rate": total_cancellations / max(total_users, 1) * 100
                },
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting user stats: {e}")
            return {}
    
    async def get_churn_candidates(self) -> List[Dict[str, Any]]:
        """Identify users at risk of churning"""
        try:
            # Users who haven't been active in 7 days but have paid plans
            week_ago = datetime.now(timezone.utc) - timedelta(days=7)
            
            pipeline = [
                {
                    "$match": {
                        "plan_type": {"$in": ["basic", "pro"]},
                        "last_active_at": {"$lt": week_ago},
                        "subscription_status": {"$in": ["active", "trialing"]}
                    }
                },
                {
                    "$project": {
                        "phone_number": 1,
                        "plan_type": 1,
                        "last_active_at": 1,
                        "total_messages_sent": 1,
                        "engagement_score": 1,
                        "days_inactive": {
                            "$divide": [
                                {"$subtract": [datetime.now(timezone.utc), "$last_active_at"]},
                                86400000  # milliseconds in a day
                            ]
                        }
                    }
                },
                {"$sort": {"days_inactive": -1}}
            ]
            
            churn_candidates = await self.db.db.users.aggregate(pipeline).to_list(None)
            return churn_candidates
            
        except Exception as e:
            logger.error(f"❌ Error getting churn candidates: {e}")
            return []
