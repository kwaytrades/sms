# ===== services/weekly_scheduler.py - FIXED COMPLETE IMPLEMENTATION =====
import asyncio
from datetime import datetime, timedelta, timezone
from loguru import logger
from typing import List, Dict, Any

from services.database import DatabaseService
from services.twilio_service import TwilioService

class WeeklyScheduler:
    def __init__(self, db: DatabaseService, twilio: TwilioService):
        self.db = db
        self.twilio = twilio
        self.est = timezone(timedelta(hours=-5))  # EST timezone
        self.last_notification_check = None
    
    async def start_scheduler(self):
        """Start the weekly notification scheduler"""
        logger.info("🕐 Starting weekly reset scheduler...")
        
        while True:
            try:
                await self._check_and_send_notifications()
                # Check every hour
                await asyncio.sleep(3600)
            except Exception as e:
                logger.error(f"❌ Scheduler error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _check_and_send_notifications(self):
        """Check if it's time to send reset notifications"""
        now = datetime.now(self.est)
        current_hour_key = f"{now.weekday()}_{now.hour}_{now.minute // 30}"  # 30-minute windows
        
        # Prevent duplicate notifications in the same time window
        if self.last_notification_check == current_hour_key:
            return
        
        self.last_notification_check = current_hour_key
        
        # Send 24-hour reminder on Sunday at 9:30 AM EST
        if now.weekday() == 6 and now.hour == 9 and 30 <= now.minute < 60:
            await self._send_24_hour_reminders()
        
        # Send reset notifications on Monday at 9:30 AM EST
        elif now.weekday() == 0 and now.hour == 9 and 30 <= now.minute < 60:
            await self._send_reset_notifications()
            await self._reset_weekly_counters()
    
    async def _send_24_hour_reminders(self):
        """Send 24-hour reset reminders ONLY to users who hit their limits"""
        logger.info("📅 Sending 24-hour reset reminders to limited users...")
        
        try:
            # Get ONLY users who hit their weekly limits
            limited_users = await self._get_users_who_hit_limits()
            
            if not limited_users:
                logger.info("📅 No users hit limits this week - no reminders needed")
                return
            
            sent_count = 0
            for user in limited_users:
                plan_info = self._get_plan_upgrade_info(user.get('plan_type', 'free'))
                
                message = f"""🔄 Your AI trading insights reset tomorrow at 9:30 AM EST!

Ready for another week of market analysis.

Want more insights? {plan_info}
[View Plans]"""
                
                success = await self.twilio.send_sms(user['phone_number'], message)
                if success:
                    sent_count += 1
                
                await asyncio.sleep(0.1)  # Rate limit API calls
            
            logger.info(f"✅ Sent 24-hour reminders to {sent_count}/{len(limited_users)} limited users")
            
        except Exception as e:
            logger.error(f"❌ Error sending 24-hour reminders: {e}")
    
    async def _send_reset_notifications(self):
        """Send reset notifications ONLY to users who were limited"""
        logger.info("🔄 Sending weekly reset notifications to limited users...")
        
        try:
            # Get ONLY users who hit their limits
            limited_users = await self._get_users_who_hit_limits()
            
            if not limited_users:
                logger.info("🔄 No users were limited this week - no reset notifications needed")
                return
            
            sent_count = 0
            for user in limited_users:
                plan_type = user.get('plan_type', 'free')
                
                if plan_type == "free":
                    message = """🚀 Limits reset! Your 10 weekly AI trading insights are ready.

Markets are open - get your edge! 📈

Need more? Upgrade for 10x analysis:
[Upgrade Now]"""
                
                elif plan_type == "paid":
                    message = """🚀 Your 100 monthly insights continue!

Keep dominating the markets this week! 📈

Want unlimited? Upgrade to Pro:
[Upgrade to Pro]"""
                
                elif plan_type == "pro":
                    message = """💎 Unlimited Pro insights continue!

Full access - let's make moves! 🚀"""
                
                success = await self.twilio.send_sms(user['phone_number'], message)
                if success:
                    sent_count += 1
                
                await asyncio.sleep(0.1)  # Rate limit API calls
            
            logger.info(f"✅ Sent reset notifications to {sent_count}/{len(limited_users)} limited users")
            
        except Exception as e:
            logger.error(f"❌ Error sending reset notifications: {e}")
    
    async def _reset_weekly_counters(self):
        """Reset weekly message counters for all users"""
        logger.info("🔄 Resetting weekly message counters...")
        
        try:
            now = datetime.now(timezone.utc)
            
            # Reset weekly counters for free plan users
            result = await self.db.db.users.update_many(
                {"plan_type": "free"},  # Only reset free users (weekly plan)
                {
                    "$set": {
                        "messages_this_period": 0,
                        "period_start": now,
                        "updated_at": now
                    }
                }
            )
            
            logger.info(f"✅ Reset weekly counters for {result.modified_count} free plan users")
            
            # Also clear Redis weekly usage counters
            if self.db.redis:
                # Get all weekly usage keys and delete them
                keys = []
                async for key in self.db.redis.scan_iter(match="usage:*:weekly"):
                    keys.append(key)
                
                if keys:
                    await self.db.redis.delete(*keys)
                    logger.info(f"✅ Cleared {len(keys)} Redis weekly usage counters")
            
        except Exception as e:
            logger.error(f"❌ Error resetting weekly counters: {e}")
    
    def _get_plan_upgrade_info(self, current_plan: str) -> str:
        """Get upgrade messaging based on current plan"""
        if current_plan == "free":
            return "Upgrade for 10x more analysis"
        elif current_plan == "paid":
            return "Upgrade to Pro for unlimited insights"
        else:
            return "You're on our highest tier"
    
    async def _get_users_who_hit_limits(self) -> List[Dict[str, Any]]:
        """Get ONLY users who actually hit their weekly limits - COMPLETE IMPLEMENTATION"""
        try:
            limited_users = []
            
            # Query for free plan users who hit their 10 message weekly limit
            free_limited_users_cursor = self.db.db.users.find({
                "plan_type": "free",
                "messages_this_period": {"$gte": 10}
            })
            
            async for user in free_limited_users_cursor:
                limited_users.append(user)
            
            # Query for paid plan users who hit their 100 message monthly limit
            # (though they're on monthly, not weekly, include for completeness)
            paid_limited_users_cursor = self.db.db.users.find({
                "plan_type": "paid",
                "messages_this_period": {"$gte": 100}
            })
            
            async for user in paid_limited_users_cursor:
                limited_users.append(user)
            
            # Query for pro plan users who hit daily cooloff (50+ messages in last 24 hours)
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            pro_users_cursor = self.db.db.users.find({"plan_type": "pro"})
            
            async for user in pro_users_cursor:
                # Check if they hit daily cooloff by counting recent usage
                daily_usage = await self.db.db.usage_tracking.count_documents({
                    "phone_number": user["phone_number"],
                    "action": "message_received",
                    "timestamp": {"$gte": yesterday}
                })
                
                if daily_usage >= 50:
                    limited_users.append(user)
            
            logger.info(f"📊 Found {len(limited_users)} users who hit their limits")
            return limited_users
            
        except Exception as e:
            logger.error(f"❌ Error getting limited users: {e}")
            return []
    
    async def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status and statistics"""
        try:
            now = datetime.now(self.est)
            next_check = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            
            # Calculate next reset times
            days_until_sunday = (6 - now.weekday()) % 7  # Sunday = 6
            if days_until_sunday == 0 and now.hour >= 9:
                days_until_sunday = 7
            
            next_24h_reminder = (now + timedelta(days=days_until_sunday)).replace(hour=9, minute=30, second=0, microsecond=0)
            
            days_until_monday = (7 - now.weekday()) % 7  # Monday = 0, but we want next Monday
            if days_until_monday == 0 and now.hour >= 9:
                days_until_monday = 7
            
            next_reset = (now + timedelta(days=days_until_monday)).replace(hour=9, minute=30, second=0, microsecond=0)
            
            # Get user statistics
            total_users = await self.db.db.users.count_documents({})
            free_users = await self.db.db.users.count_documents({"plan_type": "free"})
            limited_users = await self._get_users_who_hit_limits()
            
            return {
                "status": "active",
                "current_time": now.isoformat(),
                "timezone": "US/Eastern",
                "last_check": self.last_notification_check,
                "next_check": next_check.isoformat(),
                "next_24h_reminder": next_24h_reminder.isoformat(),
                "next_reset": next_reset.isoformat(),
                "user_statistics": {
                    "total_users": total_users,
                    "free_users": free_users,
                    "currently_limited": len(limited_users),
                    "limited_users_list": [u.get("phone_number", "unknown") for u in limited_users[:5]]  # First 5 for preview
                },
                "schedule": {
                    "24h_reminder": "Sunday 9:30 AM EST",
                    "weekly_reset": "Monday 9:30 AM EST",
                    "check_frequency": "Every hour"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting scheduler status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "current_time": datetime.now(self.est).isoformat()
            }
    
    async def manual_reset_trigger(self) -> Dict[str, Any]:
        """Manually trigger reset notifications (for testing)"""
        try:
            logger.info("🔧 Manual reset trigger activated")
            
            # Send notifications
            await self._send_reset_notifications()
            
            # Reset counters
            await self._reset_weekly_counters()
            
            return {
                "status": "success",
                "message": "Manual reset completed",
                "timestamp": datetime.now(self.est).isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Manual reset trigger failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(self.est).isoformat()
            }
    
    async def manual_reminder_trigger(self) -> Dict[str, Any]:
        """Manually trigger 24-hour reminders (for testing)"""
        try:
            logger.info("🔧 Manual reminder trigger activated")
            
            await self._send_24_hour_reminders()
            
            return {
                "status": "success",
                "message": "Manual reminders sent",
                "timestamp": datetime.now(self.est).isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Manual reminder trigger failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(self.est).isoformat()
            }
