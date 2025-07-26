# ===== services/weekly_scheduler.py =====
import asyncio
from datetime import datetime, timedelta, timezone
from loguru import logger
from typing import List

from services.database import DatabaseService
from services.twilio_service import TwilioService

class WeeklyScheduler:
    def __init__(self, db: DatabaseService, twilio: TwilioService):
        self.db = db
        self.twilio = twilio
        self.est = timezone(timedelta(hours=-5))  # EST timezone
    
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
        """Check if it's time to send reset notifications - only to limited users"""
        now = datetime.now(self.est)
        
        # Send 24-hour reminder on Sunday at 9:30 AM EST (only to limited users)
        if now.weekday() == 6 and now.hour == 9 and now.minute == 30:
            await self._send_24_hour_reminders()
        
        # Send reset notifications on Monday at 9:30 AM EST (only to limited users)
        elif now.weekday() == 0 and now.hour == 9 and now.minute == 30:
            await self._send_reset_notifications()
    
    async def _send_24_hour_reminders(self):
        """Send 24-hour reset reminders ONLY to users who hit their limits"""
        logger.info("📅 Sending 24-hour reset reminders to limited users...")
        
        try:
            # Get ONLY users who hit their weekly limits
            limited_users = await self._get_users_who_hit_limits()
            
            if not limited_users:
                logger.info("📅 No users hit limits this week - no reminders needed")
                return
            
            for user in limited_users:
                plan_info = self._get_plan_upgrade_info(user.plan_type)
                
                message = f"""🔄 Your AI trading insights reset tomorrow at 9:30 AM EST!

Ready for another week of market analysis.

Want more insights? {plan_info}
[View Plans]"""
                
                await self.twilio.send_sms(user.phone_number, message)
                await asyncio.sleep(0.1)  # Rate limit API calls
            
            logger.info(f"✅ Sent 24-hour reminders to {len(limited_users)} limited users")
            
        except Exception as e:
            logger.error(f"❌ Error sending 24-hour reminders: {e}")
    
    async def _send_reset_notifications(self):
        """Send reset notifications ONLY to users who were limited"""
        logger.info("🔄 Sending weekly reset notifications to limited users...")
        
        try:
            # Get ONLY users who hit their limits and are waiting for reset
            limited_users = await self._get_users_who_hit_limits()
            
            if not limited_users:
                logger.info("🔄 No users were limited this week - no reset notifications needed")
                return
            
            for user in limited_users:
                if user.plan_type == "free":
                    message = """🚀 Limits reset! Your 4 AI trading insights are ready.

Markets are open - get your edge! 📈

Need more? Upgrade for 10x analysis:
[Upgrade Now]"""
                
                elif user.plan_type == "standard":
                    message = """🚀 Your 40 weekly insights have reset!

Ready to dominate the markets this week? 📈

Upgrade to VIP for 3x more analysis:
[Upgrade to VIP]"""
                
                elif user.plan_type == "vip":
                    message = """💎 Your 120 VIP insights have reset!

Full access restored - let's make moves! 🚀"""
                
                await self.twilio.send_sms(user.phone_number, message)
                await asyncio.sleep(0.1)  # Rate limit API calls
            
            logger.info(f"✅ Sent reset notifications to {len(limited_users)} limited users")
            
        except Exception as e:
            logger.error(f"❌ Error sending reset notifications: {e}")
    
    def _get_plan_upgrade_info(self, current_plan: str) -> str:
        """Get upgrade messaging based on current plan"""
        if current_plan == "free":
            return "Upgrade for 10x more analysis"
        elif current_plan == "standard":
            return "Upgrade to VIP for 3x more insights"
        else:
            return "You're on our highest tier"
    
    async def _get_users_who_hit_limits(self) -> List:
        """Get ONLY users who actually hit their weekly limits"""
        try:
            # Query database for users whose usage equals their plan limit
            # This should return users with usage_count >= plan_limit
            
            # Example query logic:
            # SELECT users.* FROM users 
            # JOIN usage_tracking ON users._id = usage_tracking.user_id 
            # WHERE usage_tracking.period = 'weekly' 
            # AND usage_tracking.count >= plan_limits[users.plan_type]
            
            # Placeholder - implement actual database query
            limited_users = []
            
            logger.info(f"📊 Found {len(limited_users)} users who hit their limits")
            return limited_users
            
        except Exception as e:
            logger.error(f"❌ Error getting limited users: {e}")
            return []
    
    async def _get_limited_users(self) -> List:
        """DEPRECATED - use _get_users_who_hit_limits instead"""
        return await self._get_users_who_hit_limits()
    
    async def _get_active_users(self) -> List:
        """DEPRECATED - notifications now only go to limited users"""
        return []

# ===== Add to main.py startup =====
"""
Add this to your main.py lifespan function:

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize services
    db_service = DatabaseService()
    await db_service.initialize()
    
    twilio_service = TwilioService()
    
    # Start weekly scheduler
    scheduler = WeeklyScheduler(db_service, twilio_service)
    scheduler_task = asyncio.create_task(scheduler.start_scheduler())
    
    app.state.db = db_service
    app.state.scheduler_task = scheduler_task
    
    yield
    
    # Cleanup
    scheduler_task.cancel()
    await db_service.close()
"""
