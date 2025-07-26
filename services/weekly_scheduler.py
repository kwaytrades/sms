# ===== services/weekly_scheduler.py - MINIMAL VERSION =====
import asyncio
from datetime import datetime, timedelta, timezone
from loguru import logger

class WeeklyScheduler:
    def __init__(self, db_service, twilio_service):
        self.db = db_service
        self.twilio = twilio_service
        self.est = timezone(timedelta(hours=-5))
        logger.info("✅ Weekly scheduler initialized")
    
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
        
        # Send 24-hour reminder on Sunday at 9:30 AM EST
        if now.weekday() == 6 and now.hour == 9 and now.minute == 30:
            logger.info("📅 Sending 24-hour reset reminders...")
        
        # Send reset notifications on Monday at 9:30 AM EST
        elif now.weekday() == 0 and now.hour == 9 and now.minute == 30:
            logger.info("🔄 Sending weekly reset notifications...")
