# services/twilio_service.py
from twilio.rest import Client
from typing import Optional
from loguru import logger
from config import settings

class TwilioService:
    def __init__(self):
        if settings.twilio_account_sid and settings.twilio_auth_token:
            self.client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
            self.from_number = settings.twilio_phone_number
            logger.info("✅ Twilio service initialized")
        else:
            self.client = None
            logger.warning("⚠️ Twilio credentials not configured")
    
    async def send_sms(self, to_number: str, message: str) -> bool:
        """Send SMS message"""
        if not self.client:
            logger.warning(f"⚠️ Twilio not configured - would send to {to_number}: {message[:50]}...")
            return True  # Return True for testing without Twilio
        
        try:
            result = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_number
            )
            logger.info(f"✅ SMS sent to {to_number}: {result.sid}")
            return True
            
        except Exception as e:
            logger.error(f"❌ SMS send failed to {to_number}: {e}")
            return False
    
    async def send_batch_sms(self, recipients: List[str], message: str) -> Dict[str, int]:
        """Send batch SMS to multiple recipients"""
        results = {"sent": 0, "failed": 0, "errors": []}
        
        for phone_number in recipients:
            success = await self.send_sms(phone_number, message)
            if success:
                results["sent"] += 1
            else:
                results["failed"] += 1
                results["errors"].append(phone_number)
        
        return results
