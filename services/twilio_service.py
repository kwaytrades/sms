# services/twilio_service.py
from twilio.rest import Client
from loguru import logger
from config import settings

class TwilioService:
    def __init__(self):
        self.account_sid = settings.twilio_account_sid
        self.auth_token = settings.twilio_auth_token
        self.from_number = settings.twilio_phone_number
        
        if self.account_sid and self.auth_token:
            self.client = Client(self.account_sid, self.auth_token)
            logger.info("✅ Twilio service initialized")
        else:
            self.client = None
            logger.warning("⚠️ Twilio credentials not configured")
    
    async def send_message(self, to_number: str, message: str) -> bool:
        """Send SMS message"""
        try:
            if not self.client:
                logger.warning(f"Twilio not configured - would send to {to_number}: {message}")
                return True
            
            message = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_number
            )
            logger.info(f"✅ SMS sent to {to_number}: {message.sid}")
            return True
        except Exception as e:
            logger.error(f"❌ SMS send failed to {to_number}: {e}")
            return False
