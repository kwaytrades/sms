# ===== services/twilio_service.py - MINIMAL VERSION =====
from loguru import logger
from config import settings

class TwilioService:
    def __init__(self):
        self.account_sid = settings.twilio_account_sid
        self.auth_token = settings.twilio_auth_token
        self.from_number = settings.twilio_phone_number
        logger.info("✅ Twilio service initialized")
    
    async def send_sms(self, to_number: str, message: str) -> bool:
        """Send SMS message"""
        try:
            if not self.account_sid or not self.auth_token:
                logger.warning(f"Twilio not configured - would send to {to_number}: {message}")
                return True  # Return success in testing mode
            
            # TODO: Implement actual Twilio integration
            from twilio.rest import Client
            client = Client(self.account_sid, self.auth_token)
            
            message = client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_number
            )
            logger.info(f"✅ SMS sent to {to_number}: {message.sid}")
            return True
        except Exception as e:
            logger.error(f"❌ SMS send failed to {to_number}: {e}")
            return False
