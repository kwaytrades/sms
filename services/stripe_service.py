# services/stripe_service.py
import stripe
from typing import Dict, Optional, Any
from loguru import logger
from config import settings

class StripeService:
    def __init__(self):
        if settings.stripe_secret_key:
            stripe.api_key = settings.stripe_secret_key
            self.webhook_secret = settings.stripe_webhook_secret
            logger.info("✅ Stripe service initialized")
        else:
            logger.warning("⚠️ Stripe credentials not configured")
    
    async def handle_webhook(self, payload: bytes, sig_header: str) -> Optional[Dict[str, Any]]:
        """Handle Stripe webhook with signature verification"""
        try:
            if not self.webhook_secret:
                logger.warning("⚠️ Stripe webhook secret not configured")
                return None
            
            event = stripe.Webhook.construct_event(
                payload, sig_header, self.webhook_secret
            )
            
            logger.info(f"✅ Stripe webhook received: {event['type']}")
            return event
            
        except ValueError as e:
            logger.error(f"❌ Invalid Stripe payload: {e}")
            raise
        except stripe.error.SignatureVerificationError as e:
            logger.error(f"❌ Invalid Stripe signature: {e}")
            raise
    
    async def create_customer(self, phone_number: str, email: str = None) -> Optional[str]:
        """Create Stripe customer"""
        try:
            customer = stripe.Customer.create(
                phone=phone_number,
                email=email,
                metadata={"phone_number": phone_number}
            )
            
            logger.info(f"✅ Created Stripe customer: {customer.id}")
            return customer.id
            
        except Exception as e:
            logger.error(f"❌ Stripe customer creation failed: {e}")
            return None
