# ===== services/stripe_integration.py =====
import stripe
from typing import Dict, Optional
from loguru import logger
from datetime import datetime, timedelta
from config import settings

stripe.api_key = settings.stripe_secret_key

class StripeIntegrationService:
    def __init__(self):
        self.paid_price_id = settings.stripe_paid_price_id
        self.pro_price_id = settings.stripe_pro_price_id
        self.success_url = "https://yourbot.com/success"
        self.cancel_url = "https://yourbot.com/cancel"
        
    async def create_payment_link(self, user_id: str, plan_type: str) -> str:
        """Create Stripe payment link for subscription"""
        try:
            price_id = self.paid_price_id if plan_type == "paid" else self.pro_price_id
            
            payment_link = stripe.PaymentLink.create(
                line_items=[{
                    'price': price_id,
                    'quantity': 1,
                }],
                after_completion={
                    'type': 'redirect',
                    'redirect': {'url': self.success_url}
                },
                metadata={
                    'user_id': user_id,
                    'plan_type': plan_type,
                    'created_at': datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"✅ Payment link created for user {user_id}, plan {plan_type}")
            return payment_link.url
            
        except stripe.error.StripeError as e:
            logger.error(f"❌ Payment link creation failed: {e}")
            return "https://yourbot.com/upgrade-error"
    
    async def create_billing_portal_link(self, customer_id: str) -> str:
        """Create billing portal session for customer management"""
        try:
            portal_session = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=self.success_url
            )
            
            return portal_session.url
            
        except stripe.error.StripeError as e:
            logger.error(f"❌ Billing portal creation failed: {e}")
            return "https://yourbot.com/billing-error"
    
    async def create_discount_link(self, user_id: str, discount_percent: int) -> str:
        """Create discounted subscription link for retention"""
        try:
            # Create a coupon for the discount
            coupon = stripe.Coupon.create(
                percent_off=discount_percent,
                duration='repeating',
                duration_in_months=3,
                name=f'{discount_percent}% off retention offer'
            )
            
            # Create payment link with coupon
            payment_link = stripe.PaymentLink.create(
                line_items=[{
                    'price': self.pro_price_id,  # Assume retention offer for pro
                    'quantity': 1,
                }],
                discounts=[{
                    'coupon': coupon.id
                }],
                after_completion={
                    'type': 'redirect',
                    'redirect': {'url': self.success_url}
                },
                metadata={
                    'user_id': user_id,
                    'offer_type': 'retention',
                    'discount_percent': str(discount_percent)
                }
            )
            
            return payment_link.url
            
        except stripe.error.StripeError as e:
            logger.error(f"❌ Discount link creation failed: {e}")
            return "https://yourbot.com/offer-error"
    
    async def get_customer_subscription(self, customer_id: str) -> Optional[Dict]:
        """Get customer's current subscription details"""
        try:
            subscriptions = stripe.Subscription.list(customer=customer_id, limit=1)
            
            if subscriptions.data:
                sub = subscriptions.data[0]
                return {
                    'id': sub.id,
                    'status': sub.status,
                    'current_period_end': datetime.fromtimestamp(sub.current_period_end),
                    'plan_id': sub.items.data[0].price.id,
                    'amount': sub.items.data[0].price.unit_amount / 100
                }
            
            return None
            
        except stripe.error.StripeError as e:
            logger.error(f"❌ Failed to get subscription: {e}")
            return None
