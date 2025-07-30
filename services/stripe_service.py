# services/stripe_service.py - Enhanced Stripe Integration with Advanced Subscription Management
import stripe
from typing import Dict, Optional, List, Any
from loguru import logger
from datetime import datetime, timezone, timedelta
from enum import Enum
import json

from config import settings

stripe.api_key = settings.stripe_secret_key

class WebhookEventType(Enum):
    CUSTOMER_CREATED = "customer.created"
    CUSTOMER_UPDATED = "customer.updated"
    CUSTOMER_DELETED = "customer.deleted"
    
    SUBSCRIPTION_CREATED = "customer.subscription.created"
    SUBSCRIPTION_UPDATED = "customer.subscription.updated"
    SUBSCRIPTION_DELETED = "customer.subscription.deleted"
    SUBSCRIPTION_TRIAL_WILL_END = "customer.subscription.trial_will_end"
    
    INVOICE_CREATED = "invoice.created"
    INVOICE_FINALIZED = "invoice.finalized"
    INVOICE_PAID = "invoice.payment_succeeded"
    INVOICE_FAILED = "invoice.payment_failed"
    
    PAYMENT_SUCCEEDED = "payment_intent.succeeded"
    PAYMENT_FAILED = "payment_intent.payment_failed"

class StripeService:
    def __init__(self):
        self.basic_price_id = settings.stripe_paid_price_id or "price_basic_monthly"
        self.pro_price_id = settings.stripe_pro_price_id or "price_pro_monthly"
        self.success_url = "https://yourbot.com/success"
        self.cancel_url = "https://yourbot.com/cancel"
        
    async def create_customer(self, 
                            phone_number: str, 
                            email: str = None,
                            metadata: Dict[str, str] = None) -> Optional[Dict]:
        """Create a new Stripe customer"""
        try:
            customer_data = {
                "phone": phone_number,
                "metadata": {
                    "phone_number": phone_number,
                    "created_via": "sms_bot",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **(metadata or {})
                }
            }
            
            if email:
                customer_data["email"] = email
            
            customer = stripe.Customer.create(**customer_data)
            
            logger.info(f"✅ Created Stripe customer for {phone_number}: {customer.id}")
            return {
                "id": customer.id,
                "phone": customer.phone,
                "email": customer.email,
                "created": customer.created
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"❌ Failed to create Stripe customer: {e}")
            return None
    
    async def create_checkout_session(self, 
                                    customer_id: str,
                                    plan_type: str,
                                    user_metadata: Dict[str, str] = None,
                                    trial_days: int = None) -> Optional[str]:
        """Create Stripe Checkout session for subscription"""
        try:
            price_id = self.basic_price_id if plan_type == "basic" else self.pro_price_id
            
            session_data = {
                "customer": customer_id,
                "payment_method_types": ["card"],
                "line_items": [{
                    "price": price_id,
                    "quantity": 1,
                }],
                "mode": "subscription",
                "success_url": f"{self.success_url}?session_id={{CHECKOUT_SESSION_ID}}",
                "cancel_url": self.cancel_url,
                "metadata": {
                    "plan_type": plan_type,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **(user_metadata or {})
                }
            }
            
            # Add trial period if specified
            if trial_days:
                session_data["subscription_data"] = {
                    "trial_period_days": trial_days
                }
            
            session = stripe.checkout.Session.create(**session_data)
            
            logger.info(f"✅ Created checkout session for customer {customer_id}: {plan_type}")
            return session.url
            
        except stripe.error.StripeError as e:
            logger.error(f"❌ Failed to create checkout session: {e}")
            return None
    
    async def create_payment_link(self, 
                                plan_type: str,
                                user_id: str,
                                discount_percent: int = None,
                                trial_days: int = None) -> str:
        """Create Stripe payment link for subscription"""
        try:
            price_id = self.basic_price_id if plan_type == "basic" else self.pro_price_id
            
            payment_link_data = {
                "line_items": [{
                    "price": price_id,
                    "quantity": 1,
                }],
                "after_completion": {
                    "type": "redirect",
                    "redirect": {"url": self.success_url}
                },
                "metadata": {
                    "user_id": user_id,
                    "plan_type": plan_type,
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
            }
            
            # Add discount if provided
            if discount_percent:
                coupon = await self._create_discount_coupon(discount_percent)
                if coupon:
                    payment_link_data["discounts"] = [{"coupon": coupon.id}]
                    payment_link_data["metadata"]["discount_percent"] = str(discount_percent)
            
            # Add trial if specified
            if trial_days:
                payment_link_data["subscription_data"] = {
                    "trial_period_days": trial_days
                }
                payment_link_data["metadata"]["trial_days"] = str(trial_days)
            
            payment_link = stripe.PaymentLink.create(**payment_link_data)
            
            logger.info(f"✅ Payment link created for plan {plan_type} with {discount_percent or 0}% discount")
            return payment_link.url
            
        except stripe.error.StripeError as e:
            logger.error(f"❌ Payment link creation failed: {e}")
            return f"https://yourbot.com/upgrade-error?error=payment_link_failed"
    
    async def _create_discount_coupon(self, discount_percent: int) -> Optional[Any]:
        """Create a discount coupon"""
        try:
            coupon = stripe.Coupon.create(
                percent_off=discount_percent,
                duration="repeating",
                duration_in_months=3,
                name=f"{discount_percent}% off retention offer",
                metadata={
                    "type": "retention_offer",
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
            )
            return coupon
        except stripe.error.StripeError as e:
            logger.error(f"❌ Failed to create coupon: {e}")
            return None
    
    async def create_billing_portal_session(self, customer_id: str) -> Optional[str]:
        """Create billing portal session for customer management"""
        try:
            portal_session = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=self.success_url
            )
            
            logger.info(f"✅ Created billing portal session for customer {customer_id}")
            return portal_session.url
            
        except stripe.error.StripeError as e:
            logger.error(f"❌ Billing portal creation failed: {e}")
            return None
    
    async def get_customer_subscription(self, customer_id: str) -> Optional[Dict]:
        """Get customer's current subscription details"""
        try:
            subscriptions = stripe.Subscription.list(
                customer=customer_id,
                status="all",
                limit=1
            )
            
            if not subscriptions.data:
                return None
                
            sub = subscriptions.data[0]
            return {
                "id": sub.id,
                "status": sub.status,
                "current_period_start": datetime.fromtimestamp(sub.current_period_start, timezone.utc),
                "current_period_end": datetime.fromtimestamp(sub.current_period_end, timezone.utc),
                "trial_end": datetime.fromtimestamp(sub.trial_end, timezone.utc) if sub.trial_end else None,
                "plan_id": sub.items.data[0].price.id,
                "plan_name": "basic" if sub.items.data[0].price.id == self.basic_price_id else "pro",
                "amount": sub.items.data[0].price.unit_amount / 100,
                "cancel_at_period_end": sub.cancel_at_period_end,
                "canceled_at": datetime.fromtimestamp(sub.canceled_at, timezone.utc) if sub.canceled_at else None
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"❌ Failed to get subscription: {e}")
            return None
    
    async def cancel_subscription(self, 
                                subscription_id: str,
                                at_period_end: bool = True,
                                cancellation_reason: str = None) -> bool:
        """Cancel a subscription"""
        try:
            cancel_data = {}
            
            if at_period_end:
                # Cancel at end of billing period
                subscription = stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True,
                    metadata={
                        "cancellation_reason": cancellation_reason or "user_requested",
                        "cancelled_at": datetime.now(timezone.utc).isoformat()
                    }
                )
            else:
                # Cancel immediately
                subscription = stripe.Subscription.cancel(
                    subscription_id,
                    invoice_now=True,
                    prorate=True
                )
            
            logger.info(f"✅ Cancelled subscription {subscription_id} (at_period_end: {at_period_end})")
            return True
            
        except stripe.error.StripeError as e:
            logger.error(f"❌ Failed to cancel subscription: {e}")
            return False
    
    async def pause_subscription(self, 
                               subscription_id: str,
                               resume_date: datetime = None) -> bool:
        """Pause a subscription with optional resume date"""
        try:
            pause_data = {"pause_collection": {"behavior": "void"}}
            
            if resume_date:
                # Resume at specific date
                resume_timestamp = int(resume_date.timestamp())
                pause_data["pause_collection"]["resumes_at"] = resume_timestamp
            
            subscription = stripe.Subscription.modify(
                subscription_id,
                **pause_data,
                metadata={
                    "paused_at": datetime.now(timezone.utc).isoformat(),
                    "resume_at": resume_date.isoformat() if resume_date else "manual"
                }
            )
            
            logger.info(f"✅ Paused subscription {subscription_id}")
            return True
            
        except stripe.error.StripeError as e:
            logger.error(f"❌ Failed to pause subscription: {e}")
            return False
    
    async def resume_subscription(self, subscription_id: str) -> bool:
        """Resume a paused subscription"""
        try:
            subscription = stripe.Subscription.modify(
                subscription_id,
                pause_collection=None,
                metadata={
                    "resumed_at": datetime.now(timezone.utc).isoformat()
                }
            )
            
            logger.info(f"✅ Resumed subscription {subscription_id}")
            return True
            
        except stripe.error.StripeError as e:
            logger.error(f"❌ Failed to resume subscription: {e}")
            return False
    
    async def update_subscription_plan(self, 
                                     subscription_id: str,
                                     new_plan_type: str,
                                     prorate: bool = True) -> bool:
        """Update subscription to different plan"""
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            new_price_id = self.basic_price_id if new_plan_type == "basic" else self.pro_price_id
            
            stripe.Subscription.modify(
                subscription_id,
                items=[{
                    "id": subscription["items"]["data"][0]["id"],
                    "price": new_price_id,
                }],
                proration_behavior="create_prorations" if prorate else "none",
                metadata={
                    "plan_changed_at": datetime.now(timezone.utc).isoformat(),
                    "previous_plan": "basic" if subscription["items"]["data"][0]["price"]["id"] == self.basic_price_id else "pro",
                    "new_plan": new_plan_type
                }
            )
            
            logger.info(f"✅ Updated subscription {subscription_id} to {new_plan_type}")
            return True
            
        except stripe.error.StripeError as e:
            logger.error(f"❌ Failed to update subscription plan: {e}")
            return False
    
    async def process_webhook_event(self, event_data: Dict) -> Dict[str, Any]:
        """Process Stripe webhook events"""
        try:
            event_type = event_data.get("type")
            event_object = event_data.get("data", {}).get("object", {})
            
            logger.info(f"🔔 Processing Stripe webhook: {event_type}")
            
            result = {
                "event_type": event_type,
                "processed": False,
                "customer_id": None,
                "subscription_id": None,
                "action_taken": None,
                "metadata": {}
            }
            
            # Extract common identifiers
            customer_id = event_object.get("customer")
            subscription_id = event_object.get("id") if event_type.startswith("customer.subscription") else event_object.get("subscription")
            
            result["customer_id"] = customer_id
            result["subscription_id"] = subscription_id
            
            # Process different event types
            if event_type == WebhookEventType.SUBSCRIPTION_CREATED.value:
                result.update(await self._handle_subscription_created(event_object))
                
            elif event_type == WebhookEventType.SUBSCRIPTION_UPDATED.value:
                result.update(await self._handle_subscription_updated(event_object))
                
            elif event_type == WebhookEventType.SUBSCRIPTION_DELETED.value:
                result.update(await self._handle_subscription_cancelled(event_object))
                
            elif event_type == WebhookEventType.SUBSCRIPTION_TRIAL_WILL_END.value:
                result.update(await self._handle_trial_ending(event_object))
                
            elif event_type == WebhookEventType.INVOICE_PAID.value:
                result.update(await self._handle_payment_succeeded(event_object))
                
            elif event_type == WebhookEventType.INVOICE_FAILED.value:
                result.update(await self._handle_payment_failed(event_object))
                
            else:
                logger.info(f"ℹ️ Unhandled webhook event type: {event_type}")
                result["action_taken"] = "ignored"
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing webhook event: {e}")
            return {
                "event_type": event_data.get("type", "unknown"),
                "processed": False,
                "error": str(e)
            }
    
    async def _handle_subscription_created(self, subscription: Dict) -> Dict:
        """Handle subscription creation webhook"""
        return {
            "processed": True,
            "action_taken": "subscription_created",
            "metadata": {
                "plan_id": subscription["items"]["data"][0]["price"]["id"],
                "status": subscription["status"],
                "trial_end": subscription.get("trial_end")
            }
        }
    
    async def _handle_subscription_updated(self, subscription: Dict) -> Dict:
        """Handle subscription update webhook"""
        return {
            "processed": True,
            "action_taken": "subscription_updated",
            "metadata": {
                "status": subscription["status"],
                "cancel_at_period_end": subscription.get("cancel_at_period_end"),
                "current_period_end": subscription.get("current_period_end")
            }
        }
    
    async def _handle_subscription_cancelled(self, subscription: Dict) -> Dict:
        """Handle subscription cancellation webhook"""
        return {
            "processed": True,
            "action_taken": "subscription_cancelled",
            "metadata": {
                "cancelled_at": subscription.get("canceled_at"),
                "status": subscription["status"]
            }
        }
    
    async def _handle_trial_ending(self, subscription: Dict) -> Dict:
        """Handle trial ending webhook"""
        return {
            "processed": True,
            "action_taken": "trial_ending",
            "metadata": {
                "trial_end": subscription.get("trial_end"),
                "status": subscription["status"]
            }
        }
    
    async def _handle_payment_succeeded(self, invoice: Dict) -> Dict:
        """Handle successful payment webhook"""
        return {
            "processed": True,
            "action_taken": "payment_succeeded",
            "metadata": {
                "amount_paid": invoice.get("amount_paid"),
                "billing_reason": invoice.get("billing_reason"),
                "period_start": invoice.get("period_start"),
                "period_end": invoice.get("period_end")
            }
        }
    
    async def _handle_payment_failed(self, invoice: Dict) -> Dict:
        """Handle failed payment webhook"""
        return {
            "processed": True,
            "action_taken": "payment_failed",
            "metadata": {
                "amount_due": invoice.get("amount_due"),
                "attempt_count": invoice.get("attempt_count"),
                "next_payment_attempt": invoice.get("next_payment_attempt")
            }
        }
    
    async def get_subscription_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get subscription analytics from Stripe"""
        try:
            # Get date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            # Get all subscriptions
            subscriptions = stripe.Subscription.list(
                created={"gte": int(start_date.timestamp())},
                limit=100
            )
            
            # Analyze subscription data
            analytics = {
                "total_subscriptions": len(subscriptions.data),
                "active_subscriptions": len([s for s in subscriptions.data if s.status == "active"]),
                "trialing_subscriptions": len([s for s in subscriptions.data if s.status == "trialing"]),
                "cancelled_subscriptions": len([s for s in subscriptions.data if s.status == "canceled"]),
                "plan_breakdown": {},
                "mrr": 0,
                "total_revenue": 0
            }
            
            # Calculate plan breakdown and revenue
            for sub in subscriptions.data:
                if sub.items.data:
                    price_id = sub.items.data[0].price.id
                    amount = sub.items.data[0].price.unit_amount / 100
                    
                    plan_name = "basic" if price_id == self.basic_price_id else "pro"
                    
                    if plan_name not in analytics["plan_breakdown"]:
                        analytics["plan_breakdown"][plan_name] = {"count": 0, "revenue": 0}
                    
                    analytics["plan_breakdown"][plan_name]["count"] += 1
                    
                    if sub.status == "active":
                        analytics["plan_breakdown"][plan_name]["revenue"] += amount
                        analytics["mrr"] += amount
            
            return analytics
            
        except stripe.error.StripeError as e:
            logger.error(f"❌ Failed to get subscription analytics: {e}")
            return {}
    
    async def create_retention_offer(self, 
                                   customer_id: str,
                                   discount_percent: int = 50,
                                   duration_months: int = 3) -> Optional[str]:
        """Create a retention offer for churning customer"""
        try:
            # Get customer's current subscription
            subscription_data = await self.get_customer_subscription(customer_id)
            if not subscription_data:
                return None
            
            current_plan = subscription_data["plan_name"]
            
            # Create retention coupon
            coupon = stripe.Coupon.create(
                percent_off=discount_percent,
                duration="repeating",
                duration_in_months=duration_months,
                name=f"Retention Offer - {discount_percent}% off for {duration_months} months",
                metadata={
                    "type": "retention_offer",
                    "customer_id": customer_id,
                    "original_plan": current_plan,
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
            )
            
            # Create payment link with discount
            payment_link = stripe.PaymentLink.create(
                line_items=[{
                    "price": self.basic_price_id if current_plan == "basic" else self.pro_price_id,
                    "quantity": 1,
                }],
                discounts=[{"coupon": coupon.id}],
                after_completion={
                    "type": "redirect",
                    "redirect": {"url": f"{self.success_url}?retention=accepted"}
                },
                metadata={
                    "customer_id": customer_id,
                    "offer_type": "retention",
                    "discount_percent": str(discount_percent),
                    "duration_months": str(duration_months)
                }
            )
            
            logger.info(f"✅ Created retention offer for customer {customer_id}: {discount_percent}% off")
            return payment_link.url
            
        except stripe.error.StripeError as e:
            logger.error(f"❌ Failed to create retention offer: {e}")
            return None
