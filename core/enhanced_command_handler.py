# core/enhanced_command_handler.py - Advanced SMS Command Processing
from typing import Dict, Optional, List, Any
from datetime import datetime, timezone
from loguru import logger

from core.user_manager import UserManager, PlanType, PlanLimits, SubscriptionStatus
from services.stripe_service import StripeService

class EnhancedCommandHandler:
    def __init__(self, user_manager: UserManager, stripe_service: StripeService):
        self.user_manager = user_manager
        self.stripe = stripe_service
        
        # Define command mappings
        self.commands = {
            # Account management
            "START": self._handle_start_command,
            "HELP": self._handle_help_command,
            "STATUS": self._handle_status_command,
            "PROFILE": self._handle_profile_command,
            
            # Subscription management
            "UPGRADE": self._handle_upgrade_command,
            "DOWNGRADE": self._handle_downgrade_command,
            "CANCEL": self._handle_cancel_command,
            "REACTIVATE": self._handle_reactivate_command,
            "BILLING": self._handle_billing_command,
            "PLANS": self._handle_plans_command,
            
            # Feature commands
            "WATCHLIST": self._handle_watchlist_command,
            "PORTFOLIO": self._handle_portfolio_command,
            "ALERTS": self._handle_alerts_command,
            "SETTINGS": self._handle_settings_command,
            
            # Support
            "SUPPORT": self._handle_support_command,
            "FEEDBACK": self._handle_feedback_command,
            
            # Compliance
            "STOP": self._handle_stop_command,
            "UNSTOP": self._handle_unstop_command,
        }
    
    async def process_command(self, phone_number: str, message: str) -> str:
        """Process SMS command and return response"""
        try:
            # Get or create user
            user = await self.user_manager.get_or_create_user(phone_number)
            
            # Clean and normalize command
            command = message.strip().upper()
            
            # Check if user is stopped
            if user.get("sms_stopped", False) and command != "UNSTOP":
                return "You've opted out of messages. Reply UNSTOP to resume."
            
            # Handle multi-word commands
            if command.startswith("CANCEL"):
                command = "CANCEL"
            elif command.startswith("UPGRADE"):
                command = "UPGRADE"
            elif command.startswith("HELP"):
                command = "HELP"
            
            # Process command
            if command in self.commands:
                response = await self.commands[command](user, message)
            else:
                # Not a command, treat as regular message
                response = await self._handle_regular_message(user, message)
            
            # Track command usage
            await self._track_command_usage(user, command)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error processing command '{message}' for {phone_number}: {e}")
            return "Sorry, I encountered an error. Please try again or reply HELP for assistance."
    
    async def _handle_start_command(self, user: Dict, message: str) -> str:
        """Handle START command - welcome new users"""
        is_new_user = user.get("total_messages_sent", 0) == 0
        
        if is_new_user:
            # New user welcome
            response = f"""🚀 Welcome to AI Trading Bot!

You're on a 7-day FREE trial with all features:
📱 Unlimited messages
📊 Advanced analysis
📈 Portfolio insights
🔔 Real-time alerts

Try asking:
• "How is AAPL?"
• "Find me tech stocks"
• "Analyze my portfolio"

After trial: Reply PLANS to see options
Questions? Reply HELP"""
        else:
            # Returning user
            plan_config = PlanLimits.get_plan_config(user["plan_type"])
            usage_info = await self.user_manager.check_usage_limits(user["phone_number"])
            
            response = f"""👋 Welcome back!

Current Plan: {user['plan_type'].upper()}
Status: {user.get('subscription_status', 'active').title()}"""
            
            if usage_info.get("unlimited"):
                response += "\nMessages: Unlimited ♾️"
            else:
                remaining = usage_info.get("remaining", 0)
                limit = usage_info.get("limit", 0)
                response += f"\nMessages: {remaining}/{limit} this week"
            
            response += "\n\nReply HELP for commands or ask about any stock!"
        
        return response
    
    async def _handle_help_command(self, user: Dict, message: str) -> str:
        """Handle HELP command - show available commands"""
        plan_type = user["plan_type"]
        
        response = f"""📋 Available Commands:

📊 ANALYSIS:
• Ask about any stock: "How is AAPL?"
• Screen stocks: "Find me tech stocks"
• Get price: "TSLA price"

👤 ACCOUNT:
• STATUS - Account overview
• PLANS - View subscription options
• BILLING - Manage payments
• PROFILE - Your trading profile

📈 FEATURES:"""
        
        if plan_type in [PlanType.BASIC.value, PlanType.PRO.value]:
            response += """
• PORTFOLIO - Track holdings
• WATCHLIST - Manage favorites"""
        
        if plan_type == PlanType.PRO.value:
            response += """
• ALERTS - Set price alerts
• SETTINGS - Customize preferences"""
        
        response += """

🆘 SUPPORT:
• SUPPORT - Contact help
• FEEDBACK - Share thoughts

💬 Just ask naturally:
"What's happening with Tesla?"
"Should I buy Microsoft?"

Questions? We're here to help! 🤖"""
        
        return response
    
    async def _handle_status_command(self, user: Dict, message: str) -> str:
        """Handle STATUS command - show account status"""
        plan_config = PlanLimits.get_plan_config(user["plan_type"])
        usage_info = await self.user_manager.check_usage_limits(user["phone_number"])
        
        # Basic account info
        response = f"""📊 Account Status

👤 Plan: {user['plan_type'].upper()}
💰 Price: ${plan_config['price']}/month
📱 Status: {user.get('subscription_status', 'active').title()}"""
        
        # Usage information
        if usage_info.get("unlimited"):
            response += "\n📈 Messages: Unlimited"
            if usage_info.get("cooloff"):
                response += f" (Daily cooloff: {usage_info.get('remaining', 0)} left today)"
        else:
            remaining = usage_info.get("remaining", 0)
            limit = usage_info.get("limit", 0)
            response += f"\n📈 Messages: {remaining}/{limit} this week"
        
        # Trial information
        if user.get("subscription_status") == SubscriptionStatus.TRIALING.value:
            trial_end = user.get("trial_ends_at")
            if trial_end:
                days_left = (trial_end - datetime.now(timezone.utc)).days
                response += f"\n⏰ Trial: {days_left} days left"
        
        # Account age and usage
        created_at = user.get("created_at")
        if created_at:
            days_active = (datetime.now(timezone.utc) - created_at).days
            response += f"\n📅 Member for: {days_active} days"
        
        total_messages = user.get("total_messages_sent", 0)
        response += f"\n💬 Total messages: {total_messages}"
        
        # Feature access
        features = plan_config.get("features", [])
        if features:
            response += f"\n\n✨ Your Features:\n• " + "\n• ".join(features)
        
        response += "\n\nReply UPGRADE to see premium options!"
        
        return response
    
    async def _handle_upgrade_command(self, user: Dict, message: str) -> str:
        """Handle UPGRADE command - show upgrade options"""
        current_plan = user["plan_type"]
        
        if current_plan == PlanType.PRO.value:
            return "🎉 You're already on our highest plan (PRO)! Enjoying unlimited access to all features."
        
        # Create or get Stripe customer
        customer_id = user.get("stripe_customer_id")
        if not customer_id:
            stripe_customer = await self.stripe.create_customer(
                user["phone_number"],
                user.get("email"),
                {"phone_number": user["phone_number"]}
            )
            if stripe_customer:
                customer_id = stripe_customer["id"]
                await self.user_manager.update_subscription(
                    user["phone_number"],
                    current_plan,
                    stripe_customer_id=customer_id
                )
        
        response = f"""🚀 Upgrade Your Experience

Current: {current_plan.upper()}

"""
        
        if current_plan == PlanType.FREE.value:
            # Show both basic and pro options
            basic_link = await self.stripe.create_payment_link("basic", user["_id"])
            pro_link = await self.stripe.create_payment_link("pro", user["_id"])
            
            response += f"""💼 BASIC - $29/month
✅ 100 messages/week
✅ Portfolio tracking  
✅ Personalized insights
✅ Daily market updates
🔗 Upgrade to Basic: {basic_link}

🚀 PRO - $99/month
✅ Unlimited messages
✅ Real-time alerts
✅ Advanced analysis
✅ Priority support
🔗 Upgrade to Pro: {pro_link}

🎁 First month 50% off with code WELCOME50"""
            
        elif current_plan == PlanType.BASIC.value:
            # Show pro upgrade
            pro_link = await self.stripe.create_payment_link("pro", user["_id"])
            
            response += f"""🚀 PRO - $99/month

Upgrade from Basic and get:
✅ Unlimited messages (vs 100/week)
✅ Real-time price alerts  
✅ Advanced technical analysis
✅ Options analysis
✅ Priority support
✅ Custom alerts

🔗 Upgrade to Pro: {pro_link}

💡 Perfect for active traders who need unlimited access!"""
        
        return response
    
    async def _handle_downgrade_command(self, user: Dict, message: str) -> str:
        """Handle DOWNGRADE command"""
        current_plan = user["plan_type"]
        
        if current_plan == PlanType.FREE.value:
            return "You're already on the FREE plan - the lowest tier available."
        
        # Generate downgrade options
        response = f"""⬇️ Downgrade Options

Current Plan: {current_plan.upper()}

"""
        
        if current_plan == PlanType.PRO.value:
            # Pro can downgrade to basic or free
            basic_link = await self.stripe.create_payment_link("basic", user["_id"])
            
            response += f"""💼 BASIC Plan - $29/month
✅ Keep: 100 messages/week, insights, portfolio
❌ Lose: Unlimited messages, real-time alerts
🔗 Switch to Basic: {basic_link}

🆓 FREE Plan - No charge
✅ Keep: 10 messages/week, basic analysis
❌ Lose: All premium features
⚠️ Confirm: Reply "CONFIRM FREE"

Changes take effect next billing cycle."""
        
        elif current_plan == PlanType.BASIC.value:
            response += f"""🆓 FREE Plan - No charge
✅ Keep: 10 messages/week, basic analysis  
❌ Lose: Portfolio tracking, personalized insights
⚠️ Confirm: Reply "CONFIRM FREE"

Changes take effect next billing cycle."""
        
        return response
    
    async def _handle_cancel_command(self, user: Dict, message: str) -> str:
        """Handle CANCEL command with retention offer"""
        
        if user["plan_type"] == PlanType.FREE.value:
            return "You're on the FREE plan - no subscription to cancel. Account will remain active."
        
        # Track cancellation attempt
        await self.user_manager.track_retention_offer(
            user["phone_number"],
            "cancellation_attempt"
        )
        
        # Generate retention offer (50% discount)
        customer_id = user.get("stripe_customer_id")
        if customer_id:
            discount_link = await self.stripe.create_retention_offer(customer_id, 50, 3)
            cancel_link = await self.stripe.create_billing_portal_session(customer_id)
            
            response = f"""😢 Sorry to see you go!

🎁 WAIT! Special offer just for you:
💰 50% off your {user['plan_type'].upper()} plan for 3 months
🔗 Accept offer: {discount_link}

Or continue cancellation: {cancel_link}

⚠️ Account remains active until billing period ends
Data will be deleted after 30 days of cancellation.

What made you want to cancel? (Reply to help us improve)"""
        else:
            response = f"""We're sorry to see you go! 

Your {user['plan_type'].upper()} plan will be cancelled and you'll be moved to our FREE plan.

Reply SUPPORT if you need help, or BILLING to manage your subscription."""
        
        return response
    
    async def _handle_reactivate_command(self, user: Dict, message: str) -> str:
        """Handle REACTIVATE command"""
        subscription_status = user.get("subscription_status")
        
        if subscription_status == SubscriptionStatus.ACTIVE.value:
            return "Your subscription is already active! 🎉"
        
        if subscription_status in [SubscriptionStatus.CANCELLED.value, SubscriptionStatus.EXPIRED.value]:
            # Show reactivation options
            basic_link = await self.stripe.create_payment_link("basic", user["_id"])
            pro_link = await self.stripe.create_payment_link("pro", user["_id"])
            
            return f"""🔄 Reactivate Your Subscription

💼 BASIC - $29/month
🔗 Reactivate Basic: {basic_link}

🚀 PRO - $99/month  
🔗 Reactivate Pro: {pro_link}

🎁 Welcome back bonus: First month 25% off!

Questions? Reply SUPPORT"""
        
        elif subscription_status == SubscriptionStatus.SUSPENDED.value:
            billing_link = await self.stripe.create_billing_portal_session(user.get("stripe_customer_id"))
            
            return f"""⚠️ Account Suspended

Your account was suspended due to payment issues.

To reactivate:
1. Update payment method: {billing_link}
2. Contact support: Reply SUPPORT

Once payment is resolved, full access will be restored immediately."""
        
        return "Unable to reactivate at this time. Reply SUPPORT for help."
    
    async def _handle_billing_command(self, user: Dict, message: str) -> str:
        """Handle BILLING command"""
        
        if user["plan_type"] == PlanType.FREE.value:
            return "You're on the FREE plan - no billing information to show. Reply UPGRADE to see paid options."
        
        # Get billing portal link
        customer_id = user.get("stripe_customer_id")
        if customer_id:
            portal_link = await self.stripe.create_billing_portal_session(customer_id)
            subscription_data = await self.stripe.get_customer_subscription(customer_id)
            
            response = f"""💳 Billing Information

Plan: {user['plan_type'].upper()}
Status: {user.get('subscription_status', 'active').title()}"""
            
            if subscription_data:
                next_billing = subscription_data.get('current_period_end')
                amount = subscription_data.get('amount', 0)
                response += f"\nNext charge: ${amount} on {next_billing.strftime('%B %d, %Y') if next_billing else 'Unknown'}"
            
            response += f"""

📋 Manage billing:
• Update payment method
• Download invoices  
• View charge history
• Cancel subscription
🔗 Billing Portal: {portal_link}

Questions? Reply SUPPORT"""
        else:
            response = "Billing information not available. Reply SUPPORT for assistance."
        
        return response
    
    async def _handle_plans_command(self, user: Dict, message: str) -> str:
        """Handle PLANS command - show all available plans"""
        current_plan = user["plan_type"]
        
        response = f"""💰 Trading Bot Plans

Current: {current_plan.upper()} ⭐

🆓 FREE - $0/month
• 10 messages per week
• Basic stock analysis
• Daily market insights

💼 BASIC - $29/month  
• 100 messages per week
• Portfolio tracking
• Personalized insights
• Email support

🚀 PRO - $99/month
• Unlimited messages
• Real-time alerts
• Advanced analysis
• Priority support
• Custom features

"""
        
        if current_plan != PlanType.PRO.value:
            response += "Reply UPGRADE to change plans!"
        else:
            response += "You're on our best plan! 🎉"
        
        return response
    
    async def _handle_watchlist_command(self, user: Dict, message: str) -> str:
        """Handle WATCHLIST command"""
        plan_type = user["plan_type"]
        
        if plan_type == PlanType.FREE.value:
            return "📈 Watchlist is a BASIC+ feature. Reply UPGRADE to unlock portfolio tracking!"
        
        watchlist = user.get("watchlist", [])
        
        if not watchlist:
            response = """📈 Your Watchlist

No stocks added yet.

Add stocks: "Add AAPL to watchlist"
Remove: "Remove TSLA from watchlist"
View prices: "Show watchlist prices"

Try adding your favorite stocks!"""
        else:
            response = f"""📈 Your Watchlist ({len(watchlist)} stocks)

Stocks: {', '.join(watchlist)}

Commands:
• "Add [SYMBOL] to watchlist"
• "Remove [SYMBOL] from watchlist"  
• "Show watchlist prices"
• "Analyze my watchlist"

Reply with any stock symbol for analysis!"""
        
        return response
    
    async def _handle_portfolio_command(self, user: Dict, message: str) -> str:
        """Handle PORTFOLIO command"""
        plan_type = user["plan_type"]
        
        if plan_type == PlanType.FREE.value:
            return "📊 Portfolio tracking is a BASIC+ feature. Reply UPGRADE to connect your brokerage!"
        
        # Check if portfolio is connected
        connected_accounts = user.get("connected_accounts", [])
        
        if not connected_accounts:
            response = """📊 Portfolio Tracking

Connect your brokerage for:
• Real-time portfolio value
• Performance analytics  
• Personalized insights
• Risk analysis

Supported brokers:
• Robinhood • TD Ameritrade
• E*TRADE • Schwab • Fidelity

Reply "CONNECT PORTFOLIO" to get started!"""
        else:
            response = f"""📊 Your Portfolio

Connected: {len(connected_accounts)} account(s)

Commands:
• "Show portfolio"
• "Portfolio performance"  
• "Analyze my holdings"
• "Portfolio alerts"

Your portfolio data is always private and secure. 🔒"""
        
        return response
    
    async def _handle_support_command(self, user: Dict, message: str) -> str:
        """Handle SUPPORT command"""
        response = """🆘 Customer Support

How can we help?

📧 Email: support@tradingbot.com
💬 Live chat: Available 9am-5pm ET
📱 SMS: Describe your issue and we'll respond ASAP

Common issues:
• Billing questions: Reply BILLING
• Account problems: Reply STATUS  
• Feature help: Reply HELP

For urgent billing issues, call:
📞 1-800-TRADING (24/7)

We typically respond within 2 hours! 🚀"""
        
        # Log support request
        await self._log_support_request(user, message)
        
        return response
    
    async def _handle_feedback_command(self, user: Dict, message: str) -> str:
        """Handle FEEDBACK command"""
        return """💭 We Value Your Feedback!

How are we doing? Rate your experience:

⭐ Reply with 1-5 stars: "4 stars"
💬 Or tell us what you think!

Your feedback helps us improve. Recent updates based on user suggestions:
• Faster response times
• Better stock recommendations  
• Enhanced mobile experience

Thank you for helping us build a better service! 🙏"""
    
    async def _handle_stop_command(self, user: Dict, message: str) -> str:
        """Handle STOP command - opt out of SMS"""
        await self.user_manager.db.db.users.update_one(
            {"phone_number": user["phone_number"]},
            {
                "$set": {
                    "sms_stopped": True,
                    "stopped_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )
        
        return "You've been unsubscribed from SMS messages. Reply UNSTOP to resume. Your account remains active."
    
    async def _handle_unstop_command(self, user: Dict, message: str) -> str:
        """Handle UNSTOP command - opt back in"""
        await self.user_manager.db.db.users.update_one(
            {"phone_number": user["phone_number"]},
            {
                "$set": {
                    "sms_stopped": False,
                    "updated_at": datetime.now(timezone.utc)
                },
                "$unset": {
                    "stopped_at": ""
                }
            }
        )
        
        return "Welcome back! You're now subscribed to SMS messages. Reply HELP for available commands."
    
    async def _handle_profile_command(self, user: Dict, message: str) -> str:
        """Handle PROFILE command - show user profile"""
        response = f"""👤 Your Trading Profile

📱 Phone: {user['phone_number']}
📧 Email: {user.get('email', 'Not set')}
🎯 Experience: {user.get('trading_experience', 'intermediate').title()}
⚡ Risk Level: {user.get('risk_tolerance', 'medium').title()}
📈 Style: {user.get('trading_style', 'swing').title()}

🧠 AI Personality:
• Formality: {user.get('communication_style', {}).get('formality', 'casual').title()}
• Detail Level: {user.get('communication_style', {}).get('technical_depth', 'medium').title()}

Commands:
• "Set email [email]"
• "Set experience beginner/intermediate/advanced"
• "Set risk low/medium/high"

Your profile helps us personalize responses! 🤖"""
        
        return response
    
    async def _handle_alerts_command(self, user: Dict, message: str) -> str:
        """Handle ALERTS command"""
        if user["plan_type"] != PlanType.PRO.value:
            return "🔔 Real-time alerts are a PRO feature. Reply UPGRADE to unlock advanced monitoring!"
        
        # This would integrate with your alerts system
        return """🔔 Your Alerts

No active alerts set.

Set alerts:
• "Alert me when AAPL hits $200"
• "Alert if TSLA drops 5%"  
• "Daily update on my portfolio"

Alert types:
• Price targets
• Percentage moves
• Volume spikes
• News mentions
• Earnings dates

Reply with your alert preferences!"""
    
    async def _handle_settings_command(self, user: Dict, message: str) -> str:
        """Handle SETTINGS command"""
        return """⚙️ Your Settings

Communication:
• Style: Casual
• Length: Medium responses
• Emojis: Enabled ✅
• Time zone: Eastern

Notifications:
• Daily insights: Enabled
• Premarket: 9:00 AM
• Market close: 4:00 PM
• Breaking news: Enabled

Change settings:
• "Set style formal/casual"
• "Set timezone Pacific/Eastern/Central"
• "Disable daily insights"

Your preferences are saved automatically! 💾"""
    
    async def _handle_regular_message(self, user: Dict, message: str) -> str:
        """Handle regular trading questions"""
        # Check usage limits before processing
        usage_check = await self.user_manager.check_usage_limits(user["phone_number"])
        
        if not usage_check.get("allowed"):
            reason = usage_check.get("reason", "")
            
            if "limit exceeded" in reason:
                limit = usage_check.get("limit", 0)
                reset_date = usage_check.get("reset_date")
                
                return f"""📊 Weekly Limit Reached

You've used all {limit} messages this week. Limit resets on {reset_date.strftime('%A, %B %d') if reset_date else 'Monday'}.

🚀 Upgrade for more messages:
• BASIC: 100/week ($29/mo)
• PRO: Unlimited ($99/mo)

Reply UPGRADE to continue trading! 📈"""
            
            elif "suspended" in reason:
                return """⚠️ Account Suspended

Your account is temporarily suspended due to payment issues.

To reactivate: Reply BILLING
For help: Reply SUPPORT

We're here to help resolve this quickly! 💙"""
        
        # Increment usage counter
        await self.user_manager.increment_usage(user["phone_number"], "basic_analysis")
        
        # Return placeholder for regular message processing
        # This would integrate with your existing trading analysis logic
        return f"""🤖 Processing your request: "{message}"

This would connect to your existing trading analysis system to provide:
• Stock analysis
• Technical indicators  
• Market insights
• Personalized recommendations

[Your existing analysis logic would go here]"""
    
    async def _track_command_usage(self, user: Dict, command: str):
        """Track command usage for analytics"""
        try:
            await self.user_manager.db.db.command_usage.insert_one({
                "phone_number": user["phone_number"],
                "command": command,
                "timestamp": datetime.now(timezone.utc),
                "plan_type": user["plan_type"],
                "subscription_status": user.get("subscription_status")
            })
        except Exception as e:
            logger.error(f"❌ Error tracking command usage: {e}")
    
    async def _log_support_request(self, user: Dict, message: str):
        """Log support request for follow-up"""
        try:
            await self.user_manager.db.db.support_requests.insert_one({
                "phone_number": user["phone_number"],
                "message": message,
                "timestamp": datetime.now(timezone.utc),
                "plan_type": user["plan_type"],
                "status": "open",
                "priority": "normal"
            })
            logger.info(f"📝 Support request logged for {user['phone_number']}")
        except Exception as e:
            logger.error(f"❌ Error logging support request: {e}")
