# ===== core/enhanced_command_handler.py =====
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from loguru import logger
import json

from models.user import UserProfile, PlanLimits
from services.stripe_integration import StripeIntegrationService
from services.database import DatabaseService
from services.twilio_service import TwilioService
from utils.helpers import format_currency, extract_stock_symbols

class EnhancedCommandHandler:
    def __init__(self, db: DatabaseService, twilio: TwilioService, stripe_service):
        self.db = db
        self.twilio = twilio
        self.stripe = stripe_service
        
    async def handle_command(self, user: UserProfile, command: str, phone_number: str) -> bool:
        """Handle all SMS commands with full logic"""
        try:
            command = command.lower().strip()
            
            # Route to specific command handlers
            if command in ["start", "begin"]:
                response = await self._handle_start_command(user)
            elif command == "/upgrade":
                response = await self._handle_upgrade_command(user)
            elif command == "/downgrade":
                response = await self._handle_downgrade_command(user)
            elif command == "/cancel":
                response = await self._handle_cancel_command(user)
            elif command == "/billing":
                response = await self._handle_billing_command(user)
            elif command == "/status":
                response = await self._handle_status_command(user)
            elif command == "/help":
                response = await self._handle_help_command(user)
            elif command == "/watchlist":
                response = await self._handle_watchlist_command(user)
            elif command == "/portfolio":
                response = await self._handle_portfolio_command(user)
            elif command == "/screen":
                response = await self._handle_screen_command(user)
            elif command == "/alerts":
                response = await self._handle_alerts_command(user)
            elif command == "/settings":
                response = await self._handle_settings_command(user)
            elif command == "/support":
                response = await self._handle_support_command(user)
            elif command == "/pause":
                response = await self._handle_pause_command(user)
            elif command == "/resume":
                response = await self._handle_resume_command(user)
            elif command in ["stop", "unsubscribe"]:
                response = await self._handle_stop_command(user)
            else:
                response = "Unknown command. Reply '/help' for available commands."
            
            # Send response
            await self.twilio.send_sms(phone_number, response)
            return True
            
        except Exception as e:
            logger.error(f"❌ Command handling failed for {command}: {e}")
            await self.twilio.send_sms(phone_number, "Sorry, something went wrong. Please try again.")
            return False
    
    async def _handle_start_command(self, user: UserProfile) -> str:
        """Handle START command - onboarding"""
        user_name = user.first_name or "there"
        
        return f"""🚀 Welcome to AI Trading Insights, {user_name}!

You're on our FREE trial (10 messages/week):
✅ Basic market updates
✅ Stock analysis on demand

Try asking: "What's AAPL doing?" or "Find me tech stocks"

Commands:
• /upgrade - See paid plans  
• /help - All commands
• /watchlist - Add favorite stocks

Let's get started! What stock interests you?"""

    async def _handle_upgrade_command(self, user: UserProfile) -> str:
        """Handle /upgrade command with dynamic Stripe links"""
        
        # Generate fresh payment links
        paid_link = await self.stripe.create_payment_link(user._id, "paid")
        pro_link = await self.stripe.create_payment_link(user._id, "pro")
        
        return f"""💎 Unlock Advanced Features:

🥉 PAID - $29/month
✅ 100 messages/month
✅ Personalized insights
✅ Portfolio tracking  
✅ Market analytics
📲 Upgrade: {paid_link}

🏆 PRO - $99/month
✅ Unlimited messages
✅ Real-time trade alerts
✅ Advanced screeners
✅ Priority support
📲 Upgrade: {pro_link}

Questions? Reply with plan name for details."""

    async def _handle_downgrade_command(self, user: UserProfile) -> str:
        """Handle /downgrade command"""
        
        current_plan = user.plan_type.upper()
        
        if user.plan_type == "free":
            return "You're already on the FREE plan - the lowest tier available!"
        
        elif user.plan_type == "paid":
            # Paid can only downgrade to free
            return f"""Current Plan: {current_plan} ($29/month)

Downgrade Option:
🆓 FREE Plan - 10 messages/week only

⚠️ You'll lose:
❌ Personalized insights
❌ Portfolio tracking
❌ Advanced analytics

Changes take effect next billing cycle.

Confirm downgrade: /confirm-downgrade
Need help instead? /support"""
        
        elif user.plan_type == "pro":
            # Pro can downgrade to paid or free
            paid_link = await self.stripe.create_payment_link(user._id, "paid")
            
            return f"""Current Plan: {current_plan} ($99/month)

Downgrade Options:

🥉 PAID Plan - $29/month
✅ Keep: 100 messages, insights, portfolio
❌ Lose: Unlimited messages, real-time alerts
📲 Switch to Paid: {paid_link}

🆓 FREE Plan - 10 messages/week
❌ Lose: All premium features
⚠️ Confirm: /confirm-free

Changes take effect next billing cycle."""
    
    async def _handle_cancel_command(self, user: UserProfile) -> str:
        """Handle /cancel command with retention offer"""
        
        if user.plan_type == "free":
            return "You're on the FREE plan - no subscription to cancel. Account will remain active."
        
        # Generate retention offer (50% discount)
        discount_link = await self.stripe.create_discount_link(user._id, 50)  # 50% off
        cancel_link = await self.stripe.create_billing_portal_link(user.stripe_customer_id)
        
        return f"""😢 Sorry to see you go!

🎁 SPECIAL OFFER: 50% off next 3 months?
💰 Your {user.plan_type.upper()} plan for 50% off
🔗 Accept offer: {discount_link}

Or continue cancellation: {cancel_link}

⚠️ Account remains active until billing period ends
Data will be deleted after 30 days of cancellation.

What made you want to cancel? (Reply to help us improve)"""

    async def _handle_billing_command(self, user: UserProfile) -> str:
        """Handle /billing command"""
        
        if user.plan_type == "free":
            return "You're on the FREE plan - no billing information to show. Reply /upgrade to see paid options."
        
        # Get billing portal link
        portal_link = await self.stripe.create_billing_portal_link(user.stripe_customer_id)
        
        plan_config = PlanLimits.get_plan_config()[user.plan_type]
        next_billing = "Unknown"  # You'd calculate this from Stripe subscription data
        
        return f"""💳 Billing Information:

Current Plan: {user.plan_type.upper()} (${plan_config.price}/month)
Status: {user.subscription_status.title()}
Next charge: ${plan_config.price} on {next_billing}

📋 Manage billing:
• Update payment method
• Download invoices  
• View charge history
🔗 Billing Portal: {portal_link}

Questions? Reply /support"""

    async def _handle_status_command(self, user: UserProfile) -> str:
        """Handle /status command"""
        
        plan_config = PlanLimits.get_plan_config()[user.plan_type]
        
        # Get current usage
        current_usage = await self.db.get_usage_count(user._id, plan_config.period)
        
        # Usage display
        if plan_config.messages_per_period == 999999:
            usage_text = f"{current_usage}/unlimited messages"
        else:
            usage_text = f"{current_usage}/{plan_config.messages_per_period} messages"
        
        # Status emoji
        status_emoji = "✅" if user.subscription_status == "active" else "⚠️"
        
        # Connected accounts
        connected_count = len(user.plaid_access_tokens)
        
        return f"""📊 Account Status:

Plan: {user.plan_type.upper()} (${plan_config.price}/month)
Status: {status_emoji} {user.subscription_status.title()}

Usage this {plan_config.period}: {usage_text}
Connected accounts: {connected_count} brokerages

Last active: {user.last_active_at.strftime('%b %d, %Y') if user.last_active_at else 'Today'}

Manage: /billing | /settings | /upgrade"""

    async def _handle_help_command(self, user: UserProfile) -> str:
        """Handle /help command"""
        
        return """🤖 SMS Trading Bot Commands:

💰 SUBSCRIPTION:
/upgrade - See paid plans
/downgrade - Downgrade plan
/cancel - Cancel subscription  
/billing - Payment info
/status - Account overview

📊 TRADING:
/watchlist - Manage stocks
/portfolio - Portfolio summary
/screen - Find stocks
/alerts - Price alerts

⚙️ ACCOUNT:
/settings - Preferences
/support - Get help
/pause - Pause messages

💡 TIP: Just ask naturally!
"What's TSLA doing?" or "Find cheap tech stocks"

Reply any command for details."""

    async def _handle_watchlist_command(self, user: UserProfile) -> str:
        """Handle /watchlist command"""
        
        watchlist = user.watchlist or []
        
        if not watchlist:
            return """📈 Your Watchlist is empty!

Add stocks by replying:
"ADD AAPL" or "WATCH TSLA"

Or ask: "What are good tech stocks?"

PRO tip: Unlimited watchlist with PRO plan!
Current limit: 10 stocks (FREE/PAID), unlimited (PRO)"""
        
        # Format watchlist (mock prices for now)
        watchlist_text = "📈 Your Watchlist:\n\n"
        for i, symbol in enumerate(watchlist[:10], 1):
            # Mock price data - you'd get real prices from EODHD
            price = f"${150 + i * 10:.2f}"
            change = "+" if i % 2 else "-"
            change_pct = f"{change}{(i * 0.5):.1f}%"
            emoji = "🟢" if change == "+" else "🔴"
            
            watchlist_text += f"{i}. {symbol} - {price} ({change_pct}) {emoji}\n"
        
        watchlist_text += f"""
✅ Add: "ADD AMZN"
❌ Remove: "REMOVE TSLA"  
📊 Analyze: Reply symbol name

{10 - len(watchlist)} slots remaining."""
        
        return watchlist_text

    async def _handle_portfolio_command(self, user: UserProfile) -> str:
        """Handle /portfolio command"""
        
        if not user.plaid_access_tokens:
            return """💼 Portfolio Summary:

❌ No connected accounts

Connect your brokerage:
• Robinhood, TD Ameritrade, Schwab
• Read-only access via Plaid
• Secure bank-level encryption

🔗 Connect account: /connect
Or upgrade to PRO for advanced portfolio analytics!"""
        
        # Mock portfolio data - you'd get real data from Plaid
        return """💼 Portfolio Summary:

Total Value: $47,293 (+$987 today)
Day Change: +2.1% 📈

🏆 Top Performers:
• NVDA: +$523 (+4.2%)
• AAPL: +$287 (+1.8%)
• MSFT: +$145 (+0.9%)

📉 Underperformers:  
• TSLA: -$156 (-2.1%)
• SPY: -$78 (-0.5%)

🔗 Connected: 2 accounts
📊 Detailed analysis: Available in PRO plan

Need help? /support"""

    async def _handle_screen_command(self, user: UserProfile) -> str:
        """Handle /screen command"""
        
        # Mock screener results based on user profile
        risk = user.risk_tolerance
        sectors = user.preferred_sectors or ["Technology"]
        
        return f"""🔍 Smart Stock Screener:

Based on your profile ({risk.title()} risk, {', '.join(sectors)}):

💎 Today's Picks:
1. ROKU - $67.45 (Oversold, High volume)
2. SQ - $89.23 (Breaking resistance)
3. SHOP - $72.11 (Earnings beat expected)

🎯 Criteria: P/E < 25, Volume > avg, {sectors[0]} sector
📈 Historical win rate: 73% for your profile

🚀 PRO features:
• Custom screening criteria
• Real-time alerts on matches
• Technical indicator filters

Want custom criteria? /upgrade
Reply symbol for analysis."""

    async def _handle_alerts_command(self, user: UserProfile) -> str:
        """Handle /alerts command"""
        
        # Mock alerts - you'd store these in database
        return """🔔 Your Price Alerts (3 active):

1. AAPL hits $190 📈 (Currently: $185.23)
2. TSLA drops to $230 📉 (Currently: $242.45)  
3. NVDA above $800 🚀 (Currently: $789.12)

✅ Add alert: "ALERT MSFT 420"
❌ Remove: "REMOVE alert 1"
📊 List all: "ALERTS"

FREE/PAID: 5 alerts max
PRO users get:
• Unlimited alerts
• Real-time notifications  
• Technical indicator alerts
• Earnings & news alerts

Upgrade: /upgrade"""

    async def _handle_settings_command(self, user: UserProfile) -> str:
        """Handle /settings command"""
        
        # Format user preferences
        premarket_status = "✅ Enabled (9 AM)" if user.premarket_enabled else "❌ Disabled"
        close_status = "✅ Enabled (4 PM)" if user.market_close_enabled else "❌ Disabled"
        promo_status = "✅ Enabled" if user.promotional_messages else "❌ Disabled"
        
        return f"""⚙️ Account Settings:

📱 NOTIFICATIONS:
• Premarket insights: {premarket_status}
• Market close summary: {close_status}
• Breaking news: ❌ Disabled (PRO only)
• Promotional offers: {promo_status}

🎯 TRADING PROFILE:
• Risk tolerance: {user.risk_tolerance.title()}
• Experience: {user.trading_experience.title()}
• Trading style: {user.trading_style.title()}
• Timezone: {user.timezone}

📞 CONTACT:
• Phone: {user.phone_number}
• Email: {user.email or 'Not set'}

Edit settings: "CHANGE [setting name]"
Example: "CHANGE risk low" or "CHANGE email john@email.com"

Need help? /support"""

    async def _handle_support_command(self, user: UserProfile) -> str:
        """Handle /support command"""
        
        # Determine response time based on plan
        if user.plan_type == "pro":
            response_time = "<2 hours"
        elif user.plan_type == "paid":
            response_time = "<8 hours"
        else:
            response_time = "<24 hours"
        
        return f"""🆘 Support Options:

❓ QUICK HELP:
• /help - All commands
• Common issues: /faq
• Video tutorials: Coming soon

📞 CONTACT SUPPORT:
• Reply with your issue description
• Email: support@tradingbot.com  
• Response time: {response_time}

🐛 REPORT ISSUES:
• Bug report: "BUG: [description]"
• Feature request: "FEATURE: [idea]"
• Billing issue: "BILLING: [problem]"

💎 Priority support included with PRO plan!

How can we help you today?"""

    async def _handle_pause_command(self, user: UserProfile) -> str:
        """Handle /pause command"""
        
        # Update user preferences to disable notifications
        user.daily_insights_enabled = False
        user.premarket_enabled = False
        user.market_close_enabled = False
        await self.db.save_user(user)
        
        return """⏸️ Account Paused

✅ All automated messages stopped
✅ Subscription remains active
✅ Usage limits preserved
✅ Can still send manual queries

To resume: Reply "RESUME" anytime

⚠️ Note: Still charged normally while paused
Need to cancel instead? Reply /cancel"""

    async def _handle_resume_command(self, user: UserProfile) -> str:
        """Handle /resume command"""
        
        # Re-enable notifications
        user.daily_insights_enabled = True
        user.premarket_enabled = True
        user.market_close_enabled = True
        await self.db.save_user(user)
        
        return """▶️ Account Resumed!

✅ Daily insights restored (9 AM & 4 PM)
✅ All notifications active
✅ Welcome back!

📈 While you were away:
• Market moved +1.2% this week
• Your watchlist avg: +0.8%
• 2 new opportunities detected

Ready to trade? Ask me anything!"""

    async def _handle_stop_command(self, user: UserProfile) -> str:
        """Handle STOP/unsubscribe command"""
        
        # Disable promotional messages only
        user.promotional_messages = False
        await self.db.save_user(user)
        
        return """✋ Promotional messages stopped.

You'll still receive:
✅ Daily insights (if subscribed)
✅ Responses to your questions
✅ Account notifications

❌ No more promotional offers

To resume promos: Reply "START PROMOS"
To pause everything: Reply /pause
To cancel subscription: Reply /cancel"""
