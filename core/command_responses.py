# ===== core/command_responses.py =====
"""
Pre-defined response templates for commands
"""

class CommandResponses:
    """Static response templates for commands"""
    
    @staticmethod
    def welcome_new_user(name: str = "there") -> str:
        return f"""🚀 Welcome to AI Trading Insights, {name}!

You're on our FREE trial (10 messages/week):
✅ Basic market updates
✅ Stock analysis on demand

Try asking: "What's AAPL doing?" or "Find me tech stocks"

Commands:
• /upgrade - See paid plans
• /help - All commands  
• /watchlist - Add favorite stocks

Let's get started! What stock interests you?"""
    
    @staticmethod
    def upgrade_options(paid_link: str, pro_link: str) -> str:
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
    
    @staticmethod
    def command_not_found() -> str:
        return """❓ Unknown command.

Reply '/help' for all available commands.

Or just ask naturally:
• "What's AAPL doing?"
• "Find me tech stocks"
• "How's my portfolio?"

I'm here to help! 🤖"""
    
    @staticmethod
    def feature_not_available(feature: str, required_plan: str) -> str:
        return f"""🔒 {feature} requires {required_plan.upper()} plan

Current plan: FREE
Upgrade for access to:
✅ {feature}
✅ Advanced analytics
✅ Priority support

Reply /upgrade to see options!"""
