# core/message_handler.py - Unified Message Handler
from typing import Dict, List, Optional, Any
import uuid
import re
from datetime import datetime, timezone
from loguru import logger

from services.database import DatabaseService
from services.openai_service import OpenAIService
from services.twilio_service import TwilioService
from services.technical_analysis import TechnicalAnalysisService
from core.user_manager import UserManager
from config import PLAN_LIMITS

class MessageHandler:
    def __init__(self, db: DatabaseService, openai: OpenAIService, 
                 twilio: TwilioService, ta_service: TechnicalAnalysisService,
                 user_manager: UserManager):
        self.db = db
        self.openai = openai
        self.twilio = twilio
        self.ta_service = ta_service
        self.user_manager = user_manager
        self.intent_analyzer = IntentAnalyzer()
    
    async def process_incoming_message(self, phone_number: str, message_body: str) -> bool:
        """Process incoming SMS message with full business logic"""
        try:
            logger.info(f"📱 Processing message from {phone_number}: {message_body[:50]}...")
            
            # Get or create user
            user = await self.user_manager.get_or_create_user(phone_number)
            
            # Check if it's a command
            if self._is_command(message_body):
                await self._handle_command(user, message_body)
                return True
            
            # Check message limits
            limit_check = await self.user_manager.check_message_limits(phone_number)
            if not limit_check["can_send"]:
                await self._send_limit_message(user, limit_check)
                return True
            
            # Update user activity for received message
            await self.user_manager.update_user_activity(phone_number, "received")
            
            # Analyze message intent
            intent = self.intent_analyzer.analyze_message(message_body)
            
            # Learn from interaction
            await self.user_manager.learn_from_interaction(
                phone_number, message_body, intent.symbols, intent.action
            )
            
            # Generate response based on intent
            response = await self._generate_response(user, message_body, intent)
            
            # Send response
            success = await self.twilio.send_sms(phone_number, response)
            
            if success:
                # Update user activity for sent message
                await self.user_manager.update_user_activity(phone_number, "sent")
                
                # Save conversation
                await self.db.save_conversation(
                    phone_number, message_body, response, intent.action, intent.symbols
                )
                
                logger.info(f"✅ Successfully processed message for {phone_number}")
                return True
            else:
                logger.error(f"❌ Failed to send SMS to {phone_number}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error processing message from {phone_number}: {e}")
            
            # Send error message to user
            error_msg = "Sorry, I'm having technical difficulties. Please try again in a moment!"
            await self.twilio.send_sms(phone_number, error_msg)
            return False
    
    def _is_command(self, message: str) -> bool:
        """Check if message is a command"""
        message = message.strip().lower()
        commands = [
            '/start', '/help', '/upgrade', '/downgrade', '/cancel', '/billing',
            '/status', '/watchlist', '/portfolio', '/screen', '/alerts',
            '/settings', '/support', '/pause', '/resume', 'start', 'stop',
            'help', 'upgrade', 'cancel'
        ]
        return message in commands or message.startswith('/')
    
    async def _handle_command(self, user: Dict[str, Any], command: str) -> bool:
        """Handle SMS commands"""
        try:
            command = command.strip().lower()
            phone_number = user["phone_number"]
            
            # Route to specific command handlers
            if command in ["start", "/start", "begin"]:
                response = await self._handle_start_command(user)
            elif command in ["/help", "help"]:
                response = await self._handle_help_command(user)
            elif command in ["/upgrade", "upgrade"]:
                response = await self._handle_upgrade_command(user)
            elif command in ["/status", "status"]:
                response = await self._handle_status_command(user)
            elif command in ["/watchlist", "watchlist"]:
                response = await self._handle_watchlist_command(user)
            elif command in ["/cancel", "cancel"]:
                response = await self._handle_cancel_command(user)
            elif command in ["stop", "unsubscribe"]:
                response = await self._handle_stop_command(user)
            else:
                response = f"Unknown command: {command}. Reply 'help' for available commands."
            
            # Send response
            await self.twilio.send_sms(phone_number, response)
            return True
            
        except Exception as e:
            logger.error(f"❌ Command handling failed: {e}")
            return False
    
    async def _generate_response(self, user: Dict[str, Any], message: str, intent: 'MessageIntent') -> str:
        """Generate AI response based on intent and user profile"""
        try:
            # Handle different intent types
            if intent.action == "analyze" and intent.symbols:
                return await self._handle_stock_analysis(user, intent.symbols[0], message)
            
            elif intent.action == "screener":
                return await self._handle_stock_screener(user, intent.parameters)
            
            elif intent.action == "price" and intent.symbols:
                return await self._handle_price_query(user, intent.symbols[0])
            
            elif intent.action == "news" and intent.symbols:
                return await self._handle_news_query(user, intent.symbols[0])
            
            elif intent.action == "compare" and len(intent.symbols) >= 2:
                return await self._handle_comparison(user, intent.symbols[:2])
            
            else:
                return await self._handle_general_query(user, message, intent)
            
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            return "I'm having trouble processing your request. Please try asking about a specific stock ticker like 'How is AAPL doing?'"
    
    async def _handle_stock_analysis(self, user: Dict[str, Any], symbol: str, original_message: str) -> str:
        """Handle stock analysis requests"""
        try:
            # Get technical analysis
            analysis = await self.ta_service.analyze_symbol(symbol)
            
            if "error" in analysis:
                return f"Sorry, I couldn't get data for {symbol}. Please verify the ticker symbol and try again."
            
            # Get user profile for personalization
            user_profile = await self.user_manager.get_user_for_analysis(user["phone_number"])
            
            # Generate personalized response using OpenAI
            response = await self.openai.generate_personalized_response(
                user_query=original_message,
                user_profile=user_profile,
                market_data=analysis,
                conversation_history=[]
            )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Stock analysis failed for {symbol}: {e}")
            return f"Sorry, I encountered an error analyzing {symbol}. Please try again later."
    
    async def _handle_price_query(self, user: Dict[str, Any], symbol: str) -> str:
        """Handle simple price queries"""
        try:
            analysis = await self.ta_service.analyze_symbol(symbol)
            
            if "error" in analysis:
                return f"Sorry, I couldn't get the price for {symbol}."
            
            price = analysis["current_price"]
            change = analysis["price_change"]
            direction = "↑" if change["amount"] > 0 else "↓" if change["amount"] < 0 else "→"
            
            return f"{symbol}: ${price} {direction}{change['percent']:.1f}% ({change['amount']:+.2f})"
            
        except Exception as e:
            logger.error(f"❌ Price query failed for {symbol}: {e}")
            return f"Sorry, I couldn't get the price for {symbol}."
    
    async def _handle_stock_screener(self, user: Dict[str, Any], parameters: Dict[str, Any]) -> str:
        """Handle stock screening requests"""
        try:
            # This would integrate with a stock screener API
            # For now, provide a helpful response
            user_sectors = user.get("preferred_sectors", [])
            
            response = "🔍 Stock Screener coming soon! "
            
            if "growth" in parameters or any("growth" in str(v).lower() for v in parameters.values()):
                response += "For growth stocks, consider looking at: NVDA, AMD, TSLA. "
            elif "dividend" in parameters or any("dividend" in str(v).lower() for v in parameters.values()):
                response += "For dividend stocks, consider: JNJ, PG, KO. "
            elif "value" in parameters or any("value" in str(v).lower() for v in parameters.values()):
                response += "For value stocks, consider: BRK.B, JPM, WMT. "
            else:
                response += "Try asking: 'Find me growth stocks' or 'What are good dividend stocks?'"
            
            if user_sectors:
                response += f" Based on your interest in {', '.join(user_sectors[:2])}, you might also like specific sector analysis."
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Stock screener failed: {e}")
            return "Sorry, the stock screener is temporarily unavailable. Try asking about specific stocks instead."
    
    async def _handle_general_query(self, user: Dict[str, Any], message: str, intent: 'MessageIntent') -> str:
        """Handle general trading questions"""
        try:
            user_profile = await self.user_manager.get_user_for_analysis(user["phone_number"])
            
            response = await self.openai.generate_personalized_response(
                user_query=message,
                user_profile=user_profile,
                market_data=None,
                conversation_history=[]
            )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ General query failed: {e}")
            return "I'm here to help with trading questions! Try asking about a specific stock like 'How is AAPL doing?' or 'What's the latest on Tesla?'"
    
    # Command handlers
    
    async def _handle_start_command(self, user: Dict[str, Any]) -> str:
        """Handle START command"""
        name = user.get("first_name", "there")
        plan = user.get("plan_type", "free")
        plan_config = PLAN_LIMITS[plan]
        
        return f"""🚀 Welcome to AI Trading Insights, {name}!

You're on the {plan.upper()} plan ({plan_config['weekly_limit']} messages/week):
{chr(10).join('✅ ' + feature for feature in plan_config['features'])}

Try asking:
• "How is AAPL doing?"
• "Find me tech stocks"
• "What's Tesla's RSI?"

Commands: /help /upgrade /status /watchlist

Ready to analyze the markets? 📈"""
    
    async def _handle_help_command(self, user: Dict[str, Any]) -> str:
        """Handle HELP command"""
        return """🤖 AI Trading Bot Commands:

💰 SUBSCRIPTION:
/upgrade - See paid plans
/cancel - Cancel subscription
/status - Account overview

📊 TRADING:
/watchlist - Manage stocks
/screen - Find stocks (coming soon)

⚙️ ACCOUNT:
/settings - Preferences (coming soon)
/support - Get help

💡 NATURAL LANGUAGE:
Just ask! "How's AAPL?" or "Find cheap tech stocks"

Reply any command for details."""
    
    async def _handle_upgrade_command(self, user: Dict[str, Any]) -> str:
        """Handle UPGRADE command"""
        current_plan = user.get("plan_type", "free")
        
        if current_plan == "pro":
            return "You're already on our highest tier (PRO)! Enjoy unlimited messages and premium features. 💎"
        
        plans_text = """💎 Upgrade Your Trading Analysis:

🥉 PAID - $29/month
✅ 40 messages/week (10x more!)
✅ Personalized insights
✅ Portfolio tracking
✅ Advanced analysis

🏆 PRO - $99/month  
✅ 120 messages/week (30x more!)
✅ Real-time alerts
✅ Priority support
✅ Advanced screeners

Reply 'PAID' or 'PRO' to upgrade, or visit our website for secure payment."""
        
        return plans_text
    
    async def _handle_status_command(self, user: Dict[str, Any]) -> str:
        """Handle STATUS command"""
        plan = user.get("plan_type", "free")
        plan_config = PLAN_LIMITS[plan]
        
        # Get current usage
        weekly_usage = await self.db.get_weekly_usage(user["phone_number"])
        
        limit = plan_config["weekly_limit"]
        remaining = max(0, limit - weekly_usage)
        
        status_emoji = "✅" if user.get("subscription_status") == "active" else "⚠️"
        
        return f"""📊 Account Status:

Plan: {plan.upper()} (${plan_config['price']}/month)
Status: {status_emoji} {user.get('subscription_status', 'active').title()}

Usage this week: {weekly_usage}/{limit} messages
Remaining: {remaining} messages

Last active: {user.get('last_active_at', datetime.now()).strftime('%b %d') if user.get('last_active_at') else 'Today'}

Manage: /upgrade | /cancel"""
    
    async def _handle_watchlist_command(self, user: Dict[str, Any]) -> str:
        """Handle WATCHLIST command"""
        watchlist = user.get("watchlist", [])
        
        if not watchlist:
            return """📈 Your Watchlist is empty!

Add stocks by replying:
"ADD AAPL" or "WATCH TSLA"

Or ask: "How is Apple doing?" and I'll analyze it for you.

Popular stocks: AAPL, TSLA, NVDA, AMZN, GOOGL"""
        
        watchlist_text = "📈 Your Watchlist:\n\n"
        for i, symbol in enumerate(watchlist[:10], 1):
            watchlist_text += f"{i}. {symbol}\n"
        
        watchlist_text += f"""
✅ Add: "ADD MSFT"
❌ Remove: "REMOVE TSLA"  
📊 Analyze: Just reply with the symbol

{10 - len(watchlist)} slots remaining."""
        
        return watchlist_text
    
    async def _handle_cancel_command(self, user: Dict[str, Any]) -> str:
        """Handle CANCEL command"""
        plan = user.get("plan_type", "free")
        
        if plan == "free":
            return "You're on the FREE plan - no subscription to cancel. Your account will remain active with basic features."
        
        return f"""😢 Sorry to see you consider leaving!

Current plan: {plan.upper()}
Subscription will remain active until your billing period ends.

🎁 RETENTION OFFER: 50% off your next billing cycle?

To proceed with cancellation, please contact support at:
📧 support@tradingbot.com
📱 Reply 'SUPPORT' for help

What can we do to improve your experience?"""
    
    async def _handle_stop_command(self, user: Dict[str, Any]) -> str:
        """Handle STOP command"""
        # Update user to disable promotional messages
        await self.user_manager.user_manager.db.db.users.update_one(
            {"phone_number": user["phone_number"]},
            {"$set": {"promotional_messages": False}}
        )
        
        return """✋ You've been unsubscribed from promotional messages.

You'll still receive:
✅ Responses to your questions
✅ Account notifications

To resume promotions: Reply "START PROMOS"
To pause everything: Reply "/pause"
To cancel subscription: Reply "/cancel"

Thanks for using our service!"""
    
    async def _send_limit_message(self, user: Dict[str, Any], limit_info: Dict[str, Any]) -> bool:
        """Send appropriate limit exceeded message"""
        try:
            phone_number = user["phone_number"]
            reason = limit_info["reason"]
            plan = limit_info.get("plan", "free")
            
            if reason == "Subscription inactive":
                message = """⚠️ Your subscription is inactive.

Reactivate now to continue getting AI trading insights:
Reply /upgrade to see plans"""
                
            elif "limit exceeded" in reason.lower() or "cooloff" in reason.lower():
                used = limit_info.get("used", 0)
                limit = limit_info.get("limit", 0)
                upgrade_msg = limit_info.get("upgrade_message", "")
                
                if "daily cooloff" in reason.lower():
                    message = f"⏸️ Daily cooloff active (Pro plan)\n\nUsed {used}/50 messages today. Resets at midnight EST.\n\nYou're on our highest tier! 💎"
                else:
                    message = f"📊 Weekly limit reached!\n\nUsed: {used}/{limit} messages\nResets: Monday 9:30 AM EST\n\n{upgrade_msg}\n\nReply /upgrade for more messages!"
            else:
                message = "⚠️ Message limit reached. Reply /upgrade for more analysis!"
            
            return await self.twilio.send_sms(phone_number, message)
            
        except Exception as e:
            logger.error(f"❌ Error sending limit message: {e}")
            return False

class MessageIntent:
    """Data class for message intent analysis"""
    def __init__(self, action: str, symbols: List[str], parameters: Dict[str, Any], confidence: float):
        self.action = action
        self.symbols = symbols
        self.parameters = parameters
        self.confidence = confidence

class IntentAnalyzer:
    """Analyze user message intent"""
    
    def __init__(self):
        self.ticker_pattern = re.compile(r'\b([A-Z]{1,5})\b')
        
        self.intent_keywords = {
            'analyze': ['analyze', 'analysis', 'look at', 'check', 'what about', 'how is', 'tell me about', 'thoughts on'],
            'price': ['price', 'cost', 'trading at', 'current', 'quote', 'worth'],
            'technical': ['rsi', 'macd', 'support', 'resistance', 'technical', 'indicators', 'chart', 'bollinger'],
            'screener': ['find', 'search', 'screen', 'discover', 'suggest', 'recommend', 'good stocks'],
            'news': ['news', 'updates', 'happened', 'events', 'earnings'],
            'compare': ['vs', 'versus', 'compare', 'better than', 'against']
        }
    
    def analyze_message(self, message: str) -> MessageIntent:
        """Analyze message to determine user intent"""
        message_lower = message.lower()
        symbols = self._extract_symbols(message)
        
        # Score each intent
        intent_scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Determine primary intent
        if not intent_scores:
            primary_intent = 'general'
            confidence = 0.3
        else:
            primary_intent = max(intent_scores, key=intent_scores.get)
            confidence = min(intent_scores[primary_intent] / len(message_lower.split()) * 5, 1.0)
        
        # Extract parameters based on intent
        parameters = {}
        if primary_intent == 'screener':
            parameters = self._extract_screener_parameters(message_lower)
        
        return MessageIntent(
            action=primary_intent,
            symbols=symbols,
            parameters=parameters,
            confidence=confidence
        )
    
    def _extract_symbols(self, message: str) -> List[str]:
        """Extract stock symbols from message"""
        # Common false positives to exclude
        exclude_words = {
            'TO', 'AT', 'IN', 'ON', 'OR', 'OF', 'IS', 'IT', 'BE', 'DO', 'GO', 
            'UP', 'MY', 'AI', 'ALL', 'AND', 'FOR', 'THE', 'YOU', 'CAN', 'GET',
            'NEW', 'NOW', 'OLD', 'SEE', 'WAY', 'WHO', 'BOY', 'DID', 'ITS',
            'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE', 'HER', 'WAS', 'ONE',
            'OUR', 'HAD', 'BUT', 'NOT', 'MAY'
        }
        
        potential_symbols = self.ticker_pattern.findall(message.upper())
        return [symbol for symbol in potential_symbols 
                if symbol not in exclude_words and len(symbol) >= 2]
    
    def _extract_screener_parameters(self, message: str) -> Dict[str, Any]:
        """Extract screening criteria from message"""
        parameters = {}
        
        if any(word in message for word in ['cheap', 'undervalued', 'value']):
            parameters['strategy'] = 'value'
        elif any(word in message for word in ['growth', 'growing']):
            parameters['strategy'] = 'growth'
        elif any(word in message for word in ['dividend', 'income']):
            parameters['strategy'] = 'dividend'
        elif any(word in message for word in ['tech', 'technology']):
            parameters['sector'] = 'technology'
        elif any(word in message for word in ['healthcare', 'pharma']):
            parameters['sector'] = 'healthcare'
        
        return parameters
