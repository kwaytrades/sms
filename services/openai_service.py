# ===== services/openai_service.py =====
import os
from typing import Dict, List
from loguru import logger
from config import settings

class OpenAIService:
    def __init__(self):
        """Initialize OpenAI service with fallback compatibility"""
        self.api_key = settings.openai_api_key
        self.client = None
        self._init_client()
        
    def _init_client(self):
        """Initialize OpenAI client with version compatibility"""
        try:
            # Try new OpenAI v1.0+ approach
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.use_new_api = True
            logger.info("✅ Using OpenAI v1.0+ client")
        except Exception as e:
            logger.warning(f"⚠️ New OpenAI client failed: {e}")
            try:
                # Fallback to legacy approach
                import openai
                openai.api_key = self.api_key
                self.client = openai
                self.use_new_api = False
                logger.info("✅ Using legacy OpenAI client")
            except Exception as e2:
                logger.error(f"❌ All OpenAI initialization methods failed: {e2}")
                self.client = None
        
    async def generate_personalized_response(
        self, 
        user_query: str, 
        user_profile: Dict, 
        conversation_history: List[Dict],
        market_context: Dict = None
    ) -> str:
        """Generate personalized AI response"""
        
        # Return fallback if no client available
        if self.client is None:
            return self._get_smart_fallback_response(user_query)
        
        try:
            # Build personalized prompt
            prompt = self._build_personalized_prompt(
                user_query, user_profile, conversation_history, market_context
            )
            
            messages = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]}
            ]
            
            if self.use_new_api:
                # New OpenAI v1.0+ API
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=120,  # ~450 characters = 3 SMS segments
                    temperature=0.7
                )
                ai_response = response.choices[0].message.content.strip()
            else:
                # Legacy OpenAI API
                response = await self.client.ChatCompletion.acreate(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=120,  # ~450 characters = 3 SMS segments
                    temperature=0.7
                )
                ai_response = response.choices[0].message.content.strip()
            
            # Enforce cost-optimized limits
            if len(ai_response) > 450:
                ai_response = ai_response[:447] + "..."
            
            # Smart emoji optimization for SMS costs
            ai_response = self._optimize_sms_cost(ai_response)
            
            return ai_response
            
        except Exception as e:
            logger.error(f"❌ OpenAI generation failed: {e}")
            return self._get_smart_fallback_response(user_query)
    
    def _optimize_sms_cost(self, message: str) -> str:
        """Optimize message for SMS cost efficiency"""
        import re
        
        # If message is short enough, allow minimal emoji for engagement
        if len(message) <= 65:  # Leave room for one emoji
            # Only allow one emoji at the end for short messages
            if not re.search(r'[^\x00-\x7F]', message):  # No emojis present
                # Add single emoji for engagement on short messages
                if any(word in message.lower() for word in ['buy', 'bullish', 'strong', 'growth']):
                    return message + " 📈"
                elif any(word in message.lower() for word in ['sell', 'bearish', 'weak', 'decline']):
                    return message + " 📉"
                elif any(word in message.lower() for word in ['watch', 'monitor', 'alert']):
                    return message + " 👀"
                else:
                    return message + " 💡"
            return message
        
        # For longer messages, remove all emojis to use cheaper encoding
        # Remove all emoji/unicode characters
        message_clean = re.sub(r'[^\x00-\x7F]+', '', message)
        # Clean up extra spaces
        message_clean = ' '.join(message_clean.split())
        
        return message_clean
    
    def _get_smart_fallback_response(self, user_query: str) -> str:
        """Generate intelligent fallback responses when OpenAI fails - cost optimized"""
        query_lower = user_query.lower()
        
        # Stock/trading related queries - no emojis for longer responses
        if any(word in query_lower for word in ["buy", "sell", "stock", "ticker", "price"]):
            return """AI offline. Key trading tips: Research thoroughly before buying/selling, use stop-losses to limit downside risk, never invest more than you can afford to lose, check multiple sources for stock data. Try your question again in a few minutes!"""
        
        # Market analysis queries
        elif any(word in query_lower for word in ["market", "analysis", "trend", "outlook"]):
            return """Market analysis temporarily down. Focus on long-term trends over daily noise, diversify your portfolio across sectors, stay updated with financial news, consider your personal risk tolerance. Full analysis returning soon!"""
        
        # Crypto queries
        elif any(word in query_lower for word in ["bitcoin", "crypto", "btc", "eth", "ethereum"]):
            return """Crypto insights offline. Remember: crypto is highly volatile, only invest what you can afford to lose, do your own research (DYOR), consider dollar-cost averaging for major coins. Full crypto analysis back soon!"""
        
        # Portfolio queries
        elif any(word in query_lower for word in ["portfolio", "allocation", "diversification"]):
            return """Portfolio analysis offline. Key principles: diversify across asset classes & geographies, rebalance quarterly, match investments to your goals & timeline, review performance regularly. Detailed help returning!"""
        
        # Short general response - can use emoji
        else:
            return """AI temporarily offline. Use /help for commands. Back soon! 🔧"""
    
    def _build_personalized_prompt(
        self, 
        user_query: str, 
        user_profile: Dict,
        conversation_history: List[Dict],
        market_context: Dict = None
    ) -> Dict[str, str]:
        """Build personalized prompt based on user patterns"""
        
        # Analyze user's communication style
        comm_style = user_profile.get("communication_style", {})
        technical_pref = comm_style.get("technical_preference", 0.5)
        message_length = comm_style.get("avg_message_length", 100)
        
        # Determine response style
        if technical_pref > 0.7:
            style = "technical and analytical with specific indicators and metrics"
        elif technical_pref < 0.3:
            style = "casual and easy to understand with minimal jargon"
        else:
            style = "balanced between technical accuracy and accessibility"
        
        # Determine response length - optimize for SMS cost efficiency
        length = "detailed but concise (under 450 characters - max 3 SMS segments)" if message_length < 50 else "comprehensive but focused (under 450 characters - max 3 SMS segments)"
        
        system_prompt = f"""You are a personalized trading assistant. Your communication style should be {style} and {length}.

IMPORTANT SMS COST RULES:
- Keep responses under 450 characters for cost efficiency
- AVOID emojis unless response is under 70 characters (saves 50% SMS costs)
- Use clear, professional language without emoji decoration
- Focus on valuable trading insights and context

User Profile:
- Plan: {user_profile.get('plan_type', 'free')}
- Experience: {user_profile.get('trading_experience', 'intermediate')}
- Risk tolerance: {user_profile.get('risk_tolerance', 'medium')}

Always:
- Give actionable trading insights with context
- Include relevant risk warnings
- Use efficient but professional language
- NO emojis unless message is very short (under 70 chars)
- Prioritize value and clarity
"""
        
        user_prompt = f"User query: {user_query}"
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
