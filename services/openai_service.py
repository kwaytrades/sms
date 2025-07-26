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
                    max_tokens=500,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
            else:
                # Legacy OpenAI API
                response = await self.client.ChatCompletion.acreate(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ OpenAI generation failed: {e}")
            return self._get_smart_fallback_response(user_query)
    
    def _get_smart_fallback_response(self, user_query: str) -> str:
        """Generate intelligent fallback responses when OpenAI fails"""
        query_lower = user_query.lower()
        
        # Stock/trading related queries
        if any(word in query_lower for word in ["buy", "sell", "stock", "ticker", "price"]):
            return """📈 AI temporarily unavailable. Quick trading tips:

• Research before buying/selling
• Use stop-losses to limit risk
• Don't invest more than you can lose
• Check multiple sources for stock info

Try your question again in a few minutes! 🚀"""
        
        # Market analysis queries
        elif any(word in query_lower for word in ["market", "analysis", "trend", "outlook"]):
            return """📊 Market analysis temporarily offline.

General guidance:
• Focus on long-term trends over daily noise
• Diversify your portfolio across sectors
• Stay updated with financial news
• Consider your risk tolerance

I'll be back with full analysis soon! 💪"""
        
        # Crypto queries
        elif any(word in query_lower for word in ["bitcoin", "crypto", "btc", "eth", "ethereum"]):
            return """₿ Crypto insights temporarily down.

Remember:
• Crypto is highly volatile
• Only invest what you can afford to lose
• Do your own research (DYOR)
• Consider dollar-cost averaging

Full crypto analysis returning soon! ⚡"""
        
        # Portfolio queries
        elif any(word in query_lower for word in ["portfolio", "allocation", "diversification"]):
            return """💼 Portfolio analysis temporarily unavailable.

Basic principles:
• Diversify across asset classes
• Rebalance regularly
• Match investments to your goals
• Review and adjust quarterly

Detailed portfolio help coming back online! 📋"""
        
        # General trading questions
        else:
            return """🤖 AI trading assistant temporarily offline.

While I recover:
• Use /help for available commands
• Check reputable financial news sites
• Consult with licensed financial advisors
• Practice good risk management

I'll be back with smart insights soon! 🔧"""
    
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
        
        # Determine response length
        length = "concise (under 160 characters)" if message_length < 50 else "detailed but focused"
        
        system_prompt = f"""You are a personalized trading assistant. Your communication style should be {style} and {length}.

User Profile:
- Plan: {user_profile.get('plan_type', 'free')}
- Experience: {user_profile.get('trading_experience', 'intermediate')}
- Risk tolerance: {user_profile.get('risk_tolerance', 'medium')}
- Preferred sectors: {', '.join(user_profile.get('preferred_sectors', []))}
- Trading style: {user_profile.get('trading_style', 'swing')}

Always:
- Match their communication style preferences
- Reference their portfolio/watchlist when relevant
- Provide actionable insights
- Include appropriate risk warnings
- Use their preferred level of technical detail
- Keep responses under 1500 characters for SMS compatibility
"""
        
        user_prompt = f"User query: {user_query}"
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
