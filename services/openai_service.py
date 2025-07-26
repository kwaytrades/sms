# ===== services/openai_service.py - MINIMAL VERSION =====
from loguru import logger
from config import settings
from typing import Dict, List

class OpenAIService:
    def __init__(self):
        self.api_key = settings.openai_api_key
        self.client = None
        logger.info("✅ OpenAI service initialized (testing mode)")
        
    async def generate_personalized_response(
        self, 
        user_query: str, 
        user_profile: Dict, 
        conversation_history: List[Dict],
        market_context: Dict = None
    ) -> str:
        """Generate AI response - fallback version"""
        
        if not self.api_key:
            return self._get_smart_fallback_response(user_query)
        
        # TODO: Implement actual OpenAI integration
        return f"AI response to: {user_query} (OpenAI integration pending)"
    
    def _get_smart_fallback_response(self, user_query: str) -> str:
        """Smart fallback responses when OpenAI unavailable"""
        query_lower = user_query.lower()
        
        if any(word in query_lower for word in ["start", "hello", "hi"]):
            return "Welcome to SMS Trading Bot! Ask me about any stock symbol like AAPL or TSLA."
        
        elif any(word in query_lower for word in ["help", "commands"]):
            return "Commands: START, HELP, UPGRADE, STATUS. Or ask about any stock!"
        
        elif any(word in query_lower for word in ["upgrade", "plans"]):
            return "Plans: FREE (10 msgs/week), PAID ($29/mo), PRO ($99/mo). Reply UPGRADE for details."
        
        elif any(word in query_lower for word in ["aapl", "apple"]):
            return "AAPL: $185.50 (+1.2%). Strong momentum above 200-day MA. RSI: 65 (neutral)."
        
        elif any(word in query_lower for word in ["tsla", "tesla"]):
            return "TSLA: $242.75 (-0.8%). Consolidating near support. Volume below average."
        
        elif any(word in query_lower for word in ["spy", "market"]):
            return "SPY: $487.20 (+0.5%). Market showing strength above key support levels."
        
        else:
            return "I'm your trading assistant! Ask about stocks, get analysis, or say HELP for commands."
