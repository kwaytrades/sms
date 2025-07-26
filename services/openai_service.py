# services/openai_service.py - Updated for Unified System
import os
from typing import Dict, List, Optional, Any
from loguru import logger
from config import settings

class OpenAIService:
    def __init__(self):
        self.api_key = settings.openai_api_key
        self.client = None
        self._init_client()
        
    def _init_client(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            if self.api_key:
                self.client = OpenAI(api_key=self.api_key)
                logger.info("✅ OpenAI client initialized")
            else:
                logger.warning("⚠️ OpenAI API key not configured")
        except Exception as e:
            logger.error(f"❌ OpenAI initialization failed: {e}")
            self.client = None
        
    async def generate_personalized_response(
        self, 
        user_query: str, 
        user_profile: Optional[Dict] = None,
        market_data: Optional[Dict] = None,
        conversation_history: Optional[List] = None
    ) -> str:
        """Generate personalized AI response"""
        
        if not self.client:
            return self._get_fallback_response(user_query)
        
        try:
            # Build context-aware prompt
            system_prompt = self._build_system_prompt(user_profile, market_data)
            user_prompt = self._build_user_prompt(user_query, market_data)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=120,  # ~450 chars = 3 SMS segments max
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Optimize for SMS cost (remove emojis for longer messages)
            if len(ai_response) > 70:
                ai_response = self._remove_emojis(ai_response)
            
            return ai_response[:450]  # Hard limit for cost control
            
        except Exception as e:
            logger.error(f"❌ OpenAI generation failed: {e}")
            return self._get_fallback_response(user_query)
    
    def _build_system_prompt(self, user_profile: Optional[Dict], market_data: Optional[Dict]) -> str:
        """Build personalized system prompt"""
        base_prompt = """You are a personalized SMS trading assistant. Keep responses under 450 characters for cost efficiency.

CRITICAL SMS RULES:
- Maximum 450 characters total
- NO emojis unless message is very short (<70 chars)
- Focus on actionable insights
- Use professional but friendly tone"""
        
        if user_profile:
            experience = user_profile.get('trading_experience', 'intermediate')
            style = user_profile.get('communication_style', {})
            technical_depth = style.get('technical_depth', 'medium')
            
            base_prompt += f"""

USER CONTEXT:
- Experience: {experience}
- Technical depth: {technical_depth}
- Communication: {style.get('formality', 'casual')}"""
        
        if market_data and not market_data.get('error'):
            base_prompt += """

MARKET DATA AVAILABLE: Use the provided technical analysis to give specific insights about price, indicators, and signals."""
        
        return base_prompt
    
    def _build_user_prompt(self, user_query: str, market_data: Optional[Dict]) -> str:
        """Build user prompt with market data"""
        prompt = f"User query: {user_query}"
        
        if market_data and not market_data.get('error'):
            # Include key market data
            symbol = market_data.get('symbol', 'STOCK')
            price = market_data.get('current_price', 'N/A')
            change = market_data.get('price_change', {})
            indicators = market_data.get('indicators', {})
            signals = market_data.get('signals', [])
            
            prompt += f"""

MARKET DATA for {symbol}:
Price: ${price} ({change.get('percent', 0):+.1f}%)
"""
            
            # Add key indicators
            if indicators.get('rsi'):
                rsi = indicators['rsi']
                prompt += f"RSI: {rsi.get('value', 'N/A')} ({rsi.get('signal', 'neutral')})\n"
            
            if indicators.get('macd'):
                macd = indicators['macd']
                prompt += f"MACD: {macd.get('trend', 'neutral')}\n"
            
            # Add key signals
            if signals:
                key_signals = [s for s in signals[:2] if s.get('type') in ['bullish', 'bearish', 'warning', 'opportunity']]
                if key_signals:
                    prompt += f"Key signals: {', '.join(s.get('message', '') for s in key_signals)}"
        
        return prompt
    
    def _get_fallback_response(self, user_query: str) -> str:
        """Smart fallback when OpenAI is unavailable"""
        query_lower = user_query.lower()
        
        if any(word in query_lower for word in ["buy", "sell", "price", "stock"]):
            return "AI offline. Remember: research thoroughly, use stop-losses, never invest more than you can afford to lose. Try again in a few minutes."
        
        elif any(word in query_lower for word in ["rsi", "macd", "technical"]):
            return "Technical analysis offline. Key rules: RSI >70 = overbought, <30 = oversold. MACD crossovers signal trend changes. Back soon!"
        
        elif any(word in query_lower for word in ["market", "outlook", "trend"]):
            return "Market analysis offline. Focus on long-term trends, diversify your portfolio, stay informed with multiple sources. Full analysis returning soon."
        
        else:
            return "AI temporarily offline. Use 'help' for commands. Back soon!"
    
    def _remove_emojis(self, text: str) -> str:
        """Remove emojis to save SMS costs"""
        import re
        # Remove emoji/unicode characters
        return re.sub(r'[^\x00-\x7F]+', '', text).strip()
