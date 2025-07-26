# ===== services/openai_service.py - FIXED VERSION =====
from loguru import logger
from config import settings
from typing import Dict, List, Optional
import openai
import asyncio

class OpenAIService:
    def __init__(self):
        self.api_key = settings.openai_api_key
        if self.api_key:
            openai.api_key = self.api_key
            self.client = openai
            logger.info("✅ OpenAI service initialized with API key")
        else:
            self.client = None
            logger.warning("⚠️ OpenAI API key not found, using fallback responses")
        
    async def generate_personalized_response(
        self, 
        user_query: str, 
        user_profile: Dict, 
        conversation_history: List[Dict],
        market_context: Dict = None
    ) -> str:
        """Generate AI response with OpenAI integration"""
        
        if not self.api_key or not self.client:
            logger.warning("No OpenAI API key, using fallback")
            return self._get_smart_fallback_response(user_query)
        
        try:
            # Build personalized prompt
            prompt = self._build_personalized_prompt(
                user_query, user_profile, market_context
            )
            
            # Call OpenAI API
            response = await self._call_openai_api(prompt)
            
            # Apply personality formatting
            formatted_response = self._apply_personality_formatting(response, user_profile)
            
            logger.info(f"✅ Generated personalized OpenAI response for query: {user_query[:50]}...")
            return formatted_response
            
        except Exception as e:
            logger.error(f"❌ OpenAI API call failed: {e}")
            return self._get_smart_fallback_response(user_query)
    
    def _build_personalized_prompt(
        self, 
        user_query: str, 
        user_profile: Dict, 
        market_context: Dict = None
    ) -> str:
        """Build a personalized prompt based on user profile"""
        
        # Extract user style preferences
        comm_style = user_profile.get('communication_style', {})
        trading_personality = user_profile.get('trading_personality', {})
        
        formality = comm_style.get('formality', 'casual')
        energy = comm_style.get('energy', 'medium')
        emoji_usage = comm_style.get('emoji_usage', 'some')
        experience = trading_personality.get('experience_level', 'beginner')
        
        # Build context about market data
        market_info = ""
        if market_context:
            market_info = f"\nMarket Data Available: {market_context}"
        
        # Create personalized prompt
        prompt = f"""You are a personalized SMS trading assistant. Respond to the user's question in their preferred style.

User Question: "{user_query}"

User Communication Style:
- Formality: {formality}
- Energy Level: {energy}
- Emoji Usage: {emoji_usage}
- Trading Experience: {experience}

{market_info}

Guidelines:
- Keep response under 160 characters for SMS
- Match their communication style exactly
- If they like casual/high energy, use phrases like "yo", "crushing it", add 🚀
- If they prefer professional, use formal language with technical terms
- If they're a beginner, explain things simply
- If they're advanced, use technical analysis terminology
- Include actionable insights when possible

Response:"""

        return prompt
    
    async def _call_openai_api(self, prompt: str) -> str:
        """Make the actual OpenAI API call"""
        try:
            # Use the newer OpenAI client interface
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(api_key=self.api_key)
            
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _apply_personality_formatting(self, response: str, user_profile: Dict) -> str:
        """Apply final personality touches to the response"""
        
        comm_style = user_profile.get('communication_style', {})
        energy = comm_style.get('energy', 'medium')
        emoji_usage = comm_style.get('emoji_usage', 'some')
        
        # Adjust energy level
        if energy == "high":
            response = response.replace(".", "!")
            if not response.endswith("!"):
                response += "!"
        elif energy == "low":
            response = response.replace("!", ".")
        
        # Ensure emoji usage matches preference
        has_emojis = any(ord(char) > 127 for char in response)
        
        if emoji_usage == "lots" and not has_emojis:
            # Add appropriate emojis based on content
            if any(word in response.lower() for word in ["up", "gain", "bull", "strong"]):
                response += " 🚀"
            elif any(word in response.lower() for word in ["down", "drop", "bear", "weak"]):
                response += " 📉"
            else:
                response += " 📊"
        elif emoji_usage == "none" and has_emojis:
            # Remove emojis
            response = ''.join(char for char in response if ord(char) <= 127)
        
        return response
    
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
