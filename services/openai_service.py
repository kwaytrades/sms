# ===== services/openai_service.py =====
import openai
from typing import Dict, List
from loguru import logger
from config import settings

class OpenAIService:
    def __init__(self):
        openai.api_key = settings.openai_api_key
        
    async def generate_personalized_response(
        self, 
        user_query: str, 
        user_profile: Dict, 
        conversation_history: List[Dict],
        market_context: Dict = None
    ) -> str:
        """Generate personalized AI response"""
        try:
            # Build personalized prompt
            prompt = self._build_personalized_prompt(
                user_query, user_profile, conversation_history, market_context
            )
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ OpenAI generation failed: {e}")
            return "I'm having trouble processing your request right now. Please try again in a moment."
    
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
"""
        
        user_prompt = f"User query: {user_query}"
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
