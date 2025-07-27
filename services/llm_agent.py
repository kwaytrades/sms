# ===== services/llm_agent.py =====
import json
import asyncio
from typing import Dict, List, Optional, Any
from loguru import logger
import openai
from datetime import datetime

class TradingAgent:
    """Hybrid LLM Agent for intelligent trading bot interactions"""
    
    def __init__(self, openai_client, personality_engine):
        self.openai_client = openai_client
        self.personality_engine = personality_engine
        
    async def parse_intent(self, message: str, user_phone: str = None) -> Dict[str, Any]:
        """Step 1: Use GPT-4o-mini to parse user intent (cheap & fast)"""
        
        # Get user context for better parsing
        user_context = ""
        if user_phone and self.personality_engine:
            profile = self.personality_engine.get_user_profile(user_phone)
            experience = profile.get('trading_personality', {}).get('experience_level', 'intermediate')
            style = profile.get('communication_style', {}).get('formality', 'casual')
            user_context = f"User is {experience} trader with {style} communication style."
        
        prompt = f"""Parse this trading message into structured JSON format.

{user_context}

Extract:
1. Primary intent (what user wants to do)
2. Stock symbols mentioned (if any)
3. Parameters or context
4. Required tools/services
5. Confidence level

Message: "{message}"

Return ONLY valid JSON in this exact format:
{{
    "intent": "analyze|price|compare|screener|portfolio|news|help|general",
    "symbols": ["AAPL", "TSLA"],
    "parameters": {{
        "timeframe": "1d|1w|1m|3m",
        "analysis_type": "technical|fundamental|sentiment",
        "risk_level": "conservative|moderate|aggressive",
        "context": "buying_calls|swing_trading|long_term|day_trading"
    }},
    "requires_tools": ["technical_analysis", "portfolio_check", "stock_screener", "news_sentiment"],
    "confidence": 0.95,
    "user_emotion": "excited|cautious|frustrated|curious|neutral",
    "urgency": "low|medium|high"
}}

Examples:
- "yo what's TSLA doing?" → {{"intent": "analyze", "symbols": ["TSLA"], "confidence": 0.9}}
- "find me cheap tech stocks" → {{"intent": "screener", "parameters": {{"sector": "tech"}}, "requires_tools": ["stock_screener"]}}
- "how's my portfolio?" → {{"intent": "portfolio", "requires_tools": ["portfolio_check"]}}
"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            intent_data = json.loads(response.choices[0].message.content)
            
            # Validate and clean the response
            intent_data = self._validate_intent(intent_data, message)
            
            logger.info(f"Intent parsed: {intent_data['intent']} | Symbols: {intent_data.get('symbols', [])} | Confidence: {intent_data.get('confidence', 0)}")
            
            return intent_data
            
        except Exception as e:
            logger.error(f"Intent parsing failed: {e}")
            # Fallback to basic parsing
            return self._fallback_intent_parsing(message)
    
    def _validate_intent(self, intent_data: Dict, original_message: str) -> Dict:
        """Validate and clean the parsed intent"""
        
        # Ensure required fields exist
        if "intent" not in intent_data:
            intent_data["intent"] = "general"
        
        if "symbols" not in intent_data:
            intent_data["symbols"] = []
        
        if "confidence" not in intent_data:
            intent_data["confidence"] = 0.5
        
        if "requires_tools" not in intent_data:
            intent_data["requires_tools"] = []
        
        # Clean symbols (remove duplicates, validate format)
        if intent_data["symbols"]:
            cleaned_symbols = []
            for symbol in intent_data["symbols"]:
                if isinstance(symbol, str) and 1 <= len(symbol) <= 5 and symbol.isalpha():
                    cleaned_symbols.append(symbol.upper())
            intent_data["symbols"] = list(dict.fromkeys(cleaned_symbols))  # Remove duplicates
        
        # Auto-determine required tools based on intent
        if intent_data["intent"] == "analyze" and intent_data["symbols"]:
            if "technical_analysis" not in intent_data["requires_tools"]:
                intent_data["requires_tools"].append("technical_analysis")
        
        if intent_data["intent"] == "portfolio":
            if "portfolio_check" not in intent_data["requires_tools"]:
                intent_data["requires_tools"].append("portfolio_check")
        
        return intent_data
    
    def _fallback_intent_parsing(self, message: str) -> Dict:
        """Fallback regex-based parsing if LLM fails"""
        import re
        
        message_lower = message.lower()
        
        # Basic symbol extraction
        symbols = re.findall(r'\b[A-Z]{2,5}\b', message.upper())
        exclude_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'YO', 'HEY', 'SO'}
        symbols = [s for s in symbols if s not in exclude_words]
        
        # Basic intent detection
        if any(word in message_lower for word in ['portfolio', 'positions', 'holdings']):
            intent = "portfolio"
        elif any(word in message_lower for word in ['find', 'screen', 'search', 'discover']):
            intent = "screener"
        elif symbols:
            intent = "analyze"
        else:
            intent = "general"
        
        return {
            "intent": intent,
            "symbols": symbols,
            "confidence": 0.3,
            "requires_tools": [],
            "fallback": True
        }
    
    async def generate_response(
        self, 
        user_message: str,
        intent_data: Dict,
        tool_results: Dict,
        user_phone: str,
        user_profile: Dict = None
    ) -> str:
        """Step 3: Use GPT-4o to generate high-quality personalized response"""
        
        # Get personality context
        personality_context = ""
        if user_profile:
            comm_style = user_profile.get('communication_style', {})
            trading_style = user_profile.get('trading_personality', {})
            
            personality_context = f"""
User Communication Style:
- Formality: {comm_style.get('formality', 'casual')}
- Energy: {comm_style.get('energy', 'moderate')}
- Emoji Usage: {comm_style.get('emoji_usage', 'some')}
- Technical Depth: {comm_style.get('technical_depth', 'medium')}

User Trading Profile:
- Experience: {trading_style.get('experience_level', 'intermediate')}
- Risk Tolerance: {trading_style.get('risk_tolerance', 'moderate')}
- Trading Style: {trading_style.get('trading_style', 'swing')}
"""
        
        # Format tool results for context
        tool_context = ""
        if tool_results:
            if "technical_analysis" in tool_results:
                ta_data = tool_results["technical_analysis"]
                if ta_data:
                    tool_context += f"""
Technical Analysis Results:
{json.dumps(ta_data, indent=2)}
"""
            
            if "portfolio_check" in tool_results:
                portfolio_data = tool_results["portfolio_check"]
                if portfolio_data:
                    tool_context += f"""
Portfolio Data:
{json.dumps(portfolio_data, indent=2)}
"""
            
            if "market_data_unavailable" in tool_results:
                tool_context += "\nMARKET DATA UNAVAILABLE - Acknowledge this honestly to the user."
        
        prompt = f"""You are a hyper-personalized SMS trading assistant. Generate a response that perfectly matches the user's communication style and provides valuable trading insights.

Original Message: "{user_message}"

Intent Analysis:
{json.dumps(intent_data, indent=2)}

{personality_context}

{tool_context}

RESPONSE GUIDELINES:
1. Match the user's communication style exactly (formality, energy, emoji usage)
2. Provide actionable trading insights based on available data
3. If data is unavailable, acknowledge honestly but stay helpful
4. Keep responses SMS-friendly (under 160 chars if possible, max 2 messages)
5. Use appropriate trading terminology for their experience level
6. Include relevant emojis if user uses them
7. Be conversational and engaging

Examples of style matching:
- Casual/High-energy: "TSLA's going wild! 🚀 $245 and climbing, RSI at 68 tho - might need a breather soon. You thinking calls?"
- Professional: "TSLA trading at $245.50, up 3.2%. RSI indicates slight overbought conditions at 68. Technical outlook remains bullish."
- Beginner-friendly: "Tesla's doing well today! It's up to $245. The RSI (momentum indicator) shows it might slow down soon, but the trend looks good overall."

Generate the perfect response now:"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            
            generated_response = response.choices[0].message.content.strip()
            
            # Ensure response isn't too long for SMS
            if len(generated_response) > 320:  # SMS limit is ~160 but allow for 2 messages
                generated_response = generated_response[:317] + "..."
            
            logger.info(f"Generated personalized response: {len(generated_response)} chars")
            
            return generated_response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            # Fallback to simple response
            return self._generate_fallback_response(intent_data, tool_results)
    
    def _generate_fallback_response(self, intent_data: Dict, tool_results: Dict) -> str:
        """Simple fallback response if LLM fails"""
        
        if intent_data["intent"] == "analyze" and intent_data["symbols"]:
            symbol = intent_data["symbols"][0]
            if tool_results.get("technical_analysis"):
                return f"{symbol} analysis ready! Check the data above."
            else:
                return f"Sorry, can't get {symbol} data right now. Try again in a moment!"
        
        elif intent_data["intent"] == "help":
            return "I'm your trading assistant! Ask about stocks, portfolio, or market analysis. 📈"
        
        else:
            return "I'm here to help with your trading questions! What would you like to know?"


# ===== Tool Execution Engine =====

class ToolExecutor:
    """Handles execution of various trading tools based on intent"""
    
    def __init__(self, ta_service, portfolio_service, screener_service=None):
        self.ta_service = ta_service
        self.portfolio_service = portfolio_service
        self.screener_service = screener_service
    
    async def execute_tools(self, intent_data: Dict, user_phone: str) -> Dict[str, Any]:
        """Execute required tools based on parsed intent"""
        
        results = {}
        required_tools = intent_data.get("requires_tools", [])
        
        # Execute tools in parallel when possible
        tasks = []
        
        if "technical_analysis" in required_tools and intent_data.get("symbols"):
            tasks.append(self._execute_technical_analysis(intent_data["symbols"]))
        
        if "portfolio_check" in required_tools:
            tasks.append(self._execute_portfolio_check(user_phone))
        
        if "stock_screener" in required_tools:
            tasks.append(self._execute_stock_screener(intent_data.get("parameters", {})))
        
        # Wait for all tools to complete
        if tasks:
            tool_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(tool_results):
                if isinstance(result, Exception):
                    logger.error(f"Tool execution failed: {result}")
                    results["error"] = str(result)
                else:
                    results.update(result)
        
        return results
    
    async def _execute_technical_analysis(self, symbols: List[str]) -> Dict:
        """Execute technical analysis for symbols"""
        try:
            if not self.ta_service:
                return {"market_data_unavailable": True}
            
            ta_results = {}
            for symbol in symbols[:3]:  # Limit to 3 symbols max
                ta_data = await self.ta_service.analyze_symbol(symbol.upper())
                if ta_data:
                    ta_results[symbol] = ta_data
            
            return {"technical_analysis": ta_results} if ta_results else {"market_data_unavailable": True}
            
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            return {"market_data_unavailable": True}
    
    async def _execute_portfolio_check(self, user_phone: str) -> Dict:
        """Execute portfolio check for user"""
        try:
            if not self.portfolio_service:
                return {"portfolio_unavailable": True}
            
            portfolio_data = await self.portfolio_service.get_user_portfolio(user_phone)
            return {"portfolio_check": portfolio_data}
            
        except Exception as e:
            logger.error(f"Portfolio check failed: {e}")
            return {"portfolio_unavailable": True}
    
    async def _execute_stock_screener(self, parameters: Dict) -> Dict:
        """Execute stock screener with parameters"""
        try:
            if not self.screener_service:
                return {"screener_unavailable": True}
            
            screener_results = await self.screener_service.screen_stocks(parameters)
            return {"stock_screener": screener_results}
            
        except Exception as e:
            logger.error(f"Stock screener failed: {e}")
            return {"screener_unavailable": True}
