# services/llm_agent.py - COMPLETE VERSION with News Sentiment + Fundamental Analysis Integration

import json
import asyncio
import re
from typing import Dict, List, Optional, Any
from loguru import logger
import openai
from datetime import datetime
from openai import AsyncOpenAI

class TradingAgent:
    """Hybrid LLM Agent for intelligent trading bot interactions - COMPLETE VERSION"""
    
    def __init__(self, openai_client, personality_engine):
        self.openai_client = openai_client
        self.personality_engine = personality_engine
        
    async def parse_intent(self, message: str, user_phone: str = None) -> Dict[str, Any]:
        """Step 1: Use GPT-4o-mini to parse user intent AND extract symbols intelligently"""
        
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
2. Stock symbols mentioned (INTELLIGENTLY convert company names to tickers)
3. Parameters or context
4. Required tools/services
5. Confidence level

Message: "{message}"

SYMBOL EXTRACTION RULES:
- "google" or "Google" → "GOOGL"
- "tesla" or "Tesla" → "TSLA" 
- "apple" or "Apple" → "AAPL"
- "microsoft" or "Microsoft" → "MSFT"
- "amazon" or "Amazon" → "AMZN"
- "meta" or "facebook" or "Facebook" → "META"
- "nvidia" or "Nvidia" → "NVDA"
- Ignore casual words like "yo", "what's", "doing", "thinking", "about"
- Only extract actual stock tickers or company names

TOOL SELECTION RULES:
- When users ask about "fundamentals", "financials", "valuation", "earnings", "pe ratio", "ratios" → include "fundamental_analysis"
- When users ask for "complete", "full", "comprehensive", "everything" → include ALL tools
- When users ask about "news", "sentiment", "headlines" → include "news_sentiment"
- Default stock analysis → include "technical_analysis" and "news_sentiment"

Return ONLY valid JSON in this exact format:
{{
    "intent": "analyze|price|compare|screener|portfolio|news|help|general|fundamental_analysis|comprehensive_analysis",
    "symbols": ["AAPL", "TSLA"],
    "parameters": {{
        "timeframe": "1d|1w|1m|3m",
        "analysis_type": "technical|fundamental|sentiment|comprehensive",
        "risk_level": "conservative|moderate|aggressive",
        "context": "buying_calls|swing_trading|long_term|day_trading"
    }},
    "requires_tools": ["technical_analysis", "portfolio_check", "stock_screener", "news_sentiment", "fundamental_analysis"],
    "confidence": 0.95,
    "user_emotion": "excited|cautious|frustrated|curious|neutral",
    "urgency": "low|medium|high"
}}

Examples:
- "yo what's google doing?" → {{"intent": "analyze", "symbols": ["GOOGL"], "requires_tools": ["technical_analysis", "news_sentiment"], "confidence": 0.9}}
- "how's tesla and apple?" → {{"intent": "analyze", "symbols": ["TSLA", "AAPL"], "requires_tools": ["technical_analysis", "news_sentiment"], "confidence": 0.9}}
- "NVDA fundamentals?" → {{"intent": "fundamental_analysis", "symbols": ["NVDA"], "requires_tools": ["fundamental_analysis"], "confidence": 0.95}}
- "complete analysis of AAPL" → {{"intent": "comprehensive_analysis", "symbols": ["AAPL"], "requires_tools": ["technical_analysis", "news_sentiment", "fundamental_analysis"], "confidence": 0.9}}
- "find me cheap tech stocks" → {{"intent": "screener", "parameters": {{"sector": "tech"}}, "requires_tools": ["stock_screener"]}}
- "how's my portfolio?" → {{"intent": "portfolio", "requires_tools": ["portfolio_check"]}}
- "AAPL calls looking good" → {{"intent": "analyze", "symbols": ["AAPL"], "requires_tools": ["technical_analysis", "news_sentiment"], "confidence": 0.95}}
- "any news on tesla?" → {{"intent": "news", "symbols": ["TSLA"], "requires_tools": ["news_sentiment"]}}
"""

        try:
            # FIXED: Always use the wrapped client path for your setup
            if hasattr(self.openai_client, 'client'):
                # Wrapped client (your setup)
                response = await self.openai_client.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=200,
                    response_format={"type": "json_object"}
                )
            else:
                # Direct client fallback
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
            if "news_sentiment" not in intent_data["requires_tools"]:
                intent_data["requires_tools"].append("news_sentiment")
        
        if intent_data["intent"] == "fundamental_analysis" and intent_data["symbols"]:
            if "fundamental_analysis" not in intent_data["requires_tools"]:
                intent_data["requires_tools"].append("fundamental_analysis")
        
        if intent_data["intent"] == "comprehensive_analysis" and intent_data["symbols"]:
            required_tools = ["technical_analysis", "news_sentiment", "fundamental_analysis"]
            for tool in required_tools:
                if tool not in intent_data["requires_tools"]:
                    intent_data["requires_tools"].append(tool)
        
        if intent_data["intent"] == "news" and intent_data["symbols"]:
            if "news_sentiment" not in intent_data["requires_tools"]:
                intent_data["requires_tools"].append("news_sentiment")
        
        if intent_data["intent"] == "portfolio":
            if "portfolio_check" not in intent_data["requires_tools"]:
                intent_data["requires_tools"].append("portfolio_check")
        
        return intent_data
    
    def _fallback_intent_parsing(self, message: str) -> Dict:
        """Fallback with SIMPLE company name mapping - no complex regex"""
        
        message_lower = message.lower()
        
        # SIMPLE company mapping - let LLM handle complex cases
        simple_mappings = {
            'google': 'GOOGL', 'tesla': 'TSLA', 'apple': 'AAPL', 
            'microsoft': 'MSFT', 'amazon': 'AMZN', 'meta': 'META',
            'facebook': 'META', 'nvidia': 'NVDA', 'netflix': 'NFLX'
        }
        
        symbols = []
        for company, symbol in simple_mappings.items():
            if company in message_lower:
                symbols.append(symbol)
        
        # Enhanced intent detection with fundamental analysis support
        if any(word in message_lower for word in ['fundamental', 'fundamentals', 'valuation', 'ratios', 'pe ratio', 'financial', 'earnings', 'revenue']):
            intent = "fundamental_analysis"
            required_tools = ["fundamental_analysis"]
        elif any(word in message_lower for word in ['complete', 'full', 'comprehensive', 'detailed', 'deep dive']):
            intent = "comprehensive_analysis"
            required_tools = ["technical_analysis", "news_sentiment", "fundamental_analysis"]
        elif any(word in message_lower for word in ['news', 'headlines', 'sentiment']):
            intent = "news"
            required_tools = ["news_sentiment"]
        elif any(word in message_lower for word in ['portfolio', 'positions', 'holdings']):
            intent = "portfolio"
            required_tools = ["portfolio_check"]
        elif any(word in message_lower for word in ['find', 'screen', 'search', 'discover']):
            intent = "screener"
            required_tools = ["stock_screener"]
        elif symbols:
            intent = "analyze"
            required_tools = ["technical_analysis", "news_sentiment"]
        else:
            intent = "general"
            required_tools = []
        
        return {
            "intent": intent,
            "symbols": symbols,
            "confidence": 0.4,  # Lower confidence for fallback
            "requires_tools": required_tools,
            "fallback": True
        }
    
    def _clean_response(self, response: str) -> str:
        """Clean LLM response artifacts that shouldn't go to users"""
        
        # Remove meta-instructions
        patterns_to_remove = [
            r"Certainly!.*?for the user:\s*",
            r"Here's the.*?response.*?:\s*",
            r"Based on.*?here's.*?:\s*",
            r"Given.*?here's.*?:\s*",
            r"I'll.*?response.*?:\s*",
            r"Let me.*?response.*?:\s*",
            r"Here's the tailored response.*?:\s*",
            r".*?tailored response.*?:\s*"
        ]
        
        for pattern in patterns_to_remove:
            response = re.sub(pattern, "", response, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove quotes around the entire response
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        
        # Remove extra whitespace and newlines at start
        response = response.strip()
        
        return response
    
    def _validate_response(self, response: str) -> str:
        """Validate response doesn't contain artifacts"""
        
        # Check for common artifacts
        artifacts = [
            "here's the", "certainly", "based on", "given", 
            "let me", "i'll provide", "tailored response"
        ]
        
        response_lower = response.lower()
        if any(artifact in response_lower for artifact in artifacts):
            logger.warning(f"Response contains artifacts: {response[:50]}...")
            # Apply cleaning
            response = self._clean_response(response)
        
        return response
    
    def _serialize_fundamental_data(self, fundamental_data: Dict) -> Dict:
        """Convert FundamentalAnalysisResult objects to JSON-serializable format"""
        serializable_data = {}
        
        for symbol, analysis_result in fundamental_data.items():
            try:
                if hasattr(analysis_result, 'symbol'):
                    # This is a FundamentalAnalysisResult object
                    serializable_data[symbol] = {
                        "symbol": analysis_result.symbol,
                        "overall_score": analysis_result.overall_score,
                        "financial_health": analysis_result.financial_health.value if hasattr(analysis_result.financial_health, 'value') else str(analysis_result.financial_health),
                        "current_price": analysis_result.current_price,
                        "strength_areas": analysis_result.strength_areas,
                        "concern_areas": analysis_result.concern_areas,
                        "bull_case": analysis_result.bull_case,
                        "bear_case": analysis_result.bear_case,
                        "data_completeness": analysis_result.data_completeness
                    }
                    
                    # Add ratios if available
                    if analysis_result.ratios:
                        serializable_data[symbol]["ratios"] = {
                            "pe_ratio": getattr(analysis_result.ratios, 'pe_ratio', None),
                            "roe": getattr(analysis_result.ratios, 'roe', None),
                            "debt_to_equity": getattr(analysis_result.ratios, 'debt_to_equity', None),
                            "current_ratio": getattr(analysis_result.ratios, 'current_ratio', None)
                        }
                    
                    # Add growth metrics if available
                    if analysis_result.growth:
                        serializable_data[symbol]["growth"] = {
                            "revenue_growth_1y": getattr(analysis_result.growth, 'revenue_growth_1y', None),
                            "earnings_growth_1y": getattr(analysis_result.growth, 'earnings_growth_1y', None)
                        }
                else:
                    # Already serializable
                    serializable_data[symbol] = analysis_result
            except Exception as e:
                # Fallback: just include basic info
                serializable_data[symbol] = {"error": f"Serialization failed: {str(e)}"}
        
        return serializable_data
    
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
        
        # Format tool results for context - WITH FIXED JSON SERIALIZATION
        tool_context = ""
        if tool_results:
            if "technical_analysis" in tool_results:
                ta_data = tool_results["technical_analysis"]
                if ta_data:
                    tool_context += f"""
Technical Analysis Results:
{json.dumps(ta_data, indent=2)[:500]}...
"""
            
            if "news_sentiment" in tool_results:
                news_data = tool_results["news_sentiment"]
                if news_data:
                    tool_context += f"""
News Sentiment Results:
{json.dumps(news_data, indent=2)[:500]}...
"""
            
            if "fundamental_analysis" in tool_results:
                fundamental_data = tool_results["fundamental_analysis"]
                if fundamental_data:
                    # FIXED: Convert FundamentalAnalysisResult objects to serializable format
                    serializable_data = self._serialize_fundamental_data(fundamental_data)
                    tool_context += f"""
Fundamental Analysis Results:
{json.dumps(serializable_data, indent=2)[:500]}...
"""
            
            if "portfolio_check" in tool_results:
                portfolio_data = tool_results["portfolio_check"]
                if portfolio_data:
                    tool_context += f"""
Portfolio Data:
{json.dumps(portfolio_data, indent=2)[:300]}...
"""
            
            if "market_data_unavailable" in tool_results:
                tool_context += "\nMARKET DATA UNAVAILABLE - Acknowledge this honestly to the user."
            
            if "news_sentiment_unavailable" in tool_results:
                tool_context += "\nNEWS SENTIMENT UNAVAILABLE - Acknowledge this honestly to the user."
            
            if "fundamental_analysis_unavailable" in tool_results:
                tool_context += "\nFUNDAMENTAL ANALYSIS UNAVAILABLE - Acknowledge this honestly to the user."
        
        prompt = f"""You are a hyper-personalized SMS trading assistant. Write ONLY the direct SMS message that will be sent to the user.

DO NOT include:
- "Here's the response:"
- "Certainly! Here's..."
- Any meta-commentary
- Quotes around the response

Just write the exact message the user should receive.

Original Message: "{user_message}"

Intent Analysis:
{json.dumps(intent_data, indent=2)}

{personality_context}

{tool_context}

RESPONSE GUIDELINES:
1. Match the user's communication style exactly (formality, energy, emoji usage)
2. Provide actionable trading insights based on available data
3. If data is unavailable, acknowledge honestly but stay helpful
4. Keep responses SMS-friendly (under 320 chars if possible, max 2 messages)
5. Use appropriate trading terminology for their experience level
6. Include relevant emojis if user uses them
7. Be conversational and engaging
8. When multiple analysis types are available, integrate them naturally

Examples of style matching:
- Casual/High-energy: "TSLA's going wild! 🚀 $245 and climbing, RSI at 68 tho - might need a breather soon. News looking bullish on new factory announcement!"
- Professional: "TSLA trading at $245.50, up 3.2%. RSI indicates slight overbought conditions at 68. News sentiment bullish on expansion plans. Technical outlook remains positive."
- Beginner-friendly: "Tesla's doing well today! It's up to $245. The RSI (momentum indicator) shows it might slow down soon, but the trend looks good. Recent news is positive too!"
- Fundamental Focus: "AAPL fundamentals: Strong with 95/100 score. P/E at 23.4 (reasonable), ROE 28% (excellent). Revenue up 8.5% YoY. Health: Good. Ready for growth! 💪"

Write the direct SMS message now:"""

        try:
            # FIXED: Always use the wrapped client path for your setup
            if hasattr(self.openai_client, 'client'):
                # Wrapped client (your setup)
                response = await self.openai_client.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=200
                )
            else:
                # Direct client fallback
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=200
                )
            
            generated_response = response.choices[0].message.content.strip()
            
            # Clean and validate response
            generated_response = self._clean_response(generated_response)
            generated_response = self._validate_response(generated_response)
            
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
            if tool_results.get("technical_analysis") or tool_results.get("news_sentiment") or tool_results.get("fundamental_analysis"):
                return f"{symbol} analysis ready! Check the data above."
            else:
                return f"Sorry, can't get {symbol} data right now. Try again in a moment!"
        
        elif intent_data["intent"] == "fundamental_analysis" and intent_data["symbols"]:
            symbol = intent_data["symbols"][0]
            if tool_results.get("fundamental_analysis"):
                return f"{symbol} fundamental analysis ready!"
            else:
                return f"Sorry, can't get {symbol} fundamental data right now. Try again in a moment!"
        
        elif intent_data["intent"] == "news" and intent_data["symbols"]:
            symbol = intent_data["symbols"][0]
            if tool_results.get("news_sentiment"):
                return f"{symbol} news sentiment ready!"
            else:
                return f"Sorry, can't get {symbol} news right now. Try again in a moment!"
        
        elif intent_data["intent"] == "help":
            return "I'm your trading assistant! Ask about stocks, portfolio, or market analysis. 📈"
        
        else:
            return "I'm here to help with your trading questions! What would you like to know?"


# ===== Tool Execution Engine =====

class ToolExecutor:
    """Handles execution of various trading tools based on intent - COMPLETE WITH ALL ANALYSIS ENGINES"""
    
    def __init__(self, ta_service, portfolio_service, screener_service=None, news_service=None, fundamental_tool=None):
        self.ta_service = ta_service
        self.portfolio_service = portfolio_service
        self.screener_service = screener_service
        self.news_service = news_service  # ADD NEWS SERVICE
        self.fundamental_tool = fundamental_tool  # ADD FUNDAMENTAL TOOL
    
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
        
        # ADD NEWS SENTIMENT ANALYSIS
        if "news_sentiment" in required_tools and intent_data.get("symbols"):
            tasks.append(self._execute_news_sentiment(intent_data["symbols"]))
        
        # ADD FUNDAMENTAL ANALYSIS
        if "fundamental_analysis" in required_tools and intent_data.get("symbols"):
            tasks.append(self._execute_fundamental_analysis(intent_data["symbols"], user_phone))
        
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
                return {"technical_analysis_unavailable": True}
            
            ta_results = {}
            for symbol in symbols[:3]:  # Limit to 3 symbols max
                ta_data = await self.ta_service.analyze_symbol(symbol.upper())
                if ta_data:
                    ta_results[symbol] = ta_data
            
            return {"technical_analysis": ta_results} if ta_results else {"technical_analysis_unavailable": True}
            
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            return {"technical_analysis_unavailable": True}
    
    async def _execute_portfolio_check(self, user_phone: str) -> Dict:
        """Execute portfolio check for user"""
        try:
            if not self.portfolio_service:
                return {"portfolio_unavailable": True}
            
            portfolio_data = await self.portfolio_service.get_user_portfolio(user_phone)
            return {"portfolio": portfolio_data} if portfolio_data else {"portfolio_unavailable": True}
            
        except Exception as e:
            logger.error(f"Portfolio check failed: {e}")
            return {"portfolio_unavailable": True}
    
    async def _execute_stock_screener(self, parameters: Dict) -> Dict:
        """Execute stock screening with parameters"""
        try:
            if not self.screener_service:
                return {"screener_unavailable": True}
            
            screening_results = await self.screener_service.screen_stocks(parameters)
            return {"screener_results": screening_results} if screening_results else {"screener_unavailable": True}
            
        except Exception as e:
            logger.error(f"Stock screening failed: {e}")
            return {"screener_unavailable": True}
    
    # ADD NEW METHOD FOR NEWS SENTIMENT
    async def _execute_news_sentiment(self, symbols: List[str]) -> Dict:
        """Execute news sentiment analysis for symbols"""
        try:
            if not self.news_service:
                return {"news_sentiment_unavailable": True}
            
            news_results = {}
            for symbol in symbols[:3]:  # Limit to 3 symbols max
                try:
                    news_data = await self.news_service.get_sentiment(symbol.upper())
                    if news_data and not news_data.get('error'):
                        news_results[symbol] = news_data
                except Exception as e:
                    logger.warning(f"News sentiment failed for {symbol}: {e}")
                    continue
            
            return {"news_sentiment": news_results} if news_results else {"news_sentiment_unavailable": True}
            
        except Exception as e:
            logger.error(f"News sentiment analysis failed: {e}")
            return {"news_sentiment_unavailable": True}
    
    # ADD NEW METHOD FOR FUNDAMENTAL ANALYSIS
    async def _execute_fundamental_analysis(self, symbols: List[str], user_phone: str) -> Dict:
        """Execute fundamental analysis for symbols"""
        try:
            if not self.fundamental_tool:
                return {"fundamental_analysis_unavailable": True}
            
            fundamental_results = {}
            for symbol in symbols[:2]:  # Limit to 2 symbols max (fundamental analysis is more expensive)
                try:
                    # Determine analysis depth based on user (can be enhanced with user profile later)
                    depth = "standard"  # Can be basic, standard, comprehensive
                    user_style = "casual"  # Can be casual, professional, technical
                    
                    fund_result = await self.fundamental_tool.execute({
                        "symbol": symbol.upper(),
                        "depth": depth,
                        "user_style": user_style
                    })
                    
                    if fund_result.get("success"):
                        fundamental_results[symbol] = fund_result["analysis_result"]
                except Exception as e:
                    logger.warning(f"Fundamental analysis failed for {symbol}: {e}")
                    continue
            
            return {"fundamental_analysis": fundamental_results} if fundamental_results else {"fundamental_analysis_unavailable": True}
            
        except Exception as e:
            logger.error(f"Fundamental analysis failed: {e}")
            return {"fundamental_analysis_unavailable": True}

# ===== ADVANCED SYMBOL EXTRACTION =====

class AdvancedSymbolExtractor:
    """Advanced context-aware symbol extraction"""
    
    def __init__(self):
        # Comprehensive company mappings
        self.company_mappings = {
            # Tech giants
            'apple': 'AAPL', 'tesla': 'TSLA', 'microsoft': 'MSFT', 'amazon': 'AMZN',
            'google': 'GOOGL', 'alphabet': 'GOOGL', 'meta': 'META', 'facebook': 'META',
            'nvidia': 'NVDA', 'netflix': 'NFLX', 'adobe': 'ADBE', 'salesforce': 'CRM',
            
            # Telecom
            'verizon': 'VZ', 'at&t': 'T', 'att': 'T', 'comcast': 'CMCSA', 't-mobile': 'TMUS',
            
            # Finance
            'jpmorgan': 'JPM', 'jp morgan': 'JPM', 'goldman': 'GS', 'goldman sachs': 'GS',
            'bank of america': 'BAC', 'wells fargo': 'WFC', 'morgan stanley': 'MS',
            
            # Retail
            'walmart': 'WMT', 'target': 'TGT', 'home depot': 'HD', 'lowes': 'LOW',
            'starbucks': 'SBUX', 'mcdonalds': 'MCD', 'nike': 'NKE',
            
            # Industrial
            'boeing': 'BA', 'caterpillar': 'CAT', 'general electric': 'GE', '3m': 'MMM',
            
            # Healthcare
            'johnson & johnson': 'JNJ', 'pfizer': 'PFE', 'merck': 'MRK', 'abbott': 'ABT',
            
            # Energy
            'exxon': 'XOM', 'chevron': 'CVX', 'conocophillips': 'COP',
            
            # Crypto/Fintech
            'coinbase': 'COIN', 'paypal': 'PYPL', 'square': 'SQ', 'robinhood': 'HOOD'
        }
        
        # Trading context indicators
        self.trading_context_indicators = [
            'stock', 'ticker', 'symbol', 'price', 'trading', 'buying', 'selling',
            'calls', 'puts', 'options', 'shares', 'market', 'earnings', 'dividend',
            'up', 'down', 'gain', 'loss', 'profit', 'bull', 'bear', 'chart',
            'rsi', 'macd', 'volume', 'support', 'resistance', 'breakout'
        ]
    
    def extract_symbols(self, message: str) -> List[str]:
        """Extract symbols using advanced context-aware analysis"""
        import re
        
        message_lower = message.lower()
        found_symbols = []
        
        # Check for company names first
        for company, symbol in self.company_mappings.items():
            if company in message_lower:
                found_symbols.append(symbol)
        
        # Find potential ticker symbols
        potential_symbols = re.findall(r'\b[A-Z]{2,5}\b', message.upper())
        
        # Check for trading context
        has_trading_context = any(indicator in message_lower for indicator in self.trading_context_indicators)
        
        # Process potential symbols with context awareness
        for symbol in potential_symbols:
            if self._is_valid_symbol_in_context(symbol, message, has_trading_context):
                found_symbols.append(symbol)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(found_symbols))
    
    def _is_valid_symbol_in_context(self, symbol: str, message: str, has_trading_context: bool) -> bool:
        """Determine if a symbol is valid based on context"""
        
        # Always exclude common words
        always_exclude = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'GET', 'HAS', 
            'WAS', 'ONE', 'OUR', 'HAD', 'NOW', 'OLD', 'SEE', 'WAY', 'WHO', 'BOY', 
            'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE', 'HOW', 'WHAT', 'WHEN'
        }
        
        if symbol in always_exclude:
            return False
        
        # Context-dependent exclusions
        context_dependent_exclude = {
            'SMH': ['smh', 'shaking my head'],
            'LOL': ['lol', 'laugh'],
            'OMG': ['omg', 'oh my god'],
            'WOW': ['wow', 'amazing'],
            'CAT': ['cat', 'pet', 'animal'],
            'AI': ['artificial intelligence']
        }
        
        # If we have trading context, be more permissive
        if has_trading_context:
            if symbol in context_dependent_exclude:
                exclusion_phrases = context_dependent_exclude[symbol]
                message_lower = message.lower()
                
                for phrase in exclusion_phrases:
                    if phrase in message_lower:
                        return False
            
            return len(symbol) >= 2 and len(symbol) <= 5 and symbol.isalpha()
        
        # Without trading context, be more restrictive
        else:
            major_tickers = {
                'AAPL', 'TSLA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'NFLX',
                'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'DIS', 'WMT', 'HD'
            }
            
            return symbol in major_tickers


# ===== PERSONALITY-AWARE RESPONSE GENERATOR =====

class PersonalityAwareResponseGenerator:
    """Generates responses that match user's personality perfectly"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
    
    async def generate_personality_matched_response(
        self, 
        user_message: str,
        analysis_data: Dict,
        user_profile: Dict,
        user_phone: str
    ) -> str:
        """Generate response that perfectly matches user's personality"""
        
        # Build detailed personality prompt
        personality_prompt = self._build_personality_prompt(user_profile, user_message, analysis_data)
        
        try:
            if hasattr(self.openai_client, 'chat'):
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": personality_prompt}],
                    temperature=0.8,  # Higher creativity for personality matching
                    max_tokens=250
                )
            else:
                response = await self.openai_client.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": personality_prompt}],
                    temperature=0.8,
                    max_tokens=250
                )
            
            generated_response = response.choices[0].message.content.strip()
            
            # SMS optimization
            if len(generated_response) > 320:
                generated_response = self._optimize_for_sms(generated_response)
            
            return generated_response
            
        except Exception as e:
            logger.error(f"Personality response generation failed: {e}")
            return self._generate_fallback_personality_response(user_profile, analysis_data)
    
    def _build_personality_prompt(self, user_profile: Dict, user_message: str, analysis_data: Dict) -> str:
        """Build comprehensive personality-aware prompt"""
        
        comm_style = user_profile.get('communication_style', {})
        trading_style = user_profile.get('trading_personality', {})
        
        style_description = f"""
USER PERSONALITY PROFILE:
Communication Style:
- Formality: {comm_style.get('formality', 'casual')} (casual/professional/friendly)
- Energy Level: {comm_style.get('energy', 'moderate')} (low/moderate/high/excited)
- Emoji Usage: {comm_style.get('emoji_usage', 'some')} (none/minimal/some/lots)
- Message Length: {comm_style.get('message_length', 'medium')} (short/medium/long)
- Technical Depth: {comm_style.get('technical_depth', 'medium')} (basic/medium/advanced)

Trading Personality:
- Experience: {trading_style.get('experience_level', 'intermediate')} (beginner/intermediate/advanced)
- Risk Tolerance: {trading_style.get('risk_tolerance', 'moderate')} (conservative/moderate/aggressive)
- Trading Style: {trading_style.get('trading_style', 'swing')} (day/swing/long_term)
- Win Rate: {user_profile.get('learning_data', {}).get('successful_trades_mentioned', 0)}W/{user_profile.get('learning_data', {}).get('loss_trades_mentioned', 0)}L
"""
        
        analysis_context = ""
        if analysis_data:
            analysis_context = f"""
MARKET DATA ANALYSIS:
{json.dumps(analysis_data, indent=2)[:600]}...
"""
        
        return f"""You are this user's personal AI trading buddy. You know them intimately and communicate exactly like their best trading friend would.

{style_description}

ORIGINAL MESSAGE: "{user_message}"

{analysis_context}

CRITICAL INSTRUCTIONS:
1. Match their EXACT communication style - if they say "yo" you say "yo", if they're professional you're professional
2. Use their preferred emoji level - match their energy exactly
3. Technical depth must match their experience level
4. Reference their trading personality naturally
5. Keep under 320 characters for SMS
6. Sound like you've known them for years

STYLE EXAMPLES:
- Casual/High: "Yo! TSLA looking spicy 🌶️ RSI at 67, still got room to run. You thinking calls or waiting for a dip?"
- Professional: "TSLA analysis: $245.50 (+1.2%), RSI 67 indicates room for upward movement. Entry opportunity present."
- Beginner: "Tesla's doing well! Up $3 today. The RSI shows it's not overbought yet - that's good for more gains!"

Generate their perfect personalized response:"""
    
    def _optimize_for_sms(self, response: str) -> str:
        """Optimize long responses for SMS"""
        sentences = response.split('. ')
        if len(sentences) > 1:
            # Try to split into 2 messages
            mid_point = len(sentences) // 2
            first_half = '. '.join(sentences[:mid_point]) + '.'
            second_half = '. '.join(sentences[mid_point:])
            
            if len(first_half) <= 320 and len(second_half) <= 320:
                return f"{first_half}\n\n{second_half}"
        
        return response[:317] + "..."
    
    def _generate_fallback_personality_response(self, user_profile: Dict, analysis_data: Dict) -> str:
        """Fallback response when AI fails"""
        formality = user_profile.get('communication_style', {}).get('formality', 'casual')
        
        if formality == 'casual':
            return "Data's looking good! 📈 Let me know if you want more details!"
        else:
            return "Analysis complete. Technical indicators are available upon request."


# ===== COMPREHENSIVE MESSAGE PROCESSOR =====

class ComprehensiveMessageProcessor:
    """Complete message processing with all components integrated"""
    
    def __init__(self, openai_client, ta_service, personality_engine, news_service=None, fundamental_tool=None):
        self.trading_agent = TradingAgent(openai_client, personality_engine)
        self.tool_executor = ToolExecutor(ta_service, None, None, news_service, fundamental_tool)  # ALL SERVICES
        self.symbol_extractor = AdvancedSymbolExtractor()
        self.response_generator = PersonalityAwareResponseGenerator(openai_client)
        self.personality_engine = personality_engine
    
    async def process_complete_message(self, message: str, user_phone: str) -> str:
        """Complete message processing pipeline"""
        
        try:
            logger.info(f"🔄 Processing complete message: '{message}' from {user_phone}")
            
            # Step 1: Parse intent with trading agent
            intent_data = await self.trading_agent.parse_intent(message, user_phone)
            
            # Step 2: Execute tools based on intent
            tool_results = await self.tool_executor.execute_tools(intent_data, user_phone)
            
            # Step 3: Learn from interaction
            self.personality_engine.learn_from_message(user_phone, message, intent_data)
            
            # Step 4: Get user profile
            user_profile = self.personality_engine.get_user_profile(user_phone)
            
            # Step 5: Generate personality-matched response
            response = await self.response_generator.generate_personality_matched_response(
                user_message=message,
                analysis_data=tool_results,
                user_profile=user_profile,
                user_phone=user_phone
            )
            
            logger.info(f"✅ Complete processing finished: {len(response)} chars")
            
            return response
            
        except Exception as e:
            logger.error(f"💥 Complete message processing failed: {e}")
            return "Having some technical issues right now. Try again in a moment! 🔧"
