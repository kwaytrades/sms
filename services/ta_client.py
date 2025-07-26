# services/ta_client.py
import httpx
import asyncio
from typing import Dict, Optional, Any
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TAServiceClient:
    def __init__(self):
        self.ta_service_url = os.getenv('TA_SERVICE_URL', 'http://localhost:8000')
        self.timeout = 10.0
        
    async def get_stock_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive stock analysis from TA microservice"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.ta_service_url}/analysis/{symbol}")
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✅ Got TA data for {symbol} from {'cache' if data.get('source') == 'cache' else 'fresh API'}")
                    return data
                else:
                    logger.error(f"❌ TA service error {response.status_code} for {symbol}")
                    return None
                    
        except httpx.TimeoutException:
            logger.error(f"⏰ TA service timeout for {symbol}")
            return None
        except Exception as e:
            logger.error(f"💥 TA service error for {symbol}: {e}")
            return None
    
    async def get_price_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current price data"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.ta_service_url}/price/{symbol}")
                
                if response.status_code == 200:
                    return response.json()
                return None
                
        except Exception as e:
            logger.error(f"Price data error for {symbol}: {e}")
            return None
    
    async def get_technical_indicators(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get technical indicators only"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.ta_service_url}/technical/{symbol}")
                
                if response.status_code == 200:
                    return response.json()
                return None
                
        except Exception as e:
            logger.error(f"Technical indicators error for {symbol}: {e}")
            return None

    async def get_market_screener(self, criteria: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get stock screener results"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.ta_service_url}/screener", 
                    json=criteria
                )
                
                if response.status_code == 200:
                    return response.json()
                return None
                
        except Exception as e:
            logger.error(f"Screener error: {e}")
            return None

# services/message_analyzer.py
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class MessageIntent:
    action: str  # 'analyze', 'price', 'screener', 'news', 'general'
    symbols: List[str]
    parameters: Dict[str, Any]
    confidence: float

class MessageAnalyzer:
    def __init__(self):
        # Common stock ticker patterns
        self.ticker_pattern = re.compile(r'\b([A-Z]{1,5})\b')
        
        # Intent keywords
        self.intent_keywords = {
            'analyze': ['analyze', 'analysis', 'look at', 'check', 'what about', 'how is', 'tell me about'],
            'price': ['price', 'cost', 'trading at', 'current', 'quote'],
            'technical': ['rsi', 'macd', 'support', 'resistance', 'technical', 'indicators', 'chart'],
            'screener': ['find', 'search', 'screen', 'discover', 'suggest', 'recommend', 'good stocks'],
            'news': ['news', 'updates', 'happened', 'events', 'earnings'],
            'compare': ['vs', 'versus', 'compare', 'better than', 'against']
        }
    
    def extract_tickers(self, message: str) -> List[str]:
        """Extract potential stock tickers from message"""
        # Remove common false positives
        exclude_words = {'TO', 'AT', 'IN', 'ON', 'OR', 'OF', 'IS', 'IT', 'BE', 'DO', 'GO', 'UP', 'MY', 'AI'}
        
        potential_tickers = self.ticker_pattern.findall(message.upper())
        return [ticker for ticker in potential_tickers if ticker not in exclude_words]
    
    def detect_intent(self, message: str) -> MessageIntent:
        """Analyze message to determine user intent"""
        message_lower = message.lower()
        tickers = self.extract_tickers(message)
        
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
            # Look for screening criteria
            if 'cheap' in message_lower or 'undervalued' in message_lower:
                parameters['pe_ratio'] = '<15'
            if 'growth' in message_lower:
                parameters['growth_rate'] = '>20'
            if 'dividend' in message_lower:
                parameters['dividend_yield'] = '>2'
        
        return MessageIntent(
            action=primary_intent,
            symbols=tickers,
            parameters=parameters,
            confidence=confidence
        )

# services/response_generator.py
from typing import Dict, Any, Optional
import openai
import os

class ResponseGenerator:
    def __init__(self):
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
    def format_technical_analysis(self, symbol: str, ta_data: Dict[str, Any]) -> str:
        """Format technical analysis data into readable text"""
        if not ta_data:
            return f"Sorry, couldn't get technical data for {symbol} right now."
        
        # Extract key data points
        technical = ta_data.get('technical', {})
        price = ta_data.get('price', {})
        
        # Build response
        response_parts = []
        
        # Price info
        if price:
            current_price = price.get('price', 'N/A')
            change = price.get('change', 0)
            change_pct = price.get('change_percent', 0)
            
            direction = "↑" if change > 0 else "↓" if change < 0 else "→"
            response_parts.append(f"{symbol}: ${current_price} {direction}{change_pct:.1f}%")
        
        # Technical indicators
        if technical:
            rsi = technical.get('rsi')
            if rsi:
                rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                response_parts.append(f"RSI: {rsi:.1f} ({rsi_signal})")
            
            macd = technical.get('macd')
            if macd:
                response_parts.append(f"MACD: {macd}")
            
            support = technical.get('support')
            resistance = technical.get('resistance')
            if support and resistance:
                response_parts.append(f"Support: ${support:.2f} | Resistance: ${resistance:.2f}")
        
        # Data source indicator
        source = ta_data.get('source', 'unknown')
        is_popular = ta_data.get('is_popular', False)
        cache_indicator = "⚡" if source == 'cache' else "📡"
        popular_indicator = "🔥" if is_popular else ""
        
        return "\n".join(response_parts) + f" {cache_indicator}{popular_indicator}"
    
    async def generate_personalized_response(
        self, 
        message: str, 
        intent: MessageIntent, 
        ta_data: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate personalized response using OpenAI"""
        
        # Create context for AI
        context_parts = []
        
        if ta_data:
            context_parts.append(f"Technical Analysis Data: {ta_data}")
        
        if user_profile:
            style = user_profile.get('communication_style', 'casual')
            experience = user_profile.get('trading_experience', 'beginner')
            context_parts.append(f"User Style: {style}, Experience: {experience}")
        
        context = "\n".join(context_parts)
        
        # Craft prompt based on intent
        if intent.action == 'analyze' and ta_data:
            prompt = f"""
            User asked: "{message}"
            
            Technical data available: {ta_data}
            
            Provide a concise SMS-friendly analysis (max 160 chars) that:
            1. Includes key price and technical info
            2. Gives actionable insight
            3. Matches casual trading style
            
            Keep it brief but informative.
            """
        else:
            prompt = f"""
            User asked: "{message}"
            Intent: {intent.action}
            Symbols mentioned: {intent.symbols}
            
            Context: {context}
            
            Provide a helpful SMS response (max 160 chars) that:
            1. Addresses their question directly
            2. Is conversational and friendly
            3. Suggests what they can ask for more info
            """
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            # Fallback to formatted response
            if ta_data and intent.symbols:
                return self.format_technical_analysis(intent.symbols[0], ta_data)
            else:
                return "I'm here to help with stock analysis! Try asking about a specific ticker like 'How is AAPL doing?'"

# Updated main SMS handler
from services.ta_client import TAServiceClient
from services.message_analyzer import MessageAnalyzer
from services.response_generator import ResponseGenerator

# Add these to your main app.py
ta_client = TAServiceClient()
message_analyzer = MessageAnalyzer()
response_generator = ResponseGenerator()

@app.post("/sms")
async def handle_sms(request: Request):
    """Enhanced SMS handler with TA integration"""
    form = await request.form()
    from_number = form.get('From')
    message_body = form.get('Body', '').strip()
    
    logger.info(f"📱 SMS from {from_number}: {message_body}")
    
    try:
        # Check rate limits (your existing code)
        # ... rate limiting logic ...
        
        # Analyze the message
        intent = message_analyzer.detect_intent(message_body)
        logger.info(f"🎯 Intent: {intent.action}, Symbols: {intent.symbols}, Confidence: {intent.confidence}")
        
        # Fetch data based on intent
        ta_data = None
        if intent.symbols and intent.action in ['analyze', 'price', 'technical']:
            symbol = intent.symbols[0]  # Use first symbol mentioned
            ta_data = await ta_client.get_stock_analysis(symbol)
        
        # Get user profile (if you have user system)
        user_profile = None  # TODO: Fetch from MongoDB based on phone number
        
        # Generate response
        if ta_data:
            response_text = await response_generator.generate_personalized_response(
                message_body, intent, ta_data, user_profile
            )
        else:
            # Handle non-stock queries or when TA data unavailable
            if intent.action == 'screener':
                response_text = "Stock screener coming soon! For now, try asking about specific tickers like AAPL, TSLA, etc."
            elif intent.symbols:
                response_text = f"Couldn't get data for {', '.join(intent.symbols)} right now. Please try again!"
            else:
                response_text = await response_generator.generate_personalized_response(
                    message_body, intent, None, user_profile
                )
        
        # Send response
        twiml_response = MessagingResponse()
        twiml_response.message(response_text)
        
        # Log interaction (your existing code)
        # ... logging logic ...
        
        return Response(content=str(twiml_response), media_type="application/xml")
        
    except Exception as e:
        logger.error(f"💥 SMS handler error: {e}")
        
        twiml_response = MessagingResponse()
        twiml_response.message("Sorry, I'm having technical difficulties. Please try again in a moment!")
        return Response(content=str(twiml_response), media_type="application/xml")
