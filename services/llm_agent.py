import json
import asyncio
import re
from typing import Dict, List, Optional, Any
from loguru import logger
import openai
from datetime import datetime
from openai import AsyncOpenAI

class ConversationAwareAgent:
    """
    Simplified conversation-aware agent that handles everything in one place.
    Replaces complex orchestration with smart, context-aware processing.
    """
    
    def __init__(self, openai_client, ta_service, news_service, fundamental_tool, cache_service, personality_engine):
        self.openai_client = openai_client
        self.ta_service = ta_service
        self.news_service = news_service
        self.fundamental_tool = fundamental_tool
        self.cache_service = cache_service
        self.personality_engine = personality_engine
    
    async def process_message(self, message: str, user_phone: str) -> str:
        """
        Single method that handles everything with conversation awareness
        """
        try:
            logger.info(f"🎯 Conversation-aware processing: '{message}' from {user_phone}")
            
            # Step 1: Get conversation context (last 3-5 exchanges)
            context = await self._get_conversation_context(user_phone)
            
            # Step 2: Smart analysis and tool execution in one step
            response = await self._analyze_and_respond(message, context)
            
            # Step 3: Cache conversation for future context
            await self._cache_conversation(user_phone, message, response, context)
            
            # Step 4: Learn from interaction
            if self.personality_engine:
                self._learn_from_interaction(user_phone, message, response)
            
            logger.info(f"✅ Conversation-aware response: {len(response)} chars")
            return response
            
        except Exception as e:
            logger.error(f"💥 Conversation-aware processing failed: {e}")
            return "I'm having some technical difficulties right now. Please try again in a moment."
    
    async def _get_conversation_context(self, user_phone: str) -> Dict[str, Any]:
        """Get recent conversation context for natural flow"""
        
        context = {
            "recent_messages": [],
            "recent_symbols": [],
            "user_style": "professional",
            "conversation_flow": "new",
            "last_topic": None
        }
        
        try:
            if self.cache_service:
                # Get last 5 exchanges for context
                thread_key = f"conversation_thread:{user_phone}"
                recent_messages = await self.cache_service.get_list(thread_key, limit=5)
                context["recent_messages"] = recent_messages or []
                
                # Extract recent symbols and topics
                if recent_messages:
                    context["conversation_flow"] = "continuing"
                    # Get symbols from recent messages
                    all_symbols = []
                    for msg in recent_messages:
                        all_symbols.extend(msg.get("symbols", []))
                    context["recent_symbols"] = list(dict.fromkeys(all_symbols))[:3]
                    
                    # Get last topic
                    if recent_messages:
                        context["last_topic"] = recent_messages[0].get("topic", "analysis")
            
            # Get user communication style
            if self.personality_engine:
                profile = self.personality_engine.get_user_profile(user_phone)
                style = profile.get("communication_style", {}).get("formality", "professional")
                context["user_style"] = style
            
            return context
            
        except Exception as e:
            logger.warning(f"Context retrieval failed: {e}")
            return context
    
    async def _analyze_and_respond(self, message: str, context: Dict) -> str:
        """
        Smart analysis that determines what data to fetch and generates response
        """
        
        # Build conversation context string
        conversation_context = self._format_conversation_context(context)
        
        # Improved smart prompt with professional tone and conversational flow
        analysis_prompt = f"""You are a professional trading advisor providing direct, actionable market analysis. Analyze the conversation context and current message to provide the most relevant response.

CONVERSATION CONTEXT:
{conversation_context}

CURRENT MESSAGE: "{message}"

ANALYSIS FRAMEWORK:

1. CONTEXT ASSESSMENT:
   - If this continues a previous topic, build directly on that discussion
   - If this is a new topic, provide comprehensive analysis
   - If user is asking follow-up questions, focus specifically on their inquiry

2. DATA REQUIREMENTS:
   - Fundamental analysis: Use getFundamentals(symbol) for earnings, ratios, financial health
   - Technical analysis: Use getTechnical(symbol) for price action, indicators, levels
   - Market context: Use getNews(symbol) for recent developments affecting price
   - Determine what data is most relevant to their specific question

3. RESPONSE GUIDELINES:
   - Be direct and professional - no casual greetings or emojis
   - Lead with the most important information first
   - Reference previous conversation naturally when relevant
   - Provide specific data points and comprehensive insights
   - Keep responses under 480 characters for SMS delivery
   - Focus on what matters most to the user's immediate question
   - responses should not feel generic or googleable 

4. CONVERSATION CONTINUITY:
   - If they previously discussed a stock, reference that context
   - Build on previous analysis rather than repeating it
   - Connect new information to their ongoing interests
   - Maintain thread of conversation without restating everything

RESPONSE STRUCTURE:
- Give the most comprehensive answer to their question
- Support with relevant and specific data points
- End with actionable insight or next step
- No introductory phrases like "Here's what I found" or "Let me help"

TOOL CALLING:
Based on your analysis, call the appropriate tools to get the data you need:
- getFundamentals(symbol): For P/E ratios, earnings, financial metrics
- getTechnical(symbol): For price, RSI, support/resistance, trends  
- getNews(symbol): For recent news and sentiment

Then generate a professional, contextually aware response that directly addresses their query."""

        # Call OpenAI to analyze and determine tools needed
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3,
                max_tokens=400,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "getFundamentals",
                            "description": "Get fundamental analysis data for a stock symbol",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "symbol": {"type": "string", "description": "Stock ticker symbol (e.g., AAPL)"}
                                },
                                "required": ["symbol"]
                            }
                        }
                    },
                    {
                        "type": "function", 
                        "function": {
                            "name": "getTechnical",
                            "description": "Get technical analysis data for a stock symbol",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "symbol": {"type": "string", "description": "Stock ticker symbol (e.g., AAPL)"}
                                },
                                "required": ["symbol"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "getNews", 
                            "description": "Get news sentiment data for a stock symbol",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "symbol": {"type": "string", "description": "Stock ticker symbol (e.g., AAPL)"}
                                },
                                "required": ["symbol"]
                            }
                        }
                    }
                ],
                tool_choice="auto"
            )
            
            message_response = response.choices[0].message
            
            # Check if tools were called
            if message_response.tool_calls:
                # Execute the tools
                tool_results = await self._execute_tool_calls(message_response.tool_calls)
                
                # Generate final response with tool data
                final_response = await self._generate_final_response(message, context, tool_results)
                return final_response
            else:
                # No tools needed, use the direct response
                return self._clean_response(message_response.content)
                
        except Exception as e:
            logger.error(f"Analysis and response generation failed: {e}")
            return "I'm processing your request. Please try again in a moment."
    
    async def _execute_tool_calls(self, tool_calls) -> Dict[str, Any]:
        """Execute the tools that OpenAI determined were needed"""
        
        tool_results = {}
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            symbol = arguments.get("symbol", "").upper()
            
            try:
                if function_name == "getFundamentals" and self.fundamental_tool:
                    result = await self.fundamental_tool.execute({
                        "symbol": symbol,
                        "depth": "standard",
                        "user_style": "professional"
                    })
                    if result.get("success"):
                        tool_results[f"fundamentals_{symbol}"] = result["analysis_result"]
                
                elif function_name == "getTechnical" and self.ta_service:
                    result = await self.ta_service.analyze_symbol(symbol)
                    if result:
                        tool_results[f"technical_{symbol}"] = result
                
                elif function_name == "getNews" and self.news_service:
                    result = await self.news_service.get_sentiment(symbol)
                    if result and not result.get('error'):
                        tool_results[f"news_{symbol}"] = result
                        
            except Exception as e:
                logger.warning(f"Tool execution failed for {function_name}({symbol}): {e}")
                continue
        
        return tool_results
    
    async def _generate_final_response(self, message: str, context: Dict, tool_results: Dict) -> str:
        """Generate final response using tool data and conversation context"""
        
        conversation_context = self._format_conversation_context(context)
        formatted_data = self._format_tool_results(tool_results)
        
        final_prompt = f"""Based on the user's message, conversation context, and market data, generate a professional response.

USER MESSAGE: "{message}"

CONVERSATION CONTEXT:
{conversation_context}

MARKET DATA:
{formatted_data}

INSTRUCTIONS:
- Provide a direct, professional response
- Use the market data to support your analysis
- Reference conversation context naturally if relevant
- Be specific and actionable
- Stay under 480 characters
- No casual language or emojis
- Start with the most important information

Generate the response:"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.4,
                max_tokens=200
            )
            
            return self._clean_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Final response generation failed: {e}")
            return "Market analysis processed. Please try your request again."
    
    def _format_conversation_context(self, context: Dict) -> str:
        """Format conversation context for prompts"""
        
        parts = []
        
        if context["conversation_flow"] == "continuing":
            parts.append("CONTINUING CONVERSATION")
            if context["recent_symbols"]:
                parts.append(f"Recently discussed: {', '.join(context['recent_symbols'])}")
            if context["last_topic"]:
                parts.append(f"Last topic: {context['last_topic']}")
        else:
            parts.append("NEW CONVERSATION")
        
        parts.append(f"User style: {context['user_style']}")
        
        return "\n".join(parts)
    
    def _format_tool_results(self, tool_results: Dict) -> str:
        """Format tool results for final response generation"""
        
        if not tool_results:
            return "No market data retrieved"
        
        formatted = []
        
        for key, data in tool_results.items():
            if key.startswith("technical_"):
                symbol = key.replace("technical_", "")
                price_info = data.get('price', {})
                indicators = data.get('indicators', {})
                
                current_price = price_info.get('current', 'N/A')
                change_pct = price_info.get('change_percent', 0)
                rsi = indicators.get('RSI', {}).get('value', 'N/A')
                
                formatted.append(f"{symbol} Technical: ${current_price} ({change_pct:+.1f}%), RSI: {rsi}")
            
            elif key.startswith("fundamentals_"):
                symbol = key.replace("fundamentals_", "")
                if hasattr(data, 'overall_score'):
                    score = data.overall_score
                    health = getattr(data, 'financial_health', 'unknown')
                    formatted.append(f"{symbol} Fundamentals: Score {score}/100, Health: {health}")
            
            elif key.startswith("news_"):
                symbol = key.replace("news_", "")
                news_data = data.get('news_sentiment', {})
                sentiment = news_data.get('sentiment', 'neutral')
                impact = news_data.get('impact_score', 0)
                formatted.append(f"{symbol} News: {sentiment} sentiment (impact: {impact:.1f})")
        
        return "\n".join(formatted) if formatted else "Market data processing..."
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the final response"""
        
        if not response:
            return "I'm processing your request. Please try again."
        
        # Remove common AI artifacts
        artifacts = [
            "Here's the analysis:",
            "Based on the data:",
            "Let me provide:",
            "Here you go:",
            "Certainly!",
            "I'll help you with that."
        ]
        
        for artifact in artifacts:
            if response.startswith(artifact):
                response = response[len(artifact):].strip()
        
        # Remove quotes if they wrap entire response
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        
        # Ensure reasonable length for SMS
        if len(response) > 480:
            response = response[:477] + "..."
        
        return response.strip()
    
    async def _cache_conversation(self, user_phone: str, message: str, response: str, context: Dict):
        """Cache conversation for future context"""
        
        try:
            if not self.cache_service:
                return
            
            timestamp = datetime.now().isoformat()
            
            # Extract symbols from message and response
            symbols = self._extract_symbols_from_text(f"{message} {response}")
            
            conversation_entry = {
                "timestamp": timestamp,
                "user_message": message,
                "bot_response": response,
                "symbols": symbols,
                "topic": self._determine_topic(message),
                "conversation_flow": context.get("conversation_flow", "new")
            }
            
            # Cache in conversation thread
            thread_key = f"conversation_thread:{user_phone}"
            await self.cache_service.add_to_list(thread_key, conversation_entry, max_length=10)
            
            logger.info(f"💾 Conversation cached: {len(symbols)} symbols, topic: {conversation_entry['topic']}")
            
        except Exception as e:
            logger.warning(f"Conversation caching failed: {e}")
    
    def _learn_from_interaction(self, user_phone: str, message: str, response: str):
        """Simple learning from user interaction"""
        
        try:
            # Simple intent extraction for learning
            intent_data = {
                "intent": self._determine_topic(message),
                "symbols": self._extract_symbols_from_text(message),
                "message_length": len(message),
                "response_length": len(response)
            }
            
            self.personality_engine.learn_from_message(user_phone, message, intent_data)
            
        except Exception as e:
            logger.warning(f"Learning from interaction failed: {e}")
    
    def _extract_symbols_from_text(self, text: str) -> List[str]:
        """Extract stock symbols from text"""
        
        symbols = []
        
        # Common company mappings
        company_mappings = {
            'apple': 'AAPL', 'tesla': 'TSLA', 'microsoft': 'MSFT',
            'google': 'GOOGL', 'amazon': 'AMZN', 'meta': 'META',
            'nvidia': 'NVDA', 'amd': 'AMD', 'intel': 'INTC'
        }
        
        text_lower = text.lower()
        for company, symbol in company_mappings.items():
            if company in text_lower:
                symbols.append(symbol)
        
        # Extract ticker patterns (2-5 uppercase letters)
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        potential_tickers = re.findall(ticker_pattern, text)
        for ticker in potential_tickers:
            if ticker not in ['SMS', 'API', 'USA', 'NYSE', 'NASDAQ']:
                symbols.append(ticker)
        
        return list(dict.fromkeys(symbols))  # Remove duplicates
    
    def _determine_topic(self, message: str) -> str:
        """Determine the topic/intent of the message"""
        
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['fundamental', 'fundamentals', 'earnings', 'pe ratio']):
            return "fundamental_analysis"
        elif any(word in message_lower for word in ['technical', 'chart', 'rsi', 'support', 'resistance']):
            return "technical_analysis"
        elif any(word in message_lower for word in ['news', 'sentiment', 'headlines']):
            return "news_analysis"
        elif any(word in message_lower for word in ['portfolio', 'positions']):
            return "portfolio"
        else:
            return "general_analysis"


class ComprehensiveMessageProcessor:
    """
    Simplified main processor that uses the conversation-aware agent
    """
    
    def __init__(self, openai_client, ta_service, personality_engine, 
                 cache_service=None, news_service=None, fundamental_tool=None, 
                 portfolio_service=None, screener_service=None):
        
        # Create the single conversation-aware agent
        self.conversation_agent = ConversationAwareAgent(
            openai_client=openai_client,
            ta_service=ta_service,
            news_service=news_service,
            fundamental_tool=fundamental_tool,
            cache_service=cache_service,
            personality_engine=personality_engine
        )
        
        self.personality_engine = personality_engine
        self.cache_service = cache_service
    
    async def process_message(self, message: str, user_phone: str) -> str:
        """
        Simplified message processing - just call the conversation agent
        """
        
        try:
            logger.info(f"🎯 Processing with Conversation-Aware Agent: '{message}' from {user_phone}")
            
            # Let the conversation agent handle everything
            response = await self.conversation_agent.process_message(message, user_phone)
            
            logger.info(f"✅ Conversation-aware response generated: {len(response)} chars")
            
            return response
            
        except Exception as e:
            logger.error(f"💥 Message processing failed: {e}")
            return "I'm having some technical difficulties, but I'm here to help! Please try again."


# Backward compatibility wrapper
class TradingAgent:
    """Backward compatibility wrapper for the old TradingAgent interface"""
    
    def __init__(self, openai_client, personality_engine):
        self.openai_client = openai_client
        self.personality_engine = personality_engine
        
    async def parse_intent(self, message: str, user_phone: str = None) -> Dict[str, Any]:
        """Legacy interface - simple intent parsing"""
        
        message_lower = message.lower()
        
        # Extract symbols
        symbols = []
        company_mappings = {
            'apple': 'AAPL', 'tesla': 'TSLA', 'microsoft': 'MSFT',
            'google': 'GOOGL', 'amazon': 'AMZN', 'nvidia': 'NVDA'
        }
        
        for company, symbol in company_mappings.items():
            if company in message_lower:
                symbols.append(symbol)
        
        # Extract ticker patterns
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        potential_tickers = re.findall(ticker_pattern, message)
        for ticker in potential_tickers:
            if ticker not in ['SMS', 'API', 'USA']:
                symbols.append(ticker)
        
        # Determine intent
        if any(word in message_lower for word in ['fundamental', 'fundamentals', 'earnings']):
            intent = "fundamental"
        elif any(word in message_lower for word in ['technical', 'chart', 'rsi']):
            intent = "technical"
        elif symbols:
            intent = "analyze"
        else:
            intent = "general"
        
        return {
            "intent": intent,
            "symbols": list(dict.fromkeys(symbols)),
            "parameters": {"urgency": "medium", "complexity": "medium"},
            "requires_tools": ["technical_analysis"] if symbols else [],
            "confidence": 0.8,
            "user_emotion": "neutral",
            "urgency": "medium"
        }


# Export classes
__all__ = [
    'ConversationAwareAgent',  # New simplified agent
    'ComprehensiveMessageProcessor',  # Simplified processor
    'TradingAgent'  # Backward compatibility
]

# Backward compatibility - dummy export for services that still import it
class ToolExecutor:
    """Dummy class for backward compatibility"""
    def __init__(self, *args, **kwargs):
        pass

# Add to exports
__all__ = [
    'ConversationAwareAgent',
    'ComprehensiveMessageProcessor', 
    'TradingAgent',
    'ToolExecutor'  # For backward compatibility
]
