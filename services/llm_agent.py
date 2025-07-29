# services/data_driven_agent.py
"""
Data-driven LLM agent that calls tools directly, sees real data,
then generates contextual prompts for the response agent
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from loguru import logger
from datetime import datetime
from openai import AsyncOpenAI

class DataDrivenAgent:
    """
    LLM agent that:
    1. Calls tools directly based on user query
    2. Sees and analyzes real market data
    3. Generates contextual prompt for response agent
    4. Response agent executes with format restrictions
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
        Process message using data-driven approach:
        1. LLM calls tools and sees real data
        2. LLM generates contextual prompt
        3. Response agent executes with restrictions
        """
        try:
            logger.info(f"🎯 Data-driven processing: '{message}' from {user_phone}")
            
            # Step 1: Get conversation context
            context = await self._get_conversation_context(user_phone)
            
            # Step 2: LLM analyzes query and calls tools directly
            tool_results = await self._llm_driven_tool_calling(message, context)
            
            # Step 3: LLM sees real data and generates contextual prompt
            contextual_prompt = await self._generate_contextual_prompt(message, context, tool_results)
            
            # Step 4: Response agent executes with format restrictions
            response = await self._execute_response_agent(contextual_prompt)
            
            # Step 5: Cache and learn
            await self._cache_conversation(user_phone, message, response, context)
            
            logger.info(f"✅ Data-driven response: {len(response)} chars")
            return response
            
        except Exception as e:
            logger.error(f"💥 Data-driven processing failed: {e}")
            return "Market analysis processing. Please try again shortly."
    
    async def _llm_driven_tool_calling(self, message: str, context: Dict) -> Dict[str, Any]:
        """
        Let LLM decide which tools to call and execute them with real data
        """
        
        # Build context for LLM decision making
        context_summary = self._format_context_summary(context)
        
        tool_calling_prompt = f"""You are a trading analyst with access to market data tools. Analyze the user's request and call the appropriate tools to get REAL data.

USER REQUEST: "{message}"
CONVERSATION CONTEXT: {context_summary}

AVAILABLE TOOLS:
- getTechnical(symbol): Get technical analysis, price, RSI, support/resistance
- getFundamentals(symbol): Get P/E ratio, financial health, growth metrics  
- getNews(symbol): Get recent news sentiment and market impact

INSTRUCTIONS:
1. Determine which tools to call based on the user's request
2. Call tools to get REAL market data
3. You will see the actual data returned from each tool
4. Based on the REAL data you receive, you'll then generate an analysis prompt

Extract any stock symbols and call the appropriate tools now."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": tool_calling_prompt}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "getTechnical",
                            "description": "Get technical analysis data for a stock symbol",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "symbol": {"type": "string", "description": "Stock ticker symbol"}
                                },
                                "required": ["symbol"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "getFundamentals", 
                            "description": "Get fundamental analysis data for a stock symbol",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "symbol": {"type": "string", "description": "Stock ticker symbol"}
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
                                    "symbol": {"type": "string", "description": "Stock ticker symbol"}
                                },
                                "required": ["symbol"]
                            }
                        }
                    }
                ],
                tool_choice="auto",
                temperature=0.1
            )
            
            message_response = response.choices[0].message
            
            # Execute tools and collect real data
            tool_results = {}
            
            if message_response.tool_calls:
                for tool_call in message_response.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    symbol = arguments.get("symbol", "").upper()
                    
                    logger.info(f"🔧 LLM requested: {function_name}({symbol})")
                    
                    # Execute the actual tool and get real data
                    if function_name == "getTechnical" and self.ta_service:
                        result = await self.ta_service.analyze_symbol(symbol)
                        if result:
                            tool_results[f"technical_{symbol}"] = result
                    
                    elif function_name == "getFundamentals" and self.fundamental_tool:
                        result = await self.fundamental_tool.execute({
                            "symbol": symbol,
                            "depth": "standard", 
                            "user_style": "professional"
                        })
                        if result.get("success"):
                            tool_results[f"fundamental_{symbol}"] = result["analysis_result"]
                    
                    elif function_name == "getNews" and self.news_service:
                        result = await self.news_service.get_sentiment(symbol)
                        if result and not result.get('error'):
                            tool_results[f"news_{symbol}"] = result
            
            return tool_results
            
        except Exception as e:
            logger.error(f"Tool calling failed: {e}")
            return {}
    
    async def _generate_contextual_prompt(self, message: str, context: Dict, tool_results: Dict) -> str:
        """
        LLM sees real data and generates a contextual prompt for the response agent
        """
        
        # Format the raw data for LLM analysis - MODIFIED TO PASS RAW DATA
        data_summary = self._format_raw_tool_results(tool_results)
        context_summary = self._format_context_summary(context)
        
        prompt_generation_request = f"""You have called tools and received REAL market data. Now generate a contextual analysis prompt for the response agent.

USER'S ORIGINAL REQUEST: "{message}"
CONVERSATION CONTEXT: {context_summary}

RAW MARKET DATA RECEIVED:
{data_summary}

YOUR TASK: Generate a contextual prompt that will guide the response agent to create an intelligent, data-driven response.

The prompt should:
1. Incorporate the RAW data you received (prices, indicators, news content)
2. Highlight the most important insights from the raw data
3. Connect different raw data points into a coherent narrative  
4. Address the user's specific question with the raw data
5. Include any important context or timing considerations
6. Provide clear guidance on tone and focus for the response

Generate a contextual prompt that will result in an intelligent, professional trading response:"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_generation_request}],
                temperature=0.3,
                max_tokens=400
            )
            
            contextual_prompt = response.choices[0].message.content.strip()
            logger.info(f"📝 Generated contextual prompt: {len(contextual_prompt)} chars")
            
            return contextual_prompt
            
        except Exception as e:
            logger.error(f"Contextual prompt generation failed: {e}")
            return f"Provide a professional analysis of the market data for the user's request: {message}"
    
    async def _execute_response_agent(self, contextual_prompt: str) -> str:
        """
        Execute response agent with the contextual prompt and format restrictions
        """
        
        # Add format restrictions to the contextual prompt
        format_restricted_prompt = f"""{contextual_prompt}

CRITICAL FORMAT REQUIREMENTS:
- Keep response under 480 characters (3 SMS segments)
- Be direct and professional - no casual greetings or emojis
- Lead with the most important insight first
- Include specific data points and actionable information
- No introductory phrases like "Based on the analysis" or "Here's what I found"
- Focus on what matters most for trading decisions

Generate a professional, concise response:"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": format_restricted_prompt}],
                temperature=0.4,
                max_tokens=200
            )
            
            generated_response = response.choices[0].message.content.strip()
            return self._clean_final_response(generated_response)
            
        except Exception as e:
            logger.error(f"Response agent execution failed: {e}")
            return "Market analysis completed. Please try your request again."
    
    def _format_raw_tool_results(self, tool_results: Dict) -> str:
        """Format RAW tool results for LLM analysis - NO PROCESSING"""
        
        if not tool_results:
            return "No market data retrieved"
        
        formatted_results = []
        
        for key, data in tool_results.items():
            if key.startswith("technical_"):
                symbol = key.replace("technical_", "")
                
                # Pass raw technical data structure
                result_text = f"RAW TECHNICAL DATA for {symbol}:\n"
                result_text += f"Full data structure: {json.dumps(data, indent=2, default=str)}"
                
                formatted_results.append(result_text)
            
            elif key.startswith("fundamental_"):
                symbol = key.replace("fundamental_", "")
                
                # Pass raw fundamental data structure  
                result_text = f"RAW FUNDAMENTAL DATA for {symbol}:\n"
                result_text += f"Full data structure: {json.dumps(data, indent=2, default=str)}"
                
                formatted_results.append(result_text)
            
            elif key.startswith("news_"):
                symbol = key.replace("news_", "")
                
                # Pass raw news data structure
                result_text = f"RAW NEWS DATA for {symbol}:\n"
                result_text += f"Full data structure: {json.dumps(data, indent=2, default=str)}"
                
                formatted_results.append(result_text)
        
        return "\n\n".join(formatted_results)
    
    def _format_context_summary(self, context: Dict) -> str:
        """Format conversation context"""
        
        if context.get("conversation_flow") == "continuing":
            recent_symbols = context.get("recent_symbols", [])
            last_topic = context.get("last_topic", "")
            
            if recent_symbols:
                return f"Previously discussed: {', '.join(recent_symbols[:3])}. Last topic: {last_topic}."
            else:
                return "Continuing conversation."
        else:
            return "New conversation."
    
    def _clean_final_response(self, response: str) -> str:
        """Clean and optimize final response"""
        
        # Remove AI artifacts
        artifacts = [
            "Based on the analysis:", "Looking at the data:", "According to the information:",
            "Here's what I found:", "The data shows:", "In summary:", "To summarize:"
        ]
        
        for artifact in artifacts:
            if response.startswith(artifact):
                response = response[len(artifact):].strip()
        
        # Ensure length limit
        if len(response) > 480:
            response = response[:477] + "..."
        
        return response.strip()
    
    async def _get_conversation_context(self, user_phone: str) -> Dict[str, Any]:
        """Get conversation context"""
        
        context = {
            "recent_symbols": [],
            "conversation_flow": "new",
            "last_topic": None
        }
        
        try:
            if self.cache_service:
                thread_key = f"conversation_thread:{user_phone}"
                recent_messages = await self.cache_service.get_list(thread_key, limit=3)
                
                if recent_messages:
                    context["conversation_flow"] = "continuing"
                    # Extract symbols and topics
                    all_symbols = []
                    for msg in recent_messages:
                        all_symbols.extend(msg.get("symbols", []))
                    context["recent_symbols"] = list(dict.fromkeys(all_symbols))[:3]
                    
                    if recent_messages:
                        context["last_topic"] = recent_messages[0].get("topic", "analysis")
            
            return context
            
        except Exception as e:
            logger.warning(f"Context retrieval failed: {e}")
            return context
    
    async def _cache_conversation(self, user_phone: str, message: str, response: str, context: Dict):
        """Cache conversation"""
        
        try:
            if not self.cache_service:
                return
            
            # Extract symbols and cache conversation
            symbols = self._extract_symbols_from_text(f"{message} {response}")
            
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_message": message,
                "bot_response": response,
                "symbols": symbols,
                "topic": self._determine_topic(message)
            }
            
            thread_key = f"conversation_thread:{user_phone}"
            await self.cache_service.add_to_list(thread_key, conversation_entry, max_length=5)
            
        except Exception as e:
            logger.warning(f"Conversation caching failed: {e}")
    
    def _extract_symbols_from_text(self, text: str) -> List[str]:
        """Extract symbols from text"""
        
        import re
        symbols = []
        
        # Company mappings
        company_mappings = {
            'apple': 'AAPL', 'tesla': 'TSLA', 'microsoft': 'MSFT',
            'google': 'GOOGL', 'amazon': 'AMZN', 'nvidia': 'NVDA',
            'palantir': 'PLTR'
        }
        
        text_lower = text.lower()
        for company, symbol in company_mappings.items():
            if company in text_lower:
                symbols.append(symbol)
        
        # Extract tickers
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        potential_tickers = re.findall(ticker_pattern, text)
        for ticker in potential_tickers:
            if ticker not in ['SMS', 'API', 'USA']:
                symbols.append(ticker)
        
        return list(dict.fromkeys(symbols))
    
    def _determine_topic(self, message: str) -> str:
        """Determine topic"""
        
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['fundamental', 'fundamentals']):
            return "fundamental_analysis"
        elif any(word in message_lower for word in ['technical', 'chart']):
            return "technical_analysis"
        elif any(word in message_lower for word in ['news', 'sentiment']):
            return "news_analysis"
        else:
            return "general_analysis"


# Drop-in replacement for existing processor
class ComprehensiveMessageProcessor:
    """Enhanced processor using data-driven agent"""
    
    def __init__(self, openai_client, ta_service, personality_engine, 
                 cache_service=None, news_service=None, fundamental_tool=None, 
                 portfolio_service=None, screener_service=None):
        
        self.data_driven_agent = DataDrivenAgent(
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
        """Process using data-driven agent"""
        
        try:
            response = await self.data_driven_agent.process_message(message, user_phone)
            return response
            
        except Exception as e:
            logger.error(f"💥 Data-driven processing failed: {e}")
            return "Market analysis processing. Please try again shortly."


# Backward compatibility
class TradingAgent:
    def __init__(self, openai_client, personality_engine):
        pass
    
    async def parse_intent(self, message: str, user_phone: str = None) -> Dict[str, Any]:
        return {"intent": "analyze", "symbols": [], "confidence": 0.8}

class ToolExecutor:
    def __init__(self, *args, **kwargs):
        pass

__all__ = ['DataDrivenAgent', 'ComprehensiveMessageProcessor', 'TradingAgent', 'ToolExecutor']
