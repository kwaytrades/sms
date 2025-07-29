# services/claude_data_driven_agent.py
"""
Claude-powered data-driven LLM agent for superior trading analysis
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from loguru import logger
from datetime import datetime
import anthropic

class ClaudeDataDrivenAgent:
    """
    Claude-powered LLM agent that:
    1. Calls tools directly based on user query
    2. Passes raw data to Claude
    3. Claude handles everything - analysis, tone, conversation style
    """
    
    def __init__(self, anthropic_client, ta_service, news_service, fundamental_tool, cache_service, personality_engine):
        self.anthropic_client = anthropic_client
        self.ta_service = ta_service
        self.news_service = news_service
        self.fundamental_tool = fundamental_tool
        self.cache_service = cache_service
        self.personality_engine = personality_engine
    
    async def process_message(self, message: str, user_phone: str) -> str:
        """
        Process message using Claude's superior reasoning:
        1. Claude calls tools and gets raw data
        2. Claude generates response directly with all context
        """
        try:
            logger.info(f"🎯 Claude processing: '{message}' from {user_phone}")
            
            # Step 1: Get conversation context and user profile
            context = await self._get_conversation_context(user_phone)
            user_profile = await self._get_user_profile(user_phone)
            
            # Step 2: Claude analyzes query and calls tools directly
            tool_results = await self._claude_driven_tool_calling(message, context)
            
            # Step 3: Claude generates response with all context
            response = await self._generate_intelligent_response(
                message, context, tool_results, user_profile
            )
            
            # Step 4: Cache and learn
            await self._cache_conversation(user_phone, message, response, context)
            
            logger.info(f"✅ Claude response generated: {len(response)} chars")
            return response
            
        except Exception as e:
            logger.error(f"💥 Claude processing failed: {e}")
            return "Market analysis processing. Please try again shortly."
    
    async def _claude_driven_tool_calling(self, message: str, context: Dict) -> Dict[str, Any]:
        """
        Let Claude decide which tools to call and execute them with real data
        """
        
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

Extract any stock symbols and call the appropriate tools now."""

        try:
            # Claude's function calling format
            tools = [
                {
                    "name": "getTechnical",
                    "description": "Get technical analysis data for a stock symbol",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Stock ticker symbol"
                            }
                        },
                        "required": ["symbol"]
                    }
                },
                {
                    "name": "getFundamentals",
                    "description": "Get fundamental analysis data for a stock symbol",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Stock ticker symbol"
                            }
                        },
                        "required": ["symbol"]
                    }
                },
                {
                    "name": "getNews",
                    "description": "Get news sentiment data for a stock symbol",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Stock ticker symbol"
                            }
                        },
                        "required": ["symbol"]
                    }
                }
            ]
            
            response = await self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                temperature=0.1,
                tools=tools,
                messages=[{
                    "role": "user",
                    "content": tool_calling_prompt
                }]
            )
            
            # Execute tools based on Claude's decisions
            tool_results = {}
            
            if response.content:
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        function_name = content_block.name
                        arguments = content_block.input
                        symbol = arguments.get("symbol", "").upper()
                        
                        logger.info(f"🔧 Claude requested: {function_name}({symbol})")
                        
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
            logger.error(f"Claude tool calling failed: {e}")
            return {}
    
    async def _generate_intelligent_response(
        self, message: str, context: Dict, tool_results: Dict, user_profile: Dict
    ) -> str:
        """
        Let Claude generate the response with full context - leverage Claude's superior reasoning
        """
        
        # Format everything for Claude
        tool_data = self._format_raw_data_for_claude(tool_results)
        context_summary = self._format_context_summary(context)
        personality_info = self._format_user_personality(user_profile)
        
        # Single comprehensive prompt - let Claude handle everything
        comprehensive_prompt = f"""You are a hyper-personalized SMS trading assistant with exceptional financial reasoning abilities. The user just asked: "{message}"

CONVERSATION CONTEXT:
{context_summary}

USER PERSONALITY & COMMUNICATION STYLE:
{personality_info}

REAL MARKET DATA AVAILABLE:
{tool_data}

YOUR TASK:
Respond to the user's question naturally and helpfully using your superior analytical reasoning. You have access to real market data above.

IMPORTANT GUIDELINES:
1. Match the user's communication style and personality perfectly
2. Answer their specific question (advice, analysis, data, whatever they asked for)
3. Use the real market data to inform your response with sophisticated reasoning
4. Be conversational when they want advice, analytical when they want analysis
5. Keep responses SMS-friendly (under 480 characters for SMS delivery)
6. Be helpful, actionable, and demonstrate deep market understanding
7. Don't dump unnecessary technical data unless they specifically want it
8. Use your superior reasoning to connect data points meaningfully

Key Claude strengths to leverage:
- Superior at synthesizing multiple data sources
- Excellent at understanding context and nuance
- Natural conversational ability
- Strong financial reasoning

Generate your response now - be brilliant but concise:"""

        try:
            response = await self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                temperature=0.4,
                messages=[{
                    "role": "user",
                    "content": comprehensive_prompt
                }]
            )
            
            # Extract text from Claude's response
            generated_response = ""
            if response.content:
                for content_block in response.content:
                    if content_block.type == "text":
                        generated_response += content_block.text
            
            return self._clean_final_response(generated_response)
            
        except Exception as e:
            logger.error(f"Claude response generation failed: {e}")
            return "Market analysis completed. Please try your request again."
    
    def _format_raw_data_for_claude(self, tool_results: Dict) -> str:
        """Format raw tool results for Claude - keep it comprehensive"""
        
        if not tool_results:
            return "No market data retrieved"
        
        formatted_results = []
        
        for key, data in tool_results.items():
            symbol = key.split('_', 1)[1] if '_' in key else 'UNKNOWN'
            data_type = key.split('_', 1)[0] if '_' in key else 'unknown'
            
            # Claude can handle more complex data structures better than GPT
            result_text = f"=== {data_type.upper()} DATA for {symbol} ===\n{json.dumps(data, indent=2, default=str)}"
            formatted_results.append(result_text)
        
        return "\n\n".join(formatted_results)
    
    def _format_context_summary(self, context: Dict) -> str:
        """Format conversation context for Claude"""
        
        if context.get("conversation_flow") == "continuing":
            recent_symbols = context.get("recent_symbols", [])
            last_topic = context.get("last_topic", "")
            
            if recent_symbols:
                return f"Previously discussed: {', '.join(recent_symbols[:3])}. Last topic: {last_topic}. User expects continuity."
            else:
                return "Continuing conversation with this user."
        else:
            return "New conversation - establish rapport and understand user needs."
    
    def _format_user_personality(self, user_profile: Dict) -> str:
        """Format user personality for Claude's superior understanding"""
        
        if not user_profile:
            return "No personality data available - use professional but friendly default style, adapt based on user's message tone"
        
        comm_style = user_profile.get('communication_style', {})
        trading_style = user_profile.get('trading_personality', {})
        
        personality_summary = f"""Communication Preferences:
- Formality level: {comm_style.get('formality', 'casual')}
- Energy/enthusiasm: {comm_style.get('energy', 'moderate')}  
- Emoji usage: {comm_style.get('emoji_usage', 'some')}
- Technical depth preference: {comm_style.get('technical_depth', 'medium')}

Trading Profile:
- Experience level: {trading_style.get('experience_level', 'intermediate')}
- Risk tolerance: {trading_style.get('risk_tolerance', 'moderate')}
- Trading style: {trading_style.get('trading_style', 'swing')}
- Previous interests: {trading_style.get('interests', 'general markets')}

Match this user's style precisely - Claude excels at personality adaptation."""
        
        return personality_summary
    
    def _clean_final_response(self, response: str) -> str:
        """Minimal cleaning - let Claude handle most of it"""
        
        # Claude is generally better at following length constraints, but ensure SMS limit
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
    
    async def _get_user_profile(self, user_phone: str) -> Dict:
        """Get user personality profile"""
        
        try:
            if self.personality_engine and hasattr(self.personality_engine, 'user_profiles'):
                return self.personality_engine.user_profiles.get(user_phone, {})
            return {}
        except Exception as e:
            logger.warning(f"Profile retrieval failed: {e}")
            return {}
    
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
                "topic": self._determine_topic(message),
                "agent": "claude"  # Track which agent generated response
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


# Drop-in replacement processor with Claude
class ClaudeMessageProcessor:
    """Claude-powered processor for superior trading analysis"""
    
    def __init__(self, anthropic_client, ta_service, personality_engine, 
                 cache_service=None, news_service=None, fundamental_tool=None, 
                 portfolio_service=None, screener_service=None):
        
        self.claude_agent = ClaudeDataDrivenAgent(
            anthropic_client=anthropic_client,
            ta_service=ta_service,
            news_service=news_service,
            fundamental_tool=fundamental_tool,
            cache_service=cache_service,
            personality_engine=personality_engine
        )
        
        self.personality_engine = personality_engine
        self.cache_service = cache_service
    
    async def process_message(self, message: str, user_phone: str) -> str:
        """Process using Claude's superior reasoning"""
        
        try:
            response = await self.claude_agent.process_message(message, user_phone)
            return response
            
        except Exception as e:
            logger.error(f"💥 Claude processing failed: {e}")
            return "Market analysis processing. Please try again shortly."


# Hybrid processor that can use both Claude and OpenAI
class HybridProcessor:
    """
    Processor that can switch between Claude and OpenAI based on request type
    """
    
    def __init__(self, claude_processor, openai_processor):
        self.claude_processor = claude_processor
        self.openai_processor = openai_processor
    
    async def process_message(self, message: str, user_phone: str, prefer_claude: bool = True) -> str:
        """
        Process with preferred agent, fallback to other if needed
        """
        
        try:
            if prefer_claude:
                return await self.claude_processor.process_message(message, user_phone)
            else:
                return await self.openai_processor.process_message(message, user_phone)
                
        except Exception as e:
            logger.warning(f"Primary agent failed: {e}, trying fallback")
            
            # Fallback to other agent
            try:
                if prefer_claude:
                    return await self.openai_processor.process_message(message, user_phone)
                else:
                    return await self.claude_processor.process_message(message, user_phone)
            except Exception as e2:
                logger.error(f"Both agents failed: {e2}")
                return "Market analysis processing. Please try again shortly."


__all__ = ['ClaudeDataDrivenAgent', 'ClaudeMessageProcessor', 'HybridProcessor']
