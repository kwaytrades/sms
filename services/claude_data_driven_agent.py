# services/claude_data_driven_agent.py - COMPLETE FIX with Memory Tool
"""
Claude-powered data-driven LLM agent for superior trading analysis
FIXED: All async/await issues that were causing coroutine errors
ADDED: Memory search tool for contextual awareness
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
    4. NOW WITH MEMORY ACCESS during tool calling
    """
    
    def __init__(self, anthropic_client, ta_service, news_service, fundamental_tool, cache_service, personality_engine, memory_manager=None, db_service=None):
        self.anthropic_client = anthropic_client
        self.ta_service = ta_service
        self.news_service = news_service
        self.fundamental_tool = fundamental_tool
        self.cache_service = cache_service
        self.personality_engine = personality_engine
        self.memory_manager = memory_manager
        self.db_service = db_service  # DatabaseService with KeyBuilder
        self._current_user_phone = None  # Store for tool calling context
    
    async def process_message(self, message: str, user_phone: str) -> str:
        """
        Process message using Claude's superior reasoning - FIXED async handling
        """
        try:
            logger.info(f"🎯 Claude processing: '{message}' from {user_phone}")
            
            # Store user_phone for tool calling access
            self._current_user_phone = user_phone
            
            # Save incoming message to memory
            if self.memory_manager:
                try:
                    from services.memory_manager import MessageDirection, AgentType
                    await self.memory_manager.save_message(
                        user_id=user_phone,
                        content=message,
                        direction=MessageDirection.USER,
                        agent_type=AgentType.TRADING
                    )
                except Exception as e:
                    logger.warning(f"Memory save failed: {e}")
            
            # Step 1: Get enhanced conversation context (memory + cache) - FIXED
            if self.memory_manager:
                # Use memory manager for enhanced context
                memory_context = await self.memory_manager.get_context(
                    user_id=user_phone,
                    agent_type=AgentType.TRADING,
                    query=message
                )
                context = self._merge_memory_with_cache_context(memory_context, user_phone)
            else:
                # Fallback to existing cache context - ✅ AWAIT ADDED
                context = await self._get_conversation_context(user_phone)
            
            # ✅ AWAIT ADDED for user profile
            user_profile = await self._get_user_profile(user_phone)
            
            # Step 2: Claude analyzes query and calls tools directly - ✅ AWAIT ADDED (NOW WITH MEMORY ACCESS)
            tool_results = await self._claude_driven_tool_calling(message, context)
            
            # Step 3: Claude generates response with all context - ✅ AWAIT ADDED
            response = await self._generate_intelligent_response(
                message, context, tool_results, user_profile
            )
            
            # Step 4: Save bot response to memory and cache
            if self.memory_manager:
                try:
                    await self.memory_manager.save_message(
                        user_id=user_phone,
                        content=response,
                        direction=MessageDirection.BOT,
                        agent_type=AgentType.TRADING
                    )
                except Exception as e:
                    logger.warning(f"Memory save failed: {e}")
            
            # ✅ AWAIT ADDED for conversation caching
            await self._cache_conversation(user_phone, message, response, context)
            
            logger.info(f"✅ Claude response generated: {len(response)} chars")
            return response
            
        except Exception as e:
            logger.error(f"💥 Claude processing failed: {e}")
            return "Market analysis processing. Please try again shortly."
    
    def _merge_memory_with_cache_context(self, memory_context: Dict, user_phone: str) -> Dict:
        """Merge memory manager context with existing cache context format"""
        
        # Extract from memory context
        stm = memory_context.get('short_term_memory', [])
        summaries = memory_context.get('conversation_summaries', [])
        profile = memory_context.get('user_profile', {})
        emotional_context = memory_context.get('emotional_context', {})
        
        # Extract recent symbols from STM and summaries
        recent_symbols = []
        for msg in stm[:5]:  # Last 5 messages
            topics = msg.get('topics', [])
            recent_symbols.extend([t for t in topics if t.isupper() and len(t) <= 5])
        
        for summary in summaries[:2]:  # Recent summaries
            topics = summary.get('topics', [])
            recent_symbols.extend([t for t in topics if t.isupper() and len(t) <= 5])
        
        # Remove duplicates, keep order
        unique_symbols = []
        for symbol in recent_symbols:
            if symbol not in unique_symbols:
                unique_symbols.append(symbol)
        
        # Determine conversation flow
        conversation_flow = "continuing" if len(stm) > 0 else "new"
        
        # Get last topic from recent messages
        last_topic = None
        if stm:
            last_msg = stm[0]
            if 'technical' in last_msg.get('content', '').lower():
                last_topic = "technical_analysis"
            elif 'fundamental' in last_msg.get('content', '').lower():
                last_topic = "fundamental_analysis"
            elif 'news' in last_msg.get('content', '').lower():
                last_topic = "news_analysis"
            else:
                last_topic = "general_analysis"
        
        # Return in your existing format
        return {
            "recent_symbols": unique_symbols[:3],
            "conversation_flow": conversation_flow,
            "last_topic": last_topic,
            "memory_enhanced": True,
            "emotional_state": emotional_context.get('current_emotion'),
            "user_profile": profile,
            "message_count": len(stm),
            "conversation_summaries": summaries[:2]
        }
    
    async def _claude_driven_tool_calling(self, message: str, context: Dict) -> Dict[str, Any]:
        """
        Let Claude decide which tools to call and execute them with real data - NOW WITH MEMORY ACCESS
        """
        
        context_summary = self._format_context_summary(context)
        
        tool_calling_prompt = f"""You are a trading analyst with access to market data tools AND conversation memory. Analyze the user's request and call the appropriate tools.

USER REQUEST: "{message}"
CONVERSATION CONTEXT: {context_summary}

AVAILABLE TOOLS:
- getTechnical(symbol): Get technical analysis, price, RSI, support/resistance
- getFundamentals(symbol): Get P/E ratio, financial health, growth metrics  
- getNews(symbol): Get recent news sentiment and market impact
- searchMemory(query): Search conversation history for context about previously discussed stocks

IMPORTANT USAGE RULES:
- If user asks vague questions like "What stock am I talking about?", "How's it doing?", "Should I buy it?" 
- FIRST call searchMemory("recent stock discussed") to find what they were talking about
- THEN call the appropriate analysis tools with the discovered stock symbol
- Use searchMemory when you need context about previous conversations

Extract any stock symbols and call the appropriate tools now."""

        try:
            # Claude's function calling format - NOW WITH MEMORY TOOL
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
                },
                # ✅ NEW: Memory search tool
                {
                    "name": "searchMemory",
                    "description": "Search conversation history to find context about previously discussed stocks or topics",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to find relevant conversation context (e.g. 'recent stock discussed', 'last symbol mentioned')"
                            }
                        },
                        "required": ["query"]
                    }
                }
            ]
            
            # ✅ FIXED: This was already correct, but ensuring it's awaited
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
                        
                        logger.info(f"🔧 Claude requested: {function_name}({arguments})")
                        
                        # Execute the actual tool and get real data
                        if function_name == "getTechnical" and self.ta_service:
                            symbol = arguments.get("symbol", "").upper()
                            # ✅ FIXED: Added await for async ta_service call
                            result = await self.ta_service.analyze_symbol(symbol)
                            if result:
                                tool_results[f"technical_{symbol}"] = result
                        
                        # ✅ FIXED: FAEngine is synchronous, no await needed (this was correct)
                        elif function_name == "getFundamentals" and self.fundamental_tool:
                            symbol = arguments.get("symbol", "").upper()
                            # FAEngine uses execute() method, not analyze_fundamentals()
                            result = self.fundamental_tool.execute({
                                "symbol": symbol,
                                "depth": "standard", 
                                "user_style": "professional"
                            })
                            if result.get("success"):
                                tool_results[f"fundamental_{symbol}"] = result["analysis_result"]
                        
                        elif function_name == "getNews" and self.news_service:
                            symbol = arguments.get("symbol", "").upper()
                            # ✅ FIXED: Added await for async news_service call
                            result = await self.news_service.get_sentiment(symbol)
                            if result and not result.get('error'):
                                tool_results[f"news_{symbol}"] = result
                        
                        # ✅ NEW: Handle memory search tool
                        elif function_name == "searchMemory" and self.memory_manager:
                            query = arguments.get("query", "")
                            user_phone = self._current_user_phone
                            
                            if user_phone:
                                try:
                                    from services.memory_manager import AgentType
                                    memory_result = await self.memory_manager.get_context(
                                        user_id=user_phone,
                                        agent_type=AgentType.TRADING,
                                        query=query
                                    )
                                    
                                    # Extract recent symbols and messages from memory
                                    stm = memory_result.get('short_term_memory', [])
                                    recent_symbols = []
                                    recent_messages = []
                                    
                                    for msg in stm[:5]:  # Last 5 messages
                                        content = msg.get('content', '')
                                        direction = msg.get('direction', 'unknown')
                                        if content:
                                            recent_messages.append(f"{direction}: {content}")
                                        
                                        topics = msg.get('topics', [])
                                        recent_symbols.extend([t for t in topics if t.isupper() and len(t) <= 5])
                                    
                                    # Remove duplicate symbols, keep order
                                    unique_symbols = []
                                    for symbol in recent_symbols:
                                        if symbol not in unique_symbols:
                                            unique_symbols.append(symbol)
                                    
                                    tool_results["memory_search"] = {
                                        "recent_symbols": unique_symbols[:3],
                                        "most_recent_symbol": unique_symbols[0] if unique_symbols else None,
                                        "recent_messages": recent_messages[:3],
                                        "context_found": len(stm) > 0,
                                        "query_used": query
                                    }
                                    
                                    logger.info(f"🧠 Memory search found: {unique_symbols[:3]}")
                                    
                                except Exception as e:
                                    logger.error(f"Memory search failed: {e}")
                                    tool_results["memory_search"] = {"error": "Memory search failed", "query_used": query}
                            else:
                                tool_results["memory_search"] = {"error": "No user context available", "query_used": query}
            
            return tool_results
            
        except Exception as e:
            logger.error(f"Claude tool calling failed: {e}")
            return {}
    
    async def _generate_intelligent_response(
        self, message: str, context: Dict, tool_results: Dict, user_profile: Dict
    ) -> str:
        """
        Let Claude generate the response with full context - FIXED async
        """
        
        # Format everything for Claude
        tool_data = self._format_raw_data_for_claude(tool_results)
        context_summary = self._format_context_summary(context)
        personality_info = self._format_user_personality(user_profile)
        
        # Enhanced prompt with memory context
        memory_info = ""
        if context.get("memory_enhanced"):
            emotional_state = context.get("emotional_state")
            message_count = context.get("message_count", 0)
            
            if emotional_state:
                emotion_type = emotional_state.get("emotion_type", "neutral")
                memory_info = f"\nEMOTIONAL CONTEXT: User seems {emotion_type} based on recent messages. Adapt your tone accordingly."
            
            if message_count > 3:
                memory_info += f"\nCONVERSATION DEPTH: This is message #{message_count} in your conversation - user expects continuity and personalized responses."
        
        # Single comprehensive prompt - let Claude handle everything
        comprehensive_prompt = f"""You are a hyper-personalized SMS trading assistant with exceptional financial reasoning abilities. The user just asked: "{message}"

CONVERSATION CONTEXT:
{context_summary}

USER PERSONALITY & COMMUNICATION STYLE:
{personality_info}

REAL MARKET DATA AVAILABLE:
{tool_data}
{memory_info}

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
9. Reference previous conversations naturally if relevant

Key Claude strengths to leverage:
- Superior at synthesizing multiple data sources
- Excellent at understanding context and nuance
- Natural conversational ability
- Strong financial reasoning

Generate your response now - be brilliant but concise:"""

        try:
            # ✅ FIXED: This was already correct, ensuring proper await
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
            if key == "memory_search":
                # Special formatting for memory search results
                result_text = f"=== MEMORY SEARCH RESULTS ===\n{json.dumps(data, indent=2, default=str)}"
                formatted_results.append(result_text)
            else:
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
                context_text = f"Previously discussed: {', '.join(recent_symbols[:3])}. Last topic: {last_topic}. User expects continuity."
                
                # Add memory enhancement info
                if context.get("memory_enhanced"):
                    context_text += f" [Enhanced with {context.get('message_count', 0)} message conversation history]"
                
                return context_text
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
        """Get conversation context using KeyBuilder - FIXED async"""
        
        context = {
            "recent_symbols": [],
            "conversation_flow": "new",
            "last_topic": None
        }
        
        try:
            # Try KeyBuilder first for conversation context
            if self.db_service and hasattr(self.db_service, 'key_builder'):
                try:
                    # ✅ FIXED: Added await for KeyBuilder call
                    context_data = await self.db_service.key_builder.get_user_context(user_phone)
                    if context_data:
                        context["recent_symbols"] = context_data.get("recent_symbols", [])
                        context["conversation_flow"] = context_data.get("conversation_flow", "new")
                        context["last_topic"] = context_data.get("last_topic", None)
                        return context
                except Exception as e:
                    logger.warning(f"KeyBuilder context lookup failed: {e}")
            
            # Fallback to cache service
            if self.cache_service:
                thread_key = f"conversation_thread:{user_phone}"
                # ✅ FIXED: Added await for cache service call
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
        """Get user personality profile using KeyBuilder - FIXED async"""
        
        try:
            # Try KeyBuilder first for user profile
            if self.db_service and hasattr(self.db_service, 'key_builder'):
                try:
                    # ✅ FIXED: Added await for KeyBuilder profile call
                    user_profile = await self.db_service.key_builder.get_user_profile(user_phone)
                    if user_profile:
                        return user_profile
                except Exception as e:
                    logger.warning(f"KeyBuilder profile lookup failed: {e}")
            
            # Fallback to personality engine
            if self.personality_engine and hasattr(self.personality_engine, 'user_profiles'):
                return self.personality_engine.user_profiles.get(user_phone, {})
            return {}
        except Exception as e:
            logger.warning(f"Profile retrieval failed: {e}")
            return {}
    
    async def _cache_conversation(self, user_phone: str, message: str, response: str, context: Dict):
        """Cache conversation (read-only for DB, writes to cache only) - FIXED async"""
        
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
            # ✅ FIXED: Added await for cache service call
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
                 portfolio_service=None, screener_service=None, memory_manager=None, db_service=None):
        
        self.claude_agent = ClaudeDataDrivenAgent(
            anthropic_client=anthropic_client,
            ta_service=ta_service,
            news_service=news_service,
            fundamental_tool=fundamental_tool,
            cache_service=cache_service,
            personality_engine=personality_engine,
            memory_manager=memory_manager,
            db_service=db_service  # Pass DatabaseService with KeyBuilder
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
