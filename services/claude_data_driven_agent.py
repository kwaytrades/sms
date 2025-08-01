# services/claude_data_driven_agent.py
"""
Claude-powered data-driven LLM agent for superior trading analysis
Updated with KeyBuilder integration for centralized data access
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
    
    def __init__(self, anthropic_client, ta_service, news_service, fundamental_tool, cache_service, personality_engine, memory_manager=None, db_service=None):
        self.anthropic_client = anthropic_client
        self.ta_service = ta_service
        self.news_service = news_service
        self.fundamental_tool = fundamental_tool
        self.cache_service = cache_service
        self.personality_engine = personality_engine
        self.memory_manager = memory_manager
        self.db_service = db_service  # DatabaseService with KeyBuilder
    
    async def process_message(self, message: str, user_phone: str) -> str:
        """
        Process message using Claude's superior reasoning:
        1. Claude calls tools and gets raw data
        2. Claude generates response directly with all context
        """
        try:
            logger.info(f"🎯 Claude processing: '{message}' from {user_phone}")
            
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
            
            # Step 1: Get enhanced conversation context (memory + cache)
            if self.memory_manager:
                # Use memory manager for enhanced context
                memory_context = await self.memory_manager.get_context(
                    user_id=user_phone,
                    agent_type=AgentType.TRADING,
                    query=message
                )
                context = self._merge_memory_with_cache_context(memory_context, user_phone)
            else:
                # Fallback to existing cache context
                context = await self._get_conversation_context(user_phone)
            
            user_profile = await self._get_user_profile(user_phone)
            
            # Step 2: Claude analyzes query and calls tools directly
            tool_results = await self._claude_driven_tool_calling(message, context)
            
            # Step 3: Claude generates response with all context
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
    # services/claude_data_driven_agent.py - ASYNC/AWAIT FIXES

async def _claude_driven_tool_calling(self, message: str, context: Dict) -> Dict[str, Any]:
    """
    Let Claude decide which tools to call and execute them with real data
    FIXED: Proper async/await handling for all operations
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
        
        # ✅ FIX #1: Proper await for Claude API call
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
                    
                    # ✅ FIX #2: Proper await for tool execution
                    if function_name == "getTechnical" and self.ta_service:
                        result = await self.ta_service.analyze_symbol(symbol)  # ✅ AWAIT added
                        if result:
                            tool_results[f"technical_{symbol}"] = result
                    
                    # ✅ FIX #3: Handle synchronous FAEngine correctly
                    elif function_name == "getFundamentals" and self.fundamental_tool:
                        # FAEngine execute() is synchronous, no await needed
                        result = self.fundamental_tool.execute({
                            "symbol": symbol,
                            "depth": "standard", 
                            "user_style": "professional"
                        })
                        if result.get("success"):
                            tool_results[f"fundamental_{symbol}"] = result["analysis_result"]
                    
                    # ✅ FIX #4: Proper await for news service
                    elif function_name == "getNews" and self.news_service:
                        result = await self.news_service.get_sentiment(symbol)  # ✅ AWAIT added
                        if result and not result.get('error'):
                            tool_results[f"news_{symbol}"] = result
        
        return tool_results
        
    except Exception as e:
        logger.error(f"Claude tool calling failed: {e}")
        return {}

# ✅ FIX #5: Ensure _get_conversation_context is properly awaited
async def _get_conversation_context(self, user_phone: str) -> Dict[str, Any]:
    """Get conversation context using KeyBuilder - FIXED async handling"""
    
    context = {
        "recent_symbols": [],
        "conversation_flow": "new",
        "last_topic": None
    }
    
    try:
        # Try KeyBuilder first for conversation context
        if self.db_service and hasattr(self.db_service, 'key_builder'):
            try:
                # ✅ FIX #6: Proper await for KeyBuilder operations
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
            # ✅ FIX #7: Proper await for cache operations
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

# ✅ FIX #8: Ensure _get_user_profile is properly awaited
async def _get_user_profile(self, user_phone: str) -> Dict:
    """Get user personality profile using KeyBuilder - FIXED async handling"""
    
    try:
        # Try KeyBuilder first for user profile
        if self.db_service and hasattr(self.db_service, 'key_builder'):
            try:
                # ✅ FIX #9: Proper await for KeyBuilder profile operations
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

# ✅ FIX #10: Update the main process_message method calls
async def process_message(self, message: str, user_phone: str) -> str:
    """
    Process message using Claude's superior reasoning - FIXED async calls
    """
    try:
        logger.info(f"🎯 Claude processing: '{message}' from {user_phone}")
        
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
        
        # ✅ FIX #11: Proper await for context retrieval
        if self.memory_manager:
            # Use memory manager for enhanced context
            memory_context = await self.memory_manager.get_context(
                user_id=user_phone,
                agent_type=AgentType.TRADING,
                query=message
            )
            context = self._merge_memory_with_cache_context(memory_context, user_phone)
        else:
            # Fallback to existing cache context - ✅ AWAIT added
            context = await self._get_conversation_context(user_phone)
        
        # ✅ FIX #12: Proper await for user profile
        user_profile = await self._get_user_profile(user_phone)
        
        # ✅ FIX #13: Proper await for tool calling
        tool_results = await self._claude_driven_tool_calling(message, context)
        
        # ✅ FIX #14: Proper await for response generation
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
        
        # ✅ FIX #15: Proper await for conversation caching
        await self._cache_conversation(user_phone, message, response, context)
        
        logger.info(f"✅ Claude response generated: {len(response)} chars")
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
