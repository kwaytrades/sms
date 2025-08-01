# services/data_driven_agent.py
"""
Simplified data-driven LLM agent - let the LLM handle conversation style
Enhanced with Context Orchestrator integration
Updated with KeyBuilder integration for centralized data access
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from loguru import logger
from datetime import datetime
from openai import AsyncOpenAI

# Import the new Context Orchestrator
try:
    from services.context_orchestrator import ContextOrchestrator, StructuredContext
    from services.memory_manager import MemoryManager, AgentType, MessageDirection
    CONTEXT_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    logger.warning("Context Orchestrator not available - using fallback context")
    CONTEXT_ORCHESTRATOR_AVAILABLE = False

class DataDrivenAgent:
    """
    Simplified LLM agent that:
    1. Gets intelligent context via Context Orchestrator
    2. Calls tools directly based on user query
    3. Passes raw data + context to LLM
    4. LLM handles everything - analysis, tone, conversation style
    """
    
    def __init__(self, openai_client, ta_service, news_service, fundamental_tool, cache_service, personality_engine, memory_manager=None, db_service=None):
        self.openai_client = openai_client
        self.ta_service = ta_service
        self.news_service = news_service
        self.fundamental_tool = fundamental_tool
        self.cache_service = cache_service
        self.personality_engine = personality_engine
        self.db_service = db_service  # DatabaseService with KeyBuilder
        
        # Initialize Context Orchestrator if available
        self.context_orchestrator = None
        if CONTEXT_ORCHESTRATOR_AVAILABLE and memory_manager:
            self.context_orchestrator = ContextOrchestrator(memory_manager)
            logger.info("✅ Context Orchestrator initialized")
        else:
            logger.warning("⚠️ Using fallback context - Context Orchestrator not available")
    
    async def process_message(self, message: str, user_phone: str) -> str:
        """
        Process message using enhanced context-aware approach:
        1. Get intelligent 3-layer context
        2. LLM calls tools with context awareness
        3. LLM generates response with full context intelligence
        """
        try:
            logger.info(f"🎯 Processing: '{message}' from {user_phone}")
            
            # Step 1: Get intelligent structured context
            structured_context = await self._get_intelligent_context(user_phone, message)
            
            # Step 2: LLM analyzes query and calls tools with context awareness
            tool_results = await self._context_aware_tool_calling(message, structured_context)
            
            # Step 3: LLM generates response with full context intelligence
            response = await self._generate_context_aware_response(
                message, structured_context, tool_results
            )
            
            # Step 4: Save message to memory system
            await self._save_to_memory(user_phone, message, response, structured_context)
            
            logger.info(f"✅ Context-aware response generated: {len(response)} chars")
            return response
            
        except Exception as e:
            logger.error(f"💥 Processing failed: {e}")
            return "Market analysis processing. Please try again shortly."
    
    async def _get_intelligent_context(self, user_phone: str, message: str) -> StructuredContext:
        """Get intelligent 3-layer context via Context Orchestrator"""
        
        if self.context_orchestrator:
            try:
                # Get structured context with symbol inference and conversation threading
                structured_context = await self.context_orchestrator.get_structured_context(
                    user_id=user_phone,
                    current_message=message,
                    agent_type=AgentType.TRADING
                )
                
                logger.info(f"📊 Context retrieved: {structured_context.context_confidence:.2f} confidence, "
                           f"{len(structured_context.inferred_symbols)} symbols inferred, "
                           f"{len(structured_context.immediate_context)} recent messages")
                
                return structured_context
                
            except Exception as e:
                logger.error(f"Context orchestrator failed: {e}")
                return await self._get_fallback_context(user_phone)
        else:
            return await self._get_fallback_context(user_phone)
    
    async def _context_aware_tool_calling(self, message: str, context: StructuredContext) -> Dict[str, Any]:
        """
        Enhanced tool calling with context intelligence and symbol inference
        """
        
        # Format context for LLM tool calling
        context_summary = self._format_context_for_tool_calling(context)
        
        # Enhanced tool calling prompt with context intelligence
        tool_calling_prompt = f"""You are a trading analyst with access to market data tools. Use the conversation context to make intelligent decisions about which tools to call.

USER REQUEST: "{message}"

CONVERSATION CONTEXT & INTELLIGENCE:
{context_summary}

AVAILABLE TOOLS:
- getTechnical(symbol): Get technical analysis, price, RSI, support/resistance
- getFundamentals(symbol): Get P/E ratio, financial health, growth metrics  
- getNews(symbol): Get recent news sentiment and market impact

CONTEXT-AWARE INSTRUCTIONS:
1. Use the conversation context to understand what the user is asking about
2. If symbols are inferred from context (like "Support $160?" after TSLA discussion), use those symbols
3. Don't ask for clarification if context makes the intent clear
4. Call appropriate tools to get REAL market data
5. Be intelligent about which tools are needed based on the request type

Extract symbols from context and current message, then call appropriate tools:"""

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
                    
                    logger.info(f"🔧 Context-aware LLM requested: {function_name}({symbol})")
                    
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
            logger.error(f"Context-aware tool calling failed: {e}")
            return {}
    
    async def _generate_context_aware_response(
        self, message: str, context: StructuredContext, tool_results: Dict
    ) -> str:
        """
        Generate response with full context intelligence and conversation awareness
        """
        
        # Format everything for context-aware LLM response
        tool_data = self._format_raw_data_for_llm(tool_results)
        context_intelligence = self._format_context_intelligence_for_llm(context)
        
        # Enhanced comprehensive prompt with context intelligence
        context_aware_prompt = f"""You are a hyper-personalized SMS trading assistant with advanced conversation intelligence. The user just asked: "{message}"

CONVERSATION INTELLIGENCE & CONTEXT:
{context_intelligence}

REAL MARKET DATA AVAILABLE:
{tool_data}

CONTEXT-AWARE RESPONSE GUIDELINES:
1. Use conversation context to understand the user's intent (don't ask for clarification if context is clear)
2. Reference previous conversation naturally when relevant
3. Match the user's established communication style and preferences
4. Answer their specific question with context-aware insights
5. If you inferred symbols from context, acknowledge this naturally
6. Use real market data to provide actionable insights
7. Keep responses SMS-friendly (under 480 characters)
8. Be conversational and adaptive to the user's personality

Generate your context-aware response now:"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": context_aware_prompt}],
                temperature=0.4,
                max_tokens=200
            )
            
            generated_response = response.choices[0].message.content.strip()
            return self._clean_final_response(generated_response)
            
        except Exception as e:
            logger.error(f"Context-aware response generation failed: {e}")
            return "Market analysis completed. Please try your request again."
    
    def _format_context_for_tool_calling(self, context: StructuredContext) -> str:
        """Format structured context for tool calling decisions"""
        
        context_parts = []
        
        # Add conversation thread info
        thread = context.conversation_thread
        if thread.active_symbols:
            context_parts.append(f"ACTIVE SYMBOLS IN CONVERSATION: {', '.join(thread.active_symbols)}")
        
        if thread.current_topic != "general":
            context_parts.append(f"CURRENT TOPIC: {thread.current_topic}")
        
        # Add recent conversation
        if context.immediate_context:
            context_parts.append("RECENT CONVERSATION:")
            for msg in context.immediate_context[:3]:
                direction = "User" if msg.get('direction') == 'user' else "Bot"
                content = msg.get('content', '')[:80]
                context_parts.append(f"- {direction}: {content}")
        
        # Add symbol inferences
        if context.inferred_symbols:
            context_parts.append("SYMBOL INFERENCES:")
            for inference in context.inferred_symbols[:2]:
                context_parts.append(f"- {inference.symbol}: {inference.reasoning} (confidence: {inference.confidence:.1f})")
        
        # Add user profile
        if context.user_profile:
            profile = context.user_profile
            context_parts.append(f"USER PROFILE: {profile.get('trading_style', 'swing')} trader, {profile.get('risk_tolerance', 'medium')} risk tolerance")
        
        return "\n".join(context_parts) if context_parts else "No relevant context available"
    
    def _format_context_intelligence_for_llm(self, context: StructuredContext) -> str:
        """Format full context intelligence for response generation"""
        
        if self.context_orchestrator:
            # Use the built-in context formatter
            return self.context_orchestrator.format_context_for_llm(context)
        else:
            return self._format_context_for_tool_calling(context)
    
    async def _save_to_memory(self, user_phone: str, message: str, response: str, context: StructuredContext):
        """Save conversation to memory system"""
        
        if self.context_orchestrator and self.context_orchestrator.memory_manager:
            try:
                # Save user message
                await self.context_orchestrator.memory_manager.save_message(
                    user_id=user_phone,
                    content=message,
                    direction=MessageDirection.USER,
                    agent_type=AgentType.TRADING
                )
                
                # Save bot response
                await self.context_orchestrator.memory_manager.save_message(
                    user_id=user_phone,
                    content=response,
                    direction=MessageDirection.BOT,
                    agent_type=AgentType.TRADING
                )
                
                logger.debug(f"💾 Conversation saved to memory for {user_phone}")
                
            except Exception as e:
                logger.warning(f"Memory save failed: {e}")
        
        # Fallback to cache service
        await self._cache_conversation_fallback(user_phone, message, response, context)
    
    async def _get_fallback_context(self, user_phone: str) -> StructuredContext:
        """Fallback context when Context Orchestrator is not available"""
        
        # Get basic conversation context using KeyBuilder
        context = await self._get_conversation_context(user_phone)
        
        # Create minimal StructuredContext-like object
        from types import SimpleNamespace
        
        fallback_context = SimpleNamespace()
        fallback_context.conversation_thread = SimpleNamespace()
        fallback_context.conversation_thread.active_symbols = context.get("recent_symbols", [])
        fallback_context.conversation_thread.current_topic = context.get("last_topic", "general")
        fallback_context.conversation_thread.emotional_state = "neutral"
        
        fallback_context.immediate_context = []
        fallback_context.semantic_memories = []
        fallback_context.user_profile = await self._get_user_profile(user_phone)
        fallback_context.conversation_summaries = []
        fallback_context.inferred_symbols = []
        fallback_context.context_confidence = 0.3
        fallback_context.missing_context_flags = ["context_orchestrator_unavailable"]
        
        return fallback_context
    
    def _format_raw_data_for_llm(self, tool_results: Dict) -> str:
        """Format raw tool results for LLM - keep it simple"""
        
        if not tool_results:
            return "No market data retrieved"
        
        formatted_results = []
        
        for key, data in tool_results.items():
            symbol = key.split('_', 1)[1] if '_' in key else 'UNKNOWN'
            data_type = key.split('_', 1)[0] if '_' in key else 'unknown'
            
            result_text = f"{data_type.upper()} DATA for {symbol}:\n{json.dumps(data, indent=2, default=str)}"
            formatted_results.append(result_text)
        
        return "\n\n".join(formatted_results)
    
    def _clean_final_response(self, response: str) -> str:
        """Minimal cleaning - let LLM handle most of it"""
        
        # Just ensure length limit
        if len(response) > 480:
            response = response[:477] + "..."
        
        return response.strip()
    
    async def _get_conversation_context(self, user_phone: str) -> Dict[str, Any]:
        """Get conversation context using KeyBuilder"""
        
        context = {
            "recent_symbols": [],
            "conversation_flow": "new",
            "last_topic": None
        }
        
        try:
            if self.db_service and hasattr(self.db_service, 'key_builder'):
                # Use KeyBuilder to get user context
                user_context = await self.db_service.key_builder.get_user_context(user_phone)
                if user_context:
                    context["recent_symbols"] = user_context.get("recent_symbols", [])
                    context["conversation_flow"] = user_context.get("conversation_flow", "new")
                    context["last_topic"] = user_context.get("last_topic", None)
                    return context
            
            # Fallback to cache service if KeyBuilder not available
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
        """Get user personality profile using KeyBuilder"""
        
        try:
            # Try KeyBuilder first
            if self.db_service and hasattr(self.db_service, 'key_builder'):
                user_profile = await self.db_service.key_builder.get_user_profile(user_phone)
                if user_profile:
                    return user_profile
            
            # Fallback to personality engine
            if self.personality_engine and hasattr(self.personality_engine, 'user_profiles'):
                return self.personality_engine.user_profiles.get(user_phone, {})
            return {}
        except Exception as e:
            logger.warning(f"Profile retrieval failed: {e}")
            return {}
    
    async def _cache_conversation_fallback(self, user_phone: str, message: str, response: str, context):
        """Cache conversation (fallback method)"""
        
        try:
            # Extract symbols and create conversation entry
            symbols = self._extract_symbols_from_text(f"{message} {response}")
            
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_message": message,
                "bot_response": response,
                "symbols": symbols,
                "topic": self._determine_topic(message)
            }
            
            # Use cache service (this file only reads from DB, writes to cache)
            if self.cache_service:
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


# Enhanced drop-in replacement for existing processor
class ComprehensiveMessageProcessor:
    """Enhanced processor using context-aware data-driven agent"""
    
    def __init__(self, openai_client, ta_service, personality_engine, 
                 cache_service=None, news_service=None, fundamental_tool=None, 
                 portfolio_service=None, screener_service=None, memory_manager=None, db_service=None):
        
        self.data_driven_agent = DataDrivenAgent(
            openai_client=openai_client,
            ta_service=ta_service,
            news_service=news_service,
            fundamental_tool=fundamental_tool,
            cache_service=cache_service,
            personality_engine=personality_engine,
            memory_manager=memory_manager,  # Pass memory manager for context orchestrator
            db_service=db_service  # Pass DatabaseService with KeyBuilder
        )
        
        self.personality_engine = personality_engine
        self.cache_service = cache_service
    
    async def process_message(self, message: str, user_phone: str) -> str:
        """Process using context-aware data-driven agent"""
        
        try:
            response = await self.data_driven_agent.process_message(message, user_phone)
            return response
            
        except Exception as e:
            logger.error(f"💥 Processing failed: {e}")
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
